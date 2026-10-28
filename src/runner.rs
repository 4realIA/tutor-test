//! Drives Foundry Local: accelerators, downloads, loading and timed hints.

use std::io::{self, Write};
use std::path::Path;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};

use anyhow::{Context, Result, anyhow};
use foundry_local_sdk::{
    ChatSession, DeviceType, FoundryLocalConfig, FoundryLocalManager, Item, Model, Request,
    RequestOptions, SearchOptions, TextKind,
};
use tokio_stream::StreamExt;

use crate::cases::Case;
use crate::prompt::{self, Role};
use crate::results::{Accelerators, HintResult, ModelRun};
use crate::score;
use crate::system;

/// A model that can't answer in reasonable time on this computer.
#[derive(Debug)]
struct TooSlow(String);

impl std::fmt::Display for TooSlow {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for TooSlow {}

/// Start of the error recorded when a hint times out.
const TIMED_OUT: &str = "no full answer within";
/// A model whose warm-up answer takes longer than this is too slow to test.
const WARMUP_LIMIT: Duration = Duration::from_secs(90);

/// Longest we wait for one hint before calling it failed.
const HINT_TIMEOUT: Duration = Duration::from_secs(240);
#[derive(Default)]
struct Generation {
    text: String,
    thinking_words: u64,
    first_token_s: Option<f64>,
    total_s: f64,
    tokens: u64,
    error: Option<String>,
}

/// Room left on the disk after a download.
const DISK_MARGIN_MB: u64 = 500;

pub struct Runner {
    pub manager: Arc<FoundryLocalManager>,
    pub cancel: Arc<AtomicBool>,
    pub data_dir: std::path::PathBuf,
}

fn native_lib_name() -> &'static str {
    if cfg!(windows) {
        "foundry_local.dll"
    } else if cfg!(target_os = "macos") {
        "libfoundry_local.dylib"
    } else {
        "libfoundry_local.so"
    }
}

impl Runner {
    pub fn start(cancel: Arc<AtomicBool>) -> Result<Runner> {
        let data_dir = system::home().join(".tutor-test");
        std::fs::create_dir_all(&data_dir)
            .with_context(|| format!("couldn't create {}", data_dir.display()))?;
        let mut config =
            FoundryLocalConfig::new("tutor-test").app_data_dir(data_dir.to_string_lossy());
        // The engine ships next to the program; prefer that copy over any path
        // remembered from the build machine.
        if let Some(dir) = system::exe_dir().filter(|d| d.join(native_lib_name()).exists()) {
            config = config.library_path(dir.to_string_lossy());
        }
        let manager = FoundryLocalManager::create(config).map_err(|e| {
            anyhow!(
                "couldn't start the AI engine: {e}. Keep all the files from the zip together in one folder."
            )
        })?;
        Ok(Runner {
            manager,
            cancel,
            data_dir,
        })
    }

    fn cancelled(&self) -> bool {
        self.cancel.load(Ordering::SeqCst)
    }

    /// Set up GPU/NPU support where the computer has it. Failures fall back to the processor.
    pub async fn accelerators(&self) -> Accelerators {
        let mut acc = Accelerators::default();
        match self.manager.discover_eps() {
            Ok(eps) if eps.is_empty() => {
                acc.note = Some("No extra accelerators offered for this computer.".into());
                return acc;
            }
            Ok(eps) => {
                let names: Vec<&str> = eps.iter().map(|e| e.name.as_str()).collect();
                println!("Setting up AI accelerators: {}", names.join(", "));
            }
            Err(e) => {
                acc.note = Some(format!("Couldn't list accelerators: {e}"));
                return acc;
            }
        }
        let mut current = String::new();
        let result = self
            .manager
            .download_and_register_eps_builder()
            .progress(move |name: &str, pct: f64| {
                if name != current {
                    if !current.is_empty() {
                        println!();
                    }
                    current = name.to_string();
                }
                print!("\r  {name:<36} {pct:5.1}%");
                io::stdout().flush().ok();
            })
            .cancel(Arc::clone(&self.cancel))
            .run()
            .await;
        println!();
        match result {
            Ok(r) => {
                acc.registered = r.registered_eps;
                acc.failed = r.failed_eps;
                if !r.success {
                    acc.note = Some(format!(
                        "Some accelerators couldn't be set up: {}",
                        r.status
                    ));
                }
            }
            Err(e) => {
                acc.note = Some(format!(
                    "Accelerator setup failed, using the processor: {e}"
                ))
            }
        }
        acc
    }

    pub async fn list(&self) -> Result<()> {
        let models = self.manager.catalog().get_models().await?;
        println!("{:<34} {:<52} {:>8}  DEVICE", "ALIAS", "VARIANT", "SIZE");
        for m in models {
            for v in m.variants() {
                let info = v.info()?;
                let (device, ep) = info
                    .runtime
                    .as_ref()
                    .map(|r| (device_name(&r.device_type), r.execution_provider.clone()))
                    .unwrap_or_default();
                let size = info
                    .file_size_mb
                    .map(|mb| format!("{:.2} GB", mb as f64 / 1024.0))
                    .unwrap_or_default();
                let cached = if info.cached { " (downloaded)" } else { "" };
                println!(
                    "{:<34} {:<52} {:>8}  {device} {ep}{cached}",
                    m.alias(),
                    v.id(),
                    size
                );
            }
        }
        Ok(())
    }

    /// The model variants to test for one alias.
    /// The versions of a model to test. Normally just the one we'd try first
    /// (see [`rank`]); with `all`, every version in that order.
    pub async fn variants(&self, alias: &str, all: bool) -> Result<Vec<Arc<Model>>> {
        let mut ranked = self.ranked_variants(alias).await?;
        if !all {
            ranked.truncate(1);
        }
        Ok(ranked)
    }

    async fn ranked_variants(&self, alias: &str) -> Result<Vec<Arc<Model>>> {
        let model = self
            .manager
            .catalog()
            .get_model(alias)
            .await
            .map_err(|e| anyhow!("{alias} isn't in the model catalog ({e})"))?;
        let mut variants = model.variants();
        if variants.is_empty() {
            return Ok(vec![model]);
        }
        variants.sort_by_key(|v| {
            let runtime = v.info().ok().and_then(|i| i.runtime);
            rank(
                runtime.as_ref().map(|r| &r.device_type),
                runtime
                    .as_ref()
                    .map(|r| r.execution_provider.as_str())
                    .unwrap_or(""),
            )
        });
        Ok(variants)
    }

    /// Download (if needed), load, warm up, then time every case. Never fails:
    /// problems are recorded in the result so the other models still run.
    pub async fn run_model(
        &self,
        alias: &str,
        model: Arc<Model>,
        cases: &[Case],
        downloaded: &mut Vec<Arc<Model>>,
    ) -> ModelRun {
        let info = model.info().ok();
        let runtime = info.as_ref().and_then(|i| i.runtime.clone());
        let mut run = ModelRun {
            alias: alias.to_string(),
            variant: model.id().to_string(),
            device: runtime
                .as_ref()
                .map(|r| device_name(&r.device_type))
                .unwrap_or_else(|| "?".into()),
            execution_provider: runtime.map(|r| r.execution_provider).unwrap_or_default(),
            size_mb: info.as_ref().and_then(|i| i.file_size_mb),
            download_s: None,
            load_s: None,
            warmup_s: None,
            hints: Vec::new(),
            error: None,
            too_slow: None,
        };
        println!("\n== {alias} ({}, {})", run.variant, run.device);
        if let Err(e) = self.prepare(&model, &mut run, downloaded).await {
            println!("   {e}");
            if e.downcast_ref::<TooSlow>().is_some() {
                run.too_slow = Some(e.to_string());
            } else {
                run.error = Some(e.to_string());
            }
            return run;
        }
        if let Some(w) = run.warmup_s.filter(|w| *w > WARMUP_LIMIT.as_secs_f64()) {
            let why = format!("Too slow on this computer: the warm-up answer took {w:.0} s.");
            println!("   {why} Skipping its cases.");
            run.too_slow = Some(why);
            return run;
        }

        for (i, case) in cases.iter().enumerate() {
            if self.cancelled() {
                run.error = Some("Stopped before all cases ran.".into());
                break;
            }
            print!("   [{:>2}/{}] {:<44}", i + 1, cases.len(), case.title);
            io::stdout().flush().ok();
            let hint = self.hint(&model, case).await;
            match &hint.error {
                None => println!(
                    " {:>5.1} s{}",
                    hint.total_s,
                    if hint.checks.as_ref().is_some_and(|c| c.passed()) {
                        ""
                    } else {
                        "  (check flagged)"
                    }
                ),
                Some(e) => println!(" failed: {e}"),
            }
            let timed_out = hint
                .error
                .as_deref()
                .is_some_and(|e| e.starts_with(TIMED_OUT));
            run.hints.push(hint);
            if timed_out {
                let why = format!(
                    "Too slow on this computer: a hint took over {} s.",
                    HINT_TIMEOUT.as_secs()
                );
                println!("   {why} Skipping its other cases.");
                run.too_slow = Some(why);
                break;
            }
        }
        let s = run.summary();
        println!(
            "   typical hint {}, {} of {} passed the checks",
            s.median_total_s
                .map(|t| format!("{t:.1} s"))
                .unwrap_or("–".into()),
            s.passed_checks,
            run.hints.len()
        );
        run
    }

    async fn prepare(
        &self,
        model: &Arc<Model>,
        run: &mut ModelRun,
        downloaded: &mut Vec<Arc<Model>>,
    ) -> Result<()> {
        if !model.is_cached().await? {
            if let (Some(mb), Some(free)) = (run.size_mb, system::free_space(&self.data_dir)) {
                let needed = (mb + DISK_MARGIN_MB) * 1024 * 1024;
                if free < needed {
                    return Err(anyhow!(
                        "Not enough disk space: needs about {:.1} GB, {:.1} GB free.",
                        needed as f64 / 1024f64.powi(3),
                        free as f64 / 1024f64.powi(3)
                    ));
                }
            }
            println!(
                "   Downloading{}…",
                run.size_mb
                    .map(|mb| format!(" {:.1} GB", mb as f64 / 1024.0))
                    .unwrap_or_default()
            );
            let started = Instant::now();
            let mut last = -1i64;
            let result = model
                .download_builder()
                .progress(move |pct: f64| {
                    let whole = pct.floor() as i64;
                    if whole != last {
                        last = whole;
                        print!("\r   {pct:5.1}%");
                        io::stdout().flush().ok();
                    }
                })
                .cancel(Arc::clone(&self.cancel))
                .run()
                .await;
            println!();
            if self.cancelled() {
                return Err(anyhow!("Download stopped."));
            }
            result.map_err(|e| anyhow!("Download failed: {e}"))?;
            run.download_s = Some(started.elapsed().as_secs_f64());
            downloaded.push(Arc::clone(model));
        }

        println!("   Loading…");
        let started = Instant::now();
        model
            .load()
            .await
            .map_err(|e| anyhow!("{LOAD_FAILED} {e}"))?;
        run.load_s = Some(started.elapsed().as_secs_f64());

        // The first answer after loading is slower; don't count it.
        println!("   Warming up…");
        let started = Instant::now();
        let warm = Case {
            id: "warmup".into(),
            title: "warm-up".into(),
            lesson: "Printing text".into(),
            task: "Print Hello.".into(),
            code: "print(Hello)".into(),
            output: "NameError: name 'Hello' is not defined".into(),
            mentions: vec!["quote".into()],
            giveaway: "print(\"Hello\")".into(),
        };
        let turns =
            prompt::with_no_think(prompt::conversation(&warm), wants_no_think(model.alias()));
        let answer = self.generate(model, &turns, WARMUP_LIMIT).await;
        match answer.error {
            Some(e) if e.starts_with(TIMED_OUT) => {
                return Err(anyhow!(TooSlow(format!(
                    "Too slow on this computer: no warm-up answer within {} s.",
                    WARMUP_LIMIT.as_secs()
                ))));
            }
            Some(e) => return Err(anyhow!("{NO_ANSWER} {e}")),
            None => {}
        }
        run.warmup_s = Some(started.elapsed().as_secs_f64());
        Ok(())
    }

    /// One hint, as the app would produce it: ask, check, and if the hint gave
    /// the answer away or showed code, ask once for a rewrite.
    async fn hint(&self, model: &Model, case: &Case) -> HintResult {
        let no_think = wants_no_think(model.alias());
        let first = self
            .generate(
                model,
                &prompt::with_no_think(prompt::conversation(case), no_think),
                HINT_TIMEOUT,
            )
            .await;
        let mut result = HintResult {
            case_id: case.id.clone(),
            text: first.text,
            first_token_s: first.first_token_s,
            total_s: first.total_s,
            tokens: first.tokens,
            checks: None,
            error: first.error,
            first_text: None,
            first_checks: None,
            rewrite_s: None,
            rewrite_error: None,
            thinking_words: first.thinking_words,
        };
        if result.error.is_some() {
            return result;
        }
        let checks = score::check(case, &result.text);
        if !(checks.gave_answer || checks.code_block) || self.cancelled() {
            result.checks = Some(checks);
            return result;
        }

        let rewrite = self
            .generate(
                model,
                &prompt::with_no_think(prompt::rewrite_conversation(case, &result.text), no_think),
                HINT_TIMEOUT,
            )
            .await;
        result.rewrite_s = Some(rewrite.total_s);
        match rewrite.error {
            Some(e) => {
                // Keep the first hint; note why there's no rewrite.
                result.rewrite_error = Some(e);
                result.checks = Some(checks);
            }
            None => {
                result.first_text = Some(std::mem::replace(&mut result.text, rewrite.text));
                result.first_checks = Some(checks);
                result.checks = Some(score::check(case, &result.text));
            }
        }
        result
    }

    /// Stream one answer for a conversation, timing it.
    async fn generate(
        &self,
        model: &Model,
        turns: &[(Role, String)],
        limit: Duration,
    ) -> Generation {
        let items = turns
            .iter()
            .map(|(role, text)| {
                let content = vec![Item::text(text.as_str())];
                match role {
                    Role::System => Item::system_message(content),
                    Role::User => Item::user_message(content),
                    Role::Assistant => Item::assistant_message(content),
                }
            })
            .collect::<Vec<_>>();
        let request = Request::from_items(items).with_options(RequestOptions {
            search: SearchOptions {
                temperature: Some(0.2),
                // Room for models that think before answering; the prompt keeps
                // the visible hint to 4 sentences.
                max_output_tokens: Some(1024),
                seed: Some(7),
                ..SearchOptions::default()
            },
            ..RequestOptions::default()
        });

        let mut out = Generation::default();
        let started = Instant::now();
        let mut pieces = 0u64;
        let mut reported = None;
        let mut thinking = String::new();
        let outcome = tokio::time::timeout(limit, async {
            // A fresh session per request: each learner question stands alone.
            let session = ChatSession::new(model).await?;
            let mut stream = session.process_streaming_request(request);
            while let Some(item) = stream.next().await {
                if self.cancelled() {
                    return Err(anyhow!("stopped"));
                }
                match item? {
                    Item::Text {
                        text,
                        kind: TextKind::Default,
                    } if !text.is_empty() => {
                        if out.first_token_s.is_none() {
                            out.first_token_s = Some(started.elapsed().as_secs_f64());
                        }
                        pieces += 1;
                        out.text.push_str(&text);
                    }
                    Item::Text {
                        text,
                        kind: TextKind::Reasoning,
                    } => thinking.push_str(&text),
                    _ => {}
                }
            }
            if let Ok(response) = stream.response().await {
                reported = Some(u64::from(response.usage.completion_tokens));
                if out.text.is_empty() {
                    out.text = response.text();
                }
            }
            anyhow::Ok(())
        })
        .await;
        out.total_s = started.elapsed().as_secs_f64();
        // Some models put their thinking inline in <think> tags instead.
        let (visible, inline) = prompt::split_thinking(&out.text);
        if !inline.is_empty() {
            thinking.push_str(&inline);
            out.text = visible;
        }
        out.thinking_words = thinking.split_whitespace().count() as u64;
        out.tokens = reported.filter(|t| *t > 0).unwrap_or(pieces);
        out.error = match outcome {
            Err(_) => Some(format!("{TIMED_OUT} {} s", limit.as_secs())),
            Ok(Err(e)) => Some(e.to_string()),
            Ok(Ok(())) if out.text.trim().is_empty() => Some("empty answer".into()),
            Ok(Ok(())) => None,
        };
        out
    }

    /// Unload a model. After a timeout the engine may still be stopping the
    /// abandoned answer, so keep trying for a while.
    pub async fn unload(&self, model: &Model) {
        for attempt in 0..30 {
            match model.unload().await {
                Ok(()) => return,
                Err(e) if attempt == 29 => println!("   (couldn't unload: {e})"),
                Err(_) => tokio::time::sleep(Duration::from_secs(2)).await,
            }
        }
    }

    /// The next version of a model to try after one failed or was too slow:
    /// the best-ranked version not tried yet.
    pub async fn fallback_variant(&self, alias: &str, tried: &[String]) -> Option<Arc<Model>> {
        self.ranked_variants(alias)
            .await
            .ok()?
            .into_iter()
            .find(|v| !tried.iter().any(|t| t == v.id()))
    }

    pub async fn remove(&self, models: &[Arc<Model>]) {
        for m in models {
            match m.remove_from_cache().await {
                Ok(()) => println!("Deleted {}", m.id()),
                Err(e) => println!("Couldn't delete {}: {e}", m.id()),
            }
        }
    }

    pub fn cache_dir(&self) -> &Path {
        &self.data_dir
    }
}

/// Qwen3-family models think before answering unless told not to.
fn wants_no_think(alias: &str) -> bool {
    alias.to_lowercase().starts_with("qwen3")
}

pub const CPU_NAME: &str = "Processor (CPU)";
pub const GPU_NAME: &str = "Graphics (GPU)";

/// Order in which to try a model's versions. Standard graphics (WebGPU) first:
/// it ran on the Macs and on Mom's Intel laptop. Then other graphics, then the
/// processor. Intel's OpenVINO versions failed to load on her laptop with this
/// engine release, and AI-chip (NPU) versions are untested, so they come last.
pub fn rank(device: Option<&DeviceType>, execution_provider: &str) -> u8 {
    let ep = execution_provider.to_lowercase();
    match device {
        _ if ep.contains("openvino") => 4,
        Some(DeviceType::GPU) if ep.contains("webgpu") => 0,
        Some(DeviceType::GPU) => 1,
        Some(DeviceType::CPU) => 2,
        Some(DeviceType::NPU) => 3,
        _ => 5,
    }
}

/// Errors after which another device is worth trying (not download or disk problems).
pub fn worth_another_device(error: &str) -> bool {
    error.starts_with(LOAD_FAILED) || error.starts_with(NO_ANSWER)
}

const LOAD_FAILED: &str = "Couldn't load the model:";
const NO_ANSWER: &str = "The model loaded but couldn't answer:";

fn device_name(d: &DeviceType) -> String {
    match d {
        DeviceType::CPU => CPU_NAME,
        DeviceType::GPU => GPU_NAME,
        DeviceType::NPU => "AI chip (NPU)",
        DeviceType::Invalid => "Unknown",
    }
    .to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn standard_graphics_first_openvino_last() {
        let gpu = DeviceType::GPU;
        let cpu = DeviceType::CPU;
        let npu = DeviceType::NPU;
        let webgpu = rank(Some(&gpu), "WebGpuExecutionProvider");
        let dml = rank(Some(&gpu), "DmlExecutionProvider");
        let proc = rank(Some(&cpu), "CPUExecutionProvider");
        let ai_chip = rank(Some(&npu), "QNNExecutionProvider");
        let ov_gpu = rank(Some(&gpu), "OpenVINOExecutionProvider");
        let ov_npu = rank(Some(&npu), "OpenVINOExecutionProvider");
        assert!(webgpu < dml && dml < proc && proc < ai_chip && ai_chip < ov_gpu);
        assert_eq!(ov_gpu, ov_npu);
    }

    #[test]
    fn load_and_answer_failures_try_another_device() {
        assert!(worth_another_device(&format!("{LOAD_FAILED} native error")));
        assert!(worth_another_device(&format!("{NO_ANSWER} empty answer")));
        assert!(!worth_another_device("Download failed: timeout"));
        assert!(!worth_another_device(
            "Not enough disk space: needs about 5 GB"
        ));
    }
}
