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
use crate::prompt;
use crate::results::{Accelerators, HintResult, ModelRun};
use crate::score;
use crate::system;

/// Longest we wait for one hint before calling it failed.
const HINT_TIMEOUT: Duration = Duration::from_secs(240);
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
    pub async fn variants(&self, alias: &str, all: bool) -> Result<Vec<Arc<Model>>> {
        let model = self
            .manager
            .catalog()
            .get_model(alias)
            .await
            .map_err(|e| anyhow!("{alias} isn't in the model catalog ({e})"))?;
        Ok(if all { model.variants() } else { vec![model] })
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
        };
        println!("\n== {alias} ({}, {})", run.variant, run.device);
        if let Err(e) = self.prepare(&model, &mut run, downloaded).await {
            run.error = Some(e.to_string());
            println!("   {e}");
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
            run.hints.push(hint);
        }
        if let Err(e) = model.unload().await {
            println!("   (couldn't unload: {e})");
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
            .map_err(|e| anyhow!("Couldn't load the model: {e}"))?;
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
        let hint = self.hint(model, &warm).await;
        if let Some(e) = hint.error {
            return Err(anyhow!("The model loaded but couldn't answer: {e}"));
        }
        run.warmup_s = Some(started.elapsed().as_secs_f64());
        Ok(())
    }

    async fn hint(&self, model: &Model, case: &Case) -> HintResult {
        let mut result = HintResult {
            case_id: case.id.clone(),
            text: String::new(),
            first_token_s: None,
            total_s: 0.0,
            tokens: 0,
            checks: None,
            error: None,
        };
        let request = Request::from_items(vec![
            Item::system_message(vec![Item::text(prompt::SYSTEM)]),
            Item::user_message(vec![Item::text(prompt::user_message(case))]),
        ])
        .with_options(RequestOptions {
            search: SearchOptions {
                temperature: Some(0.2),
                max_output_tokens: Some(220),
                seed: Some(7),
                ..SearchOptions::default()
            },
            ..RequestOptions::default()
        });

        let started = Instant::now();
        let mut pieces = 0u64;
        let mut reported = None;
        let outcome = tokio::time::timeout(HINT_TIMEOUT, async {
            // A fresh session per hint: each learner question stands alone.
            let session = ChatSession::new(model).await?;
            let mut stream = session.process_streaming_request(request);
            while let Some(item) = stream.next().await {
                if self.cancelled() {
                    return Err(anyhow!("stopped"));
                }
                if let Item::Text {
                    text,
                    kind: TextKind::Default,
                } = item?
                {
                    if text.is_empty() {
                        continue;
                    }
                    if result.first_token_s.is_none() {
                        result.first_token_s = Some(started.elapsed().as_secs_f64());
                    }
                    pieces += 1;
                    result.text.push_str(&text);
                }
            }
            if let Ok(response) = stream.response().await {
                reported = Some(u64::from(response.usage.completion_tokens));
                if result.text.is_empty() {
                    result.text = response.text();
                }
            }
            anyhow::Ok(())
        })
        .await;
        result.total_s = started.elapsed().as_secs_f64();
        result.tokens = reported.filter(|t| *t > 0).unwrap_or(pieces);

        match outcome {
            Err(_) => {
                result.error = Some(format!(
                    "no full answer within {} s",
                    HINT_TIMEOUT.as_secs()
                ))
            }
            Ok(Err(e)) => result.error = Some(e.to_string()),
            Ok(Ok(())) if result.text.trim().is_empty() => {
                result.error = Some("empty answer".into())
            }
            Ok(Ok(())) => result.checks = Some(score::check(case, &result.text)),
        }
        result
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

fn device_name(d: &DeviceType) -> String {
    match d {
        DeviceType::CPU => "Processor (CPU)",
        DeviceType::GPU => "Graphics (GPU)",
        DeviceType::NPU => "AI chip (NPU)",
        DeviceType::Invalid => "Unknown",
    }
    .to_string()
}
