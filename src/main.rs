//! Tutor test: measures how well and how fast local AI models (Foundry Local)
//! give beginner Python hints on this computer, and writes an HTML report.

mod args;
mod cases;
mod prompt;
mod report;
mod results;
mod runner;
mod score;
mod system;

use std::io::{self, BufRead, IsTerminal, Write};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use anyhow::{Context, Result};
use args::Cleanup;
use results::Run;
use runner::Runner;

fn now() -> String {
    chrono::Local::now().format("%Y-%m-%d %H:%M").to_string()
}

fn interactive() -> bool {
    io::stdin().is_terminal() && io::stdout().is_terminal()
}

fn ask(question: &str) -> String {
    print!("{question}");
    io::stdout().flush().ok();
    let mut line = String::new();
    io::stdin().lock().read_line(&mut line).ok();
    line.trim().to_lowercase()
}

/// Keep the window open when the program was double-clicked.
fn pause() {
    if interactive() {
        ask("\nPress Enter to close.");
    }
}

#[tokio::main]
async fn main() {
    let code = match run().await {
        Ok(()) => 0,
        Err(e) => {
            eprintln!("\nSomething went wrong: {e:#}");
            1
        }
    };
    pause();
    std::process::exit(code);
}

async fn run() -> Result<()> {
    let args = args::parse(std::env::args().skip(1))?;
    if args.help {
        print!("{}", args::HELP);
        return Ok(());
    }
    let mut cases = cases::built_in()?;
    if let Some(n) = args.cases {
        cases.truncate(n);
    }

    println!("Tutor test {}", env!("CARGO_PKG_VERSION"));
    println!("Press Ctrl+C to stop; a report of what finished is still written.\n");

    let cancel = Arc::new(AtomicBool::new(false));
    {
        let cancel = Arc::clone(&cancel);
        tokio::spawn(async move {
            if tokio::signal::ctrl_c().await.is_ok() {
                cancel.store(true, Ordering::SeqCst);
                eprintln!("\nStopping after the current step…");
            }
        });
    }

    let runner = Runner::start(Arc::clone(&cancel))?;
    if args.list {
        return runner.list().await;
    }

    let out_dir = report_dir(args.out.as_deref())?;
    let machine = system::machine();
    println!(
        "Computer: {}, {:.0} GB memory{}",
        machine.cpu,
        machine.ram_gb,
        if machine.gpus.is_empty() {
            String::new()
        } else {
            format!(", {}", machine.gpus.join(", "))
        }
    );

    let mut run = Run {
        tool_version: env!("CARGO_PKG_VERSION").into(),
        started: now(),
        finished: String::new(),
        machine,
        accelerators: runner.accelerators().await,
        models: Vec::new(),
        stopped: None,
    };

    let mut downloaded = Vec::new();
    'models: for alias in &args.models {
        let variants = match runner.variants(alias, args.all_variants).await {
            Ok(v) => v,
            Err(e) => {
                println!("\n== {alias}: {e}");
                run.models.push(results::ModelRun {
                    alias: alias.clone(),
                    variant: String::new(),
                    device: String::new(),
                    execution_provider: String::new(),
                    size_mb: None,
                    download_s: None,
                    load_s: None,
                    warmup_s: None,
                    hints: Vec::new(),
                    error: Some(e.to_string()),
                    too_slow: None,
                });
                continue;
            }
        };
        let mut queue: std::collections::VecDeque<_> = variants.into();
        while let Some(variant) = queue.pop_front() {
            if cancel.load(Ordering::SeqCst) {
                break 'models;
            }
            let result = runner
                .run_model(
                    alias,
                    std::sync::Arc::clone(&variant),
                    &cases,
                    &mut downloaded,
                )
                .await;
            let failed_device = result.device.clone();
            let fall_back = result.too_slow.is_some()
                || result
                    .error
                    .as_deref()
                    .is_some_and(runner::worth_another_device);
            run.models.push(result);
            // Save before unloading: a stopped or crashed run still leaves a report.
            write_report(&out_dir, &run, &cases)?;
            runner.unload(&variant).await;

            // A version that failed to load or was too slow here: try the next
            // device (AI chip -> graphics -> processor).
            let tried: Vec<String> = run.models.iter().map(|m| m.variant.clone()).collect();
            if fall_back
                && !args.all_variants
                && let Some(next) = runner.fallback_variant(alias, &failed_device, &tried).await
            {
                let device = next
                    .info()
                    .ok()
                    .and_then(|i| i.runtime)
                    .map(|r| r.device_type);
                let name = match device {
                    Some(foundry_local_sdk::DeviceType::GPU) => "graphics (GPU)",
                    _ => "processor (CPU)",
                };
                println!("   Trying its {name} version instead.");
                queue.push_back(next);
            }
        }
    }
    if cancel.load(Ordering::SeqCst) {
        run.stopped = Some("Stopped with Ctrl+C.".into());
    }
    run.finished = now();
    let page = write_report(&out_dir, &run, &cases)?;
    println!("\nReport: {}", page.display());
    if args.open
        && interactive()
        && let Err(e) = opener::open(&page)
    {
        println!("(Couldn't open it automatically: {e}. Open the file above in a browser.)");
    }

    cleanup(&runner, &downloaded, &args.cleanup).await;
    Ok(())
}

/// The report folder: given, else `reports/` next to the program, else in the home folder.
fn report_dir(given: Option<&Path>) -> Result<PathBuf> {
    let stamp = chrono::Local::now()
        .format("tutor-test-%Y%m%d-%H%M%S")
        .to_string();
    if let Some(dir) = given {
        let dir = dir.join(&stamp);
        std::fs::create_dir_all(&dir)
            .with_context(|| format!("couldn't create {}", dir.display()))?;
        return Ok(dir);
    }
    let candidates = system::exe_dir()
        .map(|d| d.join("reports"))
        .into_iter()
        .chain(std::iter::once(
            system::home().join("Documents").join("Tutor test reports"),
        ))
        .chain(std::iter::once(system::home().join("Tutor test reports")));
    for base in candidates {
        let dir = base.join(&stamp);
        if std::fs::create_dir_all(&dir).is_ok() {
            return Ok(dir);
        }
    }
    anyhow::bail!("couldn't find a folder to write the report in; try --out")
}

fn write_report(dir: &Path, run: &Run, cases: &[cases::Case]) -> Result<PathBuf> {
    let json = serde_json::to_string_pretty(run)?;
    std::fs::write(dir.join("results.json"), json).context("couldn't write results.json")?;
    let page = dir.join("report.html");
    std::fs::write(&page, report::render(run, cases)).context("couldn't write report.html")?;
    Ok(page)
}

async fn cleanup(runner: &Runner, downloaded: &[Arc<foundry_local_sdk::Model>], mode: &Cleanup) {
    if downloaded.is_empty() {
        return;
    }
    let delete = match mode {
        Cleanup::Delete => true,
        Cleanup::Keep => false,
        Cleanup::Ask if interactive() => {
            let size: u64 = downloaded
                .iter()
                .filter_map(|m| m.info().ok().and_then(|i| i.file_size_mb))
                .sum();
            ask(&format!(
                "\nDelete the models this test downloaded ({:.1} GB in {})? [y/N] ",
                size as f64 / 1024.0,
                runner.cache_dir().display()
            )) == "y"
        }
        Cleanup::Ask => false,
    };
    if delete {
        runner.remove(downloaded).await;
    } else {
        println!(
            "Models kept in {} (run again with --cleanup to delete them).",
            runner.cache_dir().display()
        );
    }
}
