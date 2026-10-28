//! The HTML report: one self-contained page, readable in any browser.

use std::fmt::Write as _;

use crate::cases::Case;
use crate::prompt::numbered;
use crate::results::{ModelRun, Run, Summary};

pub fn escape(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for c in s.chars() {
        match c {
            '&' => out.push_str("&amp;"),
            '<' => out.push_str("&lt;"),
            '>' => out.push_str("&gt;"),
            '"' => out.push_str("&quot;"),
            '\'' => out.push_str("&#39;"),
            _ => out.push(c),
        }
    }
    out
}

fn secs(v: Option<f64>) -> String {
    match v {
        Some(s) if s < 10.0 => format!("{s:.1} s"),
        Some(s) if s < 120.0 => format!("{s:.0} s"),
        Some(s) => format!("{:.0} min {:.0} s", (s / 60.0).floor(), s % 60.0),
        None => "–".into(),
    }
}

/// A plain-language speed verdict from the typical time for a whole hint.
pub fn speed_verdict(s: &Summary) -> (&'static str, &'static str) {
    match s.median_total_s {
        None => ("No result", "bad"),
        Some(t) if t <= 8.0 => ("Quick", "good"),
        Some(t) if t <= 20.0 => ("OK (shown word by word)", "ok"),
        Some(_) => ("Too slow for hints", "bad"),
    }
}

const STYLE: &str = r#"
:root{--bg:#fbfaf7;--fg:#1d1c1a;--muted:#6b6862;--card:#fff;--line:#e4e1da;--code:#f3f1ec;
--good:#1f7a4d;--ok:#9a6700;--bad:#b42318;--accent:#2f5bd3}
@media (prefers-color-scheme:dark){:root{--bg:#161615;--fg:#ecebe8;--muted:#a3a09a;--card:#1f1f1d;
--line:#34332f;--code:#262624;--good:#5cc28e;--ok:#e0b34a;--bad:#f07a6e;--accent:#8aa8ff}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--fg);
font:16px/1.55 system-ui,-apple-system,"Segoe UI",sans-serif}
main{max-width:1100px;margin:0 auto;padding:24px 16px 64px}
h1{font-size:28px;margin:0 0 4px}h2{font-size:20px;margin:40px 0 12px}h3{font-size:17px;margin:0 0 8px}
.muted{color:var(--muted)}.card{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:16px;margin:12px 0}
table{border-collapse:collapse;width:100%;font-size:15px}th,td{text-align:left;padding:8px 10px;border-bottom:1px solid var(--line);vertical-align:top}
th{font-weight:600;color:var(--muted);font-size:13px;text-transform:uppercase;letter-spacing:.03em}
.scroll{overflow-x:auto}pre{background:var(--code);border-radius:8px;padding:10px 12px;margin:6px 0 10px;
overflow-x:auto;font:13.5px/1.45 ui-monospace,SFMono-Regular,Consolas,monospace;white-space:pre}
.hint{white-space:pre-wrap;margin:6px 0}.tag{display:inline-block;font-size:12px;font-weight:600;border-radius:999px;
padding:1px 9px;margin:0 6px 4px 0;border:1px solid currentColor}
.good{color:var(--good)}.ok{color:var(--ok)}.bad{color:var(--bad)}
.grid{display:grid;gap:12px;grid-template-columns:repeat(auto-fit,minmax(280px,1fr))}
.model{border-top:3px solid var(--accent);padding-top:8px}.label{font-size:13px;color:var(--muted);margin-top:10px}
summary{cursor:pointer;font-size:13px;margin-top:4px}
"#;

pub fn render(run: &Run, cases: &[Case]) -> String {
    let mut h = String::new();
    let _ = write!(
        h,
        "<!doctype html><html lang=\"en\"><head><meta charset=\"utf-8\">\
         <meta name=\"viewport\" content=\"width=device-width,initial-scale=1\">\
         <title>Tutor test</title><style>{STYLE}</style></head><body><main>"
    );
    let _ = write!(
        h,
        "<h1>Tutor test</h1><p class=\"muted\">How well and how fast each local AI model gives \
         beginner Python hints on this computer. Run {} (tool {}).</p>",
        escape(&run.started),
        escape(&run.tool_version)
    );
    if let Some(stopped) = &run.stopped {
        let _ = write!(
            h,
            "<p class=\"card bad\"><strong>Run stopped early:</strong> {}</p>",
            escape(stopped)
        );
    }

    // Machine
    let m = &run.machine;
    let accel = if run.accelerators.registered.is_empty() {
        "None (processor only)".to_string()
    } else {
        run.accelerators.registered.join(", ")
    };
    let _ = write!(
        h,
        "<h2>This computer</h2><div class=\"card scroll\"><table>\
         <tr><th>System</th><td>{}</td></tr><tr><th>Processor</th><td>{} ({} cores)</td></tr>\
         <tr><th>Memory</th><td>{:.0} GB</td></tr><tr><th>Graphics</th><td>{}</td></tr>\
         <tr><th>AI accelerators set up</th><td>{}{}</td></tr></table></div>",
        escape(&m.os),
        escape(&m.cpu),
        m.cores,
        m.ram_gb,
        if m.gpus.is_empty() {
            "–".into()
        } else {
            escape(&m.gpus.join(", "))
        },
        escape(&accel),
        run.accelerators
            .note
            .as_ref()
            .map(|n| format!("<br><span class=\"muted\">{}</span>", escape(n)))
            .unwrap_or_default(),
    );

    // Summary
    h.push_str(
        "<h2>Results</h2><div class=\"card scroll\"><table><tr><th>Model</th><th>Ran on</th>\
         <th>Speed</th><th>Typical hint</th><th>Slowest†</th><th>First words after</th>\
         <th>Words/s*</th><th>Checks passed: first try</th><th>After rewrite</th><th>Download</th><th>Load</th></tr>",
    );
    for model in &run.models {
        summary_row(&mut h, model);
    }
    h.push_str(
        "</table><p class=\"muted\">Typical = median of all hints. *Tokens per second, roughly \
         words per second. Checks: on topic, didn't give away the fixed line, no code block, \
         under 110 words. When a hint gave the answer away or showed code, the model was asked \
         once to rewrite it (as the app would); \"after rewrite\" counts what the learner would see. \
         †Including any rewrite. The checks are only a rough guide; read the hints below.</p></div>",
    );

    // Each case
    h.push_str("<h2>The hints</h2>");
    for case in cases {
        let _ = write!(
            h,
            "<div class=\"card\"><h3>{}</h3><div class=\"muted\">{} · {}</div>\
             <div class=\"label\">Code</div><pre>{}</pre><div class=\"label\">What Python said</div><pre>{}</pre><div class=\"grid\">",
            escape(&case.title),
            escape(&case.lesson),
            escape(&case.task),
            escape(&numbered(case.code())),
            escape(case.output()),
        );
        for model in &run.models {
            hint_cell(&mut h, model, &case.id);
        }
        h.push_str("</div></div>");
    }
    let _ = write!(
        h,
        "<p class=\"muted\">Finished {}.</p></main></body></html>",
        escape(&run.finished)
    );
    h
}

fn summary_row(h: &mut String, model: &ModelRun) {
    let s = model.summary();
    let (verdict, class) = if model.too_slow.is_some() {
        ("Too slow here", "bad")
    } else if model.error.is_some() && s.answered == 0 {
        ("Failed", "bad")
    } else {
        speed_verdict(&s)
    };
    let error = model
        .too_slow
        .iter()
        .chain(model.error.iter())
        .map(|e| format!("<br><span class=\"bad\">{}</span>", escape(e)))
        .collect::<String>();
    let size = model
        .size_mb
        .map(|mb| format!(" ({:.1} GB)", mb as f64 / 1024.0))
        .unwrap_or_default();
    let _ = write!(
        h,
        "<tr><td><strong>{}</strong><br><span class=\"muted\">{}</span>{}</td>\
         <td>{}<br><span class=\"muted\">{}</span></td><td class=\"{class}\"><strong>{verdict}</strong></td>\
         <td>{}</td><td>{}</td><td>{}</td><td>{}</td><td>{} of {}</td><td>{} of {}{}</td><td>{}{}</td><td>{}</td></tr>",
        escape(&model.alias),
        escape(&model.variant),
        error,
        escape(&model.device),
        escape(&model.execution_provider),
        secs(s.median_total_s),
        secs(s.slowest_total_s),
        secs(s.median_first_token_s),
        s.median_tokens_per_s
            .map(|t| format!("{t:.1}"))
            .unwrap_or("–".into()),
        s.passed_first,
        model.hints.len(),
        s.passed_checks,
        model.hints.len(),
        if s.rewrites > 0 {
            format!("<br><span class=\"muted\">{} rewritten</span>", s.rewrites)
        } else {
            String::new()
        },
        model
            .download_s
            .map(|d| secs(Some(d)))
            .unwrap_or_else(|| "already here".into()),
        size,
        secs(model.load_s),
    );
}

fn hint_cell(h: &mut String, model: &ModelRun, case_id: &str) {
    let _ = write!(
        h,
        "<div class=\"model\"><strong>{}</strong>",
        escape(&model.alias)
    );
    match model.hints.iter().find(|r| r.case_id == case_id) {
        None => h.push_str("<p class=\"muted\">Not run.</p>"),
        Some(r) if r.error.is_some() => {
            let _ = write!(
                h,
                "<p class=\"bad\">{}</p>",
                escape(r.error.as_deref().unwrap_or_default())
            );
        }
        Some(r) => {
            let rewrite = match (r.rewritten(), r.rewrite_s) {
                (true, Some(extra)) => format!(" · rewritten, +{}", secs(Some(extra))),
                _ => String::new(),
            };
            let thought = if r.thinking_words > 0 {
                format!(" · thought {} words first", r.thinking_words)
            } else {
                String::new()
            };
            let _ = write!(
                h,
                "<div class=\"muted\">{} · first words after {}{}{}</div>",
                secs(Some(r.total_s)),
                secs(r.first_token_s),
                thought,
                rewrite
            );
            if let Some(c) = &r.checks {
                let tag = |ok: bool, good: &str, bad: &str| {
                    if ok {
                        format!("<span class=\"tag good\">{good}</span>")
                    } else {
                        format!("<span class=\"tag bad\">{bad}</span>")
                    }
                };
                h.push_str("<div>");
                h.push_str(&tag(c.on_topic, "on topic", "may be off topic"));
                h.push_str(&tag(!c.gave_answer, "hint only", "gave the answer"));
                if c.code_block {
                    h.push_str(&tag(false, "", "code block"));
                }
                h.push_str(&tag(
                    !c.too_long,
                    &format!("{} words", c.words),
                    &format!("{} words: long", c.words),
                ));
                h.push_str("</div>");
            }
            let _ = write!(h, "<p class=\"hint\">{}</p>", escape(r.text.trim()));
            if let Some(first) = &r.first_text {
                let _ = write!(
                    h,
                    "<details><summary class=\"muted\">First try (gave the answer away)</summary>\
                     <p class=\"hint muted\">{}</p></details>",
                    escape(first.trim())
                );
            }
            if let Some(e) = &r.rewrite_error {
                let _ = write!(h, "<p class=\"muted\">Rewrite failed: {}</p>", escape(e));
            }
        }
    }
    h.push_str("</div>");
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cases::built_in;
    use crate::results::{Accelerators, HintResult, Machine};
    use crate::score::check;

    fn sample_run() -> Run {
        let cases = built_in().unwrap();
        let text = "Look at line 2 <b>: an if line needs a colon at the end.".to_string();
        Run {
            tool_version: "0.1.0".into(),
            started: "2026-10-27 19:00".into(),
            finished: "2026-10-27 19:30".into(),
            machine: Machine {
                os: "Windows 11".into(),
                cpu: "Intel Core Ultra 7 155U".into(),
                cores: 14,
                ram_gb: 15.6,
                gpus: vec!["Intel Graphics".into()],
            },
            accelerators: Accelerators::default(),
            models: vec![ModelRun {
                alias: "phi-4-mini".into(),
                variant: "Phi-4-mini-instruct-generic-gpu:5".into(),
                device: "GPU".into(),
                execution_provider: "WebGpuExecutionProvider".into(),
                size_mb: Some(3800),
                download_s: Some(300.0),
                load_s: Some(12.3),
                warmup_s: Some(9.0),
                hints: vec![HintResult {
                    case_id: cases[0].id.clone(),
                    checks: Some(check(&cases[0], &text)),
                    text,
                    first_token_s: Some(1.2),
                    total_s: 6.4,
                    tokens: 40,
                    error: None,
                    first_text: None,
                    first_checks: None,
                    rewrite_s: None,
                    rewrite_error: None,
                    thinking_words: 0,
                }],
                error: None,
                too_slow: None,
            }],
            stopped: None,
        }
    }

    #[test]
    fn report_escapes_model_text_and_includes_results() {
        let html = render(&sample_run(), &built_in().unwrap());
        assert!(html.contains("&lt;b&gt;"));
        assert!(!html.contains("<b>:"));
        assert!(html.contains("phi-4-mini"));
        assert!(html.contains("Quick"));
        assert!(html.contains("Intel Core Ultra 7 155U"));
        assert!(html.contains("None (processor only)"));
        assert!(html.contains("Not run."));
        assert!(html.contains("1 of 1"));
        assert!(!html.contains("First try"));
    }

    #[test]
    fn too_slow_model_is_called_out() {
        let mut run = sample_run();
        run.models[0].too_slow =
            Some("Too slow on this computer: the warm-up answer took 199 s.".into());
        let html = render(&run, &built_in().unwrap());
        assert!(html.contains("Too slow here"));
        assert!(html.contains("warm-up answer took 199 s"));
    }

    #[test]
    fn rewritten_hint_shows_both() {
        let mut run = sample_run();
        let h = &mut run.models[0].hints[0];
        h.first_text = Some("Type `if temperature > 30:` <x>".into());
        h.rewrite_s = Some(2.5);
        let html = render(&run, &built_in().unwrap());
        assert!(html.contains("First try (gave the answer away)"));
        assert!(html.contains("&lt;x&gt;"));
        assert!(html.contains("rewritten, +2.5 s"));
        assert!(html.contains("1 rewritten"));
    }

    #[test]
    fn verdicts_by_typical_time() {
        let mut s = sample_run().models[0].summary();
        assert_eq!(speed_verdict(&s).0, "Quick");
        s.median_total_s = Some(15.0);
        assert_eq!(speed_verdict(&s).1, "ok");
        s.median_total_s = Some(45.0);
        assert_eq!(speed_verdict(&s).1, "bad");
        s.median_total_s = None;
        assert_eq!(speed_verdict(&s).0, "No result");
    }

    #[test]
    fn seconds_format() {
        assert_eq!(secs(Some(3.24)), "3.2 s");
        assert_eq!(secs(Some(42.4)), "42 s");
        assert_eq!(secs(Some(125.0)), "2 min 5 s");
        assert_eq!(secs(None), "–");
    }
}
