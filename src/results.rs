//! What one run measured, written to results.json and drawn in the report.

use serde::{Deserialize, Serialize};

use crate::score::Checks;

#[derive(Debug, Clone, Default, Serialize)]
pub struct Machine {
    pub os: String,
    pub cpu: String,
    pub cores: usize,
    pub ram_gb: f64,
    pub gpus: Vec<String>,
}

#[derive(Debug, Clone, Default, Serialize)]
pub struct Accelerators {
    /// Execution providers registered (e.g. OpenVINO, DML). Empty means CPU only.
    pub registered: Vec<String>,
    pub failed: Vec<String>,
    pub note: Option<String>,
}

#[derive(Debug, Clone, Serialize)]
pub struct HintResult {
    pub case_id: String,
    pub text: String,
    pub first_token_s: Option<f64>,
    pub total_s: f64,
    /// Output tokens (from the engine when it reports them, else streamed pieces).
    pub tokens: u64,
    pub checks: Option<Checks>,
    pub error: Option<String>,
    /// When the first hint gave the answer away, it is kept here and `text`
    /// holds the rewrite the app would show instead.
    pub first_text: Option<String>,
    pub first_checks: Option<Checks>,
    /// Extra time the rewrite took (on top of `total_s`).
    pub rewrite_s: Option<f64>,
    pub rewrite_error: Option<String>,
    /// Words the model "thought" before the first hint (time counted in total_s).
    pub thinking_words: u64,
}

impl HintResult {
    pub fn rewritten(&self) -> bool {
        self.first_text.is_some()
    }

    /// Time until the learner sees the hint they'd actually get.
    pub fn shown_after_s(&self) -> f64 {
        self.total_s + self.rewrite_s.unwrap_or(0.0)
    }

    pub fn ok(&self) -> bool {
        self.error.is_none()
    }

    pub fn tokens_per_s(&self) -> Option<f64> {
        let first = self.first_token_s?;
        let gen_time = self.total_s - first;
        (self.tokens > 1 && gen_time > 0.0).then(|| (self.tokens - 1) as f64 / gen_time)
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct ModelRun {
    pub alias: String,
    pub variant: String,
    pub device: String,
    pub execution_provider: String,
    pub size_mb: Option<u64>,
    pub download_s: Option<f64>,
    pub load_s: Option<f64>,
    pub warmup_s: Option<f64>,
    pub hints: Vec<HintResult>,
    pub error: Option<String>,
    /// Set when the model was too slow here and its remaining cases were skipped.
    pub too_slow: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Summary {
    pub answered: usize,
    pub failed: usize,
    /// Hints passing the checks as the learner would see them (after any rewrite).
    pub passed_checks: usize,
    /// Hints passing the checks on the first try.
    pub passed_first: usize,
    pub rewrites: usize,
    pub median_first_token_s: Option<f64>,
    pub median_total_s: Option<f64>,
    /// Slowest wait for the hint the learner sees, including any rewrite.
    pub slowest_total_s: Option<f64>,
    pub median_tokens_per_s: Option<f64>,
}

pub fn median(values: &mut Vec<f64>) -> Option<f64> {
    values.retain(|v| v.is_finite());
    if values.is_empty() {
        return None;
    }
    values.sort_by(|a, b| a.total_cmp(b));
    let mid = values.len() / 2;
    Some(if values.len().is_multiple_of(2) {
        (values[mid - 1] + values[mid]) / 2.0
    } else {
        values[mid]
    })
}

impl ModelRun {
    pub fn summary(&self) -> Summary {
        let ok: Vec<&HintResult> = self.hints.iter().filter(|h| h.ok()).collect();
        let mut first: Vec<f64> = ok.iter().filter_map(|h| h.first_token_s).collect();
        let mut total: Vec<f64> = ok.iter().map(|h| h.total_s).collect();
        let shown: Vec<f64> = ok.iter().map(|h| h.shown_after_s()).collect();
        let mut tps: Vec<f64> = ok.iter().filter_map(|h| h.tokens_per_s()).collect();
        Summary {
            answered: ok.len(),
            failed: self.hints.len() - ok.len(),
            passed_checks: ok
                .iter()
                .filter(|h| h.checks.as_ref().is_some_and(Checks::passed))
                .count(),
            passed_first: ok
                .iter()
                .filter(|h| {
                    h.first_checks
                        .as_ref()
                        .or(h.checks.as_ref())
                        .is_some_and(Checks::passed)
                })
                .count(),
            rewrites: ok.iter().filter(|h| h.rewritten()).count(),
            median_first_token_s: median(&mut first),
            median_total_s: median(&mut total),
            slowest_total_s: shown.iter().copied().reduce(f64::max),
            median_tokens_per_s: median(&mut tps),
        }
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct Run {
    pub tool_version: String,
    pub started: String,
    pub finished: String,
    pub machine: Machine,
    pub accelerators: Accelerators,
    pub models: Vec<ModelRun>,
    /// Set when the run was stopped early (Ctrl+C) or hit a fatal error.
    pub stopped: Option<String>,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn hint(first: Option<f64>, total: f64, tokens: u64, error: bool) -> HintResult {
        HintResult {
            case_id: "c".into(),
            text: String::new(),
            first_token_s: first,
            total_s: total,
            tokens,
            checks: None,
            error: error.then(|| "boom".into()),
            first_text: None,
            first_checks: None,
            rewrite_s: None,
            rewrite_error: None,
            thinking_words: 0,
        }
    }

    #[test]
    fn median_odd_even_empty() {
        assert_eq!(median(&mut vec![3.0, 1.0, 2.0]), Some(2.0));
        assert_eq!(median(&mut vec![4.0, 1.0, 2.0, 3.0]), Some(2.5));
        assert_eq!(median(&mut vec![]), None);
        assert_eq!(median(&mut vec![f64::NAN, 5.0]), Some(5.0));
    }

    #[test]
    fn tokens_per_second_excludes_wait_for_first_token() {
        let h = hint(Some(2.0), 7.0, 51, false);
        assert_eq!(h.tokens_per_s(), Some(10.0));
        assert_eq!(hint(None, 7.0, 51, false).tokens_per_s(), None);
        assert_eq!(hint(Some(2.0), 2.0, 51, false).tokens_per_s(), None);
    }

    #[test]
    fn summary_ignores_failed_hints() {
        let run = ModelRun {
            alias: "m".into(),
            variant: "v".into(),
            device: "GPU".into(),
            execution_provider: "ep".into(),
            size_mb: None,
            download_s: None,
            load_s: None,
            warmup_s: None,
            hints: vec![
                hint(Some(1.0), 3.0, 21, false),
                hint(Some(3.0), 9.0, 61, false),
                hint(None, 120.0, 0, true),
            ],
            error: None,
            too_slow: None,
        };
        let s = run.summary();
        assert_eq!(s.answered, 2);
        assert_eq!(s.failed, 1);
        assert_eq!(s.median_total_s, Some(6.0));
        assert_eq!(s.slowest_total_s, Some(9.0));
        assert_eq!(s.median_first_token_s, Some(2.0));
        assert_eq!(s.rewrites, 0);
    }

    #[test]
    fn rewrite_counts_and_adds_to_the_wait() {
        let good = Checks {
            on_topic: true,
            gave_answer: false,
            code_block: false,
            words: 30,
            too_long: false,
        };
        let bad = Checks {
            gave_answer: true,
            ..good.clone()
        };
        let mut h = hint(Some(1.0), 4.0, 30, false);
        h.first_text = Some("use `x == 1`".into());
        h.first_checks = Some(bad);
        h.checks = Some(good.clone());
        h.rewrite_s = Some(3.0);
        assert!(h.rewritten());
        assert_eq!(h.shown_after_s(), 7.0);
        let mut plain = hint(Some(1.0), 5.0, 30, false);
        plain.checks = Some(good);
        let run = ModelRun {
            alias: "m".into(),
            variant: "v".into(),
            device: "GPU".into(),
            execution_provider: "ep".into(),
            size_mb: None,
            download_s: None,
            load_s: None,
            warmup_s: None,
            hints: vec![h, plain],
            error: None,
            too_slow: None,
        };
        let s = run.summary();
        assert_eq!(s.rewrites, 1);
        assert_eq!(s.passed_checks, 2);
        assert_eq!(s.passed_first, 1);
        assert_eq!(s.slowest_total_s, Some(7.0));
        assert_eq!(s.median_total_s, Some(4.5));
    }
}
