//! Rough automatic checks on a hint. They flag likely problems; the person
//! reading the report makes the real judgement.

use serde::Serialize;

use crate::cases::Case;

/// Hints longer than this are too much for a beginner to read mid-exercise.
pub const MAX_WORDS: usize = 110;

#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
pub struct Checks {
    /// Mentions at least one of the case's key words.
    pub on_topic: bool,
    /// Contains the corrected line (the answer, not a hint).
    pub gave_answer: bool,
    /// Contains a fenced code block despite being told not to.
    pub code_block: bool,
    pub words: usize,
    pub too_long: bool,
}

impl Checks {
    pub fn passed(&self) -> bool {
        self.on_topic && !self.gave_answer && !self.code_block && !self.too_long
    }
}

fn squash(s: &str) -> String {
    s.split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
        .to_lowercase()
}

pub fn check(case: &Case, hint: &str) -> Checks {
    let lower = hint.to_lowercase();
    let on_topic = case
        .mentions
        .iter()
        .map(|m| m.trim().to_lowercase())
        .filter(|m| !m.is_empty())
        .any(|m| lower.contains(&m));

    // The fixed line only counts as a giveaway if it isn't already in the
    // learner's code (e.g. an indentation fix leaves the same text).
    let answer = squash(&case.giveaway);
    let gave_answer = !squash(case.code()).contains(&answer) && squash(hint).contains(&answer);

    let words = hint.split_whitespace().count();
    Checks {
        on_topic,
        gave_answer,
        code_block: hint.contains("```"),
        words,
        too_long: words > MAX_WORDS,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cases::built_in;

    fn case(id: &str) -> Case {
        built_in()
            .unwrap()
            .into_iter()
            .find(|c| c.id == id)
            .unwrap()
    }

    #[test]
    fn good_hint_passes() {
        let c = case("missing-colon");
        let checks = check(
            &c,
            "Look at line 2. Python expects every if line to end with a colon. What is missing at the end?",
        );
        assert!(checks.on_topic);
        assert!(!checks.gave_answer);
        assert!(checks.passed());
    }

    #[test]
    fn answer_is_caught_despite_spacing_and_case() {
        let c = case("missing-colon");
        let checks = check(&c, "Change it to `IF   temperature > 30:` and it works.");
        assert!(checks.gave_answer);
        assert!(!checks.passed());
    }

    #[test]
    fn indentation_fix_matching_original_text_is_not_a_giveaway() {
        let c = case("indentation");
        let checks = check(
            &c,
            "Line 3, print(fruit), needs to be indented so it is inside the loop.",
        );
        assert!(!checks.gave_answer);
        assert!(checks.on_topic);
    }

    #[test]
    fn off_topic_code_block_and_length_are_flagged() {
        let c = case("print-not-return");
        let long = "word ".repeat(MAX_WORDS + 1);
        let checks = check(&c, &format!("```\nx\n```\n{long}"));
        assert!(!checks.on_topic);
        assert!(checks.code_block);
        assert!(checks.too_long);
        assert!(!checks.passed());
    }
}
