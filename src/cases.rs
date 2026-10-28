//! The beginner mistakes the tutor is tested on, loaded from `cases.toml`.

use std::collections::HashSet;

use anyhow::{Context, Result, bail};
use serde::{Deserialize, Serialize};

const BUILT_IN: &str = include_str!("../cases.toml");

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct Case {
    pub id: String,
    pub title: String,
    pub lesson: String,
    pub task: String,
    pub code: String,
    pub output: String,
    /// Words a useful hint is likely to contain; any one counts.
    pub mentions: Vec<String>,
    /// The corrected line. A hint containing it gave the answer away.
    pub giveaway: String,
}

#[derive(Deserialize)]
struct CaseFile {
    #[serde(rename = "case")]
    cases: Vec<Case>,
}

impl Case {
    /// The learner's code without the blank first/last lines TOML leaves behind.
    pub fn code(&self) -> &str {
        self.code.trim_matches('\n')
    }

    pub fn output(&self) -> &str {
        self.output.trim_matches('\n')
    }
}

/// The cases built into the program.
pub fn built_in() -> Result<Vec<Case>> {
    parse(BUILT_IN).context("the built-in cases.toml is invalid")
}

pub fn parse(text: &str) -> Result<Vec<Case>> {
    // A checkout with Windows line endings must give the same cases.
    let file: CaseFile = toml::from_str(&text.replace("\r\n", "\n"))?;
    validate(&file.cases)?;
    Ok(file.cases)
}

fn validate(cases: &[Case]) -> Result<()> {
    if cases.is_empty() {
        bail!("no cases");
    }
    let mut seen = HashSet::new();
    for case in cases {
        if !seen.insert(case.id.as_str()) {
            bail!("case id {:?} is used twice", case.id);
        }
        for (field, value) in [
            ("id", &case.id),
            ("title", &case.title),
            ("lesson", &case.lesson),
            ("task", &case.task),
            ("code", &case.code),
            ("output", &case.output),
            ("giveaway", &case.giveaway),
        ] {
            if value.trim().is_empty() {
                bail!("case {:?} has an empty {field}", case.id);
            }
        }
        if case.mentions.iter().all(|m| m.trim().is_empty()) {
            bail!("case {:?} has no mentions", case.id);
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn built_in_cases_load() {
        let cases = built_in().unwrap();
        assert_eq!(cases.len(), 15);
        assert!(cases.iter().all(|c| !c.code().starts_with('\n')));
        assert!(cases.iter().all(|c| !c.code().ends_with('\n')));
    }

    #[test]
    fn duplicate_ids_are_refused() {
        let one = r#"
[[case]]
id = "a"
title = "t"
lesson = "l"
task = "k"
code = "x"
output = "o"
mentions = ["m"]
giveaway = "g"
"#;
        let two = format!("{one}{one}");
        assert!(parse(one).is_ok());
        assert!(parse(&two).unwrap_err().to_string().contains("used twice"));
    }

    #[test]
    fn empty_fields_are_refused() {
        let text = r#"
[[case]]
id = "a"
title = "t"
lesson = "l"
task = "k"
code = "   "
output = "o"
mentions = ["m"]
giveaway = "g"
"#;
        assert!(parse(text).unwrap_err().to_string().contains("empty code"));
    }

    #[test]
    fn windows_line_endings_give_the_same_cases() {
        let crlf = BUILT_IN.replace('\n', "\r\n");
        let a = parse(BUILT_IN).unwrap();
        let b = parse(&crlf).unwrap();
        assert!(
            a.iter()
                .zip(&b)
                .all(|(x, y)| x.code == y.code && x.output == y.output)
        );
    }

    #[test]
    fn empty_file_is_refused() {
        assert!(parse("case = []").is_err());
    }
}
