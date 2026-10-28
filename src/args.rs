//! Command-line options. Double-clicking the program uses the defaults.

use std::path::PathBuf;

use anyhow::{Result, bail};

pub const DEFAULT_MODELS: [&str; 1] = ["qwen2.5-coder-7b"];
pub const BIG_MODEL: &str = "qwen2.5-coder-14b";

pub const HELP: &str = "\
Tutor test: how well and how fast local AI models give beginner Python hints.

Usage: tutor-test [options]

  --models a,b,...   Models to test (default: qwen2.5-coder-7b; e.g. add qwen2.5-coder-3b)
  --with-14b         Also test qwen2.5-coder-14b (about 9 GB more, slow)
  --all-variants     Test every version of each model (graphics, processor, AI chip)
  --cases N          Only run the first N cases (quick check)
  --list             List the models in the catalog and exit
  --out DIR          Where to write the report (default: a reports folder next to the program)
  --no-open          Don't open the report in the browser
  --cleanup          Delete the models this run downloaded, without asking
  --keep             Keep the downloaded models, without asking
  -h, --help         Show this help
";

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Cleanup {
    Ask,
    Delete,
    Keep,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Args {
    pub models: Vec<String>,
    pub all_variants: bool,
    pub cases: Option<usize>,
    pub list: bool,
    pub out: Option<PathBuf>,
    pub open: bool,
    pub cleanup: Cleanup,
    pub help: bool,
}

impl Default for Args {
    fn default() -> Self {
        Args {
            models: DEFAULT_MODELS.iter().map(|s| s.to_string()).collect(),
            all_variants: false,
            cases: None,
            list: false,
            out: None,
            open: true,
            cleanup: Cleanup::Ask,
            help: false,
        }
    }
}

pub fn parse<I: IntoIterator<Item = String>>(args: I) -> Result<Args> {
    let mut a = Args::default();
    let mut with_big = false;
    let mut it = args.into_iter();
    while let Some(arg) = it.next() {
        let mut value = |name: &str| -> Result<String> {
            match it.next() {
                Some(v) if !v.starts_with("--") => Ok(v),
                _ => bail!("{name} needs a value"),
            }
        };
        match arg.as_str() {
            "--models" => {
                let list: Vec<String> = value("--models")?
                    .split(',')
                    .map(|s| s.trim().to_string())
                    .filter(|s| !s.is_empty())
                    .collect();
                if list.is_empty() {
                    bail!("--models needs at least one model name");
                }
                a.models = list;
            }
            "--with-14b" => with_big = true,
            "--all-variants" => a.all_variants = true,
            "--cases" => {
                let n: usize = value("--cases")?
                    .parse()
                    .map_err(|_| anyhow::anyhow!("--cases needs a whole number"))?;
                if n == 0 {
                    bail!("--cases must be at least 1");
                }
                a.cases = Some(n);
            }
            "--list" => a.list = true,
            "--out" => a.out = Some(PathBuf::from(value("--out")?)),
            "--no-open" => a.open = false,
            "--cleanup" => a.cleanup = Cleanup::Delete,
            "--keep" => a.cleanup = Cleanup::Keep,
            "-h" | "--help" => a.help = true,
            other => bail!("unknown option {other:?} (try --help)"),
        }
    }
    if with_big && !a.models.iter().any(|m| m == BIG_MODEL) {
        a.models.push(BIG_MODEL.to_string());
    }
    let mut seen = std::collections::HashSet::new();
    a.models.retain(|m| seen.insert(m.clone()));
    Ok(a)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn p(s: &str) -> Result<Args> {
        parse(s.split_whitespace().map(String::from))
    }

    #[test]
    fn defaults() {
        assert_eq!(p("").unwrap(), Args::default());
    }

    #[test]
    fn options() {
        let a = p(
            "--models phi-4-mini,,qwen2.5-1.5b --with-14b --cases 3 --no-open --keep --all-variants",
        )
        .unwrap();
        assert_eq!(a.models, ["phi-4-mini", "qwen2.5-1.5b", BIG_MODEL]);
        assert_eq!(a.cases, Some(3));
        assert!(!a.open);
        assert!(a.all_variants);
        assert_eq!(a.cleanup, Cleanup::Keep);
    }

    #[test]
    fn duplicates_removed() {
        let a = p("--models phi-4-mini,phi-4-mini").unwrap();
        assert_eq!(a.models, ["phi-4-mini"]);
    }

    #[test]
    fn bad_input() {
        assert!(p("--cases 0").is_err());
        assert!(p("--cases x").is_err());
        assert!(p("--cases").is_err());
        assert!(p("--models --keep").is_err());
        assert!(p("--wat").is_err());
    }
}
