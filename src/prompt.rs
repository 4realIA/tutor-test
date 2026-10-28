//! The tutor prompt, as the learning app would send it.

use crate::cases::Case;

pub const SYSTEM: &str = "You are a patient Python tutor inside a learning app for adult beginners. \
Some learners have only written a little HTML before. The learner is stuck on an exercise and asked for a hint.

Give ONE short hint:
- Use plain, friendly English. If you must use a technical word, explain it.
- Say which line the problem is on.
- Explain what Python is complaining about, or why the result is not what the exercise wants.
- Never write code: not a corrected line, not a code block, not the exact text to type. \
You may quote the learner's own code exactly as they wrote it.
- Describe the kind of change, and end with a short question that nudges the learner to spot it.
- At most 4 sentences. No headings, no lists.";

/// One worked example, shown before the real question. Small models copy the
/// shape of an example far more reliably than they follow rules. It uses a
/// mistake that isn't among the test cases.
pub const EXAMPLE_CODE: &str = "pritn(\"Hello\")";
pub const EXAMPLE_OUTPUT: &str = "Traceback (most recent call last):\n  File \"main.py\", line 1, in <module>\n    \
pritn(\"Hello\")\n    ^^^^^\nNameError: name 'pritn' is not defined. Did you mean: 'print'?";
pub const EXAMPLE_HINT: &str = "Look at line 1. Python doesn't recognise the word `pritn`, so it says that \
name isn't defined: it only knows commands that are spelled exactly right. Compare that word, letter by \
letter, with the command you've used before to show text. Which two letters have swapped places?";

/// Asked when a hint gave the answer away; the app would show the rewrite instead.
pub const REWRITE: &str = "That gives the answer away. Rewrite the hint without any code and without \
the corrected line: describe what to change in words and end with a question. At most 4 sentences.";

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Role {
    System,
    User,
    Assistant,
}

/// The conversation sent for one hint: rules, the worked example, the question.
pub fn conversation(case: &crate::cases::Case) -> Vec<(Role, String)> {
    vec![
        (Role::System, SYSTEM.to_string()),
        (
            Role::User,
            question(
                "Printing text",
                "Print the word Hello.",
                EXAMPLE_CODE,
                EXAMPLE_OUTPUT,
            ),
        ),
        (Role::Assistant, EXAMPLE_HINT.to_string()),
        (Role::User, user_message(case)),
    ]
}

/// Qwen3's switch to answer without thinking first, added to the last question.
pub fn with_no_think(mut turns: Vec<(Role, String)>, no_think: bool) -> Vec<(Role, String)> {
    if no_think && let Some((_, last)) = turns.iter_mut().rev().find(|(r, _)| *r == Role::User) {
        last.push_str("\n/no_think");
    }
    turns
}

/// Split `<think>…</think>` text (some models write it inline) from the hint.
/// Returns (visible hint, thinking).
pub fn split_thinking(text: &str) -> (String, String) {
    let mut visible = String::new();
    let mut thinking = String::new();
    let mut rest = text;
    while let Some(start) = rest.find("<think>") {
        visible.push_str(&rest[..start]);
        let after = &rest[start + "<think>".len()..];
        match after.find("</think>") {
            Some(end) => {
                thinking.push_str(&after[..end]);
                rest = &after[end + "</think>".len()..];
            }
            None => {
                // Cut off mid-thought: nothing visible after it.
                thinking.push_str(after);
                rest = "";
            }
        }
    }
    visible.push_str(rest);
    (visible.trim().to_string(), thinking.trim().to_string())
}

/// The follow-up conversation asking for a rewrite of `first_hint`.
pub fn rewrite_conversation(case: &crate::cases::Case, first_hint: &str) -> Vec<(Role, String)> {
    let mut turns = conversation(case);
    turns.push((Role::Assistant, first_hint.to_string()));
    turns.push((Role::User, REWRITE.to_string()));
    turns
}

/// The learner's code with line numbers, so the tutor can point at a line.
pub fn numbered(code: &str) -> String {
    let lines: Vec<&str> = code.lines().collect();
    let width = lines.len().to_string().len();
    lines
        .iter()
        .enumerate()
        .map(|(i, line)| format!("{:>width$} | {line}", i + 1))
        .collect::<Vec<_>>()
        .join("\n")
}

pub fn user_message(case: &Case) -> String {
    question(&case.lesson, &case.task, case.code(), case.output())
}

fn question(lesson: &str, task: &str, code: &str, output: &str) -> String {
    format!(
        "Lesson: {lesson}\nExercise: {task}\n\nMy code (line numbers added):\n```\n{code}\n```\n\n\
         What happened when I ran it:\n```\n{output}\n```\n\nI'm stuck. Can you give me a hint?",
        code = numbered(code),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cases::built_in;

    #[test]
    fn numbers_lines_aligned() {
        let code = (1..=10)
            .map(|n| format!("x{n}"))
            .collect::<Vec<_>>()
            .join("\n");
        let out = numbered(&code);
        assert!(out.starts_with(" 1 | x1\n"));
        assert!(out.ends_with("10 | x10"));
    }

    #[test]
    fn conversation_has_rules_example_then_question() {
        let case = &built_in().unwrap()[0];
        let turns = conversation(case);
        let roles: Vec<Role> = turns.iter().map(|t| t.0).collect();
        assert_eq!(
            roles,
            [Role::System, Role::User, Role::Assistant, Role::User]
        );
        assert!(turns[1].1.contains("1 | pritn(\"Hello\")"));
        assert!(turns[3].1.contains(&case.task));
    }

    #[test]
    fn example_is_not_a_test_case_and_follows_its_own_rules() {
        let cases = built_in().unwrap();
        assert!(cases.iter().all(|c| !c.code.contains("pritn")));
        assert!(!EXAMPLE_HINT.contains("print(\""));
        assert!(EXAMPLE_HINT.trim_end().ends_with('?'));
    }

    #[test]
    fn rewrite_follows_the_first_hint() {
        let case = &built_in().unwrap()[0];
        let turns = rewrite_conversation(case, "Use `if temperature > 30:`");
        assert_eq!(turns.len(), 6);
        assert_eq!(
            turns[4],
            (Role::Assistant, "Use `if temperature > 30:`".to_string())
        );
        assert_eq!(turns[5].1, REWRITE);
    }

    #[test]
    fn no_think_goes_on_the_last_question_only() {
        let case = &built_in().unwrap()[0];
        let turns = with_no_think(conversation(case), true);
        assert!(turns[3].1.ends_with("\n/no_think"));
        assert!(!turns[1].1.contains("/no_think"));
        let plain = with_no_think(conversation(case), false);
        assert!(plain.iter().all(|t| !t.1.contains("/no_think")));
    }

    #[test]
    fn thinking_split_from_hint() {
        assert_eq!(
            split_thinking("<think>hmm, line 2</think>\n\nLook at line 2."),
            ("Look at line 2.".into(), "hmm, line 2".into())
        );
        assert_eq!(
            split_thinking("<think></think>Hi"),
            ("Hi".into(), "".into())
        );
        assert_eq!(
            split_thinking("<think>never closed"),
            ("".into(), "never closed".into())
        );
        assert_eq!(split_thinking("No tags."), ("No tags.".into(), "".into()));
    }

    #[test]
    fn keeps_indentation() {
        assert_eq!(numbered("if x:\n    y"), "1 | if x:\n2 |     y");
    }

    #[test]
    fn message_has_everything_the_tutor_needs() {
        let case = &built_in().unwrap()[0];
        let msg = user_message(case);
        assert!(msg.contains(&case.lesson));
        assert!(msg.contains(&case.task));
        assert!(msg.contains("2 | if temperature > 30"));
        assert!(msg.contains("SyntaxError: expected ':'"));
        assert!(!msg.contains(&case.giveaway));
    }
}
