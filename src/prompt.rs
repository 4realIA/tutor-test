//! The tutor prompt, as the learning app would send it.

use crate::cases::Case;

pub const SYSTEM: &str = "You are a patient Python tutor inside a learning app for adult beginners. \
Some learners have only written a little HTML before. The learner is stuck on an exercise and asked for a hint.

Give ONE short hint:
- Use plain, friendly English. If you must use a technical word, explain it.
- Say which line the problem is on.
- Explain what Python is complaining about, or why the result is not what the exercise wants.
- Do NOT write the corrected code or the corrected line. The learner should make the fix.
- At most 4 sentences. No headings, no lists, no code blocks.";

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
    format!(
        "Lesson: {lesson}\nExercise: {task}\n\nMy code (line numbers added):\n```\n{code}\n```\n\n\
         What happened when I ran it:\n```\n{output}\n```\n\nI'm stuck. Can you give me a hint?",
        lesson = case.lesson,
        task = case.task,
        code = numbered(case.code()),
        output = case.output(),
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
