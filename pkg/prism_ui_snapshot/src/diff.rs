//! Human-readable, line-level differences between two serialized snapshots.
//!
//! [`diff`] compares the `expected` and `actual` texts line by line using a
//! longest-common-subsequence alignment, then renders a stable report. Each
//! emitted line is prefixed with a sign (` ` kept, `-` only in expected, `+`
//! only in actual), the one-based line numbers on each side, and the node path
//! derived from the indentation and kind label of the surrounding lines.
//!
//! The function is pure: identical inputs always yield the identical string,
//! and it never allocates nondeterministically or depends on hashing order.

use alloc::string::{String, ToString};
use alloc::vec;
use alloc::vec::Vec;

use crate::escape::unescape;
use crate::serialize::INDENT_UNIT;

/// A single aligned step produced by the diff alignment.
enum Op {
    /// A line present and identical on both sides, at the given expected index.
    Equal(usize),
    /// A line present only in the expected text, at the given index.
    Delete(usize),
    /// A line present only in the actual text, at the given index.
    Insert(usize),
}

/// Produces a human-readable, line-level diff of two serialized snapshots.
///
/// When `expected` and `actual` are identical the result is the empty string.
#[must_use]
pub fn diff(expected: &str, actual: &str) -> String {
    if expected == actual {
        return String::new();
    }
    let expected_lines: Vec<&str> = expected.lines().collect();
    let actual_lines: Vec<&str> = actual.lines().collect();
    let ops = align(&expected_lines, &actual_lines);

    let mut out = String::new();
    let mut expected_no = 0usize;
    let mut actual_no = 0usize;
    let mut expected_path: Vec<String> = Vec::new();
    let mut actual_path: Vec<String> = Vec::new();

    for op in ops {
        match op {
            Op::Equal(i) => {
                expected_no += 1;
                actual_no += 1;
                let (depth, label) = line_label(expected_lines[i]);
                update_path(&mut expected_path, depth, &label);
                update_path(&mut actual_path, depth, &label);
                push_line(
                    &mut out,
                    ' ',
                    Some(expected_no),
                    Some(actual_no),
                    &expected_path,
                    expected_lines[i],
                );
            }
            Op::Delete(i) => {
                expected_no += 1;
                let (depth, label) = line_label(expected_lines[i]);
                update_path(&mut expected_path, depth, &label);
                push_line(
                    &mut out,
                    '-',
                    Some(expected_no),
                    None,
                    &expected_path,
                    expected_lines[i],
                );
            }
            Op::Insert(j) => {
                actual_no += 1;
                let (depth, label) = line_label(actual_lines[j]);
                update_path(&mut actual_path, depth, &label);
                push_line(
                    &mut out,
                    '+',
                    None,
                    Some(actual_no),
                    &actual_path,
                    actual_lines[j],
                );
            }
        }
    }

    out
}

/// Computes a longest-common-subsequence alignment of the two line slices.
fn align(expected: &[&str], actual: &[&str]) -> Vec<Op> {
    let n = expected.len();
    let m = actual.len();
    let mut table = vec![vec![0usize; m + 1]; n + 1];
    let mut i = n;
    while i > 0 {
        i -= 1;
        let mut j = m;
        while j > 0 {
            j -= 1;
            table[i][j] = if expected[i] == actual[j] {
                table[i + 1][j + 1] + 1
            } else {
                let down = table[i + 1][j];
                let right = table[i][j + 1];
                if down >= right {
                    down
                } else {
                    right
                }
            };
        }
    }

    let mut ops = Vec::new();
    let mut i = 0;
    let mut j = 0;
    while i < n && j < m {
        if expected[i] == actual[j] {
            ops.push(Op::Equal(i));
            i += 1;
            j += 1;
        } else if table[i + 1][j] >= table[i][j + 1] {
            ops.push(Op::Delete(i));
            i += 1;
        } else {
            ops.push(Op::Insert(j));
            j += 1;
        }
    }
    while i < n {
        ops.push(Op::Delete(i));
        i += 1;
    }
    while j < m {
        ops.push(Op::Insert(j));
        j += 1;
    }
    ops
}

/// Extracts the depth and a readable kind label from a serialized line.
///
/// Understands both the tree format (`kind=`) and the layout format
/// (`label=`); any other first token is used verbatim.
fn line_label(line: &str) -> (usize, String) {
    let spaces = line.chars().take_while(|&c| c == ' ').count();
    let depth = spaces / INDENT_UNIT;
    let rest = &line[spaces..];
    let first = rest.split(' ').next().unwrap_or("");
    let label = if let Some(value) = first.strip_prefix("kind=") {
        decode_label(value)
    } else if let Some(value) = first.strip_prefix("label=") {
        decode_label(value)
    } else {
        first.to_string()
    };
    (depth, label)
}

/// Unescapes a label token, falling back to the raw token when it is not valid
/// escaped text.
fn decode_label(value: &str) -> String {
    match unescape(value) {
        Some(decoded) => decoded,
        None => value.to_string(),
    }
}

/// Updates `path` so its tail reflects a node at `depth` with `label`.
fn update_path(path: &mut Vec<String>, depth: usize, label: &str) {
    if path.len() > depth {
        path.truncate(depth);
    }
    while path.len() < depth {
        path.push(String::new());
    }
    path.push(label.to_string());
}

/// Appends one formatted diff line to `out`.
fn push_line(
    out: &mut String,
    sign: char,
    expected_no: Option<usize>,
    actual_no: Option<usize>,
    path: &[String],
    content: &str,
) {
    let expected_col = match expected_no {
        Some(n) => n.to_string(),
        None => "-".to_string(),
    };
    let actual_col = match actual_no {
        Some(n) => n.to_string(),
        None => "-".to_string(),
    };
    let trimmed = content.trim_start_matches(' ');
    out.push(sign);
    out.push(' ');
    out.push_str(&expected_col);
    out.push('/');
    out.push_str(&actual_col);
    out.push(' ');
    out.push_str(&path.join(">"));
    out.push_str(" | ");
    out.push_str(trimmed);
    out.push('\n');
}

#[cfg(test)]
mod tests {
    use super::diff;

    #[test]
    fn identical_inputs_produce_empty_diff() {
        let text = "kind=Box text=- classes=\n  kind=Text text=+a classes=\n";
        assert_eq!(diff(text, text), "");
    }

    #[test]
    fn reports_changed_line_with_numbers_and_path() {
        let expected = "kind=Box text=- classes=\n  kind=Text text=+a classes=\n";
        let actual = "kind=Box text=- classes=\n  kind=Text text=+b classes=\n";
        let report = diff(expected, actual);
        assert_eq!(
            report,
            "  1/1 Box | kind=Box text=- classes=\n\
             - 2/- Box>Text | kind=Text text=+a classes=\n\
             + -/2 Box>Text | kind=Text text=+b classes=\n",
        );
    }

    #[test]
    fn reports_added_and_removed_lines() {
        let expected = "kind=Box text=- classes=\n  kind=Text text=+a classes=\n";
        let actual = "kind=Box text=- classes=\n  \
            kind=Text text=+a classes=\n  kind=Text text=+b classes=\n";
        let report = diff(expected, actual);
        // The common lines are kept; the extra actual line is an insertion.
        assert!(report.contains("  1/1 Box | kind=Box text=- classes=\n"));
        assert!(report.contains("  2/2 Box>Text | kind=Text text=+a classes=\n"));
        assert!(report.contains("+ -/3 Box>Text | kind=Text text=+b classes=\n"));
    }
}
