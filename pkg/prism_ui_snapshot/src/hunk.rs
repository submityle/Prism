//! Unified-diff hunking of serialized snapshots.
//!
//! Where [`crate::diff`] renders every aligned line of two snapshots, this
//! module groups the differences into *hunks* — contiguous spans of change
//! surrounded by a bounded number of unchanged context lines — in the style of
//! the GNU `diff -u` unified format used by Git and by snapshot tools such as
//! `insta`. A large golden file then diffs down to just the regions that moved
//! instead of a full re-dump.
//!
//! [`diff_hunks`] returns the structured [`Hunk`] list and [`format_unified`]
//! renders it as unified-diff text with `@@ -old +new @@` headers. Both are
//! pure and deterministic: identical inputs always yield identical output.
//!
//! The hunks are *applicable* in the unified-diff sense — walking the expected
//! snapshot and, at each hunk, replacing its `Context`/`Delete` lines with its
//! `Context`/`Insert` lines reconstructs the actual snapshot exactly. That
//! round-trip is the module's core correctness invariant.

use alloc::string::{String, ToString};
use alloc::vec::Vec;

use crate::diff::{align, Op};

/// One line inside a [`Hunk`], tagged by which side it belongs to.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum HunkLine {
    /// A line present and identical on both sides (surrounding context).
    Context(String),
    /// A line present only in the expected (old) snapshot.
    Delete(String),
    /// A line present only in the actual (new) snapshot.
    Insert(String),
}

/// A contiguous span of changes plus its surrounding context lines.
///
/// `expected_start` and `actual_start` are the one-based line numbers of the
/// hunk's first line on each side, matching the numbers in a unified-diff `@@`
/// header. When a side contributes no lines (a pure insertion or deletion) its
/// length is zero and its start is the line number it follows, exactly as GNU
/// `diff` reports.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Hunk {
    /// One-based first expected line, or the line a pure insertion follows.
    pub expected_start: usize,
    /// Number of expected-side lines (`Context` + `Delete`) in the hunk.
    pub expected_len: usize,
    /// One-based first actual line, or the line a pure deletion follows.
    pub actual_start: usize,
    /// Number of actual-side lines (`Context` + `Insert`) in the hunk.
    pub actual_len: usize,
    /// The hunk body, in order.
    pub lines: Vec<HunkLine>,
}

/// A single aligned entry carrying its content and both line numbers.
struct Entry {
    line: HunkLine,
    exp: Option<usize>,
    act: Option<usize>,
}

impl Entry {
    fn is_change(&self) -> bool {
        !matches!(self.line, HunkLine::Context(_))
    }
}

/// Aligns the two snapshots and annotates every line with its role and its
/// one-based line number on each side.
fn build_entries(expected: &str, actual: &str) -> Vec<Entry> {
    let expected_lines: Vec<&str> = expected.lines().collect();
    let actual_lines: Vec<&str> = actual.lines().collect();
    let ops = align(&expected_lines, &actual_lines);

    let mut entries = Vec::with_capacity(ops.len());
    let mut exp_no = 0usize;
    let mut act_no = 0usize;
    for op in ops {
        match op {
            Op::Equal(i) => {
                exp_no += 1;
                act_no += 1;
                entries.push(Entry {
                    line: HunkLine::Context(expected_lines[i].to_string()),
                    exp: Some(exp_no),
                    act: Some(act_no),
                });
            }
            Op::Delete(i) => {
                exp_no += 1;
                entries.push(Entry {
                    line: HunkLine::Delete(expected_lines[i].to_string()),
                    exp: Some(exp_no),
                    act: None,
                });
            }
            Op::Insert(j) => {
                act_no += 1;
                entries.push(Entry {
                    line: HunkLine::Insert(actual_lines[j].to_string()),
                    exp: None,
                    act: Some(act_no),
                });
            }
        }
    }
    entries
}

/// Scans the entries before `lo` for the nearest line number on one side,
/// returning `0` when there is none (the change precedes the first line).
fn preceding_number(entries: &[Entry], lo: usize, pick: impl Fn(&Entry) -> Option<usize>) -> usize {
    entries[..lo].iter().rev().find_map(pick).unwrap_or(0)
}

/// Builds one [`Hunk`] from the inclusive entry range `lo..=hi`.
fn build_hunk(entries: &[Entry], lo: usize, hi: usize) -> Hunk {
    let slice = &entries[lo..=hi];
    let expected_len = slice.iter().filter(|e| e.exp.is_some()).count();
    let actual_len = slice.iter().filter(|e| e.act.is_some()).count();
    let expected_start = slice
        .iter()
        .find_map(|e| e.exp)
        .unwrap_or_else(|| preceding_number(entries, lo, |e| e.exp));
    let actual_start = slice
        .iter()
        .find_map(|e| e.act)
        .unwrap_or_else(|| preceding_number(entries, lo, |e| e.act));
    let lines = slice.iter().map(|e| e.line.clone()).collect();
    Hunk {
        expected_start,
        expected_len,
        actual_start,
        actual_len,
        lines,
    }
}

/// Computes the unified-diff hunks between two serialized snapshots.
///
/// `context` is the number of unchanged lines kept around each change. Change
/// windows that overlap or abut after expansion are merged into a single hunk,
/// so no context line is ever duplicated or split. Identical inputs produce no
/// hunks.
#[must_use]
pub fn diff_hunks(expected: &str, actual: &str, context: usize) -> Vec<Hunk> {
    let entries = build_entries(expected, actual);
    let change_indices: Vec<usize> = entries
        .iter()
        .enumerate()
        .filter(|(_, e)| e.is_change())
        .map(|(i, _)| i)
        .collect();
    if change_indices.is_empty() {
        return Vec::new();
    }

    let last = entries.len() - 1;
    let mut ranges: Vec<(usize, usize)> = Vec::new();
    for idx in change_indices {
        let lo = idx.saturating_sub(context);
        let hi = (idx + context).min(last);
        match ranges.last_mut() {
            Some(range) if lo <= range.1 + 1 => {
                if hi > range.1 {
                    range.1 = hi;
                }
            }
            _ => ranges.push((lo, hi)),
        }
    }

    ranges
        .into_iter()
        .map(|(lo, hi)| build_hunk(&entries, lo, hi))
        .collect()
}

/// Formats a unified-diff range field: `start,len`, or just `start` when
/// `len` is 1 (matching GNU `diff` output).
fn format_range(start: usize, len: usize) -> String {
    if len == 1 {
        return start.to_string();
    }
    let mut out = start.to_string();
    out.push(',');
    out.push_str(&len.to_string());
    out
}

/// Renders the unified-diff text between two serialized snapshots.
///
/// Each hunk is introduced by an `@@ -old +new @@` header and its body lines are
/// prefixed with a space (`Context`), `-` (`Delete`), or `+` (`Insert`). The
/// raw serialized line content is preserved verbatim, including indentation, so
/// the output can be applied back onto the expected snapshot. Identical inputs
/// produce the empty string.
#[must_use]
pub fn format_unified(expected: &str, actual: &str, context: usize) -> String {
    let hunks = diff_hunks(expected, actual, context);
    let mut out = String::new();
    for hunk in &hunks {
        out.push_str("@@ -");
        out.push_str(&format_range(hunk.expected_start, hunk.expected_len));
        out.push_str(" +");
        out.push_str(&format_range(hunk.actual_start, hunk.actual_len));
        out.push_str(" @@\n");
        for line in &hunk.lines {
            let (sign, content) = match line {
                HunkLine::Context(s) => (' ', s),
                HunkLine::Delete(s) => ('-', s),
                HunkLine::Insert(s) => ('+', s),
            };
            out.push(sign);
            out.push_str(content);
            out.push('\n');
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::{diff_hunks, format_unified, HunkLine};
    use alloc::string::{String, ToString};
    use alloc::vec::Vec;

    /// Reconstructs the actual snapshot by applying the hunks onto `expected`,
    /// the canonical unified-diff "patch" operation. This is the oracle: for any
    /// inputs and any context, the result must equal `actual`'s lines.
    fn apply(expected: &str, hunks: &[super::Hunk]) -> Vec<String> {
        let exp: Vec<&str> = expected.lines().collect();
        let mut out: Vec<String> = Vec::new();
        let mut cursor = 0usize;
        for hunk in hunks {
            let target = if hunk.expected_len > 0 {
                hunk.expected_start - 1
            } else {
                hunk.expected_start
            };
            while cursor < target {
                out.push(exp[cursor].to_string());
                cursor += 1;
            }
            for line in &hunk.lines {
                match line {
                    HunkLine::Context(s) => {
                        out.push(s.clone());
                        cursor += 1;
                    }
                    HunkLine::Delete(_) => {
                        cursor += 1;
                    }
                    HunkLine::Insert(s) => {
                        out.push(s.clone());
                    }
                }
            }
        }
        while cursor < exp.len() {
            out.push(exp[cursor].to_string());
            cursor += 1;
        }
        out
    }

    fn lines_of(text: &str) -> Vec<String> {
        text.lines().map(ToString::to_string).collect()
    }

    #[test]
    fn identical_inputs_produce_no_hunks() {
        let text = "a\nb\nc";
        assert!(diff_hunks(text, text, 3).is_empty());
        assert_eq!(format_unified(text, text, 3), "");
    }

    #[test]
    fn single_change_with_context() {
        let expected = "a\nb\nc";
        let actual = "a\nB\nc";
        assert_eq!(
            format_unified(expected, actual, 1),
            "@@ -1,3 +1,3 @@\n a\n-b\n+B\n c\n",
        );
    }

    #[test]
    fn zero_context_trims_to_the_change() {
        let expected = "a\nb\nc";
        let actual = "a\nB\nc";
        assert_eq!(format_unified(expected, actual, 0), "@@ -2 +2 @@\n-b\n+B\n");
    }

    #[test]
    fn pure_insertion_reports_zero_length_old_side() {
        let expected = "a";
        let actual = "a\nb";
        assert_eq!(format_unified(expected, actual, 0), "@@ -1,0 +2 @@\n+b\n");
    }

    #[test]
    fn pure_deletion_reports_zero_length_new_side() {
        let expected = "a\nb";
        let actual = "a";
        assert_eq!(format_unified(expected, actual, 0), "@@ -2 +1,0 @@\n-b\n");
    }

    #[test]
    fn distant_changes_split_into_separate_hunks() {
        let expected = "a\nb\nc\nd\ne\nf\ng";
        let actual = "A\nb\nc\nd\ne\nf\nG";
        let hunks = diff_hunks(expected, actual, 1);
        assert_eq!(hunks.len(), 2);
        assert_eq!(apply(expected, &hunks), lines_of(actual));
    }

    #[test]
    fn nearby_changes_merge_into_one_hunk() {
        let expected = "a\nb\nc\nd\ne";
        let actual = "A\nb\nc\nd\nE";
        // With context 2 the two edits' windows cover the whole file and merge.
        let hunks = diff_hunks(expected, actual, 2);
        assert_eq!(hunks.len(), 1);
        assert_eq!(apply(expected, &hunks), lines_of(actual));
    }

    /// A tiny deterministic PRNG so the fuzz oracle is reproducible and needs no
    /// external crate (keeping the test `no_std`/`alloc`-only).
    struct Lcg(u64);

    impl Lcg {
        fn roll(&mut self) -> u64 {
            self.0 = self
                .0
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            self.0 >> 33
        }

        fn below(&mut self, n: u64) -> u64 {
            self.roll() % n
        }
    }

    fn random_text(rng: &mut Lcg) -> String {
        let count = rng.below(6);
        let mut parts: Vec<&str> = Vec::new();
        let mut produced = 0u64;
        while produced < count {
            let sym = match rng.below(4) {
                0 => "a",
                1 => "b",
                2 => "c",
                _ => "d",
            };
            parts.push(sym);
            produced += 1;
        }
        parts.join("\n")
    }

    #[test]
    fn apply_reconstructs_actual_for_fuzzed_pairs() {
        let mut rng = Lcg(0x1234_5678_9abc_def0);
        let contexts = [0usize, 1, 2, 3, 64];
        let mut iterations = 0u64;
        while iterations < 2000 {
            let expected = random_text(&mut rng);
            let actual = random_text(&mut rng);
            for context in contexts {
                let hunks = diff_hunks(&expected, &actual, context);
                assert_eq!(
                    apply(&expected, &hunks),
                    lines_of(&actual),
                    "reconstruction failed for context {context}",
                );
            }
            iterations += 1;
        }
    }

    #[test]
    fn hunks_preserve_change_lines_only_as_edits() {
        let expected = "keep\nold\nkeep2";
        let actual = "keep\nnew\nkeep2";
        let hunks = diff_hunks(expected, actual, 0);
        assert_eq!(hunks.len(), 1);
        let hunk = &hunks[0];
        assert!(hunk
            .lines
            .iter()
            .any(|l| matches!(l, HunkLine::Delete(s) if s == "old")));
        assert!(hunk
            .lines
            .iter()
            .any(|l| matches!(l, HunkLine::Insert(s) if s == "new")));
    }
}
