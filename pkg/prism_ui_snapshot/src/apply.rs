//! Applying unified-diff hunks back onto a snapshot (the "patch" operation).
//!
//! [`crate::hunk::diff_hunks`] turns two snapshots into a list of [`Hunk`]s;
//! this module performs the inverse. Given the *expected* (old) snapshot and
//! those hunks, [`apply_hunks`] reconstructs the *actual* (new) snapshot by
//! copying unchanged lines, dropping `Delete` lines and emitting `Insert`
//! lines. This is the executable form of the round-trip invariant documented
//! on the hunk module:
//!
//! ```text
//! apply_hunks(expected, diff_hunks(expected, actual, ctx)) == actual   (line-wise)
//! ```
//!
//! Unlike a blind replay, this applier *validates* the patch: every `Context`
//! and `Delete` line must match the expected snapshot at the position the
//! hunk header points at. A patch produced against a different base therefore
//! fails loudly with an [`ApplyError`] instead of silently corrupting output.
//! Reconstruction is line-based (operating on [`str::lines`]), so a trailing
//! newline on the original input is not re-synthesised; callers comparing
//! against a raw string should compare line sequences.

use alloc::string::String;
use alloc::vec::Vec;

use crate::hunk::{Hunk, HunkLine};

/// Why [`apply_hunks`] could not apply a hunk list to a snapshot.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ApplyError {
    /// A hunk points at an expected line at or before the previous hunk's end,
    /// so the hunks are not in strictly increasing, non-overlapping order.
    OutOfOrder {
        /// Index of the offending hunk in the input slice.
        hunk: usize,
    },
    /// A hunk references an expected line beyond the end of the snapshot.
    OutOfBounds {
        /// Index of the offending hunk in the input slice.
        hunk: usize,
    },
    /// A `Context` or `Delete` line did not match the expected snapshot at the
    /// one-based line it was expected at.
    ContextMismatch {
        /// Index of the offending hunk in the input slice.
        hunk: usize,
        /// One-based expected-side line number that failed to match.
        line: usize,
    },
}

/// Reconstructs the actual snapshot by applying `hunks` onto `expected`.
///
/// Returns the reconstructed line sequence joined with `\n`, or an
/// [`ApplyError`] when the hunks do not cleanly apply to `expected` (the base
/// differs, the hunks are mis-ordered, or a header points past the end).
///
/// # Errors
///
/// Returns [`ApplyError`] when a hunk is out of order, references a line beyond
/// the snapshot, or its `Context`/`Delete` content disagrees with the base.
pub fn apply_hunks(expected: &str, hunks: &[Hunk]) -> Result<String, ApplyError> {
    let exp: Vec<&str> = expected.lines().collect();
    let mut out: Vec<&str> = Vec::new();
    let mut cursor = 0usize;

    for (index, hunk) in hunks.iter().enumerate() {
        // The expected line the hunk's body starts at. A pure insertion
        // (`expected_len == 0`) sits *after* `expected_start`, matching GNU
        // `diff` header semantics; a hunk with expected content starts at
        // `expected_start - 1` (zero-based).
        let target = if hunk.expected_len > 0 {
            hunk.expected_start.checked_sub(1).ok_or(ApplyError::OutOfBounds { hunk: index })?
        } else {
            hunk.expected_start
        };

        if target < cursor {
            return Err(ApplyError::OutOfOrder { hunk: index });
        }
        if target > exp.len() {
            return Err(ApplyError::OutOfBounds { hunk: index });
        }

        // Copy untouched lines leading up to the hunk.
        while cursor < target {
            out.push(exp[cursor]);
            cursor += 1;
        }

        for line in &hunk.lines {
            match line {
                HunkLine::Context(s) | HunkLine::Delete(s) => {
                    let base = exp.get(cursor).ok_or(ApplyError::OutOfBounds { hunk: index })?;
                    if *base != s.as_str() {
                        return Err(ApplyError::ContextMismatch {
                            hunk: index,
                            line: cursor + 1,
                        });
                    }
                    if matches!(line, HunkLine::Context(_)) {
                        out.push(base);
                    }
                    cursor += 1;
                }
                HunkLine::Insert(s) => out.push(s.as_str()),
            }
        }
    }

    // Trailing untouched lines after the final hunk.
    while cursor < exp.len() {
        out.push(exp[cursor]);
        cursor += 1;
    }

    Ok(join_lines(&out))
}

/// Joins borrowed lines with `\n` without a trailing newline, matching the
/// line-based representation the rest of the crate operates on.
fn join_lines(lines: &[&str]) -> String {
    let mut out = String::new();
    for (i, line) in lines.iter().enumerate() {
        if i > 0 {
            out.push('\n');
        }
        out.push_str(line);
    }
    out
}


#[cfg(test)]
mod tests {
    use super::{apply_hunks, ApplyError};
    use crate::hunk::{diff_hunks, Hunk, HunkLine};
    use alloc::string::{String, ToString};
    use alloc::vec::Vec;

    struct SplitMix64(u64);
    impl SplitMix64 {
        fn next_u64(&mut self) -> u64 {
            self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut z = self.0;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            z ^ (z >> 31)
        }
        fn below(&mut self, hi: usize) -> usize {
            (self.next_u64() % hi as u64) as usize
        }
    }

    /// The canonical line representation the applier targets.
    fn normalize(text: &str) -> String {
        let lines: Vec<&str> = text.lines().collect();
        lines.join("\n")
    }

    /// Builds a random snapshot from a small alphabet so edits overlap often.
    fn random_text(rng: &mut SplitMix64) -> String {
        let n = rng.below(14);
        let mut lines: Vec<String> = Vec::with_capacity(n);
        for _ in 0..n {
            let token = rng.below(6);
            lines.push(match token {
                0 => "alpha".to_string(),
                1 => "beta".to_string(),
                2 => "gamma".to_string(),
                3 => "delta".to_string(),
                4 => "".to_string(),
                _ => "epsilon".to_string(),
            });
        }
        lines.join("\n")
    }

    #[test]
    fn roundtrip_fixed_cases() {
        let cases = [
            ("a\nb\nc", "a\nB\nc"),
            ("a\nb\nc", "a\nc"),
            ("a\nc", "a\nb\nc"),
            ("", "x\ny"),
            ("x\ny", ""),
            ("one\ntwo\nthree\nfour", "zero\none\ntwo\nfour\nfive"),
        ];
        for (expected, actual) in cases {
            for ctx in 0..=3 {
                let hunks = diff_hunks(expected, actual, ctx);
                let got = apply_hunks(expected, &hunks).unwrap();
                assert_eq!(got, normalize(actual), "ctx={ctx} {expected:?}->{actual:?}");
            }
        }
    }

    #[test]
    fn roundtrip_randomized() {
        let mut rng = SplitMix64(0xDEAD_BEEF_1234_5678);
        for _ in 0..2000 {
            let expected = random_text(&mut rng);
            let actual = random_text(&mut rng);
            let ctx = rng.below(4);
            let hunks = diff_hunks(&expected, &actual, ctx);
            let got = apply_hunks(&expected, &hunks).unwrap();
            assert_eq!(got, normalize(&actual), "ctx={ctx}");
        }
    }

    #[test]
    fn context_mismatch_is_detected() {
        // A hunk that deletes "b", but the base has "x" where "b" is claimed.
        let hunks = alloc::vec![Hunk {
            expected_start: 1,
            expected_len: 1,
            actual_start: 1,
            actual_len: 0,
            lines: alloc::vec![HunkLine::Delete("b".to_string())],
        }];
        let err = apply_hunks("x", &hunks).unwrap_err();
        assert_eq!(err, ApplyError::ContextMismatch { hunk: 0, line: 1 });
    }

    #[test]
    fn out_of_bounds_is_detected() {
        let hunks = alloc::vec![Hunk {
            expected_start: 5,
            expected_len: 1,
            actual_start: 5,
            actual_len: 1,
            lines: alloc::vec![HunkLine::Context("a".to_string())],
        }];
        assert_eq!(
            apply_hunks("a", &hunks).unwrap_err(),
            ApplyError::OutOfBounds { hunk: 0 },
        );
    }

    #[test]
    fn out_of_order_is_detected() {
        // Second hunk points before the first finished.
        let hunks = alloc::vec![
            Hunk {
                expected_start: 2,
                expected_len: 1,
                actual_start: 2,
                actual_len: 1,
                lines: alloc::vec![HunkLine::Context("b".to_string())],
            },
            Hunk {
                expected_start: 1,
                expected_len: 1,
                actual_start: 1,
                actual_len: 1,
                lines: alloc::vec![HunkLine::Context("a".to_string())],
            },
        ];
        assert_eq!(
            apply_hunks("a\nb\nc", &hunks).unwrap_err(),
            ApplyError::OutOfOrder { hunk: 1 },
        );
    }

    #[test]
    fn empty_hunks_returns_normalized_base() {
        assert_eq!(apply_hunks("a\nb\nc", &[]).unwrap(), "a\nb\nc");
    }
}
