//! Optimal (minimum-raggedness) line breaking.
//!
//! [`wrap_by_width`](crate::line_break::wrap_by_width) fills each line greedily:
//! it is fast but can leave a near-empty last line while every earlier line is
//! packed to the edge, producing ragged paragraphs. [`wrap_optimal`] instead
//! chooses the set of soft breaks that *minimizes total raggedness* — the sum of
//! squared trailing slack over every line except the last line of a paragraph
//! (or sub-paragraph terminated by a mandatory break). This is the same family
//! as TeX's Knuth–Plass line breaker and the balanced wrapping behind
//! `text-wrap: pretty` in modern browsers.
//!
//! The algorithm is a dynamic program over the UAX#14 break opportunities
//! reported by [`break_opportunities`](crate::line_break::break_opportunities):
//!
//! * Boundaries `0..=k` sit at the start of the text and after every break
//!   opportunity, so a candidate line is any range `[i, j)` of boundaries.
//! * A line may not cross a mandatory break (hard breaks stay line boundaries),
//!   and may not overflow `max_width` unless it is a single unbreakable piece
//!   (mirroring the greedy breaker's oversized-piece rule).
//! * The last line of each sub-paragraph is charged zero cost (ragged bottom),
//!   matching how both this breaker and the greedy one leave the final line free.
//!
//! Because greedy wrapping is itself one feasible point in this search space,
//! `wrap_optimal` never produces a *more* ragged paragraph than `wrap_by_width`;
//! [`raggedness`] makes that comparison checkable.

use alloc::vec::Vec;
use unicode_width::{UnicodeWidthChar, UnicodeWidthStr};

use crate::line_break::{break_opportunities, BreakKind, WrappedLine};

/// Sentinel used for "no feasible breaking reaches this boundary yet".
const INF: u128 = u128::MAX;

/// Wraps `text` to `max_width` display columns, minimizing total raggedness.
///
/// Widths are measured with [`unicode_width`](unicode_width), counting one
/// display column as one unit, and trailing whitespace is ignored when scoring a
/// line (it collapses at a soft break). Mandatory breaks always start a new line
/// and a single piece wider than `max_width` is still emitted on its own line
/// rather than dropped, exactly like [`wrap_by_width`](crate::line_break::wrap_by_width).
///
/// Returned lines are contiguous and cover the whole input: concatenating
/// `text[line.start..line.end]` over the result reproduces `text`.
#[must_use]
pub fn wrap_optimal(text: &str, max_width: usize) -> Vec<WrappedLine> {
    let breaks = break_opportunities(text);
    let k = breaks.len();

    // Empty input (no break opportunities) mirrors the greedy breaker: one empty
    // line rather than an empty vector.
    if k == 0 {
        return alloc::vec![WrappedLine {
            start: 0,
            end: 0,
            width: 0,
            hard_break: false,
        }];
    }

    // Byte offset of boundary `t` (0 == text start, `k` == text end).
    let offset = |t: usize| -> usize {
        if t == 0 {
            0
        } else {
            breaks[t - 1].offset
        }
    };
    // Is the break taken at boundary `t` (1..=k) a mandatory one?
    let mandatory = |t: usize| -> bool { t >= 1 && breaks[t - 1].kind == BreakKind::Mandatory };

    // Cumulative raw display widths at each boundary (O(n) total).
    let mut prefix = Vec::with_capacity(k + 1);
    prefix.push(0usize);
    for t in 1..=k {
        let piece = text[offset(t - 1)..offset(t)].width();
        prefix.push(prefix[t - 1] + piece);
    }

    // Width of the trailing-whitespace run ending at each boundary, so a line's
    // scored width is `raw - min(trail, raw)`.
    let mut trail = Vec::with_capacity(k + 1);
    for t in 0..=k {
        let end = offset(t);
        let mut w = 0usize;
        for ch in text[..end].chars().rev() {
            if ch.is_whitespace() {
                w += UnicodeWidthChar::width(ch).unwrap_or(0);
            } else {
                break;
            }
        }
        trail.push(w);
    }

    // `next_mandatory[i]` = smallest boundary strictly greater than `i` that is a
    // mandatory break. Boundary `k` is always mandatory (UAX#14 ends the text
    // with a mandatory break), so this is well defined for every `i < k`.
    let mut next_mandatory = Vec::with_capacity(k + 1);
    next_mandatory.resize(k + 1, k);
    for i in (0..k).rev() {
        next_mandatory[i] = if mandatory(i + 1) {
            i + 1
        } else {
            next_mandatory[i + 1]
        };
    }

    // Dynamic program: `cost[j]` is the minimum raggedness to lay out boundaries
    // `0..j`, breaking at `j`; `parent[j]` records the preceding break.
    let mut cost = Vec::with_capacity(k + 1);
    cost.resize(k + 1, INF);
    let mut parent = alloc::vec![0usize; k + 1];
    cost[0] = 0;

    for j in 1..=k {
        let segment_end = mandatory(j) || j == k;
        for i in 0..j {
            if cost[i] == INF {
                continue;
            }
            // A line may not cross an interior mandatory break. `next_mandatory`
            // is monotonic in `i`, so small `i` are the infeasible ones; skip
            // them and keep scanning toward `j`.
            if next_mandatory[i] < j {
                continue;
            }
            let raw = prefix[j] - prefix[i];
            let scored = raw - trail[j].min(raw);
            let fits = scored <= max_width;
            let single_piece = j == i + 1;
            if !fits && !single_piece {
                continue;
            }
            let line_cost = if segment_end || !fits {
                // Last line of a sub-paragraph is free, and an unavoidable
                // oversized single piece carries no penalty.
                0u128
            } else {
                let slack = (max_width - scored) as u128;
                slack * slack
            };
            let total = cost[i].saturating_add(line_cost);
            if total < cost[j] {
                cost[j] = total;
                parent[j] = i;
            }
        }
    }

    // Reconstruct the chosen lines back to front.
    let mut lines = Vec::new();
    let mut j = k;
    while j > 0 {
        let i = parent[j];
        lines.push(WrappedLine {
            start: offset(i),
            end: offset(j),
            width: prefix[j] - prefix[i],
            hard_break: mandatory(j),
        });
        j = i;
    }
    lines.reverse();
    lines
}

/// Total raggedness of a laid-out paragraph: the sum of squared trailing slack
/// over every line except mandatory (hard-break) lines and the final line, which
/// are free by the ragged-bottom convention.
///
/// This is the exact quantity [`wrap_optimal`] minimizes, so it can score any
/// breaking — for example to confirm `wrap_optimal` is never more ragged than
/// [`wrap_by_width`](crate::line_break::wrap_by_width) on the same input.
#[must_use]
pub fn raggedness(text: &str, lines: &[WrappedLine], max_width: usize) -> u128 {
    let n = lines.len();
    let mut total = 0u128;
    for (idx, line) in lines.iter().enumerate() {
        let is_last = idx + 1 == n;
        if is_last || line.hard_break {
            continue;
        }
        let scored = text[line.start..line.end]
            .trim_end_matches(char::is_whitespace)
            .width();
        if scored < max_width {
            let slack = (max_width - scored) as u128;
            total = total.saturating_add(slack * slack);
        }
    }
    total
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::line_break::wrap_by_width;
    use alloc::string::String;
    use alloc::vec::Vec;

    /// Concatenating the lines must reproduce the input exactly, with contiguous
    /// byte ranges and no gaps or overlaps.
    fn assert_covers(text: &str, lines: &[WrappedLine]) {
        let mut rebuilt = String::new();
        let mut cursor = 0usize;
        for line in lines {
            assert_eq!(line.start, cursor, "lines must be contiguous");
            assert!(line.end >= line.start);
            rebuilt.push_str(&text[line.start..line.end]);
            cursor = line.end;
        }
        assert_eq!(cursor, text.len());
        assert_eq!(rebuilt, text);
    }

    /// No line may overflow `max_width` unless it is a single unbreakable piece.
    fn assert_no_illegal_overflow(text: &str, lines: &[WrappedLine], max_width: usize) {
        for line in lines {
            let scored = text[line.start..line.end]
                .trim_end_matches(char::is_whitespace)
                .width();
            if scored > max_width {
                let inner = break_opportunities(&text[line.start..line.end]);
                // A single piece has exactly one break opportunity: its terminator.
                assert!(
                    inner.len() <= 1,
                    "multi-piece line {:?} overflows width {max_width}",
                    &text[line.start..line.end]
                );
            }
        }
    }

    /// Brute-force the true minimum raggedness by enumerating every subset of the
    /// soft break opportunities (mandatory breaks and the text end are forced).
    fn brute_force_min(text: &str, max_width: usize) -> u128 {
        let breaks = break_opportunities(text);
        let k = breaks.len();
        if k == 0 {
            return 0;
        }
        // Soft boundaries are optional; mandatory boundaries and `k` are forced.
        let soft: Vec<usize> = (1..k)
            .filter(|&t| breaks[t - 1].kind != BreakKind::Mandatory)
            .collect();
        assert!(soft.len() <= 16, "brute force input too large");

        let offset = |t: usize| -> usize {
            if t == 0 {
                0
            } else {
                breaks[t - 1].offset
            }
        };
        let mandatory = |t: usize| breaks[t - 1].kind == BreakKind::Mandatory;

        let mut best = INF;
        for mask in 0u32..(1u32 << soft.len()) {
            // Collect the chosen break boundaries (always including mandatory + k).
            let mut cuts: Vec<usize> = (1..=k).filter(|&t| t == k || mandatory(t)).collect();
            for (bit, &t) in soft.iter().enumerate() {
                if mask & (1 << bit) != 0 {
                    cuts.push(t);
                }
            }
            cuts.sort_unstable();
            cuts.dedup();

            // Build the lines for this breaking and validate / score them.
            let mut lines = Vec::new();
            let mut prev = 0usize;
            for &t in &cuts {
                lines.push(WrappedLine {
                    start: offset(prev),
                    end: offset(t),
                    width: 0,
                    hard_break: mandatory(t),
                });
                prev = t;
            }
            let mut valid = true;
            for line in &lines {
                let scored = text[line.start..line.end]
                    .trim_end_matches(char::is_whitespace)
                    .width();
                let inner = break_opportunities(&text[line.start..line.end]);
                if scored > max_width && inner.len() > 1 {
                    valid = false;
                    break;
                }
            }
            if valid {
                best = best.min(raggedness(text, &lines, max_width));
            }
        }
        best
    }

    #[test]
    fn empty_text_yields_one_empty_line() {
        let lines = wrap_optimal("", 10);
        assert_eq!(lines.len(), 1);
        assert!(lines[0].is_empty());
    }

    #[test]
    fn short_text_fits_on_one_line() {
        let lines = wrap_optimal("hello", 80);
        assert_eq!(lines.len(), 1);
        assert_eq!(&"hello"[lines[0].start..lines[0].end], "hello");
    }

    #[test]
    fn mandatory_breaks_are_preserved() {
        let text = "ab\ncd\nef";
        let lines = wrap_optimal(text, 100);
        assert_eq!(lines.len(), 3);
        assert!(lines[0].hard_break);
        assert!(lines[1].hard_break);
        assert_covers(text, &lines);
    }

    #[test]
    fn oversized_piece_is_kept() {
        let text = "abcdefghij";
        let lines = wrap_optimal(text, 3);
        assert_eq!(lines.len(), 1);
        assert_eq!(lines[0].width, 10);
        assert_covers(text, &lines);
    }

    #[test]
    fn never_more_ragged_than_greedy() {
        let cases = [
            ("the quick brown fox jumps over the lazy dog", 12usize),
            ("the quick brown fox jumps over the lazy dog", 20),
            ("aaaa bb cc dddd ee f gggg hh", 10),
            ("one two three four five six seven eight nine ten", 15),
            ("supercalifragilistic expialidocious tiny a", 8),
            ("a\nbb cc dd ee ff\nggg hhh", 7),
        ];
        for (text, width) in cases {
            let greedy = wrap_by_width(text, width);
            let optimal = wrap_optimal(text, width);
            assert_covers(text, &optimal);
            assert_no_illegal_overflow(text, &optimal, width);
            let gr = raggedness(text, &greedy, width);
            let op = raggedness(text, &optimal, width);
            assert!(
                op <= gr,
                "optimal {op} should not exceed greedy {gr} for {text:?}@{width}"
            );
        }
    }

    #[test]
    fn matches_brute_force_optimum() {
        let cases = [
            ("aaaa bb cc dddd ee f gggg hh", 10usize),
            ("one two three four five six seven", 12),
            ("the quick brown fox jumps", 9),
            ("a bb ccc dddd eeeee", 7),
            ("alpha beta gamma delta epsilon", 11),
            ("x\nyy zz ww\nqqq rr s", 6),
        ];
        for (text, width) in cases {
            let optimal = wrap_optimal(text, width);
            assert_covers(text, &optimal);
            assert_no_illegal_overflow(text, &optimal, width);
            let got = raggedness(text, &optimal, width);
            let best = brute_force_min(text, width);
            assert_eq!(got, best, "non-optimal breaking for {text:?}@{width}");
        }
    }

    #[test]
    fn beats_greedy_on_a_known_case() {
        // Greedy over-fills the first lines and leaves a very short, costly
        // interior line; the optimal breaker rebalances to a lower-or-equal cost.
        let text = "aaaaaa bb cccccc dd";
        let width = 9;
        let greedy = wrap_by_width(text, width);
        let optimal = wrap_optimal(text, width);
        let gr = raggedness(text, &greedy, width);
        let op = raggedness(text, &optimal, width);
        assert!(op <= gr);
        assert_eq!(op, brute_force_min(text, width));
    }
}
