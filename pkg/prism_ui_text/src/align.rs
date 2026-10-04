//! Horizontal line placement: alignment, justification and ellipsis truncation.
//!
//! The [`rich_text`](crate::rich_text) layer stores the *policy* for a
//! paragraph ([`Align`] and [`Truncate`]), but a policy only becomes geometry
//! once it is applied to the shaped glyphs of a concrete line. This module is
//! that application step: given a line's [`ShapedGlyph`]s (in visual order,
//! after any bidi reordering) and the available inline width, it computes the
//! pen position of every glyph and, where requested, the number of leading
//! glyphs that survive an ellipsis.
//!
//! The implementation is pure data: it depends only on glyph advances, the
//! requested [`Align`], and the paragraph [`Direction`]. It performs no font
//! loading and uses only the four basic arithmetic operations, so placements
//! are fully reproducible.
//!
//! # Alignment
//!
//! For non-justified lines the whole line keeps its natural width and is offset
//! within the available box:
//!
//! * [`Align::Start`] hugs the leading edge (left for [`Direction::Ltr`], right
//!   for [`Direction::Rtl`]).
//! * [`Align::End`] hugs the trailing edge.
//! * [`Align::Center`] centres the natural width.
//!
//! # Justification
//!
//! [`Align::Justify`] stretches a line to exactly fill the available width by
//! inserting equal extra space at a set of *expansion opportunities*. Following
//! CSS Text, the last line of a justified paragraph (and any line that has no
//! opportunities or already overflows) falls back to [`Align::Start`]. The two
//! supported [`JustifyMode`]s mirror CSS `text-justify`:
//!
//! * [`JustifyMode::InterWord`] expands only at word-separator characters.
//! * [`JustifyMode::InterCharacter`] expands between every glyph, used as the
//!   fallback for scripts without word separators.
//!
//! # Truncation
//!
//! [`truncate_to_width`] implements `text-overflow: ellipsis`: it returns the
//! largest leading glyph prefix whose advance, plus the advance of an ellipsis
//! glyph, fits in the available width.

use alloc::vec::Vec;

use crate::bidi::Direction;
use crate::rich_text::Align;
use crate::shaper::ShapedGlyph;

/// Absolute tolerance (in logical pixels) used when comparing advances.
const EPS: f32 = 1.0e-4;

/// How justification distributes the extra space on a line (CSS `text-justify`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum JustifyMode {
    /// Expand only at word-separator characters (`text-justify: inter-word`).
    #[default]
    InterWord,
    /// Expand between every glyph (`text-justify: inter-character`); also the
    /// fallback when a line has no inter-word opportunities.
    InterCharacter,
}

/// The CSS word-separator characters that provide inter-word justification
/// opportunities (the same set used by the `word-spacing` property).
///
/// `U+0020` SPACE, `U+00A0` NO-BREAK SPACE, `U+1361` ETHIOPIC WORDSPACE,
/// `U+10100`/`U+10101` AEGEAN WORD SEPARATORS and `U+1039F` UGARITIC WORD
/// DIVIDER.
#[must_use]
pub fn is_word_separator(ch: char) -> bool {
    matches!(
        ch,
        '\u{0020}'
            | '\u{00A0}'
            | '\u{1361}'
            | '\u{10100}'
            | '\u{10101}'
            | '\u{1039F}'
    )
}

/// The natural (unjustified) advance width of `glyphs`.
#[must_use]
pub fn line_width(glyphs: &[ShapedGlyph]) -> f32 {
    glyphs.iter().map(|g| g.advance).sum()
}

/// The leading-edge offset of a non-justified line of width `line_width` placed
/// in `available` logical pixels under `align` and paragraph `base_dir`.
///
/// A positive result shifts the line toward the trailing edge; the value may be
/// negative when the line is wider than the box, leaving the overflow for the
/// caller to clip. [`Align::Justify`] is treated as [`Align::Start`] here, since
/// a stretched line's leading edge coincides with the start edge; the stretch
/// itself is applied by [`place_line`].
#[must_use]
pub fn align_offset(line_width: f32, available: f32, align: Align, base_dir: Direction) -> f32 {
    let slack = available - line_width;
    match align {
        Align::Start | Align::Justify => match base_dir {
            Direction::Ltr => 0.0,
            Direction::Rtl => slack,
        },
        Align::End => match base_dir {
            Direction::Ltr => slack,
            Direction::Rtl => 0.0,
        },
        Align::Center => slack * 0.5,
    }
}

/// A line whose glyphs have been assigned absolute pen positions.
#[derive(Clone, Debug, PartialEq)]
pub struct PlacedLine {
    /// Pen position (leading edge) of each glyph, in the same order as the
    /// glyphs passed to [`place_line`].
    pub positions: Vec<f32>,
    /// Leading-edge offset applied to the whole line.
    pub offset: f32,
    /// Occupied inline width: the available width when the line was justified,
    /// otherwise the natural advance width.
    pub width: f32,
    /// `true` when justification actually stretched the line.
    pub justified: bool,
}

/// Places `glyphs` within `available` logical pixels under `align`.
///
/// `opportunities` lists glyph indices after which justification space may be
/// inserted (see [`justification_opportunities`]); it is ignored unless the line
/// is actually justified. A line is justified only when `align` is
/// [`Align::Justify`], it is not the last line of the paragraph (`is_last_line`
/// is `false`), it has at least one opportunity, and it is narrower than the
/// box. Otherwise the glyphs keep their natural advances and the line is offset
/// per [`align_offset`] (with [`Align::Justify`] falling back to
/// [`Align::Start`], matching `text-align-last: auto`).
///
/// The returned [`PlacedLine::positions`] always has the same length as
/// `glyphs` and is non-decreasing.
#[must_use]
pub fn place_line(
    glyphs: &[ShapedGlyph],
    opportunities: &[usize],
    available: f32,
    align: Align,
    base_dir: Direction,
    is_last_line: bool,
) -> PlacedLine {
    let natural = line_width(glyphs);
    let can_justify = align == Align::Justify
        && !is_last_line
        && !opportunities.is_empty()
        && available - natural > EPS;

    if can_justify {
        let per_gap = (available - natural) / opportunities.len() as f32;
        let mut positions = Vec::with_capacity(glyphs.len());
        let mut opp = opportunities.iter().peekable();
        let mut x = 0.0f32;
        for (index, glyph) in glyphs.iter().enumerate() {
            positions.push(x);
            x += glyph.advance;
            while opp.peek().is_some_and(|&&o| o == index) {
                x += per_gap;
                opp.next();
            }
        }
        PlacedLine {
            positions,
            offset: 0.0,
            width: available,
            justified: true,
        }
    } else {
        let effective = if align == Align::Justify {
            Align::Start
        } else {
            align
        };
        let offset = align_offset(natural, available, effective, base_dir);
        let mut positions = Vec::with_capacity(glyphs.len());
        let mut x = offset;
        for glyph in glyphs {
            positions.push(x);
            x += glyph.advance;
        }
        PlacedLine {
            positions,
            offset,
            width: natural,
            justified: false,
        }
    }
}

/// Computes the glyph indices that provide justification opportunities for
/// `glyphs` covering `text`.
///
/// An opportunity index `i` means extra space may be inserted *after*
/// `glyphs[i]`. The final glyph is never an opportunity (there is no gap past
/// the end of the line). For [`JustifyMode::InterWord`] an opportunity is
/// produced for each glyph whose cluster begins with a word separator (see
/// [`is_word_separator`]); for [`JustifyMode::InterCharacter`] every glyph
/// except the last is an opportunity.
#[must_use]
pub fn justification_opportunities(
    text: &str,
    glyphs: &[ShapedGlyph],
    mode: JustifyMode,
) -> Vec<usize> {
    if glyphs.len() < 2 {
        return Vec::new();
    }
    let last = glyphs.len() - 1;
    match mode {
        JustifyMode::InterCharacter => (0..last).collect(),
        JustifyMode::InterWord => {
            let mut out = Vec::new();
            for (index, glyph) in glyphs.iter().enumerate().take(last) {
                if let Some(ch) = text.get(glyph.cluster..).and_then(|s| s.chars().next())
                    && is_word_separator(ch)
                {
                    out.push(index);
                }
            }
            out
        }
    }
}

/// Returns the number of leading glyphs that fit in `available` once room is
/// reserved for an ellipsis of advance `ellipsis_advance` (`text-overflow:
/// ellipsis`).
///
/// When the whole line already fits, the full glyph count is returned and the
/// caller needs no ellipsis (`result == glyphs.len()`). Otherwise the result is
/// the largest prefix length `k` such that the advance of the first `k` glyphs
/// plus `ellipsis_advance` does not exceed `available`; this may be `0` when the
/// ellipsis alone does not fit.
#[must_use]
pub fn truncate_to_width(glyphs: &[ShapedGlyph], available: f32, ellipsis_advance: f32) -> usize {
    if line_width(glyphs) <= available + EPS {
        return glyphs.len();
    }
    let budget = available - ellipsis_advance;
    if budget <= 0.0 {
        return 0;
    }
    let mut x = 0.0f32;
    let mut kept = 0usize;
    for glyph in glyphs {
        let next = x + glyph.advance;
        if next > budget + EPS {
            break;
        }
        x = next;
        kept += 1;
    }
    kept
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec;
    use alloc::vec::Vec;

    fn glyphs(advances: &[f32]) -> Vec<ShapedGlyph> {
        advances
            .iter()
            .enumerate()
            .map(|(i, &a)| ShapedGlyph::new(i, a))
            .collect()
    }

    fn close(a: f32, b: f32) -> bool {
        (a - b).abs() < 1.0e-3
    }

    #[test]
    fn align_offset_follows_direction() {
        // 40px line in a 100px box leaves 60px of slack.
        assert!(close(align_offset(40.0, 100.0, Align::Start, Direction::Ltr), 0.0));
        assert!(close(align_offset(40.0, 100.0, Align::End, Direction::Ltr), 60.0));
        assert!(close(align_offset(40.0, 100.0, Align::Center, Direction::Ltr), 30.0));
        // RTL mirrors start/end.
        assert!(close(align_offset(40.0, 100.0, Align::Start, Direction::Rtl), 60.0));
        assert!(close(align_offset(40.0, 100.0, Align::End, Direction::Rtl), 0.0));
        // Justify's leading edge equals Start's.
        assert!(close(
            align_offset(40.0, 100.0, Align::Justify, Direction::Ltr),
            align_offset(40.0, 100.0, Align::Start, Direction::Ltr),
        ));
    }

    #[test]
    fn align_offset_overflow_is_negative_for_center() {
        // A 120px line in a 100px box overflows; centring pulls it left by 10px.
        assert!(close(align_offset(120.0, 100.0, Align::Center, Direction::Ltr), -10.0));
    }

    #[test]
    fn place_start_lays_glyphs_at_cumulative_advances() {
        let gs = glyphs(&[10.0, 20.0, 30.0]);
        let placed = place_line(&gs, &[], 200.0, Align::Start, Direction::Ltr, false);
        assert_eq!(placed.positions, vec![0.0, 10.0, 30.0]);
        assert!(!placed.justified);
        assert!(close(placed.width, 60.0));
    }

    #[test]
    fn place_center_offsets_every_glyph() {
        let gs = glyphs(&[10.0, 20.0, 30.0]); // natural width 60, slack 40, offset 20
        let placed = place_line(&gs, &[], 100.0, Align::Center, Direction::Ltr, false);
        assert_eq!(placed.positions, vec![20.0, 30.0, 50.0]);
        assert!(close(placed.offset, 20.0));
    }

    #[test]
    fn justify_fills_available_width_exactly() {
        let gs = glyphs(&[10.0, 10.0, 10.0, 10.0]); // natural 40
        let opp = vec![0usize, 1, 2]; // three gaps
        let placed = place_line(&gs, &opp, 100.0, Align::Justify, Direction::Ltr, false);
        assert!(placed.justified);
        // 60px of slack over three gaps => 20px each.
        assert_eq!(placed.positions, vec![0.0, 30.0, 60.0, 90.0]);
        // The right edge of the last glyph reaches the available width.
        let right = placed.positions[placed.positions.len() - 1] + gs[gs.len() - 1].advance;
        assert!(close(right, 100.0));
        assert!(close(placed.width, 100.0));
    }

    #[test]
    fn last_line_of_justified_paragraph_aligns_to_start() {
        let gs = glyphs(&[10.0, 10.0, 10.0]);
        let opp = vec![0usize, 1];
        let placed = place_line(&gs, &opp, 100.0, Align::Justify, Direction::Ltr, true);
        assert!(!placed.justified);
        assert_eq!(placed.positions, vec![0.0, 10.0, 20.0]);
    }

    #[test]
    fn justify_without_opportunities_falls_back_to_start() {
        let gs = glyphs(&[10.0, 10.0]);
        let placed = place_line(&gs, &[], 100.0, Align::Justify, Direction::Ltr, false);
        assert!(!placed.justified);
        assert_eq!(placed.positions, vec![0.0, 10.0]);
    }

    #[test]
    fn justify_overflowing_line_is_not_stretched() {
        let gs = glyphs(&[60.0, 60.0]); // natural 120 > available 100
        let placed = place_line(&gs, &[0], 100.0, Align::Justify, Direction::Ltr, false);
        assert!(!placed.justified);
        assert_eq!(placed.positions, vec![0.0, 60.0]);
    }

    #[test]
    fn inter_word_opportunities_pick_separators_only() {
        // "ab cd" -> glyphs a,b,space,c,d with clusters at byte offsets.
        let text = "ab cd";
        let gs = vec![
            ShapedGlyph::new(0, 8.0), // 'a'
            ShapedGlyph::new(1, 8.0), // 'b'
            ShapedGlyph::new(2, 4.0), // ' '
            ShapedGlyph::new(3, 8.0), // 'c'
            ShapedGlyph::new(4, 8.0), // 'd'
        ];
        let opp = justification_opportunities(text, &gs, JustifyMode::InterWord);
        assert_eq!(opp, vec![2]);
    }

    #[test]
    fn inter_word_recognises_no_break_space() {
        let text = "a\u{00A0}b";
        let gs = vec![
            ShapedGlyph::new(0, 8.0), // 'a'
            ShapedGlyph::new(1, 4.0), // NBSP (2 bytes)
            ShapedGlyph::new(3, 8.0), // 'b'
        ];
        let opp = justification_opportunities(text, &gs, JustifyMode::InterWord);
        assert_eq!(opp, vec![1]);
    }

    #[test]
    fn inter_character_opportunities_exclude_last_glyph() {
        let gs = glyphs(&[8.0, 8.0, 8.0]);
        let opp = justification_opportunities("abc", &gs, JustifyMode::InterCharacter);
        assert_eq!(opp, vec![0, 1]);
    }

    #[test]
    fn truncate_returns_full_count_when_line_fits() {
        let gs = glyphs(&[10.0, 10.0, 10.0]);
        assert_eq!(truncate_to_width(&gs, 100.0, 5.0), gs.len());
    }

    #[test]
    fn truncate_reserves_room_for_the_ellipsis() {
        // Natural width 50 > available 25. Ellipsis is 7px, so budget is 18px:
        // two 10px glyphs (20px) overflow it, one glyph (10px) fits.
        let gs = glyphs(&[10.0, 10.0, 10.0, 10.0, 10.0]);
        let kept = truncate_to_width(&gs, 25.0, 7.0);
        assert_eq!(kept, 1);
        // Boundary property: kept prefix + ellipsis fits; one more does not.
        let prefix: f32 = gs[..kept].iter().map(|g| g.advance).sum();
        let prefix_plus: f32 = gs[..kept + 1].iter().map(|g| g.advance).sum();
        assert!(prefix + 7.0 <= 25.0 + EPS);
        assert!(prefix_plus + 7.0 > 25.0 + EPS);
    }

    #[test]
    fn truncate_can_keep_nothing_when_ellipsis_alone_overflows() {
        let gs = glyphs(&[10.0, 10.0]);
        assert_eq!(truncate_to_width(&gs, 3.0, 5.0), 0);
    }

    /// Dependency-free xorshift32 PRNG for the randomized oracle below.
    struct Rng(u32);

    impl Rng {
        fn next_u32(&mut self) -> u32 {
            let mut x = self.0;
            x ^= x << 13;
            x ^= x >> 17;
            x ^= x << 5;
            self.0 = x;
            x
        }

        fn advance(&mut self) -> f32 {
            // 1.0..=16.0 in quarter-pixel steps, always positive.
            1.0 + (self.next_u32() % 61) as f32 * 0.25
        }
    }

    #[test]
    fn justify_oracle_fills_width_for_random_lines() {
        let mut rng = Rng(0x1234_5678);
        for _ in 0..2000 {
            let count = 2 + (rng.next_u32() % 10) as usize;
            let advs: Vec<f32> = (0..count).map(|_| rng.advance()).collect();
            let gs = glyphs(&advs);
            let natural = line_width(&gs);
            // Opportunities: every interior glyph (inter-character style).
            let opp: Vec<usize> = (0..count - 1).collect();
            // Target width strictly larger than natural so justification engages.
            let available = natural + 1.0 + (rng.next_u32() % 400) as f32 * 0.25;
            let placed = place_line(&gs, &opp, available, Align::Justify, Direction::Ltr, false);
            assert!(placed.justified);
            // Positions are non-decreasing.
            for w in placed.positions.windows(2) {
                assert!(w[1] >= w[0] - EPS);
            }
            // The line fills exactly the available width.
            let right = placed.positions[count - 1] + gs[count - 1].advance;
            assert!(
                (right - available).abs() < 1.0e-2,
                "right={right} available={available}"
            );
        }
    }
}
