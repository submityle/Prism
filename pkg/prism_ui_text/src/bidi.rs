//! A deliberately small subset of the Unicode Bidirectional Algorithm (UAX#9).
//!
//! # Scope and limitations
//!
//! This module implements only the pieces the Loom text stack needs to lay out
//! predominantly single-direction paragraphs with occasional opposite-direction
//! words:
//!
//! * **Base direction detection** following rules P2/P3: the paragraph takes
//!   the direction of its first strong character, defaulting to left-to-right.
//! * **Strong-type level assignment**: strong left-to-right characters take an
//!   even embedding level and strong right-to-left characters take an odd one.
//! * **Neutral / weak inheritance**: characters without an intrinsic strong
//!   direction (spaces, digits, punctuation) inherit the level of the preceding
//!   strong character, or the base level at the start of the paragraph.
//! * **Run coalescing**: adjacent characters sharing a level are merged into a
//!   single [`Run`].
//!
//! The following UAX#9 features are **not** implemented and are explicitly out
//! of scope: explicit formatting codes and isolates (LRE/RLE/LRO/RLO/PDF/LRI/
//! RLI/FSI/PDI), the paired-bracket algorithm (rule N0), the full weak-type
//! rules (W1–W7), the full neutral rules (N1–N2) and the reordering step (L1–
//! L4). Callers needing complete bidi behaviour should pre-process text with a
//! full implementation; the hook points here (per-character direction and
//! coalesced level runs) are intended to interoperate with the theme-level RTL
//! flag rather than to replace a conformant engine.

use alloc::vec::Vec;

/// A resolved writing direction.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Direction {
    /// Left-to-right.
    Ltr,
    /// Right-to-left.
    Rtl,
}

/// Returns the intrinsic strong direction of `ch`, if it has one.
///
/// Characters in the Hebrew, Arabic and related right-to-left blocks resolve to
/// [`Direction::Rtl`]; ASCII and other strong left-to-right letters resolve to
/// [`Direction::Ltr`]. Digits, whitespace, punctuation and symbols have no
/// strong direction and return `None`.
#[must_use]
pub fn char_direction(ch: char) -> Option<Direction> {
    let c = ch as u32;
    // Right-to-left blocks (Hebrew, Arabic, Syriac, Thaana, NKo, Samaritan,
    // Mandaic plus the Arabic presentation forms).
    let rtl = (0x0590..=0x05FF).contains(&c)
        || (0x0600..=0x06FF).contains(&c)
        || (0x0700..=0x08FF).contains(&c)
        || (0xFB1D..=0xFDFF).contains(&c)
        || (0xFE70..=0xFEFF).contains(&c);
    if rtl {
        return Some(Direction::Rtl);
    }
    if ch.is_alphabetic() {
        return Some(Direction::Ltr);
    }
    None
}

/// Determines the base paragraph direction of `text` (rules P2/P3).
///
/// Returns the direction of the first strong character, or [`Direction::Ltr`]
/// when the text contains no strong character.
#[must_use]
pub fn base_direction(text: &str) -> Direction {
    for ch in text.chars() {
        if let Some(dir) = char_direction(ch) {
            return dir;
        }
    }
    Direction::Ltr
}

/// A maximal run of characters sharing a single embedding level.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Run {
    /// Byte offset of the first byte of the run.
    pub start: usize,
    /// Byte offset one past the final byte of the run.
    pub end: usize,
    /// Resolved embedding level (even is left-to-right, odd right-to-left).
    pub level: u8,
}

impl Run {
    /// Returns the resolved direction implied by the run's embedding level.
    #[must_use]
    pub fn direction(&self) -> Direction {
        if self.level.is_multiple_of(2) {
            Direction::Ltr
        } else {
            Direction::Rtl
        }
    }

    /// Returns the byte length of the run.
    #[must_use]
    pub fn len_bytes(&self) -> usize {
        self.end - self.start
    }

    /// Returns `true` when the run spans no bytes.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.start == self.end
    }
}

/// Resolves `text` into coalesced embedding-level [`Run`]s under `base`.
///
/// The base level is `0` for a left-to-right paragraph and `1` for a
/// right-to-left one. Each strong character sets the current level (even for
/// left-to-right, odd for right-to-left); neutral and weak characters inherit
/// the current level. Adjacent characters with equal levels are merged.
#[must_use]
pub fn resolve_levels(text: &str, base: Direction) -> Vec<Run> {
    let base_level: u8 = match base {
        Direction::Ltr => 0,
        Direction::Rtl => 1,
    };

    let mut runs: Vec<Run> = Vec::new();
    let mut current = base_level;
    for (offset, ch) in text.char_indices() {
        let level = match char_direction(ch) {
            Some(Direction::Ltr) => 0,
            Some(Direction::Rtl) => 1,
            None => current,
        };
        current = level;
        let end = offset + ch.len_utf8();
        match runs.last_mut() {
            Some(last) if last.level == level => last.end = end,
            _ => runs.push(Run {
                start: offset,
                end,
                level,
            }),
        }
    }
    runs
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn char_direction_classifies_scripts() {
        assert_eq!(char_direction('a'), Some(Direction::Ltr));
        assert_eq!(char_direction('\u{05D0}'), Some(Direction::Rtl)); // Hebrew alef
        assert_eq!(char_direction('\u{0627}'), Some(Direction::Rtl)); // Arabic alef
        assert_eq!(char_direction('1'), None);
        assert_eq!(char_direction(' '), None);
    }

    #[test]
    fn base_direction_uses_first_strong() {
        assert_eq!(base_direction("hello"), Direction::Ltr);
        assert_eq!(base_direction("  \u{05D0}"), Direction::Rtl);
        assert_eq!(base_direction("123"), Direction::Ltr);
        assert_eq!(base_direction(""), Direction::Ltr);
    }

    #[test]
    fn pure_ltr_is_single_run() {
        let runs = resolve_levels("abc", Direction::Ltr);
        assert_eq!(runs.len(), 1);
        assert_eq!(runs[0].direction(), Direction::Ltr);
        assert_eq!(runs[0].start, 0);
        assert_eq!(runs[0].end, 3);
    }

    #[test]
    fn pure_rtl_is_single_run() {
        let text = "\u{05D0}\u{05D1}";
        let runs = resolve_levels(text, Direction::Rtl);
        assert_eq!(runs.len(), 1);
        assert_eq!(runs[0].direction(), Direction::Rtl);
        assert_eq!(runs[0].len_bytes(), text.len());
    }

    #[test]
    fn mixed_text_splits_runs() {
        // "a" (ltr) + Hebrew alef (rtl) + "b" (ltr).
        let text = "a\u{05D0}b";
        let runs = resolve_levels(text, Direction::Ltr);
        assert_eq!(runs.len(), 3);
        assert_eq!(runs[0].direction(), Direction::Ltr);
        assert_eq!(runs[1].direction(), Direction::Rtl);
        assert_eq!(runs[2].direction(), Direction::Ltr);
    }

    #[test]
    fn neutrals_inherit_previous_strong() {
        // Hebrew alef, space, Hebrew bet: the space inherits the RTL level and
        // the whole string collapses to one run.
        let text = "\u{05D0} \u{05D1}";
        let runs = resolve_levels(text, Direction::Rtl);
        assert_eq!(runs.len(), 1);
        assert_eq!(runs[0].direction(), Direction::Rtl);
    }

    #[test]
    fn leading_neutral_takes_base_level() {
        let runs = resolve_levels("  a", Direction::Rtl);
        // Two leading spaces inherit the RTL base, then "a" flips to LTR.
        assert_eq!(runs.len(), 2);
        assert_eq!(runs[0].direction(), Direction::Rtl);
        assert_eq!(runs[1].direction(), Direction::Ltr);
    }
}
