//! A conformant implementation of the Unicode Bidirectional Algorithm (UAX#9).
//!
//! The algorithm is split into focused submodules, each covering one rule
//! group, so the structure mirrors the specification and stays testable in
//! isolation:
//!
//! * [`class`] — `Bidi_Class` lookup and paired-bracket data (BD14–BD16).
//! * [`explicit`] — explicit levels, isolates and run structure (X1–X10, BD9/BD13).
//! * [`weak`] — weak type resolution (W1–W7).
//! * [`neutral`] — paired brackets and neutral runs (N0, N1, N2).
//! * [`implicit`] — implicit level raising (I1, I2).
//! * [`reorder`] — line-level reset and visual reordering (L1, L2).
//!
//! The high-level entry points are [`base_direction`] (rules P2/P3),
//! [`resolve_levels`] (the full pipeline reduced to coalesced [`Run`]s for the
//! Loom text stack) and [`BidiInfo`], which exposes per-character levels and
//! visual reordering for callers that need the complete result.

mod class;
mod explicit;
mod implicit;
mod neutral;
mod reorder;
mod weak;

#[cfg(test)]
mod tests;

use alloc::vec::Vec;

pub use class::{bidi_class, paired_bracket, BidiClass, Bracket, BracketKind};
pub use explicit::MAX_DEPTH;
pub use reorder::reorder_visual;

use class::BidiClass::{AL, L, NSM, R};

/// A resolved writing direction.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Direction {
    /// Left-to-right.
    Ltr,
    /// Right-to-left.
    Rtl,
}

impl Direction {
    /// The base embedding level for a paragraph of this direction.
    #[must_use]
    pub fn base_level(self) -> u8 {
        match self {
            Direction::Ltr => 0,
            Direction::Rtl => 1,
        }
    }

    /// The direction implied by an embedding `level` (even is LTR, odd is RTL).
    #[must_use]
    pub fn from_level(level: u8) -> Self {
        if level.is_multiple_of(2) {
            Direction::Ltr
        } else {
            Direction::Rtl
        }
    }
}

/// Returns the intrinsic strong direction of `ch`, if it has one.
///
/// Strong left-to-right characters resolve to [`Direction::Ltr`]; strong
/// right-to-left and Arabic letters resolve to [`Direction::Rtl`]. Weak and
/// neutral characters (numbers, whitespace, punctuation, symbols) return
/// `None`.
#[must_use]
pub fn char_direction(ch: char) -> Option<Direction> {
    match bidi_class(ch) {
        L => Some(Direction::Ltr),
        R | AL => Some(Direction::Rtl),
        _ => None,
    }
}

/// Determines the base paragraph direction of `text` (rules P2/P3).
///
/// Returns the direction of the first strong character, skipping any text
/// between an isolate initiator and its matching PDI, and defaulting to
/// [`Direction::Ltr`] when no strong character is found.
#[must_use]
pub fn base_direction(text: &str) -> Direction {
    let classes: Vec<BidiClass> = text.chars().map(bidi_class).collect();
    let (matching_pdi, _) = explicit::matching_isolates(&classes);
    let mut i = 0;
    while i < classes.len() {
        match classes[i] {
            L => return Direction::Ltr,
            R | AL => return Direction::Rtl,
            c if c.is_isolate_initiator() => match matching_pdi[i] {
                Some(pdi) => i = pdi, // resume at the matching PDI (not strong)
                None => break,
            },
            _ => {}
        }
        i += 1;
    }
    Direction::Ltr
}

/// A maximal run of characters sharing a single resolved embedding level.
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
        Direction::from_level(self.level)
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

/// The complete bidi resolution of a single paragraph of text.
///
/// [`BidiInfo::new`] runs the full UAX#9 pipeline (explicit levels, weak,
/// neutral, implicit and the L1 reset) for a given base direction and retains
/// per-character embedding levels plus the original classes so callers can
/// reorder lines ([`BidiInfo::reorder_visual`]) or coalesce level runs
/// ([`BidiInfo::runs`]).
pub struct BidiInfo {
    /// The paragraph (base) embedding level.
    pub para_level: u8,
    /// Final embedding level for every character, in logical order.
    pub levels: Vec<u8>,
    /// The original `Bidi_Class` of every character (pre-resolution).
    pub original_classes: Vec<BidiClass>,
    /// Byte offset of each character plus a trailing sentinel equal to the
    /// text length (`byte_offsets.len() == levels.len() + 1`).
    byte_offsets: Vec<usize>,
}

impl BidiInfo {
    /// Resolves `text` under the explicit base `direction`.
    #[must_use]
    pub fn new(text: &str, direction: Direction) -> Self {
        let chars: Vec<char> = text.chars().collect();
        let n = chars.len();

        let mut byte_offsets = Vec::with_capacity(n + 1);
        for (offset, _) in text.char_indices() {
            byte_offsets.push(offset);
        }
        byte_offsets.push(text.len());

        let original_classes: Vec<BidiClass> = chars.iter().map(|&c| bidi_class(c)).collect();
        let para_level = direction.base_level();

        let exp = explicit::resolve_explicit(para_level, &original_classes);
        let mut levels = exp.levels.clone();
        let orig_nsm: Vec<bool> = original_classes.iter().map(|&c| c == NSM).collect();

        for seq in explicit::isolating_run_sequences(para_level, &exp) {
            let mut seq_classes: Vec<BidiClass> =
                seq.indices.iter().map(|&i| exp.classes[i]).collect();
            let seq_chars: Vec<char> = seq.indices.iter().map(|&i| chars[i]).collect();
            let seq_nsm: Vec<bool> = seq.indices.iter().map(|&i| orig_nsm[i]).collect();
            let mut seq_levels: Vec<u8> = seq.indices.iter().map(|&i| exp.levels[i]).collect();

            weak::resolve(&mut seq_classes, seq.sos);
            neutral::resolve(&seq_chars, &mut seq_classes, &seq_nsm, seq.level, seq.sos, seq.eos);
            implicit::resolve(&seq_classes, &mut seq_levels);

            for (k, &idx) in seq.indices.iter().enumerate() {
                levels[idx] = seq_levels[k];
            }
        }

        reorder::apply_l1(&original_classes, &mut levels, para_level);

        BidiInfo {
            para_level,
            levels,
            original_classes,
            byte_offsets,
        }
    }

    /// The resolved embedding level of each character, in logical order.
    #[must_use]
    pub fn levels(&self) -> &[u8] {
        &self.levels
    }

    /// The base paragraph direction.
    #[must_use]
    pub fn base_direction(&self) -> Direction {
        Direction::from_level(self.para_level)
    }

    /// Coalesces adjacent characters of equal level into byte-ranged [`Run`]s.
    #[must_use]
    pub fn runs(&self) -> Vec<Run> {
        let mut runs: Vec<Run> = Vec::new();
        for (k, &level) in self.levels.iter().enumerate() {
            let start = self.byte_offsets[k];
            let end = self.byte_offsets[k + 1];
            match runs.last_mut() {
                Some(last) if last.level == level => last.end = end,
                _ => runs.push(Run { start, end, level }),
            }
        }
        runs
    }

    /// The visual (display) order of character indices for this paragraph (L2).
    #[must_use]
    pub fn reorder_visual(&self) -> Vec<usize> {
        reorder_visual(&self.levels)
    }
}

/// Resolves `text` into coalesced embedding-level [`Run`]s under `base`.
///
/// This runs the full UAX#9 pipeline (explicit levels and isolates, the weak
/// rules W1–W7, the paired-bracket rule N0, the neutral rules N1–N2, the
/// implicit rules I1–I2 and the L1 reset) and then merges adjacent characters
/// that share a resolved embedding level. Even levels are left-to-right and odd
/// levels right-to-left.
#[must_use]
pub fn resolve_levels(text: &str, base: Direction) -> Vec<Run> {
    BidiInfo::new(text, base).runs()
}
