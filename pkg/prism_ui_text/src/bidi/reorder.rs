//! Line-level reordering: UAX#9 rules L1 and L2.

use super::class::BidiClass::{self, *};
use alloc::vec::Vec;

/// L1: reset the levels of segment/paragraph separators and of any whitespace or
/// isolate-format characters that trail them or the line to the paragraph level.
///
/// The decision uses the *original* character classes (`orig`), per L1.
pub fn apply_l1(orig: &[BidiClass], levels: &mut [u8], para_level: u8) {
    let n = levels.len();
    // Start of the current resettable run of whitespace / isolate formats, if any.
    let mut reset_from: Option<usize> = None;
    for i in 0..n {
        match orig[i] {
            // L1 (1,2): separators reset, along with any preceding whitespace run.
            B | S => {
                levels[i] = para_level;
                if let Some(start) = reset_from {
                    for level in levels.iter_mut().take(i).skip(start) {
                        *level = para_level;
                    }
                }
                reset_from = None;
            }
            // Whitespace and isolate formatting characters may begin or extend a
            // resettable run (L1 clauses 3 and 4).
            WS | FSI | LRI | RLI | PDI => {
                if reset_from.is_none() {
                    reset_from = Some(i);
                }
            }
            // Characters removed by X9 do not break a whitespace run for L1.
            BN => {}
            _ => reset_from = None,
        }
    }
    // L1 (4): a trailing whitespace / isolate run at end of line.
    if let Some(start) = reset_from {
        for level in levels.iter_mut().skip(start) {
            *level = para_level;
        }
    }
}

/// L2: compute the visual (display) order of character positions from their
/// resolved embedding `levels`, by reversing contiguous runs from the highest
/// level down to the lowest odd level.
#[must_use]
pub fn reorder_visual(levels: &[u8]) -> Vec<usize> {
    let n = levels.len();
    let mut order: Vec<usize> = (0..n).collect();
    if n == 0 {
        return order;
    }
    let max_level = *levels.iter().max().unwrap();
    let min_odd = levels
        .iter()
        .copied()
        .filter(|l| l % 2 == 1)
        .min()
        .unwrap_or(max_level.saturating_add(1));

    let mut lvl = max_level;
    while lvl >= min_odd {
        let mut i = 0;
        while i < n {
            if levels[i] >= lvl {
                let start = i;
                while i < n && levels[i] >= lvl {
                    i += 1;
                }
                order[start..i].reverse();
            } else {
                i += 1;
            }
        }
        if lvl == 0 {
            break;
        }
        lvl -= 1;
    }
    order
}
