//! Explicit levels, isolates and run structure: UAX#9 rules X1–X10.
//!
//! [`resolve_explicit`] implements the directional-status-stack machine of
//! rules X1–X8 (embeddings, overrides and isolates with a maximum depth of
//! [`MAX_DEPTH`] and overflow accounting), records which characters are removed
//! by X9, and matches isolate initiators to their PDIs (BD9). [`isolating_run_sequences`]
//! then groups the resulting level runs into isolating run sequences (BD13) and
//! computes the `sos`/`eos` boundary types each sequence needs (X10).

use super::class::BidiClass::{self, *};
use alloc::vec;
use alloc::vec::Vec;

/// Maximum explicit embedding depth (`max_depth` in UAX#9).
pub const MAX_DEPTH: u8 = 125;

/// Directional override status carried on the status stack (X2–X6).
#[derive(Clone, Copy, PartialEq, Eq)]
enum Override {
    Neutral,
    Ltr,
    Rtl,
}

/// One entry of the directional status stack (X1).
#[derive(Clone, Copy)]
struct Status {
    level: u8,
    over: Override,
    isolate: bool,
}

/// Result of the explicit-level phase (X1–X9) plus isolate matching (BD9).
pub struct Explicit {
    /// Embedding level assigned to every original character.
    pub levels: Vec<u8>,
    /// Working class per character after override application (X6).
    pub classes: Vec<BidiClass>,
    /// Whether each character is removed by X9 (`RLE LRE RLO LRO PDF BN`).
    pub removed: Vec<bool>,
    /// For each isolate initiator, the index of its matching PDI, if any (BD9).
    pub matching_pdi: Vec<Option<usize>>,
    /// For each PDI, the index of its matching isolate initiator, if any (BD9).
    pub matching_initiator: Vec<Option<usize>>,
}

/// Least odd level strictly greater than `level`.
fn next_odd(level: u8) -> u8 {
    (level + 1) | 1
}

/// Least even level strictly greater than `level`.
fn next_even(level: u8) -> u8 {
    (level + 2) & !1
}

/// Matches isolate initiators (`LRI RLI FSI`) with their `PDI`s (BD9).
///
/// Returns `(matching_pdi, matching_initiator)` indexed by original position.
#[must_use]
pub fn matching_isolates(classes: &[BidiClass]) -> (Vec<Option<usize>>, Vec<Option<usize>>) {
    let n = classes.len();
    let mut matching_pdi = vec![None; n];
    let mut matching_initiator = vec![None; n];
    let mut stack: Vec<usize> = Vec::new();
    for (i, &c) in classes.iter().enumerate() {
        if c.is_isolate_initiator() {
            stack.push(i);
        } else if c == PDI
            && let Some(open) = stack.pop()
        {
            matching_pdi[open] = Some(i);
            matching_initiator[i] = Some(open);
        }
    }
    (matching_pdi, matching_initiator)
}

/// Resolves the direction of a First Strong Isolate (X5c) by applying P2/P3 to
/// the characters it isolates, skipping any nested isolates.
fn fsi_is_rtl(classes: &[BidiClass], fsi: usize, pdi: Option<usize>, matching_pdi: &[Option<usize>]) -> bool {
    let end = pdi.unwrap_or(classes.len());
    let mut j = fsi + 1;
    while j < end {
        match classes[j] {
            L => return false,
            R | AL => return true,
            c if c.is_isolate_initiator() => {
                j = matching_pdi[j].map_or(end, |p| p + 1);
                continue;
            }
            _ => {}
        }
        j += 1;
    }
    false
}

/// Applies rules X1–X9: computes explicit embedding levels, applies directional
/// overrides to character classes, and flags characters removed by X9.
#[must_use]
pub fn resolve_explicit(para_level: u8, orig: &[BidiClass]) -> Explicit {
    let n = orig.len();
    let (matching_pdi, matching_initiator) = matching_isolates(orig);
    let mut levels = vec![para_level; n];
    let mut classes = orig.to_vec();

    let mut stack: Vec<Status> = vec![Status {
        level: para_level,
        over: Override::Neutral,
        isolate: false,
    }];
    let mut overflow_isolate: usize = 0;
    let mut overflow_embedding: usize = 0;
    let mut valid_isolate: usize = 0;

    let apply_override = |classes: &mut [BidiClass], i: usize, over: Override| match over {
        Override::Ltr => classes[i] = L,
        Override::Rtl => classes[i] = R,
        Override::Neutral => {}
    };

    for i in 0..n {
        match orig[i] {
            // X2–X5: explicit embeddings and overrides.
            RLE | LRE | RLO | LRO => {
                let last = *stack.last().unwrap();
                levels[i] = last.level;
                let is_rtl = matches!(orig[i], RLE | RLO);
                let new_level = if is_rtl { next_odd(last.level) } else { next_even(last.level) };
                if new_level <= MAX_DEPTH && overflow_isolate == 0 && overflow_embedding == 0 {
                    let over = match orig[i] {
                        LRO => Override::Ltr,
                        RLO => Override::Rtl,
                        _ => Override::Neutral,
                    };
                    stack.push(Status {
                        level: new_level,
                        over,
                        isolate: false,
                    });
                } else if overflow_isolate == 0 {
                    overflow_embedding += 1;
                }
            }
            // X5a/X5b/X5c: isolate initiators.
            RLI | LRI | FSI => {
                let last = *stack.last().unwrap();
                levels[i] = last.level;
                apply_override(&mut classes, i, last.over);
                let is_rtl = match orig[i] {
                    RLI => true,
                    LRI => false,
                    _ => fsi_is_rtl(orig, i, matching_pdi[i], &matching_pdi),
                };
                let new_level = if is_rtl { next_odd(last.level) } else { next_even(last.level) };
                if new_level <= MAX_DEPTH && overflow_isolate == 0 && overflow_embedding == 0 {
                    valid_isolate += 1;
                    stack.push(Status {
                        level: new_level,
                        over: Override::Neutral,
                        isolate: true,
                    });
                } else {
                    overflow_isolate += 1;
                }
            }
            // X6a: pop directional isolate.
            PDI => {
                if overflow_isolate > 0 {
                    overflow_isolate -= 1;
                } else if valid_isolate != 0 {
                    overflow_embedding = 0;
                    while !stack.last().unwrap().isolate {
                        stack.pop();
                    }
                    stack.pop();
                    valid_isolate -= 1;
                }
                let last = *stack.last().unwrap();
                levels[i] = last.level;
                apply_override(&mut classes, i, last.over);
            }
            // X7: pop directional embedding/override.
            PDF => {
                if overflow_isolate > 0 {
                    // Inside an overflow isolate: no effect.
                } else if overflow_embedding > 0 {
                    overflow_embedding -= 1;
                } else if !stack.last().unwrap().isolate && stack.len() >= 2 {
                    stack.pop();
                }
                levels[i] = stack.last().unwrap().level;
            }
            // X8: paragraph separator terminates all explicit state.
            B => {
                stack.clear();
                stack.push(Status {
                    level: para_level,
                    over: Override::Neutral,
                    isolate: false,
                });
                overflow_isolate = 0;
                overflow_embedding = 0;
                valid_isolate = 0;
                levels[i] = para_level;
            }
            // X9: boundary neutrals keep the embedding level for display.
            BN => {
                levels[i] = stack.last().unwrap().level;
            }
            // X6: any other character.
            _ => {
                let last = *stack.last().unwrap();
                levels[i] = last.level;
                apply_override(&mut classes, i, last.over);
            }
        }
    }

    let removed = orig.iter().map(|c| c.is_removed_by_x9()).collect();
    Explicit {
        levels,
        classes,
        removed,
        matching_pdi,
        matching_initiator,
    }
}

/// One isolating run sequence (BD13) with its boundary types (X10).
pub struct Sequence {
    /// Original character indices belonging to the sequence, in logical order
    /// (characters removed by X9 are excluded).
    pub indices: Vec<usize>,
    /// Shared embedding level of every character in the sequence.
    pub level: u8,
    /// Start-of-sequence boundary type (`L` or `R`).
    pub sos: BidiClass,
    /// End-of-sequence boundary type (`L` or `R`).
    pub eos: BidiClass,
}

/// Maps an embedding level to the boundary class implied by its parity.
fn level_dir(level: u8) -> BidiClass {
    if level.is_multiple_of(2) { L } else { R }
}

/// Builds isolating run sequences from the explicit-level result (X10 / BD13).
#[must_use]
pub fn isolating_run_sequences(para_level: u8, exp: &Explicit) -> Vec<Sequence> {
    // Positions retained after X9, in logical order.
    let retained: Vec<usize> = (0..exp.levels.len()).filter(|&i| !exp.removed[i]).collect();
    if retained.is_empty() {
        return Vec::new();
    }

    // Level runs: maximal ranges [a, b) into `retained` with equal level.
    let mut runs: Vec<(usize, usize)> = Vec::new();
    let mut a = 0;
    while a < retained.len() {
        let lvl = exp.levels[retained[a]];
        let mut b = a + 1;
        while b < retained.len() && exp.levels[retained[b]] == lvl {
            b += 1;
        }
        runs.push((a, b));
        a = b;
    }

    // Index level runs by the original index of their first character so a
    // matching PDI can be chained to the run it starts (BD13).
    let run_starting_at = |orig_idx: usize| -> Option<usize> {
        runs.iter().position(|&(ra, _)| retained[ra] == orig_idx)
    };

    let mut sequences = Vec::new();
    let mut consumed = vec![false; runs.len()];
    for r in 0..runs.len() {
        if consumed[r] {
            continue;
        }
        let (ra, _) = runs[r];
        let first = retained[ra];
        // A run beginning with a PDI that matches an initiator continues an
        // earlier sequence rather than starting one.
        if exp.classes[first] == PDI && exp.matching_initiator[first].is_some() {
            continue;
        }

        let mut chain = vec![r];
        consumed[r] = true;
        loop {
            let (_, cb) = runs[*chain.last().unwrap()];
            let last_char = retained[cb - 1];
            if exp.classes[last_char].is_isolate_initiator()
                && let Some(pdi) = exp.matching_pdi[last_char]
                && let Some(next_run) = run_starting_at(pdi)
                && !consumed[next_run]
            {
                consumed[next_run] = true;
                chain.push(next_run);
                continue;
            }
            break;
        }

        let mut indices = Vec::new();
        for &run in &chain {
            let (ca, cb) = runs[run];
            indices.extend_from_slice(&retained[ca..cb]);
        }

        let seq_level = exp.levels[indices[0]];

        // sos: compare with the retained predecessor of the sequence start.
        let first_pos = runs[chain[0]].0;
        let prev_level = if first_pos == 0 {
            para_level
        } else {
            exp.levels[retained[first_pos - 1]]
        };
        let sos = level_dir(seq_level.max(prev_level));

        // eos: compare with the retained successor of the sequence end, or the
        // paragraph level when the sequence ends in an unmatched initiator.
        let last_pos = runs[*chain.last().unwrap()].1 - 1;
        let last_char = retained[last_pos];
        let next_level = if exp.classes[last_char].is_isolate_initiator()
            && exp.matching_pdi[last_char].is_none()
        {
            para_level
        } else if last_pos + 1 < retained.len() {
            exp.levels[retained[last_pos + 1]]
        } else {
            para_level
        };
        let eos = level_dir(seq_level.max(next_level));

        sequences.push(Sequence {
            indices,
            level: seq_level,
            sos,
            eos,
        });
    }

    sequences
}
