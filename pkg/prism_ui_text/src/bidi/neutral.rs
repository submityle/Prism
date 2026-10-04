//! Neutral and paired-bracket resolution: UAX#9 rules N0, N1 and N2.

use super::class::BidiClass::{self, *};
use super::class::{paired_bracket, BracketKind};
use alloc::vec::Vec;

/// Maximum paired-bracket stack depth before BD16 stops pairing.
const BRACKET_STACK_LIMIT: usize = 63;

/// Applies the neutral rules to one isolating run sequence.
///
/// * `chars` are the sequence characters (used by N0 to look up brackets).
/// * `classes` is the working class slice (modified in place).
/// * `orig_nsm[k]` records whether character `k` had class NSM before W1, used
///   for the N0 NSM-after-bracket fix-up.
/// * `level` is the shared embedding level of the sequence; `sos`/`eos` are the
///   boundary types from X10.
pub fn resolve(
    chars: &[char],
    classes: &mut [BidiClass],
    orig_nsm: &[bool],
    level: u8,
    sos: BidiClass,
    eos: BidiClass,
) {
    let e = if level.is_multiple_of(2) { L } else { R };
    n0(chars, classes, orig_nsm, e, sos);
    n1_n2(classes, e, sos, eos);
}

/// The strong direction a class contributes under the neutral rules, where EN
/// and AN count as R (`None` for characters that are not strong here).
fn strong_dir(c: BidiClass) -> Option<BidiClass> {
    match c {
        L => Some(L),
        R | EN | AN => Some(R),
        _ => None,
    }
}

/// N0: resolve paired brackets using the enclosed and preceding strong context.
fn n0(chars: &[char], classes: &mut [BidiClass], orig_nsm: &[bool], e: BidiClass, sos: BidiClass) {
    let o = if e == L { R } else { L };
    for (open, close) in bracket_pairs(chars, classes) {
        let mut found_e = false;
        let mut found_o = false;
        for c in classes.iter().take(close).skip(open + 1) {
            if let Some(d) = strong_dir(*c) {
                if d == e {
                    found_e = true;
                    break;
                }
                found_o = true;
            }
        }

        let resolved = if found_e {
            Some(e)
        } else if found_o {
            // Strong opposite enclosed: adopt it only if the context preceding
            // the opening bracket also establishes that direction.
            if preceding_strong(classes, open, sos) == o {
                Some(o)
            } else {
                Some(e)
            }
        } else {
            None
        };

        if let Some(dir) = resolved {
            classes[open] = dir;
            classes[close] = dir;
            set_following_nsm(classes, orig_nsm, open, dir);
            set_following_nsm(classes, orig_nsm, close, dir);
        }
    }
}

/// Any run of (originally) NSM characters immediately after a bracket that N0
/// changed takes the bracket's resolved direction.
fn set_following_nsm(classes: &mut [BidiClass], orig_nsm: &[bool], pos: usize, dir: BidiClass) {
    let mut k = pos + 1;
    while k < classes.len() && orig_nsm[k] {
        classes[k] = dir;
        k += 1;
    }
}

/// The strong direction (EN/AN as R) preceding `open`, falling back to `sos`.
fn preceding_strong(classes: &[BidiClass], open: usize, sos: BidiClass) -> BidiClass {
    for c in classes.iter().take(open).rev() {
        if let Some(d) = strong_dir(*c) {
            return d;
        }
    }
    sos
}

/// Identifies bracket pairs within the sequence (BD16).
fn bracket_pairs(chars: &[char], classes: &[BidiClass]) -> Vec<(usize, usize)> {
    // Each stack entry stores the canonical representative of the closing
    // bracket expected to match an opener, plus the opener's position.
    let mut stack: Vec<(char, usize)> = Vec::new();
    let mut pairs: Vec<(usize, usize)> = Vec::new();
    for (i, &ch) in chars.iter().enumerate() {
        if classes[i] != ON {
            continue;
        }
        let Some(bracket) = paired_bracket(ch) else {
            continue;
        };
        match bracket.kind {
            BracketKind::Open => {
                if stack.len() == BRACKET_STACK_LIMIT {
                    break; // BD16: abandon pairing on overflow.
                }
                stack.push((bracket.canonical_opposite(), i));
            }
            BracketKind::Close => {
                let want = bracket.canonical;
                if let Some(p) = stack.iter().rposition(|&(exp, _)| exp == want) {
                    pairs.push((stack[p].1, i));
                    stack.truncate(p);
                }
            }
        }
    }
    pairs.sort_unstable_by_key(|&(open, _)| open);
    pairs
}

/// N1 and N2: resolve runs of neutral-or-isolate characters.
fn n1_n2(classes: &mut [BidiClass], e: BidiClass, sos: BidiClass, eos: BidiClass) {
    let n = classes.len();
    let mut i = 0;
    while i < n {
        if !classes[i].is_neutral_or_isolate() {
            i += 1;
            continue;
        }
        let start = i;
        while i < n && classes[i].is_neutral_or_isolate() {
            i += 1;
        }
        let before = if start == 0 { sos } else { boundary_dir(classes[start - 1]) };
        let after = if i == n { eos } else { boundary_dir(classes[i]) };
        // N1: identical surrounding strong directions win; otherwise N2 uses the
        // embedding direction.
        let resolved = if before == after { before } else { e };
        for c in classes.iter_mut().take(i).skip(start) {
            *c = resolved;
        }
    }
}

/// Maps a non-neutral class to the strong direction it presents to a neutral
/// run (EN/AN as R).
fn boundary_dir(c: BidiClass) -> BidiClass {
    match c {
        R | EN | AN => R,
        _ => L,
    }
}
