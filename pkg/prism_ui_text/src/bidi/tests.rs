//! Tests for the UAX#9 bidirectional implementation.
//!
//! The suite combines the behavioural guarantees inherited from the previous
//! minimal implementation (base-direction detection, strong-type level
//! assignment, run coalescing) with conformance checks that exercise the parts
//! added for full UAX#9 support: explicit isolates (X/BD rules), the weak rules
//! (W1–W7), the paired-bracket rule (N0), the neutral rules (N1/N2), implicit
//! level raising (I1/I2) and visual reordering (L1/L2).

use super::*;

// -- Characters used across the suite -----------------------------------------
const ALEF: char = '\u{05D0}'; // Hebrew letter alef (class R)
const ARABIC_ALEF: char = '\u{0627}'; // Arabic letter alef (class AL)
const LRI: char = '\u{2066}'; // left-to-right isolate
const RLI: char = '\u{2067}'; // right-to-left isolate
const FSI: char = '\u{2068}'; // first-strong isolate
const PDI: char = '\u{2069}'; // pop directional isolate

// =============================================================================
// Behaviour preserved from the previous minimal implementation
// =============================================================================

#[test]
fn char_direction_classifies_scripts() {
    assert_eq!(char_direction('a'), Some(Direction::Ltr));
    assert_eq!(char_direction(ALEF), Some(Direction::Rtl));
    assert_eq!(char_direction(ARABIC_ALEF), Some(Direction::Rtl));
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
    // Hebrew alef, space, Hebrew bet: the space sits between two R characters
    // (N1) so the whole string collapses to one RTL run.
    let text = "\u{05D0} \u{05D1}";
    let runs = resolve_levels(text, Direction::Rtl);
    assert_eq!(runs.len(), 1);
    assert_eq!(runs[0].direction(), Direction::Rtl);
}

#[test]
fn leading_neutral_takes_base_level() {
    let runs = resolve_levels("  a", Direction::Rtl);
    // Two leading spaces keep the RTL base level (N2 → embedding direction, and
    // L1 does not reset leading whitespace), then "a" is raised to an LTR level.
    assert_eq!(runs.len(), 2);
    assert_eq!(runs[0].direction(), Direction::Rtl);
    assert_eq!(runs[0].start, 0);
    assert_eq!(runs[1].direction(), Direction::Ltr);
    assert_eq!(runs[1].end, runs.last().unwrap().end);
}

// =============================================================================
// Level assignment and run structure (I1/I2 raising)
// =============================================================================

#[test]
fn ltr_paragraph_raises_embedded_rtl_word() {
    // "ab" + alef + bet + "cd" under an LTR base: the two Hebrew letters are
    // raised to level 1, producing exactly three runs L/R/L that tile the text.
    let text = "ab\u{05D0}\u{05D1}cd";
    let info = BidiInfo::new(text, Direction::Ltr);
    assert_eq!(info.levels(), &[0, 0, 1, 1, 0, 0]);

    let runs = info.runs();
    assert_eq!(runs.len(), 3);
    assert_eq!(runs[0].start, 0);
    assert_eq!(runs[0].direction(), Direction::Ltr);
    assert_eq!(runs[1].direction(), Direction::Rtl);
    assert_eq!(runs[2].direction(), Direction::Ltr);
    assert_eq!(runs.last().unwrap().end, text.len());
}

#[test]
fn rtl_paragraph_raises_embedded_ltr_word() {
    // alef + "ab" + bet under an RTL base: "ab" is raised to an even level 2
    // (still LTR) while the Hebrew stays at level 1.
    let text = "\u{05D0}ab\u{05D1}";
    let info = BidiInfo::new(text, Direction::Rtl);
    assert_eq!(info.levels(), &[1, 2, 2, 1]);
    assert_eq!(info.base_direction(), Direction::Rtl);
}

// =============================================================================
// Weak rules (W1–W7)
// =============================================================================

#[test]
fn w2_w3_european_number_after_arabic_becomes_arabic() {
    // Arabic alef (AL) followed by "5" (EN). W2 turns EN into AN because the
    // last strong type is AL; W3 turns AL into R. Under I rules the AN is raised
    // to an even level above the surrounding RTL letter.
    let text = "\u{0627}5";
    let info = BidiInfo::new(text, Direction::Rtl);
    assert_eq!(info.levels(), &[1, 2]);

    let runs = info.runs();
    assert_eq!(runs.len(), 2);
    assert_eq!(runs[0].direction(), Direction::Rtl);
    assert_eq!(runs[1].direction(), Direction::Ltr);
}

#[test]
fn w7_european_number_after_latin_resolves_ltr() {
    // "a" (L) then "5" (EN): W7 turns EN into L, so the whole string is one LTR
    // run at level 0.
    let text = "a5";
    let info = BidiInfo::new(text, Direction::Ltr);
    assert_eq!(info.levels(), &[0, 0]);
    assert_eq!(info.runs().len(), 1);
}

// =============================================================================
// Paired brackets (N0)
// =============================================================================

#[test]
fn paired_bracket_canonical_fold() {
    // U+2329 / U+232A fold to the canonical U+3008 / U+3009 pair, so an opening
    // angle bracket of either spelling is recognised.
    let open = paired_bracket('\u{2329}').expect("opening bracket");
    assert_eq!(open.kind, BracketKind::Open);
    let plain = paired_bracket('(').expect("opening paren");
    assert_eq!(plain.kind, BracketKind::Open);
    assert!(paired_bracket('a').is_none());
}

#[test]
fn n0_brackets_follow_matching_outer_context() {
    // LTR base, "a[alef]b": the strong R inside the brackets is opposite the
    // embedding direction, but the established context before "[" is LTR, so N0
    // resolves the brackets to the embedding (LTR) direction. Only the Hebrew
    // letter is raised to level 1.
    let text = "a[\u{05D0}]b";
    let info = BidiInfo::new(text, Direction::Ltr);
    assert_eq!(info.levels(), &[0, 0, 1, 0, 0]);
    assert_eq!(info.runs().len(), 3);
}

#[test]
fn n0_brackets_follow_opposite_context() {
    // LTR base, "alef[bet]": the context before "[" is RTL and the bracket pair
    // encloses RTL text, so N0 resolves the brackets to RTL. Everything is
    // raised to level 1 and coalesces into a single RTL run.
    let text = "\u{05D0}[\u{05D1}]";
    let info = BidiInfo::new(text, Direction::Ltr);
    assert_eq!(info.levels(), &[1, 1, 1, 1]);
    let runs = info.runs();
    assert_eq!(runs.len(), 1);
    assert_eq!(runs[0].direction(), Direction::Rtl);
}

// =============================================================================
// Explicit isolates (X rules, BD9/BD13) and base direction (P2/P3)
// =============================================================================

#[test]
fn base_direction_skips_isolated_runs() {
    // FSI..PDI wraps a Hebrew letter; P2/P3 skips the isolated span and settles
    // on the following strong LTR character.
    let text = alloc::format!("{FSI}{ALEF}{PDI}a");
    assert_eq!(base_direction(&text), Direction::Ltr);

    // LRI..PDI wrapping the only strong character leaves no strong character
    // outside the isolate, so the paragraph defaults to LTR.
    let only_isolated = alloc::format!("{LRI}{ALEF}{PDI}");
    assert_eq!(base_direction(&only_isolated), Direction::Ltr);
}

#[test]
fn rtl_isolate_keeps_inner_run_rtl() {
    // "a" + RLI + alef + PDI + "b" under LTR base: the isolate forms its own
    // level-1 run while the surrounding text and the isolate markers stay at
    // level 0, yielding three runs L/R/L.
    let text = alloc::format!("a{RLI}{ALEF}{PDI}b");
    let info = BidiInfo::new(&text, Direction::Ltr);
    assert_eq!(info.levels(), &[0, 0, 1, 0, 0]);

    let runs = info.runs();
    assert_eq!(runs.len(), 3);
    assert_eq!(runs[0].direction(), Direction::Ltr);
    assert_eq!(runs[1].direction(), Direction::Rtl);
    assert_eq!(runs[2].direction(), Direction::Ltr);
    assert_eq!(runs.last().unwrap().end, text.len());
}

// =============================================================================
// Visual reordering (L2)
// =============================================================================

#[test]
fn reorder_visual_reverses_rtl_runs() {
    // Levels 0,1,1,0: the contiguous level-1 run is reversed for display while
    // the surrounding level-0 characters keep their logical order.
    let info = BidiInfo::new("ab\u{05D0}\u{05D1}cd", Direction::Ltr);
    assert_eq!(info.reorder_visual(), [0, 1, 3, 2, 4, 5]);
}

#[test]
fn reorder_visual_pure_rtl_is_fully_reversed() {
    let info = BidiInfo::new("\u{05D0}\u{05D1}\u{05D2}", Direction::Rtl);
    assert_eq!(info.reorder_visual(), [2, 1, 0]);
}

#[test]
fn reorder_visual_identity_for_ltr() {
    let info = BidiInfo::new("abc", Direction::Ltr);
    assert_eq!(info.reorder_visual(), [0, 1, 2]);
}

// =============================================================================
// Degenerate inputs
// =============================================================================

#[test]
fn empty_text_has_no_runs() {
    let info = BidiInfo::new("", Direction::Ltr);
    assert!(info.levels().is_empty());
    assert!(info.runs().is_empty());
    assert!(info.reorder_visual().is_empty());
    assert!(resolve_levels("", Direction::Rtl).is_empty());
}
