//! Fuzzy subsequence matching for command-palette / sidebar story search.
//!
//! This module implements an [`fzf`](https://github.com/junegunn/fzf)-style
//! fuzzy matcher: a short *pattern* matches a longer *candidate* when the
//! pattern's characters appear in the candidate **in order** (as a
//! subsequence), not necessarily contiguously. This is the interaction model
//! behind VS Code's command palette and Storybook's sidebar filter — the user
//! types a few characters and the best structurally-matching entries surface
//! first.
//!
//! # Matching rule
//!
//! Matching is ASCII-case-insensitive: a pattern character matches a candidate
//! character when they are equal after [`char::to_ascii_lowercase`] (non-ASCII
//! characters compare exactly). The pattern matches the candidate iff the
//! pattern is a subsequence of the candidate under that rule.
//!
//! # Scoring
//!
//! Among all valid alignments the matcher chooses, by dynamic programming, the
//! one maximising a deterministic score that rewards tight, well-anchored
//! matches the way a human expects:
//!
//! * every matched character earns a flat reward;
//! * matching the first character of a *word boundary* (string start, or after
//!   a separator such as space, `/`, `-`, `_`, `.`, `:`, `\`, or at a
//!   lower-to-upper camel-case transition) earns a bonus;
//! * matching *consecutively* (immediately after the previous match) earns a
//!   bonus, so contiguous runs beat scattered ones;
//! * each non-consecutive jump pays a small gap penalty.
//!
//! Ties are broken toward the left-most alignment, so results are fully
//! deterministic. Returned positions are **character** indices into the
//! candidate (not byte offsets), strictly increasing.

#![forbid(unsafe_code)]

use alloc::string::String;
use alloc::vec;
use alloc::vec::Vec;

use crate::registry::Workbench;
use crate::story::Story;

/// Reward for every matched character.
const SCORE_CHAR: i32 = 16;
/// Bonus for matching the first character after a word boundary.
const SCORE_BOUNDARY: i32 = 8;
/// Bonus for matching immediately after the previous matched character.
const SCORE_CONSECUTIVE: i32 = 8;
/// Penalty for a non-consecutive jump between two matched characters.
const SCORE_GAP: i32 = -3;

/// A successful fuzzy match of a pattern against a candidate.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FuzzyMatch {
    /// Total alignment score; larger is a better match.
    score: i32,
    /// Character indices (not byte offsets) in the candidate that the pattern
    /// matched, in strictly increasing order. Empty when the pattern is empty.
    positions: Vec<usize>,
}

impl FuzzyMatch {
    /// The alignment score; larger means a tighter, better-anchored match.
    #[must_use]
    pub fn score(&self) -> i32 {
        self.score
    }

    /// The matched character indices into the candidate, strictly increasing.
    #[must_use]
    pub fn positions(&self) -> &[usize] {
        &self.positions
    }
}

/// Returns `true` when `prev` is the last character before a word boundary at
/// `cur`, i.e. `cur` begins a new "word" for scoring purposes.
fn is_boundary(cand: &[char], index: usize) -> bool {
    if index == 0 {
        return true;
    }
    let prev = cand[index - 1];
    let cur = cand[index];
    if matches!(prev, ' ' | '/' | '-' | '_' | '.' | ':' | '\\') {
        return true;
    }
    // camel-case transition: a non-uppercase character followed by an uppercase
    // one starts a new word (e.g. the `P` in `tabPanel`).
    !prev.is_uppercase() && cur.is_uppercase()
}

/// Base reward for placing a matched character at `index` in the candidate.
fn base_score(cand: &[char], index: usize) -> i32 {
    if is_boundary(cand, index) {
        SCORE_CHAR + SCORE_BOUNDARY
    } else {
        SCORE_CHAR
    }
}

/// Case-insensitive (ASCII) character equality.
fn eq_fold(a: char, b: char) -> bool {
    a.eq_ignore_ascii_case(&b)
}

/// Fuzzy-matches `pattern` against `candidate`.
///
/// Returns [`Some`] with the best-scoring alignment when `pattern` is a
/// subsequence of `candidate` (ASCII-case-insensitively), or [`None`]
/// otherwise. An empty pattern always matches, with score `0` and no positions.
///
/// The returned [`FuzzyMatch::positions`] are character indices into
/// `candidate`, strictly increasing, each pointing at a character equal (under
/// case folding) to the corresponding pattern character.
#[must_use]
pub fn fuzzy_match(pattern: &str, candidate: &str) -> Option<FuzzyMatch> {
    let pat: Vec<char> = pattern.chars().collect();
    if pat.is_empty() {
        return Some(FuzzyMatch {
            score: 0,
            positions: Vec::new(),
        });
    }
    let cand: Vec<char> = candidate.chars().collect();
    if pat.len() > cand.len() {
        return None;
    }

    // `dp[j]` for the current pattern row: best score ending with the current
    // pattern character placed at candidate index `j`, or `None` if infeasible.
    // `parent[k][j]` records the candidate index chosen for pattern character
    // `k - 1` when `k`'s match sits at `j`, enabling reconstruction.
    let mut prev_row: Vec<Option<i32>> = vec![None; cand.len()];
    let mut parents: Vec<Vec<Option<usize>>> = Vec::with_capacity(pat.len());

    for (k, &pc) in pat.iter().enumerate() {
        let mut cur_row: Vec<Option<i32>> = vec![None; cand.len()];
        let mut par_row: Vec<Option<usize>> = vec![None; cand.len()];
        for (j, &cc) in cand.iter().enumerate() {
            if !eq_fold(pc, cc) {
                continue;
            }
            if k == 0 {
                cur_row[j] = Some(base_score(&cand, j));
                continue;
            }
            // Combine with the best feasible placement of the previous pattern
            // character at some index `i < j`. `i == j - 1` is a consecutive
            // match (bonus); otherwise it is a gap (penalty).
            let mut best: Option<(i32, usize)> = None;
            for (i, prev) in prev_row[..j].iter().enumerate() {
                let Some(prev_score) = *prev else {
                    continue;
                };
                let step = if i + 1 == j {
                    SCORE_CONSECUTIVE
                } else {
                    SCORE_GAP
                };
                let total = prev_score + step;
                match best {
                    Some((best_total, _)) if best_total >= total => {}
                    _ => best = Some((total, i)),
                }
            }
            if let Some((total, i)) = best {
                cur_row[j] = Some(total + base_score(&cand, j));
                par_row[j] = Some(i);
            }
        }
        parents.push(par_row);
        prev_row = cur_row;
    }

    // Pick the best feasible end position in the final row (left-most on ties).
    let mut end: Option<(i32, usize)> = None;
    for (j, cell) in prev_row.iter().enumerate() {
        let Some(score) = *cell else {
            continue;
        };
        match end {
            Some((best_score, _)) if best_score >= score => {}
            _ => end = Some((score, j)),
        }
    }
    let (score, mut j) = end?;

    // Reconstruct positions back to front using the parent pointers.
    let mut positions = vec![0usize; pat.len()];
    for k in (0..pat.len()).rev() {
        positions[k] = j;
        if k > 0 {
            j = parents[k][j].expect("feasible cell has a parent");
        }
    }

    Some(FuzzyMatch { score, positions })
}

/// A workbench story that matched a fuzzy query, with its alignment score.
#[derive(Debug, Clone, Copy)]
pub struct StoryHit<'a> {
    /// The group path the story is registered under (e.g. `"Forms/Button"`).
    pub group: &'a str,
    /// The story's name within its group.
    pub name: &'a str,
    /// The matched story.
    pub story: &'a Story,
    /// The fuzzy alignment score against the `"group/name"` label.
    pub score: i32,
}

/// Fuzzy-filters every story in `workbench` by `pattern`, matching against each
/// story's `"group/name"` label.
///
/// Returns the matching stories ordered by descending score; ties keep the
/// workbench's deterministic sorted `(group, name)` order. An empty pattern
/// returns every story (all with score `0`).
#[must_use]
pub fn fuzzy_filter<'a>(workbench: &'a Workbench, pattern: &str) -> Vec<StoryHit<'a>> {
    let mut hits: Vec<StoryHit<'a>> = Vec::new();
    let mut label = String::new();
    for (group, name, story) in workbench.stories() {
        label.clear();
        label.push_str(group);
        label.push('/');
        label.push_str(name);
        if let Some(matched) = fuzzy_match(pattern, &label) {
            hits.push(StoryHit {
                group,
                name,
                story,
                score: matched.score(),
            });
        }
    }
    // Stable sort by descending score preserves the sorted-label order of
    // `stories()` as the deterministic tie-breaker.
    hits.sort_by_key(|hit| core::cmp::Reverse(hit.score));
    hits
}

#[cfg(all(test, feature = "std"))]
mod tests {
    use super::*;
    use crate::controls::ControlValue;
    use prism_ui::Element;

    /// Independent oracle: does `pattern` occur in `candidate` as an
    /// ASCII-case-insensitive subsequence? Greedy left-most scan is a standard
    /// correct subsequence test.
    fn is_subsequence(pattern: &str, candidate: &str) -> bool {
        let mut cand = candidate.chars();
        'outer: for pc in pattern.chars() {
            for cc in cand.by_ref() {
                if eq_fold(pc, cc) {
                    continue 'outer;
                }
            }
            return false;
        }
        true
    }

    struct SplitMix64(u64);
    impl SplitMix64 {
        fn next_u64(&mut self) -> u64 {
            self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut z = self.0;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            z ^ (z >> 31)
        }
        fn below(&mut self, n: u64) -> u64 {
            self.next_u64() % n
        }
    }

    fn random_string(rng: &mut SplitMix64, alphabet: &[char], max_len: usize) -> String {
        let len = rng.below((max_len + 1) as u64) as usize;
        let mut s = String::new();
        for _ in 0..len {
            let idx = rng.below(alphabet.len() as u64) as usize;
            s.push(alphabet[idx]);
        }
        s
    }

    #[test]
    fn empty_pattern_matches_with_zero_score() {
        let m = fuzzy_match("", "anything").expect("empty pattern matches");
        assert_eq!(m.score(), 0);
        assert!(m.positions().is_empty());

        let m = fuzzy_match("", "").expect("empty matches empty");
        assert_eq!(m.score(), 0);
    }

    #[test]
    fn single_char_boundary_vs_interior_score() {
        // Index 0 is always a boundary.
        assert_eq!(
            fuzzy_match("a", "a").expect("matches").score(),
            SCORE_CHAR + SCORE_BOUNDARY
        );
        // Interior, non-boundary placement earns only the base char reward.
        assert_eq!(fuzzy_match("a", "ba").expect("matches").score(), SCORE_CHAR);
    }

    #[test]
    fn pattern_longer_than_candidate_never_matches() {
        assert!(fuzzy_match("abcd", "abc").is_none());
    }

    #[test]
    fn non_subsequence_returns_none() {
        assert!(fuzzy_match("zzz", "abc").is_none());
        // Right characters, wrong order.
        assert!(fuzzy_match("ca", "abc").is_none());
    }

    #[test]
    fn case_insensitive_match() {
        let m = fuzzy_match("AB", "xaby").expect("case-insensitive match");
        assert_eq!(m.positions(), [1, 2]);
    }

    #[test]
    fn consecutive_beats_scattered() {
        let tight = fuzzy_match("ab", "ab").expect("tight").score();
        let loose = fuzzy_match("ab", "axb").expect("loose").score();
        assert!(tight > loose, "tight={tight} loose={loose}");
    }

    #[test]
    fn boundary_anchor_beats_interior() {
        // The matched `b` sits after a space (boundary) in the first case and
        // mid-word in the second.
        let anchored = fuzzy_match("b", "a b").expect("anchored").score();
        let interior = fuzzy_match("b", "ab").expect("interior").score();
        assert!(anchored > interior, "anchored={anchored} interior={interior}");
    }

    #[test]
    fn prefix_match_positions_start_at_zero() {
        let m = fuzzy_match("cmd", "Command Palette").expect("prefix match");
        assert_eq!(m.positions()[0], 0);
    }

    #[test]
    fn deterministic_repeated_calls() {
        let a = fuzzy_match("abc", "a_b_c_abc").expect("match");
        let b = fuzzy_match("abc", "a_b_c_abc").expect("match");
        assert_eq!(a, b);
    }

    #[test]
    fn property_match_iff_subsequence_and_positions_valid() {
        let alphabet = ['a', 'b', 'c', 'd'];
        let mut rng = SplitMix64(0x1234_5678_9ABC_DEF0);
        for _ in 0..4000 {
            let pattern = random_string(&mut rng, &alphabet, 5);
            let candidate = random_string(&mut rng, &alphabet, 10);

            let expected = is_subsequence(&pattern, &candidate);
            let got = fuzzy_match(&pattern, &candidate);
            assert_eq!(
                got.is_some(),
                expected,
                "pattern={pattern:?} candidate={candidate:?}"
            );

            if let Some(m) = got {
                let pat: Vec<char> = pattern.chars().collect();
                let cand: Vec<char> = candidate.chars().collect();
                assert_eq!(m.positions().len(), pat.len());
                // Strictly increasing and each matches its pattern character.
                let mut last: Option<usize> = None;
                for (pi, &ci) in m.positions().iter().enumerate() {
                    if let Some(prev) = last {
                        assert!(ci > prev, "positions not strictly increasing");
                    }
                    assert!(eq_fold(pat[pi], cand[ci]), "position char mismatch");
                    last = Some(ci);
                }
            }
        }
    }

    #[test]
    fn fuzzy_filter_orders_and_filters_workbench() {
        let leaf = |name: &str| Story::builder(name).build(|_ctx| Element::box_());
        let mut wb = Workbench::new();
        wb.add("Forms/Button", leaf("Primary"));
        wb.add("Layout/Stack", leaf("Vertical"));

        // "btn" is a subsequence of "Forms/Button" but not of "Layout/Stack".
        let hits = fuzzy_filter(&wb, "btn");
        assert_eq!(hits.len(), 1);
        assert_eq!((hits[0].group, hits[0].name), ("Forms/Button", "Primary"));

        // No match.
        assert!(fuzzy_filter(&wb, "zzz").is_empty());

        // Empty pattern returns every story.
        assert_eq!(fuzzy_filter(&wb, "").len(), wb.len());
    }

    #[test]
    fn fuzzy_filter_sorts_by_descending_score() {
        let leaf = |name: &str| Story::builder(name).build(|_ctx| Element::box_());
        let mut wb = Workbench::new();
        // "Button/Press" scores higher for "button" (contiguous prefix) than
        // "Big/Unit" which also contains the subsequence b-u-t-t-o-n scattered.
        wb.add("Button", leaf("Press"));
        wb.add("Big", leaf("UnitTotoron"));

        let hits = fuzzy_filter(&wb, "button");
        assert!(!hits.is_empty());
        // Descending score ordering.
        for pair in hits.windows(2) {
            assert!(pair[0].score >= pair[1].score);
        }
    }

    #[test]
    fn uses_control_value_in_story() {
        // Smoke test that stories carrying controls still filter cleanly.
        let story = Story::builder("Themed")
            .arg("dark", ControlValue::Bool(true))
            .build(|_ctx| Element::box_());
        let mut wb = Workbench::new();
        wb.add("Theme/Mode", story);
        let hits = fuzzy_filter(&wb, "theme");
        assert_eq!(hits.len(), 1);
    }
}
