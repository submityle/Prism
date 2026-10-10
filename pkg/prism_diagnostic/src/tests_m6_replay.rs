//! M6 tests for deterministic replay markers.

use crate::replay::{compare_timelines, fnv1a_64, ReplayDivergence, ReplayMarker, ReplayTimeline};

fn timeline(entries: &[(u64, &'static str, u64)]) -> ReplayTimeline {
    let mut t = ReplayTimeline::new();
    for &(frame, label, hash) in entries {
        t.mark(frame, label, hash);
    }
    t
}

#[test]
fn identical_runs_match() {
    let a = timeline(&[(0, "world", 1), (1, "world", 2), (2, "physics", 3)]);
    let b = a.clone();
    assert_eq!(a.diff(&b), ReplayDivergence::Match);
    assert!(a.diff(&b).is_match());
    assert_eq!(a.diff(&b).divergence_index(), None);
    assert_eq!(a.len(), 3);
    assert!(!a.is_empty());
}

#[test]
fn first_hash_divergence_is_localized() {
    let a = timeline(&[(0, "world", 10), (1, "world", 20), (2, "world", 30)]);
    let b = timeline(&[(0, "world", 10), (1, "world", 999), (2, "world", 30)]);
    let div = compare_timelines(&a, &b);
    match div {
        ReplayDivergence::Diverged { index, left, right } => {
            assert_eq!(index, 1, "the FIRST mismatch is reported, not a later one");
            assert_eq!(left.hash, 20);
            assert_eq!(right.hash, 999);
        }
        other => panic!("expected Diverged, got {other:?}"),
    }
    assert_eq!(div.divergence_index(), Some(1));
    assert!(!div.is_match());
}

#[test]
fn frame_field_is_informational_not_part_of_match() {
    // Same label+hash but a dropped frame number still matches: the hash is the
    // source of truth, the frame index is just informational.
    let a = timeline(&[(0, "world", 7)]);
    let b = timeline(&[(99, "world", 7)]);
    assert_eq!(a.diff(&b), ReplayDivergence::Match);

    // ...but a different label at the same frame is a divergence.
    let c = timeline(&[(0, "physics", 7)]);
    assert!(matches!(a.diff(&c), ReplayDivergence::Diverged { .. }));
}

#[test]
fn length_mismatch_reports_common_prefix() {
    let a = timeline(&[(0, "world", 1), (1, "world", 2)]);
    let b = timeline(&[(0, "world", 1), (1, "world", 2), (2, "world", 3)]);
    match a.diff(&b) {
        ReplayDivergence::LengthMismatch {
            common,
            left_len,
            right_len,
        } => {
            assert_eq!(common, 2);
            assert_eq!(left_len, 2);
            assert_eq!(right_len, 3);
        }
        other => panic!("expected LengthMismatch, got {other:?}"),
    }
    assert_eq!(a.diff(&b).divergence_index(), Some(2));
}

#[test]
fn push_precomposed_marker_round_trips() {
    let mut t = ReplayTimeline::new();
    assert!(t.is_empty());
    let m = ReplayMarker {
        frame: 5,
        label: "world",
        hash: 0xdead_beef,
    };
    t.push(m);
    assert_eq!(t.markers(), &[m]);
}

#[test]
fn fnv1a_64_is_deterministic_and_order_sensitive() {
    // Known FNV-1a 64 vector: the empty input is the offset basis.
    assert_eq!(fnv1a_64(b""), 0xcbf2_9ce4_8422_2325);
    // Deterministic: same bytes -> same hash.
    assert_eq!(fnv1a_64(b"prism"), fnv1a_64(b"prism"));
    // Order-sensitive: a byte-stream digest distinguishes permutations.
    assert_ne!(fnv1a_64(b"ab"), fnv1a_64(b"ba"));
}
