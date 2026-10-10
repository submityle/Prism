//! §24.4 deterministic replay & trace comparison tests: the stable `FNV`-1a
//! hasher, per-frame trace recording, input + seed recording/replay, and
//! double-run comparison. These exercise the feature-neutral core arithmetic
//! with hand-computed oracles (pinned `FNV`-1a vectors + structural oracles).

use crate::determinism::hash::{fnv1a_64, StateHasher};
use crate::determinism::record::{FrameInput, InputRecorder};
use crate::determinism::trace::{compare, DeterminismTrace, FrameHash, TraceDiff};

// ---- stable hash: pinned FNV-1a vectors -------------------------------------

#[test]
fn fnv1a_matches_published_vectors() {
    // Canonical 64-bit FNV-1a test vectors (Landon Curt Noll). These pin the
    // offset basis + prime so a refactor can never silently change the digest.
    assert_eq!(fnv1a_64(b""), 0xcbf2_9ce4_8422_2325);
    assert_eq!(fnv1a_64(b"a"), 0xaf63_dc4c_8601_ec8c);
    assert_eq!(fnv1a_64(b"foobar"), 0x8594_4171_f739_67e8);
}

#[test]
fn state_hasher_empty_is_offset_basis() {
    assert_eq!(StateHasher::new().finish(), 0xcbf2_9ce4_8422_2325);
    assert_eq!(StateHasher::default().finish(), 0xcbf2_9ce4_8422_2325);
}

#[test]
fn state_hasher_write_bytes_equals_free_function() {
    let mut hasher = StateHasher::new();
    hasher.write_bytes(b"foobar");
    assert_eq!(hasher.finish(), fnv1a_64(b"foobar"));
}

#[test]
fn state_hasher_integer_folds_are_little_endian() {
    let mut hasher = StateHasher::new();
    hasher.write_u64(0x0102_0304_0506_0708);
    assert_eq!(
        hasher.finish(),
        fnv1a_64(&0x0102_0304_0506_0708u64.to_le_bytes())
    );

    let mut h32 = StateHasher::new();
    h32.write_u32(0x0a0b_0c0d);
    assert_eq!(h32.finish(), fnv1a_64(&0x0a0b_0c0du32.to_le_bytes()));
}

#[test]
fn state_hasher_field_order_matters() {
    let mut a = StateHasher::new();
    a.write_u32(1);
    a.write_u32(2);

    let mut b = StateHasher::new();
    b.write_u32(2);
    b.write_u32(1);

    assert_ne!(a.finish(), b.finish());
}

#[test]
fn state_hasher_str_length_prefix_disambiguates() {
    // "ab" + "c" must not collide with "a" + "bc": the length prefix separates.
    let mut a = StateHasher::new();
    a.write_str("ab");
    a.write_str("c");

    let mut b = StateHasher::new();
    b.write_str("a");
    b.write_str("bc");

    assert_ne!(a.finish(), b.finish());
}

#[test]
fn state_hasher_from_state_chains() {
    let first = {
        let mut h = StateHasher::new();
        h.write_u64(7);
        h.finish()
    };
    // Continuing from a prior digest equals one hasher fed both values in order.
    let chained = {
        let mut h = StateHasher::from_state(first);
        h.write_u64(9);
        h.finish()
    };
    let combined = {
        let mut h = StateHasher::new();
        h.write_u64(7);
        h.write_u64(9);
        h.finish()
    };
    assert_eq!(chained, combined);
}

#[test]
fn state_hasher_floats_hash_by_bits() {
    let mut neg_zero = StateHasher::new();
    neg_zero.write_f64_bits(-0.0);
    let mut pos_zero = StateHasher::new();
    pos_zero.write_f64_bits(0.0);
    // -0.0 and 0.0 compare equal as values but have different bit patterns.
    assert_ne!(neg_zero.finish(), pos_zero.finish());
}

// ---- deterministic trace: recording -----------------------------------------

#[test]
fn trace_records_in_order() {
    let mut trace = DeterminismTrace::new();
    assert!(trace.is_empty());
    trace.record(0, 0xaa);
    trace.record(1, 0xbb);
    trace.push(FrameHash::new(2, 0xcc));
    assert_eq!(trace.len(), 3);
    assert_eq!(
        trace.frames(),
        &[
            FrameHash::new(0, 0xaa),
            FrameHash::new(1, 0xbb),
            FrameHash::new(2, 0xcc),
        ]
    );
    assert_eq!(trace.last(), Some(FrameHash::new(2, 0xcc)));
    trace.clear();
    assert!(trace.is_empty());
    assert_eq!(trace.last(), None);
}

// ---- double-run comparison --------------------------------------------------

#[test]
fn compare_identical_traces() {
    let mut a = DeterminismTrace::new();
    let mut b = DeterminismTrace::new();
    for frame in 0..5u64 {
        let h = fnv1a_64(&frame.to_le_bytes());
        a.record(frame, h);
        b.record(frame, h);
    }
    let diff = compare(&a, &b);
    assert_eq!(diff, TraceDiff::Identical);
    assert!(diff.is_identical());
    assert_eq!(diff.diverged_frame(), None);
    assert_eq!(diff.divergence_position(), None);
    // Method form agrees with the free function.
    assert_eq!(a.compare(&b), diff);
}

#[test]
fn compare_locates_first_divergence() {
    let mut a = DeterminismTrace::new();
    let mut b = DeterminismTrace::new();
    for frame in 0..6u64 {
        a.record(frame, 100 + frame);
        // Run B diverges starting at frame 3.
        let h = if frame >= 3 { 999 + frame } else { 100 + frame };
        b.record(frame, h);
    }
    let diff = compare(&a, &b);
    assert_eq!(
        diff,
        TraceDiff::Diverged {
            frame: 3,
            left: FrameHash::new(3, 103),
            right: FrameHash::new(3, 1002),
        }
    );
    assert_eq!(diff.diverged_frame(), Some(3));
    assert_eq!(diff.divergence_position(), Some(3));
}

#[test]
fn compare_divergence_at_frame_zero() {
    let mut a = DeterminismTrace::new();
    a.record(0, 1);
    let mut b = DeterminismTrace::new();
    b.record(0, 2);
    assert_eq!(
        compare(&a, &b),
        TraceDiff::Diverged {
            frame: 0,
            left: FrameHash::new(0, 1),
            right: FrameHash::new(0, 2),
        }
    );
}

#[test]
fn compare_detects_renumbered_frame_as_divergence() {
    // Same hash but a dropped/renumbered frame number must surface, not realign.
    let mut a = DeterminismTrace::new();
    a.record(0, 42);
    a.record(1, 43);
    let mut b = DeterminismTrace::new();
    b.record(0, 42);
    b.record(2, 43);
    assert_eq!(
        compare(&a, &b),
        TraceDiff::Diverged {
            frame: 1,
            left: FrameHash::new(1, 43),
            right: FrameHash::new(2, 43),
        }
    );
}

#[test]
fn compare_length_mismatch_right_shorter() {
    let mut a = DeterminismTrace::new();
    let mut b = DeterminismTrace::new();
    for frame in 0..4u64 {
        a.record(frame, frame);
    }
    for frame in 0..2u64 {
        b.record(frame, frame);
    }
    let diff = compare(&a, &b);
    assert_eq!(
        diff,
        TraceDiff::LengthMismatch {
            matched: 2,
            left_len: 4,
            right_len: 2,
        }
    );
    assert!(!diff.is_identical());
    assert_eq!(diff.diverged_frame(), None);
    assert_eq!(diff.divergence_position(), Some(2));
}

#[test]
fn compare_length_mismatch_left_shorter() {
    let mut a = DeterminismTrace::new();
    a.record(0, 7);
    let mut b = DeterminismTrace::new();
    b.record(0, 7);
    b.record(1, 8);
    assert_eq!(
        compare(&a, &b),
        TraceDiff::LengthMismatch {
            matched: 1,
            left_len: 1,
            right_len: 2,
        }
    );
}

#[test]
fn compare_empty_traces_are_identical() {
    let a = DeterminismTrace::new();
    let b = DeterminismTrace::new();
    assert_eq!(compare(&a, &b), TraceDiff::Identical);
}

#[test]
fn compare_empty_vs_nonempty_is_length_mismatch() {
    let a = DeterminismTrace::new();
    let mut b = DeterminismTrace::new();
    b.record(0, 1);
    assert_eq!(
        compare(&a, &b),
        TraceDiff::LengthMismatch {
            matched: 0,
            left_len: 0,
            right_len: 1,
        }
    );
}

// ---- input + seed recording / replay ----------------------------------------

#[test]
fn recorder_auto_numbers_frames_and_hashes_input() {
    let mut rec = InputRecorder::new();
    assert!(rec.is_empty());
    assert_eq!(rec.next_frame(), 0);
    let f0 = rec.record(b"jump", 11);
    let f1 = rec.record(b"left", 22);
    assert_eq!(
        f0,
        FrameInput::new(0, InputRecorder::hash_input(b"jump"), 11)
    );
    assert_eq!(
        f1,
        FrameInput::new(1, InputRecorder::hash_input(b"left"), 22)
    );
    assert_eq!(f0.input_hash, fnv1a_64(b"jump"));
    assert_eq!(rec.len(), 2);
    assert_eq!(rec.next_frame(), 2);
}

#[test]
fn recorder_push_sets_explicit_frame_and_advances_cursor() {
    let mut rec = InputRecorder::new();
    rec.push(FrameInput::new(10, 0xdead, 5));
    assert_eq!(rec.next_frame(), 11);
    let next = rec.record_hashed(0xbeef, 6);
    assert_eq!(next.frame, 11);
}

#[test]
fn same_inputs_two_runs_produce_identical_per_frame_hashes() {
    // Two independent runs fed the identical input + seed stream must produce
    // byte-identical per-frame digests and compare Identical.
    let inputs: [(&[u8], u64); 4] = [(b"up", 1), (b"down", 2), (b"left", 3), (b"right", 4)];
    let mut run_a = InputRecorder::new();
    let mut run_b = InputRecorder::new();
    for (input, seed) in inputs {
        run_a.record(input, seed);
        run_b.record(input, seed);
    }
    assert_eq!(run_a.digest(), run_b.digest());
    let trace_a = run_a.to_trace();
    let trace_b = run_b.to_trace();
    // Per-frame hashes equal, position by position.
    assert_eq!(trace_a.frames(), trace_b.frames());
    assert_eq!(compare(&trace_a, &trace_b), TraceDiff::Identical);
}

#[test]
fn injected_seed_divergence_is_localized() {
    let mut run_a = InputRecorder::new();
    let mut run_b = InputRecorder::new();
    for frame in 0..5u64 {
        run_a.record_hashed(frame, 1000 + frame);
        // Run B uses a different seed starting at frame 2 (e.g. an RNG desync).
        let seed = if frame == 2 { 7777 } else { 1000 + frame };
        run_b.record_hashed(frame, seed);
    }
    assert_ne!(run_a.digest(), run_b.digest());
    let diff = compare(&run_a.to_trace(), &run_b.to_trace());
    assert_eq!(diff.diverged_frame(), Some(2));
    // The seed difference is the only change, so frames 0,1 match and 3,4 are
    // never reached by the comparison.
    match diff {
        TraceDiff::Diverged { frame, left, right } => {
            assert_eq!(frame, 2);
            assert_eq!(
                left,
                FrameHash::new(2, FrameInput::new(2, 2, 1002).digest())
            );
            assert_eq!(
                right,
                FrameHash::new(2, FrameInput::new(2, 2, 7777).digest())
            );
        }
        other => panic!("expected divergence, got {other:?}"),
    }
}

#[test]
fn replay_reproduces_recorded_stream_exactly() {
    // Record a run, then replay it back through a fresh recorder; the replayed
    // stream must reproduce the original bit-for-bit (precise reproduction).
    let mut original = InputRecorder::new();
    original.record(b"a", 10);
    original.record(b"b", 20);
    original.record(b"c", 30);

    let mut replay = original.replay();
    assert_eq!(replay.remaining(), 3);
    assert_eq!(replay.peek(), Some(original.entries()[0]));

    let mut reproduced = InputRecorder::new();
    while let Some(frame) = replay.next_frame() {
        // Feeding the recorded input + seed back reconstructs the same frame.
        reproduced.push(frame);
    }
    assert!(replay.is_done());
    assert_eq!(replay.remaining(), 0);
    assert_eq!(reproduced.entries(), original.entries());
    assert_eq!(reproduced.digest(), original.digest());
    assert_eq!(
        compare(&reproduced.to_trace(), &original.to_trace()),
        TraceDiff::Identical
    );
}

#[test]
fn replay_cursor_as_iterator_and_reset() {
    let mut rec = InputRecorder::new();
    rec.record_hashed(1, 100);
    rec.record_hashed(2, 200);
    let mut cursor = rec.replay();
    assert_eq!(cursor.position(), 0);
    let collected: Vec<FrameInput> = cursor.by_ref().collect();
    assert_eq!(collected, rec.entries());
    assert!(cursor.is_done());
    cursor.reset();
    assert_eq!(cursor.position(), 0);
    assert_eq!(cursor.remaining(), 2);
}

#[test]
fn empty_recorder_digest_is_stable_and_distinct_from_single_frame() {
    let empty = InputRecorder::new();
    assert_eq!(empty.len(), 0);
    assert!(empty.to_trace().is_empty());
    // Empty digest folds just the zero length; it is stable across constructions.
    assert_eq!(empty.digest(), InputRecorder::new().digest());
    let mut one = InputRecorder::new();
    one.record_hashed(0, 0);
    assert_ne!(empty.digest(), one.digest());
}

#[test]
fn recorder_clear_resets_frame_numbering() {
    let mut rec = InputRecorder::new();
    rec.record(b"x", 1);
    rec.record(b"y", 2);
    rec.clear();
    assert!(rec.is_empty());
    assert_eq!(rec.next_frame(), 0);
    let f = rec.record(b"z", 3);
    assert_eq!(f.frame, 0);
}
