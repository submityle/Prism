//! §24.8 tests: multi-world time-domain isolation and deterministic audit —
//! pause isolation, independent scale, audit double-run matching, first-divergence
//! location, length mismatch, and boundaries.

use crate::{
    compare_trails, fnv1a_64, AuditDiff, AuditTrail, Duration, StateHasher, WorldSet,
    WorldTimeDomain,
};

fn ms(n: u64) -> Duration {
    Duration::from_millis(n)
}

#[test]
fn pausing_one_world_does_not_affect_another() {
    let mut running = WorldTimeDomain::from_hz(100); // 10 ms step
    let mut frozen = WorldTimeDomain::from_hz(100);
    frozen.pause();

    // Advance both by the same real delta for several frames.
    let mut running_ticks = 0;
    let mut frozen_ticks = 0;
    for _ in 0..10 {
        running_ticks += running.advance(ms(10));
        frozen_ticks += frozen.advance(ms(10));
    }

    assert_eq!(running_ticks, 10);
    assert_eq!(running.tick(), 10);
    // The paused world is untouched: zero ticks, empty accumulator.
    assert_eq!(frozen_ticks, 0);
    assert_eq!(frozen.tick(), 0);
    assert_eq!(frozen.overstep_subunits(), 0);
    assert!(frozen.is_paused());

    // Unpausing resumes exactly where it left off, still independent.
    frozen.unpause();
    assert_eq!(frozen.advance(ms(10)), 1);
    assert_eq!(frozen.tick(), 1);
    assert_eq!(running.tick(), 10); // untouched by the other world
}

#[test]
fn worlds_scale_independently() {
    let mut full = WorldTimeDomain::from_hz(100); // scale 1.0
    let mut half = WorldTimeDomain::from_hz(100).with_scale(0.5);
    let mut double = WorldTimeDomain::from_hz(100).with_scale(2.0);

    for _ in 0..10 {
        full.advance(ms(10));
        half.advance(ms(10));
        double.advance(ms(10));
    }

    // 100 ms of game time each frame-batch: full=100ms, half=50ms, double=200ms.
    assert_eq!(full.tick(), 10);
    assert_eq!(half.tick(), 5);
    assert_eq!(double.tick(), 20);
    assert_eq!(full.scale(), 1.0);
    assert_eq!(half.scale(), 0.5);
}

#[test]
fn negative_and_nonfinite_scales_are_sanitized() {
    let w = WorldTimeDomain::from_hz(60).with_scale(-4.0);
    assert_eq!(w.scale(), 0.0);
    assert_eq!(w.effective_scale(), 0.0);

    let mut w2 = WorldTimeDomain::from_hz(60);
    w2.set_scale(f64::NAN);
    assert_eq!(w2.scale(), 1.0); // ignored, keeps previous
    w2.set_scale(f64::INFINITY);
    assert_eq!(w2.scale(), 1.0); // ignored
    w2.set_scale(-1.0);
    assert_eq!(w2.scale(), 0.0); // clamped
}

#[test]
fn world_set_advances_each_world_independently() {
    let mut set = WorldSet::new();
    let main = set.push(WorldTimeDomain::from_hz(100));
    let preview = set.push(WorldTimeDomain::from_hz(100).paused());
    let slow = set.push(WorldTimeDomain::from_hz(100).with_scale(0.5));
    assert_eq!(set.len(), 3);

    for _ in 0..20 {
        set.advance_all(ms(10));
    }

    assert_eq!(set.world(main).unwrap().tick(), 20);
    assert_eq!(set.world(preview).unwrap().tick(), 0);
    assert_eq!(set.world(slow).unwrap().tick(), 10);

    // Mutating one world via the set stays isolated.
    set.world_mut(preview).unwrap().unpause();
    set.advance_all(ms(10));
    assert_eq!(set.world(preview).unwrap().tick(), 1);
    assert_eq!(set.world(main).unwrap().tick(), 21);
}

#[test]
fn audit_double_run_is_identical() {
    fn run() -> AuditTrail {
        let mut set = WorldSet::new();
        set.push(WorldTimeDomain::from_hz(60));
        set.push(WorldTimeDomain::from_hz(50).with_scale(0.5));
        let pattern = [ms(16), ms(17), ms(16), Duration::from_micros(16_666)];
        let mut trail = AuditTrail::new();
        for i in 0..200usize {
            set.advance_all(pattern[i % 4]);
            trail.record(set.audit_hash());
        }
        trail
    }

    let a = run();
    let b = run();
    assert_eq!(a.len(), 200);
    assert!(compare_trails(&a, &b).is_identical());
}

#[test]
fn audit_locates_first_divergence() {
    // Two runs identical until frame 50, where one world is nudged.
    fn run(diverge_at: Option<usize>) -> AuditTrail {
        let mut world = WorldTimeDomain::from_hz(60);
        let mut trail = AuditTrail::new();
        for i in 0..100usize {
            world.advance(ms(16));
            if Some(i) == diverge_at {
                // A single extra accumulate perturbs this and all later frames.
                world.advance(ms(16));
            }
            trail.record(world.audit_hash());
        }
        trail
    }

    let baseline = run(None);
    let perturbed = run(Some(50));
    match compare_trails(&baseline, &perturbed) {
        AuditDiff::Diverged { frame, left, right } => {
            assert_eq!(frame, 50);
            assert_ne!(left, right);
        }
        other => panic!("expected divergence at 50, got {other:?}"),
    }
    assert_eq!(
        compare_trails(&baseline, &perturbed).diverged_frame(),
        Some(50)
    );
    // Frames before the divergence are provably equal.
    assert_eq!(&baseline.digests()[..50], &perturbed.digests()[..50]);
}

#[test]
fn audit_reports_length_mismatch_when_prefix_matches() {
    let mut short = AuditTrail::new();
    let mut long = AuditTrail::new();
    for d in [1u64, 2, 3] {
        short.record(d);
        long.record(d);
    }
    long.record(4);
    match compare_trails(&short, &long) {
        AuditDiff::LengthMismatch {
            matched,
            left_len,
            right_len,
        } => {
            assert_eq!(matched, 3);
            assert_eq!(left_len, 3);
            assert_eq!(right_len, 4);
        }
        other => panic!("expected length mismatch, got {other:?}"),
    }
}

#[test]
fn fnv1a_and_state_hasher_are_deterministic() {
    assert_eq!(fnv1a_64(b"prism"), fnv1a_64(b"prism"));
    assert_ne!(fnv1a_64(b"prism"), fnv1a_64(b"Prism"));

    let mut a = StateHasher::new();
    a.write_u64(1);
    a.write_u128(2);
    a.write_f64_bits(0.5);
    let mut b = StateHasher::new();
    b.write_u64(1);
    b.write_u128(2);
    b.write_f64_bits(0.5);
    assert_eq!(a.finish(), b.finish());

    // Field order matters: a different order yields a different digest.
    let mut c = StateHasher::new();
    c.write_u128(2);
    c.write_u64(1);
    c.write_f64_bits(0.5);
    assert_ne!(a.finish(), c.finish());
}

#[test]
fn empty_trails_compare_identical_and_set_defaults() {
    let a = AuditTrail::new();
    let b = AuditTrail::default();
    assert!(a.is_empty());
    assert_eq!(a.last(), None);
    assert!(compare_trails(&a, &b).is_identical());

    let set = WorldSet::default();
    assert!(set.is_empty());
    assert_eq!(set.len(), 0);
    assert!(set.world(0).is_none());
}

#[test]
fn world_reset_zeroes_ticks_but_keeps_policy() {
    let mut w = WorldTimeDomain::from_hz(60).with_scale(0.5);
    for _ in 0..10 {
        w.advance(ms(32));
    }
    assert!(w.tick() > 0);
    let hash_before = w.audit_hash();
    w.reset();
    assert_eq!(w.tick(), 0);
    assert_eq!(w.overstep_subunits(), 0);
    assert_eq!(w.scale(), 0.5); // policy kept
    assert_ne!(w.audit_hash(), hash_before);
}

#[test]
fn exact_step_world_has_no_drift() {
    // 50 Hz => 20 ms exact; feed 20 ms per frame, one tick per frame, forever.
    let mut w = WorldTimeDomain::from_hz(50);
    for _ in 0..100_000 {
        assert_eq!(w.advance(ms(20)), 1);
    }
    assert_eq!(w.tick(), 100_000);
    assert_eq!(w.overstep_subunits(), 0);
}
