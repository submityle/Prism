//! Real-device parity: the `GPU` sphere-sphere narrow-phase kernel must
//! reproduce the `CPU` golden twin's contact manifolds slot for slot.
//!
//! Each test hands a batch of candidate pairs to both
//! [`GpuNarrowphase::query`] and [`cpu_narrowphase`] and compares the two
//! outputs index by index. The validity decision (a reported contact versus a
//! `None` slot) must match exactly; when both report a contact, the particle
//! indices must match exactly and the normal, depth, and point within a tight
//! tolerance, since the only inexact steps are the square root and the
//! reciprocal in the normalisation. The scenes are built with clear overlaps and
//! clear gaps (never a grazing `dist ~= ra + rb` boundary), so the tiny
//! floating-point tolerance can never flip a validity flag.
//!
//! On a headless host with no `wgpu` adapter the tests skip (with a printed
//! notice) instead of failing, so the suite stays green everywhere while still
//! exercising the full dispatch-and-readback on any machine with a real device.
//!
//! Provenance: textbook sphere-sphere collision manifold. No Unreal Engine
//! source or derived code.

use glam::Vec3;
use prism_physics_gpu::{
    cpu_narrowphase, CandidatePair, Contact, GpuContext, GpuNarrowphase, Particle,
};

/// Tolerance on the normal, depth, and point; the only inexact steps are the
/// square root and the reciprocal in the normalisation, both well under this
/// bound over the test scene scales.
const TOL: f32 = 1e-4;

/// A small xorshift generator so the scenes are deterministic without pulling
/// in an `RNG` dependency.
struct Rng {
    /// Mutable generator state; never zero.
    state: u64,
}

impl Rng {
    /// Seeds the generator, forcing a non-zero state.
    fn new(seed: u64) -> Rng {
        Rng { state: seed | 1 }
    }

    /// Advances the state and returns the next 64-bit value.
    fn next_u64(&mut self) -> u64 {
        let mut x = self.state;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.state = x;
        x.wrapping_mul(0x2545_f491_4f6c_dd1d)
    }

    /// Returns an `f32` in `[lo, lo + span)`.
    fn coord(&mut self, lo: f32, span: f32) -> f32 {
        let unit = (self.next_u64() >> 40) as f32 / (1u64 << 24) as f32;
        lo + unit * span
    }
}

/// Compares one `GPU` contact slot to the `CPU` twin's, allowing only the
/// square-root and reciprocal tolerance.
#[expect(
    clippy::print_stderr,
    reason = "surface the differing slot on a parity failure"
)]
fn assert_slot_matches(index: usize, want: Option<Contact>, got: Option<Contact>) {
    match (want, got) {
        (None, None) => {}
        (Some(w), Some(g)) => {
            assert_eq!(w.a, g.a, "slot {index}: contact a must match");
            assert_eq!(w.b, g.b, "slot {index}: contact b must match");
            assert!(
                (w.normal - g.normal).length() < TOL,
                "slot {index}: normal {:?} vs {:?}",
                w.normal,
                g.normal
            );
            assert!(
                (w.depth - g.depth).abs() < TOL,
                "slot {index}: depth {} vs {}",
                w.depth,
                g.depth
            );
            assert!(
                (w.point - g.point).length() < TOL,
                "slot {index}: point {:?} vs {:?}",
                w.point,
                g.point
            );
        }
        (want, got) => {
            eprintln!("slot {index}: validity mismatch: cpu {want:?} vs gpu {got:?}");
            panic!("slot {index}: one path reported a contact and the other did not");
        }
    }
}

/// Runs both paths over the same inputs and checks every slot agrees.
#[expect(
    clippy::print_stderr,
    reason = "the skip notice must reach the test log on hosts without a wgpu adapter"
)]
fn check(particles: &[Particle], pairs: &[CandidatePair]) {
    let cpu = cpu_narrowphase(particles, pairs);

    let Some(ctx) = GpuContext::try_headless() else {
        eprintln!("skipping GPU narrow-phase parity: no wgpu adapter on this host");
        return;
    };
    let narrow = GpuNarrowphase::new(&ctx);
    let gpu = narrow.query(&ctx, particles, pairs);

    assert_eq!(
        gpu.len(),
        cpu.len(),
        "GPU must emit one slot per pair like the CPU twin"
    );
    for (index, (want, got)) in cpu.into_iter().zip(gpu).enumerate() {
        assert_slot_matches(index, want, got);
    }
}

#[test]
fn hand_built_cases_agree_slot_for_slot() {
    // A deliberate mix: clear overlap on axis, clear overlap off axis, a clear
    // gap, and coincident centres. Every case is far from the touching
    // boundary, so the validity flag is unambiguous.
    let particles = [
        Particle::new(Vec3::new(0.0, 0.0, 0.0), 1.0),
        Particle::new(Vec3::new(1.0, 0.0, 0.0), 1.0),
        Particle::new(Vec3::new(0.7, 0.7, 0.0), 1.0),
        Particle::new(Vec3::new(10.0, 0.0, 0.0), 1.0),
        Particle::new(Vec3::new(-3.0, 2.0, 1.0), 0.5),
        Particle::new(Vec3::new(-3.0, 2.0, 1.0), 0.75),
    ];
    let pairs = [
        CandidatePair::new(0, 1), // overlap on +x, depth 1
        CandidatePair::new(0, 2), // overlap off axis
        CandidatePair::new(0, 3), // clear gap, no contact
        CandidatePair::new(4, 5), // coincident centres, fallback axis
    ];
    check(&particles, &pairs);
}

#[test]
fn empty_batch_returns_empty() {
    let particles = [Particle::new(Vec3::ZERO, 1.0)];
    check(&particles, &[]);
}

#[test]
fn randomised_overlaps_and_gaps_agree() {
    // Build pairs of spheres whose separation is chosen to be either a clear
    // overlap or a clear gap, never near the touching boundary, so the GPU and
    // CPU always agree on the validity flag.
    let mut rng = Rng::new(0xC0FF_EE00_1234_5678);
    let mut particles = Vec::new();
    let mut pairs = Vec::new();
    let count = 128_u32;
    for i in 0..count {
        let ra = rng.coord(0.4, 0.6);
        let rb = rng.coord(0.4, 0.6);
        let sum_r = ra + rb;
        let centre_a = Vec3::new(
            rng.coord(-20.0, 40.0),
            rng.coord(-20.0, 40.0),
            rng.coord(-20.0, 40.0),
        );
        // A random unit-ish direction (never zero: the +1.0 keeps x positive).
        let dir = Vec3::new(
            rng.coord(0.0, 1.0) + 1.0,
            rng.coord(-1.0, 2.0),
            rng.coord(-1.0, 2.0),
        )
        .normalize();
        // Half the pairs clearly overlap (dist ~ 0.5 * sum_r), half clearly
        // separate (dist ~ 1.6 * sum_r); neither is near the boundary.
        let factor = if i % 2 == 0 { 0.5 } else { 1.6 };
        let centre_b = centre_a + dir * (sum_r * factor);
        particles.push(Particle::new(centre_a, ra));
        particles.push(Particle::new(centre_b, rb));
        pairs.push(CandidatePair::new(2 * i, 2 * i + 1));
    }
    check(&particles, &pairs);
}
