//! Real-device parity: the `GPU` sphere-halfspace narrow-phase kernel must
//! reproduce the `CPU` golden twin's contact manifolds slot for slot.
//!
//! Each test hands a batch of `(sphere, plane)` couples to both
//! [`GpuHalfspaceNarrowphase::query`] and [`cpu_halfspace_narrowphase`] and
//! compares the two outputs index by index. The validity decision (a reported
//! contact versus a `None` slot) must match exactly; when both report a contact,
//! the sphere and plane indices must match exactly and the normal, depth, and
//! point to within a tight tolerance. This path carries no square root or
//! reciprocal, so the only perturbation is fused-multiply-add reassociation in
//! the dot product, well under the tolerance. The scenes use clear penetrations
//! and clear gaps (never a grazing `s ~= r` boundary), so the tolerance can
//! never flip a validity flag.
//!
//! On a headless host with no `wgpu` adapter the tests skip (with a printed
//! notice) instead of failing, so the suite stays green everywhere while still
//! exercising the full dispatch-and-readback on any machine with a real device.
//!
//! Provenance: textbook sphere-plane collision manifold. No Unreal Engine
//! source or derived code.

use glam::Vec3;
use prism_physics_gpu::{
    cpu_halfspace_narrowphase, GpuContext, GpuHalfspaceNarrowphase, Particle, Plane,
    SpherePlanePair,
};

/// Tolerance on the normal, depth, and point; the only inexact step on this
/// path is fused-multiply-add reassociation in the plane dot product.
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

/// Compares one `GPU` contact slot to the `CPU` twin's, allowing only the tight
/// float tolerance.
#[expect(
    clippy::print_stderr,
    reason = "surface the differing slot on a parity failure"
)]
fn assert_slot_matches(
    index: usize,
    want: Option<prism_physics_gpu::Contact>,
    got: Option<prism_physics_gpu::Contact>,
) {
    match (want, got) {
        (None, None) => {}
        (Some(w), Some(g)) => {
            assert_eq!(w.a, g.a, "slot {index}: sphere index differs");
            assert_eq!(w.b, g.b, "slot {index}: plane index differs");
            let dn = (w.normal - g.normal).length();
            let dd = (w.depth - g.depth).abs();
            let dp = (w.point - g.point).length();
            if dn > TOL || dd > TOL || dp > TOL {
                eprintln!(
                    "slot {index} diverged: normal {dn}, depth {dd}, point {dp}\n cpu {w:?}\n gpu {g:?}"
                );
            }
            assert!(dn <= TOL, "slot {index}: normal diverged by {dn}");
            assert!(dd <= TOL, "slot {index}: depth diverged by {dd}");
            assert!(dp <= TOL, "slot {index}: point diverged by {dp}");
        }
        (w, g) => panic!("slot {index}: validity mismatch: cpu {w:?} vs gpu {g:?}"),
    }
}

/// Runs both engines over the same scene and asserts slot-for-slot parity.
fn run_parity(
    ctx: &GpuContext,
    gpu: &GpuHalfspaceNarrowphase,
    spheres: &[Particle],
    planes: &[Plane],
    pairs: &[SpherePlanePair],
) {
    let want = cpu_halfspace_narrowphase(spheres, planes, pairs);
    let got = gpu.query(ctx, spheres, planes, pairs);
    assert_eq!(want.len(), got.len(), "slot count differs");
    for (i, (w, g)) in want.into_iter().zip(got).enumerate() {
        assert_slot_matches(i, w, g);
    }
}

#[test]
#[expect(
    clippy::print_stderr,
    reason = "the skip notice must reach the test log on hosts without a wgpu adapter"
)]
fn gpu_halfspace_matches_cpu_golden() {
    let Some(ctx) = GpuContext::try_headless() else {
        eprintln!("skipping GPU halfspace parity: no wgpu adapter on this host");
        return;
    };
    let gpu = GpuHalfspaceNarrowphase::new(&ctx);

    // A hand-built scene mixing clear penetrations and clear gaps against the
    // ground and a slanted wall, so both the hit and miss branches run.
    let ground = Plane::new(Vec3::Y, 0.0);
    let inv = 1.0 / (2.0f32).sqrt();
    let wall = Plane::new(Vec3::new(inv, 0.0, inv), -1.0);
    let planes = [ground, wall];
    let spheres = [
        Particle::new(Vec3::new(0.0, 0.5, 0.0), 1.0), // dips into ground
        Particle::new(Vec3::new(0.0, 5.0, 0.0), 1.0), // floats clear of ground
        Particle::new(Vec3::new(-1.0, 2.0, -1.0), 1.5), // pushes into the wall
    ];
    let pairs = [
        SpherePlanePair::new(0, 0),
        SpherePlanePair::new(1, 0),
        SpherePlanePair::new(2, 1),
        SpherePlanePair::new(2, 0),
    ];
    run_parity(&ctx, &gpu, &spheres, &planes, &pairs);

    // A larger randomised scene: spheres scattered around the ground plane, each
    // tested against it. The vertical span straddles the surface so roughly half
    // penetrate, exercising both branches at batch scale. Radii and heights are
    // chosen to stay clear of the grazing boundary.
    let mut rng = Rng::new(0x5eed_1234_abcd_ef01);
    let mut big_spheres = Vec::new();
    let mut big_pairs = Vec::new();
    for i in 0..256u32 {
        let x = rng.coord(-20.0, 40.0);
        let y = rng.coord(-3.0, 6.0);
        let z = rng.coord(-20.0, 40.0);
        let r = rng.coord(0.4, 0.8);
        // Nudge any sphere out of the grazing band |y - r| < 0.05 so the
        // validity flag is unambiguous on both paths.
        let y = if (y - r).abs() < 0.05 { y + 0.2 } else { y };
        big_spheres.push(Particle::new(Vec3::new(x, y, z), r));
        big_pairs.push(SpherePlanePair::new(i, 0));
    }
    run_parity(&ctx, &gpu, &big_spheres, &[ground], &big_pairs);
}
