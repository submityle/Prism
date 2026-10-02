//! Real-device parity: the `GPU` sphere-versus-heightfield narrow-phase kernel
//! must reproduce the `CPU` golden twin's contact manifolds slot for slot.
//!
//! Each test hands a batch of candidate `(sphere, heightfield)` pairs to both
//! [`GpuSphereHeightfieldNarrowphase::query`] and
//! [`cpu_sphere_heightfield_narrowphase`] and compares the two outputs index by
//! index. The validity decision (a reported contact versus a `None` slot) must
//! match exactly; when both report a contact, the sphere and heightfield indices
//! must match exactly and the normal, depth, and point within a tight tolerance,
//! since the only inexact steps are the square root, the floor, and the handful
//! of barycentric reciprocals in the closest-point clamps. The scenes are built
//! with clear overlaps and clear gaps (never a grazing `dist ~= r` boundary, and
//! never a symmetric tie that would make the deepest contact non-unique), so the
//! tiny floating-point tolerance can never flip a validity flag or the reduction
//! winner. Flat-cell interior, cross-cell footprints, a varying field, a raised
//! second field, and clearly separated and off-grid cases are exercised so the
//! cell-range mapping, the closest-point cascade, and the deepest-contact
//! reduction all run on each path.
//!
//! On a headless host with no `wgpu` adapter the tests skip (with a printed
//! notice) instead of failing, so the suite stays green everywhere while still
//! exercising the full dispatch-and-readback on any machine with a real device.
//!
//! Provenance: closest-point-on-triangle is the Voronoi-region method from
//! Christer Ericson, *Real-Time Collision Detection* (2004), section 5.1.5; the
//! heightfield cell triangulation is textbook. No Unreal Engine source or
//! derived code.

use glam::Vec3;
use prism_physics_gpu::{
    cpu_sphere_heightfield_narrowphase, Contact, GpuContext, GpuSphereHeightfieldNarrowphase,
    Heightfield, HeightfieldSpherePair, Particle,
};

/// Tolerance on the normal, depth, and point; the only inexact steps are the
/// square root, the floor, and the barycentric reciprocals, all well under this
/// bound over the test scene scales.
const TOL: f32 = 1e-4;

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
            assert_eq!(w.a, g.a, "slot {index}: sphere index must match");
            assert_eq!(w.b, g.b, "slot {index}: heightfield index must match");
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
fn check(spheres: &[Particle], fields: &[Heightfield], pairs: &[HeightfieldSpherePair]) {
    let cpu = cpu_sphere_heightfield_narrowphase(spheres, fields, pairs);

    let Some(ctx) = GpuContext::try_headless() else {
        eprintln!("skipping GPU sphere-heightfield parity: no wgpu adapter on this host");
        return;
    };
    let narrow = GpuSphereHeightfieldNarrowphase::new(&ctx);
    let gpu = narrow.query(&ctx, spheres, fields, pairs);

    assert_eq!(
        gpu.len(),
        cpu.len(),
        "GPU must emit one slot per pair like the CPU twin"
    );
    for (index, (want, got)) in cpu.into_iter().zip(gpu).enumerate() {
        assert_slot_matches(index, want, got);
    }
}

/// A flat 4x4-sample field (3x3 cells) at `y = 0` with unit spacing, origin at
/// the world origin. Samples span `x, z` in `[0, 3]`.
fn flat_field() -> Heightfield {
    Heightfield::new(4, 4, 1.0, Vec3::ZERO, vec![0.0; 16])
}

/// A gently varying 4x4-sample field so the closest-point cascade runs on real
/// (non-axis-aligned) triangle faces, not just flat quads.
fn bumpy_field() -> Heightfield {
    #[rustfmt::skip]
    let heights = vec![
        0.0, 0.2, 0.1, 0.0,
        0.1, 0.6, 0.4, 0.1,
        0.2, 0.5, 0.7, 0.2,
        0.0, 0.1, 0.2, 0.0,
    ];
    Heightfield::new(4, 4, 1.0, Vec3::ZERO, heights)
}

#[test]
fn flat_field_cases_agree_slot_for_slot() {
    // A clear interior hit, a clear gap high above, a footprint straddling a
    // seam between two cells, and a footprint entirely off the -X edge. Every
    // case is far from the touching boundary, so the validity flag is
    // unambiguous.
    let field = flat_field();
    let spheres = [
        Particle::new(Vec3::new(0.5, 0.35, 0.5), 0.5), // interior of cell (0,0)
        Particle::new(Vec3::new(1.5, 4.0, 1.5), 0.5),  // clear gap above
        Particle::new(Vec3::new(1.0, 0.3, 1.6), 0.6),  // straddles the x = 1 seam
        Particle::new(Vec3::new(-4.0, 0.0, 1.5), 0.5), // off the grid
    ];
    let fields = [field];
    let pairs = [
        HeightfieldSpherePair::new(0, 0),
        HeightfieldSpherePair::new(1, 0),
        HeightfieldSpherePair::new(2, 0),
        HeightfieldSpherePair::new(3, 0),
    ];
    check(&spheres, &fields, &pairs);
}

#[test]
fn bumpy_field_cases_agree_slot_for_slot() {
    // Spheres resting over several interior cells of the varying field, so the
    // closest point lands on tilted faces and edges and the deepest-contact
    // reduction has to choose among adjacent cells. Each centre is lowered a
    // clear margin into the surface and a clear margin away from any symmetric
    // tie, so the winner is unique on both paths.
    let field = bumpy_field();
    let spheres = [
        Particle::new(Vec3::new(1.5, 0.95, 1.5), 0.5), // over the tall bump at (2,2)
        Particle::new(Vec3::new(0.7, 0.55, 1.2), 0.4), // over a sloped shoulder
        Particle::new(Vec3::new(2.3, 0.6, 2.2), 0.45), // over the far ridge
        Particle::new(Vec3::new(1.5, 3.0, 1.5), 0.5),  // clear gap above
    ];
    let fields = [field];
    let pairs = [
        HeightfieldSpherePair::new(0, 0),
        HeightfieldSpherePair::new(1, 0),
        HeightfieldSpherePair::new(2, 0),
        HeightfieldSpherePair::new(3, 0),
    ];
    check(&spheres, &fields, &pairs);
}

#[test]
fn multiple_fields_preserve_indices() {
    // Two fields with different origins: pairs against each must carry the right
    // heightfield index and read from the right slice of the shared height
    // buffer. The raised field sits one unit higher, so a sphere clearing the
    // flat field still rests on the raised one.
    let flat = flat_field();
    let raised = Heightfield::new(4, 4, 1.0, Vec3::new(0.0, 1.0, 0.0), vec![0.0; 16]);
    let fields = [flat, raised];
    let spheres = [
        Particle::new(Vec3::new(1.5, 0.3, 1.5), 0.5), // on the flat field (index 0)
        Particle::new(Vec3::new(1.5, 1.3, 1.5), 0.5), // on the raised field (index 1)
        Particle::new(Vec3::new(1.5, 0.3, 1.5), 0.5), // clears the raised field
    ];
    let pairs = [
        HeightfieldSpherePair::new(0, 0),
        HeightfieldSpherePair::new(1, 1),
        HeightfieldSpherePair::new(2, 1),
    ];
    check(&spheres, &fields, &pairs);
}

#[test]
fn empty_batch_returns_empty() {
    let spheres = [Particle::new(Vec3::ZERO, 1.0)];
    let fields = [flat_field()];
    check(&spheres, &fields, &[]);
}
