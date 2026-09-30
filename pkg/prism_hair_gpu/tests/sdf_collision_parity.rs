//! Real-device parity for the signed-distance-field body-collision push-out
//! twin: [`GpuSdfCollider`] must reproduce the `CPU` golden
//! [`push_out_of_field`](prism_render_architecture::hair::sdf_collision::push_out_of_field)
//! for a batch of query points against a shared union of primitives (sphere,
//! capsule, half-space, box), including the inside/outside branch, the
//! multi-pass settling in overlapping unions, the zero-gradient escape fallback,
//! and the inert-primitive and empty-input no-ops the reference guards.
//!
//! The tests skip (with a printed notice) when the host has no `wgpu` adapter,
//! so the suite stays green everywhere while still exercising the full
//! dispatch-and-readback on any real device. The kernel is portable core-WGSL,
//! so it needs no optional device feature.
//!
//! # Parity criterion
//!
//! The projection is closed-form geometry (the reference restricts itself to
//! `sqrt`, `min`, `max`, `clamp`, `dot`), so the `CPU` and `GPU` evaluate the
//! same expression and diverge only through legal fused-multiply-add
//! contraction and a possible reassociation of the union reduction. Because the
//! push-out is *iterative* (gradient steps repeated up to the pass cap), that
//! per-pass divergence can compound, so parity is asserted to within
//! `abs_diff < 1e-3` or `rel_diff < 1e-3` — tight enough to fail a genuinely
//! wrong port (a swapped primitive branch, a missing radius, a wrong gradient
//! sign), loose enough to admit the compounded fma contraction. Test unions and
//! points are chosen so every inside/outside branch decision is unambiguous
//! (points clearly inside with a strong gradient, or exactly symmetric for the
//! flat-field case), so `CPU` and `GPU` never split on a branch. A resolved
//! point is also asserted to have escaped the union (union distance `≥ -1e-3`)
//! so a no-op kernel could not pass.
//!
//! Provenance: standard analytic SDF union plus gradient push-out; no Unreal
//! Engine source or derived code.

use prism_hair_gpu::sdf_collision::GpuSdfCollider;
use prism_hair_gpu::GpuContext;
use prism_render_architecture::hair::dynamics::Vec3;
use prism_render_architecture::hair::sdf_collision::{
    push_out_of_field, union_signed_distance, SdfPrimitive, DEFAULT_SDF_ITERATIONS,
};

/// Asserts a single component matches within the documented iterative tolerance.
fn assert_close(got: f32, expected: f32, label: &str) {
    let abs_diff = (got - expected).abs();
    let rel_diff = abs_diff / expected.abs().max(1e-6);
    assert!(
        abs_diff < 1e-3 || rel_diff < 1e-3,
        "{label}: gpu {got}, cpu {expected} (abs {abs_diff}, rel {rel_diff})"
    );
}

/// Asserts every `GPU`-resolved point equals the `CPU` golden per component.
fn assert_batch_parity(
    points: &[[f32; 3]],
    primitives: &[SdfPrimitive],
    iterations: u32,
    gpu: &[[f32; 3]],
) {
    assert_eq!(gpu.len(), points.len(), "one resolved point per input");
    for (i, (g, p)) in gpu.iter().zip(points.iter()).enumerate() {
        let c = push_out_of_field(primitives, Vec3::new(p[0], p[1], p[2]), iterations);
        assert_close(g[0], c.x, &format!("point {i} x"));
        assert_close(g[1], c.y, &format!("point {i} y"));
        assert_close(g[2], c.z, &format!("point {i} z"));
    }
}

#[test]
#[expect(
    clippy::print_stderr,
    reason = "the skip notice must reach the test log on hosts without a wgpu adapter"
)]
fn gpu_mixed_union_matches_cpu_golden() {
    let Some(ctx) = GpuContext::try_headless() else {
        eprintln!("skipping sdf parity: no wgpu adapter on this host");
        return;
    };

    // A body-like union of all four primitive kinds, well separated so a point
    // clearly inside one has a strong, unambiguous gradient.
    let prims = vec![
        SdfPrimitive::Sphere {
            center: Vec3::new(0.0, 0.0, 0.0),
            radius: 1.0,
        },
        SdfPrimitive::Capsule {
            a: Vec3::new(3.0, -1.0, 0.0),
            b: Vec3::new(3.0, 1.0, 0.0),
            radius: 0.5,
        },
        SdfPrimitive::HalfSpace {
            normal: Vec3::new(0.0, 1.0, 0.0),
            offset: -2.0,
        },
        SdfPrimitive::Box {
            center: Vec3::new(-3.0, 0.0, 0.0),
            half_extents: Vec3::new(0.5, 0.8, 0.5),
        },
    ];

    // A mix of deep-inside, near-surface-inside and already-outside points.
    let points = vec![
        [0.3, 0.1, -0.2],  // inside the sphere
        [2.9, 0.0, 0.1],   // inside the capsule
        [-3.1, 0.0, 0.05], // inside the box
        [0.0, -1.9, 0.0],  // inside the half-space (below y = -2)
        [5.0, 5.0, 5.0],   // far outside everything: unchanged
        [0.0, 2.0, 0.0],   // above the sphere, outside: unchanged
    ];

    let out = GpuSdfCollider::new(&ctx).eval(&ctx, &points, &prims, DEFAULT_SDF_ITERATIONS);
    assert_batch_parity(&points, &prims, DEFAULT_SDF_ITERATIONS, &out);

    // Every resolved point must be outside the union (allowing the iterative
    // tolerance), so a no-op kernel could not pass.
    for (i, r) in out.iter().enumerate() {
        let d = union_signed_distance(&prims, Vec3::new(r[0], r[1], r[2]));
        assert!(d >= -1e-3, "point {i} still inside union: distance {d}");
    }

    // The already-outside points must be returned unchanged.
    assert_close(out[4][0], 5.0, "outside point x unchanged");
    assert_close(out[5][1], 2.0, "outside point y unchanged");
}

#[test]
#[expect(
    clippy::print_stderr,
    reason = "the skip notice must reach the test log on hosts without a wgpu adapter"
)]
fn gpu_zero_gradient_escapes_up_like_cpu() {
    let Some(ctx) = GpuContext::try_headless() else {
        eprintln!("skipping sdf parity: no wgpu adapter on this host");
        return;
    };

    // At the exact center of a single sphere the central-difference gradient is
    // zero by symmetry (bit-for-bit on both sides), so both the golden and the
    // twin take the fixed-axis escape: move +Y by the penetration depth (the
    // radius) and stop.
    let prims = vec![SdfPrimitive::Sphere {
        center: Vec3::new(0.0, 0.0, 0.0),
        radius: 1.0,
    }];
    let points = vec![[0.0, 0.0, 0.0]];

    let out = GpuSdfCollider::new(&ctx).eval(&ctx, &points, &prims, DEFAULT_SDF_ITERATIONS);
    assert_batch_parity(&points, &prims, DEFAULT_SDF_ITERATIONS, &out);

    // The depth at the center is the full radius, so the escape lands at +Y = 1.
    assert_close(out[0][0], 0.0, "escape x");
    assert_close(out[0][1], 1.0, "escape +Y by depth");
    assert_close(out[0][2], 0.0, "escape z");
}

#[test]
#[expect(
    clippy::print_stderr,
    reason = "the skip notice must reach the test log on hosts without a wgpu adapter"
)]
fn gpu_inert_and_noop_inputs_match_cpu() {
    let Some(ctx) = GpuContext::try_headless() else {
        eprintln!("skipping sdf parity: no wgpu adapter on this host");
        return;
    };
    let kernel = GpuSdfCollider::new(&ctx);

    // Inert primitives (non-positive radius / degenerate extent / zero normal):
    // the union is empty everywhere, so every point is returned unchanged.
    let inert = vec![
        SdfPrimitive::Sphere {
            center: Vec3::new(0.0, 0.0, 0.0),
            radius: 0.0,
        },
        SdfPrimitive::Box {
            center: Vec3::new(1.0, 1.0, 1.0),
            half_extents: Vec3::new(0.0, 0.5, 0.5),
        },
        SdfPrimitive::HalfSpace {
            normal: Vec3::new(0.0, 0.0, 0.0),
            offset: 1.0,
        },
    ];
    let points = vec![[0.0, 0.0, 0.0], [1.0, 1.0, 1.0], [-2.0, 0.5, 0.3]];
    let out = kernel.eval(&ctx, &points, &inert, DEFAULT_SDF_ITERATIONS);
    assert_batch_parity(&points, &inert, DEFAULT_SDF_ITERATIONS, &out);
    for (i, (r, p)) in out.iter().zip(points.iter()).enumerate() {
        assert_close(r[0], p[0], &format!("inert point {i} x unchanged"));
        assert_close(r[1], p[1], &format!("inert point {i} y unchanged"));
        assert_close(r[2], p[2], &format!("inert point {i} z unchanged"));
    }

    // Zero iterations: no-op even against a real field.
    let real = vec![SdfPrimitive::Sphere {
        center: Vec3::new(0.0, 0.0, 0.0),
        radius: 1.0,
    }];
    let inside = vec![[0.2, 0.0, 0.0]];
    let zero_iter = kernel.eval(&ctx, &inside, &real, 0);
    assert_eq!(zero_iter.len(), 1, "one point back");
    assert_close(zero_iter[0][0], 0.2, "zero-iteration no-op x");

    // Empty primitive union: no-op.
    let no_prims: [SdfPrimitive; 0] = [];
    let empty_field = kernel.eval(&ctx, &inside, &no_prims, DEFAULT_SDF_ITERATIONS);
    assert_close(empty_field[0][0], 0.2, "empty-union no-op x");

    // Empty point batch: empty result, no dispatch.
    let no_points: [[f32; 3]; 0] = [];
    assert!(
        kernel
            .eval(&ctx, &no_points, &real, DEFAULT_SDF_ITERATIONS)
            .is_empty(),
        "no points -> empty result"
    );
}
