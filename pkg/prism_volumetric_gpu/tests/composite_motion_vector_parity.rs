//! Real-device parity for the composite-motion-vector twin:
//! [`GpuCompositeMotionVector`] must reproduce the `CPU` golden
//! [`composite_motion_vector`](prism_render_architecture::volumetric::temporal::composite_motion_vector)
//! across a deterministic spread of advection and camera-motion pairs.
//!
//! The tests skip (with a printed notice) when the host has no `wgpu` adapter,
//! so the suite stays green everywhere while still exercising the full
//! dispatch-and-readback on any real device. The kernel is portable core-WGSL,
//! so it needs no optional device feature.
//!
//! # Parity criterion
//!
//! The kernel evaluates the same closed-form vector add as the `CPU`, so each
//! component is asserted to within `abs_diff < 1e-6` or `rel_diff < 1e-5` —
//! tight enough to fail a wrong port (a dropped source, a swapped component).
//! The scenes also assert the composite equals the plain component-wise sum, so
//! a degenerate kernel could not pass.
//!
//! Provenance: standard screen-space motion-vector composition; no Unreal Engine
//! source or derived code.

use prism_render_architecture::volumetric::temporal::composite_motion_vector;
use prism_render_architecture::volumetric::Vec2;
use prism_volumetric_gpu::{CompositeMotionVectorQuery, GpuCompositeMotionVector, GpuContext};

/// Asserts every `gpu` vector matches the `CPU` golden to within the documented
/// tolerance.
fn assert_parity(
    queries: &[CompositeMotionVectorQuery],
    gpu: &[prism_volumetric_gpu::MotionVector],
) {
    assert_eq!(gpu.len(), queries.len(), "one vector per query");
    for (i, q) in queries.iter().enumerate() {
        let advection = Vec2::new(q.advection_x, q.advection_y);
        let camera = Vec2::new(q.camera_x, q.camera_y);
        let exp = composite_motion_vector(advection, camera);
        let got = gpu[i];

        for (axis, g, e) in [("x", got.x, exp.x), ("y", got.y, exp.y)] {
            let abs_diff = (g - e).abs();
            let rel_diff = abs_diff / e.abs().max(1e-6);
            assert!(
                abs_diff < 1e-6 || rel_diff < 1e-5,
                "motion-vector {axis} mismatch for query {i}: gpu {g}, cpu {e} \
                 (abs {abs_diff}, rel {rel_diff})"
            );
        }

        // The composite is exactly the component-wise sum.
        assert!(
            (got.x - (q.advection_x + q.camera_x)).abs() < 1e-6
                && (got.y - (q.advection_y + q.camera_y)).abs() < 1e-6,
            "composite must equal the component-wise sum for query {i}: {got:?}"
        );
    }
}

#[test]
#[expect(
    clippy::print_stderr,
    reason = "the skip notice must reach the test log on hosts without a wgpu adapter"
)]
fn gpu_composite_motion_vector_matches_cpu_golden() {
    let Some(ctx) = GpuContext::try_headless() else {
        eprintln!("skipping composite-motion-vector parity: no wgpu adapter on this host");
        return;
    };
    let gpu_kernel = GpuCompositeMotionVector::new(&ctx);

    // Hand-picked scenes, then a deterministic sweep.
    let mut queries: Vec<CompositeMotionVectorQuery> = vec![
        // Zero motion.
        CompositeMotionVectorQuery {
            advection_x: 0.0,
            advection_y: 0.0,
            camera_x: 0.0,
            camera_y: 0.0,
        },
        // Advection only.
        CompositeMotionVectorQuery {
            advection_x: 1.5,
            advection_y: -0.75,
            camera_x: 0.0,
            camera_y: 0.0,
        },
        // Camera only.
        CompositeMotionVectorQuery {
            advection_x: 0.0,
            advection_y: 0.0,
            camera_x: -2.25,
            camera_y: 3.0,
        },
        // Cancelling components.
        CompositeMotionVectorQuery {
            advection_x: 4.0,
            advection_y: -1.0,
            camera_x: -4.0,
            camera_y: 1.0,
        },
    ];
    for k in 0..=200 {
        let t = (k as f32) * 0.01;
        queries.push(CompositeMotionVectorQuery {
            advection_x: t * 2.0 - 1.0,
            advection_y: 1.0 - t,
            camera_x: -t,
            camera_y: t * 0.5,
        });
    }

    let gpu = gpu_kernel.eval(&ctx, &queries);
    assert_parity(&queries, &gpu);

    // Cancelling components yield the zero vector.
    assert!(
        gpu[3].x.abs() < 1e-6 && gpu[3].y.abs() < 1e-6,
        "cancelling components yield zero motion: {:?}",
        gpu[3]
    );
}

#[test]
fn empty_queries_yields_no_results() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu_kernel = GpuCompositeMotionVector::new(&ctx);
    let out = gpu_kernel.eval(&ctx, &[]);
    assert!(out.is_empty(), "an empty query slice yields no values");
}
