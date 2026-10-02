//! Real-device parity for the `GPU` multi-layer garment coupling pass against
//! its `CPU` golden twin.
//!
//! Every test acquires a headless device with `GpuContext::try_headless()` and
//! skips cleanly when no adapter is available (for example inside a sandbox),
//! so the suite is a no-op rather than a failure off a real `GPU`.
//!
//! The `CPU` twin ([`cpu_cloth_layer_coupling`]) delegates to
//! `prism_physics_core`'s `resolve_layer_coupling_jacobi`, while the kernel
//! reimplements the same oriented-plane / radial-fallback separation in `WGSL`
//! over a host-built per-particle cross-layer adjacency. The only divergence is
//! a few `ULP` in `sqrt`/division, so parity is checked within a tight
//! tolerance.
//!
//! Provenance: the layer-number stacking constraint and inverse-mass-weighted
//! separation are standard position-based dynamics; the Jacobi own-slot
//! accumulate is standard parallel position-based dynamics. No Unreal Engine
//! source or derived code.

use glam::Vec3;
use prism_physics_core::LayerParams;
use prism_physics_gpu::context::GpuContext;
use prism_physics_gpu::{cpu_cloth_layer_coupling, GpuClothLayerCoupling};

/// Absolute/relative tolerance for position parity.
const TOL: f32 = 1.0e-4;

#[expect(
    clippy::print_stderr,
    reason = "the suite is a deliberate no-op when no GPU adapter is present"
)]
fn headless() -> Option<GpuContext> {
    match GpuContext::try_headless() {
        Some(ctx) => Some(ctx),
        None => {
            eprintln!("skipping cloth layer-coupling parity: no GPU adapter available");
            None
        }
    }
}

/// Asserts two vectors agree within [`TOL`] (combined absolute/relative).
#[track_caller]
fn assert_vec_close(a: Vec3, b: Vec3, what: &str) {
    let scale = a.length().max(b.length()).max(1.0);
    assert!(
        a.distance(b) <= TOL * scale,
        "{what}: cpu={a:?} gpu={b:?} (dist {})",
        a.distance(b)
    );
}

/// Runs both paths over the same inputs and asserts full parity of the applied
/// positions.
#[track_caller]
fn assert_parity(
    ctx: &GpuContext,
    kernel: &GpuClothLayerCoupling,
    positions: &[Vec3],
    inverse_masses: &[f32],
    layer_of: &[u32],
    normals: &[Vec3],
    params: LayerParams,
) {
    let cpu = cpu_cloth_layer_coupling(positions, inverse_masses, layer_of, normals, params);
    let gpu = kernel.solve(ctx, positions, inverse_masses, layer_of, normals, params);
    assert_eq!(cpu.len(), gpu.len(), "position count parity");
    for (i, (c, g)) in cpu.iter().zip(gpu.iter()).enumerate() {
        assert_vec_close(*c, *g, &format!("position[{i}]"));
    }
}

/// Advances both paths for several Jacobi iterations and asserts parity at each
/// step, so accumulated reduction-order drift (if any) would show up.
#[track_caller]
fn assert_parity_iterated(
    ctx: &GpuContext,
    kernel: &GpuClothLayerCoupling,
    positions: &[Vec3],
    inverse_masses: &[f32],
    layer_of: &[u32],
    normals: &[Vec3],
    params: LayerParams,
    steps: usize,
) {
    let mut cpu = positions.to_vec();
    let mut gpu = positions.to_vec();
    for _ in 0..steps {
        cpu = cpu_cloth_layer_coupling(&cpu, inverse_masses, layer_of, normals, params);
        gpu = kernel.solve(ctx, &gpu, inverse_masses, layer_of, normals, params);
        for (i, (c, g)) in cpu.iter().zip(gpu.iter()).enumerate() {
            let scale = c.length().max(g.length()).max(1.0);
            assert!(
                c.distance(*g) <= 1.0e-3 * scale,
                "iterated position[{i}]: cpu={c:?} gpu={g:?}"
            );
        }
    }
}

fn params() -> LayerParams {
    LayerParams {
        thickness: 0.1,
        cell_size: 0.2,
    }
}

#[test]
fn oriented_cross_layer_pair_parity() {
    let Some(ctx) = headless() else {
        return;
    };
    let kernel = GpuClothLayerCoupling::new(&ctx);
    // Inner (layer 0) above outer (layer 1); the outer is pushed to +normal.
    let positions = [Vec3::ZERO, Vec3::new(0.0, -0.05, 0.0)];
    let im = [1.0, 1.0];
    let layer_of = [0u32, 1u32];
    let normals = [Vec3::new(0.0, 1.0, 0.0), Vec3::ZERO];
    assert_parity(&ctx, &kernel, &positions, &im, &layer_of, &normals, params());
}

#[test]
fn radial_fallback_pair_parity() {
    let Some(ctx) = headless() else {
        return;
    };
    let kernel = GpuClothLayerCoupling::new(&ctx);
    // Inner normal is zero, so the pass falls back to a radial push.
    let positions = [Vec3::ZERO, Vec3::new(0.02, 0.0, 0.0)];
    let im = [1.0, 1.0];
    let layer_of = [0u32, 1u32];
    let normals = [Vec3::ZERO, Vec3::ZERO];
    assert_parity(&ctx, &kernel, &positions, &im, &layer_of, &normals, params());
}

#[test]
fn asymmetric_mass_pair_parity() {
    let Some(ctx) = headless() else {
        return;
    };
    let kernel = GpuClothLayerCoupling::new(&ctx);
    let positions = [Vec3::ZERO, Vec3::new(0.0, -0.04, 0.0)];
    let im = [0.25, 1.0];
    let layer_of = [0u32, 1u32];
    let normals = [Vec3::new(0.0, 1.0, 0.0), Vec3::ZERO];
    assert_parity(&ctx, &kernel, &positions, &im, &layer_of, &normals, params());
}

#[test]
fn pinned_inner_pair_parity() {
    let Some(ctx) = headless() else {
        return;
    };
    let kernel = GpuClothLayerCoupling::new(&ctx);
    let positions = [Vec3::ZERO, Vec3::new(0.0, -0.05, 0.0)];
    let im = [0.0, 1.0];
    let layer_of = [0u32, 1u32];
    let normals = [Vec3::new(0.0, 1.0, 0.0), Vec3::ZERO];
    assert_parity(&ctx, &kernel, &positions, &im, &layer_of, &normals, params());
}

#[test]
fn coincident_radial_pair_parity() {
    let Some(ctx) = headless() else {
        return;
    };
    let kernel = GpuClothLayerCoupling::new(&ctx);
    // Coincident with no normal: both paths must separate along +X.
    let positions = [Vec3::ZERO, Vec3::ZERO];
    let im = [1.0, 1.0];
    let layer_of = [0u32, 1u32];
    let normals = [Vec3::ZERO, Vec3::ZERO];
    assert_parity(&ctx, &kernel, &positions, &im, &layer_of, &normals, params());
}

#[test]
fn same_layer_pair_is_untouched_parity() {
    let Some(ctx) = headless() else {
        return;
    };
    let kernel = GpuClothLayerCoupling::new(&ctx);
    // Same layer: intra-layer contact is self-collision's job, not this tier.
    let positions = [Vec3::ZERO, Vec3::new(0.0, 0.01, 0.0)];
    let im = [1.0, 1.0];
    let layer_of = [2u32, 2u32];
    let normals = [Vec3::new(0.0, 1.0, 0.0), Vec3::ZERO];
    assert_parity(&ctx, &kernel, &positions, &im, &layer_of, &normals, params());
}

#[test]
fn separated_pair_is_untouched_parity() {
    let Some(ctx) = headless() else {
        return;
    };
    let kernel = GpuClothLayerCoupling::new(&ctx);
    // Within a shared neighborhood but farther apart than thickness along the
    // normal: both paths must leave the pair exactly where it started.
    let positions = [Vec3::ZERO, Vec3::new(0.0, -0.5, 0.0)];
    let im = [1.0, 1.0];
    let layer_of = [0u32, 1u32];
    let normals = [Vec3::new(0.0, 1.0, 0.0), Vec3::ZERO];
    assert_parity(&ctx, &kernel, &positions, &im, &layer_of, &normals, params());
}

#[test]
fn cluster_parity() {
    let Some(ctx) = headless() else {
        return;
    };
    let kernel = GpuClothLayerCoupling::new(&ctx);
    // One inner particle penetrated by several outer-layer particles, so the
    // inner's own-slot sum folds multiple halves in the golden's order.
    let positions = [
        Vec3::new(0.0, 0.0, 0.0),
        Vec3::new(0.0, -0.03, 0.0),
        Vec3::new(0.02, -0.02, 0.0),
        Vec3::new(-0.02, -0.025, 0.0),
    ];
    let im = [1.0, 1.0, 1.0, 1.0];
    let layer_of = [0u32, 1u32, 1u32, 1u32];
    let normals = [
        Vec3::new(0.0, 1.0, 0.0),
        Vec3::new(0.0, 1.0, 0.0),
        Vec3::new(0.0, 1.0, 0.0),
        Vec3::new(0.0, 1.0, 0.0),
    ];
    assert_parity(&ctx, &kernel, &positions, &im, &layer_of, &normals, params());
}

#[test]
fn three_layer_stack_parity() {
    let Some(ctx) = headless() else {
        return;
    };
    let kernel = GpuClothLayerCoupling::new(&ctx);
    // Three stacked layers (0/1/2) all within thickness along the shared
    // normal, so both cross-layer boundaries resolve in one pass.
    let positions = [
        Vec3::new(0.0, 0.0, 0.0),
        Vec3::new(0.0, -0.04, 0.0),
        Vec3::new(0.0, -0.08, 0.0),
    ];
    let im = [1.0, 1.0, 1.0];
    let layer_of = [0u32, 1u32, 2u32];
    let normals = [
        Vec3::new(0.0, 1.0, 0.0),
        Vec3::new(0.0, 1.0, 0.0),
        Vec3::new(0.0, 1.0, 0.0),
    ];
    assert_parity(&ctx, &kernel, &positions, &im, &layer_of, &normals, params());
}

#[test]
fn iterated_cluster_parity() {
    let Some(ctx) = headless() else {
        return;
    };
    let kernel = GpuClothLayerCoupling::new(&ctx);
    let positions = [
        Vec3::new(0.0, 0.0, 0.0),
        Vec3::new(0.0, -0.03, 0.0),
        Vec3::new(0.02, -0.02, 0.0),
    ];
    let im = [1.0, 1.0, 1.0];
    let layer_of = [0u32, 1u32, 1u32];
    let normals = [
        Vec3::new(0.0, 1.0, 0.0),
        Vec3::new(0.0, 1.0, 0.0),
        Vec3::new(0.0, 1.0, 0.0),
    ];
    assert_parity_iterated(&ctx, &kernel, &positions, &im, &layer_of, &normals, params(), 32);
}

#[test]
fn empty_neighborhood_is_untouched_parity() {
    let Some(ctx) = headless() else {
        return;
    };
    let kernel = GpuClothLayerCoupling::new(&ctx);
    // Far apart: the host broad phase yields no neighbors and both paths no-op.
    let positions = [Vec3::ZERO, Vec3::new(100.0, 0.0, 0.0)];
    let im = [1.0, 1.0];
    let layer_of = [0u32, 1u32];
    let normals = [Vec3::new(0.0, 1.0, 0.0), Vec3::ZERO];
    assert_parity(&ctx, &kernel, &positions, &im, &layer_of, &normals, params());
}

#[test]
fn disabled_params_are_untouched_parity() {
    let Some(ctx) = headless() else {
        return;
    };
    let kernel = GpuClothLayerCoupling::new(&ctx);
    let positions = [Vec3::ZERO, Vec3::new(0.0, -0.05, 0.0)];
    let im = [1.0, 1.0];
    let layer_of = [0u32, 1u32];
    let normals = [Vec3::new(0.0, 1.0, 0.0), Vec3::ZERO];
    assert_parity(
        &ctx,
        &kernel,
        &positions,
        &im,
        &layer_of,
        &normals,
        LayerParams {
            thickness: 0.0,
            cell_size: 0.2,
        },
    );
}
