//! Real-device parity for the `3DGS` / `EWA` Gaussian-splat projection twin:
//! [`GpuGaussianSplat`](prism_volumetric_gpu::gaussian_splat::GpuGaussianSplat)
//! must reproduce the `CPU` golden
//! [`gaussian_splat`](prism_render_architecture::particle::gaussian_splat)
//! projection chain
//! ([`Gaussian3d::cov3_from_scale_quat`](prism_render_architecture::particle::gaussian_splat::Gaussian3d::cov3_from_scale_quat)
//! → [`project_to_2d`](prism_render_architecture::particle::gaussian_splat::project_to_2d)
//! → [`conic_from_cov2`](prism_render_architecture::particle::gaussian_splat::conic_from_cov2)
//! → [`Conic2d::bounding_aabb`](prism_render_architecture::particle::gaussian_splat::Conic2d)
//! / [`Conic2d::bounding_tiles`](prism_render_architecture::particle::gaussian_splat::Conic2d))
//! across an empty batch, a single unit-covariance splat, a large anisotropic
//! covariance, the degenerate (un-invertible) footprint branch, a large random
//! batch compared field by field, and `AABB` / `tile`-range boundaries where
//! the box straddles the screen origin and the `tile` edge varies (including a
//! zero `tile_size`, which the reference treats as one).
//!
//! The tests skip (returning early) when the host has no `wgpu` adapter, so the
//! suite stays green everywhere while still exercising the full
//! dispatch-and-readback on any real device such as an Apple `M`-series `GPU`.
//! The kernel is portable core-`WGSL`, so it needs no optional device feature.
//!
//! # Parity criterion
//!
//! Each lane is a fixed, non-reorderable sequence of multiplies, adds, guarded
//! divides and `sqrt`s, so `CPU` and `GPU` evaluate the same closed form in the
//! same order. The continuous outputs (the six 3D covariance entries, the three
//! 2D covariance entries, the three conic coefficients, the conic center and
//! the four `AABB` corners) are compared with `abs_diff <= 1e-4` or
//! `rel_diff <= 1e-4` — loose enough to admit a legal fused multiply-add
//! contraction, yet tight enough to fail a wrong port (a transposed rotation, a
//! dropped covariance term, a missing depth clamp, a flipped conic sign, a
//! wrong `AABB` radius). The discrete outputs — whether the footprint exists
//! (the reference `None` branch) and the four integer `tile` indices — are
//! compared exactly, so a mis-floored or mis-clamped bin fails immediately.
//!
//! Provenance: standard `3DGS` / `EWA` splatting projection
//! (`Zwicker` et al. 2001; `Kerbl` et al. 2023); no third-party engine source
//! or derived code.

use prism_render_architecture::particle::gaussian_splat::{
    conic_from_cov2, project_to_2d, Gaussian3d,
};
use prism_volumetric_gpu::gaussian_splat::{
    GaussianSplatProjection, GpuGaussianSplat, GpuGaussianSplatQuery,
};
use prism_volumetric_gpu::GpuContext;

/// Absolute/relative parity bound for the continuous outputs. A `GPU` may fuse
/// a multiply-add the scalar reference leaves separate, and the projection
/// chain is long (rotation, covariance, Jacobian, inversion, `sqrt` radius), so
/// `1e-4` admits that accumulated legal slack while still failing a genuinely
/// wrong port.
const EPS: f32 = 1.0e-4;

/// Floor on the relative-error denominator so a near-zero expected value does
/// not inflate the relative error.
const REL_FLOOR: f32 = 1.0e-6;

/// Returns whether `a` and `b` agree within the absolute or relative bound.
fn close(a: f32, b: f32) -> bool {
    let abs = (a - b).abs();
    let rel = abs / a.abs().max(b.abs()).max(REL_FLOOR);
    abs <= EPS || rel <= EPS
}

/// A tiny integer linear-congruential generator; only integer and divide work,
/// so no transcendental appears. Returns a value in `[0, 1)`.
fn lcg(state: &mut u64) -> f32 {
    // Knuth multiplier / increment; the shift takes the high bits where the
    // generator mixes best.
    *state = state
        .wrapping_mul(6_364_136_223_846_793_005)
        .wrapping_add(1_442_695_040_888_963_407);
    let bits = (*state >> 40) as u32;
    // 24 usable mantissa bits mapped onto [0, 1).
    (bits & 0x00ff_ffff) as f32 / 16_777_216.0
}

/// Draws a scalar uniformly in `[lo, hi)` from the generator.
fn uniform(state: &mut u64, lo: f32, hi: f32) -> f32 {
    lo + lcg(state) * (hi - lo)
}

/// Builds a projection query from raw fields.
fn query(
    mean: [f32; 3],
    scale: [f32; 3],
    quat: [f32; 4],
    focal: [f32; 2],
    screen_center: [f32; 2],
) -> GpuGaussianSplatQuery {
    GpuGaussianSplatQuery {
        splat: Gaussian3d::new(mean, scale, quat, 1.0),
        focal,
        screen_center,
    }
}

/// Runs the `GPU` dispatch and compares every splat against the `CPU` golden
/// projection chain, returning the `GPU` results for any extra per-test
/// assertions.
fn check(
    ctx: &GpuContext,
    gpu: &GpuGaussianSplat,
    tile_size: u32,
    queries: &[GpuGaussianSplatQuery],
) -> Vec<GaussianSplatProjection> {
    let got = gpu.eval(ctx, tile_size, queries);
    assert_eq!(got.len(), queries.len(), "one projection per splat");

    for (lane, (g, q)) in got.iter().zip(queries.iter()).enumerate() {
        // CPU golden chain: 3D covariance → 2D covariance → optional footprint.
        let cov3 = q.splat.cov3_from_scale_quat();
        let cov2 = project_to_2d(cov3, q.splat.mean, q.focal);

        for (k, (&gc, &cc)) in g.cov3.iter().zip(cov3.iter()).enumerate() {
            assert!(close(gc, cc), "lane {lane} cov3[{k}]: gpu {gc} vs cpu {cc}");
        }
        for (k, (&gc, &cc)) in g.cov2.iter().zip(cov2.iter()).enumerate() {
            assert!(close(gc, cc), "lane {lane} cov2[{k}]: gpu {gc} vs cpu {cc}");
        }

        match conic_from_cov2(cov2) {
            None => {
                assert!(
                    g.footprint.is_none(),
                    "lane {lane}: cpu reports a degenerate footprint but gpu kept one"
                );
            }
            Some(mut conic) => {
                // The reference caller places the origin-centered conic at the
                // projected pixel position before deriving its bounds.
                conic.center = q.screen_center;
                let aabb = conic.bounding_aabb();
                let tiles = conic.bounding_tiles(tile_size);

                let fp = g
                    .footprint
                    .expect("gpu should keep a footprint when the cpu conic exists");

                assert!(
                    close(fp.conic.a, conic.a),
                    "lane {lane} conic.a: gpu {} vs cpu {}",
                    fp.conic.a,
                    conic.a
                );
                assert!(
                    close(fp.conic.b, conic.b),
                    "lane {lane} conic.b: gpu {} vs cpu {}",
                    fp.conic.b,
                    conic.b
                );
                assert!(
                    close(fp.conic.c, conic.c),
                    "lane {lane} conic.c: gpu {} vs cpu {}",
                    fp.conic.c,
                    conic.c
                );
                assert!(
                    close(fp.conic.center[0], conic.center[0])
                        && close(fp.conic.center[1], conic.center[1]),
                    "lane {lane} conic.center: gpu {:?} vs cpu {:?}",
                    fp.conic.center,
                    conic.center
                );

                assert!(
                    close(fp.aabb.min[0], aabb.min[0]) && close(fp.aabb.min[1], aabb.min[1]),
                    "lane {lane} aabb.min: gpu {:?} vs cpu {:?}",
                    fp.aabb.min,
                    aabb.min
                );
                assert!(
                    close(fp.aabb.max[0], aabb.max[0]) && close(fp.aabb.max[1], aabb.max[1]),
                    "lane {lane} aabb.max: gpu {:?} vs cpu {:?}",
                    fp.aabb.max,
                    aabb.max
                );

                assert_eq!(fp.tiles.min_x, tiles.min_x, "lane {lane} tile min_x");
                assert_eq!(fp.tiles.min_y, tiles.min_y, "lane {lane} tile min_y");
                assert_eq!(fp.tiles.max_x, tiles.max_x, "lane {lane} tile max_x");
                assert_eq!(fp.tiles.max_y, tiles.max_y, "lane {lane} tile max_y");
            }
        }
    }

    got
}

#[test]
fn empty_batch_is_empty() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuGaussianSplat::new(&ctx);
    let got = gpu.eval(&ctx, 16, &[]);
    assert!(got.is_empty(), "an empty batch yields an empty result");
}

#[test]
fn single_unit_covariance_projects() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuGaussianSplat::new(&ctx);
    // An isotropic unit Gaussian with the identity orientation, in front of the
    // camera: its 3D covariance is the identity, so the projection is a clean,
    // well-conditioned circle.
    let queries = [query(
        [0.0, 0.0, 2.0],
        [1.0, 1.0, 1.0],
        [0.0, 0.0, 0.0, 1.0],
        [600.0, 600.0],
        [320.5, 240.5],
    )];
    let got = check(&ctx, &gpu, 16, &queries);
    let fp = got[0]
        .footprint
        .expect("a unit covariance in front of the camera is invertible");
    // Identity rotation and unit scale give the identity 3D covariance.
    assert!(close(got[0].cov3[0], 1.0), "cov3 xx should be 1");
    assert!(close(got[0].cov3[3], 1.0), "cov3 yy should be 1");
    assert!(close(got[0].cov3[5], 1.0), "cov3 zz should be 1");
    assert!(close(got[0].cov3[1], 0.0), "cov3 xy should be 0");
    // Isotropic in-plane projection: equal focal lengths give a symmetric conic.
    assert!(
        close(fp.conic.a, fp.conic.c),
        "an isotropic splat should give a symmetric conic ({} vs {})",
        fp.conic.a,
        fp.conic.c
    );
    assert!(
        close(fp.conic.b, 0.0),
        "an axis-aligned isotropic splat should have no conic cross term, got {}",
        fp.conic.b
    );
}

#[test]
fn anisotropic_large_covariance_projects() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuGaussianSplat::new(&ctx);
    // A strongly anisotropic, rotated Gaussian: the long axis dominates the
    // projected footprint and the rotation mixes all three covariance axes, so
    // every covariance and conic term is exercised.
    let queries = [
        query(
            [0.7, -0.4, 3.0],
            [5.0, 0.5, 2.0],
            [0.2, 0.5, -0.3, 0.9],
            [800.0, 650.0],
            [410.0, 300.0],
        ),
        query(
            [-1.5, 1.1, 4.5],
            [3.0, 3.0, 0.25],
            [0.6, -0.1, 0.2, 0.7],
            [700.0, 700.0],
            [128.0, 96.0],
        ),
    ];
    let got = check(&ctx, &gpu, 32, &queries);
    for (lane, proj) in got.iter().enumerate() {
        assert!(
            proj.footprint.is_some(),
            "lane {lane}: a full-rank anisotropic splat should stay invertible"
        );
    }
}

#[test]
fn degenerate_zero_scale_has_no_footprint() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuGaussianSplat::new(&ctx);
    // A zero-scale Gaussian collapses to a point: its 3D and 2D covariances are
    // all zero, the determinant is zero (below `DET_EPS`), and the reference
    // `conic_from_cov2` returns `None`. The twin must report no footprint.
    let queries = [query(
        [0.3, -0.2, 2.5],
        [0.0, 0.0, 0.0],
        [0.1, 0.2, 0.3, 0.9],
        [600.0, 600.0],
        [200.0, 150.0],
    )];
    let got = check(&ctx, &gpu, 16, &queries);
    assert!(
        got[0].footprint.is_none(),
        "a zero-scale splat is degenerate and must have no footprint"
    );
    for (k, &c) in got[0].cov2.iter().enumerate() {
        assert!(close(c, 0.0), "cov2[{k}] of a zero-scale splat should be 0");
    }
}

#[test]
fn aabb_and_tiles_clamp_at_the_origin() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuGaussianSplat::new(&ctx);
    // A small splat seen through a normal lens, centered a few pixels from the
    // screen origin. Its screen-space covariance is kept moderate on purpose:
    // Sigma_xx = (focal / depth)^2 * scale^2 = (300 / 2)^2 * 0.02^2 = 9, so the
    // 3σ half-extent is 3 * sqrt(9) = 9 px and the box [−5, −6] × [13, 12]
    // reaches into negative pixel space. The covariance stays well above the
    // conic-determinant floor (det = 81, inverse det = 1/81 ≫ DET_EPS), so —
    // unlike a very broad splat whose conic underflows `to_cov2` and collapses
    // to a zero box — the footprint is a real, origin-straddling rectangle. The
    // minimum tile indices must therefore clamp to zero while the maxima stay
    // positive. Sweeping the tile edge (including a zero, which the reference
    // treats as one) exercises the floor-and-clamp binning across sizes.
    let base = query(
        [0.0, 0.0, 2.0],
        [0.02, 0.02, 0.02],
        [0.0, 0.0, 0.0, 1.0],
        [300.0, 300.0],
        [4.0, 3.0],
    );
    for &tile_size in &[0_u32, 1, 8, 16, 64] {
        let got = check(&ctx, &gpu, tile_size, &[base]);
        let fp = got[0]
            .footprint
            .expect("the broad splat is invertible at every tile size");
        // The box straddles the origin, so at least one corner is negative and
        // its tile index must have been clamped to zero.
        assert!(
            fp.aabb.min[0] < 0.0 || fp.aabb.min[1] < 0.0,
            "fixture should straddle the origin, got min {:?}",
            fp.aabb.min
        );
        assert_eq!(
            fp.tiles.min_x, 0,
            "a negative min corner must clamp to tile 0 (tile_size {tile_size})"
        );
        assert_eq!(
            fp.tiles.min_y, 0,
            "a negative min corner must clamp to tile 0 (tile_size {tile_size})"
        );
    }
}

#[test]
fn random_batch_matches_reference_field_by_field() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuGaussianSplat::new(&ctx);
    // A large batch of well-conditioned random splats: scales stay away from
    // zero so every 2D covariance is comfortably invertible, depths stay in
    // front of the camera, and centers sit mid-tile so the floored `tile`
    // indices are not razor-close to an edge. The `check` helper compares every
    // covariance, conic, `AABB` and `tile` field against the CPU reference.
    let mut state = 0x5eed_1234_abcd_0001_u64;
    let mut queries = Vec::with_capacity(256);
    for _ in 0..256 {
        let mean = [
            uniform(&mut state, -2.0, 2.0),
            uniform(&mut state, -2.0, 2.0),
            uniform(&mut state, 1.0, 5.0),
        ];
        let scale = [
            uniform(&mut state, 0.5, 2.5),
            uniform(&mut state, 0.5, 2.5),
            uniform(&mut state, 0.5, 2.5),
        ];
        let quat = [
            uniform(&mut state, -1.0, 1.0),
            uniform(&mut state, -1.0, 1.0),
            uniform(&mut state, -1.0, 1.0),
            uniform(&mut state, -1.0, 1.0),
        ];
        let focal = [
            uniform(&mut state, 400.0, 800.0),
            uniform(&mut state, 400.0, 800.0),
        ];
        let screen_center = [
            uniform(&mut state, 100.0, 300.0),
            uniform(&mut state, 100.0, 300.0),
        ];
        queries.push(query(mean, scale, quat, focal, screen_center));
    }
    for &tile_size in &[8_u32, 16, 64] {
        let got = check(&ctx, &gpu, tile_size, &queries);
        assert_eq!(got.len(), queries.len());
        assert!(
            got.iter().any(|p| p.footprint.is_some()),
            "the well-conditioned batch should yield invertible footprints"
        );
    }
}
