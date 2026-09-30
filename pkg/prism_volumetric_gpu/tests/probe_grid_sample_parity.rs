//! Real-device parity for the probe-grid sampler twin: [`GpuProbeGridSample`]
//! must reproduce the `CPU` golden
//! [`ProbeGrid::sample`](prism_render_architecture::volumetric::multiscatter::ProbeGrid::sample)
//! across interior points, exact lattice nodes, and out-of-range
//! (clamp-to-edge) positions, for all six irradiance bands.
//!
//! The tests skip (with a printed notice) when the host has no `wgpu` adapter,
//! so the suite stays green everywhere while still exercising the full
//! dispatch-and-readback on any real device. The kernel is portable core-WGSL,
//! so it needs no optional device feature.
//!
//! # Grid construction
//!
//! The `CPU` [`ProbeGrid`] keeps its `probes`/`min_corner`/`max_corner`
//! private, so the test builds a grid with [`ProbeGrid::from_fn`] and flattens
//! its probes via the public [`ProbeGrid::probe_at`] accessor in row-major,
//! band-inner order to feed the `GPU`. The chosen box corners are passed to
//! both sides, so the axis mapping is identical.
//!
//! # Parity criterion
//!
//! The sampler is only multiply/add on the raw probe data plus integer address
//! math — no transcendental — so `CPU` and `GPU` evaluate the identical
//! closed-form algebra, differing at most by a legal multiply-add contraction
//! of a few `ULP`. Each of the six bands is asserted to within `abs_diff < 1e-6`
//! or `rel_diff < 1e-5`, tight enough to fail a wrong port (a swapped axis, a
//! dropped corner, a mis-ordered accumulation).
//!
//! Provenance: standard trilinear grid sampling with clamp-to-edge; no Unreal
//! Engine source or derived code.

use prism_render_architecture::volumetric::multiscatter::{ProbeGrid, PROBE_BANDS};
use prism_render_architecture::volumetric::Vec3;
use prism_volumetric_gpu::{GpuContext, GpuProbeGridSample, ProbeSampleQuery};

/// Box corners shared by the `CPU` grid and the `GPU` dispatch.
fn box_corners() -> (Vec3, Vec3) {
    (Vec3::new(-2.0, 0.0, 1.0), Vec3::new(4.0, 3.0, 7.0))
}

/// A deterministic, per-band irradiance closure over world position. Distinct
/// per band so a band mix-up cannot pass.
fn probe_closure(p: Vec3) -> [f32; PROBE_BANDS] {
    [
        0.10 * p.x + 0.20 * p.y,
        1.00 - 0.05 * p.z,
        0.30 * p.x * 0.10 + 0.40,
        2.00 + 0.15 * p.y - 0.05 * p.x,
        0.50 * p.z,
        0.25 * (p.x + p.y + p.z),
    ]
}

/// Flattens the grid's probes via the public accessor into the row-major,
/// band-inner layout the kernel expects.
fn flatten(grid: &ProbeGrid, dims: [usize; 3]) -> Vec<f32> {
    let mut flat = Vec::with_capacity(dims[0] * dims[1] * dims[2] * PROBE_BANDS);
    for ix in 0..dims[0] {
        for iy in 0..dims[1] {
            for iz in 0..dims[2] {
                let probe = grid.probe_at(ix, iy, iz);
                flat.extend_from_slice(&probe.irradiance);
            }
        }
    }
    flat
}

/// Asserts every `gpu` band matches the `CPU` `grid.sample` for its query.
fn assert_parity(grid: &ProbeGrid, queries: &[ProbeSampleQuery], gpu: &[[f32; PROBE_BANDS]]) {
    assert_eq!(gpu.len(), queries.len(), "one band octet per query");
    for (i, q) in queries.iter().enumerate() {
        let exp = grid.sample(q.pos);
        let got = gpu[i];
        for band in 0..PROBE_BANDS {
            let abs_diff = (got[band] - exp[band]).abs();
            let rel_diff = abs_diff / exp[band].abs().max(1e-6);
            assert!(
                abs_diff < 1e-6 || rel_diff < 1e-5,
                "probe grid mismatch for query {i} ({q:?}) band {band}: \
                 gpu {}, cpu {} (abs {abs_diff}, rel {rel_diff})",
                got[band],
                exp[band]
            );
        }
    }
}

#[test]
#[expect(clippy::print_stderr, reason = "surface a skip notice on headless CI")]
fn probe_grid_sample_matches_cpu_interior_and_nodes() {
    let Some(ctx) = GpuContext::try_headless() else {
        eprintln!("skipping probe_grid_sample parity: no wgpu adapter");
        return;
    };
    let dims = [4usize, 3usize, 5usize];
    let (min_corner, max_corner) = box_corners();
    let grid = ProbeGrid::from_fn(dims, min_corner, max_corner, probe_closure);
    let flat = flatten(&grid, dims);

    let mut queries = Vec::new();
    for &x in &[-2.0f32, -0.5, 1.0, 2.7, 4.0] {
        for &y in &[0.0f32, 1.1, 3.0] {
            for &z in &[1.0f32, 3.4, 5.0, 7.0] {
                queries.push(ProbeSampleQuery {
                    pos: Vec3::new(x, y, z),
                });
            }
        }
    }

    let sampler = GpuProbeGridSample::new(&ctx);
    let gpu = sampler.eval(&ctx, &flat, dims, min_corner, max_corner, &queries);
    assert_parity(&grid, &queries, &gpu);
}

#[test]
#[expect(clippy::print_stderr, reason = "surface a skip notice on headless CI")]
fn probe_grid_sample_clamps_out_of_range() {
    let Some(ctx) = GpuContext::try_headless() else {
        eprintln!("skipping probe_grid_sample clamp parity: no wgpu adapter");
        return;
    };
    let dims = [3usize, 4usize, 3usize];
    let (min_corner, max_corner) = box_corners();
    let grid = ProbeGrid::from_fn(dims, min_corner, max_corner, probe_closure);
    let flat = flatten(&grid, dims);

    let queries = vec![
        ProbeSampleQuery {
            pos: Vec3::new(-100.0, -50.0, -30.0),
        },
        ProbeSampleQuery {
            pos: Vec3::new(100.0, 80.0, 60.0),
        },
        ProbeSampleQuery {
            pos: Vec3::new(-10.0, 20.0, -5.0),
        },
    ];

    let sampler = GpuProbeGridSample::new(&ctx);
    let gpu = sampler.eval(&ctx, &flat, dims, min_corner, max_corner, &queries);
    assert_parity(&grid, &queries, &gpu);

    // Clamp-to-edge: a below-range query equals the min-corner probe sample.
    let lo = grid.sample(Vec3::new(-100.0, -50.0, -30.0));
    let lo_edge = grid.sample(min_corner);
    for band in 0..PROBE_BANDS {
        assert!(
            (lo[band] - lo_edge[band]).abs() < 1e-6,
            "below-range query must clamp to the min corner (band {band})"
        );
    }
}

#[test]
#[expect(clippy::print_stderr, reason = "surface a skip notice on headless CI")]
fn probe_grid_sample_empty_queries_is_empty() {
    let Some(ctx) = GpuContext::try_headless() else {
        eprintln!("skipping probe_grid_sample empty parity: no wgpu adapter");
        return;
    };
    let dims = [2usize, 2usize, 2usize];
    let (min_corner, max_corner) = box_corners();
    let grid = ProbeGrid::from_fn(dims, min_corner, max_corner, probe_closure);
    let flat = flatten(&grid, dims);
    let sampler = GpuProbeGridSample::new(&ctx);
    let gpu = sampler.eval(&ctx, &flat, dims, min_corner, max_corner, &[]);
    assert!(gpu.is_empty(), "empty queries must yield an empty result");
}
