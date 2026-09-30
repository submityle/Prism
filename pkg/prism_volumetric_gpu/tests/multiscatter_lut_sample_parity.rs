//! Real-device parity for the multi-scatter `LUT` sampler twin:
//! [`GpuMultiScatterLutSample`] must reproduce the `CPU` golden
//! [`MultiScatterLut::sample`](prism_render_architecture::volumetric::multiscatter::MultiScatterLut::sample)
//! across interior points, exact grid nodes, and out-of-range (clamp-to-edge)
//! coordinates.
//!
//! The tests skip (with a printed notice) when the host has no `wgpu` adapter,
//! so the suite stays green everywhere while still exercising the full
//! dispatch-and-readback on any real device. The kernel is portable core-WGSL,
//! so it needs no optional device feature.
//!
//! # Table construction
//!
//! The `CPU` [`MultiScatterLut`] keeps its `data`/`mins`/`maxs` private, so the
//! test builds a table with [`MultiScatterLut::from_fn`] (which stores
//! `saturate(closure(cos, depth, albedo))` at every cell) and independently
//! reconstructs the same row-major flat data to feed the `GPU`. The axis ranges
//! match the fixed `MultiScatterLut::new` layout — `mins = [-1, 0, 0]`,
//! `maxs = [1, DEFAULT_MAX_OPTICAL_DEPTH, 1]` with
//! `DEFAULT_MAX_OPTICAL_DEPTH = 8.0` (a private const mirrored here). A
//! cross-check asserts the reconstructed flat cell at each node equals
//! `lut.sample` at that node's exact coordinate (where the interpolation
//! fraction is zero), so a mis-ordered flattening could not silently pass.
//!
//! # Parity criterion
//!
//! The sampler is only multiply/add on the raw table plus integer address math
//! — no transcendental — so `CPU` and `GPU` evaluate the identical closed-form
//! algebra, differing at most by a legal multiply-add contraction of a few
//! `ULP`. Each sampled gain is asserted to within `abs_diff < 1e-6` and to stay
//! in `[0, 1]`, tight enough to fail a wrong port (a swapped axis, a dropped
//! corner, a mis-ordered accumulation).
//!
//! Provenance: standard trilinear `LUT` sampling with clamp-to-edge; no Unreal
//! Engine source or derived code.

use prism_render_architecture::volumetric::math::saturate;
use prism_render_architecture::volumetric::multiscatter::MultiScatterLut;
use prism_volumetric_gpu::{GpuContext, GpuMultiScatterLutSample, MultiScatterSampleQuery};

/// Fixed axis ranges baked into `MultiScatterLut::new`. The upper `depth` bound
/// is the crate-private `DEFAULT_MAX_OPTICAL_DEPTH`, mirrored here.
const MINS: [f32; 3] = [-1.0, 0.0, 0.0];
const MAXS: [f32; 3] = [1.0, 8.0, 1.0];

/// Sample value of axis coordinate `i` on a `dim`-cell axis spanning
/// `[min, max]`, mirroring the crate-private `axis_value`.
fn axis_value(min: f32, max: f32, dim: usize, i: usize) -> f32 {
    if dim <= 1 {
        min
    } else {
        min + (max - min) * (i as f32 / (dim - 1) as f32)
    }
}

/// A deterministic, smoothly varying closure over the three physical axes. It
/// stays inside `[0, 1]` so `from_fn`'s `saturate` is a no-op and the table is
/// exactly the closure evaluated at cell centres.
fn cell_closure(cos: f32, depth: f32, albedo: f32) -> f32 {
    let a = 0.5 + 0.5 * cos; // cos in [-1, 1] -> [0, 1]
    let b = depth / 8.0; //     depth in [0, 8] -> [0, 1]
    let c = albedo; //          albedo already in [0, 1]
    0.2 * a + 0.5 * b * c + 0.3 * (a * c)
}

/// Reconstructs the row-major flat table for `dims`, matching the fill order of
/// `MultiScatterLut::from_fn` (`cos` outer, `depth` middle, `albedo` inner) and
/// its per-cell `saturate`.
fn build_flat(dims: [usize; 3]) -> Vec<f32> {
    let mut flat = Vec::with_capacity(dims[0] * dims[1] * dims[2]);
    for ic in 0..dims[0] {
        let cos = axis_value(MINS[0], MAXS[0], dims[0], ic);
        for id in 0..dims[1] {
            let depth = axis_value(MINS[1], MAXS[1], dims[1], id);
            for ia in 0..dims[2] {
                let albedo = axis_value(MINS[2], MAXS[2], dims[2], ia);
                flat.push(saturate(cell_closure(cos, depth, albedo)));
            }
        }
    }
    flat
}

/// Confirms the reconstructed flat table matches the `CPU` `LUT`'s stored cells
/// by sampling at each node's exact coordinate (interpolation fraction zero).
fn assert_flat_matches_nodes(lut: &MultiScatterLut, flat: &[f32], dims: [usize; 3]) {
    for ic in 0..dims[0] {
        let cos = axis_value(MINS[0], MAXS[0], dims[0], ic);
        for id in 0..dims[1] {
            let depth = axis_value(MINS[1], MAXS[1], dims[1], id);
            for ia in 0..dims[2] {
                let albedo = axis_value(MINS[2], MAXS[2], dims[2], ia);
                let flat_idx = (ic * dims[1] + id) * dims[2] + ia;
                let node = lut.sample(cos, depth, albedo);
                assert!(
                    (flat[flat_idx] - node).abs() < 1e-6,
                    "flat cell {flat_idx} ({cos},{depth},{albedo}) = {}, node sample = {node}",
                    flat[flat_idx]
                );
            }
        }
    }
}

/// Asserts every `gpu` gain matches the `CPU` `lut.sample` for its query and
/// stays in `[0, 1]`.
fn assert_parity(lut: &MultiScatterLut, queries: &[MultiScatterSampleQuery], gpu: &[f32]) {
    assert_eq!(gpu.len(), queries.len(), "one gain per query");
    for (i, q) in queries.iter().enumerate() {
        let exp = lut.sample(q.cos, q.depth, q.albedo);
        let got = gpu[i];
        assert!(
            (got - exp).abs() < 1e-6,
            "multiscatter LUT sample mismatch for query {i} ({q:?}): gpu {got}, cpu {exp}"
        );
        assert!(
            (0.0..=1.0).contains(&got),
            "gpu gain must stay in [0, 1]: {got}"
        );
    }
}

#[test]
#[expect(clippy::print_stderr, reason = "surface a skip notice on headless CI")]
fn multiscatter_lut_sample_matches_cpu_interior_and_nodes() {
    let Some(ctx) = GpuContext::try_headless() else {
        eprintln!("skipping multiscatter_lut_sample parity: no wgpu adapter");
        return;
    };
    let dims = [5usize, 6usize, 4usize];
    let lut = MultiScatterLut::from_fn(dims, cell_closure);
    let flat = build_flat(dims);
    assert_flat_matches_nodes(&lut, &flat, dims);

    // Interior points, exact nodes, and axis extremes.
    let mut queries = Vec::new();
    for &cos in &[-1.0f32, -0.5, 0.0, 0.25, 0.75, 1.0] {
        for &depth in &[0.0f32, 1.3, 4.0, 6.5, 8.0] {
            for &albedo in &[0.0f32, 0.35, 0.6, 1.0] {
                queries.push(MultiScatterSampleQuery { cos, depth, albedo });
            }
        }
    }

    let sampler = GpuMultiScatterLutSample::new(&ctx);
    let gpu = sampler.eval(&ctx, &flat, dims, MINS, MAXS, &queries);
    assert_parity(&lut, &queries, &gpu);
}

#[test]
#[expect(clippy::print_stderr, reason = "surface a skip notice on headless CI")]
fn multiscatter_lut_sample_clamps_out_of_range() {
    let Some(ctx) = GpuContext::try_headless() else {
        eprintln!("skipping multiscatter_lut_sample clamp parity: no wgpu adapter");
        return;
    };
    let dims = [4usize, 5usize, 3usize];
    let lut = MultiScatterLut::from_fn(dims, cell_closure);
    let flat = build_flat(dims);

    // All coordinates pushed well past both ends of every axis.
    let queries = vec![
        MultiScatterSampleQuery {
            cos: -5.0,
            depth: -3.0,
            albedo: -2.0,
        },
        MultiScatterSampleQuery {
            cos: 5.0,
            depth: 20.0,
            albedo: 4.0,
        },
        MultiScatterSampleQuery {
            cos: -2.0,
            depth: 12.0,
            albedo: -1.0,
        },
        MultiScatterSampleQuery {
            cos: 3.0,
            depth: -1.0,
            albedo: 2.0,
        },
    ];

    let sampler = GpuMultiScatterLutSample::new(&ctx);
    let gpu = sampler.eval(&ctx, &flat, dims, MINS, MAXS, &queries);
    assert_parity(&lut, &queries, &gpu);

    // Clamp-to-edge: an out-of-range query equals the clamped in-range one.
    let lo = lut.sample(-5.0, -3.0, -2.0);
    let lo_edge = lut.sample(MINS[0], MINS[1], MINS[2]);
    assert!(
        (lo - lo_edge).abs() < 1e-6,
        "below-range query must clamp to the min corner: {lo} vs {lo_edge}"
    );
    let hi = lut.sample(5.0, 20.0, 4.0);
    let hi_edge = lut.sample(MAXS[0], MAXS[1], MAXS[2]);
    assert!(
        (hi - hi_edge).abs() < 1e-6,
        "above-range query must clamp to the max corner: {hi} vs {hi_edge}"
    );
}

#[test]
#[expect(clippy::print_stderr, reason = "surface a skip notice on headless CI")]
fn multiscatter_lut_sample_empty_queries_is_empty() {
    let Some(ctx) = GpuContext::try_headless() else {
        eprintln!("skipping multiscatter_lut_sample empty parity: no wgpu adapter");
        return;
    };
    let dims = [3usize, 3usize, 3usize];
    let flat = build_flat(dims);
    let sampler = GpuMultiScatterLutSample::new(&ctx);
    let gpu = sampler.eval(&ctx, &flat, dims, MINS, MAXS, &[]);
    assert!(gpu.is_empty(), "empty queries must yield an empty result");
}
