//! Real-device parity for the `billboard`-`atlas` twin:
//! [`GpuBillboardAtlas`](prism_volumetric_gpu::billboard_atlas::GpuBillboardAtlas)
//! must reproduce the `CPU` golden
//! [`billboard_atlas`](prism_render_architecture::particle::billboard_atlas)
//! across the `octahedral` encode and decode, the single-cell view selection,
//! the integer grid arithmetic (`cell_coord`, `cell_index`, `cell_count`) and
//! the normalized cell `UV` rectangle.
//!
//! The fixtures draw random unit directions and random integer cell probes from
//! a host `LCG`, across a spread of grid dimensions. Every direction is
//! rejection-sampled to stay well clear of the two branch-critical regions so
//! the discrete cell codes agree exactly between `CPU` and `GPU`:
//!
//! * the `octahedral` seam, where the lower-hemisphere fold flips: each sampled
//!   direction keeps `abs(z) >= 0.15` and `abs(x), abs(y) >= 0.1`, so the fold
//!   predicate `pz < 0` and the `sign_not_zero` of each lane are stable under
//!   the tiny `f32` differences between the `CPU` and `GPU` normalize;
//! * the cell quantization boundary, where `floor(s * grid_dim)` jumps: the
//!   fractional parts of `s * grid_dim` and `t * grid_dim` are kept inside
//!   `[0.1, 0.9]`, so both devices truncate to the same column and row.
//!
//! Under those preconditions the index, column, row and count comparisons are
//! exact `==`; the encode, decode and `UV` rectangle are compared under
//! tolerance (`abs_diff <= 1e-4` or `rel_diff <= 1e-3`, `REL_FLOOR = 1e-6`),
//! since they thread through a reciprocal and a `sqrt` a `GPU` may contract
//! differently. A degenerate `grid_dim` of zero exercises the clamp-up-to-one
//! fallback, and an empty batch exercises the host short-circuit.
//!
//! The tests skip (returning early) when the host has no `wgpu` adapter, so the
//! suite stays green everywhere while still exercising the full
//! dispatch-and-readback on any real device such as an Apple `M`-series `GPU`.
//!
//! Provenance: 孪生自本仓 `prism_render_architecture::particle::billboard_atlas`；无第三方引擎源码或衍生代码。

use prism_render_architecture::particle::billboard_atlas::{oct_decode, oct_encode, ImpostorGrid};
use prism_volumetric_gpu::billboard_atlas::{
    GpuBillboardAtlas, GpuBillboardAtlasQuery, GpuBillboardAtlasResult,
};
use prism_volumetric_gpu::GpuContext;

/// Absolute tolerance for the continuous quantities.
const ABS_EPS: f32 = 1.0e-4;
/// Relative tolerance for the continuous quantities.
const REL_EPS: f32 = 1.0e-3;
/// Floor for the relative-tolerance denominator.
const REL_FLOOR: f32 = 1.0e-6;

/// Mixed absolute / relative tolerance comparison for one `f32` lane.
fn approx(a: f32, b: f32) -> bool {
    let diff = (a - b).abs();
    if diff <= ABS_EPS {
        return true;
    }
    let scale = a.abs().max(b.abs()).max(REL_FLOOR);
    diff / scale <= REL_EPS
}

/// A small `u64` linear-congruential generator for deterministic fixtures; it
/// uses only integer arithmetic and never a transcendental method.
struct Lcg {
    state: u64,
}

impl Lcg {
    fn new(seed: u64) -> Lcg {
        Lcg { state: seed }
    }

    /// Advances the generator and returns the high `32` bits of the new state.
    fn next_u32(&mut self) -> u32 {
        self.state = self
            .state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        (self.state >> 32) as u32
    }

    /// A pseudo-random `f32` in `[-1.0, 1.0]`, from integer arithmetic only.
    fn next_signed(&mut self) -> f32 {
        let unit = self.next_u32() as f32 / u32::MAX as f32;
        unit * 2.0 - 1.0
    }
}

/// Rejection-samples a random query for `grid_dim`, keeping the direction clear
/// of the `octahedral` seam and the cell quantization boundaries so the discrete
/// cell codes agree exactly between the two devices (see the module docs).
fn sample_query(lcg: &mut Lcg, grid_dim: u32) -> GpuBillboardAtlasQuery {
    // The reference clamps `grid_dim` up to one; mirror it for the sampler maths.
    let dim = grid_dim.max(1);
    let g = dim as f32;
    loop {
        let x = lcg.next_signed();
        let y = lcg.next_signed();
        let z = lcg.next_signed();
        let len_sq = x * x + y * y + z * z;
        if len_sq < 0.2 {
            continue;
        }
        let inv = 1.0 / len_sq.sqrt();
        let nx = x * inv;
        let ny = y * inv;
        let nz = z * inv;
        // Stay off the seam: a robustly signed z (fold predicate) and robustly
        // signed x and y (the `sign_not_zero` lanes of the fold).
        if nx.abs() < 0.1 || ny.abs() < 0.1 || nz.abs() < 0.15 {
            continue;
        }
        // Stay off the quantization boundary: fractional part of s*g and t*g in
        // [0.1, 0.9] so both devices floor to the same column and row.
        let enc = oct_encode([nx, ny, nz]);
        let s = enc[0] * 0.5 + 0.5;
        let t = enc[1] * 0.5 + 0.5;
        let sg = s * g;
        let tg = t * g;
        let fs = sg - sg.floor();
        let ft = tg - tg.floor();
        if !(0.1..=0.9).contains(&fs) || !(0.1..=0.9).contains(&ft) {
            continue;
        }

        let total = dim * dim;
        let cell = lcg.next_u32() % total;
        let col = lcg.next_u32() % dim;
        let row = lcg.next_u32() % dim;
        return GpuBillboardAtlasQuery {
            dir: [nx, ny, nz],
            grid_dim,
            cell,
            col,
            row,
            atlas_width: 1024,
            atlas_height: 512,
        };
    }
}

/// Asserts one `GPU` result matches the `CPU` golden for its query.
fn assert_parity(got: &GpuBillboardAtlasResult, q: &GpuBillboardAtlasQuery) {
    let grid = ImpostorGrid::new(q.grid_dim, q.atlas_width, q.atlas_height);

    // Continuous quantities: tolerance only.
    let cpu_enc = oct_encode(q.dir);
    assert!(
        approx(got.oct_encode[0], cpu_enc[0]) && approx(got.oct_encode[1], cpu_enc[1]),
        "oct_encode mismatch: gpu {:?} vs cpu {cpu_enc:?} for dir {:?}",
        got.oct_encode,
        q.dir
    );
    let cpu_dec = oct_decode(cpu_enc);
    assert!(
        approx(got.oct_decode[0], cpu_dec[0])
            && approx(got.oct_decode[1], cpu_dec[1])
            && approx(got.oct_decode[2], cpu_dec[2]),
        "oct_decode mismatch: gpu {:?} vs cpu {cpu_dec:?}",
        got.oct_decode
    );

    // Discrete cell codes: exact. The sampler keeps directions off the seam and
    // the quantization boundary, so the floor-based column and row are stable.
    let vc = grid.view_to_cell(q.dir);
    assert_eq!(
        got.view_cell_index, vc.index,
        "view index for dir {:?}",
        q.dir
    );
    assert_eq!(got.view_cell_col, vc.col, "view col for dir {:?}", q.dir);
    assert_eq!(got.view_cell_row, vc.row, "view row for dir {:?}", q.dir);

    let (ccol, crow) = grid.cell_coord(q.cell);
    assert_eq!(got.cell_col, ccol, "cell_coord col for cell {}", q.cell);
    assert_eq!(got.cell_row, crow, "cell_coord row for cell {}", q.cell);

    assert_eq!(
        got.cell_index,
        grid.cell_index(q.col, q.row),
        "cell_index for ({}, {})",
        q.col,
        q.row
    );
    assert_eq!(got.cell_count, grid.cell_count(), "cell_count mismatch");

    // UV rectangle: tolerance.
    let cpu_uv = grid.cell_uv_rect(q.cell);
    assert!(
        approx(got.cell_uv_rect[0], cpu_uv[0])
            && approx(got.cell_uv_rect[1], cpu_uv[1])
            && approx(got.cell_uv_rect[2], cpu_uv[2])
            && approx(got.cell_uv_rect[3], cpu_uv[3]),
        "cell_uv_rect mismatch: gpu {:?} vs cpu {cpu_uv:?}",
        got.cell_uv_rect
    );
}

#[test]
fn random_queries_match_golden_across_grids() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuBillboardAtlas::new(&ctx);

    let grid_dims = [1u32, 2, 3, 5, 8, 16, 32];
    let mut lcg = Lcg::new(0x9E37_79B9_7F4A_7C15);
    let mut queries = Vec::new();
    for &dim in &grid_dims {
        for _ in 0..24 {
            queries.push(sample_query(&mut lcg, dim));
        }
    }

    let got = gpu.evaluate(&ctx, &queries);
    assert_eq!(got.len(), queries.len(), "one result per query");
    for (result, q) in got.iter().zip(queries.iter()) {
        assert_parity(result, q);
    }
}

#[test]
fn degenerate_grid_dim_clamps_to_one_cell() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuBillboardAtlas::new(&ctx);

    // grid_dim 0 is clamped up to a single cell on both devices, so every probe
    // resolves to cell 0 and the whole atlas is one unit UV rectangle.
    let mut lcg = Lcg::new(0x1234_5678_9ABC_DEF0);
    let q = sample_query(&mut lcg, 0);
    let got = gpu.evaluate(&ctx, std::slice::from_ref(&q));
    assert_eq!(got.len(), 1, "one result per query");
    assert_eq!(got[0].cell_count, 1, "clamped grid has one cell");
    assert_eq!(got[0].view_cell_index, 0, "single cell is index 0");
    assert_parity(&got[0], &q);
}

#[test]
fn evaluation_is_deterministic() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuBillboardAtlas::new(&ctx);

    let mut lcg = Lcg::new(0xDEAD_BEEF_CAFE_1234);
    let q = sample_query(&mut lcg, 12);
    let first = gpu.evaluate(&ctx, std::slice::from_ref(&q));
    let second = gpu.evaluate(&ctx, std::slice::from_ref(&q));
    assert_eq!(first.len(), 1);
    assert_eq!(second.len(), 1);
    let a = first[0];
    let b = second[0];
    // Both runs match the golden, and their discrete cell codes are identical;
    // the continuous lanes agree to tolerance (no exact f32 compare is used).
    assert_parity(&a, &q);
    assert_parity(&b, &q);
    assert_eq!(
        a.view_cell_index, b.view_cell_index,
        "view index differs across runs"
    );
    assert_eq!(
        a.view_cell_col, b.view_cell_col,
        "view col differs across runs"
    );
    assert_eq!(
        a.view_cell_row, b.view_cell_row,
        "view row differs across runs"
    );
    assert_eq!(a.cell_col, b.cell_col, "cell col differs across runs");
    assert_eq!(a.cell_row, b.cell_row, "cell row differs across runs");
    assert_eq!(a.cell_index, b.cell_index, "cell index differs across runs");
    assert_eq!(a.cell_count, b.cell_count, "cell count differs across runs");
    assert!(
        approx(a.oct_encode[0], b.oct_encode[0]) && approx(a.oct_encode[1], b.oct_encode[1]),
        "oct_encode differs across runs",
    );
    assert!(
        approx(a.oct_decode[0], b.oct_decode[0])
            && approx(a.oct_decode[1], b.oct_decode[1])
            && approx(a.oct_decode[2], b.oct_decode[2]),
        "oct_decode differs across runs",
    );
    assert!(
        approx(a.cell_uv_rect[0], b.cell_uv_rect[0])
            && approx(a.cell_uv_rect[1], b.cell_uv_rect[1])
            && approx(a.cell_uv_rect[2], b.cell_uv_rect[2])
            && approx(a.cell_uv_rect[3], b.cell_uv_rect[3]),
        "cell_uv_rect differs across runs",
    );
}

#[test]
fn empty_batch_short_circuits() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuBillboardAtlas::new(&ctx);
    let got = gpu.evaluate(&ctx, &[]);
    assert!(
        got.is_empty(),
        "an empty batch issues no dispatch and returns empty"
    );
}
