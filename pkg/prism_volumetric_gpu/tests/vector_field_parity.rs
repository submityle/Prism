//! Real-device parity for the discrete vector-field twin:
//! [`GpuVectorField`](prism_volumetric_gpu::vector_field::GpuVectorField) must
//! reproduce the `CPU` golden
//! [`vector_field`](prism_render_architecture::particle::vector_field) across
//! the row-major linear index, the clamped texel read, the trilinear sample
//! under each wrap code, the world⇄grid transform maps and the
//! central-difference divergence and curl.
//!
//! The fixtures cover both
//! [`WrapMode`](prism_render_architecture::particle::vector_field::WrapMode)
//! policies, in-range and out-of-range sampling (negative and past-`dim`
//! coordinates resolved by `Clamp` and by `Tile`), anisotropic grids with
//! unequal `X` / `Y` / `Z` extents, a degenerate `dim = 1` axis, and a
//! zero-extent transform axis that maps to `0.0`. Every continuous grid
//! coordinate keeps its fractional part inside `[0.15, 0.85]` so the floor split
//! stays clear of a texel tie, and the `Tile` out-of-range coordinates stay well
//! away from the integer boundary where the wrapped index flips.
//!
//! The tests skip (returning early) when the host has no `wgpu` adapter, so the
//! suite stays green everywhere while still exercising the full
//! dispatch-and-readback on any real device such as an Apple `M`-series `GPU`.
//! The kernel is portable core-`WGSL`, so it needs no optional device feature.
//!
//! # Parity criterion
//!
//! The linear index is an integer address, so `CPU` and `GPU` must agree
//! exactly: the comparison is an exact `==`. The sampled vectors, the transform
//! maps and the derivatives thread through multiplies, adds and one guarded
//! division, so they are compared under tolerance (`abs_diff <= 1e-4` or
//! `rel_diff <= 1e-3`, `REL_FLOOR = 1e-6`).
//!
//! Provenance: 孪生自本仓 `prism_render_architecture::particle::vector_field`；无第三方引擎源码或衍生代码。

use prism_render_architecture::particle::sort_cull::Aabb;
use prism_render_architecture::particle::vector_field::{
    VectorField, VectorFieldTransform, WrapMode,
};
use prism_render_architecture::particle::Vec3;
use prism_volumetric_gpu::vector_field::{
    GpuVectorField, VectorFieldQuery, VectorFieldResult, WRAP_CLAMP, WRAP_TILE,
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

/// Tolerant comparison of a `GPU` `[f32; 3]` lane triple against a golden
/// [`Vec3`].
fn approx3(a: [f32; 3], b: Vec3) -> bool {
    approx(a[0], b.x) && approx(a[1], b.y) && approx(a[2], b.z)
}

/// Deterministic host-side `u64` LCG (numerical-recipes constants) producing a
/// repeatable stream of texel components; no transcendental math is involved.
struct Lcg {
    /// Current state word.
    state: u64,
}

impl Lcg {
    /// Seeds the generator.
    fn new(seed: u64) -> Lcg {
        Lcg { state: seed }
    }

    /// Advances the state and returns an `f32` in roughly `[-2, 2)` built from
    /// the high bits, so the texel values stay small and exactly shared between
    /// the `CPU` and `GPU` inputs.
    fn next_component(&mut self) -> f32 {
        self.state = self
            .state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        let bits = (self.state >> 40) as u32;
        // 0..=16777215 scaled to a signed quarter-step grid in [-2, 2).
        let unit = (bits & 0x00ff_ffff) as f32 / 16_777_216.0;
        unit * 4.0 - 2.0
    }
}

/// Builds a shared field of `dims` texels, returning the row-major `[x, y, z]`
/// grid for the `GPU` and the matching [`Vec3`] grid for the `CPU` golden.
fn build_field(dims: (u32, u32, u32), seed: u64) -> (Vec<[f32; 3]>, Vec<Vec3>) {
    let count = (dims.0 * dims.1 * dims.2) as usize;
    let mut rng = Lcg::new(seed);
    let mut flat: Vec<[f32; 3]> = Vec::with_capacity(count);
    let mut vecs: Vec<Vec3> = Vec::with_capacity(count);
    for _ in 0..count {
        let x = rng.next_component();
        let y = rng.next_component();
        let z = rng.next_component();
        flat.push([x, y, z]);
        vecs.push(Vec3::new(x, y, z));
    }
    (flat, vecs)
}

/// Builds one query with an explicit wrap code.
fn query(grid: [f32; 3], wrap: u32, world: [f32; 3], texel: [u32; 3]) -> VectorFieldQuery {
    VectorFieldQuery {
        grid,
        wrap,
        world,
        texel,
    }
}

/// Maps a kernel wrap code back to the golden [`WrapMode`].
fn wrap_mode(code: u32) -> WrapMode {
    if code == WRAP_TILE {
        WrapMode::Tile
    } else {
        WrapMode::Clamp
    }
}

/// Full fixture: a field, a transform box and the query batch to check.
struct Fixture {
    /// `(X, Y, Z)` texel dimensions.
    dims: (u32, u32, u32),
    /// Minimum box corner.
    bounds_min: [f32; 3],
    /// Maximum box corner.
    bounds_max: [f32; 3],
    /// Row-major `GPU` texels.
    flat: Vec<[f32; 3]>,
    /// Row-major golden texels.
    vecs: Vec<Vec3>,
    /// Query batch.
    queries: Vec<VectorFieldQuery>,
}

/// Runs the `GPU` batch once and asserts every twinned answer against the
/// golden `CPU` field.
fn assert_fixture(gpu: &GpuVectorField, ctx: &GpuContext, fx: &Fixture) {
    let got = gpu.evaluate(
        ctx,
        fx.dims,
        fx.bounds_min,
        fx.bounds_max,
        &fx.flat,
        &fx.queries,
    );
    assert_eq!(got.len(), fx.queries.len(), "one result per query");

    let field = VectorField::from_data(fx.dims, fx.vecs.clone()).expect("valid dims and data");
    let transform = VectorFieldTransform::new(Aabb {
        min: Vec3::new(fx.bounds_min[0], fx.bounds_min[1], fx.bounds_min[2]),
        max: Vec3::new(fx.bounds_max[0], fx.bounds_max[1], fx.bounds_max[2]),
    });

    for (q, g) in fx.queries.iter().zip(got.iter()) {
        assert_result(&field, &transform, fx.dims, q, g);
    }
}

/// Asserts one [`VectorFieldResult`] against the golden field and transform.
fn assert_result(
    field: &VectorField,
    transform: &VectorFieldTransform,
    dims: (u32, u32, u32),
    q: &VectorFieldQuery,
    g: &VectorFieldResult,
) {
    let grid = Vec3::new(q.grid[0], q.grid[1], q.grid[2]);
    let world = Vec3::new(q.world[0], q.world[1], q.world[2]);
    let (i, j, k) = (q.texel[0], q.texel[1], q.texel[2]);

    let cpu_sampled = field.sample_grid(grid, wrap_mode(q.wrap));
    assert!(
        approx3(g.sampled, cpu_sampled),
        "sample_grid mismatch at {:?} wrap {}: gpu {:?} vs cpu {cpu_sampled:?}",
        q.grid,
        q.wrap,
        g.sampled
    );

    let cpu_grid_from_world = transform.world_to_grid(dims, world);
    assert!(
        approx3(g.grid_from_world, cpu_grid_from_world),
        "world_to_grid mismatch at {:?}: gpu {:?} vs cpu {cpu_grid_from_world:?}",
        q.world,
        g.grid_from_world
    );

    let cpu_roundtrip = transform.grid_to_world(dims, cpu_grid_from_world);
    assert!(
        approx3(g.world_roundtrip, cpu_roundtrip),
        "grid_to_world round-trip mismatch at {:?}: gpu {:?} vs cpu {cpu_roundtrip:?}",
        q.world,
        g.world_roundtrip
    );

    let cpu_texel = field.sample_texel(i, j, k);
    assert!(
        approx3(g.texel_value, cpu_texel),
        "sample_texel mismatch at {:?}: gpu {:?} vs cpu {cpu_texel:?}",
        q.texel,
        g.texel_value
    );

    assert_eq!(
        g.linear_index as usize,
        field.linear_index(i, j, k),
        "linear_index mismatch at {:?}",
        q.texel
    );

    let cpu_div = field.divergence(i, j, k);
    assert!(
        approx(g.divergence, cpu_div),
        "divergence mismatch at {:?}: gpu {} vs cpu {cpu_div}",
        q.texel,
        g.divergence
    );

    let cpu_curl = field.curl(i, j, k);
    assert!(
        approx3(g.curl, cpu_curl),
        "curl mismatch at {:?}: gpu {:?} vs cpu {cpu_curl:?}",
        q.texel,
        g.curl
    );
}

#[test]
fn clamp_sampling_in_and_out_of_range() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuVectorField::new(&ctx);
    // Anisotropic 4x3x2 grid in a plain box.
    let dims = (4, 3, 2);
    let (flat, vecs) = build_field(dims, 0x1234_5678_9abc_def0);
    let fx = Fixture {
        dims,
        bounds_min: [0.0, 0.0, 0.0],
        bounds_max: [8.0, 6.0, 4.0],
        flat,
        vecs,
        queries: vec![
            // Interior sample, fracs clear of the 0.5 tie.
            query([1.3, 0.65, 0.4], WRAP_CLAMP, [3.1, 2.2, 1.7], [1, 1, 1]),
            // Negative coordinate: Clamp pins to the first texel.
            query([-1.35, -0.7, 0.25], WRAP_CLAMP, [-5.0, 1.0, 2.0], [0, 0, 0]),
            // Past the far edge: Clamp pins to the last texel.
            query([6.4, 5.3, 3.65], WRAP_CLAMP, [20.0, 10.0, 8.0], [3, 2, 1]),
        ],
    };
    assert_fixture(&gpu, &ctx, &fx);
}

#[test]
fn tile_sampling_wraps_periodically() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuVectorField::new(&ctx);
    let dims = (3, 4, 3);
    let (flat, vecs) = build_field(dims, 0x0bad_c0de_f00d_1357);
    let fx = Fixture {
        dims,
        bounds_min: [-2.0, -1.0, 0.5],
        bounds_max: [4.0, 7.0, 6.5],
        flat,
        vecs,
        queries: vec![
            // In-range tile sample.
            query([1.3, 2.65, 1.4], WRAP_TILE, [0.3, 1.2, 2.7], [1, 2, 1]),
            // Negative tile: index wraps forward, frac stays mid-cell.
            query([-1.35, -2.3, -3.7], WRAP_TILE, [-1.0, 0.0, 3.0], [0, 1, 2]),
            // Far positive tile, well away from the integer flip.
            query([7.4, 9.3, 8.25], WRAP_TILE, [3.0, 6.0, 5.0], [2, 3, 0]),
        ],
    };
    assert_fixture(&gpu, &ctx, &fx);
}

#[test]
fn degenerate_dim_one_axis() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuVectorField::new(&ctx);
    // A flat field with a single Y layer: every Y read clamps to index 0 and the
    // Y central difference is zero.
    let dims = (3, 1, 4);
    let (flat, vecs) = build_field(dims, 0xfeed_face_dead_beef);
    let fx = Fixture {
        dims,
        bounds_min: [0.0, 0.0, 0.0],
        bounds_max: [6.0, 2.0, 8.0],
        flat,
        vecs,
        queries: vec![
            query([1.3, 0.4, 2.65], WRAP_CLAMP, [2.1, 0.9, 5.3], [1, 0, 2]),
            query([0.35, 0.7, 1.4], WRAP_TILE, [1.0, 1.5, 3.0], [0, 0, 1]),
        ],
    };
    assert_fixture(&gpu, &ctx, &fx);
}

#[test]
fn transform_zero_extent_axis_maps_to_zero() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuVectorField::new(&ctx);
    // The Y box extent is zero, so world_to_grid returns 0 on that axis instead
    // of dividing by zero; X and Z remain ordinary non-degenerate maps.
    let dims = (4, 2, 3);
    let (flat, vecs) = build_field(dims, 0x00c0_ffee_1234_5678);
    let fx = Fixture {
        dims,
        bounds_min: [1.0, 3.0, -2.0],
        bounds_max: [9.0, 3.0, 4.0],
        flat,
        vecs,
        queries: vec![
            query([1.3, 0.4, 1.65], WRAP_CLAMP, [4.2, 3.0, 0.7], [1, 1, 1]),
            query([2.35, 0.7, 0.4], WRAP_CLAMP, [7.0, 5.0, -1.0], [2, 0, 2]),
        ],
    };
    assert_fixture(&gpu, &ctx, &fx);
}

#[test]
fn batch_mixes_wrap_modes() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuVectorField::new(&ctx);
    // One dispatch with both wrap modes interleaved exercises the
    // one-thread-per-query flattening; each result is independent.
    let dims = (4, 4, 4);
    let (flat, vecs) = build_field(dims, 0x5555_aaaa_3333_cccc);
    let fx = Fixture {
        dims,
        bounds_min: [-4.0, -4.0, -4.0],
        bounds_max: [4.0, 4.0, 4.0],
        flat,
        vecs,
        queries: vec![
            query([1.3, 2.4, 0.65], WRAP_CLAMP, [0.3, -1.2, 2.7], [1, 2, 0]),
            query([-1.35, 4.3, 5.7], WRAP_TILE, [-3.0, 3.0, -2.0], [0, 3, 3]),
            query([2.65, 1.4, 3.3], WRAP_CLAMP, [1.0, 1.0, 1.0], [2, 1, 3]),
            query([5.4, -0.7, 2.25], WRAP_TILE, [3.5, -3.5, 0.0], [3, 0, 2]),
        ],
    };
    assert_fixture(&gpu, &ctx, &fx);
}

#[test]
fn empty_batch_short_circuits() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuVectorField::new(&ctx);
    // No dispatch is issued and the result vector is empty.
    let out = gpu.evaluate(
        &ctx,
        (2, 2, 2),
        [0.0, 0.0, 0.0],
        [1.0, 1.0, 1.0],
        &[
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 1.0],
            [0.0, 1.0, 1.0],
            [1.0, 1.0, 1.0],
        ],
        &[],
    );
    assert!(out.is_empty());
}
