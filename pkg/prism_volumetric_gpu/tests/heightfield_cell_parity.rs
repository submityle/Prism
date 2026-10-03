//! Real-device parity for the per-cell-triangle ray intersection twin:
//! [`GpuHeightfieldCell`](prism_volumetric_gpu::heightfield_cell::GpuHeightfieldCell)
//! must reproduce the `CPU` closed form of
//! `prism_render_architecture::ray_scene::heightfield` —
//! `Heightfield::intersect_cell_triangle` (Möller–Trumbore over an implicit
//! displacement grid) — across front- and back-face hits on both triangles of
//! a cell, clear misses, interval-clipped misses and a randomized sweep
//! compared query-for-query.
//!
//! The tests skip (returning early) when the host has no `wgpu` adapter, so the
//! suite stays green everywhere while still exercising the full
//! dispatch-and-readback on any real device such as an Apple `M`-series `GPU`.
//! The kernel is portable core-`WGSL`, so it needs no optional device feature.
//!
//! # Oracle
//!
//! This wave forbids depending on the golden crate, so the host oracle is an
//! *independent* reimplementation of the same closed form: the implicit grid
//! vertex `origin + (ix/(width-1) * extent.x, heights[iz*width+ix],
//! iz/(height-1) * extent.y)`, the two triangle corner selections, the
//! Möller–Trumbore barycentric solve with its `EPS` determinant guard, the
//! interval test applied verbatim, the barycentric reconstruction of the hit
//! `position` and domain `uv`, and the geometric normal flipped against the
//! ray and normalized with the golden near-zero fallback. Because the reference
//! and this oracle are both scalar `f32`, a `GPU == oracle` pass is direct
//! evidence the ported kernel computes the same hits the reference does.
//!
//! # Parity criterion
//!
//! Every continuous output threads through products, quotients and one `sqrt`,
//! so a `GPU` result may land a few units in the last place from the scalar
//! oracle; `t`, `u`, `v`, each `position` and `normal` component and each `uv`
//! component are asserted within `abs_diff <= 1e-4` or `rel_diff <= 1e-3`, with
//! a relative floor of `1e-6` so a near-zero expected value does not inflate
//! the relative error. The discrete `hit`, `front_face`, `cell` and `triangle`
//! fields are compared for exact equality.
//!
//! # Conditioning
//!
//! The branch-switch loci are the determinant (`det ~ 0`, ray parallel to the
//! triangle), the barycentric edges (`u` or `v` at `0`, `u + v` at `1`), the
//! ray-interval ends (`t` at `t_min`/`t_max`) and the degenerate geometric
//! normal (`len_sq ~ 0`). The named fixtures sit well inside a single branch
//! and the randomized sweep rejects any sample within a safety margin of each
//! locus, so both sides fold the identical verdict and no cliff can appear.
//!
//! Provenance: 孪生自本仓 `prism_render_architecture::ray_scene::heightfield`；无第三方引擎源码或衍生代码。

use prism_volumetric_gpu::heightfield_cell::{
    GpuHeightfieldCell, HeightfieldCellQuery, HeightfieldCellResult,
};
use prism_volumetric_gpu::GpuContext;

/// Absolute bound on any continuous output. A `GPU` `sqrt`/divide may land a few
/// units in the last place from the scalar oracle; `1e-4` admits that legal
/// slack while still failing a wrong port.
const ABS_EPS: f32 = 1.0e-4;

/// Relative bound on any continuous output, applied for larger magnitudes where
/// a few units in the last place exceed the absolute floor.
const REL_EPS: f32 = 1.0e-3;

/// Floor on the relative-error denominator so a near-zero expected value does
/// not inflate the relative error.
const REL_FLOOR: f32 = 1.0e-6;

/// Determinant magnitude floor shared with the kernel: `abs(det)` at or below
/// this routes to a miss, mirroring the reference `det.abs() < 1e-8` without a
/// forbidden bare `f32` equality.
const EPS: f32 = 1.0e-8;

/// Squared-length floor for the near-zero normal fallback, matching the golden
/// `normalize_or` guard `len_sq < 1e-24`.
const LEN_SQ_FLOOR: f32 = 1.0e-24;

/// Safety margin for the randomized sweep's rejection bands.
const MARGIN: f32 = 0.02;

/// Returns whether `a` and `b` agree within the absolute or relative bound
/// (relative error floored at `REL_FLOOR`).
fn close(a: f32, b: f32) -> bool {
    let abs = (a - b).abs();
    let rel = abs / a.abs().max(b.abs()).max(REL_FLOOR);
    abs <= ABS_EPS || rel <= REL_EPS
}

/// Subtracts `b` from `a` componentwise.
fn sub(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

/// Euclidean dot product of two vectors.
fn dot(a: [f32; 3], b: [f32; 3]) -> f32 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

/// Cross product `a x b`.
fn cross(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

/// An implicit displacement grid: `width * height` row-major height samples over
/// the planar domain `[origin.xz, origin.xz + extent]`, displaced along `Y`.
struct Grid {
    /// Grid columns (vertices along `X`); at least two.
    width: u32,
    /// Grid rows (vertices along `Z`); at least two.
    height: u32,
    /// World-space corner the grid's `(0, 0)` vertex maps to.
    origin: [f32; 3],
    /// Grid extent along `X` and `Z` in world units.
    extent: [f32; 2],
    /// Row-major height samples (`heights[iz * width + ix]`).
    heights: Vec<f32>,
}

impl Grid {
    /// Domain `UV` of grid vertex `(cx, cz)` in `[0, 1]^2`.
    fn grid_param(&self, cx: u32, cz: u32) -> [f32; 2] {
        [
            cx as f32 / (self.width - 1) as f32,
            cz as f32 / (self.height - 1) as f32,
        ]
    }

    /// World-space position of grid vertex `(cx, cz)`.
    fn vertex_pos(&self, cx: u32, cz: u32) -> [f32; 3] {
        let p = self.grid_param(cx, cz);
        let h = self.heights[(cz * self.width + cx) as usize];
        [
            self.origin[0] + p[0] * self.extent[0],
            self.origin[1] + h,
            self.origin[2] + p[1] * self.extent[1],
        ]
    }

    /// The three corner positions and domain `UV`s of `cell`'s `tri`-th
    /// triangle in winding order (`tri` is `0` or `1`).
    fn triangle_corners(&self, cell: u32, tri: u32) -> [([f32; 3], [f32; 2]); 3] {
        let cells_x = self.width - 1;
        let ix = cell % cells_x;
        let iz = cell / cells_x;
        let corner = |cx: u32, cz: u32| (self.vertex_pos(cx, cz), self.grid_param(cx, cz));
        if tri == 0 {
            [corner(ix, iz), corner(ix + 1, iz), corner(ix + 1, iz + 1)]
        } else {
            [corner(ix, iz), corner(ix + 1, iz + 1), corner(ix, iz + 1)]
        }
    }
}

/// Independent reimplementation of
/// `ray_scene::heightfield::Heightfield::intersect_cell_triangle`: decode the
/// implicit triangle corners, run Möller–Trumbore, and reconstruct the hit with
/// its oriented unit normal. Mirrors the kernel lane for lane.
fn intersect_host(grid: &Grid, q: &HeightfieldCellQuery) -> HeightfieldCellResult {
    let miss = HeightfieldCellResult {
        hit: 0,
        t: 0.0,
        u: 0.0,
        v: 0.0,
        position: [0.0, 0.0, 0.0],
        normal: [0.0, 0.0, 0.0],
        uv: [0.0, 0.0],
        front_face: 0,
        cell: 0,
        triangle: 0,
    };

    let [(p0, t0), (p1, t1), (p2, t2)] = grid.triangle_corners(q.cell, q.triangle);
    let e1 = sub(p1, p0);
    let e2 = sub(p2, p0);
    let dir = q.direction;
    let pvec = cross(dir, e2);
    let det = dot(e1, pvec);
    if det.abs() < EPS {
        return miss;
    }
    let inv_det = 1.0 / det;
    let tvec = sub(q.origin, p0);
    let u = dot(tvec, pvec) * inv_det;
    if !(0.0..=1.0).contains(&u) {
        return miss;
    }
    let qvec = cross(tvec, e1);
    let v = dot(dir, qvec) * inv_det;
    if v < 0.0 || u + v > 1.0 {
        return miss;
    }
    let t = dot(e2, qvec) * inv_det;
    if t < q.t_min || t > q.t_max {
        return miss;
    }

    let w0 = 1.0 - u - v;
    let position = [
        w0 * p0[0] + u * p1[0] + v * p2[0],
        w0 * p0[1] + u * p1[1] + v * p2[1],
        w0 * p0[2] + u * p1[2] + v * p2[2],
    ];

    let ng_raw = cross(e1, e2);
    let facing = dot(dir, ng_raw) < 0.0;
    let geo = if facing {
        ng_raw
    } else {
        [-ng_raw[0], -ng_raw[1], -ng_raw[2]]
    };
    let len_sq = dot(geo, geo);
    let normal = if len_sq < LEN_SQ_FLOOR {
        geo
    } else {
        let inv = 1.0 / len_sq.sqrt();
        [geo[0] * inv, geo[1] * inv, geo[2] * inv]
    };

    let uv = [
        w0 * t0[0] + u * t1[0] + v * t2[0],
        w0 * t0[1] + u * t1[1] + v * t2[1],
    ];

    HeightfieldCellResult {
        hit: 1,
        t,
        u,
        v,
        position,
        normal,
        uv,
        front_face: u32::from(facing),
        cell: q.cell,
        triangle: q.triangle,
    }
}

/// A tiny integer linear-congruential generator; only integer work, so no
/// transcendental appears. Returns the raw high bits as a `u32`.
fn lcg(state: &mut u64) -> u32 {
    *state = state
        .wrapping_mul(6_364_136_223_846_793_005)
        .wrapping_add(1_442_695_040_888_963_407);
    (*state >> 33) as u32
}

/// Draws an `f32` in `[lo, hi]` at ten-thousandth resolution from `state`.
fn draw(state: &mut u64, lo: f32, hi: f32) -> f32 {
    lo + (lcg(state) % 10_001) as f32 / 10_000.0 * (hi - lo)
}

/// Returns `true` when a sweep sample sits too close to a branch-switch locus
/// and must be rejected, so the kept comparison stays far from any cliff. It
/// recomputes the Möller–Trumbore solve and inspects the determinant guard, the
/// barycentric edges, the interval ends and the degenerate normal.
fn reject_sample(grid: &Grid, q: &HeightfieldCellQuery) -> bool {
    let [(p0, _), (p1, _), (p2, _)] = grid.triangle_corners(q.cell, q.triangle);
    let e1 = sub(p1, p0);
    let e2 = sub(p2, p0);
    let dir = q.direction;
    let pvec = cross(dir, e2);
    let det = dot(e1, pvec);
    // Near-parallel: the determinant guard is ill-conditioned here.
    if det.abs() < 0.05 {
        return true;
    }
    let inv_det = 1.0 / det;
    let tvec = sub(q.origin, p0);
    let u = dot(tvec, pvec) * inv_det;
    // Only roots near the valid interval can flip the verdict.
    if u > -0.5 && u < 1.5 && (u.abs() < MARGIN || (u - 1.0).abs() < MARGIN) {
        return true;
    }
    if !(0.0..=1.0).contains(&u) {
        // Clean miss far from the edge: nothing to disagree about, keep it.
        return false;
    }
    let qvec = cross(tvec, e1);
    let v = dot(dir, qvec) * inv_det;
    if v.abs() < MARGIN || (u + v - 1.0).abs() < MARGIN {
        return true;
    }
    if v < 0.0 || u + v > 1.0 {
        return false;
    }
    let t = dot(e2, qvec) * inv_det;
    if (t - q.t_min).abs() < MARGIN || (t - q.t_max).abs() < MARGIN {
        return true;
    }
    let ng_raw = cross(e1, e2);
    if dot(ng_raw, ng_raw) < 0.01 {
        return true;
    }
    false
}

/// Dispatches `queries` over `grid` and asserts every output matches the host
/// oracle: the discrete flags exactly and, when hit, every continuous field
/// within tolerance.
fn check_batch(
    ctx: &GpuContext,
    gpu: &GpuHeightfieldCell,
    grid: &Grid,
    queries: &[HeightfieldCellQuery],
) {
    let got: Vec<HeightfieldCellResult> = gpu.evaluate(
        ctx,
        grid.width,
        grid.height,
        grid.origin,
        grid.extent,
        &grid.heights,
        queries,
    );
    assert_eq!(got.len(), queries.len(), "one result per query");
    for (q, r) in queries.iter().zip(got.iter()) {
        let golden = intersect_host(grid, q);
        assert_eq!(
            r.hit, golden.hit,
            "hit flag mismatch: gpu={} golden={} (query={q:?})",
            r.hit, golden.hit
        );
        if golden.hit == 1 {
            assert_eq!(
                r.front_face, golden.front_face,
                "front_face mismatch: gpu={} golden={} (query={q:?})",
                r.front_face, golden.front_face
            );
            assert_eq!(
                r.cell, golden.cell,
                "cell mismatch: gpu={} golden={} (query={q:?})",
                r.cell, golden.cell
            );
            assert_eq!(
                r.triangle, golden.triangle,
                "triangle mismatch: gpu={} golden={} (query={q:?})",
                r.triangle, golden.triangle
            );
            assert!(
                close(r.t, golden.t),
                "t mismatch: gpu={} golden={} (query={q:?})",
                r.t,
                golden.t
            );
            assert!(
                close(r.u, golden.u),
                "u mismatch: gpu={} golden={} (query={q:?})",
                r.u,
                golden.u
            );
            assert!(
                close(r.v, golden.v),
                "v mismatch: gpu={} golden={} (query={q:?})",
                r.v,
                golden.v
            );
            for axis in 0..3 {
                assert!(
                    close(r.position[axis], golden.position[axis]),
                    "position[{axis}] mismatch: gpu={} golden={} (query={q:?})",
                    r.position[axis],
                    golden.position[axis]
                );
                assert!(
                    close(r.normal[axis], golden.normal[axis]),
                    "normal[{axis}] mismatch: gpu={} golden={} (query={q:?})",
                    r.normal[axis],
                    golden.normal[axis]
                );
            }
            for axis in 0..2 {
                assert!(
                    close(r.uv[axis], golden.uv[axis]),
                    "uv[{axis}] mismatch: gpu={} golden={} (query={q:?})",
                    r.uv[axis],
                    golden.uv[axis]
                );
            }
        }
    }
}

/// A flat unit cell: `2x2` vertices spanning `[0, 1]^2` in the `X`/`Z` plane at
/// height zero, so the single cell's two triangles both lie in the plane
/// `y = 0`.
fn flat_cell() -> Grid {
    Grid {
        width: 2,
        height: 2,
        origin: [0.0, 0.0, 0.0],
        extent: [1.0, 1.0],
        heights: vec![0.0, 0.0, 0.0, 0.0],
    }
}

/// Builds a straight-down query over cell `0` and triangle `tri` from a planar
/// `(x, z)` entry point above the flat cell.
fn down_query(x: f32, z: f32, tri: u32) -> HeightfieldCellQuery {
    HeightfieldCellQuery {
        cell: 0,
        triangle: tri,
        origin: [x, 1.0, z],
        direction: [0.0, -1.0, 0.0],
        t_min: 0.0,
        t_max: 10.0,
    }
}

#[test]
#[expect(
    clippy::print_stderr,
    reason = "the skip notice must reach the test log on hosts without a wgpu adapter"
)]
fn empty_batch_returns_empty_without_dispatch() {
    let Some(ctx) = GpuContext::try_headless() else {
        eprintln!("skipping heightfield_cell parity: no wgpu adapter available");
        return;
    };
    let gpu = GpuHeightfieldCell::new(&ctx);
    let grid = flat_cell();
    let got = gpu.evaluate(
        &ctx,
        grid.width,
        grid.height,
        grid.origin,
        grid.extent,
        &grid.heights,
        &[],
    );
    assert!(got.is_empty(), "empty batch yields no results");
}

#[test]
fn hit_triangle_zero_from_above() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuHeightfieldCell::new(&ctx);
    let grid = flat_cell();
    // Triangle 0 covers the (z < x) half of the cell; a point with z < x lands
    // on it. The straight-down ray strikes the plane at t = 1.
    let q = down_query(0.6, 0.4, 0);
    let golden = intersect_host(&grid, &q);
    assert_eq!(golden.hit, 1, "point in triangle 0 should hit");
    assert_eq!(golden.triangle, 0, "triangle selector echoed");
    assert!(close(golden.t, 1.0), "flat plane struck at t = 1");
    check_batch(&ctx, &gpu, &grid, &[q]);
}

#[test]
fn hit_triangle_one_from_above() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuHeightfieldCell::new(&ctx);
    let grid = flat_cell();
    // Triangle 1 covers the (z > x) half; a point with z > x lands on it.
    let q = down_query(0.4, 0.6, 1);
    let golden = intersect_host(&grid, &q);
    assert_eq!(golden.hit, 1, "point in triangle 1 should hit");
    assert_eq!(golden.triangle, 1, "triangle selector echoed");
    check_batch(&ctx, &gpu, &grid, &[q]);
}

#[test]
fn front_face_hit_from_below() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuHeightfieldCell::new(&ctx);
    let grid = flat_cell();
    // A ray climbing from below meets the triangle on the counter-clockwise
    // (front) face, so front_face is 1 — the opposite of the top-down fixtures.
    let q = HeightfieldCellQuery {
        cell: 0,
        triangle: 0,
        origin: [0.6, -1.0, 0.4],
        direction: [0.0, 1.0, 0.0],
        t_min: 0.0,
        t_max: 10.0,
    };
    let golden = intersect_host(&grid, &q);
    assert_eq!(golden.hit, 1, "upward ray should hit triangle 0");
    assert_eq!(golden.front_face, 1, "ray meets the front face from below");
    check_batch(&ctx, &gpu, &grid, &[q]);
}

#[test]
fn back_face_hit_from_above() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuHeightfieldCell::new(&ctx);
    let grid = flat_cell();
    let q = down_query(0.6, 0.4, 0);
    let golden = intersect_host(&grid, &q);
    assert_eq!(golden.hit, 1, "downward ray should hit triangle 0");
    assert_eq!(golden.front_face, 0, "top-down ray meets the back face");
    check_batch(&ctx, &gpu, &grid, &[q]);
}

#[test]
fn grazing_parallel_miss() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuHeightfieldCell::new(&ctx);
    let grid = flat_cell();
    // A horizontal ray in the triangle's own plane gives det == 0, a clean miss.
    let q = HeightfieldCellQuery {
        cell: 0,
        triangle: 0,
        origin: [0.5, 0.0, 0.4],
        direction: [1.0, 0.0, 0.0],
        t_min: 0.0,
        t_max: 10.0,
    };
    assert_eq!(intersect_host(&grid, &q).hit, 0, "parallel ray misses");
    check_batch(&ctx, &gpu, &grid, &[q]);
}

#[test]
fn miss_outside_triangle() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuHeightfieldCell::new(&ctx);
    let grid = flat_cell();
    // Far outside the cell footprint: the barycentric u leaves [0, 1].
    let q = down_query(2.5, 0.4, 0);
    assert_eq!(intersect_host(&grid, &q).hit, 0, "out-of-cell ray misses");
    check_batch(&ctx, &gpu, &grid, &[q]);
}

#[test]
fn interval_clipped_miss() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuHeightfieldCell::new(&ctx);
    let grid = flat_cell();
    // The plane sits at t = 1 but the interval caps at 0.5, clipping the hit.
    let mut q = down_query(0.6, 0.4, 0);
    q.t_max = 0.5;
    assert_eq!(intersect_host(&grid, &q).hit, 0, "hit lies beyond t_max");
    check_batch(&ctx, &gpu, &grid, &[q]);
}

#[test]
fn mixed_single_dispatch_batch() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuHeightfieldCell::new(&ctx);
    let grid = flat_cell();
    let mut clipped = down_query(0.6, 0.4, 0);
    clipped.t_max = 0.5;
    let queries = vec![
        down_query(0.6, 0.4, 0),
        down_query(0.4, 0.6, 1),
        down_query(2.5, 0.4, 0),
        clipped,
        HeightfieldCellQuery {
            cell: 0,
            triangle: 0,
            origin: [0.7, -1.0, 0.3],
            direction: [0.0, 1.0, 0.0],
            t_min: 0.0,
            t_max: 10.0,
        },
    ];
    check_batch(&ctx, &gpu, &grid, &queries);
}

#[test]
fn randomized_sweep() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuHeightfieldCell::new(&ctx);
    // A 3x3 displaced grid: four cells, eight triangles, random heights.
    let mut state: u64 = 0x9e37_79b9_7f4a_7c15;
    let width = 3u32;
    let height = 3u32;
    let cells = (width - 1) * (height - 1);
    let mut heights = Vec::new();
    for _ in 0..(width * height) {
        heights.push(draw(&mut state, -0.3, 0.3));
    }
    let grid = Grid {
        width,
        height,
        origin: [0.0, 0.0, 0.0],
        extent: [2.0, 2.0],
        heights,
    };

    let mut queries: Vec<HeightfieldCellQuery> = Vec::new();
    let mut guard = 0u32;
    while queries.len() < 512 {
        guard += 1;
        assert!(guard < 400_000, "rejection sampling failed to converge");

        let cell = lcg(&mut state) % cells;
        let tri = lcg(&mut state) % 2;
        let origin = [
            draw(&mut state, 0.0, 2.0),
            draw(&mut state, 3.0, 5.0),
            draw(&mut state, 0.0, 2.0),
        ];
        let direction = [
            draw(&mut state, -0.3, 0.3),
            -1.0,
            draw(&mut state, -0.3, 0.3),
        ];
        let q = HeightfieldCellQuery {
            cell,
            triangle: tri,
            origin,
            direction,
            t_min: 0.0,
            t_max: 50.0,
        };
        if reject_sample(&grid, &q) {
            continue;
        }
        queries.push(q);
    }
    check_batch(&ctx, &gpu, &grid, &queries);
}
