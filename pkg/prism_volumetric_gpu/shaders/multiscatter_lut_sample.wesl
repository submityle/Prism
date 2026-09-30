// Multi-scatter LUT sampler twin: samples the 3D energy-gain table with
// trilinear interpolation and clamp-to-edge addressing, mirroring the CPU
// golden `MultiScatterLut::sample`
// (in `prism_render_architecture::volumetric::multiscatter`).
//
// The multi-scatter LUT tabulates the energy gain over three physical axes -
// `cos` (view/light angle cosine), accumulated `optical_depth`, and `albedo`
// (design section 7b). A query maps each raw axis coordinate to a normalized
// fraction (clamped into range), resolves the two bracketing cells per axis,
// and blends the eight surrounding cells with trilinear weights before
// saturating. Because the corner weights sum to one and every stored cell is in
// `[0, 1]`, the returned gain is always in `[0, 1]`; out-of-range coordinates
// resolve to the nearest edge value without ever reading out of bounds.
//
// This kernel is a pure multiply/add sampler on the raw table - no
// transcendental, no lattice hash, no optional device feature - so CPU and GPU
// evaluate the identical closed-form algebra and differ at most by a legal
// multiply-add contraction of a few ULP. The per-corner weight is grouped as
// `(wx * wy * wz)` and then multiplied by the cell value, matching the CPU
// `weights[k] * corners[k]` accumulation order. It uses only the portable
// core-WGSL subset, so it runs unmodified on Metal, Vulkan and DX12.
//
// Provenance: standard trilinear LUT sampling with clamp-to-edge; no Unreal
// Engine source or derived code.

const EPS: f32 = 1e-6;

// Dispatch parameters plus the table's fixed axis layout. Matches `Params` in
// `src/multiscatter_lut_sample.rs`.
struct Params {
    count: u32,
    dim0: u32,
    dim1: u32,
    dim2: u32,
    min0: f32,
    min1: f32,
    min2: f32,
    pad0: f32,
    max0: f32,
    max1: f32,
    max2: f32,
    pad1: f32,
}

// One query: the three physical axis coordinates. Matches `GpuQuery` in
// `src/multiscatter_lut_sample.rs`.
struct Query {
    cos: f32,
    depth: f32,
    albedo: f32,
    pad0: f32,
}

@group(0) @binding(0) var<uniform> params: Params;
// Row-major table, length `dim0 * dim1 * dim2`, `flat_index` addressed.
@group(0) @binding(1) var<storage, read> lut: array<f32>;
@group(0) @binding(2) var<storage, read> queries: array<Query>;
// One f32 gain per query, tightly packed.
@group(0) @binding(3) var<storage, read_write> results: array<f32>;

// Maps a raw `value` on `[min, max]` to a normalized fraction in `[0, 1]`,
// guarding a collapsed span and saturating out-of-range inputs.
fn axis_fraction(value: f32, lo: f32, hi: f32) -> f32 {
    let span = hi - lo;
    if (abs(span) < EPS) {
        return 0.0;
    }
    return saturate((value - lo) / span);
}

// Row-major flat index into the table.
fn flat_index(d1: u32, d2: u32, a0: u32, a1: u32, a2: u32) -> u32 {
    return (a0 * d1 + a1) * d2 + a2;
}

@compute @workgroup_size(64)
fn multiscatter_lut_sample_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.count) {
        return;
    }
    let q = queries[idx];

    let tc = axis_fraction(q.cos, params.min0, params.max0);
    let td = axis_fraction(q.depth, params.min1, params.max1);
    let ta = axis_fraction(q.albedo, params.min2, params.max2);

    // Per-axis clamp-to-edge bracketing cells and interpolation fractions.
    var c0 = 0u;
    var c1 = 0u;
    var fc = 0.0;
    if (params.dim0 > 1u) {
        let last = params.dim0 - 1u;
        let pos = saturate(tc) * f32(last);
        let base = floor(pos);
        c0 = min(u32(base), last);
        c1 = min(c0 + 1u, last);
        fc = pos - base;
    }

    var d0 = 0u;
    var d1i = 0u;
    var fd = 0.0;
    if (params.dim1 > 1u) {
        let last = params.dim1 - 1u;
        let pos = saturate(td) * f32(last);
        let base = floor(pos);
        d0 = min(u32(base), last);
        d1i = min(d0 + 1u, last);
        fd = pos - base;
    }

    var a0 = 0u;
    var a1 = 0u;
    var fa = 0.0;
    if (params.dim2 > 1u) {
        let last = params.dim2 - 1u;
        let pos = saturate(ta) * f32(last);
        let base = floor(pos);
        a0 = min(u32(base), last);
        a1 = min(a0 + 1u, last);
        fa = pos - base;
    }

    let dim1 = params.dim1;
    let dim2 = params.dim2;

    var acc = 0.0;
    var k = 0u;
    while (k < 8u) {
        let bx = (k >> 2u) & 1u;
        let by = (k >> 1u) & 1u;
        let bz = k & 1u;

        var ci = c0;
        if (bx == 1u) {
            ci = c1;
        }
        var di = d0;
        if (by == 1u) {
            di = d1i;
        }
        var ai = a0;
        if (bz == 1u) {
            ai = a1;
        }

        var wx = 1.0 - fc;
        if (bx == 1u) {
            wx = fc;
        }
        var wy = 1.0 - fd;
        if (by == 1u) {
            wy = fd;
        }
        var wz = 1.0 - fa;
        if (bz == 1u) {
            wz = fa;
        }

        let cell = lut[flat_index(dim1, dim2, ci, di, ai)];
        acc = acc + (wx * wy * wz) * cell;
        k = k + 1u;
    }

    results[idx] = saturate(acc);
}
