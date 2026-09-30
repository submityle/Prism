// Probe-grid sampler twin: samples band-resolved multi-scatter irradiance from
// a regular 3D probe grid with trilinear interpolation and clamp-to-edge,
// mirroring the CPU golden `ProbeGrid::sample`
// (in `prism_render_architecture::volumetric::multiscatter`).
//
// The probe grid caches directional irradiance in six signed-axis bands at each
// lattice corner over an axis-aligned box (design section 7b). A query maps its
// world position to per-axis fractions (clamped into the box), resolves the two
// bracketing probes per axis, and blends the eight surrounding probes per band
// with trilinear weights. The eight corner weights sum to one, so each band is
// a true convex combination of the stored irradiance; points outside the box
// clamp to the boundary probes without ever reading out of bounds.
//
// This kernel is a pure multiply/add sampler on the raw probe data plus integer
// address arithmetic - no transcendental, no lattice hash, no optional device
// feature - so CPU and GPU evaluate the identical closed-form algebra and
// differ at most by a legal multiply-add contraction of a few ULP. The per-band
// accumulation multiplies each corner weight by the band value in corner order,
// matching the CPU `weights[k] * irradiance[band]` order. It uses only the
// portable core-WGSL subset, so it runs unmodified on Metal, Vulkan and DX12.
//
// Provenance: standard trilinear grid sampling with clamp-to-edge; no Unreal
// Engine source or derived code.

const EPS: f32 = 1e-6;
const BANDS: u32 = 6u;

// Dispatch parameters plus the grid's box layout. Matches `Params` in
// `src/probe_grid_sample.rs`.
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

// One query: a world-space sample position. Matches `GpuQuery` in
// `src/probe_grid_sample.rs`.
struct Query {
    x: f32,
    y: f32,
    z: f32,
    pad0: f32,
}

@group(0) @binding(0) var<uniform> params: Params;
// Row-major probes, `dim0 * dim1 * dim2` cells of `BANDS` f32 each, band-inner.
@group(0) @binding(1) var<storage, read> probes: array<f32>;
@group(0) @binding(2) var<storage, read> queries: array<Query>;
// `BANDS` f32 per query, tightly packed.
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

// Row-major flat probe index.
fn flat_index(d1: u32, d2: u32, a0: u32, a1: u32, a2: u32) -> u32 {
    return (a0 * d1 + a1) * d2 + a2;
}

@compute @workgroup_size(64)
fn probe_grid_sample_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.count) {
        return;
    }
    let q = queries[idx];

    let tx = axis_fraction(q.x, params.min0, params.max0);
    let ty = axis_fraction(q.y, params.min1, params.max1);
    let tz = axis_fraction(q.z, params.min2, params.max2);

    // Per-axis clamp-to-edge bracketing probes and interpolation fractions.
    var x0 = 0u;
    var x1 = 0u;
    var fx = 0.0;
    if (params.dim0 > 1u) {
        let last = params.dim0 - 1u;
        let pos = saturate(tx) * f32(last);
        let base = floor(pos);
        x0 = min(u32(base), last);
        x1 = min(x0 + 1u, last);
        fx = pos - base;
    }

    var y0 = 0u;
    var y1 = 0u;
    var fy = 0.0;
    if (params.dim1 > 1u) {
        let last = params.dim1 - 1u;
        let pos = saturate(ty) * f32(last);
        let base = floor(pos);
        y0 = min(u32(base), last);
        y1 = min(y0 + 1u, last);
        fy = pos - base;
    }

    var z0 = 0u;
    var z1 = 0u;
    var fz = 0.0;
    if (params.dim2 > 1u) {
        let last = params.dim2 - 1u;
        let pos = saturate(tz) * f32(last);
        let base = floor(pos);
        z0 = min(u32(base), last);
        z1 = min(z0 + 1u, last);
        fz = pos - base;
    }

    let dim1 = params.dim1;
    let dim2 = params.dim2;

    // Eight corner probe indices in the CPU corner order.
    var corner: array<u32, 8>;
    corner[0] = flat_index(dim1, dim2, x0, y0, z0);
    corner[1] = flat_index(dim1, dim2, x0, y0, z1);
    corner[2] = flat_index(dim1, dim2, x0, y1, z0);
    corner[3] = flat_index(dim1, dim2, x0, y1, z1);
    corner[4] = flat_index(dim1, dim2, x1, y0, z0);
    corner[5] = flat_index(dim1, dim2, x1, y0, z1);
    corner[6] = flat_index(dim1, dim2, x1, y1, z0);
    corner[7] = flat_index(dim1, dim2, x1, y1, z1);

    // Eight corner weights in the same order (bit2->fx, bit1->fy, bit0->fz).
    var weight: array<f32, 8>;
    var k = 0u;
    while (k < 8u) {
        let bx = (k >> 2u) & 1u;
        let by = (k >> 1u) & 1u;
        let bz = k & 1u;
        var wx = 1.0 - fx;
        if (bx == 1u) {
            wx = fx;
        }
        var wy = 1.0 - fy;
        if (by == 1u) {
            wy = fy;
        }
        var wz = 1.0 - fz;
        if (bz == 1u) {
            wz = fz;
        }
        weight[k] = wx * wy * wz;
        k = k + 1u;
    }

    let base = idx * BANDS;
    var band = 0u;
    while (band < BANDS) {
        var acc = 0.0;
        var c = 0u;
        while (c < 8u) {
            acc = acc + weight[c] * probes[corner[c] * BANDS + band];
            c = c + 1u;
        }
        results[base + band] = acc;
        band = band + 1u;
    }
}
