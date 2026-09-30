// Prism virtual-geometry vis-buffer scanline twin: solves each row's contiguous
// coverage interval in closed form and evaluates screen-space-linear depth from
// an edge-function triple on-device, mirroring the CPU golden
// `TriangleGradients::row_span` and `TriangleGradients::depth_from_edges` (in
// `prism_render_architecture::virtual_geometry`).
//
// The shipping vis-buffer software rasterizer does not test every pixel on a
// wide row. Once a row is wide enough it stops per-pixel testing and instead
// solves the three affine edge inequalities `w_row[i] + k * w_x[i] >= 0` for
// the row, whose intersection is a single interval `[k_lo, k_hi]` in `+x` step
// units, then walks only that span while stepping the depth plane by `z_x` per
// column. This kernel is the on-device twin of that closed form: one thread per
// query solves the interval and (independently) evaluates the barycentric depth
// `dot(vertices_z, w)` the walk carries, so the twin's fast-path span and its
// depth step are validated against the reference rather than merely compiled.
//
// Coverage uses inclusive `w_i >= 0` (not the top-left tie-break of the
// per-pixel golden path); at generic pixel centers no edge value is exactly
// zero, so the interval matches a per-pixel `all(w_i >= 0)` scan bit-for-bit.
//
// Numerical model: `w_x` and `w_row` are built from integer / dyadic pixel
// coordinates so each is exactly representable; a lone reciprocal-free division
// `-w0 / g` is correctly rounded identically on CPU and GPU (no
// fused-multiply-add can contract a single divide), and `ceil` / `floor` of an
// exactly-representable quotient agree, so the interval bounds are bit-exact.
// When the double area is an exact power of two the depth weights `vertices_z`
// are exact, so each `vertices_z[i] * w[i]` product and their sum are fma-immune
// and the depth is bit-exact too. The kernel uses only the portable core-WGSL
// subset - no optional feature - so it runs unmodified on Metal, Vulkan and
// DX12.
//
// Provenance: standard affine edge-inequality span solve for software
// rasterization; no Unreal Engine source or derived code.

struct Params {
    // Number of queries in `inputs` / outputs in `spans`.
    count: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

// One scanline query. `52`-byte stride matching the `bytemuck` upload struct.
struct ScanInput {
    // Per-column (`+x`) edge-function increments (`TriangleGradients::w_x`).
    wx0: f32, wx1: f32, wx2: f32,
    // Depth weights `(v0.z, v1.z, v2.z) / double_area` (`vertices_z`).
    vz0: f32, vz1: f32, vz2: f32,
    // Edge-function triple at the row's first candidate pixel center `p0`.
    wrow0: f32, wrow1: f32, wrow2: f32,
    // Number of `+x` unit steps to the last candidate (`k in 0..=steps`).
    steps: u32,
    // Edge-function triple at the depth query pixel (for `depth_from_edges`).
    dw0: f32, dw1: f32, dw2: f32,
}

// One scanline result. `16`-byte stride.
struct ScanOutput {
    // `1u` row covered (span valid), `0u` row not covered.
    valid: u32,
    // Inclusive span bounds in `+x` step units, clamped to `[0, steps]`.
    k_lo: u32,
    k_hi: u32,
    // `dot(vertices_z, dw)` - barycentric depth of the walk, always written.
    depth: f32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> inputs: array<ScanInput>;
@group(0) @binding(2) var<storage, read_write> spans: array<ScanOutput>;

@compute @workgroup_size(64)
fn scan(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= params.count) {
        return;
    }

    let s = inputs[i];
    let steps_f = f32(s.steps);

    // depth_from_edges is coverage-independent and always written.
    let depth = s.vz0 * s.dw0 + s.vz1 * s.dw1 + s.vz2 * s.dw2;

    var lo = 0.0;
    var hi = steps_f;
    var covered = true;

    // Edge 0: w_row0 + k * wx0 >= 0.
    if (s.wx0 > 0.0) {
        lo = max(lo, -s.wrow0 / s.wx0);
    } else if (s.wx0 < 0.0) {
        hi = min(hi, -s.wrow0 / s.wx0);
    } else if (s.wrow0 < 0.0) {
        covered = false;
    }
    // Edge 1.
    if (s.wx1 > 0.0) {
        lo = max(lo, -s.wrow1 / s.wx1);
    } else if (s.wx1 < 0.0) {
        hi = min(hi, -s.wrow1 / s.wx1);
    } else if (s.wrow1 < 0.0) {
        covered = false;
    }
    // Edge 2.
    if (s.wx2 > 0.0) {
        lo = max(lo, -s.wrow2 / s.wx2);
    } else if (s.wx2 < 0.0) {
        hi = min(hi, -s.wrow2 / s.wx2);
    } else if (s.wrow2 < 0.0) {
        covered = false;
    }

    if (!covered) {
        spans[i] = ScanOutput(0u, 0u, 0u, depth);
        return;
    }

    let k_lo = ceil(lo);
    let k_hi = floor(hi);
    if (k_lo > k_hi || k_hi < 0.0 || k_lo > steps_f) {
        spans[i] = ScanOutput(0u, 0u, 0u, depth);
        return;
    }

    let lo_u = u32(max(k_lo, 0.0));
    let hi_u = u32(min(k_hi, steps_f));
    spans[i] = ScanOutput(1u, lo_u, hi_u, depth);
}
