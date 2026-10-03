// Displaced-micro-map 11-bit unorm quantization kernel.
//
// One invocation quantizes one micro-vertex height. The height is normalised
// into [0, 1] using the per-triangle [min, max] range (a degenerate flat range
// maps every height to 0 to avoid a division by zero), then rounded to an
// 11-bit unorm code in 0..=2047. Rounding uses `floor(x + 0.5)` rather than the
// WGSL `round` builtin because the CPU golden quantizes with `libm::roundf`
// (round-half-away-from-zero); for the non-negative scaled values here that is
// exactly `floor(x + 0.5)`, whereas WGSL `round` is round-half-to-even and
// would diverge on exact halves.
//
// This mirrors `prism_dmm::quantize` element for element. Provenance: classical
// round-to-nearest quantization; no Unreal Engine source and no AI/ML.

struct Params {
    ux0: f32,
    uy0: f32,
    ux1: f32,
    uy1: f32,
    ux2: f32,
    uy2: f32,
    n: u32,
    width: u32,
    height: u32,
    wrap: u32,
    mode: u32,
    vertex_count: u32,
    fixed_min: f32,
    fixed_max: f32,
    pad0: u32,
    pad1: u32,
};

// Largest representable 11-bit unorm code.
const UNORM11_MAX: f32 = 2047.0;

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> heights: array<f32>;
@group(0) @binding(2) var<storage, read> scalebias: array<f32>;
@group(0) @binding(3) var<storage, read_write> codes: array<u32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if idx >= params.vertex_count {
        return;
    }

    let mn = scalebias[0];
    let mx = scalebias[1];
    let h = heights[idx];

    var t: f32;
    if mx == mn {
        t = 0.0;
    } else {
        t = clamp((h - mn) / (mx - mn), 0.0, 1.0);
    }

    let clamped = clamp(t, 0.0, 1.0);
    let scaled = floor(clamped * UNORM11_MAX + 0.5);
    codes[idx] = u32(clamp(scaled, 0.0, UNORM11_MAX));
}
