// Displaced-micro-map scale/bias reduction kernel.
//
// A single invocation determines the per-triangle quantization range. In
// per-triangle mode it scans the sampled heights for the exact [min, max] in
// the same sequential order as the CPU golden (initialising both from
// heights[0] and comparing the rest with `<` / `>`), which for a set of finite
// floats is order-independent and therefore byte-identical to the CPU result.
// In fixed mode it copies the caller-provided range unchanged.
//
// This mirrors the scale/bias derivation inside `prism_dmm::bake_triangle`.
// Provenance: classical min/max reduction; no Unreal Engine source and no
// AI/ML.

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

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> heights: array<f32>;
@group(0) @binding(2) var<storage, read_write> scalebias: array<f32>;

@compute @workgroup_size(1)
fn main() {
    if params.mode == 1u {
        scalebias[0] = params.fixed_min;
        scalebias[1] = params.fixed_max;
        return;
    }

    var mn = heights[0];
    var mx = heights[0];
    for (var j = 1u; j < params.vertex_count; j += 1u) {
        let h = heights[j];
        if h < mn {
            mn = h;
        }
        if h > mx {
            mx = h;
        }
    }
    scalebias[0] = mn;
    scalebias[1] = mx;
}
