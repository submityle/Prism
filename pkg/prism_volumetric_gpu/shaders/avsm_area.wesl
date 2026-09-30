// AVSM curve-area twin: computes the trapezoidal area under each adaptive
// volumetric shadow-map curve's transmittance profile, mirroring the CPU
// golden `AvsmCurve::area` (in `prism_render_architecture::volumetric::avsm`).
//
// The area under the piecewise-linear (depth, transmittance) curve is the
// compression error metric: comparing the area before and after
// `AvsmCurve::compress` bounds how far the adaptive merge shifted the curve.
// The insert/compress state machine that builds each curve is inherently
// sequential and stays on the CPU; this kernel twins only the pure whole-curve
// reduction. Many curves (one per froxel/shadow texel) are reduced in parallel:
// each invocation owns one curve, walks its own [offset, offset+count) slice of
// the shared flattened node array, and sums the trapezoid areas of adjacent
// node pairs. A curve with fewer than two nodes encloses no area, so its result
// is 0, exactly as on the CPU. The accumulation order (ascending node index)
// and the per-segment `0.5 * (lo.t + hi.t) * width` form match the CPU golden,
// so the two agree to a few ULP. It uses only the portable core-WGSL subset, so
// it runs unmodified on Metal, Vulkan and DX12.
//
// Provenance: standard trapezoidal-rule integration of a shadow-curve profile;
// no Unreal Engine source or derived code.

struct Params {
    curve_count: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

// One curve control point. Matches `GpuNode` in `src/avsm_area.rs`.
struct Node {
    depth: f32,
    transmittance: f32,
}

// Where one curve's nodes live in the shared flattened array. Matches
// `GpuCurveRange` in `src/avsm_area.rs`.
struct CurveRange {
    offset: u32,
    count: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> nodes: array<Node>;
@group(0) @binding(2) var<storage, read> ranges: array<CurveRange>;
@group(0) @binding(3) var<storage, read_write> results: array<f32>;

// Trapezoidal area under one curve's [offset, offset+count) node slice,
// mirroring `AvsmCurve::area`.
fn curve_area(offset: u32, count: u32) -> f32 {
    if (count < 2u) {
        return 0.0;
    }
    var sum = 0.0;
    var i = 0u;
    loop {
        if (i + 1u >= count) {
            break;
        }
        let lo = nodes[offset + i];
        let hi = nodes[offset + i + 1u];
        let width = hi.depth - lo.depth;
        sum = sum + 0.5 * (lo.transmittance + hi.transmittance) * width;
        i = i + 1u;
    }
    return sum;
}

@compute @workgroup_size(64)
fn avsm_area_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.curve_count) {
        return;
    }
    let r = ranges[idx];
    results[idx] = curve_area(r.offset, r.count);
}
