// Semi-Lagrangian weather-map advection twin: advances a weather field one step
// along a uniform wind by a backward trace, mirroring the CPU golden
// `advect_semi_lagrangian` (in `prism_render_architecture::volumetric::weather`).
//
// The weather map is a row-major grid of 4-channel samples (coverage,
// cloud_type, precipitation, wind_disturbance). Each cell centre (x, y) is
// traced backward to (x - wind_x*dt, y - wind_y*dt) in cell units and the
// previous field is bilinearly sampled there with clamp-to-edge, exactly as the
// CPU `WeatherField::sample_bilinear`. The scheme is unconditionally stable and
// preserves each channel's convex range, so it is embarrassingly parallel: one
// invocation owns one output cell. Every arithmetic step (clamp, truncation,
// per-channel lerp) mirrors the CPU golden, so the two agree to a few ULP. It
// uses only the portable core-WGSL subset, so it runs unmodified on Metal,
// Vulkan and DX12.
//
// Provenance: standard semi-Lagrangian backtrace with bilinear resampling
// (Stam-style stable advection); no Unreal Engine source or derived code.

struct Params {
    width: u32,
    height: u32,
    wind_x: f32,
    wind_y: f32,
    dt: f32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

// One weather-map cell. Matches `GpuSample` in
// `src/advect_semi_lagrangian.rs` and `WeatherSample` channel order.
struct Sample {
    coverage: f32,
    cloud_type: f32,
    precipitation: f32,
    wind_disturbance: f32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> input: array<Sample>;
@group(0) @binding(2) var<storage, read_write> output: array<Sample>;

// Component-wise linear blend, mirroring `WeatherSample::lerp` / `math::lerp`.
fn sample_lerp(a: Sample, b: Sample, t: f32) -> Sample {
    var r: Sample;
    r.coverage = a.coverage + (b.coverage - a.coverage) * t;
    r.cloud_type = a.cloud_type + (b.cloud_type - a.cloud_type) * t;
    r.precipitation = a.precipitation + (b.precipitation - a.precipitation) * t;
    r.wind_disturbance = a.wind_disturbance + (b.wind_disturbance - a.wind_disturbance) * t;
    return r;
}

// Clamp-to-edge cell fetch, mirroring `WeatherField::get`.
fn cell(x: u32, y: u32) -> Sample {
    let cx = min(x, params.width - 1u);
    let cy = min(y, params.height - 1u);
    return input[cy * params.width + cx];
}

// Bilinear sample at fractional cell coordinates, mirroring
// `WeatherField::sample_bilinear`.
fn sample_bilinear(u: f32, v: f32) -> Sample {
    let max_x = f32(params.width - 1u);
    let max_y = f32(params.height - 1u);
    let cx = clamp(u, 0.0, max_x);
    let cy = clamp(v, 0.0, max_y);
    let x0 = u32(cx);
    let y0 = u32(cy);
    let x1 = min(x0 + 1u, params.width - 1u);
    let y1 = min(y0 + 1u, params.height - 1u);
    let fx = cx - f32(x0);
    let fy = cy - f32(y0);
    let s00 = cell(x0, y0);
    let s10 = cell(x1, y0);
    let s01 = cell(x0, y1);
    let s11 = cell(x1, y1);
    let top = sample_lerp(s00, s10, fx);
    let bottom = sample_lerp(s01, s11, fx);
    return sample_lerp(top, bottom, fy);
}

@compute @workgroup_size(64)
fn advect_semi_lagrangian_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    let total = params.width * params.height;
    if (idx >= total) {
        return;
    }
    let x = idx % params.width;
    let y = idx / params.width;
    let px = f32(x) - params.wind_x * params.dt;
    let py = f32(y) - params.wind_y * params.dt;
    output[idx] = sample_bilinear(px, py);
}
