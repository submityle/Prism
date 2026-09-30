// Wind-field weather-map advection twin: advances a weather field one step
// along a WindField (base flow plus divergence-free curl gust modulated by each
// cell's wind-disturbance channel), mirroring the CPU golden `advect_with_wind`
// (in `prism_render_architecture::volumetric::weather`).
//
// This is the per-cell-velocity variant of `advect_semi_lagrangian`: instead of
// a single uniform wind, each cell samples its own velocity from the wind field
// (`WindField::velocity_at` with that cell's wind_disturbance) and traces that
// cell backward, then bilinearly samples the previous field with clamp-to-edge.
// Because the curl perturbation is divergence-free the total coverage stays
// near-conserved for interior features. Every output cell is independent, so
// the grid advects in parallel: one invocation per cell.
//
// The velocity combines the base flow `direction * speed` with the curl shape
// `(sin(f x) cos(f y), -cos(f x) sin(f y))` (f = CURL_FREQUENCY) scaled by the
// gust `speed * curl_strength * (1 + saturate(local_disturbance))`. The CPU
// golden reduces angles with a hand-rolled sin_approx/cos_approx (Taylor
// polynomial after range reduction to [-PI/2, PI/2], no float intrinsic, for
// cross-target determinism); this kernel mirrors that same range reduction and
// polynomial, and the same clamp/truncation/lerp bilinear tap, so CPU and GPU
// agree to a few ULP. `cos_approx(x)` is `sin_approx(x + PI/2)`, exactly as on
// the CPU. It uses only the portable core-WGSL subset, so it runs unmodified on
// Metal, Vulkan and DX12.
//
// Provenance: standard semi-Lagrangian backtrace with divergence-free curl-
// noise wind advection; no Unreal Engine source or derived code.

// Angle constants matching `PI`, `TWO_PI` and `FRAC_PI_2` in the CPU `math`
// module (high-precision literals round to the same f32 bits).
const PI: f32 = 3.14159265358979323846;
const TWO_PI: f32 = 6.28318530717958647692;
const FRAC_PI_2: f32 = 1.57079632679489661923;
// Matches `CURL_FREQUENCY` in the CPU `weather` module.
const CURL_FREQUENCY: f32 = 0.15;

struct Params {
    width: u32,
    height: u32,
    dir_x: f32,
    dir_y: f32,
    speed: f32,
    curl_strength: f32,
    dt: f32,
    pad0: u32,
}

// One weather-map cell. Matches `GpuSample` in `src/advect_with_wind.rs` and
// `WeatherSample` channel order.
struct Sample {
    coverage: f32,
    cloud_type: f32,
    precipitation: f32,
    wind_disturbance: f32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> input: array<Sample>;
@group(0) @binding(2) var<storage, read_write> output: array<Sample>;

fn saturate_s(x: f32) -> f32 {
    return clamp(x, 0.0, 1.0);
}

// Round halves away from zero, matching Rust's `f32::round`.
fn round_ties_away(v: f32) -> f32 {
    let r = floor(abs(v) + 0.5);
    return select(r, -r, v < 0.0);
}

// Reduce an angle to [-PI, PI], mirroring `math::wrap_pi`.
fn wrap_pi(x: f32) -> f32 {
    let k = round_ties_away(x / TWO_PI);
    return x - k * TWO_PI;
}

// Hand-rolled sine, mirroring `math::sin_approx` bit for bit.
fn sin_approx(x: f32) -> f32 {
    var r = wrap_pi(x);
    if (r > FRAC_PI_2) {
        r = PI - r;
    } else if (r < -FRAC_PI_2) {
        r = -PI - r;
    }
    let x2 = r * r;
    return r * (1.0 + x2 * (-1.0 / 6.0 + x2 * (1.0 / 120.0 + x2 * (-1.0 / 5040.0))));
}

// Hand-rolled cosine via `cos(x) = sin(x + PI/2)`, mirroring `math::cos_approx`.
fn cos_approx(x: f32) -> f32 {
    return sin_approx(x + FRAC_PI_2);
}

// Total wind velocity at `pos`, mirroring `WindField::velocity_at`.
fn velocity_at(pos_x: f32, pos_y: f32, local_disturbance: f32) -> vec2<f32> {
    let gust = params.speed * params.curl_strength * (1.0 + saturate_s(local_disturbance));
    let ax = CURL_FREQUENCY * pos_x;
    let ay = CURL_FREQUENCY * pos_y;
    let curl_x = sin_approx(ax) * cos_approx(ay);
    let curl_y = -cos_approx(ax) * sin_approx(ay);
    return vec2<f32>(
        params.dir_x * params.speed + curl_x * gust,
        params.dir_y * params.speed + curl_y * gust,
    );
}

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
fn advect_with_wind_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    let total = params.width * params.height;
    if (idx >= total) {
        return;
    }
    let x = idx % params.width;
    let y = idx / params.width;
    let here = input[idx];
    let vel = velocity_at(f32(x), f32(y), here.wind_disturbance);
    let px = f32(x) - vel.x * params.dt;
    let py = f32(y) - vel.y * params.dt;
    output[idx] = sample_bilinear(px, py);
}
