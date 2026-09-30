// Storm vertical-profile twin: evaluates on-device the composite vertical
// density weight of a cumulonimbus body, mirroring the CPU golden
// `StormState::vertical_profile` (in
// `prism_render_architecture::volumetric::storm`).
//
// The cumulonimbus vertical-development model (design section 9b) folds the
// storm state machine's evolving fields back through the module's authored
// curves into a single 0..=1 weight the density field multiplies the deep
// convective column by. `vertical_profile` unions the spreading `anvil_profile`
// (driven by `anvil_spread`) with the `overshooting_bump` dome (driven by
// `overshooting_top`, reinforced by the `pyrocumulus` plume) via a probabilistic
// OR `a + b - a*b`:
//   anvil      = anvil_profile(height_fraction, anvil_spread)
//   dome_drive = saturate(overshooting_top + pyrocumulus * (1 - overshooting_top))
//   dome       = overshooting_bump(height_fraction, dome_drive)
//   result     = saturate(anvil + dome - anvil * dome)
// so two overlapping contributions never sum past one and the result stays in
// 0..=1, monotonically non-decreasing in every driving field. The CPU golden
// `StormState::vertical_profile` owns that math; this kernel is the on-device
// twin that runs one thread per query and reproduces the same value.
//
// The `lerp` / `smoothstep` are expanded to the SAME closed forms the CPU
// `math` module uses, and the exponential in the dome uses the SAME hand-rolled
// `exp_approx` the reference uses (base-two range reduction: a fractional
// seven-term polynomial times an integer power assembled from the float
// exponent field), NOT the device-native `exp`, so the twin stays bit-close to
// the reference and the parity test can assert a tight tolerance.
//
// It uses only the portable core-WGSL subset - floor, bitcast, integer ops,
// clamp/saturate and multiply/add - so it runs unmodified on Metal, Vulkan and
// DX12.
//
// Provenance: standard cumulonimbus vertical-development composite; no Unreal
// Engine source or derived code.

const LOG2_E: f32 = 1.4426950408889634;
// Guard span matching `EPS` in the CPU `math` module.
const EPS: f32 = 1e-6;
// Gaussian sharpness of the overshooting dome, matching `OVERSHOOT_SHARPNESS`.
const OVERSHOOT_SHARPNESS: f32 = 12.0;

struct Params {
    count: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

// One storm vertical-profile query: the normalized band height plus the three
// driving storm-state fields. Matches `GpuQuery` in
// `src/storm_vertical_profile.rs`.
struct Query {
    height_fraction: f32,
    anvil_spread: f32,
    overshooting_top: f32,
    pyrocumulus: f32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> queries: array<Query>;
@group(0) @binding(2) var<storage, read_write> results: array<f32>;

fn exp2_int(k: i32) -> f32 {
    if (k < -126) {
        return 0.0;
    }
    if (k > 127) {
        return 3.4028235e38 * 3.4028235e38;
    }
    return bitcast<f32>(u32(k + 127) << 23u);
}

fn exp2_frac(f: f32) -> f32 {
    let c1 = 0.69314718;
    let c2 = 0.2402265;
    let c3 = 0.05550411;
    let c4 = 0.00961813;
    let c5 = 0.00133336;
    let c6 = 0.00015403;
    return 1.0 + f * (c1 + f * (c2 + f * (c3 + f * (c4 + f * (c5 + f * c6)))));
}

fn exp_approx(x: f32) -> f32 {
    let t = x * LOG2_E;
    let k = floor(t);
    let frac = t - k;
    return exp2_frac(frac) * exp2_int(i32(k));
}

// Linear interpolation `a + (b - a) * t`, mirroring `math::lerp` bit for bit.
fn lerp_approx(a: f32, b: f32, t: f32) -> f32 {
    return a + (b - a) * t;
}

// Cubic Hermite smoothstep, mirroring `math::smoothstep` bit for bit.
fn smoothstep_approx(edge0: f32, edge1: f32, x: f32) -> f32 {
    let span = edge1 - edge0;
    if (abs(span) < EPS) {
        if (x < edge0) {
            return 0.0;
        }
        return 1.0;
    }
    let t = saturate((x - edge0) / span);
    return t * t * (3.0 - 2.0 * t);
}

// Anvil density weight, mirroring `anvil_profile`.
fn anvil_profile(height_fraction: f32, spread: f32) -> f32 {
    let h = saturate(height_fraction);
    let s = saturate(spread);
    let base = lerp_approx(0.85, 0.5, s);
    return saturate(s * smoothstep_approx(base, 1.0, h));
}

// Overshooting-top dome weight, mirroring `overshooting_bump`.
fn overshooting_bump(height_fraction: f32, overshooting_top: f32) -> f32 {
    let top = saturate(overshooting_top);
    let d = height_fraction - 1.0;
    return saturate(top * exp_approx(-OVERSHOOT_SHARPNESS * d * d));
}

// Composite vertical density weight, mirroring `StormState::vertical_profile`.
fn vertical_profile(
    height_fraction: f32,
    anvil_spread: f32,
    overshooting_top: f32,
    pyrocumulus: f32,
) -> f32 {
    let anvil = anvil_profile(height_fraction, anvil_spread);
    let dome_drive = saturate(overshooting_top + pyrocumulus * (1.0 - overshooting_top));
    let dome = overshooting_bump(height_fraction, dome_drive);
    return saturate(anvil + dome - anvil * dome);
}

@compute @workgroup_size(64)
fn storm_vertical_profile_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.count) {
        return;
    }
    let q = queries[idx];
    results[idx] = vertical_profile(
        q.height_fraction,
        q.anvil_spread,
        q.overshooting_top,
        q.pyrocumulus,
    );
}
