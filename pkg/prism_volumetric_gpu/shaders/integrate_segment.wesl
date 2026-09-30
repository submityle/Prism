// Segment-integration twin: integrates one homogeneous raymarch segment into
// the accumulator on-device, mirroring the CPU golden `integrate_segment` (in
// `prism_render_architecture::volumetric::raymarch`).
//
// Each marched segment adds its in-scattered radiance and attenuates the
// running transmittance (design section 6). Negative coefficients are clamped
// to zero and `light_transmittance` / `powder_factor` are saturated, so the
// update never amplifies energy or produces NaN. The scattered contribution
// uses the analytic segment integral
//   sigma_s * (1 - exp(-sigma_t*step)) / sigma_t
// degrading to `sigma_s * step` as sigma_t -> 0 (the thin-medium limit). The
// accumulator update is:
//   seg_optical = sigma_t * step
//   seg_trans   = exp(-seg_optical)
//   integral    = (sigma_t > EPS) ? (1 - seg_trans)/sigma_t : step
//   scattered  += transmittance * light * sigma_s * phase * integral * powder
//   transmittance = saturate(transmittance * seg_trans)
//   optical_depth += seg_optical
//   steps_taken   += 1
// The CPU golden owns that logic; this kernel is the on-device twin that runs
// one thread per query.
//
// The CPU golden uses a hand-rolled `exp_approx` (no float intrinsic, for
// determinism across targets); this kernel mirrors that same polynomial exp so
// the on-device result matches bit-close, rather than calling the WGSL `exp`
// builtin. The result is the updated state (transmittance, optical_depth,
// scattered, steps_taken). It uses only the portable core-WGSL subset, so it
// runs unmodified on Metal, Vulkan and DX12.
//
// Provenance: standard analytic volumetric segment integration; no Unreal
// Engine source or derived code.

const LOG2_E: f32 = 1.4426950408889634;
const EPS: f32 = 1e-6;

struct Params {
    count: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

// One segment query: the incoming accumulator state plus the segment
// coefficients. Matches `GpuQuery` in `src/integrate_segment.rs`.
struct Query {
    transmittance: f32,
    optical_depth: f32,
    scattered: f32,
    steps_taken: u32,
    sigma_t: f32,
    sigma_s: f32,
    phase: f32,
    step: f32,
    light_transmittance: f32,
    powder_factor: f32,
    pad0: u32,
    pad1: u32,
}

// One result: the updated accumulator state. Matches `GpuResult` in
// `src/integrate_segment.rs`.
struct Res {
    transmittance: f32,
    optical_depth: f32,
    scattered: f32,
    steps_taken: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> queries: array<Query>;
@group(0) @binding(2) var<storage, read_write> results: array<Res>;

fn saturate_s(x: f32) -> f32 {
    return clamp(x, 0.0, 1.0);
}

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

// One homogeneous-segment integration, mirroring `integrate_segment`.
fn integrate_segment(q: Query) -> Res {
    let sigma_t = max(q.sigma_t, 0.0);
    let sigma_s = max(q.sigma_s, 0.0);
    let phase = max(q.phase, 0.0);
    let step = max(q.step, 0.0);
    let light = saturate_s(q.light_transmittance);
    let powder = saturate_s(q.powder_factor);

    let seg_optical = sigma_t * step;
    let seg_trans = exp_approx(-seg_optical);
    // Analytic in-scatter integral; the limit as sigma_t -> 0 is `step`.
    let integral = select(step, (1.0 - seg_trans) / sigma_t, sigma_t > EPS);

    var out: Res;
    out.scattered = q.scattered + q.transmittance * light * sigma_s * phase * integral * powder;
    out.transmittance = saturate_s(q.transmittance * seg_trans);
    out.optical_depth = q.optical_depth + seg_optical;
    out.steps_taken = q.steps_taken + 1u;
    return out;
}

@compute @workgroup_size(64)
fn integrate_segment_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.count) {
        return;
    }
    results[idx] = integrate_segment(queries[idx]);
}
