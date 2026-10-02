// Surfel-update (temporal-integration) producer kernel: the on-device twin of
// the CPU golden `integrate_radiance` in
// `prism_render_shading::gi::surface_cache::integration`.
//
// A Lumen-style surface cache refreshes every surfel's cached radiance each
// frame with a confidence-weighted exponential moving average, resetting the
// history whenever the surfel is disoccluded (its anchor jumped off the
// previous disc, or its normal flipped). This kernel runs one thread per
// surfel update and writes the advanced `(radiance, sample_count)` entry,
// mirroring the CPU golden exactly:
//
//   sample = sanitize(new_radiance)
//   if no history or disoccluded:  entry = (sample, 1)
//   else:
//     alpha  = 1 / min(prev_count + 1, max(max_samples, 1))
//     entry  = (sanitize(mix(sanitize(prev_radiance), sample, alpha)),
//               min(prev_count + 1, max(max_samples, 1)))
//
// The blend is written as the explicit `mix(a, b, t) = a*(1-t) + b*t`, which is
// exactly the `glam` `Vec3::lerp` the golden uses, so the CPU mirror reproduces
// it bit-for-bit. Only `+ - * /`, `sqrt`, `min`/`max` and comparisons are used
// — the portable core-WGSL subset, no transcendental and no optional device
// feature — so it runs unmodified on Metal, Vulkan and DX12.
//
// Provenance: standard confidence-weighted temporal accumulation with
// disocclusion reset; no Unreal Engine source or derived code.

// Dispatch parameters. 16-byte `repr(C)` matching `GpuSurfelUpdateParams` in
// `src/gi/surface_cache/gpu/abi.rs`.
struct UpdateParams {
    count: u32,
    max_samples: u32,
    position_tolerance: f32,
    normal_tolerance: f32,
}

// One surfel-update input: previous entry + previous/current surfel geometry +
// the new sample. All fields are scalar (not `vec3`) so the std430 stride stays
// a 4-byte-aligned 88 bytes. Matches `GpuSurfelUpdateInput`.
struct UpdateInput {
    has_history: u32,
    prev_count: u32,
    prev_radiance_x: f32,
    prev_radiance_y: f32,
    prev_radiance_z: f32,
    prev_pos_x: f32,
    prev_pos_y: f32,
    prev_pos_z: f32,
    prev_normal_x: f32,
    prev_normal_y: f32,
    prev_normal_z: f32,
    prev_radius: f32,
    curr_pos_x: f32,
    curr_pos_y: f32,
    curr_pos_z: f32,
    curr_normal_x: f32,
    curr_normal_y: f32,
    curr_normal_z: f32,
    curr_radius: f32,
    new_radiance_x: f32,
    new_radiance_y: f32,
    new_radiance_z: f32,
}

// Advanced cache entry. 16-byte `repr(C)` matching `GpuSurfelUpdateResult`.
struct UpdateResult {
    radiance_x: f32,
    radiance_y: f32,
    radiance_z: f32,
    sample_count: u32,
}

@group(0) @binding(0) var<uniform> params: UpdateParams;
@group(0) @binding(1) var<storage, read> inputs: array<UpdateInput>;
@group(0) @binding(2) var<storage, read_write> results: array<UpdateResult>;

// Largest finite `f32`; used with the self-equality `NaN` test to reproduce the
// golden `f32::is_finite` guard without an `isFinite` builtin.
const F32_FINITE_MAX: f32 = 3.40282347e38;

// `true` when `x` is finite (neither `NaN` nor an infinity): `NaN` fails the
// self-equality test and the infinities exceed the finite max.
fn is_finite(x: f32) -> bool {
    return x == x && abs(x) <= F32_FINITE_MAX;
}

// Replace a non-finite scalar with zero, mirroring the golden `finite_or_zero`.
fn finite_or_zero(x: f32) -> f32 {
    if (is_finite(x)) {
        return x;
    }
    return 0.0;
}

// Sanitise a linear-RGB triple: non-finite components become zero and the
// result is clamped non-negative, mirroring the golden `sanitize_rgb`.
fn sanitize_rgb(c: vec3<f32>) -> vec3<f32> {
    let s = vec3<f32>(finite_or_zero(c.x), finite_or_zero(c.y), finite_or_zero(c.z));
    return max(s, vec3<f32>(0.0, 0.0, 0.0));
}

// Whether the current surfel geometry drifted far enough from the previous one
// to invalidate its history, mirroring the golden `is_disoccluded`.
fn is_disoccluded(inp: UpdateInput, par: UpdateParams) -> bool {
    let dx = inp.curr_pos_x - inp.prev_pos_x;
    let dy = inp.curr_pos_y - inp.prev_pos_y;
    let dz = inp.curr_pos_z - inp.prev_pos_z;
    let offset = sqrt(dx * dx + dy * dy + dz * dz);
    if (!is_finite(offset)) {
        return true;
    }
    let radius = max(inp.prev_radius, inp.curr_radius);
    let max_offset = radius * max(par.position_tolerance, 0.0);
    if (offset > max_offset) {
        return true;
    }
    let cos_n = inp.prev_normal_x * inp.curr_normal_x
        + inp.prev_normal_y * inp.curr_normal_y
        + inp.prev_normal_z * inp.curr_normal_z;
    if (!is_finite(cos_n) || cos_n < par.normal_tolerance) {
        return true;
    }
    return false;
}

@compute @workgroup_size(64)
fn surfel_update_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.count) {
        return;
    }
    let inp = inputs[idx];
    let sample = sanitize_rgb(vec3<f32>(inp.new_radiance_x, inp.new_radiance_y, inp.new_radiance_z));

    var result: UpdateResult;

    if (inp.has_history == 0u || is_disoccluded(inp, params)) {
        result.radiance_x = sample.x;
        result.radiance_y = sample.y;
        result.radiance_z = sample.z;
        result.sample_count = 1u;
        results[idx] = result;
        return;
    }

    let cap = max(params.max_samples, 1u);
    let n = min(inp.prev_count + 1u, cap);
    let alpha = 1.0 / f32(n);
    let prev_rad = sanitize_rgb(vec3<f32>(inp.prev_radiance_x, inp.prev_radiance_y, inp.prev_radiance_z));
    let blended = sanitize_rgb(mix(prev_rad, sample, vec3<f32>(alpha, alpha, alpha)));

    result.radiance_x = blended.x;
    result.radiance_y = blended.y;
    result.radiance_z = blended.z;
    result.sample_count = n;
    results[idx] = result;
}
