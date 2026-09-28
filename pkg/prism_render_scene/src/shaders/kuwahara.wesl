// GPU twin of `prism_render_shading::kuwahara` (the classic four-quadrant
// Kuwahara edge-preserving painterly filter: split the neighbourhood into four
// overlapping quadrants, take each quadrant's RGB mean and luminance variance,
// and output the mean of the lowest-variance quadrant).
// Self-contained (no intra-crate imports, matching ssgi.wesl / bloom.wesl /
// exposure.wesl / color_grade.wesl) so the WESL compile test can validate it
// standalone. Every helper is `kuwahara_`-prefixed to avoid symbol collisions
// with the other standalone kernels, and mirrors the CPU golden arm-for-arm:
// same Rec. 709 luma weights, same variance metric (mean squared deviation,
// no sqrt), same forward quadrant scan with a strict `<` so ties keep the first
// minimum. CPU `if` guards (empty / single / disabled) are rewritten with
// `select` so both sides stay in lockstep. WGSL `min`/`max` are componentwise,
// matching the per-channel Rust.

// Rec. 709 luma weights (linear sRGB primaries), used by the variance metric.
const KUWAHARA_LUMA_WEIGHTS: vec3<f32> = vec3<f32>(0.2126, 0.7152, 0.0722);

// Rec. 709 relative luminance of a linear RGB sample.
fn kuwahara_luminance(c: vec3<f32>) -> f32 {
    return dot(c, KUWAHARA_LUMA_WEIGHTS);
}

// Arithmetic mean of the first `count` samples of a fixed four-slot quadrant.
// An empty quadrant (count == 0) has no colour to average, so it returns ZERO,
// mirroring the CPU `region_mean` empty-slice guard via `select`.
fn kuwahara_region_mean(samples: array<vec3<f32>, 4>, count: u32) -> vec3<f32> {
    var sum = vec3<f32>(0.0, 0.0, 0.0);
    for (var i = 0u; i < count; i = i + 1u) {
        sum = sum + samples[i];
    }
    let denom = max(f32(count), 1.0);
    let mean = sum / denom;
    return select(mean, vec3<f32>(0.0, 0.0, 0.0), count == 0u);
}

// Mean squared deviation of the samples' luminance about `mean`'s luminance.
// An empty or single-sample quadrant (count < 2) has no spread and returns 0,
// mirroring the CPU `region_luma_variance` guard via `select`.
fn kuwahara_region_luma_variance(samples: array<vec3<f32>, 4>, count: u32, mean: vec3<f32>) -> f32 {
    let mean_luma = kuwahara_luminance(mean);
    var acc = 0.0;
    for (var i = 0u; i < count; i = i + 1u) {
        let d = kuwahara_luminance(samples[i]) - mean_luma;
        acc = acc + d * d;
    }
    let denom = max(f32(count), 1.0);
    let variance = acc / denom;
    return select(variance, 0.0, count < 2u);
}

// Return the RGB mean of the quadrant with the smallest luminance variance,
// scanning the first `count` of a fixed four-slot region set. A strict `<`
// keeps the first minimum on ties, matching the CPU `select_min_variance`
// forward scan. An empty set (count == 0) returns ZERO.
fn kuwahara_select_min_variance(region_means: array<vec3<f32>, 4>, region_vars: array<f32, 4>, count: u32) -> vec3<f32> {
    var best_mean = vec3<f32>(0.0, 0.0, 0.0);
    var best_var = 0.0;
    var found = false;
    for (var i = 0u; i < count; i = i + 1u) {
        let v = region_vars[i];
        let take = (!found) || (v < best_var);
        best_mean = select(best_mean, region_means[i], take);
        best_var = select(best_var, v, take);
        found = true;
    }
    return select(vec3<f32>(0.0, 0.0, 0.0), best_mean, found);
}

// Painterly Kuwahara controls (mirrors the CPU `KuwaharaParams`). `enabled` is
// the master switch; the default of disabled with radius 2 is the identity.
struct KuwaharaParams {
    radius: u32,
    enabled: u32,
}

// Apply the Kuwahara selection to precomputed quadrant statistics. When the
// filter is disabled, or there are no quadrants, the `center` pixel passes
// through (identity); otherwise the lowest-variance quadrant mean is returned.
// Rewrites the CPU `if !enabled || regions.is_empty()` guard with `select`.
fn kuwahara_apply(region_means: array<vec3<f32>, 4>, region_vars: array<f32, 4>, count: u32, center: vec3<f32>, enabled: bool) -> vec3<f32> {
    let selected = kuwahara_select_min_variance(region_means, region_vars, count);
    let use_center = (!enabled) || (count == 0u);
    return select(selected, center, use_center);
}

// Minimal entry so the module type-checks as a standalone compute kernel under
// the WESL compile test; the real Kuwahara pass wires the source target and the
// params uniform later. References every helper so none drifts unused, using
// fixed literal inputs.
@compute @workgroup_size(64, 1, 1)
fn kuwahara_probe(@builtin(global_invocation_id) gid: vec3<u32>) {
    let seed = f32(gid.x) * 0.01;
    let center = vec3<f32>(0.15 + seed, 0.4, 0.85);

    let samples = array<vec3<f32>, 4>(
        vec3<f32>(0.0, 0.0, 0.0),
        vec3<f32>(1.0, 1.0, 1.0),
        vec3<f32>(0.2, 0.3, 0.4),
        vec3<f32>(0.5, 0.5, 0.5),
    );
    let count = 4u;

    let lum = kuwahara_luminance(center);
    let mean = kuwahara_region_mean(samples, count);
    let variance = kuwahara_region_luma_variance(samples, count, mean);

    let region_means = array<vec3<f32>, 4>(
        vec3<f32>(0.2, 0.3, 0.4),
        vec3<f32>(0.9, 0.1, 0.0),
        vec3<f32>(0.5, 0.5, 0.5),
        vec3<f32>(0.1, 0.7, 0.2),
    );
    let region_vars = array<f32, 4>(0.8, 0.05, 0.6, 0.3);
    let quads = 4u;

    let selected = kuwahara_select_min_variance(region_means, region_vars, quads);

    var params: KuwaharaParams;
    params.radius = 2u;
    params.enabled = 1u;
    let applied = kuwahara_apply(region_means, region_vars, quads, center, params.enabled != 0u);

    _ = lum + mean.x + variance + selected.x + applied.x + f32(params.radius);
}
