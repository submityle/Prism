// GPU twin of `prism_render_shading::cas` (AMD FidelityFX Contrast-Adaptive
// Sharpening, the "no-scaling" cross kernel over a 3x3 neighbourhood on the
// display-referred, tone-mapped [0, 1] buffer).
// Self-contained (no intra-crate imports, matching color_grade.wesl /
// bloom.wesl / exposure.wesl) so the WESL compile test can validate it
// standalone. Every function mirrors the CPU golden arm-for-arm: WGSL `min`,
// `max`, `sqrt` and `clamp` on vec3 are componentwise, matching the per-channel
// Rust, and lerps are written `a + (b - a) * t` rather than `mix` so the GPU
// twin stays bit-identical with the golden.

// Neighbourhood is row-major (a b c / d e f / g h i); index 4 is the centre e.

fn cas_min3(a: vec3<f32>, b: vec3<f32>, c: vec3<f32>) -> vec3<f32> {
    return min(min(a, b), c);
}

fn cas_max3(a: vec3<f32>, b: vec3<f32>, c: vec3<f32>) -> vec3<f32> {
    return max(max(a, b), c);
}

// FidelityFX doubled soft minimum: min of the five-tap cross (b, d, e, f, h)
// plus min of the four corners (a, c, g, i).
fn cas_soft_min(n: array<vec3<f32>, 9>) -> vec3<f32> {
    let cross = cas_min3(cas_min3(n[3], n[4], n[5]), n[1], n[7]);
    let corners = cas_min3(cas_min3(n[0], n[2], n[6]), n[8], n[8]);
    return cross + corners;
}

// FidelityFX doubled soft maximum (cross max plus corner max).
fn cas_soft_max(n: array<vec3<f32>, 9>) -> vec3<f32> {
    let cross = cas_max3(cas_max3(n[3], n[4], n[5]), n[1], n[7]);
    let corners = cas_max3(cas_max3(n[0], n[2], n[6]), n[8], n[8]);
    return cross + corners;
}

// Adaptive amplitude sqrt(saturate(min(soft_min, 2 - soft_max) / soft_max)),
// zero where soft_max is non-positive (pure black), mirroring the CPU guard.
fn cas_amplitude(soft_min: vec3<f32>, soft_max: vec3<f32>) -> vec3<f32> {
    let limited = min(soft_min, vec3<f32>(2.0) - soft_max);
    let amp = clamp(limited / soft_max, vec3<f32>(0.0), vec3<f32>(1.0));
    return select(sqrt(amp), vec3<f32>(0.0), soft_max <= vec3<f32>(0.0));
}

// Shape the amplitude into the negative cross weight w. sharpness in [0, 1]
// maps the peak from -1/8 (gentle) to -1/5 (strong).
fn cas_weight(amplitude: vec3<f32>, sharpness: f32) -> vec3<f32> {
    let s = clamp(sharpness, 0.0, 1.0);
    let denom = 8.0 + (5.0 - 8.0) * s;
    let peak = -1.0 / denom;
    return amplitude * peak;
}

// Blend cross taps against the centre: saturate(((b+d+f+h)*w + e) / (1 + 4w)).
fn cas_blend(b: vec3<f32>, d: vec3<f32>, e: vec3<f32>, f: vec3<f32>, h: vec3<f32>, w: vec3<f32>) -> vec3<f32> {
    let rcp = vec3<f32>(1.0) / (vec3<f32>(1.0) + 4.0 * w);
    return clamp(((b + d + f + h) * w + e) * rcp, vec3<f32>(0.0), vec3<f32>(1.0));
}

// Full CAS kernel over the 3x3 neighbourhood at the given sharpness.
fn cas_sharpen(n: array<vec3<f32>, 9>, sharpness: f32) -> vec3<f32> {
    let smn = cas_soft_min(n);
    let smx = cas_soft_max(n);
    let amp = cas_amplitude(smn, smx);
    let w = cas_weight(amp, sharpness);
    return cas_blend(n[1], n[3], n[4], n[5], n[7], w);
}

struct CasParams {
    strength: f32,
    sharpness: f32,
}

// Apply the kernel and blend it over the centre by params.strength.
fn cas_apply(n: array<vec3<f32>, 9>, params: CasParams) -> vec3<f32> {
    let e = n[4];
    let s = clamp(params.strength, 0.0, 1.0);
    let sharp = cas_sharpen(n, params.sharpness);
    return e + (sharp - e) * s;
}

// Minimal entry so the module type-checks as a standalone compute kernel under
// the WESL compile test; the real sharpen pass wires the gather taps and the
// params uniform later. References every function so none drifts unused.
@compute @workgroup_size(64, 1, 1)
fn cas_probe(@builtin(global_invocation_id) gid: vec3<u32>) {
    let seed = f32(gid.x) * 0.01;
    let e = vec3<f32>(0.55 + seed, 0.5, 0.45);
    var n: array<vec3<f32>, 9>;
    n[0] = vec3<f32>(0.40, 0.40, 0.40);
    n[1] = vec3<f32>(0.42, 0.42, 0.42);
    n[2] = vec3<f32>(0.40, 0.40, 0.40);
    n[3] = vec3<f32>(0.42, 0.42, 0.42);
    n[4] = e;
    n[5] = vec3<f32>(0.42, 0.42, 0.42);
    n[6] = vec3<f32>(0.40, 0.40, 0.40);
    n[7] = vec3<f32>(0.42, 0.42, 0.42);
    n[8] = vec3<f32>(0.40, 0.40, 0.40);

    let mn3 = cas_min3(n[0], n[1], n[2]);
    let mx3 = cas_max3(n[0], n[1], n[2]);
    let smn = cas_soft_min(n);
    let smx = cas_soft_max(n);
    let amp = cas_amplitude(smn, smx);
    let w = cas_weight(amp, 0.5);
    let blended = cas_blend(n[1], n[3], n[4], n[5], n[7], w);
    let sharp = cas_sharpen(n, 0.5);

    var params: CasParams;
    params.strength = 0.6;
    params.sharpness = 0.5;
    let applied = cas_apply(n, params);

    _ = mn3.x + mx3.x + smn.x + smx.x + amp.x + w.x + blended.x + sharp.x + applied.x;
}
