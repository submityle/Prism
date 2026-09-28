//! Split-sum environment-BRDF ("DFG") integration compute twin.
//!
//! GPU twin of `prism_render_shading::environment::brdf_lut`.  For every texel
//! this kernel integrates the Karis 2013 split-sum environment BRDF with GGX
//! importance sampling and stores the `(scale, bias)` pair so the resolve stage
//! can reconstruct specular reflectance as `F0 * scale + bias`.  The table's
//! x axis samples `n_dot_v` and its y axis samples perceptual `roughness`, both
//! across `[0, 1]` at texel centres.
//!
//! Every constant and expression mirrors the CPU golden bit-for-bit within
//! tolerance so machines with and without a GPU agree.  The file is
//! self-contained (no `import`s), matching the `gtao.wesl` precedent.

const BRDF_TAU: f32 = 6.28318530717958647692;

// Number of GGX importance samples per texel, uploaded per dispatch.
struct BrdfLutConfig {
    sample_count: u32,
    padding0: u32,
    padding1: u32,
    padding2: u32,
}

// `(scale, bias)` split-sum output; `.x = scale`, `.y = bias`.
@group(0) @binding(0) var dfg_out: texture_storage_2d<rg16float, write>;

var<immediate> config: BrdfLutConfig;

// Low-discrepancy Van der Corput radical inverse in base 2.
fn radical_inverse_vdc(bits_in: u32) -> f32 {
    var bits = bits_in;
    bits = (bits << 16u) | (bits >> 16u);
    bits = ((bits & 0x55555555u) << 1u) | ((bits & 0xAAAAAAAAu) >> 1u);
    bits = ((bits & 0x33333333u) << 2u) | ((bits & 0xCCCCCCCCu) >> 2u);
    bits = ((bits & 0x0F0F0F0Fu) << 4u) | ((bits & 0xF0F0F0F0u) >> 4u);
    bits = ((bits & 0x00FF00FFu) << 8u) | ((bits & 0xFF00FF00u) >> 8u);
    // 1 / 2^32.
    return f32(bits) * 2.3283064365386963e-10;
}

// The `i`-th of `n` Hammersley points on the unit square.
fn hammersley(i: u32, n: u32) -> vec2<f32> {
    return vec2<f32>(f32(i) / f32(n), radical_inverse_vdc(i));
}

// GGX half vector in the tangent frame whose normal is `+Z`.
fn importance_sample_ggx(xi: vec2<f32>, roughness: f32) -> vec3<f32> {
    let a = roughness * roughness;
    let phi = BRDF_TAU * xi.x;
    let cos_theta = sqrt((1.0 - xi.y) / (1.0 + (a * a - 1.0) * xi.y));
    let sin_theta = sqrt(max(1.0 - cos_theta * cos_theta, 0.0));
    return vec3<f32>(sin_theta * cos(phi), sin_theta * sin(phi), cos_theta);
}

// Smith masking-shadowing for IBL, Karis's k = roughness^2 / 2 remap.
fn geometry_smith_ibl(n_dot_v: f32, n_dot_l: f32, roughness: f32) -> f32 {
    let k = roughness * roughness * 0.5;
    let g1_v = n_dot_v / (n_dot_v * (1.0 - k) + k);
    let g1_l = n_dot_l / (n_dot_l * (1.0 - k) + k);
    return g1_v * g1_l;
}

// Integrates the split-sum environment BRDF for one `(n_dot_v, roughness)`.
fn integrate_brdf(n_dot_v_in: f32, roughness_in: f32, sample_count: u32) -> vec2<f32> {
    let n_dot_v = clamp(n_dot_v_in, 1.0e-4, 1.0);
    let roughness = clamp(roughness_in, 0.0, 1.0);
    let samples = max(sample_count, 1u);

    let view = vec3<f32>(sqrt(max(1.0 - n_dot_v * n_dot_v, 0.0)), 0.0, n_dot_v);

    var scale = 0.0;
    var bias = 0.0;
    for (var i = 0u; i < samples; i = i + 1u) {
        let xi = hammersley(i, samples);
        let h = importance_sample_ggx(xi, roughness);
        let v_dot_h_raw = dot(view, h);
        let light = 2.0 * v_dot_h_raw * h - view;
        let n_dot_l = light.z;
        if n_dot_l <= 0.0 {
            continue;
        }
        let n_dot_h = max(h.z, 0.0);
        let v_dot_h = max(v_dot_h_raw, 0.0);
        let g = geometry_smith_ibl(n_dot_v, n_dot_l, roughness);
        let g_vis = (g * v_dot_h) / max(n_dot_h * n_dot_v, 1.0e-6);
        let one_minus_voh = 1.0 - v_dot_h;
        let fc = one_minus_voh * one_minus_voh * one_minus_voh * one_minus_voh * one_minus_voh;
        scale = scale + (1.0 - fc) * g_vis;
        bias = bias + fc * g_vis;
    }

    let inv = 1.0 / f32(samples);
    return vec2<f32>(scale * inv, bias * inv);
}

@compute @workgroup_size(8, 8, 1)
fn integrate_brdf_lut(@builtin(global_invocation_id) id: vec3<u32>) {
    let dims = textureDimensions(dfg_out);
    if id.x >= dims.x || id.y >= dims.y {
        return;
    }
    // Texel centres so the table samples the open interval and skips the
    // degenerate n_dot_v = 0 column, matching `DfgLut::generate`.
    let n_dot_v = (f32(id.x) + 0.5) / f32(dims.x);
    let roughness = (f32(id.y) + 0.5) / f32(dims.y);
    let dfg = integrate_brdf(n_dot_v, roughness, config.sample_count);
    textureStore(dfg_out, vec2<i32>(i32(id.x), i32(id.y)), vec4<f32>(dfg.x, dfg.y, 0.0, 1.0));
}
