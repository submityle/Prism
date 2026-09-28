//! GGX prefiltered environment radiance compute twin.
//!
//! GPU twin of `prism_render_shading::environment::prefilter`.  One dispatch
//! prefilters a source environment cube into a single output mip baked at a
//! fixed `roughness`: each invocation owns one texel of one output face
//! (`global_invocation_id.z` selects the face layer), reconstructs the
//! world-space direction that texel looks along, convolves the source cube
//! with the GGX lobe about that direction, and writes the filtered radiance.
//!
//! The Karis split-sum approximation assumes the normal, view, and reflection
//! all equal the sampled direction, so the lobe is centred on the reflected
//! view.  Sampling the source through `textureSampleLevel` lets the hardware
//! bilinear/trilinear filter blend texels, matching the CPU golden's per-face
//! bilinear `CubemapFaces::sample`.  Constants mirror the golden bit-for-bit.
//!
//! The file is self-contained (no `import`s), matching the `brdf_lut.wesl`
//! precedent.

const PREFILTER_TAU: f32 = 6.28318530717958647692;

// Per-dispatch parameters for one output mip.
struct PrefilterConfig {
    // Perceptual roughness baked into this mip, in `[0, 1]`.
    roughness: f32,
    // Number of GGX importance samples per texel.
    sample_count: u32,
    // Edge length of the output face in texels.
    mip_size: u32,
    padding0: u32,
}

// Source environment probe (all six faces, mip 0 = sharpest).
@group(0) @binding(0) var source_cube: texture_cube<f32>;
@group(0) @binding(1) var source_sampler: sampler;
// Output prefiltered face array: layer = cube face in wgpu order.
@group(0) @binding(2) var prefiltered_out: texture_storage_2d_array<rgba16float, write>;

var<immediate> config: PrefilterConfig;

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
    let phi = PREFILTER_TAU * xi.x;
    let cos_theta = sqrt((1.0 - xi.y) / (1.0 + (a * a - 1.0) * xi.y));
    let sin_theta = sqrt(max(1.0 - cos_theta * cos_theta, 0.0));
    return vec3<f32>(sin_theta * cos(phi), sin_theta * sin(phi), cos_theta);
}

// Maps a face layer plus in-face `[-1, 1]` coordinates to the world-space
// direction that texel looks along, matching `face_direction` in the golden.
fn face_direction(face: u32, u: f32, v: f32) -> vec3<f32> {
    switch face {
        case 0u: { return vec3<f32>(1.0, -v, -u); }   // +X
        case 1u: { return vec3<f32>(-1.0, -v, u); }   // -X
        case 2u: { return vec3<f32>(u, 1.0, v); }     // +Y
        case 3u: { return vec3<f32>(u, -1.0, -v); }   // -Y
        case 4u: { return vec3<f32>(u, -v, 1.0); }    // +Z
        default: { return vec3<f32>(-u, -v, -1.0); }  // -Z
    }
}

// Convolves the source cube with the GGX lobe about `normal`.
fn prefilter_radiance(normal: vec3<f32>, roughness: f32, sample_count: u32) -> vec3<f32> {
    let n = normalize(normal);
    let samples = max(sample_count, 1u);

    // Mirror path: the lobe is a delta at the normal, so sample sharply.
    if roughness <= 0.0 {
        return textureSampleLevel(source_cube, source_sampler, n, 0.0).rgb;
    }

    // Orthonormal tangent frame around the normal.
    var up = vec3<f32>(0.0, 0.0, 1.0);
    if abs(n.z) >= 0.999 {
        up = vec3<f32>(1.0, 0.0, 0.0);
    }
    let tangent = normalize(cross(up, n));
    let bitangent = cross(n, tangent);

    // V == R == N in the isotropic split-sum approximation.
    let view = n;

    var color = vec3<f32>(0.0);
    var total_weight = 0.0;
    for (var i = 0u; i < samples; i = i + 1u) {
        let xi = hammersley(i, samples);
        let ht = importance_sample_ggx(xi, roughness);
        let h = tangent * ht.x + bitangent * ht.y + n * ht.z;
        let v_dot_h = dot(view, h);
        let light = 2.0 * v_dot_h * h - view;
        let n_dot_l = dot(n, light);
        if n_dot_l > 0.0 {
            let radiance = textureSampleLevel(source_cube, source_sampler, light, 0.0).rgb;
            color = color + radiance * n_dot_l;
            total_weight = total_weight + n_dot_l;
        }
    }

    if total_weight > 1.0e-6 {
        return color / total_weight;
    }
    return textureSampleLevel(source_cube, source_sampler, n, 0.0).rgb;
}

@compute @workgroup_size(8, 8, 1)
fn prefilter_env_map(@builtin(global_invocation_id) id: vec3<u32>) {
    let size = config.mip_size;
    if id.x >= size || id.y >= size {
        return;
    }
    let face = id.z;
    // Texel centres mapped to the in-face [-1, 1] extent, matching
    // `PrefilteredEnvMap::generate`.
    let inv = 1.0 / f32(size);
    let u = (f32(id.x) + 0.5) * inv * 2.0 - 1.0;
    let v = (f32(id.y) + 0.5) * inv * 2.0 - 1.0;
    let dir = face_direction(face, u, v);
    let filtered = prefilter_radiance(dir, config.roughness, config.sample_count);
    textureStore(
        prefiltered_out,
        vec2<i32>(i32(id.x), i32(id.y)),
        i32(face),
        vec4<f32>(filtered, 1.0),
    );
}
