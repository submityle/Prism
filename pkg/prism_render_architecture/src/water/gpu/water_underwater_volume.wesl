// Water underwater-volume froxel compute kernel
// (entry point `water_underwater_volume`).
//
// Below the surface the water is a participating medium: light is absorbed and
// out-scattered along every eye path, so distant geometry fades, warm colours
// vanish first, and shafts of light (god rays) glow in the haze. This pass runs
// the shared froxel volume's per-cell underwater math — `Beer-Lambert` depth
// colour shift, a bounded multiple-scattering boost, and god-ray in-scatter —
// mirroring the CPU golden `underwater::depth_color_shift`,
// `underwater::multiple_scatter_boost` and `underwater::godray_inscatter`
// byte-for-byte.
//
// ABI (matches `WaterKernel::UnderwaterVolume.descriptor()`: 0 storage buffers,
// 1 uniform, 1 storage texture, 1 sampled texture, 4x4x4 brick, Grid3d domain):
//   @binding(0) params     : UnderwaterParams uniform
//   @binding(1) froxel_in  : texture_3d<f32>, per-cell lit radiance and the
//                            eye-path length (color_r, color_g, color_b,
//                            path_length)
//   @binding(2) froxel_out : texture_storage_3d<rgba32float, write>, the
//                            attenuated, scattered in-water radiance
//
// Extinction uses the crate's monotone `exp_approx` (the limit identity
// exp(x) = (1 + x/4096)^4096 by twelve squarings), bit-for-bit identical to
// `water::exp_approx`. Only `+ - * /`, comparisons, `clamp` and `max` appear —
// no float intrinsic the workspace determinism policy forbids. `EPS` matches
// the crate's `water::EPS` (1e-6).

const EPS: f32 = 1e-6;

struct UnderwaterParams {
    ext_r: f32,
    ext_g: f32,
    ext_b: f32,
    scatter_albedo: f32,
    surface_light: f32,
    godray_extinction: f32,
    dim_x: u32,
    dim_y: u32,
    dim_z: u32,
}

@group(0) @binding(0) var<uniform> params: UnderwaterParams;
@group(0) @binding(1) var froxel_in: texture_3d<f32>;
@group(0) @binding(2) var froxel_out: texture_storage_3d<rgba32float, write>;

// Hand-rolled exp via exp(x) = (1 + x/4096)^4096, twelve squarings. Bit-for-bit
// the same algorithm as `water::exp_approx`.
fn exp_approx(x: f32) -> f32 {
    var base: f32 = 1.0 + x / 4096.0;
    if (base < 0.0) {
        base = 0.0;
    }
    for (var i: i32 = 0; i < 12; i = i + 1) {
        base = base * base;
    }
    return base;
}

// `Beer-Lambert` transmittance exp(-extinction * distance) clamped to 0..=1.
fn beer_lambert_transmittance(extinction: f32, distance: f32) -> f32 {
    let e = max(extinction, 0.0);
    let d = max(distance, 0.0);
    return clamp(exp_approx(-(e * d)), 0.0, 1.0);
}

// Bounded multiple-scattering amplification single / (1 - albedo).
fn multiple_scatter_boost(single: f32, albedo: f32) -> f32 {
    let a = clamp(albedo, 0.0, 0.999);
    return max(single, 0.0) / (1.0 - a);
}

// God-ray in-scatter surface_light * albedo * (1 - transmittance).
fn godray_inscatter(
    surface_light: f32,
    scatter_albedo: f32,
    extinction: f32,
    path_length: f32,
) -> f32 {
    let light = max(surface_light, 0.0);
    let albedo = clamp(scatter_albedo, 0.0, 1.0);
    let transmittance = beer_lambert_transmittance(extinction, path_length);
    return light * albedo * (1.0 - transmittance);
}

@compute @workgroup_size(4, 4, 4)
fn water_underwater_volume(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= params.dim_x || gid.y >= params.dim_y || gid.z >= params.dim_z) {
        return;
    }
    let coord = vec3<i32>(i32(gid.x), i32(gid.y), i32(gid.z));
    let texel = textureLoad(froxel_in, coord, 0);
    let color_r = max(texel.x, 0.0);
    let color_g = max(texel.y, 0.0);
    let color_b = max(texel.z, 0.0);
    let path_length = texel.w;

    // Per-channel `Beer-Lambert` depth colour shift (red collapses first).
    let shifted_r = color_r * beer_lambert_transmittance(params.ext_r, path_length);
    let shifted_g = color_g * beer_lambert_transmittance(params.ext_g, path_length);
    let shifted_b = color_b * beer_lambert_transmittance(params.ext_b, path_length);

    // Bounded multiple-scattering glow lift.
    let boosted_r = multiple_scatter_boost(shifted_r, params.scatter_albedo);
    let boosted_g = multiple_scatter_boost(shifted_g, params.scatter_albedo);
    let boosted_b = multiple_scatter_boost(shifted_b, params.scatter_albedo);

    // Achromatic god-ray in-scatter added along the shaft.
    let glow = godray_inscatter(
        params.surface_light,
        params.scatter_albedo,
        params.godray_extinction,
        path_length,
    );

    textureStore(
        froxel_out,
        coord,
        vec4<f32>(boosted_r + glow, boosted_g + glow, boosted_b + glow, 1.0),
    );
}
