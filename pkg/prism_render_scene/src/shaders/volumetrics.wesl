// Froxel volumetric fog single scattering — GPU twin of
// `prism_render_shading::volumetrics`.
//
// Mirrors the CPU golden arm-for-arm: a Henyey-Greenstein phase function drives
// per-channel in-scattering `sigma_s * phase * L * vis`, Beer-Lambert
// transmittance `exp(-sigma_t * d)` attenuates each slice, and an
// energy-conserving analytic slice integral `(S - S*T) / sigma_t` (with the
// `sigma_t -> 0` limit `S * d`) folds a constant per-slice source under its own
// extinction. A front-to-back column march accumulates each slice's
// in-scattering under the transmittance built up by nearer slices so nearer
// media correctly occlude farther media. This is the Frostbite / UE
// "Volumetric Fog" scattering integration.
//
// Two chained compute entries realise the AAA froxel-fog pipeline over a
// view-frustum-fitted 3D grid (the CPU golden's `Froxel` column, unrolled onto
// the GPU):
//
//   * `volumetrics_scatter` (@workgroup_size(4,4,4), one froxel per invocation)
//     reconstructs each froxel's view ray from the camera frustum half-tangents,
//     builds the homogeneous participating medium, in-scatters the single
//     directional light through the phase function and writes the per-froxel
//     source radiance (`in_scatter + emissive`) + slice thickness and the
//     per-froxel extinction into two storage volumes;
//   * `volumetrics_integrate` (@workgroup_size(8,8,1), one froxel *column* per
//     invocation) marches each column front-to-back exactly like the golden
//     `integrate_froxel_column`, writing the running camera-to-slice
//     accumulated in-scattering and transmittance into two more storage volumes
//     so a later composite can sample the fog at any froxel depth.
//
// The scatter and integrate entries reuse `@group(0)` binding slots with
// different types (each is a distinct pipeline with its own layout), mirroring
// the two-entry `exposure.wesl` precedent. Self-contained (no intra-crate
// `import`s, matching ssgi.wesl / outline.wesl) so the WESL compile test can
// validate it standalone with no GPU present.

// Matches `MediumSample` in volumetrics.rs.
struct MediumSample {
    scattering: vec3<f32>,
    absorption: vec3<f32>,
    emissive: vec3<f32>,
}

// Matches `Froxel` in volumetrics.rs.
struct Froxel {
    medium: MediumSample,
    in_scattering: vec3<f32>,
    thickness: f32,
}

// Matches `VolumetricIntegration` in volumetrics.rs.
struct VolumetricIntegration {
    in_scattering: vec3<f32>,
    transmittance: vec3<f32>,
}

const VOLUMETRICS_PI: f32 = 3.14159265358979323846;
const VOLUMETRICS_INV_4PI: f32 = 1.0 / (4.0 * VOLUMETRICS_PI);
// Numerical floor below which extinction is treated as zero (the `S * d` limit).
const EXTINCTION_EPSILON: f32 = 1.0e-5;

// Per-channel extinction `sigma_t = sigma_s + sigma_a`. Mirrors
// `MediumSample::extinction`.
fn medium_extinction(medium: MediumSample) -> vec3<f32> {
    return medium.scattering + medium.absorption;
}

// Mirrors `henyey_greenstein`: g clamped to +/-0.99, normalised so the sphere
// integral is 1; g == 0 collapses to 1/(4*pi).
fn henyey_greenstein(cos_theta: f32, g_in: f32) -> f32 {
    let g = clamp(g_in, -0.99, 0.99);
    let g2 = g * g;
    var denom = 1.0 + g2 - 2.0 * g * cos_theta;
    denom = max(denom, 1.0e-8);
    return VOLUMETRICS_INV_4PI * (1.0 - g2) / (denom * sqrt(denom));
}

// Mirrors `transmittance`: per-channel Beer-Lambert `exp(-sigma_t * d)` with
// non-negative distance and extinction.
fn volumetric_transmittance(extinction: vec3<f32>, distance: f32) -> vec3<f32> {
    let d = max(distance, 0.0);
    let ext = max(extinction, vec3<f32>(0.0, 0.0, 0.0));
    return exp(-ext * d);
}

// Mirrors `in_scatter`: `sigma_s * phase * L * vis` per channel.
fn in_scatter(
    medium: MediumSample,
    cos_theta: f32,
    g: f32,
    light_radiance: vec3<f32>,
    visibility: f32,
) -> vec3<f32> {
    let phase = henyey_greenstein(cos_theta, g);
    let vis = clamp(visibility, 0.0, 1.0);
    return medium.scattering * phase * light_radiance * vis;
}

// Mirrors `froxel_source`: in-scattered light plus the medium emissive density.
fn froxel_source(froxel: Froxel) -> vec3<f32> {
    return froxel.in_scattering + froxel.medium.emissive;
}

// Mirrors `integrate_slice`: energy-conserving analytic integral of a constant
// source across a slice under its own extinction. Per channel to honour the
// `sigma_t -> 0` limit `S * d` exactly as the CPU golden's while loop does.
fn integrate_slice(source: vec3<f32>, extinction: vec3<f32>, thickness: f32) -> vec3<f32> {
    let d = max(thickness, 0.0);
    let t = volumetric_transmittance(extinction, d);
    var out = vec3<f32>(0.0, 0.0, 0.0);
    let ext = max(extinction, vec3<f32>(0.0, 0.0, 0.0));
    var src = array<f32, 3>(source.x, source.y, source.z);
    var sig = array<f32, 3>(ext.x, ext.y, ext.z);
    var tr = array<f32, 3>(t.x, t.y, t.z);
    var res = array<f32, 3>(0.0, 0.0, 0.0);
    for (var c = 0; c < 3; c = c + 1) {
        let sigma = sig[c];
        if (sigma > EXTINCTION_EPSILON) {
            res[c] = (src[c] - src[c] * tr[c]) / sigma;
        } else {
            res[c] = src[c] * d;
        }
    }
    out = vec3<f32>(res[0], res[1], res[2]);
    return out;
}

// Mirrors `integrate_froxel_column`: front-to-back accumulation of each slice's
// in-scattering under the running transmittance, then folding the slice's own
// transmittance into the running product. `count` bounds the live prefix of the
// fixed array so the twin matches the CPU golden's slice iteration. Retained as
// the reference reduction the streaming `volumetrics_integrate` entry below
// unrolls over the froxel volume; unreferenced module functions are valid.
fn integrate_froxel_column(froxels: ptr<function, array<Froxel, 64>>, count: u32) -> VolumetricIntegration {
    var accum = vec3<f32>(0.0, 0.0, 0.0);
    var trans = vec3<f32>(1.0, 1.0, 1.0);
    let n = min(count, 64u);
    for (var i = 0u; i < n; i = i + 1u) {
        let froxel = (*froxels)[i];
        let ext = medium_extinction(froxel.medium);
        let source = froxel_source(froxel);
        let slice = integrate_slice(source, ext, froxel.thickness);
        accum = accum + trans * slice;
        let slice_t = volumetric_transmittance(ext, froxel.thickness);
        trans = trans * slice_t;
    }
    var result: VolumetricIntegration;
    result.in_scattering = accum;
    result.transmittance = trans;
    return result;
}

// View-space depth of a froxel slice boundary. The grid packs more slices near
// the camera via the exponential distribution `near * (far/near)^(frac^power)`
// (`power == 1` is a pure exponential fit; `> 1` biases detail toward the eye).
fn volumetrics_slice_depth(frac: f32, near: f32, far: f32, power: f32) -> f32 {
    let t = pow(clamp(frac, 0.0, 1.0), power);
    return near * pow(far / near, t);
}

// View-space thickness of froxel depth slice `z` of `grid_z`, the near-to-far
// span the golden `Froxel::thickness` carries.
fn volumetrics_slice_thickness(z: u32, grid_z: u32, near: f32, far: f32, power: f32) -> f32 {
    let n = f32(max(grid_z, 1u));
    let d0 = volumetrics_slice_depth(f32(z) / n, near, far, power);
    let d1 = volumetrics_slice_depth(f32(z + 1u) / n, near, far, power);
    return max(d1 - d0, 0.0);
}

// ---------------------------------------------------------------------------
// Scatter pass: build every froxel's medium + in-scattered source radiance.
// ---------------------------------------------------------------------------

// Byte-for-byte twin of `GpuVolumetricsScatterParams` (24 back-to-back 4-byte
// scalars, no `vec3` so no 16-byte-align padding).
struct VolumetricsScatterParams {
    grid_x: u32,
    grid_y: u32,
    grid_z: u32,
    near_plane: f32,
    far_plane: f32,
    scattering_r: f32,
    scattering_g: f32,
    scattering_b: f32,
    absorption_r: f32,
    absorption_g: f32,
    absorption_b: f32,
    emissive_r: f32,
    emissive_g: f32,
    emissive_b: f32,
    light_dir_x: f32,
    light_dir_y: f32,
    light_dir_z: f32,
    light_radiance_r: f32,
    light_radiance_g: f32,
    light_radiance_b: f32,
    phase_g: f32,
    tan_half_fov_x: f32,
    tan_half_fov_y: f32,
    depth_power: f32,
}

var<immediate> scatter_params: VolumetricsScatterParams;

// group 0 (scatter): the two write-only froxel volumes.
//   0 = froxel_scattering  (rgb = source radiance = in_scatter + emissive, a = thickness)
//   1 = froxel_extinction  (rgb = sigma_t)
@group(0) @binding(0) var froxel_scattering_out: texture_storage_3d<rgba16float, write>;
@group(0) @binding(1) var froxel_extinction_out: texture_storage_3d<rgba16float, write>;

@compute @workgroup_size(4, 4, 4)
fn volumetrics_scatter(@builtin(global_invocation_id) gid: vec3<u32>) {
    let grid = vec3<u32>(scatter_params.grid_x, scatter_params.grid_y, scatter_params.grid_z);
    if (gid.x >= grid.x || gid.y >= grid.y || gid.z >= grid.z) {
        return;
    }

    // Reconstruct this froxel's view ray from its screen-tile centre and the
    // camera frustum half-tangents. View space is -Z forward; the NDC y is
    // flipped so +y is up.
    let uv = (vec2<f32>(f32(gid.x), f32(gid.y)) + vec2<f32>(0.5, 0.5))
        / vec2<f32>(f32(grid.x), f32(grid.y));
    let ndc = uv * 2.0 - vec2<f32>(1.0, 1.0);
    let view_ray = normalize(vec3<f32>(
        ndc.x * scatter_params.tan_half_fov_x,
        -ndc.y * scatter_params.tan_half_fov_y,
        -1.0,
    ));

    // Direction the light travels; the HG scattering-angle cosine peaks for
    // forward scattering when the view ray looks down-sun.
    let light_dir = normalize(vec3<f32>(
        scatter_params.light_dir_x,
        scatter_params.light_dir_y,
        scatter_params.light_dir_z,
    ));
    let cos_theta = dot(view_ray, light_dir);

    var medium: MediumSample;
    medium.scattering = vec3<f32>(
        scatter_params.scattering_r,
        scatter_params.scattering_g,
        scatter_params.scattering_b,
    );
    medium.absorption = vec3<f32>(
        scatter_params.absorption_r,
        scatter_params.absorption_g,
        scatter_params.absorption_b,
    );
    medium.emissive = vec3<f32>(
        scatter_params.emissive_r,
        scatter_params.emissive_g,
        scatter_params.emissive_b,
    );

    let light_radiance = vec3<f32>(
        scatter_params.light_radiance_r,
        scatter_params.light_radiance_g,
        scatter_params.light_radiance_b,
    );

    // No shadow map in the standalone fog path, so the medium-shadow term is 1.
    let in_scattered = in_scatter(medium, cos_theta, scatter_params.phase_g, light_radiance, 1.0);

    var froxel: Froxel;
    froxel.medium = medium;
    froxel.in_scattering = in_scattered;
    froxel.thickness = volumetrics_slice_thickness(
        gid.z,
        grid.z,
        scatter_params.near_plane,
        scatter_params.far_plane,
        scatter_params.depth_power,
    );

    let source = froxel_source(froxel);
    let extinction = medium_extinction(medium);

    let coord = vec3<i32>(i32(gid.x), i32(gid.y), i32(gid.z));
    textureStore(froxel_scattering_out, coord, vec4<f32>(source, froxel.thickness));
    textureStore(froxel_extinction_out, coord, vec4<f32>(extinction, 0.0));
}

// ---------------------------------------------------------------------------
// Integrate pass: march each froxel column front-to-back.
// ---------------------------------------------------------------------------

// Byte-for-byte twin of `GpuVolumetricsIntegrateParams` (3 back-to-back u32).
struct VolumetricsIntegrateParams {
    grid_x: u32,
    grid_y: u32,
    grid_z: u32,
}

var<immediate> integrate_params: VolumetricsIntegrateParams;

// group 0 (integrate): the scatter volumes read back + the integrated volumes.
//   0 = froxel_scattering (sampled, rgb = source, a = thickness)
//   1 = froxel_extinction (sampled, rgb = sigma_t)
//   2 = integrated_scattering    (rgb = camera-to-slice accumulated in-scatter)
//   3 = integrated_transmittance (rgb = camera-to-slice transmittance)
@group(0) @binding(0) var froxel_scattering_in: texture_3d<f32>;
@group(0) @binding(1) var froxel_extinction_in: texture_3d<f32>;
@group(0) @binding(2) var integrated_scattering_out: texture_storage_3d<rgba16float, write>;
@group(0) @binding(3) var integrated_transmittance_out: texture_storage_3d<rgba16float, write>;

@compute @workgroup_size(8, 8, 1)
fn volumetrics_integrate(@builtin(global_invocation_id) gid: vec3<u32>) {
    let grid = vec3<u32>(integrate_params.grid_x, integrate_params.grid_y, integrate_params.grid_z);
    if (gid.x >= grid.x || gid.y >= grid.y) {
        return;
    }

    // Front-to-back column march, arm-for-arm with the CPU golden
    // `integrate_froxel_column`: add this slice's energy-conserving
    // in-scattering under the running transmittance, then fold the slice's own
    // transmittance into the running product. The running integral is written
    // at every froxel so a later composite reads the fog accumulated from the
    // camera up to that depth.
    var accum = vec3<f32>(0.0, 0.0, 0.0);
    var trans = vec3<f32>(1.0, 1.0, 1.0);
    for (var z = 0u; z < grid.z; z = z + 1u) {
        let coord = vec3<i32>(i32(gid.x), i32(gid.y), i32(z));
        let scat = textureLoad(froxel_scattering_in, coord, 0);
        let ext = textureLoad(froxel_extinction_in, coord, 0).xyz;
        let source = scat.xyz;
        let thickness = scat.w;

        let slice = integrate_slice(source, ext, thickness);
        accum = accum + trans * slice;
        let slice_t = volumetric_transmittance(ext, thickness);
        trans = trans * slice_t;

        textureStore(integrated_scattering_out, coord, vec4<f32>(accum, 1.0));
        textureStore(integrated_transmittance_out, coord, vec4<f32>(trans, 1.0));
    }
}
