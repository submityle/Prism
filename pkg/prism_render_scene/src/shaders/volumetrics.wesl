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
// Self-contained (no intra-crate `import`s, matching ssgi.wesl / outline.wesl)
// so the WESL compile test can validate it standalone with no GPU present.

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
// fixed array so the twin matches the CPU golden's slice iteration.
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

// Minimal entry so the module type-checks as a standalone compute kernel under
// the WESL compile test; the render-world pass wires real froxel grid bindings
// later.
@compute @workgroup_size(4, 4, 4)
fn volumetrics_probe(@builtin(global_invocation_id) gid: vec3<u32>) {
    var medium: MediumSample;
    medium.scattering = vec3<f32>(0.5, 0.5, 0.5);
    medium.absorption = vec3<f32>(0.1, 0.1, 0.1);
    medium.emissive = vec3<f32>(0.0, 0.0, 0.0);

    let cos_theta = 1.0 - f32(gid.x) * 0.01;
    let scattered = in_scatter(medium, cos_theta, 0.3, vec3<f32>(4.0, 4.0, 4.0), 1.0);

    var froxels: array<Froxel, 64>;
    var f0: Froxel;
    f0.medium = medium;
    f0.in_scattering = scattered;
    f0.thickness = 1.0;
    froxels[0] = f0;

    let integ = integrate_froxel_column(&froxels, 1u);
    _ = integ.in_scattering.x + integ.transmittance.x;
}
