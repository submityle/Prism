//! `GPU` compute twins of the water render-FX golden references.
//!
//! Every kernel below mirrors, one-for-one, a pure/deterministic `CPU` golden
//! in `prism_render_architecture::water`, moved onto the `GPU` as a compute
//! dispatch. There is no `GPU` in the sandbox, so this module is validated by
//! naga compilation only; it carries no host scheduling, and each entry-point
//! name is byte-identical to the contract emitted by
//! `prism_render_architecture::water::kernels::WaterKernel::wesl_entry_point`
//! so the pipeline cache keys resolve against this source:
//! `water_caustics_project`, `water_dispersion_refract`,
//! `water_underwater_volume`, `water_wetness_step`, `water_coupling_readback`.
//!
//! Mirrored golden modules:
//!
//! - `caustics.rs` — light-space Jacobian projection of refracted-ray area
//!   distortion into receiver brightness, plus the `RT`/photon splat density
//!   estimate (`photon_count * power / (PI * r^2)`).
//! - `dispersion.rs` — the `Cauchy` two-term `IOR` law sampled at three `RGB`
//!   wavelengths, split into ordered screen-space refraction offsets so the
//!   scene colour is sampled three times into a coloured fringe.
//! - `underwater.rs` — per-`froxel` `Beer-Lambert` extinction and `RGB` colour
//!   shift, the `Henyey-Greenstein` phase, a bounded multiple-scatter boost,
//!   and god-ray in-scatter.
//! - `wetness.rs` — capillary rise, albedo darkening, and exponential
//!   absorb/dry envelopes over a caller-owned moisture state.
//! - `coupling.rs` — two-way buoyancy/drag/added-mass queries packed into a
//!   bounded read-back batch (never a full-field `GPU` read-back).
//!
//! The math is purely classical and deterministic (no AI/ML): the same inputs
//! always produce the same outputs, extinction/drying reuse the golden's
//! monotone `water_exp_approx` rather than a hardware `exp`, and every divisor
//! is floored by `WATER_EPS` so no path divides by zero.

/// Small positive epsilon guarding divisions and magnitude tests. Mirrors the
/// crate-level `EPS` used by every golden module here.
const WATER_EPS: f32 = 1.0e-6;

/// Circle constant used by the photon-splat area and the `Henyey-Greenstein`
/// normalisation. Mirrors the crate-level `PI`.
const WATER_PI: f32 = 3.14159265358979;

// ---------------------------------------------------------------------------
// Shared pure helpers (no bindings): direct twins of the golden functions.
// ---------------------------------------------------------------------------

/// Monotone non-negative `exp` twin of `super::exp_approx`: `(1 + x/4096)^4096`
/// via twelve squarings, with the base floored at zero so a large negative
/// argument saturates to `0` instead of going negative.
fn water_exp_approx(x: f32) -> f32 {
    var base = 1.0 + x / 4096.0;
    if base < 0.0 {
        base = 0.0;
    }
    for (var i = 0u; i < 12u; i = i + 1u) {
        base = base * base;
    }
    return base;
}

/// Caustic gain from the refracted-ray Jacobian determinant: `1 / |jacobian|`
/// clamped to `max_gain`, saturating (not exploding) at a near-singular focus.
/// Twin of `caustics::jacobian_caustic_gain`; always non-negative.
fn water_jacobian_gain(jacobian: f32, max_gain: f32) -> f32 {
    let cap = max(max_gain, 0.0);
    let mag = abs(jacobian);
    if mag <= WATER_EPS {
        return cap;
    }
    return min(1.0 / mag, cap);
}

/// Caustic irradiance via Jacobian projection: `incident * gain`, brightening
/// under focus and dimming under defocus. Twin of
/// `caustics::project_caustic_intensity`; non-negative and bounded.
fn water_project_caustic(incident: f32, jacobian: f32, max_gain: f32) -> f32 {
    return max(incident, 0.0) * water_jacobian_gain(jacobian, max_gain);
}

/// Photon-map irradiance estimate for the `RT`/photon route:
/// `photon_count * photon_power / (PI * radius^2)`. Twin of
/// `caustics::photon_splat_density`; a degenerate radius returns zero.
fn water_photon_density(photon_count: u32, photon_power: f32, radius: f32) -> f32 {
    let r = max(radius, 0.0);
    if r <= WATER_EPS {
        return 0.0;
    }
    let area = WATER_PI * r * r;
    return f32(photon_count) * max(photon_power, 0.0) / area;
}

/// `Cauchy` two-term refractive index `n(lambda) = a + b / lambda^2`, falling
/// monotonically with wavelength. Twin of `dispersion::cauchy_ior`; the
/// wavelength is floored so it never divides by zero.
fn water_cauchy_ior(a: f32, b: f32, wavelength_um: f32) -> f32 {
    let lambda = max(wavelength_um, WATER_EPS);
    return a + b / (lambda * lambda);
}

/// Transmitted-ray sine for one channel from `Snell`'s law:
/// `sin_incidence / ior`, clamped to `0..=1`. Twin of
/// `dispersion::channel_transmitted_sine`; a higher `IOR` bends closer to the
/// normal (smaller sine).
fn water_channel_sine(sin_incidence: f32, ior: f32) -> f32 {
    let n = max(ior, WATER_EPS);
    return clamp(clamp(sin_incidence, 0.0, 1.0) / n, 0.0, 1.0);
}

/// `Beer-Lambert` transmittance `exp(-extinction * distance)` in `0..=1`,
/// falling monotonically with either factor. Twin of
/// `underwater::beer_lambert_transmittance`.
fn water_beer_lambert(extinction: f32, path_len: f32) -> f32 {
    let e = max(extinction, 0.0);
    let d = max(path_len, 0.0);
    return clamp(water_exp_approx(-(e * d)), 0.0, 1.0);
}

/// The `Henyey-Greenstein` phase function
/// `(1 - g^2) / (4*PI * (1 + g^2 - 2*g*cos)^(3/2))`. Twin of
/// `underwater::henyey_greenstein`: `g > 0` forward-peaked, `g = 0` isotropic;
/// the `3/2` power is `d * sqrt(d)` so only `sqrt` is used. Always non-negative.
fn water_henyey_greenstein(cos_theta: f32, g: f32) -> f32 {
    let gg = clamp(g, -0.999, 0.999);
    let c = clamp(cos_theta, -1.0, 1.0);
    let denom_base = max(1.0 + gg * gg - 2.0 * gg * c, WATER_EPS);
    let denom = 4.0 * WATER_PI * denom_base * sqrt(denom_base);
    return (1.0 - gg * gg) / denom;
}

/// Bounded multiple-scattering boost `single / (1 - albedo)`, the closed form
/// of the scattering-order geometric series. Twin of
/// `underwater::multiple_scatter_boost`; albedo clamped below one, result
/// non-negative.
fn water_multiple_scatter(single: f32, albedo: f32) -> f32 {
    let a = clamp(albedo, 0.0, 0.999);
    return max(single, 0.0) / (1.0 - a);
}

/// God-ray in-scatter `surface_light * scatter_albedo * (1 - transmittance)`,
/// rising with shaft length and saturating. Twin of
/// `underwater::godray_inscatter`; non-negative.
fn water_godray(surface_light: f32, scatter_albedo: f32, extinction: f32, path_length: f32) -> f32 {
    let light = max(surface_light, 0.0);
    let albedo = clamp(scatter_albedo, 0.0, 1.0);
    let transmittance = water_beer_lambert(extinction, path_length);
    return light * albedo * (1.0 - transmittance);
}

/// Exponential absorption toward full saturation:
/// `1 - (1 - w) * exp(-rate * dt)`. Twin of `wetness::absorb`; stays in `0..=1`
/// and never drops below the input.
fn water_absorb(wetness: f32, rate: f32, dt: f32) -> f32 {
    let w = clamp(wetness, 0.0, 1.0);
    let dry_fraction = (1.0 - w) * water_exp_approx(-max(rate, 0.0) * max(dt, 0.0));
    return clamp(1.0 - dry_fraction, 0.0, 1.0);
}

/// Exponential drying toward zero: `w * exp(-rate * dt)`. Twin of
/// `wetness::dry`; non-negative and never above the input.
fn water_dry(wetness: f32, rate: f32, dt: f32) -> f32 {
    let w = clamp(wetness, 0.0, 1.0);
    return w * water_exp_approx(-max(rate, 0.0) * max(dt, 0.0));
}

/// One wetness step: direct contact soaks at the full rate, else rain wets
/// scaled by its clamped intensity and a dry surface dries. Twin of
/// `wetness::update_wetness`; always a valid `0..=1` saturation.
fn water_update_wetness(current: f32, absorb_rate: f32, dry_rate: f32, water_contact: u32, rain_rate: f32, dt: f32) -> f32 {
    if water_contact != 0u {
        return water_absorb(current, absorb_rate, dt);
    }
    let rain = max(rain_rate, 0.0);
    if rain > WATER_EPS {
        return water_absorb(current, absorb_rate * min(rain, 1.0), dt);
    }
    return water_dry(current, dry_rate, dt);
}

/// Albedo multiplier scaling from `1` (dry) to `1 - darkening` (soaked). Twin
/// of `wetness::wet_albedo_scale`; monotonically non-increasing in wetness.
fn water_wet_albedo(wetness: f32, darkening_strength: f32) -> f32 {
    let w = clamp(wetness, 0.0, 1.0);
    let darkening = clamp(darkening_strength, 0.0, 1.0);
    return clamp(1.0 - darkening * w, 1.0 - darkening, 1.0);
}

/// Capillary wet-band height `reach * wetness * falloff`, falling with distance
/// above the waterline. Twin of `wetness::capillary_height`; a non-positive
/// reach disables the band.
fn water_capillary(wetness: f32, dist_above_water: f32, reach: f32) -> f32 {
    if reach <= WATER_EPS {
        return 0.0;
    }
    let w = clamp(wetness, 0.0, 1.0);
    let falloff = clamp(1.0 - max(dist_above_water, 0.0) / reach, 0.0, 1.0);
    return reach * w * falloff;
}

/// Standing puddle depth integrated over `dt`:
/// `max(depth + (rain - drain) * dt, 0)`. Twin of `wetness::puddle_depth`;
/// never negative.
fn water_puddle_depth(accumulated: f32, rain_rate: f32, drain_rate: f32, dt: f32) -> f32 {
    let base = max(accumulated, 0.0);
    return max(base + (max(rain_rate, 0.0) - max(drain_rate, 0.0)) * max(dt, 0.0), 0.0);
}

/// `Archimedes` buoyancy magnitude `fluid_density * submerged_volume * gravity`.
/// Twin of `coupling::buoyancy_force`; non-negative, rising with submersion.
fn water_buoyancy(fluid_density: f32, submerged_volume: f32, gravity: f32) -> f32 {
    return max(fluid_density, 0.0) * max(submerged_volume, 0.0) * max(gravity, 0.0);
}

/// Quadratic drag magnitude `0.5 * drag_coeff * density * area * speed^2`. Twin
/// of `coupling::drag_force`; non-negative, growing with the square of speed.
fn water_drag(drag_coeff: f32, fluid_density: f32, area: f32, rel_speed: f32) -> f32 {
    let v = max(rel_speed, 0.0);
    return 0.5 * max(drag_coeff, 0.0) * max(fluid_density, 0.0) * max(area, 0.0) * v * v;
}

/// Added-mass reaction `added_mass_coeff * density * displaced_volume`. Twin of
/// `coupling::added_mass`; non-negative.
fn water_added_mass(added_mass_coeff: f32, fluid_density: f32, displaced_volume: f32) -> f32 {
    return max(added_mass_coeff, 0.0) * max(fluid_density, 0.0) * max(displaced_volume, 0.0);
}

/// Source write-back fraction `submerged / total`, clamped to `0..=1`; a zero
/// total is inert. Twin of `coupling::source_writeback_fraction`.
fn water_writeback_fraction(submerged_volume: f32, total_volume: f32) -> f32 {
    let total = max(total_volume, 0.0);
    if total <= WATER_EPS {
        return 0.0;
    }
    return clamp(max(submerged_volume, 0.0) / total, 0.0, 1.0);
}

/// Bounded read-back size `min(query_count, max_readback)`. Twin of the
/// `readback_batch` rule in `coupling::plan_coupling`; forbids a full-field
/// `GPU` read-back per frame.
fn water_readback_batch(query_count: u32, max_readback: u32) -> u32 {
    return min(query_count, max_readback);
}

// ===========================================================================
// Kernel 1: water_caustics_project (Screen, 8x8) — twin of `caustics.rs`.
// ===========================================================================

/// Parameters for the caustics projection pass. Mirrors the scalars the golden
/// `caustics` route derives once per frame.
struct WaterCausticsParams {
    /// Incident irradiance reaching the receiver before focusing.
    incident: f32,
    /// Upper bound on the Jacobian gain, so a near-singular focus is capped.
    max_gain: f32,
    /// Per-photon power for the `RT`/photon splat-density term.
    photon_power: f32,
    /// Photon gather radius for the splat-density term.
    splat_radius: f32,
    /// Receiver-plane texel size, the finite-difference step for the Jacobian.
    texel_size: f32,
    /// Target width in texels.
    width: u32,
    /// Target height in texels.
    height: u32,
    /// Padding to a 16-byte boundary.
    pad: u32,
}

// Per-texel photon counts for the RT/photon splat-density term.
@group(0) @binding(0) var<storage, read> caustics_photon_counts: array<u32>;
// Per-frame caustics scalars.
@group(0) @binding(1) var<uniform> caustics_params: WaterCausticsParams;
// Caustic irradiance render target (single channel).
@group(0) @binding(2) var caustics_out: texture_storage_2d<r32float, write>;
// Refracted-ray screen offset field (xy = lateral offset), read by texel index.
@group(0) @binding(3) var caustics_offsets: texture_2d<f32>;

/// One invocation per receiver texel: builds the refracted-ray Jacobian from
/// finite differences of the offset field, converts its determinant into a
/// focus gain, projects the incident irradiance, and adds the photon splat
/// estimate. Twin of the Jacobian + photon routes in `caustics.rs`.
@compute @workgroup_size(8, 8, 1)
fn water_caustics_project(@builtin(global_invocation_id) gid: vec3<u32>) {
    if gid.x >= caustics_params.width || gid.y >= caustics_params.height {
        return;
    }
    let x = i32(gid.x);
    let y = i32(gid.y);
    let xm = max(x - 1, 0);
    let ym = max(y - 1, 0);
    let xp = min(x + 1, i32(caustics_params.width) - 1);
    let yp = min(y + 1, i32(caustics_params.height) - 1);
    // Refracted-ray screen offsets around the texel.
    let off_xp = textureLoad(caustics_offsets, vec2<i32>(xp, y), 0).xy;
    let off_xm = textureLoad(caustics_offsets, vec2<i32>(xm, y), 0).xy;
    let off_yp = textureLoad(caustics_offsets, vec2<i32>(x, yp), 0).xy;
    let off_ym = textureLoad(caustics_offsets, vec2<i32>(x, ym), 0).xy;
    let fd_step = max(caustics_params.texel_size, WATER_EPS) * 2.0;
    // Jacobian of the surface->receiver map; identity plus offset gradients.
    let dudx = 1.0 + (off_xp.x - off_xm.x) / fd_step;
    let dvdx = (off_xp.y - off_xm.y) / fd_step;
    let dudy = (off_yp.x - off_ym.x) / fd_step;
    let dvdy = 1.0 + (off_yp.y - off_ym.y) / fd_step;
    let jacobian = dudx * dvdy - dvdx * dudy;
    let projected = water_project_caustic(caustics_params.incident, jacobian, caustics_params.max_gain);
    let idx = gid.y * caustics_params.width + gid.x;
    let photons = caustics_photon_counts[idx];
    let photon_term = water_photon_density(photons, caustics_params.photon_power, caustics_params.splat_radius);
    let intensity = max(projected + photon_term, 0.0);
    textureStore(caustics_out, vec2<i32>(x, y), vec4<f32>(intensity, 0.0, 0.0, 0.0));
}

// ===========================================================================
// Kernel 2: water_dispersion_refract (Screen, 8x8) — twin of `dispersion.rs`.
// ===========================================================================

/// Parameters for the spectral dispersion refraction pass. Mirrors the golden
/// `Cauchy` coefficients and refraction strength.
struct WaterDispersionParams {
    /// `Cauchy` baseline index `a`.
    cauchy_a: f32,
    /// `Cauchy` dispersion coefficient `b`, in micrometre-squared.
    cauchy_b: f32,
    /// Screen-space refraction offset gain folded with water thickness.
    strength: f32,
    /// Target width in texels.
    width: u32,
    /// Target height in texels.
    height: u32,
    /// Padding to a 16-byte boundary.
    pad0: u32,
    /// Padding to a 16-byte boundary.
    pad1: u32,
    /// Padding to a 16-byte boundary.
    pad2: u32,
}

// Per-frame dispersion scalars.
@group(1) @binding(0) var<uniform> dispersion_params: WaterDispersionParams;
// Refracted colour render target.
@group(1) @binding(1) var dispersion_out: texture_storage_2d<rgba16float, write>;
// Scene colour behind the water, sampled three times at the RGB offsets.
@group(1) @binding(2) var dispersion_scene: texture_2d<f32>;
// Surface normal field (xyz), read by texel index for the incidence angle.
@group(1) @binding(3) var dispersion_normal: texture_2d<f32>;
// Sampler for the scene colour lookups.
@group(1) @binding(4) var dispersion_sampler: sampler;

/// One invocation per screen texel: evaluates the `Cauchy` `IOR` at the three
/// `RGB` wavelengths, derives the per-channel transmitted-sine offsets (blue
/// bends most, red least), and samples the scene colour along those offsets to
/// assemble the chromatic fringe. Twin of the offset chain in `dispersion.rs`.
@compute @workgroup_size(8, 8, 1)
fn water_dispersion_refract(@builtin(global_invocation_id) gid: vec3<u32>) {
    if gid.x >= dispersion_params.width || gid.y >= dispersion_params.height {
        return;
    }
    let coord = vec2<i32>(i32(gid.x), i32(gid.y));
    let normal = textureLoad(dispersion_normal, coord, 0).xyz;
    // Incidence sine from the surface normal's z (view-space up component).
    let cos_i = clamp(abs(normal.z), 0.0, 1.0);
    let sin_i = sqrt(max(1.0 - cos_i * cos_i, 0.0));
    // Reference RGB wavelengths in micrometres: red, green, blue.
    let ior_r = water_cauchy_ior(dispersion_params.cauchy_a, dispersion_params.cauchy_b, 0.700);
    let ior_g = water_cauchy_ior(dispersion_params.cauchy_a, dispersion_params.cauchy_b, 0.546);
    let ior_b = water_cauchy_ior(dispersion_params.cauchy_a, dispersion_params.cauchy_b, 0.440);
    let gain = max(dispersion_params.strength, 0.0);
    let off_r = water_channel_sine(sin_i, ior_r) * gain;
    let off_g = water_channel_sine(sin_i, ior_g) * gain;
    let off_b = water_channel_sine(sin_i, ior_b) * gain;
    // Lateral refraction direction from the normal's tangent projection.
    let tangent = normal.xy;
    let t_len = sqrt(max(dot(tangent, tangent), WATER_EPS));
    let dir = tangent / t_len;
    let dims = vec2<f32>(f32(dispersion_params.width), f32(dispersion_params.height));
    let uv = (vec2<f32>(f32(gid.x), f32(gid.y)) + vec2<f32>(0.5, 0.5)) / dims;
    let px = dir / dims;
    // Sample the scene colour three times; each channel takes its own offset.
    let s_r = textureSampleLevel(dispersion_scene, dispersion_sampler, uv + px * off_r, 0.0).r;
    let s_g = textureSampleLevel(dispersion_scene, dispersion_sampler, uv + px * off_g, 0.0).g;
    let s_b = textureSampleLevel(dispersion_scene, dispersion_sampler, uv + px * off_b, 0.0).b;
    textureStore(dispersion_out, coord, vec4<f32>(max(s_r, 0.0), max(s_g, 0.0), max(s_b, 0.0), 1.0));
}

// ===========================================================================
// Kernel 3: water_underwater_volume (Grid3d, 4x4x4) — twin of `underwater.rs`.
// ===========================================================================

/// Parameters for the underwater `froxel` volume pass. Mirrors the golden
/// extinction, phase, and scattering scalars.
struct WaterUnderwaterParams {
    /// `RGB` extinction coefficients (`r > g > b` for the blue-green shift).
    extinction: vec3<f32>,
    /// Thickness of one `froxel` depth slice, in metres.
    slice_thickness: f32,
    /// Base surface radiance colour entering the medium.
    base_color: vec3<f32>,
    /// `Henyey-Greenstein` asymmetry `g` in `(-1, 1)`.
    phase_g: f32,
    /// Cosine of the angle between the view and sun directions.
    sun_cos: f32,
    /// Single-scattering albedo for the multiple-scatter boost.
    scatter_albedo: f32,
    /// `froxel` grid width.
    width: u32,
    /// `froxel` grid height.
    height: u32,
    /// `froxel` grid depth (slice count).
    depth: u32,
    /// Padding to a 16-byte boundary.
    pad0: u32,
    /// Padding to a 16-byte boundary.
    pad1: u32,
    /// Padding to a 16-byte boundary.
    pad2: u32,
}

// Per-frame underwater scalars.
@group(2) @binding(0) var<uniform> underwater_params: WaterUnderwaterParams;
// froxel scattering + transmittance output volume.
@group(2) @binding(1) var underwater_out: texture_storage_3d<rgba16float, write>;
// Per-column surface light (sun-shaft) intensity, read by froxel xy.
@group(2) @binding(2) var underwater_surface_light: texture_2d<f32>;

/// One invocation per `froxel` voxel: applies `Beer-Lambert` extinction and the
/// `RGB` colour shift over the slice depth, weights single scatter by the
/// `Henyey-Greenstein` phase, boosts it for multiple scattering, and adds the
/// god-ray in-scatter. Twin of the volume math in `underwater.rs`.
@compute @workgroup_size(4, 4, 4)
fn water_underwater_volume(@builtin(global_invocation_id) gid: vec3<u32>) {
    if gid.x >= underwater_params.width || gid.y >= underwater_params.height || gid.z >= underwater_params.depth {
        return;
    }
    // Depth from the eye to this slice's centre.
    let depth = (f32(gid.z) + 0.5) * max(underwater_params.slice_thickness, 0.0);
    // Per-channel Beer-Lambert transmittance and colour shift (red fades first).
    let tr = water_beer_lambert(underwater_params.extinction.x, depth);
    let tg = water_beer_lambert(underwater_params.extinction.y, depth);
    let tb = water_beer_lambert(underwater_params.extinction.z, depth);
    let shifted = vec3<f32>(
        max(underwater_params.base_color.x, 0.0) * tr,
        max(underwater_params.base_color.y, 0.0) * tg,
        max(underwater_params.base_color.z, 0.0) * tb,
    );
    // Sun light for this column drives the phase-weighted in-scatter.
    let surface_light = textureLoad(underwater_surface_light, vec2<i32>(i32(gid.x), i32(gid.y)), 0).r;
    let phase = water_henyey_greenstein(underwater_params.sun_cos, underwater_params.phase_g);
    let single = max(surface_light, 0.0) * phase * clamp(underwater_params.scatter_albedo, 0.0, 1.0);
    let boosted = water_multiple_scatter(single, underwater_params.scatter_albedo);
    // God-ray glow uses the green extinction as a representative shaft rate.
    let godray = water_godray(surface_light, underwater_params.scatter_albedo, underwater_params.extinction.y, depth);
    let inscatter = shifted * boosted + shifted * godray;
    let avg_transmittance = (tr + tg + tb) / 3.0;
    textureStore(underwater_out, vec3<i32>(i32(gid.x), i32(gid.y), i32(gid.z)),
        vec4<f32>(max(inscatter.x, 0.0), max(inscatter.y, 0.0), max(inscatter.z, 0.0), avg_transmittance));
}

// ===========================================================================
// Kernel 4: water_wetness_step (Grid2d, 8x8) — twin of `wetness.rs`.
// ===========================================================================

/// Parameters for the wetness/capillary/puddle step. Mirrors the golden
/// `WetnessParams` plus the per-step weather drive.
struct WaterWetnessParams {
    /// Maximum capillary rise / wet-band reach, in metres.
    max_capillary_height: f32,
    /// Absorption rate per second on contact or full rain.
    absorb_rate: f32,
    /// Drying rate per second when exposed and rain-free.
    dry_rate: f32,
    /// Peak fractional albedo darkening at full saturation, in `0..=1`.
    darkening_strength: f32,
    /// Puddle depth above which a cell seeds `SWE` ripples.
    puddle_threshold: f32,
    /// Normalised rain wetting drive for this step.
    rain_rate: f32,
    /// Puddle drainage rate for this step.
    drain_rate: f32,
    /// Step duration, seconds.
    dt: f32,
    /// Height above the waterline sampled for the capillary band, in metres.
    dist_above_water: f32,
    /// Non-zero when the surface is in direct water contact this step.
    water_contact: u32,
    /// Field width in cells.
    width: u32,
    /// Field height in cells.
    height: u32,
}

// Per-cell moisture state carried between frames: x = wetness, y = puddle depth.
@group(3) @binding(0) var<storage, read_write> wetness_state: array<vec2<f32>>;
// Per-frame wetness scalars.
@group(3) @binding(1) var<uniform> wetness_params: WaterWetnessParams;
// Output: r = albedo scale, g = capillary height, b = wetness, a = puddle flag.
@group(3) @binding(2) var wetness_out: texture_storage_2d<rgba16float, write>;

/// One invocation per surface cell: advances wetness with the exponential
/// absorb/dry envelopes, integrates the puddle depth, then emits the albedo
/// darkening, capillary-band height, and puddle flag. Twin of
/// `wetness::step_moisture` plus the shading helpers.
@compute @workgroup_size(8, 8, 1)
fn water_wetness_step(@builtin(global_invocation_id) gid: vec3<u32>) {
    if gid.x >= wetness_params.width || gid.y >= wetness_params.height {
        return;
    }
    let idx = gid.y * wetness_params.width + gid.x;
    let state = wetness_state[idx];
    let wet = water_update_wetness(
        state.x,
        wetness_params.absorb_rate,
        wetness_params.dry_rate,
        wetness_params.water_contact,
        wetness_params.rain_rate,
        wetness_params.dt,
    );
    let puddle = water_puddle_depth(state.y, wetness_params.rain_rate, wetness_params.drain_rate, wetness_params.dt);
    wetness_state[idx] = vec2<f32>(wet, puddle);
    let albedo = water_wet_albedo(wet, wetness_params.darkening_strength);
    let capillary = water_capillary(wet, wetness_params.dist_above_water, wetness_params.max_capillary_height);
    let threshold = max(wetness_params.puddle_threshold, 0.0);
    let is_puddle = select(puddle > WATER_EPS, puddle >= threshold, threshold > WATER_EPS);
    let flag = select(0.0, 1.0, is_puddle);
    textureStore(wetness_out, vec2<i32>(i32(gid.x), i32(gid.y)), vec4<f32>(albedo, capillary, wet, flag));
}

// ===========================================================================
// Kernel 5: water_coupling_readback (Particle, 64) — twin of `coupling.rs`.
// ===========================================================================

/// One two-way coupling query: the submerged/total volume, drag cross-section,
/// and relative speed sampled for a floating body. Mirrors the golden coupling
/// force inputs.
struct WaterCouplingQuery {
    /// Volume of the body below the surface.
    submerged_volume: f32,
    /// Total volume of the body.
    total_volume: f32,
    /// Drag cross-section presented to the flow.
    cross_section: f32,
    /// Relative body/fluid speed.
    rel_speed: f32,
}

/// Parameters for the coupling read-back pack. Mirrors the golden fluid profile
/// and the bounded read-back cap.
struct WaterCouplingParams {
    /// Density of the surrounding fluid.
    fluid_density: f32,
    /// Quadratic form-drag coefficient.
    drag_coeff: f32,
    /// Added-mass coefficient for the body shape.
    added_mass_coeff: f32,
    /// Gravitational acceleration magnitude.
    gravity: f32,
    /// Number of queries requested this frame.
    query_count: u32,
    /// Upper bound on the read-back batch (never a full-field read-back).
    max_readback: u32,
    /// Padding to a 16-byte boundary.
    pad0: u32,
    /// Padding to a 16-byte boundary.
    pad1: u32,
}

// Input coupling queries sampled from the fluid field.
@group(4) @binding(0) var<storage, read> coupling_queries: array<WaterCouplingQuery>;
// Packed read-back: xyzw = buoyancy, drag, added mass, write-back fraction.
@group(4) @binding(1) var<storage, read_write> coupling_readback: array<vec4<f32>>;
// Per-frame coupling scalars.
@group(4) @binding(2) var<uniform> coupling_params: WaterCouplingParams;

/// One invocation per query, but only up to the bounded read-back batch
/// `min(query_count, max_readback)` — invocations past the cap return, so the
/// pass never reads the whole field back. Each in-range lane computes the
/// buoyancy, drag, added-mass, and write-back fraction and packs them. Twin of
/// the query + `readback_batch` logic in `coupling.rs`.
@compute @workgroup_size(64)
fn water_coupling_readback(@builtin(global_invocation_id) gid: vec3<u32>) {
    let batch = water_readback_batch(coupling_params.query_count, coupling_params.max_readback);
    if gid.x >= batch {
        return;
    }
    let q = coupling_queries[gid.x];
    let buoyancy = water_buoyancy(coupling_params.fluid_density, q.submerged_volume, coupling_params.gravity);
    let drag = water_drag(coupling_params.drag_coeff, coupling_params.fluid_density, q.cross_section, q.rel_speed);
    let added = water_added_mass(coupling_params.added_mass_coeff, coupling_params.fluid_density, q.submerged_volume);
    let writeback = water_writeback_fraction(q.submerged_volume, q.total_volume);
    coupling_readback[gid.x] = vec4<f32>(buoyancy, drag, added, writeback);
}
