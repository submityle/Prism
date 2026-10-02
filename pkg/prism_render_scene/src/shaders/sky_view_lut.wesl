//! Hillaire-style physical sky *sky-view* LUT — the consumer of the
//! multiple-scattering and transmittance LUTs.
//!
//! The equations intentionally mirror the CPU golden
//! prism_render_shading::gi::atmosphere::scattering::sky_view_radiance
//! scalar-for-scalar: identical Earth constants, planet-centred ray/shell
//! intersection (faithful near/far roots, matching `distance_to_boundary`),
//! midpoint view-ray march, Rayleigh/Cornette-Shanks phases, and the per-step
//! single-scatter + multiple-scatter accumulation.
//!
//! Per Hillaire 2020 the inner `sun_transmittance` and `multiscatter_estimate`
//! marches of the golden are replaced here by texture fetches of the baked
//! transmittance LUT and multiple-scattering LUT at the sample's
//! `(altitude, sun_cos_zenith)` coordinate — this is precisely what makes the
//! pass the LUT chain's consumer. Both sibling LUTs share the same
//! parameterization: x -> altitude in [0, THICKNESS], y -> cos(zenith) in
//! [-1, 1], evaluated at a point on the +Y axis, so the lookup coordinate is
//! `(altitude / THICKNESS, (sun_cos_zenith + 1) * 0.5)`.
//!
//! The sky-view LUT itself is parameterized by (azimuth, view-zenith cosine):
//! x -> azimuth in [0, 2π), y -> cos(view-zenith) in [1, -1]. The bake frame is
//! fixed (and documented) to match the CPU golden's test frame:
//!   origin  = (0, BOTTOM + 0.5, 0)            (`ground_origin`)
//!   sun_dir = normalize(vec3(0.3, 0.95, 0.0)) (golden test sun)
//! IRRADIANCE = 20 matches the irradiance baked into the multiple-scattering
//! LUT, so the LUT-sampled multiscatter term couples in at the correct energy.

struct SkyViewParams { width: u32, height: u32, samples: u32, sun_samples: u32, }
var<immediate> params: SkyViewParams;
@group(0) @binding(0) var lut_sampler: sampler;
@group(0) @binding(1) var ms_lut: texture_2d<f32>;
@group(0) @binding(2) var tr_lut: texture_2d<f32>;
@group(0) @binding(3) var lut_out: texture_storage_2d<rgba16float, write>;

const PI: f32 = 3.141592653589793;
const EPSILON: f32 = 0.0001;
const BOTTOM: f32 = 6360.0;
const TOP: f32 = 6460.0;
const THICKNESS: f32 = 100.0;
const IRRADIANCE: vec3<f32> = vec3<f32>(20.0);
const RAYLEIGH: vec3<f32> = vec3<f32>(5.802e-3, 13.558e-3, 33.1e-3);
const RAYLEIGH_H: f32 = 8.0;
const MIE_S: f32 = 3.996e-3;
const MIE_E: f32 = 4.44e-3;
const MIE_H: f32 = 1.2;
const MIE_G: f32 = 0.8;
const OZONE: vec3<f32> = vec3<f32>(0.650e-3, 1.881e-3, 0.085e-3);
const OZONE_C: f32 = 25.0;
const OZONE_W: f32 = 15.0;
const THREE_OVER_16PI: f32 = 3.0 / (16.0 * PI);
const THREE_OVER_8PI: f32 = 3.0 / (8.0 * PI);
const MAX_G: f32 = 1.0 - 1.0e-4;

fn sane3(v: vec3<f32>) -> vec3<f32> { return max(v, vec3<f32>(0.0)); }
fn density(h: f32, scale: f32) -> f32 { return clamp(exp(-max(h, 0.0) / scale), 0.0, 1.0); }
fn ozone_density(h: f32) -> f32 { return clamp(1.0 - abs(max(h, 0.0) - OZONE_C) / OZONE_W, 0.0, 1.0); }
fn rayleigh_scattering(h: f32) -> vec3<f32> { return RAYLEIGH * density(h, RAYLEIGH_H); }
fn mie_scattering(h: f32) -> f32 { return MIE_S * density(h, MIE_H); }
fn scattering(h: f32) -> vec3<f32> { return sane3(RAYLEIGH * density(h, RAYLEIGH_H) + vec3<f32>(MIE_S * density(h, MIE_H))); }
fn extinction(h: f32) -> vec3<f32> { return sane3(RAYLEIGH * density(h, RAYLEIGH_H) + vec3<f32>(MIE_E * density(h, MIE_H)) + OZONE * ozone_density(h)); }
fn exp_neg3(t: vec3<f32>) -> vec3<f32> { return clamp(exp(-min(max(t, vec3<f32>(0.0)), vec3<f32>(80.0))), vec3<f32>(0.0), vec3<f32>(1.0)); }
fn altitude(p: vec3<f32>) -> f32 { return clamp(length(p) - BOTTOM, 0.0, THICKNESS); }

// Nearest positive root (CPU `nearest_positive_intersection`); -1 when none ahead.
fn nearest_positive(o: vec3<f32>, d: vec3<f32>, r: f32) -> f32 {
  let b = dot(o, d); let c = dot(o, o) - r * r; let disc = b * b - c;
  if (disc < 0.0) { return -1.0; }
  let s = sqrt(disc); let t0 = -b - s; let t1 = -b + s;
  if (t0 > EPSILON) { return t0; } if (t1 > EPSILON) { return t1; } return -1.0;
}
// Far positive exit root (CPU `top_exit_distance`); 0 when none ahead.
fn top_exit(o: vec3<f32>, d: vec3<f32>, r: f32) -> f32 {
  let b = dot(o, d); let c = dot(o, o) - r * r; let disc = b * b - c;
  if (disc < 0.0) { return 0.0; }
  let s = sqrt(disc); let t0 = -b - s; let t1 = -b + s;
  if (t1 > EPSILON) { return t1; } if (t0 > EPSILON) { return t0; } return 0.0;
}
// Faithful `distance_to_boundary`: ground shell = nearest positive root, top
// shell = far exit root, ground wins only when strictly ahead of the top exit.
fn boundary(o: vec3<f32>, d: vec3<f32>) -> f32 {
  let top = top_exit(o, d, TOP); let ground = nearest_positive(o, d, BOTTOM);
  if (ground >= 0.0 && ground < top) { return ground; } return top;
}

fn rayleigh_phase(mu: f32) -> f32 { let m = clamp(mu, -1.0, 1.0); return max(THREE_OVER_16PI * (1.0 + m * m), 0.0); }
fn cornette_shanks_phase(mu: f32, g0: f32) -> f32 {
  let m = clamp(mu, -1.0, 1.0); let g = clamp(g0, -MAX_G, MAX_G); let g2 = g * g;
  let num = (1.0 - g2) * (1.0 + m * m);
  let denom = max(1.0 + g2 - 2.0 * g * m, 1.0e-12);
  return max(THREE_OVER_8PI * num / ((2.0 + g2) * pow(denom, 1.5)), 0.0);
}

// LUT lookup coordinate shared by both sibling LUTs.
fn lut_uv(alt: f32, sun_cos: f32) -> vec2<f32> {
  return vec2<f32>(clamp(alt / THICKNESS, 0.0, 1.0), clamp((sun_cos + 1.0) * 0.5, 0.0, 1.0));
}

@compute @workgroup_size(8, 8, 1)
fn sky_view_lut(@builtin(global_invocation_id) gid: vec3<u32>) {
  if (gid.x >= params.width || gid.y >= params.height) { return; }

  // Fixed bake frame (mirrors the CPU golden's test frame).
  let origin = vec3<f32>(0.0, BOTTOM + 0.5, 0.0);
  let sun = normalize(vec3<f32>(0.3, 0.95, 0.0));

  // Texel -> view direction. Azimuth about +Y, zenith cosine from +1 (up) to -1.
  let az = (f32(gid.x) + 0.5) / f32(params.width) * 2.0 * PI;
  let cos_v = 1.0 - 2.0 * (f32(gid.y) + 0.5) / f32(params.height);
  let sin_v = sqrt(max(1.0 - cos_v * cos_v, 0.0));
  let dir = normalize(vec3<f32>(sin_v * cos(az), cos_v, sin_v * sin(az)));

  let distance = boundary(origin, dir);
  if (!(distance > 0.0)) {
    textureStore(lut_out, vec2<i32>(i32(gid.x), i32(gid.y)), vec4<f32>(0.0, 0.0, 0.0, 1.0));
    return;
  }

  let cos_theta = clamp(dot(dir, sun), -1.0, 1.0);
  let phase_r = rayleigh_phase(cos_theta);
  let phase_m = cornette_shanks_phase(cos_theta, MIE_G);

  let steps = max(params.samples, 1u);
  let ds = distance / f32(steps);
  var optical_depth = vec3<f32>(0.0);
  var radiance = vec3<f32>(0.0);
  for (var i: u32 = 0u; i < steps; i++) {
    let pos = origin + dir * ((f32(i) + 0.5) * ds);
    let alt = altitude(pos);
    let ext = extinction(alt);
    let t_view = exp_neg3(optical_depth + ext * (0.5 * ds));

    // Single-scattering integrand (identical to `integrate_inscatter`).
    let scatter_phase = rayleigh_scattering(alt) * phase_r + vec3<f32>(mie_scattering(alt) * phase_m);
    let up = normalize(pos);
    let sun_cos = clamp(dot(sun, up), -1.0, 1.0);
    // Sun transmittance via the baked LUT; zero when the sun ray hits the ground
    // (CPU `sun_transmittance` shadow test via `nearest_positive_intersection`).
    var t_sun = vec3<f32>(0.0);
    if (nearest_positive(pos, sun, BOTTOM) < 0.0) {
      t_sun = textureSampleLevel(tr_lut, lut_sampler, lut_uv(alt, sun_cos), 0.0).rgb;
    }
    radiance += t_view * t_sun * scatter_phase * IRRADIANCE * ds;

    // Multiple-scattering term from the baked LUT (`multiscatter_estimate`),
    // coupled by the local scattering coefficient: T_view * σ_s(h) * L_ms * ds.
    let l_ms = textureSampleLevel(ms_lut, lut_sampler, lut_uv(alt, sun_cos), 0.0).rgb;
    radiance += t_view * scattering(alt) * l_ms * ds;

    optical_depth += ext * ds;
  }

  textureStore(lut_out, vec2<i32>(i32(gid.x), i32(gid.y)), vec4<f32>(sane3(radiance), 1.0));
}
