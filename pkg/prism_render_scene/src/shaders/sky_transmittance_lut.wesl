//! Hillaire-style physical sky transmittance LUT.
//! The equations intentionally mirror
//! prism_render_shading::gi::atmosphere::transmittance::transmittance_to_boundary
//! (Earth constants, planet-centred ray/shell intersection, midpoint optical-depth
//! march, and exp(-tau)). 2D LUT parameterized by (altitude, view-zenith cosine)
//! per Hillaire: x -> altitude in [0, THICKNESS], y -> mu = cos(zenith) in [-1, 1].
//!
//! The boundary selection reproduces `distance_to_boundary` exactly: the ground
//! shell uses the *nearest* positive root (`nearest_positive_intersection`), the
//! top shell uses the far exit root (`top_exit_distance`), and the ground wins
//! only when it lies strictly ahead of the top exit.

struct SkyTransmittanceParams { width: u32, height: u32, samples: u32, pad: u32, }
var<immediate> params: SkyTransmittanceParams;
@group(0) @binding(0) var lut_out: texture_storage_2d<rgba16float, write>;

const EPSILON: f32 = 0.0001;
const BOTTOM: f32 = 6360.0;
const TOP: f32 = 6460.0;
const THICKNESS: f32 = 100.0;
const RAYLEIGH: vec3<f32> = vec3<f32>(5.802e-3, 13.558e-3, 33.1e-3);
const RAYLEIGH_H: f32 = 8.0;
const MIE_E: f32 = 4.44e-3;
const MIE_H: f32 = 1.2;
const OZONE: vec3<f32> = vec3<f32>(0.650e-3, 1.881e-3, 0.085e-3);
const OZONE_C: f32 = 25.0;
const OZONE_W: f32 = 15.0;

fn sane3(v: vec3<f32>) -> vec3<f32> { return max(v, vec3<f32>(0.0)); }
fn density(h: f32, scale: f32) -> f32 { return clamp(exp(-max(h, 0.0) / scale), 0.0, 1.0); }
fn ozone_density(h: f32) -> f32 { return clamp(1.0 - abs(max(h, 0.0) - OZONE_C) / OZONE_W, 0.0, 1.0); }
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
fn boundary(o: vec3<f32>, d: vec3<f32>) -> f32 {
  let top = top_exit(o, d, TOP); let ground = nearest_positive(o, d, BOTTOM);
  if (ground >= 0.0 && ground < top) { return ground; } return top;
}
fn transmittance(o: vec3<f32>, d: vec3<f32>, samples: u32) -> vec3<f32> {
  let dist = boundary(o, d); if (!(dist > 0.0)) { return vec3<f32>(1.0); }
  let n = max(samples, 1u); let ds = dist / f32(n); var tau = vec3<f32>(0.0);
  for (var i: u32 = 0u; i < n; i++) { let p = o + d * ((f32(i) + 0.5) * ds); tau += extinction(altitude(p)) * ds; }
  return exp_neg3(tau);
}

@compute @workgroup_size(8, 8, 1)
fn sky_transmittance_lut(@builtin(global_invocation_id) gid: vec3<u32>) {
  if (gid.x >= params.width || gid.y >= params.height) { return; }
  let altitude_value = (f32(gid.x) + 0.5) / f32(params.width) * THICKNESS;
  let mu = (f32(gid.y) + 0.5) / f32(params.height) * 2.0 - 1.0;
  let s = sqrt(max(1.0 - mu * mu, 0.0));
  let dir = normalize(vec3<f32>(s, mu, 0.0));
  let point = vec3<f32>(0.0, BOTTOM + altitude_value, 0.0);
  let value = transmittance(point, dir, max(params.samples, 1u));
  textureStore(lut_out, vec2<i32>(i32(gid.x), i32(gid.y)), vec4<f32>(value, 1.0));
}
