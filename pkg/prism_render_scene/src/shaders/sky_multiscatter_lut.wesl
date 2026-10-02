//! Hillaire-style physical sky multiple-scattering LUT.
//! The equations intentionally mirror prism_render_shading::gi::atmosphere::scattering
//! (multiscatter_terms / march_second_order), including constants, Fibonacci
//! directions, midpoint march, isotropic phase, and geometric coupling.

struct SkyMultiscatterParams { width: u32, height: u32, dir_samples: u32, march_samples: u32, }
var<immediate> params: SkyMultiscatterParams;
@group(0) @binding(0) var lut_out: texture_storage_2d<rgba16float, write>;

const PI: f32 = 3.141592653589793;
const INV_4PI: f32 = 1.0 / (4.0 * PI);
const GOLDEN_ANGLE: f32 = PI * (3.0 - 2.2360679);
const MAX_FRACTION: f32 = 0.999;
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
const OZONE: vec3<f32> = vec3<f32>(0.650e-3, 1.881e-3, 0.085e-3);
const OZONE_C: f32 = 25.0;
const OZONE_W: f32 = 15.0;

fn sane(x: f32) -> f32 { return max(x, 0.0); }
fn sane3(v: vec3<f32>) -> vec3<f32> { return max(v, vec3<f32>(0.0)); }
fn density(h: f32, scale: f32) -> f32 { return clamp(exp(-max(h, 0.0) / scale), 0.0, 1.0); }
fn ozone_density(h: f32) -> f32 { return clamp(1.0 - abs(max(h, 0.0) - OZONE_C) / OZONE_W, 0.0, 1.0); }
fn scattering(h: f32) -> vec3<f32> { return sane3(RAYLEIGH * density(h, RAYLEIGH_H) + vec3<f32>(MIE_S * density(h, MIE_H))); }
fn extinction(h: f32) -> vec3<f32> { return sane3(RAYLEIGH * density(h, RAYLEIGH_H) + vec3<f32>(MIE_E * density(h, MIE_H)) + OZONE * ozone_density(h)); }
fn exp_neg3(t: vec3<f32>) -> vec3<f32> { return clamp(exp(-min(max(t, vec3<f32>(0.0)), vec3<f32>(80.0))), vec3<f32>(0.0), vec3<f32>(1.0)); }
fn altitude(p: vec3<f32>) -> f32 { return clamp(length(p) - BOTTOM, 0.0, THICKNESS); }

fn sphere_distance(o: vec3<f32>, d: vec3<f32>, r: f32) -> f32 {
  let b = dot(o, d); let c = dot(o, o) - r * r; let disc = b * b - c;
  if (disc < 0.0) { return 0.0; }
  let s = sqrt(disc); let t0 = -b - s; let t1 = -b + s;
  if (t1 > EPSILON) { return t1; } if (t0 > EPSILON) { return t0; } return 0.0;
}
fn boundary(o: vec3<f32>, d: vec3<f32>) -> f32 {
  let top = sphere_distance(o, d, TOP); let ground = sphere_distance(o, d, BOTTOM);
  if (ground > EPSILON && ground < top) { return ground; } return top;
}
fn transmittance(o: vec3<f32>, d: vec3<f32>, samples: u32) -> vec3<f32> {
  let dist = boundary(o, d); if (!(dist > 0.0)) { return vec3<f32>(1.0); }
  let n = max(samples, 1u); let ds = dist / f32(n); var tau = vec3<f32>(0.0);
  for (var i: u32 = 0u; i < n; i++) { let p = o + d * ((f32(i) + 0.5) * ds); tau += extinction(altitude(p)) * ds; }
  return exp_neg3(tau);
}
fn sun_t(o: vec3<f32>, d: vec3<f32>) -> vec3<f32> {
  if (sphere_distance(o, d, BOTTOM) > EPSILON) { return vec3<f32>(0.0); }
  return transmittance(o, d, 8u);
}
fn fib(i: u32, n0: u32) -> vec3<f32> {
  let n = f32(max(n0, 1u)); let z = 1.0 - 2.0 * (f32(i) + 0.5) / n; let r = sqrt(max(1.0 - z*z, 0.0)); let p = GOLDEN_ANGLE * f32(i);
  return vec3<f32>(r * cos(p), z, r * sin(p));
}
fn total(f: f32) -> f32 { let x = clamp(f, 0.0, MAX_FRACTION); return max(1.0 / (1.0 - x), 1.0); }

@compute @workgroup_size(8, 8, 1)
fn sky_multiscatter_lut(@builtin(global_invocation_id) gid: vec3<u32>) {
  if (gid.x >= params.width || gid.y >= params.height) { return; }
  let altitude_value = (f32(gid.x) + 0.5) / f32(params.width) * THICKNESS;
  let sun_cos = (f32(gid.y) + 0.5) / f32(params.height) * 2.0 - 1.0;
  let s = sqrt(max(1.0 - sun_cos * sun_cos, 0.0)); let sun = normalize(vec3<f32>(s, sun_cos, 0.0));
  let point = vec3<f32>(0.0, BOTTOM + altitude_value, 0.0); let n = max(params.dir_samples, 1u); let steps = max(params.march_samples, 1u);
  var l2 = vec3<f32>(0.0); var f = vec3<f32>(0.0);
  for (var j: u32 = 0u; j < n; j++) {
    let dir = fib(j, n); let dist = boundary(point, dir); if (!(dist > 0.0)) { continue; }
    let ds = dist / f32(steps); var tau = vec3<f32>(0.0); var ld = vec3<f32>(0.0); var fd = vec3<f32>(0.0);
    for (var i: u32 = 0u; i < steps; i++) {
      let p = point + dir * ((f32(i) + 0.5) * ds); let ext = extinction(altitude(p)); let tv = exp_neg3(tau + ext * (0.5 * ds)); let sc = scattering(altitude(p)); let st = sun_t(p, sun); let w = INV_4PI * ds;
      fd += tv * sc * w; ld += tv * sc * st * IRRADIANCE * w; tau += ext * ds;
    }
    l2 += ld; f += fd;
  }
  let inv_n = 1.0 / f32(n); l2 = sane3(l2 * inv_n); f = sane3(f * inv_n);
  let value = sane3(vec3<f32>(l2.x * total(f.x), l2.y * total(f.y), l2.z * total(f.z)));
  textureStore(lut_out, vec2<i32>(i32(gid.x), i32(gid.y)), vec4<f32>(value, 1.0));
}
