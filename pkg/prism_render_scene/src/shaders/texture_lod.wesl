// GPU twin of `prism_render_material::texture_lod` (ray-cone texture LOD core).
//
// This mirrors the CPU golden's operation order, clamps, floors, and finite
// sanitization exactly so the GPU path tracks the deterministic libm-backed CPU
// reference. Transcendentals (`log2`/`tan`/`atan`/`sqrt`) use WGSL builtins;
// numeric fidelity is anchored by exact structural mirroring, with a device
// parity test reserved for the real-GPU harness.
//
// References: Ray Tracing Gems 2019, ch. 20 ("Texture Level of Detail
// Strategies for Real-Time Ray Tracing"); Ewins et al. 1998.

// --- vector helpers: mirror texture_lod/math.rs operation order exactly ---

// Dot product of two 3D vectors, left-to-right accumulation.
fn tex_lod_dot3(a: vec3<f32>, b: vec3<f32>) -> f32 {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

// Component-wise difference `a - b`.
fn tex_lod_sub3(a: vec3<f32>, b: vec3<f32>) -> vec3<f32> {
    return vec3<f32>(a.x - b.x, a.y - b.y, a.z - b.z);
}

// Cross product of two 3D vectors.
fn tex_lod_cross3(a: vec3<f32>, b: vec3<f32>) -> vec3<f32> {
    return vec3<f32>(
        a.y * b.z - a.z * b.y,
        a.z * b.x - a.x * b.z,
        a.x * b.y - a.y * b.x
    );
}

// Euclidean length `sqrt(dot(a, a))`.
fn tex_lod_length3(a: vec3<f32>) -> f32 {
    return sqrt(tex_lod_dot3(a, a));
}

// Signed twice-area of the UV triangle (2D cross of its two edge vectors).
fn tex_lod_uv_double_area(uv0: vec2<f32>, uv1: vec2<f32>, uv2: vec2<f32>) -> f32 {
    let e1 = vec2<f32>(uv1.x - uv0.x, uv1.y - uv0.y);
    let e2 = vec2<f32>(uv2.x - uv0.x, uv2.y - uv0.y);
    return e1.x * e2.y - e1.y * e2.x;
}

// --- triangle LOD constant: mirror texture_lod/triangle.rs ---

// Smallest area (either space) treated as non-degenerate, in squared units.
const TEX_LOD_MIN_AREA: f32 = 1.0e-12;
// Clamp bound for `Delta` so a pathological triangle stays finite and coarse.
const TEX_LOD_DELTA_CLAMP: f32 = 24.0;

// Triangle texel-density constant `Delta = 0.5 * log2(T_a / W_a)`.
//
// `tex_width`/`tex_height` are the sampled texture's mip-0 dimensions in texels
// (pass the integral dimensions; they are exact in f32 for real textures).
fn tex_lod_triangle_delta(
    w0: vec3<f32>,
    w1: vec3<f32>,
    w2: vec3<f32>,
    uv0: vec2<f32>,
    uv1: vec2<f32>,
    uv2: vec2<f32>,
    tex_width: f32,
    tex_height: f32
) -> f32 {
    let uv_area2 = abs(tex_lod_uv_double_area(uv0, uv1, uv2));
    let texels = tex_width * tex_height;
    let t_a = max(texels * uv_area2, TEX_LOD_MIN_AREA);

    let e1 = tex_lod_sub3(w1, w0);
    let e2 = tex_lod_sub3(w2, w0);
    let w_a = max(tex_lod_length3(tex_lod_cross3(e1, e2)), TEX_LOD_MIN_AREA);

    let delta = 0.5 * log2(t_a / w_a);
    return clamp(delta, -TEX_LOD_DELTA_CLAMP, TEX_LOD_DELTA_CLAMP);
}

// --- mip-level selection: mirror texture_lod/mip.rs ---

// Lower bound on `|n . d|`; grazing hits yield a large-but-finite LOD boost.
const TEX_LOD_MIN_COS_INCIDENCE: f32 = 1.0e-3;
// `f32::MIN_POSITIVE` as a literal (smallest positive normal f32).
const TEX_LOD_F32_MIN_POSITIVE: f32 = 1.17549435e-38;

// Map an isotropic footprint radius (texels) to a continuous mip, clamped to
// `[0, max(max_mip, 0)]`.
fn tex_lod_mip_from_isotropic_footprint(footprint_texels: f32, max_mip: f32) -> f32 {
    let f = max(footprint_texels, TEX_LOD_F32_MIN_POSITIVE);
    return clamp(log2(f), 0.0, max(max_mip, 0.0));
}

// Ray-cone mip level: `lambda = Delta + log2(|cone_width|) - log2(|n . d|)`.
fn tex_lod_cone_mip_level(
    delta: f32,
    cone_width: f32,
    n_dot_d: f32,
    max_mip: f32
) -> f32 {
    let width = max(abs(cone_width), TEX_LOD_F32_MIN_POSITIVE);
    let cos_i = max(abs(n_dot_d), TEX_LOD_MIN_COS_INCIDENCE);
    let lambda = delta + log2(width) - log2(cos_i);
    return clamp(lambda, 0.0, max(max_mip, 0.0));
}

// --- ray cone propagation: mirror texture_lod/ray_cone.rs ---

// Maximum spread half-angle (radians); `FRAC_PI_2` rounded to nearest f32.
const TEX_LOD_MAX_SPREAD_ANGLE: f32 = 1.5707963267948966;

// A propagated ray cone (footprint diameter + growth half-angle).
struct RayCone {
    width: f32,
    spread_angle: f32,
}

// Construct a cone from explicit width/spread, both sanitized to range.
fn tex_lod_ray_cone_new(width: f32, spread_angle: f32) -> RayCone {
    var cone: RayCone;
    cone.width = max(width, 0.0);
    cone.spread_angle = clamp(spread_angle, 0.0, TEX_LOD_MAX_SPREAD_ANGLE);
    return cone;
}

// Primary-ray cone for a pinhole camera: zero width, per-pixel spread
// `atan(2 * tan(fov / 2) / height)`.
fn tex_lod_ray_cone_from_pinhole_pixel(vertical_fov: f32, screen_height_px: f32) -> RayCone {
    let height = max(screen_height_px, 1.0);
    let half = tan(max(vertical_fov, 0.0) * 0.5);
    let spread = atan(2.0 * half / height);
    return tex_lod_ray_cone_new(0.0, spread);
}

// Advance the cone by `hit_distance`, widening by `distance * spread_angle`.
fn tex_lod_ray_cone_advanced(cone: RayCone, hit_distance: f32) -> RayCone {
    let t = max(hit_distance, 0.0);
    var out: RayCone;
    out.width = cone.width + t * cone.spread_angle;
    out.spread_angle = cone.spread_angle;
    return out;
}

// Update the spread angle at a scattering event (width unchanged).
fn tex_lod_ray_cone_scattered(cone: RayCone, curvature_spread: f32, roughness_spread: f32) -> RayCone {
    let extra = max(curvature_spread, 0.0) + max(roughness_spread, 0.0);
    var out: RayCone;
    out.width = cone.width;
    out.spread_angle = clamp(cone.spread_angle + extra, 0.0, TEX_LOD_MAX_SPREAD_ANGLE);
    return out;
}

// Reflection update: adds `2 * curvature_spread` plus a roughness widening.
fn tex_lod_ray_cone_reflected(cone: RayCone, surface_curvature_spread: f32, roughness_spread: f32) -> RayCone {
    return tex_lod_ray_cone_scattered(cone, 2.0 * max(surface_curvature_spread, 0.0), roughness_spread);
}
