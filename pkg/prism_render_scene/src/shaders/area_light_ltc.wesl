// Area-light Linearly Transformed Cosines (`LTC`) evaluation kernel.
//
// A self-contained `WESL` twin of the `CPU` golden
// `prism_render_shading::gi::area_light::polygon`: it horizon-clips a quad
// light polygon to the upper hemisphere, integrates the closed-form
// clamped-cosine edge integral (Heitz et al. 2016) and applies the `LTC`
// inverse transform `M-inverse` to obtain the specular response. The same
// machinery with the identity transform yields the diffuse (clamped-cosine)
// form factor.
//
// Conventions mirror the golden exactly: every polygon vertex is a position
// relative to the shaded point in a shading frame whose surface normal is `+Z`,
// `polygon_form_factor` returns the normalised integral `F` in `[0, 1]`, and the
// specular response is `clamp(amplitude, 0, 1) * F(M-inverse . polygon)`.
//
// No intra-crate imports: this file is standalone so its parse / type-check is
// pinned by the `shader_tests` `WESL`-compile test, and the inlined maths is
// cross-checked against the `CPU` golden there.

// Near-parallel edge directions span zero solid angle (golden `MIN_SIN`).
const MIN_SIN: f32 = 1.0e-7;
// Vertices this close to the shaded point are dropped (golden `MIN_LEN_SQ`).
const MIN_LEN_SQ: f32 = 1.0e-18;
const TWO_PI: f32 = 6.28318530717958648;
const F32_FINITE_MAX: f32 = 3.40282347e38;

// Single `var<immediate>` block mirroring `GpuAreaLightLtcParams` (96 bytes):
// four quad corners (`w` unused) plus the two packed `LTC` coefficient rows
// `coeffs0 = (a00, a02, a11, a20)` and `coeffs1 = (a22, amplitude, _, _)`.
struct AreaLightLtcParams {
    points: array<vec4<f32>, 4>,
    coeffs0: vec4<f32>,
    coeffs1: vec4<f32>,
}

var<immediate> params: AreaLightLtcParams;

// Scalar `rgba32float`-style scratch output so the inlined fns are reachable
// and the kernel compiles / links standalone (`x` specular, `y` diffuse).
@group(0) @binding(0) var<storage, read_write> out: array<f32>;

fn is_finite(x: f32) -> bool {
    return x == x && abs(x) <= F32_FINITE_MAX;
}

fn vec_is_finite(v: vec3<f32>) -> bool {
    return is_finite(v.x) && is_finite(v.y) && is_finite(v.z);
}

// Vector edge integral of a clamped-cosine distribution for one polygon edge
// (golden `integrate_edge_vec`): `(acos(v1.v2) / |v1 x v2|) * (v1 x v2)`.
fn integrate_edge_vec(v1: vec3<f32>, v2: vec3<f32>) -> vec3<f32> {
    let cos_theta = clamp(dot(v1, v2), -1.0, 1.0);
    let theta = acos(cos_theta);
    let cross_v = cross(v1, v2);
    let sin_theta = length(cross_v);
    if (sin_theta < MIN_SIN) {
        return vec3<f32>(0.0, 0.0, 0.0);
    }
    let result = cross_v * (theta / sin_theta);
    if (vec_is_finite(result)) {
        return result;
    }
    return vec3<f32>(0.0, 0.0, 0.0);
}

// `z`-component of the edge integral: the edge's contribution to the form
// factor of a `+Z`-normal surface (golden `integrate_edge`).
fn integrate_edge(v1: vec3<f32>, v2: vec3<f32>) -> f32 {
    return integrate_edge_vec(v1, v2).z;
}

// Intersection of segment `a -> b` with the `z = 0` plane. `w = 1` marks a
// valid finite intersection, `w = 0` none (golden `intersect_horizon`).
fn intersect_horizon(a: vec3<f32>, b: vec3<f32>) -> vec4<f32> {
    let dz = a.z - b.z;
    if (abs(dz) < 1.0e-12) {
        return vec4<f32>(0.0, 0.0, 0.0, 0.0);
    }
    let t = a.z / dz;
    if (!(t >= 0.0 && t <= 1.0)) {
        return vec4<f32>(0.0, 0.0, 0.0, 0.0);
    }
    let p = a + (b - a) * t;
    if (vec_is_finite(p)) {
        return vec4<f32>(p, 1.0);
    }
    return vec4<f32>(0.0, 0.0, 0.0, 0.0);
}

// Normalised clamped-cosine form factor `F` in `[0, 1]` of the input quad
// (golden `polygon_form_factor` specialised to four vertices): Sutherland-
// Hodgman horizon clip against `z = 0`, project survivors onto the unit sphere,
// sum the edge integrals and divide by `2*pi`.
fn polygon_form_factor_quad(input: array<vec3<f32>, 4>) -> f32 {
    let n = 4u;
    // Clip to the upper hemisphere `z >= 0`; a convex quad yields at most five
    // survivors, so an eight-slot ring is always sufficient.
    var clipped: array<vec3<f32>, 8>;
    var cc = 0u;
    for (var i = 0u; i < n; i = i + 1u) {
        let cur = input[i];
        let prev = input[(i + n - 1u) % n];
        let cur_in = cur.z >= 0.0;
        let prev_in = prev.z >= 0.0;
        if (cur_in) {
            if (!prev_in) {
                let hit = intersect_horizon(prev, cur);
                if (hit.w > 0.5) {
                    clipped[cc] = hit.xyz;
                    cc = cc + 1u;
                }
            }
            clipped[cc] = cur;
            cc = cc + 1u;
        } else if (prev_in) {
            let hit = intersect_horizon(prev, cur);
            if (hit.w > 0.5) {
                clipped[cc] = hit.xyz;
                cc = cc + 1u;
            }
        }
    }
    if (cc < 3u) {
        return 0.0;
    }
    // Project surviving vertices onto the unit sphere, dropping those coincident
    // with the shaded point.
    var dirs: array<vec3<f32>, 8>;
    var k = 0u;
    for (var i = 0u; i < cc; i = i + 1u) {
        let p = clipped[i];
        let len_sq = dot(p, p);
        if (len_sq <= MIN_LEN_SQ) {
            continue;
        }
        let d = p / sqrt(len_sq);
        if (dot(d, d) > 0.5) {
            dirs[k] = d;
            k = k + 1u;
        }
    }
    if (k < 3u) {
        return 0.0;
    }
    var sum = 0.0;
    for (var i = 0u; i < k; i = i + 1u) {
        let v1 = dirs[i];
        let v2 = dirs[(i + 1u) % k];
        sum = sum + integrate_edge(v1, v2);
    }
    let f = sum / TWO_PI;
    if (is_finite(f)) {
        return clamp(f, 0.0, 1.0);
    }
    return 0.0;
}

// Applies the `LTC` inverse transform `M-inverse` to a direction (golden
// `LtcCoeffs::apply`): the block-sparse `(a00, a02, a11, a20, a22)` matrix.
fn ltc_apply(d: vec3<f32>) -> vec3<f32> {
    let a00 = params.coeffs0.x;
    let a02 = params.coeffs0.y;
    let a11 = params.coeffs0.z;
    let a20 = params.coeffs0.w;
    let a22 = params.coeffs1.x;
    return vec3<f32>(
        a00 * d.x + a02 * d.z,
        a11 * d.y,
        a20 * d.x + a22 * d.z,
    );
}

// `LTC` specular response of the quad: transform each corner by `M-inverse`,
// evaluate the clamped-cosine form factor and scale by the clamped lobe
// amplitude (golden `ltc_evaluate` / `quad_ltc_evaluate`).
fn ltc_evaluate_quad(p0: vec3<f32>, p1: vec3<f32>, p2: vec3<f32>, p3: vec3<f32>) -> f32 {
    var pts: array<vec3<f32>, 4>;
    pts[0] = ltc_apply(p0);
    pts[1] = ltc_apply(p1);
    pts[2] = ltc_apply(p2);
    pts[3] = ltc_apply(p3);
    for (var i = 0u; i < 4u; i = i + 1u) {
        if (!vec_is_finite(pts[i])) {
            pts[i] = vec3<f32>(0.0, 0.0, 0.0);
        }
    }
    let f = polygon_form_factor_quad(pts);
    let amplitude = clamp(params.coeffs1.y, 0.0, 1.0);
    let r = amplitude * f;
    if (is_finite(r)) {
        return max(r, 0.0);
    }
    return 0.0;
}

@compute @workgroup_size(64, 1, 1)
fn area_light_ltc_main(@builtin(global_invocation_id) id: vec3<u32>) {
    if (id.x >= 1u) {
        return;
    }
    let p0 = params.points[0].xyz;
    let p1 = params.points[1].xyz;
    let p2 = params.points[2].xyz;
    let p3 = params.points[3].xyz;

    // Specular: the full `LTC` evaluation under `M-inverse`.
    let specular = ltc_evaluate_quad(p0, p1, p2, p3);
    // Diffuse: the identity-transform clamped-cosine form factor.
    var diffuse_pts: array<vec3<f32>, 4>;
    diffuse_pts[0] = p0;
    diffuse_pts[1] = p1;
    diffuse_pts[2] = p2;
    diffuse_pts[3] = p3;
    let diffuse = polygon_form_factor_quad(diffuse_pts);

    out[0] = specular;
    out[1] = diffuse;
}
