// GPU twin of `prism_render_shading::face_shadow` (Genshin / Guilty Gear-style
// SDF face-shadow visibility for the stylized NPR axis).
// Self-contained (no intra-crate imports, matching ssgi.wesl / bloom.wesl /
// color_grade.wesl / vignette.wesl) so the WESL compile test can validate it
// standalone. Every helper is `face_shadow_`-prefixed and mirrors the CPU
// golden arm-for-arm: same function split, same constants, same operation
// order. The golden's `if` guards are rewritten with `select`, the `bool`
// return is a `select` over the same comparison, and `normalize_or` re-derives
// its degenerate guard. smoothstep is written `t*t*(3 - 2t)` with the same
// collapsed-edge hard-step semantics. `dot`/`sqrt`/`clamp`/`select` are WGSL
// builtins used exactly as the CPU golden's `vecmath` helpers are.

// The head's orthonormal frame used to place the SDF face shadow. Mirrors the
// CPU `FaceFrame`: `forward` out of the face, `right` across it, `up` the head
// axis defining the horizontal plane the light is projected onto.
struct FaceFrame {
    forward: vec3<f32>,
    right: vec3<f32>,
    up: vec3<f32>,
}

// Tunable controls for the SDF face shadow transition. Mirrors the CPU
// `FaceShadowParams`: `softness` is the transition half-width in threshold space.
struct FaceShadowParams {
    softness: f32,
}

// Normalizes `value`, returning `fallback` for degenerate/non-finite inputs.
// Mirrors `vecmath::normalize_or`: guard is `lsq > 1e-12 && lsq finite`, and the
// scale is `1 / sqrt(lsq)` (the golden's `sqrt().recip()`), chosen with select.
fn face_shadow_normalize_or(value: vec3<f32>, fallback: vec3<f32>) -> vec3<f32> {
    let length_squared = dot(value, value);
    let ok = length_squared > 1.0e-12 && length_squared < 3.402823466e38;
    return select(fallback, value * (1.0 / sqrt(length_squared)), ok);
}

// Hermite smoothstep with the collapsed-edge semantics of the stylized stack: a
// collapsed or inverted edge pair (edge1 <= edge0) degrades to a hard step at
// edge0 (0 for x < edge0, else 1); otherwise `t*t*(3 - 2t)` over the clamped t.
fn face_shadow_smoothstep(edge0: f32, edge1: f32, x: f32) -> f32 {
    let hard = select(1.0, 0.0, x < edge0);
    let t = clamp((x - edge0) / (edge1 - edge0), 0.0, 1.0);
    let soft = t * t * (3.0 - 2.0 * t);
    return select(soft, hard, edge1 <= edge0);
}

// Projects `light_dir` (towards the light) onto the head's horizontal plane and
// returns (forward_cos, right_cos) as a vec2. Mirrors the golden arm-for-arm:
// remove the up component, renormalize in-plane, then dot against forward/right.
fn face_shadow_light_cosines(face: FaceFrame, light_dir: vec3<f32>) -> vec2<f32> {
    let up = face_shadow_normalize_or(face.up, vec3<f32>(0.0, 1.0, 0.0));
    let l = face_shadow_normalize_or(light_dir, face.forward);
    let planar = l - up * dot(l, up);
    let l_h = face_shadow_normalize_or(
        planar,
        face_shadow_normalize_or(face.forward, vec3<f32>(0.0, 0.0, 1.0)),
    );
    let forward = face_shadow_normalize_or(face.forward, vec3<f32>(0.0, 0.0, 1.0));
    let right = face_shadow_normalize_or(face.right, vec3<f32>(1.0, 0.0, 0.0));
    return vec2<f32>(dot(l_h, forward), dot(l_h, right));
}

// Whether the SDF map's horizontal UV must be mirrored for this light side. The
// map is authored for the light on the character's right (right_cos >= 0), so
// it mirrors once the light crosses to the left (right_cos < 0).
fn face_shadow_flip_u(right_cos: f32) -> bool {
    return select(false, true, right_cos < 0.0);
}

// Turns the resolved SDF sample into a [0, 1] face-shadow visibility. Threshold
// is (1 - clamp(forward_cos, -1, 1)) / 2; the clamped sdf sample is smoothstep'd
// across [threshold - half, threshold + half] with half = clamp(softness,0,0.5).
fn face_shadow_evaluate(sdf: f32, forward_cos: f32, params: FaceShadowParams) -> f32 {
    let threshold = (1.0 - clamp(forward_cos, -1.0, 1.0)) * 0.5;
    let sample = clamp(sdf, 0.0, 1.0);
    let half_width = clamp(params.softness, 0.0, 0.5);
    return face_shadow_smoothstep(threshold - half_width, threshold + half_width, sample);
}

// Minimal entry so the module type-checks as a standalone compute kernel under
// the WESL compile test; the real face-shadow pass wires the head frame, the
// resolved SDF sample and the params uniform later. References every helper
// (and each struct field) once with fixed literal inputs.
@compute @workgroup_size(64, 1, 1)
fn face_shadow_probe(@builtin(global_invocation_id) gid: vec3<u32>) {
    let seed = f32(gid.x) * 0.001;

    var frame: FaceFrame;
    frame.forward = vec3<f32>(0.0, 0.0, 1.0);
    frame.right = vec3<f32>(1.0, 0.0, 0.0);
    frame.up = vec3<f32>(0.0, 1.0, 0.0);

    var params: FaceShadowParams;
    params.softness = 0.2;

    let light_dir = vec3<f32>(1.0 + seed, 0.0, 1.0);

    let norm = face_shadow_normalize_or(frame.forward, vec3<f32>(0.0, 0.0, 1.0));
    let ss = face_shadow_smoothstep(0.3, 0.7, 0.5);
    let cosines = face_shadow_light_cosines(frame, light_dir);
    let flip = face_shadow_flip_u(cosines.y);
    let flip_term = select(0.0, 1.0, flip);
    let vis = face_shadow_evaluate(0.5, cosines.x, params);

    // Reference each struct field so none drifts unused.
    let field_sum = frame.forward.z + frame.right.x + frame.up.y + params.softness;

    _ = norm.z + ss + cosines.x + cosines.y + flip_term + vis + field_sum;
}
