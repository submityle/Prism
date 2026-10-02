//! SDF face-shadow visibility for the stylized (NPR) illumination axis.
//!
//! Cel-shaded anime faces cannot rely on the geometric `N.L` term or shadow
//! maps: the low-poly nose/brow geometry produces ugly self-shadow blobs that
//! swim as the key light moves.  The production technique (Guilty Gear / Genshin
//! / Honkai lineage) instead bakes a *signed distance field* into the face
//! texture whose texel value encodes **at what light angle that pixel flips from
//! lit to shadowed**, then drives the shadow purely from the light's azimuth in
//! the head's own frame.  The result is the hand-authored, art-directed face
//! shadow shape that stays stable under animation.
//!
//! This module is the backend-neutral **golden reference** for that decision
//! math.  It owns:
//!
//! * [`face_shadow_light_cosines`] - projects the light onto the head's
//!   horizontal plane (robust to head tilt) and returns the forward/right
//!   cosines the rest of the technique keys off.
//! * [`face_shadow_flip_u`] - the UV-mirror decision.  The SDF map is authored
//!   for the light on one side of the face and mirrored across the vertical
//!   axis for the other side, so a single map covers both.
//! * [`evaluate_face_shadow`] - turns the resolved SDF sample plus the forward
//!   cosine into a `[0, 1]` visibility term (`1` fully lit, `0` fully shadowed).
//!
//! The texture *sample* itself (and the mirrored UV addressing) is a GPU / pass
//! concern; this reference deliberately takes the already-resolved `sdf` value
//! so the numeric decision stays a pure, testable function whose WESL twin
//! (`face_shadow_*` in `brdf.wesl`) can mirror it arm-for-arm.  The visibility
//! it returns is designed to feed the `visibility` input of
//! [`crate::evaluate_stylized_direct`] (via `stylized_shadow`), so the SDF face
//! shadow composes with the stepped-shadow and cel-ramp stack rather than
//! replacing it.

use crate::vecmath::{dot, mul_scalar, normalize_or, sub};

/// Hermite `smoothstep` with the same collapsed-edge semantics as the rest of
/// the stylized stack: a collapsed or inverted edge pair degrades to a hard
/// step at `edge0` (the GPU builtin is undefined there, so the twin re-derives
/// it explicitly).
#[inline]
fn smoothstep(edge0: f32, edge1: f32, x: f32) -> f32 {
    if edge1 <= edge0 {
        return if x < edge0 { 0.0 } else { 1.0 };
    }
    let t = ((x - edge0) / (edge1 - edge0)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

/// The head's orthonormal frame used to place the SDF face shadow.
///
/// `forward` points out of the face, `right` points to the character's right,
/// and `up` is the head's up axis.  They need not be perfectly orthonormal; the
/// cosine resolver only relies on `up` to define the horizontal plane and on
/// `forward`/`right` as its in-plane basis.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct FaceFrame {
    /// Outward face direction (character's gaze).
    pub forward: [f32; 3],
    /// Character's right-hand direction across the face.
    pub right: [f32; 3],
    /// Head up axis defining the horizontal plane the light is projected onto.
    pub up: [f32; 3],
}

impl Default for FaceFrame {
    /// Identity head frame: gaze down `+Z`, right along `+X`, up along `+Y`.
    fn default() -> Self {
        Self {
            forward: [0.0, 0.0, 1.0],
            right: [1.0, 0.0, 0.0],
            up: [0.0, 1.0, 0.0],
        }
    }
}

/// Tunable controls for the SDF face shadow transition.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct FaceShadowParams {
    /// Half-width of the lit/shadow transition in SDF-threshold space. `0`
    /// gives a hard ink edge; larger values soften the terminator.  Clamped to
    /// `[0, 0.5]` at evaluation.
    pub softness: f32,
}

impl Default for FaceShadowParams {
    /// A hard-edged terminator, matching the crisp look most stylized faces
    /// ship with.
    fn default() -> Self {
        Self { softness: 0.0 }
    }
}

impl FaceShadowParams {
    /// Builds parameters with an explicit transition half-width.
    #[must_use]
    pub const fn with_softness(softness: f32) -> Self {
        Self { softness }
    }
}

/// Projects `light_dir` (direction *towards* the light) onto the head's
/// horizontal plane and returns `(forward_cos, right_cos)`.
///
/// `forward_cos` is `+1` when the light is straight ahead of the face and `-1`
/// when directly behind; `right_cos` is `+1` when the light is off the
/// character's right.  Head tilt is handled by removing the `up` component
/// before normalizing, so the shadow keeps its authored shape as the head
/// rolls.  A light parallel to `up` degrades to `forward_cos == 1`
/// (front-lit / fully lit), which reads correctly for a top-down key.
#[must_use]
pub fn face_shadow_light_cosines(face: FaceFrame, light_dir: [f32; 3]) -> (f32, f32) {
    let up = normalize_or(face.up, [0.0, 1.0, 0.0]);
    let l = normalize_or(light_dir, face.forward);
    // Remove the vertical component so only the azimuth around the head drives
    // the shadow, then renormalize in-plane.
    let planar = sub(l, mul_scalar(up, dot(l, up)));
    let l_h = normalize_or(planar, normalize_or(face.forward, [0.0, 0.0, 1.0]));
    let forward = normalize_or(face.forward, [0.0, 0.0, 1.0]);
    let right = normalize_or(face.right, [1.0, 0.0, 0.0]);
    (dot(l_h, forward), dot(l_h, right))
}

/// Whether the SDF map's horizontal UV must be mirrored for this light side.
///
/// The map is authored for the key light on the character's **right**
/// (`right_cos >= 0`), so it is sampled directly there and mirrored across the
/// vertical axis (`u -> 1 - u`) once the light crosses to the left
/// (`right_cos < 0`).  Keeping this a shared decision guarantees the CPU
/// reference and the GPU sampler pick the same side.
#[must_use]
pub fn face_shadow_flip_u(right_cos: f32) -> bool {
    right_cos < 0.0
}

/// Turns the resolved SDF sample into a `[0, 1]` face-shadow visibility.
///
/// `sdf` is the (already mirror-resolved) face-map texel in `[0, 1]`: larger
/// values stay lit through steeper grazing angles.  `forward_cos` is the
/// forward cosine from [`face_shadow_light_cosines`].  The lit threshold is
/// `(1 - forward_cos) / 2`, so a front light (`forward_cos == 1`) lights every
/// texel and a back light (`forward_cos == -1`) shadows every texel; in between,
/// the authored SDF decides the terminator shape.  `1` is fully lit, `0` fully
/// shadowed.
#[must_use]
pub fn evaluate_face_shadow(sdf: f32, forward_cos: f32, params: FaceShadowParams) -> f32 {
    let threshold = (1.0 - forward_cos.clamp(-1.0, 1.0)) * 0.5;
    let sample = sdf.clamp(0.0, 1.0);
    let half = params.softness.clamp(0.0, 0.5);
    smoothstep(threshold - half, threshold + half, sample)
}

#[cfg(test)]
mod tests {
    use super::*;

    const EPS: f32 = 1.0e-6;

    #[test]
    fn front_light_lights_every_texel() {
        // forward_cos == 1 -> threshold 0 -> any non-negative sdf is lit.
        for &sdf in &[0.0_f32, 0.25, 0.5, 1.0] {
            assert_eq!(
                evaluate_face_shadow(sdf, 1.0, FaceShadowParams::default()),
                1.0
            );
        }
    }

    #[test]
    fn back_light_shadows_every_interior_texel() {
        // forward_cos == -1 -> threshold 1 -> hard step: only sdf >= 1 stays lit.
        assert_eq!(
            evaluate_face_shadow(0.99, -1.0, FaceShadowParams::default()),
            0.0
        );
        // At the exact boundary the hard step includes the edge.
        assert_eq!(
            evaluate_face_shadow(1.0, -1.0, FaceShadowParams::default()),
            1.0
        );
    }

    #[test]
    fn threshold_is_monotonic_in_forward_cos() {
        // A fixed mid SDF flips from lit to shadowed as the light swings back.
        let sdf = 0.5;
        let front = evaluate_face_shadow(sdf, 0.9, FaceShadowParams::default());
        let side = evaluate_face_shadow(sdf, 0.0, FaceShadowParams::default());
        let back = evaluate_face_shadow(sdf, -0.9, FaceShadowParams::default());
        assert_eq!(front, 1.0);
        // threshold at forward_cos 0 is exactly 0.5; hard step includes the edge.
        assert_eq!(side, 1.0);
        assert_eq!(back, 0.0);
    }

    #[test]
    fn softness_produces_partial_visibility_at_the_terminator() {
        // threshold 0.5, sdf exactly on it -> smoothstep midpoint 0.5.
        let lit = evaluate_face_shadow(0.5, 0.0, FaceShadowParams::with_softness(0.2));
        assert!(
            (lit - 0.5).abs() < EPS,
            "expected 0.5 at terminator, got {lit}"
        );
        // Just inside the lit band.
        let brighter = evaluate_face_shadow(0.55, 0.0, FaceShadowParams::with_softness(0.2));
        assert!(brighter > lit);
    }

    #[test]
    fn cosines_identity_frame_front_light() {
        let (fwd, right) = face_shadow_light_cosines(FaceFrame::default(), [0.0, 0.0, 1.0]);
        assert!((fwd - 1.0).abs() < EPS);
        assert!(right.abs() < EPS);
    }

    #[test]
    fn cosines_side_light_sets_right_axis() {
        let (fwd, right) = face_shadow_light_cosines(FaceFrame::default(), [1.0, 0.0, 0.0]);
        assert!(fwd.abs() < EPS);
        assert!((right - 1.0).abs() < EPS);
        assert!(
            !face_shadow_flip_u(right),
            "light on the right samples directly"
        );
        let (_, left) = face_shadow_light_cosines(FaceFrame::default(), [-1.0, 0.0, 0.0]);
        assert!(
            face_shadow_flip_u(left),
            "light on the left mirrors the map"
        );
    }

    #[test]
    fn head_tilt_projects_out_the_up_component() {
        // Head rolled 90deg: up now points along +X. A light straight along the
        // world +Z (still the gaze) must remain a pure front light.
        let tilted = FaceFrame {
            forward: [0.0, 0.0, 1.0],
            right: [0.0, 1.0, 0.0],
            up: [1.0, 0.0, 0.0],
        };
        let (fwd, right) = face_shadow_light_cosines(tilted, [0.0, 0.0, 1.0]);
        assert!((fwd - 1.0).abs() < EPS);
        assert!(right.abs() < EPS);
    }

    #[test]
    fn light_along_up_axis_degrades_to_front_lit() {
        let (fwd, _right) = face_shadow_light_cosines(FaceFrame::default(), [0.0, 1.0, 0.0]);
        assert!((fwd - 1.0).abs() < EPS);
    }
}
