//! Per-particle motion-vector computation and encoding for temporal
//! integration (design §21).
//!
//! Where [`super::shading`] owns the *requirement decision* layer — whether a
//! renderer should write motion vectors at all, whether fast `flipbook`
//! animation makes it temporally unstable, and the coarse reactive-mask weight
//! (see [`super::shading::MotionVectorRequest`]) — this module owns the
//! *compute and encode* layer that the `GPU` draw kernel evaluates per particle:
//!
//! 1. **Clip-space delta** — project the current and previous world positions
//!    through the current and previous view-projection matrices, apply the
//!    perspective divide, and take the `NDC`-to-`UV` difference. This is the
//!    same screen-space motion vector that `TAA` / temporal super-resolution
//!    (`TSR`) and `DLSS`-style upsamplers consume, and it mirrors Unreal
//!    `Niagara` and Unity `VFX Graph` velocity output at the algorithm level
//!    without reusing any of their code.
//! 2. **Previous-frame tracking** — [`PrevParticleState`] persists the prior
//!    frame's world position (design §5.2) so a pooled particle carries the
//!    history the reprojection needs, and marks freshly spawned particles as
//!    having no valid history.
//! 3. **`Flipbook` temporal stability** — [`flipbook_stability`] turns a
//!    per-frame `flipbook` / `UV`-animation advance into a temporally-unstable
//!    flag plus a suggested history-weight reduction, so high-frequency
//!    sub-image changes stop the upsampler from smearing stale frames.
//! 4. **Reactive mask** — [`reactive_history_weight`] consumes the upstream
//!    [`super::shading::MotionVectorRequest`] (it never redefines it) and turns
//!    it into the concrete history-blend weight the temporal upsampler applies.
//! 5. **Bandwidth-friendly encoding** — [`encode_motion_f16`] /
//!    [`encode_motion_snorm16`] pack the two-channel screen-space vector into
//!    `fp16` or range-relative fixed point (design §27 style), each with an
//!    exact decode inverse for round-trip verification.
//! 6. **`OIT` co-operation** — [`oit_motion_contract`] documents how a
//!    translucent particle routed through the shared order-independent
//!    transparency (`OIT`) path contributes to the motion-vector buffer used
//!    for history reprojection (design §12).
//!
//! Everything here is `CPU`-verifiable and built from ordinary arithmetic plus
//! `sqrt` only — no transcendental functions — so a future `GPU` compute kernel
//! reproduces the same motion vectors bit for bit. The contract applies equally
//! to the `Sprite`, `Mesh`, `Ribbon`, and `Beam` surface renderers
//! (see [`renderer_writes_motion_vectors`]).

use alloc::vec::Vec;

use super::compression::{f16_bits_to_f32, f32_to_f16_bits, snorm16_decode, snorm16_encode};
use super::renderers::RendererKind;
use super::shading::MotionVectorRequest;
use super::Vec3;

/// Minimum homogeneous `w` for which the perspective divide is well defined.
///
/// A clip-space position with `w` at or below this bound is at or behind the
/// camera plane; dividing by it would explode or flip the sign, so such a
/// position yields no valid `NDC` coordinate.
pub const EPS_W: f32 = 1e-6;

/// Smallest usable range for range-relative fixed-point encoding.
///
/// A non-positive or vanishing range would divide by zero when normalizing a
/// motion vector, so [`encode_motion_snorm16`] guards against it.
pub const EPS_RANGE: f32 = 1e-12;

/// A hand-rolled two-component vector used for screen-space (`UV`) quantities.
///
/// The shared [`Vec3`] is three-dimensional; motion vectors live in the
/// two-dimensional screen/`UV` plane, so this module carries its own minimal
/// `2D` type. Like [`Vec3`] it uses only add/subtract/multiply/divide and never
/// a transcendental function, and it derives only [`PartialEq`] (no `Eq`/`Hash`)
/// because it holds `f32` fields.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Vec2 {
    /// X (horizontal / `U`) component.
    pub x: f32,
    /// Y (vertical / `V`) component.
    pub y: f32,
}

impl Vec2 {
    /// The zero vector.
    pub const ZERO: Self = Self { x: 0.0, y: 0.0 };

    /// Builds a vector from components.
    #[must_use]
    pub const fn new(x: f32, y: f32) -> Self {
        Self { x, y }
    }

    /// Component-wise sum.
    #[must_use]
    #[expect(
        clippy::should_implement_trait,
        reason = "The particle math API uses named add/sub methods for call-site uniformity, matching the sibling Vec3 type; operator traits are intentionally not part of this internal type."
    )]
    pub fn add(self, rhs: Self) -> Self {
        Self::new(self.x + rhs.x, self.y + rhs.y)
    }

    /// Component-wise difference `self - rhs`.
    #[must_use]
    #[expect(
        clippy::should_implement_trait,
        reason = "See add: the specified API uses named sub for call-site uniformity, not operator traits."
    )]
    pub fn sub(self, rhs: Self) -> Self {
        Self::new(self.x - rhs.x, self.y - rhs.y)
    }

    /// Uniform scale by a scalar.
    #[must_use]
    pub fn scale(self, s: f32) -> Self {
        Self::new(self.x * s, self.y * s)
    }

    /// Squared Euclidean length; cheaper than [`Vec2::length`] for comparisons.
    #[must_use]
    pub fn length_squared(self) -> f32 {
        self.x * self.x + self.y * self.y
    }

    /// Euclidean length (uses `sqrt`, the only permitted non-arithmetic op).
    #[must_use]
    pub fn length(self) -> f32 {
        self.length_squared().sqrt()
    }
}

/// A minimal column-major `4x4` matrix, used as a view-projection transform.
///
/// Only the operations this module needs are provided (identity and point
/// transform); it is a self-contained arithmetic type so the crate keeps its
/// zero-dependency contract. Columns are stored as `cols[column][row]`, the
/// convention most `GPU` shading languages use. It derives only [`PartialEq`]
/// (no `Eq`/`Hash`) because it holds `f32` fields.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Mat4 {
    /// The four columns, each a four-element `[row]` array (`cols[col][row]`).
    pub cols: [[f32; 4]; 4],
}

impl Mat4 {
    /// The identity transform (maps every point to itself with `w == 1`).
    pub const IDENTITY: Self = Self {
        cols: [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
    };

    /// Builds a matrix from its four columns (`cols[col][row]`).
    #[must_use]
    pub const fn from_cols(cols: [[f32; 4]; 4]) -> Self {
        Self { cols }
    }

    /// Transforms a world-space point (treated as homogeneous `(x, y, z, 1)`)
    /// into clip space, returning the un-divided [`ClipPos`].
    #[must_use]
    pub fn transform_point(self, p: Vec3) -> ClipPos {
        let m = self.cols;
        ClipPos {
            x: m[0][0] * p.x + m[1][0] * p.y + m[2][0] * p.z + m[3][0],
            y: m[0][1] * p.x + m[1][1] * p.y + m[2][1] * p.z + m[3][1],
            z: m[0][2] * p.x + m[1][2] * p.y + m[2][2] * p.z + m[3][2],
            w: m[0][3] * p.x + m[1][3] * p.y + m[2][3] * p.z + m[3][3],
        }
    }
}

/// A homogeneous clip-space position (before the perspective divide).
///
/// Derives only [`PartialEq`] (no `Eq`/`Hash`) because it holds `f32` fields.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ClipPos {
    /// Clip-space `X`.
    pub x: f32,
    /// Clip-space `Y`.
    pub y: f32,
    /// Clip-space `Z` (depth).
    pub z: f32,
    /// Homogeneous `W`; the perspective divisor.
    pub w: f32,
}

impl ClipPos {
    /// Returns `true` when the point is in front of the camera plane, i.e. its
    /// `w` exceeds [`EPS_W`] and the perspective divide is well defined.
    #[must_use]
    pub fn is_visible(self) -> bool {
        self.w > EPS_W
    }

    /// Applies the perspective divide, returning the normalized-device
    /// coordinate (`NDC`) in the canonical `-1..=1` cube, or [`None`] when the
    /// point is at or behind the camera plane (see [`ClipPos::is_visible`]).
    #[must_use]
    pub fn to_ndc(self) -> Option<Vec3> {
        if !self.is_visible() {
            return None;
        }
        let inv_w = 1.0 / self.w;
        Some(Vec3::new(self.x * inv_w, self.y * inv_w, self.z * inv_w))
    }
}

/// The `NDC`-to-`UV` mapping convention of the target framebuffer (design §21).
///
/// Screen-space `X` always maps `NDC` `-1..=1` to `UV` `0..=1`, but the `Y`
/// axis differs by graphics `API`: some frameworks place the `UV` origin at the
/// top-left with `V` growing downward, others at the bottom-left with `V`
/// growing upward. The motion-vector `UV` delta must use the same convention as
/// the history buffer it reprojects into.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum NdcConvention {
    /// `UV` origin at the top-left, `V` grows downward (`DirectX` / `Vulkan`
    /// framebuffer style).
    TopLeftYDown,
    /// `UV` origin at the bottom-left, `V` grows upward (`OpenGL` style).
    BottomLeftYUp,
}

impl NdcConvention {
    /// Maps an `NDC` position (components in `-1..=1`) to a `UV` position
    /// (components in `0..=1`) under this convention.
    #[must_use]
    pub fn ndc_to_uv(self, ndc: Vec2) -> Vec2 {
        let u = ndc.x * 0.5 + 0.5;
        let v = match self {
            NdcConvention::TopLeftYDown => -ndc.y * 0.5 + 0.5,
            NdcConvention::BottomLeftYUp => ndc.y * 0.5 + 0.5,
        };
        Vec2::new(u, v)
    }
}

/// The camera state needed to reproject a particle between two frames.
///
/// Holds the current and previous view-projection matrices and the sub-pixel
/// `TAA` jitter offsets (in `NDC` units) that were baked into each of those
/// matrices when the frame was rasterized. Because the rasterizing matrices
/// carry the jitter, the geometric (jitter-free) position is recovered by
/// subtracting the jitter after the perspective divide; this keeps the motion
/// vector jitter-free so the upsampler reprojects onto the un-jittered pixel
/// grid. Derives only [`PartialEq`] (no `Eq`/`Hash`) because it holds `f32`
/// fields via its matrices and jitter vectors.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CameraMotionState {
    /// This frame's (jittered) view-projection matrix.
    pub cur_view_proj: Mat4,
    /// Last frame's (jittered) view-projection matrix (design §5.2).
    pub prev_view_proj: Mat4,
    /// This frame's `NDC`-space jitter offset baked into `cur_view_proj`.
    pub cur_jitter: Vec2,
    /// Last frame's `NDC`-space jitter offset baked into `prev_view_proj`.
    pub prev_jitter: Vec2,
    /// The framebuffer `NDC`-to-`UV` convention (see [`NdcConvention`]).
    pub convention: NdcConvention,
}

/// The previous-frame tracking state carried by one pooled particle (§5.2).
///
/// A pooled particle must remember where it was last frame so the reprojection
/// has a history sample. A particle spawned this frame has no valid history and
/// its motion vector is undefined; [`PrevParticleState::valid`] records that.
/// Derives only [`PartialEq`] (no `Eq`/`Hash`) because of its `f32` position.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct PrevParticleState {
    /// The particle's world-space position at the end of the previous frame.
    pub world_position: Vec3,
    /// Whether `world_position` is a real previous-frame sample. `false` on the
    /// frame the particle spawns, when no history exists yet.
    pub valid: bool,
}

impl PrevParticleState {
    /// The state for a particle spawned this frame: no valid history, seeded
    /// with its spawn position so next frame's tracking starts from there.
    #[must_use]
    pub fn spawned(world_position: Vec3) -> Self {
        Self {
            world_position,
            valid: false,
        }
    }

    /// Records `world_position` as this frame's position, producing the valid
    /// previous-frame state to store for next frame's reprojection.
    #[must_use]
    pub fn record(world_position: Vec3) -> Self {
        Self {
            world_position,
            valid: true,
        }
    }
}

/// A per-particle screen-space motion vector: the `UV`-space displacement of a
/// particle from its previous-frame to its current-frame screen position.
///
/// Derives only [`PartialEq`] (no `Eq`/`Hash`) because of its `f32` fields.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct ScreenMotionVector {
    /// `current_uv - previous_uv`, jitter-removed, in `UV` units.
    pub uv_delta: Vec2,
}

impl ScreenMotionVector {
    /// A zero motion vector (a perfectly static particle).
    pub const ZERO: Self = Self {
        uv_delta: Vec2::ZERO,
    };

    /// The `UV`-space magnitude of the displacement.
    #[must_use]
    pub fn magnitude(self) -> f32 {
        self.uv_delta.length()
    }
}

/// Computes one particle's jitter-free screen-space motion vector (design §21).
///
/// Projects `cur_world` through the current matrix and `prev.world_position`
/// through the previous matrix, applies the perspective divide, removes the
/// respective frame jitter, converts both to `UV`, and returns their
/// difference. Returns [`None`] when the particle has no valid history (spawned
/// this frame) or when either projected point falls at or behind the camera
/// plane; in that case the temporal upsampler must reject history for the pixel
/// rather than reproject with an undefined vector.
#[must_use]
pub fn screen_motion_vector(
    camera: CameraMotionState,
    cur_world: Vec3,
    prev: PrevParticleState,
) -> Option<ScreenMotionVector> {
    if !prev.valid {
        return None;
    }
    let cur_ndc = camera.cur_view_proj.transform_point(cur_world).to_ndc()?;
    let prev_ndc = camera
        .prev_view_proj
        .transform_point(prev.world_position)
        .to_ndc()?;

    let cur_unjittered = Vec2::new(cur_ndc.x, cur_ndc.y).sub(camera.cur_jitter);
    let prev_unjittered = Vec2::new(prev_ndc.x, prev_ndc.y).sub(camera.prev_jitter);

    let cur_uv = camera.convention.ndc_to_uv(cur_unjittered);
    let prev_uv = camera.convention.ndc_to_uv(prev_unjittered);

    Some(ScreenMotionVector {
        uv_delta: cur_uv.sub(prev_uv),
    })
}

/// Batch form of [`screen_motion_vector`] over parallel current-position and
/// previous-state slices.
///
/// The two slices are zipped elementwise (extra trailing entries in the longer
/// slice are ignored), producing one [`Option`] per particle with the same
/// semantics as the scalar function. This mirrors the wide, per-lane evaluation
/// the `GPU` kernel performs.
#[must_use]
pub fn screen_motion_vectors(
    camera: CameraMotionState,
    cur_worlds: &[Vec3],
    prev_states: &[PrevParticleState],
) -> Vec<Option<ScreenMotionVector>> {
    cur_worlds
        .iter()
        .zip(prev_states.iter())
        .map(|(&cur_world, &prev)| screen_motion_vector(camera, cur_world, prev))
        .collect()
}

/// The temporal-stability verdict for a `flipbook` / `UV` animation (design §21).
///
/// Derives only [`PartialEq`] (no `Eq`/`Hash`) because of its `f32` field.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct FlipbookStability {
    /// Whether the per-frame `flipbook` advance is fast enough to reject
    /// history (the sub-image changes faster than the upsampler can track).
    pub temporally_unstable: bool,
    /// Suggested history-blend weight in `0..=1`: `1` keeps all history, `0`
    /// rejects it. Falls linearly as the advance approaches the threshold.
    pub history_weight: f32,
}

/// The absolute number of `flipbook` frames advanced between two render frames.
///
/// Fractional frames are supported (sub-frame `UV`-animation phase), and the
/// result is always non-negative regardless of animation direction.
#[must_use]
pub fn flipbook_frames_advanced(prev_frame: f32, cur_frame: f32) -> f32 {
    (cur_frame - prev_frame).abs()
}

/// Classifies a per-frame `flipbook` advance into a [`FlipbookStability`].
///
/// `frames_advanced` is the count from [`flipbook_frames_advanced`] and
/// `unstable_threshold` is the advance at or above which the animation is
/// declared temporally unstable. The suggested history weight ramps linearly
/// from `1` (no advance) down to `0` (advance at or beyond the threshold). A
/// non-positive threshold disables the signal, reporting a stable animation
/// with full history so a mis-configured effect never spuriously rejects it.
#[must_use]
pub fn flipbook_stability(frames_advanced: f32, unstable_threshold: f32) -> FlipbookStability {
    if unstable_threshold <= 0.0 {
        return FlipbookStability {
            temporally_unstable: false,
            history_weight: 1.0,
        };
    }
    let ratio = (frames_advanced / unstable_threshold).clamp(0.0, 1.0);
    FlipbookStability {
        temporally_unstable: frames_advanced >= unstable_threshold,
        history_weight: 1.0 - ratio,
    }
}

/// The concrete reactive-mask value and history-blend weight for one particle.
///
/// Derives only [`PartialEq`] (no `Eq`/`Hash`) because of its `f32` fields.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct HistoryWeight {
    /// The reactive-mask value in `0..=1`: higher means less history reuse.
    pub reactive: f32,
    /// The history-blend weight in `0..=1`, defined as `1 - reactive`: higher
    /// means more history reuse (the upsampler trusts the reprojected sample).
    pub history_weight: f32,
}

/// Turns an upstream [`MotionVectorRequest`] plus this particle's computed
/// motion into the concrete history-blend weight the temporal upsampler applies
/// (design §21).
///
/// This consumes the shading layer's decision rather than redefining it: the
/// request's coarse `reactive_mask` is the baseline, a temporally-unstable
/// `flipbook` saturates the reactive value, and a missing motion vector (no
/// history, or projected off-screen; see [`screen_motion_vector`]) also forces
/// full reactivity so the upsampler drops the untrusted history.
#[must_use]
pub fn reactive_history_weight(
    request: MotionVectorRequest,
    motion: Option<ScreenMotionVector>,
) -> HistoryWeight {
    let mut reactive = request.reactive_mask.clamp(0.0, 1.0);
    if request.temporally_unstable {
        reactive = 1.0;
    }
    if motion.is_none() {
        reactive = 1.0;
    }
    HistoryWeight {
        reactive,
        history_weight: 1.0 - reactive,
    }
}

/// The two-channel `fp16`-packed screen-space motion vector (design §27 style).
///
/// `fp16` gives ample precision for the small `UV` deltas typical of a
/// motion-vector buffer while halving the bandwidth of an `f32` pair. Derives
/// only [`PartialEq`] here for symmetry with the float types; the packed bits
/// themselves are exact `u16` values.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct EncodedMotionVectorF16 {
    /// The `X` (`U`) channel as an `fp16` bit pattern.
    pub x: u16,
    /// The `Y` (`V`) channel as an `fp16` bit pattern.
    pub y: u16,
}

/// Packs a screen-space motion vector into two `fp16` channels.
#[must_use]
pub fn encode_motion_f16(mv: ScreenMotionVector) -> EncodedMotionVectorF16 {
    EncodedMotionVectorF16 {
        x: f32_to_f16_bits(mv.uv_delta.x),
        y: f32_to_f16_bits(mv.uv_delta.y),
    }
}

/// Unpacks two `fp16` channels back into a screen-space motion vector; the
/// exact inverse of [`encode_motion_f16`] on every finite half value.
#[must_use]
pub fn decode_motion_f16(enc: EncodedMotionVectorF16) -> ScreenMotionVector {
    ScreenMotionVector {
        uv_delta: Vec2::new(f16_bits_to_f32(enc.x), f16_bits_to_f32(enc.y)),
    }
}

/// The two-channel range-relative fixed-point (`snorm16`) motion vector.
///
/// When the maximum expected `UV` displacement is known, encoding relative to
/// that range in `snorm16` gives uniform precision across the whole range at
/// the same `4`-byte footprint as [`EncodedMotionVectorF16`]. The packed codes
/// are exact `i16` values.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct EncodedMotionVectorSnorm16 {
    /// The `X` (`U`) channel as an `snorm16` code.
    pub x: i16,
    /// The `Y` (`V`) channel as an `snorm16` code.
    pub y: i16,
}

/// Packs a screen-space motion vector into two range-relative `snorm16`
/// channels, normalizing each component by `max_uv`.
///
/// Components outside `-max_uv..=max_uv` clamp to the range endpoints (the
/// underlying `snorm16` codec saturates). A non-positive or vanishing `max_uv`
/// (see [`EPS_RANGE`]) is degenerate and encodes to zero rather than dividing
/// by zero.
#[must_use]
pub fn encode_motion_snorm16(mv: ScreenMotionVector, max_uv: f32) -> EncodedMotionVectorSnorm16 {
    if max_uv <= EPS_RANGE {
        return EncodedMotionVectorSnorm16 { x: 0, y: 0 };
    }
    let inv = 1.0 / max_uv;
    EncodedMotionVectorSnorm16 {
        x: snorm16_encode(mv.uv_delta.x * inv),
        y: snorm16_encode(mv.uv_delta.y * inv),
    }
}

/// Unpacks two range-relative `snorm16` channels back into a screen-space
/// motion vector, scaling each component by `max_uv`; the inverse of
/// [`encode_motion_snorm16`] up to the `snorm16` quantization step.
#[must_use]
pub fn decode_motion_snorm16(enc: EncodedMotionVectorSnorm16, max_uv: f32) -> ScreenMotionVector {
    ScreenMotionVector {
        uv_delta: Vec2::new(
            snorm16_decode(enc.x) * max_uv,
            snorm16_decode(enc.y) * max_uv,
        ),
    }
}

/// The contract for how a translucent particle contributes to the shared
/// motion-vector buffer used for history reprojection (design §12, §21).
///
/// Derives only [`PartialEq`] (no `Eq`/`Hash`) because of its `f32` field.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct OitMotionContract {
    /// Whether this fragment writes into the shared motion-vector buffer. A
    /// blended particle routed through the shared `OIT` path has many
    /// overlapping layers, but the motion-vector buffer stores one vector per
    /// pixel, so only the front-most layer contributes; deeper layers would
    /// smear the reprojection.
    pub contributes_motion_vector: bool,
    /// The history-blend weight in `0..=1` for the pixel this fragment covers
    /// (`1 - reactive`): translucent and temporally-unstable particles keep
    /// less history to avoid ghosting.
    pub history_weight: f32,
}

/// Resolves the [`OitMotionContract`] for a particle from its upstream
/// [`MotionVectorRequest`] and whether it is the front-most translucent layer.
///
/// Only a particle that both writes motion vectors and is the front-most layer
/// contributes to the shared buffer; a temporally-unstable request saturates
/// reactivity (dropping history), otherwise the request's `reactive_mask` sets
/// the retained history weight.
#[must_use]
pub fn oit_motion_contract(request: MotionVectorRequest, front_most: bool) -> OitMotionContract {
    let contributes_motion_vector = request.write_motion_vectors && front_most;
    let reactive = if request.temporally_unstable {
        1.0
    } else {
        request.reactive_mask.clamp(0.0, 1.0)
    };
    OitMotionContract {
        contributes_motion_vector,
        history_weight: 1.0 - reactive,
    }
}

/// Whether a given renderer primitive writes per-particle motion vectors.
///
/// The four surface renderers — `Sprite`, `Mesh`, `Ribbon`, and `Beam` — draw
/// screen-covering geometry whose per-particle displacement must feed `TAA` /
/// `TSR` reprojection, so they write motion vectors. `Light` emits no geometry;
/// `Decal` and `Volume` are reprojected with a screen-space / camera-only
/// motion vector handled by their own paths, not a per-particle one.
#[must_use]
pub fn renderer_writes_motion_vectors(kind: RendererKind) -> bool {
    match kind {
        RendererKind::Sprite | RendererKind::Mesh | RendererKind::Ribbon | RendererKind::Beam => {
            true
        }
        RendererKind::Light | RendererKind::Decal | RendererKind::Volume => false,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Absolute-tolerance comparison, since bare `f32` equality is forbidden.
    fn approx(a: f32, b: f32, eps: f32) -> bool {
        (a - b).abs() <= eps
    }

    /// Builds an identity matrix with an `NDC` translation baked in, used to
    /// simulate a jittered rasterizing view-projection in the tests.
    fn identity_with_translation(tx: f32, ty: f32) -> Mat4 {
        let mut m = Mat4::IDENTITY;
        m.cols[3][0] = tx;
        m.cols[3][1] = ty;
        m
    }

    #[test]
    fn vec2_arithmetic_round_trips() {
        let a = Vec2::new(1.0, -2.0);
        let b = Vec2::new(0.5, 0.5);
        assert!(approx(a.add(b).x, 1.5, 1e-6));
        assert!(approx(a.sub(b).y, -2.5, 1e-6));
        assert!(approx(a.scale(2.0).x, 2.0, 1e-6));
        assert!(approx(Vec2::new(3.0, 4.0).length(), 5.0, 1e-6));
    }

    #[test]
    fn mat4_identity_transform_is_point_with_unit_w() {
        let c = Mat4::IDENTITY.transform_point(Vec3::new(0.3, -0.4, 0.5));
        assert!(approx(c.x, 0.3, 1e-6));
        assert!(approx(c.y, -0.4, 1e-6));
        assert!(approx(c.z, 0.5, 1e-6));
        assert!(approx(c.w, 1.0, 1e-6));
    }

    #[test]
    fn clip_perspective_divide_matches_manual() {
        let clip = ClipPos {
            x: 1.0,
            y: 2.0,
            z: 3.0,
            w: 2.0,
        };
        let ndc = clip.to_ndc().expect("w > eps is visible");
        assert!(approx(ndc.x, 0.5, 1e-6));
        assert!(approx(ndc.y, 1.0, 1e-6));
        assert!(approx(ndc.z, 1.5, 1e-6));
    }

    #[test]
    fn clip_behind_camera_has_no_ndc() {
        let behind = ClipPos {
            x: 1.0,
            y: 1.0,
            z: 1.0,
            w: 0.0,
        };
        assert!(!behind.is_visible());
        assert!(behind.to_ndc().is_none());
        let negative = ClipPos {
            x: 1.0,
            y: 1.0,
            z: 1.0,
            w: -0.5,
        };
        assert!(negative.to_ndc().is_none());
    }

    #[test]
    fn ndc_to_uv_flips_y_by_convention() {
        let top = NdcConvention::TopLeftYDown.ndc_to_uv(Vec2::new(0.0, 1.0));
        assert!(approx(top.x, 0.5, 1e-6));
        assert!(approx(top.y, 0.0, 1e-6));
        let bottom = NdcConvention::BottomLeftYUp.ndc_to_uv(Vec2::new(0.0, 1.0));
        assert!(approx(bottom.y, 1.0, 1e-6));
    }

    fn camera_no_jitter() -> CameraMotionState {
        CameraMotionState {
            cur_view_proj: Mat4::IDENTITY,
            prev_view_proj: Mat4::IDENTITY,
            cur_jitter: Vec2::ZERO,
            prev_jitter: Vec2::ZERO,
            convention: NdcConvention::BottomLeftYUp,
        }
    }

    #[test]
    fn static_particle_has_zero_motion() {
        let camera = camera_no_jitter();
        let world = Vec3::new(0.25, -0.1, 0.0);
        let mv = screen_motion_vector(camera, world, PrevParticleState::record(world))
            .expect("static particle in front of camera has a motion vector");
        assert!(mv.magnitude() <= 1e-6);
    }

    #[test]
    fn moving_particle_has_expected_delta() {
        let camera = camera_no_jitter();
        let prev = PrevParticleState::record(Vec3::new(0.2, 0.1, 0.0));
        let mv =
            screen_motion_vector(camera, Vec3::new(0.4, 0.1, 0.0), prev).expect("visible particle");
        // `NDC` x moves 0.2 -> `UV` moves 0.1 (half scale); v unchanged.
        assert!(approx(mv.uv_delta.x, 0.1, 1e-6));
        assert!(approx(mv.uv_delta.y, 0.0, 1e-6));
    }

    #[test]
    fn newly_spawned_particle_has_no_motion_vector() {
        let camera = camera_no_jitter();
        let spawn = PrevParticleState::spawned(Vec3::new(0.1, 0.2, 0.0));
        assert!(!spawn.valid);
        assert!(screen_motion_vector(camera, Vec3::new(0.1, 0.2, 0.0), spawn).is_none());
    }

    #[test]
    fn behind_camera_particle_has_no_motion_vector() {
        // A projection matrix whose w output equals -1 for any point.
        let mut behind = Mat4::IDENTITY;
        behind.cols[3][3] = -1.0;
        let camera = CameraMotionState {
            cur_view_proj: behind,
            prev_view_proj: Mat4::IDENTITY,
            cur_jitter: Vec2::ZERO,
            prev_jitter: Vec2::ZERO,
            convention: NdcConvention::BottomLeftYUp,
        };
        let prev = PrevParticleState::record(Vec3::new(0.0, 0.0, 0.0));
        assert!(screen_motion_vector(camera, Vec3::new(0.1, 0.0, 0.0), prev).is_none());
    }

    #[test]
    fn jitter_is_removed_from_motion_vector() {
        // Both frames rasterize the same static world point with different
        // baked jitter; the jitter-free motion vector must be ~zero.
        let cur_jitter = Vec2::new(0.05, -0.02);
        let prev_jitter = Vec2::new(-0.03, 0.04);
        let camera = CameraMotionState {
            cur_view_proj: identity_with_translation(cur_jitter.x, cur_jitter.y),
            prev_view_proj: identity_with_translation(prev_jitter.x, prev_jitter.y),
            cur_jitter,
            prev_jitter,
            convention: NdcConvention::BottomLeftYUp,
        };
        let world = Vec3::new(0.2, 0.3, 0.0);
        let mv =
            screen_motion_vector(camera, world, PrevParticleState::record(world)).expect("visible");
        assert!(mv.magnitude() <= 1e-6);
    }

    #[test]
    fn batch_matches_scalar() {
        let camera = camera_no_jitter();
        let worlds = [Vec3::new(0.4, 0.1, 0.0), Vec3::new(0.0, 0.0, 0.0)];
        let prevs = [
            PrevParticleState::record(Vec3::new(0.2, 0.1, 0.0)),
            PrevParticleState::spawned(Vec3::new(0.0, 0.0, 0.0)),
        ];
        let batch = screen_motion_vectors(camera, &worlds, &prevs);
        assert_eq!(batch.len(), 2);
        let scalar0 = screen_motion_vector(camera, worlds[0], prevs[0]).unwrap();
        assert!(approx(
            batch[0].unwrap().uv_delta.x,
            scalar0.uv_delta.x,
            1e-6
        ));
        assert!(batch[1].is_none());
    }

    #[test]
    fn flipbook_frames_advanced_is_absolute() {
        assert!(approx(flipbook_frames_advanced(2.0, 5.0), 3.0, 1e-6));
        assert!(approx(flipbook_frames_advanced(5.0, 2.0), 3.0, 1e-6));
    }

    #[test]
    fn flipbook_below_threshold_is_stable_with_partial_history() {
        let s = flipbook_stability(0.5, 2.0);
        assert!(!s.temporally_unstable);
        assert!(approx(s.history_weight, 0.75, 1e-6));
    }

    #[test]
    fn flipbook_at_or_above_threshold_rejects_history() {
        let s = flipbook_stability(3.0, 2.0);
        assert!(s.temporally_unstable);
        assert!(approx(s.history_weight, 0.0, 1e-6));
    }

    #[test]
    fn flipbook_non_positive_threshold_is_disabled() {
        let s = flipbook_stability(10.0, 0.0);
        assert!(!s.temporally_unstable);
        assert!(approx(s.history_weight, 1.0, 1e-6));
    }

    fn request(write: bool, unstable: bool, reactive: f32) -> MotionVectorRequest {
        MotionVectorRequest {
            write_motion_vectors: write,
            temporally_unstable: unstable,
            reactive_mask: reactive,
        }
    }

    #[test]
    fn reactive_weight_opaque_static_keeps_history() {
        let hw = reactive_history_weight(request(true, false, 0.0), Some(ScreenMotionVector::ZERO));
        assert!(approx(hw.reactive, 0.0, 1e-6));
        assert!(approx(hw.history_weight, 1.0, 1e-6));
    }

    #[test]
    fn reactive_weight_unstable_rejects_history() {
        let hw = reactive_history_weight(request(true, true, 0.5), Some(ScreenMotionVector::ZERO));
        assert!(approx(hw.reactive, 1.0, 1e-6));
        assert!(approx(hw.history_weight, 0.0, 1e-6));
    }

    #[test]
    fn reactive_weight_missing_motion_rejects_history() {
        let hw = reactive_history_weight(request(true, false, 0.2), None);
        assert!(approx(hw.reactive, 1.0, 1e-6));
    }

    #[test]
    fn reactive_weight_clamps_out_of_range_mask() {
        let hw = reactive_history_weight(request(true, false, 2.0), Some(ScreenMotionVector::ZERO));
        assert!(approx(hw.reactive, 1.0, 1e-6));
    }

    #[test]
    fn motion_f16_round_trip_within_tolerance() {
        let mv = ScreenMotionVector {
            uv_delta: Vec2::new(0.0123, -0.0456),
        };
        let dec = decode_motion_f16(encode_motion_f16(mv));
        assert!(approx(dec.uv_delta.x, mv.uv_delta.x, 1e-4));
        assert!(approx(dec.uv_delta.y, mv.uv_delta.y, 1e-4));
    }

    #[test]
    fn motion_snorm16_round_trip_within_tolerance() {
        let max_uv = 0.5;
        let mv = ScreenMotionVector {
            uv_delta: Vec2::new(0.1, -0.25),
        };
        let dec = decode_motion_snorm16(encode_motion_snorm16(mv, max_uv), max_uv);
        assert!(approx(dec.uv_delta.x, 0.1, 1e-4));
        assert!(approx(dec.uv_delta.y, -0.25, 1e-4));
    }

    #[test]
    fn motion_snorm16_saturates_out_of_range() {
        let max_uv = 0.1;
        let mv = ScreenMotionVector {
            uv_delta: Vec2::new(0.5, -0.5),
        };
        let dec = decode_motion_snorm16(encode_motion_snorm16(mv, max_uv), max_uv);
        assert!(approx(dec.uv_delta.x, 0.1, 1e-4));
        assert!(approx(dec.uv_delta.y, -0.1, 1e-4));
    }

    #[test]
    fn motion_snorm16_zero_range_encodes_zero() {
        let mv = ScreenMotionVector {
            uv_delta: Vec2::new(0.3, 0.3),
        };
        let enc = encode_motion_snorm16(mv, 0.0);
        assert_eq!(enc, EncodedMotionVectorSnorm16 { x: 0, y: 0 });
    }

    #[test]
    fn oit_contract_only_front_most_writes() {
        let front = oit_motion_contract(request(true, false, 0.5), true);
        assert!(front.contributes_motion_vector);
        assert!(approx(front.history_weight, 0.5, 1e-6));
        let back = oit_motion_contract(request(true, false, 0.5), false);
        assert!(!back.contributes_motion_vector);
    }

    #[test]
    fn oit_contract_unstable_drops_history() {
        let c = oit_motion_contract(request(true, true, 0.2), true);
        assert!(approx(c.history_weight, 0.0, 1e-6));
    }

    #[test]
    fn surface_renderers_write_motion_vectors() {
        for kind in [
            RendererKind::Sprite,
            RendererKind::Mesh,
            RendererKind::Ribbon,
            RendererKind::Beam,
        ] {
            assert!(renderer_writes_motion_vectors(kind));
        }
        for kind in [
            RendererKind::Light,
            RendererKind::Decal,
            RendererKind::Volume,
        ] {
            assert!(!renderer_writes_motion_vectors(kind));
        }
    }

    #[test]
    fn prev_state_record_and_spawn_flags() {
        assert!(PrevParticleState::record(Vec3::ZERO).valid);
        assert!(!PrevParticleState::spawned(Vec3::ZERO).valid);
    }
}
