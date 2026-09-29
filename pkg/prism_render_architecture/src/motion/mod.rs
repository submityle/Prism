//! Motion-vector, reprojection, and disocclusion contracts (AAA temporal core).
//!
//! Motion is a first-class rendering subsystem: it produces the per-pixel
//! screen-space velocity field and the confidence/rejection signals that every
//! temporal consumer relies on. `TAA`, motion blur, temporal upsamplers
//! (`DLSS` / `FSR` / `XeSS`-class reconstructors), temporal denoisers, and
//! variable-rate history reuse all read this subsystem's outputs. Getting the
//! velocity vectors and the disocclusion mask right is what separates a stable
//! AAA image from a ghosting, smearing one.
//!
//! The pipeline mirrors production temporal stacks at the algorithm level,
//! without reusing any engine's code:
//!
//! 1. **Reprojection** — the previous frame's clip-space transform is combined
//!    with the current one to map each pixel back to where its surface was last
//!    frame, yielding a screen-space motion vector and a reprojection
//!    confidence (design: camera + object motion, `NDC` <-> pixel mapping); see
//!    [`reproject`].
//! 2. **Encoding** — motion vectors are quantized and packed into the compact
//!    fixed-point representation the `GPU` velocity target stores, together with
//!    the reactive / transparency masks that steer temporal blend weights; see
//!    [`encode`].
//! 3. **Dilation** — the neighborhood-max velocity dilation that keeps thin,
//!    fast-moving foreground silhouettes from tearing, plus the tile-max
//!    reduction that motion blur samples; see [`dilation`].
//! 4. **Disocclusion** — the depth + normal + surface-id heuristic that decides
//!    when the reprojected history is invalid (a newly revealed surface) and
//!    must be rejected instead of blended; see [`disocclusion`].
//! 5. **Tiles** — the screen-tile classifier (static / slow / fast) that drives
//!    `TAA` tile flags and the motion-blur half-length budget; see [`tiles`].
//!
//! This module owns the shared contract types the sibling modules build on: the
//! hand-rolled vector/matrix math, the screen-dimension descriptor, the motion
//! sample record, and the small policy enums. The `GPU` compute kernels and the
//! `WESL` shader codegen are out of scope for this `CPU`-verifiable contract
//! layer and are documented as "pending the GPU backend" where the contract
//! signatures anticipate them.

pub mod encode;
pub mod reproject;

/// Squared-length threshold below which a vector is treated as zero, so
/// normalization never divides by (near) zero and never propagates `NaN`.
/// Matches the sibling particle/cloth/hair subsystems.
pub const EPS_LEN_SQ: f32 = 1e-12;

/// Generic small epsilon for comparing derived `f32` quantities (confidences,
/// weights, clip-space `w`) without relying on exact bit equality.
pub const EPS: f32 = 1e-6;

/// A hand-rolled two-component vector for screen-space motion math.
///
/// `prism_render_architecture` is a dependency-free contracts crate, so the
/// vector math is spelled out here rather than pulled from a linear-algebra
/// dependency. Only `sqrt` is used (the single float intrinsic the workspace
/// determinism policy allows for this crate); no transcendental functions are
/// called, keeping the `CPU` reference bit-reproducible against a future `GPU`
/// kernel.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Vec2 {
    /// X component (pixels or `NDC`, depending on the call site).
    pub x: f32,
    /// Y component (pixels or `NDC`, depending on the call site).
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

    /// Uniform vector with both components set to `v`.
    #[must_use]
    pub const fn splat(v: f32) -> Self {
        Self { x: v, y: v }
    }

    /// Component-wise sum.
    #[must_use]
    #[expect(
        clippy::should_implement_trait,
        reason = "The motion math API is specified with named add/sub methods for call-site uniformity, matching the sibling water/particle modules; operator traits are intentionally not part of this internal type."
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

    /// Component-wise (Hadamard) product.
    #[must_use]
    #[expect(
        clippy::should_implement_trait,
        reason = "See add: the specified API uses a named component-wise mul (Hadamard product), not the scalar-overloading Mul operator trait."
    )]
    pub fn mul(self, rhs: Self) -> Self {
        Self::new(self.x * rhs.x, self.y * rhs.y)
    }

    /// Dot (inner) product.
    #[must_use]
    pub fn dot(self, rhs: Self) -> f32 {
        self.x * rhs.x + self.y * rhs.y
    }

    /// Squared Euclidean length; cheaper than [`Vec2::length`] for comparisons.
    #[must_use]
    pub fn length_squared(self) -> f32 {
        self.dot(self)
    }

    /// Euclidean length.
    #[must_use]
    pub fn length(self) -> f32 {
        self.length_squared().sqrt()
    }

    /// Component-wise minimum.
    #[must_use]
    pub fn min(self, rhs: Self) -> Self {
        Self::new(self.x.min(rhs.x), self.y.min(rhs.y))
    }

    /// Component-wise maximum.
    #[must_use]
    pub fn max(self, rhs: Self) -> Self {
        Self::new(self.x.max(rhs.x), self.y.max(rhs.y))
    }

    /// Returns the unit vector along `self`, or [`Vec2::ZERO`] when `self` is
    /// (numerically) the zero vector, so normalization never yields `NaN`.
    #[must_use]
    pub fn normalize_or_zero(self) -> Self {
        let len_sq = self.length_squared();
        if len_sq > EPS_LEN_SQ {
            self.scale(1.0 / len_sq.sqrt())
        } else {
            Self::ZERO
        }
    }
}

/// A hand-rolled four-component vector, used for homogeneous clip-space
/// positions in reprojection.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Vec4 {
    /// X component.
    pub x: f32,
    /// Y component.
    pub y: f32,
    /// Z component.
    pub z: f32,
    /// W (homogeneous) component.
    pub w: f32,
}

impl Vec4 {
    /// The zero vector.
    pub const ZERO: Self = Self {
        x: 0.0,
        y: 0.0,
        z: 0.0,
        w: 0.0,
    };

    /// Builds a vector from components.
    #[must_use]
    pub const fn new(x: f32, y: f32, z: f32, w: f32) -> Self {
        Self { x, y, z, w }
    }

    /// Builds a homogeneous point `(x, y, z, 1)`.
    #[must_use]
    pub const fn point(x: f32, y: f32, z: f32) -> Self {
        Self { x, y, z, w: 1.0 }
    }

    /// Component-wise sum.
    #[must_use]
    #[expect(
        clippy::should_implement_trait,
        reason = "See Vec2::add: the specified API uses named add for call-site uniformity, not operator traits."
    )]
    pub fn add(self, rhs: Self) -> Self {
        Self::new(
            self.x + rhs.x,
            self.y + rhs.y,
            self.z + rhs.z,
            self.w + rhs.w,
        )
    }

    /// Uniform scale by a scalar.
    #[must_use]
    pub fn scale(self, s: f32) -> Self {
        Self::new(self.x * s, self.y * s, self.z * s, self.w * s)
    }

    /// Dot (inner) product.
    #[must_use]
    pub fn dot(self, rhs: Self) -> f32 {
        self.x * rhs.x + self.y * rhs.y + self.z * rhs.z + self.w * rhs.w
    }

    /// Perspective divide to `NDC`, returning `(x/w, y/w, z/w)` as a [`Vec4`]
    /// with `w = 1`, or [`Vec4::ZERO`] when `w` is (numerically) zero so the
    /// divide never yields `NaN` / infinities. Call sites must check
    /// [`Vec4::is_in_front`] first when they need to distinguish a genuine
    /// behind-camera projection from a clamped one.
    #[must_use]
    pub fn perspective_divide(self) -> Self {
        if self.w.abs() > EPS {
            let inv = 1.0 / self.w;
            Self::new(self.x * inv, self.y * inv, self.z * inv, 1.0)
        } else {
            Self::ZERO
        }
    }

    /// Whether the homogeneous point is in front of the camera (positive `w`),
    /// the precondition for a meaningful perspective divide.
    #[must_use]
    pub fn is_in_front(self) -> bool {
        self.w > EPS
    }

    /// The `x, y` components as a [`Vec2`], dropping `z` and `w`.
    #[must_use]
    pub fn xy(self) -> Vec2 {
        Vec2::new(self.x, self.y)
    }
}

/// A column-major `4x4` matrix, used to carry the previous/current
/// view-projection transforms into the reprojection math.
///
/// Storage is four column [`Vec4`]s so that a transform is `cols[0] * v.x +
/// cols[1] * v.y + cols[2] * v.z + cols[3] * v.w`. Only `+`, `-`, `*`, and `/`
/// are used, so the result stays bit-reproducible against a future `GPU`
/// kernel.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Mat4 {
    /// The four columns, in column-major order.
    pub cols: [Vec4; 4],
}

impl Default for Mat4 {
    fn default() -> Self {
        Self::IDENTITY
    }
}

impl Mat4 {
    /// The identity transform.
    pub const IDENTITY: Self = Self {
        cols: [
            Vec4::new(1.0, 0.0, 0.0, 0.0),
            Vec4::new(0.0, 1.0, 0.0, 0.0),
            Vec4::new(0.0, 0.0, 1.0, 0.0),
            Vec4::new(0.0, 0.0, 0.0, 1.0),
        ],
    };

    /// Builds a matrix from four columns (column-major).
    #[must_use]
    pub const fn from_cols(c0: Vec4, c1: Vec4, c2: Vec4, c3: Vec4) -> Self {
        Self {
            cols: [c0, c1, c2, c3],
        }
    }

    /// Transforms a homogeneous vector: `M * v`.
    #[must_use]
    pub fn mul_vec4(&self, v: Vec4) -> Vec4 {
        self.cols[0]
            .scale(v.x)
            .add(self.cols[1].scale(v.y))
            .add(self.cols[2].scale(v.z))
            .add(self.cols[3].scale(v.w))
    }

    /// Matrix product `self * rhs` (column-major composition, so applying the
    /// result is equivalent to applying `rhs` first, then `self`).
    #[must_use]
    pub fn mul_mat4(&self, rhs: &Self) -> Self {
        Self::from_cols(
            self.mul_vec4(rhs.cols[0]),
            self.mul_vec4(rhs.cols[1]),
            self.mul_vec4(rhs.cols[2]),
            self.mul_vec4(rhs.cols[3]),
        )
    }

    /// Flattens to a column-major `[f32; 16]` (`m[col * 4 + row]`), the layout
    /// the inverse cofactor expansion below indexes directly.
    #[must_use]
    pub fn to_cols_array(&self) -> [f32; 16] {
        [
            self.cols[0].x,
            self.cols[0].y,
            self.cols[0].z,
            self.cols[0].w,
            self.cols[1].x,
            self.cols[1].y,
            self.cols[1].z,
            self.cols[1].w,
            self.cols[2].x,
            self.cols[2].y,
            self.cols[2].z,
            self.cols[2].w,
            self.cols[3].x,
            self.cols[3].y,
            self.cols[3].z,
            self.cols[3].w,
        ]
    }

    /// Rebuilds a matrix from a column-major `[f32; 16]` (`m[col * 4 + row]`).
    #[must_use]
    pub fn from_cols_array(m: &[f32; 16]) -> Self {
        Self::from_cols(
            Vec4::new(m[0], m[1], m[2], m[3]),
            Vec4::new(m[4], m[5], m[6], m[7]),
            Vec4::new(m[8], m[9], m[10], m[11]),
            Vec4::new(m[12], m[13], m[14], m[15]),
        )
    }

    /// Analytic inverse via cofactor expansion (adjugate over determinant),
    /// returning `None` when the matrix is (numerically) singular so callers
    /// never divide by a zero determinant. Uses only `+`, `-`, `*`, `/`, so the
    /// result is bit-reproducible against a future `GPU` kernel. The classic
    /// cofactor layout operates on the column-major flat array.
    #[must_use]
    pub fn inverse(&self) -> Option<Self> {
        let m = self.to_cols_array();
        let mut inv = [0.0f32; 16];

        inv[0] = m[5] * m[10] * m[15] - m[5] * m[11] * m[14] - m[9] * m[6] * m[15]
            + m[9] * m[7] * m[14]
            + m[13] * m[6] * m[11]
            - m[13] * m[7] * m[10];
        inv[4] = -m[4] * m[10] * m[15] + m[4] * m[11] * m[14] + m[8] * m[6] * m[15]
            - m[8] * m[7] * m[14]
            - m[12] * m[6] * m[11]
            + m[12] * m[7] * m[10];
        inv[8] = m[4] * m[9] * m[15] - m[4] * m[11] * m[13] - m[8] * m[5] * m[15]
            + m[8] * m[7] * m[13]
            + m[12] * m[5] * m[11]
            - m[12] * m[7] * m[9];
        inv[12] = -m[4] * m[9] * m[14] + m[4] * m[10] * m[13] + m[8] * m[5] * m[14]
            - m[8] * m[6] * m[13]
            - m[12] * m[5] * m[10]
            + m[12] * m[6] * m[9];
        inv[1] = -m[1] * m[10] * m[15] + m[1] * m[11] * m[14] + m[9] * m[2] * m[15]
            - m[9] * m[3] * m[14]
            - m[13] * m[2] * m[11]
            + m[13] * m[3] * m[10];
        inv[5] = m[0] * m[10] * m[15] - m[0] * m[11] * m[14] - m[8] * m[2] * m[15]
            + m[8] * m[3] * m[14]
            + m[12] * m[2] * m[11]
            - m[12] * m[3] * m[10];
        inv[9] = -m[0] * m[9] * m[15] + m[0] * m[11] * m[13] + m[8] * m[1] * m[15]
            - m[8] * m[3] * m[13]
            - m[12] * m[1] * m[11]
            + m[12] * m[3] * m[9];
        inv[13] = m[0] * m[9] * m[14] - m[0] * m[10] * m[13] - m[8] * m[1] * m[14]
            + m[8] * m[2] * m[13]
            + m[12] * m[1] * m[10]
            - m[12] * m[2] * m[9];
        inv[2] = m[1] * m[6] * m[15] - m[1] * m[7] * m[14] - m[5] * m[2] * m[15]
            + m[5] * m[3] * m[14]
            + m[13] * m[2] * m[7]
            - m[13] * m[3] * m[6];
        inv[6] = -m[0] * m[6] * m[15] + m[0] * m[7] * m[14] + m[4] * m[2] * m[15]
            - m[4] * m[3] * m[14]
            - m[12] * m[2] * m[7]
            + m[12] * m[3] * m[6];
        inv[10] = m[0] * m[5] * m[15] - m[0] * m[7] * m[13] - m[4] * m[1] * m[15]
            + m[4] * m[3] * m[13]
            + m[12] * m[1] * m[7]
            - m[12] * m[3] * m[5];
        inv[14] = -m[0] * m[5] * m[14] + m[0] * m[6] * m[13] + m[4] * m[1] * m[14]
            - m[4] * m[2] * m[13]
            - m[12] * m[1] * m[6]
            + m[12] * m[2] * m[5];
        inv[3] = -m[1] * m[6] * m[11] + m[1] * m[7] * m[10] + m[5] * m[2] * m[11]
            - m[5] * m[3] * m[10]
            - m[9] * m[2] * m[7]
            + m[9] * m[3] * m[6];
        inv[7] = m[0] * m[6] * m[11] - m[0] * m[7] * m[10] - m[4] * m[2] * m[11]
            + m[4] * m[3] * m[10]
            + m[8] * m[2] * m[7]
            - m[8] * m[3] * m[6];
        inv[11] = -m[0] * m[5] * m[11] + m[0] * m[7] * m[9] + m[4] * m[1] * m[11]
            - m[4] * m[3] * m[9]
            - m[8] * m[1] * m[7]
            + m[8] * m[3] * m[5];
        inv[15] = m[0] * m[5] * m[10] - m[0] * m[6] * m[9] - m[4] * m[1] * m[10]
            + m[4] * m[2] * m[9]
            + m[8] * m[1] * m[6]
            - m[8] * m[2] * m[5];

        let det = m[0] * inv[0] + m[1] * inv[4] + m[2] * inv[8] + m[3] * inv[12];
        if det.abs() <= EPS {
            return None;
        }
        let inv_det = 1.0 / det;
        for entry in &mut inv {
            *entry *= inv_det;
        }
        Some(Self::from_cols_array(&inv))
    }
}

/// Integer screen dimensions in pixels, shared by every stage that maps between
/// `NDC` and pixel coordinates.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct ScreenDims {
    /// Render-target width in pixels; always at least 1 after construction.
    pub width: u32,
    /// Render-target height in pixels; always at least 1 after construction.
    pub height: u32,
}

impl ScreenDims {
    /// Builds screen dimensions, clamping each axis to a minimum of one pixel so
    /// downstream `NDC` <-> pixel scaling never divides by zero.
    #[must_use]
    pub const fn new(width: u32, height: u32) -> Self {
        Self {
            width: if width == 0 { 1 } else { width },
            height: if height == 0 { 1 } else { height },
        }
    }

    /// Width as `f32`.
    #[must_use]
    pub fn width_f32(self) -> f32 {
        self.width as f32
    }

    /// Height as `f32`.
    #[must_use]
    pub fn height_f32(self) -> f32 {
        self.height as f32
    }

    /// Total pixel count.
    #[must_use]
    pub fn pixel_count(self) -> u64 {
        u64::from(self.width) * u64::from(self.height)
    }
}

/// A stable identifier for the surface a pixel belongs to, used by the
/// disocclusion heuristic to reject history that reprojects onto a different
/// object than it came from. `0` is reserved for "no surface" (sky / cleared).
#[derive(Clone, Copy, Debug, Default, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct SurfaceId(pub u64);

impl SurfaceId {
    /// The reserved "no surface" identifier (sky, cleared background).
    pub const NONE: Self = Self(0);

    /// Whether this identifier denotes a real surface (not the reserved none).
    #[must_use]
    pub fn is_some(self) -> bool {
        self.0 != Self::NONE.0
    }
}

/// The per-pixel motion record produced by the subsystem and consumed by every
/// temporal stage.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct MotionSample {
    /// Screen-space velocity in pixels: where the surface *was* relative to
    /// where it is now, i.e. `prev_pixel - curr_pixel`. Adding this to the
    /// current pixel coordinate yields the history fetch location.
    pub velocity_pixels: [f32; 2],
    /// Confidence in the reprojection, in `[0, 1]`. Drops toward zero when the
    /// history fetch leaves the frame or the surface was disoccluded.
    pub reprojection_confidence: f32,
    /// Reactive mask in `[0, 1]`: how strongly this pixel should favor the
    /// current frame over history (transparents, fast flipbooks, shading
    /// discontinuities). `0` = fully temporal, `1` = fully reactive.
    pub reactive: f32,
    /// Transparency coverage in `[0, 1]` for order-independent compositing hints
    /// and reactive-mask derivation.
    pub transparency: f32,
    /// The surface this pixel belongs to, for disocclusion rejection.
    pub surface_id: u64,
}

impl MotionSample {
    /// The velocity as a [`Vec2`].
    #[must_use]
    pub fn velocity(self) -> Vec2 {
        Vec2::new(self.velocity_pixels[0], self.velocity_pixels[1])
    }

    /// The surface identifier as a [`SurfaceId`].
    #[must_use]
    pub fn surface(self) -> SurfaceId {
        SurfaceId(self.surface_id)
    }
}

/// What generates a pixel's motion, classifying the source so temporal
/// consumers can weight history reuse and reactive masks appropriately.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum MotionSource {
    /// Camera-only motion (static geometry reprojected by the view change).
    Camera,
    /// A rigidly transformed object (one `prev`/`curr` model matrix).
    Rigid,
    /// A skinned mesh (skeletal deformation baked into per-vertex motion).
    Skinned,
    /// A blend-shape / morph-target deformed mesh.
    Morph,
    /// General baked vertex animation.
    VertexAnimation,
    /// A simulated particle (velocity comes from the particle integrator).
    Particle,
    /// A surface newly revealed by streaming (history must be treated as
    /// invalid; pairs with [`crate::history::InvalidationMask::STREAMING_REVEAL`]).
    StreamingReveal,
}

impl MotionSource {
    /// Whether this source can be reprojected purely from the camera transform.
    /// Only static geometry (camera motion) qualifies; every deforming or
    /// simulated source needs its own per-vertex/per-particle velocity.
    #[must_use]
    pub fn is_camera_reprojectable(self) -> bool {
        matches!(self, MotionSource::Camera)
    }

    /// Whether this source should force history rejection on the frame it first
    /// appears. A streaming reveal has no valid history to blend against.
    #[must_use]
    pub fn forces_history_rejection(self) -> bool {
        matches!(self, MotionSource::StreamingReveal)
    }
}

/// Clamps `x` to `[lo, hi]`, resolving `NaN` to `lo` deterministically instead
/// of relying on `f32::clamp` (whose `NaN` propagation is undesirable here and
/// which trips the `manual_clamp` lint when hand-written).
#[must_use]
pub(crate) fn clamp01(x: f32) -> f32 {
    // Written as explicit branches so a `NaN` input lands on `0.0` rather than
    // propagating, matching the deterministic policy of the sibling modules.
    if x.is_nan() || x < 0.0 {
        0.0
    } else if x > 1.0 {
        1.0
    } else {
        x
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx(a: f32, b: f32) -> bool {
        (a - b).abs() <= 1e-5
    }

    #[test]
    fn vec2_basic_algebra_is_exact() {
        let a = Vec2::new(1.0, 2.0);
        let b = Vec2::new(4.0, 6.0);
        assert_eq!(a.add(b), Vec2::new(5.0, 8.0));
        assert_eq!(b.sub(a), Vec2::new(3.0, 4.0));
        assert_eq!(a.scale(2.0), Vec2::new(2.0, 4.0));
        assert_eq!(a.mul(b), Vec2::new(4.0, 12.0));
        assert_eq!(a.dot(b), 16.0);
        assert_eq!(Vec2::splat(3.0), Vec2::new(3.0, 3.0));
    }

    #[test]
    fn vec2_length_and_normalize() {
        let a = Vec2::new(3.0, 4.0);
        assert_eq!(a.length_squared(), 25.0);
        assert_eq!(a.length(), 5.0);
        let n = a.normalize_or_zero();
        assert!(approx(n.length(), 1.0));
        assert_eq!(Vec2::ZERO.normalize_or_zero(), Vec2::ZERO);
    }

    #[test]
    fn vec2_min_max() {
        let a = Vec2::new(1.0, -2.0);
        let b = Vec2::new(-1.0, 2.0);
        assert_eq!(a.min(b), Vec2::new(-1.0, -2.0));
        assert_eq!(a.max(b), Vec2::new(1.0, 2.0));
    }

    #[test]
    fn mat4_identity_is_a_no_op() {
        let v = Vec4::new(1.0, 2.0, 3.0, 1.0);
        assert_eq!(Mat4::IDENTITY.mul_vec4(v), v);
        assert_eq!(Mat4::default(), Mat4::IDENTITY);
    }

    #[test]
    fn mat4_compose_matches_sequential_apply() {
        // A pure translation by (10, 0, 0) in column-major form: the translation
        // lives in the last column.
        let translate = Mat4::from_cols(
            Vec4::new(1.0, 0.0, 0.0, 0.0),
            Vec4::new(0.0, 1.0, 0.0, 0.0),
            Vec4::new(0.0, 0.0, 1.0, 0.0),
            Vec4::new(10.0, 0.0, 0.0, 1.0),
        );
        // A uniform scale by 2.
        let scale = Mat4::from_cols(
            Vec4::new(2.0, 0.0, 0.0, 0.0),
            Vec4::new(0.0, 2.0, 0.0, 0.0),
            Vec4::new(0.0, 0.0, 2.0, 0.0),
            Vec4::new(0.0, 0.0, 0.0, 1.0),
        );
        let p = Vec4::point(1.0, 1.0, 1.0);
        // translate * scale: scale first, then translate.
        let composed = translate.mul_mat4(&scale);
        let direct = translate.mul_vec4(scale.mul_vec4(p));
        assert_eq!(composed.mul_vec4(p), direct);
        assert_eq!(composed.mul_vec4(p), Vec4::new(12.0, 2.0, 2.0, 1.0));
    }

    #[test]
    fn perspective_divide_guards_zero_w() {
        let p = Vec4::new(2.0, 4.0, 6.0, 2.0);
        assert_eq!(p.perspective_divide(), Vec4::new(1.0, 2.0, 3.0, 1.0));
        // Zero (and near-zero) w must not produce NaN/inf.
        let degenerate = Vec4::new(1.0, 1.0, 1.0, 0.0).perspective_divide();
        assert_eq!(degenerate, Vec4::ZERO);
        assert!(!p.perspective_divide().x.is_nan());
    }

    #[test]
    fn is_in_front_tracks_w_sign() {
        assert!(Vec4::point(0.0, 0.0, 5.0).is_in_front());
        assert!(!Vec4::new(0.0, 0.0, 5.0, -1.0).is_in_front());
        assert!(!Vec4::new(0.0, 0.0, 5.0, 0.0).is_in_front());
    }

    #[test]
    fn mat4_inverse_round_trips_to_identity() {
        // An affine transform: uniform scale 2, then translate (3, -4, 5).
        let m = Mat4::from_cols(
            Vec4::new(2.0, 0.0, 0.0, 0.0),
            Vec4::new(0.0, 2.0, 0.0, 0.0),
            Vec4::new(0.0, 0.0, 2.0, 0.0),
            Vec4::new(3.0, -4.0, 5.0, 1.0),
        );
        let inv = m.inverse().expect("non-singular");
        let product = m.mul_mat4(&inv);
        for row in 0..4 {
            for col in 0..4 {
                let expected = if row == col { 1.0 } else { 0.0 };
                let entry = match row {
                    0 => product.cols[col].x,
                    1 => product.cols[col].y,
                    2 => product.cols[col].z,
                    _ => product.cols[col].w,
                };
                assert!(approx(entry, expected), "[{row}][{col}] = {entry}");
            }
        }
    }

    #[test]
    fn mat4_inverse_reports_singular() {
        // A projection onto a plane (z column zeroed) is singular.
        let singular = Mat4::from_cols(
            Vec4::new(1.0, 0.0, 0.0, 0.0),
            Vec4::new(0.0, 1.0, 0.0, 0.0),
            Vec4::new(0.0, 0.0, 0.0, 0.0),
            Vec4::new(0.0, 0.0, 0.0, 1.0),
        );
        assert!(singular.inverse().is_none());
    }

    #[test]
    fn screen_dims_clamp_to_one_pixel() {
        let d = ScreenDims::new(0, 0);
        assert_eq!(d.width, 1);
        assert_eq!(d.height, 1);
        let d = ScreenDims::new(1920, 1080);
        assert_eq!(d.pixel_count(), 1920 * 1080);
        assert!(approx(d.width_f32(), 1920.0));
    }

    #[test]
    fn surface_id_none_is_distinct() {
        assert!(!SurfaceId::NONE.is_some());
        assert!(SurfaceId(1).is_some());
        assert_eq!(SurfaceId::default(), SurfaceId::NONE);
    }

    #[test]
    fn motion_sample_accessors() {
        let s = MotionSample {
            velocity_pixels: [1.5, -2.5],
            reprojection_confidence: 0.9,
            reactive: 0.1,
            transparency: 0.0,
            surface_id: 7,
        };
        assert_eq!(s.velocity(), Vec2::new(1.5, -2.5));
        assert_eq!(s.surface(), SurfaceId(7));
    }

    #[test]
    fn motion_source_classification() {
        assert!(MotionSource::Camera.is_camera_reprojectable());
        assert!(!MotionSource::Skinned.is_camera_reprojectable());
        assert!(MotionSource::StreamingReveal.forces_history_rejection());
        assert!(!MotionSource::Rigid.forces_history_rejection());
    }

    #[test]
    fn clamp01_resolves_nan_to_zero() {
        assert_eq!(clamp01(f32::NAN), 0.0);
        assert_eq!(clamp01(-1.0), 0.0);
        assert_eq!(clamp01(2.0), 1.0);
        assert_eq!(clamp01(0.5), 0.5);
    }
}
