//! Projection matrices: perspective and orthographic.
//!
//! Both variants build right-handed clip-space matrices with a `[0, 1]` depth
//! range via [`prism_math`], so they pair directly with the default Prism RHI
//! backend (Vulkan / D3D / Metal / WebGPU).

use prism_math::{orthographic_rh, perspective_rh, Mat4, Vec2};

/// A vertical-field-of-view perspective projection.
///
/// The camera looks down `-Z` in view space. `fov_y` is the *full* vertical
/// opening angle (not the half-angle) and `aspect` is `width / height`.
#[derive(Clone, Copy, PartialEq, Debug)]
pub struct Perspective {
    /// Full vertical field of view, in radians.
    pub fov_y: f32,
    /// Viewport aspect ratio, `width / height`.
    pub aspect: f32,
    /// Distance to the near clip plane (maps to depth `0`); must be positive.
    pub near: f32,
    /// Distance to the far clip plane (maps to depth `1`); must exceed `near`.
    pub far: f32,
}

impl Perspective {
    /// Create a perspective projection from its defining scalars.
    #[inline]
    #[must_use]
    pub const fn new(fov_y: f32, aspect: f32, near: f32, far: f32) -> Self {
        Self {
            fov_y,
            aspect,
            near,
            far,
        }
    }

    /// The clip-space projection matrix.
    #[inline]
    #[must_use]
    pub fn matrix(self) -> Mat4 {
        perspective_rh(self.fov_y, self.aspect, self.near, self.far)
    }

    /// A copy re-fit to a different aspect ratio (for a window resize).
    #[inline]
    #[must_use]
    pub const fn with_aspect(self, aspect: f32) -> Self {
        Self { aspect, ..self }
    }

    /// True when the parameters form a usable (non-degenerate) projection.
    #[inline]
    #[must_use]
    pub fn is_valid(self) -> bool {
        self.fov_y.is_finite()
            && self.fov_y > 0.0
            && self.fov_y < core::f32::consts::PI
            && self.aspect.is_finite()
            && self.aspect > 0.0
            && self.near.is_finite()
            && self.near > 0.0
            && self.far.is_finite()
            && self.far > self.near
    }
}

/// An orthographic projection defined by an explicit view rectangle.
///
/// Use [`Orthographic::from_size`] for the common symmetric case, or
/// [`Orthographic::new`] when the bounds are already known.
#[derive(Clone, Copy, PartialEq, Debug)]
pub struct Orthographic {
    /// Left edge of the view rectangle.
    pub left: f32,
    /// Right edge of the view rectangle.
    pub right: f32,
    /// Bottom edge of the view rectangle.
    pub bottom: f32,
    /// Top edge of the view rectangle.
    pub top: f32,
    /// Distance to the near clip plane (maps to depth `0`).
    pub near: f32,
    /// Distance to the far clip plane (maps to depth `1`).
    pub far: f32,
}

impl Orthographic {
    /// Create an orthographic projection from explicit bounds.
    #[inline]
    #[must_use]
    pub const fn new(left: f32, right: f32, bottom: f32, top: f32, near: f32, far: f32) -> Self {
        Self {
            left,
            right,
            bottom,
            top,
            near,
            far,
        }
    }

    /// Create a symmetric orthographic projection of the given `size`
    /// (`width`, `height`) centered on the view axis.
    #[inline]
    #[must_use]
    pub fn from_size(size: Vec2, near: f32, far: f32) -> Self {
        let half = size * 0.5;
        Self::new(-half.x, half.x, -half.y, half.y, near, far)
    }

    /// The clip-space projection matrix.
    #[inline]
    #[must_use]
    pub fn matrix(self) -> Mat4 {
        orthographic_rh(
            self.left,
            self.right,
            self.bottom,
            self.top,
            self.near,
            self.far,
        )
    }

    /// A copy re-fit to a different aspect ratio: the height (and vertical
    /// centring) are preserved and the horizontal extent is widened to match.
    #[inline]
    #[must_use]
    pub fn with_aspect(self, aspect: f32) -> Self {
        let height = self.top - self.bottom;
        let half_width = aspect * height * 0.5;
        let center_y = (self.top + self.bottom) * 0.5;
        Self {
            left: -half_width,
            right: half_width,
            bottom: center_y - height * 0.5,
            top: center_y + height * 0.5,
            ..self
        }
    }

    /// True when the bounds form a usable (non-degenerate) projection.
    #[inline]
    #[must_use]
    pub fn is_valid(self) -> bool {
        self.left.is_finite()
            && self.right.is_finite()
            && self.bottom.is_finite()
            && self.top.is_finite()
            && self.right > self.left
            && self.top > self.bottom
            && self.near.is_finite()
            && self.far.is_finite()
            && self.far > self.near
    }
}

/// A camera projection: perspective or orthographic.
#[derive(Clone, Copy, PartialEq, Debug)]
pub enum Projection {
    /// A vertical-field-of-view perspective projection.
    Perspective(Perspective),
    /// An orthographic projection.
    Orthographic(Orthographic),
}

impl Projection {
    /// The clip-space projection matrix for this projection.
    #[inline]
    #[must_use]
    pub fn matrix(self) -> Mat4 {
        match self {
            Self::Perspective(perspective) => perspective.matrix(),
            Self::Orthographic(orthographic) => orthographic.matrix(),
        }
    }

    /// A copy re-fit to `aspect`: perspective picks up the new ratio directly,
    /// orthographic widens horizontally while preserving its height.
    #[inline]
    #[must_use]
    pub fn with_aspect(self, aspect: f32) -> Self {
        match self {
            Self::Perspective(perspective) => Self::Perspective(perspective.with_aspect(aspect)),
            Self::Orthographic(orthographic) => {
                Self::Orthographic(orthographic.with_aspect(aspect))
            }
        }
    }

    /// True when the wrapped projection is usable.
    #[inline]
    #[must_use]
    pub fn is_valid(self) -> bool {
        match self {
            Self::Perspective(perspective) => perspective.is_valid(),
            Self::Orthographic(orthographic) => orthographic.is_valid(),
        }
    }
}
