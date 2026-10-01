//! The [`Lerp`] trait: linear interpolation between two values of the same
//! type.

/// Linear interpolation between `self` and `other`.
///
/// `t` is the interpolation parameter. At `t == 0.0` the result equals `self`
/// and at `t == 1.0` it equals `other`.
///
/// # Clamping
///
/// `Lerp` deliberately does **not** clamp `t`: passing values outside
/// `[0, 1]` performs linear extrapolation. This keeps the trait a pure linear
/// map, which is what overshooting easings (such as `Back`) and springs rely
/// on. Callers that need clamped behaviour should clamp `t` themselves or use
/// an [`crate::Easing`], which maps its input into `[0, 1]` before sampling.
pub trait Lerp {
    /// Interpolate from `self` towards `other` by the parameter `t`.
    fn lerp(&self, other: &Self, t: f32) -> Self;
}

impl Lerp for f32 {
    #[inline]
    fn lerp(&self, other: &Self, t: f32) -> Self {
        self + (other - self) * t
    }
}

impl Lerp for f64 {
    #[inline]
    fn lerp(&self, other: &Self, t: f32) -> Self {
        self + (other - self) * f64::from(t)
    }
}

impl Lerp for [f32; 2] {
    #[inline]
    fn lerp(&self, other: &Self, t: f32) -> Self {
        [self[0].lerp(&other[0], t), self[1].lerp(&other[1], t)]
    }
}

impl Lerp for [f32; 3] {
    #[inline]
    fn lerp(&self, other: &Self, t: f32) -> Self {
        [
            self[0].lerp(&other[0], t),
            self[1].lerp(&other[1], t),
            self[2].lerp(&other[2], t),
        ]
    }
}

impl Lerp for [f32; 4] {
    #[inline]
    fn lerp(&self, other: &Self, t: f32) -> Self {
        [
            self[0].lerp(&other[0], t),
            self[1].lerp(&other[1], t),
            self[2].lerp(&other[2], t),
            self[3].lerp(&other[3], t),
        ]
    }
}

impl Lerp for (f32, f32) {
    #[inline]
    fn lerp(&self, other: &Self, t: f32) -> Self {
        (self.0.lerp(&other.0, t), self.1.lerp(&other.1, t))
    }
}

impl Lerp for (f32, f32, f32) {
    #[inline]
    fn lerp(&self, other: &Self, t: f32) -> Self {
        (
            self.0.lerp(&other.0, t),
            self.1.lerp(&other.1, t),
            self.2.lerp(&other.2, t),
        )
    }
}

impl Lerp for (f32, f32, f32, f32) {
    #[inline]
    fn lerp(&self, other: &Self, t: f32) -> Self {
        (
            self.0.lerp(&other.0, t),
            self.1.lerp(&other.1, t),
            self.2.lerp(&other.2, t),
            self.3.lerp(&other.3, t),
        )
    }
}
