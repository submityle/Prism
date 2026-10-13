//! Double-precision vectors: [`DVec2`], [`DVec3`], [`DVec4`].
//!
//! These are the `f64` analogues of the `f32` [`Vec2`](crate::Vec2),
//! [`Vec3`](crate::Vec3), and [`Vec4`](crate::Vec4) facade types and expose the
//! same method set. They power the M3 big-world coordinate path, where the
//! extra mantissa bits keep absolute world positions accurate before they are
//! rebased into small `f32` offsets (see [`crate::bigworld`]).
//!
//! Column vectors; matrices multiply on the left (`m * v`). Unlike the `f32`
//! types there is no SIMD backend: `f64` math is scalar and is its own
//! behavioural reference.

use crate::float::f64 as mf;
use crate::{Vec2, Vec3, Vec4};
use core::ops::{
    Add, AddAssign, Div, DivAssign, Index, IndexMut, Mul, MulAssign, Neg, Sub, SubAssign,
};

/// A 2-component `f64` vector.
#[derive(Clone, Copy, PartialEq, Default)]
#[repr(C)]
pub struct DVec2 {
    /// X component.
    pub x: f64,
    /// Y component.
    pub y: f64,
}

/// A 3-component `f64` vector (tightly packed, 24 bytes).
#[derive(Clone, Copy, PartialEq, Default)]
#[repr(C)]
pub struct DVec3 {
    /// X component.
    pub x: f64,
    /// Y component.
    pub y: f64,
    /// Z component.
    pub z: f64,
}

/// A 4-component `f64` vector.
#[derive(Clone, Copy, PartialEq, Default)]
#[repr(C)]
pub struct DVec4 {
    /// X component.
    pub x: f64,
    /// Y component.
    pub y: f64,
    /// Z component.
    pub z: f64,
    /// W component.
    pub w: f64,
}

/// Shorthand constructor for [`DVec2`].
#[inline]
pub const fn dvec2(x: f64, y: f64) -> DVec2 {
    DVec2::new(x, y)
}
/// Shorthand constructor for [`DVec3`].
#[inline]
pub const fn dvec3(x: f64, y: f64, z: f64) -> DVec3 {
    DVec3::new(x, y, z)
}
/// Shorthand constructor for [`DVec4`].
#[inline]
pub const fn dvec4(x: f64, y: f64, z: f64, w: f64) -> DVec4 {
    DVec4::new(x, y, z, w)
}

impl DVec2 {
    /// All zeros.
    pub const ZERO: Self = Self::splat(0.0);
    /// All ones.
    pub const ONE: Self = Self::splat(1.0);
    /// Unit X.
    pub const X: Self = Self::new(1.0, 0.0);
    /// Unit Y.
    pub const Y: Self = Self::new(0.0, 1.0);

    /// Create a new vector.
    #[inline]
    pub const fn new(x: f64, y: f64) -> Self {
        Self { x, y }
    }
    /// Broadcast a scalar to every component.
    #[inline]
    pub const fn splat(v: f64) -> Self {
        Self { x: v, y: v }
    }
    /// Dot product.
    #[inline]
    pub fn dot(self, rhs: Self) -> f64 {
        self.x * rhs.x + self.y * rhs.y
    }
    /// Squared length.
    #[inline]
    pub fn length_squared(self) -> f64 {
        self.dot(self)
    }
    /// Euclidean length.
    #[inline]
    pub fn length(self) -> f64 {
        // Scale before squaring so representable vectors near f64's extremes
        // do not overflow/underflow merely while computing their norm.
        if self.x.is_nan() || self.y.is_nan() {
            return f64::NAN;
        }
        let scale = mf::abs(self.x).max(mf::abs(self.y));
        if scale == 0.0 || !scale.is_finite() {
            return scale;
        }
        let x = self.x / scale;
        let y = self.y / scale;
        scale * mf::sqrt(x * x + y * y)
    }
    /// Distance to `rhs`.
    #[inline]
    pub fn distance(self, rhs: Self) -> f64 {
        (self - rhs).length()
    }
    /// Normalize to unit length, returning zero for zero or non-finite input.
    #[inline]
    pub fn normalize(self) -> Self {
        if !self.is_finite() {
            return Self::ZERO;
        }
        let scale = mf::abs(self.x).max(mf::abs(self.y));
        if scale == 0.0 || !scale.is_finite() {
            return Self::ZERO;
        }
        let scaled = Self::new(self.x / scale, self.y / scale);
        scaled * (1.0 / mf::sqrt(scaled.length_squared()))
    }
    /// Normalize, or return zero if the length is too small to normalize.
    #[inline]
    pub fn normalize_or_zero(self) -> Self {
        if !self.is_finite() {
            return Self::ZERO;
        }
        let scale = mf::abs(self.x).max(mf::abs(self.y));
        if scale == 0.0 || !scale.is_finite() {
            Self::ZERO
        } else {
            let scaled = Self::new(self.x / scale, self.y / scale);
            let scaled_len = mf::sqrt(scaled.length_squared());
            if scale > 1.0e-200 / scaled_len {
                scaled * (1.0 / scaled_len)
            } else {
                Self::ZERO
            }
        }
    }
    /// True if both components are finite.
    #[inline]
    pub fn is_finite(self) -> bool {
        self.x.is_finite() && self.y.is_finite()
    }
    /// True if any component is NaN.
    #[inline]
    pub fn is_nan(self) -> bool {
        self.x.is_nan() || self.y.is_nan()
    }
    /// Component-wise minimum.
    #[inline]
    pub fn min(self, rhs: Self) -> Self {
        Self::new(self.x.min(rhs.x), self.y.min(rhs.y))
    }
    /// Component-wise maximum.
    #[inline]
    pub fn max(self, rhs: Self) -> Self {
        Self::new(self.x.max(rhs.x), self.y.max(rhs.y))
    }
    /// Component-wise clamp.
    #[inline]
    pub fn clamp(self, lo: Self, hi: Self) -> Self {
        self.max(lo).min(hi)
    }
    /// Component-wise absolute value.
    #[inline]
    pub fn abs(self) -> Self {
        Self::new(mf::abs(self.x), mf::abs(self.y))
    }
    /// Linear interpolation.
    #[inline]
    pub fn lerp(self, rhs: Self, t: f64) -> Self {
        self + (rhs - self) * t
    }
    /// Horizontal sum of components.
    #[inline]
    pub fn element_sum(self) -> f64 {
        self.x + self.y
    }
    /// 2D perpendicular dot (`self.x*rhs.y - self.y*rhs.x`).
    #[inline]
    pub fn perp_dot(self, rhs: Self) -> f64 {
        self.x * rhs.y - self.y * rhs.x
    }
    /// Extend to a [`DVec3`].
    #[inline]
    pub const fn extend(self, z: f64) -> DVec3 {
        DVec3::new(self.x, self.y, z)
    }
    /// As an array.
    #[inline]
    pub const fn to_array(self) -> [f64; 2] {
        [self.x, self.y]
    }
    /// From an array.
    #[inline]
    pub const fn from_array(a: [f64; 2]) -> Self {
        Self::new(a[0], a[1])
    }
    /// Lossy conversion to the `f32` [`Vec2`].
    #[inline]
    pub fn as_vec2(self) -> Vec2 {
        Vec2::new(self.x as f32, self.y as f32)
    }
}

impl DVec3 {
    /// All zeros.
    pub const ZERO: Self = Self::splat(0.0);
    /// All ones.
    pub const ONE: Self = Self::splat(1.0);
    /// Unit X.
    pub const X: Self = Self::new(1.0, 0.0, 0.0);
    /// Unit Y.
    pub const Y: Self = Self::new(0.0, 1.0, 0.0);
    /// Unit Z.
    pub const Z: Self = Self::new(0.0, 0.0, 1.0);
    /// Negative unit X.
    pub const NEG_X: Self = Self::new(-1.0, 0.0, 0.0);
    /// Negative unit Y.
    pub const NEG_Y: Self = Self::new(0.0, -1.0, 0.0);
    /// Negative unit Z.
    pub const NEG_Z: Self = Self::new(0.0, 0.0, -1.0);

    /// Create a new vector.
    #[inline]
    pub const fn new(x: f64, y: f64, z: f64) -> Self {
        Self { x, y, z }
    }
    /// Broadcast a scalar to every component.
    #[inline]
    pub const fn splat(v: f64) -> Self {
        Self { x: v, y: v, z: v }
    }
    /// Dot product.
    #[inline]
    pub fn dot(self, rhs: Self) -> f64 {
        self.x * rhs.x + self.y * rhs.y + self.z * rhs.z
    }
    /// Cross product.
    #[inline]
    pub fn cross(self, rhs: Self) -> Self {
        Self::new(
            self.y * rhs.z - self.z * rhs.y,
            self.z * rhs.x - self.x * rhs.z,
            self.x * rhs.y - self.y * rhs.x,
        )
    }
    /// Squared length.
    #[inline]
    pub fn length_squared(self) -> f64 {
        self.dot(self)
    }
    /// Euclidean length.
    #[inline]
    pub fn length(self) -> f64 {
        if self.x.is_nan() || self.y.is_nan() || self.z.is_nan() {
            return f64::NAN;
        }
        let scale = mf::abs(self.x).max(mf::abs(self.y)).max(mf::abs(self.z));
        if scale == 0.0 || !scale.is_finite() {
            return scale;
        }
        let x = self.x / scale;
        let y = self.y / scale;
        let z = self.z / scale;
        scale * mf::sqrt(x * x + y * y + z * z)
    }
    /// Distance to `rhs`.
    #[inline]
    pub fn distance(self, rhs: Self) -> f64 {
        (self - rhs).length()
    }
    /// Normalize to unit length, returning zero for zero or non-finite input.
    #[inline]
    pub fn normalize(self) -> Self {
        if !self.is_finite() {
            return Self::ZERO;
        }
        let scale = mf::abs(self.x).max(mf::abs(self.y)).max(mf::abs(self.z));
        if scale == 0.0 || !scale.is_finite() {
            return Self::ZERO;
        }
        let scaled = Self::new(self.x / scale, self.y / scale, self.z / scale);
        scaled * (1.0 / mf::sqrt(scaled.length_squared()))
    }
    /// Normalize, or return zero if too short to normalize.
    #[inline]
    pub fn normalize_or_zero(self) -> Self {
        if !self.is_finite() {
            return Self::ZERO;
        }
        let scale = mf::abs(self.x).max(mf::abs(self.y)).max(mf::abs(self.z));
        if scale == 0.0 || !scale.is_finite() {
            Self::ZERO
        } else {
            let scaled = Self::new(self.x / scale, self.y / scale, self.z / scale);
            let scaled_len = mf::sqrt(scaled.length_squared());
            if scale > 1.0e-200 / scaled_len {
                scaled * (1.0 / scaled_len)
            } else {
                Self::ZERO
            }
        }
    }
    /// True if all components are finite.
    #[inline]
    pub fn is_finite(self) -> bool {
        self.x.is_finite() && self.y.is_finite() && self.z.is_finite()
    }
    /// True if any component is NaN.
    #[inline]
    pub fn is_nan(self) -> bool {
        self.x.is_nan() || self.y.is_nan() || self.z.is_nan()
    }
    /// Component-wise minimum.
    #[inline]
    pub fn min(self, rhs: Self) -> Self {
        Self::new(self.x.min(rhs.x), self.y.min(rhs.y), self.z.min(rhs.z))
    }
    /// Component-wise maximum.
    #[inline]
    pub fn max(self, rhs: Self) -> Self {
        Self::new(self.x.max(rhs.x), self.y.max(rhs.y), self.z.max(rhs.z))
    }
    /// Component-wise clamp.
    #[inline]
    pub fn clamp(self, lo: Self, hi: Self) -> Self {
        self.max(lo).min(hi)
    }
    /// Component-wise absolute value.
    #[inline]
    pub fn abs(self) -> Self {
        Self::new(mf::abs(self.x), mf::abs(self.y), mf::abs(self.z))
    }
    /// Linear interpolation.
    #[inline]
    pub fn lerp(self, rhs: Self, t: f64) -> Self {
        self + (rhs - self) * t
    }
    /// Horizontal sum of components.
    #[inline]
    pub fn element_sum(self) -> f64 {
        self.x + self.y + self.z
    }
    /// Largest component.
    #[inline]
    pub fn max_element(self) -> f64 {
        self.x.max(self.y).max(self.z)
    }
    /// Smallest component.
    #[inline]
    pub fn min_element(self) -> f64 {
        self.x.min(self.y).min(self.z)
    }
    /// Build any unit vector orthogonal to `self` (assumes `self` normalized).
    #[inline]
    pub fn any_orthonormal(self) -> Self {
        // Hughes-Moller style: pick the smallest axis to avoid degeneracy.
        if mf::abs(self.x) <= mf::abs(self.y) && mf::abs(self.x) <= mf::abs(self.z) {
            Self::new(0.0, -self.z, self.y).normalize()
        } else if mf::abs(self.y) <= mf::abs(self.z) {
            Self::new(-self.z, 0.0, self.x).normalize()
        } else {
            Self::new(-self.y, self.x, 0.0).normalize()
        }
    }
    /// Truncate to a [`DVec2`] (drop Z).
    #[inline]
    pub const fn truncate(self) -> DVec2 {
        DVec2::new(self.x, self.y)
    }
    /// Extend to a [`DVec4`].
    #[inline]
    pub const fn extend(self, w: f64) -> DVec4 {
        DVec4::new(self.x, self.y, self.z, w)
    }
    /// As an array.
    #[inline]
    pub const fn to_array(self) -> [f64; 3] {
        [self.x, self.y, self.z]
    }
    /// From an array.
    #[inline]
    pub const fn from_array(a: [f64; 3]) -> Self {
        Self::new(a[0], a[1], a[2])
    }
    /// Lossy conversion to the `f32` [`Vec3`].
    #[inline]
    pub fn as_vec3(self) -> Vec3 {
        Vec3::new(self.x as f32, self.y as f32, self.z as f32)
    }
}

impl DVec4 {
    /// All zeros.
    pub const ZERO: Self = Self::splat(0.0);
    /// All ones.
    pub const ONE: Self = Self::splat(1.0);
    /// Unit X.
    pub const X: Self = Self::new(1.0, 0.0, 0.0, 0.0);
    /// Unit Y.
    pub const Y: Self = Self::new(0.0, 1.0, 0.0, 0.0);
    /// Unit Z.
    pub const Z: Self = Self::new(0.0, 0.0, 1.0, 0.0);
    /// Unit W.
    pub const W: Self = Self::new(0.0, 0.0, 0.0, 1.0);

    /// Create a new vector.
    #[inline]
    pub const fn new(x: f64, y: f64, z: f64, w: f64) -> Self {
        Self { x, y, z, w }
    }
    /// Broadcast a scalar to every component.
    #[inline]
    pub const fn splat(v: f64) -> Self {
        Self {
            x: v,
            y: v,
            z: v,
            w: v,
        }
    }
    /// Dot product.
    #[inline]
    pub fn dot(self, rhs: Self) -> f64 {
        self.x * rhs.x + self.y * rhs.y + self.z * rhs.z + self.w * rhs.w
    }
    /// Squared length.
    #[inline]
    pub fn length_squared(self) -> f64 {
        self.dot(self)
    }
    /// Euclidean length.
    #[inline]
    pub fn length(self) -> f64 {
        if self.x.is_nan() || self.y.is_nan() || self.z.is_nan() || self.w.is_nan() {
            return f64::NAN;
        }
        let scale = mf::abs(self.x)
            .max(mf::abs(self.y))
            .max(mf::abs(self.z))
            .max(mf::abs(self.w));
        if scale == 0.0 || !scale.is_finite() {
            return scale;
        }
        let x = self.x / scale;
        let y = self.y / scale;
        let z = self.z / scale;
        let w = self.w / scale;
        scale * mf::sqrt(x * x + y * y + z * z + w * w)
    }
    /// Normalize to unit length, returning zero for zero or non-finite input.
    #[inline]
    pub fn normalize(self) -> Self {
        if !self.is_finite() {
            return Self::ZERO;
        }
        let scale = mf::abs(self.x)
            .max(mf::abs(self.y))
            .max(mf::abs(self.z))
            .max(mf::abs(self.w));
        if scale == 0.0 || !scale.is_finite() {
            return Self::ZERO;
        }
        let scaled = Self::new(
            self.x / scale,
            self.y / scale,
            self.z / scale,
            self.w / scale,
        );
        scaled * (1.0 / mf::sqrt(scaled.length_squared()))
    }
    /// True if all components are finite.
    #[inline]
    pub fn is_finite(self) -> bool {
        self.x.is_finite() && self.y.is_finite() && self.z.is_finite() && self.w.is_finite()
    }
    /// Linear interpolation.
    #[inline]
    pub fn lerp(self, rhs: Self, t: f64) -> Self {
        self + (rhs - self) * t
    }
    /// Truncate to a [`DVec3`] (drop W).
    #[inline]
    pub const fn truncate(self) -> DVec3 {
        DVec3::new(self.x, self.y, self.z)
    }
    /// As an array.
    #[inline]
    pub const fn to_array(self) -> [f64; 4] {
        [self.x, self.y, self.z, self.w]
    }
    /// From an array.
    #[inline]
    pub const fn from_array(a: [f64; 4]) -> Self {
        Self::new(a[0], a[1], a[2], a[3])
    }
    /// Lossy conversion to the `f32` [`Vec4`].
    #[inline]
    pub fn as_vec4(self) -> Vec4 {
        Vec4::new(self.x as f32, self.y as f32, self.z as f32, self.w as f32)
    }
}

#[cfg(test)]
mod tests {
    use super::{DVec2, DVec3, DVec4};

    #[test]
    fn lengths_and_normalization_scale_without_overflow() {
        let v2 = DVec2::new(8.0e307, -8.0e307);
        let v3 = DVec3::new(8.0e307, -8.0e307, 8.0e307);
        let v4 = DVec4::new(8.0e307, -8.0e307, 8.0e307, -8.0e307);
        assert!(v2.length().is_finite());
        assert!(v3.length().is_finite());
        assert!(v4.length().is_finite());
        assert!((v2.normalize().length() - 1.0).abs() < 1.0e-15);
        assert!((v3.normalize().length() - 1.0).abs() < 1.0e-15);
        assert!((v4.normalize().length() - 1.0).abs() < 1.0e-15);
    }

    #[test]
    fn tiny_nonzero_vectors_normalize_without_underflow() {
        let v = DVec3::new(5.0e-320, -5.0e-320, 0.0).normalize();
        assert!(v.is_finite());
        assert!((v.length() - 1.0).abs() < 1.0e-15);
        let normalized = DVec3::new(5.0e-320, -5.0e-320, 0.0).normalize_or_zero();
        assert_eq!(normalized, DVec3::ZERO);
    }

    #[test]
    fn non_finite_lengths_and_normalization_fail_closed() {
        assert!(DVec2::new(f64::NAN, 1.0).length().is_nan());
        assert!(DVec3::new(f64::INFINITY, 1.0, 0.0).length().is_infinite());
        assert_eq!(DVec4::new(f64::NAN, 0.0, 0.0, 0.0).normalize(), DVec4::ZERO);
        assert_eq!(
            DVec4::new(f64::INFINITY, 0.0, 0.0, 0.0).normalize(),
            DVec4::ZERO
        );
    }
}

// ---- f32 -> f64 widening conversions (mirror of the `as_vecN` narrowings) --

impl Vec2 {
    /// Widen to the `f64` [`DVec2`].
    #[inline]
    pub fn as_dvec2(self) -> DVec2 {
        DVec2::new(self.x as f64, self.y as f64)
    }
}
impl Vec3 {
    /// Widen to the `f64` [`DVec3`].
    #[inline]
    pub fn as_dvec3(self) -> DVec3 {
        DVec3::new(self.x as f64, self.y as f64, self.z as f64)
    }
}
impl Vec4 {
    /// Widen to the `f64` [`DVec4`].
    #[inline]
    pub fn as_dvec4(self) -> DVec4 {
        DVec4::new(self.x as f64, self.y as f64, self.z as f64, self.w as f64)
    }
}

// ---- operator impls via a shared macro ------------------------------------

macro_rules! impl_dvec_ops {
    ($ty:ty { $($c:ident),+ }) => {
        impl Add for $ty {
            type Output = $ty;
            #[inline]
            fn add(self, r: $ty) -> $ty { Self { $($c: self.$c + r.$c),+ } }
        }
        impl Sub for $ty {
            type Output = $ty;
            #[inline]
            fn sub(self, r: $ty) -> $ty { Self { $($c: self.$c - r.$c),+ } }
        }
        impl Mul for $ty {
            type Output = $ty;
            #[inline]
            fn mul(self, r: $ty) -> $ty { Self { $($c: self.$c * r.$c),+ } }
        }
        impl Div for $ty {
            type Output = $ty;
            #[inline]
            fn div(self, r: $ty) -> $ty { Self { $($c: self.$c / r.$c),+ } }
        }
        impl Mul<f64> for $ty {
            type Output = $ty;
            #[inline]
            fn mul(self, s: f64) -> $ty { Self { $($c: self.$c * s),+ } }
        }
        impl Mul<$ty> for f64 {
            type Output = $ty;
            #[inline]
            fn mul(self, v: $ty) -> $ty { <$ty>::new($(self * v.$c),+) }
        }
        impl Div<f64> for $ty {
            type Output = $ty;
            #[inline]
            fn div(self, s: f64) -> $ty { Self { $($c: self.$c / s),+ } }
        }
        impl Neg for $ty {
            type Output = $ty;
            #[inline]
            fn neg(self) -> $ty { Self { $($c: -self.$c),+ } }
        }
        impl AddAssign for $ty {
            #[inline]
            fn add_assign(&mut self, r: $ty) { $(self.$c += r.$c;)+ }
        }
        impl SubAssign for $ty {
            #[inline]
            fn sub_assign(&mut self, r: $ty) { $(self.$c -= r.$c;)+ }
        }
        impl MulAssign<f64> for $ty {
            #[inline]
            fn mul_assign(&mut self, s: f64) { $(self.$c *= s;)+ }
        }
        impl DivAssign<f64> for $ty {
            #[inline]
            fn div_assign(&mut self, s: f64) { $(self.$c /= s;)+ }
        }
    };
}

impl_dvec_ops!(DVec2 { x, y });
impl_dvec_ops!(DVec3 { x, y, z });
impl_dvec_ops!(DVec4 { x, y, z, w });

impl Index<usize> for DVec3 {
    type Output = f64;
    #[inline]
    fn index(&self, i: usize) -> &f64 {
        match i {
            0 => &self.x,
            1 => &self.y,
            2 => &self.z,
            _ => panic!("DVec3 index out of range: {i}"),
        }
    }
}
impl IndexMut<usize> for DVec3 {
    #[inline]
    fn index_mut(&mut self, i: usize) -> &mut f64 {
        match i {
            0 => &mut self.x,
            1 => &mut self.y,
            2 => &mut self.z,
            _ => panic!("DVec3 index out of range: {i}"),
        }
    }
}

impl core::fmt::Debug for DVec2 {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "DVec2({}, {})", self.x, self.y)
    }
}
impl core::fmt::Debug for DVec3 {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "DVec3({}, {}, {})", self.x, self.y, self.z)
    }
}
impl core::fmt::Debug for DVec4 {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "DVec4({}, {}, {}, {})", self.x, self.y, self.z, self.w)
    }
}
