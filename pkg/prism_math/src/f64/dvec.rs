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
use core::ops::{Add, AddAssign, Div, DivAssign, Index, IndexMut, Mul, MulAssign, Neg, Sub, SubAssign};

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
        mf::sqrt(self.length_squared())
    }
    /// Distance to `rhs`.
    #[inline]
    pub fn distance(self, rhs: Self) -> f64 {
        (self - rhs).length()
    }
    /// Normalize to unit length (panics in debug if non-finite result).
    #[inline]
    pub fn normalize(self) -> Self {
        let inv = 1.0 / self.length();
        let n = self * inv;
        debug_assert!(n.is_finite(), "normalize of near-zero vector");
        n
    }
    /// Normalize, or return zero if the length is too small to normalize.
    #[inline]
    pub fn normalize_or_zero(self) -> Self {
        let len = self.length();
        if len > 1.0e-200 { self * (1.0 / len) } else { Self::ZERO }
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
        mf::sqrt(self.length_squared())
    }
    /// Distance to `rhs`.
    #[inline]
    pub fn distance(self, rhs: Self) -> f64 {
        (self - rhs).length()
    }
    /// Normalize to unit length (panics in debug if non-finite result).
    #[inline]
    pub fn normalize(self) -> Self {
        let inv = 1.0 / self.length();
        let n = self * inv;
        debug_assert!(n.is_finite(), "normalize of near-zero vector");
        n
    }
    /// Normalize, or return zero if too short to normalize.
    #[inline]
    pub fn normalize_or_zero(self) -> Self {
        let len = self.length();
        if len > 1.0e-200 { self * (1.0 / len) } else { Self::ZERO }
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
        Self { x: v, y: v, z: v, w: v }
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
        mf::sqrt(self.length_squared())
    }
    /// Normalize to unit length (panics in debug if non-finite result).
    #[inline]
    pub fn normalize(self) -> Self {
        let inv = 1.0 / self.length();
        let n = self * inv;
        debug_assert!(n.is_finite(), "normalize of near-zero vector");
        n
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
