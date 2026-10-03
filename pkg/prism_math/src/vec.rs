//! Floating-point vectors: [`Vec2`], [`Vec3`], [`Vec4`], and the 16-byte
//! aligned [`Vec3A`].
//!
//! Column vectors; matrices multiply on the left (`m * v`). The M0 backend is
//! scalar and is the behavioural reference for later SIMD backends.

use crate::backend;
use crate::float::f32 as mf;
use core::ops::{Add, AddAssign, Div, DivAssign, Index, IndexMut, Mul, MulAssign, Neg, Sub, SubAssign};

/// A 2-component `f32` vector.
#[derive(Clone, Copy, PartialEq, Default)]
#[repr(C)]
pub struct Vec2 {
    /// X component.
    pub x: f32,
    /// Y component.
    pub y: f32,
}

/// A 3-component `f32` vector (tightly packed, 12 bytes).
#[derive(Clone, Copy, PartialEq, Default)]
#[repr(C)]
pub struct Vec3 {
    /// X component.
    pub x: f32,
    /// Y component.
    pub y: f32,
    /// Z component.
    pub z: f32,
}

/// A 3-component `f32` vector aligned to 16 bytes for SIMD-friendly layout.
#[derive(Clone, Copy, PartialEq, Default)]
#[repr(C, align(16))]
pub struct Vec3A {
    /// X component.
    pub x: f32,
    /// Y component.
    pub y: f32,
    /// Z component.
    pub z: f32,
}

/// A 4-component `f32` vector.
#[derive(Clone, Copy, PartialEq, Default)]
#[repr(C)]
pub struct Vec4 {
    /// X component.
    pub x: f32,
    /// Y component.
    pub y: f32,
    /// Z component.
    pub z: f32,
    /// W component.
    pub w: f32,
}

/// Shorthand constructor for [`Vec2`].
#[inline]
pub const fn vec2(x: f32, y: f32) -> Vec2 {
    Vec2::new(x, y)
}
/// Shorthand constructor for [`Vec3`].
#[inline]
pub const fn vec3(x: f32, y: f32, z: f32) -> Vec3 {
    Vec3::new(x, y, z)
}
/// Shorthand constructor for [`Vec3A`].
#[inline]
pub const fn vec3a(x: f32, y: f32, z: f32) -> Vec3A {
    Vec3A::new(x, y, z)
}
/// Shorthand constructor for [`Vec4`].
#[inline]
pub const fn vec4(x: f32, y: f32, z: f32, w: f32) -> Vec4 {
    Vec4::new(x, y, z, w)
}

impl Vec2 {
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
    pub const fn new(x: f32, y: f32) -> Self {
        Self { x, y }
    }
    /// Broadcast a scalar to every component.
    #[inline]
    pub const fn splat(v: f32) -> Self {
        Self { x: v, y: v }
    }
    /// Dot product.
    #[inline]
    pub fn dot(self, rhs: Self) -> f32 {
        self.x * rhs.x + self.y * rhs.y
    }
    /// Squared length.
    #[inline]
    pub fn length_squared(self) -> f32 {
        self.dot(self)
    }
    /// Euclidean length.
    #[inline]
    pub fn length(self) -> f32 {
        mf::sqrt(self.length_squared())
    }
    /// Distance to `rhs`.
    #[inline]
    pub fn distance(self, rhs: Self) -> f32 {
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
        if len > 1.0e-20 { self * (1.0 / len) } else { Self::ZERO }
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
    pub fn lerp(self, rhs: Self, t: f32) -> Self {
        self + (rhs - self) * t
    }
    /// Horizontal sum of components.
    #[inline]
    pub fn element_sum(self) -> f32 {
        self.x + self.y
    }
    /// 2D perpendicular dot (`self.x*rhs.y - self.y*rhs.x`).
    #[inline]
    pub fn perp_dot(self, rhs: Self) -> f32 {
        self.x * rhs.y - self.y * rhs.x
    }
    /// Extend to a [`Vec3`].
    #[inline]
    pub const fn extend(self, z: f32) -> Vec3 {
        Vec3::new(self.x, self.y, z)
    }
    /// As an array.
    #[inline]
    pub const fn to_array(self) -> [f32; 2] {
        [self.x, self.y]
    }
    /// From an array.
    #[inline]
    pub const fn from_array(a: [f32; 2]) -> Self {
        Self::new(a[0], a[1])
    }
}

impl Vec3 {
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
    pub const fn new(x: f32, y: f32, z: f32) -> Self {
        Self { x, y, z }
    }
    /// Broadcast a scalar to every component.
    #[inline]
    pub const fn splat(v: f32) -> Self {
        Self { x: v, y: v, z: v }
    }
    /// Dot product.
    #[inline]
    pub fn dot(self, rhs: Self) -> f32 {
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
    pub fn length_squared(self) -> f32 {
        self.dot(self)
    }
    /// Euclidean length.
    #[inline]
    pub fn length(self) -> f32 {
        mf::sqrt(self.length_squared())
    }
    /// Distance to `rhs`.
    #[inline]
    pub fn distance(self, rhs: Self) -> f32 {
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
        if len > 1.0e-20 { self * (1.0 / len) } else { Self::ZERO }
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
    pub fn lerp(self, rhs: Self, t: f32) -> Self {
        self + (rhs - self) * t
    }
    /// Horizontal sum of components.
    #[inline]
    pub fn element_sum(self) -> f32 {
        self.x + self.y + self.z
    }
    /// Largest component.
    #[inline]
    pub fn max_element(self) -> f32 {
        self.x.max(self.y).max(self.z)
    }
    /// Smallest component.
    #[inline]
    pub fn min_element(self) -> f32 {
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
    /// Truncate to a [`Vec2`] (drop Z).
    #[inline]
    pub const fn truncate(self) -> Vec2 {
        Vec2::new(self.x, self.y)
    }
    /// Extend to a [`Vec4`].
    #[inline]
    pub const fn extend(self, w: f32) -> Vec4 {
        Vec4::new(self.x, self.y, self.z, w)
    }
    /// Reinterpret as the 16-byte aligned [`Vec3A`].
    #[inline]
    pub const fn to_vec3a(self) -> Vec3A {
        Vec3A::new(self.x, self.y, self.z)
    }
    /// As an array.
    #[inline]
    pub const fn to_array(self) -> [f32; 3] {
        [self.x, self.y, self.z]
    }
    /// From an array.
    #[inline]
    pub const fn from_array(a: [f32; 3]) -> Self {
        Self::new(a[0], a[1], a[2])
    }
}

impl Vec3A {
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

    /// Create a new vector.
    #[inline]
    pub const fn new(x: f32, y: f32, z: f32) -> Self {
        Self { x, y, z }
    }
    /// Broadcast a scalar to every component.
    #[inline]
    pub const fn splat(v: f32) -> Self {
        Self { x: v, y: v, z: v }
    }
    /// Convert from a packed [`Vec3`].
    #[inline]
    pub const fn from_vec3(v: Vec3) -> Self {
        Self::new(v.x, v.y, v.z)
    }
    /// Convert to a packed [`Vec3`].
    #[inline]
    pub const fn to_vec3(self) -> Vec3 {
        Vec3::new(self.x, self.y, self.z)
    }
    /// Dot product.
    #[inline]
    pub fn dot(self, rhs: Self) -> f32 {
        backend::vec3_dot(self.to_simd(), rhs.to_simd())
    }
    /// Cross product.
    #[inline]
    pub fn cross(self, rhs: Self) -> Self {
        Self::from_vec3(self.to_vec3().cross(rhs.to_vec3()))
    }
    /// Squared length.
    #[inline]
    pub fn length_squared(self) -> f32 {
        self.dot(self)
    }
    /// Euclidean length.
    #[inline]
    pub fn length(self) -> f32 {
        backend::vec3_length(self.to_simd())
    }
    /// Normalize to unit length.
    #[inline]
    pub fn normalize(self) -> Self {
        Self::from_simd(backend::vec3_normalize(self.to_simd()))
    }
    /// Pack into SIMD lane order `[x, y, z, 0]` (padding lane cleared).
    #[inline]
    fn to_simd(self) -> [f32; 4] {
        [self.x, self.y, self.z, 0.0]
    }
    /// Rebuild from SIMD lanes `[x, y, z, _]` (padding lane dropped).
    #[inline]
    fn from_simd(a: [f32; 4]) -> Self {
        Self { x: a[0], y: a[1], z: a[2] }
    }
    /// Linear interpolation.
    #[inline]
    pub fn lerp(self, rhs: Self, t: f32) -> Self {
        self + (rhs - self) * t
    }
}

impl Vec4 {
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
    pub const fn new(x: f32, y: f32, z: f32, w: f32) -> Self {
        Self { x, y, z, w }
    }
    /// Broadcast a scalar to every component.
    #[inline]
    pub const fn splat(v: f32) -> Self {
        Self { x: v, y: v, z: v, w: v }
    }
    /// Dot product.
    #[inline]
    pub fn dot(self, rhs: Self) -> f32 {
        backend::vec4_dot(self.to_simd(), rhs.to_simd())
    }
    /// Squared length.
    #[inline]
    pub fn length_squared(self) -> f32 {
        self.dot(self)
    }
    /// Euclidean length.
    #[inline]
    pub fn length(self) -> f32 {
        backend::vec4_length(self.to_simd())
    }
    /// Normalize to unit length.
    #[inline]
    pub fn normalize(self) -> Self {
        Self::from_simd(backend::vec4_normalize(self.to_simd()))
    }
    /// Pack into SIMD lane order `[x, y, z, w]`.
    #[inline]
    fn to_simd(self) -> [f32; 4] {
        [self.x, self.y, self.z, self.w]
    }
    /// Rebuild from SIMD lanes `[x, y, z, w]`.
    #[inline]
    fn from_simd(a: [f32; 4]) -> Self {
        Self { x: a[0], y: a[1], z: a[2], w: a[3] }
    }
    /// True if all components are finite.
    #[inline]
    pub fn is_finite(self) -> bool {
        self.x.is_finite() && self.y.is_finite() && self.z.is_finite() && self.w.is_finite()
    }
    /// Linear interpolation.
    #[inline]
    pub fn lerp(self, rhs: Self, t: f32) -> Self {
        self + (rhs - self) * t
    }
    /// Truncate to a [`Vec3`] (drop W).
    #[inline]
    pub const fn truncate(self) -> Vec3 {
        Vec3::new(self.x, self.y, self.z)
    }
    /// As an array.
    #[inline]
    pub const fn to_array(self) -> [f32; 4] {
        [self.x, self.y, self.z, self.w]
    }
    /// From an array.
    #[inline]
    pub const fn from_array(a: [f32; 4]) -> Self {
        Self::new(a[0], a[1], a[2], a[3])
    }
}

// ---- operator impls via macros -------------------------------------------

macro_rules! impl_vec_ops {
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
        impl Mul<f32> for $ty {
            type Output = $ty;
            #[inline]
            fn mul(self, s: f32) -> $ty { Self { $($c: self.$c * s),+ } }
        }
        impl Mul<$ty> for f32 {
            type Output = $ty;
            #[inline]
            fn mul(self, v: $ty) -> $ty { <$ty>::new($(self * v.$c),+) }
        }
        impl Div<f32> for $ty {
            type Output = $ty;
            #[inline]
            fn div(self, s: f32) -> $ty { Self { $($c: self.$c / s),+ } }
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
        impl MulAssign<f32> for $ty {
            #[inline]
            fn mul_assign(&mut self, s: f32) { $(self.$c *= s;)+ }
        }
        impl DivAssign<f32> for $ty {
            #[inline]
            fn div_assign(&mut self, s: f32) { $(self.$c /= s;)+ }
        }
    };
}

impl_vec_ops!(Vec2 { x, y });
impl_vec_ops!(Vec3 { x, y, z });

// `Vec3A` and `Vec4` route their hot operators through the SIMD backend. The
// componentwise ops operate on 4 lanes (`Vec3A` keeps lane 3 at `0.0`), while
// the per-scalar / assignment helpers reuse those routed operators.
macro_rules! impl_simd_vec_ops {
    ($ty:ty) => {
        impl Add for $ty {
            type Output = $ty;
            #[inline]
            fn add(self, r: $ty) -> $ty {
                Self::from_simd(backend::vec4_add(self.to_simd(), r.to_simd()))
            }
        }
        impl Sub for $ty {
            type Output = $ty;
            #[inline]
            fn sub(self, r: $ty) -> $ty {
                Self::from_simd(backend::vec4_sub(self.to_simd(), r.to_simd()))
            }
        }
        impl Mul for $ty {
            type Output = $ty;
            #[inline]
            fn mul(self, r: $ty) -> $ty {
                Self::from_simd(backend::vec4_mul(self.to_simd(), r.to_simd()))
            }
        }
        impl Div for $ty {
            type Output = $ty;
            #[inline]
            fn div(self, r: $ty) -> $ty {
                Self::from_simd(backend::vec4_div(self.to_simd(), r.to_simd()))
            }
        }
        impl Mul<f32> for $ty {
            type Output = $ty;
            #[inline]
            fn mul(self, s: f32) -> $ty {
                Self::from_simd(backend::vec4_scale(self.to_simd(), s))
            }
        }
        impl Mul<$ty> for f32 {
            type Output = $ty;
            #[inline]
            fn mul(self, v: $ty) -> $ty {
                v * self
            }
        }
        impl Div<f32> for $ty {
            type Output = $ty;
            #[inline]
            fn div(self, s: f32) -> $ty {
                Self::from_simd(backend::vec4_div(self.to_simd(), [s, s, s, s]))
            }
        }
        impl Neg for $ty {
            type Output = $ty;
            #[inline]
            fn neg(self) -> $ty {
                Self::from_simd(backend::vec4_scale(self.to_simd(), -1.0))
            }
        }
        impl AddAssign for $ty {
            #[inline]
            fn add_assign(&mut self, r: $ty) {
                *self = *self + r;
            }
        }
        impl SubAssign for $ty {
            #[inline]
            fn sub_assign(&mut self, r: $ty) {
                *self = *self - r;
            }
        }
        impl MulAssign<f32> for $ty {
            #[inline]
            fn mul_assign(&mut self, s: f32) {
                *self = *self * s;
            }
        }
        impl DivAssign<f32> for $ty {
            #[inline]
            fn div_assign(&mut self, s: f32) {
                *self = *self / s;
            }
        }
    };
}

impl_simd_vec_ops!(Vec3A);
impl_simd_vec_ops!(Vec4);

impl Index<usize> for Vec3 {
    type Output = f32;
    #[inline]
    fn index(&self, i: usize) -> &f32 {
        match i {
            0 => &self.x,
            1 => &self.y,
            2 => &self.z,
            _ => panic!("Vec3 index out of range: {i}"),
        }
    }
}
impl IndexMut<usize> for Vec3 {
    #[inline]
    fn index_mut(&mut self, i: usize) -> &mut f32 {
        match i {
            0 => &mut self.x,
            1 => &mut self.y,
            2 => &mut self.z,
            _ => panic!("Vec3 index out of range: {i}"),
        }
    }
}

impl core::fmt::Debug for Vec2 {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "Vec2({}, {})", self.x, self.y)
    }
}
impl core::fmt::Debug for Vec3 {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "Vec3({}, {}, {})", self.x, self.y, self.z)
    }
}
impl core::fmt::Debug for Vec3A {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "Vec3A({}, {}, {})", self.x, self.y, self.z)
    }
}
impl core::fmt::Debug for Vec4 {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "Vec4({}, {}, {}, {})", self.x, self.y, self.z, self.w)
    }
}

impl From<Vec3> for Vec3A {
    #[inline]
    fn from(v: Vec3) -> Self {
        Self::from_vec3(v)
    }
}
impl From<Vec3A> for Vec3 {
    #[inline]
    fn from(v: Vec3A) -> Self {
        v.to_vec3()
    }
}
