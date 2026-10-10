//! Fixed-point vectors [`FxVec2`], [`FxVec3`], [`FxVec4`] built on [`Fixed`].
//!
//! These are the deterministic analogues of the `f32` [`Vec2`](crate::Vec2) /
//! [`Vec3`](crate::Vec3) / [`Vec4`](crate::Vec4) facade types: every component
//! is a Q32.32 [`Fixed`], so all vector algebra is bit-exact across platforms.
//! They power the fixed-point position/velocity state used by rollback
//! netcode and server-authoritative simulation (design doc §13).

use super::Fixed;
use core::ops::{Add, AddAssign, Mul, Neg, Sub, SubAssign};

/// A 2-component fixed-point vector.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Default, Debug)]
#[repr(C)]
pub struct FxVec2 {
    /// X component.
    pub x: Fixed,
    /// Y component.
    pub y: Fixed,
}

/// A 3-component fixed-point vector.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Default, Debug)]
#[repr(C)]
pub struct FxVec3 {
    /// X component.
    pub x: Fixed,
    /// Y component.
    pub y: Fixed,
    /// Z component.
    pub z: Fixed,
}

/// A 4-component fixed-point vector.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Default, Debug)]
#[repr(C)]
pub struct FxVec4 {
    /// X component.
    pub x: Fixed,
    /// Y component.
    pub y: Fixed,
    /// Z component.
    pub z: Fixed,
    /// W component.
    pub w: Fixed,
}

/// Shorthand constructor for [`FxVec2`].
#[inline]
pub const fn fxvec2(x: Fixed, y: Fixed) -> FxVec2 {
    FxVec2::new(x, y)
}
/// Shorthand constructor for [`FxVec3`].
#[inline]
pub const fn fxvec3(x: Fixed, y: Fixed, z: Fixed) -> FxVec3 {
    FxVec3::new(x, y, z)
}
/// Shorthand constructor for [`FxVec4`].
#[inline]
pub const fn fxvec4(x: Fixed, y: Fixed, z: Fixed, w: Fixed) -> FxVec4 {
    FxVec4::new(x, y, z, w)
}

impl FxVec2 {
    /// All zeros.
    pub const ZERO: Self = Self::splat(Fixed::ZERO);
    /// All ones.
    pub const ONE: Self = Self::splat(Fixed::ONE);
    /// Unit X.
    pub const X: Self = Self::new(Fixed::ONE, Fixed::ZERO);
    /// Unit Y.
    pub const Y: Self = Self::new(Fixed::ZERO, Fixed::ONE);

    /// Create a new vector.
    #[inline]
    pub const fn new(x: Fixed, y: Fixed) -> Self {
        Self { x, y }
    }
    /// Broadcast a scalar to every component.
    #[inline]
    pub const fn splat(v: Fixed) -> Self {
        Self { x: v, y: v }
    }
    /// Dot product.
    #[inline]
    pub fn dot(self, rhs: Self) -> Fixed {
        self.x * rhs.x + self.y * rhs.y
    }
    /// Squared length.
    #[inline]
    pub fn length_squared(self) -> Fixed {
        self.dot(self)
    }
    /// Euclidean length (deterministic fixed-point `sqrt`).
    #[inline]
    pub fn length(self) -> Fixed {
        self.length_squared().sqrt()
    }
    /// Component-wise linear interpolation.
    #[inline]
    pub fn lerp(self, rhs: Self, t: Fixed) -> Self {
        self + (rhs - self) * t
    }
    /// Normalize, or return zero when below `min_len` (avoids divide-by-zero).
    #[inline]
    pub fn normalize_or_zero(self, min_len: Fixed) -> Self {
        let len = self.length();
        if len > min_len {
            self * len.recip()
        } else {
            Self::ZERO
        }
    }
    /// As an array of raw Q32.32 integers (for hashing / serialization).
    #[inline]
    pub const fn to_bits(self) -> [i64; 2] {
        [self.x.to_bits(), self.y.to_bits()]
    }
}

impl FxVec3 {
    /// All zeros.
    pub const ZERO: Self = Self::splat(Fixed::ZERO);
    /// All ones.
    pub const ONE: Self = Self::splat(Fixed::ONE);
    /// Unit X.
    pub const X: Self = Self::new(Fixed::ONE, Fixed::ZERO, Fixed::ZERO);
    /// Unit Y.
    pub const Y: Self = Self::new(Fixed::ZERO, Fixed::ONE, Fixed::ZERO);
    /// Unit Z.
    pub const Z: Self = Self::new(Fixed::ZERO, Fixed::ZERO, Fixed::ONE);

    /// Create a new vector.
    #[inline]
    pub const fn new(x: Fixed, y: Fixed, z: Fixed) -> Self {
        Self { x, y, z }
    }
    /// Broadcast a scalar to every component.
    #[inline]
    pub const fn splat(v: Fixed) -> Self {
        Self { x: v, y: v, z: v }
    }
    /// Dot product.
    #[inline]
    pub fn dot(self, rhs: Self) -> Fixed {
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
    pub fn length_squared(self) -> Fixed {
        self.dot(self)
    }
    /// Euclidean length (deterministic fixed-point `sqrt`).
    #[inline]
    pub fn length(self) -> Fixed {
        self.length_squared().sqrt()
    }
    /// Component-wise linear interpolation.
    #[inline]
    pub fn lerp(self, rhs: Self, t: Fixed) -> Self {
        self + (rhs - self) * t
    }
    /// Normalize, or return zero when below `min_len` (avoids divide-by-zero).
    #[inline]
    pub fn normalize_or_zero(self, min_len: Fixed) -> Self {
        let len = self.length();
        if len > min_len {
            self * len.recip()
        } else {
            Self::ZERO
        }
    }
    /// As an array of raw Q32.32 integers (for hashing / serialization).
    #[inline]
    pub const fn to_bits(self) -> [i64; 3] {
        [self.x.to_bits(), self.y.to_bits(), self.z.to_bits()]
    }
}

impl FxVec4 {
    /// All zeros.
    pub const ZERO: Self = Self::splat(Fixed::ZERO);
    /// All ones.
    pub const ONE: Self = Self::splat(Fixed::ONE);

    /// Create a new vector.
    #[inline]
    pub const fn new(x: Fixed, y: Fixed, z: Fixed, w: Fixed) -> Self {
        Self { x, y, z, w }
    }
    /// Broadcast a scalar to every component.
    #[inline]
    pub const fn splat(v: Fixed) -> Self {
        Self {
            x: v,
            y: v,
            z: v,
            w: v,
        }
    }
    /// Dot product.
    #[inline]
    pub fn dot(self, rhs: Self) -> Fixed {
        self.x * rhs.x + self.y * rhs.y + self.z * rhs.z + self.w * rhs.w
    }
    /// Squared length.
    #[inline]
    pub fn length_squared(self) -> Fixed {
        self.dot(self)
    }
    /// Euclidean length (deterministic fixed-point `sqrt`).
    #[inline]
    pub fn length(self) -> Fixed {
        self.length_squared().sqrt()
    }
    /// Component-wise linear interpolation.
    #[inline]
    pub fn lerp(self, rhs: Self, t: Fixed) -> Self {
        self + (rhs - self) * t
    }
    /// As an array of raw Q32.32 integers (for hashing / serialization).
    #[inline]
    pub const fn to_bits(self) -> [i64; 4] {
        [
            self.x.to_bits(),
            self.y.to_bits(),
            self.z.to_bits(),
            self.w.to_bits(),
        ]
    }
}

// -- operator impls ---------------------------------------------------------

macro_rules! impl_vec_ops {
    ($Ty:ident { $($field:ident),+ }) => {
        impl Add for $Ty {
            type Output = Self;
            #[inline]
            fn add(self, rhs: Self) -> Self {
                Self { $($field: self.$field + rhs.$field),+ }
            }
        }
        impl Sub for $Ty {
            type Output = Self;
            #[inline]
            fn sub(self, rhs: Self) -> Self {
                Self { $($field: self.$field - rhs.$field),+ }
            }
        }
        impl Neg for $Ty {
            type Output = Self;
            #[inline]
            fn neg(self) -> Self {
                Self { $($field: -self.$field),+ }
            }
        }
        impl Mul<Fixed> for $Ty {
            type Output = Self;
            #[inline]
            fn mul(self, rhs: Fixed) -> Self {
                Self { $($field: self.$field * rhs),+ }
            }
        }
        impl Mul<$Ty> for Fixed {
            type Output = $Ty;
            #[inline]
            fn mul(self, rhs: $Ty) -> $Ty {
                $Ty { $($field: self * rhs.$field),+ }
            }
        }
        impl AddAssign for $Ty {
            #[inline]
            fn add_assign(&mut self, rhs: Self) {
                *self = *self + rhs;
            }
        }
        impl SubAssign for $Ty {
            #[inline]
            fn sub_assign(&mut self, rhs: Self) {
                *self = *self - rhs;
            }
        }
    };
}

impl_vec_ops!(FxVec2 { x, y });
impl_vec_ops!(FxVec3 { x, y, z });
impl_vec_ops!(FxVec4 { x, y, z, w });
