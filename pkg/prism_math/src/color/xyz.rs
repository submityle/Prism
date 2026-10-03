//! CIE 1931 XYZ color ([`Xyza`]) with a D65 reference white.

/// A color in CIE 1931 **XYZ** tristimulus space (D65 white point), with a
/// straight alpha carried alongside unchanged. XYZ is the device-independent
/// hub used to bridge sRGB primaries and other color models.
#[derive(Clone, Copy, PartialEq, Default, Debug)]
#[repr(C)]
pub struct Xyza {
    /// `X` tristimulus value.
    pub x: f32,
    /// `Y` tristimulus value (luminance).
    pub y: f32,
    /// `Z` tristimulus value.
    pub z: f32,
    /// Alpha (linear).
    pub alpha: f32,
}

impl Xyza {
    /// The D65 reference white in XYZ (`Y` normalized to 1), opaque.
    pub const D65_WHITE: Self = Self::new(0.950_489, 1.0, 1.088_84, 1.0);

    /// Construct from components.
    #[inline]
    pub const fn new(x: f32, y: f32, z: f32, alpha: f32) -> Self {
        Self { x, y, z, alpha }
    }

    /// Components as `[x, y, z, a]`.
    #[inline]
    pub const fn to_array(self) -> [f32; 4] {
        [self.x, self.y, self.z, self.alpha]
    }

    /// Build from `[x, y, z, a]`.
    #[inline]
    pub const fn from_array(a: [f32; 4]) -> Self {
        Self::new(a[0], a[1], a[2], a[3])
    }
}
