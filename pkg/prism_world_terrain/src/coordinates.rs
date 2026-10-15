//! Integer and Q32.32 coordinates with bounded local-float conversion.

/// Fractional bits in the fixed-point representation.
pub const FIXED_FRAC_BITS: u32 = 32;
/// Fixed-point units per whole unit.
pub const FIXED_SCALE: i64 = 1_i64 << FIXED_FRAC_BITS;

/// Errors returned by coordinate conversion.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CoordinateError {
    /// Subtracting the origin would overflow `i64`.
    DifferenceOverflow,
}

/// A signed integer world position.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct I64Point {
    units: [i64; 3],
}

impl I64Point {
    /// Creates an integer world position.
    pub const fn new(units: [i64; 3]) -> Self {
        Self { units }
    }

    /// Returns the raw integer components.
    pub const fn units(self) -> [i64; 3] {
        self.units
    }

    /// Converts after subtracting the integer origin, with a conservative error bound.
    pub fn to_local(self, origin: Self) -> Result<VerifiedLocalPoint, CoordinateError> {
        let mut value = [0.0; 3];
        let mut error: f32 = 0.0;
        for (axis, component) in value.iter_mut().enumerate() {
            let delta = self.units[axis]
                .checked_sub(origin.units[axis])
                .ok_or(CoordinateError::DifferenceOverflow)?;
            *component = delta as f32;
            error = error.max(float_cast_error(*component));
        }
        Ok(VerifiedLocalPoint {
            value,
            max_abs_error: error,
        })
    }
}

/// A signed Q32.32 fixed-point world position.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct FixedPoint3 {
    raw: [i64; 3],
}

impl FixedPoint3 {
    /// Creates a Q32.32 position from raw fixed-point bits.
    pub const fn from_raw(raw: [i64; 3]) -> Self {
        Self { raw }
    }

    /// Returns the raw Q32.32 components.
    pub const fn raw(self) -> [i64; 3] {
        self.raw
    }

    /// Converts after subtracting the fixed-point origin, with a conservative error bound.
    pub fn to_local(self, origin: Self) -> Result<VerifiedLocalPoint, CoordinateError> {
        let mut value = [0.0; 3];
        let mut error: f32 = 0.0;
        for (axis, component) in value.iter_mut().enumerate() {
            let delta = self.raw[axis]
                .checked_sub(origin.raw[axis])
                .ok_or(CoordinateError::DifferenceOverflow)?;
            *component = delta as f32 / FIXED_SCALE as f32;
            error = error.max(float_cast_error(*component));
        }
        Ok(VerifiedLocalPoint {
            value,
            max_abs_error: error,
        })
    }
}

/// A local float position accompanied by a bound on absolute conversion error.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct VerifiedLocalPoint {
    value: [f32; 3],
    max_abs_error: f32,
}

impl VerifiedLocalPoint {
    /// Returns the local float components.
    pub const fn value(self) -> [f32; 3] {
        self.value
    }

    /// Returns a conservative per-component absolute error bound.
    pub const fn max_abs_error(self) -> f32 {
        self.max_abs_error
    }

    /// Tests whether a higher-precision reference point lies within the bound.
    pub fn contains(self, reference: [f64; 3]) -> bool {
        self.value.iter().zip(reference).all(|(value, exact)| {
            (f64::from(*value) - exact).abs() <= f64::from(self.max_abs_error)
        })
    }
}

fn float_cast_error(value: f32) -> f32 {
    (value.abs() + 1.0) * f32::EPSILON
}
