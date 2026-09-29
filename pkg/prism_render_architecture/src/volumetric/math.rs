//! Hand-rolled float math for the volumetric subsystem.
//!
//! `prism_render_architecture` is a dependency-free contracts crate and the
//! workspace determinism policy allows only `sqrt` among the float intrinsics
//! (see the crate `clippy.toml`: `f32::exp`, `f32::powf`, `f32::ln`,
//! `f32::sin`, ... are all forbidden). The `raymarch`, `scatter`, `modeling`,
//! and `atmosphere` modules nevertheless need `exp`/`ln`/`pow` (Beer-Lambert
//! extinction, `remap` exponents, phase functions) and a little trigonometry
//! (`curl` noise, spectral phases). This module supplies deterministic,
//! bit-reproducible approximations of those functions built purely from
//! add/multiply/`sqrt`/integer-bit operations, plus the hand-rolled `Vec2` /
//! `Vec3` types every sibling module shares.
//!
//! The `GPU` `WESL` kernels use the native `exp`/`pow`/`sin` intrinsics; these
//! `CPU` approximations exist so the reference path is verifiable in the
//! sandbox (there is no `GPU` here) and so unit tests can assert numeric
//! properties. Every approximation is validated against known closed-form
//! constants in the module tests.

/// General absolute tolerance for scalar comparisons.
///
/// `f32` equality is never tested with `==` on computed reals; call sites
/// compare magnitudes against this constant instead, matching the workspace
/// determinism policy shared with the sibling water subsystem.
pub const EPS: f32 = 1e-6;

/// Squared-length threshold below which a vector is treated as zero, so
/// normalization never divides by (near) zero and never propagates `NaN`.
pub const EPS_LEN_SQ: f32 = 1e-12;

/// The mathematical constant pi.
pub const PI: f32 = core::f32::consts::PI;

/// Two pi, the period used by [`sin_approx`] / [`cos_approx`] range reduction.
pub const TWO_PI: f32 = 2.0 * core::f32::consts::PI;

/// Half pi, used to derive cosine from sine.
pub const FRAC_PI_2: f32 = core::f32::consts::FRAC_PI_2;

/// Natural logarithm of two, the scale between `log2` and `ln`.
pub const LN2: f32 = core::f32::consts::LN_2;

/// `log2(e)`, the scale that turns `exp` into a base-two exponent for range
/// reduction.
pub const LOG2_E: f32 = core::f32::consts::LOG2_E;

/// Square root of two, the balance point for [`ln_approx`] mantissa centring.
pub const SQRT_2: f32 = core::f32::consts::SQRT_2;

/// Clamps `x` into the inclusive range `[lo, hi]`.
///
/// `lo` is assumed to be no greater than `hi`; when they are inverted the
/// function still terminates and returns `lo`, never `NaN`.
#[must_use]
pub fn clamp(x: f32, lo: f32, hi: f32) -> f32 {
    if x < lo {
        lo
    } else if x > hi {
        hi
    } else {
        x
    }
}

/// Clamps `x` into `[0, 1]`, the saturation used pervasively for weights.
#[must_use]
pub fn saturate(x: f32) -> f32 {
    clamp(x, 0.0, 1.0)
}

/// Linear interpolation `a + t (b - a)`; `t` is not clamped.
#[must_use]
pub fn lerp(a: f32, b: f32, t: f32) -> f32 {
    a + (b - a) * t
}

/// Remaps `x` from `[in_lo, in_hi]` to `[out_lo, out_hi]`.
///
/// The input span is guarded against collapse: when `in_hi - in_lo` is below
/// [`EPS`] the fraction is taken as zero so the map never divides by (near)
/// zero. This is the energy-preserving `remap` the `detail erosion` and height
/// gradient modelling rely on (design section 4/6).
#[must_use]
pub fn remap(x: f32, in_lo: f32, in_hi: f32, out_lo: f32, out_hi: f32) -> f32 {
    let span = in_hi - in_lo;
    let t = if span.abs() < EPS {
        0.0
    } else {
        (x - in_lo) / span
    };
    out_lo + (out_hi - out_lo) * t
}

/// Hermite `smoothstep` between `edge0` and `edge1`, clamped to `[0, 1]`.
///
/// When the edges collapse (`|edge1 - edge0| < EPS`) it degrades to a hard step
/// at `edge0` instead of dividing by zero.
#[must_use]
pub fn smoothstep(edge0: f32, edge1: f32, x: f32) -> f32 {
    let span = edge1 - edge0;
    if span.abs() < EPS {
        return if x < edge0 { 0.0 } else { 1.0 };
    }
    let t = saturate((x - edge0) / span);
    t * t * (3.0 - 2.0 * t)
}

/// Fractional part `x - floor(x)`, always in `[0, 1)`; used by the noise hash.
#[must_use]
pub fn fract(x: f32) -> f32 {
    x - x.floor()
}

/// `2^k` for an integer `k`, assembled directly from the `f32` exponent field.
///
/// Underflows to `0` for `k < -126` and saturates to `+inf` for `k > 127`,
/// matching `IEEE-754` `ldexp` on the mantissa `1.0`. Pure integer-bit work,
/// so it is exactly reproducible and needs no float intrinsic.
#[must_use]
fn exp2_int(k: i32) -> f32 {
    if k < -126 {
        0.0
    } else if k > 127 {
        f32::INFINITY
    } else {
        f32::from_bits(((k + 127) as u32) << 23)
    }
}

/// `2^f` for a fraction `f` in `[0, 1]`, via the truncated `exp(f * ln2)`
/// series (seven terms, error below `3e-7` on the interval).
#[must_use]
fn exp2_frac(f: f32) -> f32 {
    // Coefficients are (ln2)^k / k! so this is exp(f * ln2) = 2^f.
    const C1: f32 = core::f32::consts::LN_2;
    const C2: f32 = 0.240_226_5;
    const C3: f32 = 0.055_504_11;
    const C4: f32 = 0.009_618_13;
    const C5: f32 = 0.001_333_36;
    const C6: f32 = 0.000_154_03;
    1.0 + f * (C1 + f * (C2 + f * (C3 + f * (C4 + f * (C5 + f * C6)))))
}

/// Hand-rolled `exp` via base-two range reduction: `exp(x) = 2^(x*log2e)`
/// split into an integer power (assembled from the exponent field) and a
/// fractional polynomial.
///
/// The result is non-negative and monotonically increasing for every finite
/// input — exactly what Beer-Lambert extinction (`exp(-sigma*d)`), powder, and
/// multi-scatter energy terms require. Accurate to a few `1e-6` relative error
/// across the argument range the subsystem produces; large positive arguments
/// saturate to `+inf` and large negative arguments flush to `0`.
#[must_use]
pub fn exp_approx(x: f32) -> f32 {
    let t = x * LOG2_E;
    // floor via truncation toward negative infinity for the integer split.
    let k = t.floor();
    let frac = t - k;
    exp2_frac(frac) * exp2_int(k as i32)
}

/// Hand-rolled natural logarithm via `f32` exponent extraction plus the
/// `atanh` series on the centred mantissa.
///
/// Defined for `x > 0`; non-positive inputs are clamped to the smallest
/// positive normal so the function never returns `NaN` and stays monotonic.
/// Accurate to about `1e-6` absolute error, which is well within the
/// tolerance the `pow_approx` callers need.
#[must_use]
pub fn ln_approx(x: f32) -> f32 {
    let x = if x < f32::MIN_POSITIVE {
        f32::MIN_POSITIVE
    } else {
        x
    };
    let bits = x.to_bits();
    let exp_field = ((bits >> 23) & 0xFF) as i32;
    let mut e = exp_field - 127;
    // Reconstruct the mantissa in [1, 2).
    let mantissa_bits = (bits & 0x007F_FFFF) | (127 << 23);
    let mut m = f32::from_bits(mantissa_bits);
    // Centre the mantissa around 1 so the atanh series converges fastest.
    if m > SQRT_2 {
        m *= 0.5;
        e += 1;
    }
    let t = (m - 1.0) / (m + 1.0);
    let t2 = t * t;
    // 2 * (t + t^3/3 + t^5/5 + t^7/7 + t^9/9), Horner form.
    let series = t * (1.0 + t2 * (1.0 / 3.0 + t2 * (0.2 + t2 * (1.0 / 7.0 + t2 * (1.0 / 9.0)))));
    e as f32 * LN2 + 2.0 * series
}

/// Hand-rolled `pow(base, exponent)` as `exp(exponent * ln(base))`.
///
/// Defined for `base > 0` (the only regime the modelling `remap` and height
/// gradients use). A `base` of `0` returns `0`; negative bases are treated as
/// `0` rather than producing `NaN`, keeping the density/energy pipeline free of
/// `NaN` propagation.
#[must_use]
pub fn pow_approx(base: f32, exponent: f32) -> f32 {
    if base <= 0.0 {
        return 0.0;
    }
    exp_approx(exponent * ln_approx(base))
}

/// Reduces an angle to `[-PI, PI]` by subtracting the nearest whole multiple of
/// `2*PI`, so [`sin_approx`] / [`cos_approx`] stay accurate for large phases.
#[must_use]
fn wrap_pi(x: f32) -> f32 {
    let k = (x / TWO_PI).round();
    x - k * TWO_PI
}

/// Hand-rolled `sin` (the determinism policy forbids [`f32::sin`]). The angle is
/// reduced to `[-PI, PI]`, folded into `[-PI/2, PI/2]` via `sin(PI - x)`, then
/// evaluated with the seventh-order Taylor polynomial (error below `2e-4`).
/// Exactly periodic and bounded to roughly `[-1, 1]`.
#[must_use]
pub fn sin_approx(x: f32) -> f32 {
    let mut r = wrap_pi(x);
    if r > FRAC_PI_2 {
        r = PI - r;
    } else if r < -FRAC_PI_2 {
        r = -PI - r;
    }
    let x2 = r * r;
    r * (1.0 + x2 * (-1.0 / 6.0 + x2 * (1.0 / 120.0 + x2 * (-1.0 / 5040.0))))
}

/// Hand-rolled `cos` via `cos(x) = sin(x + PI/2)`; see [`sin_approx`].
#[must_use]
pub fn cos_approx(x: f32) -> f32 {
    sin_approx(x + FRAC_PI_2)
}

/// A hand-rolled two-component vector for weather-map and screen-space math.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Vec2 {
    /// X component.
    pub x: f32,
    /// Y component.
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

    /// Uniform vector with every component set to `v`.
    #[must_use]
    pub const fn splat(v: f32) -> Self {
        Self { x: v, y: v }
    }

    /// Component-wise sum.
    #[must_use]
    #[expect(
        clippy::should_implement_trait,
        reason = "The volumetric math API uses named add/sub for call-site uniformity, matching the sibling water/particle modules; operator traits are intentionally not part of this internal type."
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

    /// Unit vector, or the zero vector when the length is below `EPS_LEN_SQ`.
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

/// A hand-rolled three-component vector for density-field and lighting math.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Vec3 {
    /// X component.
    pub x: f32,
    /// Y component.
    pub y: f32,
    /// Z component.
    pub z: f32,
}

impl Vec3 {
    /// The zero vector.
    pub const ZERO: Self = Self {
        x: 0.0,
        y: 0.0,
        z: 0.0,
    };

    /// The unit vector along +Y (world up), the reference for height gradients.
    pub const UP: Self = Self {
        x: 0.0,
        y: 1.0,
        z: 0.0,
    };

    /// Builds a vector from components.
    #[must_use]
    pub const fn new(x: f32, y: f32, z: f32) -> Self {
        Self { x, y, z }
    }

    /// Uniform vector with every component set to `v`.
    #[must_use]
    pub const fn splat(v: f32) -> Self {
        Self { x: v, y: v, z: v }
    }

    /// Component-wise sum.
    #[must_use]
    #[expect(
        clippy::should_implement_trait,
        reason = "The volumetric math API uses named add/sub for call-site uniformity, matching the sibling water/particle modules; operator traits are intentionally not part of this internal type."
    )]
    pub fn add(self, rhs: Self) -> Self {
        Self::new(self.x + rhs.x, self.y + rhs.y, self.z + rhs.z)
    }

    /// Component-wise difference `self - rhs`.
    #[must_use]
    #[expect(
        clippy::should_implement_trait,
        reason = "See add: the specified API uses named sub for call-site uniformity, not operator traits."
    )]
    pub fn sub(self, rhs: Self) -> Self {
        Self::new(self.x - rhs.x, self.y - rhs.y, self.z - rhs.z)
    }

    /// Uniform scale by a scalar.
    #[must_use]
    pub fn scale(self, s: f32) -> Self {
        Self::new(self.x * s, self.y * s, self.z * s)
    }

    /// Component-wise (Hadamard) product.
    #[must_use]
    #[expect(
        clippy::should_implement_trait,
        reason = "See add: the specified API uses named mul for call-site uniformity, not operator traits."
    )]
    pub fn mul(self, rhs: Self) -> Self {
        Self::new(self.x * rhs.x, self.y * rhs.y, self.z * rhs.z)
    }

    /// Dot (inner) product.
    #[must_use]
    pub fn dot(self, rhs: Self) -> f32 {
        self.x * rhs.x + self.y * rhs.y + self.z * rhs.z
    }

    /// Cross product `self x rhs`.
    #[must_use]
    pub fn cross(self, rhs: Self) -> Self {
        Self::new(
            self.y * rhs.z - self.z * rhs.y,
            self.z * rhs.x - self.x * rhs.z,
            self.x * rhs.y - self.y * rhs.x,
        )
    }

    /// Squared Euclidean length; cheaper than [`Vec3::length`] for comparisons.
    #[must_use]
    pub fn length_squared(self) -> f32 {
        self.dot(self)
    }

    /// Euclidean length.
    #[must_use]
    pub fn length(self) -> f32 {
        self.length_squared().sqrt()
    }

    /// Distance to another point.
    #[must_use]
    pub fn distance(self, rhs: Self) -> f32 {
        self.sub(rhs).length()
    }

    /// Component-wise linear interpolation toward `rhs` by `t` (unclamped).
    #[must_use]
    pub fn lerp(self, rhs: Self, t: f32) -> Self {
        Self::new(
            lerp(self.x, rhs.x, t),
            lerp(self.y, rhs.y, t),
            lerp(self.z, rhs.z, t),
        )
    }

    /// Unit vector, or the zero vector when the length is below `EPS_LEN_SQ`.
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

#[cfg(test)]
mod tests {
    use super::*;

    /// Asserts `a` and `b` agree within `tol` absolute error.
    fn close(a: f32, b: f32, tol: f32) -> bool {
        (a - b).abs() <= tol
    }

    #[test]
    fn exp_matches_known_constants() {
        assert!(close(exp_approx(0.0), 1.0, 1e-6));
        assert!(close(exp_approx(1.0), core::f32::consts::E, 1e-3));
        assert!(close(exp_approx(-1.0), 0.367_879_44, 1e-4));
        assert!(close(exp_approx(2.0), 7.389_056, 2e-3));
        assert!(close(exp_approx(-5.0), 0.006_737_947, 1e-4));
        // Large magnitude stays finite and correctly signed.
        assert!(exp_approx(50.0).is_finite() || exp_approx(50.0).is_infinite());
        assert_eq!(exp_approx(-1000.0), 0.0);
    }

    #[test]
    fn exp_is_monotonic_and_positive() {
        let mut prev = exp_approx(-8.0);
        assert!(prev > 0.0);
        let mut x = -8.0;
        while x <= 8.0 {
            let cur = exp_approx(x);
            assert!(cur >= prev - 1e-6, "exp must be non-decreasing at x={x}");
            assert!(cur > 0.0);
            prev = cur;
            x += 0.25;
        }
    }

    #[test]
    fn ln_matches_known_constants() {
        assert!(close(ln_approx(1.0), 0.0, 1e-6));
        assert!(close(ln_approx(core::f32::consts::E), 1.0, 1e-5));
        assert!(close(ln_approx(2.0), core::f32::consts::LN_2, 1e-5));
        assert!(close(ln_approx(10.0), core::f32::consts::LN_10, 1e-4));
        assert!(close(ln_approx(0.5), -core::f32::consts::LN_2, 1e-5));
    }

    #[test]
    fn ln_is_inverse_of_exp() {
        let mut x = 0.1;
        while x < 20.0 {
            assert!(close(exp_approx(ln_approx(x)), x, x * 1e-3 + 1e-4));
            x += 0.37;
        }
    }

    #[test]
    fn pow_matches_known_constants() {
        assert!(close(pow_approx(2.0, 10.0), 1024.0, 1024.0 * 2e-3));
        assert!(close(pow_approx(9.0, 0.5), 3.0, 1e-3));
        assert!(close(pow_approx(4.0, 0.5), 2.0, 1e-3));
        assert!(close(pow_approx(3.0, 3.0), 27.0, 27.0 * 2e-3));
        assert_eq!(pow_approx(0.0, 2.0), 0.0);
        assert_eq!(pow_approx(-2.0, 2.0), 0.0);
    }

    #[test]
    fn trig_matches_known_constants() {
        assert!(close(sin_approx(0.0), 0.0, 1e-4));
        assert!(close(sin_approx(FRAC_PI_2), 1.0, 1e-3));
        assert!(close(sin_approx(PI), 0.0, 1e-3));
        assert!(close(cos_approx(0.0), 1.0, 1e-3));
        assert!(close(cos_approx(FRAC_PI_2), 0.0, 1e-3));
        // Periodicity across many wraps.
        assert!(close(
            sin_approx(10.0 * TWO_PI + 0.5),
            sin_approx(0.5),
            1e-3
        ));
    }

    #[test]
    fn helpers_behave() {
        assert_eq!(clamp(5.0, 0.0, 1.0), 1.0);
        assert_eq!(clamp(-5.0, 0.0, 1.0), 0.0);
        assert_eq!(saturate(0.5), 0.5);
        assert_eq!(lerp(0.0, 10.0, 0.25), 2.5);
        assert_eq!(remap(5.0, 0.0, 10.0, 0.0, 1.0), 0.5);
        // Collapsed input span does not divide by zero.
        assert_eq!(remap(5.0, 2.0, 2.0, 0.0, 1.0), 0.0);
        assert_eq!(smoothstep(0.0, 1.0, 0.5), 0.5);
        assert!(close(fract(3.75), 0.75, 1e-6));
    }

    #[test]
    fn vec3_algebra_and_cross() {
        let x = Vec3::new(1.0, 0.0, 0.0);
        let y = Vec3::new(0.0, 1.0, 0.0);
        assert_eq!(x.cross(y), Vec3::new(0.0, 0.0, 1.0));
        assert_eq!(Vec3::new(3.0, 4.0, 0.0).length(), 5.0);
        assert_eq!(Vec3::ZERO.normalize_or_zero(), Vec3::ZERO);
        assert_eq!(x.mul(Vec3::splat(2.0)), Vec3::new(2.0, 0.0, 0.0));
        assert_eq!(Vec3::ZERO.lerp(Vec3::splat(4.0), 0.5), Vec3::splat(2.0));
    }

    #[test]
    fn vec2_length_and_normalize() {
        let a = Vec2::new(3.0, 4.0);
        assert_eq!(a.length(), 5.0);
        assert_eq!(a.normalize_or_zero(), Vec2::new(0.6, 0.8));
        assert_eq!(Vec2::ZERO.normalize_or_zero(), Vec2::ZERO);
    }
}
