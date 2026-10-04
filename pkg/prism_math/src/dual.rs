//! Forward-mode automatic differentiation via dual numbers.
//!
//! A dual number carries a value together with its first derivative with
//! respect to a single scalar parameter. Evaluating any composition of the
//! operations below simultaneously evaluates the function and its exact
//! analytic derivative, with none of the step-size tuning or cancellation
//! noise of finite differences.
//!
//! This is classical numerical differentiation and has nothing to do with
//! machine learning: it is strictly forward mode, first order, with no reverse
//! pass or training. It is the tool for parametric-curve tangents, analytic
//! surface normals, speed/acceleration from a motion path, and Jacobian
//! columns for inverse kinematics (`IK`) and physics constraints.
//!
//! [`Dual`] is the scalar carrier; [`DualVec3`] propagates a derivative through
//! a 3-vector that varies with one parameter (for example a point on a curve
//! `C(t)` whose derivative is the tangent `C'(t)`).

use crate::float::f32 as mf;
use crate::vec::Vec3;

/// A scalar value paired with its derivative with respect to one parameter.
///
/// `re` is the value `f(t)` and `du` is `f'(t)`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Dual {
    /// Value component `f(t)`.
    pub re: f32,
    /// Derivative component `f'(t)`.
    pub du: f32,
}

impl Dual {
    /// Build a dual number from an explicit value and derivative.
    #[inline]
    #[must_use]
    pub const fn new(re: f32, du: f32) -> Self {
        Self { re, du }
    }

    /// A constant: value `v` with zero derivative.
    #[inline]
    #[must_use]
    pub const fn constant(v: f32) -> Self {
        Self { re: v, du: 0.0 }
    }

    /// The independent variable seeded at `v`: value `v` with unit derivative.
    /// Compose functions of this to read off their derivative in `du`.
    #[inline]
    #[must_use]
    pub const fn variable(v: f32) -> Self {
        Self { re: v, du: 1.0 }
    }

    /// Reciprocal `1/x` with `d(1/x) = -x'/x^2`.
    #[inline]
    #[must_use]
    pub fn recip(self) -> Self {
        let inv = 1.0 / self.re;
        Self {
            re: inv,
            du: -self.du * inv * inv,
        }
    }

    /// Square root with `d(sqrt x) = x'/(2*sqrt x)`.
    #[inline]
    #[must_use]
    pub fn sqrt(self) -> Self {
        let r = mf::sqrt(self.re);
        Self {
            re: r,
            du: self.du / (2.0 * r),
        }
    }

    /// Square `x^2` with `d(x^2) = 2*x*x'`.
    #[inline]
    #[must_use]
    pub fn squared(self) -> Self {
        Self {
            re: self.re * self.re,
            du: 2.0 * self.re * self.du,
        }
    }

    /// Power `x^p` for a constant exponent, with `d(x^p) = p*x^(p-1)*x'`.
    #[inline]
    #[must_use]
    pub fn powf(self, p: f32) -> Self {
        Self {
            re: mf::powf(self.re, p),
            du: p * mf::powf(self.re, p - 1.0) * self.du,
        }
    }

    /// Natural exponential with `d(e^x) = e^x*x'`.
    #[inline]
    #[must_use]
    pub fn exp(self) -> Self {
        let e = mf::exp(self.re);
        Self {
            re: e,
            du: e * self.du,
        }
    }

    /// Natural logarithm with `d(ln x) = x'/x`.
    #[inline]
    #[must_use]
    pub fn ln(self) -> Self {
        Self {
            re: mf::ln(self.re),
            du: self.du / self.re,
        }
    }

    /// Sine with `d(sin x) = cos x * x'`.
    #[inline]
    #[must_use]
    pub fn sin(self) -> Self {
        Self {
            re: mf::sin(self.re),
            du: mf::cos(self.re) * self.du,
        }
    }

    /// Cosine with `d(cos x) = -sin x * x'`.
    #[inline]
    #[must_use]
    pub fn cos(self) -> Self {
        Self {
            re: mf::cos(self.re),
            du: -mf::sin(self.re) * self.du,
        }
    }

    /// Tangent with `d(tan x) = (1 + tan^2 x) * x'`.
    #[inline]
    #[must_use]
    pub fn tan(self) -> Self {
        let t = mf::tan(self.re);
        Self {
            re: t,
            du: (1.0 + t * t) * self.du,
        }
    }

    /// Absolute value with `d(|x|) = sign(x) * x'` (undefined at zero; the
    /// derivative there is taken as zero).
    #[inline]
    #[must_use]
    pub fn abs(self) -> Self {
        let s = if self.re > 0.0 {
            1.0
        } else if self.re < 0.0 {
            -1.0
        } else {
            0.0
        };
        Self {
            re: mf::abs(self.re),
            du: s * self.du,
        }
    }
}

impl From<f32> for Dual {
    #[inline]
    fn from(v: f32) -> Self {
        Self::constant(v)
    }
}

impl core::ops::Neg for Dual {
    type Output = Self;
    #[inline]
    fn neg(self) -> Self {
        Self {
            re: -self.re,
            du: -self.du,
        }
    }
}

impl core::ops::Add for Dual {
    type Output = Self;
    #[inline]
    fn add(self, rhs: Self) -> Self {
        Self {
            re: self.re + rhs.re,
            du: self.du + rhs.du,
        }
    }
}

impl core::ops::Sub for Dual {
    type Output = Self;
    #[inline]
    fn sub(self, rhs: Self) -> Self {
        Self {
            re: self.re - rhs.re,
            du: self.du - rhs.du,
        }
    }
}

impl core::ops::Mul for Dual {
    type Output = Self;
    #[inline]
    fn mul(self, rhs: Self) -> Self {
        // Product rule.
        Self {
            re: self.re * rhs.re,
            du: self.du * rhs.re + self.re * rhs.du,
        }
    }
}

impl core::ops::Div for Dual {
    type Output = Self;
    #[inline]
    fn div(self, rhs: Self) -> Self {
        // Quotient rule.
        let inv = 1.0 / rhs.re;
        Self {
            re: self.re * inv,
            du: (self.du * rhs.re - self.re * rhs.du) * inv * inv,
        }
    }
}

impl core::ops::Mul<f32> for Dual {
    type Output = Self;
    #[inline]
    fn mul(self, rhs: f32) -> Self {
        Self {
            re: self.re * rhs,
            du: self.du * rhs,
        }
    }
}

/// A 3-vector whose components vary with one scalar parameter, carrying both
/// the value and its derivative. For a curve `C(t)`, `value` is the position
/// and `deriv` is the tangent `C'(t)`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DualVec3 {
    /// Value `C(t)`.
    pub value: Vec3,
    /// Derivative `C'(t)`.
    pub deriv: Vec3,
}

impl DualVec3 {
    /// Build from explicit value and derivative vectors.
    #[inline]
    #[must_use]
    pub const fn new(value: Vec3, deriv: Vec3) -> Self {
        Self { value, deriv }
    }

    /// A constant vector with zero derivative.
    #[inline]
    #[must_use]
    pub const fn constant(value: Vec3) -> Self {
        Self {
            value,
            deriv: Vec3::ZERO,
        }
    }

    /// Assemble from three independent [`Dual`] components.
    #[inline]
    #[must_use]
    pub const fn from_components(x: Dual, y: Dual, z: Dual) -> Self {
        Self {
            value: Vec3::new(x.re, y.re, z.re),
            deriv: Vec3::new(x.du, y.du, z.du),
        }
    }

    /// Dot product with the chain rule `d(a.b) = a'.b + a.b'`.
    #[inline]
    #[must_use]
    pub fn dot(self, rhs: Self) -> Dual {
        Dual {
            re: self.value.dot(rhs.value),
            du: self.deriv.dot(rhs.value) + self.value.dot(rhs.deriv),
        }
    }

    /// Cross product with the product rule `d(a x b) = a' x b + a x b'`.
    #[inline]
    #[must_use]
    pub fn cross(self, rhs: Self) -> Self {
        Self {
            value: self.value.cross(rhs.value),
            deriv: self.deriv.cross(rhs.value) + self.value.cross(rhs.deriv),
        }
    }

    /// Euclidean length with its derivative `d|v| = (v.v')/|v|`.
    #[inline]
    #[must_use]
    pub fn length(self) -> Dual {
        self.dot(self).sqrt()
    }
}

impl core::ops::Add for DualVec3 {
    type Output = Self;
    #[inline]
    fn add(self, rhs: Self) -> Self {
        Self {
            value: self.value + rhs.value,
            deriv: self.deriv + rhs.deriv,
        }
    }
}

impl core::ops::Sub for DualVec3 {
    type Output = Self;
    #[inline]
    fn sub(self, rhs: Self) -> Self {
        Self {
            value: self.value - rhs.value,
            deriv: self.deriv - rhs.deriv,
        }
    }
}

impl core::ops::Mul<Dual> for DualVec3 {
    type Output = Self;
    #[inline]
    fn mul(self, s: Dual) -> Self {
        // Scale a varying vector by a varying scalar (product rule per axis).
        Self {
            value: self.value * s.re,
            deriv: self.deriv * s.re + self.value * s.du,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::float::f32 as mf;

    const EPS: f32 = 1e-3;

    fn fd(f: impl Fn(f32) -> f32, x: f32) -> f32 {
        // Central finite difference as an independent derivative oracle.
        let h = 1e-3;
        (f(x + h) - f(x - h)) / (2.0 * h)
    }

    #[test]
    fn polynomial_derivative_is_exact() {
        // f(x) = 3x^3 - 2x^2 + x - 5 ; f'(x) = 9x^2 - 4x + 1
        let eval = |x: Dual| {
            x * x * x * Dual::constant(3.0) - x * x * Dual::constant(2.0) + x
                - Dual::constant(5.0)
        };
        for &x in &[-2.0f32, -0.5, 0.0, 1.0, 3.3] {
            let d = eval(Dual::variable(x));
            let expected = 9.0 * x * x - 4.0 * x + 1.0;
            assert!(
                (d.du - expected).abs() < EPS,
                "x={x}: got {}, want {expected}",
                d.du
            );
        }
    }

    #[test]
    fn quotient_rule() {
        // f(x) = (x^2 + 1) / (x - 3)
        let eval = |x: Dual| (x * x + Dual::constant(1.0)) / (x - Dual::constant(3.0));
        let scalar = |x: f32| (x * x + 1.0) / (x - 3.0);
        for &x in &[-1.0f32, 0.0, 1.5, 5.0] {
            let d = eval(Dual::variable(x));
            assert!((d.re - scalar(x)).abs() < EPS);
            assert!(
                (d.du - fd(scalar, x)).abs() < 1e-2,
                "x={x}: got {}, fd {}",
                d.du,
                fd(scalar, x)
            );
        }
    }

    #[test]
    fn transcendental_chain_rules() {
        // f(x) = sin(x^2) ; f'(x) = cos(x^2) * 2x
        for &x in &[0.2f32, 0.7, 1.3] {
            let d = (Dual::variable(x) * Dual::variable(x)).sin();
            let expected = mf::cos(x * x) * 2.0 * x;
            assert!((d.du - expected).abs() < EPS, "x={x}");
        }
        // f(x) = exp(x) * ln(x)
        for &x in &[0.5f32, 1.0, 2.5] {
            let xv = Dual::variable(x);
            let d = xv.exp() * xv.ln();
            let scalar = |t: f32| mf::exp(t) * mf::ln(t);
            assert!((d.du - fd(scalar, x)).abs() < 1e-2, "x={x}");
        }
    }

    #[test]
    fn sqrt_and_recip() {
        for &x in &[0.25f32, 1.0, 4.0, 9.0] {
            let d = Dual::variable(x).sqrt();
            assert!((d.du - 0.5 / mf::sqrt(x)).abs() < EPS, "sqrt x={x}");
            let r = Dual::variable(x).recip();
            assert!((r.du - (-1.0 / (x * x))).abs() < EPS, "recip x={x}");
        }
    }

    #[test]
    fn curve_tangent_and_speed() {
        // Helix-like path built from polynomials to stay deterministic:
        // C(t) = (t, t^2, 2t) ; C'(t) = (1, 2t, 2) ; speed = |C'(t)|.
        let curve = |t: Dual| {
            DualVec3::from_components(t, t * t, t * Dual::constant(2.0))
        };
        for &t in &[0.0f32, 1.0, 2.5] {
            let c = curve(Dual::variable(t));
            assert!((c.deriv.x - 1.0).abs() < EPS);
            assert!((c.deriv.y - 2.0 * t).abs() < EPS);
            assert!((c.deriv.z - 2.0).abs() < EPS);
            // Speed is the magnitude of the velocity (derivative) vector.
            let speed = c.deriv.length();
            let expected_speed = mf::sqrt(1.0 + 4.0 * t * t + 4.0);
            assert!((speed - expected_speed).abs() < EPS, "t={t}");
        }
    }

    #[test]
    fn dualvec3_length_derivative() {
        // d|C(t)|/dt = (C . C') / |C| via DualVec3::length(), checked against
        // a central finite difference of the scalar magnitude. Avoid t=0 where
        // |C| is singular for this path.
        let pos = |t: f32| mf::sqrt(t * t + (t * t) * (t * t) + 4.0 * t * t);
        let curve = |t: Dual| DualVec3::from_components(t, t * t, t * Dual::constant(2.0));
        for &t in &[0.5f32, 1.0, 2.5] {
            let len = curve(Dual::variable(t)).length();
            assert!((len.re - pos(t)).abs() < EPS, "value t={t}");
            assert!((len.du - fd(pos, t)).abs() < 1e-2, "deriv t={t}");
        }
    }

    #[test]
    fn dot_and_cross_product_rules() {
        let a = DualVec3::new(Vec3::new(1.0, 2.0, 3.0), Vec3::new(0.5, -1.0, 2.0));
        let b = DualVec3::new(Vec3::new(-1.0, 0.0, 4.0), Vec3::new(2.0, 1.0, -0.5));
        let dot = a.dot(b);
        let expected_dot_du =
            a.deriv.dot(b.value) + a.value.dot(b.deriv);
        assert!((dot.du - expected_dot_du).abs() < EPS);
        let cross = a.cross(b);
        let expected_cross_du = a.deriv.cross(b.value) + a.value.cross(b.deriv);
        assert!((cross.deriv - expected_cross_du).length() < EPS);
    }

    #[test]
    fn constant_has_zero_derivative() {
        let c = Dual::constant(7.0);
        let d = c * c + c;
        assert_eq!(d.du, 0.0);
    }
}
