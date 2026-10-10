//! Real spherical harmonics for directional lighting (classical GI numerics).
//!
//! Spherical harmonics (`SH`) compress a function on the sphere — most often
//! incident radiance — into a short vector of band coefficients. Low-order
//! `SH` captures smooth lighting (sky, bounce, irradiance probes) in a handful
//! of numbers that reconstruct cheaply and interpolate well between probes.
//!
//! Two orders are provided: [`Sh2`] (bands 0..=2, 9 coefficients) which is the
//! standard choice for diffuse irradiance, and [`Sh3`] (bands 0..=3, 16
//! coefficients) for sharper directional detail. Both use the real `SH` basis
//! with the Condon–Shortley phase folded into the constants, matching the
//! conventions in Ramamoorthi & Hanrahan and Sloan's "Stupid SH Tricks".
//!
//! Everything here is scalar; store one instance per color channel for `RGB`
//! lighting. Transcendental-free: only polynomials of the direction cosines,
//! so results are identical on every platform.

use crate::vec::Vec3;

/// Band-0 constant basis value `Y(0,0)`.
const K0: f32 = 0.282_094_79; // 1/2 * sqrt(1/pi)
/// Band-1 linear coefficient (shared by all three m).
const K1: f32 = 0.488_602_51; // 1/2 * sqrt(3/pi)
/// Band-2 off-axis coefficient (m = -2, -1, +1).
const K2A: f32 = 1.092_548_4; // 1/2 * sqrt(15/pi)
/// Band-2 `z^2` coefficient (m = 0).
const K2B: f32 = 0.315_391_57; // 1/4 * sqrt(5/pi)
/// Band-2 `x^2 - y^2` coefficient (m = +2).
const K2C: f32 = 0.546_274_2; // 1/4 * sqrt(15/pi)
/// Band-3 coefficient for m = -3, +3.
const K3A: f32 = 0.590_043_6; // 1/4 * sqrt(35/(2*pi))
/// Band-3 coefficient for m = -2, +2.
const K3B: f32 = 2.890_611_4; // 1/2 * sqrt(105/pi)
/// Band-3 coefficient for m = -1, +1.
const K3C: f32 = 0.457_045_8; // 1/4 * sqrt(21/(2*pi))
/// Band-3 coefficient for m = 0.
const K3D: f32 = 0.373_176_3; // 1/4 * sqrt(7/pi)

/// Zonal cosine-lobe convolution factors per band (clamped-cosine kernel).
///
/// Multiplying a radiance `SH` band by its factor yields the diffuse
/// irradiance `SH` band; see [`Sh2::convolve_cosine`]. Odd bands above 1 and
/// the pattern of the even tail follow the classic analytic result; band 3 is
/// exactly zero.
const COS_A0: f32 = core::f32::consts::PI;
/// Band-1 cosine-lobe factor `2*pi/3`.
const COS_A1: f32 = 2.094_395_1;
/// Band-2 cosine-lobe factor `pi/4`.
const COS_A2: f32 = core::f32::consts::FRAC_PI_4;

/// Evaluate the 16 real `SH` basis functions (bands 0..=3) in `dir`.
///
/// `dir` should be unit length; the polynomials assume `x^2 + y^2 + z^2 = 1`.
#[must_use]
pub fn basis3(dir: Vec3) -> [f32; 16] {
    let (x, y, z) = (dir.x, dir.y, dir.z);
    let (x2, y2, z2) = (x * x, y * y, z * z);
    [
        // band 0
        K0,
        // band 1
        K1 * y,
        K1 * z,
        K1 * x,
        // band 2
        K2A * x * y,
        K2A * y * z,
        K2B * (3.0 * z2 - 1.0),
        K2A * x * z,
        K2C * (x2 - y2),
        // band 3
        K3A * y * (3.0 * x2 - y2),
        K3B * x * y * z,
        K3C * y * (5.0 * z2 - 1.0),
        K3D * z * (5.0 * z2 - 3.0),
        K3C * x * (5.0 * z2 - 1.0),
        0.5 * K3B * z * (x2 - y2),
        K3A * x * (x2 - 3.0 * y2),
    ]
}

/// Evaluate the first 9 real `SH` basis functions (bands 0..=2) in `dir`.
#[must_use]
pub fn basis2(dir: Vec3) -> [f32; 9] {
    let b = basis3(dir);
    let mut out = [0.0f32; 9];
    out.copy_from_slice(&b[..9]);
    out
}

/// Order-2 real spherical harmonics: 9 coefficients (bands 0..=2).
///
/// This is the workhorse for diffuse irradiance probes; it reconstructs smooth
/// lighting to within a few percent of ground truth.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Sh2 {
    /// Band coefficients in `(l, m)` order: 0; (1,-1),(1,0),(1,1); (2,-2)..(2,2).
    pub c: [f32; 9],
}

impl Sh2 {
    /// All-zero projection (no light).
    pub const ZERO: Self = Self { c: [0.0; 9] };

    /// Build from an explicit coefficient array.
    #[inline]
    #[must_use]
    pub const fn from_coeffs(c: [f32; 9]) -> Self {
        Self { c }
    }

    /// Accumulate a directional sample of magnitude `value` from `dir`.
    ///
    /// Call once per Monte Carlo sample with `value` already divided by the
    /// sample `pdf` and multiplied by the per-sample solid angle, or use it to
    /// splat a single analytic directional light.
    #[inline]
    pub fn add_sample(&mut self, dir: Vec3, value: f32) {
        for (c, b) in self.c.iter_mut().zip(basis2(dir)) {
            *c += value * b;
        }
    }

    /// Reconstruct the scalar function value in direction `dir`.
    #[inline]
    #[must_use]
    pub fn eval(&self, dir: Vec3) -> f32 {
        self.c.iter().zip(basis2(dir)).map(|(c, b)| c * b).sum()
    }

    /// Convolve radiance coefficients with the clamped-cosine kernel to obtain
    /// diffuse irradiance coefficients. Evaluating the result in a direction
    /// `n` gives the irradiance `E(n)` (integral of radiance times the clamped
    /// cosine over the hemisphere about `n`).
    #[inline]
    #[must_use]
    pub fn convolve_cosine(&self) -> Self {
        let c = self.c;
        Self {
            c: [
                c[0] * COS_A0,
                c[1] * COS_A1,
                c[2] * COS_A1,
                c[3] * COS_A1,
                c[4] * COS_A2,
                c[5] * COS_A2,
                c[6] * COS_A2,
                c[7] * COS_A2,
                c[8] * COS_A2,
            ],
        }
    }

    /// Scale all coefficients by `s`.
    #[inline]
    #[must_use]
    pub fn scaled(self, s: f32) -> Self {
        self * s
    }
}

impl core::ops::Add for Sh2 {
    type Output = Self;
    #[inline]
    fn add(self, rhs: Self) -> Self {
        let mut c = self.c;
        for (a, b) in c.iter_mut().zip(rhs.c) {
            *a += b;
        }
        Self { c }
    }
}

impl core::ops::Sub for Sh2 {
    type Output = Self;
    #[inline]
    fn sub(self, rhs: Self) -> Self {
        let mut c = self.c;
        for (a, b) in c.iter_mut().zip(rhs.c) {
            *a -= b;
        }
        Self { c }
    }
}

impl core::ops::Mul<f32> for Sh2 {
    type Output = Self;
    #[inline]
    fn mul(self, s: f32) -> Self {
        let mut c = self.c;
        for v in &mut c {
            *v *= s;
        }
        Self { c }
    }
}

/// Order-3 real spherical harmonics: 16 coefficients (bands 0..=3).
///
/// Adds the band-3 lobes for sharper directional content than [`Sh2`] at the
/// cost of 7 more coefficients.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Sh3 {
    /// Band coefficients in `(l, m)` order for bands 0..=3.
    pub c: [f32; 16],
}

impl Sh3 {
    /// All-zero projection (no light).
    pub const ZERO: Self = Self { c: [0.0; 16] };

    /// Build from an explicit coefficient array.
    #[inline]
    #[must_use]
    pub const fn from_coeffs(c: [f32; 16]) -> Self {
        Self { c }
    }

    /// Accumulate a directional sample of magnitude `value` from `dir`.
    #[inline]
    pub fn add_sample(&mut self, dir: Vec3, value: f32) {
        for (c, b) in self.c.iter_mut().zip(basis3(dir)) {
            *c += value * b;
        }
    }

    /// Reconstruct the scalar function value in direction `dir`.
    #[inline]
    #[must_use]
    pub fn eval(&self, dir: Vec3) -> f32 {
        self.c.iter().zip(basis3(dir)).map(|(c, b)| c * b).sum()
    }

    /// Diffuse-irradiance convolution. Band 3 vanishes under the clamped-cosine
    /// kernel, so only bands 0..=2 survive (matching [`Sh2::convolve_cosine`]).
    #[inline]
    #[must_use]
    pub fn convolve_cosine(&self) -> Self {
        let c = self.c;
        Self {
            c: [
                c[0] * COS_A0,
                c[1] * COS_A1,
                c[2] * COS_A1,
                c[3] * COS_A1,
                c[4] * COS_A2,
                c[5] * COS_A2,
                c[6] * COS_A2,
                c[7] * COS_A2,
                c[8] * COS_A2,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
            ],
        }
    }

    /// Truncate to the first 9 (bands 0..=2) coefficients as an [`Sh2`].
    #[inline]
    #[must_use]
    pub fn to_sh2(self) -> Sh2 {
        let mut c = [0.0f32; 9];
        c.copy_from_slice(&self.c[..9]);
        Sh2 { c }
    }

    /// Scale all coefficients by `s`.
    #[inline]
    #[must_use]
    pub fn scaled(self, s: f32) -> Self {
        self * s
    }
}

impl core::ops::Add for Sh3 {
    type Output = Self;
    #[inline]
    fn add(self, rhs: Self) -> Self {
        let mut c = self.c;
        for (a, b) in c.iter_mut().zip(rhs.c) {
            *a += b;
        }
        Self { c }
    }
}

impl core::ops::Sub for Sh3 {
    type Output = Self;
    #[inline]
    fn sub(self, rhs: Self) -> Self {
        let mut c = self.c;
        for (a, b) in c.iter_mut().zip(rhs.c) {
            *a -= b;
        }
        Self { c }
    }
}

impl core::ops::Mul<f32> for Sh3 {
    type Output = Self;
    #[inline]
    fn mul(self, s: f32) -> Self {
        let mut c = self.c;
        for v in &mut c {
            *v *= s;
        }
        Self { c }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::float::f32 as mf;
    use crate::rng::{Rng, Xoshiro256StarStar};

    fn rand_dir(rng: &mut Xoshiro256StarStar) -> Vec3 {
        // Uniform point on the unit sphere via the z/azimuth method.
        let z = rng.next_f32() * 2.0 - 1.0;
        let phi = rng.next_f32() * core::f32::consts::TAU;
        let r = mf::sqrt((1.0 - z * z).max(0.0));
        let (s, c) = mf::sin_cos(phi);
        Vec3::new(r * c, r * s, z)
    }

    #[test]
    fn basis_is_orthonormal_over_the_sphere() {
        // Monte Carlo estimate of <Y_i, Y_j> = integral over the sphere. With
        // the 4*pi solid angle and uniform sampling the mean of Y_i*Y_j*4*pi
        // approaches the Kronecker delta.
        let mut rng = Xoshiro256StarStar::new(0xA11CE);
        let n = 400_000;
        let mut gram = [[0.0f64; 16]; 16];
        for _ in 0..n {
            let b = basis3(rand_dir(&mut rng));
            for (i, &bi) in b.iter().enumerate() {
                for (j, &bj) in b.iter().enumerate() {
                    gram[i][j] += f64::from(bi * bj);
                }
            }
        }
        let scale = (4.0 * core::f64::consts::PI) / f64::from(n);
        for (i, row) in gram.iter().enumerate() {
            for (j, &g) in row.iter().enumerate() {
                let v = g * scale;
                let want = if i == j { 1.0 } else { 0.0 };
                assert!((v - want).abs() < 0.03, "<{i},{j}> = {v}, want {want}");
            }
        }
    }

    #[test]
    fn projection_reconstructs_low_order_field() {
        // A smooth band-limited field: f(d) = 1 + 2*d.z + 0.5*(3 d.z^2 - 1).
        // It lives entirely in bands 0..=2, so SH2 reconstructs it exactly.
        let field = |d: Vec3| 1.0 + 2.0 * d.z + 0.5 * (3.0 * d.z * d.z - 1.0);
        let mut rng = Xoshiro256StarStar::new(7);
        let n = 200_000;
        let mut sh = Sh2::ZERO;
        // Projection: coeff_i = integral f*Y_i ~ (4*pi/n) * sum f*Y_i.
        let w = (4.0 * core::f32::consts::PI) / n as f32;
        for _ in 0..n {
            let d = rand_dir(&mut rng);
            sh.add_sample(d, field(d) * w);
        }
        for _ in 0..1000 {
            let d = rand_dir(&mut rng);
            assert!((sh.eval(d) - field(d)).abs() < 0.05, "reconstruct mismatch");
        }
    }

    #[test]
    fn cosine_convolution_matches_direct_hemisphere_integral() {
        // Project a single directional light, convolve to irradiance, then
        // compare SH irradiance against a brute-force clamped-cosine integral.
        let light_dir = Vec3::new(0.3, 0.6, 0.74).normalize();
        let mut radiance = Sh2::ZERO;
        // Delta light: its SH projection is intensity * Y(light_dir).
        let intensity = 1.0;
        for (c, b) in radiance.c.iter_mut().zip(basis2(light_dir)) {
            *c = intensity * b;
        }
        let irr = radiance.convolve_cosine();

        let mut rng = Xoshiro256StarStar::new(0xBEEF);
        for _ in 0..8 {
            let n = rand_dir(&mut rng);
            // Analytic clamped-cosine response of a delta light: intensity*max(0,n.l).
            let direct = intensity * n.dot(light_dir).max(0.0);
            let sh_irr = irr.eval(n);
            // Low-order SH of a delta is only approximate; allow ringing slack.
            assert!(
                (sh_irr - direct).abs() < 0.4,
                "n={n:?}: sh={sh_irr}, direct={direct}"
            );
        }
    }

    #[test]
    fn linearity_add_scale() {
        let mut a = Sh3::ZERO;
        let mut b = Sh3::ZERO;
        a.add_sample(Vec3::new(1.0, 0.0, 0.0), 2.0);
        b.add_sample(Vec3::new(0.0, 1.0, 0.0), 3.0);
        let d = Vec3::new(0.0, 0.0, 1.0);
        let sum = a + b;
        assert!((sum.eval(d) - (a.eval(d) + b.eval(d))).abs() < 1e-5);
        let scaled = a * 2.5;
        assert!((scaled.eval(d) - a.eval(d) * 2.5).abs() < 1e-5);
    }

    #[test]
    fn sh3_truncates_to_sh2() {
        let mut s = Sh3::ZERO;
        s.add_sample(Vec3::new(0.2, -0.3, 0.93).normalize(), 1.5);
        let t = s.to_sh2();
        for (a, b) in t.c.iter().zip(&s.c[..9]) {
            assert_eq!(a, b);
        }
    }
}
