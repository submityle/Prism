//! Critical-state stress ratio `M` for the Cam-Clay family of soil models.
//!
//! In the triaxial `q`-`p'` plane the critical-state line has slope `M`, which
//! is tied to the critical-state friction angle `phi_cv` through the
//! Mohr-Coulomb relationships:
//!
//! ```text
//! M_c = 6 sin(phi) / (3 - sin(phi))   (triaxial compression)
//! M_e = 6 sin(phi) / (3 + sin(phi))   (triaxial extension)
//! ```
//!
//! The inverse map recovers the friction angle from the compression slope:
//!
//! ```text
//! sin(phi) = 3 M_c / (6 + M_c)
//! ```
//!
//! Because `sin(phi)` is bounded by `1`, the compression slope `M_c` is
//! physically restricted to the open interval `(0, 3)`.
//!
//! The ratio of the two slopes depends only on the friction angle:
//!
//! ```text
//! M_e / M_c = (3 - sin(phi)) / (3 + sin(phi))
//! ```
//!
//! References: Schofield & Wroth, "Critical State Soil Mechanics", 1968;
//! Muir Wood, "Soil Behaviour and Critical State Soil Mechanics", 1990.
//!
//! This module is a self-contained diagnostic primitive. It performs no
//! coupling with the simulation pipeline and holds no simulation state.

use std::f32::consts::FRAC_PI_2;

/// Critical-state stress ratio, stored as the critical-state `sin(phi_cv)`.
///
/// Storing the sine keeps both the compression and extension slopes exact and
/// avoids repeated trigonometric evaluation.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CriticalStateFrictionRatio {
    sin_phi: f32,
}

impl CriticalStateFrictionRatio {
    /// Builds the ratio from a critical-state friction angle in radians.
    ///
    /// Returns `None` if `phi` is non-finite or outside the open interval
    /// `(0, pi/2)`.
    pub fn from_friction_angle_radians(phi: f32) -> Option<Self> {
        if !phi.is_finite() || phi <= 0.0 || phi >= FRAC_PI_2 {
            return None;
        }
        // `sin` is computed in f64 to satisfy the f32 transcendental lint.
        let sin_phi = f64::from(phi).sin() as f32;
        if !(0.0..1.0).contains(&sin_phi) {
            return None;
        }
        Some(Self { sin_phi })
    }

    /// Builds the ratio from a critical-state friction angle in degrees.
    ///
    /// Returns `None` if `phi_degrees` is non-finite or outside `(0, 90)`.
    pub fn from_friction_angle_degrees(phi_degrees: f32) -> Option<Self> {
        if !phi_degrees.is_finite() {
            return None;
        }
        Self::from_friction_angle_radians(phi_degrees.to_radians())
    }

    /// Builds the ratio from a triaxial-compression slope `M_c`.
    ///
    /// Returns `None` if `m_c` is non-finite or outside the physical interval
    /// `(0, 3)`.
    pub fn from_compression_ratio(m_c: f32) -> Option<Self> {
        if !m_c.is_finite() || m_c <= 0.0 || m_c >= 3.0 {
            return None;
        }
        // sin(phi) = 3 M_c / (6 + M_c)
        let sin_phi = 3.0 * m_c / (6.0 + m_c);
        if !(0.0..1.0).contains(&sin_phi) {
            return None;
        }
        Some(Self { sin_phi })
    }

    /// Returns the critical-state `sin(phi_cv)`.
    pub fn sin_phi(&self) -> f32 {
        self.sin_phi
    }

    /// Returns the critical-state friction angle in radians.
    pub fn friction_angle_radians(&self) -> f32 {
        // `asin` is computed in f64 to satisfy the f32 transcendental lint.
        f64::from(self.sin_phi).asin() as f32
    }

    /// Returns the critical-state friction angle in degrees.
    pub fn friction_angle_degrees(&self) -> f32 {
        self.friction_angle_radians().to_degrees()
    }

    /// Triaxial-compression critical-state slope `M_c = 6 sin(phi) / (3 - sin(phi))`.
    pub fn compression_ratio(&self) -> f32 {
        6.0 * self.sin_phi / (3.0 - self.sin_phi)
    }

    /// Triaxial-extension critical-state slope `M_e = 6 sin(phi) / (3 + sin(phi))`.
    pub fn extension_ratio(&self) -> f32 {
        6.0 * self.sin_phi / (3.0 + self.sin_phi)
    }

    /// Ratio of extension to compression slopes `M_e / M_c = (3 - sin(phi)) / (3 + sin(phi))`.
    pub fn extension_to_compression_ratio(&self) -> f32 {
        (3.0 - self.sin_phi) / (3.0 + self.sin_phi)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn assert_close(a: f32, b: f32, tol: f32) {
        assert!((a - b).abs() <= tol, "expected {b}, got {a}");
    }

    #[test]
    fn thirty_degrees_matches_hand_values() {
        // phi = 30 deg -> sin = 0.5; M_c = 3/2.5 = 1.2; M_e = 3/3.5 = 0.857143.
        let csr = CriticalStateFrictionRatio::from_friction_angle_degrees(30.0).unwrap();
        assert_close(csr.sin_phi(), 0.5, 1.0e-6);
        assert_close(csr.compression_ratio(), 1.2, 1.0e-5);
        assert_close(csr.extension_ratio(), 0.857_142_8, 1.0e-5);
    }

    #[test]
    fn compression_exceeds_extension() {
        let csr = CriticalStateFrictionRatio::from_friction_angle_degrees(33.0).unwrap();
        assert!(csr.compression_ratio() > csr.extension_ratio());
    }

    #[test]
    fn extension_to_compression_ratio_is_consistent() {
        let csr = CriticalStateFrictionRatio::from_friction_angle_degrees(28.0).unwrap();
        let direct = csr.extension_to_compression_ratio();
        let computed = csr.extension_ratio() / csr.compression_ratio();
        assert_close(direct, computed, 1.0e-6);
    }

    #[test]
    fn radians_and_degrees_agree() {
        let a = CriticalStateFrictionRatio::from_friction_angle_degrees(35.0).unwrap();
        let b =
            CriticalStateFrictionRatio::from_friction_angle_radians(35.0_f32.to_radians()).unwrap();
        assert_close(a.sin_phi(), b.sin_phi(), 1.0e-6);
    }

    #[test]
    fn friction_angle_roundtrips() {
        let csr = CriticalStateFrictionRatio::from_friction_angle_degrees(31.5).unwrap();
        assert_close(csr.friction_angle_degrees(), 31.5, 1.0e-3);
    }

    #[test]
    fn compression_ratio_inverts_to_angle() {
        // M_c = 1.2 -> sin(phi) = 3.6/7.2 = 0.5 -> phi = 30 deg.
        let csr = CriticalStateFrictionRatio::from_compression_ratio(1.2).unwrap();
        assert_close(csr.sin_phi(), 0.5, 1.0e-6);
        assert_close(csr.friction_angle_degrees(), 30.0, 1.0e-3);
    }

    #[test]
    fn compression_ratio_roundtrips() {
        let csr = CriticalStateFrictionRatio::from_friction_angle_degrees(29.0).unwrap();
        let m_c = csr.compression_ratio();
        let back = CriticalStateFrictionRatio::from_compression_ratio(m_c).unwrap();
        assert_close(back.friction_angle_degrees(), 29.0, 1.0e-3);
    }

    #[test]
    fn rejects_out_of_range_angle() {
        assert!(CriticalStateFrictionRatio::from_friction_angle_degrees(0.0).is_none());
        assert!(CriticalStateFrictionRatio::from_friction_angle_degrees(90.0).is_none());
        assert!(CriticalStateFrictionRatio::from_friction_angle_degrees(-5.0).is_none());
    }

    #[test]
    fn rejects_out_of_range_compression_ratio() {
        assert!(CriticalStateFrictionRatio::from_compression_ratio(0.0).is_none());
        assert!(CriticalStateFrictionRatio::from_compression_ratio(3.0).is_none());
        assert!(CriticalStateFrictionRatio::from_compression_ratio(3.5).is_none());
    }

    #[test]
    fn rejects_non_finite_inputs() {
        assert!(CriticalStateFrictionRatio::from_friction_angle_degrees(f32::NAN).is_none());
        assert!(CriticalStateFrictionRatio::from_friction_angle_radians(f32::INFINITY).is_none());
        assert!(CriticalStateFrictionRatio::from_compression_ratio(f32::NAN).is_none());
    }
}
