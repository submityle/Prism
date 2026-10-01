//! Path vertices and reconnectable path suffixes for ReSTIR PT — CPU golden.
//!
//! Path-space ReSTIR (Lin et al. 2022, *ReSTIR PT* / *Generalized Resampled
//! Importance Sampling*) reuses whole light-transport *paths* rather than the
//! single secondary bounce of screen-probe `ReSTIR` GI.  A reused path is split
//! at a *reconnection vertex*: everything from that vertex onward (its outgoing
//! radiance) is view independent and can be re-anchored to a neighbour's
//! shading point, while the prefix (camera → primary hit) is regenerated per
//! pixel.  This module is the backend-neutral representation of those pieces.
//!
//! * [`PathVertex`] is one surface interaction — a world position and a unit
//!   surface normal — with the geometry queries the shift map needs.
//! * [`PathSuffix`] is the reconnectable tail of a path: the *primary* (shading)
//!   vertex `x_v`, the *reconnection* vertex `x_s`, and the RGB radiance leaving
//!   `x_s` towards `x_v`.  It converts losslessly to and from the shared
//!   [`GiSample`] payload so the existing reservoir machinery can store it.
//!
//! # Conventions
//! * All geometry is `f32`-backed [`Vec3`] to match the GPU reservoir-buffer
//!   twin; a [`PathSuffix`] packs into exactly the five vectors of a
//!   [`GiSample`].
//! * Normals are normalised on construction with a deterministic zero fallback
//!   for degenerate input, so every downstream dot product stays finite.
//! * Cosines returned by the geometry queries are clamped non-negative: a
//!   back-facing or coincident configuration yields `0`, never a negative or
//!   `NaN` value.
//! * Every function is a deterministic pure function: no RNG, no I/O, no GPU,
//!   no global state, and no `unsafe`.

use bevy_math::Vec3;

use crate::gi::screen_probe::restir::GiSample;

/// Normalises `v`, returning zero for a degenerate (near-zero / non-finite)
/// input so downstream dot products stay finite.
#[inline]
pub(crate) fn normalize_or_zero(v: Vec3) -> Vec3 {
    let len_sq = v.length_squared();
    if len_sq.is_finite() && len_sq > f32::MIN_POSITIVE {
        v * len_sq.sqrt().recip()
    } else {
        Vec3::ZERO
    }
}

/// A single surface interaction along a transport path.
///
/// Holds the world-space `position` of the interaction and the unit surface
/// `normal` there.  Constructed through [`PathVertex::new`], which normalises
/// the supplied normal so the stored value is always unit length or exactly
/// zero (a degenerate marker).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PathVertex {
    /// World-space position of the interaction.
    pub position: Vec3,
    /// Unit surface normal at the interaction (or [`Vec3::ZERO`] if degenerate).
    pub normal: Vec3,
}

impl Default for PathVertex {
    #[inline]
    fn default() -> Self {
        Self::ZERO
    }
}

impl PathVertex {
    /// A fully-zeroed, degenerate vertex at the origin with no normal.
    pub const ZERO: Self = Self {
        position: Vec3::ZERO,
        normal: Vec3::ZERO,
    };

    /// Creates a vertex at `position`, normalising `normal` with a zero
    /// fallback for a degenerate (near-zero / non-finite) input.
    #[inline]
    pub fn new(position: Vec3, normal: Vec3) -> Self {
        Self {
            position,
            normal: normalize_or_zero(normal),
        }
    }

    /// Vector from this vertex to `target`.
    #[inline]
    pub fn offset_to(&self, target: Vec3) -> Vec3 {
        target - self.position
    }

    /// Squared distance from this vertex to `target`.
    ///
    /// Always finite and non-negative; `0` for a coincident point.
    #[inline]
    pub fn dist_sq_to(&self, target: Vec3) -> f32 {
        let d = self.offset_to(target).length_squared();
        if d.is_finite() {
            d.max(0.0)
        } else {
            0.0
        }
    }

    /// Cosine of the angle between this vertex's normal and the direction to
    /// `target`, clamped to `[0, 1]`.
    ///
    /// Returns `0` for a coincident `target`, a degenerate normal, or a
    /// back-facing direction, so the result is always finite and non-negative.
    #[inline]
    pub fn cos_toward(&self, target: Vec3) -> f32 {
        let dir = normalize_or_zero(self.offset_to(target));
        let c = self.normal.dot(dir);
        if c.is_finite() {
            c.clamp(0.0, 1.0)
        } else {
            0.0
        }
    }

    /// Whether this vertex is degenerate (its normal collapsed to zero).
    #[inline]
    pub fn is_degenerate(&self) -> bool {
        self.normal == Vec3::ZERO
    }
}

/// The reconnectable tail of a transport path.
///
/// A path suffix couples the *primary* shading vertex `x_v` (where the pixel is
/// shaded) with the *reconnection* vertex `x_s` (the first vertex of the reused
/// tail) and the RGB `radiance` leaving `x_s` towards `x_v`.  Because the tail
/// is view independent, [`with_primary`](PathSuffix::with_primary) can re-anchor
/// it to a neighbour's shading point — the reconnection shift map.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PathSuffix {
    /// Primary (shading) vertex `x_v`.
    pub primary: PathVertex,
    /// Reconnection vertex `x_s`, the anchor of the reusable tail.
    pub reconnection: PathVertex,
    /// Linear RGB radiance leaving `x_s` towards `x_v`.
    pub radiance: Vec3,
}

impl Default for PathSuffix {
    #[inline]
    fn default() -> Self {
        Self::ZERO
    }
}

impl PathSuffix {
    /// A fully-zeroed suffix that carries no energy.
    pub const ZERO: Self = Self {
        primary: PathVertex::ZERO,
        reconnection: PathVertex::ZERO,
        radiance: Vec3::ZERO,
    };

    /// Builds a suffix from its primary vertex, reconnection vertex, and the
    /// radiance leaving the reconnection vertex towards the primary vertex.
    #[inline]
    pub fn new(primary: PathVertex, reconnection: PathVertex, radiance: Vec3) -> Self {
        Self {
            primary,
            reconnection,
            radiance,
        }
    }

    /// Reconstructs a suffix from the shared [`GiSample`] reservoir payload.
    ///
    /// The visible point maps to the primary vertex and the secondary sample
    /// point to the reconnection vertex; normals are re-normalised so a stored
    /// sample always yields a well-formed suffix.
    #[inline]
    pub fn from_gi_sample(sample: &GiSample) -> Self {
        Self {
            primary: PathVertex::new(sample.visible_point, sample.visible_normal),
            reconnection: PathVertex::new(sample.sample_point, sample.sample_normal),
            radiance: sample.radiance,
        }
    }

    /// Packs this suffix into the shared [`GiSample`] reservoir payload.
    ///
    /// This is the exact inverse of [`from_gi_sample`](Self::from_gi_sample) for
    /// a suffix whose normals are already unit length.
    #[inline]
    pub fn to_gi_sample(&self) -> GiSample {
        GiSample {
            visible_point: self.primary.position,
            visible_normal: self.primary.normal,
            sample_point: self.reconnection.position,
            sample_normal: self.reconnection.normal,
            radiance: self.radiance,
        }
    }

    /// Re-anchors the reusable tail to a new primary (shading) vertex.
    ///
    /// This is the geometric core of the reconnection shift map: the
    /// reconnection vertex and its outgoing radiance are kept verbatim (they are
    /// view independent) while the primary vertex is replaced, mapping the path
    /// into the destination pixel's domain.  The measure change this induces is
    /// accounted for separately by the shift Jacobian.
    #[inline]
    pub fn with_primary(&self, primary: PathVertex) -> Self {
        Self {
            primary,
            reconnection: self.reconnection,
            radiance: self.radiance,
        }
    }

    /// Whether either endpoint is degenerate or the endpoints are coincident.
    ///
    /// A degenerate suffix cannot carry a well-defined reconnection direction,
    /// so the shift map treats it as an identity (unit-Jacobian) fallback.
    #[inline]
    pub fn is_degenerate(&self) -> bool {
        self.primary.is_degenerate()
            || self.reconnection.is_degenerate()
            || self.reconnection.dist_sq_to(self.primary.position) <= f32::MIN_POSITIVE
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn new_normalises_normal_and_falls_back_to_zero() {
        let v = PathVertex::new(Vec3::ONE, Vec3::new(0.0, 0.0, 5.0));
        assert!((v.normal.length() - 1.0).abs() < 1e-6);
        assert_eq!(v.position, Vec3::ONE);
        // A degenerate normal collapses to zero and marks the vertex degenerate.
        let d = PathVertex::new(Vec3::ZERO, Vec3::ZERO);
        assert_eq!(d.normal, Vec3::ZERO);
        assert!(d.is_degenerate());
    }

    #[test]
    fn cos_toward_matches_analytic_and_clamps() {
        // Normal +Z, target straight up: cos = 1.
        let v = PathVertex::new(Vec3::ZERO, Vec3::Z);
        assert!((v.cos_toward(Vec3::new(0.0, 0.0, 2.0)) - 1.0).abs() < 1e-6);
        // 45 degrees: cos = 1/sqrt(2).
        let c = v.cos_toward(Vec3::new(0.0, 1.0, 1.0));
        assert!((c - (0.5f32).sqrt()).abs() < 1e-6, "c={c}");
        // Back-facing target clamps to zero.
        assert_eq!(v.cos_toward(Vec3::new(0.0, 0.0, -1.0)), 0.0);
        // Coincident target clamps to zero.
        assert_eq!(v.cos_toward(Vec3::ZERO), 0.0);
    }

    #[test]
    fn dist_sq_is_non_negative_and_finite() {
        let v = PathVertex::new(Vec3::ZERO, Vec3::Z);
        assert!((v.dist_sq_to(Vec3::new(3.0, 0.0, 4.0)) - 25.0).abs() < 1e-5);
        assert_eq!(v.dist_sq_to(Vec3::ZERO), 0.0);
    }

    #[test]
    fn gi_sample_roundtrip_is_lossless() {
        let s = GiSample {
            visible_point: Vec3::new(1.0, 2.0, 3.0),
            visible_normal: Vec3::Z,
            sample_point: Vec3::new(-1.0, 0.0, 2.0),
            sample_normal: Vec3::X,
            radiance: Vec3::new(0.4, 0.5, 0.6),
        };
        let suffix = PathSuffix::from_gi_sample(&s);
        let back = suffix.to_gi_sample();
        assert_eq!(back, s);
    }

    #[test]
    fn with_primary_keeps_tail_and_swaps_shading_point() {
        let suffix = PathSuffix::new(
            PathVertex::new(Vec3::ZERO, Vec3::Z),
            PathVertex::new(Vec3::new(0.0, 0.0, 2.0), Vec3::NEG_Z),
            Vec3::splat(1.0),
        );
        let moved = suffix.with_primary(PathVertex::new(Vec3::new(1.0, 0.0, 0.0), Vec3::Z));
        assert_eq!(moved.reconnection, suffix.reconnection);
        assert_eq!(moved.radiance, suffix.radiance);
        assert_eq!(moved.primary.position, Vec3::new(1.0, 0.0, 0.0));
    }

    #[test]
    fn degenerate_suffix_detection() {
        // Coincident endpoints.
        let coincident = PathSuffix::new(
            PathVertex::new(Vec3::ZERO, Vec3::Z),
            PathVertex::new(Vec3::ZERO, Vec3::NEG_Z),
            Vec3::ONE,
        );
        assert!(coincident.is_degenerate());
        // Well-separated, well-oriented suffix is not degenerate.
        let good = PathSuffix::new(
            PathVertex::new(Vec3::ZERO, Vec3::Z),
            PathVertex::new(Vec3::new(0.0, 0.0, 2.0), Vec3::NEG_Z),
            Vec3::ONE,
        );
        assert!(!good.is_degenerate());
    }

    #[test]
    fn results_are_deterministic() {
        let build = || {
            let v = PathVertex::new(Vec3::new(1.0, 2.0, 3.0), Vec3::new(1.0, 1.0, 0.0));
            (v.cos_toward(Vec3::new(2.0, 2.0, 3.0)), v.dist_sq_to(Vec3::ZERO))
        };
        assert_eq!(build(), build());
    }
}
