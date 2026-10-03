//! Engine-agnostic contact input views the translator consumes.
//!
//! The bridge is deliberately decoupled from any one physics engine: callers
//! translate their native manifold into these plain data views before handing
//! them to [`crate::translator::ContactAudioTranslator`]. A
//! [`ContactManifoldView`] carries the two bodies, the shared contact normal
//! (pointing from body `a` toward body `b`, matching the physics-core
//! convention), and the world-space contact points. A [`ContactPhase`] tags
//! whether the contact just started, is persisting, or just ended.
//!
//! # Provenance
//! Original work; no Unreal Engine, Unity, Godot, Wwise, FMOD, Steam Audio,
//! Dolby, or Google Resonance Audio source or derived code; no AI/ML.
//!
//! # Relationship
//! Defines the input boundary of design section 47.1: the concrete adapters in
//! [`crate::physics_core`] fill these views from `prism_physics_core`, and the
//! translator reads them with no engine-specific types in scope.

#[cfg(not(feature = "std"))]
use alloc::vec::Vec;

use bevy_math::Vec3;

use crate::body::BodyAudioId;

/// Lifecycle phase of a contact within the current block.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub enum ContactPhase {
    /// The contact began this step; synthesises a discrete impact.
    Started,
    /// The contact persisted this step; synthesises a continuous sustain.
    Persisting,
    /// The contact broke this step; synthesises a separation and is forgotten.
    Ended,
}

/// A single world-space contact point within a manifold view.
#[derive(Clone, Copy, Debug, PartialEq)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub struct ContactPointView {
    /// World-space position of the contact.
    pub world_point: Vec3,
    /// Non-negative overlap depth along the manifold normal.
    pub penetration: f32,
}

impl ContactPointView {
    /// Builds a contact point view, clamping non-finite or negative depth to
    /// zero.
    #[inline]
    #[must_use]
    pub fn new(world_point: Vec3, penetration: f32) -> Self {
        let penetration = if penetration.is_finite() && penetration > 0.0 {
            penetration
        } else {
            0.0
        };
        Self {
            world_point,
            penetration,
        }
    }
}

/// An engine-agnostic view of one contact manifold between two bodies.
#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub struct ContactManifoldView {
    /// Identifier of the first body in the ordered pair.
    pub body_a: BodyAudioId,
    /// Identifier of the second body in the ordered pair.
    pub body_b: BodyAudioId,
    /// Unit contact normal pointing from body `a` toward body `b`.
    pub normal: Vec3,
    /// World-space contact points carried by the manifold.
    pub points: Vec<ContactPointView>,
}

impl ContactManifoldView {
    /// Builds a manifold view from its bodies, normal, and contact points.
    #[inline]
    #[must_use]
    pub fn new(
        body_a: BodyAudioId,
        body_b: BodyAudioId,
        normal: Vec3,
        points: Vec<ContactPointView>,
    ) -> Self {
        Self {
            body_a,
            body_b,
            normal,
            points,
        }
    }

    /// Returns the deepest (most representative) contact point, if any.
    ///
    /// Picks the point with the largest penetration; ties keep the earliest in
    /// slice order so the choice is deterministic.
    #[inline]
    #[must_use]
    pub fn representative_point(&self) -> Option<ContactPointView> {
        let mut best: Option<ContactPointView> = None;
        for p in &self.points {
            match best {
                Some(current) if current.penetration >= p.penetration => {}
                _ => best = Some(*p),
            }
        }
        best
    }

    /// Returns the energy-neutral centroid of the contact points, if any.
    ///
    /// Used as the world position of the manifold for distance-based
    /// clustering; returns `None` for an empty manifold.
    #[inline]
    #[must_use]
    pub fn centroid(&self) -> Option<Vec3> {
        if self.points.is_empty() {
            return None;
        }
        let mut sum = Vec3::ZERO;
        for p in &self.points {
            sum += p.world_point;
        }
        Some(sum / self.points.len() as f32)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn point_sanitises_penetration() {
        let p = ContactPointView::new(Vec3::ZERO, -3.0);
        assert!((p.penetration).abs() < 1e-6);
        let q = ContactPointView::new(Vec3::ZERO, f32::INFINITY);
        assert!((q.penetration).abs() < 1e-6);
    }

    #[test]
    fn representative_picks_deepest() {
        let view = ContactManifoldView::new(
            BodyAudioId(1),
            BodyAudioId(2),
            Vec3::Y,
            alloc::vec![
                ContactPointView::new(Vec3::new(0.0, 0.0, 0.0), 0.1),
                ContactPointView::new(Vec3::new(1.0, 0.0, 0.0), 0.4),
                ContactPointView::new(Vec3::new(2.0, 0.0, 0.0), 0.2),
            ],
        );
        let rep = view.representative_point().unwrap();
        assert!((rep.penetration - 0.4).abs() < 1e-6);
        assert_eq!(rep.world_point, Vec3::new(1.0, 0.0, 0.0));
    }

    #[test]
    fn empty_manifold_has_no_representative() {
        let view = ContactManifoldView::new(BodyAudioId(1), BodyAudioId(2), Vec3::Y, Vec::new());
        assert!(view.representative_point().is_none());
        assert!(view.centroid().is_none());
    }

    #[test]
    fn centroid_averages_points() {
        let view = ContactManifoldView::new(
            BodyAudioId(1),
            BodyAudioId(2),
            Vec3::Y,
            alloc::vec![
                ContactPointView::new(Vec3::new(0.0, 0.0, 0.0), 0.1),
                ContactPointView::new(Vec3::new(2.0, 0.0, 0.0), 0.1),
            ],
        );
        let c = view.centroid().unwrap();
        assert!((c.x - 1.0).abs() < 1e-6);
    }
}
