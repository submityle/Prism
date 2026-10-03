//! Pluggable backend: wraps a baked parameter field behind a small trait so a
//! host can swap the wave backend in alongside (or instead of) the geometric
//! one.
//!
//! # Provenance
//! Original work; no Unreal Engine, Unity, Godot, Wwise, FMOD, Steam Audio,
//! Microsoft Project Acoustics, or Google Resonance Audio source or derived
//! code; no AI/ML.
//!
//! # Relationship
//! Packages the runtime half of design section 43 as a backend. The
//! [`WaveParameterSource`] trait is the narrow runtime contract (listener
//! position in, [`prism_audio_spatial::SpatialParams`] out); [`WaveBackend`]
//! implements it over a baked [`ParameterField`] via the trilinear
//! [`ParameterLookup`]. Querying it is allocation free, lock free, and panic
//! free, matching the geometric backend's real-time behaviour.

use bevy_math::Vec3;
use prism_audio_spatial::SpatialParams;

use crate::encoding::PerceptualParams;
use crate::field::ParameterField;
use crate::lookup::ParameterLookup;

/// The runtime contract a propagation backend exposes to the mixer: map a
/// world-space listener position to the shared spatial parameter bus.
///
/// Implementations must be real-time safe (no allocation, locks, or panics) so
/// they can be queried from the audio callback thread.
pub trait WaveParameterSource {
    /// Resolves the spatial parameters at `listener_pos`.
    #[must_use]
    fn sample_spatial(&self, listener_pos: Vec3) -> SpatialParams;
}

/// A wave-acoustics backend over a baked [`ParameterField`].
///
/// Construct it once after baking; each query trilinearly interpolates the
/// field and projects the result onto [`SpatialParams`].
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub struct WaveBackend {
    field: ParameterField,
}

impl WaveBackend {
    /// Wraps a baked `field`.
    #[must_use]
    #[inline]
    pub fn new(field: ParameterField) -> Self {
        Self { field }
    }

    /// Borrows the underlying field.
    #[must_use]
    #[inline]
    pub fn field(&self) -> &ParameterField {
        &self.field
    }

    /// Consumes the backend and returns the owned field.
    #[must_use]
    #[inline]
    pub fn into_field(self) -> ParameterField {
        self.field
    }

    /// A fresh trilinear lookup over the field.
    #[must_use]
    #[inline]
    pub fn lookup(&self) -> ParameterLookup<'_> {
        ParameterLookup::new(&self.field)
    }

    /// Samples the raw perceptual parameters at `listener_pos` (before the
    /// projection onto the spatial bus).
    #[must_use]
    #[inline]
    pub fn sample_perceptual(&self, listener_pos: Vec3) -> PerceptualParams {
        self.lookup().sample(listener_pos)
    }
}

impl WaveParameterSource for WaveBackend {
    #[inline]
    fn sample_spatial(&self, listener_pos: Vec3) -> SpatialParams {
        self.sample_perceptual(listener_pos).to_spatial()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::field::{BitDepth, ParameterFieldBuilder};
    use crate::grid::{Aabb, ProbeGrid};

    fn approx(a: f32, b: f32, eps: f32) -> bool {
        (a - b).abs() <= eps
    }

    fn backend() -> WaveBackend {
        let grid = ProbeGrid::new(Aabb::new(Vec3::ZERO, Vec3::splat(2.0)), 2, 2, 2);
        let mut builder = ParameterFieldBuilder::new(grid);
        builder.set_probe(0, PerceptualParams::OCCLUDED);
        builder.set_probe(7, PerceptualParams::OPEN);
        WaveBackend::new(builder.build(BitDepth::Sixteen))
    }

    #[test]
    fn sample_spatial_matches_perceptual_projection() {
        let be = backend();
        let pos = Vec3::splat(1.3);
        let sp = be.sample_spatial(pos);
        let pp = be.sample_perceptual(pos).to_spatial();
        assert!(approx(sp.direct_gain, pp.direct_gain, 1e-6));
        assert!(approx(sp.wet_gain, pp.wet_gain, 1e-6));
        assert!(approx(sp.pitch_ratio, 1.0, 1e-6));
    }

    #[test]
    fn trait_object_is_usable() {
        let be = backend();
        let src: &dyn WaveParameterSource = &be;
        let open_corner = src.sample_spatial(be.field().grid().probe_position(1, 1, 1));
        assert!(open_corner.direct_gain >= 0.0 && open_corner.direct_gain <= 1.0);
    }

    #[test]
    fn into_field_round_trips() {
        let be = backend();
        let field = be.clone().into_field();
        assert_eq!(&field, be.field());
    }
}
