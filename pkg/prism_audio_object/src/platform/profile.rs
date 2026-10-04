//! Named platform-renderer presets.
//!
//! [`PlatformProfile`] enumerates the spatial-audio targets the engine ships
//! defaults for and resolves each to a [`PlatformCapability`]. The presets
//! encode publicly documented capacities (object ceilings, native beds,
//! Ambisonic orders, head tracking) as our own configurable defaults; a title
//! that knows a device's exact limits can bypass the presets and construct a
//! [`PlatformCapability`] directly.
//!
//! # Provenance
//! Original work; no Unreal Engine, Unity, Godot, Wwise, FMOD, Steam Audio,
//! Dolby, or Google Resonance Audio source or derived code; no AI/ML. Platform
//! names identify delivery targets only; no vendor code or data is reproduced.
//!
//! # Relationship
//! Supports design section 16 (platform spatial backend bridge). Each profile
//! yields a [`PlatformCapability`] consumed by
//! [`crate::platform::spatial_backend::ProfileBackend`] and
//! [`crate::platform::delivery::negotiate_delivery`].

use crate::bed::BedLayout;
use crate::platform::capability::PlatformCapability;
use crate::platform::channel_order::ChannelOrder;

/// Default discrete-object ceiling for console/desktop object renderers.
///
/// A conservative shared default; individual titles override it from the real
/// device query when one is available.
const DEFAULT_OBJECT_CEILING: usize = 128;

/// A named spatial-audio delivery target with a default capability.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub enum PlatformProfile {
    /// A Windows Sonic / Spatial Sound style discrete-object renderer over a
    /// 7.1.4 bed.
    WindowsSpatialSound,
    /// An Apple `CoreAudio` head-tracked binaural spatial renderer.
    CoreAudioSpatial,
    /// A Sony Tempest 3D style head-tracked binaural renderer.
    Tempest3d,
    /// A Dolby Atmos style discrete-object renderer over a 7.1.4 bed.
    AtmosRenderer,
    /// Plain stereo headphones rendered binaurally (no head tracking).
    StereoHeadphones,
    /// A fixed loudspeaker bed with no object renderer; everything folds onto
    /// the given layout.
    ChannelBed(BedLayout),
}

impl PlatformProfile {
    /// Resolves this profile to its default [`PlatformCapability`].
    #[must_use]
    pub fn capability(self) -> PlatformCapability {
        match self {
            PlatformProfile::WindowsSpatialSound => PlatformCapability::discrete_objects(
                DEFAULT_OBJECT_CEILING,
                BedLayout::Surround7_1_4,
                ChannelOrder::LfeLast,
                false,
            ),
            PlatformProfile::AtmosRenderer => PlatformCapability::discrete_objects(
                DEFAULT_OBJECT_CEILING,
                BedLayout::Surround7_1_4,
                ChannelOrder::EngineCanonical,
                false,
            ),
            PlatformProfile::CoreAudioSpatial | PlatformProfile::Tempest3d => {
                PlatformCapability::binaural(BedLayout::Surround7_1_4, true)
            }
            PlatformProfile::StereoHeadphones => {
                PlatformCapability::binaural(BedLayout::Stereo, false)
            }
            PlatformProfile::ChannelBed(layout) => {
                PlatformCapability::bed_only(layout, ChannelOrder::EngineCanonical)
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::platform::capability::ObjectRenderMode;

    #[test]
    fn discrete_profiles_expose_an_object_budget() {
        for profile in [
            PlatformProfile::WindowsSpatialSound,
            PlatformProfile::AtmosRenderer,
        ] {
            let cap = profile.capability();
            assert_eq!(cap.object_mode, ObjectRenderMode::DiscreteObjects);
            assert_eq!(cap.budget().max_objects(), DEFAULT_OBJECT_CEILING);
        }
    }

    #[test]
    fn binaural_profiles_fold_objects() {
        for profile in [
            PlatformProfile::CoreAudioSpatial,
            PlatformProfile::Tempest3d,
            PlatformProfile::StereoHeadphones,
        ] {
            let cap = profile.capability();
            assert_eq!(cap.object_mode, ObjectRenderMode::Binaural);
            assert_eq!(cap.budget().max_objects(), 0);
        }
    }

    #[test]
    fn head_tracking_follows_the_target() {
        assert!(PlatformProfile::CoreAudioSpatial.capability().head_tracked);
        assert!(!PlatformProfile::StereoHeadphones.capability().head_tracked);
    }

    #[test]
    fn channel_bed_profile_is_bed_only_on_its_layout() {
        let cap = PlatformProfile::ChannelBed(BedLayout::Surround5_1_4).capability();
        assert_eq!(cap.object_mode, ObjectRenderMode::BedOnly);
        assert_eq!(cap.bed, BedLayout::Surround5_1_4);
    }
}
