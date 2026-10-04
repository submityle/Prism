//! Capability descriptor for a platform spatial renderer.
//!
//! A platform spatial renderer (an operating-system or console spatial-audio
//! service) exposes a bounded set of abilities: how it ingests dynamic objects
//! (native discrete objects, a fixed channel bed, headphone binaural, or a
//! scene-based Ambisonic transport), how many discrete objects it can present
//! at once, which channel bed it expects, how high an Ambisonic order it
//! accepts, and whether it applies head tracking. [`PlatformCapability`]
//! captures that contract so the negotiation layer ([`crate::platform::delivery`])
//! can map an engine [`crate::scene::ObjectScene`] onto whatever the platform
//! can actually render, degrading gracefully when the scene exceeds the
//! platform's limits.
//!
//! This is a pure data descriptor: it performs no input/output and holds no
//! per-sample audio. The actual operating-system handoff (for example a
//! Windows `ISpatialAudioClient` object stream or a `CoreAudio` spatial mixer)
//! lives in the platform/device layer; this crate only produces the
//! control-rate delivery description that layer consumes.
//!
//! # Provenance
//! Original work; no Unreal Engine, Unity, Godot, Wwise, FMOD, Steam Audio,
//! Dolby, or Google Resonance Audio source or derived code; no AI/ML. The
//! object-count ceilings and bed layouts referenced by presets are publicly
//! documented platform capacities expressed as our own configurable defaults.
//!
//! # Relationship
//! Supports design section 16 (platform spatial backend bridge) and section
//! 44.2 (bed-plus-objects delivery with hardware object budgets). Produces an
//! [`crate::budget::ObjectBudget`] for the clustering fallback and selects the
//! [`crate::channel_order::ChannelOrder`] a bed delivery must follow.

use crate::bed::BedLayout;
use crate::budget::ObjectBudget;
use crate::platform::channel_order::ChannelOrder;

/// How a platform renderer ingests the engine's dynamic objects.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub enum ObjectRenderMode {
    /// The platform has a native discrete-object renderer and accepts up to
    /// [`PlatformCapability::max_objects`] simultaneous positioned objects on
    /// top of its channel bed (for example an Atmos-, Tempest-, or Windows
    /// Sonic-style object renderer).
    DiscreteObjects,
    /// The platform has no object renderer; every object must be folded onto
    /// its channel bed before delivery.
    BedOnly,
    /// The platform is a headphone binaural renderer that takes per-object
    /// direction parameters rather than channel or object streams.
    Binaural,
    /// The platform is a scene-based Ambisonic renderer that takes an `AmbiX`
    /// (ACN + SN3D) transport of a bounded order.
    Ambisonic,
}

/// The rendering contract a platform spatial renderer exposes to the engine.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub struct PlatformCapability {
    /// How the platform ingests dynamic objects.
    pub object_mode: ObjectRenderMode,
    /// Maximum number of simultaneously renderable discrete objects. Only
    /// meaningful when `object_mode` is [`ObjectRenderMode::DiscreteObjects`];
    /// zero for every other mode.
    pub max_objects: usize,
    /// The channel bed the platform natively presents (and the fold target for
    /// [`ObjectRenderMode::BedOnly`]).
    pub bed: BedLayout,
    /// Highest Ambisonic order the platform accepts. Only meaningful when
    /// `object_mode` is [`ObjectRenderMode::Ambisonic`]; zero otherwise.
    pub max_ambisonic_order: usize,
    /// Channel ordering the platform expects for bed deliveries.
    pub channel_order: ChannelOrder,
    /// Whether the platform applies listener head tracking downstream (purely
    /// informational for the engine's control-rate planning).
    pub head_tracked: bool,
}

impl PlatformCapability {
    /// Creates a discrete-object capability for `max_objects` objects layered
    /// on `bed`, using `channel_order` for any bed delivery.
    #[must_use]
    pub const fn discrete_objects(
        max_objects: usize,
        bed: BedLayout,
        channel_order: ChannelOrder,
        head_tracked: bool,
    ) -> Self {
        Self {
            object_mode: ObjectRenderMode::DiscreteObjects,
            max_objects,
            bed,
            max_ambisonic_order: 0,
            channel_order,
            head_tracked,
        }
    }

    /// Creates a bed-only capability that folds every object onto `bed`.
    #[must_use]
    pub const fn bed_only(bed: BedLayout, channel_order: ChannelOrder) -> Self {
        Self {
            object_mode: ObjectRenderMode::BedOnly,
            max_objects: 0,
            bed,
            max_ambisonic_order: 0,
            channel_order,
            head_tracked: false,
        }
    }

    /// Creates a binaural (headphone) capability.
    ///
    /// `bed` is retained as the fold target of last resort should a caller
    /// request a channel delivery from a binaural device.
    #[must_use]
    pub const fn binaural(bed: BedLayout, head_tracked: bool) -> Self {
        Self {
            object_mode: ObjectRenderMode::Binaural,
            max_objects: 0,
            bed,
            max_ambisonic_order: 0,
            channel_order: ChannelOrder::EngineCanonical,
            head_tracked,
        }
    }

    /// Creates a scene-based Ambisonic capability of the given maximum order.
    #[must_use]
    pub const fn ambisonic(max_ambisonic_order: usize, bed: BedLayout, head_tracked: bool) -> Self {
        Self {
            object_mode: ObjectRenderMode::Ambisonic,
            max_objects: 0,
            bed,
            max_ambisonic_order,
            channel_order: ChannelOrder::EngineCanonical,
            head_tracked,
        }
    }

    /// Returns the object budget implied by this capability.
    ///
    /// Discrete-object platforms budget their advertised ceiling; every other
    /// mode has a zero budget, which folds all objects (into the bed, binaural
    /// parameters, or the Ambisonic field) by the delivery negotiation.
    #[must_use]
    pub const fn budget(self) -> ObjectBudget {
        match self.object_mode {
            ObjectRenderMode::DiscreteObjects => ObjectBudget::new(self.max_objects),
            ObjectRenderMode::BedOnly
            | ObjectRenderMode::Binaural
            | ObjectRenderMode::Ambisonic => ObjectBudget::new(0),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn discrete_capability_budgets_its_ceiling() {
        let cap =
            PlatformCapability::discrete_objects(16, BedLayout::Surround7_1_4, ChannelOrder::EngineCanonical, true);
        assert_eq!(cap.object_mode, ObjectRenderMode::DiscreteObjects);
        assert_eq!(cap.budget().max_objects(), 16);
        assert!(cap.head_tracked);
    }

    #[test]
    fn non_discrete_modes_have_zero_budget() {
        let bed = BedLayout::Surround5_1_4;
        assert_eq!(
            PlatformCapability::bed_only(bed, ChannelOrder::LfeLast)
                .budget()
                .max_objects(),
            0
        );
        assert_eq!(
            PlatformCapability::binaural(bed, true).budget().max_objects(),
            0
        );
        assert_eq!(
            PlatformCapability::ambisonic(3, bed, false)
                .budget()
                .max_objects(),
            0
        );
    }

    #[test]
    fn ambisonic_records_order() {
        let cap = PlatformCapability::ambisonic(3, BedLayout::Stereo, false);
        assert_eq!(cap.object_mode, ObjectRenderMode::Ambisonic);
        assert_eq!(cap.max_ambisonic_order, 3);
    }
}
