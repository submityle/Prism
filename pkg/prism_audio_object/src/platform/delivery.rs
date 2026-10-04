//! Negotiating an engine object set into a platform-conformant delivery.
//!
//! Given a flat object list and a [`PlatformCapability`], this module selects
//! the [`OutputFormat`] and [`ObjectBudget`] the platform can actually consume,
//! runs the shared budget-to-clustering-to-fold pipeline ([`render_objects`]),
//! and wraps the result in a [`DeliveryPlan`] annotated with the bed channel
//! permutation and whether the scene had to be degraded (objects reduced below
//! their input count because the platform's object ceiling was exceeded).
//!
//! The mapping per [`ObjectRenderMode`] is:
//!
//! * [`ObjectRenderMode::DiscreteObjects`]: budget the platform's object
//!   ceiling and keep up to that many representative objects (carried in
//!   [`RenderOutput::clusters`]); the bed payload is the ready fold fallback,
//!   presented in the platform's channel order.
//! * [`ObjectRenderMode::BedOnly`]: a zero budget folds every object onto the
//!   platform bed, presented in the platform's channel order.
//! * [`ObjectRenderMode::Binaural`]: every object reduces to a binaural
//!   direction parameter set.
//! * [`ObjectRenderMode::Ambisonic`]: every object encodes into an `AmbiX`
//!   transport of the platform's maximum order.
//!
//! No input/output and no per-sample audio occur here; this is a control-rate
//! reduction the platform/device layer consumes.
//!
//! # Provenance
//! Original work; no Unreal Engine, Unity, Godot, Wwise, FMOD, Steam Audio,
//! Dolby, or Google Resonance Audio source or derived code; no AI/ML.
//!
//! # Relationship
//! Implements the negotiation core of design section 16 (platform spatial
//! backend bridge) over the bed-plus-objects model of section 44.2 and the
//! object-budget clustering of section 33. Built on [`crate::render`],
//! [`crate::budget`], and [`crate::platform::capability`].

#[cfg(not(feature = "std"))]
use alloc::vec::Vec;

use crate::object::AudioObject;
use crate::platform::capability::{ObjectRenderMode, PlatformCapability};
use crate::render::{render_objects, OutputFormat, RenderOutput};
use crate::scene::ObjectScene;

/// A platform-conformant delivery description produced by negotiation.
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub struct DeliveryPlan {
    /// The capability the plan was negotiated against.
    pub capability: PlatformCapability,
    /// The output format chosen for the platform.
    pub format: OutputFormat,
    /// The rendered result (clusters plus the format-specific payload).
    pub output: RenderOutput,
    /// Permutation from platform channel slot to canonical bed channel index
    /// for bed payloads; empty for binaural and Ambisonic payloads.
    pub channel_order: Vec<usize>,
    /// Number of input objects presented to the negotiation.
    pub object_count: usize,
    /// Number of discrete objects actually delivered (the representative
    /// cluster count for a discrete-object platform; zero otherwise).
    pub discrete_count: usize,
    /// Whether the platform's object ceiling forced the scene to be reduced
    /// below its input object count.
    pub degraded: bool,
}

/// Negotiates `objects` into a [`DeliveryPlan`] for `capability`.
///
/// The objects are rendered onto the capability's native bed. The returned
/// plan is self-describing: callers read [`DeliveryPlan::format`] to interpret
/// [`RenderOutput::payload`], and apply [`DeliveryPlan::channel_order`] when the
/// payload is a bed.
#[must_use]
pub fn negotiate_delivery(
    objects: &[AudioObject],
    capability: PlatformCapability,
) -> DeliveryPlan {
    let object_count = objects.len();
    let budget = capability.budget();
    let bed = capability.bed;

    let (format, channel_order) = match capability.object_mode {
        ObjectRenderMode::DiscreteObjects | ObjectRenderMode::BedOnly => (
            OutputFormat::Bed,
            capability.channel_order.permutation(bed),
        ),
        ObjectRenderMode::Binaural => (OutputFormat::Binaural, Vec::new()),
        ObjectRenderMode::Ambisonic => (
            OutputFormat::Ambisonic {
                order: capability.max_ambisonic_order,
            },
            Vec::new(),
        ),
    };

    let output = render_objects(objects, bed, budget, format);

    let discrete_count = match capability.object_mode {
        ObjectRenderMode::DiscreteObjects => output.clusters.len(),
        ObjectRenderMode::BedOnly
        | ObjectRenderMode::Binaural
        | ObjectRenderMode::Ambisonic => 0,
    };

    let degraded = matches!(capability.object_mode, ObjectRenderMode::DiscreteObjects)
        && budget.overflow(object_count) > 0;

    DeliveryPlan {
        capability,
        format,
        output,
        channel_order,
        object_count,
        discrete_count,
        degraded,
    }
}

/// Negotiates the current objects of `scene` into a [`DeliveryPlan`].
///
/// Callers should [`ObjectScene::advance_to`] the desired time first so the
/// snapshot reflects the current metadata. The scene's own bed is ignored in
/// favour of the platform's native bed from `capability`.
#[must_use]
pub fn negotiate_scene(scene: &ObjectScene, capability: PlatformCapability) -> DeliveryPlan {
    negotiate_delivery(&scene.current_objects(), capability)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bed::{direction_from_angles, BedLayout};
    use crate::object::ObjectId;
    use crate::platform::channel_order::ChannelOrder;
    use crate::render::RenderPayload;

    /// Asserts `perm` is a genuine permutation of `0..count`.
    fn assert_permutation_valid(perm: &[usize], count: usize) {
        assert_eq!(perm.len(), count);
        let mut sorted = perm.to_vec();
        sorted.sort_unstable();
        let identity: Vec<usize> = (0..count).collect();
        assert_eq!(sorted, identity);
    }

    fn obj(id: u32, az: f32, gain: f32) -> AudioObject {
        AudioObject::new(ObjectId(id), direction_from_angles(az, 0.0), gain)
    }

    #[test]
    fn discrete_within_budget_keeps_every_object() {
        let objects = [obj(0, 0.0, 1.0), obj(1, 90.0, 0.8), obj(2, -90.0, 0.6)];
        let cap = PlatformCapability::discrete_objects(
            8,
            BedLayout::Surround7_1_4,
            ChannelOrder::EngineCanonical,
            true,
        );
        let plan = negotiate_delivery(&objects, cap);

        assert_eq!(plan.object_count, 3);
        assert_eq!(plan.discrete_count, 3);
        assert!(!plan.degraded);
        assert_eq!(plan.format, OutputFormat::Bed);
        assert_eq!(plan.channel_order.len(), BedLayout::Surround7_1_4.channel_count());
    }

    #[test]
    fn discrete_over_budget_degrades_and_reduces() {
        let objects: Vec<AudioObject> = (0..6).map(|i| obj(i, i as f32 * 20.0, 1.0)).collect();
        let cap = PlatformCapability::discrete_objects(
            3,
            BedLayout::Surround7_1_4,
            ChannelOrder::LfeLast,
            false,
        );
        let plan = negotiate_delivery(&objects, cap);

        assert_eq!(plan.object_count, 6);
        assert_eq!(plan.discrete_count, 3);
        assert!(plan.degraded);
        assert_permutation_valid(&plan.channel_order, BedLayout::Surround7_1_4.channel_count());
    }

    #[test]
    fn bed_only_folds_everything_and_never_reports_degraded() {
        let objects = [obj(0, 10.0, 1.0), obj(1, 120.0, 1.0)];
        let cap = PlatformCapability::bed_only(BedLayout::Surround5_1_4, ChannelOrder::LfeLast);
        let plan = negotiate_delivery(&objects, cap);

        assert_eq!(plan.discrete_count, 0);
        assert!(!plan.degraded);
        match &plan.output.payload {
            RenderPayload::Bed(rows) => assert_eq!(rows.len(), 2),
            other => panic!("expected bed payload, got {other:?}"),
        }
    }

    #[test]
    fn binaural_produces_direction_parameters() {
        let objects = [obj(0, 45.0, 1.0), obj(1, -45.0, 1.0)];
        let cap = PlatformCapability::binaural(BedLayout::Stereo, true);
        let plan = negotiate_delivery(&objects, cap);

        assert_eq!(plan.format, OutputFormat::Binaural);
        assert!(plan.channel_order.is_empty());
        match &plan.output.payload {
            RenderPayload::Binaural(dirs) => assert_eq!(dirs.len(), 2),
            other => panic!("expected binaural payload, got {other:?}"),
        }
    }

    #[test]
    fn ambisonic_encodes_at_platform_order() {
        let objects = [obj(0, 30.0, 1.0)];
        let cap = PlatformCapability::ambisonic(2, BedLayout::Surround7_1_4, false);
        let plan = negotiate_delivery(&objects, cap);

        match &plan.output.payload {
            RenderPayload::Ambisonic { order, rows } => {
                assert_eq!(*order, 2);
                assert_eq!(rows.len(), 1);
            }
            other => panic!("expected ambisonic payload, got {other:?}"),
        }
    }

    #[test]
    fn empty_scene_is_not_degraded() {
        let cap = PlatformCapability::discrete_objects(
            4,
            BedLayout::Surround7_1_4,
            ChannelOrder::EngineCanonical,
            false,
        );
        let plan = negotiate_delivery(&[], cap);
        assert_eq!(plan.object_count, 0);
        assert_eq!(plan.discrete_count, 0);
        assert!(!plan.degraded);
    }
}
