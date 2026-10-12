//! Render system sets and phase labels.
//!
//! These are the Prism equivalents of Bevy's `RenderSystems` and the
//! core-pipeline phase labels (`Opaque3d`, `Core3dSystems`): named
//! [`SystemSet`]s that renderer systems hang off so a frame's work is ordered
//! the same way regardless of which plugin registered it.
//!
//! The stages are the renderer's frame order — extract, build views, queue
//! draws, prepare the GPU resources the queued order needs, record phases,
//! clean up — and the phases are the material-model-agnostic draw buckets of
//! the single execution path (`prism_material_pipeline_refactor_zh.md`
//! §3.5/§8.5): the four lines (PBR / NPR / hybrid / custom) are *response
//! modes*, not separate phases, so the phase set stays small and the model
//! differences ride on the pipeline key.

use prism_ecs::schedule::{Schedule, SetConfig, SystemSet, SystemSetId};
use prism_render_graph::RenderGraphSystems;

/// The ordered stages of a render frame.
///
/// The names and the order are the renderer's, not a fresh invention: bevy's
/// `RenderSystems` orders `… → Queue → Prepare → Render → Cleanup`, where
/// queueing (specialize + bin) happens **before** GPU resources and bind groups
/// are prepared, because preparation walks the queued/sorted order. Prism keeps
/// that order and the sub-stages the renderer's systems hang off
/// ([`QueueMeshes`](Self::QueueMeshes),
/// [`PrepareResources`](Self::PrepareResources),
/// [`PrepareResourcesFlush`](Self::PrepareResourcesFlush),
/// [`PrepareBindGroups`](Self::PrepareBindGroups)), so ported systems keep their
/// relative order without every one of them declaring edges by hand.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum RenderSystems {
    /// Pull the frame's inputs out of the main world.
    Extract,
    /// Prepare the GPU form of the assets the extract carried across.
    PrepareAssets,
    /// Create the frame's views (its view family) before anything is queued or
    /// prepared against them.
    CreateViews,
    /// Prepare the auxiliary views queueing consumes (shadow, reflection,
    /// capture).
    PrepareViews,
    /// Turn the visible set into draw commands / bins.
    Queue,
    /// The mesh-queueing sub-stage within [`Queue`](Self::Queue).
    QueueMeshes,
    /// Build/upload the GPU resources and bind groups the frame's passes read,
    /// in the order the queue produced.
    Prepare,
    /// The resource-construction sub-stage within [`Prepare`](Self::Prepare).
    PrepareResources,
    /// Flush the resources built in
    /// [`PrepareResources`](Self::PrepareResources) before bind groups are
    /// created from them.
    PrepareResourcesFlush,
    /// The bind-group sub-stage within [`Prepare`](Self::Prepare).
    PrepareBindGroups,
    /// Record the graph's passes.
    Phase,
    /// Release per-frame scratch and publish stats.
    Cleanup,
}

impl RenderSystems {
    /// The top-level stage order, earliest first.
    ///
    /// Sub-stages are not listed: they are ordered *inside* the stage that owns
    /// them (see [`configure_render_sets`]), so a system registered in
    /// [`QueueMeshes`](Self::QueueMeshes) needs no edge to one in
    /// [`PrepareResources`](Self::PrepareResources).
    pub const ORDER: [Self; 8] = [
        Self::Extract,
        Self::PrepareAssets,
        Self::CreateViews,
        Self::PrepareViews,
        Self::Queue,
        Self::Prepare,
        Self::Phase,
        Self::Cleanup,
    ];

    const fn discriminant(self) -> u64 {
        match self {
            Self::Extract => 0,
            Self::PrepareAssets => 1,
            Self::CreateViews => 2,
            Self::PrepareViews => 3,
            Self::Queue => 4,
            Self::QueueMeshes => 5,
            Self::Prepare => 6,
            Self::PrepareResources => 7,
            Self::PrepareResourcesFlush => 8,
            Self::PrepareBindGroups => 9,
            Self::Phase => 10,
            Self::Cleanup => 11,
        }
    }
}

impl SystemSet for RenderSystems {
    fn set_id(&self) -> SystemSetId {
        SystemSetId::with::<Self>(self.discriminant())
    }
}

/// The draw phases of the single execution path.
///
/// A phase is a bucket of the frame graph, not a material model: the four lines
/// (PBR / NPR / hybrid / custom) share these phases and differ only in the
/// response function they dispatch to (`respond()`), so the custom line needs no
/// phase of its own beyond [`RenderPhase::Opaque`]/[`RenderPhase::Transparent`].
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum RenderPhase {
    /// Depth-tested, opaque draws.
    Opaque,
    /// Order-independent / sorted transparent draws.
    Transparent,
    /// Stylized (NPR) overlays: outlines, ramps, screen-space stylization.
    Stylized,
    /// Project-injected custom passes.
    Custom,
}

impl RenderPhase {
    /// The phase order, earliest first.
    pub const ORDER: [Self; 4] = [
        Self::Opaque,
        Self::Transparent,
        Self::Stylized,
        Self::Custom,
    ];

    /// The number of phases, i.e. the length of [`RenderPhase::ORDER`].
    pub const COUNT: usize = 4;

    /// This phase's index in [`RenderPhase::ORDER`], for array-indexed queues.
    #[must_use]
    pub const fn index(self) -> usize {
        self.discriminant() as usize
    }

    const fn discriminant(self) -> u64 {
        match self {
            Self::Opaque => 0,
            Self::Transparent => 1,
            Self::Stylized => 2,
            Self::Custom => 3,
        }
    }
}

impl SystemSet for RenderPhase {
    fn set_id(&self) -> SystemSetId {
        SystemSetId::with::<Self>(self.discriminant())
    }
}

/// Orders the render stages and phases on `schedule`.
///
/// Called once when the render schedule is assembled, so an app that registers
/// a system `in_set(RenderSystems::Queue)` does not need to know about every
/// other stage. Phases are ordered within the [`RenderSystems::Phase`] stage and
/// across each other.
///
/// The sub-stages nest inside their owner:
/// [`QueueMeshes`](RenderSystems::QueueMeshes) inside
/// [`Queue`](RenderSystems::Queue), and
/// [`PrepareResources`](RenderSystems::PrepareResources) →
/// [`PrepareResourcesFlush`](RenderSystems::PrepareResourcesFlush) →
/// [`PrepareBindGroups`](RenderSystems::PrepareBindGroups) inside
/// [`Prepare`](RenderSystems::Prepare).
pub fn configure_render_sets(schedule: &mut Schedule) {
    for pair in RenderSystems::ORDER.windows(2) {
        schedule.configure_set(pair[1], SetConfig::new().after(pair[0]));
    }
    // Mesh queueing is part of queueing.
    schedule.configure_set(
        RenderSystems::QueueMeshes,
        SetConfig::new().after(RenderSystems::Queue),
    );
    schedule.configure_set(
        RenderSystems::QueueMeshes,
        SetConfig::new().before(RenderSystems::Prepare),
    );
    // Resources → flush → bind groups, all inside prepare.
    for (before, after) in [
        (
            RenderSystems::PrepareResources,
            RenderSystems::PrepareResourcesFlush,
        ),
        (
            RenderSystems::PrepareResourcesFlush,
            RenderSystems::PrepareBindGroups,
        ),
    ] {
        schedule.configure_set(after, SetConfig::new().after(before));
    }
    for sub in [
        RenderSystems::PrepareResources,
        RenderSystems::PrepareResourcesFlush,
        RenderSystems::PrepareBindGroups,
    ] {
        schedule.configure_set(sub, SetConfig::new().after(RenderSystems::Prepare));
        schedule.configure_set(sub, SetConfig::new().before(RenderSystems::Phase));
    }
    for pair in RenderPhase::ORDER.windows(2) {
        schedule.configure_set(pair[1], SetConfig::new().after(pair[0]));
    }
    // Phases run inside the record stage, which is *after* every prepare
    // sub-stage: a phase item records with the bind groups preparation built, so
    // ordering against `Prepare` alone would leave the sub-stages unordered
    // against the record.
    for phase in RenderPhase::ORDER {
        schedule.configure_set(
            phase,
            SetConfig::new().after(RenderSystems::PrepareBindGroups),
        );
        schedule.configure_set(phase, SetConfig::new().before(RenderSystems::Cleanup));
    }
}

/// The renderer's schedulable spelling of the frame graph's own system sets.
///
/// [`RenderGraphSystems`] lives in `prism_render_graph`, which is deliberately
/// dependency-free (`no_std` over the RHI) and therefore cannot name
/// `prism_ecs`'s [`SystemSet`]; the orphan rule also forbids implementing a
/// foreign trait for a foreign type here. So the graph *names* the sets and the
/// renderer — which owns the schedule — makes them schedulable, exactly as the
/// port doc splits it (`docs/prism_render_scene_rhi_port_zh.md` §3, "自建系统集").
///
/// Graph nodes register with `in_set(RenderGraphSet(RenderGraphSystems::Render))`
/// and never wire edges by hand; the order is
/// [`RenderGraphSystems::ORDER_EDGES`].
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct RenderGraphSet(pub RenderGraphSystems);

impl RenderGraphSet {
    /// The graph sets in run order: begin → render → submit → finish.
    pub const ORDER: [Self; 4] = [
        Self(RenderGraphSystems::Begin),
        Self(RenderGraphSystems::Render),
        Self(RenderGraphSystems::Submit),
        Self(RenderGraphSystems::Finish),
    ];
}

impl SystemSet for RenderGraphSet {
    fn set_id(&self) -> SystemSetId {
        // Keyed on the variant index, so every variant is a distinct set while
        // the label stays one type — the set id the graph's own vocabulary
        // documents.
        SystemSetId::with::<RenderGraphSystems>(self.0.variant_index())
    }
}

/// Orders the frame graph's system sets on `schedule`.
///
/// The renderer's half of [`prism_render_graph::configure_render_graph_sets`]:
/// the graph carries the ordering policy as data, and this applies it to the
/// schedule that owns the node systems.
pub fn configure_render_graph_sets(schedule: &mut Schedule) {
    prism_render_graph::configure_render_graph_sets(|before, after| {
        schedule.configure_set(RenderGraphSet(after), SetConfig::new().after(RenderGraphSet(before)));
    });
}
