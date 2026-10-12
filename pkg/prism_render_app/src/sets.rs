//! Render system sets and phase labels.
//!
//! These are the Prism equivalents of Bevy's `RenderSystems` and the
//! core-pipeline phase labels (`Opaque3d`, `Core3dSystems`): named
//! [`SystemSet`]s that renderer systems hang off so a frame's work is ordered
//! the same way regardless of which plugin registered it.
//!
//! The stages are the classic frame order — extract, prepare resources, queue
//! draws, record phases, clean up — and the phases are the material-model-
//! agnostic draw buckets of the single execution path
//! (`prism_material_pipeline_refactor_zh.md` §3.5/§8.5): the four lines
//! (PBR / NPR / hybrid / custom) are *response modes*, not separate phases, so
//! the phase set stays small and the model differences ride on the pipeline key.

use prism_ecs::schedule::{Schedule, SetConfig, SystemSet, SystemSetId};

/// The ordered stages of a render frame.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum RenderSystems {
    /// Pull the frame's inputs out of the main world.
    Extract,
    /// Build/upload the GPU resources the frame's passes will read.
    Prepare,
    /// Turn the visible set into draw commands / bins.
    Queue,
    /// Record the graph's passes.
    Phase,
    /// Release per-frame scratch and publish stats.
    Cleanup,
}

impl RenderSystems {
    /// The stage order, earliest first.
    pub const ORDER: [Self; 5] = [
        Self::Extract,
        Self::Prepare,
        Self::Queue,
        Self::Phase,
        Self::Cleanup,
    ];

    const fn discriminant(self) -> u64 {
        match self {
            Self::Extract => 0,
            Self::Prepare => 1,
            Self::Queue => 2,
            Self::Phase => 3,
            Self::Cleanup => 4,
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
pub fn configure_render_sets(schedule: &mut Schedule) {
    for pair in RenderSystems::ORDER.windows(2) {
        schedule.configure_set(pair[1], SetConfig::new().after(pair[0]));
    }
    for pair in RenderPhase::ORDER.windows(2) {
        schedule.configure_set(pair[1], SetConfig::new().after(pair[0]));
    }
    // Phases run inside the record stage.
    for phase in RenderPhase::ORDER {
        schedule.configure_set(phase, SetConfig::new().after(RenderSystems::Queue));
        schedule.configure_set(phase, SetConfig::new().before(RenderSystems::Cleanup));
    }
}
