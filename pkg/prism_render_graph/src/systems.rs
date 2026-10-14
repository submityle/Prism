//! The render-graph schedule's system-set vocabulary.
//!
//! Bevy drives every frame through a root render-graph **schedule** whose
//! systems hang off four ordered sets — `RenderGraphSystems`. Prism keeps that
//! frame order, so the labels and their ordering live here, beside the frame
//! graph they name:
//!
//! | set | meaning |
//! | --- | --- |
//! | [`Begin`](RenderGraphSystems::Begin) | per-frame setup, before anything records |
//! | [`Render`](RenderGraphSystems::Render) | the graph's passes record |
//! | [`Submit`](RenderGraphSystems::Submit) | the buffers recorded above submit |
//! | [`Finish`](RenderGraphSystems::Finish) | per-frame finalization |
//!
//! The order is a *weak* chain — each set before the next, exactly as Bevy
//! configures its root `RenderGraph` schedule. It is carried as data
//! ([`ORDER`](RenderGraphSystems::ORDER) and
//! [`ORDER_EDGES`](RenderGraphSystems::ORDER_EDGES)), so there is nothing to
//! keep in sync but this table and no schedule machinery is needed to describe
//! it.
//!
//! # Why these are labels and edges, not `prism_ecs::SystemSet` impls
//!
//! The graph crate is deliberately dependency-free — `no_std + alloc` over
//! [`prism_render_driver`] alone — while the schedule API lives in `prism_ecs`.
//! A `SystemSet` impl can only be written where the trait *or* the label type
//! is local, so the renderer (which owns the schedule and depends on
//! `prism_ecs`) implements it for this vocabulary, keying each variant's set id
//! on [`variant_index`](RenderGraphSystems::variant_index):
//!
//! ```ignore
//! // In the renderer, which has `prism_ecs`:
//! #[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
//! struct RenderGraphSet(RenderGraphSystems);
//!
//! impl SystemSet for RenderGraphSet {
//!     fn set_id(&self) -> SystemSetId {
//!         SystemSetId::with::<RenderGraphSystems>(self.0.variant_index())
//!     }
//! }
//!
//! for (before, after) in RenderGraphSystems::ORDER_EDGES {
//!     schedule.configure_set(
//!         RenderGraphSet(after),
//!         SetConfig::new().after(RenderGraphSet(before)),
//!     );
//! }
//!
//! // ...and every graph-node system declares its set:
//! schedule.add_systems(reclaim_completed_handles.in_set(RenderGraphSet(RenderGraphSystems::Finish)));
//! ```
//!
//! [`prism_render_driver`]: prism_render_driver

/// The ordered system sets of the renderer's root render-graph schedule.
///
/// The Prism counterpart of Bevy's `RenderGraphSystems`: named anchors a
/// frame's graph-node systems hang off, so the order is imposed by the set a
/// system declares rather than by every plugin wiring edges by hand.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum RenderGraphSystems {
    /// Per-frame setup, before anything records.
    Begin,
    /// The graph's passes record here.
    Render,
    /// The command buffers recorded during [`Render`](Self::Render) submit.
    Submit,
    /// Per-frame finalization, after rendering and submission.
    Finish,
}

impl RenderGraphSystems {
    /// The sets in run order: `Begin → Render → Submit → Finish`.
    ///
    /// This is the weak chain Bevy configures for its root render-graph
    /// schedule.
    pub const ORDER: [Self; 4] = [Self::Begin, Self::Render, Self::Submit, Self::Finish];

    /// The ordering policy as data: each set in [`ORDER`](Self::ORDER) weakly
    /// before the next.
    ///
    /// A renderer consumes this to configure its schedule (see the module
    /// docs); the pairs are the whole policy.
    pub const ORDER_EDGES: [(Self, Self); 3] = [
        (Self::Begin, Self::Render),
        (Self::Render, Self::Submit),
        (Self::Submit, Self::Finish),
    ];

    /// This variant's index in [`ORDER`](Self::ORDER).
    ///
    /// A `prism_ecs` consumer keys the set's `SystemSetId` on it —
    /// `SystemSetId::with::<RenderGraphSystems>(set.variant_index())` — so every
    /// variant is a distinct set while the label stays one type.
    #[must_use]
    pub const fn variant_index(self) -> u64 {
        match self {
            Self::Begin => 0,
            Self::Render => 1,
            Self::Submit => 2,
            Self::Finish => 3,
        }
    }
}

/// Applies [`RenderGraphSystems`]' ordering to a render schedule.
///
/// The graph crate carries the policy but not the schedule API — that lives in
/// `prism_ecs`, which this crate deliberately does not depend on — so
/// `order(before, after)` is called once per edge and the caller runs it
/// against its own schedule:
///
/// ```
/// use prism_render_graph::{configure_render_graph_sets, RenderGraphSystems};
///
/// let mut applied = Vec::new();
/// configure_render_graph_sets(|before, after| applied.push((before, after)));
/// assert_eq!(applied.as_slice(), RenderGraphSystems::ORDER_EDGES);
/// ```
pub fn configure_render_graph_sets(mut order: impl FnMut(RenderGraphSystems, RenderGraphSystems)) {
    for (before, after) in RenderGraphSystems::ORDER_EDGES {
        order(before, after);
    }
}

#[cfg(test)]
#[path = "../text/unit/systems.rs"]
mod tests;
