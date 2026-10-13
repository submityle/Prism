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
mod tests {
    use super::{configure_render_graph_sets, RenderGraphSystems};
    use alloc::vec;
    use alloc::vec::Vec;

    /// The order marker systems run in, given how they were registered and the
    /// set policy they hang off.
    ///
    /// Mirrors `prism_ecs`'s ordering model: a set-ordering edge expands to an
    /// edge between every member system, and the topological sort breaks ties
    /// by registration order (its documented determinism guarantee). Only
    /// [`RenderGraphSystems::ORDER_EDGES`] is consulted for precedence, so a
    /// lost policy edge shows up as a wrong run order rather than passing by
    /// accident.
    fn resolve_run_order(
        registrations: &[(RenderGraphSystems, &'static str)],
    ) -> Vec<&'static str> {
        let node_count = registrations.len();
        let mut in_degree = vec![0usize; node_count];
        let mut edges: Vec<Vec<usize>> = vec![Vec::new(); node_count];
        for (index, (before, _)) in registrations.iter().enumerate() {
            for (other, (after, _)) in registrations.iter().enumerate() {
                if index != other && precedes(*before, *after) {
                    edges[index].push(other);
                    in_degree[other] += 1;
                }
            }
        }

        let mut order = Vec::with_capacity(node_count);
        let mut emitted = vec![false; node_count];
        for _ in 0..node_count {
            let next = (0..node_count)
                .find(|&index| !emitted[index] && in_degree[index] == 0)
                .expect("the set policy is acyclic");
            emitted[next] = true;
            order.push(registrations[next].1);
            for &dependent in &edges[next] {
                in_degree[dependent] -= 1;
            }
        }
        order
    }

    /// Whether the set policy orders `before` ahead of `after` — the transitive
    /// closure of [`RenderGraphSystems::ORDER_EDGES`].
    fn precedes(before: RenderGraphSystems, after: RenderGraphSystems) -> bool {
        let mut frontier = vec![before];
        let mut seen = vec![before];
        while let Some(current) = frontier.pop() {
            for (from, to) in RenderGraphSystems::ORDER_EDGES {
                if from == current && !seen.contains(&to) {
                    if to == after {
                        return true;
                    }
                    seen.push(to);
                    frontier.push(to);
                }
            }
        }
        false
    }

    #[test]
    fn order_and_edges_are_the_weak_chain_begin_to_finish() {
        assert_eq!(
            RenderGraphSystems::ORDER,
            [
                RenderGraphSystems::Begin,
                RenderGraphSystems::Render,
                RenderGraphSystems::Submit,
                RenderGraphSystems::Finish,
            ],
            "the sets run Begin → Render → Submit → Finish"
        );
        assert_eq!(
            RenderGraphSystems::ORDER_EDGES,
            [
                (RenderGraphSystems::Begin, RenderGraphSystems::Render),
                (RenderGraphSystems::Render, RenderGraphSystems::Submit),
                (RenderGraphSystems::Submit, RenderGraphSystems::Finish),
            ],
            "the policy is exactly the weak chain over ORDER"
        );
    }

    #[test]
    fn render_graph_sets_impose_their_order_when_registered_out_of_order() {
        // Registered out of order across sets (and, within a set, in the order
        // the schedule keeps: registration order is the tie-break).
        let registrations = [
            (RenderGraphSystems::Finish, "map-readback"),
            (RenderGraphSystems::Submit, "submit-pending-buffers"),
            (RenderGraphSystems::Render, "record-graph"),
            (RenderGraphSystems::Begin, "reset-frame-scratch"),
            (RenderGraphSystems::Finish, "reclaim-completed-handles"),
            (RenderGraphSystems::Begin, "collect-last-frame-readback"),
        ];

        assert_eq!(
            resolve_run_order(&registrations),
            [
                "reset-frame-scratch",
                "collect-last-frame-readback",
                "record-graph",
                "submit-pending-buffers",
                "map-readback",
                "reclaim-completed-handles",
            ],
            "per-frame setup runs first, then the graph records, then the buffers \
             submit, then finalization — whatever order they were registered in"
        );
    }

    #[test]
    fn configure_emits_every_ordering_edge_in_order() {
        let mut applied = Vec::new();
        configure_render_graph_sets(|before, after| applied.push((before, after)));
        assert_eq!(applied.as_slice(), RenderGraphSystems::ORDER_EDGES);
    }
}
