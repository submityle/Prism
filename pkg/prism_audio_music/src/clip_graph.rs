//! **Clip graph**: the Godot-style interactive flow of clips and transitions.
//!
//! A clip graph models interactive music as a directed graph: nodes are
//! [`Clip`]s (each wrapping a [`crate::segment::Segment`]) and edges are
//! [`ClipEdge`]s carrying a [`TriggerCondition`] and a
//! [`crate::transition::Transition`]. Playback starts at [`ClipGraph::start`]
//! and, whenever gameplay posts an event, the planner walks the edges leaving
//! the current clip in authored order and takes the first whose condition
//! matches the live [`TransitionContext`] (the current gameplay branch and
//! intensity). This expresses horizontal re-sequencing (branch to a different
//! clip) and conditional musical form as plain data.
//!
//! Authoring data can contain cycles (a loop of `Immediate` edges), so every
//! traversal is bounded by [`MAX_GRAPH_DEPTH`] to guarantee termination.
//!
//! # Provenance
//!
//! This module contains **no Unreal Engine, Unity, Godot, Wwise, FMOD, Steam
//! Audio, or Google Resonance Audio source or derived code**. The
//! nodes-and-conditional-edges flow is reconstructed from first principles over
//! plain data and classic first-match graph traversal. No AI/ML.
//!
//! # Relationship
//!
//! A [`ClipGraph`] is stored in [`crate::model::MusicModel`] and traversed by
//! [`crate::system::MusicSystem::clip_graph_event`], which evaluates each
//! [`ClipEdge`]'s [`TriggerCondition`] against a [`TransitionContext`] and
//! applies the edge's [`crate::transition::Transition`].

use alloc::vec::Vec;

use prism_audio_core::math::Sample;

use crate::id::{BranchId, ClipId, GraphId, SegmentId};
use crate::transition::Transition;

/// Hard ceiling on clip-graph traversal depth per event.
///
/// Authoring data can reference clips in a cycle (for example a ring of
/// `Immediate` edges); this bound guarantees a single
/// [`crate::system::MusicSystem::clip_graph_event`] call terminates without an
/// unbounded action stream. It is deliberately generous: real clip graphs chain
/// only a handful of immediate hops before settling on a quantized edge.
pub const MAX_GRAPH_DEPTH: usize = 32;

/// A node in a [`ClipGraph`]: one clip wrapping a music segment.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub struct Clip {
    /// Stable id this clip is referenced by (edge endpoints, graph start).
    pub id: ClipId,
    /// The segment this clip plays.
    pub segment: SegmentId,
}

impl Clip {
    /// Builds a clip wrapping `segment`.
    #[must_use]
    pub fn new(id: ClipId, segment: SegmentId) -> Self {
        Self { id, segment }
    }
}

/// A condition gating whether a [`ClipEdge`] may be taken.
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub enum TriggerCondition {
    /// Always matches (an unconditional default edge).
    Always,
    /// Matches when the live gameplay branch equals this value.
    OnBranch(BranchId),
    /// Matches when the live intensity is at or above this threshold.
    IntensityAtLeast(Sample),
    /// Matches when the live intensity is strictly below this threshold.
    IntensityBelow(Sample),
}

/// The live gameplay context evaluated against a [`TriggerCondition`].
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub struct TransitionContext {
    /// The active gameplay branch, if any.
    pub branch: Option<BranchId>,
    /// The live intensity value.
    pub intensity: Sample,
}

impl TransitionContext {
    /// Builds a context with the given branch and intensity.
    #[must_use]
    pub fn new(branch: Option<BranchId>, intensity: Sample) -> Self {
        Self { branch, intensity }
    }

    /// Returns whether this context satisfies `condition`.
    #[must_use]
    pub fn matches(&self, condition: &TriggerCondition) -> bool {
        match *condition {
            TriggerCondition::Always => true,
            TriggerCondition::OnBranch(branch) => self.branch == Some(branch),
            TriggerCondition::IntensityAtLeast(threshold) => self.intensity >= threshold,
            TriggerCondition::IntensityBelow(threshold) => self.intensity < threshold,
        }
    }
}

/// A directed edge between two clips with a trigger condition and transition.
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub struct ClipEdge {
    /// Source clip the edge leaves.
    pub from: ClipId,
    /// Destination clip the edge enters.
    pub to: ClipId,
    /// Condition that must match the live context for the edge to be taken.
    pub condition: TriggerCondition,
    /// How the switch is quantized and faded when the edge is taken.
    pub transition: Transition,
}

impl ClipEdge {
    /// Builds an edge from `from` to `to` gated by `condition` using
    /// `transition`.
    #[must_use]
    pub fn new(
        from: ClipId,
        to: ClipId,
        condition: TriggerCondition,
        transition: Transition,
    ) -> Self {
        Self {
            from,
            to,
            condition,
            transition,
        }
    }
}

/// A directed graph of [`Clip`]s connected by conditional [`ClipEdge`]s.
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub struct ClipGraph {
    /// Stable id this graph is referenced by.
    pub id: GraphId,
    /// The clip playback begins at.
    pub start: ClipId,
    /// All clip nodes, in authored order.
    pub clips: Vec<Clip>,
    /// All edges, in authored order (first match wins per source clip).
    pub edges: Vec<ClipEdge>,
}

impl ClipGraph {
    /// Builds a graph whose playback starts at `start` with no clips or edges.
    #[must_use]
    pub fn new(id: GraphId, start: ClipId) -> Self {
        Self {
            id,
            start,
            clips: Vec::new(),
            edges: Vec::new(),
        }
    }

    /// Appends a clip, returning `self` for builder-style chaining.
    #[must_use]
    pub fn with_clip(mut self, clip: Clip) -> Self {
        self.clips.push(clip);
        self
    }

    /// Appends an edge, returning `self` for builder-style chaining.
    #[must_use]
    pub fn with_edge(mut self, edge: ClipEdge) -> Self {
        self.edges.push(edge);
        self
    }

    /// Looks up a clip node by id.
    #[must_use]
    pub fn clip(&self, id: ClipId) -> Option<&Clip> {
        self.clips.iter().find(|c| c.id == id)
    }

    /// Selects the first edge leaving `from` whose condition matches `ctx`,
    /// in authored order.
    #[must_use]
    pub fn select_edge(&self, from: ClipId, ctx: &TransitionContext) -> Option<&ClipEdge> {
        self.edges
            .iter()
            .find(|e| e.from == from && ctx.matches(&e.condition))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::transition::{Fade, Transition, TransitionType};

    fn graph() -> ClipGraph {
        // Clip 1 has two outgoing edges: a conditional high-intensity edge
        // listed first, then an unconditional default.
        ClipGraph::new(GraphId::new(1), ClipId::new(1))
            .with_clip(Clip::new(ClipId::new(1), SegmentId::new(10)))
            .with_clip(Clip::new(ClipId::new(2), SegmentId::new(11)))
            .with_clip(Clip::new(ClipId::new(3), SegmentId::new(12)))
            .with_edge(ClipEdge::new(
                ClipId::new(1),
                ClipId::new(2),
                TriggerCondition::IntensityAtLeast(0.8),
                Transition::new(TransitionType::NextBar, Fade::crossfade(256)),
            ))
            .with_edge(ClipEdge::new(
                ClipId::new(1),
                ClipId::new(3),
                TriggerCondition::Always,
                Transition::immediate(),
            ))
    }

    #[test]
    fn condition_matching() {
        let ctx = TransitionContext::new(Some(BranchId::new(2)), 0.5);
        assert!(ctx.matches(&TriggerCondition::Always));
        assert!(ctx.matches(&TriggerCondition::OnBranch(BranchId::new(2))));
        assert!(!ctx.matches(&TriggerCondition::OnBranch(BranchId::new(3))));
        assert!(ctx.matches(&TriggerCondition::IntensityBelow(0.6)));
        assert!(!ctx.matches(&TriggerCondition::IntensityAtLeast(0.6)));
    }

    #[test]
    fn first_matching_edge_wins_in_authored_order() {
        let g = graph();
        // High intensity: the conditional edge (listed first) is chosen.
        let hot = TransitionContext::new(None, 0.9);
        assert_eq!(g.select_edge(ClipId::new(1), &hot).unwrap().to, ClipId::new(2));
        // Low intensity: falls through to the unconditional default.
        let cold = TransitionContext::new(None, 0.1);
        assert_eq!(g.select_edge(ClipId::new(1), &cold).unwrap().to, ClipId::new(3));
    }

    #[test]
    fn no_edge_when_none_match() {
        let g = ClipGraph::new(GraphId::new(1), ClipId::new(1))
            .with_clip(Clip::new(ClipId::new(1), SegmentId::new(10)))
            .with_edge(ClipEdge::new(
                ClipId::new(1),
                ClipId::new(2),
                TriggerCondition::IntensityAtLeast(0.9),
                Transition::immediate(),
            ));
        let ctx = TransitionContext::new(None, 0.0);
        assert!(g.select_edge(ClipId::new(1), &ctx).is_none());
    }

    #[test]
    fn clip_lookup() {
        let g = graph();
        assert_eq!(g.clip(ClipId::new(2)).unwrap().segment, SegmentId::new(11));
        assert!(g.clip(ClipId::new(99)).is_none());
    }
}
