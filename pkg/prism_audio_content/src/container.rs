//! **Containers**: authored nodes that choose, sequence, or layer their child
//! playables so a single event produces varied, context-sensitive output.
//!
//! A container resolves *one level*: given its runtime cursor/history and the
//! current game context, it returns the child [`Playable`]s to play this time
//! (each with a relative gain). Children may themselves be containers; the
//! recursive walk that flattens a tree into concrete leaf picks lives in
//! [`crate::system`], keeping this module a pure per-node selector.
//!
//! Supported kinds (aligned with the Wwise/FMOD/Unity container vocabulary,
//! modelled from scratch):
//!
//! - **Random** — pick one child, optionally weighted, with a no-repeat window
//!   or a shuffle-bag that exhausts every child before repeating.
//! - **Sequence** — step through children in order (forward-loop, ping-pong, or
//!   play-once-and-stop).
//! - **Blend** — crossfade several layers by a blend position in `[0, 1]`, each
//!   layer carrying its own position→gain curve.
//! - **Switch** — select the branch bound to the currently active switch.
//! - **Scatter** — pick several children at once (e.g. a flock), count chosen
//!   randomly within an authored range; spatial placement is left to the
//!   spatial layer.
//!
//! # Provenance
//!
//! This module contains **no Unreal Engine, Unity, Godot, Wwise, FMOD, Steam
//! Audio, or Google Resonance Audio source or derived code**. The container
//! taxonomy is reconstructed from first principles over plain data and the
//! deterministic [`crate::rng`]. No AI/ML.
//!
//! # Relationship
//!
//! Resolution is driven by [`crate::system::EventSystem`], which owns the
//! [`ContainerState`] cursors, supplies the active [`crate::switch`] value and
//! blend position, and recurses into child containers. Blend layer gains use
//! [`crate::curve::ParameterCurve`]; weighted picks use [`crate::rng::Rng`].

use alloc::vec::Vec;

use prism_audio_core::math::Sample;

use crate::curve::ParameterCurve;
use crate::id::{ContainerId, Playable, RtpcId, SwitchGroupId, SwitchId};
use crate::rng::Rng;

/// How a [`ContainerKind::Random`] draws its child.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub enum RandomMode {
    /// Independent weighted draws; a no-repeat window suppresses immediate
    /// repetition without otherwise biasing the distribution.
    Standard,
    /// Shuffle-bag: every child is played exactly once (order shuffled) before
    /// any child repeats, guaranteeing even coverage.
    Shuffle,
}

/// How a [`ContainerKind::Sequence`] advances its cursor.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub enum SequenceMode {
    /// `0,1,2,...,n-1,0,1,...` wrapping forever.
    Loop,
    /// `0,1,...,n-1,n-2,...,1,0,1,...` bouncing at the ends.
    PingPong,
    /// `0,1,...,n-1` then silence (no further picks).
    Once,
}

/// A random child with its selection weight.
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub struct WeightedChild {
    /// The child to play if selected.
    pub playable: Playable,
    /// Relative selection weight; non-positive weights are treated as zero.
    pub weight: Sample,
}

impl WeightedChild {
    /// Builds a weighted child.
    #[must_use]
    pub fn new(playable: Playable, weight: Sample) -> Self {
        Self { playable, weight }
    }

    /// Builds a child with unit weight.
    #[must_use]
    pub fn uniform(playable: Playable) -> Self {
        Self { playable, weight: 1.0 }
    }
}

/// One layer of a blend container.
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub struct BlendLayer {
    /// The layer's playable.
    pub playable: Playable,
    /// Maps the blend position in `[0, 1]` to this layer's gain in decibels.
    pub gain_db: ParameterCurve,
}

impl BlendLayer {
    /// Builds a blend layer.
    #[must_use]
    pub fn new(playable: Playable, gain_db: ParameterCurve) -> Self {
        Self { playable, gain_db }
    }
}

/// A branch of a switch container: the switch value it answers to and the child
/// it selects.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub struct SwitchBranch {
    /// The switch value selecting this branch.
    pub switch: SwitchId,
    /// The child played when that switch is active.
    pub playable: Playable,
}

impl SwitchBranch {
    /// Builds a switch branch.
    #[must_use]
    pub fn new(switch: SwitchId, playable: Playable) -> Self {
        Self { switch, playable }
    }
}

/// The behaviour of a container, with its authored child data.
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
#[non_exhaustive]
pub enum ContainerKind {
    /// Weighted/shuffled random selection of one child.
    Random {
        /// Candidate children with weights.
        children: Vec<WeightedChild>,
        /// Draw strategy.
        mode: RandomMode,
        /// Number of most-recent picks that may not repeat (Standard mode).
        avoid_repeat: usize,
    },
    /// Ordered stepping through children.
    Sequence {
        /// Children in play order.
        children: Vec<Playable>,
        /// Cursor advance strategy.
        mode: SequenceMode,
    },
    /// Crossfade of layers by a blend position derived from a game
    /// parameter (RTPC).
    Blend {
        /// Game parameter whose normalised value in `[0, 1]` drives the
        /// crossfade; [`crate::system::EventSystem`] reads the live RTPC
        /// value, normalises it against its definition range, and passes the
        /// result as the blend position.
        rtpc: RtpcId,
        /// The layers to crossfade.
        layers: Vec<BlendLayer>,
    },
    /// Branch selection by the active switch of a bound switch group.
    Switch {
        /// Switch group whose active value selects the branch;
        /// [`crate::system::EventSystem`] reads the active switch for this
        /// group (scoped to the posting game object) and passes it in.
        group: SwitchGroupId,
        /// The branches to choose among.
        branches: Vec<SwitchBranch>,
        /// Fallback child when no branch matches the active switch.
        default: Option<Playable>,
    },
    /// Multi-selection of several children at once.
    Scatter {
        /// Candidate children.
        children: Vec<Playable>,
        /// Smallest number of children picked per resolve.
        min_count: usize,
        /// Largest number of children picked per resolve.
        max_count: usize,
    },
}

/// Authoring definition of a container: its id and behaviour.
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub struct Container {
    /// Stable id of this container.
    pub id: ContainerId,
    /// What the container does.
    pub kind: ContainerKind,
}

impl Container {
    /// Builds a container.
    #[must_use]
    pub fn new(id: ContainerId, kind: ContainerKind) -> Self {
        Self { id, kind }
    }
}

/// One child selected during resolution, with the relative gain to apply.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ContainerPick {
    /// The selected child (leaf or nested container).
    pub playable: Playable,
    /// Relative gain in decibels to apply to this pick.
    pub gain_db: Sample,
}

impl ContainerPick {
    /// Builds a pick at unity gain.
    #[must_use]
    pub fn unity(playable: Playable) -> Self {
        Self { playable, gain_db: 0.0 }
    }
}

/// Mutable per-container runtime state (cursors, histories, shuffle bags).
///
/// Instances are created lazily by [`crate::system`] keyed by container id and
/// reset when playback is reset. The state is deliberately separate from the
/// immutable [`Container`] authoring data so the same definition can be shared.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ContainerState {
    /// Forward cursor for sequence/shuffle traversal.
    cursor: usize,
    /// Ping-pong direction (`true` = ascending).
    ascending: bool,
    /// `true` once a `SequenceMode::Once` has run off its end.
    exhausted: bool,
    /// Indices played recently (no-repeat window) or the current shuffle bag.
    history: Vec<usize>,
}

impl Default for ContainerState {
    fn default() -> Self {
        Self::new()
    }
}

impl ContainerState {
    /// Creates a fresh state (cursor at the start, ascending, empty history).
    #[must_use]
    pub fn new() -> Self {
        Self { cursor: 0, ascending: true, exhausted: false, history: Vec::new() }
    }

    /// Resets the state to its initial condition.
    pub fn reset(&mut self) {
        self.cursor = 0;
        self.ascending = true;
        self.exhausted = false;
        self.history.clear();
    }
}

impl ContainerKind {
    /// Returns the switch group that drives this container, if it is a
    /// [`ContainerKind::Switch`].
    ///
    /// [`crate::system::EventSystem`] uses this to look up the active switch
    /// for the group before resolving.
    #[must_use]
    pub fn switch_group(&self) -> Option<SwitchGroupId> {
        match self {
            Self::Switch { group, .. } => Some(*group),
            _ => None,
        }
    }

    /// Returns the RTPC that drives this container's blend position, if it is
    /// a [`ContainerKind::Blend`].
    ///
    /// [`crate::system::EventSystem`] reads the live value of this RTPC and
    /// normalises it to `[0, 1]` before resolving.
    #[must_use]
    pub fn blend_rtpc(&self) -> Option<RtpcId> {
        match self {
            Self::Blend { rtpc, .. } => Some(*rtpc),
            _ => None,
        }
    }

    /// Appends every child [`Playable`] this container could ever select into
    /// `out`, independent of runtime cursor, active switch, or blend position.
    ///
    /// [`crate::system::EventSystem`] uses this to enumerate the full static
    /// subtree of a container when resolving an [`crate::action::Action::Stop`],
    /// so it can stop every leaf the container may have started. Order follows
    /// authoring order; duplicates (if a child appears twice) are preserved and
    /// de-duplicated by the caller.
    pub fn collect_children(&self, out: &mut Vec<Playable>) {
        match self {
            Self::Random { children, .. } => {
                for child in children {
                    out.push(child.playable);
                }
            }
            Self::Sequence { children, .. } | Self::Scatter { children, .. } => {
                out.extend_from_slice(children);
            }
            Self::Blend { layers, .. } => {
                for layer in layers {
                    out.push(layer.playable);
                }
            }
            Self::Switch { branches, default, .. } => {
                for branch in branches {
                    out.push(branch.playable);
                }
                if let Some(def) = default {
                    out.push(*def);
                }
            }
        }
    }

    /// Resolves one level of selection into `out`.
    ///
    /// `state` carries the mutable cursor/history for this container, `rng`
    /// provides deterministic randomness, and `active_switch` is the resolved
    /// switch value (for [`ContainerKind::Switch`]) and `blend_position` the
    /// current blend position in `[0, 1]` (for [`ContainerKind::Blend`]).
    /// Picks are appended; the function never allocates beyond `out`/`state`
    /// growth and never panics on empty child lists.
    pub fn resolve_into(
        &self,
        state: &mut ContainerState,
        rng: &mut Rng,
        active_switch: Option<SwitchId>,
        blend_position: Sample,
        out: &mut Vec<ContainerPick>,
    ) {
        match self {
            Self::Random { children, mode, avoid_repeat } => {
                Self::resolve_random(children, *mode, *avoid_repeat, state, rng, out);
            }
            Self::Sequence { children, mode } => {
                Self::resolve_sequence(children, *mode, state, out);
            }
            Self::Blend { layers, .. } => {
                Self::resolve_blend(layers, blend_position, out);
            }
            Self::Switch { branches, default, .. } => {
                Self::resolve_switch(branches, *default, active_switch, out);
            }
            Self::Scatter { children, min_count, max_count } => {
                Self::resolve_scatter(children, *min_count, *max_count, rng, out);
            }
        }
    }

    fn resolve_random(
        children: &[WeightedChild],
        mode: RandomMode,
        avoid_repeat: usize,
        state: &mut ContainerState,
        rng: &mut Rng,
        out: &mut Vec<ContainerPick>,
    ) {
        if children.is_empty() {
            return;
        }
        if children.len() == 1 {
            out.push(ContainerPick::unity(children[0].playable));
            return;
        }
        let index = match mode {
            RandomMode::Shuffle => Self::draw_shuffle(children.len(), state, rng),
            RandomMode::Standard => {
                Self::draw_weighted(children, avoid_repeat, state, rng)
            }
        };
        out.push(ContainerPick::unity(children[index].playable));
    }

    fn draw_shuffle(len: usize, state: &mut ContainerState, rng: &mut Rng) -> usize {
        // Refill the bag with a fresh Fisher-Yates shuffle when empty.
        if state.history.is_empty() {
            state.history.extend(0..len);
            // Fisher-Yates (Durstenfeld) in-place shuffle.
            let n = state.history.len();
            for i in (1..n).rev() {
                let j = rng.next_index(i + 1);
                state.history.swap(i, j);
            }
        }
        // Pop from the end of the bag.
        state.history.pop().unwrap_or(0)
    }

    fn draw_weighted(
        children: &[WeightedChild],
        avoid_repeat: usize,
        state: &mut ContainerState,
        rng: &mut Rng,
    ) -> usize {
        let window = avoid_repeat.min(children.len().saturating_sub(1));
        // Candidate weight sum, excluding indices inside the no-repeat window.
        let mut total = 0.0f32;
        for (i, child) in children.iter().enumerate() {
            let w = if child.weight > 0.0 { child.weight } else { 0.0 };
            if !Self::in_recent(&state.history, window, i) {
                total += w;
            }
        }
        let index = if total > 0.0 {
            let mut target = rng.next_unit() * total;
            let mut chosen = 0usize;
            for (i, child) in children.iter().enumerate() {
                if Self::in_recent(&state.history, window, i) {
                    continue;
                }
                let w = if child.weight > 0.0 { child.weight } else { 0.0 };
                if target < w {
                    chosen = i;
                    break;
                }
                target -= w;
                chosen = i;
            }
            chosen
        } else {
            // All candidate weight fell inside the window (or all zero): fall
            // back to an unweighted draw across all children.
            rng.next_index(children.len())
        };
        // Record in the no-repeat window.
        if window > 0 {
            state.history.push(index);
            while state.history.len() > window {
                state.history.remove(0);
            }
        }
        index
    }

    fn in_recent(history: &[usize], window: usize, index: usize) -> bool {
        if window == 0 {
            return false;
        }
        let start = history.len().saturating_sub(window);
        history[start..].contains(&index)
    }

    fn resolve_sequence(
        children: &[Playable],
        mode: SequenceMode,
        state: &mut ContainerState,
        out: &mut Vec<ContainerPick>,
    ) {
        if children.is_empty() || state.exhausted {
            return;
        }
        let n = children.len();
        let index = state.cursor.min(n - 1);
        out.push(ContainerPick::unity(children[index]));
        // Advance the cursor for next time.
        match mode {
            SequenceMode::Loop => {
                state.cursor = (index + 1) % n;
            }
            SequenceMode::Once => {
                if index + 1 >= n {
                    state.exhausted = true;
                } else {
                    state.cursor = index + 1;
                }
            }
            SequenceMode::PingPong => {
                if n == 1 {
                    state.cursor = 0;
                } else if state.ascending {
                    if index + 1 >= n {
                        state.ascending = false;
                        state.cursor = n - 2;
                    } else {
                        state.cursor = index + 1;
                    }
                } else if index == 0 {
                    state.ascending = true;
                    state.cursor = 1;
                } else {
                    state.cursor = index - 1;
                }
            }
        }
    }

    fn resolve_blend(layers: &[BlendLayer], blend_position: Sample, out: &mut Vec<ContainerPick>) {
        let pos = blend_position.clamp(0.0, 1.0);
        for layer in layers {
            let gain_db = layer.gain_db.sample(pos);
            // Skip effectively-silent layers so the resolved stream stays lean.
            if gain_db > -120.0 {
                out.push(ContainerPick { playable: layer.playable, gain_db });
            }
        }
    }

    fn resolve_switch(
        branches: &[SwitchBranch],
        default: Option<Playable>,
        active_switch: Option<SwitchId>,
        out: &mut Vec<ContainerPick>,
    ) {
        if let Some(active) = active_switch
            && let Some(branch) = branches.iter().find(|b| b.switch == active)
        {
            out.push(ContainerPick::unity(branch.playable));
            return;
        }
        if let Some(def) = default {
            out.push(ContainerPick::unity(def));
        }
    }

    fn resolve_scatter(
        children: &[Playable],
        min_count: usize,
        max_count: usize,
        rng: &mut Rng,
        out: &mut Vec<ContainerPick>,
    ) {
        if children.is_empty() {
            return;
        }
        let lo = min_count.min(children.len()).max(1);
        let hi = max_count.min(children.len()).max(lo);
        let count = if hi > lo { lo + rng.next_index(hi - lo + 1) } else { lo };
        // Partial Fisher-Yates over a scratch index list to pick `count`
        // distinct children without replacement.
        let mut indices: Vec<usize> = (0..children.len()).collect();
        let n = indices.len();
        for i in 0..count.min(n) {
            let j = i + rng.next_index(n - i);
            indices.swap(i, j);
            out.push(ContainerPick::unity(children[indices[i]]));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::curve::ParameterCurve;
    use crate::id::{SoundId, SwitchId};
    use alloc::vec::Vec;

    fn snd(n: u32) -> Playable {
        Playable::Sound(SoundId::new(n))
    }

    fn resolve_once(
        kind: &ContainerKind,
        state: &mut ContainerState,
        rng: &mut Rng,
        active_switch: Option<SwitchId>,
        blend: Sample,
    ) -> Vec<ContainerPick> {
        let mut out = Vec::new();
        kind.resolve_into(state, rng, active_switch, blend, &mut out);
        out
    }

    #[test]
    fn random_standard_never_repeats_within_window() {
        let kind = ContainerKind::Random {
            children: alloc::vec![
                WeightedChild::uniform(snd(1)),
                WeightedChild::uniform(snd(2)),
                WeightedChild::uniform(snd(3)),
                WeightedChild::uniform(snd(4)),
            ],
            mode: RandomMode::Standard,
            avoid_repeat: 2,
        };
        let mut state = ContainerState::new();
        let mut rng = Rng::new(0xA11CE);
        let mut history: Vec<Playable> = Vec::new();
        for _ in 0..200 {
            let picks = resolve_once(&kind, &mut state, &mut rng, None, 0.0);
            assert_eq!(picks.len(), 1);
            let p = picks[0].playable;
            // Must not equal either of the last two picks.
            let n = history.len();
            if n >= 1 {
                assert_ne!(p, history[n - 1]);
            }
            if n >= 2 {
                assert_ne!(p, history[n - 2]);
            }
            history.push(p);
        }
    }

    #[test]
    fn random_shuffle_exhausts_bag_before_repeating() {
        let kind = ContainerKind::Random {
            children: alloc::vec![
                WeightedChild::uniform(snd(1)),
                WeightedChild::uniform(snd(2)),
                WeightedChild::uniform(snd(3)),
            ],
            mode: RandomMode::Shuffle,
            avoid_repeat: 0,
        };
        let mut state = ContainerState::new();
        let mut rng = Rng::new(99);
        // Over each window of 3 consecutive draws every child appears once.
        for _ in 0..10 {
            let mut seen: Vec<Playable> = Vec::new();
            for _ in 0..3 {
                let picks = resolve_once(&kind, &mut state, &mut rng, None, 0.0);
                let p = picks[0].playable;
                assert!(!seen.contains(&p), "child repeated inside a shuffle bag");
                seen.push(p);
            }
            assert_eq!(seen.len(), 3);
        }
    }

    #[test]
    fn sequence_loop_wraps() {
        let kind = ContainerKind::Sequence {
            children: alloc::vec![snd(1), snd(2), snd(3)],
            mode: SequenceMode::Loop,
        };
        let mut state = ContainerState::new();
        let mut rng = Rng::new(1);
        let mut seq: Vec<u32> = Vec::new();
        for _ in 0..7 {
            let picks = resolve_once(&kind, &mut state, &mut rng, None, 0.0);
            if let Playable::Sound(id) = picks[0].playable {
                seq.push(id.get());
            }
        }
        assert_eq!(seq, alloc::vec![1, 2, 3, 1, 2, 3, 1]);
    }

    #[test]
    fn sequence_pingpong_bounces() {
        let kind = ContainerKind::Sequence {
            children: alloc::vec![snd(1), snd(2), snd(3)],
            mode: SequenceMode::PingPong,
        };
        let mut state = ContainerState::new();
        let mut rng = Rng::new(1);
        let mut seq: Vec<u32> = Vec::new();
        for _ in 0..7 {
            let picks = resolve_once(&kind, &mut state, &mut rng, None, 0.0);
            if let Playable::Sound(id) = picks[0].playable {
                seq.push(id.get());
            }
        }
        assert_eq!(seq, alloc::vec![1, 2, 3, 2, 1, 2, 3]);
    }

    #[test]
    fn sequence_once_stops_after_end() {
        let kind = ContainerKind::Sequence {
            children: alloc::vec![snd(1), snd(2)],
            mode: SequenceMode::Once,
        };
        let mut state = ContainerState::new();
        let mut rng = Rng::new(1);
        let mut count = 0;
        for _ in 0..5 {
            let picks = resolve_once(&kind, &mut state, &mut rng, None, 0.0);
            count += picks.len();
        }
        // Only two picks are ever produced; subsequent resolves are silent.
        assert_eq!(count, 2);
    }

    #[test]
    fn blend_includes_audible_layers_and_skips_silent() {
        let kind = ContainerKind::Blend {
            rtpc: RtpcId::new(1),
            layers: alloc::vec![
                BlendLayer::new(snd(1), ParameterCurve::constant(0.0)),
                BlendLayer::new(snd(2), ParameterCurve::constant(-130.0)),
                BlendLayer::new(snd(3), ParameterCurve::line(0.0, -120.0, 1.0, 0.0)),
            ],
        };
        let mut state = ContainerState::new();
        let mut rng = Rng::new(1);
        // At position 1.0 layer3 is at 0 dB (audible); layer2 is always silent.
        let picks = resolve_once(&kind, &mut state, &mut rng, None, 1.0);
        let ids: Vec<u32> = picks
            .iter()
            .filter_map(|p| match p.playable {
                Playable::Sound(s) => Some(s.get()),
                Playable::Container(_) => None,
            })
            .collect();
        assert!(ids.contains(&1));
        assert!(!ids.contains(&2));
        assert!(ids.contains(&3));
    }

    #[test]
    fn switch_selects_matching_branch_then_default() {
        let kind = ContainerKind::Switch {
            group: SwitchGroupId::new(1),
            branches: alloc::vec![
                SwitchBranch::new(SwitchId::new(10), snd(1)),
                SwitchBranch::new(SwitchId::new(11), snd(2)),
            ],
            default: Some(snd(99)),
        };
        let mut state = ContainerState::new();
        let mut rng = Rng::new(1);
        let picks = resolve_once(&kind, &mut state, &mut rng, Some(SwitchId::new(11)), 0.0);
        assert_eq!(picks.len(), 1);
        assert_eq!(picks[0].playable, snd(2));
        // Unknown active switch falls back to default.
        let picks = resolve_once(&kind, &mut state, &mut rng, Some(SwitchId::new(77)), 0.0);
        assert_eq!(picks[0].playable, snd(99));
        // No active switch also falls back to default.
        let picks = resolve_once(&kind, &mut state, &mut rng, None, 0.0);
        assert_eq!(picks[0].playable, snd(99));
    }

    #[test]
    fn scatter_picks_distinct_children_in_range() {
        let kind = ContainerKind::Scatter {
            children: alloc::vec![snd(1), snd(2), snd(3), snd(4), snd(5)],
            min_count: 2,
            max_count: 4,
        };
        let mut state = ContainerState::new();
        let mut rng = Rng::new(0xBEEF);
        for _ in 0..50 {
            let picks = resolve_once(&kind, &mut state, &mut rng, None, 0.0);
            assert!(picks.len() >= 2 && picks.len() <= 4);
            // All picks distinct.
            let mut seen: Vec<Playable> = Vec::new();
            for p in &picks {
                assert!(!seen.contains(&p.playable));
                seen.push(p.playable);
            }
        }
    }

    #[test]
    fn collect_children_enumerates_full_static_subtree() {
        let random = ContainerKind::Random {
            children: alloc::vec![WeightedChild::uniform(snd(1)), WeightedChild::uniform(snd(2))],
            mode: RandomMode::Standard,
            avoid_repeat: 0,
        };
        let mut out = Vec::new();
        random.collect_children(&mut out);
        assert_eq!(out, alloc::vec![snd(1), snd(2)]);

        let switch = ContainerKind::Switch {
            group: SwitchGroupId::new(1),
            branches: alloc::vec![SwitchBranch::new(SwitchId::new(1), snd(3))],
            default: Some(snd(4)),
        };
        let mut out = Vec::new();
        switch.collect_children(&mut out);
        assert_eq!(out, alloc::vec![snd(3), snd(4)]);
    }

    #[test]
    fn switch_group_and_blend_rtpc_accessors() {
        let switch = ContainerKind::Switch {
            group: SwitchGroupId::new(5),
            branches: Vec::new(),
            default: None,
        };
        assert_eq!(switch.switch_group(), Some(SwitchGroupId::new(5)));
        assert_eq!(switch.blend_rtpc(), None);
        let blend = ContainerKind::Blend { rtpc: RtpcId::new(7), layers: Vec::new() };
        assert_eq!(blend.blend_rtpc(), Some(RtpcId::new(7)));
        assert_eq!(blend.switch_group(), None);
    }

    #[test]
    fn same_seed_yields_identical_scatter_streams() {
        let kind = ContainerKind::Scatter {
            children: alloc::vec![snd(1), snd(2), snd(3), snd(4)],
            min_count: 1,
            max_count: 3,
        };
        let mut sa = ContainerState::new();
        let mut ra = Rng::new(1234);
        let mut sb = ContainerState::new();
        let mut rb = Rng::new(1234);
        for _ in 0..20 {
            let a = resolve_once(&kind, &mut sa, &mut ra, None, 0.0);
            let b = resolve_once(&kind, &mut sb, &mut rb, None, 0.0);
            assert_eq!(a, b);
        }
    }

    #[test]
    fn empty_children_resolve_to_nothing() {
        let random = ContainerKind::Random {
            children: Vec::new(),
            mode: RandomMode::Standard,
            avoid_repeat: 0,
        };
        let mut state = ContainerState::new();
        let mut rng = Rng::new(1);
        assert!(resolve_once(&random, &mut state, &mut rng, None, 0.0).is_empty());
    }

    #[test]
    fn container_state_reset_restores_sequence_cursor() {
        let kind = ContainerKind::Sequence {
            children: alloc::vec![snd(1), snd(2), snd(3)],
            mode: SequenceMode::Loop,
        };
        let mut state = ContainerState::new();
        let mut rng = Rng::new(1);
        let _ = resolve_once(&kind, &mut state, &mut rng, None, 0.0);
        let _ = resolve_once(&kind, &mut state, &mut rng, None, 0.0);
        state.reset();
        let picks = resolve_once(&kind, &mut state, &mut rng, None, 0.0);
        assert_eq!(picks[0].playable, snd(1));
    }
}
