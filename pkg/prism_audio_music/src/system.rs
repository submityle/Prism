//! The live **music system**: the deterministic runtime half of the model.
//!
//! Where [`crate::model::MusicModel`] is immutable authored data, a
//! [`MusicSystem`] holds the *mutable* playback state -- the currently playing
//! segment and its start sample, the active playlist cursor, the live intensity
//! and the set of layers it has switched on, each clip graph's current clip,
//! the quantization [`NamedClock`], and the seeded [`crate::rng::Rng`] -- and
//! turns high-level music requests into a flat, sample-accurate stream of
//! [`MusicAction`]s. One model can back many independent systems (split-screen,
//! server mixdown, offline bounce); each is a self-contained deterministic
//! state machine seeded by a single `u64`, so the same seed, model, and request
//! order always produce byte-identical action streams.
//!
//! Requests are **control-rate** (raised when gameplay changes), so this layer
//! may allocate; it does no per-sample DSP. The lower runtime consumes the
//! resolved stream and manages voices.
//!
//! # Provenance
//!
//! This module contains **no Unreal Engine, Unity, Godot, Wwise, FMOD, Steam
//! Audio, or Google Resonance Audio source or derived code**. The planner is
//! reconstructed from first principles over this crate's own plain data, the
//! deterministic [`crate::rng::Rng`], and `prism_audio_core`'s tempo clock. No
//! AI/ML.
//!
//! # Relationship
//!
//! [`MusicSystem`] reads a [`crate::model::MusicModel`], resolves
//! [`crate::transition::TransitionType`] grids against a
//! [`NamedClock`], walks [`crate::playlist::PlaylistCursor`]s and
//! [`crate::clip_graph::ClipGraph`]s, applies [`crate::layer::Layer`]
//! hysteresis, and emits [`MusicAction`]s for the lower runtime.

use alloc::collections::{BTreeMap, BTreeSet};
use alloc::vec::Vec;

use prism_audio_core::math::Sample;
use prism_audio_core::scheduler::{Grid, NamedClock};
use prism_audio_core::time::TimeSignature;

use crate::action::MusicAction;
use crate::clip_graph::{TransitionContext, MAX_GRAPH_DEPTH};
use crate::id::{ClipId, GraphId, PlaylistId, SegmentId};
use crate::model::MusicModel;
use crate::playlist::PlaylistCursor;
use crate::rng::Rng;
use crate::segment::Segment;
use crate::transition::{FadeCurve, Transition, TransitionType};

/// Hard ceiling on how many segments one [`MusicSystem::advance_to`] call may
/// chain through.
///
/// A playlist of zero-length-body segments could otherwise advance forever at a
/// single transport position; this bound guarantees termination. It is
/// generous: real transport blocks cross only a handful of segment boundaries.
pub const MAX_ADVANCE_STEPS: usize = 4096;

/// Default tempo (beats per minute) of a freshly built system before any
/// segment anchors the clock.
const DEFAULT_TEMPO_BPM: f32 = 120.0;

/// The segment currently playing and the absolute sample its playback began.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
struct Playing {
    /// The playing segment's id.
    segment: SegmentId,
    /// Absolute sample position playback (sample zero, including pre-roll)
    /// began.
    start_sample: u64,
}

/// Live, mutable runtime driving one deterministic [`MusicAction`] stream from
/// an authored [`MusicModel`].
#[derive(Debug, Clone)]
pub struct MusicSystem {
    /// Sample rate in Hz (shared with the clock).
    sample_rate: u32,
    /// Seed the system (and its RNG) was created or reset with.
    seed: u64,
    /// Deterministic generator for random/shuffle playlists.
    rng: Rng,
    /// Tempo grid anchored at the current segment's entry cue.
    clock: NamedClock,
    /// The segment currently playing, if any.
    playing: Option<Playing>,
    /// The active playlist walk, if any.
    playlist: Option<PlaylistCursor>,
    /// The live intensity value driving vertical layering.
    intensity: Sample,
    /// Layers currently switched on (by the hysteresis in
    /// [`MusicSystem::set_intensity`]).
    active_layers: BTreeSet<crate::id::LayerId>,
    /// Each clip graph's current clip.
    graph_cursor: BTreeMap<GraphId, ClipId>,
}

impl MusicSystem {
    /// Builds a system at `sample_rate` Hz seeded with `seed`.
    ///
    /// The clock starts at the default tempo in common time, anchored at sample
    /// zero; the first played segment re-anchors it. `sample_rate` is forced to
    /// at least one so the clock can never be built with a zero rate.
    #[must_use]
    pub fn new(sample_rate: u32, seed: u64) -> Self {
        let sample_rate = sample_rate.max(1);
        Self {
            sample_rate,
            seed,
            rng: Rng::new(seed),
            clock: NamedClock::new(sample_rate, 0, DEFAULT_TEMPO_BPM, TimeSignature::default()),
            playing: None,
            playlist: None,
            intensity: 0.0,
            active_layers: BTreeSet::new(),
            graph_cursor: BTreeMap::new(),
        }
    }

    /// Clears all playback state and re-seeds the RNG with `seed`.
    pub fn reset(&mut self, seed: u64) {
        self.seed = seed;
        self.rng = Rng::new(seed);
        self.clock = NamedClock::new(
            self.sample_rate,
            0,
            DEFAULT_TEMPO_BPM,
            TimeSignature::default(),
        );
        self.playing = None;
        self.playlist = None;
        self.intensity = 0.0;
        self.active_layers.clear();
        self.graph_cursor.clear();
    }

    /// Returns the sample rate in Hz.
    #[inline]
    #[must_use]
    pub fn sample_rate(&self) -> u32 {
        self.sample_rate
    }

    /// Returns the seed the system was last created or reset with.
    #[inline]
    #[must_use]
    pub fn seed(&self) -> u64 {
        self.seed
    }

    /// Returns the live intensity value.
    #[inline]
    #[must_use]
    pub fn intensity(&self) -> Sample {
        self.intensity
    }

    /// Returns the id of the currently playing segment, if any.
    #[inline]
    #[must_use]
    pub fn playing_segment(&self) -> Option<SegmentId> {
        self.playing.map(|p| p.segment)
    }

    /// Returns the absolute sample at which the current segment's playback
    /// began, if any.
    #[inline]
    #[must_use]
    pub fn playing_since(&self) -> Option<u64> {
        self.playing.map(|p| p.start_sample)
    }

    /// Returns the number of layers currently switched on.
    #[inline]
    #[must_use]
    pub fn active_layer_count(&self) -> usize {
        self.active_layers.len()
    }

    /// Returns whether `layer` is currently switched on.
    #[must_use]
    pub fn is_layer_active(&self, layer: crate::id::LayerId) -> bool {
        self.active_layers.contains(&layer)
    }

    /// Returns the current clip of `graph`, if the graph has been started.
    #[must_use]
    pub fn graph_clip(&self, graph: GraphId) -> Option<ClipId> {
        self.graph_cursor.get(&graph).copied()
    }

    /// Returns the tempo in beats per minute of the active clock.
    #[inline]
    #[must_use]
    pub fn tempo_bpm(&self) -> f32 {
        self.clock.tempo_bpm()
    }

    /// Anchors the clock at a segment's entry cue and records it as playing.
    fn set_playing(&mut self, seg: &Segment, start_sample: u64) {
        self.playing = Some(Playing {
            segment: seg.id,
            start_sample,
        });
        self.clock.set_origin(start_sample + seg.pre_roll);
        self.clock.set_tempo_bpm(seg.tempo_bpm);
        self.clock.set_signature(seg.signature);
    }

    /// Emits a `PlaySegment` for `seg` whose entry cue lands at `entry` and
    /// records it as the playing segment.
    ///
    /// Playback (sample zero) begins `pre_roll` samples before `entry`,
    /// saturating at zero so a lead-in longer than `entry` simply starts at the
    /// transport origin.
    fn begin_segment(
        &mut self,
        seg: &Segment,
        entry: u64,
        fade_in: u64,
        curve: FadeCurve,
        out: &mut Vec<MusicAction>,
    ) {
        let start = entry.saturating_sub(seg.pre_roll);
        out.push(MusicAction::PlaySegment {
            segment: seg.id,
            sound: seg.sound,
            at_sample: start,
            gain_db: 0.0,
            fade_in,
            curve,
        });
        self.set_playing(seg, start);
    }

    /// Starts a single segment at `at_sample`, replacing any current segment
    /// and cancelling any playlist walk.
    ///
    /// Returns `false` (emitting nothing) when `segment` is unknown. The entry
    /// cue is placed at `at_sample`; see [`MusicSystem::begin_segment`] for how
    /// the pre-roll is handled.
    pub fn play_segment(
        &mut self,
        model: &MusicModel,
        segment: SegmentId,
        at_sample: u64,
        fade_in: u64,
        fade_out: u64,
        out: &mut Vec<MusicAction>,
    ) -> bool {
        let Some(seg) = model.segment(segment) else {
            return false;
        };
        self.stop_current_with_model(model, at_sample, fade_out, FadeCurve::EqualPower, out);
        self.playlist = None;
        self.begin_segment(seg, at_sample, fade_in, FadeCurve::EqualPower, out);
        true
    }

    /// Emits a `StopSegment` for the current segment, resolving its leaf sound
    /// through `model`.
    fn stop_current_with_model(
        &self,
        model: &MusicModel,
        at_sample: u64,
        fade_out: u64,
        curve: FadeCurve,
        out: &mut Vec<MusicAction>,
    ) {
        if let Some(playing) = self.playing {
            let sound = model
                .segment(playing.segment)
                .map_or(crate::id::SoundId::new(0), |s| s.sound);
            out.push(MusicAction::StopSegment {
                segment: playing.segment,
                sound,
                at_sample,
                fade_out,
                curve,
            });
        }
    }

    /// Starts a playlist at `at_sample`, playing its first segment.
    ///
    /// Returns `false` (emitting nothing) when the playlist is unknown, empty,
    /// or its first segment is unknown.
    pub fn start_playlist(
        &mut self,
        model: &MusicModel,
        playlist: PlaylistId,
        at_sample: u64,
        fade_in: u64,
        fade_out: u64,
        out: &mut Vec<MusicAction>,
    ) -> bool {
        let Some(pl) = model.playlist(playlist) else {
            return false;
        };
        if pl.is_empty() {
            return false;
        }
        let mut cursor = PlaylistCursor::new(playlist);
        let Some(first) = cursor.next(pl, &mut self.rng) else {
            return false;
        };
        let Some(seg) = model.segment(first) else {
            return false;
        };
        self.stop_current_with_model(model, at_sample, fade_out, FadeCurve::EqualPower, out);
        self.playlist = Some(cursor);
        self.begin_segment(seg, at_sample, fade_in, FadeCurve::EqualPower, out);
        true
    }

    /// Advances an active playlist up to `now_sample`, seamlessly chaining each
    /// segment at its exit cue.
    ///
    /// Call this once per transport block with the block's end sample. Each time
    /// the current segment's exit cue has been reached the next playlist segment
    /// is scheduled so its entry cue coincides with that exit cue (a gapless
    /// hand-off). When the playlist is exhausted the final segment is stopped at
    /// its exit cue. No-ops when nothing is playing or no playlist is active.
    pub fn advance_to(&mut self, model: &MusicModel, now_sample: u64, out: &mut Vec<MusicAction>) {
        for _ in 0..MAX_ADVANCE_STEPS {
            let Some(playing) = self.playing else {
                break;
            };
            if self.playlist.is_none() {
                break;
            }
            let Some(seg) = model.segment(playing.segment) else {
                break;
            };
            let exit_point = playing.start_sample + seg.exit_cue();
            if now_sample < exit_point {
                break;
            }

            let pl_id = self.playlist.as_ref().map(PlaylistCursor::playlist);
            let Some(pl_id) = pl_id else {
                break;
            };
            let Some(pl) = model.playlist(pl_id) else {
                break;
            };
            let next = self
                .playlist
                .as_mut()
                .and_then(|c| c.next(pl, &mut self.rng));

            match next {
                Some(next_id) => {
                    let Some(next_seg) = model.segment(next_id) else {
                        break;
                    };
                    self.begin_segment(next_seg, exit_point, 0, FadeCurve::EqualPower, out);
                }
                None => {
                    out.push(MusicAction::StopSegment {
                        segment: playing.segment,
                        sound: seg.sound,
                        at_sample: exit_point,
                        fade_out: 0,
                        curve: FadeCurve::EqualPower,
                    });
                    self.playing = None;
                    self.playlist = None;
                    break;
                }
            }
        }
    }

    /// Requests a transition to `target` using `transition`, returning the
    /// absolute sample the switch resolves to.
    ///
    /// Returns `None` (emitting nothing) when `target` is unknown. The current
    /// segment is stopped at the resolved point with the transition's fade-out;
    /// if the transition carries a bridge segment it plays first and `target`
    /// follows at the bridge's exit cue, otherwise `target` begins directly.
    /// Any active playlist walk is cancelled.
    pub fn request_transition(
        &mut self,
        model: &MusicModel,
        target: SegmentId,
        transition: Transition,
        now_sample: u64,
        out: &mut Vec<MusicAction>,
    ) -> Option<u64> {
        let target_seg = model.segment(target)?;
        let point = self.resolve_point(model, transition.transition_type, now_sample);
        let fade = transition.fade;
        self.playlist = None;
        self.stop_current_with_model(model, point, fade.fade_out, fade.curve, out);

        if let Some(bridge_id) = transition.bridge
            && let Some(bridge) = model.segment(bridge_id)
        {
            // Play the bridge at the resolved point, then hand off to the
            // target at the bridge's exit cue.
            out.push(MusicAction::PlaySegment {
                segment: bridge.id,
                sound: bridge.sound,
                at_sample: point.saturating_sub(bridge.pre_roll),
                gain_db: 0.0,
                fade_in: fade.fade_in,
                curve: fade.curve,
            });
            let handoff = point + bridge.body;
            self.begin_segment(target_seg, handoff, fade.fade_in, fade.curve, out);
            return Some(point);
        }

        self.begin_segment(target_seg, point, fade.fade_in, fade.curve, out);
        Some(point)
    }

    /// Fires a one-shot stinger overlay aligned to its quantization grid,
    /// returning the absolute sample it resolves to.
    ///
    /// Returns `None` (emitting nothing) when `stinger` is unknown. The current
    /// segment, playlist cursor, and clock are left untouched.
    pub fn trigger_stinger(
        &mut self,
        model: &MusicModel,
        stinger: crate::id::StingerId,
        now_sample: u64,
        out: &mut Vec<MusicAction>,
    ) -> Option<u64> {
        let st = model.stinger(stinger)?;
        let point = self.resolve_point(model, st.quantize, now_sample);
        out.push(MusicAction::PlayStinger {
            stinger: st.id,
            sound: st.sound,
            at_sample: point,
            gain_db: st.gain_db,
        });
        Some(point)
    }

    /// Sets the live intensity and switches vertical layers on or off at
    /// `at_sample`, honouring each layer's hysteresis band.
    ///
    /// Every registered [`crate::layer::LayerSet`] is evaluated in ascending id
    /// order; within a set the layers are evaluated in authored order, so the
    /// emitted `StartLayer`/`StopLayer` actions are fully deterministic. Layer
    /// fades are left to the runtime (zero-length here).
    pub fn set_intensity(
        &mut self,
        model: &MusicModel,
        value: Sample,
        at_sample: u64,
        out: &mut Vec<MusicAction>,
    ) {
        self.intensity = value;
        for set in model.layer_sets() {
            for layer in &set.layers {
                let was_active = self.active_layers.contains(&layer.id);
                let now_active = layer.is_active(value, was_active);
                if now_active && !was_active {
                    self.active_layers.insert(layer.id);
                    out.push(MusicAction::StartLayer {
                        layer: layer.id,
                        sound: layer.sound,
                        at_sample,
                        gain_db: layer.gain_db,
                        fade_in: 0,
                        curve: FadeCurve::EqualPower,
                    });
                } else if was_active && !now_active {
                    self.active_layers.remove(&layer.id);
                    out.push(MusicAction::StopLayer {
                        layer: layer.id,
                        sound: layer.sound,
                        at_sample,
                        fade_out: 0,
                        curve: FadeCurve::EqualPower,
                    });
                }
            }
        }
    }

    /// Starts a clip graph at its start clip, playing that clip's segment at
    /// `at_sample`.
    ///
    /// Returns `false` (emitting nothing) when the graph, its start clip, or
    /// that clip's segment is unknown. Cancels any active playlist walk.
    pub fn clip_graph_start(
        &mut self,
        model: &MusicModel,
        graph: GraphId,
        at_sample: u64,
        fade_in: u64,
        out: &mut Vec<MusicAction>,
    ) -> bool {
        let Some(g) = model.graph(graph) else {
            return false;
        };
        let Some(clip) = g.clip(g.start) else {
            return false;
        };
        let Some(seg) = model.segment(clip.segment) else {
            return false;
        };
        self.playlist = None;
        self.graph_cursor.insert(graph, clip.id);
        self.begin_segment(seg, at_sample, fade_in, FadeCurve::EqualPower, out);
        true
    }

    /// Posts a gameplay event to a running clip graph, taking matching edges and
    /// returning the clip the graph settles on.
    ///
    /// Edges leaving the current clip are tried in authored order; the first
    /// whose [`crate::clip_graph::TriggerCondition`] matches `ctx` is taken, its
    /// transition applied, and traversal continues only through further
    /// `Immediate` edges (so a cycle of immediate edges is bounded by
    /// [`MAX_GRAPH_DEPTH`]). Returns `None` when the graph is unknown or has not
    /// been started.
    pub fn clip_graph_event(
        &mut self,
        model: &MusicModel,
        graph: GraphId,
        ctx: TransitionContext,
        now_sample: u64,
        out: &mut Vec<MusicAction>,
    ) -> Option<ClipId> {
        let g = model.graph(graph)?;
        let mut current = self.graph_cursor.get(&graph).copied()?;

        for _ in 0..MAX_GRAPH_DEPTH {
            let Some(edge) = g.select_edge(current, &ctx) else {
                break;
            };
            let Some(target_clip) = g.clip(edge.to) else {
                break;
            };
            let Some(seg) = model.segment(target_clip.segment) else {
                break;
            };
            let tt = edge.transition.transition_type;
            let point = self.resolve_point(model, tt, now_sample);
            let fade = edge.transition.fade;
            self.stop_current_with_model(model, point, fade.fade_out, fade.curve, out);
            self.begin_segment(seg, point, fade.fade_in, fade.curve, out);
            current = target_clip.id;
            if tt != TransitionType::Immediate {
                break;
            }
        }

        self.graph_cursor.insert(graph, current);
        Some(current)
    }

    /// Resolves a [`TransitionType`] into an absolute sample at or after
    /// `now_sample` against the current clock and playing segment.
    fn resolve_point(&self, model: &MusicModel, tt: TransitionType, now_sample: u64) -> u64 {
        match tt {
            TransitionType::Immediate => now_sample,
            TransitionType::NextBeat => self.clock.quantize(now_sample, Grid::Beat),
            TransitionType::NextBar => self.clock.quantize(now_sample, Grid::Bar),
            TransitionType::NextGrid(n) => self.clock.quantize(now_sample, Grid::Nth(n)),
            TransitionType::SegmentEnd => self.segment_end_point(model, now_sample),
            TransitionType::Marker(id) => self.marker_point(model, id, now_sample),
        }
    }

    /// Absolute sample of the current segment's next exit cue at or after
    /// `now_sample` (looping forward by the body length), or `now_sample` when
    /// nothing is playing.
    fn segment_end_point(&self, model: &MusicModel, now_sample: u64) -> u64 {
        let Some(playing) = self.playing else {
            return now_sample;
        };
        let Some(seg) = model.segment(playing.segment) else {
            return now_sample;
        };
        let mut exit = playing.start_sample + seg.exit_cue();
        if exit >= now_sample {
            return exit;
        }
        if seg.body == 0 {
            return now_sample;
        }
        for _ in 0..MAX_ADVANCE_STEPS {
            exit += seg.body;
            if exit >= now_sample {
                break;
            }
        }
        exit.max(now_sample)
    }

    /// Absolute sample of the current segment's next occurrence of marker `id`
    /// at or after `now_sample`, falling back to the exit cue when the marker is
    /// absent and to `now_sample` when nothing is playing.
    fn marker_point(&self, model: &MusicModel, id: crate::id::MarkerId, now_sample: u64) -> u64 {
        let Some(playing) = self.playing else {
            return now_sample;
        };
        let Some(seg) = model.segment(playing.segment) else {
            return now_sample;
        };
        let Some(offset) = seg.marker_offset(id) else {
            return self.segment_end_point(model, now_sample);
        };
        let mut at = playing.start_sample + offset;
        if at >= now_sample {
            return at;
        }
        if seg.body == 0 {
            return now_sample;
        }
        for _ in 0..MAX_ADVANCE_STEPS {
            at += seg.body;
            if at >= now_sample {
                break;
            }
        }
        at.max(now_sample)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::clip_graph::{Clip, ClipEdge, ClipGraph, TriggerCondition};
    use crate::id::{BranchId, LayerId, LayerSetId, SoundId, StingerId};
    use crate::layer::{Layer, LayerSet};
    use crate::playlist::{Playlist, PlaylistItem, PlaylistMode};
    use crate::segment::Segment;
    use crate::stinger::Stinger;
    use crate::transition::Fade;

    const SR: u32 = 48_000;

    // At 120 BPM @ 48 kHz: 24000 samples/beat, 96000 samples/bar (4/4).
    fn seg(id: u32, body: u64) -> Segment {
        Segment::new(
            SegmentId::new(id),
            SoundId::new(id + 100),
            120.0,
            TimeSignature::default(),
            0,
            body,
            0,
        )
    }

    fn model_with_segments() -> MusicModel {
        let mut m = MusicModel::new();
        m.add_segment(seg(1, 96_000));
        m.add_segment(seg(2, 96_000));
        m
    }

    #[test]
    fn play_segment_emits_play_and_sets_state() {
        let m = model_with_segments();
        let mut sys = MusicSystem::new(SR, 1);
        let mut out = Vec::new();
        assert!(sys.play_segment(&m, SegmentId::new(1), 0, 0, 0, &mut out));
        assert_eq!(out.len(), 1);
        assert_eq!(sys.playing_segment(), Some(SegmentId::new(1)));
        assert_eq!(out[0].at_sample(), 0);
    }

    #[test]
    fn unknown_segment_emits_nothing() {
        let m = model_with_segments();
        let mut sys = MusicSystem::new(SR, 1);
        let mut out = Vec::new();
        assert!(!sys.play_segment(&m, SegmentId::new(99), 0, 0, 0, &mut out));
        assert!(out.is_empty());
    }

    #[test]
    fn next_bar_quantizes_to_bar_boundary() {
        let m = model_with_segments();
        let mut sys = MusicSystem::new(SR, 1);
        let mut out = Vec::new();
        sys.play_segment(&m, SegmentId::new(1), 0, 0, 0, &mut out);
        out.clear();
        // Request at sample 1 -> next bar boundary is 96000.
        let point = sys
            .request_transition(
                &m,
                SegmentId::new(2),
                Transition::new(TransitionType::NextBar, Fade::cut()),
                1,
                &mut out,
            )
            .unwrap();
        assert_eq!(point, 96_000);
        // Stop current (seg 1) then play target (seg 2), both at the boundary.
        assert_eq!(out.len(), 2);
        assert_eq!(out[0].at_sample(), 96_000);
        assert_eq!(out[1].at_sample(), 96_000);
        assert_eq!(sys.playing_segment(), Some(SegmentId::new(2)));
    }

    #[test]
    fn next_beat_quantizes_to_beat_boundary() {
        let m = model_with_segments();
        let mut sys = MusicSystem::new(SR, 1);
        let mut out = Vec::new();
        sys.play_segment(&m, SegmentId::new(1), 0, 0, 0, &mut out);
        out.clear();
        let point = sys
            .request_transition(
                &m,
                SegmentId::new(2),
                Transition::new(TransitionType::NextBeat, Fade::cut()),
                100,
                &mut out,
            )
            .unwrap();
        assert_eq!(point, 24_000);
    }

    #[test]
    fn segment_end_resolves_to_exit_cue() {
        let m = model_with_segments();
        let mut sys = MusicSystem::new(SR, 1);
        let mut out = Vec::new();
        sys.play_segment(&m, SegmentId::new(1), 0, 0, 0, &mut out);
        out.clear();
        let point = sys
            .request_transition(
                &m,
                SegmentId::new(2),
                Transition::new(TransitionType::SegmentEnd, Fade::cut()),
                1_000,
                &mut out,
            )
            .unwrap();
        // Seg 1 body is one bar: exit cue at 96000.
        assert_eq!(point, 96_000);
    }

    #[test]
    fn segment_end_loops_forward_past_now() {
        let m = model_with_segments();
        let mut sys = MusicSystem::new(SR, 1);
        let mut out = Vec::new();
        sys.play_segment(&m, SegmentId::new(1), 0, 0, 0, &mut out);
        out.clear();
        // now beyond the first exit cue (96000): next is 192000.
        let point = sys
            .request_transition(
                &m,
                SegmentId::new(2),
                Transition::new(TransitionType::SegmentEnd, Fade::cut()),
                100_000,
                &mut out,
            )
            .unwrap();
        assert_eq!(point, 192_000);
    }

    #[test]
    fn marker_transition_aligns_to_marker() {
        let mut m = MusicModel::new();
        m.add_segment(
            seg(1, 96_000).with_marker(crate::segment::Marker::new(crate::id::MarkerId::new(1), 48_000)),
        );
        m.add_segment(seg(2, 96_000));
        let mut sys = MusicSystem::new(SR, 1);
        let mut out = Vec::new();
        sys.play_segment(&m, SegmentId::new(1), 0, 0, 0, &mut out);
        out.clear();
        let point = sys
            .request_transition(
                &m,
                SegmentId::new(2),
                Transition::new(TransitionType::Marker(crate::id::MarkerId::new(1)), Fade::cut()),
                1_000,
                &mut out,
            )
            .unwrap();
        assert_eq!(point, 48_000);
    }

    #[test]
    fn immediate_transition_is_now() {
        let m = model_with_segments();
        let mut sys = MusicSystem::new(SR, 1);
        let mut out = Vec::new();
        sys.play_segment(&m, SegmentId::new(1), 0, 0, 0, &mut out);
        out.clear();
        let point = sys
            .request_transition(&m, SegmentId::new(2), Transition::immediate(), 12_345, &mut out)
            .unwrap();
        assert_eq!(point, 12_345);
    }

    #[test]
    fn bridged_transition_plays_bridge_then_target() {
        let mut m = model_with_segments();
        m.add_segment(seg(3, 48_000)); // bridge
        let mut sys = MusicSystem::new(SR, 1);
        let mut out = Vec::new();
        sys.play_segment(&m, SegmentId::new(1), 0, 0, 0, &mut out);
        out.clear();
        let point = sys
            .request_transition(
                &m,
                SegmentId::new(2),
                Transition::bridged(TransitionType::NextBar, Fade::cut(), SegmentId::new(3)),
                1,
                &mut out,
            )
            .unwrap();
        assert_eq!(point, 96_000);
        // Stop seg1 @96000, play bridge @96000, play target @ 96000+48000.
        assert_eq!(out.len(), 3);
        assert_eq!(out[1].at_sample(), 96_000);
        assert_eq!(out[2].at_sample(), 144_000);
        assert_eq!(sys.playing_segment(), Some(SegmentId::new(2)));
    }

    #[test]
    fn stinger_aligns_and_leaves_playing_untouched() {
        let mut m = model_with_segments();
        m.add_stinger(Stinger::new(StingerId::new(1), SoundId::new(500), TransitionType::NextBar));
        let mut sys = MusicSystem::new(SR, 1);
        let mut out = Vec::new();
        sys.play_segment(&m, SegmentId::new(1), 0, 0, 0, &mut out);
        out.clear();
        let point = sys.trigger_stinger(&m, StingerId::new(1), 5_000, &mut out).unwrap();
        assert_eq!(point, 96_000);
        assert_eq!(out.len(), 1);
        assert_eq!(sys.playing_segment(), Some(SegmentId::new(1)));
    }

    #[test]
    fn playlist_chains_segments_at_exit_cues() {
        let mut m = model_with_segments();
        m.add_playlist(
            Playlist::new(PlaylistId::new(1), PlaylistMode::Sequence)
                .with_item(PlaylistItem::once(SegmentId::new(1)))
                .with_item(PlaylistItem::once(SegmentId::new(2))),
        );
        let mut sys = MusicSystem::new(SR, 1);
        let mut out = Vec::new();
        assert!(sys.start_playlist(&m, PlaylistId::new(1), 0, 0, 0, &mut out));
        assert_eq!(out.len(), 1); // play seg1
        out.clear();
        // Advance just before the first exit cue: no chaining.
        sys.advance_to(&m, 95_999, &mut out);
        assert!(out.is_empty());
        // Advance across the exit cue (96000): seg2 should begin there.
        sys.advance_to(&m, 96_000, &mut out);
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].at_sample(), 96_000);
        assert_eq!(sys.playing_segment(), Some(SegmentId::new(2)));
        // Advance past seg2's exit cue (192000): playlist exhausts -> stop.
        out.clear();
        sys.advance_to(&m, 300_000, &mut out);
        assert_eq!(out.len(), 1);
        assert!(matches!(out[0], MusicAction::StopSegment { at_sample: 192_000, .. }));
        assert_eq!(sys.playing_segment(), None);
    }

    #[test]
    fn intensity_adds_and_removes_layers_with_hysteresis() {
        let mut m = MusicModel::new();
        m.add_layer_set(
            LayerSet::new(LayerSetId::new(1))
                .with_layer(Layer::new(LayerId::new(1), SoundId::new(10), 0.3, 0.2))
                .with_layer(Layer::new(LayerId::new(2), SoundId::new(11), 0.7, 0.6)),
        );
        let mut sys = MusicSystem::new(SR, 1);
        let mut out = Vec::new();

        sys.set_intensity(&m, 0.5, 0, &mut out);
        // Only layer 1 (enter 0.3) switches on.
        assert_eq!(out.len(), 1);
        assert!(sys.is_layer_active(LayerId::new(1)));
        assert!(!sys.is_layer_active(LayerId::new(2)));

        out.clear();
        sys.set_intensity(&m, 0.8, 100, &mut out);
        // Layer 2 (enter 0.7) switches on.
        assert_eq!(out.len(), 1);
        assert!(sys.is_layer_active(LayerId::new(2)));

        out.clear();
        // Drop into the hysteresis band of layer 2 (0.6..0.7): stays on.
        sys.set_intensity(&m, 0.65, 200, &mut out);
        assert!(out.is_empty());
        assert!(sys.is_layer_active(LayerId::new(2)));

        out.clear();
        // Drop below exit of both: both switch off.
        sys.set_intensity(&m, 0.1, 300, &mut out);
        assert_eq!(out.len(), 2);
        assert_eq!(sys.active_layer_count(), 0);
    }

    #[test]
    fn clip_graph_selects_edge_by_condition() {
        let mut m = model_with_segments();
        m.add_segment(seg(3, 96_000));
        m.add_graph(
            ClipGraph::new(GraphId::new(1), ClipId::new(1))
                .with_clip(Clip::new(ClipId::new(1), SegmentId::new(1)))
                .with_clip(Clip::new(ClipId::new(2), SegmentId::new(2)))
                .with_clip(Clip::new(ClipId::new(3), SegmentId::new(3)))
                .with_edge(ClipEdge::new(
                    ClipId::new(1),
                    ClipId::new(2),
                    TriggerCondition::OnBranch(BranchId::new(7)),
                    Transition::new(TransitionType::NextBar, Fade::cut()),
                ))
                .with_edge(ClipEdge::new(
                    ClipId::new(1),
                    ClipId::new(3),
                    TriggerCondition::Always,
                    Transition::immediate(),
                )),
        );
        let mut sys = MusicSystem::new(SR, 1);
        let mut out = Vec::new();
        assert!(sys.clip_graph_start(&m, GraphId::new(1), 0, 0, &mut out));
        assert_eq!(sys.graph_clip(GraphId::new(1)), Some(ClipId::new(1)));
        out.clear();

        // Branch 7 -> edge to clip 2.
        let ctx = TransitionContext::new(Some(BranchId::new(7)), 0.0);
        let landed = sys.clip_graph_event(&m, GraphId::new(1), ctx, 1_000, &mut out).unwrap();
        assert_eq!(landed, ClipId::new(2));
        assert_eq!(sys.playing_segment(), Some(SegmentId::new(2)));
    }

    #[test]
    fn clip_graph_immediate_cycle_is_bounded() {
        // Two clips with mutual Immediate+Always edges form a cycle; the depth
        // bound must terminate traversal.
        let mut m = model_with_segments();
        m.add_graph(
            ClipGraph::new(GraphId::new(1), ClipId::new(1))
                .with_clip(Clip::new(ClipId::new(1), SegmentId::new(1)))
                .with_clip(Clip::new(ClipId::new(2), SegmentId::new(2)))
                .with_edge(ClipEdge::new(
                    ClipId::new(1),
                    ClipId::new(2),
                    TriggerCondition::Always,
                    Transition::immediate(),
                ))
                .with_edge(ClipEdge::new(
                    ClipId::new(2),
                    ClipId::new(1),
                    TriggerCondition::Always,
                    Transition::immediate(),
                )),
        );
        let mut sys = MusicSystem::new(SR, 1);
        let mut out = Vec::new();
        sys.clip_graph_start(&m, GraphId::new(1), 0, 0, &mut out);
        out.clear();
        // Should return without hanging; action count is bounded by depth.
        let ctx = TransitionContext::new(None, 0.0);
        let landed = sys.clip_graph_event(&m, GraphId::new(1), ctx, 0, &mut out);
        assert!(landed.is_some());
        assert!(out.len() <= MAX_GRAPH_DEPTH * 2);
    }

    #[test]
    fn same_seed_and_requests_are_deterministic() {
        let mut m = model_with_segments();
        m.add_segment(seg(3, 96_000));
        m.add_playlist(
            Playlist::new(PlaylistId::new(1), PlaylistMode::Shuffle)
                .with_item(PlaylistItem::once(SegmentId::new(1)))
                .with_item(PlaylistItem::once(SegmentId::new(2)))
                .with_item(PlaylistItem::once(SegmentId::new(3))),
        );

        let run = |seed: u64| {
            let mut sys = MusicSystem::new(SR, seed);
            let mut out = Vec::new();
            sys.start_playlist(&m, PlaylistId::new(1), 0, 0, 0, &mut out);
            for block in 1..=12 {
                sys.advance_to(&m, block * 96_000, &mut out);
            }
            out
        };

        assert_eq!(run(0xBEEF), run(0xBEEF));
        // A different seed changes the shuffle order (very likely differs).
        assert_ne!(run(0xBEEF), run(0x1234_5678));
    }

    #[test]
    fn reset_restores_initial_state() {
        let m = model_with_segments();
        let mut sys = MusicSystem::new(SR, 1);
        let mut out = Vec::new();
        sys.play_segment(&m, SegmentId::new(1), 0, 0, 0, &mut out);
        sys.reset(2);
        assert_eq!(sys.playing_segment(), None);
        assert_eq!(sys.seed(), 2);
        assert_eq!(sys.active_layer_count(), 0);
    }
}
