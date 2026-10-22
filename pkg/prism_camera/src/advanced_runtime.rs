//! Fixed-tick orchestration for the optional advanced camera contracts.
//!
//! This module deliberately owns scheduling state, not ECS storage or renderer
//! resources.  It can therefore be mounted by a caller that already owns the
//! camera graph while still giving every stage a bounded result and an explicit
//! degradation reason.

#[cfg(any(feature = "lookahead_stream", feature = "occlusion_fade"))]
use alloc::vec::Vec;
#[cfg(not(any(feature = "cutscene", feature = "occlusion_fade")))]
use core::marker::PhantomData;

#[cfg(feature = "shutter")]
use crate::Shutter;
#[cfg(feature = "view_lod")]
use crate::ViewLod;
#[cfg(feature = "occlusion_fade")]
use crate::{CameraOcclusionIntent, OcclusionFadeState, OcclusionSample};
#[cfg(feature = "cutscene")]
use crate::{CameraPose, CameraTrack, CameraTrackSample};
#[cfg(feature = "autofocus")]
use crate::{FocusSource, FocusTracker};
#[cfg(feature = "edge_stabilize")]
use crate::{HistoryRejectReason, HistoryStabilizer};
#[cfg(feature = "lookahead_stream")]
use crate::{LookaheadPredictor, StreamingContribution};
#[cfg(feature = "lookahead_stream")]
use prism_math::DVec3;

/// Hard upper bounds used by the runtime-owned buffers.
pub const MAX_LOOKAHEAD_SAMPLES: usize = 16;
pub const MAX_OCCLUSION_SAMPLES: usize = 64;
pub const MAX_OCCLUSION_INTENTS: usize = 64;
pub const MAX_HISTORY_FRAMES: u32 = 4096;

/// Which capacity in [`AdvancedRuntimeLimits`] rejected a configuration.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CapacityField {
    LookaheadSamples,
    OcclusionSamples,
    OcclusionIntents,
    HistoryFrames,
}

/// Configuration errors are returned before any runtime state is created.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum AdvancedRuntimeConfigError {
    InvalidFixedDelta,
    InvalidTickTolerance,
    InvalidFrameRate,
    InvalidNearClip,
    CapacityExceeded {
        field: CapacityField,
        requested: u32,
        maximum: u32,
    },
}

/// Explicit allocation and retained-history limits for one runtime instance.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct AdvancedRuntimeLimits {
    pub max_lookahead_samples: usize,
    pub max_occlusion_samples: usize,
    pub max_occlusion_intents: usize,
    pub max_history_frames: u32,
}

impl Default for AdvancedRuntimeLimits {
    fn default() -> Self {
        Self {
            max_lookahead_samples: MAX_LOOKAHEAD_SAMPLES,
            max_occlusion_samples: MAX_OCCLUSION_SAMPLES,
            max_occlusion_intents: MAX_OCCLUSION_INTENTS,
            max_history_frames: 2,
        }
    }
}

impl AdvancedRuntimeLimits {
    pub fn validate(self) -> Result<(), AdvancedRuntimeConfigError> {
        let checks = [
            (
                CapacityField::LookaheadSamples,
                self.max_lookahead_samples,
                MAX_LOOKAHEAD_SAMPLES,
            ),
            (
                CapacityField::OcclusionSamples,
                self.max_occlusion_samples,
                MAX_OCCLUSION_SAMPLES,
            ),
            (
                CapacityField::OcclusionIntents,
                self.max_occlusion_intents,
                MAX_OCCLUSION_INTENTS,
            ),
        ];
        for (field, requested, maximum) in checks {
            if requested > maximum {
                return Err(AdvancedRuntimeConfigError::CapacityExceeded {
                    field,
                    requested: requested as u32,
                    maximum: maximum as u32,
                });
            }
        }
        if self.max_history_frames > MAX_HISTORY_FRAMES {
            return Err(AdvancedRuntimeConfigError::CapacityExceeded {
                field: CapacityField::HistoryFrames,
                requested: self.max_history_frames,
                maximum: MAX_HISTORY_FRAMES,
            });
        }
        Ok(())
    }
}

/// Fixed-tick policy and all bounds used by [`AdvancedRuntime`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct AdvancedRuntimeConfig {
    pub fixed_delta_seconds: f32,
    pub tick_tolerance_seconds: f32,
    pub frame_rate: f32,
    pub near_clip_m: f32,
    pub limits: AdvancedRuntimeLimits,
}

impl Default for AdvancedRuntimeConfig {
    fn default() -> Self {
        Self {
            fixed_delta_seconds: 1.0 / 60.0,
            tick_tolerance_seconds: 1.0e-5,
            frame_rate: 60.0,
            near_clip_m: 1.0e-3,
            limits: AdvancedRuntimeLimits::default(),
        }
    }
}

impl AdvancedRuntimeConfig {
    pub fn validate(self) -> Result<(), AdvancedRuntimeConfigError> {
        if !self.fixed_delta_seconds.is_finite() || self.fixed_delta_seconds <= 0.0 {
            return Err(AdvancedRuntimeConfigError::InvalidFixedDelta);
        }
        if !self.tick_tolerance_seconds.is_finite() || self.tick_tolerance_seconds < 0.0 {
            return Err(AdvancedRuntimeConfigError::InvalidTickTolerance);
        }
        if !self.frame_rate.is_finite() || self.frame_rate <= 0.0 {
            return Err(AdvancedRuntimeConfigError::InvalidFrameRate);
        }
        if !self.near_clip_m.is_finite() || self.near_clip_m <= 0.0 {
            return Err(AdvancedRuntimeConfigError::InvalidNearClip);
        }
        self.limits.validate()
    }
}

/// A caller-provided fixed simulation tick.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct FixedTick {
    pub index: u64,
    pub frame_index: u64,
    pub delta_seconds: f32,
}

impl FixedTick {
    pub const fn new(index: u64, frame_index: u64, delta_seconds: f32) -> Self {
        Self {
            index,
            frame_index,
            delta_seconds,
        }
    }
}

/// Hard failures that prevent a tick from being committed.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum AdvancedRuntimeError {
    InvalidConfig(AdvancedRuntimeConfigError),
    NonFiniteDelta,
    NonPositiveDelta,
    DeltaMismatch { expected: f32, actual: f32 },
    TickReversed { previous: u64, current: u64 },
    TickGap { previous: u64, current: u64 },
    TickOverflow,
    InvalidTrackFallback,
}

/// Bit flags summarizing non-fatal stage degradation for one tick.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct RuntimeDegradation {
    pub bits: u32,
}

impl RuntimeDegradation {
    pub const TRACK_FALLBACK: u32 = 1 << 0;
    pub const FOCUS_UNAVAILABLE: u32 = 1 << 1;
    pub const SHUTTER_SANITIZED: u32 = 1 << 2;
    pub const LOD_SANITIZED: u32 = 1 << 3;
    pub const LOOKAHEAD_DROPPED: u32 = 1 << 4;
    pub const OCCLUSION_DROPPED: u32 = 1 << 5;
    pub const HISTORY_REJECTED: u32 = 1 << 6;

    pub const fn is_degraded(self) -> bool {
        self.bits != 0
    }

    fn add(&mut self, bit: u32) {
        self.bits |= bit;
    }
}

#[cfg(feature = "cutscene")]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum TrackRuntimeStatus {
    Sampled,
    HandoffToPrevious,
    Deactivated,
    InvalidSampleUsedFallback,
}

#[cfg(feature = "cutscene")]
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TrackRuntimeInput<'a> {
    pub track: &'a crate::ValidatedCameraTrack,
    pub fallback: CameraTrackSample,
}

#[cfg(feature = "cutscene")]
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TrackRuntimeResult {
    pub sample: CameraTrackSample,
    pub status: TrackRuntimeStatus,
}

#[cfg(feature = "autofocus")]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum FocusRuntimeStatus {
    Updated,
    FocusLost,
    SourceUnavailable,
    InvalidSource,
    InvalidState,
}

#[cfg(feature = "autofocus")]
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct FocusRuntimeResult {
    pub focus_distance_m: f32,
    pub status: FocusRuntimeStatus,
}

#[cfg(feature = "shutter")]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ShutterRuntimeStatus {
    Applied,
    InvalidFrameRate,
    InvalidAngle,
    InvalidFrameRateAndAngle,
}

#[cfg(feature = "shutter")]
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ShutterSnapshot {
    pub shutter_angle_deg: f32,
    pub fps: f32,
    pub exposure_seconds: f32,
    pub exposure: crate::Exposure,
    pub aperture_f_stop: f32,
    pub motion_blur_shutter_angle_deg: f32,
    pub status: ShutterRuntimeStatus,
}

#[cfg(feature = "view_lod")]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum LodRuntimeStatus {
    Applied,
    Sanitized,
    FrameRewound,
}

#[cfg(feature = "view_lod")]
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct LodRuntimeResult {
    pub update: bool,
    pub interval_frames: u64,
    pub resolution_scale: f32,
    pub history_frames: u32,
    pub status: LodRuntimeStatus,
}

#[cfg(feature = "lookahead_stream")]
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct LookaheadInput {
    pub current_position: DVec3,
    pub position: DVec3,
    pub velocity: DVec3,
    pub acceleration: DVec3,
    pub base_importance: f32,
    pub target_valid: bool,
}

#[cfg(feature = "lookahead_stream")]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum LookaheadRuntimeStatus {
    Predicted,
    InvalidInput,
    Disabled,
    Sanitized,
    Truncated,
}

#[cfg(feature = "lookahead_stream")]
#[derive(Clone, Debug, PartialEq)]
pub struct LookaheadRuntimeResult {
    pub samples: Vec<DVec3>,
    pub status: LookaheadRuntimeStatus,
    pub dropped_samples: usize,
    pub contribution: StreamingContribution,
}

#[cfg(feature = "occlusion_fade")]
#[derive(Clone, Debug, PartialEq)]
pub struct OcclusionRuntimeResult {
    pub intents: Vec<CameraOcclusionIntent>,
    pub accepted_samples: usize,
    pub dropped_samples: usize,
    pub invalid_samples: usize,
    pub capacity_reached: bool,
}

#[cfg(feature = "edge_stabilize")]
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct HistoryInput {
    pub history_valid: bool,
    pub linear_velocity: f32,
    pub angular_velocity: f32,
}

#[cfg(feature = "edge_stabilize")]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum HistoryRuntimeStatus {
    Accepted,
    Rejected(HistoryRejectReason),
    Sanitized,
}

#[cfg(feature = "edge_stabilize")]
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct HistoryRuntimeResult {
    pub weight: f32,
    pub status: HistoryRuntimeStatus,
    pub acceptance_rate: f32,
    pub dropped_count: u32,
    pub sharpen_strength: f32,
}

/// Inputs for all optional stages.  `Default` leaves every stage disabled.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct AdvancedRuntimeInput<'a> {
    #[cfg(feature = "cutscene")]
    pub track: Option<TrackRuntimeInput<'a>>,
    #[cfg(feature = "autofocus")]
    pub focus_source: Option<FocusSource>,
    #[cfg(feature = "shutter")]
    pub shutter: Option<Shutter>,
    #[cfg(feature = "view_lod")]
    pub lod: Option<ViewLod>,
    #[cfg(feature = "lookahead_stream")]
    pub lookahead: Option<LookaheadInput>,
    #[cfg(feature = "occlusion_fade")]
    pub occlusion_samples: &'a [OcclusionSample],
    #[cfg(feature = "edge_stabilize")]
    pub history: Option<HistoryInput>,
    #[cfg(not(any(feature = "cutscene", feature = "occlusion_fade")))]
    _marker: PhantomData<&'a ()>,
}

impl<'a> Default for AdvancedRuntimeInput<'a> {
    fn default() -> Self {
        Self::empty()
    }
}

impl<'a> AdvancedRuntimeInput<'a> {
    pub const fn empty() -> Self {
        Self {
            #[cfg(feature = "cutscene")]
            track: None,
            #[cfg(feature = "autofocus")]
            focus_source: None,
            #[cfg(feature = "shutter")]
            shutter: None,
            #[cfg(feature = "view_lod")]
            lod: None,
            #[cfg(feature = "lookahead_stream")]
            lookahead: None,
            #[cfg(feature = "occlusion_fade")]
            occlusion_samples: &[],
            #[cfg(feature = "edge_stabilize")]
            history: None,
            #[cfg(not(any(feature = "cutscene", feature = "occlusion_fade")))]
            _marker: PhantomData,
        }
    }

    #[cfg(feature = "cutscene")]
    pub fn with_track(mut self, track: TrackRuntimeInput<'a>) -> Self {
        self.track = Some(track);
        self
    }

    #[cfg(feature = "autofocus")]
    pub fn with_focus_source(mut self, source: FocusSource) -> Self {
        self.focus_source = Some(source);
        self
    }

    #[cfg(feature = "shutter")]
    pub fn with_shutter(mut self, shutter: Shutter) -> Self {
        self.shutter = Some(shutter);
        self
    }

    #[cfg(feature = "view_lod")]
    pub fn with_lod(mut self, lod: ViewLod) -> Self {
        self.lod = Some(lod);
        self
    }

    #[cfg(feature = "lookahead_stream")]
    pub fn with_lookahead(mut self, lookahead: LookaheadInput) -> Self {
        self.lookahead = Some(lookahead);
        self
    }

    #[cfg(feature = "occlusion_fade")]
    pub fn with_occlusion_samples(mut self, samples: &'a [OcclusionSample]) -> Self {
        self.occlusion_samples = samples;
        self
    }

    #[cfg(feature = "edge_stabilize")]
    pub fn with_history(mut self, history: HistoryInput) -> Self {
        self.history = Some(history);
        self
    }
}

/// One bounded, immutable result bundle produced by [`AdvancedRuntime::step`].
#[derive(Clone, Debug, PartialEq)]
pub struct AdvancedRuntimeOutput {
    pub tick: FixedTick,
    pub degradation: RuntimeDegradation,
    #[cfg(feature = "cutscene")]
    pub track: Option<TrackRuntimeResult>,
    #[cfg(feature = "autofocus")]
    pub focus: Option<FocusRuntimeResult>,
    #[cfg(feature = "shutter")]
    pub shutter: Option<ShutterSnapshot>,
    #[cfg(feature = "view_lod")]
    pub lod: Option<LodRuntimeResult>,
    #[cfg(feature = "lookahead_stream")]
    pub lookahead: Option<LookaheadRuntimeResult>,
    #[cfg(feature = "occlusion_fade")]
    pub occlusion: Option<OcclusionRuntimeResult>,
    #[cfg(feature = "edge_stabilize")]
    pub history: Option<HistoryRuntimeResult>,
}

/// Persistent fixed-tick state for the advanced camera stages.
#[derive(Clone, Debug)]
pub struct AdvancedRuntime {
    config: AdvancedRuntimeConfig,
    last_tick: Option<u64>,
    #[cfg(feature = "view_lod")]
    last_lod_frame: Option<u64>,
    #[cfg(feature = "autofocus")]
    focus_tracker: FocusTracker,
    #[cfg(feature = "occlusion_fade")]
    occlusion_fade: OcclusionFadeState,
    #[cfg(feature = "lookahead_stream")]
    lookahead_predictor: LookaheadPredictor,
    #[cfg(feature = "edge_stabilize")]
    history_stabilizer: HistoryStabilizer,
}

impl AdvancedRuntime {
    pub fn new(config: AdvancedRuntimeConfig) -> Result<Self, AdvancedRuntimeError> {
        config
            .validate()
            .map_err(AdvancedRuntimeError::InvalidConfig)?;
        Ok(Self {
            config,
            last_tick: None,
            #[cfg(feature = "view_lod")]
            last_lod_frame: None,
            #[cfg(feature = "autofocus")]
            focus_tracker: FocusTracker::default(),
            #[cfg(feature = "occlusion_fade")]
            occlusion_fade: OcclusionFadeState::default(),
            #[cfg(feature = "lookahead_stream")]
            lookahead_predictor: LookaheadPredictor::default(),
            #[cfg(feature = "edge_stabilize")]
            history_stabilizer: HistoryStabilizer::default(),
        })
    }

    pub fn config(&self) -> AdvancedRuntimeConfig {
        self.config
    }

    pub fn last_tick(&self) -> Option<u64> {
        self.last_tick
    }

    pub fn reset_timeline(&mut self) {
        self.last_tick = None;
        #[cfg(feature = "view_lod")]
        {
            self.last_lod_frame = None;
        }
    }

    #[cfg(feature = "autofocus")]
    pub fn focus_tracker(&self) -> &FocusTracker {
        &self.focus_tracker
    }

    #[cfg(feature = "autofocus")]
    pub fn focus_tracker_mut(&mut self) -> &mut FocusTracker {
        &mut self.focus_tracker
    }

    #[cfg(feature = "occlusion_fade")]
    pub fn occlusion_fade(&self) -> &OcclusionFadeState {
        &self.occlusion_fade
    }

    #[cfg(feature = "occlusion_fade")]
    pub fn occlusion_fade_mut(&mut self) -> &mut OcclusionFadeState {
        &mut self.occlusion_fade
    }

    #[cfg(feature = "lookahead_stream")]
    pub fn lookahead_predictor(&self) -> &LookaheadPredictor {
        &self.lookahead_predictor
    }

    #[cfg(feature = "lookahead_stream")]
    pub fn lookahead_predictor_mut(&mut self) -> &mut LookaheadPredictor {
        &mut self.lookahead_predictor
    }

    #[cfg(feature = "edge_stabilize")]
    pub fn history_stabilizer(&self) -> &HistoryStabilizer {
        &self.history_stabilizer
    }

    #[cfg(feature = "edge_stabilize")]
    pub fn history_stabilizer_mut(&mut self) -> &mut HistoryStabilizer {
        &mut self.history_stabilizer
    }

    /// Runs every enabled stage once, in a deterministic order.
    pub fn step(
        &mut self,
        tick: FixedTick,
        input: AdvancedRuntimeInput<'_>,
    ) -> Result<AdvancedRuntimeOutput, AdvancedRuntimeError> {
        self.validate_tick(tick)?;

        #[cfg(feature = "cutscene")]
        if let Some(track) = input.track {
            if !valid_track_sample(track.fallback) {
                return Err(AdvancedRuntimeError::InvalidTrackFallback);
            }
        }

        #[allow(unused_mut)]
        let mut degradation = RuntimeDegradation::default();
        let _ = &input;

        #[cfg(feature = "cutscene")]
        let track_result = input.track.map(|track| {
            let sample = track.track.sample_state(tick.index, track.fallback);
            if !valid_track_sample(sample) {
                degradation.add(RuntimeDegradation::TRACK_FALLBACK);
                TrackRuntimeResult {
                    sample: track.fallback,
                    status: TrackRuntimeStatus::InvalidSampleUsedFallback,
                }
            } else if sample.handoff_to_previous {
                degradation.add(RuntimeDegradation::TRACK_FALLBACK);
                TrackRuntimeResult {
                    sample,
                    status: TrackRuntimeStatus::HandoffToPrevious,
                }
            } else if !sample.active {
                degradation.add(RuntimeDegradation::TRACK_FALLBACK);
                TrackRuntimeResult {
                    sample,
                    status: TrackRuntimeStatus::Deactivated,
                }
            } else {
                TrackRuntimeResult {
                    sample,
                    status: TrackRuntimeStatus::Sampled,
                }
            }
        });

        #[cfg(feature = "autofocus")]
        let focus_result = input.focus_source.map(|source| {
            let source_unavailable =
                matches!(source, FocusSource::ScreenCenterHit { distance_m: None });
            let updated =
                self.focus_tracker
                    .update(source, self.config.near_clip_m, tick.delta_seconds);
            let status = if source_unavailable {
                degradation.add(RuntimeDegradation::FOCUS_UNAVAILABLE);
                FocusRuntimeStatus::FocusLost
            } else if !updated {
                degradation.add(RuntimeDegradation::FOCUS_UNAVAILABLE);
                FocusRuntimeStatus::InvalidSource
            } else if !self.focus_tracker.focus_distance_m.is_finite() {
                degradation.add(RuntimeDegradation::FOCUS_UNAVAILABLE);
                FocusRuntimeStatus::InvalidState
            } else {
                FocusRuntimeStatus::Updated
            };
            FocusRuntimeResult {
                focus_distance_m: if self.focus_tracker.focus_distance_m.is_finite() {
                    self.focus_tracker.focus_distance_m
                } else {
                    FocusTracker::default().focus_distance_m
                },
                status,
            }
        });

        #[cfg(feature = "shutter")]
        let shutter_result = input.shutter.map(|shutter| {
            let valid_fps = shutter.fps.is_finite() && shutter.fps > 0.0;
            let valid_angle = shutter.shutter_angle_deg.is_finite();
            let valid_aperture =
                shutter.aperture_f_stop.is_finite() && shutter.aperture_f_stop > 0.0;
            let safe = Shutter {
                shutter_angle_deg: if valid_angle {
                    shutter.shutter_angle_deg.clamp(0.0, 360.0)
                } else {
                    Shutter::default().shutter_angle_deg
                },
                fps: if valid_fps {
                    shutter.fps
                } else {
                    Shutter::default().fps
                },
                aperture_f_stop: if valid_aperture {
                    shutter.aperture_f_stop
                } else {
                    Shutter::default().aperture_f_stop
                },
            };
            let status = match (valid_fps, valid_angle) {
                (true, true) => ShutterRuntimeStatus::Applied,
                (false, true) => ShutterRuntimeStatus::InvalidFrameRate,
                (true, false) => ShutterRuntimeStatus::InvalidAngle,
                (false, false) => ShutterRuntimeStatus::InvalidFrameRateAndAngle,
            };
            if status != ShutterRuntimeStatus::Applied || !valid_aperture {
                degradation.add(RuntimeDegradation::SHUTTER_SANITIZED);
            }
            ShutterSnapshot {
                shutter_angle_deg: safe.shutter_angle_deg,
                fps: safe.fps,
                exposure_seconds: safe.exposure_seconds(),
                exposure: safe.physical_exposure(100.0),
                aperture_f_stop: safe.aperture_f_stop,
                motion_blur_shutter_angle_deg: safe.shutter_angle_deg,
                status,
            }
        });

        #[cfg(feature = "view_lod")]
        let lod_result = input.lod.map(|lod| {
            let defaults = ViewLod::default();
            let mut safe = lod;
            let mut sanitized = false;
            if !safe.max_update_hz.is_finite() || safe.max_update_hz <= 0.0 {
                safe.max_update_hz = defaults.max_update_hz;
                sanitized = true;
            }
            if !safe.min_resolution_scale.is_finite() {
                safe.min_resolution_scale = defaults.min_resolution_scale;
                sanitized = true;
            } else if !(0.0..=1.0).contains(&safe.min_resolution_scale) {
                sanitized = true;
            }
            let interval_frames = safe.update_interval_frames(self.config.frame_rate);
            let previous_frame = self.last_lod_frame;
            let rewound = previous_frame.is_some_and(|previous| tick.frame_index < previous);
            let update = previous_frame.is_none_or(|previous| {
                safe.should_update(tick.frame_index, previous, self.config.frame_rate)
            });
            if update {
                self.last_lod_frame = Some(tick.frame_index);
            }
            let history_frames = safe
                .effective_history_frames()
                .min(self.config.limits.max_history_frames);
            if history_frames != safe.history_frames {
                sanitized = true;
            }
            if sanitized || rewound {
                degradation.add(RuntimeDegradation::LOD_SANITIZED);
            }
            LodRuntimeResult {
                update,
                interval_frames,
                resolution_scale: safe.resolution_scale(),
                history_frames,
                status: if rewound {
                    LodRuntimeStatus::FrameRewound
                } else if sanitized {
                    LodRuntimeStatus::Sanitized
                } else {
                    LodRuntimeStatus::Applied
                },
            }
        });

        #[cfg(feature = "lookahead_stream")]
        let lookahead_result = input.lookahead.map(|lookahead| {
            let capacity = self.config.limits.max_lookahead_samples;
            if capacity == 0 {
                degradation.add(RuntimeDegradation::LOOKAHEAD_DROPPED);
                return LookaheadRuntimeResult {
                    samples: Vec::new(),
                    status: LookaheadRuntimeStatus::Disabled,
                    dropped_samples: 0,
                    contribution: StreamingContribution {
                        importance: 0.0,
                        distance_m: 0.0,
                        sample_count: 0,
                    },
                };
            }
            if !lookahead.position.is_finite()
                || !lookahead.velocity.is_finite()
                || !lookahead.acceleration.is_finite()
            {
                degradation.add(RuntimeDegradation::LOOKAHEAD_DROPPED);
                return LookaheadRuntimeResult {
                    samples: Vec::with_capacity(capacity),
                    status: LookaheadRuntimeStatus::InvalidInput,
                    dropped_samples: 0,
                    contribution: StreamingContribution {
                        importance: 0.0,
                        distance_m: 0.0,
                        sample_count: 0,
                    },
                };
            }
            let defaults = LookaheadPredictor::default();
            let mut predictor = self.lookahead_predictor;
            let sanitized = if predictor.horizon_s.is_finite()
                && predictor.sample_step_s.is_finite()
                && predictor.max_speed.is_finite()
                && predictor.max_acceleration.is_finite()
                && predictor.horizon_s >= 0.0
                && predictor.sample_step_s > 0.0
                && predictor.max_speed >= 0.0
                && predictor.max_acceleration >= 0.0
            {
                false
            } else {
                predictor = defaults;
                true
            };
            if sanitized {
                self.lookahead_predictor = predictor;
            }
            let mut samples = predictor.predict(
                lookahead.position,
                lookahead.velocity,
                lookahead.acceleration,
            );
            let dropped_samples = samples.len().saturating_sub(capacity);
            if dropped_samples != 0 {
                samples.truncate(capacity);
            }
            let contribution = predictor.contribution(
                &samples,
                lookahead.current_position,
                lookahead.base_importance,
                lookahead.target_valid,
            );
            if sanitized || dropped_samples != 0 {
                degradation.add(RuntimeDegradation::LOOKAHEAD_DROPPED);
            }
            LookaheadRuntimeResult {
                samples,
                status: if dropped_samples != 0 {
                    LookaheadRuntimeStatus::Truncated
                } else if sanitized {
                    LookaheadRuntimeStatus::Sanitized
                } else {
                    LookaheadRuntimeStatus::Predicted
                },
                dropped_samples,
                contribution,
            }
        });

        #[cfg(feature = "occlusion_fade")]
        let occlusion_result = {
            let capacity = self.config.limits.max_occlusion_intents;
            let sample_capacity = self.config.limits.max_occlusion_samples;
            if capacity == 0 || sample_capacity == 0 {
                self.occlusion_fade.intents.clear();
                if !input.occlusion_samples.is_empty() {
                    degradation.add(RuntimeDegradation::OCCLUSION_DROPPED);
                }
                Some(OcclusionRuntimeResult {
                    intents: Vec::new(),
                    accepted_samples: 0,
                    dropped_samples: input.occlusion_samples.len(),
                    invalid_samples: 0,
                    capacity_reached: true,
                })
            } else {
                self.occlusion_fade.intents.truncate(capacity);
                let accepted_samples = input.occlusion_samples.len().min(sample_capacity);
                let dropped_samples = input
                    .occlusion_samples
                    .len()
                    .saturating_sub(accepted_samples);
                let samples = &input.occlusion_samples[..accepted_samples];
                let invalid_samples = samples
                    .iter()
                    .filter(|sample| !sample.coverage.is_finite())
                    .count();
                self.occlusion_fade.update(samples, tick.delta_seconds);
                let capacity_reached = self.occlusion_fade.intents.len() >= capacity;
                let mut intents = Vec::with_capacity(capacity);
                intents.extend(self.occlusion_fade.intents.iter().take(capacity).copied());
                if dropped_samples != 0 || invalid_samples != 0 || capacity_reached {
                    degradation.add(RuntimeDegradation::OCCLUSION_DROPPED);
                }
                Some(OcclusionRuntimeResult {
                    intents,
                    accepted_samples,
                    dropped_samples,
                    invalid_samples,
                    capacity_reached,
                })
            }
        };

        #[cfg(feature = "edge_stabilize")]
        let history_result = input.history.map(|history| {
            let defaults = HistoryStabilizer::default();
            let mut stabilizer = self.history_stabilizer;
            let sanitized = if stabilizer.max_linear_velocity.is_finite()
                && stabilizer.max_linear_velocity >= 0.0
                && stabilizer.max_angular_velocity.is_finite()
                && stabilizer.max_angular_velocity >= 0.0
                && stabilizer.clamp_weight.is_finite()
            {
                false
            } else {
                stabilizer = defaults;
                true
            };
            if sanitized {
                self.history_stabilizer = stabilizer;
            }
            let result = stabilizer.history_weight(
                history.history_valid,
                history.linear_velocity,
                history.angular_velocity,
            );
            let (weight, status) = match result {
                Ok(weight) if !sanitized => (weight, HistoryRuntimeStatus::Accepted),
                Ok(weight) => (weight, HistoryRuntimeStatus::Sanitized),
                Err(reason) => {
                    degradation.add(RuntimeDegradation::HISTORY_REJECTED);
                    (0.0, HistoryRuntimeStatus::Rejected(reason))
                }
            };
            if sanitized {
                degradation.add(RuntimeDegradation::HISTORY_REJECTED);
            }
            let acceptance_rate = if matches!(status, HistoryRuntimeStatus::Accepted) {
                1.0
            } else {
                0.0
            };
            HistoryRuntimeResult {
                weight,
                status,
                acceptance_rate,
                dropped_count: u32::from(!matches!(status, HistoryRuntimeStatus::Accepted)),
                sharpen_strength: stabilizer.sharpen_strength(),
            }
        });

        self.last_tick = Some(tick.index);
        Ok(AdvancedRuntimeOutput {
            tick,
            degradation,
            #[cfg(feature = "cutscene")]
            track: track_result,
            #[cfg(feature = "autofocus")]
            focus: focus_result,
            #[cfg(feature = "shutter")]
            shutter: shutter_result,
            #[cfg(feature = "view_lod")]
            lod: lod_result,
            #[cfg(feature = "lookahead_stream")]
            lookahead: lookahead_result,
            #[cfg(feature = "occlusion_fade")]
            occlusion: occlusion_result,
            #[cfg(feature = "edge_stabilize")]
            history: history_result,
        })
    }

    fn validate_tick(&self, tick: FixedTick) -> Result<(), AdvancedRuntimeError> {
        if !tick.delta_seconds.is_finite() {
            return Err(AdvancedRuntimeError::NonFiniteDelta);
        }
        if tick.delta_seconds <= 0.0 {
            return Err(AdvancedRuntimeError::NonPositiveDelta);
        }
        if (tick.delta_seconds - self.config.fixed_delta_seconds).abs()
            > self.config.tick_tolerance_seconds
        {
            return Err(AdvancedRuntimeError::DeltaMismatch {
                expected: self.config.fixed_delta_seconds,
                actual: tick.delta_seconds,
            });
        }
        let Some(previous) = self.last_tick else {
            return Ok(());
        };
        if previous == u64::MAX {
            return Err(AdvancedRuntimeError::TickOverflow);
        }
        if tick.index <= previous {
            return Err(AdvancedRuntimeError::TickReversed {
                previous,
                current: tick.index,
            });
        }
        if tick.index != previous + 1 {
            return Err(AdvancedRuntimeError::TickGap {
                previous,
                current: tick.index,
            });
        }
        Ok(())
    }
}

#[cfg(feature = "cutscene")]
fn valid_track_sample(sample: CameraTrackSample) -> bool {
    sample.pose.position.is_finite()
        && sample.pose.rotation.is_finite()
        && sample.pose.world_origin.is_finite()
        && sample.focal_length_mm.is_finite()
        && sample.focal_length_mm > 0.0
        && sample.exposure.ev100().is_finite()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rejects_invalid_limits_before_allocating_runtime() {
        let mut config = AdvancedRuntimeConfig::default();
        config.limits.max_lookahead_samples = MAX_LOOKAHEAD_SAMPLES + 1;
        assert!(matches!(
            AdvancedRuntime::new(config),
            Err(AdvancedRuntimeError::InvalidConfig(
                AdvancedRuntimeConfigError::CapacityExceeded {
                    field: CapacityField::LookaheadSamples,
                    ..
                }
            ))
        ));
    }

    #[test]
    fn fixed_tick_requires_contiguous_ticks() {
        let mut runtime = AdvancedRuntime::new(AdvancedRuntimeConfig::default()).unwrap();
        let dt = runtime.config().fixed_delta_seconds;
        runtime
            .step(FixedTick::new(10, 10, dt), AdvancedRuntimeInput::default())
            .unwrap();
        assert!(matches!(
            runtime.step(FixedTick::new(12, 12, dt), AdvancedRuntimeInput::default()),
            Err(AdvancedRuntimeError::TickGap { .. })
        ));
        assert_eq!(runtime.last_tick(), Some(10));
    }

    #[cfg(feature = "cutscene")]
    #[test]
    fn non_finite_track_keyframe_is_rejected_before_runtime() {
        use crate::{CameraTrack, Exposure};
        use prism_math::{DQuat, DVec3};

        let fallback = CameraTrackSample {
            pose: CameraPose::default(),
            focal_length_mm: 35.0,
            exposure: Exposure::default(),
            event_key: None,
            active: true,
            handoff_to_previous: false,
        };
        let invalid = CameraTrackSample {
            pose: CameraPose {
                position: DVec3::new(f32::NAN as f64, 0.0, 0.0),
                rotation: DQuat::IDENTITY,
                world_origin: DVec3::ZERO,
            },
            ..fallback
        };
        let mut track = CameraTrack::default();
        track.keyframes.push(crate::CameraKeyframe {
            tick: 0,
            pose: invalid.pose,
            focal_length_mm: invalid.focal_length_mm,
            exposure: invalid.exposure,
            event_key: 1,
        });
        assert_eq!(
            track.into_validated().unwrap_err().code,
            crate::CameraTrackValidationCode::NonFinitePose
        );
        assert!(fallback.active);
    }

    #[cfg(feature = "cutscene")]
    #[test]
    fn unsupported_track_version_is_rejected_before_runtime() {
        let mut track = CameraTrack::default();
        track.asset_version = CameraTrack::CURRENT_ASSET_VERSION + 1;
        track.keyframes.push(crate::CameraKeyframe {
            tick: 0,
            pose: CameraPose::default(),
            focal_length_mm: 50.0,
            exposure: crate::Exposure::default(),
            event_key: 9,
        });
        assert_eq!(
            track.into_validated().unwrap_err().code,
            crate::CameraTrackValidationCode::UnsupportedVersion
        );
    }

    #[cfg(all(
        feature = "cutscene",
        feature = "autofocus",
        feature = "shutter",
        feature = "view_lod",
        feature = "lookahead_stream",
        feature = "occlusion_fade",
        feature = "edge_stabilize"
    ))]
    #[test]
    fn runs_all_enabled_stages_with_bounded_results() {
        use crate::{CameraKeyframe, Exposure};
        use prism_math::DVec3;

        let fallback = CameraTrackSample {
            pose: CameraPose::default(),
            focal_length_mm: 35.0,
            exposure: Exposure::default(),
            event_key: None,
            active: true,
            handoff_to_previous: false,
        };
        let mut track = CameraTrack::default();
        assert!(track.push(CameraKeyframe {
            tick: 0,
            pose: fallback.pose,
            focal_length_mm: fallback.focal_length_mm,
            exposure: fallback.exposure,
            event_key: 7,
        }));
        let track = track.into_validated().unwrap();

        let occlusion = [
            OcclusionSample {
                entity_id: 1,
                generation: 0,
                coverage: 0.5,
            },
            OcclusionSample {
                entity_id: 2,
                generation: 0,
                coverage: 0.5,
            },
            OcclusionSample {
                entity_id: 3,
                generation: 0,
                coverage: 0.5,
            },
        ];
        let mut config = AdvancedRuntimeConfig::default();
        config.limits.max_lookahead_samples = 2;
        config.limits.max_occlusion_samples = 2;
        config.limits.max_occlusion_intents = 2;
        let mut runtime = AdvancedRuntime::new(config).unwrap();
        let dt = runtime.config().fixed_delta_seconds;
        let input = AdvancedRuntimeInput::empty()
            .with_track(TrackRuntimeInput {
                track: &track,
                fallback,
            })
            .with_focus_source(FocusSource::TargetDistance { distance_m: 10.0 })
            .with_shutter(Shutter {
                shutter_angle_deg: 180.0,
                fps: f32::NAN,
                aperture_f_stop: 2.8,
            })
            .with_lod(ViewLod {
                max_update_hz: 30.0,
                min_resolution_scale: 0.5,
                history_frames: 2,
            })
            .with_lookahead(LookaheadInput {
                current_position: DVec3::ZERO,
                position: DVec3::ZERO,
                velocity: DVec3::X,
                acceleration: DVec3::ZERO,
                base_importance: 1.0,
                target_valid: true,
            })
            .with_occlusion_samples(&occlusion)
            .with_history(HistoryInput {
                history_valid: true,
                linear_velocity: 1.0,
                angular_velocity: 1.0,
            });
        let output = runtime.step(FixedTick::new(0, 0, dt), input).unwrap();

        assert_eq!(output.track.unwrap().status, TrackRuntimeStatus::Sampled);
        assert_eq!(output.focus.unwrap().status, FocusRuntimeStatus::Updated);
        assert_eq!(
            output.shutter.unwrap().status,
            ShutterRuntimeStatus::InvalidFrameRate
        );
        assert!(output.lod.unwrap().update);
        let lookahead = output.lookahead.unwrap();
        assert_eq!(lookahead.samples.len(), 2);
        assert_eq!(lookahead.dropped_samples, 3);
        let occlusion = output.occlusion.unwrap();
        assert_eq!(occlusion.accepted_samples, 2);
        assert_eq!(occlusion.dropped_samples, 1);
        assert_eq!(occlusion.intents.len(), 2);
        assert_eq!(
            output.history.unwrap().status,
            HistoryRuntimeStatus::Accepted
        );
        assert!(output.degradation.is_degraded());
    }
}
