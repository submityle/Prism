#[cfg(any(
    feature = "cutscene",
    feature = "occlusion_fade",
    feature = "lookahead_stream",
    feature = "composition_aid",
    feature = "view_lod",
    feature = "edge_stabilize"
))]
use alloc::vec::Vec;

#[cfg(any(feature = "cutscene", feature = "lookahead_stream"))]
use prism_math::DVec3;
#[cfg(any(feature = "virtual_production", feature = "composition_aid"))]
use prism_math::Vec2;

#[cfg(any(feature = "cutscene", feature = "virtual_production"))]
use crate::CameraPose;
#[cfg(any(feature = "cutscene", feature = "shutter"))]
use crate::Exposure;
#[cfg(feature = "physical_lens")]
use crate::Lens;
#[cfg(all(feature = "cutscene", feature = "post"))]
use crate::PostStack;
#[cfg(feature = "determinism")]
use crate::ViewData;
#[cfg(feature = "view_lod")]
use crate::ViewKind;

#[cfg(feature = "cutscene")]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CutsceneEndPolicy {
    Hold,
    BlendToPrevious,
    Deactivate,
}

#[cfg(feature = "cutscene")]
impl Default for CutsceneEndPolicy {
    fn default() -> Self {
        Self::Hold
    }
}

#[cfg(feature = "cutscene")]
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CameraKeyframe {
    pub tick: u64,
    pub pose: CameraPose,
    pub focal_length_mm: f32,
    pub exposure: Exposure,
    pub event_key: u64,
}

#[cfg(feature = "cutscene")]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct CameraTrackValidationError {
    pub code: CameraTrackValidationCode,
    pub index: usize,
}

#[cfg(feature = "cutscene")]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CameraTrackValidationCode {
    UnsupportedVersion,
    Empty,
    NonFinitePose,
    NonFiniteLens,
    NonFiniteExposure,
    NonMonotonicTick,
}

#[cfg(feature = "cutscene")]
#[derive(Clone, Debug, PartialEq)]
pub struct CameraTrack {
    pub asset_version: u32,
    pub keyframes: Vec<CameraKeyframe>,
    pub end_policy: CutsceneEndPolicy,
    #[cfg(feature = "post")]
    pub post_stack: Option<PostStack>,
}

#[cfg(feature = "cutscene")]
impl Default for CameraTrack {
    fn default() -> Self {
        Self {
            asset_version: Self::CURRENT_ASSET_VERSION,
            keyframes: Vec::new(),
            end_policy: CutsceneEndPolicy::default(),
            #[cfg(feature = "post")]
            post_stack: None,
        }
    }
}

#[cfg(feature = "cutscene")]
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CameraTrackSample {
    pub pose: CameraPose,
    pub focal_length_mm: f32,
    pub exposure: Exposure,
    pub event_key: Option<u64>,
    pub active: bool,
    pub handoff_to_previous: bool,
}

#[cfg(feature = "cutscene")]
impl CameraTrack {
    pub const CURRENT_ASSET_VERSION: u32 = 1;
    pub const MAX_KEYFRAMES: usize = 4096;

    pub fn validate(&self) -> Result<(), CameraTrackValidationError> {
        if self.asset_version != Self::CURRENT_ASSET_VERSION {
            return Err(CameraTrackValidationError {
                code: CameraTrackValidationCode::UnsupportedVersion,
                index: 0,
            });
        }
        if self.keyframes.is_empty() {
            return Err(CameraTrackValidationError {
                code: CameraTrackValidationCode::Empty,
                index: 0,
            });
        }
        for (index, keyframe) in self.keyframes.iter().enumerate() {
            if !keyframe.pose.position.is_finite()
                || !keyframe.pose.rotation.is_finite()
                || !keyframe.pose.world_origin.is_finite()
            {
                return Err(CameraTrackValidationError {
                    code: CameraTrackValidationCode::NonFinitePose,
                    index,
                });
            }
            if !keyframe.focal_length_mm.is_finite() || keyframe.focal_length_mm <= 0.0 {
                return Err(CameraTrackValidationError {
                    code: CameraTrackValidationCode::NonFiniteLens,
                    index,
                });
            }
            if !keyframe.exposure.ev100().is_finite() {
                return Err(CameraTrackValidationError {
                    code: CameraTrackValidationCode::NonFiniteExposure,
                    index,
                });
            }
            if index != 0 && self.keyframes[index - 1].tick >= keyframe.tick {
                return Err(CameraTrackValidationError {
                    code: CameraTrackValidationCode::NonMonotonicTick,
                    index,
                });
            }
        }
        Ok(())
    }

    /// Validates once and freezes the track for allocation-free fixed-tick sampling.
    pub fn into_validated(self) -> Result<ValidatedCameraTrack, CameraTrackValidationError> {
        self.validate()?;
        Ok(ValidatedCameraTrack { track: self })
    }

    pub fn push(&mut self, keyframe: CameraKeyframe) -> bool {
        if self.keyframes.len() >= Self::MAX_KEYFRAMES
            || self
                .keyframes
                .last()
                .is_some_and(|last| last.tick >= keyframe.tick)
        {
            return false;
        }
        self.keyframes.push(keyframe);
        true
    }

    pub fn sample(&self, tick: u64, fallback: CameraPose) -> (CameraPose, Option<u64>) {
        let sample = self.sample_state(
            tick,
            CameraTrackSample {
                pose: fallback,
                focal_length_mm: 50.0,
                exposure: Exposure::default(),
                event_key: None,
                active: true,
                handoff_to_previous: false,
            },
        );
        (sample.pose, sample.event_key)
    }

    pub fn sample_state(&self, tick: u64, fallback: CameraTrackSample) -> CameraTrackSample {
        let Some(first) = self.keyframes.first().copied() else {
            return fallback;
        };
        let Some(last) = self.keyframes.last().copied() else {
            return fallback;
        };
        if tick <= first.tick {
            return CameraTrackSample {
                pose: first.pose,
                focal_length_mm: first.focal_length_mm,
                exposure: first.exposure,
                event_key: Some(first.event_key),
                active: true,
                handoff_to_previous: false,
            };
        }
        if tick >= last.tick {
            return match self.end_policy {
                CutsceneEndPolicy::Hold => CameraTrackSample {
                    pose: last.pose,
                    focal_length_mm: last.focal_length_mm,
                    exposure: last.exposure,
                    event_key: Some(last.event_key),
                    active: true,
                    handoff_to_previous: false,
                },
                CutsceneEndPolicy::BlendToPrevious => CameraTrackSample {
                    handoff_to_previous: true,
                    ..fallback
                },
                CutsceneEndPolicy::Deactivate => CameraTrackSample {
                    active: false,
                    ..fallback
                },
            };
        }
        let index = self.keyframes.partition_point(|frame| frame.tick <= tick);
        let a = self.keyframes[index - 1];
        let b = self.keyframes[index];
        let segment_ticks = (b.tick - a.tick) as f64;
        let linear_t = (tick - a.tick) as f64 / segment_ticks;
        let t = linear_t.clamp(0.0, 1.0);
        let previous = self.keyframes.get(index.saturating_sub(2)).unwrap_or(&a);
        let next = self.keyframes.get(index + 1).unwrap_or(&b);
        let tangent_a_scale = segment_ticks / (b.tick - previous.tick) as f64;
        let tangent_b_scale = segment_ticks / (next.tick - a.tick) as f64;
        let position = hermite_dvec3(
            a.pose.position,
            b.pose.position,
            (b.pose.position - previous.pose.position) * tangent_a_scale,
            (next.pose.position - a.pose.position) * tangent_b_scale,
            t,
        );
        let origin = hermite_dvec3(
            a.pose.world_origin,
            b.pose.world_origin,
            (b.pose.world_origin - previous.pose.world_origin) * tangent_a_scale,
            (next.pose.world_origin - a.pose.world_origin) * tangent_b_scale,
            t,
        );
        let tangent_a_focal =
            (b.focal_length_mm - previous.focal_length_mm) * tangent_a_scale as f32;
        let tangent_b_focal = (next.focal_length_mm - a.focal_length_mm) * tangent_b_scale as f32;
        let focal_length_mm = hermite_scalar(
            a.focal_length_mm,
            b.focal_length_mm,
            tangent_a_focal,
            tangent_b_focal,
            t as f32,
        )
        .clamp(
            a.focal_length_mm.min(b.focal_length_mm),
            a.focal_length_mm.max(b.focal_length_mm),
        )
        .max(1.0e-4);
        CameraTrackSample {
            pose: CameraPose {
                position,
                rotation: a.pose.rotation.slerp(b.pose.rotation, t),
                world_origin: origin,
            },
            focal_length_mm,
            exposure: lerp_exposure(a.exposure, b.exposure, t as f32),
            event_key: Some(if linear_t < 0.5 {
                a.event_key
            } else {
                b.event_key
            }),
            active: true,
            handoff_to_previous: false,
        }
    }
}

/// Immutable, prevalidated camera animation ready for runtime sampling.
#[cfg(feature = "cutscene")]
#[derive(Clone, Debug, PartialEq)]
pub struct ValidatedCameraTrack {
    track: CameraTrack,
}

/// Stateful, allocation-free event cursor for one validated camera track.
///
/// Keep one cursor per playing track instance. The cursor consumes authored
/// keyframe events exactly once as fixed ticks advance, including when a
/// caller skips over several keyframes in one update.
#[cfg(feature = "cutscene")]
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct CameraTrackEventCursor {
    last_tick: Option<u64>,
}

/// Failure to advance an event cursor monotonically.
#[cfg(feature = "cutscene")]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CameraTrackEventError {
    /// The requested tick precedes the cursor's last committed tick.
    TickReversed { previous: u64, requested: u64 },
}

/// One authored event occurrence crossed by a track playback cursor.
#[cfg(feature = "cutscene")]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct CameraTrackEventOccurrence {
    /// Authored fixed tick of this occurrence; distinct keyframes may share a payload key.
    pub tick: u64,
    /// Application-defined event payload key.
    pub event_key: u64,
}

#[cfg(feature = "cutscene")]
impl CameraTrackEventCursor {
    /// Returns the last committed tick, or `None` before the first advance.
    #[must_use]
    pub const fn last_tick(self) -> Option<u64> {
        self.last_tick
    }

    /// Advances through `(last_tick, tick]`, replacing `output` with crossed
    /// event keys in authored order. The first call establishes a baseline and
    /// emits no historical events. Equal ticks emit nothing; reversed ticks
    /// leave both the cursor and output unchanged.
    pub fn advance(
        &mut self,
        track: &ValidatedCameraTrack,
        tick: u64,
        output: &mut Vec<CameraTrackEventOccurrence>,
    ) -> Result<(), CameraTrackEventError> {
        if let Some(previous) = self.last_tick {
            if tick < previous {
                return Err(CameraTrackEventError::TickReversed {
                    previous,
                    requested: tick,
                });
            }
            if tick == previous {
                output.clear();
                return Ok(());
            }
            track.append_event_keys(previous, tick, output);
        } else {
            output.clear();
        }
        self.last_tick = Some(tick);
        Ok(())
    }
}

#[cfg(feature = "cutscene")]
impl ValidatedCameraTrack {
    /// Samples the validated track without repeating its O(keyframe-count) validation.
    pub fn sample_state(&self, tick: u64, fallback: CameraTrackSample) -> CameraTrackSample {
        self.track.sample_state(tick, fallback)
    }

    /// Returns the number of immutable keyframes in this track.
    pub fn keyframe_count(&self) -> usize {
        self.track.keyframes.len()
    }

    /// Appends events in `(from_tick, through_tick]` without rescanning the
    /// prefix. The output buffer is cleared first and retains its capacity.
    /// Complexity is `O(log N + K)` for `N` keyframes and `K` crossed events.
    fn append_event_keys(
        &self,
        from_tick: u64,
        through_tick: u64,
        output: &mut Vec<CameraTrackEventOccurrence>,
    ) {
        output.clear();
        if through_tick <= from_tick {
            return;
        }
        let first = self
            .track
            .keyframes
            .partition_point(|frame| frame.tick <= from_tick);
        let end = self
            .track
            .keyframes
            .partition_point(|frame| frame.tick <= through_tick);
        output.extend(self.track.keyframes[first..end].iter().map(|frame| {
            CameraTrackEventOccurrence {
                tick: frame.tick,
                event_key: frame.event_key,
            }
        }));
    }
}

#[cfg(feature = "cutscene")]
fn hermite_scalar(p0: f32, p1: f32, m0: f32, m1: f32, t: f32) -> f32 {
    let t2 = t * t;
    let t3 = t2 * t;
    (2.0 * t3 - 3.0 * t2 + 1.0) * p0
        + (t3 - 2.0 * t2 + t) * m0
        + (-2.0 * t3 + 3.0 * t2) * p1
        + (t3 - t2) * m1
}

#[cfg(feature = "cutscene")]
fn hermite_dvec3(p0: DVec3, p1: DVec3, m0: DVec3, m1: DVec3, t: f64) -> DVec3 {
    let t2 = t * t;
    let t3 = t2 * t;
    p0 * (2.0 * t3 - 3.0 * t2 + 1.0)
        + m0 * (t3 - 2.0 * t2 + t)
        + p1 * (-2.0 * t3 + 3.0 * t2)
        + m1 * (t3 - t2)
}

#[cfg(feature = "cutscene")]
fn lerp_exposure(a: Exposure, b: Exposure, t: f32) -> Exposure {
    Exposure::ManualEv100 {
        ev100: a.ev100() + (b.ev100() - a.ev100()) * t.clamp(0.0, 1.0),
    }
}

#[cfg(feature = "autofocus")]
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum FocusSource {
    FixedDistance { distance_m: f32 },
    TargetDistance { distance_m: f32 },
    ScreenCenterHit { distance_m: Option<f32> },
}

#[cfg(feature = "autofocus")]
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct FocusTracker {
    pub focus_distance_m: f32,
    pub max_distance_m: f32,
    pub half_life_s: f32,
    /// Focal-length correction coefficient used for focus breathing.
    pub breathing_scale: f32,
}

#[cfg(feature = "autofocus")]
impl Default for FocusTracker {
    fn default() -> Self {
        Self {
            focus_distance_m: 3.0,
            max_distance_m: 1000.0,
            half_life_s: 0.08,
            breathing_scale: 0.0,
        }
    }
}

#[cfg(feature = "autofocus")]
impl FocusTracker {
    pub fn update(&mut self, source: FocusSource, near: f32, dt: f32) -> bool {
        let distance = match source {
            FocusSource::FixedDistance { distance_m }
            | FocusSource::TargetDistance { distance_m } => distance_m,
            FocusSource::ScreenCenterHit { distance_m } => match distance_m {
                Some(value) => value,
                None => return false,
            },
        };
        if !distance.is_finite() {
            return false;
        }
        let target = distance
            .max(near + 1.0e-4)
            .min(self.max_distance_m.max(near + 1.0e-4));
        self.focus_distance_m = smooth_scalar(self.focus_distance_m, target, self.half_life_s, dt);
        true
    }

    pub fn breathing_focal_length(self, base_focal_length_mm: f32) -> f32 {
        let base = if base_focal_length_mm.is_finite() {
            base_focal_length_mm.max(1.0e-4)
        } else {
            1.0
        };
        let scale = if self.breathing_scale.is_finite() {
            self.breathing_scale
        } else {
            0.0
        };
        let reference = 3.0_f32;
        (base * (1.0 + scale * (1.0 / self.focus_distance_m.max(1.0e-4) - 1.0 / reference)))
            .max(1.0e-4)
    }
}

#[cfg(any(feature = "autofocus", feature = "occlusion_fade"))]
fn smooth_scalar(current: f32, target: f32, half_life: f32, dt: f32) -> f32 {
    if !target.is_finite() {
        return if current.is_finite() { current } else { 0.0 };
    }
    if !current.is_finite() {
        return target;
    }
    if !dt.is_finite() {
        return if dt.is_sign_positive() {
            target
        } else {
            current
        };
    }
    if half_life.is_nan() || half_life <= 0.0 || dt <= 0.0 {
        target
    } else if half_life.is_infinite() {
        current
    } else {
        target + (current - target) * libm::exp2f(-dt / half_life)
    }
}

#[cfg(feature = "shutter")]
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Shutter {
    pub shutter_angle_deg: f32,
    pub fps: f32,
    pub aperture_f_stop: f32,
}

#[cfg(feature = "shutter")]
impl Default for Shutter {
    fn default() -> Self {
        Self {
            shutter_angle_deg: 180.0,
            fps: 60.0,
            aperture_f_stop: 2.8,
        }
    }
}

#[cfg(feature = "shutter")]
impl Shutter {
    pub fn exposure_seconds(self) -> f32 {
        if self.fps.is_finite() && self.fps > 0.0 {
            self.shutter_angle_deg.clamp(0.0, 360.0) / (360.0 * self.fps)
        } else {
            1.0 / 60.0
        }
    }

    pub fn physical_exposure(self, iso: f32) -> Exposure {
        Exposure::Physical {
            f_stop: if self.aperture_f_stop.is_finite() {
                self.aperture_f_stop.max(1.0e-4)
            } else {
                Self::default().aperture_f_stop
            },
            shutter_s: self.exposure_seconds(),
            iso: if iso.is_finite() {
                iso.max(1.0e-4)
            } else {
                100.0
            },
        }
    }
}

#[cfg(feature = "virtual_production")]
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct FoveationHint {
    pub center: Vec2,
    pub inner_radius: f32,
    pub outer_radius: f32,
    pub peripheral_scale: f32,
}

#[cfg(feature = "virtual_production")]
impl Default for FoveationHint {
    fn default() -> Self {
        Self {
            center: Vec2::new(0.5, 0.5),
            inner_radius: 0.25,
            outer_radius: 0.75,
            peripheral_scale: 0.5,
        }
    }
}

#[cfg(feature = "view_lod")]
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ViewLod {
    pub max_update_hz: f32,
    pub min_resolution_scale: f32,
    pub history_frames: u32,
}

#[cfg(feature = "view_lod")]
impl Default for ViewLod {
    fn default() -> Self {
        Self {
            max_update_hz: 60.0,
            min_resolution_scale: 0.25,
            history_frames: 2,
        }
    }
}

#[cfg(feature = "view_lod")]
impl ViewLod {
    pub fn update_interval_frames(self, frame_rate: f32) -> u64 {
        if !self.max_update_hz.is_finite()
            || !frame_rate.is_finite()
            || self.max_update_hz <= 0.0
            || frame_rate <= 0.0
        {
            return 1;
        }
        (frame_rate / self.max_update_hz).ceil().max(1.0) as u64
    }

    pub fn resolution_scale(self) -> f32 {
        if self.min_resolution_scale.is_finite() {
            self.min_resolution_scale.clamp(0.0, 1.0)
        } else {
            0.0
        }
    }

    pub fn effective_history_frames(self) -> u32 {
        self.history_frames
    }

    pub fn should_update(self, frame: u64, last_frame: u64, frame_rate: f32) -> bool {
        if frame < last_frame {
            return true;
        }
        frame.saturating_sub(last_frame) >= self.update_interval_frames(frame_rate)
    }
}

#[cfg(feature = "view_lod")]
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ViewLodCandidate {
    pub key: u64,
    pub kind: ViewKind,
    pub importance: f32,
    pub screen_area: f32,
    pub last_update_frame: u64,
    pub lod: ViewLod,
}

#[cfg(feature = "view_lod")]
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ViewLodDecision {
    pub key: u64,
    pub update: bool,
    pub resolution_scale: f32,
    pub history_frames: u32,
    pub protected: bool,
}

#[cfg(feature = "view_lod")]
#[derive(Clone, Debug, Default, PartialEq)]
pub struct ViewLodScheduler {
    pub max_updates: usize,
}

#[cfg(feature = "view_lod")]
impl ViewLodScheduler {
    pub const MAX_VIEWS: usize = 256;

    pub fn schedule(
        &self,
        candidates: &[ViewLodCandidate],
        frame: u64,
        frame_rate: f32,
    ) -> Vec<ViewLodDecision> {
        let mut ranked: Vec<(usize, f32, u64)> = candidates
            .iter()
            .enumerate()
            .take(Self::MAX_VIEWS)
            .map(|(index, candidate)| {
                let due =
                    candidate
                        .lod
                        .should_update(frame, candidate.last_update_frame, frame_rate);
                let importance = if candidate.importance.is_finite() {
                    candidate.importance.max(0.0)
                } else {
                    0.0
                };
                let area = if candidate.screen_area.is_finite() {
                    candidate.screen_area.clamp(0.0, 1.0)
                } else {
                    0.0
                };
                (
                    index,
                    if due { importance * 2.0 + area } else { -1.0 },
                    candidate.key,
                )
            })
            .collect();
        ranked.sort_by(|a, b| {
            b.1.partial_cmp(&a.1)
                .unwrap_or(core::cmp::Ordering::Equal)
                .then_with(|| a.2.cmp(&b.2))
        });
        let mut selected = [false; Self::MAX_VIEWS];
        let mut updates = 0usize;
        let max_updates = if self.max_updates == 0 {
            candidates.len().min(Self::MAX_VIEWS)
        } else {
            self.max_updates.min(Self::MAX_VIEWS)
        };
        for (index, score, _) in ranked {
            let protected = matches!(
                candidates[index].kind,
                ViewKind::Main | ViewKind::StereoEye | ViewKind::Reflection
            );
            if protected || (score >= 0.0 && updates < max_updates) {
                selected[index] = true;
                if !protected {
                    updates += 1;
                }
            }
        }
        candidates
            .iter()
            .enumerate()
            .take(Self::MAX_VIEWS)
            .map(|(index, candidate)| ViewLodDecision {
                key: candidate.key,
                update: selected[index],
                resolution_scale: candidate.lod.resolution_scale(),
                history_frames: candidate.lod.effective_history_frames(),
                protected: matches!(
                    candidate.kind,
                    ViewKind::Main | ViewKind::StereoEye | ViewKind::Reflection
                ),
            })
            .collect()
    }
}

#[cfg(feature = "composition_aid")]
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CompositionGuide {
    pub thirds: bool,
    pub golden_ratio: bool,
    pub title_safe: f32,
    pub action_safe: f32,
}

#[cfg(feature = "composition_aid")]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CompositionGizmoKind {
    ThirdsVertical,
    ThirdsHorizontal,
    GoldenVertical,
    GoldenHorizontal,
    TitleSafe,
    ActionSafe,
}

#[cfg(feature = "composition_aid")]
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CompositionGizmo {
    pub kind: CompositionGizmoKind,
    pub a: Vec2,
    pub b: Vec2,
}

#[cfg(feature = "composition_aid")]
#[derive(Clone, Debug, PartialEq)]
pub struct CompositionAidResult {
    pub gizmos: Vec<CompositionGizmo>,
    pub error_count: u32,
}

#[cfg(feature = "composition_aid")]
impl CompositionGuide {
    pub fn compute(self, view: &ViewData) -> CompositionAidResult {
        let mut gizmos = Vec::with_capacity(8);
        let error_count = self.compute_into(view, &mut gizmos);
        CompositionAidResult {
            gizmos,
            error_count,
        }
    }

    /// Writes normalized guide lines into caller-owned storage and returns the
    /// number of invalid-view conditions. Reusing the vector avoids steady-
    /// state allocations in ECS/editor frame pipelines.
    pub fn compute_into(self, view: &ViewData, gizmos: &mut Vec<CompositionGizmo>) -> u32 {
        gizmos.clear();
        let safe = |value: f32, fallback: f32| {
            if value.is_finite() {
                value.clamp(0.0, 1.0)
            } else {
                fallback
            }
        };
        if self.thirds {
            gizmos.push(CompositionGizmo {
                kind: CompositionGizmoKind::ThirdsVertical,
                a: Vec2::new(1.0 / 3.0, 0.0),
                b: Vec2::new(1.0 / 3.0, 1.0),
            });
            gizmos.push(CompositionGizmo {
                kind: CompositionGizmoKind::ThirdsVertical,
                a: Vec2::new(2.0 / 3.0, 0.0),
                b: Vec2::new(2.0 / 3.0, 1.0),
            });
            gizmos.push(CompositionGizmo {
                kind: CompositionGizmoKind::ThirdsHorizontal,
                a: Vec2::new(0.0, 1.0 / 3.0),
                b: Vec2::new(1.0, 1.0 / 3.0),
            });
            gizmos.push(CompositionGizmo {
                kind: CompositionGizmoKind::ThirdsHorizontal,
                a: Vec2::new(0.0, 2.0 / 3.0),
                b: Vec2::new(1.0, 2.0 / 3.0),
            });
        }
        if self.golden_ratio {
            let phi = 0.618_033_95;
            gizmos.push(CompositionGizmo {
                kind: CompositionGizmoKind::GoldenVertical,
                a: Vec2::new(phi, 0.0),
                b: Vec2::new(phi, 1.0),
            });
            gizmos.push(CompositionGizmo {
                kind: CompositionGizmoKind::GoldenHorizontal,
                a: Vec2::new(0.0, phi),
                b: Vec2::new(1.0, phi),
            });
        }
        for (kind, value) in [
            (CompositionGizmoKind::TitleSafe, self.title_safe),
            (CompositionGizmoKind::ActionSafe, self.action_safe),
        ] {
            let inset = (1.0 - safe(value, 0.9)) * 0.5;
            gizmos.push(CompositionGizmo {
                kind,
                a: Vec2::new(inset, inset),
                b: Vec2::new(1.0 - inset, 1.0 - inset),
            });
        }
        if view.viewport_px.size[0] == 0 || view.viewport_px.size[1] == 0 {
            gizmos.clear();
            return 1;
        }
        0
    }
}

#[cfg(feature = "virtual_production")]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ExternalPoseError {
    InvalidSample,
    TimestampReversed,
    CalibrationMismatch,
    TranslationJump,
    RotationJump,
}

#[cfg(feature = "virtual_production")]
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ExternalPoseResult {
    pub pose: CameraPose,
    pub accepted: bool,
    pub error: Option<ExternalPoseError>,
}

#[cfg(feature = "virtual_production")]
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ExternalPoseSample {
    pub timestamp_s: f64,
    pub pose: CameraPose,
    pub calibration_version: u64,
}

#[cfg(feature = "virtual_production")]
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ExternalPoseValidator {
    pub calibration_version: u64,
    pub max_translation_delta_m: f64,
    pub max_rotation_delta_rad: f64,
    pub last: Option<ExternalPoseSample>,
}

#[cfg(feature = "virtual_production")]
impl Default for ExternalPoseValidator {
    fn default() -> Self {
        Self {
            calibration_version: 0,
            max_translation_delta_m: 2.0,
            max_rotation_delta_rad: 1.5,
            last: None,
        }
    }
}

#[cfg(feature = "virtual_production")]
impl ExternalPoseValidator {
    pub fn accept(&mut self, sample: ExternalPoseSample) -> Result<CameraPose, ExternalPoseError> {
        if !sample.timestamp_s.is_finite()
            || !sample.pose.position.is_finite()
            || !sample.pose.world_origin.is_finite()
            || !sample.pose.rotation.is_finite()
        {
            return Err(ExternalPoseError::InvalidSample);
        }
        if sample.calibration_version != self.calibration_version {
            return Err(ExternalPoseError::CalibrationMismatch);
        }
        if let Some(previous) = self.last {
            if sample.timestamp_s < previous.timestamp_s {
                return Err(ExternalPoseError::TimestampReversed);
            }
            if (sample.pose.position - previous.pose.position).length()
                > self.max_translation_delta_m.max(0.0)
            {
                return Err(ExternalPoseError::TranslationJump);
            }
            let dot = sample
                .pose
                .rotation
                .dot(previous.pose.rotation)
                .abs()
                .clamp(0.0, 1.0);
            let angle = 2.0 * libm::acosf(dot as f32) as f64;
            if angle > self.max_rotation_delta_rad.max(0.0) {
                return Err(ExternalPoseError::RotationJump);
            }
        }
        self.last = Some(sample);
        Ok(sample.pose)
    }

    pub fn accept_or_last(&mut self, sample: ExternalPoseSample) -> ExternalPoseResult {
        match self.accept(sample) {
            Ok(pose) => ExternalPoseResult {
                pose,
                accepted: true,
                error: None,
            },
            Err(error) => ExternalPoseResult {
                pose: self.last.map(|value| value.pose).unwrap_or_default(),
                accepted: false,
                error: Some(error),
            },
        }
    }
}

#[cfg(feature = "occlusion_fade")]
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct OcclusionSample {
    pub entity_id: u64,
    pub generation: u32,
    pub coverage: f32,
}

#[cfg(feature = "occlusion_fade")]
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CameraOcclusionIntent {
    pub entity_id: u64,
    pub generation: u32,
    pub fade: f32,
}

#[cfg(feature = "occlusion_fade")]
#[derive(Clone, Debug, PartialEq)]
pub struct OcclusionFadeState {
    pub intents: Vec<CameraOcclusionIntent>,
    pub enter_coverage: f32,
    pub exit_coverage: f32,
    pub half_life_s: f32,
}

#[cfg(feature = "occlusion_fade")]
impl Default for OcclusionFadeState {
    fn default() -> Self {
        Self {
            intents: Vec::new(),
            enter_coverage: 0.15,
            exit_coverage: 0.05,
            half_life_s: 0.08,
        }
    }
}

#[cfg(feature = "occlusion_fade")]
impl OcclusionFadeState {
    pub const MAX_INTENTS: usize = 64;

    pub fn update(&mut self, samples: &[OcclusionSample], dt: f32) {
        for sample in samples.iter().take(Self::MAX_INTENTS) {
            let covered = sample.coverage.is_finite() && sample.coverage >= self.enter_coverage;
            let visible = !sample.coverage.is_finite() || sample.coverage <= self.exit_coverage;
            let Some(index) = self.intents.iter().position(|intent| {
                intent.entity_id == sample.entity_id && intent.generation == sample.generation
            }) else {
                if self.intents.len() < Self::MAX_INTENTS {
                    self.intents.push(CameraOcclusionIntent {
                        entity_id: sample.entity_id,
                        generation: sample.generation,
                        fade: if covered { 1.0 } else { 0.0 },
                    });
                }
                continue;
            };
            let target = if covered {
                1.0
            } else if visible {
                0.0
            } else {
                self.intents[index].fade
            };
            self.intents[index].fade =
                smooth_scalar(self.intents[index].fade, target, self.half_life_s, dt)
                    .clamp(0.0, 1.0);
        }
        for intent in &mut self.intents {
            if !samples.iter().any(|sample| {
                sample.entity_id == intent.entity_id && sample.generation == intent.generation
            }) {
                intent.fade = smooth_scalar(intent.fade, 0.0, self.half_life_s, dt);
            }
        }
        self.intents.retain(|intent| intent.fade > 1.0e-3);
    }
}

#[cfg(feature = "lookahead_stream")]
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct LookaheadPredictor {
    pub horizon_s: f32,
    pub sample_step_s: f32,
    pub max_speed: f32,
    pub max_acceleration: f32,
}

#[cfg(feature = "lookahead_stream")]
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct StreamingContribution {
    pub importance: f32,
    pub distance_m: f32,
    pub sample_count: u16,
}

#[cfg(feature = "lookahead_stream")]
impl Default for LookaheadPredictor {
    fn default() -> Self {
        Self {
            horizon_s: 0.5,
            sample_step_s: 0.1,
            max_speed: 100.0,
            max_acceleration: 100.0,
        }
    }
}

#[cfg(feature = "lookahead_stream")]
impl LookaheadPredictor {
    pub const MAX_SAMPLES: usize = 16;

    pub fn predict(&self, position: DVec3, velocity: DVec3, acceleration: DVec3) -> Vec<DVec3> {
        let mut result = Vec::new();
        if !position.is_finite() || !velocity.is_finite() || !acceleration.is_finite() {
            return result;
        }
        let dt = self.sample_step_s.max(1.0e-3);
        let horizon = self.horizon_s.clamp(0.0, 0.5);
        let acceleration = clamp_dvec3(acceleration, self.max_acceleration.max(0.0) as f64);
        let velocity = clamp_dvec3(velocity, self.max_speed.max(0.0) as f64);
        let mut time = dt;
        while time <= horizon + 1.0e-6 && result.len() < Self::MAX_SAMPLES {
            result.push(
                position + velocity * time as f64 + acceleration * (0.5 * time * time) as f64,
            );
            time += dt;
        }
        result
    }

    pub fn contribution(
        &self,
        points: &[DVec3],
        current_position: DVec3,
        base_importance: f32,
        target_valid: bool,
    ) -> StreamingContribution {
        if !target_valid || !current_position.is_finite() {
            return StreamingContribution {
                importance: 0.0,
                distance_m: 0.0,
                sample_count: 0,
            };
        }
        let mut total = 0.0_f32;
        let mut distance = 0.0_f64;
        for point in points.iter().take(Self::MAX_SAMPLES) {
            if !point.is_finite() {
                continue;
            }
            let d = (*point - current_position).length();
            if d.is_finite() {
                distance = distance.max(d);
                total += 1.0 / (1.0 + d as f32);
            }
        }
        let base = if base_importance.is_finite() {
            base_importance.max(0.0)
        } else {
            0.0
        };
        StreamingContribution {
            importance: (base * total).clamp(0.0, 1.0e6),
            distance_m: distance as f32,
            sample_count: points.len().min(Self::MAX_SAMPLES) as u16,
        }
    }
}

#[cfg(feature = "lookahead_stream")]
fn clamp_dvec3(value: DVec3, limit: f64) -> DVec3 {
    let length = value.length();
    if !length.is_finite() || length <= limit || length <= 1.0e-12 {
        value
    } else {
        value * (limit / length)
    }
}

#[cfg(feature = "edge_stabilize")]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum HistoryRejectReason {
    InvalidHistory,
    LinearVelocity,
    AngularVelocity,
}

#[cfg(feature = "edge_stabilize")]
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct HistoryStabilizer {
    pub max_linear_velocity: f32,
    pub max_angular_velocity: f32,
    pub clamp_weight: f32,
    pub sharpen_strength: f32,
}

#[cfg(feature = "edge_stabilize")]
impl Default for HistoryStabilizer {
    fn default() -> Self {
        Self {
            max_linear_velocity: 50.0,
            max_angular_velocity: 8.0,
            clamp_weight: 0.9,
            sharpen_strength: 0.0,
        }
    }
}

#[cfg(feature = "edge_stabilize")]
impl HistoryStabilizer {
    pub fn history_weight(
        self,
        history_valid: bool,
        linear_velocity: f32,
        angular_velocity: f32,
    ) -> Result<f32, HistoryRejectReason> {
        if !history_valid {
            return Err(HistoryRejectReason::InvalidHistory);
        }
        if !linear_velocity.is_finite() || linear_velocity.abs() > self.max_linear_velocity.max(0.0)
        {
            return Err(HistoryRejectReason::LinearVelocity);
        }
        if !angular_velocity.is_finite()
            || angular_velocity.abs() > self.max_angular_velocity.max(0.0)
        {
            return Err(HistoryRejectReason::AngularVelocity);
        }
        Ok(self.clamp_weight.clamp(0.0, 1.0))
    }

    pub fn evaluate_batch(self, samples: &[(bool, f32, f32)]) -> (Vec<f32>, u32, u32, f32) {
        let mut weights = Vec::with_capacity(samples.len());
        let mut accepted = 0_u32;
        let mut rejected = 0_u32;
        for &(valid, linear, angular) in samples {
            match self.history_weight(valid, linear, angular) {
                Ok(weight) => {
                    accepted = accepted.saturating_add(1);
                    weights.push(weight);
                }
                Err(_) => {
                    rejected = rejected.saturating_add(1);
                    weights.push(0.0);
                }
            }
        }
        let acceptance = if samples.is_empty() {
            0.0
        } else {
            accepted as f32 / samples.len() as f32
        };
        (weights, accepted, rejected, acceptance)
    }

    pub fn sharpen_strength(self) -> f32 {
        if self.sharpen_strength.is_finite() {
            self.sharpen_strength.clamp(0.0, 1.0)
        } else {
            0.0
        }
    }
}

#[cfg(feature = "composition_aid")]
impl Default for CompositionGuide {
    fn default() -> Self {
        Self {
            thirds: true,
            golden_ratio: false,
            title_safe: 0.9,
            action_safe: 0.95,
        }
    }
}

#[cfg(feature = "determinism")]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct CameraDeterminismDigest {
    pub bytes: [u8; 32],
}

#[cfg(feature = "determinism")]
impl CameraDeterminismDigest {
    pub fn from_view(view: &ViewData, frame: u64) -> Self {
        let mut lanes = [0xcbf29ce484222325_u64; 4];
        let mut feed = |value: u64| {
            for (index, lane) in lanes.iter_mut().enumerate() {
                *lane ^= value.rotate_left((index * 13) as u32);
                *lane = lane.wrapping_mul(0x100000001b3);
            }
        };
        feed(frame);
        feed(view.handle.index as u64);
        feed(view.handle.generation as u64);
        feed(view.jitter.x.to_bits() as u64 | ((view.jitter.y.to_bits() as u64) << 32));
        feed(view.teleport as u64 | ((view.history_valid as u64) << 1));
        feed(view.world_position.x.to_bits());
        feed(view.world_position.y.to_bits());
        feed(view.world_position.z.to_bits());
        let mut bytes = [0_u8; 32];
        for (index, lane) in lanes.into_iter().enumerate() {
            bytes[index * 8..index * 8 + 8].copy_from_slice(&lane.to_le_bytes());
        }
        Self { bytes }
    }
}

#[cfg(feature = "physical_lens")]
#[allow(dead_code)]
fn _keep_lens_type_visible(_: Lens) {}

#[cfg(test)]
mod tests {
    #[cfg(any(
        feature = "cutscene",
        feature = "determinism",
        feature = "virtual_production",
        feature = "occlusion_fade",
        all(feature = "lookahead_stream", feature = "edge_stabilize")
    ))]
    use super::*;
    #[cfg(feature = "determinism")]
    use crate::{derive_view, CameraIntent, CameraPose, ViewHistory};
    #[cfg(feature = "cutscene")]
    use alloc::vec;
    #[cfg(any(
        feature = "cutscene",
        feature = "determinism",
        feature = "virtual_production",
        all(feature = "lookahead_stream", feature = "edge_stabilize")
    ))]
    use prism_math::DVec3;

    #[cfg(feature = "cutscene")]
    #[test]
    fn track_rejects_unsorted_keyframes_and_interpolates() {
        let mut track = CameraTrack::default();
        let base = CameraKeyframe {
            tick: 10,
            pose: CameraPose::default(),
            focal_length_mm: 50.0,
            exposure: Exposure::default(),
            event_key: 1,
        };
        assert!(track.push(base));
        assert!(!track.push(CameraKeyframe { tick: 9, ..base }));
        assert!(track.push(CameraKeyframe {
            tick: 20,
            pose: CameraPose {
                position: DVec3::new(10.0, 0.0, 0.0),
                ..base.pose
            },
            ..base
        }));
        assert!((track.sample(15, CameraPose::default()).0.position.x - 5.0).abs() < 1.0e-6);
        let sampled = track.sample_state(
            15,
            CameraTrackSample {
                pose: CameraPose::default(),
                focal_length_mm: 35.0,
                exposure: Exposure::default(),
                event_key: None,
                active: true,
                handoff_to_previous: false,
            },
        );
        assert!((sampled.focal_length_mm - 50.0).abs() < 1.0e-6);
        assert_eq!(sampled.event_key, Some(1));
        assert_eq!(track.validate(), Ok(()));
        track.asset_version += 1;
        assert_eq!(
            track.validate().unwrap_err().code,
            CameraTrackValidationCode::UnsupportedVersion
        );
    }

    #[cfg(feature = "cutscene")]
    #[test]
    fn track_end_policy_is_observable() {
        let mut track = CameraTrack {
            end_policy: CutsceneEndPolicy::Deactivate,
            ..Default::default()
        };
        assert!(track.push(CameraKeyframe {
            tick: 1,
            pose: CameraPose::default(),
            focal_length_mm: 50.0,
            exposure: Exposure::default(),
            event_key: 1,
        }));
        let fallback = CameraTrackSample {
            pose: CameraPose::default(),
            focal_length_mm: 35.0,
            exposure: Exposure::default(),
            event_key: None,
            active: true,
            handoff_to_previous: false,
        };
        assert!(!track.sample_state(2, fallback).active);
        track.end_policy = CutsceneEndPolicy::BlendToPrevious;
        assert!(track.sample_state(2, fallback).handoff_to_previous);
    }

    #[cfg(feature = "cutscene")]
    #[test]
    fn event_cursor_consumes_crossed_keys_once_in_authored_order() {
        let mut track = CameraTrack::default();
        for (tick, event_key) in [(2, 20), (5, 50), (9, 90)] {
            assert!(track.push(CameraKeyframe {
                tick,
                pose: CameraPose::default(),
                focal_length_mm: 50.0,
                exposure: Exposure::default(),
                event_key,
            }));
        }
        let track = track.into_validated().unwrap();
        let mut cursor = CameraTrackEventCursor::default();
        let mut output = vec![999];

        cursor.advance(&track, 1, &mut output).unwrap();
        assert!(output.is_empty());
        cursor.advance(&track, 5, &mut output).unwrap();
        assert_eq!(
            output,
            [
                CameraTrackEventOccurrence {
                    tick: 2,
                    event_key: 20,
                },
                CameraTrackEventOccurrence {
                    tick: 5,
                    event_key: 50,
                },
            ]
        );
        let capacity = output.capacity();
        cursor.advance(&track, 5, &mut output).unwrap();
        assert!(output.is_empty());
        assert!(output.capacity() >= capacity);
        cursor.advance(&track, 12, &mut output).unwrap();
        assert_eq!(
            output,
            [CameraTrackEventOccurrence {
                tick: 9,
                event_key: 90,
            }]
        );
    }

    #[cfg(feature = "cutscene")]
    #[test]
    fn event_cursor_rejects_reverse_time_without_mutating_state_or_output() {
        let mut track = CameraTrack::default();
        assert!(track.push(CameraKeyframe {
            tick: 4,
            pose: CameraPose::default(),
            focal_length_mm: 50.0,
            exposure: Exposure::default(),
            event_key: 44,
        }));
        let track = track.into_validated().unwrap();
        let mut cursor = CameraTrackEventCursor::default();
        let mut output = Vec::new();
        cursor.advance(&track, 8, &mut output).unwrap();
        output.push(7);

        assert_eq!(
            cursor.advance(&track, 7, &mut output),
            Err(CameraTrackEventError::TickReversed {
                previous: 8,
                requested: 7,
            })
        );
        assert_eq!(cursor.last_tick(), Some(8));
        assert_eq!(output, [7]);
    }

    #[cfg(feature = "determinism")]
    #[test]
    fn digest_changes_when_view_changes() {
        let mut history = ViewHistory::default();
        let first = derive_view(
            CameraIntent::default(),
            CameraPose::default(),
            0,
            [64, 64],
            &mut history,
        )
        .unwrap();
        let second = derive_view(
            CameraIntent::default(),
            CameraPose {
                position: DVec3::new(1.0, 0.0, 0.0),
                ..Default::default()
            },
            1,
            [64, 64],
            &mut history,
        )
        .unwrap();
        assert_ne!(
            CameraDeterminismDigest::from_view(&first, 0),
            CameraDeterminismDigest::from_view(&second, 1)
        );
    }

    #[cfg(feature = "virtual_production")]
    #[test]
    fn external_pose_validator_rejects_stale_and_large_jumps() {
        let mut validator = ExternalPoseValidator::default();
        let sample = ExternalPoseSample {
            timestamp_s: 1.0,
            pose: CameraPose::default(),
            calibration_version: 0,
        };
        assert!(validator.accept(sample).is_ok());
        assert_eq!(
            validator.accept(ExternalPoseSample {
                timestamp_s: 0.5,
                ..sample
            }),
            Err(ExternalPoseError::TimestampReversed)
        );
        assert_eq!(
            validator.accept(ExternalPoseSample {
                timestamp_s: 2.0,
                pose: CameraPose {
                    position: DVec3::new(3.0, 0.0, 0.0),
                    ..sample.pose
                },
                ..sample
            }),
            Err(ExternalPoseError::TranslationJump)
        );
        let rejected = validator.accept_or_last(ExternalPoseSample {
            timestamp_s: 2.0,
            pose: CameraPose {
                position: DVec3::new(4.0, 0.0, 0.0),
                ..sample.pose
            },
            ..sample
        });
        assert!(!rejected.accepted);
        assert_eq!(rejected.pose, sample.pose);
    }

    #[cfg(feature = "occlusion_fade")]
    #[test]
    fn occlusion_fade_uses_capacity_and_hysteresis() {
        let mut state = OcclusionFadeState::default();
        state.update(
            &[OcclusionSample {
                entity_id: 1,
                generation: 2,
                coverage: 1.0,
            }],
            0.1,
        );
        assert_eq!(state.intents.len(), 1);
        assert!(state.intents[0].fade > 0.0);
        state.update(
            &[OcclusionSample {
                entity_id: 1,
                generation: 2,
                coverage: 0.0,
            }],
            1.0,
        );
        assert!(state.intents.is_empty());
        state.update(
            &[OcclusionSample {
                entity_id: 1,
                generation: 2,
                coverage: 1.0,
            }],
            0.1,
        );
        state.update(&[], 1.0);
        assert!(state.intents.is_empty());
    }

    #[cfg(all(feature = "lookahead_stream", feature = "edge_stabilize"))]
    #[test]
    fn lookahead_is_bounded_and_history_rejects_fast_motion() {
        let points = LookaheadPredictor::default().predict(
            DVec3::ZERO,
            DVec3::new(1000.0, 0.0, 0.0),
            DVec3::new(1000.0, 0.0, 0.0),
        );
        assert!(!points.is_empty());
        assert!(points.len() <= LookaheadPredictor::MAX_SAMPLES);
        assert_eq!(
            HistoryStabilizer::default().history_weight(true, 100.0, 0.0),
            Err(HistoryRejectReason::LinearVelocity)
        );
    }

    #[cfg(feature = "occlusion_fade")]
    #[test]
    fn occlusion_invalid_samples_release_existing_fade() {
        let mut state = OcclusionFadeState::default();
        state.update(
            &[OcclusionSample {
                entity_id: 9,
                generation: 1,
                coverage: 1.0,
            }],
            0.0,
        );
        assert_eq!(state.intents[0].fade, 1.0);
        state.update(
            &[OcclusionSample {
                entity_id: 9,
                generation: 1,
                coverage: f32::NAN,
            }],
            1.0,
        );
        assert!(state.intents.is_empty());
    }

    #[cfg(feature = "view_lod")]
    #[test]
    fn view_lod_handles_invalid_rate_and_frame_rewind() {
        let policy = ViewLod {
            max_update_hz: f32::NAN,
            min_resolution_scale: 4.0,
            history_frames: 0,
        };
        assert_eq!(policy.update_interval_frames(60.0), 1);
        assert!(policy.should_update(2, 10, 60.0));
        assert_eq!(policy.resolution_scale(), 1.0);
        assert_eq!(policy.effective_history_frames(), 0);
    }

    #[cfg(feature = "composition_aid")]
    #[test]
    fn composition_guides_are_normalized_and_hide_for_zero_viewport() {
        let guide = CompositionGuide {
            thirds: true,
            golden_ratio: true,
            title_safe: f32::NAN,
            action_safe: 0.8,
        };
        let mut history = ViewHistory::default();
        let valid = derive_view(
            CameraIntent::default(),
            CameraPose::default(),
            0,
            [100, 50],
            &mut history,
        )
        .unwrap();
        let result = guide.compute(&valid);
        assert_eq!(result.gizmos.len(), 8);
        assert!(result.gizmos.iter().all(|gizmo| {
            gizmo.a.x.is_finite()
                && gizmo.a.y.is_finite()
                && gizmo.b.x.is_finite()
                && gizmo.b.y.is_finite()
        }));
        let mut zero = valid;
        zero.viewport_px.size = [0, 0];
        let result = guide.compute(&zero);
        assert!(result.gizmos.is_empty());
        assert_eq!(result.error_count, 1);
    }
}
