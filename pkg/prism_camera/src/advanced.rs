#[cfg(any(
    feature = "cutscene",
    feature = "occlusion_fade",
    feature = "lookahead_stream"
))]
use alloc::vec::Vec;

#[cfg(any(
    feature = "cutscene",
    feature = "virtual_production",
    feature = "view_batch",
    feature = "lookahead_stream",
    feature = "determinism"
))]
use prism_math::DVec3;
#[cfg(any(feature = "virtual_production", feature = "view_batch"))]
use prism_math::Vec2;

#[cfg(any(
    feature = "cutscene",
    feature = "virtual_production",
    feature = "determinism"
))]
use crate::CameraPose;
#[cfg(feature = "cutscene")]
use crate::Exposure;
#[cfg(feature = "physical_lens")]
use crate::Lens;
#[cfg(any(feature = "view_batch", feature = "determinism"))]
use crate::ViewData;
#[cfg(feature = "view_batch")]
use crate::{RenderTarget, ViewKind};

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
#[derive(Clone, Debug, Default, PartialEq)]
pub struct CameraTrack {
    pub keyframes: Vec<CameraKeyframe>,
    pub end_policy: CutsceneEndPolicy,
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
    pub const MAX_KEYFRAMES: usize = 4096;

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
        let Some(first) = self.keyframes.first().copied() else {
            return (fallback, None);
        };
        if tick <= first.tick {
            return (first.pose, Some(first.event_key));
        }
        let Some(last) = self.keyframes.last().copied() else {
            return (fallback, None);
        };
        if tick >= last.tick {
            return match self.end_policy {
                CutsceneEndPolicy::Hold => (last.pose, Some(last.event_key)),
                CutsceneEndPolicy::BlendToPrevious | CutsceneEndPolicy::Deactivate => {
                    (fallback, None)
                }
            };
        }
        let index = self.keyframes.partition_point(|frame| frame.tick <= tick);
        let a = self.keyframes[index - 1];
        let b = self.keyframes[index];
        let linear_t = (tick - a.tick) as f64 / (b.tick - a.tick) as f64;
        let t = linear_t * linear_t * (3.0 - 2.0 * linear_t);
        (
            CameraPose {
                position: a.pose.position.lerp(b.pose.position, t),
                rotation: a.pose.rotation.slerp(b.pose.rotation, t),
                world_origin: a.pose.world_origin.lerp(b.pose.world_origin, t),
            },
            Some(if t < 0.5 { a.event_key } else { b.event_key }),
        )
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
        let linear_t = (tick - a.tick) as f32 / (b.tick - a.tick) as f32;
        let t = linear_t * linear_t * (3.0 - 2.0 * linear_t);
        CameraTrackSample {
            pose: CameraPose {
                position: a.pose.position.lerp(b.pose.position, t as f64),
                rotation: a.pose.rotation.slerp(b.pose.rotation, t as f64),
                world_origin: a.pose.world_origin.lerp(b.pose.world_origin, t as f64),
            },
            focal_length_mm: a.focal_length_mm + (b.focal_length_mm - a.focal_length_mm) * t,
            exposure: lerp_exposure(a.exposure, b.exposure, t),
            event_key: Some(if t < 0.5 { a.event_key } else { b.event_key }),
            active: true,
            handoff_to_previous: false,
        }
    }
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
}

#[cfg(feature = "autofocus")]
impl Default for FocusTracker {
    fn default() -> Self {
        Self {
            focus_distance_m: 3.0,
            max_distance_m: 1000.0,
            half_life_s: 0.08,
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
}

#[cfg(feature = "shutter")]
impl Default for Shutter {
    fn default() -> Self {
        Self {
            shutter_angle_deg: 180.0,
            fps: 60.0,
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

#[cfg(feature = "view_batch")]
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct ViewBatchKey {
    pub target: RenderTarget,
    pub kind: ViewKind,
    pub origin_bits: [u64; 3],
    pub policy_hash: u64,
}

#[cfg(feature = "view_batch")]
impl ViewBatchKey {
    pub fn from_view(view: &ViewData, policy_hash: u64) -> Self {
        Self {
            target: view.target,
            kind: view.kind,
            origin_bits: [
                view.world_origin.x.to_bits(),
                view.world_origin.y.to_bits(),
                view.world_origin.z.to_bits(),
            ],
            policy_hash,
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

#[cfg(feature = "composition_aid")]
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CompositionGuide {
    pub thirds: bool,
    pub golden_ratio: bool,
    pub title_safe: f32,
    pub action_safe: f32,
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
}

#[cfg(feature = "edge_stabilize")]
impl Default for HistoryStabilizer {
    fn default() -> Self {
        Self {
            max_linear_velocity: 50.0,
            max_angular_velocity: 8.0,
            clamp_weight: 0.9,
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
    use super::*;
    use crate::{derive_view, CameraIntent, ViewHistory};
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
}
