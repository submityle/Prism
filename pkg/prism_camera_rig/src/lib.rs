//! Allocation-free camera rig and director primitives.
#![cfg_attr(not(test), no_std)]
#![forbid(unsafe_code)]

extern crate alloc;

use alloc::vec::Vec;
use prism_camera::{CameraPose, Lens};
use prism_math::{DQuat, DVec3, Quat, Vec2, Vec3};

/// A camera candidate with a stable identity used for deterministic selection.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct VirtualCamera {
    pub key: u64,
    pub priority: i32,
    pub enabled: bool,
    pub pose: CameraPose,
    pub lens: Option<Lens>,
    pub weight: f32,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum BodyRig {
    Fixed {
        position: DVec3,
    },
    Follow {
        target: DVec3,
        offset: DVec3,
    },
    Orbit {
        target: DVec3,
        radius: f32,
        yaw: f32,
        pitch: f32,
    },
}

impl Default for BodyRig {
    fn default() -> Self {
        Self::Fixed {
            position: DVec3::ZERO,
        }
    }
}

impl BodyRig {
    pub fn position(self, fallback: DVec3) -> DVec3 {
        let position = match self {
            Self::Fixed { position } => position,
            Self::Follow { target, offset } => target + offset,
            Self::Orbit {
                target,
                radius,
                yaw,
                pitch,
            } => {
                let rotation =
                    DQuat::from_rotation_y(yaw as f64) * DQuat::from_rotation_x(pitch as f64);
                target + rotation.mul_vec3(DVec3::new(0.0, 0.0, radius.max(0.0) as f64))
            }
        };
        position.is_finite().then_some(position).unwrap_or(fallback)
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum AimRig {
    Fixed { rotation: DQuat },
    LookAt { target: DVec3 },
}

impl Default for AimRig {
    fn default() -> Self {
        Self::Fixed {
            rotation: DQuat::IDENTITY,
        }
    }
}

impl AimRig {
    pub fn rotation(self, position: DVec3, fallback: DQuat) -> DQuat {
        let rotation = match self {
            Self::Fixed { rotation } => rotation.normalize(),
            Self::LookAt { target } => {
                let direction = (target - position).normalize_or_zero();
                if direction.length_squared() <= 1.0e-12 {
                    return fallback;
                }
                let yaw = libm::atan2(-direction.x, -direction.z);
                let pitch = libm::atan2(
                    direction.y,
                    (direction.x * direction.x + direction.z * direction.z).sqrt(),
                );
                (DQuat::from_rotation_y(yaw) * DQuat::from_rotation_x(-pitch)).normalize()
            }
        };
        rotation.is_finite().then_some(rotation).unwrap_or(fallback)
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RigOutput {
    pub pose: CameraPose,
    pub spring_arm_status: Option<SpringArmStatus>,
}

pub fn compose_rig(body: BodyRig, aim: AimRig, origin: DVec3, fallback: CameraPose) -> RigOutput {
    let position = body.position(fallback.position);
    let rotation = aim.rotation(position, fallback.rotation);
    RigOutput {
        pose: CameraPose {
            position,
            rotation,
            world_origin: origin,
        },
        spring_arm_status: None,
    }
}

impl Default for VirtualCamera {
    fn default() -> Self {
        Self {
            key: 0,
            priority: 0,
            enabled: true,
            pose: CameraPose::default(),
            lens: None,
            weight: 1.0,
        }
    }
}

/// Blend policy for two virtual-camera outputs.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Blend {
    pub duration_s: f32,
    pub elapsed_s: f32,
}

impl Blend {
    pub const fn new(duration_s: f32) -> Self {
        Self {
            duration_s,
            elapsed_s: 0.0,
        }
    }
    pub fn advance(self, dt: f32) -> Self {
        Self {
            elapsed_s: (self.elapsed_s + dt.max(0.0)).min(self.duration_s.max(0.0)),
            ..self
        }
    }
    pub fn amount(self) -> f32 {
        if self.duration_s <= 0.0 {
            1.0
        } else {
            (self.elapsed_s / self.duration_s).clamp(0.0, 1.0)
        }
    }
    pub fn finished(self) -> bool {
        self.amount() >= 1.0
    }
}

/// Stateful director. Candidate order is priority descending, then key ascending.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct CameraBrain {
    pub active_key: Option<u64>,
    pub blend: Option<Blend>,
    pub cut: bool,
    pub blend_duration_s: f32,
    from_pose: Option<CameraPose>,
    current_pose: Option<CameraPose>,
}

impl CameraBrain {
    pub fn choose(cameras: &[VirtualCamera]) -> Option<VirtualCamera> {
        cameras
            .iter()
            .copied()
            .filter(|camera| camera.enabled && camera.weight > 0.0)
            .max_by(|a, b| a.priority.cmp(&b.priority).then_with(|| b.key.cmp(&a.key)))
    }

    pub fn update(
        &mut self,
        candidates: &[VirtualCamera],
        dt: f32,
        default: CameraPose,
    ) -> CameraPose {
        let Some(next) = Self::choose(candidates) else {
            self.cut = false;
            self.active_key = None;
            return default;
        };
        let changed = self.active_key != Some(next.key);
        if changed {
            self.from_pose = self.current_pose;
            self.active_key = Some(next.key);
            self.cut = self.from_pose.is_none() || self.blend_duration_s <= 0.0;
            self.blend = (!self.cut).then(|| Blend::new(self.blend_duration_s));
        } else {
            self.cut = false;
        }
        let target = next.pose;
        let output = if let Some(blend) = self.blend.map(|b| b.advance(dt)) {
            let amount = blend.amount();
            self.blend = (!blend.finished()).then_some(blend);
            if let Some(from) = self.from_pose {
                blend_pose(from, target, amount)
            } else {
                target
            }
        } else {
            target
        };
        self.current_pose = Some(output);
        if self.blend.is_none() {
            self.from_pose = None;
        }
        output
    }
}

fn blend_pose(from: CameraPose, to: CameraPose, amount: f32) -> CameraPose {
    let t = amount.clamp(0.0, 1.0) as f64;
    CameraPose {
        position: from.position.lerp(to.position, t),
        rotation: from.rotation.slerp(to.rotation, t),
        world_origin: from.world_origin.lerp(to.world_origin, t),
    }
}

/// A bounded sphere cast request for a spring arm.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CastRequest {
    pub origin: Vec3,
    pub direction: Vec3,
    pub distance: f32,
    pub radius: f32,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CastHit {
    pub hit: bool,
    pub distance: f32,
    pub normal: Vec3,
}

/// Batch query owner. Physics remains authoritative outside this crate.
pub trait SceneQuery {
    fn cast_batch(&self, requests: &[CastRequest], hits: &mut [CastHit]);
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum SpringArmStatus {
    Clear,
    Hit,
    Unavailable,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SpringArm {
    pub arm_length: f32,
    pub probe_radius: f32,
    pub lag_half_life: f32,
    pub collision_margin: f32,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct NoiseAxes {
    pub amplitude: Vec3,
    pub frequency: Vec3,
}

impl Default for NoiseAxes {
    fn default() -> Self {
        Self {
            amplitude: Vec3::ZERO,
            frequency: Vec3::ONE,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CameraNoise {
    pub position: NoiseAxes,
    pub rotation: NoiseAxes,
    pub gain: f32,
    pub seed: u64,
    pub temporal_clamp: f32,
}

impl Default for CameraNoise {
    fn default() -> Self {
        Self {
            position: NoiseAxes::default(),
            rotation: NoiseAxes::default(),
            gain: 0.0,
            seed: 0,
            temporal_clamp: 1.0,
        }
    }
}

impl CameraNoise {
    pub fn sample(self, pose: CameraPose, time_s: f32) -> CameraPose {
        let g = self.gain.max(0.0);
        let sample = |axis: usize, axes: NoiseAxes| {
            let amplitude = [axes.amplitude.x, axes.amplitude.y, axes.amplitude.z][axis];
            let frequency = [axes.frequency.x, axes.frequency.y, axes.frequency.z][axis];
            let phase = self.seed as f32 * 0.000_001 + axis as f32 * 1.73;
            let raw = gradient_noise(time_s * frequency.max(0.0) + phase);
            let limit = self.temporal_clamp.max(0.0);
            raw * amplitude * g.min(limit)
        };
        let translation = DVec3::new(
            sample(0, self.position) as f64,
            sample(1, self.position) as f64,
            sample(2, self.position) as f64,
        );
        let rotation = DQuat::from_rotation_x(sample(0, self.rotation) as f64)
            * DQuat::from_rotation_y(sample(1, self.rotation) as f64)
            * DQuat::from_rotation_z(sample(2, self.rotation) as f64);
        CameraPose {
            position: pose.position + pose.rotation.mul_vec3(translation),
            rotation: (pose.rotation * rotation).normalize(),
            ..pose
        }
    }
}

fn gradient_noise(value: f32) -> f32 {
    if !value.is_finite() {
        return 0.0;
    }
    let base = libm::floorf(value);
    let local = value - base;
    let fade = local * local * (3.0 - 2.0 * local);
    let lattice = |cell: f32| {
        let bits = (cell as i32 as u32).wrapping_mul(0x9e37_79b9);
        let hashed = bits ^ bits.rotate_left(13) ^ 0x85eb_ca6b;
        if hashed & 1 == 0 {
            1.0
        } else {
            -1.0
        }
    };
    let left = lattice(base) * local;
    let right = lattice(base + 1.0) * (local - 1.0);
    let first = left + (right - left) * fade;
    let octave = value * 2.03 + 11.0;
    let octave_base = libm::floorf(octave);
    let octave_local = octave - octave_base;
    let octave_fade = octave_local * octave_local * (3.0 - 2.0 * octave_local);
    let octave_left = lattice(octave_base) * octave_local;
    let octave_right = lattice(octave_base + 1.0) * (octave_local - 1.0);
    (first + (octave_left + (octave_right - octave_left) * octave_fade) * 0.5) / 1.5
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ImpulseSource {
    pub id: u64,
    pub position: Vec3,
    pub velocity: Vec3,
    pub range: f32,
    pub attack_s: f32,
    pub sustain_s: f32,
    pub decay_s: f32,
}

#[derive(Clone, Debug, PartialEq)]
pub struct ImpulseBuffer {
    pub entries: Vec<ImpulseEntry>,
    pub max_translation: f32,
    pub max_rotation: f32,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ImpulseEntry {
    pub source: ImpulseSource,
    pub started_at_s: f32,
}

impl Default for ImpulseBuffer {
    fn default() -> Self {
        Self {
            entries: Vec::new(),
            max_translation: 1.0,
            max_rotation: 1.0,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ImpulseResponse {
    pub translation: Vec3,
    pub rotation: Vec3,
}

impl ImpulseSource {
    pub fn response(self, camera_position: Vec3, time_s: f32) -> ImpulseResponse {
        let attack = self.attack_s.max(0.0);
        let sustain = self.sustain_s.max(0.0);
        let decay = self.decay_s.max(0.0);
        if !self.position.is_finite()
            || !self.velocity.is_finite()
            || !camera_position.is_finite()
            || !self.range.is_finite()
            || !time_s.is_finite()
        {
            return ImpulseResponse {
                translation: Vec3::ZERO,
                rotation: Vec3::ZERO,
            };
        }
        let distance = (self.position - camera_position).length();
        if distance > self.range.max(0.0) || self.range <= 0.0 || time_s < 0.0 {
            return ImpulseResponse {
                translation: Vec3::ZERO,
                rotation: Vec3::ZERO,
            };
        }
        let lifetime = attack + sustain + decay;
        if time_s > lifetime {
            return ImpulseResponse {
                translation: Vec3::ZERO,
                rotation: Vec3::ZERO,
            };
        }
        let envelope = if attack > 0.0 && time_s < attack {
            time_s / attack
        } else if time_s < attack + sustain {
            1.0
        } else if decay > 0.0 {
            1.0 - ((time_s - attack - sustain) / decay).clamp(0.0, 1.0)
        } else {
            0.0
        };
        let falloff = 1.0 - (distance / self.range).clamp(0.0, 1.0);
        let signal = self.velocity * (envelope * falloff);
        ImpulseResponse {
            translation: signal,
            rotation: signal * 0.1,
        }
    }
}

impl ImpulseBuffer {
    pub const MAX_SOURCES: usize = 64;

    pub fn submit(&mut self, source: ImpulseSource, now_s: f32) -> bool {
        self.prune(now_s);
        if !now_s.is_finite()
            || self
                .entries
                .iter()
                .any(|existing| existing.source.id == source.id)
        {
            return false;
        }
        if self.entries.len() >= Self::MAX_SOURCES {
            return false;
        }
        self.entries.push(ImpulseEntry {
            source,
            started_at_s: now_s,
        });
        true
    }

    pub fn prune(&mut self, time_s: f32) {
        self.entries.retain(|entry| {
            let lifetime = entry.source.attack_s.max(0.0)
                + entry.source.sustain_s.max(0.0)
                + entry.source.decay_s.max(0.0);
            time_s.is_finite() && time_s - entry.started_at_s <= lifetime
        });
    }

    pub fn sample(&self, camera_position: Vec3, time_s: f32) -> ImpulseResponse {
        let mut response = ImpulseResponse {
            translation: Vec3::ZERO,
            rotation: Vec3::ZERO,
        };
        for entry in &self.entries {
            let contribution = entry
                .source
                .response(camera_position, time_s - entry.started_at_s);
            response.translation += contribution.translation;
            response.rotation += contribution.rotation;
        }
        response.translation = clamp_vector(response.translation, self.max_translation);
        response.rotation = clamp_vector(response.rotation, self.max_rotation);
        response
    }
}

fn clamp_vector(value: Vec3, limit: f32) -> Vec3 {
    let limit = limit.max(0.0);
    let length = value.length();
    if !length.is_finite() || length <= limit || length <= 1.0e-6 {
        if value.is_finite() {
            value
        } else {
            Vec3::ZERO
        }
    } else {
        value * (limit / length)
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct CameraCommand {
    pub move_axis: Vec3,
    pub look_axis: Vec2,
    pub zoom: f32,
    pub boost: bool,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CameraInputMap {
    pub move_deadzone: f32,
    pub look_sensitivity: f32,
    pub zoom_sensitivity: f32,
    pub boost_multiplier: f32,
}

impl Default for CameraInputMap {
    fn default() -> Self {
        Self {
            move_deadzone: 0.05,
            look_sensitivity: 1.0,
            zoom_sensitivity: 1.0,
            boost_multiplier: 4.0,
        }
    }
}

impl CameraInputMap {
    pub fn normalize(self, mut command: CameraCommand) -> CameraCommand {
        if !command.move_axis.is_finite() {
            command.move_axis = Vec3::ZERO;
        }
        if !command.look_axis.is_finite() {
            command.look_axis = Vec2::ZERO;
        }
        if !command.zoom.is_finite() {
            command.zoom = 0.0;
        }
        let deadzone = if self.move_deadzone.is_finite() {
            self.move_deadzone.clamp(0.0, 0.999_999)
        } else {
            0.0
        };
        let len = command.move_axis.length();
        if len <= deadzone {
            command.move_axis = Vec3::ZERO;
        } else {
            command.move_axis = command.move_axis.normalize()
                * ((len - deadzone) / (1.0 - deadzone).max(1.0e-6)).clamp(0.0, 1.0);
        }
        let look_sensitivity = if self.look_sensitivity.is_finite() {
            self.look_sensitivity.max(0.0)
        } else {
            0.0
        };
        let zoom_sensitivity = if self.zoom_sensitivity.is_finite() {
            self.zoom_sensitivity.max(0.0)
        } else {
            0.0
        };
        command.look_axis *= look_sensitivity;
        command.zoom *= zoom_sensitivity;
        if command.boost {
            let boost = if self.boost_multiplier.is_finite() {
                self.boost_multiplier.max(1.0)
            } else {
                1.0
            };
            command.move_axis *= boost;
        }
        command
    }
}

#[derive(Clone, Debug, Default)]
pub struct ViewUpdateScheduler {
    pub last_frames: Vec<u64>,
}

impl ViewUpdateScheduler {
    pub fn should_update(
        &mut self,
        index: usize,
        policy: prism_camera::ViewLod,
        frame: u64,
        frame_rate: f32,
    ) -> bool {
        if index >= self.last_frames.len() {
            self.last_frames.resize(index + 1, u64::MAX);
        }
        let last = self.last_frames[index];
        let update = last == u64::MAX || policy.should_update(frame, last, frame_rate);
        if update {
            self.last_frames[index] = frame;
        }
        update
    }
}

impl Default for SpringArm {
    fn default() -> Self {
        Self {
            arm_length: 4.0,
            probe_radius: 0.2,
            lag_half_life: 0.0,
            collision_margin: 0.05,
        }
    }
}

impl SpringArm {
    pub fn resolve<Q: SceneQuery>(
        &self,
        query: Option<&Q>,
        origin: Vec3,
        direction: Vec3,
        previous_distance: f32,
        dt: f32,
    ) -> (f32, SpringArmStatus) {
        let arm_length = self.arm_length.max(0.0);
        let direction = direction.normalize_or_zero();
        let Some(query) = query else {
            return (arm_length, SpringArmStatus::Unavailable);
        };
        let request = CastRequest {
            origin,
            direction,
            distance: arm_length,
            radius: self.probe_radius.max(0.0),
        };
        let mut hit = [CastHit {
            hit: false,
            distance: arm_length,
            normal: Vec3::Y,
        }];
        query.cast_batch(&[request], &mut hit);
        if !hit[0].distance.is_finite() || hit[0].distance < 0.0 {
            return (arm_length, SpringArmStatus::Unavailable);
        }
        let target = if hit[0].hit {
            (hit[0].distance - self.collision_margin.max(0.0)).clamp(0.0, arm_length)
        } else {
            arm_length
        };
        let distance = smooth_half_life(
            previous_distance.clamp(0.0, arm_length),
            target,
            self.lag_half_life,
            dt.max(0.0),
        );
        (
            distance,
            if hit[0].hit {
                SpringArmStatus::Hit
            } else {
                SpringArmStatus::Clear
            },
        )
    }
}

pub fn smooth_half_life(current: f32, target: f32, half_life: f32, dt: f32) -> f32 {
    if !current.is_finite() || !target.is_finite() {
        return if target.is_finite() { target } else { current };
    }
    if half_life <= 0.0 || dt <= 0.0 || !half_life.is_finite() || !dt.is_finite() {
        return target;
    }
    let k = libm::exp2f(-dt / half_life);
    target + (current - target) * k
}

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct FlyController {
    pub move_axis: Vec3,
    pub speed: f32,
}

impl FlyController {
    pub fn apply(self, pose: CameraPose, dt: f32) -> CameraPose {
        let movement = pose.rotation.mul_vec3(prism_math::DVec3::new(
            self.move_axis.x as f64,
            self.move_axis.y as f64,
            self.move_axis.z as f64,
        )) * (self.speed.max(0.0) * dt.max(0.0)) as f64;
        CameraPose {
            position: pose.position + movement,
            ..pose
        }
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct OrbitController {
    pub yaw: f32,
    pub pitch: f32,
    pub radius: f32,
    pub target: DVec3,
}

impl OrbitController {
    pub fn pose(self, world_origin: DVec3) -> CameraPose {
        let pitch = self.pitch.clamp(-1.56, 1.56);
        let rotation =
            DQuat::from_rotation_y(self.yaw as f64) * DQuat::from_rotation_x(pitch as f64);
        let offset = rotation.mul_vec3(DVec3::new(0.0, 0.0, self.radius.max(0.0) as f64));
        CameraPose {
            position: self.target + offset,
            rotation,
            world_origin,
        }
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct PanController {
    pub axis: Vec2,
    pub speed: f32,
}

impl PanController {
    pub fn apply(self, pose: CameraPose, dt: f32) -> CameraPose {
        let local = Vec3::new(self.axis.x, self.axis.y, 0.0) * (self.speed.max(0.0) * dt.max(0.0));
        let local = DVec3::new(local.x as f64, local.y as f64, local.z as f64);
        CameraPose {
            position: pose.position + pose.rotation.mul_vec3(local),
            ..pose
        }
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Impulse {
    pub translation: Vec3,
    pub rotation: Vec3,
    pub remaining_s: f32,
}

impl Impulse {
    pub fn sample(self, dt: f32) -> (CameraPose, Self) {
        let factor = if self.remaining_s <= 0.0 {
            0.0
        } else {
            (dt.max(0.0) / self.remaining_s).clamp(0.0, 1.0)
        };
        let next = Self {
            translation: self.translation * (1.0 - factor),
            rotation: self.rotation * (1.0 - factor),
            remaining_s: (self.remaining_s - dt.max(0.0)).max(0.0),
        };
        let pose = CameraPose {
            position: self.translation.as_dvec3(),
            rotation: Quat::from_rotation_y(self.rotation.y).as_dquat(),
            ..CameraPose::default()
        };
        (pose, next)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn selection_is_stable() {
        let a = VirtualCamera {
            key: 2,
            priority: 10,
            ..Default::default()
        };
        let b = VirtualCamera {
            key: 1,
            priority: 10,
            ..Default::default()
        };
        assert_eq!(CameraBrain::choose(&[a, b]).unwrap().key, 1);
    }

    #[test]
    fn brain_blends_pose_after_a_switch() {
        let a = VirtualCamera {
            key: 1,
            pose: CameraPose {
                position: DVec3::ZERO,
                ..Default::default()
            },
            ..Default::default()
        };
        let b = VirtualCamera {
            key: 2,
            priority: 1,
            pose: CameraPose {
                position: DVec3::new(10.0, 0.0, 0.0),
                ..Default::default()
            },
            ..Default::default()
        };
        let mut brain = CameraBrain {
            blend_duration_s: 1.0,
            ..Default::default()
        };
        let _ = brain.update(&[a], 0.0, CameraPose::default());
        let pose = brain.update(&[a, b], 0.5, CameraPose::default());
        assert!((pose.position.x - 5.0).abs() < 1.0e-6);
        assert!(!brain.cut);
    }

    #[test]
    fn half_life_is_frame_rate_independent_at_zero() {
        assert_eq!(smooth_half_life(1.0, 4.0, 0.0, 1.0), 4.0);
    }

    #[test]
    fn body_and_aim_compose_with_finite_fallbacks() {
        let output = compose_rig(
            BodyRig::Follow {
                target: DVec3::new(1.0, 2.0, 3.0),
                offset: DVec3::new(0.0, 1.0, 0.0),
            },
            AimRig::LookAt {
                target: DVec3::new(1.0, 2.0, 0.0),
            },
            DVec3::ZERO,
            CameraPose::default(),
        );
        assert_eq!(output.pose.position, DVec3::new(1.0, 3.0, 3.0));
        assert!(output.pose.rotation.is_finite());
    }

    #[test]
    fn spring_arm_reports_hit_and_unavailable() {
        struct Query;
        impl SceneQuery for Query {
            fn cast_batch(&self, _requests: &[CastRequest], hits: &mut [CastHit]) {
                hits[0].hit = true;
                hits[0].distance = 1.0;
            }
        }
        let arm = SpringArm {
            arm_length: 4.0,
            collision_margin: 0.1,
            ..Default::default()
        };
        let (distance, status) = arm.resolve(Some(&Query), Vec3::ZERO, Vec3::Z, 4.0, 0.016);
        assert_eq!(status, SpringArmStatus::Hit);
        assert!((distance - 0.9).abs() < 1.0e-6);
        let (_, status) = arm.resolve::<Query>(None, Vec3::ZERO, Vec3::Z, 4.0, 0.016);
        assert_eq!(status, SpringArmStatus::Unavailable);
    }

    #[test]
    fn noise_is_deterministic_and_bounded() {
        let noise = CameraNoise {
            position: NoiseAxes {
                amplitude: Vec3::new(10.0, 0.0, 0.0),
                frequency: Vec3::splat(2.0),
            },
            gain: 1.0,
            seed: 42,
            temporal_clamp: 0.25,
            ..Default::default()
        };
        let a = noise.sample(CameraPose::default(), 1.0);
        let b = noise.sample(CameraPose::default(), 1.0);
        assert_eq!(a, b);
        assert!(a.position.length() <= 10.0 * 0.25 * 1.01);
    }

    #[test]
    fn impulse_buffer_deduplicates_and_clamps() {
        let source = ImpulseSource {
            id: 7,
            position: Vec3::ZERO,
            velocity: Vec3::splat(10.0),
            range: 10.0,
            attack_s: 0.0,
            sustain_s: 1.0,
            decay_s: 1.0,
        };
        let mut buffer = ImpulseBuffer::default();
        assert!(buffer.submit(source, 0.0));
        assert!(!buffer.submit(source, 0.0));
        let response = buffer.sample(Vec3::ZERO, 0.5);
        assert!(response.translation.length() <= 1.0 + 1.0e-6);
    }

    #[test]
    fn input_map_rejects_non_finite_values() {
        let command = CameraInputMap::default().normalize(CameraCommand {
            move_axis: Vec3::splat(f32::NAN),
            look_axis: Vec2::splat(f32::INFINITY),
            zoom: f32::NAN,
            boost: false,
        });
        assert_eq!(command, CameraCommand::default());
    }

    #[test]
    fn scheduler_respects_view_lod_interval() {
        let mut scheduler = ViewUpdateScheduler::default();
        let policy = prism_camera::ViewLod {
            max_update_hz: 30.0,
            ..Default::default()
        };
        assert!(scheduler.should_update(0, policy, 0, 60.0));
        assert!(!scheduler.should_update(0, policy, 1, 60.0));
        assert!(scheduler.should_update(0, policy, 2, 60.0));
    }
}
