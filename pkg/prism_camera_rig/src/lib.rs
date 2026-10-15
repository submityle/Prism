//! Allocation-free camera rig and director primitives.
#![cfg_attr(not(test), no_std)]
#![forbid(unsafe_code)]

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

impl Default for VirtualCamera {
    fn default() -> Self {
        Self { key: 0, priority: 0, enabled: true, pose: CameraPose::default(), lens: None, weight: 1.0 }
    }
}

/// Blend policy for two virtual-camera outputs.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Blend {
    pub duration_s: f32,
    pub elapsed_s: f32,
}

impl Blend {
    pub const fn new(duration_s: f32) -> Self { Self { duration_s, elapsed_s: 0.0 } }
    pub fn advance(self, dt: f32) -> Self { Self { elapsed_s: (self.elapsed_s + dt.max(0.0)).min(self.duration_s.max(0.0)), ..self } }
    pub fn amount(self) -> f32 {
        if self.duration_s <= 0.0 { 1.0 } else { (self.elapsed_s / self.duration_s).clamp(0.0, 1.0) }
    }
    pub fn finished(self) -> bool { self.amount() >= 1.0 }
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
        cameras.iter().copied().filter(|camera| camera.enabled && camera.weight > 0.0).max_by(|a, b| {
            a.priority.cmp(&b.priority).then_with(|| b.key.cmp(&a.key))
        })
    }

    pub fn update(&mut self, candidates: &[VirtualCamera], dt: f32, default: CameraPose) -> CameraPose {
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
            if let Some(from) = self.from_pose { blend_pose(from, target, amount) } else { target }
        } else { target };
        self.current_pose = Some(output);
        if self.blend.is_none() { self.from_pose = None; }
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
pub enum SpringArmStatus { Clear, Hit, Unavailable }

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SpringArm {
    pub arm_length: f32,
    pub probe_radius: f32,
    pub lag_half_life: f32,
    pub collision_margin: f32,
}

impl Default for SpringArm {
    fn default() -> Self { Self { arm_length: 4.0, probe_radius: 0.2, lag_half_life: 0.0, collision_margin: 0.05 } }
}

impl SpringArm {
    pub fn resolve<Q: SceneQuery>(&self, query: Option<&Q>, origin: Vec3, direction: Vec3, previous_distance: f32, dt: f32) -> (f32, SpringArmStatus) {
        let arm_length = self.arm_length.max(0.0);
        let direction = direction.normalize_or_zero();
        let Some(query) = query else { return (arm_length, SpringArmStatus::Unavailable); };
        let request = CastRequest { origin, direction, distance: arm_length, radius: self.probe_radius.max(0.0) };
        let mut hit = [CastHit { hit: false, distance: arm_length, normal: Vec3::Y }];
        query.cast_batch(&[request], &mut hit);
        let target = if hit[0].hit { (hit[0].distance - self.collision_margin.max(0.0)).clamp(0.0, arm_length) } else { arm_length };
        let distance = smooth_half_life(previous_distance.clamp(0.0, arm_length), target, self.lag_half_life, dt.max(0.0));
        (distance, if hit[0].hit { SpringArmStatus::Hit } else { SpringArmStatus::Clear })
    }
}

pub fn smooth_half_life(current: f32, target: f32, half_life: f32, dt: f32) -> f32 {
    if half_life <= 0.0 || dt <= 0.0 { return target; }
    let k = libm::exp2f(-dt / half_life);
    target + (current - target) * k
}

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct FlyController { pub move_axis: Vec3, pub speed: f32 }

impl FlyController {
    pub fn apply(self, pose: CameraPose, dt: f32) -> CameraPose {
        let movement = pose.rotation.mul_vec3(
            prism_math::DVec3::new(
                self.move_axis.x as f64,
                self.move_axis.y as f64,
                self.move_axis.z as f64,
            ),
        ) * (self.speed.max(0.0) * dt.max(0.0)) as f64;
        CameraPose { position: pose.position + movement, ..pose }
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct OrbitController { pub yaw: f32, pub pitch: f32, pub radius: f32, pub target: DVec3 }

impl OrbitController {
    pub fn pose(self, world_origin: DVec3) -> CameraPose {
        let pitch = self.pitch.clamp(-1.56, 1.56);
        let rotation = DQuat::from_rotation_y(self.yaw as f64) * DQuat::from_rotation_x(pitch as f64);
        let offset = rotation.mul_vec3(DVec3::new(0.0, 0.0, self.radius.max(0.0) as f64));
        CameraPose { position: self.target + offset, rotation, world_origin }
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct PanController { pub axis: Vec2, pub speed: f32 }

impl PanController {
    pub fn apply(self, pose: CameraPose, dt: f32) -> CameraPose {
        let local = Vec3::new(self.axis.x, self.axis.y, 0.0) * (self.speed.max(0.0) * dt.max(0.0));
        let local = DVec3::new(local.x as f64, local.y as f64, local.z as f64);
        CameraPose { position: pose.position + pose.rotation.mul_vec3(local), ..pose }
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Impulse { pub translation: Vec3, pub rotation: Vec3, pub remaining_s: f32 }

impl Impulse {
    pub fn sample(self, dt: f32) -> (CameraPose, Self) {
        let factor = if self.remaining_s <= 0.0 { 0.0 } else { (dt.max(0.0) / self.remaining_s).clamp(0.0, 1.0) };
        let next = Self { translation: self.translation * (1.0 - factor), rotation: self.rotation * (1.0 - factor), remaining_s: (self.remaining_s - dt.max(0.0)).max(0.0) };
        let pose = CameraPose { position: self.translation.as_dvec3(), rotation: Quat::from_rotation_y(self.rotation.y).as_dquat(), ..CameraPose::default() };
        (pose, next)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn selection_is_stable() {
        let a = VirtualCamera { key: 2, priority: 10, ..Default::default() };
        let b = VirtualCamera { key: 1, priority: 10, ..Default::default() };
        assert_eq!(CameraBrain::choose(&[a, b]).unwrap().key, 1);
    }

    #[test]
    fn brain_blends_pose_after_a_switch() {
        let a = VirtualCamera { key: 1, pose: CameraPose { position: DVec3::ZERO, ..Default::default() }, ..Default::default() };
        let b = VirtualCamera { key: 2, priority: 1, pose: CameraPose { position: DVec3::new(10.0, 0.0, 0.0), ..Default::default() }, ..Default::default() };
        let mut brain = CameraBrain { blend_duration_s: 1.0, ..Default::default() };
        let _ = brain.update(&[a], 0.0, CameraPose::default());
        let pose = brain.update(&[a, b], 0.5, CameraPose::default());
        assert!((pose.position.x - 5.0).abs() < 1.0e-6);
        assert!(!brain.cut);
    }

    #[test]
    fn half_life_is_frame_rate_independent_at_zero() {
        assert_eq!(smooth_half_life(1.0, 4.0, 0.0, 1.0), 4.0);
    }
}
