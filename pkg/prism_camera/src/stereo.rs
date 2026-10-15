use prism_math::{DQuat, DVec3};

use crate::{derive_view, CameraError, CameraIntent, CameraPose, ViewData, ViewHistory};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Eye {
    Left,
    Right,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct StereoConfig {
    pub ipd_m: f32,
    pub convergence_m: f32,
    pub asymmetric: bool,
}

impl Default for StereoConfig {
    fn default() -> Self {
        Self {
            ipd_m: 0.064,
            convergence_m: 10.0,
            asymmetric: true,
        }
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct LateLatchPose {
    pub position: DVec3,
    pub rotation: DQuat,
    pub frame: u64,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct StereoPair {
    pub left: ViewData,
    pub right: ViewData,
}

impl StereoConfig {
    pub fn derive_pair(
        self,
        intent: CameraIntent,
        pose: CameraPose,
        frame: u64,
        target_size: [u32; 2],
        left_history: &mut ViewHistory,
        right_history: &mut ViewHistory,
    ) -> Result<StereoPair, CameraError> {
        let half_ipd = self.ipd_m.max(0.0) * 0.5;
        let rotation = pose.rotation.normalize();
        let right_axis = rotation.mul_vec3(DVec3::new(1.0, 0.0, 0.0));
        let left_pose = CameraPose {
            position: pose.position - right_axis * half_ipd as f64,
            ..pose
        };
        let right_pose = CameraPose {
            position: pose.position + right_axis * half_ipd as f64,
            ..pose
        };
        let mut left_intent = intent;
        let mut right_intent = intent;
        left_intent.kind = prism_render_architecture::view_family::ViewKind::StereoEye;
        right_intent.kind = prism_render_architecture::view_family::ViewKind::StereoEye;
        Ok(StereoPair {
            left: derive_view(left_intent, left_pose, frame, target_size, left_history)?,
            right: derive_view(right_intent, right_pose, frame, target_size, right_history)?,
        })
    }

    pub fn relatch(
        self,
        pair: StereoPair,
        pose: LateLatchPose,
        frame: u64,
    ) -> Result<StereoPair, CameraError> {
        if pose.frame != frame {
            return Err(CameraError::new(
                crate::CameraErrorCode::InvalidTargetSize,
                "late_latch_frame",
                0.0,
            ));
        }
        if !pose.position.is_finite() || !pose.rotation.is_finite() {
            return Err(CameraError::new(
                crate::CameraErrorCode::NonFinitePose,
                "late_latch_pose",
                0.0,
            ));
        }
        let rotation = pose.rotation.normalize();
        let right_axis = rotation.mul_vec3(DVec3::new(1.0, 0.0, 0.0));
        let half_ipd = self.ipd_m.max(0.0) as f64 * 0.5;
        let left_position = pose.position - right_axis * half_ipd;
        let right_position = pose.position + right_axis * half_ipd;
        let mut left = pair.left;
        let mut right = pair.right;
        let rotation = rotation.as_quat().normalize();
        let left_origin = left.world_origin;
        let right_origin = right.world_origin;
        let left_transform = prism_math::Affine3::from_mat3_translation(
            prism_math::Mat4::from_quat(rotation).to_mat3(),
            prism_math::Vec3::new(
                (left_position.x - left_origin.x) as f32,
                (left_position.y - left_origin.y) as f32,
                (left_position.z - left_origin.z) as f32,
            ),
        );
        let right_transform = prism_math::Affine3::from_mat3_translation(
            prism_math::Mat4::from_quat(rotation).to_mat3(),
            prism_math::Vec3::new(
                (right_position.x - right_origin.x) as f32,
                (right_position.y - right_origin.y) as f32,
                (right_position.z - right_origin.z) as f32,
            ),
        );
        let left_view = left_transform.inverse();
        let right_view = right_transform.inverse();
        left.view_from_world = left_view;
        left.world_from_view = left_transform;
        right.view_from_world = right_view;
        right.world_from_view = right_transform;
        left.clip_from_world = left.clip_from_view * left_view.to_mat4();
        right.clip_from_world = right.clip_from_view * right_view.to_mat4();
        left.clip_from_world_unjittered = left.clip_from_view_unjittered * left_view.to_mat4();
        right.clip_from_world_unjittered = right.clip_from_view_unjittered * right_view.to_mat4();
        left.frustum = crate::extract_frustum(
            &left.clip_from_world_unjittered,
            left.frustum_reverse_z,
            left.frustum_finite_far,
        );
        right.frustum = crate::extract_frustum(
            &right.clip_from_world_unjittered,
            right.frustum_reverse_z,
            right.frustum_finite_far,
        );
        left.world_position = left_position;
        right.world_position = right_position;
        Ok(StereoPair { left, right })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stereo_eyes_are_separated_and_share_frame() {
        let mut left = ViewHistory::default();
        let mut right = ViewHistory::default();
        let pair = StereoConfig::default()
            .derive_pair(
                CameraIntent::default(),
                CameraPose::default(),
                1,
                [100, 100],
                &mut left,
                &mut right,
            )
            .unwrap();
        assert!(pair.left.world_position.x < pair.right.world_position.x);
        assert_eq!(
            pair.left.kind,
            prism_render_architecture::view_family::ViewKind::StereoEye
        );
    }

    #[test]
    fn late_latch_rebuilds_relative_matrices_and_frusta() {
        let mut left_history = ViewHistory::default();
        let mut right_history = ViewHistory::default();
        let origin = DVec3::new(1_000_000_000.0, -2_000_000_000.0, 3_000_000_000.0);
        let pair = StereoConfig::default()
            .derive_pair(
                CameraIntent::default(),
                CameraPose {
                    position: origin,
                    world_origin: origin,
                    ..Default::default()
                },
                7,
                [1280, 720],
                &mut left_history,
                &mut right_history,
            )
            .unwrap();
        let old_frustum = pair.left.frustum;
        let latched = StereoConfig::default()
            .relatch(
                pair,
                LateLatchPose {
                    position: origin + DVec3::new(0.25, 0.0, 0.5),
                    rotation: DQuat::from_rotation_y(0.25),
                    frame: 7,
                },
                7,
            )
            .unwrap();
        let left = latched.left;
        assert!(left.world_from_view.translation.x.abs() < 1.0);
        assert_eq!(
            left.clip_from_world,
            left.clip_from_view * left.view_from_world.to_mat4()
        );
        assert_eq!(
            left.clip_from_world_unjittered,
            left.clip_from_view_unjittered * left.view_from_world.to_mat4()
        );
        assert_eq!(left.frustum.count, old_frustum.count);
        assert_ne!(left.frustum, old_frustum);
    }

    #[test]
    fn late_latch_rejects_wrong_frame_and_non_finite_pose() {
        let mut left_history = ViewHistory::default();
        let mut right_history = ViewHistory::default();
        let pair = StereoConfig::default()
            .derive_pair(
                CameraIntent::default(),
                CameraPose::default(),
                3,
                [64, 64],
                &mut left_history,
                &mut right_history,
            )
            .unwrap();
        assert!(StereoConfig::default()
            .relatch(
                pair,
                LateLatchPose {
                    frame: 2,
                    ..Default::default()
                },
                3,
            )
            .is_err());
        assert!(StereoConfig::default()
            .relatch(
                pair,
                LateLatchPose {
                    position: DVec3::new(f64::NAN, 0.0, 0.0),
                    frame: 3,
                    ..Default::default()
                },
                3,
            )
            .is_err());
    }
}
