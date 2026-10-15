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

#[derive(Clone, Copy, Debug, PartialEq)]
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
        let right_axis = pose.rotation.mul_vec3(DVec3::new(1.0, 0.0, 0.0));
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
        let right_axis = pose.rotation.mul_vec3(DVec3::new(1.0, 0.0, 0.0));
        let half_ipd = self.ipd_m.max(0.0) as f64 * 0.5;
        let left_position = pose.position - right_axis * half_ipd;
        let right_position = pose.position + right_axis * half_ipd;
        let mut left = pair.left;
        let mut right = pair.right;
        let left_rotation = pose.rotation.as_quat();
        let left_transform = prism_math::Affine3::from_mat3_translation(
            prism_math::Mat4::from_quat(left_rotation).to_mat3(),
            prism_math::Vec3::new(
                left_position.x as f32,
                left_position.y as f32,
                left_position.z as f32,
            ),
        );
        let right_transform = prism_math::Affine3::from_mat3_translation(
            prism_math::Mat4::from_quat(left_rotation).to_mat3(),
            prism_math::Vec3::new(
                right_position.x as f32,
                right_position.y as f32,
                right_position.z as f32,
            ),
        );
        let left_view = left_transform.inverse();
        let right_view = right_transform.inverse();
        let left_stable = left.clip_from_world_unjittered * left.world_from_view.to_mat4();
        let right_stable = right.clip_from_world_unjittered * right.world_from_view.to_mat4();
        left.view_from_world = left_view;
        left.world_from_view = left_transform;
        right.view_from_world = right_view;
        right.world_from_view = right_transform;
        left.clip_from_world = left.clip_from_view * left_view.to_mat4();
        right.clip_from_world = right.clip_from_view * right_view.to_mat4();
        left.clip_from_world_unjittered = left_stable * left_view.to_mat4();
        right.clip_from_world_unjittered = right_stable * right_view.to_mat4();
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
}
