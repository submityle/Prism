use prism_math::{Affine3, DQuat, DVec3, Mat4, Vec2, Vec3};
use prism_render_architecture::view_family::{ViewHandle, ViewImportance, ViewKind, ViewRegistry};

#[cfg(feature = "physical_lens")]
use crate::Lens;
use crate::{
    extract_frustum, CameraError, CameraErrorCode, Exposure, GpuViewRecord, JitterSequence,
    Projection, RenderTarget, Viewport, ViewportRect,
};

/// A validated world-space camera pose. The origin is the rebasing point used
/// to keep the render-space transform in f32 without losing large-world data.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CameraPose {
    pub position: DVec3,
    pub rotation: DQuat,
    pub world_origin: DVec3,
}

impl Default for CameraPose {
    fn default() -> Self {
        Self {
            position: DVec3::ZERO,
            rotation: DQuat::IDENTITY,
            world_origin: DVec3::ZERO,
        }
    }
}

/// Mutable camera intent consumed by the view derivation stage.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CameraIntent {
    pub active: bool,
    pub kind: ViewKind,
    pub target: RenderTarget,
    pub clear: crate::ClearMode,
    pub viewport: Viewport,
    pub order: i32,
    pub projection: Projection,
    pub jitter: JitterSequence,
    pub jitter_scale: f32,
    pub exposure: Exposure,
    #[cfg(feature = "physical_lens")]
    pub lens: Option<Lens>,
    pub importance: ViewImportance,
    pub relative_rendering: bool,
}

impl Default for CameraIntent {
    fn default() -> Self {
        Self {
            active: true,
            kind: ViewKind::Main,
            target: RenderTarget::PrimaryWindow,
            clear: crate::ClearMode::ColorAndDepth,
            viewport: Viewport::FULL,
            order: 0,
            projection: Projection::Perspective(Default::default()),
            jitter: JitterSequence::default(),
            jitter_scale: 1.0,
            exposure: Exposure::default(),
            #[cfg(feature = "physical_lens")]
            lens: None,
            importance: ViewKind::Main.default_importance(),
            relative_rendering: true,
        }
    }
}

/// Per-frame immutable view result.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ViewData {
    pub handle: ViewHandle,
    pub kind: ViewKind,
    pub target: RenderTarget,
    pub clear: crate::ClearMode,
    pub order: i32,
    pub view_from_world: Affine3,
    pub world_from_view: Affine3,
    pub clip_from_view: Mat4,
    pub clip_from_view_unjittered: Mat4,
    pub clip_from_world: Mat4,
    pub clip_from_world_unjittered: Mat4,
    pub jitter: Vec2,
    pub prev_clip_from_world: Mat4,
    pub history_valid: bool,
    pub teleport: bool,
    pub frustum: crate::Frustum,
    pub frustum_reverse_z: bool,
    pub frustum_finite_far: bool,
    pub relative_rendering: bool,
    pub exposure: f32,
    pub viewport_px: ViewportRect,
    pub world_position: DVec3,
    pub world_origin: DVec3,
    pub status: ViewStatus,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ViewStatus {
    Valid,
    Invalid(CameraErrorCode),
}

/// State kept by the owner of a camera between frames.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ViewHistory {
    pub previous_unjittered: Mat4,
    pub previous_position: DVec3,
    pub previous_rotation: DQuat,
    pub valid: bool,
    pub extent: [u32; 2],
}

impl Default for ViewHistory {
    fn default() -> Self {
        Self {
            previous_unjittered: Mat4::IDENTITY,
            previous_position: DVec3::ZERO,
            previous_rotation: DQuat::IDENTITY,
            valid: false,
            extent: [0, 0],
        }
    }
}

impl ViewHistory {
    pub fn clear(&mut self) {
        self.valid = false;
    }
}

pub fn derive_view(
    intent: CameraIntent,
    pose: CameraPose,
    frame: u64,
    target_size: [u32; 2],
    history: &mut ViewHistory,
) -> Result<ViewData, CameraError> {
    if !intent.active {
        return Err(CameraError::new(
            CameraErrorCode::InvalidProjection,
            "active",
            0.0,
        ));
    }
    if target_size[0] == 0 || target_size[1] == 0 {
        return Err(CameraError::new(
            CameraErrorCode::InvalidTargetSize,
            "target_size",
            0.0,
        ));
    }
    if !pose.position.is_finite() || !pose.world_origin.is_finite() || !pose.rotation.is_finite() {
        return Err(CameraError::new(
            CameraErrorCode::NonFinitePose,
            "pose",
            0.0,
        ));
    }
    let viewport_px = intent.viewport.resolve(target_size)?;
    #[cfg(feature = "physical_lens")]
    let mut projection = intent.projection;
    #[cfg(not(feature = "physical_lens"))]
    let projection = intent.projection;
    #[cfg(feature = "physical_lens")]
    if let Projection::Perspective(mut value) = projection {
        if let Some(lens) = intent.lens {
            let mut lens_projection = value;
            lens_projection.fov_y = lens.vertical_fov();
            lens_projection.aspect = lens.filmback.sensor_width_mm / lens.filmback.sensor_height_mm;
            value = lens_projection;
            projection = Projection::Perspective(value);
        }
    }
    let mut clip_from_view = projection.clip_from_view()?;
    let clip_from_view_unjittered = clip_from_view;
    let (jitter_state, jitter_ndc) =
        intent
            .jitter
            .ndc_offset(frame, viewport_px.size, intent.jitter_scale);
    clip_from_view.w_axis.x += jitter_ndc.x;
    clip_from_view.w_axis.y += jitter_ndc.y;

    let render_origin = if intent.relative_rendering {
        pose.world_origin
    } else {
        DVec3::ZERO
    };
    let local_position = pose.position - render_origin;
    if !local_position.x.is_finite()
        || !local_position.y.is_finite()
        || !local_position.z.is_finite()
    {
        return Err(CameraError::new(
            CameraErrorCode::NonFinitePose,
            "local_position",
            0.0,
        ));
    }
    let rotation = pose.rotation.as_quat().normalize();
    let world_from_view = Affine3::from_mat3_translation(
        Mat4::from_quat(rotation).to_mat3(),
        Vec3::new(
            local_position.x as f32,
            local_position.y as f32,
            local_position.z as f32,
        ),
    );
    let view_from_world = world_from_view.inverse();
    if !view_from_world.is_finite() || !world_from_view.is_finite() {
        return Err(CameraError::new(CameraErrorCode::SingularView, "pose", 0.0));
    }
    let stable = clip_from_view_unjittered * view_from_world.to_mat4();
    let current = clip_from_view * view_from_world.to_mat4();
    if !matrix_is_finite(stable) || !matrix_is_finite(current) {
        return Err(CameraError::new(
            CameraErrorCode::NonFiniteDerivedView,
            "matrix",
            0.0,
        ));
    }
    let teleport = history.valid && (pose.position - history.previous_position).length() > 10.0
        || history.valid && pose.rotation.dot(history.previous_rotation).abs() < 0.5;
    let history_valid = history.valid && history.extent == target_size && !teleport;
    let previous = if history_valid {
        history.previous_unjittered
    } else {
        stable
    };
    let frustum = extract_frustum(&stable, projection.reverse_z(), projection.far().is_some());
    let exposure = intent.exposure.multiplier();
    if !exposure.is_finite() {
        return Err(CameraError::new(
            CameraErrorCode::NonFiniteDerivedView,
            "exposure",
            0.0,
        ));
    }
    history.previous_unjittered = stable;
    history.previous_position = pose.position;
    history.previous_rotation = pose.rotation;
    history.extent = target_size;
    history.valid = true;
    Ok(ViewData {
        handle: ViewHandle::INVALID,
        kind: intent.kind,
        target: intent.target,
        clear: intent.clear,
        order: intent.order,
        view_from_world,
        world_from_view,
        clip_from_view,
        clip_from_view_unjittered,
        clip_from_world: current,
        clip_from_world_unjittered: stable,
        jitter: jitter_state.value_pixels,
        prev_clip_from_world: previous,
        history_valid,
        teleport,
        frustum,
        frustum_reverse_z: projection.reverse_z(),
        frustum_finite_far: projection.far().is_some(),
        relative_rendering: intent.relative_rendering,
        exposure,
        viewport_px,
        world_position: pose.position,
        world_origin: render_origin,
        status: ViewStatus::Valid,
    })
}

fn matrix_is_finite(matrix: Mat4) -> bool {
    matrix.x_axis.is_finite()
        && matrix.y_axis.is_finite()
        && matrix.z_axis.is_finite()
        && matrix.w_axis.is_finite()
}

pub fn publish_view(
    registry: &mut ViewRegistry,
    handle: &mut ViewHandle,
    data: &mut ViewData,
    importance: ViewImportance,
) -> Result<(), CameraError> {
    if *handle == ViewHandle::INVALID {
        *handle = registry.register_with(data.kind, importance);
    } else if !registry.contains(*handle) {
        return Err(
            CameraError::new(CameraErrorCode::StaleViewHandle, "handle", 0.0).with_handle(*handle),
        );
    } else {
        registry.set_importance(*handle, importance);
    }
    data.handle = *handle;
    Ok(())
}

impl ViewData {
    pub fn gpu_record(&self) -> GpuViewRecord {
        let (hi, lo) = GpuViewRecord::split_origin(self.world_origin);
        GpuViewRecord {
            clip_from_world: GpuViewRecord::matrix(self.clip_from_world),
            clip_from_world_unjittered: GpuViewRecord::matrix(self.clip_from_world_unjittered),
            prev_clip_from_world: GpuViewRecord::matrix(self.prev_clip_from_world),
            view_from_world: GpuViewRecord::matrix(self.view_from_world.to_mat4()),
            world_from_view: GpuViewRecord::matrix(self.world_from_view.to_mat4()),
            world_origin_hi: hi,
            world_origin_lo: lo,
            exposure: self.exposure,
            schema_version: crate::VIEW_SCHEMA_VERSION,
            flags: GpuViewRecord::flags(
                self.history_valid,
                self.teleport,
                self.frustum.count == 6,
                self.relative_rendering,
            ),
            view_index: self.handle.index,
            jitter: GpuViewRecord::jitter(self.jitter),
            viewport: [
                self.viewport_px.origin[0] as f32,
                self.viewport_px.origin[1] as f32,
                self.viewport_px.size[0] as f32,
                self.viewport_px.size[1] as f32,
            ],
            frustum: self
                .frustum
                .planes
                .map(|p| [p.normal.x, p.normal.y, p.normal.z, p.d]),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::OrthographicProjection;
    #[cfg(feature = "physical_lens")]
    use crate::PerspectiveProjection;

    #[test]
    fn first_frame_clears_history_and_second_frame_reuses_it() {
        let mut history = ViewHistory::default();
        let intent = CameraIntent::default();
        let pose = CameraPose::default();
        let first = derive_view(intent, pose, 0, [1920, 1080], &mut history).unwrap();
        assert!(!first.history_valid);
        let second = derive_view(intent, pose, 1, [1920, 1080], &mut history).unwrap();
        assert!(second.history_valid);
        assert_eq!(
            second.prev_clip_from_world,
            first.clip_from_world_unjittered
        );
    }

    #[test]
    fn resize_invalidates_history() {
        let mut history = ViewHistory::default();
        let intent = CameraIntent {
            projection: Projection::Orthographic(OrthographicProjection::default()),
            ..Default::default()
        };
        let pose = CameraPose::default();
        let _ = derive_view(intent, pose, 0, [100, 100], &mut history).unwrap();
        let next = derive_view(intent, pose, 1, [200, 100], &mut history).unwrap();
        assert!(!next.history_valid);
    }

    #[test]
    fn publish_rejects_stale_handle() {
        let mut registry = ViewRegistry::new();
        let mut handle = registry.register(ViewKind::Main);
        registry.remove(handle);
        let mut data = derive_view(
            Default::default(),
            Default::default(),
            0,
            [64, 64],
            &mut ViewHistory::default(),
        )
        .unwrap();
        let error = publish_view(
            &mut registry,
            &mut handle,
            &mut data,
            ViewKind::Main.default_importance(),
        )
        .unwrap_err();
        assert_eq!(error.code, CameraErrorCode::StaleViewHandle);
    }

    #[test]
    fn gpu_relative_flag_matches_camera_intent() {
        let pose = CameraPose {
            position: DVec3::new(10.0, 20.0, 30.0),
            world_origin: DVec3::new(1000.0, 2000.0, 3000.0),
            ..Default::default()
        };
        let mut intent = CameraIntent::default();
        intent.relative_rendering = false;
        let view = derive_view(
            intent,
            pose,
            0,
            [64, 64],
            &mut ViewHistory::default(),
        )
        .unwrap();
        assert!(!view.relative_rendering);
        assert_eq!(view.gpu_record().flags & (1 << 3), 0);

        intent.relative_rendering = true;
        let view = derive_view(
            intent,
            pose,
            0,
            [64, 64],
            &mut ViewHistory::default(),
        )
        .unwrap();
        assert!(view.relative_rendering);
        assert_ne!(view.gpu_record().flags & (1 << 3), 0);
    }

    #[test]
    fn relative_rendering_selects_world_origin_for_view_space() {
        let pose = CameraPose {
            position: DVec3::new(1010.0, 2020.0, 3030.0),
            world_origin: DVec3::new(1000.0, 2000.0, 3000.0),
            ..Default::default()
        };
        let mut relative = CameraIntent::default();
        relative.relative_rendering = true;
        let relative_view = derive_view(
            relative,
            pose,
            0,
            [64, 64],
            &mut ViewHistory::default(),
        )
        .unwrap();
        assert_eq!(relative_view.world_origin, pose.world_origin);
        assert_eq!(relative_view.world_from_view.translation.x, 10.0);
        assert_eq!(relative_view.world_from_view.translation.y, 20.0);
        assert_eq!(relative_view.world_from_view.translation.z, 30.0);

        let mut absolute = relative;
        absolute.relative_rendering = false;
        let absolute_view = derive_view(
            absolute,
            pose,
            0,
            [64, 64],
            &mut ViewHistory::default(),
        )
        .unwrap();
        assert_eq!(absolute_view.world_origin, DVec3::ZERO);
        assert_eq!(absolute_view.world_from_view.translation.x, 1010.0);
        assert_eq!(absolute_view.world_from_view.translation.y, 2020.0);
        assert_eq!(absolute_view.world_from_view.translation.z, 3030.0);
    }

    #[cfg(feature = "physical_lens")]
    #[test]
    fn lens_updates_perspective_fov() {
        let mut history = ViewHistory::default();
        let mut intent = CameraIntent::default();
        intent.projection = Projection::Perspective(PerspectiveProjection::default());
        intent.lens = Some(Lens::full_frame_50mm());
        let view = derive_view(intent, Default::default(), 0, [1920, 1080], &mut history).unwrap();
        assert!(view.clip_from_view.x_axis.x.is_finite());
    }
}
