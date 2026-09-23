use bevy_asset::{load_embedded_asset, Handle};
use bevy_ecs::prelude::*;
use bevy_material::{
    bind_group_layout_entries::{
        binding_types::{storage_buffer, storage_buffer_read_only, texture_2d},
        BindGroupLayoutEntries,
    },
    descriptor::BindGroupLayoutDescriptor,
};
use bevy_render::{
    render_resource::{
        BindGroup, BindGroupEntries, BindGroupLayout, Buffer, BufferDescriptor, BufferId,
        BufferUsages, CachedComputePipelineId, ComputePipelineDescriptor, ShaderStages,
        TextureSampleType, TextureViewId,
    },
    renderer::RenderDevice,
};
use bevy_render::render_resource::ShaderType;
use bevy_math::{Mat4, Vec2, Vec3};
use bevy_shader::Shader;

#[repr(C)]
#[derive(
    Clone,
    Copy,
    Debug,
    Default,
    bytemuck::Pod,
    bytemuck::Zeroable,
    bevy_render::render_resource::ShaderType,
)]
pub(crate) struct RenderHzbCullInput {
    pub uv_min: [f32; 2],
    pub uv_max: [f32; 2],
    pub nearest_depth: f32,
    pub projected_velocity: f32,
    pub is_active: u32,
    pub _padding: u32,
}

pub(crate) fn project_sphere_to_hzb(
    current_clip_from_world: Mat4,
    previous_clip_from_world: Mat4,
    current_center: Vec3,
    previous_center: Vec3,
    radius: f32,
    viewport: [u32; 4],
) -> Option<RenderHzbCullInput> {
    let current = current_clip_from_world * current_center.extend(1.0);
    let previous = previous_clip_from_world * previous_center.extend(1.0);
    if current.w <= 1.0e-5 || !current.is_finite() || !previous.is_finite() {
        return None;
    }
    let ndc = current.truncate() / current.w;
    let previous_ndc = if previous.w > 1.0e-5 {
        previous.truncate() / previous.w
    } else {
        ndc
    };
    let clip_radius = radius.abs() / current.w.abs().max(1.0e-5);
    let uv = Vec3::new(ndc.x * 0.5 + 0.5, 0.5 - ndc.y * 0.5, ndc.z);
    let uv_radius = Vec3::new(clip_radius * 0.5, clip_radius * 0.5, clip_radius);
    let viewport_scale = Vec3::new(viewport[2].max(1) as f32, viewport[3].max(1) as f32, 1.0);
    Some(RenderHzbCullInput {
        uv_min: (uv - uv_radius)
            .truncate()
            .clamp(Vec2::ZERO, Vec2::ONE)
            .to_array(),
        uv_max: (uv + uv_radius)
            .truncate()
            .clamp(Vec2::ZERO, Vec2::ONE)
            .to_array(),
        // Reverse-Z: the nearest point has the greater NDC depth.
        nearest_depth: (uv.z + uv_radius.z).clamp(0.0, 1.0),
        projected_velocity: ((ndc - previous_ndc) * viewport_scale).truncate().length(),
        is_active: 1,
        _padding: 0,
    })
}

#[derive(Resource)]
pub(crate) struct HzbVisibilityPipeline {
    pub(crate) pipeline: CachedComputePipelineId,
    layout: BindGroupLayoutDescriptor,
    bind_group_layout: BindGroupLayout,
}

#[derive(Component, Default)]
pub(crate) struct HzbVisibilityBindGroup {
    pub(crate) bind_group: Option<BindGroup>,
    ids: Option<(BufferId, TextureViewId, BufferId)>,
}

#[derive(Resource)]
pub(crate) struct HzbVisibilityBuffers {
    candidates: Buffer,
    stages: Buffer,
    capacity: u32,
    view_ranges: bevy_platform::collections::HashMap<
        bevy_render::view::RetainedViewEntity,
        (u32, u32),
    >,
    active_slots: bevy_platform::collections::HashSet<u32>,
}

impl FromWorld for HzbVisibilityBuffers {
    fn from_world(world: &mut World) -> Self {
        let device = world.resource::<RenderDevice>();
        Self::with_capacity(device, 1)
    }
}

impl HzbVisibilityBuffers {
    fn with_capacity(device: &RenderDevice, capacity: u32) -> Self {
        let capacity = capacity.max(1);
        Self {
            candidates: device.create_buffer(&BufferDescriptor {
                label: Some("prism hzb candidates"),
                size: capacity as u64 * RenderHzbCullInput::min_size().get(),
                usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }),
            stages: device.create_buffer(&BufferDescriptor {
                label: Some("prism hzb stages"),
                size: capacity as u64 * size_of::<u32>() as u64,
                usage: BufferUsages::STORAGE | BufferUsages::COPY_DST | BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            }),
            capacity,
            view_ranges: bevy_platform::collections::HashMap::default(),
            active_slots: bevy_platform::collections::HashSet::default(),
        }
    }

    pub(crate) fn ensure_capacity(&mut self, device: &RenderDevice, capacity: u32) {
        if capacity > self.capacity {
            let view_ranges = core::mem::take(&mut self.view_ranges);
            let active_slots = core::mem::take(&mut self.active_slots);
            *self = Self::with_capacity(device, capacity.next_power_of_two());
            self.view_ranges = view_ranges;
            self.active_slots = active_slots;
        }
    }

    pub(crate) fn bindings(&self) -> (&Buffer, &Buffer) {
        (&self.candidates, &self.stages)
    }

    pub(crate) fn view_range(
        &self,
        retained: bevy_render::view::RetainedViewEntity,
    ) -> Option<(u32, u32)> {
        self.view_ranges.get(&retained).copied()
    }

    pub(crate) fn active_count_in_range(&self, start: u32, count: u32) -> u32 {
        active_count_in_range(&self.active_slots, start, count)
    }
}

fn active_count_in_range(
    active_slots: &bevy_platform::collections::HashSet<u32>,
    start: u32,
    count: u32,
) -> u32 {
    active_slots
        .iter()
        .filter(|slot| **slot >= start && **slot < start.saturating_add(count))
        .count() as u32
}

#[expect(
    clippy::too_many_arguments,
    reason = "HZB staging joins scene, per-view state, buffers, device, and queue."
)]
pub(crate) fn prepare_hzb_candidates(
    scene: Res<crate::RenderGpuScene>,
    state: Res<super::runtime::UnifiedVisibilityState>,
    mut buffers: ResMut<HzbVisibilityBuffers>,
    device: Res<RenderDevice>,
    queue: Res<bevy_render::renderer::RenderQueue>,
    views: Query<&bevy_render::view::ExtractedView>,
) {
    let capacity = scene.mirror().capacity() as u32;
    let view_count = state.views.len() as u32;
    buffers.ensure_capacity(&device, capacity.saturating_mul(view_count).max(1));
    if view_count == 0 {
        return;
    }
    let mut candidates =
        vec![RenderHzbCullInput::default(); capacity.saturating_mul(view_count) as usize];
    buffers.view_ranges.clear();
    buffers.active_slots.clear();
    for (view_index, view_record) in state.views.iter().enumerate() {
        let Some(retained) = state.retained_view(view_record.handle) else {
            continue;
        };
        buffers
            .view_ranges
            .insert(retained, (view_index as u32 * capacity, capacity));
        let current_clip = Mat4::from_cols_array_2d(&view_record.clip_from_world);
        let previous_clip = Mat4::from_cols_array_2d(&view_record.previous_clip_from_world);
        for handle in scene.mirror().live_handles() {
            let Some(record) = scene.mirror().get(handle) else {
                continue;
            };
            let current_center = transform_point(record.current_transform, record.bounds.center);
            let previous_center = transform_point(record.previous_transform, record.bounds.center);
            if let Some(candidate) = project_sphere_to_hzb(
                current_clip,
                previous_clip,
                current_center,
                previous_center,
                record.bounds.radius,
                view_record.viewport,
            ) {
                let slot = view_index * capacity as usize + handle.index as usize;
                candidates[slot] = candidate;
                buffers.active_slots.insert(slot as u32);
            }
        }
    }
    let (candidate_buffer, stage_buffer) = buffers.bindings();
    if !candidates.is_empty() {
        queue.write_buffer(candidate_buffer, 0, bytemuck::cast_slice(&candidates));
        queue.write_buffer(
            stage_buffer,
            0,
            bytemuck::cast_slice(&vec![
                prism_render_visibility::VisibilityStageMask::EARLY.0;
                candidates.len()
            ]),
        );
    }
    debug_assert_eq!(views.iter().count(), state.views.len());
}

fn transform_point(
    transform: prism_render_architecture::gpu_scene::SceneTransform,
    point: [f32; 3],
) -> Vec3 {
    Vec3::new(
        transform.rows[0][0] * point[0]
            + transform.rows[0][1] * point[1]
            + transform.rows[0][2] * point[2]
            + transform.rows[0][3],
        transform.rows[1][0] * point[0]
            + transform.rows[1][1] * point[1]
            + transform.rows[1][2] * point[2]
            + transform.rows[1][3],
        transform.rows[2][0] * point[0]
            + transform.rows[2][1] * point[1]
            + transform.rows[2][2] * point[2]
            + transform.rows[2][3],
    )
}

pub(crate) fn inspect_hzb_visibility_pipeline(
    pipeline: Res<HzbVisibilityPipeline>,
    cache: Res<bevy_render::render_resource::PipelineCache>,
    mut buffers: ResMut<HzbVisibilityBuffers>,
    device: Res<RenderDevice>,
) {
    let _projector: fn(Mat4, Mat4, Vec3, Vec3, f32, [u32; 4]) -> Option<RenderHzbCullInput> =
        project_sphere_to_hzb;
    buffers.ensure_capacity(&device, 1);
    let _ = buffers.bindings();
    let _ = (
        &pipeline.layout,
        &pipeline.bind_group_layout,
        cache.get_compute_pipeline(pipeline.pipeline),
    );
}

pub(crate) fn prepare_hzb_bind_groups(
    mut commands: Commands,
    pipeline: Res<HzbVisibilityPipeline>,
    buffers: Res<HzbVisibilityBuffers>,
    device: Res<RenderDevice>,
    views: Query<(
        Entity,
        &bevy_core_pipeline::mip_generation::experimental::depth::ViewDepthPyramid,
        Option<&HzbVisibilityBindGroup>,
    )>,
) {
    let (candidates, stages) = buffers.bindings();
    for (entity, pyramid, existing) in &views {
        let ids = (candidates.id(), pyramid.all_mips.id(), stages.id());
        if existing.is_some_and(|existing| existing.ids == Some(ids)) {
            continue;
        }
        let bind_group = device.create_bind_group(
            "prism hzb visibility",
            &pipeline.bind_group_layout,
            &BindGroupEntries::sequential((
                candidates.as_entire_binding(),
                &pyramid.all_mips,
                stages.as_entire_binding(),
            )),
        );
        commands.entity(entity).insert(HzbVisibilityBindGroup {
            bind_group: Some(bind_group),
            ids: Some(ids),
        });
    }
}

pub(crate) fn init_hzb_visibility_pipeline(
    mut commands: Commands,
    device: Res<RenderDevice>,
    cache: Res<bevy_render::render_resource::PipelineCache>,
    asset_server: Res<bevy_asset::AssetServer>,
) {
    let entries = BindGroupLayoutEntries::sequential(
        ShaderStages::COMPUTE,
        (
            storage_buffer_read_only::<RenderHzbCullInput>(false),
            texture_2d(TextureSampleType::Float { filterable: false }),
            storage_buffer::<u32>(false),
        ),
    );
    let layout = BindGroupLayoutDescriptor::new("prism hzb visibility", &entries);
    let bind_group_layout = device.create_bind_group_layout("prism hzb visibility", &entries);
    let shader: Handle<Shader> =
        load_embedded_asset!(asset_server.as_ref(), "../shaders/hzb_visibility.wesl");
    let pipeline = cache.queue_compute_pipeline(ComputePipelineDescriptor {
        label: Some("prism hzb visibility".into()),
        layout: vec![layout.clone()],
        immediate_size: 32,
        shader,
        entry_point: Some("classify_hzb".into()),
        ..Default::default()
    });
    commands.insert_resource(HzbVisibilityPipeline {
        pipeline,
        layout,
        bind_group_layout,
    });
}

#[cfg(test)]
mod tests {
    use super::{active_count_in_range, RenderHzbCullInput};
    use bevy_render::render_resource::ShaderType;
    use bevy_asset::{uuid::Uuid, AssetId};
    use bevy_shader::{Shader, ShaderCache, ShaderCacheSource};

    fn load_source(
        _: &(),
        source: ShaderCacheSource,
        _: &bevy_shader::ValidateShader,
    ) -> Result<String, bevy_shader::ShaderCacheError> {
        match source {
            ShaderCacheSource::Wgsl(source) => Ok(source),
            ShaderCacheSource::SpirV(_) => unreachable!("HZB shader is WESL"),
        }
    }

    #[test]
    fn hzb_visibility_wesl_compiles() {
        let shader_id = AssetId::Uuid {
            uuid: Uuid::from_u128(0x5052_4953_4d48_5a42_5649_5349_4249_4c01),
        };
        let mut cache = ShaderCache::new((), load_source);
        cache.set_shader(
            shader_id,
            Shader::from_wesl(
                include_str!("../shaders/hzb_visibility.wesl"),
                "shaders/prism_hzb_visibility.wesl",
            ),
        );
        cache
            .get(0, shader_id, &[])
            .unwrap_or_else(|error| panic!("HZB visibility shader failed: {error}"));
    }

    #[test]
    fn hzb_rows_match_shader_layout() {
        assert_eq!(RenderHzbCullInput::min_size().get(), 32);
    }

    #[test]
    fn sphere_projection_is_reverse_z_and_motion_conservative() {
        let projected = super::project_sphere_to_hzb(
            bevy_math::Mat4::IDENTITY,
            bevy_math::Mat4::IDENTITY,
            bevy_math::Vec3::new(0.0, 0.0, 0.5),
            bevy_math::Vec3::new(-0.1, 0.0, 0.5),
            0.1,
            [0, 0, 100, 100],
        )
        .unwrap();
        assert!(projected.nearest_depth > 0.5);
        assert!(projected.projected_velocity >= 5.0);
        assert!(projected.uv_min[0] < projected.uv_max[0]);
    }

    #[test]
    fn active_counts_are_scoped_to_each_view_range() {
        let mut active = bevy_platform::collections::HashSet::default();
        active.extend([1, 3, 8, 9]);
        assert_eq!(active_count_in_range(&active, 0, 4), 2);
        assert_eq!(active_count_in_range(&active, 8, 4), 2);
        assert_eq!(active_count_in_range(&active, 4, 4), 0);
    }
}
