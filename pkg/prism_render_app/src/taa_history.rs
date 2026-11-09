//! Per-camera TAA history ownership and RenderGraph resolve admission.
//!
//! This module owns real RHI ping-pong textures and appends a compute resolve
//! to Prism's graph. Camera HDR, motion and reactive inputs remain owned by the
//! camera frame graph and are passed explicitly to append_resolve.

use alloc::collections::{BTreeMap, BTreeSet};
use alloc::sync::Arc;
use alloc::vec;
use alloc::vec::Vec;
use core::any::Any;

use prism_ecs::prelude::{Entity, Resource};
use prism_render_driver as rhi;
use prism_render_graph::{ImportedTexture, PassFlags, RenderGraph, TextureDesc, TextureHandle};
use prism_render_runtime::RenderRuntime;

/// A camera output namespace.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum TaaOutput {
    /// The application's primary window.
    PrimaryWindow,
    /// A native window identified by its stable application id.
    Window(u64),
    /// An application-owned texture output.
    Texture(u64),
}

/// Maximum per-frame UV displacement accepted for temporal history reuse.
/// Larger deltas are treated as cuts/teleports rather than clamped motion;
/// clamping would reproject unrelated history into the newly visible surface.
pub const MAX_TEMPORAL_REPROJECTION_UV: f32 = 0.25;

/// CPU reference result for the per-surface motion contract used by TAA.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TemporalMotionReference {
    /// Current UV minus previous UV, matching the TAA resolve convention.
    pub velocity_uv: [f32; 2],
    /// True when the resolve must not sample prior history for this surface.
    pub reject_history: bool,
}

/// Projects one local-space surface point through current/previous object and
/// camera transforms. Invalid instance/camera history, non-finite transforms,
/// points behind either camera, and teleport-sized reprojection all fail
/// closed. This CPU function is the numerical oracle for the GPU motion pass.
#[must_use]
pub fn temporal_motion_reference(
    local_position: [f32; 3],
    current_object_rows: [[f32; 4]; 3],
    previous_object_rows: [[f32; 4]; 3],
    current_clip_from_world: [[f32; 4]; 4],
    previous_clip_from_world: [[f32; 4]; 4],
    instance_history_valid: bool,
    camera_history_valid: bool,
) -> TemporalMotionReference {
    let rejected = TemporalMotionReference {
        velocity_uv: [0.0; 2],
        reject_history: true,
    };
    if !instance_history_valid || !camera_history_valid {
        return rejected;
    }
    let local = [local_position[0], local_position[1], local_position[2], 1.0];
    if !local.iter().all(|value| value.is_finite())
        || !current_object_rows
            .iter()
            .chain(previous_object_rows.iter())
            .flatten()
            .all(|value| value.is_finite())
        || !current_clip_from_world
            .iter()
            .chain(previous_clip_from_world.iter())
            .flatten()
            .all(|value| value.is_finite())
    {
        return rejected;
    }
    let current_world = affine_point(current_object_rows, local);
    let previous_world = affine_point(previous_object_rows, local);
    let current_clip = project(current_clip_from_world, current_world);
    let previous_clip = project(previous_clip_from_world, previous_world);
    if current_clip[3] <= 1.0e-8 || previous_clip[3] <= 1.0e-8 {
        return rejected;
    }
    let current_uv = [
        current_clip[0] / current_clip[3] * 0.5 + 0.5,
        current_clip[1] / current_clip[3] * -0.5 + 0.5,
    ];
    let previous_uv = [
        previous_clip[0] / previous_clip[3] * 0.5 + 0.5,
        previous_clip[1] / previous_clip[3] * -0.5 + 0.5,
    ];
    let velocity_uv = [
        current_uv[0] - previous_uv[0],
        current_uv[1] - previous_uv[1],
    ];
    if !velocity_uv.iter().all(|value| value.is_finite())
        || velocity_uv[0].abs() > MAX_TEMPORAL_REPROJECTION_UV
        || velocity_uv[1].abs() > MAX_TEMPORAL_REPROJECTION_UV
    {
        return rejected;
    }
    TemporalMotionReference {
        velocity_uv,
        reject_history: false,
    }
}

fn affine_point(rows: [[f32; 4]; 3], point: [f32; 4]) -> [f32; 4] {
    [
        rows[0].iter().zip(point).map(|(a, b)| a * b).sum(),
        rows[1].iter().zip(point).map(|(a, b)| a * b).sum(),
        rows[2].iter().zip(point).map(|(a, b)| a * b).sum(),
        1.0,
    ]
}

fn project(matrix_columns: [[f32; 4]; 4], point: [f32; 4]) -> [f32; 4] {
    core::array::from_fn(|row| {
        (0..4)
            .map(|column| matrix_columns[column][row] * point[column])
            .sum()
    })
}

/// Full isolation key for one camera output's temporal history.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct TaaHistoryKey {
    /// Generational camera identity.
    pub camera: Entity,
    /// Render destination identity.
    pub output: TaaOutput,
    /// Render extent width.
    pub width: u32,
    /// Render extent height.
    pub height: u32,
    /// HDR history format. The current resolve uses Rgba16Float.
    pub format: rhi::TextureFormat,
    /// Application-controlled reset epoch, isolated per camera/output.
    pub epoch: u64,
}

impl PartialOrd for TaaHistoryKey {
    fn partial_cmp(&self, other: &Self) -> Option<core::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for TaaHistoryKey {
    fn cmp(&self, other: &Self) -> core::cmp::Ordering {
        self.camera
            .cmp(&other.camera)
            .then_with(|| self.output.cmp(&other.output))
            .then_with(|| self.width.cmp(&other.width))
            .then_with(|| self.height.cmp(&other.height))
            .then_with(|| (self.format as u8).cmp(&(other.format as u8)))
            .then_with(|| self.epoch.cmp(&other.epoch))
    }
}

impl TaaHistoryKey {
    /// Creates a key for a two-dimensional HDR output.
    #[must_use]
    pub const fn new(
        camera: Entity,
        output: TaaOutput,
        width: u32,
        height: u32,
        format: rhi::TextureFormat,
    ) -> Self {
        Self {
            camera,
            output,
            width,
            height,
            format,
            epoch: 0,
        }
    }

    /// Sets the temporal reset epoch.
    #[must_use]
    pub const fn with_epoch(mut self, epoch: u64) -> Self {
        self.epoch = epoch;
        self
    }

    fn extent(self) -> rhi::Extent3d {
        rhi::Extent3d::new_2d(self.width, self.height)
    }

    /// Rebinds an extracted camera key to the concrete output currently being
    /// submitted by a host. This is required for PrimaryWindow routing, whose
    /// resolved window identity is selected by the presentation layer.
    #[must_use]
    pub const fn with_output(mut self, output: TaaOutput) -> Self {
        self.output = output;
        self
    }
}

/// TAA setup failure detected before graph execution.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TaaHistoryError {
    /// History/current output dimensions must be non-zero.
    EmptyExtent,
    /// This resolve requires linear Rgba16Float HDR input/output.
    UnsupportedFormat(rhi::TextureFormat),
    /// The requested extent exceeds the device's 2D texture limit.
    ExtentExceedsDeviceLimit { width: u32, height: u32, limit: u32 },
    /// A frame must be newer than the prior successful resolve.
    FrameDidNotAdvance { previous: u64, current: u64 },
    /// A resolve for this camera/output is already awaiting submission.
    SubmissionPending,
}

/// The result of admitting one camera resolve into a graph.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TaaResolveOutput {
    /// The resolved HDR history image for the current frame.
    pub texture: TextureHandle,
    /// Whether a completed previous history sample was used.
    pub history_valid: bool,
    /// Whether this frame seeded history from the current image.
    pub reset: bool,
    /// False indicates deterministic budget fallback to the unfiltered current image.
    pub admitted: bool,
}

struct HistoryGpuResources {
    device: Arc<dyn rhi::RenderDevice + Send + Sync>,
    textures: [rhi::TextureId; 2],
    views: [rhi::TextureViewId; 2],
}

impl Drop for HistoryGpuResources {
    fn drop(&mut self) {
        for (&view, &texture) in self.views.iter().zip(&self.textures) {
            self.device.destroy_texture_view(view);
            self.device.destroy_texture(texture);
        }
    }
}

struct HistoryOwner {
    resources: Arc<HistoryGpuResources>,
    generation: u64,
    states: [rhi::TextureState; 2],
    read_index: usize,
    epoch: u64,
    last_frame: Option<u64>,
    last_used: u64,
    retention: u32,
    completion: u64,
    valid: bool,
    bytes: u64,
}

struct RetiredOwner {
    owner: HistoryOwner,
    retire_after: u64,
}

struct ResolvePipeline {
    device: Arc<dyn rhi::RenderDevice + Send + Sync>,
    generation: u64,
    layout: rhi::BindGroupLayoutId,
    pipeline_layout: rhi::PipelineLayoutId,
    shader: rhi::ShaderModuleId,
    pipeline: rhi::ComputePipelineId,
    motion_layout: rhi::BindGroupLayoutId,
    motion_pipeline_layout: rhi::PipelineLayoutId,
    motion_shader: rhi::ShaderModuleId,
    motion_pipeline: rhi::ComputePipelineId,
}

impl ResolvePipeline {
    fn create(device: Arc<dyn rhi::RenderDevice + Send + Sync>, generation: u64) -> Self {
        let entries = [
            rhi::BindGroupLayoutEntry {
                binding: 0,
                visibility: rhi::ShaderStages::COMPUTE,
                ty: rhi::BindingType::Texture {
                    sample_type: rhi::TextureSampleType::UnfilterableFloat,
                    view_dimension: rhi::TextureViewDimension::D2,
                    multisampled: false,
                },
                count: None,
            },
            rhi::BindGroupLayoutEntry {
                binding: 1,
                visibility: rhi::ShaderStages::COMPUTE,
                ty: rhi::BindingType::Texture {
                    sample_type: rhi::TextureSampleType::UnfilterableFloat,
                    view_dimension: rhi::TextureViewDimension::D2,
                    multisampled: false,
                },
                count: None,
            },
            rhi::BindGroupLayoutEntry {
                binding: 2,
                visibility: rhi::ShaderStages::COMPUTE,
                ty: rhi::BindingType::Texture {
                    sample_type: rhi::TextureSampleType::UnfilterableFloat,
                    view_dimension: rhi::TextureViewDimension::D2,
                    multisampled: false,
                },
                count: None,
            },
            rhi::BindGroupLayoutEntry {
                binding: 3,
                visibility: rhi::ShaderStages::COMPUTE,
                ty: rhi::BindingType::Texture {
                    sample_type: rhi::TextureSampleType::Float,
                    view_dimension: rhi::TextureViewDimension::D2,
                    multisampled: false,
                },
                count: None,
            },
            rhi::BindGroupLayoutEntry {
                binding: 4,
                visibility: rhi::ShaderStages::COMPUTE,
                ty: rhi::BindingType::StorageTexture {
                    access: rhi::StorageTextureAccess::WriteOnly,
                    format: rhi::TextureFormat::Rgba16Float,
                    view_dimension: rhi::TextureViewDimension::D2,
                },
                count: None,
            },
        ];
        let layout = device.create_bind_group_layout(&rhi::BindGroupLayoutDescriptor {
            label: Some("Prism camera TAA bindings".into()),
            entries: entries.to_vec(),
        });
        let pipeline_layout = device.create_pipeline_layout(&rhi::PipelineLayoutDescriptor {
            label: Some("Prism camera TAA pipeline layout".into()),
            bind_group_layouts: vec![layout],
            push_constant_ranges: vec![],
        });
        let shader =
            device.create_shader_module(&rhi::ShaderModuleDescriptor::wgsl(TAA_RESOLVE_WGSL));
        let pipeline = device.create_compute_pipeline(&rhi::ComputePipelineDescriptor {
            label: Some("Prism camera TAA resolve".into()),
            layout: Some(pipeline_layout),
            module: shader,
            entry_point: "resolve_taa".into(),
        });
        let motion_entries = [
            rhi::BindGroupLayoutEntry {
                binding: 0,
                visibility: rhi::ShaderStages::COMPUTE,
                ty: rhi::BindingType::Texture {
                    sample_type: rhi::TextureSampleType::UnfilterableFloat,
                    view_dimension: rhi::TextureViewDimension::D2,
                    multisampled: false,
                },
                count: None,
            },
            rhi::BindGroupLayoutEntry {
                binding: 1,
                visibility: rhi::ShaderStages::COMPUTE,
                ty: rhi::BindingType::Texture {
                    sample_type: rhi::TextureSampleType::UnfilterableFloat,
                    view_dimension: rhi::TextureViewDimension::D2,
                    multisampled: false,
                },
                count: None,
            },
            rhi::BindGroupLayoutEntry {
                binding: 2,
                visibility: rhi::ShaderStages::COMPUTE,
                ty: rhi::BindingType::Texture {
                    sample_type: rhi::TextureSampleType::UnfilterableFloat,
                    view_dimension: rhi::TextureViewDimension::D2,
                    multisampled: false,
                },
                count: None,
            },
            rhi::BindGroupLayoutEntry {
                binding: 3,
                visibility: rhi::ShaderStages::COMPUTE,
                ty: rhi::BindingType::Texture {
                    sample_type: rhi::TextureSampleType::UnfilterableFloat,
                    view_dimension: rhi::TextureViewDimension::D2,
                    multisampled: false,
                },
                count: None,
            },
            rhi::BindGroupLayoutEntry {
                binding: 4,
                visibility: rhi::ShaderStages::COMPUTE,
                ty: rhi::BindingType::StorageTexture {
                    access: rhi::StorageTextureAccess::WriteOnly,
                    format: crate::target::TAA_MOTION_FORMAT,
                    view_dimension: rhi::TextureViewDimension::D2,
                },
                count: None,
            },
            rhi::BindGroupLayoutEntry {
                binding: 5,
                visibility: rhi::ShaderStages::COMPUTE,
                ty: rhi::BindingType::StorageTexture {
                    access: rhi::StorageTextureAccess::WriteOnly,
                    format: crate::target::TAA_REACTIVE_FORMAT,
                    view_dimension: rhi::TextureViewDimension::D2,
                },
                count: None,
            },
        ];
        let motion_layout = device.create_bind_group_layout(&rhi::BindGroupLayoutDescriptor {
            label: Some("Prism camera TAA temporal-input bindings".into()),
            entries: motion_entries.to_vec(),
        });
        let motion_pipeline_layout =
            device.create_pipeline_layout(&rhi::PipelineLayoutDescriptor {
                label: Some("Prism camera TAA temporal-input pipeline layout".into()),
                bind_group_layouts: vec![motion_layout],
                push_constant_ranges: vec![rhi::PushConstantRange {
                    stages: rhi::ShaderStages::COMPUTE,
                    start: 0,
                    end: 128,
                }],
            });
        let motion_shader = device
            .create_shader_module(&rhi::ShaderModuleDescriptor::wgsl(TAA_TEMPORAL_INPUT_WGSL));
        let motion_pipeline = device.create_compute_pipeline(&rhi::ComputePipelineDescriptor {
            label: Some("Prism camera TAA temporal-input producer".into()),
            layout: Some(motion_pipeline_layout),
            module: motion_shader,
            entry_point: "produce_temporal_inputs".into(),
        });
        Self {
            device,
            generation,
            layout,
            pipeline_layout,
            shader,
            pipeline,
            motion_layout,
            motion_pipeline_layout,
            motion_shader,
            motion_pipeline,
        }
    }
}

impl Drop for ResolvePipeline {
    fn drop(&mut self) {
        self.device.destroy_compute_pipeline(self.pipeline);
        self.device.destroy_pipeline_layout(self.pipeline_layout);
        self.device.destroy_bind_group_layout(self.layout);
        self.device.destroy_shader_module(self.shader);
        self.device.destroy_compute_pipeline(self.motion_pipeline);
        self.device
            .destroy_pipeline_layout(self.motion_pipeline_layout);
        self.device.destroy_bind_group_layout(self.motion_layout);
        self.device.destroy_shader_module(self.motion_shader);
    }
}

struct BindGroupLease {
    device: Arc<dyn rhi::RenderDevice + Send + Sync>,
    bind_group: rhi::BindGroupId,
}
impl Drop for BindGroupLease {
    fn drop(&mut self) {
        self.device.destroy_bind_group(self.bind_group);
    }
}

/// Render-world owner of independent TAA histories and the resolve pipeline.
///
/// Call append_resolve while constructing a frame graph, then mark_submitted
/// only after successful queue submission. collect_completed uses the runtime's
/// completion watermark and never stalls the render thread.
pub struct TaaHistoryCache {
    active: BTreeMap<TaaHistoryKey, HistoryOwner>,
    retired: Vec<RetiredOwner>,
    pipeline: Option<Arc<ResolvePipeline>>,
    pending: BTreeMap<TaaHistoryKey, PendingResolve>,
    budget_bytes: u64,
    resident_bytes: u64,
}

impl Default for TaaHistoryCache {
    fn default() -> Self {
        Self {
            active: BTreeMap::new(),
            retired: Vec::new(),
            pipeline: None,
            pending: BTreeMap::new(),
            budget_bytes: 256 * 1024 * 1024,
            resident_bytes: 0,
        }
    }
}

struct PendingResolve {
    frame: u64,
    epoch: u64,
    read_index: usize,
    write_index: usize,
    valid: bool,
    retention: u32,
    /// Whether this pending resolve created the active owner. If submission is
    /// cancelled, that owner has never been visible to the GPU and must be
    /// released immediately rather than aging as a live history.
    created_owner: bool,
}

impl Resource for TaaHistoryCache {}

impl Drop for TaaHistoryCache {
    fn drop(&mut self) {
        // Dropping active/retired owners releases their Arc references. A
        // graph that is still recording or in flight keeps its own owner pin,
        // so shutdown cannot destroy history images prematurely.
        self.pending.clear();
        self.active.clear();
        self.retired.clear();
        self.pipeline.take();
    }
}

impl TaaHistoryCache {
    /// Number of resident camera/output history pairs.
    #[must_use]
    pub fn len(&self) -> usize {
        self.active.len()
    }

    /// Whether no history is resident.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.active.is_empty()
    }

    /// Number of retired owners held for in-flight GPU work.
    #[must_use]
    pub fn retired_len(&self) -> usize {
        self.retired.len()
    }

    /// Bytes retained by active plus completion-quarantined history owners.
    #[must_use]
    pub const fn resident_bytes(&self) -> u64 {
        self.resident_bytes
    }

    /// Sets the deterministic memory budget for persistent history pairs.
    pub fn set_budget_bytes(&mut self, budget_bytes: u64) {
        self.budget_bytes = budget_bytes;
    }

    /// Append the RenderApp-owned temporal input producer. The position planes
    /// and coverage channel are the actual outputs of the camera's opaque
    /// raster pass; no synthetic zero motion or reactive texture is admitted.
    /// The current RenderApp ABI does not expose a per-pixel generational scene
    /// id, so the producer emits camera reprojection motion and conservatively
    /// rejects covered pixels while the camera matrices change. This keeps
    /// dynamic surfaces from consuming a camera-only vector as object motion.
    #[allow(clippy::too_many_arguments)]
    pub fn append_temporal_inputs<I: rhi::Instance>(
        &mut self,
        graph: &mut RenderGraph,
        runtime: &RenderRuntime<I>,
        width: u32,
        height: u32,
        position_x: TextureHandle,
        position_y: TextureHandle,
        position_z: TextureHandle,
        normal: TextureHandle,
        motion: TextureHandle,
        reactive: TextureHandle,
        clip_from_world: [[f32; 4]; 4],
        previous_clip_from_world: [[f32; 4]; 4],
    ) where
        I::Device: Send + Sync + 'static,
    {
        self.ensure_pipeline(runtime);
        let pipeline = self.pipeline.as_ref().expect("TAA pipeline created");
        let bind_group = graph.create_texture_bind_group(
            pipeline.motion_layout,
            vec![
                (0, position_x),
                (1, position_y),
                (2, position_z),
                (3, normal),
                (4, motion),
                (5, reactive),
            ],
            Some("Prism camera TAA temporal inputs".into()),
        );
        let mut matrices = [0.0_f32; 32];
        for (dst, column) in matrices[..16].chunks_exact_mut(4).zip(clip_from_world) {
            dst.copy_from_slice(&column);
        }
        for (dst, column) in matrices[16..]
            .chunks_exact_mut(4)
            .zip(previous_clip_from_world)
        {
            dst.copy_from_slice(&column);
        }
        let pipeline_id = pipeline.motion_pipeline;
        let x = width.div_ceil(TAA_WORKGROUP_EDGE);
        let y = height.div_ceil(TAA_WORKGROUP_EDGE);
        graph.add_compute_pass(
            "prism-camera-taa-temporal-inputs",
            PassFlags::SIDE_EFFECT,
            |builder| {
                builder.sample(position_x);
                builder.sample(position_y);
                builder.sample(position_z);
                builder.sample(normal);
                builder.write_storage_texture(motion);
                builder.write_storage_texture(reactive);
                move |ctx| {
                    let bind_group = ctx.texture_bind_group(bind_group.clone());
                    ctx.set_compute_pipeline(pipeline_id)
                        .set_compute_bind_group(0, bind_group, Vec::new())
                        .set_compute_push_constants(0, bytemuck::cast_slice(&matrices).to_vec())
                        .dispatch(x, y, 1);
                }
            },
        );
    }

    /// Append a real TAA resolve pass. The graph handles and RHI views must
    /// refer to the same current, motion and reactive textures. history_frames
    /// is the number of idle frames to retain this owner, not a texture count.
    #[allow(clippy::too_many_arguments)]
    pub fn append_resolve<I: rhi::Instance>(
        &mut self,
        graph: &mut RenderGraph,
        runtime: &RenderRuntime<I>,
        key: TaaHistoryKey,
        history_epoch: u64,
        camera_cut: bool,
        frame: u64,
        history_frames: u32,
        current: TextureHandle,
        current_view: rhi::TextureViewId,
        motion: TextureHandle,
        motion_view: rhi::TextureViewId,
        reactive: TextureHandle,
        reactive_view: rhi::TextureViewId,
    ) -> Result<TaaResolveOutput, TaaHistoryError>
    where
        I::Device: Send + Sync + 'static,
    {
        validate_key(&*runtime.device(), key)?;
        let key = key.with_epoch(history_epoch);
        self.ensure_pipeline(runtime);
        let generation = runtime.device_generation();
        let device: Arc<dyn rhi::RenderDevice + Send + Sync> = runtime.device_handle();
        self.collect_completed(generation, runtime.completed_value());
        let key_changed = self
            .active
            .keys()
            .any(|other| other.camera == key.camera && other.output == key.output && *other != key);
        if key_changed {
            let conflicts: Vec<_> = self
                .active
                .keys()
                .copied()
                .filter(|other| {
                    other.camera == key.camera && other.output == key.output && *other != key
                })
                .collect();
            for old_key in conflicts {
                if let Some(owner) = self.active.remove(&old_key) {
                    self.retire_owner(owner);
                }
            }
        }
        // Remove stale-device owners before admission so their bytes remain
        // quarantined in the budget until the old generation is known idle.
        let mismatch = self.active.get(&key).is_some_and(|owner| {
            owner.generation != generation || !Arc::ptr_eq(&owner.resources.device, &device)
        });
        if mismatch {
            if let Some(owner) = self.active.remove(&key) {
                self.retire_owner(owner);
            }
        }
        let owner_bytes = history_bytes(key);
        if !self.active.contains_key(&key) && !self.ensure_budget(key, owner_bytes) {
            return Ok(TaaResolveOutput {
                texture: current,
                history_valid: false,
                reset: true,
                admitted: false,
            });
        }
        let created_owner = if !self.active.contains_key(&key) {
            self.active.insert(
                key,
                create_owner(
                    Arc::clone(&device),
                    generation,
                    key,
                    history_epoch,
                    frame,
                    history_frames,
                ),
            );
            self.resident_bytes = self.resident_bytes.saturating_add(owner_bytes);
            true
        } else {
            false
        };
        if self.pending.contains_key(&key) {
            return Err(TaaHistoryError::SubmissionPending);
        }
        let owner = self.active.get_mut(&key).expect("owner inserted above");
        if let Some(previous) = owner.last_frame {
            if frame <= previous {
                return Err(TaaHistoryError::FrameDidNotAdvance {
                    previous,
                    current: frame,
                });
            }
        }
        let contiguous = owner
            .last_frame
            .and_then(|previous| previous.checked_add(1))
            == Some(frame);
        if owner.epoch != history_epoch || camera_cut || !contiguous {
            owner.valid = false;
            owner.epoch = history_epoch;
        }
        let valid = owner.valid && contiguous && !camera_cut;
        let read_index = owner.read_index;
        let write_index = 1 - read_index;
        let desc = TextureDesc::color(key.format).with_usage(
            rhi::TextureUsages::STORAGE_BINDING.union(rhi::TextureUsages::TEXTURE_BINDING),
        );
        let history_read = graph.import_texture(
            "taa-history-read",
            desc.clone(),
            imported(owner, read_index),
        );
        let history_write =
            graph.import_texture("taa-history-write", desc, imported(owner, write_index));
        let history_input = if valid { history_read } else { current };
        let bind_group = graph.create_texture_bind_group(
            self.pipeline.as_ref().expect("pipeline created").layout,
            vec![
                (0, current),
                (1, history_input),
                (2, motion),
                (3, reactive),
                (4, history_write),
            ],
            Some("Prism per-camera TAA resolve".into()),
        );
        let history_owner: Arc<dyn Any + Send + Sync> = owner.resources.clone();
        graph.retain_external_owner(history_owner);
        let pipeline_owner = Arc::clone(self.pipeline.as_ref().expect("pipeline created"));
        let pipeline = pipeline_owner.pipeline;
        let pipeline_pin: Arc<dyn Any + Send + Sync> = pipeline_owner;
        graph.retain_external_owner(pipeline_pin);
        let (x, y) = (
            key.width.div_ceil(TAA_WORKGROUP_EDGE),
            key.height.div_ceil(TAA_WORKGROUP_EDGE),
        );
        graph.add_compute_pass(
            "prism-camera-taa-resolve",
            PassFlags::SIDE_EFFECT,
            move |builder| {
                builder.sample(current);
                if valid {
                    builder.sample(history_read);
                }
                builder.sample(motion);
                builder.sample(reactive);
                builder.write_storage_texture(history_write);
                move |ctx| {
                    let bind_group = ctx.texture_bind_group(bind_group.clone());
                    ctx.set_compute_pipeline(pipeline);
                    ctx.set_compute_bind_group(0, bind_group, Vec::new());
                    ctx.dispatch(x, y, 1);
                }
            },
        );
        self.pending.insert(
            key,
            PendingResolve {
                frame,
                epoch: history_epoch,
                read_index,
                write_index,
                valid,
                retention: history_frames,
                created_owner,
            },
        );
        Ok(TaaResolveOutput {
            texture: history_write,
            history_valid: valid,
            reset: !valid,
            admitted: true,
        })
    }

    fn ensure_pipeline<I: rhi::Instance>(&mut self, runtime: &RenderRuntime<I>)
    where
        I::Device: Send + Sync + 'static,
    {
        let generation = runtime.device_generation();
        let device: Arc<dyn rhi::RenderDevice + Send + Sync> = runtime.device_handle();
        if self
            .pipeline
            .as_ref()
            .is_none_or(|p| p.generation != generation || !Arc::ptr_eq(&p.device, &device))
        {
            self.pipeline = Some(Arc::new(ResolvePipeline::create(
                Arc::clone(&device),
                generation,
            )));
        }
    }

    /// Confirm that this camera's resolve was included in a successful frame submission.
    pub fn mark_submitted(
        &mut self,
        key: TaaHistoryKey,
        frame: u64,
        completion_value: u64,
    ) -> bool {
        // append_resolve accepts the camera epoch separately for compatibility
        // with callers that construct a stable key once and advance that epoch
        // in extracted view data. The pending entry is always authoritative.
        let pending_key = self
            .pending
            .keys()
            .find(|pending_key| {
                pending_key.camera == key.camera
                    && pending_key.output == key.output
                    && pending_key.width == key.width
                    && pending_key.height == key.height
                    && pending_key.format == key.format
            })
            .copied();
        let Some(pending_key) = pending_key else {
            return false;
        };
        let Some(pending) = self.pending.get(&pending_key) else {
            return false;
        };
        if pending.frame != frame {
            return false;
        }
        let pending = self
            .pending
            .remove(&pending_key)
            .expect("pending resolve was checked above");
        let Some(owner) = self.active.get_mut(&pending_key) else {
            return false;
        };
        owner.states[pending.read_index] =
            rhi::TextureState::shader_read(rhi::PipelineStages::COMPUTE_SHADER);
        owner.states[pending.write_index] =
            rhi::TextureState::storage_write(rhi::PipelineStages::COMPUTE_SHADER);
        owner.read_index = pending.write_index;
        owner.epoch = pending.epoch;
        owner.valid = true;
        owner.last_frame = Some(frame);
        owner.last_used = frame;
        owner.retention = pending.retention;
        owner.completion = completion_value;
        true
    }

    /// Confirm precisely the history entries admitted to one submitted graph.
    pub fn mark_graph_submitted(
        &mut self,
        keys: &[TaaHistoryKey],
        frame: u64,
        completion_value: u64,
    ) -> usize {
        keys.iter()
            .filter(|key| self.mark_submitted(**key, frame, completion_value))
            .count()
    }

    /// Mark all pending camera histories in one output graph after its exact
    /// successful submission receipt is known.
    pub fn mark_output_submitted(
        &mut self,
        output: TaaOutput,
        frame: u64,
        completion_value: u64,
    ) -> usize {
        let keys: Vec<_> = self
            .pending
            .iter()
            .filter_map(|(key, pending)| {
                (key.output == output && pending.frame == frame).then_some(*key)
            })
            .collect();
        self.mark_graph_submitted(&keys, frame, completion_value)
    }

    /// Cancel pending histories belonging to an output graph that did not
    /// reach queue submission.
    pub fn cancel_output(&mut self, output: TaaOutput, frame: u64) -> usize {
        let keys: Vec<_> = self
            .pending
            .iter()
            .filter_map(|(key, pending)| {
                (key.output == output && pending.frame == frame).then_some(*key)
            })
            .collect();
        self.cancel_graph(&keys, frame)
    }

    /// Cancel graph-local pending entries when graph finalization/submit fails.
    pub fn cancel_graph(&mut self, keys: &[TaaHistoryKey], frame: u64) -> usize {
        keys.iter()
            .filter(|key| self.cancel_pending(**key, frame))
            .count()
    }

    /// Cancel an unsubmitted resolve after graph compilation or submission failed.
    pub fn cancel_pending(&mut self, key: TaaHistoryKey, frame: u64) -> bool {
        let pending_key = self
            .pending
            .keys()
            .find(|pending_key| {
                pending_key.camera == key.camera
                    && pending_key.output == key.output
                    && pending_key.width == key.width
                    && pending_key.height == key.height
                    && pending_key.format == key.format
            })
            .copied();
        let Some(pending_key) = pending_key else {
            return false;
        };
        let Some(pending) = self.pending.get(&pending_key) else {
            return false;
        };
        if pending.frame != frame {
            return false;
        }
        let pending = self
            .pending
            .remove(&pending_key)
            .expect("pending resolve was checked above");
        if pending.created_owner {
            if let Some(owner) = self.active.remove(&pending_key) {
                self.resident_bytes = self.resident_bytes.saturating_sub(owner.bytes);
                drop(owner);
            }
        }
        true
    }

    /// Retain live keys, age inactive keys using history_frames, and reclaim
    /// retired GPU textures only after the submitting queue confirms completion.
    pub fn retain_live(
        &mut self,
        live: &BTreeSet<TaaHistoryKey>,
        frame: u64,
        generation: u64,
        completed: u64,
    ) {
        let stale: Vec<_> = self
            .active
            .iter()
            .filter_map(|(key, owner)| {
                (!live.contains(key)
                    && !self.pending.contains_key(key)
                    && (owner.generation != generation
                        || frame.saturating_sub(owner.last_used) > u64::from(owner.retention)))
                .then_some(*key)
            })
            .collect();
        for key in stale {
            if let Some(owner) = self.active.remove(&key) {
                self.retire_owner(owner);
            }
        }
        let pending: Vec<_> = self
            .pending
            .iter()
            .filter_map(|(key, pending)| {
                (!live.contains(key)
                    && frame.saturating_sub(pending.frame) > u64::from(pending.retention))
                .then_some((*key, pending.frame))
            })
            .collect();
        for (key, pending_frame) in pending {
            self.cancel_pending(key, pending_frame);
        }
        self.collect_completed(generation, completed);
    }

    /// Reclaim completion-safe retired owners and invalidate active owners from
    /// replaced device generations. No wait is issued.
    pub fn collect_completed(&mut self, generation: u64, completed: u64) {
        let stale: Vec<_> = self
            .active
            .iter()
            .filter_map(|(key, owner)| (owner.generation != generation).then_some(*key))
            .collect();
        for key in stale {
            if let Some(owner) = self.active.remove(&key) {
                self.retire_owner(owner);
            }
        }
        let mut keep = Vec::with_capacity(self.retired.len());
        for retired in core::mem::take(&mut self.retired) {
            // RenderRuntime increments its generation only after the old
            // queue has been drained. An owner from an older generation is
            // therefore completion-safe to destroy when this method observes
            // the new generation; completion values are not comparable across
            // generations and must not be used for that decision.
            if retired.owner.generation != generation || retired.retire_after <= completed {
                self.resident_bytes = self.resident_bytes.saturating_sub(retired.owner.bytes);
                drop(retired.owner);
            } else {
                keep.push(retired);
            }
        }
        self.retired = keep;
    }

    fn retire_owner(&mut self, owner: HistoryOwner) {
        let retire_after = owner.completion;
        self.retired.push(RetiredOwner {
            owner,
            retire_after,
        });
    }

    fn ensure_budget(&mut self, key: TaaHistoryKey, bytes: u64) -> bool {
        if bytes > self.budget_bytes {
            return false;
        }
        while self.resident_bytes.saturating_add(bytes) > self.budget_bytes {
            let candidate = self
                .active
                .iter()
                .filter(|(candidate, _)| {
                    **candidate != key && !self.pending.contains_key(candidate)
                })
                .min_by_key(|(candidate, owner)| (owner.last_used, **candidate))
                .map(|(candidate, _)| *candidate);
            let Some(candidate) = candidate else {
                return false;
            };
            if let Some(owner) = self.active.remove(&candidate) {
                self.retire_owner(owner);
            }
        }
        true
    }
}

fn validate_key(device: &dyn rhi::RenderDevice, key: TaaHistoryKey) -> Result<(), TaaHistoryError> {
    if key.width == 0 || key.height == 0 {
        return Err(TaaHistoryError::EmptyExtent);
    }
    if key.format != rhi::TextureFormat::Rgba16Float {
        return Err(TaaHistoryError::UnsupportedFormat(key.format));
    }
    let limit = device.capabilities().limits.max_texture_dimension_2d;
    if key.width > limit || key.height > limit {
        return Err(TaaHistoryError::ExtentExceedsDeviceLimit {
            width: key.width,
            height: key.height,
            limit,
        });
    }
    Ok(())
}

fn create_owner(
    device: Arc<dyn rhi::RenderDevice + Send + Sync>,
    generation: u64,
    key: TaaHistoryKey,
    epoch: u64,
    frame: u64,
    retention: u32,
) -> HistoryOwner {
    let descriptor = rhi::TextureDescriptor::new_2d(
        key.extent(),
        key.format,
        rhi::TextureUsages::STORAGE_BINDING
            .union(rhi::TextureUsages::TEXTURE_BINDING)
            .union(rhi::TextureUsages::COPY_SRC),
    );
    let mut textures = Vec::with_capacity(2);
    let mut views = Vec::with_capacity(2);
    for _ in 0..2 {
        let texture = device.create_texture(&descriptor);
        let view = device.create_texture_view(texture, &rhi::TextureViewDescriptor::default());
        textures.push(texture);
        views.push(view);
    }
    HistoryOwner {
        resources: Arc::new(HistoryGpuResources {
            device,
            textures: textures.try_into().ok().expect("two history textures"),
            views: views.try_into().ok().expect("two history views"),
        }),
        generation,
        states: [rhi::TextureState::initial(); 2],
        read_index: 0,
        epoch,
        last_frame: None,
        last_used: frame,
        retention,
        completion: 0,
        valid: false,
        bytes: history_bytes(key),
    }
}

fn history_bytes(key: TaaHistoryKey) -> u64 {
    u64::from(key.width)
        .saturating_mul(u64::from(key.height))
        .saturating_mul(8)
        .saturating_mul(2)
}

fn imported(owner: &HistoryOwner, index: usize) -> ImportedTexture {
    ImportedTexture {
        texture: owner.resources.textures[index],
        default_view: owner.resources.views[index],
        entry_state: owner.states[index],
        exit_state: rhi::TextureState::new(
            rhi::PipelineStages::COMPUTE_SHADER,
            rhi::Accesses::SHADER_READ.union(rhi::Accesses::SHADER_WRITE),
            rhi::TextureLayout::General,
        ),
        mip_levels: 1,
        array_layers: 1,
    }
}

const TAA_WORKGROUP_EDGE: u32 = 8;

const TAA_RESOLVE_WGSL: &str = r#"
@group(0) @binding(0) var current_color: texture_2d<f32>;
@group(0) @binding(1) var history_color: texture_2d<f32>;
@group(0) @binding(2) var motion_vectors: texture_2d<f32>;
@group(0) @binding(3) var reactive_mask: texture_2d<f32>;
@group(0) @binding(4) var resolved_history: texture_storage_2d<rgba16float, write>;
fn to_ycocg(rgb: vec3<f32>) -> vec3<f32> {
    return vec3<f32>(0.25 * rgb.r + 0.5 * rgb.g + 0.25 * rgb.b, 0.5 * rgb.r - 0.5 * rgb.b, -0.25 * rgb.r + 0.5 * rgb.g - 0.25 * rgb.b);
}
fn from_ycocg(v: vec3<f32>) -> vec3<f32> { return vec3<f32>(v.x + v.y - v.z, v.x + v.z, v.x - v.y - v.z); }
@compute @workgroup_size(8, 8, 1)
fn resolve_taa(@builtin(global_invocation_id) gid: vec3<u32>) {
    let dims = textureDimensions(current_color);
    if (gid.x >= dims.x || gid.y >= dims.y) { return; }
    let p = vec2<i32>(gid.xy);
    let dims_i = vec2<i32>(dims);
    let current = textureLoad(current_color, p, 0).rgb;
    let motion = textureLoad(motion_vectors, p, 0).xy;
    let previous_p = vec2<i32>(round(vec2<f32>(p) - motion * vec2<f32>(dims)));
    let in_bounds = all(previous_p >= vec2<i32>(0)) && all(previous_p < dims_i);
    var lo = vec3<f32>(1.0e30); var hi = vec3<f32>(-1.0e30);
    for (var y = -1; y <= 1; y += 1) { for (var x = -1; x <= 1; x += 1) {
        let q = clamp(p + vec2<i32>(x, y), vec2<i32>(0), dims_i - vec2<i32>(1));
        let sample = to_ycocg(textureLoad(current_color, q, 0).rgb); lo = min(lo, sample); hi = max(hi, sample);
    }}
    let reactive = clamp(textureLoad(reactive_mask, p, 0).r, 0.0, 1.0);
    var result = current;
    if (in_bounds && reactive < 1.0) {
        let history = to_ycocg(textureLoad(history_color, previous_p, 0).rgb);
        let clipped = clamp(history, lo, hi);
        let weight = 0.9 * (1.0 - reactive) / (1.0 + max(clipped.x, 0.0));
        result = from_ycocg((to_ycocg(current) + clipped * weight) / (1.0 + weight));
    }
    textureStore(resolved_history, p, vec4<f32>(result, 1.0));
}
"#;

const TAA_TEMPORAL_INPUT_WGSL: &str = r#"
struct Matrices { current: mat4x4<f32>, previous: mat4x4<f32> };
var<immediate> matrices: Matrices;
@group(0) @binding(0) var position_x: texture_2d<f32>;
@group(0) @binding(1) var position_y: texture_2d<f32>;
@group(0) @binding(2) var position_z: texture_2d<f32>;
@group(0) @binding(3) var coverage: texture_2d<f32>;
@group(0) @binding(4) var motion_out: texture_storage_2d<rgba16float, write>;
@group(0) @binding(5) var reactive_out: texture_storage_2d<r32float, write>;

@compute @workgroup_size(8, 8, 1)
fn produce_temporal_inputs(@builtin(global_invocation_id) gid: vec3<u32>) {
    let dims = textureDimensions(position_x);
    if (gid.x >= dims.x || gid.y >= dims.y) { return; }
    let pixel = vec2<i32>(gid.xy);
    let position = vec3<f32>(
        textureLoad(position_x, pixel, 0).x,
        textureLoad(position_y, pixel, 0).x,
        textureLoad(position_z, pixel, 0).x,
    );
    let covered = textureLoad(coverage, pixel, 0).w > 0.0;
    let current_clip = matrices.current * vec4<f32>(position, 1.0);
    let previous_clip = matrices.previous * vec4<f32>(position, 1.0);
    let usable = covered && current_clip.w > 1.0e-8 && previous_clip.w > 1.0e-8;
    var velocity = vec2<f32>(0.0);
    var reactive = 1.0;
    if (usable) {
        let current_uv = current_clip.xy / current_clip.w * vec2<f32>(0.5, -0.5) + vec2<f32>(0.5);
        let previous_uv = previous_clip.xy / previous_clip.w * vec2<f32>(0.5, -0.5) + vec2<f32>(0.5);
        velocity = current_uv - previous_uv;
        // World-position alone cannot establish that the material/object was
        // unchanged. Reject history for covered pixels when reprojection moves;
        // uncovered background still receives valid camera-motion vectors.
        reactive = select(0.0, 1.0, length(velocity) > 1.0e-6);
    }
    textureStore(motion_out, pixel, vec4<f32>(velocity, 0.0, 0.0));
    textureStore(reactive_out, pixel, vec4<f32>(reactive, 0.0, 0.0, 0.0));
}
"#;

#[cfg(test)]
mod tests {
    use super::*;
    use prism_render_driver::{Extent3d, Instance as _, RenderDevice as _};
    use prism_render_driver_null::NullInstance;

    fn entity(index: u32, generation: u32) -> Entity {
        Entity::from_bits((u64::from(generation) << 32) | u64::from(index)).expect("entity")
    }
    fn key(camera: Entity, output: TaaOutput, width: u32) -> TaaHistoryKey {
        TaaHistoryKey::new(camera, output, width, 32, rhi::TextureFormat::Rgba16Float)
    }

    fn identity_affine() -> [[f32; 4]; 3] {
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
        ]
    }

    fn identity_clip() -> [[f32; 4]; 4] {
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    }

    #[test]
    fn motion_reference_tracks_object_motion_with_a_static_camera() {
        let mut current = identity_affine();
        let mut previous = identity_affine();
        current[0][3] = 0.2;
        previous[0][3] = -0.1;
        let result = temporal_motion_reference(
            [0.25, -0.5, 0.0],
            current,
            previous,
            identity_clip(),
            identity_clip(),
            true,
            true,
        );
        assert!(!result.reject_history);
        assert!((result.velocity_uv[0] - 0.15).abs() < 1.0e-6);
        assert!(result.velocity_uv[1].abs() < 1.0e-6);
    }

    #[test]
    fn motion_reference_combines_camera_and_object_motion() {
        let mut current_object = identity_affine();
        let mut previous_object = identity_affine();
        current_object[0][3] = 0.05;
        previous_object[0][3] = -0.05;
        let mut current_camera = identity_clip();
        let mut previous_camera = identity_clip();
        current_camera[3][0] = -0.02;
        previous_camera[3][0] = 0.03;
        let result = temporal_motion_reference(
            [0.1, 0.0, 0.0],
            current_object,
            previous_object,
            current_camera,
            previous_camera,
            true,
            true,
        );
        assert!(!result.reject_history);
        assert!((result.velocity_uv[0] - 0.025).abs() < 1.0e-6);
    }

    #[test]
    fn motion_reference_rejects_new_instances_camera_resets_and_teleports() {
        let affine = identity_affine();
        let clip = identity_clip();
        for (instance_valid, camera_valid) in [(false, true), (true, false)] {
            let result = temporal_motion_reference(
                [0.0; 3],
                affine,
                affine,
                clip,
                clip,
                instance_valid,
                camera_valid,
            );
            assert!(result.reject_history);
            assert_eq!(result.velocity_uv, [0.0; 2]);
        }
        let mut current = affine;
        current[0][3] = MAX_TEMPORAL_REPROJECTION_UV * 2.1;
        let result = temporal_motion_reference([0.0; 3], current, affine, clip, clip, true, true);
        assert!(result.reject_history);
        assert_eq!(result.velocity_uv, [0.0; 2]);
    }

    #[test]
    fn motion_reference_rejects_non_finite_and_behind_camera_projections() {
        let affine = identity_affine();
        let mut bad = identity_clip();
        bad[0][0] = f32::NAN;
        assert!(
            temporal_motion_reference([0.0; 3], affine, affine, bad, identity_clip(), true, true)
                .reject_history
        );
        let mut behind = identity_clip();
        behind[3] = [0.0, 0.0, 0.0, -1.0];
        assert!(
            temporal_motion_reference(
                [0.0; 3],
                affine,
                affine,
                identity_clip(),
                behind,
                true,
                true
            )
            .reject_history
        );
    }

    #[test]
    fn cache_key_isolated_by_entity_generation_output_extent_and_format() {
        let base = key(entity(1, 1), TaaOutput::Window(3), 64);
        assert_ne!(base, key(entity(1, 2), TaaOutput::Window(3), 64));
        assert_ne!(base, key(entity(1, 1), TaaOutput::Window(4), 64));
        assert_ne!(base, key(entity(1, 1), TaaOutput::Texture(3), 64));
        assert_ne!(base, key(entity(1, 1), TaaOutput::Window(3), 65));
        assert_ne!(
            base,
            TaaHistoryKey::new(
                base.camera,
                base.output,
                base.width,
                base.height,
                rhi::TextureFormat::Rgba32Float
            )
        );
    }

    #[test]
    fn null_graph_executes_resolve_and_retirement_waits_for_completion() {
        let runtime = RenderRuntime::headless(NullInstance::new(), Extent3d::new_2d(64, 32))
            .expect("runtime");
        let device = runtime.device();
        let make = |format, usage| {
            let texture = device.create_texture(&rhi::TextureDescriptor::new_2d(
                Extent3d::new_2d(64, 32),
                format,
                usage,
            ));
            let view = device.create_texture_view(texture, &rhi::TextureViewDescriptor::default());
            (texture, view)
        };
        let (ct, cv) = make(
            rhi::TextureFormat::Rgba16Float,
            rhi::TextureUsages::TEXTURE_BINDING,
        );
        let (mt, mv) = make(
            crate::target::TAA_MOTION_FORMAT,
            rhi::TextureUsages::TEXTURE_BINDING,
        );
        let (rt, rv) = make(
            crate::target::TAA_REACTIVE_FORMAT,
            rhi::TextureUsages::TEXTURE_BINDING,
        );
        let mut cache = TaaHistoryCache::default();
        let camera_key = key(entity(8, 4), TaaOutput::Window(7), 64);
        let mut graph = runtime.new_graph();
        let current = graph.import_texture(
            "current",
            TextureDesc::color(rhi::TextureFormat::Rgba16Float),
            ImportedTexture {
                texture: ct,
                default_view: cv,
                entry_state: rhi::TextureState::initial(),
                exit_state: rhi::TextureState::initial(),
                mip_levels: 1,
                array_layers: 1,
            },
        );
        let motion = graph.import_texture(
            "motion",
            TextureDesc::color(crate::target::TAA_MOTION_FORMAT),
            ImportedTexture {
                texture: mt,
                default_view: mv,
                entry_state: rhi::TextureState::initial(),
                exit_state: rhi::TextureState::initial(),
                mip_levels: 1,
                array_layers: 1,
            },
        );
        let reactive = graph.import_texture(
            "reactive",
            TextureDesc::color(crate::target::TAA_REACTIVE_FORMAT),
            ImportedTexture {
                texture: rt,
                default_view: rv,
                entry_state: rhi::TextureState::initial(),
                exit_state: rhi::TextureState::initial(),
                mip_levels: 1,
                array_layers: 1,
            },
        );
        cache
            .append_resolve(
                &mut graph, &runtime, camera_key, 9, false, 10, 4, current, cv, motion, mv,
                reactive, rv,
            )
            .expect("TAA append");
        assert_eq!(graph.pass_count(), 1);
        let plan = graph.compile().expect("valid resolve graph");
        assert_eq!(plan.pass_count(), 1);
        let completion = runtime.execute_tracked(graph).expect("Null graph submit");
        assert!(!cache.mark_submitted(camera_key, 9, completion));
        assert!(cache.mark_submitted(camera_key, 10, completion));
        let mut next_graph = runtime.new_graph();
        let next_current = next_graph.import_texture(
            "current",
            TextureDesc::color(rhi::TextureFormat::Rgba16Float),
            ImportedTexture {
                texture: ct,
                default_view: cv,
                entry_state: rhi::TextureState::initial(),
                exit_state: rhi::TextureState::initial(),
                mip_levels: 1,
                array_layers: 1,
            },
        );
        let next_motion = next_graph.import_texture(
            "motion",
            TextureDesc::color(crate::target::TAA_MOTION_FORMAT),
            ImportedTexture {
                texture: mt,
                default_view: mv,
                entry_state: rhi::TextureState::initial(),
                exit_state: rhi::TextureState::initial(),
                mip_levels: 1,
                array_layers: 1,
            },
        );
        let next_reactive = next_graph.import_texture(
            "reactive",
            TextureDesc::color(crate::target::TAA_REACTIVE_FORMAT),
            ImportedTexture {
                texture: rt,
                default_view: rv,
                entry_state: rhi::TextureState::initial(),
                exit_state: rhi::TextureState::initial(),
                mip_levels: 1,
                array_layers: 1,
            },
        );
        let next = cache
            .append_resolve(
                &mut next_graph,
                &runtime,
                camera_key,
                9,
                false,
                11,
                4,
                next_current,
                cv,
                next_motion,
                mv,
                next_reactive,
                rv,
            )
            .expect("continuous frame append");
        assert!(next.history_valid);
        let next_completion = runtime.execute_tracked(next_graph).expect("second submit");
        assert!(cache.mark_submitted(camera_key, 11, next_completion));

        let mut cut_graph = runtime.new_graph();
        let cut_current = cut_graph.import_texture(
            "current",
            TextureDesc::color(rhi::TextureFormat::Rgba16Float),
            ImportedTexture {
                texture: ct,
                default_view: cv,
                entry_state: rhi::TextureState::initial(),
                exit_state: rhi::TextureState::initial(),
                mip_levels: 1,
                array_layers: 1,
            },
        );
        let cut_motion = cut_graph.import_texture(
            "motion",
            TextureDesc::color(crate::target::TAA_MOTION_FORMAT),
            ImportedTexture {
                texture: mt,
                default_view: mv,
                entry_state: rhi::TextureState::initial(),
                exit_state: rhi::TextureState::initial(),
                mip_levels: 1,
                array_layers: 1,
            },
        );
        let cut_reactive = cut_graph.import_texture(
            "reactive",
            TextureDesc::color(crate::target::TAA_REACTIVE_FORMAT),
            ImportedTexture {
                texture: rt,
                default_view: rv,
                entry_state: rhi::TextureState::initial(),
                exit_state: rhi::TextureState::initial(),
                mip_levels: 1,
                array_layers: 1,
            },
        );
        let cut = cache
            .append_resolve(
                &mut cut_graph,
                &runtime,
                camera_key,
                10,
                true,
                12,
                4,
                cut_current,
                cv,
                cut_motion,
                mv,
                cut_reactive,
                rv,
            )
            .expect("cut append");
        assert!(cut.reset);
        assert!(!cut.history_valid);
        assert!(!cache.cancel_pending(camera_key, 11));
        assert!(cache.cancel_pending(camera_key, 12));
        let empty = BTreeSet::new();
        cache.retain_live(&empty, 16, runtime.device_generation(), 0);
        assert_eq!(cache.len(), 0);
        assert_eq!(
            cache.retired_len(),
            1,
            "GPU owner remains quarantined before completion"
        );
        assert!(runtime.wait_idle());
        cache.collect_completed(runtime.device_generation(), runtime.completed_value());
        assert_eq!(cache.retired_len(), 0);
        for (texture, view) in [(ct, cv), (mt, mv), (rt, rv)] {
            device.destroy_texture_view(view);
            device.destroy_texture(texture);
        }
    }

    #[test]
    fn epoch_cut_extent_and_format_validation_are_explicit() {
        let runtime =
            RenderRuntime::headless(NullInstance::new(), Extent3d::new_2d(8, 8)).expect("runtime");
        let base = key(entity(2, 1), TaaOutput::PrimaryWindow, 8);
        assert_eq!(
            validate_key(
                runtime.device(),
                TaaHistoryKey::new(base.camera, base.output, 0, 8, base.format)
            ),
            Err(TaaHistoryError::EmptyExtent)
        );
        assert_eq!(
            validate_key(
                runtime.device(),
                TaaHistoryKey::new(
                    base.camera,
                    base.output,
                    8,
                    8,
                    rhi::TextureFormat::Rgba8Unorm
                )
            ),
            Err(TaaHistoryError::UnsupportedFormat(
                rhi::TextureFormat::Rgba8Unorm
            ))
        );
    }
}
