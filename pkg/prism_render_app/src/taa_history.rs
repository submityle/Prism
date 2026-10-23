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
            .then_with(|| {
                texture_format_sort_key(self.format).cmp(&texture_format_sort_key(other.format))
            })
    }
}

fn texture_format_sort_key(format: rhi::TextureFormat) -> u8 {
    match format {
        rhi::TextureFormat::R8Unorm => 0,
        rhi::TextureFormat::R8Uint => 1,
        rhi::TextureFormat::R16Float => 2,
        rhi::TextureFormat::R32Float => 3,
        rhi::TextureFormat::R32Uint => 4,
        rhi::TextureFormat::Rg8Unorm => 5,
        rhi::TextureFormat::Rg16Float => 6,
        rhi::TextureFormat::Rg32Float => 7,
        rhi::TextureFormat::Rgba8Unorm => 8,
        rhi::TextureFormat::Rgba8UnormSrgb => 9,
        rhi::TextureFormat::Bgra8Unorm => 10,
        rhi::TextureFormat::Bgra8UnormSrgb => 11,
        rhi::TextureFormat::Rgb10a2Unorm => 12,
        rhi::TextureFormat::Rg11b10Float => 13,
        rhi::TextureFormat::Rgba16Float => 14,
        rhi::TextureFormat::Rgba32Float => 15,
        rhi::TextureFormat::Depth32Float => 16,
        rhi::TextureFormat::Depth24PlusStencil8 => 17,
        rhi::TextureFormat::Rgba32Uint => 18,
        _ => u8::MAX,
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
        }
    }

    fn extent(self) -> rhi::Extent3d {
        rhi::Extent3d::new_2d(self.width, self.height)
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
        Self {
            device,
            generation,
            layout,
            pipeline_layout,
            shader,
            pipeline,
        }
    }
}

impl Drop for ResolvePipeline {
    fn drop(&mut self) {
        self.device.destroy_compute_pipeline(self.pipeline);
        self.device.destroy_pipeline_layout(self.pipeline_layout);
        self.device.destroy_bind_group_layout(self.layout);
        self.device.destroy_shader_module(self.shader);
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
#[derive(Default)]
pub struct TaaHistoryCache {
    active: BTreeMap<TaaHistoryKey, HistoryOwner>,
    retired: Vec<RetiredOwner>,
    pipeline: Option<Arc<ResolvePipeline>>,
    pending: BTreeMap<TaaHistoryKey, PendingResolve>,
}

struct PendingResolve {
    frame: u64,
    epoch: u64,
    read_index: usize,
    write_index: usize,
    valid: bool,
    retention: u32,
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
        let generation = runtime.device_generation();
        let device: Arc<dyn rhi::RenderDevice + Send + Sync> = runtime.device_handle();
        self.collect_completed(generation, runtime.completed_value());
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
        if !self.active.contains_key(&key) {
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
        }
        let mismatch = self.active.get(&key).is_some_and(|owner| {
            owner.generation != generation || !Arc::ptr_eq(&owner.resources.device, &device)
        });
        if mismatch {
            if let Some(owner) = self.active.remove(&key) {
                self.retire_owner(owner);
            }
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
        }
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
        let history_view = if valid {
            owner.resources.views[read_index]
        } else {
            current_view
        };
        let bind_group = device.create_bind_group(&rhi::BindGroupDescriptor {
            label: Some("Prism per-camera TAA resolve".into()),
            layout: Some(self.pipeline.as_ref().expect("pipeline created").layout),
            entries: vec![
                rhi::BindGroupEntry {
                    binding: 0,
                    resource: rhi::BindingResource::TextureView(current_view),
                },
                rhi::BindGroupEntry {
                    binding: 1,
                    resource: rhi::BindingResource::TextureView(history_view),
                },
                rhi::BindGroupEntry {
                    binding: 2,
                    resource: rhi::BindingResource::TextureView(motion_view),
                },
                rhi::BindGroupEntry {
                    binding: 3,
                    resource: rhi::BindingResource::TextureView(reactive_view),
                },
                rhi::BindGroupEntry {
                    binding: 4,
                    resource: rhi::BindingResource::TextureView(owner.resources.views[write_index]),
                },
            ],
        });
        let history_owner: Arc<dyn Any + Send + Sync> = Arc::clone(&owner.resources);
        graph.retain_external_owner(history_owner);
        let lease: Arc<dyn Any + Send + Sync> = Arc::new(BindGroupLease {
            device: Arc::clone(&device),
            bind_group,
        });
        graph.retain_external_owner(lease);
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
                    ctx.set_compute_pipeline(pipeline)
                        .set_compute_bind_group(0, bind_group, Vec::new())
                        .dispatch(x, y, 1);
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
            },
        );
        Ok(TaaResolveOutput {
            texture: history_write,
            history_valid: valid,
            reset: !valid,
        })
    }

    /// Confirm that this camera's resolve was included in a successful frame submission.
    pub fn mark_submitted(
        &mut self,
        key: TaaHistoryKey,
        frame: u64,
        completion_value: u64,
    ) -> bool {
        let Some(pending) = self.pending.remove(&key) else {
            return false;
        };
        if pending.frame != frame {
            return false;
        }
        let Some(owner) = self.active.get_mut(&key) else {
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

    /// Cancel an unsubmitted resolve after graph compilation or submission failed.
    pub fn cancel_pending(&mut self, key: TaaHistoryKey, frame: u64) -> bool {
        self.pending
            .remove(&key)
            .is_some_and(|pending| pending.frame == frame)
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
    }
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
            rhi::TextureFormat::Rg16Float,
            rhi::TextureUsages::TEXTURE_BINDING,
        );
        let (rt, rv) = make(
            rhi::TextureFormat::R8Unorm,
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
            TextureDesc::color(rhi::TextureFormat::Rg16Float),
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
            TextureDesc::color(rhi::TextureFormat::R8Unorm),
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
            TextureDesc::color(rhi::TextureFormat::Rg16Float),
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
            TextureDesc::color(rhi::TextureFormat::R8Unorm),
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
            TextureDesc::color(rhi::TextureFormat::Rg16Float),
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
            TextureDesc::color(rhi::TextureFormat::R8Unorm),
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
        assert!(cache.cancel_pending(camera_key, 12));
        let empty = BTreeSet::new();
        cache.retain_live(&empty, 14, runtime.device_generation(), 0);
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
