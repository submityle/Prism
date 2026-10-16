//! The GPU-driven scene draw: rasterize the culled survivors from GPU-authored
//! indirect arguments, with each survivor's placement and color read from the
//! scene's instance storage.
//!
//! This is the M2 slice's "frame", declared as a [`RenderGraph`]:
//!
//! 1. a compute pass resets the cull's atomic append cursor,
//! 2. a compute pass culls the instance AABBs and authors the indirect-draw
//!    arguments, and
//! 3. a raster pass consumes those arguments through the graph's indirect-draw
//!    recorders and paints the survivors into an offscreen `Rgba8Unorm` target.
//!
//! Nothing about the frame is hand-synchronized. The cull reads the argument
//! version the reset wrote and the draw reads the visible-list and argument
//! versions the cull wrote, so the graph's SSA versioning orders the three
//! passes. Every resource the stages already own — the cull's GPU-authored
//! buffers, the scene's instance storage and index table, and the offscreen
//! target — is *imported*, so the graph tracks state without reallocating
//! anything.
//!
//! The CPU never learns how many instances survived: the single-draw path takes
//! the count straight from the GPU-authored argument record, and only the
//! multi-draw path (which must know the record count up front) echoes it back.
//!
//! [`SceneDrawStage::record_indexed_indirect`] and
//! [`SceneDrawStage::record_multi_indirect`] are raw recording hooks for a
//! caller that owns its encoder *and* its submission (the benchmark): the
//! frame-graph executor records and submits in one step, so it cannot hand back
//! a command buffer for the caller to pace. The crate's own frame — and every
//! test — goes through the graph.
//!
//! [`RenderGraph`]: prism_render_graph::RenderGraph

use alloc::string::String;
use alloc::vec::Vec;

use prism_render_driver as rhi;
use prism_render_graph::{
    BufferDesc, BufferHandle, CompileError, ImportedBuffer, ImportedTexture, PassFlags,
    RecordedGraph, RenderGraph, SizeClass, TextureDesc,
};
use rhi::{
    BindGroupDescriptor, BindGroupId, BindGroupLayoutDescriptor, BindGroupLayoutId,
    BufferBindingType, BufferDescriptor, BufferId, BufferState, BufferUsages, ColorTargetState,
    CommandEncoder, FragmentState, IndexFormat, MultisampleState, PipelineLayoutDescriptor,
    PipelineLayoutId, PipelineStages, PrimitiveState, PrimitiveTopology, RenderCommand,
    RenderDevice, RenderPipelineDescriptor, RenderPipelineId, RenderQueue, ShaderModuleDescriptor,
    ShaderModuleId, ShaderStages, TextureId, TextureState, TextureUsages, TextureViewId,
    VertexState,
};

use crate::camera::{FrameUniform, FRAME_UNIFORM_BYTES};
use crate::cull::CullStage;
use crate::draw::INDICES_PER_INSTANCE;
use crate::encoding::{IndirectArgs, AABB_STRIDE, FRUSTUM_UNIFORM_BYTES, PARAMS_BYTES};
use crate::gpu::{bind_buffer, buffer_entry, push_black_pass, quad_prefix};
use crate::instances::{instance_storage_bytes, Instance, INSTANCE_STRIDE};
use crate::quad::QUAD_VERTICES;

/// Which indirect-draw primitive the graph frame uses to consume the cull's
/// GPU-authored arguments.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum IndirectMode {
    /// One indexed indirect draw. Its instance count is the GPU-authored one,
    /// so the CPU never supplies it.
    Single,
    /// One packed multi-draw record per survivor. The primitive needs the
    /// record count up front, so `records` is the surviving count the caller
    /// read back; each record's contents stay GPU-authored.
    Multi {
        /// How many packed records the multi-draw covers.
        records: u32,
    },
}

/// A GPU-driven draw stage over an offscreen target.
///
/// The stage binds the cull pass's visible-index list and the scene's instance
/// storage, then records indirect draws from GPU-authored argument buffers: the
/// single-draw path takes the surviving count straight from the GPU's record,
/// and the multi-draw path takes the record count from the caller, who echoes
/// the count the GPU already confirmed.
pub struct SceneDrawStage {
    resolution: u32,
    capacity: u32,
    visible: BufferId,
    target: TextureId,
    target_view: TextureViewId,
    index_buffer: BufferId,
    instances: BufferId,
    frame_uniform: BufferId,
    bind_group_layout: BindGroupLayoutId,
    pipeline_layout: PipelineLayoutId,
    pipeline: RenderPipelineId,
    multi_pipeline: RenderPipelineId,
    shader: ShaderModuleId,
    bind_group: BindGroupId,
}

impl SceneDrawStage {
    /// Creates the offscreen target, index table, instance storage, pipelines
    /// and bind group that rasterize the survivors listed in `visible`.
    ///
    /// `visible` is the cull stage's output: one `u32` survivor id per element,
    /// bound here as a read-only storage buffer. `capacity` is the instance pool
    /// the visible list and the scene storage were sized for.
    ///
    /// # Panics
    ///
    /// Panics when `resolution` or `capacity` is zero, or when `capacity` is
    /// too large for 16-bit index values.
    #[must_use]
    pub fn new(
        device: &impl RenderDevice,
        resolution: u32,
        visible: BufferId,
        capacity: u32,
    ) -> Self {
        assert!(resolution > 0, "the scene draw needs a non-empty target");
        assert!(
            capacity > 0,
            "the scene draw needs a non-empty instance pool"
        );
        assert!(
            capacity * INDICES_PER_INSTANCE <= u32::from(u16::MAX),
            "the index table must fit 16-bit index values"
        );
        let extent = rhi::Extent3d::new_2d(resolution, resolution);

        let target = device.create_texture(&rhi::TextureDescriptor::new_2d(
            extent,
            rhi::TextureFormat::Rgba8Unorm,
            TextureUsages::RENDER_ATTACHMENT | TextureUsages::COPY_SRC,
        ));
        let target_view =
            device.create_texture_view(target, &rhi::TextureViewDescriptor::default());

        // One block of `INDICES_PER_INSTANCE` indices per instance slot, whose
        // values are that block's own global positions. A draw that starts at a
        // slot's block recovers the slot as `vertex_index / INDICES_PER_INSTANCE`,
        // which is how the multi-draw path names its survivor.
        let index_buffer = device.create_buffer(&BufferDescriptor {
            label: Some("gpudriven-scene-index".into()),
            size: u64::from(capacity) * u64::from(QUAD_VERTICES) * 2,
            usage: BufferUsages::INDEX | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let instances = device.create_buffer(&BufferDescriptor {
            label: Some("gpudriven-scene-instances".into()),
            size: u64::from(capacity) * INSTANCE_STRIDE,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let frame_uniform = device.create_buffer(&BufferDescriptor::new(
            FRAME_UNIFORM_BYTES,
            BufferUsages::UNIFORM | BufferUsages::COPY_DST,
        ));

        let shader = device.create_shader_module(&ShaderModuleDescriptor::wgsl(include_str!(
            "shaders/scene.wgsl"
        )));

        let bind_group_layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("gpudriven-scene-bgl".into()),
            entries: vec![
                buffer_entry(0, BufferBindingType::Uniform, ShaderStages::VERTEX),
                buffer_entry(
                    1,
                    BufferBindingType::Storage { read_only: true },
                    ShaderStages::VERTEX,
                ),
                buffer_entry(
                    2,
                    BufferBindingType::Storage { read_only: true },
                    ShaderStages::VERTEX,
                ),
            ],
        });

        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("gpudriven-scene-pl".into()),
            bind_group_layouts: vec![bind_group_layout],
            push_constant_ranges: Vec::new(),
        });

        let pipeline = make_pipeline(device, pipeline_layout, shader, "vs");
        let multi_pipeline = make_pipeline(device, pipeline_layout, shader, "vs_per_draw");

        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("gpudriven-scene-bg".into()),
            layout: Some(bind_group_layout),
            entries: vec![
                bind_buffer(0, frame_uniform),
                bind_buffer(1, visible),
                bind_buffer(2, instances),
            ],
        });

        Self {
            resolution,
            capacity,
            visible,
            target,
            target_view,
            index_buffer,
            instances,
            frame_uniform,
            bind_group_layout,
            pipeline_layout,
            pipeline,
            multi_pipeline,
            shader,
            bind_group,
        }
    }

    /// Uploads the index table, the camera uniform and the scene's instance
    /// records. Call once per scene (or per frame when the scene animates)
    /// before recording a draw.
    pub fn upload(&self, queue: &impl RenderQueue, frame: &FrameUniform, instances: &[Instance]) {
        let mut index_table = Vec::with_capacity(self.capacity as usize * QUAD_VERTICES as usize);
        for slot in 0..self.capacity {
            for corner in 0..QUAD_VERTICES {
                index_table.push((slot * QUAD_VERTICES + corner) as u16);
            }
        }
        let index_bytes: Vec<u8> = index_table.iter().flat_map(|i| i.to_le_bytes()).collect();
        queue.write_buffer(self.index_buffer, 0, &index_bytes);
        queue.write_buffer(self.frame_uniform, 0, &frame.to_bytes());
        queue.write_buffer(self.instances, 0, &instance_storage_bytes(instances));
    }

    /// Records one indexed indirect draw into a caller-owned encoder, using the
    /// GPU-authored arguments the cull pass wrote.
    ///
    /// This is the low-level hook for callers that own their encoder *and* their
    /// submission: the frame-graph executor both records and submits, so a
    /// caller that must measure or pace submission itself (the benchmark)
    /// records through here instead. The crate's own frame path is
    /// [`Self::graph_frame`] / [`Self::render_graph`].
    pub fn record_indexed_indirect(
        &self,
        encoder: &mut CommandEncoder,
        args_buffer: BufferId,
        offset: u64,
    ) {
        self.record_raw(
            encoder,
            self.pipeline,
            RenderCommand::DrawIndexedIndirect {
                buffer: args_buffer,
                offset,
            },
        );
    }

    /// Records `count` packed indexed indirect draws into a caller-owned
    /// encoder.
    ///
    /// See [`Self::record_indexed_indirect`] for when to use the raw hooks.
    pub fn record_multi_indirect(
        &self,
        encoder: &mut CommandEncoder,
        args_buffer: BufferId,
        offset: u64,
        count: u32,
        stride: u32,
    ) {
        self.record_raw(
            encoder,
            self.multi_pipeline,
            RenderCommand::MultiDrawIndirect {
                buffer: args_buffer,
                offset,
                count,
                stride,
            },
        );
    }

    /// Records a black-cleared raster pass into `encoder` with `draw`.
    fn record_raw(
        &self,
        encoder: &mut CommandEncoder,
        pipeline: RenderPipelineId,
        draw: RenderCommand,
    ) {
        let mut commands = quad_prefix(pipeline, self.bind_group, self.index_buffer);
        commands.push(draw);
        push_black_pass(encoder, "gpudriven-scene", self.target_view, commands);
    }

    /// Declares the whole GPU-driven frame as a [`RenderGraph`].
    ///
    /// The graph has three passes, all bound to resources the two stages
    /// already own:
    ///
    /// 1. `gpudriven-cull-reset` — one invocation that zeroes the append cursor,
    ///    so the frame is repeatable.
    /// 2. `gpudriven-cull` — the compute cull, writing the visible list and the
    ///    indirect arguments.
    /// 3. `gpudriven-scene` — the raster pass, consuming the cull's outputs
    ///    through the graph's indirect recorders.
    ///
    /// `mode` selects which indirect primitive the raster pass issues.
    #[must_use]
    pub fn graph_frame(&self, cull: &CullStage, mode: IndirectMode) -> RenderGraph {
        let mut graph = RenderGraph::new(rhi::Extent3d::new_2d(self.resolution, self.resolution));

        // --- import the cull's GPU-authored outputs -----------------------
        let visible = import_buffer(
            &mut graph,
            "gpudriven-visible",
            cull.visible(),
            u64::from(cull.capacity()) * 4,
            BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            BufferState::storage_write(PipelineStages::COMPUTE_SHADER),
        );
        let args = import_buffer(
            &mut graph,
            "gpudriven-draw-args",
            cull.draw_args(),
            IndirectArgs::BYTES,
            BufferUsages::STORAGE | BufferUsages::INDIRECT | BufferUsages::COPY_DST,
            BufferState::storage_write(PipelineStages::COMPUTE_SHADER),
        );
        let multi = import_buffer(
            &mut graph,
            "gpudriven-multi-args",
            cull.multi_args(),
            u64::from(cull.capacity()) * IndirectArgs::BYTES,
            BufferUsages::STORAGE | BufferUsages::INDIRECT,
            BufferState::storage_write(PipelineStages::COMPUTE_SHADER),
        );

        // --- import the cull's read-only inputs ---------------------------
        let frustum = import_buffer(
            &mut graph,
            "gpudriven-frustum",
            cull.frustum_buffer(),
            FRUSTUM_UNIFORM_BYTES,
            BufferUsages::UNIFORM | BufferUsages::COPY_DST,
            BufferState::copy_dst(),
        );
        let cull_instances = import_buffer(
            &mut graph,
            "gpudriven-cull-instances",
            cull.instance_buffer(),
            u64::from(cull.capacity()) * AABB_STRIDE,
            BufferUsages::STORAGE | BufferUsages::COPY_DST,
            BufferState::copy_dst(),
        );
        let cull_params = import_buffer(
            &mut graph,
            "gpudriven-cull-params",
            cull.params_buffer(),
            PARAMS_BYTES,
            BufferUsages::UNIFORM | BufferUsages::COPY_DST,
            BufferState::copy_dst(),
        );

        // --- import the draw's read-only inputs ---------------------------
        let frame_uniform = import_buffer(
            &mut graph,
            "gpudriven-frame",
            self.frame_uniform,
            FRAME_UNIFORM_BYTES,
            BufferUsages::UNIFORM | BufferUsages::COPY_DST,
            BufferState::copy_dst(),
        );
        let render_instances = import_buffer(
            &mut graph,
            "gpudriven-scene-instances",
            self.instances,
            u64::from(self.capacity) * INSTANCE_STRIDE,
            BufferUsages::STORAGE | BufferUsages::COPY_DST,
            BufferState::copy_dst(),
        );
        let index_buffer = import_buffer(
            &mut graph,
            "gpudriven-scene-index",
            self.index_buffer,
            u64::from(self.capacity) * u64::from(QUAD_VERTICES) * 2,
            BufferUsages::INDEX | BufferUsages::COPY_DST,
            BufferState::copy_dst(),
        );

        // --- import the offscreen target ----------------------------------
        let target = graph.import_texture(
            "gpudriven-target",
            TextureDesc::color(rhi::TextureFormat::Rgba8Unorm)
                .with_size(SizeClass::absolute_2d(self.resolution, self.resolution))
                .with_usage(TextureUsages::COPY_SRC),
            ImportedTexture {
                texture: self.target,
                default_view: self.target_view,
                entry_state: TextureState::initial(),
                exit_state: TextureState::copy_src(),
                mip_levels: 1,
                array_layers: 1,
            },
        );

        // --- pass 1: reset the append cursor ------------------------------
        let reset_pipeline = cull.reset_pipeline();
        let reset_bind_group = cull.cull_bind_group();
        graph.add_compute_pass("gpudriven-cull-reset", PassFlags::EMPTY, move |b| {
            let args = b.write_storage_buffer(args);
            b.blackboard_mut().set(ResetArgs(args));
            move |ctx| {
                ctx.set_compute_pipeline(reset_pipeline)
                    .set_compute_bind_group(0, reset_bind_group, Vec::new())
                    .dispatch(1, 1, 1);
            }
        });

        // --- pass 2: cull on the GPU --------------------------------------
        let cull_pipeline = cull.cull_pipeline();
        let cull_bind_group = cull.cull_bind_group();
        let groups = cull.dispatch_groups();
        graph.add_compute_pass("gpudriven-cull", PassFlags::EMPTY, move |b| {
            b.read_uniform(frustum);
            b.read_storage_buffer(cull_instances);
            b.read_uniform(cull_params);
            let reset = b
                .blackboard()
                .get::<ResetArgs>()
                .expect("the reset pass published the argument version")
                .0;
            let args = b.write_storage_buffer(reset);
            let visible = b.write_storage_buffer(visible);
            let multi = b.write_storage_buffer(multi);
            b.blackboard_mut().set(CullOutputs {
                visible,
                args,
                multi,
            });
            move |ctx| {
                ctx.set_compute_pipeline(cull_pipeline)
                    .set_compute_bind_group(0, cull_bind_group, Vec::new())
                    .dispatch(groups, 1, 1);
            }
        });

        // --- pass 3: draw the GPU-authored survivors ----------------------
        let pipeline = match mode {
            IndirectMode::Single => self.pipeline,
            IndirectMode::Multi { .. } => self.multi_pipeline,
        };
        let bind_group = self.bind_group;
        let index_buffer_id = self.index_buffer;
        let records = match mode {
            IndirectMode::Single => 0,
            IndirectMode::Multi { records } => records,
        };
        graph.add_raster_pass("gpudriven-scene", PassFlags::EMPTY, move |b| {
            b.read_uniform(frame_uniform);
            b.read_storage_buffer(render_instances);
            b.read_index(index_buffer);
            let outputs = *b
                .blackboard()
                .get::<CullOutputs>()
                .expect("the cull pass published its outputs");
            b.read_storage_buffer(outputs.visible);
            match mode {
                IndirectMode::Single => {
                    b.read_indirect(outputs.args);
                }
                IndirectMode::Multi { .. } => {
                    b.read_indirect(outputs.multi);
                }
            }
            b.color_attachment(target);
            move |ctx| {
                ctx.set_render_pipeline(pipeline)
                    .set_render_bind_group(0, bind_group, Vec::new())
                    .set_index_buffer(index_buffer_id, IndexFormat::Uint16, 0);
                match mode {
                    IndirectMode::Single => {
                        ctx.draw_indexed_indirect(outputs.args, 0);
                    }
                    IndirectMode::Multi { .. } => {
                        ctx.multi_draw_indirect(
                            outputs.multi,
                            0,
                            records,
                            IndirectArgs::BYTES as u32,
                        );
                    }
                }
            }
        });

        graph
    }

    /// Declares the GPU-driven frame as a graph and executes it.
    ///
    /// The returned owner retains the graph's physical allocations. Keep it
    /// alive until the submitting queue confirms completion, then call
    /// [`RecordedGraph::release`] with the creating device. Submission alone
    /// is not completion, and dropping the owner does not destroy resources.
    ///
    /// # Errors
    ///
    /// Returns a [`CompileError`] only when the declared frame cannot be
    /// scheduled, which a well-formed frame never does.
    pub fn render_graph<D: RenderDevice, Q: RenderQueue>(
        &self,
        device: &D,
        queue: &Q,
        cull: &CullStage,
        mode: IndirectMode,
    ) -> Result<RecordedGraph, CompileError> {
        self.graph_frame(cull, mode).execute(device, queue)
    }

    /// The offscreen target the survivors are rendered into.
    #[must_use]
    pub const fn target(&self) -> TextureId {
        self.target
    }

    /// The target's side length in pixels.
    #[must_use]
    pub const fn resolution(&self) -> u32 {
        self.resolution
    }

    /// The visible-index list this stage rasterizes, as bound for the shader.
    #[must_use]
    pub const fn visible(&self) -> BufferId {
        self.visible
    }

    /// Releases every resource the stage owns.
    pub fn destroy(&self, device: &impl RenderDevice) {
        device.destroy_render_pipeline(self.pipeline);
        device.destroy_render_pipeline(self.multi_pipeline);
        device.destroy_pipeline_layout(self.pipeline_layout);
        device.destroy_bind_group_layout(self.bind_group_layout);
        device.destroy_shader_module(self.shader);
        device.destroy_bind_group(self.bind_group);
        device.destroy_buffer(self.index_buffer);
        device.destroy_buffer(self.instances);
        device.destroy_buffer(self.frame_uniform);
        device.destroy_texture_view(self.target_view);
        device.destroy_texture(self.target);
    }
}

/// The argument version the reset pass minted, shared with the cull pass
/// through the graph blackboard.
#[derive(Clone, Copy)]
struct ResetArgs(BufferHandle);

/// The resource versions the cull pass minted, shared with the draw pass.
#[derive(Clone, Copy)]
struct CullOutputs {
    /// The GPU-authored visible-index list.
    visible: BufferHandle,
    /// The GPU-authored single-draw arguments.
    args: BufferHandle,
    /// The GPU-authored packed multi-draw records.
    multi: BufferHandle,
}

/// Imports an externally-owned buffer into `graph`, preserving its concrete
/// size, usage, and entry state.
fn import_buffer(
    graph: &mut RenderGraph,
    name: &str,
    buffer: BufferId,
    size: u64,
    usage: BufferUsages,
    entry_state: BufferState,
) -> BufferHandle {
    graph.import_buffer(
        name,
        BufferDesc::new(size, usage),
        ImportedBuffer {
            buffer,
            entry_state,
        },
    )
}

/// Creates a quad pipeline from one of the module's vertex entry points.
///
/// Corners come from the vertex index and placement from the instance storage,
/// so no vertex buffer is bound — only the shared index buffer, which is what
/// makes the indirect draws indexed.
fn make_pipeline(
    device: &impl RenderDevice,
    layout: PipelineLayoutId,
    shader: ShaderModuleId,
    entry_point: &'static str,
) -> RenderPipelineId {
    device.create_render_pipeline(&RenderPipelineDescriptor {
        label: Some(String::from("gpudriven-scene")),
        layout: Some(layout),
        vertex: VertexState {
            module: shader,
            entry_point: entry_point.into(),
            buffers: Vec::new(),
        },
        primitive: PrimitiveState {
            topology: PrimitiveTopology::TriangleStrip,
            // A strip topology must declare the index width it restarts on, and
            // it must match the bound index buffer.
            strip_index_format: Some(IndexFormat::Uint16),
            cull_mode: None,
            ..PrimitiveState::default()
        },
        depth_stencil: None,
        multisample: MultisampleState::default(),
        fragment: Some(FragmentState {
            module: shader,
            entry_point: "fs".into(),
            targets: vec![Some(ColorTargetState::opaque(
                rhi::TextureFormat::Rgba8Unorm,
            ))],
        }),
    })
}
