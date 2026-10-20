//! GPU sort/cull and renderer-matrix resource wiring for particle renderers.
//!
//! This module is intentionally independent from the application frame loop.
//! It owns the RHI resources and declares their RenderGraph dependencies, while
//! the architecture crate remains the owner of renderer and sort contracts.
//! The cull pass writes sort items, the sort passes order those items by the
//! contract's quantized depth key with a particle-index tie break, and the
//! final passes publish one shared visible-index ABI plus renderer-specific
//! indirect argument records.

use alloc::format;
use alloc::string::String;
use alloc::vec::Vec;

use prism_render_architecture::particle::indirect_draw::{
    RendererGeometry, INDICES_PER_SEGMENT, SPRITE_QUAD_VERTEX_COUNT,
};
use prism_render_architecture::particle::renderers::{RendererKind, RendererPlan};
use prism_render_architecture::particle::sort_cull::{
    sort_cull_buffer_layout, SortCullBufferLayout, SortCullLayoutError,
};
use prism_render_architecture::particle::SortStrategy;
use prism_render_driver as rhi;
use prism_render_driver::{BufferState, PipelineStages};
use prism_render_graph::{
    BufferDesc, BufferHandle, CompileError, ImportedBuffer, ImportedTexture, PassFlags,
    RecordedGraph, RenderGraph, SizeClass, TextureDesc,
};
use rhi::{
    BindGroupDescriptor, BindGroupId, BindGroupLayoutDescriptor, BindGroupLayoutId,
    BufferBindingType, BufferDescriptor, BufferId, BufferUsages, ColorTargetState,
    ComputePipelineDescriptor, ComputePipelineId, FragmentState, IndexFormat, MultisampleState,
    PipelineLayoutDescriptor, PipelineLayoutId, PrimitiveState, PrimitiveTopology, RenderDevice,
    RenderPipelineDescriptor, RenderPipelineId, RenderQueue, ShaderModuleDescriptor,
    ShaderModuleId, ShaderStages, TextureFormat, TextureState, TextureUsages,
    TextureViewDescriptor, VertexState,
};

use crate::gpu::{bind_buffer, buffer_entry};

/// The fixed offscreen target used by the standalone renderer-matrix raster
/// consumer. Callers embedding the stage in a larger frame graph can use the
/// same pipeline/bindings with their own color target.
pub const RENDERER_MATRIX_TARGET_SIZE: u32 = 128;

const SURFACE_MATERIAL_BYTES: u64 = 64;
const SURFACE_MESH_VERTEX_STRIDE: u64 = 16;
const AUX_RECORD_BYTES: u64 = 32;
const AUX_PARAMS_BYTES: u64 = 16;

/// One source particle record consumed by the GPU cull pass.
///
/// The payload index is kept explicit so a compact render buffer can refer to
/// a stable pool slot even when the source records were assembled from a
/// subrange or a multi-emitter batch.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ParticleRenderInput {
    /// World-space bounding-sphere center and radius.
    pub center_radius: [f32; 4],
    /// View-space depth used to build the 16-bit sort key.
    pub depth: f32,
    /// Stable pool/instance index gathered by renderer shaders.
    pub particle_index: u32,
    /// Renderer-defined flags. ABI v1 preserves them but does not interpret them.
    pub flags: u32,
    /// Reserved ABI word; must be zero in host uploads.
    pub reserved: u32,
}

/// One mesh vertex consumed by the particle Mesh renderer.
///
/// The renderer-matrix surface ABI currently carries local position and UV;
/// normals and material attributes remain owned by the shared particle
/// material path.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ParticleMeshVertex {
    /// Local mesh position in the renderer's two-dimensional surface space.
    pub position: [f32; 2],
    /// Mesh UV passed through to the surface fragment stage.
    pub uv: [f32; 2],
}

impl ParticleMeshVertex {
    /// Creates a finite mesh vertex.
    #[must_use]
    pub const fn new(position: [f32; 2], uv: [f32; 2]) -> Self {
        Self { position, uv }
    }

    fn validate(self) -> bool {
        self.position
            .iter()
            .chain(self.uv.iter())
            .all(|value| value.is_finite())
    }
}

impl ParticleRenderInput {
    /// Creates a finite, valid source record.
    #[must_use]
    pub const fn new(center: [f32; 3], radius: f32, depth: f32, particle_index: u32) -> Self {
        Self {
            center_radius: [center[0], center[1], center[2], radius],
            depth,
            particle_index,
            flags: 0,
            reserved: 0,
        }
    }

    fn validate(self) -> bool {
        self.center_radius.iter().all(|value| value.is_finite())
            && self.depth.is_finite()
            && self.center_radius[3] >= 0.0
    }
}

/// Six inward-facing frustum planes, packed as `(normal.xyz, d)`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ParticleCullView {
    /// The six planes used by the conservative sphere test.
    pub planes: [[f32; 4]; 6],
    /// Camera position used by the optional distance test.
    pub camera_position: [f32; 3],
    /// A non-positive value disables distance culling.
    pub max_distance: f32,
    /// Near and far values used by depth-key quantization.
    pub near_depth: f32,
    /// Far depth used by depth-key quantization.
    pub far_depth: f32,
    /// `true` produces an ascending key for far-to-near compositing.
    pub back_to_front: bool,
}

impl ParticleCullView {
    /// A view that keeps every finite sphere and quantizes depth in `[0, 1]`.
    #[must_use]
    pub const fn unrestricted() -> Self {
        Self {
            planes: [[0.0, 0.0, 0.0, 1.0]; 6],
            camera_position: [0.0; 3],
            max_distance: 0.0,
            near_depth: 0.0,
            far_depth: 1.0,
            back_to_front: false,
        }
    }

    fn validate(self) -> bool {
        self.planes.iter().flatten().all(|value| value.is_finite())
            && self.camera_position.iter().all(|value| value.is_finite())
            && self.max_distance.is_finite()
            && self.near_depth.is_finite()
            && self.far_depth.is_finite()
            && self.far_depth > self.near_depth
    }

    fn frustum_bytes(self) -> [u8; 96] {
        let mut bytes = [0; 96];
        for (index, plane) in self.planes.into_iter().enumerate() {
            for (component, value) in plane.into_iter().enumerate() {
                let offset = (index * 16) + (component * 4);
                bytes[offset..offset + 4].copy_from_slice(&value.to_le_bytes());
            }
        }
        bytes
    }

    fn params_bytes(self, capacity: u32, input_count: u32) -> [u8; 64] {
        let words = [
            capacity,
            input_count,
            u32::from(self.back_to_front),
            0,
            self.near_depth.to_bits(),
            self.far_depth.to_bits(),
            self.max_distance.to_bits(),
            0,
            self.camera_position[0].to_bits(),
            self.camera_position[1].to_bits(),
            self.camera_position[2].to_bits(),
            0,
            0,
            0,
            0,
            0,
        ];
        let mut bytes = [0; 64];
        for (index, word) in words.into_iter().enumerate() {
            bytes[index * 4..index * 4 + 4].copy_from_slice(&word.to_le_bytes());
        }
        bytes
    }
}

/// The shared renderer route accepted by the GPU stage.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ParticleRendererRoute {
    /// Geometry family selected by the architecture renderer matrix.
    pub geometry: RendererGeometry,
    /// Sort strategy selected by the architecture sort/cull matrix.
    pub sort: SortStrategy,
}

impl ParticleRendererRoute {
    /// Builds a GPU route from the architecture plan and its geometry contract.
    pub fn from_plan(
        plan: &RendererPlan,
        geometry: RendererGeometry,
    ) -> Result<Self, ParticleRendererRouteError> {
        if plan.kind != geometry.kind() {
            return Err(ParticleRendererRouteError::GeometryKindMismatch {
                plan: plan.kind,
                geometry: geometry.kind(),
            });
        }
        if !matches!(
            plan.kind,
            RendererKind::Sprite | RendererKind::Mesh | RendererKind::Ribbon | RendererKind::Beam
        ) {
            return Err(ParticleRendererRouteError::UnsupportedRenderer(plan.kind));
        }
        let sort = plan
            .routing
            .map(|routing| routing.sort)
            .ok_or(ParticleRendererRouteError::MissingGeometryRouting)?;
        Self::new(plan.kind, sort, geometry)
    }

    /// Creates a route directly from the frozen renderer contracts.
    pub fn new(
        kind: RendererKind,
        sort: SortStrategy,
        geometry: RendererGeometry,
    ) -> Result<Self, ParticleRendererRouteError> {
        if kind != geometry.kind() {
            return Err(ParticleRendererRouteError::GeometryKindMismatch {
                plan: kind,
                geometry: geometry.kind(),
            });
        }
        if !matches!(
            kind,
            RendererKind::Sprite | RendererKind::Mesh | RendererKind::Ribbon | RendererKind::Beam
        ) {
            return Err(ParticleRendererRouteError::UnsupportedRenderer(kind));
        }
        if matches!(geometry, RendererGeometry::Mesh { index_count: 0, .. }) {
            return Err(ParticleRendererRouteError::ZeroMeshIndexCount);
        }
        Ok(Self { geometry, sort })
    }

    /// Returns the contract renderer kind.
    #[must_use]
    pub const fn kind(self) -> RendererKind {
        self.geometry.kind()
    }

    fn kind_code(self) -> u32 {
        match self.kind() {
            RendererKind::Sprite => 0,
            RendererKind::Mesh => 1,
            RendererKind::Ribbon => 2,
            RendererKind::Beam => 3,
            RendererKind::Light | RendererKind::Decal | RendererKind::Volume => unreachable!(),
        }
    }

    fn geometry_count(self) -> u32 {
        match self.geometry {
            RendererGeometry::Sprite { alive_count, .. }
            | RendererGeometry::Mesh { alive_count, .. } => alive_count,
            RendererGeometry::Ribbon { chain_count, .. }
            | RendererGeometry::Beam { chain_count, .. } => chain_count,
        }
    }

    fn index_count(self) -> u32 {
        match self.geometry {
            RendererGeometry::Sprite { .. } => 0,
            RendererGeometry::Mesh { index_count, .. } => index_count,
            RendererGeometry::Ribbon {
                segments_per_chain, ..
            }
            | RendererGeometry::Beam {
                segments_per_chain, ..
            } => segments_per_chain.saturating_mul(INDICES_PER_SEGMENT),
        }
    }

    fn first_instance(self) -> u32 {
        match self.geometry {
            RendererGeometry::Sprite { first_instance, .. }
            | RendererGeometry::Mesh { first_instance, .. }
            | RendererGeometry::Ribbon { first_instance, .. }
            | RendererGeometry::Beam { first_instance, .. } => first_instance,
        }
    }

    fn params_bytes(self) -> [u8; 48] {
        let words = [
            self.kind_code(),
            self.geometry_count(),
            self.index_count(),
            self.first_instance(),
            SPRITE_QUAD_VERTEX_COUNT,
            0,
            0,
            0,
        ];
        let mut bytes = [0; 48];
        for (index, word) in words.into_iter().enumerate() {
            bytes[index * 4..index * 4 + 4].copy_from_slice(&word.to_le_bytes());
        }
        bytes
    }
}

/// Invalid renderer route input.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ParticleRendererRouteError {
    /// The geometry kind and plan kind do not agree.
    GeometryKindMismatch {
        /// Kind selected by the renderer plan.
        plan: RendererKind,
        /// Kind represented by the geometry ABI.
        geometry: RendererKind,
    },
    /// A light/decal/volume plan cannot be sent to this surface route.
    UnsupportedRenderer(RendererKind),
    /// A plan without a geometry routing decision cannot produce a surface pass.
    MissingGeometryRouting,
    /// Mesh rendering needs a nonzero source index count.
    ZeroMeshIndexCount,
}

/// Resource/layout error returned before creating any GPU object.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ParticleRendererGpuError {
    /// The source pool has no elements.
    ZeroCapacity,
    /// The requested capacity cannot be rounded to a power of two.
    CapacityOverflow,
    /// A required workgroup exceeds the device dispatch limit.
    DispatchTooLarge { workgroups: u32, limit: u32 },
    /// A required resource exceeds one device buffer limit.
    BufferTooLarge {
        /// Resource label used in the error.
        resource: &'static str,
        /// Requested byte count.
        requested: u64,
        /// Device limit.
        limit: u64,
    },
    /// A required compute binding is unavailable.
    BindingLimitTooSmall { requested: u32, limit: u32 },
    /// The architecture sort/cull contract rejected the layout.
    Layout(SortCullLayoutError),
    /// A route is not surface geometry supported by this stage.
    Route(ParticleRendererRouteError),
    /// Host input did not satisfy the finite ABI invariants.
    InvalidInput,
    /// Host view parameters did not satisfy the finite ABI invariants.
    InvalidView,
    /// A custom surface mesh was supplied to a non-Mesh route.
    MeshGeometryUnsupported,
    /// A custom surface mesh did not fit the route-owned buffers.
    MeshGeometryTooLarge,
    /// A custom surface mesh contained an invalid index or non-finite vertex.
    InvalidMeshGeometry,
}

impl From<SortCullLayoutError> for ParticleRendererGpuError {
    fn from(value: SortCullLayoutError) -> Self {
        Self::Layout(value)
    }
}

impl From<ParticleRendererRouteError> for ParticleRendererGpuError {
    fn from(value: ParticleRendererRouteError) -> Self {
        Self::Route(value)
    }
}

impl ParticleAuxiliaryGpu {
    /// Creates a renderer-specific Light, Decal, or Volume dispatch stage.
    pub fn new<D: RenderDevice + ?Sized>(
        device: &D,
        kind: RendererKind,
        capacity: u32,
    ) -> Result<Self, ParticleRendererGpuError> {
        if capacity == 0 {
            return Err(ParticleRendererGpuError::ZeroCapacity);
        }
        if !matches!(
            kind,
            RendererKind::Light | RendererKind::Decal | RendererKind::Volume
        ) {
            return Err(ParticleRendererGpuError::Route(
                ParticleRendererRouteError::UnsupportedRenderer(kind),
            ));
        }
        let limits = device.capabilities().limits;
        let max_storage = limits
            .max_storage_buffer_binding_size
            .min(limits.max_buffer_size);
        for (resource, size) in [
            ("particle auxiliary input", u64::from(capacity) * 32),
            (
                "particle auxiliary output",
                u64::from(capacity) * AUX_RECORD_BYTES,
            ),
        ] {
            if size > max_storage {
                return Err(ParticleRendererGpuError::BufferTooLarge {
                    resource,
                    requested: size,
                    limit: max_storage,
                });
            }
        }
        let workgroups = capacity.div_ceil(64);
        if workgroups > limits.max_compute_workgroups_per_dimension {
            return Err(ParticleRendererGpuError::DispatchTooLarge {
                workgroups,
                limit: limits.max_compute_workgroups_per_dimension,
            });
        }
        let input = device.create_buffer(&BufferDescriptor::new(
            u64::from(capacity) * 32,
            BufferUsages::STORAGE | BufferUsages::COPY_DST,
        ));
        let output = device.create_buffer(&BufferDescriptor::new(
            u64::from(capacity) * AUX_RECORD_BYTES,
            BufferUsages::STORAGE | BufferUsages::COPY_SRC,
        ));
        let params = device.create_buffer(&BufferDescriptor::new(
            AUX_PARAMS_BYTES,
            BufferUsages::UNIFORM | BufferUsages::COPY_DST,
        ));
        let shader = device.create_shader_module(&ShaderModuleDescriptor::wgsl(include_str!(
            "../shaders/particle_renderer_auxiliary.wgsl"
        )));
        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some(String::from("particle-renderer-auxiliary-bgl")),
            entries: vec![
                buffer_entry(
                    0,
                    BufferBindingType::Storage { read_only: true },
                    ShaderStages::COMPUTE,
                ),
                buffer_entry(
                    1,
                    BufferBindingType::Storage { read_only: false },
                    ShaderStages::COMPUTE,
                ),
                buffer_entry(2, BufferBindingType::Uniform, ShaderStages::COMPUTE),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some(String::from("particle-renderer-auxiliary-pl")),
            bind_group_layouts: vec![layout],
            push_constant_ranges: Vec::new(),
        });
        let entry_point = match kind {
            RendererKind::Light => "emit_lights",
            RendererKind::Decal => "project_decals",
            RendererKind::Volume => "accumulate_volumes",
            _ => unreachable!(),
        };
        let pipeline = make_compute(device, pipeline_layout, shader, entry_point);
        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some(String::from("particle-renderer-auxiliary-bg")),
            layout: Some(layout),
            entries: vec![
                bind_buffer(0, input),
                bind_buffer(1, output),
                bind_buffer(2, params),
            ],
        });
        Ok(Self {
            kind,
            capacity,
            input,
            output,
            params,
            shader,
            layout,
            pipeline_layout,
            pipeline,
            bind_group,
        })
    }

    /// Uploads particle records and renderer-specific effect parameters.
    pub fn set_inputs(
        &self,
        queue: &impl RenderQueue,
        inputs: &[ParticleRenderInput],
        intensity_or_density: f32,
    ) -> Result<(), ParticleRendererGpuError> {
        if inputs.len() > self.capacity as usize || !intensity_or_density.is_finite() {
            return Err(ParticleRendererGpuError::InvalidInput);
        }
        let mut bytes = Vec::with_capacity(inputs.len() * 32);
        for input in inputs {
            if !input.validate() {
                return Err(ParticleRendererGpuError::InvalidInput);
            }
            for value in input.center_radius {
                bytes.extend_from_slice(&value.to_le_bytes());
            }
            bytes.extend_from_slice(&input.depth.to_le_bytes());
            bytes.extend_from_slice(&input.particle_index.to_le_bytes());
            bytes.extend_from_slice(&input.flags.to_le_bytes());
            bytes.extend_from_slice(&input.reserved.to_le_bytes());
        }
        queue.write_buffer(self.input, 0, &bytes);
        let mode = match self.kind {
            RendererKind::Light => 0.0_f32,
            RendererKind::Decal => 1.0,
            RendererKind::Volume => 2.0,
            _ => unreachable!(),
        };
        let params = [
            inputs.len() as u32,
            mode.to_bits(),
            intensity_or_density.to_bits(),
            0_u32,
        ];
        let mut packed = [0_u8; AUX_PARAMS_BYTES as usize];
        for (i, word) in params.into_iter().enumerate() {
            packed[i * 4..i * 4 + 4].copy_from_slice(&word.to_le_bytes());
        }
        queue.write_buffer(self.params, 0, &packed);
        Ok(())
    }

    /// Adds the selected renderer's concrete compute phase to a graph.
    pub fn append_to_graph(
        &self,
        graph: &mut RenderGraph,
    ) -> Result<ParticleAuxiliaryGraphOutputs, CompileError> {
        let input = import_buffer(
            graph,
            "particle-aux-input",
            self.input,
            u64::from(self.capacity) * 32,
            BufferUsages::STORAGE | BufferUsages::COPY_DST,
            BufferState::storage_read(PipelineStages::COMPUTE_SHADER),
        );
        let output = import_buffer(
            graph,
            "particle-aux-output",
            self.output,
            u64::from(self.capacity) * AUX_RECORD_BYTES,
            BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            BufferState::storage_write(PipelineStages::COMPUTE_SHADER),
        );
        let params = import_buffer(
            graph,
            "particle-aux-params",
            self.params,
            AUX_PARAMS_BYTES,
            BufferUsages::UNIFORM | BufferUsages::COPY_DST,
            BufferState::copy_dst(),
        );
        let groups = self.capacity.div_ceil(64);
        let pipeline = self.pipeline;
        let bind_group = self.bind_group;
        let mut records = output;
        graph.add_compute_pass(
            format!("particle-renderer-{:?}-phase", self.kind),
            PassFlags::SIDE_EFFECT,
            |builder| {
                builder.read_storage_buffer(input);
                builder.read_uniform(params);
                records = builder.write_storage_buffer(output);
                move |context| {
                    context
                        .set_compute_pipeline(pipeline)
                        .set_compute_bind_group(0, bind_group, Vec::new())
                        .dispatch(groups, 1, 1);
                }
            },
        );
        Ok(ParticleAuxiliaryGraphOutputs { records })
    }

    /// Builds a graph containing this renderer's concrete compute phase.
    #[must_use]
    pub fn graph_frame(&self) -> RenderGraph {
        let mut graph = RenderGraph::new(rhi::Extent3d::new_2d(1, 1));
        self.append_to_graph(&mut graph)
            .expect("particle auxiliary graph admission");
        graph
    }

    /// Executes this phase and submits its compute dispatch.
    pub fn execute<D: RenderDevice, Q: RenderQueue>(
        &self,
        device: &D,
        queue: &Q,
    ) -> Result<RecordedGraph, CompileError> {
        self.graph_frame().execute(device, queue)
    }

    /// GPU output records written by the selected renderer phase.
    #[must_use]
    pub const fn output(&self) -> BufferId {
        self.output
    }

    /// Releases all GPU resources owned by this auxiliary phase.
    pub fn destroy<D: RenderDevice + ?Sized>(&self, device: &D) {
        device.destroy_bind_group(self.bind_group);
        device.destroy_compute_pipeline(self.pipeline);
        device.destroy_pipeline_layout(self.pipeline_layout);
        device.destroy_bind_group_layout(self.layout);
        device.destroy_shader_module(self.shader);
        for buffer in [self.input, self.output, self.params] {
            device.destroy_buffer(buffer);
        }
    }
}

/// The GPU-backed sort/cull and renderer-argument stage.
pub struct ParticleRendererGpu {
    capacity: u32,
    padded_capacity: u32,
    layout: SortCullBufferLayout,
    route: ParticleRendererRoute,
    input: BufferId,
    frustum: BufferId,
    params: BufferId,
    control: BufferId,
    sort_a: BufferId,
    sort_b: BufferId,
    visible_indices: BufferId,
    sprite_args: BufferId,
    indexed_args: BufferId,
    route_params: BufferId,
    cull_shader: ShaderModuleId,
    sort_shader: ShaderModuleId,
    compact_shader: ShaderModuleId,
    args_shader: ShaderModuleId,
    cull_layout: BindGroupLayoutId,
    cull_pipeline_layout: PipelineLayoutId,
    reset_pipeline: ComputePipelineId,
    seed_pipeline: ComputePipelineId,
    cull_pipeline: ComputePipelineId,
    cull_bind_group: BindGroupId,
    sort_layout: BindGroupLayoutId,
    sort_pipeline_layout: PipelineLayoutId,
    sort_pipeline: ComputePipelineId,
    sort_bind_groups: Vec<BindGroupId>,
    sort_steps: Vec<BufferId>,
    compact_layout: BindGroupLayoutId,
    compact_pipeline_layout: PipelineLayoutId,
    compact_pipeline: ComputePipelineId,
    compact_bind_group: BindGroupId,
    args_layout: BindGroupLayoutId,
    args_pipeline_layout: PipelineLayoutId,
    args_pipeline: ComputePipelineId,
    args_bind_group: BindGroupId,
    sorted_output: BufferId,
    surface_target: rhi::TextureId,
    surface_target_view: rhi::TextureViewId,
    surface_mesh_vertices: BufferId,
    surface_index_buffer: BufferId,
    surface_material: BufferId,
    surface_mesh_bytes: Vec<u8>,
    surface_index_bytes: Vec<u8>,
    surface_shader: ShaderModuleId,
    surface_bind_group_layout: BindGroupLayoutId,
    surface_pipeline_layout: PipelineLayoutId,
    surface_pipeline: RenderPipelineId,
    surface_bind_group: BindGroupId,
}

/// Executable non-surface particle phase for Light, Decal, or Volume.
///
/// The stage owns its input/output records and dispatches a renderer-specific
/// WGSL entry point; it is deliberately separate from indexed surface draws.
pub struct ParticleAuxiliaryGpu {
    kind: RendererKind,
    capacity: u32,
    input: BufferId,
    output: BufferId,
    params: BufferId,
    shader: ShaderModuleId,
    layout: BindGroupLayoutId,
    pipeline_layout: PipelineLayoutId,
    pipeline: ComputePipelineId,
    bind_group: BindGroupId,
}

/// Graph outputs from a real renderer-specific auxiliary dispatch.
#[derive(Clone, Copy, Debug)]
pub struct ParticleAuxiliaryGraphOutputs {
    /// The transformed per-particle light/decal/volume records.
    pub records: BufferHandle,
}

/// SSA versions produced by the renderer-matrix compute passes.
#[derive(Clone, Copy, Debug)]
pub struct ParticleRendererGraphOutputs {
    /// Source particle records consumed by the raster and auxiliary paths.
    pub input: BufferHandle,
    /// Compact visible particle indices consumed by the vertex shader.
    pub visible_indices: BufferHandle,
    /// Sprite non-indexed indirect arguments.
    pub sprite_args: BufferHandle,
    /// Mesh/Ribbon/Beam indexed indirect arguments.
    pub indexed_args: BufferHandle,
    /// Cull counter and route control values.
    pub control: BufferHandle,
}

impl ParticleRendererGpu {
    /// Creates all persistent buffers, pipelines and bind groups for one route.
    pub fn new<D: RenderDevice + ?Sized>(
        device: &D,
        capacity: u32,
        route: ParticleRendererRoute,
    ) -> Result<Self, ParticleRendererGpuError> {
        if capacity == 0 {
            return Err(ParticleRendererGpuError::ZeroCapacity);
        }
        let caps = device.capabilities();
        if caps.limits.max_bind_groups < 1 || caps.limits.max_bindings_per_bind_group < 5 {
            return Err(ParticleRendererGpuError::BindingLimitTooSmall {
                requested: 5,
                limit: caps.limits.max_bindings_per_bind_group,
            });
        }
        if caps.limits.max_compute_invocations_per_workgroup < 64 {
            return Err(ParticleRendererGpuError::BindingLimitTooSmall {
                requested: 64,
                limit: caps.limits.max_compute_invocations_per_workgroup,
            });
        }
        let padded_capacity = capacity
            .checked_next_power_of_two()
            .ok_or(ParticleRendererGpuError::CapacityOverflow)?;
        let max_storage = caps
            .limits
            .max_storage_buffer_binding_size
            .min(caps.limits.max_buffer_size);
        let layout = sort_cull_buffer_layout(capacity, 256, 64, max_storage)?;
        let workgroups = padded_capacity.div_ceil(64);
        if workgroups > caps.limits.max_compute_workgroups_per_dimension {
            return Err(ParticleRendererGpuError::DispatchTooLarge {
                workgroups,
                limit: caps.limits.max_compute_workgroups_per_dimension,
            });
        }

        let check_buffer = |resource: &'static str, size: u64| {
            if size > max_storage {
                Err(ParticleRendererGpuError::BufferTooLarge {
                    resource,
                    requested: size,
                    limit: max_storage,
                })
            } else {
                Ok(())
            }
        };
        let input_bytes = u64::from(capacity) * 32;
        let sort_bytes = u64::from(padded_capacity) * 8;
        let visible_bytes = u64::from(capacity) * 4;
        check_buffer("particle render input", input_bytes)?;
        check_buffer("particle sort items", sort_bytes)?;
        check_buffer("particle visible indices", visible_bytes)?;
        check_buffer("particle control", 16)?;
        check_buffer("particle renderer arguments", 40)?;

        let storage = BufferUsages::STORAGE | BufferUsages::COPY_SRC;
        let input = device.create_buffer(&BufferDescriptor::new(
            input_bytes,
            storage | BufferUsages::COPY_DST,
        ));
        let frustum = device.create_buffer(&BufferDescriptor::new(
            96,
            BufferUsages::UNIFORM | BufferUsages::COPY_DST,
        ));
        let params = device.create_buffer(&BufferDescriptor::new(
            64,
            BufferUsages::UNIFORM | BufferUsages::COPY_DST,
        ));
        let control = device.create_buffer(&BufferDescriptor::new(16, storage));
        let sort_a = device.create_buffer(&BufferDescriptor::new(sort_bytes, storage));
        let sort_b = device.create_buffer(&BufferDescriptor::new(sort_bytes, storage));
        let visible_indices = device.create_buffer(&BufferDescriptor::new(visible_bytes, storage));
        let sprite_args = device.create_buffer(&BufferDescriptor::new(
            16,
            BufferUsages::STORAGE | BufferUsages::INDIRECT | BufferUsages::COPY_SRC,
        ));
        let indexed_args = device.create_buffer(&BufferDescriptor::new(
            20,
            BufferUsages::STORAGE | BufferUsages::INDIRECT | BufferUsages::COPY_SRC,
        ));
        let route_params = device.create_buffer(&BufferDescriptor::new(
            48,
            BufferUsages::UNIFORM | BufferUsages::COPY_DST,
        ));
        let cull_shader = device.create_shader_module(&ShaderModuleDescriptor::wgsl(include_str!(
            "../shaders/particle_renderer_cull.wgsl"
        )));
        let sort_shader = device.create_shader_module(&ShaderModuleDescriptor::wgsl(include_str!(
            "../shaders/particle_renderer_sort.wgsl"
        )));
        let compact_shader = device.create_shader_module(&ShaderModuleDescriptor::wgsl(
            include_str!("../shaders/particle_renderer_compact.wgsl"),
        ));
        let args_shader = device.create_shader_module(&ShaderModuleDescriptor::wgsl(include_str!(
            "../shaders/particle_renderer_args.wgsl"
        )));

        let cull_layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some(String::from("particle-renderer-cull-bgl")),
            entries: vec![
                buffer_entry(0, BufferBindingType::Uniform, ShaderStages::COMPUTE),
                buffer_entry(
                    1,
                    BufferBindingType::Storage { read_only: true },
                    ShaderStages::COMPUTE,
                ),
                buffer_entry(
                    2,
                    BufferBindingType::Storage { read_only: false },
                    ShaderStages::COMPUTE,
                ),
                buffer_entry(
                    3,
                    BufferBindingType::Storage { read_only: false },
                    ShaderStages::COMPUTE,
                ),
                buffer_entry(4, BufferBindingType::Uniform, ShaderStages::COMPUTE),
            ],
        });
        let cull_pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some(String::from("particle-renderer-cull-pl")),
            bind_group_layouts: vec![cull_layout],
            push_constant_ranges: Vec::new(),
        });
        let reset_pipeline = make_compute(device, cull_pipeline_layout, cull_shader, "reset");
        let seed_pipeline = make_compute(device, cull_pipeline_layout, cull_shader, "seed");
        let cull_pipeline = make_compute(device, cull_pipeline_layout, cull_shader, "cull");
        let cull_bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some(String::from("particle-renderer-cull-bg")),
            layout: Some(cull_layout),
            entries: vec![
                bind_buffer(0, frustum),
                bind_buffer(1, input),
                bind_buffer(2, control),
                bind_buffer(3, sort_a),
                bind_buffer(4, params),
            ],
        });

        let sort_layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some(String::from("particle-renderer-sort-bgl")),
            entries: vec![
                buffer_entry(
                    0,
                    BufferBindingType::Storage { read_only: true },
                    ShaderStages::COMPUTE,
                ),
                buffer_entry(
                    1,
                    BufferBindingType::Storage { read_only: false },
                    ShaderStages::COMPUTE,
                ),
                buffer_entry(2, BufferBindingType::Uniform, ShaderStages::COMPUTE),
            ],
        });
        let sort_pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some(String::from("particle-renderer-sort-pl")),
            bind_group_layouts: vec![sort_layout],
            push_constant_ranges: Vec::new(),
        });
        let sort_pipeline = make_compute(device, sort_pipeline_layout, sort_shader, "sort_step");
        let mut sort_bind_groups = Vec::new();
        let mut sort_steps = Vec::new();
        let mut k = 2u32;
        while k <= padded_capacity {
            let mut j = k / 2;
            while j != 0 {
                let step = device.create_buffer(&BufferDescriptor::new(
                    16,
                    BufferUsages::UNIFORM | BufferUsages::COPY_DST,
                ));
                let source = if sort_bind_groups.len() % 2 == 0 {
                    sort_a
                } else {
                    sort_b
                };
                let destination = if sort_bind_groups.len() % 2 == 0 {
                    sort_b
                } else {
                    sort_a
                };
                let bind_group = device.create_bind_group(&BindGroupDescriptor {
                    label: Some(String::from("particle-renderer-sort-bg")),
                    layout: Some(sort_layout),
                    entries: vec![
                        bind_buffer(0, source),
                        bind_buffer(1, destination),
                        bind_buffer(2, step),
                    ],
                });
                sort_bind_groups.push(bind_group);
                sort_steps.push(step);
                if j == 1 {
                    break;
                }
                j /= 2;
            }
            if k > padded_capacity / 2 {
                break;
            }
            k *= 2;
        }
        let sorted_output = if sort_bind_groups.len() % 2 == 0 {
            sort_a
        } else {
            sort_b
        };

        let compact_layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some(String::from("particle-renderer-compact-bgl")),
            entries: vec![
                buffer_entry(
                    0,
                    BufferBindingType::Storage { read_only: true },
                    ShaderStages::COMPUTE,
                ),
                buffer_entry(
                    1,
                    BufferBindingType::Storage { read_only: false },
                    ShaderStages::COMPUTE,
                ),
                buffer_entry(2, BufferBindingType::Uniform, ShaderStages::COMPUTE),
                buffer_entry(
                    3,
                    BufferBindingType::Storage { read_only: true },
                    ShaderStages::COMPUTE,
                ),
            ],
        });
        let compact_pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some(String::from("particle-renderer-compact-pl")),
            bind_group_layouts: vec![compact_layout],
            push_constant_ranges: Vec::new(),
        });
        let compact_pipeline =
            make_compute(device, compact_pipeline_layout, compact_shader, "compact");
        let compact_bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some(String::from("particle-renderer-compact-bg")),
            layout: Some(compact_layout),
            entries: vec![
                bind_buffer(0, sorted_output),
                bind_buffer(1, visible_indices),
                bind_buffer(2, params),
                bind_buffer(3, control),
            ],
        });

        let args_layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some(String::from("particle-renderer-args-bgl")),
            entries: vec![
                buffer_entry(
                    0,
                    BufferBindingType::Storage { read_only: true },
                    ShaderStages::COMPUTE,
                ),
                buffer_entry(
                    1,
                    BufferBindingType::Storage { read_only: false },
                    ShaderStages::COMPUTE,
                ),
                buffer_entry(
                    2,
                    BufferBindingType::Storage { read_only: false },
                    ShaderStages::COMPUTE,
                ),
                buffer_entry(3, BufferBindingType::Uniform, ShaderStages::COMPUTE),
            ],
        });
        let args_pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some(String::from("particle-renderer-args-pl")),
            bind_group_layouts: vec![args_layout],
            push_constant_ranges: Vec::new(),
        });
        let args_pipeline = make_compute(device, args_pipeline_layout, args_shader, "emit_args");
        let args_bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some(String::from("particle-renderer-args-bg")),
            layout: Some(args_layout),
            entries: vec![
                bind_buffer(0, control),
                bind_buffer(1, sprite_args),
                bind_buffer(2, indexed_args),
                bind_buffer(3, route_params),
            ],
        });

        let surface_target = device.create_texture(&rhi::TextureDescriptor::new_2d(
            rhi::Extent3d::new_2d(RENDERER_MATRIX_TARGET_SIZE, RENDERER_MATRIX_TARGET_SIZE),
            TextureFormat::Rgba8Unorm,
            TextureUsages::RENDER_ATTACHMENT | TextureUsages::COPY_SRC,
        ));
        let surface_target_view =
            device.create_texture_view(surface_target, &TextureViewDescriptor::default());
        let surface_mesh_vertices = device.create_buffer(&BufferDescriptor::new(
            u64::from(self_index_capacity(route).max(3)) * SURFACE_MESH_VERTEX_STRIDE,
            BufferUsages::STORAGE | BufferUsages::COPY_DST,
        ));
        let surface_index_buffer = device.create_buffer(&BufferDescriptor::new(
            u64::from(surface_index_count(route).max(1)) * 4,
            BufferUsages::INDEX | BufferUsages::COPY_DST,
        ));
        let surface_material = device.create_buffer(&BufferDescriptor::new(
            SURFACE_MATERIAL_BYTES,
            BufferUsages::UNIFORM | BufferUsages::COPY_DST,
        ));
        let surface_shader = device.create_shader_module(&ShaderModuleDescriptor::wgsl(
            include_str!("../shaders/particle_renderer_surface.wgsl"),
        ));
        let surface_bind_group_layout =
            device.create_bind_group_layout(&BindGroupLayoutDescriptor {
                label: Some(String::from("particle-renderer-surface-bgl")),
                entries: vec![
                    buffer_entry(
                        0,
                        BufferBindingType::Storage { read_only: true },
                        ShaderStages::VERTEX,
                    ),
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
                    buffer_entry(
                        3,
                        BufferBindingType::Uniform,
                        ShaderStages::VERTEX | ShaderStages::FRAGMENT,
                    ),
                ],
            });
        let surface_pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some(String::from("particle-renderer-surface-pl")),
            bind_group_layouts: vec![surface_bind_group_layout],
            push_constant_ranges: Vec::new(),
        });
        let surface_pipeline = make_surface_pipeline(
            device,
            surface_pipeline_layout,
            surface_shader,
            route.kind(),
        );
        let surface_bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some(String::from("particle-renderer-surface-bg")),
            layout: Some(surface_bind_group_layout),
            entries: vec![
                bind_buffer(0, input),
                bind_buffer(1, visible_indices),
                bind_buffer(2, surface_mesh_vertices),
                bind_buffer(3, surface_material),
            ],
        });

        Ok(Self {
            capacity,
            padded_capacity,
            layout,
            route,
            input,
            frustum,
            params,
            control,
            sort_a,
            sort_b,
            visible_indices,
            sprite_args,
            indexed_args,
            route_params,
            cull_shader,
            sort_shader,
            compact_shader,
            args_shader,
            cull_layout,
            cull_pipeline_layout,
            reset_pipeline,
            seed_pipeline,
            cull_pipeline,
            cull_bind_group,
            sort_layout,
            sort_pipeline_layout,
            sort_pipeline,
            sort_bind_groups,
            sort_steps,
            compact_layout,
            compact_pipeline_layout,
            compact_pipeline,
            compact_bind_group,
            args_layout,
            args_pipeline_layout,
            args_pipeline,
            args_bind_group,
            sorted_output,
            surface_target,
            surface_target_view,
            surface_mesh_vertices,
            surface_index_buffer,
            surface_material,
            surface_mesh_bytes: surface_mesh_vertex_bytes(route),
            surface_index_bytes: surface_index_bytes(route),
            surface_shader,
            surface_bind_group_layout,
            surface_pipeline_layout,
            surface_pipeline,
            surface_bind_group,
        })
    }

    /// Uploads the source records and the frame's cull/sort constants.
    pub fn set_inputs(
        &self,
        queue: &impl RenderQueue,
        inputs: &[ParticleRenderInput],
        view: ParticleCullView,
    ) -> Result<(), ParticleRendererGpuError> {
        if inputs.len() > self.capacity as usize {
            return Err(ParticleRendererGpuError::InvalidInput);
        }
        if !view.validate() || !inputs.iter().copied().all(ParticleRenderInput::validate) {
            return Err(ParticleRendererGpuError::InvalidView);
        }
        let mut bytes = Vec::with_capacity(inputs.len() * 32);
        for input in inputs {
            for value in input.center_radius {
                bytes.extend_from_slice(&value.to_le_bytes());
            }
            bytes.extend_from_slice(&input.depth.to_le_bytes());
            bytes.extend_from_slice(&input.particle_index.to_le_bytes());
            bytes.extend_from_slice(&input.flags.to_le_bytes());
            bytes.extend_from_slice(&input.reserved.to_le_bytes());
        }
        queue.write_buffer(self.input, 0, &bytes);
        queue.write_buffer(self.frustum, 0, &view.frustum_bytes());
        queue.write_buffer(
            self.params,
            0,
            &view.params_bytes(self.capacity, inputs.len() as u32),
        );
        queue.write_buffer(self.route_params, 0, &self.route.params_bytes());
        queue.write_buffer(self.surface_mesh_vertices, 0, &self.surface_mesh_bytes);
        queue.write_buffer(self.surface_index_buffer, 0, &self.surface_index_bytes);
        queue.write_buffer(
            self.surface_material,
            0,
            &surface_material_bytes(self.route),
        );
        for (index, step) in self.sort_steps.iter().copied().enumerate() {
            let (k, j) = sort_step_at(self.padded_capacity, index);
            let bytes = [k.to_le_bytes(), j.to_le_bytes(), [0; 4], [0; 4]].concat();
            queue.write_buffer(step, 0, &bytes);
        }
        Ok(())
    }

    /// Replaces the source geometry consumed by the Mesh surface route.
    ///
    /// The route's index count is fixed when the stage is created. Geometry
    /// replacement therefore updates existing buffers and does not rebuild
    /// the renderer graph resources.
    pub fn set_mesh_geometry(
        &mut self,
        queue: &impl RenderQueue,
        vertices: &[ParticleMeshVertex],
        indices: &[u32],
    ) -> Result<(), ParticleRendererGpuError> {
        if self.route.kind() != RendererKind::Mesh {
            return Err(ParticleRendererGpuError::MeshGeometryUnsupported);
        }
        let expected_indices = surface_index_count(self.route) as usize;
        let vertex_capacity = self_index_capacity(self.route) as usize;
        if indices.len() != expected_indices || vertices.len() > vertex_capacity {
            return Err(ParticleRendererGpuError::MeshGeometryTooLarge);
        }
        if vertices.is_empty()
            || vertices.iter().copied().any(|vertex| !vertex.validate())
            || indices
                .iter()
                .any(|index| *index as usize >= vertices.len())
        {
            return Err(ParticleRendererGpuError::InvalidMeshGeometry);
        }

        let mut vertex_bytes =
            Vec::with_capacity(vertex_capacity * SURFACE_MESH_VERTEX_STRIDE as usize);
        for vertex in vertices {
            vertex_bytes.extend_from_slice(&vertex.position[0].to_le_bytes());
            vertex_bytes.extend_from_slice(&vertex.position[1].to_le_bytes());
            vertex_bytes.extend_from_slice(&vertex.uv[0].to_le_bytes());
            vertex_bytes.extend_from_slice(&vertex.uv[1].to_le_bytes());
        }
        vertex_bytes.resize(vertex_capacity * SURFACE_MESH_VERTEX_STRIDE as usize, 0);

        let mut index_bytes = Vec::with_capacity(indices.len() * 4);
        for index in indices {
            index_bytes.extend_from_slice(&index.to_le_bytes());
        }
        queue.write_buffer(self.surface_mesh_vertices, 0, &vertex_bytes);
        queue.write_buffer(self.surface_index_buffer, 0, &index_bytes);
        self.surface_mesh_bytes = vertex_bytes;
        self.surface_index_bytes = index_bytes;
        Ok(())
    }

    /// Appends the reset, cull, explicit sort, compaction, and renderer-args
    /// passes to an existing graph.
    pub fn append_to_graph(
        &self,
        graph: &mut RenderGraph,
    ) -> Result<ParticleRendererGraphOutputs, CompileError> {
        let input = import_buffer(
            graph,
            "particle-render-input",
            self.input,
            u64::from(self.capacity) * 32,
            BufferUsages::STORAGE | BufferUsages::COPY_SRC | BufferUsages::COPY_DST,
            BufferState::storage_read(PipelineStages::COMPUTE_SHADER),
        );
        let frustum = import_buffer(
            graph,
            "particle-render-frustum",
            self.frustum,
            96,
            BufferUsages::UNIFORM | BufferUsages::COPY_DST,
            BufferState::copy_dst(),
        );
        let params = import_buffer(
            graph,
            "particle-render-params",
            self.params,
            64,
            BufferUsages::UNIFORM | BufferUsages::COPY_DST,
            BufferState::copy_dst(),
        );
        let control = import_buffer(
            graph,
            "particle-render-control",
            self.control,
            16,
            BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            BufferState::storage_write(PipelineStages::COMPUTE_SHADER),
        );
        let sort_a = import_buffer(
            graph,
            "particle-render-sort-a",
            self.sort_a,
            u64::from(self.padded_capacity) * 8,
            BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            BufferState::storage_write(PipelineStages::COMPUTE_SHADER),
        );
        let sort_b = import_buffer(
            graph,
            "particle-render-sort-b",
            self.sort_b,
            u64::from(self.padded_capacity) * 8,
            BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            BufferState::storage_write(PipelineStages::COMPUTE_SHADER),
        );
        let visible = import_buffer(
            graph,
            "particle-render-visible",
            self.visible_indices,
            u64::from(self.capacity) * 4,
            BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            BufferState::storage_write(PipelineStages::COMPUTE_SHADER),
        );
        let sprite_args = import_buffer(
            graph,
            "particle-render-sprite-args",
            self.sprite_args,
            16,
            BufferUsages::STORAGE | BufferUsages::INDIRECT | BufferUsages::COPY_SRC,
            BufferState::storage_write(PipelineStages::COMPUTE_SHADER),
        );
        let indexed_args = import_buffer(
            graph,
            "particle-render-indexed-args",
            self.indexed_args,
            20,
            BufferUsages::STORAGE | BufferUsages::INDIRECT | BufferUsages::COPY_SRC,
            BufferState::storage_write(PipelineStages::COMPUTE_SHADER),
        );
        let mesh_vertices = import_buffer(
            graph,
            "particle-render-mesh-vertices",
            self.surface_mesh_vertices,
            u64::from(self_index_capacity(self.route).max(3)) * SURFACE_MESH_VERTEX_STRIDE,
            BufferUsages::STORAGE | BufferUsages::COPY_DST,
            BufferState::storage_read(PipelineStages::VERTEX_SHADER),
        );
        let index_buffer = import_buffer(
            graph,
            "particle-render-index-buffer",
            self.surface_index_buffer,
            u64::from(surface_index_count(self.route).max(1)) * 4,
            BufferUsages::INDEX | BufferUsages::COPY_DST,
            BufferState::index(),
        );
        let material = import_buffer(
            graph,
            "particle-render-material",
            self.surface_material,
            SURFACE_MATERIAL_BYTES,
            BufferUsages::UNIFORM | BufferUsages::COPY_DST,
            BufferState::copy_dst(),
        );
        let target = graph.import_texture(
            "particle-renderer-surface-target",
            TextureDesc::color(TextureFormat::Rgba8Unorm).with_size(SizeClass::absolute_2d(
                RENDERER_MATRIX_TARGET_SIZE,
                RENDERER_MATRIX_TARGET_SIZE,
            )),
            ImportedTexture {
                texture: self.surface_target,
                default_view: self.surface_target_view,
                entry_state: TextureState::initial(),
                exit_state: TextureState::color_target(),
                mip_levels: 1,
                array_layers: 1,
            },
        );

        let reset_pipeline = self.reset_pipeline;
        let reset_bind_group = self.cull_bind_group;
        let (control_reset, sprite_reset, indexed_reset) = {
            let mut control_reset = control;
            let mut sprite_reset = sprite_args;
            let mut indexed_reset = indexed_args;
            graph.add_compute_pass(
                "particle-renderer-reset",
                PassFlags::SIDE_EFFECT,
                |builder| {
                    control_reset = builder.write_storage_buffer(control);
                    sprite_reset = builder.write_storage_buffer(sprite_args);
                    indexed_reset = builder.write_storage_buffer(indexed_args);
                    move |context| {
                        context
                            .set_compute_pipeline(reset_pipeline)
                            .set_compute_bind_group(0, reset_bind_group, Vec::new())
                            .dispatch(1, 1, 1);
                    }
                },
            );
            (control_reset, sprite_reset, indexed_reset)
        };

        let seed_pipeline = self.seed_pipeline;
        let seed_bind_group = self.cull_bind_group;
        let seed_groups = self.padded_capacity.div_ceil(64);
        let sort_a_seed = {
            let mut output = sort_a;
            graph.add_compute_pass(
                "particle-renderer-seed",
                PassFlags::SIDE_EFFECT,
                |builder| {
                    output = builder.write_storage_buffer(sort_a);
                    move |context| {
                        context
                            .set_compute_pipeline(seed_pipeline)
                            .set_compute_bind_group(0, seed_bind_group, Vec::new())
                            .dispatch(seed_groups, 1, 1);
                    }
                },
            );
            output
        };

        let cull_pipeline = self.cull_pipeline;
        let cull_bind_group = self.cull_bind_group;
        let cull_groups = self.capacity.div_ceil(64);
        let (control_cull, sort_a_cull) = {
            let mut control_cull = control_reset;
            let mut sort_a_cull = sort_a_seed;
            graph.add_compute_pass(
                "particle-renderer-cull",
                PassFlags::SIDE_EFFECT,
                |builder| {
                    builder.read_uniform(frustum);
                    builder.read_uniform(params);
                    builder.read_storage_buffer(input);
                    control_cull = builder.write_storage_buffer(control_reset);
                    sort_a_cull = builder.write_storage_buffer(sort_a_seed);
                    move |context| {
                        context
                            .set_compute_pipeline(cull_pipeline)
                            .set_compute_bind_group(0, cull_bind_group, Vec::new())
                            .dispatch(cull_groups, 1, 1);
                    }
                },
            );
            (control_cull, sort_a_cull)
        };

        let mut sort_a_current = sort_a_cull;
        let mut sort_b_current = sort_b;
        if self.route.sort.is_explicit_sort() {
            for (index, bind_group) in self.sort_bind_groups.iter().copied().enumerate() {
                let (source, destination) = if index % 2 == 0 {
                    (sort_a_current, sort_b_current)
                } else {
                    (sort_b_current, sort_a_current)
                };
                let mut output = destination;
                let sort_pipeline = self.sort_pipeline;
                let sort_groups = self.padded_capacity.div_ceil(64);
                graph.add_compute_pass(
                    format!("particle-renderer-sort-{index}"),
                    PassFlags::SIDE_EFFECT,
                    |builder| {
                        builder.read_storage_buffer(source);
                        output = builder.write_storage_buffer(destination);
                        move |context| {
                            context
                                .set_compute_pipeline(sort_pipeline)
                                .set_compute_bind_group(0, bind_group, Vec::new())
                                .dispatch(sort_groups, 1, 1);
                        }
                    },
                );
                if index % 2 == 0 {
                    sort_b_current = output;
                } else {
                    sort_a_current = output;
                }
            }
        }
        let sorted = if self.route.sort.is_explicit_sort() {
            if self.sort_bind_groups.len() % 2 == 0 {
                sort_a_current
            } else {
                sort_b_current
            }
        } else {
            sort_a_cull
        };

        let compact_pipeline = self.compact_pipeline;
        let compact_bind_group = self.compact_bind_group;
        let compact_groups = self.capacity.div_ceil(64);
        let mut visible_output = visible;
        graph.add_compute_pass(
            "particle-renderer-compact",
            PassFlags::SIDE_EFFECT,
            |builder| {
                builder.read_storage_buffer(sorted);
                builder.read_uniform(params);
                builder.read_storage_buffer(control_cull);
                visible_output = builder.write_storage_buffer(visible);
                move |context| {
                    context
                        .set_compute_pipeline(compact_pipeline)
                        .set_compute_bind_group(0, compact_bind_group, Vec::new())
                        .dispatch(compact_groups, 1, 1);
                }
            },
        );

        let args_pipeline = self.args_pipeline;
        let args_bind_group = self.args_bind_group;
        let mut sprite_output = sprite_reset;
        let mut indexed_output = indexed_reset;
        graph.add_compute_pass(
            "particle-renderer-emit-args",
            PassFlags::SIDE_EFFECT,
            |builder| {
                builder.read_storage_buffer(control_cull);
                sprite_output = builder.write_storage_buffer(sprite_reset);
                indexed_output = builder.write_storage_buffer(indexed_reset);
                move |context| {
                    context
                        .set_compute_pipeline(args_pipeline)
                        .set_compute_bind_group(0, args_bind_group, Vec::new())
                        .dispatch(1, 1, 1);
                }
            },
        );
        let surface_pipeline = self.surface_pipeline;
        let surface_bind_group = self.surface_bind_group;
        let surface_index_buffer = self.surface_index_buffer;
        let renderer_route = self.route;
        let visible_for_raster = visible_output;
        let indexed_for_raster = indexed_output;
        let input_for_raster = input;
        graph.add_raster_pass(
            format!("particle-renderer-raster-{:?}", renderer_route.kind()),
            PassFlags::SIDE_EFFECT,
            |builder| {
                builder.read_storage_buffer(input_for_raster);
                builder.read_storage_buffer(visible_for_raster);
                builder.read_storage_buffer(mesh_vertices);
                builder.read_uniform(material);
                builder.read_index(index_buffer);
                match renderer_route.kind() {
                    RendererKind::Sprite => builder.read_indirect(sprite_output),
                    RendererKind::Mesh | RendererKind::Ribbon | RendererKind::Beam => {
                        builder.read_indirect(indexed_for_raster)
                    }
                    RendererKind::Light | RendererKind::Decal | RendererKind::Volume => {
                        unreachable!()
                    }
                };
                builder.color_attachment(target);
                move |context| {
                    context
                        .set_render_pipeline(surface_pipeline)
                        .set_render_bind_group(0, surface_bind_group, Vec::new())
                        .set_index_buffer(surface_index_buffer, IndexFormat::Uint32, 0);
                    match renderer_route.kind() {
                        RendererKind::Sprite => {
                            context.draw_indirect(sprite_output, 0);
                        }
                        RendererKind::Mesh | RendererKind::Ribbon | RendererKind::Beam => {
                            context.draw_indexed_indirect(indexed_for_raster, 0);
                        }
                        RendererKind::Light | RendererKind::Decal | RendererKind::Volume => {
                            unreachable!()
                        }
                    }
                }
            },
        );
        Ok(ParticleRendererGraphOutputs {
            input,
            visible_indices: visible_output,
            sprite_args: sprite_output,
            indexed_args: indexed_output,
            control: control_cull,
        })
    }

    /// Builds a graph containing only this stage's compute passes.
    #[must_use]
    pub fn graph_frame(&self) -> RenderGraph {
        let mut graph = RenderGraph::new(rhi::Extent3d::new_2d(1, 1));
        self.append_to_graph(&mut graph)
            .expect("particle renderer graph admission");
        graph
    }

    /// Executes this stage's graph and submits its command buffer.
    pub fn execute<D: RenderDevice, Q: RenderQueue>(
        &self,
        device: &D,
        queue: &Q,
    ) -> Result<RecordedGraph, CompileError> {
        self.graph_frame().execute(device, queue)
    }

    /// The source layout derived from the architecture sort/cull contract.
    #[must_use]
    pub const fn layout(&self) -> SortCullBufferLayout {
        self.layout
    }

    /// The renderer route used by this stage.
    #[must_use]
    pub const fn route(&self) -> ParticleRendererRoute {
        self.route
    }

    /// Capacity of the persistent source/visibility buffers.
    #[must_use]
    pub const fn capacity(&self) -> u32 {
        self.capacity
    }

    /// The GPU-authored compact visible-index buffer.
    #[must_use]
    pub const fn visible_indices(&self) -> BufferId {
        self.visible_indices
    }

    /// The GPU-authored non-indexed Sprite indirect argument buffer.
    #[must_use]
    pub const fn sprite_args(&self) -> BufferId {
        self.sprite_args
    }

    /// The GPU-authored indexed Mesh/Ribbon/Beam indirect argument buffer.
    #[must_use]
    pub const fn indexed_args(&self) -> BufferId {
        self.indexed_args
    }

    /// The GPU-authored visible counter buffer.
    #[must_use]
    pub const fn control(&self) -> BufferId {
        self.control
    }

    /// The offscreen target written by the renderer-specific raster pass.
    #[must_use]
    pub const fn raster_target(&self) -> rhi::TextureId {
        self.surface_target
    }

    /// The final sorted item buffer, or the cull output when sorting is disabled.
    #[must_use]
    pub const fn sorted_output(&self) -> BufferId {
        self.sorted_output
    }

    /// Destroys every persistent resource owned by the stage.
    pub fn destroy<D: RenderDevice + ?Sized>(&self, device: &D) {
        device.destroy_bind_group(self.args_bind_group);
        device.destroy_compute_pipeline(self.args_pipeline);
        device.destroy_pipeline_layout(self.args_pipeline_layout);
        device.destroy_bind_group_layout(self.args_layout);
        device.destroy_bind_group(self.compact_bind_group);
        device.destroy_compute_pipeline(self.compact_pipeline);
        device.destroy_pipeline_layout(self.compact_pipeline_layout);
        device.destroy_bind_group_layout(self.compact_layout);
        for &bind_group in &self.sort_bind_groups {
            device.destroy_bind_group(bind_group);
        }
        for &step in &self.sort_steps {
            device.destroy_buffer(step);
        }
        device.destroy_compute_pipeline(self.sort_pipeline);
        device.destroy_pipeline_layout(self.sort_pipeline_layout);
        device.destroy_bind_group_layout(self.sort_layout);
        device.destroy_bind_group(self.cull_bind_group);
        for pipeline in [self.reset_pipeline, self.seed_pipeline, self.cull_pipeline] {
            device.destroy_compute_pipeline(pipeline);
        }
        device.destroy_pipeline_layout(self.cull_pipeline_layout);
        device.destroy_bind_group_layout(self.cull_layout);
        device.destroy_shader_module(self.cull_shader);
        device.destroy_shader_module(self.sort_shader);
        device.destroy_shader_module(self.compact_shader);
        device.destroy_shader_module(self.args_shader);
        for buffer in [
            self.input,
            self.frustum,
            self.params,
            self.control,
            self.sort_a,
            self.sort_b,
            self.visible_indices,
            self.sprite_args,
            self.indexed_args,
            self.route_params,
            self.surface_mesh_vertices,
            self.surface_index_buffer,
            self.surface_material,
        ] {
            device.destroy_buffer(buffer);
        }
        device.destroy_render_pipeline(self.surface_pipeline);
        device.destroy_pipeline_layout(self.surface_pipeline_layout);
        device.destroy_bind_group_layout(self.surface_bind_group_layout);
        device.destroy_bind_group(self.surface_bind_group);
        device.destroy_shader_module(self.surface_shader);
        device.destroy_texture_view(self.surface_target_view);
        device.destroy_texture(self.surface_target);
    }
}

fn make_compute<D: RenderDevice + ?Sized>(
    device: &D,
    layout: PipelineLayoutId,
    shader: ShaderModuleId,
    entry_point: &'static str,
) -> ComputePipelineId {
    device.create_compute_pipeline(&ComputePipelineDescriptor {
        label: Some(format!("particle-renderer-{entry_point}")),
        layout: Some(layout),
        module: shader,
        entry_point: entry_point.into(),
    })
}

fn self_index_capacity(route: ParticleRendererRoute) -> u32 {
    match route.geometry {
        RendererGeometry::Sprite { .. } => SPRITE_QUAD_VERTEX_COUNT,
        RendererGeometry::Mesh { index_count, .. } => index_count.max(3),
        RendererGeometry::Ribbon {
            segments_per_chain, ..
        }
        | RendererGeometry::Beam {
            segments_per_chain, ..
        } => segments_per_chain.saturating_mul(4).max(4),
    }
}

fn surface_index_count(route: ParticleRendererRoute) -> u32 {
    match route.geometry {
        RendererGeometry::Sprite { .. } => 6,
        _ => route.index_count(),
    }
}

fn surface_mesh_vertex_bytes(route: ParticleRendererRoute) -> Vec<u8> {
    let count = self_index_capacity(route).max(3);
    let triangle = [[-0.48_f32, -0.40_f32], [0.48, -0.40], [0.0, 0.52]];
    let mut bytes = Vec::with_capacity(count as usize * SURFACE_MESH_VERTEX_STRIDE as usize);
    for index in 0..count {
        let point = triangle[(index % 3) as usize];
        bytes.extend_from_slice(&point[0].to_le_bytes());
        bytes.extend_from_slice(&point[1].to_le_bytes());
        bytes.extend_from_slice(&0.0_f32.to_le_bytes());
        bytes.extend_from_slice(&1.0_f32.to_le_bytes());
    }
    bytes
}

fn surface_index_bytes(route: ParticleRendererRoute) -> Vec<u8> {
    let count = surface_index_count(route);
    let mut bytes = Vec::with_capacity(count as usize * 4);
    for index in 0..count {
        let vertex = match route.geometry {
            RendererGeometry::Sprite { .. } => [0_u32, 1, 2, 0, 2, 3][(index % 6) as usize],
            RendererGeometry::Mesh { .. } => index,
            RendererGeometry::Ribbon { .. } | RendererGeometry::Beam { .. } => {
                let local = index % 6;
                let segment = index / 6;
                segment * 4 + [0_u32, 1, 2, 0, 2, 3][local as usize]
            }
        };
        bytes.extend_from_slice(&vertex.to_le_bytes());
    }
    bytes
}

fn surface_material_bytes(route: ParticleRendererRoute) -> [u8; SURFACE_MATERIAL_BYTES as usize] {
    let kind_tint: [f32; 4] = match route.kind() {
        RendererKind::Sprite => [1.0, 0.72, 0.20, 1.0],
        RendererKind::Mesh => [0.20, 0.82, 0.38, 1.0],
        RendererKind::Ribbon => [0.20, 0.58, 1.0, 1.0],
        RendererKind::Beam => [1.0, 0.24, 0.12, 1.0],
        RendererKind::Light | RendererKind::Decal | RendererKind::Volume => [1.0; 4],
    };
    let words = [
        1.0_f32.to_bits(),
        1.0_f32.to_bits(),
        1.0_f32.to_bits(),
        1.0_f32.to_bits(),
        kind_tint[0].to_bits(),
        kind_tint[1].to_bits(),
        kind_tint[2].to_bits(),
        kind_tint[3].to_bits(),
        0.08_f32.to_bits(),
        0.025_f32.to_bits(),
        0.0_f32.to_bits(),
        0.0_f32.to_bits(),
        route.first_instance(),
        0.0_f32.to_bits(),
        0.0_f32.to_bits(),
        0.0_f32.to_bits(),
    ];
    let mut bytes = [0; SURFACE_MATERIAL_BYTES as usize];
    for (index, word) in words.into_iter().enumerate() {
        bytes[index * 4..index * 4 + 4].copy_from_slice(&word.to_le_bytes());
    }
    bytes
}

fn make_surface_pipeline<D: RenderDevice + ?Sized>(
    device: &D,
    layout: PipelineLayoutId,
    shader: ShaderModuleId,
    kind: RendererKind,
) -> RenderPipelineId {
    let entry = match kind {
        RendererKind::Sprite => "vs_sprite",
        RendererKind::Mesh => "vs_mesh",
        RendererKind::Ribbon => "vs_ribbon",
        RendererKind::Beam => "vs_beam",
        RendererKind::Light | RendererKind::Decal | RendererKind::Volume => unreachable!(),
    };
    device.create_render_pipeline(&RenderPipelineDescriptor {
        label: Some(format!("particle-renderer-{kind:?}")),
        layout: Some(layout),
        vertex: VertexState {
            module: shader,
            entry_point: entry.into(),
            buffers: Vec::new(),
        },
        primitive: PrimitiveState {
            topology: PrimitiveTopology::TriangleList,
            cull_mode: None,
            ..PrimitiveState::default()
        },
        depth_stencil: None,
        multisample: MultisampleState::default(),
        fragment: Some(FragmentState {
            module: shader,
            entry_point: "fs".into(),
            targets: vec![Some(ColorTargetState::opaque(TextureFormat::Rgba8Unorm))],
        }),
    })
}

fn sort_step_at(padded_capacity: u32, index: usize) -> (u32, u32) {
    let mut remaining = index;
    let mut k = 2u32;
    while k <= padded_capacity {
        let mut j = k / 2;
        while j != 0 {
            if remaining == 0 {
                return (k, j);
            }
            remaining -= 1;
            if j == 1 {
                break;
            }
            j /= 2;
        }
        if k > padded_capacity / 2 {
            break;
        }
        k *= 2;
    }
    (padded_capacity, 1)
}

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

#[cfg(test)]
mod tests {
    use super::*;
    use prism_render_architecture::particle::indirect_draw::RendererGeometry;
    use prism_render_driver::Instance as _;
    use prism_render_driver_null::NullInstance;

    fn route(kind: RendererKind) -> ParticleRendererRoute {
        let geometry = match kind {
            RendererKind::Sprite => RendererGeometry::Sprite {
                alive_count: 16,
                first_instance: 2,
            },
            RendererKind::Mesh => RendererGeometry::Mesh {
                alive_count: 16,
                index_count: 36,
                first_instance: 2,
            },
            RendererKind::Ribbon => RendererGeometry::Ribbon {
                chain_count: 4,
                segments_per_chain: 8,
                first_instance: 2,
            },
            RendererKind::Beam => RendererGeometry::Beam {
                chain_count: 4,
                segments_per_chain: 8,
                first_instance: 2,
            },
            RendererKind::Light | RendererKind::Decal | RendererKind::Volume => {
                panic!("surface test route")
            }
        };
        ParticleRendererRoute::new(kind, SortStrategy::ViewDepthBitonic, geometry)
            .expect("valid route")
    }

    #[test]
    fn route_matrix_accepts_the_four_surface_geometry_contracts() {
        for kind in [
            RendererKind::Sprite,
            RendererKind::Mesh,
            RendererKind::Ribbon,
            RendererKind::Beam,
        ] {
            assert_eq!(route(kind).kind(), kind);
        }
        assert_eq!(route(RendererKind::Ribbon).index_count(), 48);
        assert_eq!(route(RendererKind::Beam).index_count(), 48);
    }

    #[test]
    fn null_rhi_records_the_complete_sort_cull_graph() {
        let instance = NullInstance::new();
        let adapter = instance
            .request_adapter(&Default::default())
            .expect("null adapter");
        let (device, queue) = instance
            .create_device(&adapter, &Default::default())
            .expect("null device");
        let stage = ParticleRendererGpu::new(&device, 7, route(RendererKind::Sprite))
            .expect("null stage resources");
        stage
            .set_inputs(
                &queue,
                &[ParticleRenderInput::new([0.0, 0.0, 0.0], 0.5, 0.2, 9)],
                ParticleCullView::unrestricted(),
            )
            .expect("valid input");
        let recorded = stage
            .execute(&device, &queue)
            .expect("graph executes on null RHI");
        assert!(recorded
            .commands()
            .passes
            .iter()
            .any(|pass| matches!(pass, prism_render_driver::Pass::Compute { .. })));
        stage.destroy(&device);
    }

    #[test]
    fn mesh_route_accepts_custom_geometry_and_rejects_invalid_geometry() {
        let instance = NullInstance::new();
        let adapter = instance
            .request_adapter(&Default::default())
            .expect("null adapter");
        let (device, queue) = instance
            .create_device(&adapter, &Default::default())
            .expect("null device");
        let mut stage = ParticleRendererGpu::new(
            &device,
            4,
            ParticleRendererRoute::new(
                RendererKind::Mesh,
                SortStrategy::ViewDepthBitonic,
                RendererGeometry::Mesh {
                    alive_count: 1,
                    index_count: 6,
                    first_instance: 0,
                },
            )
            .expect("mesh route"),
        )
        .expect("mesh stage");

        let vertices = [
            ParticleMeshVertex::new([-0.5, -0.5], [0.0, 0.0]),
            ParticleMeshVertex::new([0.5, -0.5], [1.0, 0.0]),
            ParticleMeshVertex::new([0.5, 0.5], [1.0, 1.0]),
            ParticleMeshVertex::new([-0.5, 0.5], [0.0, 1.0]),
        ];
        stage
            .set_mesh_geometry(&queue, &vertices, &[0, 1, 2, 0, 2, 3])
            .expect("quad geometry is accepted");
        assert_eq!(
            stage.set_mesh_geometry(&queue, &vertices, &[0, 1, 2]),
            Err(ParticleRendererGpuError::MeshGeometryTooLarge)
        );
        assert_eq!(
            stage.set_mesh_geometry(&queue, &vertices, &[0, 1, 9, 0, 2, 3]),
            Err(ParticleRendererGpuError::InvalidMeshGeometry)
        );
        stage.destroy(&device);
    }
}
