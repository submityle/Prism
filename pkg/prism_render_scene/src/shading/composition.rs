//! Resident response composition resources and command recording.
//!
//! The shading resolve producer writes the left response plane after shared
//! geometry, shadow and lighting terms have been evaluated. The resident
//! compositor then consumes a current-frame token once; an old frame is never
//! blended again and a missing token is a no-op, not a graph-topology change.
//!
//! Split-arm producers may replace the right plane and controls before calling
//! [`ViewResponseComposition::publish`]. The built-in resolve path supplies a
//! neutral right/control state, so ordinary materials retain the exact resolve
//! radiance while the same resident pass remains available for split plans.

#![deny(missing_docs)]

use bevy_ecs::prelude::*;
use bevy_log::error;
use bevy_math::UVec2;
use bevy_render::{
    render_resource::{
        BindGroup, BindGroupEntries, BindGroupLayout, Buffer, BufferDescriptor, BufferId,
        BufferInitDescriptor, BufferUsages, CommandEncoder, ComputePassDescriptor, ComputePipeline,
        ComputePipelineId, PipelineCompilationOptions, RawComputePipelineDescriptor,
        ShaderModuleDescriptor, ShaderSource, Texture, TextureDimension, TextureFormat, TextureId,
        TextureUsages,
    },
    renderer::{RenderContext, RenderDevice, ViewQuery},
};
use bytemuck::cast_slice;
use thiserror::Error;
use wgpu_types::Limits;

use super::{
    resources::{ViewVisibilityBuffer, SCENE_COLOR_FORMAT, VISIBILITY_ID_FORMAT},
    response_gpu::{
        GpuAuxiliaryMerge, GpuCompositionControl, GpuResponseOutput, GpuWaterCoverage,
        ResponseCompositionPipelines, RESPONSE_COMPOSITION_WORKGROUP_SIZE,
    },
    runtime::PrismShadingSettings,
};

pub(crate) const COMPOSITION_PASS_NAME: &str = "response_composition";
const HDR_WORKGROUP_SIZE: u32 = 8;
const HDR_BRIDGE_SHADER: &str = include_str!("../shaders/response_to_hdr.wgsl");

/// Typed mappings shared by the graph declaration and GPU allocation.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CompositionPlane {
    /// Left response, overwritten by the compositor, one 64-byte value per pixel.
    Left,
    /// Right response, one read-only 64-byte value per pixel.
    Right,
    /// One 16-byte composition control per pixel.
    Controls,
    /// Independent 64-byte typed auxiliary jobs, not a radiance channel.
    Auxiliary,
    /// One `u32` coverage flag per pixel for raster response producers.
    WaterCoverage,
}

impl CompositionPlane {
    /// Every resident composition plane, including planes not bound by the
    /// current compositor entry point.
    pub const ALL: [Self; 5] = [
        Self::Left,
        Self::Right,
        Self::Controls,
        Self::Auxiliary,
        Self::WaterCoverage,
    ];

    /// Stable resource name in the authoritative graph.
    pub const fn graph_name(self) -> &'static str {
        match self {
            Self::Left => "response_left",
            Self::Right => "response_right",
            Self::Controls => "response_controls",
            Self::Auxiliary => "response_auxiliary",
            Self::WaterCoverage => "water_coverage",
        }
    }

    /// Actual production storage stride; no padded `vec3` ABI is inferred.
    pub const fn stride(self) -> u64 {
        match self {
            Self::Left | Self::Right => size_of::<GpuResponseOutput>() as u64,
            Self::Controls => size_of::<GpuCompositionControl>() as u64,
            Self::Auxiliary => size_of::<GpuAuxiliaryMerge>() as u64,
            Self::WaterCoverage => size_of::<GpuWaterCoverage>() as u64,
        }
    }
}

/// Admission or target validation failure, always detected before recording work.
#[derive(Clone, Debug, PartialEq, Eq, Error)]
pub enum CompositionResourceError {
    /// Empty views cannot own storage planes.
    #[error("composition extent is empty")]
    EmptyExtent,
    /// The response shader's linear pixel index is a `u32`.
    #[error("composition pixel count exceeds u32")]
    PixelCountOverflow,
    /// The HDR and visibility targets exceed the device's 2D texture limit.
    #[error("composition extent {size:?} exceeds 2D texture limit {limit}")]
    ExtentLimit {
        /// Requested extent.
        size: UVec2,
        /// Device limit.
        limit: u32,
    },
    /// A plane exceeds either storage binding or buffer allocation capacity.
    #[error("{plane:?} needs {requested} bytes; device limit is {limit}")]
    BufferLimit {
        /// Typed plane that failed admission.
        plane: CompositionPlane,
        /// Exact byte count, without saturation or truncation.
        requested: u64,
        /// Smaller of the device's allocation and storage-binding limits.
        limit: u64,
    },
    /// The production kernel indexes only the X dimension of its dispatch.
    #[error("{plane:?} needs {requested} workgroups; X limit is {limit}")]
    DispatchLimit {
        /// Plane whose jobs exceeded the dispatch range.
        plane: CompositionPlane,
        /// Requested X workgroups.
        requested: u32,
        /// Device X workgroup-count limit.
        limit: u32,
    },
    /// The production kernels require 64 invocations, including an 8x8 HDR bridge.
    #[error("device cannot execute the composition workgroup shapes")]
    WorkgroupShape,
    /// Only full, single-sample, single-mip 2D textures of the specified ABI bind.
    #[error("invalid composition target {target}: {reason}")]
    InvalidTarget {
        /// Target's graph name.
        target: &'static str,
        /// Invariant violated by the supplied texture.
        reason: &'static str,
    },
    /// Bind groups must belong to the current left plane and extent.
    #[error("composition target belongs to another response allocation")]
    TargetMismatch,
    /// Auto-layout bind groups are tied to the exact pipelines that created them.
    #[error("composition bindings belong to another pipeline allocation")]
    PipelineMismatch,
}

/// Checked capacity mapping for the three pixel planes and optional auxiliary jobs.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ResponseCompositionLayout {
    size: UVec2,
    pixels: u32,
    auxiliary_jobs: u32,
}

impl ResponseCompositionLayout {
    /// Validates every dimension, byte count and dispatch before any GPU allocation.
    ///
    /// The parent kernel is one-dimensional. Oversized jobs are rejected rather
    /// than dispatched in Y (which would silently repeat its X-only indices).
    pub fn new(
        size: UVec2,
        auxiliary_jobs: u32,
        limits: &Limits,
    ) -> Result<Self, CompositionResourceError> {
        if size.x == 0 || size.y == 0 {
            return Err(CompositionResourceError::EmptyExtent);
        }
        let pixels = u32::try_from(u64::from(size.x) * u64::from(size.y))
            .map_err(|_| CompositionResourceError::PixelCountOverflow)?;
        if size.max_element() > limits.max_texture_dimension_2d {
            return Err(CompositionResourceError::ExtentLimit {
                size,
                limit: limits.max_texture_dimension_2d,
            });
        }
        if limits.max_compute_invocations_per_workgroup < RESPONSE_COMPOSITION_WORKGROUP_SIZE
            || limits.max_compute_workgroup_size_x < RESPONSE_COMPOSITION_WORKGROUP_SIZE
            || limits.max_compute_workgroup_size_y < HDR_WORKGROUP_SIZE
        {
            return Err(CompositionResourceError::WorkgroupShape);
        }
        let layout = Self {
            size,
            pixels,
            auxiliary_jobs,
        };
        let byte_limit = limits
            .max_buffer_size
            .min(u64::from(limits.max_storage_buffer_binding_size));
        for plane in CompositionPlane::ALL {
            let requested = layout.bytes(plane);
            if requested > byte_limit {
                return Err(CompositionResourceError::BufferLimit {
                    plane,
                    requested,
                    limit: byte_limit,
                });
            }
        }
        for (plane, count) in [
            (CompositionPlane::Left, pixels),
            (CompositionPlane::Auxiliary, auxiliary_jobs),
        ] {
            let requested = count.div_ceil(RESPONSE_COMPOSITION_WORKGROUP_SIZE);
            if requested > limits.max_compute_workgroups_per_dimension {
                return Err(CompositionResourceError::DispatchLimit {
                    plane,
                    requested,
                    limit: limits.max_compute_workgroups_per_dimension,
                });
            }
        }
        // A very narrow/tall view can fit the linear kernel yet exceed the
        // bridge's Y dispatch limit. Validate both spatial axes as well.
        let bridge_groups = size.max_element().div_ceil(HDR_WORKGROUP_SIZE);
        if bridge_groups > limits.max_compute_workgroups_per_dimension {
            return Err(CompositionResourceError::DispatchLimit {
                plane: CompositionPlane::Left,
                requested: bridge_groups,
                limit: limits.max_compute_workgroups_per_dimension,
            });
        }
        Ok(layout)
    }

    /// Full response/HDR extent.
    pub const fn size(self) -> UVec2 {
        self.size
    }

    /// Exactly one response and control element is allocated for each pixel.
    pub const fn pixel_count(self) -> u32 {
        self.pixels
    }

    /// Number of independently typed auxiliary jobs; zero means no auxiliary allocation.
    pub const fn auxiliary_jobs(self) -> u32 {
        self.auxiliary_jobs
    }

    /// Exact allocation/binding length for a typed plane, computed without narrowing.
    pub const fn bytes(self, plane: CompositionPlane) -> u64 {
        let count = match plane {
            CompositionPlane::Auxiliary => self.auxiliary_jobs,
            _ => self.pixels,
        };
        count as u64 * plane.stride()
    }
}

/// Production compositor plus an HDR export, installed regardless of scene contents.
#[derive(Resource)]
pub struct ResidentCompositionPipelines {
    kernels: ResponseCompositionPipelines,
    to_hdr: ComputePipeline,
}

impl ResidentCompositionPipelines {
    fn identities(&self) -> [ComputePipelineId; 3] {
        [
            self.kernels.radiance.id(),
            self.kernels.auxiliary.id(),
            self.to_hdr.id(),
        ]
    }

    /// Reuses the authoritative compositor. The extra shader only sums its five
    /// already-composed RGB channels and exports alpha to the shared HDR target.
    pub fn new(device: &RenderDevice) -> Self {
        let module = device
            .wgpu_device()
            .create_shader_module(ShaderModuleDescriptor {
                label: Some("prism response HDR export"),
                source: ShaderSource::Wgsl(HDR_BRIDGE_SHADER.into()),
            });
        let to_hdr = device.create_compute_pipeline(&RawComputePipelineDescriptor {
            label: Some("prism response HDR export"),
            layout: None,
            module: &module,
            entry_point: Some("response_to_hdr"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });
        Self {
            kernels: ResponseCompositionPipelines::new(device),
            to_hdr,
        }
    }
}

#[derive(Default)]
struct Publication(Option<u32>);

impl Publication {
    fn take(&mut self, frame: u32) -> bool {
        self.0.take() == Some(frame)
    }
}

/// Resident GPU resources attached to a render-view entity by a real split producer.
///
/// Allocation is opt-in data, not installation of a pass. Buffer usage supports
/// storage producers, explicit uploads and diagnostic readback. No CPU response
/// evaluator is called from this execution path.
#[derive(Component)]
pub struct ViewResponseComposition {
    layout: ResponseCompositionLayout,
    pipelines: [ComputePipelineId; 3],
    left: Buffer,
    right: Buffer,
    controls: Buffer,
    auxiliary: Option<Buffer>,
    water_coverage: Buffer,
    radiance_group: BindGroup,
    auxiliary_group: Option<BindGroup>,
    publication: Publication,
    last_error: Option<CompositionResourceError>,
}

impl ViewResponseComposition {
    /// Creates exactly sized planes after checking the complete request.
    pub fn new(
        device: &RenderDevice,
        pipelines: &ResidentCompositionPipelines,
        size: UVec2,
        auxiliary_jobs: u32,
    ) -> Result<Self, CompositionResourceError> {
        let layout = ResponseCompositionLayout::new(size, auxiliary_jobs, &device.limits())?;
        let allocate = |plane: CompositionPlane| {
            device.create_buffer(&BufferDescriptor {
                label: Some(plane.graph_name()),
                size: layout.bytes(plane),
                usage: BufferUsages::STORAGE | BufferUsages::COPY_DST | BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            })
        };
        let left = allocate(CompositionPlane::Left);
        let right = allocate(CompositionPlane::Right);
        let neutral_control = GpuCompositionControl {
            mask: 0.0,
            channels: 0,
            shared_alpha: 1.0,
            // Preserve the producer's alpha. This is the neutral rule for the
            // built-in single-arm path and remains valid for split producers
            // that explicitly rewrite controls before publication.
            alpha_rule: 1,
        };
        let controls = device.create_buffer_with_data(&BufferInitDescriptor {
            label: Some(CompositionPlane::Controls.graph_name()),
            contents: cast_slice(&vec![neutral_control; layout.pixel_count() as usize]),
            usage: BufferUsages::STORAGE | BufferUsages::COPY_DST | BufferUsages::COPY_SRC,
        });
        let auxiliary = (auxiliary_jobs != 0).then(|| allocate(CompositionPlane::Auxiliary));
        let water_coverage = device.create_buffer_with_data(&BufferInitDescriptor {
            label: Some(CompositionPlane::WaterCoverage.graph_name()),
            contents: cast_slice(&vec![
                GpuWaterCoverage::default();
                layout.pixel_count() as usize
            ]),
            usage: BufferUsages::STORAGE | BufferUsages::COPY_DST | BufferUsages::COPY_SRC,
        });
        let radiance_group = device.create_bind_group(
            "prism response composition radiance",
            &BindGroupLayout::from(pipelines.kernels.radiance.get_bind_group_layout(0)),
            &BindGroupEntries::sequential((
                left.as_entire_binding(),
                right.as_entire_binding(),
                controls.as_entire_binding(),
                water_coverage.as_entire_binding(),
            )),
        );
        // Auto layouts are entry-point specific. Auxiliary uses binding 4 in
        // its own entry point; it is not part of the radiance bind group.
        let auxiliary_group = auxiliary.as_ref().map(|buffer| {
            device.create_bind_group(
                "prism response composition auxiliary",
                &BindGroupLayout::from(pipelines.kernels.auxiliary.get_bind_group_layout(0)),
                &BindGroupEntries::with_indices(((4, buffer.as_entire_binding()),)),
            )
        });
        Ok(Self {
            layout,
            pipelines: pipelines.identities(),
            left,
            right,
            controls,
            auxiliary,
            water_coverage,
            radiance_group,
            auxiliary_group,
            publication: Publication::default(),
            last_error: None,
        })
    }

    /// Immutable checked dimensions and plane-to-byte mappings.
    pub const fn layout(&self) -> ResponseCompositionLayout {
        self.layout
    }

    /// The left response plane written by the shading resolve producer.
    pub(crate) fn left_buffer(&self) -> &Buffer {
        &self.left
    }

    /// The right response plane written by a split raster producer.
    pub(crate) fn right_buffer(&self) -> &Buffer {
        &self.right
    }

    /// The per-pixel composition controls written by a split raster producer.
    pub(crate) fn controls_buffer(&self) -> &Buffer {
        &self.controls
    }

    /// The per-pixel coverage flags written by a raster producer.
    pub(crate) fn water_coverage_buffer(&self) -> &Buffer {
        &self.water_coverage
    }

    /// Clears coverage before the opaque resolve publishes its response.
    pub(crate) fn clear_water_coverage(&self, encoder: &mut CommandEncoder) {
        encoder.clear_buffer(&self.water_coverage, 0, None);
    }

    /// Typed storage buffers filled by GPU producers. Auxiliary is absent when empty.
    pub fn plane(&self, plane: CompositionPlane) -> Option<&Buffer> {
        match plane {
            CompositionPlane::Left => Some(&self.left),
            CompositionPlane::Right => Some(&self.right),
            CompositionPlane::Controls => Some(&self.controls),
            CompositionPlane::Auxiliary => self.auxiliary.as_ref(),
            CompositionPlane::WaterCoverage => Some(&self.water_coverage),
        }
    }

    /// Publishes GPU work recorded before the composition system for this frame.
    ///
    /// Both arms and every control must have been rewritten in full; auxiliary
    /// jobs must contain their typed merge rules. The arms must use the same
    /// already-shadowed lighting. Publishing is not GPU completion notification:
    /// the command-order dependency makes those writes visible to this pass.
    /// A token is consumed once, even if the frame index has not advanced.
    pub fn publish(&mut self, frame_index: u32) {
        self.publication.0 = Some(frame_index);
        self.last_error = None;
    }

    /// Most recent resident target/recording validation error, exposed to producers.
    pub fn last_error(&self) -> Option<&CompositionResourceError> {
        self.last_error.as_ref()
    }

    /// Records the authoritative radiance and auxiliary kernels, then exports HDR.
    ///
    /// Returns `false` without recording when no fresh current-frame input exists.
    /// No allocation, bind-group creation, submission, mapping or CPU shading is
    /// performed here. Targets must have been prepared for this exact allocation.
    pub fn record(
        &mut self,
        pipelines: &ResidentCompositionPipelines,
        encoder: &mut CommandEncoder,
        target: &ResponseCompositionTarget,
        frame_index: u32,
    ) -> Result<bool, CompositionResourceError> {
        if self.pipelines != pipelines.identities() {
            return Err(CompositionResourceError::PipelineMismatch);
        }
        if target.left != self.left.id()
            || target.water_coverage != self.water_coverage.id()
            || target.size != self.layout.size
        {
            return Err(CompositionResourceError::TargetMismatch);
        }
        if !self.publication.take(frame_index) {
            return Ok(false);
        }
        {
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("prism response composition radiance"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&pipelines.kernels.radiance);
            pass.set_bind_group(0, &self.radiance_group, &[]);
            pass.dispatch_workgroups(
                self.layout
                    .pixels
                    .div_ceil(RESPONSE_COMPOSITION_WORKGROUP_SIZE),
                1,
                1,
            );
        }
        if let Some(group) = &self.auxiliary_group {
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("prism response composition auxiliary"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&pipelines.kernels.auxiliary);
            pass.set_bind_group(0, group, &[]);
            pass.dispatch_workgroups(
                self.layout
                    .auxiliary_jobs
                    .div_ceil(RESPONSE_COMPOSITION_WORKGROUP_SIZE),
                1,
                1,
            );
        }
        {
            // Separate compute passes make the in-place response writes visible
            // to the HDR bridge, without another response/lerp implementation.
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("prism response HDR export"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&pipelines.to_hdr);
            pass.set_bind_group(0, &target.group, &[]);
            pass.dispatch_workgroups(
                self.layout.size.x.div_ceil(HDR_WORKGROUP_SIZE),
                self.layout.size.y.div_ceil(HDR_WORKGROUP_SIZE),
                1,
            );
        }
        self.last_error = None;
        Ok(true)
    }
}

/// Cached full-view HDR/visibility bindings, invalidated by any allocation change.
#[derive(Component)]
pub struct ResponseCompositionTarget {
    group: BindGroup,
    left: BufferId,
    water_coverage: BufferId,
    hdr: TextureId,
    ids: TextureId,
    metadata: TextureId,
    size: UVec2,
}

impl ResponseCompositionTarget {
    /// Checks full textures before creating views or bind groups. Only matching
    /// single-mip, single-layer, single-sample 2D targets are accepted.
    pub fn new(
        device: &RenderDevice,
        pipelines: &ResidentCompositionPipelines,
        responses: &ViewResponseComposition,
        hdr: &Texture,
        ids: &Texture,
        metadata: &Texture,
    ) -> Result<Self, CompositionResourceError> {
        if responses.pipelines != pipelines.identities() {
            return Err(CompositionResourceError::PipelineMismatch);
        }
        for (name, texture, format, usage) in [
            (
                "hdr_scene_color",
                hdr,
                SCENE_COLOR_FORMAT,
                TextureUsages::STORAGE_BINDING,
            ),
            (
                "visibility_ids",
                ids,
                VISIBILITY_ID_FORMAT,
                TextureUsages::TEXTURE_BINDING,
            ),
            (
                "visibility_metadata",
                metadata,
                VISIBILITY_ID_FORMAT,
                TextureUsages::TEXTURE_BINDING,
            ),
        ] {
            validate_target(name, texture, responses.layout.size, format, usage)?;
        }
        let hdr_view = hdr.create_view(&Default::default());
        let ids_view = ids.create_view(&Default::default());
        let metadata_view = metadata.create_view(&Default::default());
        let group = device.create_bind_group(
            "prism response HDR export",
            &BindGroupLayout::from(pipelines.to_hdr.get_bind_group_layout(0)),
            &BindGroupEntries::sequential((
                responses.left.as_entire_binding(),
                &hdr_view,
                &ids_view,
                &metadata_view,
            )),
        );
        Ok(Self {
            group,
            left: responses.left.id(),
            water_coverage: responses.water_coverage.id(),
            hdr: hdr.id(),
            ids: ids.id(),
            metadata: metadata.id(),
            size: responses.layout.size,
        })
    }

    fn matches(
        &self,
        responses: &ViewResponseComposition,
        visibility: &ViewVisibilityBuffer,
    ) -> bool {
        let (ids, metadata, size) = visibility.feedback_sources();
        self.left == responses.left.id()
            && self.water_coverage == responses.water_coverage.id()
            && self.hdr == visibility.scene_color_texture().id()
            && self.ids == ids.id()
            && self.metadata == metadata.id()
            && self.size == size
            && self.size == responses.layout.size
    }
}

fn validate_target(
    target: &'static str,
    texture: &Texture,
    size: UVec2,
    format: TextureFormat,
    usage: TextureUsages,
) -> Result<(), CompositionResourceError> {
    let reason = if texture.dimension() != TextureDimension::D2
        || texture.depth_or_array_layers() != 1
        || texture.sample_count() != 1
        || texture.mip_level_count() != 1
    {
        "requires single-layer, single-sample, single-mip 2D texture"
    } else if texture.width() != size.x || texture.height() != size.y {
        "extent differs from response planes"
    } else if texture.format() != format {
        "format differs from HDR/visibility ABI"
    } else if !texture.usage().contains(usage) {
        "required storage/sampled usage is absent"
    } else {
        return Ok(());
    };
    Err(CompositionResourceError::InvalidTarget { target, reason })
}

pub(crate) fn init_response_composition(mut commands: Commands, device: Res<RenderDevice>) {
    commands.insert_resource(ResidentCompositionPipelines::new(&device));
}

/// Gives every visibility view a resident response allocation. The allocation
/// is sized from the actual view target, not from a fixed screen limit, and is
/// replaced atomically when a camera changes resolution.
pub(crate) fn prepare_response_composition_resources(
    mut commands: Commands,
    device: Res<RenderDevice>,
    pipelines: Res<ResidentCompositionPipelines>,
    views: Query<(
        Entity,
        &ViewVisibilityBuffer,
        Option<&ViewResponseComposition>,
    )>,
) {
    for (entity, visibility, existing) in &views {
        if existing.is_some_and(|responses| responses.layout().size() == visibility.size) {
            continue;
        }
        match ViewResponseComposition::new(&device, &pipelines, visibility.size, 0) {
            Ok(responses) => {
                commands.entity(entity).insert(responses);
            }
            Err(error) => {
                commands
                    .entity(entity)
                    .remove::<ViewResponseComposition>()
                    .remove::<ResponseCompositionTarget>();
                error!(
                    "failed to allocate resident response composition for view {:?}: {error}",
                    entity
                );
            }
        }
    }
}

pub(crate) fn prepare_response_composition_targets(
    mut commands: Commands,
    device: Res<RenderDevice>,
    pipelines: Res<ResidentCompositionPipelines>,
    mut views: Query<(
        Entity,
        &mut ViewResponseComposition,
        &ViewVisibilityBuffer,
        Option<&ResponseCompositionTarget>,
    )>,
) {
    for (entity, mut responses, visibility, old_target) in &mut views {
        if old_target.is_some_and(|target| target.matches(&responses, visibility)) {
            continue;
        }
        let (ids, metadata, _) = visibility.feedback_sources();
        match ResponseCompositionTarget::new(
            &device,
            &pipelines,
            &responses,
            visibility.scene_color_texture(),
            ids,
            metadata,
        ) {
            Ok(target) => {
                commands.entity(entity).insert(target);
                responses.last_error = None;
            }
            Err(error) => {
                commands
                    .entity(entity)
                    .remove::<ResponseCompositionTarget>();
                responses.last_error = Some(error);
            }
        }
    }
}

/// Stable scheduling anchors installed even when no view has split data.
#[derive(SystemSet, Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ResponseCompositionSystems {
    /// GPU arm producers must finish before this set; HDR consumers run after it.
    Compose,
}

pub(crate) fn response_composition_pass(
    settings: Res<PrismShadingSettings>,
    view: ViewQuery<(
        Option<&mut ViewResponseComposition>,
        Option<&ResponseCompositionTarget>,
        &ViewVisibilityBuffer,
    )>,
    pipelines: Res<ResidentCompositionPipelines>,
    mut context: RenderContext,
) {
    if !settings.enable_visibility_buffer {
        return;
    }
    let (Some(mut responses), target, visibility) = view.into_inner() else {
        return;
    };
    let Some(target) = target.filter(|target| target.matches(&responses, visibility)) else {
        return;
    };
    // Do not even request the lazy encoder for empty/stale input. The kernel
    // overwrites left, so a missed publication is consumed rather than replayed.
    if responses.publication.0 != Some(settings.frame_index) {
        responses.publication.0 = None;
        return;
    }
    if let Err(error) = responses.record(
        &pipelines,
        context.command_encoder(),
        target,
        settings.frame_index,
    ) {
        responses.last_error = Some(error);
    }
}

#[cfg(test)]
#[path = "../../text/unit/shading/composition.rs"]
mod tests;

#[cfg(test)]
#[path = "../../text/unit/shading/composition_ecs.rs"]
mod ecs_tests;
