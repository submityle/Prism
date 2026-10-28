//! Per-view projected Hair uploads and a bounded six-stage GPU raster slice.
//!
//! The adapter consumes the extracted groom owner, admits a validated CPU frame,
//! records the real queue upload and ordered geometry/visibility/deep-opacity/
//! shading dispatches, and keeps each SceneRhi lease alive until a tracked
//! completion is confirmed.

use bevy_render::render_resource::{
    BindGroup, BindGroupDescriptor, BindGroupEntry, BindGroupLayout, BindGroupLayoutEntry,
    BindingResource, BindingType, BufferBinding, BufferBindingType, BufferSize,
    ComputePassDescriptor, ComputePipeline, PipelineCompilationOptions, PipelineLayoutDescriptor,
    RawComputePipelineDescriptor as ComputePipelineDescriptor, ShaderModuleDescriptor,
    ShaderSource, ShaderStages,
};
use bevy_render::renderer::RenderDevice;
use prism_render_architecture::hair::{
    raster::{
        project_resampled_groom_with_id_base, HairRasterDispatchParams, HairSoftRasterAbi,
        ProjectedHairRasterSegmentError,
    },
    HairGroupHandle,
};
use prism_render_architecture::{
    gpu_scene::GpuCompletionValue,
    render_fabric::{
        PassKey, HAIR_DEEP_OPACITY_RESOURCE, HAIR_STRAND_RESOURCE, HAIR_VISIBILITY_RESOURCE,
    },
};
use prism_render_driver::{
    BufferDescriptor, BufferUsages as RhiBufferUsages, CompletionToken, Extent3d, FrameSync,
    ReadbackError, RenderQueue as _,
};
use prism_render_driver_wgpu::WgpuQueue;
use std::{fmt, sync::Arc};
use thiserror::Error;

use crate::fabric_domains::owners::HairRenderWorldOwner;
use crate::fabric_runtime::{
    SceneFabricCoordinatorInput, SceneFabricCoordinatorInputError, SceneFabricDispatch,
    SceneFabricPass, SceneFabricPushConstants, SceneFabricResource,
};
use crate::gpu::rhi::{SceneRhi, SceneRhiBuffer};

const HAIR_SEGMENT_STRIDE: u64 = 32;
const HAIR_GEOMETRY_STRIDE: u64 = 40;
const HAIR_PIXEL_RESULT_STRIDE: u64 = 16;
const HAIR_FABRIC_RESOURCE_BYTES: u64 = 256;

// These IDs are the stable Hair variant pass IDs in the frozen Render Fabric
// catalog. The architecture crate currently exports the resource keys but not
// these two pass constants, so this adapter keeps the catalog IDs local rather
// than inventing a second pass declaration.
const HAIR_VISIBILITY_PASS: PassKey = PassKey(0x5006);
const HAIR_DEEP_OPACITY_PASS: PassKey = PassKey(0x5008);
const HAIR_VISIBILITY_DISPATCH_COUNT: usize = 4;
const HAIR_DEEP_OPACITY_DISPATCH_COUNT: usize = 2;

/// One deterministic groom-to-segment mapping for downstream deferred shading.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct HairRasterGroomRange {
    /// Domain object identity supplied by the authoring/renderer producer.
    pub object: prism_render_architecture::render_fabric::DomainObjectHandle,
    /// Authored groom identity.
    pub groom: HairGroupHandle,
    /// First logical segment ID, including segments clipped from this view.
    pub first_segment_id: u32,
    /// Number of logical segment IDs reserved for this groom.
    pub segment_count: u32,
    /// Byte offset into the compact visible-segment upload slab.
    pub upload_offset_bytes: u64,
    /// Number of projected records present in the slab.
    pub visible_segment_count: u32,
}

/// CPU-prepared per-view groom projection, ready for a GPU recorder to upload.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct HairRasterViewFrame {
    /// Viewport extent used by projection and dispatch.
    pub viewport: [u32; 2],
    /// Stable mapping from segment IDs to their source groom.
    pub grooms: Vec<HairRasterGroomRange>,
    /// Compact tightly packed 32-byte little-endian segment records.
    pub upload_bytes: Vec<u8>,
    /// Number of projected records addressed by the dispatch.
    pub dispatch: Option<HairRasterDispatchParams>,
    /// Workgroup count for the segment-owned geometry/visibility passes.
    pub workgroups: u32,
    /// Workgroup count for the per-pixel deep-opacity and shading passes.
    pub pixel_workgroups: u32,
}

impl HairRasterViewFrame {
    /// Project all live owner snapshots for one extracted camera into upload bytes.
    pub fn from_owner(
        owner: &HairRenderWorldOwner,
        clip_from_world: &[[f32; 4]; 4],
        viewport: [u32; 2],
        abi: HairSoftRasterAbi,
        max_workgroups: u32,
    ) -> Result<Self, HairRasterPrepareError> {
        let mut frame = Self {
            viewport,
            ..Default::default()
        };
        let mut first_segment_id = 0_u32;
        for object in owner.iter() {
            let projected = project_resampled_groom_with_id_base(
                clip_from_world,
                object.geometry(),
                viewport,
                first_segment_id,
            )?;
            let segment_count = u32::try_from(projected.len())
                .map_err(|_| ProjectedHairRasterSegmentError::SegmentCountOutOfRange)?;
            let visible_segment_count = u32::try_from(projected.iter().flatten().count())
                .map_err(|_| ProjectedHairRasterSegmentError::SegmentCountOutOfRange)?;
            let upload_offset_bytes = u64::try_from(frame.upload_bytes.len())
                .map_err(|_| ProjectedHairRasterSegmentError::UploadSizeOutOfRange)?;
            let additional_bytes = usize::try_from(visible_segment_count)
                .ok()
                .and_then(|count| count.checked_mul(32))
                .ok_or(ProjectedHairRasterSegmentError::UploadSizeOutOfRange)?;
            frame
                .upload_bytes
                .try_reserve(additional_bytes)
                .map_err(|_| ProjectedHairRasterSegmentError::UploadSizeOutOfRange)?;
            for segment in projected.into_iter().flatten() {
                frame.upload_bytes.extend_from_slice(&segment.encode_le());
            }
            frame.grooms.push(HairRasterGroomRange {
                object: object.identity().object,
                groom: object.groom().handle,
                first_segment_id,
                segment_count,
                upload_offset_bytes,
                visible_segment_count,
            });
            first_segment_id = first_segment_id
                .checked_add(segment_count)
                .ok_or(ProjectedHairRasterSegmentError::SegmentIdOutOfRange)?;
        }

        let dispatch =
            HairRasterDispatchParams::try_new(viewport, abi, frame.upload_bytes.len() / 32)?;
        frame.workgroups = dispatch.dispatch_workgroups(max_workgroups)?;
        let pixel_count = u64::from(viewport[0])
            .checked_mul(u64::from(viewport[1]))
            .ok_or(HairRasterPrepareError::Admission(
                "viewport pixel count overflows u64",
            ))?;
        let pixel_workgroups = pixel_count
            .checked_add(63)
            .and_then(|count| count.checked_div(64))
            .ok_or(HairRasterPrepareError::Admission(
                "pixel workgroup count overflows u64",
            ))?;
        frame.pixel_workgroups = u32::try_from(pixel_workgroups).map_err(|_| {
            HairRasterPrepareError::Admission("pixel workgroup count is out of range")
        })?;
        if frame.pixel_workgroups > max_workgroups {
            return Err(HairRasterPrepareError::Admission(
                "pixel dispatch exceeds the device workgroup limit",
            ));
        }
        frame.dispatch = Some(dispatch);
        Ok(frame)
    }
}

/// Projection/ABI/device-limit failure for one view; no partial frame is published.
#[derive(Clone, Copy, Debug, Eq, PartialEq, thiserror::Error)]
pub enum HairRasterPrepareError {
    /// The source groom or target dimensions cannot be represented by the shader ABI.
    #[error("hair raster projection or ABI validation failed: {0}")]
    Projection(ProjectedHairRasterSegmentError),
    /// The projected frame does not match the dispatch and storage ABI.
    #[error("hair raster frame admission failed: {0}")]
    Admission(&'static str),
    /// The configured in-flight generation budget has no retired generation
    /// available for this frame.
    #[error("hair raster generation budget exhausted")]
    GenerationBudgetExhausted,
    /// The monotonically increasing content version cannot advance.
    #[error("hair raster content version exhausted")]
    VersionExhausted,
    /// Immediate constants are required by the six production kernels.
    #[error("hair raster device does not support the 20-byte immediate ABI")]
    ImmediateAbiUnsupported,
}

impl From<ProjectedHairRasterSegmentError> for HairRasterPrepareError {
    fn from(error: ProjectedHairRasterSegmentError) -> Self {
        Self::Projection(error)
    }
}

/// Capacity of one real SceneRhi-backed Hair raster generation.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct HairRasterCapacity {
    /// Allocated bytes for the compact segment upload. Empty frames retain the
    /// shader-required one-record allocation but admit zero logical segments.
    pub segment_bytes: u64,
    /// Allocated bytes for the generated geometry stream.
    pub geometry_bytes: u64,
    /// Allocated u32 elements in each per-pixel visibility output.
    pub pixel_count: u64,
    /// Physical bytes allocated for the exported visibility buffer. The
    /// logical readback remains `pixel_count * 4`; the floor is required by
    /// the canonical Fabric resource template for small views.
    pub visibility_bytes: u64,
    /// Physical bytes allocated for the exported deep-opacity buffer. The
    /// logical readback remains `pixel_count * 16`; the floor is required by
    /// the canonical Fabric resource template for small views.
    pub deep_opacity_bytes: u64,
}

impl HairRasterCapacity {
    fn for_frame(frame: &HairRasterViewFrame) -> Result<Self, HairRasterPrepareError> {
        let dispatch = frame.dispatch.ok_or(HairRasterPrepareError::Admission(
            "missing dispatch parameters",
        ))?;
        let upload_bytes = u64::try_from(frame.upload_bytes.len())
            .map_err(|_| HairRasterPrepareError::Admission("upload length overflows u64"))?;
        if upload_bytes % 32 != 0 {
            return Err(HairRasterPrepareError::Admission(
                "segment upload is not a multiple of the 32-byte ABI stride",
            ));
        }
        if usize::try_from(dispatch_segment_count(dispatch)).ok()
            != Some(frame.upload_bytes.len() / 32)
        {
            return Err(HairRasterPrepareError::Admission(
                "dispatch segment count does not match the upload slab",
            ));
        }
        let pixel_count = u64::from(frame.viewport[0])
            .checked_mul(u64::from(frame.viewport[1]))
            .ok_or(HairRasterPrepareError::Admission(
                "viewport pixel count overflows u64",
            ))?;
        if pixel_count == 0 {
            return Err(HairRasterPrepareError::Admission("empty viewport"));
        }
        let visibility_bytes = pixel_count
            .checked_mul(4)
            .ok_or(HairRasterPrepareError::Admission(
                "visibility buffer size overflows u64",
            ))?
            .max(HAIR_FABRIC_RESOURCE_BYTES);
        let deep_opacity_bytes = pixel_count
            .checked_mul(HAIR_PIXEL_RESULT_STRIDE)
            .ok_or(HairRasterPrepareError::Admission(
                "deep-opacity buffer size overflows u64",
            ))?
            .max(HAIR_FABRIC_RESOURCE_BYTES);
        Ok(Self {
            segment_bytes: upload_bytes
                .max(HAIR_SEGMENT_STRIDE)
                .max(HAIR_FABRIC_RESOURCE_BYTES),
            geometry_bytes: u64::from(dispatch_segment_count(dispatch))
                .checked_mul(HAIR_GEOMETRY_STRIDE)
                .ok_or(HairRasterPrepareError::Admission(
                    "geometry stream size overflows u64",
                ))?
                .max(HAIR_GEOMETRY_STRIDE),
            pixel_count,
            visibility_bytes,
            deep_opacity_bytes,
        })
    }

    fn covers(self, required: Self) -> bool {
        self.segment_bytes >= required.segment_bytes
            && self.geometry_bytes >= required.geometry_bytes
            && self.pixel_count >= required.pixel_count
            && self.visibility_bytes >= required.visibility_bytes
            && self.deep_opacity_bytes >= required.deep_opacity_bytes
    }
}

fn dispatch_segment_count(dispatch: HairRasterDispatchParams) -> u32 {
    // HairRasterDispatchParams intentionally exposes only its encoded ABI. The
    // final word is the segment count and using the ABI bytes keeps this
    // adapter independent of the architecture module's private fields.
    u32::from_le_bytes(
        dispatch.encode_le()[16..20]
            .try_into()
            .expect("20-byte ABI"),
    )
}

/// A completed GPU readback from the ordered Hair raster production chain.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct HairRasterReadback {
    /// Raw 40-byte geometry records represented as little-endian u32 words.
    pub geometry: Vec<u32>,
    pub depth: Vec<u32>,
    pub ids: Vec<u32>,
    pub coverage: Vec<u32>,
    /// Per-pixel `[near_transmittance, far_transmittance, visible_alpha, product]`.
    pub deep_opacity: Vec<[f32; 4]>,
    /// Per-pixel `[r, g, b, coverage]` shading output.
    pub shading: Vec<[f32; 4]>,
}

/// Pipeline set for the production Hair raster recorder.
#[derive(Clone, Debug)]
pub(crate) struct HairRasterPipelineSet {
    layout: BindGroupLayout,
    geometry: ComputePipeline,
    depth: ComputePipeline,
    resolve: ComputePipeline,
    publish: ComputePipeline,
    deep_opacity: ComputePipeline,
    shading: ComputePipeline,
}

/// Failure while lowering compiled Hair raster WGSL into native pipelines.
#[derive(Clone, Debug, Eq, PartialEq, thiserror::Error)]
pub enum HairRasterPipelineError {
    #[error("compiled hair raster shader has no entry point containing {0}")]
    MissingEntry(&'static str),
}

impl HairRasterPipelineSet {
    /// Builds and caches the six native pipelines from the renderer's
    /// already-compiled WESL/WGSL source. The source is never synthesized by
    /// this adapter; shader compilation and permutation selection stay with the
    /// renderer's shader service.
    pub fn from_wgsl(rhi: &SceneRhi, wgsl: &str) -> Result<Self, HairRasterPipelineError> {
        let geometry_entry = find_entry_point(wgsl, "hair_raster_geometry")?;
        let depth_entry = find_entry_point(wgsl, "hair_raster_depth")?;
        let resolve_entry = find_entry_point(wgsl, "hair_raster_resolve")?;
        let publish_entry = find_entry_point(wgsl, "hair_raster_publish")?;
        let deep_opacity_entry = find_entry_point(wgsl, "hair_raster_deep_opacity")?;
        let shading_entry = find_entry_point(wgsl, "hair_raster_shading")?;
        let device = RenderDevice::from(rhi.device().wgpu_device().clone());
        let layout = device.create_bind_group_layout(
            "prism hair raster group 0",
            &[
                storage_binding(0, true),
                storage_binding(1, false),
                storage_binding(2, false),
                storage_binding(3, false),
                storage_binding(4, false),
                storage_binding(5, false),
                storage_binding(6, false),
            ],
        );
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("prism hair raster pipeline layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 20,
        });
        let module = device.create_and_validate_shader_module(ShaderModuleDescriptor {
            label: Some("prism hair raster compiled shader"),
            source: ShaderSource::Wgsl(wgsl.into()),
        });
        let pipeline = |entry: String| {
            device.create_compute_pipeline(&ComputePipelineDescriptor {
                label: Some("prism hair raster pass"),
                layout: Some(&pipeline_layout),
                module: &module,
                entry_point: Some(&entry),
                compilation_options: PipelineCompilationOptions::default(),
                cache: None,
            })
        };
        Ok(Self {
            layout,
            geometry: pipeline(geometry_entry),
            depth: pipeline(depth_entry),
            resolve: pipeline(resolve_entry),
            publish: pipeline(publish_entry),
            deep_opacity: pipeline(deep_opacity_entry),
            shading: pipeline(shading_entry),
        })
    }
}

fn storage_binding(binding: u32, read_only: bool) -> BindGroupLayoutEntry {
    BindGroupLayoutEntry {
        binding,
        visibility: ShaderStages::COMPUTE,
        ty: BindingType::Buffer {
            ty: BufferBindingType::Storage { read_only },
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

fn find_entry_point(source: &str, needle: &'static str) -> Result<String, HairRasterPipelineError> {
    source
        .lines()
        .map(str::trim_start)
        .filter_map(|line| line.strip_prefix("fn "))
        .filter_map(|rest| rest.split_once('(').map(|(name, _)| name))
        .find(|name| name.contains(needle))
        .map(str::to_owned)
        .ok_or(HairRasterPipelineError::MissingEntry(needle))
}

struct HairRasterGeneration {
    segment: SceneRhiBuffer,
    geometry: SceneRhiBuffer,
    depth: SceneRhiBuffer,
    ids: SceneRhiBuffer,
    coverage: SceneRhiBuffer,
    deep_opacity: SceneRhiBuffer,
    shading: SceneRhiBuffer,
    capacity: HairRasterCapacity,
    version: u32,
}

impl HairRasterGeneration {
    fn allocate(rhi: &SceneRhi, capacity: HairRasterCapacity, version: u32) -> Self {
        let storage = RhiBufferUsages::STORAGE;
        Self {
            segment: rhi.create_buffer(&BufferDescriptor {
                label: Some("prism hair raster segments".into()),
                size: capacity.segment_bytes,
                usage: storage | RhiBufferUsages::COPY_DST,
                mapped_at_creation: false,
            }),
            geometry: rhi.create_buffer(&BufferDescriptor {
                label: Some("prism hair raster geometry".into()),
                size: capacity.geometry_bytes,
                usage: storage | RhiBufferUsages::COPY_SRC,
                mapped_at_creation: false,
            }),
            depth: rhi.create_buffer(&BufferDescriptor {
                label: Some("prism hair raster depth".into()),
                size: capacity.visibility_bytes,
                usage: storage | RhiBufferUsages::COPY_DST | RhiBufferUsages::COPY_SRC,
                mapped_at_creation: false,
            }),
            ids: rhi.create_buffer(&BufferDescriptor {
                label: Some("prism hair raster ids".into()),
                size: capacity.visibility_bytes,
                usage: storage | RhiBufferUsages::COPY_DST | RhiBufferUsages::COPY_SRC,
                mapped_at_creation: false,
            }),
            coverage: rhi.create_buffer(&BufferDescriptor {
                label: Some("prism hair raster coverage".into()),
                size: capacity.visibility_bytes,
                usage: storage | RhiBufferUsages::COPY_DST | RhiBufferUsages::COPY_SRC,
                mapped_at_creation: false,
            }),
            deep_opacity: rhi.create_buffer(&BufferDescriptor {
                label: Some("prism hair raster deep opacity".into()),
                size: capacity.deep_opacity_bytes,
                usage: storage | RhiBufferUsages::COPY_SRC,
                mapped_at_creation: false,
            }),
            shading: rhi.create_buffer(&BufferDescriptor {
                label: Some("prism hair raster shading".into()),
                size: capacity.deep_opacity_bytes,
                usage: storage | RhiBufferUsages::COPY_SRC,
                mapped_at_creation: false,
            }),
            capacity,
            version,
        }
    }

    fn segment_id(&self) -> prism_render_driver::BufferId {
        self.segment.id()
    }
}

/// Error raised while attaching an admitted Hair production slice to a shared
/// SceneFabric batch.
///
/// A batch adapter is intentionally stricter than the legacy direct path:
/// there must be at least one real segment dispatch. An empty owner is a valid
/// direct-path clear/readback frame, but it is not evidence for a Fabric pass.
#[derive(Clone, Copy, Debug, Eq, Error, PartialEq)]
pub(crate) enum HairRasterBatchError {
    #[error("hair raster batch has no renderable segment work")]
    NoRenderableSegments,
    #[error("hair raster host completion token is zero")]
    InvalidCompletionToken,
    #[error(transparent)]
    Input(#[from] SceneFabricCoordinatorInputError),
}

/// Failure while building the real Hair contribution for one coordinator
/// frame. The packet is only created after owner projection, GPU admission,
/// resource upload and bind-group construction all succeed.
#[derive(Debug, Error)]
pub(crate) enum HairRasterDomainPacketError {
    #[error(transparent)]
    Prepare(#[from] HairRasterPrepareError),
    #[error(transparent)]
    Batch(#[from] HairRasterBatchError),
}

/// Owned Hair resources and recorders for one future unified-host batch.
///
/// The adapter owns the real generation and bind group. It must remain alive
/// from `append_to` through host command recording and until
/// [`HairRasterGpuState::complete_batch_adapter`] observes the host's actual
/// completion value. It never creates a semantic fallback resource and it
/// never claims coverage for passes outside the two Hair catalog passes.
pub(crate) struct HairRasterBatchAdapter {
    frame: u64,
    extent: Extent3d,
    version: u32,
    dispatch: HairRasterDispatchParams,
    segment_workgroups: u32,
    pixel_workgroups: u32,
    generation: Option<HairRasterGeneration>,
    pipelines: HairRasterPipelineSet,
    bind_group: BindGroup,
}

/// Real per-view Hair contribution prepared from the extracted render-world
/// owner for one shared SceneFabric batch.
///
/// The packet owns the GPU adapter and the deterministic groom-to-segment
/// mapping produced from [`HairRenderWorldOwner`]. It contributes exactly the
/// frozen Hair visibility and deep-opacity passes; all command recording is
/// typed and appended to the caller's existing coordinator input.
pub(crate) struct HairRasterDomainPacket {
    adapter: HairRasterBatchAdapter,
    grooms: Vec<HairRasterGroomRange>,
}

impl fmt::Debug for HairRasterDomainPacket {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("HairRasterDomainPacket")
            .field("frame", &self.adapter.frame())
            .field("version", &self.adapter.version())
            .field("grooms", &self.grooms)
            .field("active_passes", &self.adapter.active_passes())
            .finish()
    }
}

impl HairRasterDomainPacket {
    /// Projects the current production owner snapshot and prepares its real
    /// SceneRhi generation for the shared coordinator. No private command
    /// buffer or synthetic resource is created here.
    pub(crate) fn prepare_from_owner(
        state: &mut HairRasterGpuState,
        rhi: &SceneRhi,
        frame: u64,
        owner: &HairRenderWorldOwner,
        clip_from_world: &[[f32; 4]; 4],
        viewport: [u32; 2],
        abi: HairSoftRasterAbi,
        max_workgroups: u32,
        pipelines: &HairRasterPipelineSet,
    ) -> Result<Self, HairRasterDomainPacketError> {
        let view =
            HairRasterViewFrame::from_owner(owner, clip_from_world, viewport, abi, max_workgroups)?;
        require_segment_work(view.workgroups)?;
        let grooms = view.grooms.clone();
        let adapter = state.prepare_batch_adapter(rhi, frame, view, pipelines)?;
        Ok(Self { adapter, grooms })
    }

    /// Appends this packet's real resources and typed compute recorders to the
    /// caller's single coordinator input.
    pub(crate) fn append_to<'a, 'lease>(
        &'a self,
        batch: &mut SceneFabricCoordinatorInput<'a, 'lease>,
    ) -> Result<(), HairRasterBatchError> {
        self.adapter.append_to(batch)
    }

    /// Deterministic groom ranges emitted by the current owner snapshot.
    #[must_use]
    pub(crate) fn grooms(&self) -> &[HairRasterGroomRange] {
        &self.grooms
    }

    /// Frame identity carried by all resources in this packet.
    #[must_use]
    pub(crate) const fn frame(&self) -> u64 {
        self.adapter.frame()
    }

    /// Content version carried by all resources in this packet.
    #[must_use]
    pub(crate) const fn version(&self) -> u32 {
        self.adapter.version()
    }

    /// Exact catalog passes contributed by this packet.
    #[must_use]
    pub(crate) const fn active_passes(&self) -> [PassKey; 2] {
        self.adapter.active_passes()
    }

    /// Returns the prepared adapter for host submission/completion ownership.
    pub(crate) fn into_adapter(self) -> HairRasterBatchAdapter {
        self.adapter
    }
}

impl fmt::Debug for HairRasterBatchAdapter {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("HairRasterBatchAdapter")
            .field("frame", &self.frame)
            .field("extent", &self.extent)
            .field("version", &self.version)
            .field("segment_workgroups", &self.segment_workgroups)
            .field("pixel_workgroups", &self.pixel_workgroups)
            .field(
                "generation",
                &self
                    .generation
                    .as_ref()
                    .map(|generation| generation.version),
            )
            .finish()
    }
}

impl HairRasterBatchAdapter {
    fn from_prepared(
        rhi: &SceneRhi,
        prepared: HairRasterPreparedFrame,
        pipelines: &HairRasterPipelineSet,
    ) -> Result<Self, HairRasterPrepareError> {
        let HairRasterPreparedFrame {
            frame,
            view,
            generation,
            dispatch,
        } = prepared;
        let generation = generation.ok_or(HairRasterPrepareError::Admission(
            "prepared frame already consumed",
        ))?;
        let (pixel_count, _geometry_words) = recording_shape(&view, dispatch)?;
        upload_generation(rhi, &generation, &view, pixel_count);
        let bind_group = create_bind_group(rhi, &generation, pipelines);
        let version = generation.version;
        Ok(Self {
            frame,
            extent: Extent3d::new_2d(view.viewport[0], view.viewport[1]),
            version,
            dispatch,
            segment_workgroups: view.workgroups,
            pixel_workgroups: view.pixel_workgroups,
            generation: Some(generation),
            pipelines: pipelines.clone(),
            bind_group,
        })
    }

    /// Appends the real Hair resources and typed recorders to a shared batch.
    ///
    /// The four segment-owned kernels are grouped under the catalog's
    /// visibility pass and the two pixel-owned kernels under deep-opacity.
    /// Runtime ordering supplies the visibility-to-deep-opacity dependency;
    /// this method does not declare or fake any other pass.
    pub(crate) fn append_to<'a, 'lease>(
        &'a self,
        batch: &mut SceneFabricCoordinatorInput<'a, 'lease>,
    ) -> Result<(), HairRasterBatchError> {
        require_segment_work(self.segment_workgroups)?;
        let generation = self
            .generation
            .as_ref()
            .ok_or(HairRasterBatchError::NoRenderableSegments)?;
        let segment_groups = [self.segment_workgroups, 1, 1];
        let pixel_groups = [self.pixel_workgroups, 1, 1];
        let visibility_dispatches = Vec::from([
            hair_scene_fabric_dispatch(
                &self.pipelines.geometry,
                &self.bind_group,
                self.dispatch,
                segment_groups,
            ),
            hair_scene_fabric_dispatch(
                &self.pipelines.depth,
                &self.bind_group,
                self.dispatch,
                segment_groups,
            ),
            hair_scene_fabric_dispatch(
                &self.pipelines.resolve,
                &self.bind_group,
                self.dispatch,
                segment_groups,
            ),
            hair_scene_fabric_dispatch(
                &self.pipelines.publish,
                &self.bind_group,
                self.dispatch,
                segment_groups,
            ),
        ]);
        let deep_opacity_dispatches = Vec::from([
            hair_scene_fabric_dispatch(
                &self.pipelines.deep_opacity,
                &self.bind_group,
                self.dispatch,
                pixel_groups,
            ),
            hair_scene_fabric_dispatch(
                &self.pipelines.shading,
                &self.bind_group,
                self.dispatch,
                pixel_groups,
            ),
        ]);
        debug_assert_eq!(visibility_dispatches.len(), HAIR_VISIBILITY_DISPATCH_COUNT);
        debug_assert_eq!(
            deep_opacity_dispatches.len(),
            HAIR_DEEP_OPACITY_DISPATCH_COUNT
        );
        let resources = vec![
            SceneFabricResource::Buffer {
                key: HAIR_STRAND_RESOURCE,
                frame: self.frame,
                version: self.version,
                buffer: generation.segment.buffer(),
                offset: 0,
            },
            SceneFabricResource::Buffer {
                key: HAIR_VISIBILITY_RESOURCE,
                frame: self.frame,
                version: self.version,
                buffer: generation.coverage.buffer(),
                offset: 0,
            },
            SceneFabricResource::Buffer {
                key: HAIR_DEEP_OPACITY_RESOURCE,
                frame: self.frame,
                version: self.version,
                buffer: generation.deep_opacity.buffer(),
                offset: 0,
            },
        ];
        let recorders = vec![
            SceneFabricPass::Compute {
                pass: HAIR_VISIBILITY_PASS,
                dispatches: visibility_dispatches,
            },
            SceneFabricPass::Compute {
                pass: HAIR_DEEP_OPACITY_PASS,
                dispatches: deep_opacity_dispatches,
            },
        ];
        batch.append_parts(
            self.frame,
            self.extent,
            &[],
            &[HAIR_VISIBILITY_PASS, HAIR_DEEP_OPACITY_PASS],
            &[],
            resources,
            recorders,
        )?;
        Ok(())
    }

    /// Frame identity carried by the physical Hair publication.
    #[must_use]
    pub(crate) const fn frame(&self) -> u64 {
        self.frame
    }

    /// Content version shared by all three imported Hair resources.
    #[must_use]
    pub(crate) const fn version(&self) -> u32 {
        self.version
    }

    /// The exact catalog passes this adapter contributes.
    #[must_use]
    pub(crate) const fn active_passes(&self) -> [PassKey; 2] {
        [HAIR_VISIBILITY_PASS, HAIR_DEEP_OPACITY_PASS]
    }

    fn into_generation(mut self) -> HairRasterGeneration {
        self.generation
            .take()
            .expect("Hair batch adapter generation must be retained")
    }
}

fn require_segment_work(segment_workgroups: u32) -> Result<(), HairRasterBatchError> {
    if segment_workgroups == 0 {
        Err(HairRasterBatchError::NoRenderableSegments)
    } else {
        Ok(())
    }
}

/// Host-submitted Hair resources retained until the shared queue completion.
pub(crate) struct HairRasterBatchSubmission {
    completion: GpuCompletionValue,
    adapter: Option<HairRasterBatchAdapter>,
}

impl fmt::Debug for HairRasterBatchSubmission {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("HairRasterBatchSubmission")
            .field("completion", &self.completion)
            .field("adapter", &self.adapter)
            .finish()
    }
}

impl HairRasterBatchSubmission {
    /// Queue timeline value supplied by the unified host after submission.
    #[must_use]
    pub(crate) const fn completion(&self) -> GpuCompletionValue {
        self.completion
    }
}

impl Drop for HairRasterBatchSubmission {
    fn drop(&mut self) {
        // A timeline value cannot be waited on from this module. If the host
        // drops an uncompleted submission, leaking the adapter is the only
        // safe outcome; releasing its SceneRhi buffers could race the queue.
        if let Some(adapter) = self.adapter.take() {
            std::mem::forget(adapter);
        }
    }
}

fn hair_scene_fabric_dispatch<'a>(
    pipeline: &'a ComputePipeline,
    bind_group: &'a BindGroup,
    dispatch: HairRasterDispatchParams,
    workgroups: [u32; 3],
) -> SceneFabricDispatch<'a> {
    SceneFabricDispatch {
        pipeline,
        bind_group,
        bind_group_index: 0,
        workgroups,
        push_constants: Some(SceneFabricPushConstants {
            offset: 0,
            data: dispatch.encode_le().to_vec(),
        }),
    }
}

/// Admission and retirement state for one Hair raster view stream.
pub(crate) struct HairRasterGpuState {
    free: Vec<HairRasterGeneration>,
    allocated_generations: usize,
    in_flight: usize,
    next_version: u32,
    max_generations: usize,
}

impl Default for HairRasterGpuState {
    fn default() -> Self {
        Self::with_max_generations(3)
    }
}

impl HairRasterGpuState {
    /// Creates a bounded generation pool. A generation is returned to the free
    /// pool only after its submission completion has been confirmed.
    #[must_use]
    pub(crate) const fn with_max_generations(max_generations: usize) -> Self {
        Self {
            free: Vec::new(),
            allocated_generations: 0,
            in_flight: 0,
            next_version: 1,
            max_generations,
        }
    }

    /// PrepareResources admission. This validates the actual CPU projection,
    /// reuses a retired generation when its capacity covers the view, or
    /// allocates a new SceneRhi generation within the configured frame budget.
    pub(crate) fn prepare_resources(
        &mut self,
        rhi: &SceneRhi,
        frame: u64,
        view: HairRasterViewFrame,
    ) -> Result<HairRasterPreparedFrame, HairRasterPrepareError> {
        let capacity = HairRasterCapacity::for_frame(&view)?;
        if !rhi
            .device()
            .wgpu_device()
            .features()
            .contains(bevy_render::render_resource::WgpuFeatures::IMMEDIATES)
            || rhi.device().wgpu_device().limits().max_immediate_size < 20
        {
            return Err(HairRasterPrepareError::ImmediateAbiUnsupported);
        }
        let version = self.next_version;
        let next_version = version
            .checked_add(1)
            .ok_or(HairRasterPrepareError::VersionExhausted)?;
        let generation = self
            .free
            .iter()
            .position(|candidate| candidate.capacity.covers(capacity))
            .map(|index| self.free.swap_remove(index));
        let generation = match generation {
            Some(mut generation) => {
                generation.version = version;
                generation
            }
            None => {
                if self.max_generations == 0 || self.allocated_generations >= self.max_generations {
                    return Err(HairRasterPrepareError::GenerationBudgetExhausted);
                }
                self.allocated_generations += 1;
                HairRasterGeneration::allocate(rhi, capacity, version)
            }
        };
        self.next_version = next_version;
        let dispatch = view.dispatch.ok_or(HairRasterPrepareError::Admission(
            "missing dispatch parameters",
        ))?;
        Ok(HairRasterPreparedFrame {
            frame,
            view,
            generation: Some(generation),
            dispatch,
        })
    }

    /// Prepares a real Hair generation for the shared SceneFabric host.
    ///
    /// Unlike [`Self::submit`], this method does not submit a private command
    /// buffer. It performs the real queue upload/clear work and returns an
    /// adapter whose resources and recorders can be appended to the host's
    /// single Fabric batch.
    pub(crate) fn prepare_batch_adapter(
        &mut self,
        rhi: &SceneRhi,
        frame: u64,
        view: HairRasterViewFrame,
        pipelines: &HairRasterPipelineSet,
    ) -> Result<HairRasterBatchAdapter, HairRasterPrepareError> {
        let prepared = self.prepare_resources(rhi, frame, view)?;
        HairRasterBatchAdapter::from_prepared(rhi, prepared, pipelines)
    }

    /// Cancels a batch before host submission and returns its real generation
    /// to the retired/free pool. This must be used when host admission or
    /// recording fails after [`Self::prepare_batch_adapter`].
    pub(crate) fn cancel_batch_adapter(&mut self, adapter: HairRasterBatchAdapter) {
        self.free.push(adapter.into_generation());
    }

    /// Transfers ownership of a recorded Hair adapter to a host-completion
    /// ticket. The host must call [`Self::complete_batch_adapter`] only after
    /// its completion tracker has observed the returned timeline value.
    pub(crate) fn submit_batch_adapter(
        &mut self,
        adapter: HairRasterBatchAdapter,
        completion: GpuCompletionValue,
    ) -> Result<HairRasterBatchSubmission, HairRasterBatchError> {
        if completion.0 == 0 {
            self.free.push(adapter.into_generation());
            return Err(HairRasterBatchError::InvalidCompletionToken);
        }
        self.in_flight += 1;
        Ok(HairRasterBatchSubmission {
            completion,
            adapter: Some(adapter),
        })
    }

    /// Reclaims a host-submitted generation only after its actual completion
    /// value is visible. A pending submission is returned unchanged for a
    /// later poll; no GPU resource is released on that path.
    pub(crate) fn complete_batch_adapter(
        &mut self,
        mut submission: HairRasterBatchSubmission,
        completed: GpuCompletionValue,
    ) -> Result<(), HairRasterBatchSubmission> {
        if completed < submission.completion {
            return Err(submission);
        }
        let adapter = submission
            .adapter
            .take()
            .expect("Hair batch submission adapter must be retained");
        self.free.push(adapter.into_generation());
        self.in_flight = self.in_flight.saturating_sub(1);
        Ok(())
    }

    /// Records queue uploads and the ordered depth/resolve/publish dispatches,
    /// then returns a completion-owned generation. The caller must pass the
    /// submission back to [`Self::complete`] before the generation can be
    /// reused.
    pub(crate) fn submit(
        &mut self,
        rhi: &SceneRhi,
        mut prepared: HairRasterPreparedFrame,
        pipelines: &HairRasterPipelineSet,
    ) -> Result<HairRasterSubmission, HairRasterPrepareError> {
        let generation = prepared
            .generation
            .take()
            .ok_or(HairRasterPrepareError::Admission(
                "prepared frame already consumed",
            ))?;
        let (pixel_count, geometry_words) = recording_shape(&prepared.view, prepared.dispatch)?;
        upload_generation(rhi, &generation, &prepared.view, pixel_count);
        let bind_group = create_bind_group(rhi, &generation, pipelines);
        let mut encoder = rhi.device().wgpu_device().create_command_encoder(
            &bevy_render::render_resource::CommandEncoderDescriptor {
                label: Some("prism hair raster production chain recorder"),
            },
        );
        for (label, pipeline) in [
            ("hair_raster_geometry", &pipelines.geometry),
            ("hair_raster_depth", &pipelines.depth),
            ("hair_raster_resolve", &pipelines.resolve),
            ("hair_raster_publish", &pipelines.publish),
        ] {
            if prepared.view.workgroups == 0 {
                continue;
            }
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some(label),
                timestamp_writes: None,
            });
            pass.set_pipeline(pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.set_immediates(0, &prepared.dispatch.encode_le());
            pass.dispatch_workgroups(prepared.view.workgroups, 1, 1);
        }
        for (label, pipeline) in [
            ("hair_raster_deep_opacity", &pipelines.deep_opacity),
            ("hair_raster_shading", &pipelines.shading),
        ] {
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some(label),
                timestamp_writes: None,
            });
            pass.set_pipeline(pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.set_immediates(0, &prepared.dispatch.encode_le());
            pass.dispatch_workgroups(prepared.view.pixel_workgroups, 1, 1);
        }
        let completion: CompletionToken =
            Box::new(rhi.queue().wgpu_queue().submit([encoder.finish()]));
        self.in_flight += 1;
        Ok(HairRasterSubmission {
            frame: prepared.frame,
            version: generation.version,
            pixel_count,
            geometry_words,
            generation: Some(generation),
            completion: Some(completion),
            queue: rhi.queue().clone(),
        })
    }

    /// Confirms GPU completion, performs the real RHI readback, and only then
    /// returns the generation to the reusable pool.
    pub(crate) fn complete(
        &mut self,
        rhi: &SceneRhi,
        submission: HairRasterSubmission,
    ) -> Result<HairRasterReadback, HairRasterCompleteError> {
        let completed = submission
            .wait_for_completion(rhi)
            .map_err(HairRasterCompleteError::Pending)?;
        self.in_flight = self.in_flight.saturating_sub(1);
        let HairRasterCompleted {
            generation,
            readback,
        } = completed;
        // Completion has already been confirmed, so the generation is safe to
        // reuse even if the diagnostic readback itself fails.
        self.free.push(generation);
        readback.map_err(|error| HairRasterCompleteError::Readback { error })
    }

    #[must_use]
    pub(crate) const fn in_flight(&self) -> usize {
        self.in_flight
    }
}

fn recording_shape(
    view: &HairRasterViewFrame,
    dispatch: HairRasterDispatchParams,
) -> Result<(usize, usize), HairRasterPrepareError> {
    let pixel_count = usize::try_from(view.viewport[0])
        .ok()
        .and_then(|width| {
            usize::try_from(view.viewport[1])
                .ok()
                .and_then(|height| width.checked_mul(height))
        })
        .ok_or(HairRasterPrepareError::Admission(
            "pixel count exceeds host usize",
        ))?;
    let geometry_words = usize::try_from(dispatch_segment_count(dispatch))
        .ok()
        .and_then(|count| count.checked_mul(HAIR_GEOMETRY_STRIDE as usize / 4))
        .ok_or(HairRasterPrepareError::Admission(
            "geometry word count exceeds host usize",
        ))?
        .max(HAIR_GEOMETRY_STRIDE as usize / 4);
    Ok((pixel_count, geometry_words))
}

fn upload_generation(
    rhi: &SceneRhi,
    generation: &HairRasterGeneration,
    view: &HairRasterViewFrame,
    pixel_count: usize,
) {
    let segment_bytes = if view.upload_bytes.is_empty() {
        [0_u8; HAIR_SEGMENT_STRIDE as usize].as_slice()
    } else {
        view.upload_bytes.as_slice()
    };
    let empty_depth = vec![u32::MAX; pixel_count];
    let empty_ids = vec![u32::MAX; pixel_count];
    let empty_coverage = vec![0_u32; pixel_count];
    rhi.queue()
        .write_buffer(generation.segment.id(), 0, segment_bytes);
    rhi.queue()
        .write_buffer(generation.depth.id(), 0, bytemuck::cast_slice(&empty_depth));
    rhi.queue()
        .write_buffer(generation.ids.id(), 0, bytemuck::cast_slice(&empty_ids));
    rhi.queue().write_buffer(
        generation.coverage.id(),
        0,
        bytemuck::cast_slice(&empty_coverage),
    );
}

fn create_bind_group(
    rhi: &SceneRhi,
    generation: &HairRasterGeneration,
    pipelines: &HairRasterPipelineSet,
) -> BindGroup {
    let bind_entries = [
        buffer_entry(0, &generation.segment),
        buffer_entry(1, &generation.geometry),
        buffer_entry(2, &generation.depth),
        buffer_entry(3, &generation.ids),
        buffer_entry(4, &generation.coverage),
        buffer_entry(5, &generation.deep_opacity),
        buffer_entry(6, &generation.shading),
    ];
    BindGroup::from(
        rhi.device()
            .wgpu_device()
            .create_bind_group(&BindGroupDescriptor {
                label: Some("prism hair raster group 0"),
                layout: &pipelines.layout,
                entries: &bind_entries,
            }),
    )
}

fn buffer_entry(binding: u32, buffer: &SceneRhiBuffer) -> BindGroupEntry<'_> {
    BindGroupEntry {
        binding,
        resource: BindingResource::Buffer(BufferBinding {
            buffer: buffer.buffer(),
            offset: 0,
            size: BufferSize::new(buffer.buffer().size()),
        }),
    }
}

/// Prepared, admitted CPU Hair frame waiting for recording.
pub(crate) struct HairRasterPreparedFrame {
    frame: u64,
    view: HairRasterViewFrame,
    generation: Option<HairRasterGeneration>,
    dispatch: HairRasterDispatchParams,
}

impl HairRasterPreparedFrame {
    #[must_use]
    pub(crate) const fn frame(&self) -> u64 {
        self.frame
    }

    #[must_use]
    pub(crate) const fn version(&self) -> u32 {
        self.generation.as_ref().expect("prepared frame").version
    }

    #[must_use]
    pub(crate) const fn capacity(&self) -> HairRasterCapacity {
        self.generation.as_ref().expect("prepared frame").capacity
    }

    #[must_use]
    pub(crate) fn segment_buffer_id(&self) -> prism_render_driver::BufferId {
        self.generation
            .as_ref()
            .expect("prepared frame")
            .segment_id()
    }
}

/// A submitted generation whose SceneRhi leases remain live through completion.
pub(crate) struct HairRasterSubmission {
    frame: u64,
    version: u32,
    pixel_count: usize,
    geometry_words: usize,
    generation: Option<HairRasterGeneration>,
    completion: Option<CompletionToken>,
    queue: WgpuQueue,
}

impl fmt::Debug for HairRasterSubmission {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("HairRasterSubmission")
            .field("frame", &self.frame)
            .field("version", &self.version)
            .field("pixel_count", &self.pixel_count)
            .field("geometry_words", &self.geometry_words)
            .field(
                "generation",
                &self
                    .generation
                    .as_ref()
                    .map(|generation| generation.version),
            )
            .field("completion_pending", &self.completion.is_some())
            .finish()
    }
}

struct HairRasterCompleted {
    generation: HairRasterGeneration,
    readback: Result<HairRasterReadback, ReadbackError>,
}

impl HairRasterSubmission {
    /// Frame identity retained by the GPU submission.
    #[must_use]
    pub(crate) const fn frame(&self) -> u64 {
        self.frame
    }

    /// Content version retained by the GPU submission.
    #[must_use]
    pub(crate) const fn version(&self) -> u32 {
        self.version
    }

    fn wait_for_completion(mut self, rhi: &SceneRhi) -> Result<HairRasterCompleted, Self> {
        if !self
            .generation
            .as_ref()
            .is_some_and(|generation| Arc::ptr_eq(generation.segment.device(), rhi.device()))
        {
            return Err(self);
        }
        let completion = self.completion.take().expect("pending hair submission");
        if !FrameSync::wait_for(&self.queue, completion.as_ref()) {
            self.completion = Some(completion);
            return Err(self);
        }
        let generation = self.generation.take().expect("pending hair generation");
        let readback = read_generation(rhi, &generation, self.pixel_count, self.geometry_words);
        Ok(HairRasterCompleted {
            generation,
            readback,
        })
    }
}

impl Drop for HairRasterSubmission {
    fn drop(&mut self) {
        let Some(completion) = self.completion.take() else {
            return;
        };
        if FrameSync::wait_for(&self.queue, completion.as_ref()) {
            drop(self.generation.take());
        } else {
            if let Some(generation) = self.generation.take() {
                std::mem::forget(generation);
            }
            std::mem::forget(completion);
        }
    }
}

/// Completion failure that keeps an unconfirmed submission retryable.
#[derive(Debug, thiserror::Error)]
pub(crate) enum HairRasterCompleteError {
    #[error("hair raster GPU completion could not be confirmed")]
    Pending(HairRasterSubmission),
    #[error("hair raster readback failed: {error:?}")]
    Readback { error: ReadbackError },
}

fn read_generation(
    rhi: &SceneRhi,
    generation: &HairRasterGeneration,
    pixel_count: usize,
    geometry_words: usize,
) -> Result<HairRasterReadback, ReadbackError> {
    let pixel_bytes = u64::try_from(pixel_count)
        .ok()
        .and_then(|count| count.checked_mul(HAIR_PIXEL_RESULT_STRIDE))
        .ok_or(ReadbackError::InvalidRange)?;
    let words = |data: Vec<u8>| {
        data.chunks_exact(4)
            .map(|word| u32::from_le_bytes(word.try_into().expect("4-byte readback word")))
            .collect()
    };
    let float4 = |data: Vec<u8>| {
        data.chunks_exact(16)
            .map(|value| {
                [
                    f32::from_le_bytes(value[0..4].try_into().expect("float4.x")),
                    f32::from_le_bytes(value[4..8].try_into().expect("float4.y")),
                    f32::from_le_bytes(value[8..12].try_into().expect("float4.z")),
                    f32::from_le_bytes(value[12..16].try_into().expect("float4.w")),
                ]
            })
            .collect()
    };
    let visibility_bytes = pixel_bytes / 4;
    let geometry_bytes = u64::try_from(geometry_words)
        .ok()
        .and_then(|words| words.checked_mul(4))
        .ok_or(ReadbackError::InvalidRange)?;
    Ok(HairRasterReadback {
        geometry: words(
            rhi.queue()
                .read_buffer(generation.geometry.id(), geometry_bytes)?,
        ),
        depth: words(
            rhi.queue()
                .read_buffer(generation.depth.id(), visibility_bytes)?,
        ),
        ids: words(
            rhi.queue()
                .read_buffer(generation.ids.id(), visibility_bytes)?,
        ),
        coverage: words(
            rhi.queue()
                .read_buffer(generation.coverage.id(), visibility_bytes)?,
        ),
        deep_opacity: float4(
            rhi.queue()
                .read_buffer(generation.deep_opacity.id(), pixel_bytes)?,
        ),
        shading: float4(
            rhi.queue()
                .read_buffer(generation.shading.id(), pixel_bytes)?,
        ),
    })
}

#[cfg(test)]
mod tests {
    use prism_render_architecture::{
        deformation::DeformationHandle,
        hair::{
            dynamics::Vec3,
            groom_import::{ResampledGroom, StrandAttributes},
            HairGroup, HairGroupHandle, HairLodTier,
        },
        motion::MotionSource,
        render_fabric::DomainObjectHandle,
    };

    use super::*;
    use crate::fabric_domains::DomainFrameIdentity;

    fn fixture_owner() -> HairRenderWorldOwner {
        let mut owner = HairRenderWorldOwner::default();
        for (object, groom) in [
            (DomainObjectHandle::new(2, 4), 9),
            (DomainObjectHandle::new(7, 1), 3),
        ] {
            owner
                .register(
                    DomainFrameIdentity {
                        object,
                        history_key: object.index(),
                        motion_source: MotionSource::VertexAnimation,
                    },
                    HairGroup {
                        handle: HairGroupHandle(groom),
                        guide_strand_count: 1,
                        max_render_strands: 1,
                        segments_per_strand: 1,
                        deformation: DeformationHandle(0),
                        native_form: HairLodTier::Strands,
                    },
                    ResampledGroom {
                        points_per_strand: 2,
                        positions: vec![Vec3::new(-0.5, 0.0, 0.0), Vec3::new(0.5, 0.0, 0.0)],
                        attributes: vec![StrandAttributes::new(0.1, 0.05, [0.0; 2], groom)],
                    },
                )
                .unwrap();
        }
        owner
    }

    #[test]
    fn per_view_frame_projects_real_owner_geometry_with_stable_ids_and_compact_upload() {
        let owner = fixture_owner();
        let frame = HairRasterViewFrame::from_owner(
            &owner,
            &glam::Mat4::IDENTITY.to_cols_array_2d(),
            [64, 64],
            HairSoftRasterAbi::default(),
            1024,
        )
        .unwrap();

        assert_eq!(frame.grooms.len(), 2);
        assert_eq!(frame.grooms[0].object, DomainObjectHandle::new(2, 4));
        assert_eq!(frame.grooms[0].groom, HairGroupHandle(9));
        assert_eq!(frame.grooms[0].first_segment_id, 0);
        assert_eq!(frame.grooms[1].first_segment_id, 1);
        assert_eq!(
            frame
                .grooms
                .iter()
                .map(|range| range.visible_segment_count)
                .sum::<u32>(),
            2
        );
        assert_eq!(frame.upload_bytes.len(), 64);
        assert_eq!(frame.workgroups, 1);
        assert_eq!(
            frame.dispatch.unwrap().dispatch_workgroups(1024).unwrap(),
            1
        );
        assert_eq!(
            u32::from_le_bytes(frame.upload_bytes[28..32].try_into().unwrap()),
            0
        );
        assert_eq!(
            u32::from_le_bytes(frame.upload_bytes[60..64].try_into().unwrap()),
            1
        );
    }

    #[test]
    fn invalid_view_fails_without_publishing_partial_groom_ranges() {
        let owner = fixture_owner();
        let result = HairRasterViewFrame::from_owner(
            &owner,
            &glam::Mat4::IDENTITY.to_cols_array_2d(),
            [u32::MAX, 2],
            HairSoftRasterAbi::default(),
            1024,
        );
        assert!(matches!(result, Err(HairRasterPrepareError::Projection(_))));
    }

    #[test]
    fn scene_fabric_batch_contract_keeps_only_real_hair_passes_and_dispatches() {
        assert_eq!(
            [HAIR_VISIBILITY_PASS, HAIR_DEEP_OPACITY_PASS],
            [PassKey(0x5006), PassKey(0x5008)]
        );
        assert_eq!(HAIR_VISIBILITY_DISPATCH_COUNT, 4);
        assert_eq!(HAIR_DEEP_OPACITY_DISPATCH_COUNT, 2);
        assert_ne!(HAIR_VISIBILITY_PASS, HAIR_DEEP_OPACITY_PASS);
    }

    #[test]
    fn scene_fabric_batch_rejects_empty_segment_work_instead_of_recording_noop() {
        assert_eq!(
            require_segment_work(0),
            Err(HairRasterBatchError::NoRenderableSegments)
        );
        assert_eq!(require_segment_work(1), Ok(()));
    }

    #[test]
    fn fabric_capacity_reserves_canonical_minimum_without_changing_logical_counts() {
        let owner = fixture_owner();
        let frame = HairRasterViewFrame::from_owner(
            &owner,
            &glam::Mat4::IDENTITY.to_cols_array_2d(),
            [1, 1],
            HairSoftRasterAbi::default(),
            1024,
        )
        .unwrap();
        let capacity = HairRasterCapacity::for_frame(&frame).unwrap();

        assert_eq!(capacity.pixel_count, 1);
        assert_eq!(capacity.segment_bytes, HAIR_FABRIC_RESOURCE_BYTES);
        assert_eq!(capacity.visibility_bytes, HAIR_FABRIC_RESOURCE_BYTES);
        assert_eq!(capacity.deep_opacity_bytes, HAIR_FABRIC_RESOURCE_BYTES);
        assert!(capacity.covers(capacity));
    }

    #[test]
    fn empty_owner_is_rejected_before_a_fabric_generation_is_allocated() {
        let owner = HairRenderWorldOwner::default();
        let frame = HairRasterViewFrame::from_owner(
            &owner,
            &glam::Mat4::IDENTITY.to_cols_array_2d(),
            [8, 8],
            HairSoftRasterAbi::default(),
            1024,
        )
        .unwrap();
        assert_eq!(frame.workgroups, 0);
        assert_eq!(
            require_segment_work(frame.workgroups),
            Err(HairRasterBatchError::NoRenderableSegments)
        );
    }
}
