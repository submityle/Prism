//! Native opaque surface encoding with no lighting evaluation.
//!
//! This module owns the shader, explicit pipeline ABI, mesh validation, and
//! bind-group descriptor builder. The caller owns frame/material buffers and
//! binds group 0 with dynamic offsets in `[frame, material]` order.
//!
//! The shader consumes world-space position, normal, and canonical UV0
//! attributes at locations 0, 1, and 2 (`Float32x3`, `Float32x3`, and
//! `Float32x2`). UV0 is required even when the current fragment path does not
//! sample a texture; absent, duplicate, or differently formatted UV0 is a
//! diagnosed mesh rejection, never synthesized from world position. It
//! transforms position and normal with the declared `clip_from_world` matrix, and
//! writes ten logical opaque GBuffer attachments across two raster recipes:
//! world-position x/y/z (`R32Float` each), normal/coverage (`Rgba16Float`),
//! and albedo (`Rgba8UnormSrgb`) in the geometry recipe, followed by surface
//! identity (`R32Uint`) and canonical SurfaceData (`Rgba16Float`) in the
//! material recipe, including sampled channels and a packed integer
//! octahedral clearcoat normal. Normal alpha is binary
//! coverage (one for every opaque fragment, independent of material opacity;
//! zero must be supplied by the background clear). Position retains f32 storage
//! precision, normal retains binary16 precision, and identity retains all u32
//! bits. Albedo RGB is clamped/quantized to sRGB8, not preserved as binary16;
//! its alpha is linear UNORM8, not coverage. It does not use a
//! storage buffer as a color target or a full-screen substitute.
//!
//! Only sample count 1 is supported by the current single-sample composite.
//! The fallible descriptor builders return `SurfaceRejection`; the production
//! trait entry points panic with the same error text for an invalid variant,
//! rather than emitting an incompatible descriptor or changing its samples.

use alloc::vec;
use alloc::vec::Vec;

use bytemuck::{Pod, Zeroable};
use prism_render_driver::{
    BindGroupDescriptor, BindGroupEntry, BindGroupLayoutDescriptor, BindGroupLayoutEntry,
    BindingResource, BindingType, BufferBindingType, BufferId, ColorTargetState, DepthStencilState,
    FragmentState, MultisampleState, PipelineLayoutDescriptor, PipelineLayoutId, PrimitiveState,
    PrimitiveTopology, RenderDevice, RenderPipelineDescriptor, SamplerBindingType, SamplerId,
    ShaderModuleDescriptor, ShaderModuleId, ShaderStages, TextureFormat, TextureSampleType,
    TextureViewDimension, TextureViewId, VertexBufferLayout, VertexFormat, VertexState,
};

use crate::mesh_pipeline::{MeshPipelineKey, MeshPipelineProvider};

/// The fixed bind-group index used by the surface-encode shader.
pub const SURFACE_GROUP: u32 = 0;
/// The frame/view uniform binding.
pub const SURFACE_FRAME_BINDING: u32 = 0;
/// The per-draw material identity uniform binding.
pub const SURFACE_MATERIAL_BINDING: u32 = 1;
/// The canonical material header storage binding.
pub const SURFACE_HEADERS_BINDING: u32 = 2;
/// The canonical packed parameter heap storage binding.
pub const SURFACE_PARAMETERS_BINDING: u32 = 3;
/// Canonical material texture-row storage binding.
pub const SURFACE_TEXTURES_BINDING: u32 = 4;
/// VT page-table binding in the portable surface texture group.
pub const SURFACE_VT_PAGE_TABLE_BINDING: u32 = 5;
/// VT config binding in the portable surface texture group.
pub const SURFACE_VT_CONFIG_BINDING: u32 = 6;
/// VT atlas binding in the portable surface texture group.
pub const SURFACE_VT_ATLAS_BINDING: u32 = 7;
/// VT filtering sampler binding in the portable surface texture group.
pub const SURFACE_VT_SAMPLER_BINDING: u32 = 8;
/// Canonical mesh UV0 vertex location and format.
pub const SURFACE_UV0_LOCATION: u32 = 2;
/// The first raster recipe writes geometry and material color data. It stays
/// below the WebGPU 32-byte color-attachment budget.
pub const SURFACE_GEOMETRY_TARGET_FORMATS: [TextureFormat; 5] = [
    crate::target::POSITION_X_GBUFFER_FORMAT,
    crate::target::POSITION_Y_GBUFFER_FORMAT,
    crate::target::POSITION_Z_GBUFFER_FORMAT,
    crate::target::GBUFFER_FORMAT,
    crate::target::ALBEDO_GBUFFER_FORMAT,
];
/// The second raster recipe writes identity, surface coordinates, and sampled
/// non-base material channels for the canonical composite.
pub const SURFACE_MATERIAL_TARGET_FORMATS: [TextureFormat; 5] = [
    crate::target::SURFACE_ID_GBUFFER_FORMAT,
    crate::target::SURFACE_DATA_GBUFFER_FORMAT,
    crate::target::SURFACE_MATERIAL_A_GBUFFER_FORMAT,
    crate::target::SURFACE_MATERIAL_B_GBUFFER_FORMAT,
    crate::target::SURFACE_MATERIAL_C_GBUFFER_FORMAT,
];
/// The complete logical GBuffer target count across both raster recipes.
pub const SURFACE_GBUFFER_TARGET_COUNT: usize =
    SURFACE_GEOMETRY_TARGET_FORMATS.len() + SURFACE_MATERIAL_TARGET_FORMATS.len();

/// The exact resource ABI consumed by the surface-encode shader.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SurfaceBindingContract {
    /// The bind-group index.
    pub group: u32,
    /// The frame/view uniform slot.
    pub frame: u32,
    /// The material uniform slot.
    pub material: u32,
    /// The canonical header table slot.
    pub headers: u32,
    /// The canonical packed parameter heap slot.
    pub parameters: u32,
    /// Canonical material texture-row table.
    pub textures: u32,
}

impl SurfaceBindingContract {
    /// The provider's immutable binding declaration.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            group: SURFACE_GROUP,
            frame: SURFACE_FRAME_BINDING,
            material: SURFACE_MATERIAL_BINDING,
            headers: SURFACE_HEADERS_BINDING,
            parameters: SURFACE_PARAMETERS_BINDING,
            textures: SURFACE_TEXTURES_BINDING,
        }
    }

    /// Returns the RHI bind-group layout matching the WGSL declarations.
    #[must_use]
    pub fn layout_descriptor(self) -> BindGroupLayoutDescriptor {
        let vertex_fragment = ShaderStages::VERTEX | ShaderStages::FRAGMENT;
        let fragment = ShaderStages::FRAGMENT;
        BindGroupLayoutDescriptor {
            label: Some("prism-surface-encode-group-0".into()),
            entries: vec![
                BindGroupLayoutEntry {
                    binding: self.frame,
                    visibility: vertex_fragment,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Uniform,
                        has_dynamic_offset: true,
                        min_binding_size: Some(SurfaceFrame::SIZE_BYTES),
                    },
                    count: None,
                },
                BindGroupLayoutEntry {
                    binding: self.material,
                    visibility: fragment,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Uniform,
                        has_dynamic_offset: true,
                        min_binding_size: Some(SurfaceMaterial::SIZE_BYTES),
                    },
                    count: None,
                },
                BindGroupLayoutEntry {
                    binding: self.headers,
                    visibility: fragment,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: Some(core::mem::size_of::<
                            prism_render_material::GpuMaterialHeader,
                        >() as u64),
                    },
                    count: None,
                },
                BindGroupLayoutEntry {
                    binding: self.parameters,
                    visibility: fragment,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: Some(4),
                    },
                    count: None,
                },
                BindGroupLayoutEntry {
                    binding: self.textures,
                    visibility: fragment,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: Some(core::mem::size_of::<
                            prism_render_material::GpuMaterialTexture,
                        >() as u64),
                    },
                    count: None,
                },
            ],
        }
    }
}

/// Portable Material-VT group shared by Native opaque surface sampling.
#[must_use]
pub fn surface_vt_layout_descriptor() -> BindGroupLayoutDescriptor {
    BindGroupLayoutDescriptor {
        label: Some("prism-surface-material-vt-group-1".into()),
        entries: vec![
            BindGroupLayoutEntry {
                binding: SURFACE_VT_PAGE_TABLE_BINDING,
                visibility: ShaderStages::FRAGMENT,
                ty: BindingType::Buffer {
                    ty: BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: Some(16),
                },
                count: None,
            },
            BindGroupLayoutEntry {
                binding: SURFACE_VT_CONFIG_BINDING,
                visibility: ShaderStages::FRAGMENT,
                ty: BindingType::Buffer {
                    ty: BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: Some(32),
                },
                count: None,
            },
            BindGroupLayoutEntry {
                binding: SURFACE_VT_ATLAS_BINDING,
                visibility: ShaderStages::FRAGMENT,
                ty: BindingType::Texture {
                    sample_type: TextureSampleType::Float,
                    view_dimension: TextureViewDimension::D2Array,
                    multisampled: false,
                },
                count: None,
            },
            BindGroupLayoutEntry {
                binding: SURFACE_VT_SAMPLER_BINDING,
                visibility: ShaderStages::FRAGMENT,
                ty: BindingType::Sampler(SamplerBindingType::Filtering),
                count: None,
            },
        ],
    }
}

impl Default for SurfaceBindingContract {
    fn default() -> Self {
        Self::new()
    }
}

/// Per-view values written by the caller for one surface-encode frame.
///
/// Matrices use WGSL's column-major `mat4x4<f32>` memory layout. The vertex
/// position and normal contract is world-space for the non-transformed entry
/// point. The transformed entry point applies the row-major affine instance
/// stream at locations 13, 14, and 15 before evaluating the same contract.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Pod, Zeroable)]
pub struct SurfaceFrame {
    /// Column-major clip-from-world matrix.
    pub clip_from_world: [[f32; 4]; 4],
    /// Previous column-major clip-from-world matrix, reserved for motion consumers.
    pub previous_clip_from_world: [[f32; 4]; 4],
    /// Bounded scene-linear photographic exposure compensation in stops.
    pub exposure_stops: f32,
    /// Whether the previous camera matrix is valid for temporal reprojection.
    pub camera_history_valid: u32,
    /// Reserved for ABI alignment and future per-view controls.
    pub _padding: [f32; 2],
}

impl SurfaceFrame {
    /// The uniform's ABI size in bytes.
    pub const SIZE_BYTES: u64 = core::mem::size_of::<Self>() as u64;

    /// Packs the validated current/previous camera matrices for the GPU ABI.
    #[must_use]
    pub const fn from_view(view: &crate::surface_views::SurfaceView) -> Self {
        Self {
            clip_from_world: view.clip_from_world,
            previous_clip_from_world: view.previous_clip_from_world,
            exposure_stops: view.exposure_stops,
            camera_history_valid: view.history_valid as u32,
            _padding: [0.0; 2],
        }
    }
}

/// Per-draw identity of a canonical material row, with no duplicated parameters.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Pod, Zeroable)]
pub struct SurfaceMaterial {
    /// Canonical material header row and stable GBuffer identity.
    pub index: u32,
    /// Expected generational handle, checked again by the GPU.
    pub generation: u32,
    /// Expected material revision, checked again by the GPU.
    pub revision: u32,
    /// Expected canonical execution plan identity.
    pub execution_plan_index: u32,
}

impl SurfaceMaterial {
    /// The uniform's ABI size in bytes.
    pub const SIZE_BYTES: u64 = core::mem::size_of::<Self>() as u64;
}

/// The resources that the caller must place in group 0 for one draw.
///
/// Every field is optional so missing capabilities are reported by a stable
/// rejection instead of becoming an implicit zero buffer or a shader-side
/// fallback. Both buffers may contain multiple aligned uniform records; draw
/// recording supplies dynamic offsets in `[frame, material]` binding order.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct SurfaceResourceSet {
    /// Frame/view uniform buffer.
    pub frame: Option<BufferId>,
    /// Material uniform buffer.
    pub material: Option<BufferId>,
    /// Canonical material header storage buffer.
    pub headers: Option<BufferId>,
    /// Canonical packed parameter heap storage buffer.
    pub parameters: Option<BufferId>,
    /// Canonical material texture-row storage buffer.
    pub textures: Option<BufferId>,
}

/// A missing or invalid mesh capability required by surface-encode shading.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SurfaceRejection {
    /// No position attribute exists at shader location 0.
    MissingPosition,
    /// More than one position attribute claims shader location 0.
    DuplicatePosition,
    /// The position attribute is not a tightly represented float3.
    UnsupportedPositionFormat(VertexFormat),
    /// No normal attribute exists at shader location 1.
    MissingNormal,
    /// More than one normal attribute claims shader location 1.
    DuplicateNormal,
    /// The normal attribute is not a tightly represented float3.
    UnsupportedNormalFormat(VertexFormat),
    /// No canonical UV0 attribute exists at shader location 2.
    MissingUv0,
    /// More than one attribute claims canonical UV0 shader location 2.
    DuplicateUv0,
    /// Canonical UV0 is not represented as Float32x2.
    UnsupportedUv0Format(VertexFormat),
    /// The mesh key did not select the normal-plane format used as the opaque
    /// GBuffer profile marker; the provider declares all six actual targets.
    RequiresRgba16FloatTarget,
    /// Strip topology cannot be selected without an index-format field in the
    /// existing mesh pipeline key.
    UnsupportedStripTopology(PrimitiveTopology),
    /// The current GBuffer/composite contract requires exactly one sample.
    InvalidSampleCount,
}

impl core::fmt::Display for SurfaceRejection {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::MissingPosition => {
                f.write_str("surface-encode mesh is missing position at location 0")
            }
            Self::DuplicatePosition => {
                f.write_str("surface-encode mesh has duplicate position attributes at location 0")
            }
            Self::UnsupportedPositionFormat(format) => {
                write!(
                    f,
                    "surface-encode position format {format:?} is not Float32x3"
                )
            }
            Self::MissingNormal => {
                f.write_str("surface-encode mesh is missing normal at location 1")
            }
            Self::DuplicateNormal => {
                f.write_str("surface-encode mesh has duplicate normal attributes at location 1")
            }
            Self::UnsupportedNormalFormat(format) => {
                write!(
                    f,
                    "surface-encode normal format {format:?} is not Float32x3"
                )
            }
            Self::MissingUv0 => {
                f.write_str("surface-encode mesh is missing canonical UV0 at location 2")
            }
            Self::DuplicateUv0 => f.write_str(
                "surface-encode mesh has duplicate canonical UV0 attributes at location 2",
            ),
            Self::UnsupportedUv0Format(format) => {
                write!(f, "surface-encode UV0 format {format:?} is not Float32x2")
            }
            Self::RequiresRgba16FloatTarget => f.write_str(
                "surface-encode requires the current six-attachment opaque GBuffer profile",
            ),
            Self::UnsupportedStripTopology(topology) => {
                write!(
                    f,
                    "surface-encode cannot select strip topology {topology:?} without index format"
                )
            }
            Self::InvalidSampleCount => f.write_str(
                "surface-encode requires sample_count == 1 for the single-sample GBuffer composite",
            ),
        }
    }
}

/// A missing resource required by the explicit group-0 ABI.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SurfaceBindingRejection {
    /// The frame/view uniform was not supplied.
    MissingFrame,
    /// The material uniform was not supplied.
    MissingMaterial,
    /// The canonical material header table was not supplied.
    MissingHeaders,
    /// The canonical packed parameter heap was not supplied.
    MissingParameters,
    /// The canonical material texture-row table was not supplied.
    MissingTextures,
}

impl core::fmt::Display for SurfaceBindingRejection {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::MissingFrame => f.write_str("surface-encode frame/view buffer is missing"),
            Self::MissingMaterial => f.write_str("surface-encode material buffer is missing"),
            Self::MissingHeaders => {
                f.write_str("surface-encode canonical header buffer is missing")
            }
            Self::MissingParameters => {
                f.write_str("surface-encode canonical parameter heap is missing")
            }
            Self::MissingTextures => {
                f.write_str("surface-encode canonical texture-row buffer is missing")
            }
        }
    }
}

/// Encodes opaque mesh surfaces into the split GBuffer described above.
pub struct SurfaceEncodeProvider {
    module: ShaderModuleId,
    pipeline_layout: PipelineLayoutId,
    bind_group_layout: prism_render_driver::BindGroupLayoutId,
    vt_bind_group_layout: prism_render_driver::BindGroupLayoutId,
    contract: SurfaceBindingContract,
}

impl SurfaceEncodeProvider {
    /// Creates the shader module and explicit surface/portable-VT pipeline ABI.
    #[must_use]
    pub fn new(device: &dyn RenderDevice) -> Self {
        let contract = SurfaceBindingContract::new();
        let bind_group_layout = device.create_bind_group_layout(&contract.layout_descriptor());
        let vt_bind_group_layout = device.create_bind_group_layout(&surface_vt_layout_descriptor());
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("prism-surface-encode-pipeline-layout".into()),
            bind_group_layouts: vec![bind_group_layout, vt_bind_group_layout],
            push_constant_ranges: Vec::new(),
        });
        let module =
            device.create_shader_module(&ShaderModuleDescriptor::wgsl(SURFACE_ENCODE_WGSL));
        Self {
            module,
            pipeline_layout,
            bind_group_layout,
            vt_bind_group_layout,
            contract,
        }
    }

    /// The provider's exact resource contract.
    #[must_use]
    pub const fn binding_contract(&self) -> SurfaceBindingContract {
        self.contract
    }

    /// The explicit group-0 layout id used by this provider.
    #[must_use]
    pub const fn bind_group_layout(&self) -> prism_render_driver::BindGroupLayoutId {
        self.bind_group_layout
    }

    /// The explicit group-1 layout for portable Material-VT resources.
    #[must_use]
    pub const fn vt_bind_group_layout(&self) -> prism_render_driver::BindGroupLayoutId {
        self.vt_bind_group_layout
    }

    /// The explicit pipeline layout id used by this provider.
    #[must_use]
    pub const fn pipeline_layout(&self) -> PipelineLayoutId {
        self.pipeline_layout
    }

    /// Validates the canonical surface vertex ABI: position 0, normal 1, UV0 2.
    pub fn validate_mesh_layout(layout: &[VertexBufferLayout]) -> Result<(), SurfaceRejection> {
        validate_attribute(layout, 0, true)?;
        validate_attribute(layout, 1, false)?;
        validate_uv0(layout)
    }

    /// Validates the target portion of a mesh variant, including single sampling.
    pub fn validate_variant(variant: &MeshPipelineKey) -> Result<(), SurfaceRejection> {
        if variant.color_format != Some(crate::target::GBUFFER_FORMAT) {
            return Err(SurfaceRejection::RequiresRgba16FloatTarget);
        }
        if variant.topology.is_strip() {
            return Err(SurfaceRejection::UnsupportedStripTopology(variant.topology));
        }
        if variant.sample_count != 1 {
            return Err(SurfaceRejection::InvalidSampleCount);
        }
        Ok(())
    }

    /// Builds a descriptor after validating the mesh and target contracts.
    pub fn try_descriptor(
        &self,
        variant: &MeshPipelineKey,
        layout: &[VertexBufferLayout],
    ) -> Result<RenderPipelineDescriptor, SurfaceRejection> {
        Self::validate_variant(variant)?;
        Self::validate_mesh_layout(layout)?;
        Ok(self.make_descriptor(variant, false))
    }

    /// Builds the transformed descriptor after validating the mesh and target
    /// contracts. Locations 13..15 carry current affine rows; 16..18 carry
    /// previous rows and 19 carries object-history validity.
    pub fn try_transformed_descriptor(
        &self,
        variant: &MeshPipelineKey,
        layout: &[VertexBufferLayout],
    ) -> Result<RenderPipelineDescriptor, SurfaceRejection> {
        Self::validate_variant(variant)?;
        Self::validate_mesh_layout(layout)?;
        Ok(self.make_descriptor(variant, true))
    }

    /// Builds group 0 using this provider's actual layout and logical ranges.
    ///
    /// Bind with two dynamic offsets in `[frame, material]` order. Each offset
    /// must satisfy the device's uniform-buffer alignment and buffer bounds;
    /// the uniforms cover exactly 128 and 16 bytes. Storage bindings cover the
    /// canonical buffers, so their runtime-array bounds match the actual heaps.
    pub fn bind_group_descriptor(
        &self,
        resources: SurfaceResourceSet,
    ) -> Result<BindGroupDescriptor, SurfaceBindingRejection> {
        let frame = resources
            .frame
            .ok_or(SurfaceBindingRejection::MissingFrame)?;
        let material = resources
            .material
            .ok_or(SurfaceBindingRejection::MissingMaterial)?;
        let headers = resources
            .headers
            .ok_or(SurfaceBindingRejection::MissingHeaders)?;
        let parameters = resources
            .parameters
            .ok_or(SurfaceBindingRejection::MissingParameters)?;
        let textures = resources
            .textures
            .ok_or(SurfaceBindingRejection::MissingTextures)?;
        let c = self.contract;
        Ok(BindGroupDescriptor {
            label: Some("prism-surface-encode-group-0".into()),
            layout: Some(self.bind_group_layout),
            entries: vec![
                buffer_entry(c.frame, frame, SurfaceFrame::SIZE_BYTES),
                buffer_entry(c.material, material, SurfaceMaterial::SIZE_BYTES),
                BindGroupEntry {
                    binding: c.headers,
                    resource: BindingResource::Buffer {
                        buffer: headers,
                        offset: 0,
                        size: None,
                    },
                },
                BindGroupEntry {
                    binding: c.parameters,
                    resource: BindingResource::Buffer {
                        buffer: parameters,
                        offset: 0,
                        size: None,
                    },
                },
                BindGroupEntry {
                    binding: c.textures,
                    resource: BindingResource::Buffer {
                        buffer: textures,
                        offset: 0,
                        size: None,
                    },
                },
            ],
        })
    }

    /// Builds group 1 from the currently published portable VT slice lease.
    pub fn vt_bind_group_descriptor(
        &self,
        page_table: BufferId,
        config: BufferId,
        atlas: TextureViewId,
        sampler: SamplerId,
    ) -> BindGroupDescriptor {
        BindGroupDescriptor {
            label: Some("prism-surface-material-vt-group-1".into()),
            layout: Some(self.vt_bind_group_layout),
            entries: vec![
                BindGroupEntry {
                    binding: SURFACE_VT_PAGE_TABLE_BINDING,
                    resource: BindingResource::Buffer {
                        buffer: page_table,
                        offset: 0,
                        size: None,
                    },
                },
                BindGroupEntry {
                    binding: SURFACE_VT_CONFIG_BINDING,
                    resource: BindingResource::Buffer {
                        buffer: config,
                        offset: 0,
                        size: None,
                    },
                },
                BindGroupEntry {
                    binding: SURFACE_VT_ATLAS_BINDING,
                    resource: BindingResource::TextureView(atlas),
                },
                BindGroupEntry {
                    binding: SURFACE_VT_SAMPLER_BINDING,
                    resource: BindingResource::Sampler(sampler),
                },
            ],
        }
    }

    fn make_descriptor(
        &self,
        variant: &MeshPipelineKey,
        transformed: bool,
    ) -> RenderPipelineDescriptor {
        // The production trait cannot return SurfaceRejection. Keep validation
        // here so neither trait entry point can bypass the fallible contract.
        Self::validate_variant(variant).unwrap_or_else(|error| panic!("{error}"));
        RenderPipelineDescriptor {
            label: Some("prism-surface-encode-opaque".into()),
            layout: Some(self.pipeline_layout),
            vertex: VertexState {
                module: self.module,
                entry_point: if transformed { "vs_transform" } else { "vs" }.into(),
                buffers: Vec::new(),
            },
            primitive: PrimitiveState {
                topology: variant.topology,
                ..PrimitiveState::default()
            },
            depth_stencil: variant.depth_format.map(DepthStencilState::reverse_z),
            multisample: MultisampleState {
                count: variant.sample_count,
                ..Default::default()
            },
            fragment: Some(FragmentState {
                module: self.module,
                entry_point: "fs_geometry".into(),
                targets: SURFACE_GEOMETRY_TARGET_FORMATS
                    .into_iter()
                    .map(|format| Some(ColorTargetState::opaque(format)))
                    .collect(),
            }),
        }
    }

    fn make_secondary_descriptor(
        &self,
        variant: &MeshPipelineKey,
        transformed: bool,
    ) -> RenderPipelineDescriptor {
        Self::validate_variant(variant).unwrap_or_else(|error| panic!("{error}"));
        RenderPipelineDescriptor {
            label: Some("prism-surface-encode-material".into()),
            layout: Some(self.pipeline_layout),
            vertex: VertexState {
                module: self.module,
                entry_point: if transformed { "vs_transform" } else { "vs" }.into(),
                buffers: Vec::new(),
            },
            primitive: PrimitiveState {
                topology: variant.topology,
                ..PrimitiveState::default()
            },
            depth_stencil: variant.depth_format.map(DepthStencilState::reverse_z),
            multisample: MultisampleState {
                count: variant.sample_count,
                ..Default::default()
            },
            fragment: Some(FragmentState {
                module: self.module,
                entry_point: "fs_material".into(),
                targets: SURFACE_MATERIAL_TARGET_FORMATS
                    .into_iter()
                    .map(|format| Some(ColorTargetState::opaque(format)))
                    .collect(),
            }),
        }
    }
}

fn buffer_entry(binding: u32, buffer: BufferId, size: u64) -> BindGroupEntry {
    BindGroupEntry {
        binding,
        resource: BindingResource::Buffer {
            buffer,
            offset: 0,
            size: Some(size),
        },
    }
}

fn validate_attribute(
    layout: &[VertexBufferLayout],
    location: u32,
    position: bool,
) -> Result<(), SurfaceRejection> {
    let matches: Vec<_> = layout
        .iter()
        .flat_map(|buffer| buffer.attributes.iter())
        .filter(|attribute| attribute.shader_location == location)
        .collect();
    if matches.is_empty() {
        return Err(if position {
            SurfaceRejection::MissingPosition
        } else {
            SurfaceRejection::MissingNormal
        });
    }
    if matches.len() > 1 {
        return Err(if position {
            SurfaceRejection::DuplicatePosition
        } else {
            SurfaceRejection::DuplicateNormal
        });
    }
    if matches[0].format != VertexFormat::Float32x3 {
        return Err(if position {
            SurfaceRejection::UnsupportedPositionFormat(matches[0].format)
        } else {
            SurfaceRejection::UnsupportedNormalFormat(matches[0].format)
        });
    }
    Ok(())
}

fn validate_uv0(layout: &[VertexBufferLayout]) -> Result<(), SurfaceRejection> {
    let matches: Vec<_> = layout
        .iter()
        .flat_map(|buffer| buffer.attributes.iter())
        .filter(|attribute| attribute.shader_location == SURFACE_UV0_LOCATION)
        .collect();
    match matches.as_slice() {
        [] => Err(SurfaceRejection::MissingUv0),
        [attribute] if attribute.format == VertexFormat::Float32x2 => Ok(()),
        [attribute] => Err(SurfaceRejection::UnsupportedUv0Format(attribute.format)),
        _ => Err(SurfaceRejection::DuplicateUv0),
    }
}

/// Returns a shader source descriptor that callers can register in diagnostics.
#[must_use]
pub fn surface_encode_shader_descriptor() -> ShaderModuleDescriptor {
    ShaderModuleDescriptor::wgsl(SURFACE_ENCODE_WGSL)
}

/// The complete explicit-binding surface-encode shader.
pub const SURFACE_ENCODE_WGSL: &str =
    include_str!(concat!(env!("OUT_DIR"), "/surface_encode.wgsl"));

impl MeshPipelineProvider for SurfaceEncodeProvider {
    fn descriptor(&self, variant: &MeshPipelineKey) -> RenderPipelineDescriptor {
        self.make_descriptor(variant, false)
    }

    fn transformed_descriptor(
        &self,
        variant: &MeshPipelineKey,
    ) -> Option<RenderPipelineDescriptor> {
        Some(self.make_descriptor(variant, true))
    }

    fn secondary_descriptor(&self, variant: &MeshPipelineKey) -> Option<RenderPipelineDescriptor> {
        Some(self.make_secondary_descriptor(variant, false))
    }

    fn transformed_secondary_descriptor(
        &self,
        variant: &MeshPipelineKey,
    ) -> Option<RenderPipelineDescriptor> {
        Some(self.make_secondary_descriptor(variant, true))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use prism_render_driver::{PrimitiveTopology, VertexAttribute, VertexStepMode};

    fn layout(position: bool, normal: bool, uv0: bool) -> Vec<VertexBufferLayout> {
        let mut attributes = Vec::new();
        if position {
            attributes.push(VertexAttribute {
                format: VertexFormat::Float32x3,
                offset: 0,
                shader_location: 0,
            });
        }
        if normal {
            attributes.push(VertexAttribute {
                format: VertexFormat::Float32x3,
                offset: 12,
                shader_location: 1,
            });
        }
        if uv0 {
            attributes.push(VertexAttribute {
                format: VertexFormat::Float32x2,
                offset: 24,
                shader_location: SURFACE_UV0_LOCATION,
            });
        }
        vec![VertexBufferLayout {
            array_stride: if uv0 { 32 } else { 24 },
            step_mode: VertexStepMode::Vertex,
            attributes,
        }]
    }

    fn variant() -> MeshPipelineKey {
        MeshPipelineKey {
            execution_family:
                prism_render_architecture::shader_package::ExecutionFamily::FusedBuiltin,
            response_module_id: prism_render_architecture::shader_package::ModuleId::SPINE,
            permutation_index: prism_render_architecture::shader_package::PermutationIndex::ZERO,
            fallback_permutation: None,
            color_format: Some(TextureFormat::Rgba16Float),
            depth_format: Some(TextureFormat::Depth32Float),
            sample_count: 1,
            topology: PrimitiveTopology::TriangleList,
        }
    }

    #[test]
    fn frame_and_material_abis_are_stable_and_aligned() {
        use core::mem::{offset_of, size_of};

        fn assert_pod<T: Pod + Zeroable>() {}
        assert_pod::<SurfaceFrame>();
        assert_pod::<SurfaceMaterial>();
        assert_eq!(SurfaceFrame::SIZE_BYTES, 144);
        assert_eq!(SurfaceMaterial::SIZE_BYTES, 16);
        assert_eq!(core::mem::align_of::<SurfaceFrame>(), 4);
        assert_eq!(core::mem::align_of::<SurfaceMaterial>(), 4);
        assert_eq!(offset_of!(SurfaceFrame, clip_from_world), 0);
        assert_eq!(offset_of!(SurfaceFrame, previous_clip_from_world), 64);
        assert_eq!(offset_of!(SurfaceFrame, exposure_stops), 128);
        assert_eq!(offset_of!(SurfaceFrame, camera_history_valid), 132);
        assert_eq!(size_of::<[f32; 4]>(), 16, "WGSL matrix column stride");
        assert_eq!(size_of::<[SurfaceFrame; 2]>(), 288);
        assert_eq!(offset_of!(SurfaceMaterial, index), 0);
        assert_eq!(offset_of!(SurfaceMaterial, generation), 4);
        assert_eq!(offset_of!(SurfaceMaterial, revision), 8);
        assert_eq!(offset_of!(SurfaceMaterial, execution_plan_index), 12);
        assert_eq!(size_of::<[SurfaceMaterial; 2]>(), 32);
        let view = crate::surface_views::SurfaceView {
            camera: prism_ecs::prelude::Entity::from_bits(1_u64 << 32).unwrap(),
            target: prism_camera::RenderTarget::PrimaryWindow,
            clear: prism_camera::ClearMode::ColorAndDepth,
            order: 0,
            clip_from_world: [[2.0; 4]; 4],
            view_position: [0.0; 3],
            previous_clip_from_world: [[3.0; 4]; 4],
            viewport: [0, 0, 1, 1],
            history_epoch: 0,
            history_valid: true,
            resolution_scale: 1.0,
            history_frames: 0,
            exposure_stops: 0.0,
        };
        let frame = SurfaceFrame::from_view(&view);
        assert_eq!(frame.clip_from_world, view.clip_from_world);
        assert_eq!(
            frame.previous_clip_from_world,
            view.previous_clip_from_world
        );
    }

    #[test]
    fn missing_position_and_normal_are_stable_rejections() {
        assert_eq!(
            SurfaceEncodeProvider::validate_mesh_layout(&layout(false, true, true)),
            Err(SurfaceRejection::MissingPosition)
        );
        assert_eq!(
            SurfaceEncodeProvider::validate_mesh_layout(&layout(true, false, true)),
            Err(SurfaceRejection::MissingNormal)
        );
    }

    #[test]
    fn unsupported_attribute_formats_are_rejected_without_fallback() {
        let mut bad = layout(true, true, true);
        bad[0].attributes[1].format = VertexFormat::Float32x4;
        assert_eq!(
            SurfaceEncodeProvider::validate_mesh_layout(&bad),
            Err(SurfaceRejection::UnsupportedNormalFormat(
                VertexFormat::Float32x4
            ))
        );
    }

    #[test]
    fn canonical_uv0_is_required_and_has_an_explicit_float32x2_contract() {
        let mut no_uv = layout(true, true, false);
        assert_eq!(
            SurfaceEncodeProvider::validate_mesh_layout(&no_uv),
            Err(SurfaceRejection::MissingUv0)
        );
        assert_eq!(
            SurfaceRejection::MissingUv0.to_string(),
            "surface-encode mesh is missing canonical UV0 at location 2"
        );

        let valid = layout(true, true, true);
        SurfaceEncodeProvider::validate_mesh_layout(&valid).expect("canonical UV0 is valid");
        let uv = valid[0]
            .attributes
            .iter()
            .find(|attribute| attribute.shader_location == SURFACE_UV0_LOCATION)
            .expect("location 2 is declared");
        assert_eq!(uv.format, VertexFormat::Float32x2);
        assert_eq!(uv.offset, 24);
        assert_eq!(valid[0].array_stride, 32);

        no_uv[0].attributes.push(VertexAttribute {
            format: VertexFormat::Float32x3,
            offset: 24,
            shader_location: SURFACE_UV0_LOCATION,
        });
        assert_eq!(
            SurfaceEncodeProvider::validate_mesh_layout(&no_uv),
            Err(SurfaceRejection::UnsupportedUv0Format(
                VertexFormat::Float32x3
            ))
        );
        no_uv[0].attributes.push(VertexAttribute {
            format: VertexFormat::Float32x2,
            offset: 36,
            shader_location: SURFACE_UV0_LOCATION,
        });
        assert_eq!(
            SurfaceEncodeProvider::validate_mesh_layout(&no_uv),
            Err(SurfaceRejection::DuplicateUv0)
        );
    }

    #[test]
    fn shader_declares_and_forwards_uv0_without_synthesizing_coordinates() {
        let source = SURFACE_ENCODE_WGSL;
        assert!(source.contains("@location(2) uv0: vec2<f32>"));
        assert!(source.contains("output.uv0 = uv0;"));
        assert!(!source.contains("world_position.xy"));
    }

    #[test]
    fn binding_layout_declares_canonical_uniform_and_storage_ranges() {
        let contract = SurfaceBindingContract::new();
        assert_eq!(
            (
                contract.group,
                contract.frame,
                contract.material,
                contract.headers,
                contract.parameters,
                contract.textures
            ),
            (0, 0, 1, 2, 3, 4)
        );
        let descriptor = contract.layout_descriptor();
        assert!(descriptor.has_unique_bindings());
        assert_eq!(
            descriptor
                .entries
                .iter()
                .map(|entry| entry.binding)
                .collect::<Vec<_>>(),
            vec![0, 1, 2, 3, 4]
        );
        for (entry, size) in descriptor.entries[..2].iter().zip([144, 16]) {
            assert_eq!(
                entry.ty,
                BindingType::Buffer {
                    ty: BufferBindingType::Uniform,
                    has_dynamic_offset: true,
                    min_binding_size: Some(size),
                }
            );
            assert_eq!(entry.count, None);
        }
        assert_eq!(
            descriptor.entries[0].visibility,
            ShaderStages::VERTEX | ShaderStages::FRAGMENT
        );
        assert_eq!(descriptor.entries[1].visibility, ShaderStages::FRAGMENT);
        assert_eq!(
            core::mem::size_of::<prism_render_material::GpuMaterialHeader>(),
            116
        );
        for (entry, size) in descriptor.entries[2..].iter().zip([116, 4, 32]) {
            assert_eq!(
                entry.ty,
                BindingType::Buffer {
                    ty: BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: Some(size),
                }
            );
            assert_eq!(entry.visibility, ShaderStages::FRAGMENT);
            assert_eq!(entry.count, None);
        }
        // Shader/layout agreement is exercised by the physical-GPU pipeline
        // test and the raster/readback/composite test, not source substrings.
    }

    #[test]
    fn target_contract_rejects_non_hdr_variants() {
        let mut bad = variant();
        bad.color_format = Some(TextureFormat::Rgba8Unorm);
        assert_eq!(
            SurfaceEncodeProvider::validate_variant(&bad),
            Err(SurfaceRejection::RequiresRgba16FloatTarget)
        );
    }

    #[test]
    fn single_sample_contract_is_shared_by_all_descriptor_entry_points() {
        let provider = SurfaceEncodeProvider {
            module: ShaderModuleId::from_parts(1, 2),
            pipeline_layout: PipelineLayoutId::from_parts(3, 4),
            bind_group_layout: prism_render_driver::BindGroupLayoutId::from_parts(5, 6),
            vt_bind_group_layout: prism_render_driver::BindGroupLayoutId::from_parts(7, 8),
            contract: SurfaceBindingContract::new(),
        };
        let mesh_layout = layout(true, true, true);
        let expected_error = SurfaceRejection::InvalidSampleCount;
        let expected_message =
            "surface-encode requires sample_count == 1 for the single-sample GBuffer composite";
        assert_eq!(expected_error.to_string(), expected_message);

        for sample_count in [0, 1, 2, 4] {
            let key = MeshPipelineKey {
                sample_count,
                ..variant()
            };
            if sample_count == 1 {
                assert_eq!(SurfaceEncodeProvider::validate_variant(&key), Ok(()));
                let descriptors = [
                    provider.try_descriptor(&key, &mesh_layout).unwrap(),
                    provider
                        .try_transformed_descriptor(&key, &mesh_layout)
                        .unwrap(),
                    MeshPipelineProvider::descriptor(&provider, &key),
                    MeshPipelineProvider::transformed_descriptor(&provider, &key)
                        .expect("single-sample transformed variant must remain supported"),
                ];
                for (descriptor, entry_point) in
                    descriptors
                        .into_iter()
                        .zip(["vs", "vs_transform", "vs", "vs_transform"])
                {
                    assert_eq!(descriptor.multisample.count, 1);
                    assert_eq!(descriptor.vertex.entry_point, entry_point);
                }
                continue;
            }

            assert_eq!(
                SurfaceEncodeProvider::validate_variant(&key),
                Err(expected_error),
                "sample_count={sample_count}"
            );
            assert_eq!(
                provider.try_descriptor(&key, &mesh_layout).err(),
                Some(expected_error),
                "sample_count={sample_count}"
            );
            assert_eq!(
                provider
                    .try_transformed_descriptor(&key, &mesh_layout)
                    .err(),
                Some(expected_error),
                "sample_count={sample_count}"
            );
            let rejected = [
                std::panic::catch_unwind(|| MeshPipelineProvider::descriptor(&provider, &key))
                    .err(),
                std::panic::catch_unwind(|| {
                    MeshPipelineProvider::transformed_descriptor(&provider, &key)
                })
                .err(),
            ];
            for panic in rejected {
                let panic = panic.expect("production trait must reject non-single-sample variants");
                let message = panic
                    .downcast_ref::<String>()
                    .map(String::as_str)
                    .or_else(|| panic.downcast_ref::<&str>().copied());
                assert_eq!(
                    message,
                    Some(expected_message),
                    "sample_count={sample_count}"
                );
            }
        }
    }

    #[test]
    fn descriptor_declares_gbuffer_targets_and_real_mesh_entry_points() {
        let provider = SurfaceEncodeProvider {
            module: ShaderModuleId::from_parts(1, 2),
            pipeline_layout: PipelineLayoutId::from_parts(3, 4),
            bind_group_layout: prism_render_driver::BindGroupLayoutId::from_parts(5, 6),
            vt_bind_group_layout: prism_render_driver::BindGroupLayoutId::from_parts(7, 8),
            contract: SurfaceBindingContract::new(),
        };
        let descriptor = provider
            .try_descriptor(&variant(), &layout(true, true, true))
            .expect("valid surface-encode descriptor");
        assert_eq!(descriptor.layout, Some(PipelineLayoutId::from_parts(3, 4)));
        assert_eq!(descriptor.vertex.module, ShaderModuleId::from_parts(1, 2));
        assert_eq!(descriptor.vertex.entry_point, "vs");
        assert_eq!(
            descriptor.fragment.as_ref().unwrap().entry_point,
            "fs_geometry"
        );
        assert_eq!(
            descriptor.fragment.as_ref().unwrap().targets,
            SURFACE_GEOMETRY_TARGET_FORMATS
                .into_iter()
                .map(|format| Some(ColorTargetState::opaque(format)))
                .collect::<Vec<_>>()
        );
        assert_eq!(
            descriptor.color_target_count(),
            SURFACE_GEOMETRY_TARGET_FORMATS.len()
        );
        assert_eq!(
            descriptor
                .fragment
                .as_ref()
                .unwrap()
                .targets
                .iter()
                .map(|target| target.as_ref().unwrap().format)
                .collect::<Vec<_>>(),
            SURFACE_GEOMETRY_TARGET_FORMATS
        );
        let secondary = provider
            .secondary_descriptor(&variant())
            .expect("secondary surface-encode descriptor");
        assert_eq!(
            secondary.fragment.as_ref().unwrap().entry_point,
            "fs_material"
        );
        assert_eq!(
            secondary.color_target_count(),
            SURFACE_MATERIAL_TARGET_FORMATS.len()
        );
        assert_eq!(
            secondary
                .fragment
                .as_ref()
                .unwrap()
                .targets
                .iter()
                .filter_map(|target| target.as_ref().map(|target| target.format))
                .collect::<Vec<_>>(),
            SURFACE_MATERIAL_TARGET_FORMATS
        );

        let transformed = provider
            .try_transformed_descriptor(&variant(), &layout(true, true, true))
            .expect("valid transformed surface-encode descriptor");
        assert_eq!(transformed.vertex.entry_point, "vs_transform");
    }

    #[test]
    fn missing_canonical_resources_are_stable_binding_rejections() {
        let id = BufferId::from_parts(0, 0);
        let provider = SurfaceEncodeProvider {
            module: ShaderModuleId::from_parts(0, 0),
            pipeline_layout: PipelineLayoutId::from_parts(0, 0),
            bind_group_layout: prism_render_driver::BindGroupLayoutId::from_parts(0, 0),
            vt_bind_group_layout: prism_render_driver::BindGroupLayoutId::from_parts(0, 0),
            contract: SurfaceBindingContract::new(),
        };
        let complete = SurfaceResourceSet {
            frame: Some(id),
            material: Some(id),
            headers: Some(id),
            parameters: Some(id),
            textures: Some(id),
        };
        for (resources, rejection, message) in [
            (
                SurfaceResourceSet {
                    frame: None,
                    ..complete
                },
                SurfaceBindingRejection::MissingFrame,
                "surface-encode frame/view buffer is missing",
            ),
            (
                SurfaceResourceSet {
                    material: None,
                    ..complete
                },
                SurfaceBindingRejection::MissingMaterial,
                "surface-encode material buffer is missing",
            ),
            (
                SurfaceResourceSet {
                    headers: None,
                    ..complete
                },
                SurfaceBindingRejection::MissingHeaders,
                "surface-encode canonical header buffer is missing",
            ),
            (
                SurfaceResourceSet {
                    parameters: None,
                    ..complete
                },
                SurfaceBindingRejection::MissingParameters,
                "surface-encode canonical parameter heap is missing",
            ),
            (
                SurfaceResourceSet {
                    textures: None,
                    ..complete
                },
                SurfaceBindingRejection::MissingTextures,
                "surface-encode canonical texture-row buffer is missing",
            ),
        ] {
            assert_eq!(provider.bind_group_descriptor(resources), Err(rejection));
            assert_eq!(rejection.to_string(), message);
        }
    }

    #[test]
    fn dynamic_bindings_use_provider_layout_and_exact_logical_ranges() {
        let provider = SurfaceEncodeProvider {
            module: ShaderModuleId::from_parts(1, 2),
            pipeline_layout: PipelineLayoutId::from_parts(3, 4),
            bind_group_layout: prism_render_driver::BindGroupLayoutId::from_parts(5, 6),
            vt_bind_group_layout: prism_render_driver::BindGroupLayoutId::from_parts(7, 8),
            contract: SurfaceBindingContract::new(),
        };
        let frame = BufferId::from_parts(7, 8);
        let material = BufferId::from_parts(9, 10);
        let headers = BufferId::from_parts(11, 12);
        let parameters = BufferId::from_parts(13, 14);
        let textures = BufferId::from_parts(15, 16);
        let descriptor = provider
            .bind_group_descriptor(SurfaceResourceSet {
                frame: Some(frame),
                material: Some(material),
                headers: Some(headers),
                parameters: Some(parameters),
                textures: Some(textures),
            })
            .expect("uniforms and canonical storage supplied");
        assert_eq!(descriptor.layout, Some(provider.bind_group_layout()));
        assert_eq!(
            descriptor.entries,
            vec![
                buffer_entry(0, frame, 144),
                buffer_entry(1, material, 16),
                BindGroupEntry {
                    binding: 2,
                    resource: BindingResource::Buffer {
                        buffer: headers,
                        offset: 0,
                        size: None
                    },
                },
                BindGroupEntry {
                    binding: 3,
                    resource: BindingResource::Buffer {
                        buffer: parameters,
                        offset: 0,
                        size: None
                    },
                },
                BindGroupEntry {
                    binding: 4,
                    resource: BindingResource::Buffer {
                        buffer: textures,
                        offset: 0,
                        size: None
                    },
                },
            ]
        );
    }

    #[test]
    fn physical_gpu_accepts_surface_encode_shader_and_pipeline_abi() {
        use prism_render_driver::RenderDevice as _;
        use prism_render_driver_wgpu::{require_hardware_gpu, HardwareDeviceError, WgpuInstance};

        let (device, _queue) = match WgpuInstance::new().request_default_hardware_device() {
            Ok(pair) => pair,
            Err(HardwareDeviceError::NoAdapter) => {
                assert!(
                    !require_hardware_gpu(),
                    "PRISM_REQUIRE_GPU=1 but no physical GPU adapter was found"
                );
                eprintln!("SKIPPED surface-encode ABI validation: no adapter");
                return;
            }
            Err(HardwareDeviceError::NonHardware(evidence)) => {
                assert!(
                    !require_hardware_gpu(),
                    "PRISM_REQUIRE_GPU=1 requires a physical GPU; observed {evidence}"
                );
                eprintln!(
                    "SKIPPED surface-encode ABI validation: non-hardware adapter: {evidence}"
                );
                return;
            }
            Err(HardwareDeviceError::Device(error)) => {
                panic!("physical GPU was found but device creation failed: {error}");
            }
        };

        let provider = SurfaceEncodeProvider::new(&device);
        let layout = layout(true, true, true);
        let mut descriptor = provider
            .try_descriptor(&variant(), &layout)
            .expect("valid surface-encode descriptor");
        // The provider leaves physical vertex slots to mesh specialization.
        // Pipeline creation validates the real canonical location-0/1/2 ABI too.
        descriptor.vertex.buffers = layout.clone();
        device.create_render_pipeline(&descriptor);

        let mut transformed = provider
            .try_transformed_descriptor(&variant(), &layout)
            .expect("valid transformed surface-encode descriptor");
        transformed.vertex.buffers = layout;
        transformed.vertex.buffers.push(VertexBufferLayout {
            array_stride: 48,
            step_mode: VertexStepMode::Instance,
            attributes: (0..3)
                .map(|row| VertexAttribute {
                    format: VertexFormat::Float32x4,
                    offset: u64::from(row) * 16,
                    shader_location: 13 + row,
                })
                .collect(),
        });
        device.create_render_pipeline(&transformed);
    }
}
