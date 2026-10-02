//! Device-side pipeline objects for the shadow-map depth rasterization pass.
//!
//! This module owns everything the [`shadow_depth_pass`](super::depth_pass) node
//! needs to draw scene geometry into the shadow atlas: the specialized render
//! pipeline (`shadow_depth.wesl`), the per-view uniform layout, and the
//! [`GpuShadowDepthView`] GPU-ABI record that mirrors the shader's uniform
//! block.
//!
//! The depth pass is deliberately independent of Bevy's `MeshPipeline`: its
//! `@group(0)` is a small per-view uniform (the CPU twin
//! [`prism_render_shading::ShadowDepthView`]) rather than the mesh view layout,
//! and its `@group(1)` is the shared GPU-scene storage table
//! ([`crate::buffers::GpuSceneBindGroup`]).  Geometry is fetched exactly like
//! the visibility raster: one instance drawn per scene entity, positions pulled
//! from the mesh allocator's shared vertex buffer.
//!
//! The shader is registered as an embedded asset from *this* file via
//! [`register_shadow_depth_shader`] and loaded from the same file in
//! [`init_shadow_depth_pipeline`], so the two call sites compute an identical
//! embedded [`AssetPath`](bevy_asset::AssetPath) key and the runtime resolve
//! cannot miss.

use bevy_app::App;
use bevy_asset::{load_embedded_asset, AssetServer, Handle};
use bevy_ecs::{prelude::*, world::FromWorld};
use bevy_material::{
    bind_group_layout_entries::{binding_types::uniform_buffer, BindGroupLayoutEntries},
    descriptor::BindGroupLayoutDescriptor,
};
use bevy_math::{Mat4, UVec4, Vec4};
use bevy_mesh::{Mesh, MeshAttributeCompressionFlags, MeshVertexBufferLayoutRef};
use bevy_pbr::MeshPipelineKey;
use bevy_render::{
    render_resource::{
        BindGroup, BindGroupLayout, BindGroupLayoutEntry, ColorTargetState, ColorWrites,
        CompareFunction, DepthBiasState, DepthStencilState, DynamicUniformBuffer, Face,
        FragmentState, FrontFace, MultisampleState, PrimitiveState, RenderPipelineDescriptor,
        ShaderStages, ShaderType, SpecializedMeshPipeline, SpecializedMeshPipelineError,
        StencilState, VertexState,
    },
    renderer::RenderDevice,
};
use bevy_shader::Shader;

use super::resources::SHADOW_ATLAS_FORMAT;

/// Depth format of the transient depth-stencil target the shadow depth pass
/// tests against.  The atlas layers themselves are colour (`R32Float`); this
/// hardware depth buffer only exists to resolve occlusion within a single
/// shadow view and is cleared and reused for every layer.
pub(crate) const SHADOW_DEPTH_FORMAT: bevy_render::render_resource::TextureFormat =
    bevy_render::render_resource::TextureFormat::Depth32Float;

/// Registers `shadow_depth.wesl` as an embedded asset.
///
/// This *must* be called from `pipeline.rs` (the same file as
/// [`init_shadow_depth_pipeline`]'s [`load_embedded_asset!`]) so the registered
/// and loaded embedded [`AssetPath`](bevy_asset::AssetPath) keys are computed
/// from the identical `file!()` location and match byte-for-byte.
pub(crate) fn register_shadow_depth_shader(app: &mut App) {
    bevy_asset::embedded_asset!(app, "../../shaders/shadow_depth.wesl");
}

/// GPU-ABI mirror of the `ShadowDepthView` uniform block in
/// `shadow_depth.wesl`, and the CPU twin of
/// [`prism_render_shading::ShadowDepthView`].
///
/// Field order and packing match the shader's `struct ShadowDepthView` exactly
/// so an uploaded [`DynamicUniformBuffer`] element is read back identically by
/// the vertex and fragment stages.
#[derive(Clone, Copy, Default, ShaderType)]
pub(crate) struct GpuShadowDepthView {
    /// Column-major world -> light-clip matrix (wgpu clip: `z` in `[0, 1]`).
    pub view_projection: Mat4,
    /// `xyz`: light world position; `w`: `1 / range` (distance mode only).
    pub light_position: Vec4,
    /// `x`: storage mode (`0` = NDC depth, `1` = normalized distance); the
    /// remaining lanes are reserved and always zero.
    pub params: UVec4,
}

impl GpuShadowDepthView {
    /// Packs a reference [`prism_render_shading::ShadowDepthView`] into the GPU
    /// record, converting the flat column-major matrix and the light-position
    /// array into their `bevy_math` equivalents.
    pub(crate) fn from_view(view: &prism_render_shading::ShadowDepthView) -> Self {
        Self {
            view_projection: Mat4::from_cols_array(&view.view_projection),
            light_position: Vec4::from_array(view.light_position),
            params: UVec4::new(view.mode.as_u32(), 0, 0, 0),
        }
    }
}

/// The single-binding layout entry for the per-view shadow uniform, shared by
/// the pipeline's `@group(0)` and the [`ShadowDepthViewUniform`] bind group so
/// they stay compatible.
fn shadow_view_layout_entries() -> [BindGroupLayoutEntry; 1] {
    BindGroupLayoutEntries::single(
        ShaderStages::VERTEX | ShaderStages::FRAGMENT,
        // `true`: bound with a dynamic offset so one buffer holds every layer's
        // view and each draw selects its slice.
        uniform_buffer::<GpuShadowDepthView>(true),
    )
}

/// Specialization key: the mesh topology / index-format bits that drive the
/// primitive state, exactly as the visibility raster keys its pipeline.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub(crate) struct ShadowDepthPipelineKey {
    /// Mesh topology and strip-index flags for this draw.
    pub mesh: MeshPipelineKey,
}

/// The specialized-pipeline template for the shadow depth pass.
#[derive(Resource)]
pub(crate) struct ShadowDepthPipeline {
    /// `@group(0)`: the per-view shadow uniform layout.
    shadow_view_layout: BindGroupLayoutDescriptor,
    /// `@group(1)`: the shared GPU-scene storage table (a superset of the two
    /// bindings the shader reads).
    scene_layout: BindGroupLayoutDescriptor,
    /// The embedded `shadow_depth.wesl` handle.
    shader: Handle<Shader>,
}

/// Builds the [`ShadowDepthPipeline`] once the GPU-scene bind group layout
/// exists, cloning that layout so the depth pass and the resolve pass agree on
/// the scene table binding.
pub(crate) fn init_shadow_depth_pipeline(
    mut commands: Commands,
    scene: Res<crate::buffers::GpuSceneBindGroup>,
    asset_server: Res<AssetServer>,
) {
    commands.insert_resource(ShadowDepthPipeline {
        shadow_view_layout: BindGroupLayoutDescriptor::new(
            "prism shadow depth view",
            &shadow_view_layout_entries(),
        ),
        scene_layout: scene.layout_descriptor.clone(),
        shader: load_embedded_asset!(asset_server.as_ref(), "../../shaders/shadow_depth.wesl"),
    });
}

impl SpecializedMeshPipeline for ShadowDepthPipeline {
    type Key = ShadowDepthPipelineKey;

    fn specialize(
        &self,
        key: Self::Key,
        layout: &MeshVertexBufferLayoutRef,
    ) -> Result<RenderPipelineDescriptor, SpecializedMeshPipelineError> {
        let mut shader_defs = Vec::new();
        if layout
            .0
            .get_attribute_compression()
            .contains(MeshAttributeCompressionFlags::COMPRESS_POSITION)
        {
            shader_defs.push("VERTEX_POSITIONS_COMPRESSED".into());
        }
        let vertex_layout = layout
            .0
            .get_layout(&[Mesh::ATTRIBUTE_POSITION.at_shader_location(0)])?;
        Ok(RenderPipelineDescriptor {
            label: Some("prism shadow depth".into()),
            layout: vec![self.shadow_view_layout.clone(), self.scene_layout.clone()],
            immediate_size: 0,
            vertex: VertexState {
                shader: self.shader.clone(),
                shader_defs: shader_defs.clone(),
                buffers: vec![vertex_layout],
                ..Default::default()
            },
            fragment: Some(FragmentState {
                shader: self.shader.clone(),
                shader_defs,
                targets: vec![Some(ColorTargetState {
                    format: SHADOW_ATLAS_FORMAT,
                    blend: None,
                    write_mask: ColorWrites::ALL,
                })],
                ..Default::default()
            }),
            primitive: PrimitiveState {
                topology: key.mesh.primitive_topology(),
                strip_index_format: key.mesh.strip_index_format(),
                front_face: FrontFace::Ccw,
                cull_mode: Some(Face::Back),
                ..Default::default()
            },
            depth_stencil: Some(DepthStencilState {
                format: SHADOW_DEPTH_FORMAT,
                depth_write_enabled: Some(true),
                depth_compare: Some(CompareFunction::Less),
                stencil: StencilState::default(),
                bias: DepthBiasState::default(),
            }),
            multisample: MultisampleState::default(),
            ..Default::default()
        })
    }
}

/// The per-view uniform buffer and its bind group, uploaded once per frame with
/// one dynamic-offset slice per admitted atlas layer.
#[derive(Resource)]
pub(crate) struct ShadowDepthViewUniform {
    /// One [`GpuShadowDepthView`] element per depth draw; indexed by dynamic
    /// offset when the layer is rendered.
    pub buffer: DynamicUniformBuffer<GpuShadowDepthView>,
    /// The `@group(0)` layout the bind group is built against.
    pub layout: BindGroupLayout,
    /// The bind group, rebuilt every frame after the buffer is written.
    pub bind_group: Option<BindGroup>,
}

impl FromWorld for ShadowDepthViewUniform {
    fn from_world(world: &mut World) -> Self {
        let device = world.resource::<RenderDevice>();
        let layout = device
            .create_bind_group_layout("prism shadow depth view", &shadow_view_layout_entries());
        let mut buffer = DynamicUniformBuffer::default();
        buffer.set_label(Some("prism shadow depth views"));
        Self {
            buffer,
            layout,
            bind_group: None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use prism_render_shading::{ShadowDepthMode, ShadowDepthView};

    #[test]
    fn gpu_view_packs_reference_fields_in_order() {
        let mut view_projection = [0.0_f32; 16];
        for (index, slot) in view_projection.iter_mut().enumerate() {
            *slot = index as f32;
        }
        let reference = ShadowDepthView {
            view_projection,
            light_position: [1.0, 2.0, 3.0, 0.25],
            mode: ShadowDepthMode::Distance,
        };
        let gpu = GpuShadowDepthView::from_view(&reference);
        assert_eq!(gpu.view_projection, Mat4::from_cols_array(&view_projection));
        assert_eq!(gpu.light_position, Vec4::new(1.0, 2.0, 3.0, 0.25));
        assert_eq!(gpu.params, UVec4::new(1, 0, 0, 0));
    }

    #[test]
    fn gpu_view_ndc_mode_maps_to_zero() {
        let gpu = GpuShadowDepthView::from_view(&ShadowDepthView {
            view_projection: [0.0; 16],
            light_position: [0.0; 4],
            mode: ShadowDepthMode::Ndc,
        });
        assert_eq!(gpu.params.x, 0);
    }
}
