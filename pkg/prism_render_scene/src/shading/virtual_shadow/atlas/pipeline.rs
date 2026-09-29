//! Device-side pipeline objects for the virtual-shadow-map caster depth pass.
//!
//! This is the VSM twin of [`super::super::super::shadow::pipeline`]: it owns the
//! specialized render pipeline that rasterizes shadow casters into a page-sized
//! depth target, the per-page uniform layout / buffer, and the embedded shader
//! registration. Like the classic shadow depth pass it is independent of Bevy's
//! `MeshPipeline`:
//!
//! * `@group(0)` is the small per-page uniform
//!   ([`super::abi::GpuVsmCasterDepthView`], one `world -> light-clip` matrix),
//!   bound with a dynamic offset so one buffer drives every resident page.
//! * `@group(1)` is the shared GPU-scene storage table
//!   ([`crate::buffers::GpuSceneBindGroup`]), reused verbatim so the same scene
//!   bind group drives every geometry pass.
//!
//! The colour target is one `R32Float` page tile (the atlas texel format,
//! [`VSM_CASTER_DEPTH_ATTACHMENT_FORMAT`]) whose `.r` stores wgpu NDC depth; a
//! transient [`VSM_CASTER_DEPTH_FORMAT`] depth buffer resolves occlusion within a
//! page. The shader is registered as an embedded asset from *this* file via
//! [`register_vsm_caster_depth_shader`] and loaded from the same file in
//! [`init_vsm_caster_depth_pipeline`], so the two call sites compute an identical
//! embedded [`AssetPath`](bevy_asset::AssetPath) key and the runtime resolve
//! cannot miss.

use bevy_app::App;
use bevy_asset::{load_embedded_asset, AssetServer, Handle};
use bevy_ecs::{prelude::*, world::FromWorld};
use bevy_material::descriptor::BindGroupLayoutDescriptor;
use bevy_mesh::{Mesh, MeshAttributeCompressionFlags, MeshVertexBufferLayoutRef};
use bevy_pbr::MeshPipelineKey;
use bevy_render::{
    render_resource::{
        BindGroup, BindGroupLayout, ColorTargetState, ColorWrites, CompareFunction, DepthBiasState,
        DepthStencilState, DynamicUniformBuffer, Face, FragmentState, FrontFace, MultisampleState,
        PrimitiveState, RenderPipelineDescriptor, SpecializedMeshPipeline,
        SpecializedMeshPipelineError, StencilState, TextureFormat, VertexState,
    },
    renderer::RenderDevice,
};
use bevy_shader::Shader;

use super::abi::GpuVsmCasterDepthView;
use super::bind_groups::caster_depth_view_layout_entries;
use super::resources::VSM_PHYSICAL_ATLAS_FORMAT;

/// Colour attachment format of the caster-depth pass: the physical atlas's own
/// single-channel `R32Float`. The pass renders straight into the atlas texture
/// (one page-sized viewport per resident page), so this must equal the atlas
/// format exactly. Its `.r` holds the wgpu NDC depth `vsm_sample.wesl` reads back.
pub(crate) const VSM_CASTER_DEPTH_ATTACHMENT_FORMAT: TextureFormat = VSM_PHYSICAL_ATLAS_FORMAT;

/// Format of the transient depth-stencil buffer the caster-depth pass tests
/// against. It exists only to resolve occlusion within a single page render and
/// is cleared and reused for every page; the *stored* depth is the colour
/// target's `.r`, not this buffer.
pub(crate) const VSM_CASTER_DEPTH_FORMAT: TextureFormat = TextureFormat::Depth32Float;

/// Registers `vsm_caster_depth.wesl` as an embedded asset.
///
/// This *must* be called from `pipeline.rs` (the same file as
/// [`init_vsm_caster_depth_pipeline`]'s [`load_embedded_asset!`]) so the
/// registered and loaded embedded [`AssetPath`](bevy_asset::AssetPath) keys are
/// computed from the identical `file!()` location and match byte-for-byte.
pub(crate) fn register_vsm_caster_depth_shader(app: &mut App) {
    bevy_asset::embedded_asset!(app, "../../../shaders/vsm_caster_depth.wesl");
}

/// Specialization key: the mesh topology / index-format bits that drive the
/// primitive state, exactly as the classic shadow depth pass keys its pipeline.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub(crate) struct VsmCasterDepthPipelineKey {
    /// Mesh topology and strip-index flags for this draw.
    pub mesh: MeshPipelineKey,
}

/// The specialized-pipeline template for the VSM caster depth pass.
#[derive(Resource)]
pub(crate) struct VsmCasterDepthPipeline {
    /// `@group(0)`: the per-page caster-depth uniform layout.
    view_layout: BindGroupLayoutDescriptor,
    /// `@group(1)`: the shared GPU-scene storage table (a superset of the two
    /// bindings the shader reads).
    scene_layout: BindGroupLayoutDescriptor,
    /// The embedded `vsm_caster_depth.wesl` handle.
    shader: Handle<Shader>,
}

/// Builds the [`VsmCasterDepthPipeline`] once the GPU-scene bind group layout
/// exists, cloning that layout so the caster-depth pass and the resolve pass
/// agree on the scene table binding.
pub(crate) fn init_vsm_caster_depth_pipeline(
    mut commands: Commands,
    scene: Res<crate::buffers::GpuSceneBindGroup>,
    asset_server: Res<AssetServer>,
) {
    commands.insert_resource(VsmCasterDepthPipeline {
        view_layout: BindGroupLayoutDescriptor::new(
            "prism vsm caster depth view",
            &caster_depth_view_layout_entries(),
        ),
        scene_layout: scene.layout_descriptor.clone(),
        shader: load_embedded_asset!(
            asset_server.as_ref(),
            "../../../shaders/vsm_caster_depth.wesl"
        ),
    });
}

impl SpecializedMeshPipeline for VsmCasterDepthPipeline {
    type Key = VsmCasterDepthPipelineKey;

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
            label: Some("prism vsm caster depth".into()),
            layout: vec![self.view_layout.clone(), self.scene_layout.clone()],
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
                    format: VSM_CASTER_DEPTH_ATTACHMENT_FORMAT,
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
                format: VSM_CASTER_DEPTH_FORMAT,
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

/// The per-page uniform buffer and its bind group, uploaded once per frame with
/// one dynamic-offset slice per resident clipmap page (across every view).
#[derive(Resource)]
pub(crate) struct VsmCasterDepthViewUniform {
    /// One [`GpuVsmCasterDepthView`] element per resident page; indexed by
    /// dynamic offset when the page's tile is rendered.
    pub buffer: DynamicUniformBuffer<GpuVsmCasterDepthView>,
    /// The `@group(0)` layout the bind group is built against.
    pub layout: BindGroupLayout,
    /// The bind group, rebuilt every frame after the buffer is written.
    pub bind_group: Option<BindGroup>,
}

impl FromWorld for VsmCasterDepthViewUniform {
    fn from_world(world: &mut World) -> Self {
        let device = world.resource::<RenderDevice>();
        let layout = device
            .create_bind_group_layout("prism vsm caster depth view", &caster_depth_view_layout_entries());
        let mut buffer = DynamicUniformBuffer::default();
        buffer.set_label(Some("prism vsm caster depth views"));
        Self {
            buffer,
            layout,
            bind_group: None,
        }
    }
}

#[cfg(test)]
mod tests {
    use bevy_asset::{uuid::Uuid, AssetId};
    use bevy_shader::{Shader, ShaderCache, ShaderCacheError, ShaderCacheSource, ValidateShader};

    fn load_source(
        _: &(),
        source: ShaderCacheSource,
        _: &ValidateShader,
    ) -> Result<String, ShaderCacheError> {
        match source {
            ShaderCacheSource::Wgsl(source) => Ok(source),
            ShaderCacheSource::SpirV(_) => unreachable!("vsm caster depth shader is WESL"),
        }
    }

    /// Compiles `vsm_caster_depth.wesl` standalone. It has no imports, so a green
    /// result proves the per-page uniform block, the mirrored GPU-scene storage
    /// layout, the affine-transform expansion, the invalid-instance cull and the
    /// NDC-depth store all parse and type-check on their own -- and that the
    /// vertex / fragment entry points the pipeline names actually exist.
    #[test]
    fn vsm_caster_depth_wesl_compiles_standalone() {
        let mut cache = ShaderCache::new((), load_source);
        let caster = AssetId::Uuid {
            uuid: Uuid::from_u128(0x5052_4953_4d5f_5653_4d5f_4353_5444_0001),
        };
        cache.set_shader(
            caster,
            Shader::from_wesl(
                include_str!("../../../shaders/vsm_caster_depth.wesl"),
                "embedded://prism_render_scene/shaders/vsm_caster_depth.wesl",
            ),
        );
        cache
            .get(0, caster, &[])
            .unwrap_or_else(|error| panic!("vsm_caster_depth.wesl failed to compile: {error:?}"));
    }
}
