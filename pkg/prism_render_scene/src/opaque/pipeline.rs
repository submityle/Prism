use bevy_asset::{embedded_asset, load_embedded_asset, Handle};
use bevy_core_pipeline::core_3d::CORE_3D_DEPTH_FORMAT;
use bevy_ecs::prelude::*;
use bevy_material::descriptor::BindGroupLayoutDescriptor;
use bevy_mesh::{Mesh, MeshAttributeCompressionFlags, MeshVertexBufferLayoutRef};
use bevy_pbr::{MeshPipeline, MeshPipelineKey, MeshPipelineViewLayoutKey};
use bevy_render::render_resource::*;
use bevy_shader::Shader;

use crate::{buffers::GpuSceneBindGroup, MaterialBindGroup};

#[derive(Resource, Clone, Copy, Debug, Default, Eq, Hash, PartialEq)]
pub enum GpuSceneDebugView {
    #[default]
    Shaded,
    InstanceId,
    GeometryId,
    MaterialId,
    Motion,
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub(crate) struct GpuSceneOpaquePipelineKey {
    pub mesh: MeshPipelineKey,
    pub debug: GpuSceneDebugView,
    pub has_normals: bool,
    pub has_uvs: bool,
}

#[derive(Resource)]
pub(crate) struct GpuSceneOpaquePipeline {
    mesh_pipeline: MeshPipeline,
    scene_layout: BindGroupLayoutDescriptor,
    material_layout: BindGroupLayoutDescriptor,
    shader: Handle<Shader>,
}

pub(crate) fn embed_opaque_shader(app: &mut bevy_app::App) {
    embedded_asset!(app, "../shaders/opaque.wesl");
}

pub(crate) fn init_opaque_pipeline(
    mut commands: Commands,
    mesh_pipeline: Res<MeshPipeline>,
    scene_bindings: Res<GpuSceneBindGroup>,
    material_bindings: Res<MaterialBindGroup>,
    asset_server: Res<bevy_asset::AssetServer>,
) {
    commands.insert_resource(GpuSceneOpaquePipeline {
        mesh_pipeline: mesh_pipeline.clone(),
        scene_layout: scene_bindings.layout_descriptor.clone(),
        material_layout: material_bindings.layout_descriptor.clone(),
        shader: load_embedded_asset!(asset_server.as_ref(), "../shaders/opaque.wesl"),
    });
}

#[cfg(test)]
mod tests {
    use super::GpuSceneDebugView;
    use bevy_asset::{uuid::Uuid, AssetId};
    use bevy_shader::{Shader, ShaderCache, ShaderCacheSource, ShaderDefVal};

    fn load_source(
        _: &(),
        source: ShaderCacheSource,
        _: &bevy_shader::ValidateShader,
    ) -> Result<String, bevy_shader::ShaderCacheError> {
        match source {
            ShaderCacheSource::Wgsl(source) => Ok(source),
            ShaderCacheSource::SpirV(_) => unreachable!("opaque shader is WESL"),
        }
    }

    #[test]
    fn opaque_wesl_compiles_for_every_specialization() {
        let shader_id = AssetId::Uuid {
            uuid: Uuid::from_u128(0x5052_4953_4d4f_5041_5155_4553_4844_5201),
        };
        let mut cache = ShaderCache::new((), load_source);
        let source = include_str!("../shaders/opaque.wesl");
        let source = source[source.find("struct GpuSceneInstance").unwrap()..].replace(
            "bevy_render::utils::decompress_vertex_position",
            "decompress_vertex_position",
        );
        let stubs = r#"
struct TestView { world_position: vec3<f32> }
var<private> view: TestView;
fn affine3_to_square(value: mat3x4<f32>) -> mat4x4<f32> {
    return mat4x4<f32>(
        vec4<f32>(1.0, 0.0, 0.0, 0.0),
        vec4<f32>(0.0, 1.0, 0.0, 0.0),
        vec4<f32>(0.0, 0.0, 1.0, 0.0),
        vec4<f32>(0.0, 0.0, 0.0, 1.0),
    );
}
fn position_world_to_clip(position: vec3<f32>) -> vec4<f32> {
    return vec4<f32>(position, 1.0);
}
fn decompress_vertex_position(
    position: vec4<f32>,
    center: vec3<f32>,
    half_extents: vec3<f32>,
) -> vec3<f32> {
    return position.xyz;
}
"#;
        cache.set_shader(
            shader_id,
            Shader::from_wesl(format!("{stubs}{source}"), "shaders/prism_opaque.wesl"),
        );
        for compressed in [false, true] {
            for debug_view in 0..=GpuSceneDebugView::Motion as u32 {
                for normals in [false, true] {
                    for uvs in [false, true] {
                        let mut defs = Vec::new();
                        for candidate in 1..=GpuSceneDebugView::Motion as u32 {
                            defs.push(ShaderDefVal::Bool(
                                format!("PRISM_DEBUG_VIEW_{candidate}").into(),
                                debug_view == candidate,
                            ));
                        }
                        if compressed {
                            defs.push("VERTEX_POSITIONS_COMPRESSED".into());
                        }
                        if normals {
                            defs.push("VERTEX_NORMALS".into());
                        }
                        if uvs {
                            defs.push("VERTEX_UVS".into());
                        }
                        cache
                            .get(debug_view as usize, shader_id, &defs)
                            .unwrap_or_else(|error| {
                                panic!("opaque specialization failed: {error}")
                            });
                    }
                }
            }
        }
    }
}

impl SpecializedMeshPipeline for GpuSceneOpaquePipeline {
    type Key = GpuSceneOpaquePipelineKey;

    fn specialize(
        &self,
        key: Self::Key,
        layout: &MeshVertexBufferLayoutRef,
    ) -> Result<RenderPipelineDescriptor, SpecializedMeshPipelineError> {
        let mut shader_defs = Vec::new();
        if key.has_normals {
            shader_defs.push("VERTEX_NORMALS".into());
        }
        if key.has_uvs {
            shader_defs.push("VERTEX_UVS".into());
        }
        for debug_view in 1..=GpuSceneDebugView::Motion as u32 {
            shader_defs.push(bevy_shader::ShaderDefVal::Bool(
                format!("PRISM_DEBUG_VIEW_{debug_view}").into(),
                key.debug as u32 == debug_view,
            ));
        }
        if layout
            .0
            .get_attribute_compression()
            .contains(MeshAttributeCompressionFlags::COMPRESS_POSITION)
        {
            shader_defs.push("VERTEX_POSITIONS_COMPRESSED".into());
        }
        let mut attributes = vec![Mesh::ATTRIBUTE_POSITION.at_shader_location(0)];
        if key.has_normals {
            attributes.push(Mesh::ATTRIBUTE_NORMAL.at_shader_location(1));
        }
        if key.has_uvs {
            attributes.push(Mesh::ATTRIBUTE_UV_0.at_shader_location(2));
        }
        let vertex_layout = layout.0.get_layout(&attributes)?;
        let view = self
            .mesh_pipeline
            .get_view_layout(MeshPipelineViewLayoutKey::from(key.mesh));
        Ok(RenderPipelineDescriptor {
            label: Some("prism gpu scene opaque".into()),
            layout: vec![
                view.main_layout,
                view.empty_layout,
                self.scene_layout.clone(),
                self.material_layout.clone(),
            ],
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
                    format: key.mesh.target_format(),
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
                format: CORE_3D_DEPTH_FORMAT,
                depth_write_enabled: Some(true),
                depth_compare: Some(CompareFunction::GreaterEqual),
                stencil: StencilState::default(),
                bias: DepthBiasState::default(),
            }),
            multisample: MultisampleState {
                count: key.mesh.msaa_samples(),
                ..Default::default()
            },
            ..Default::default()
        })
    }
}
