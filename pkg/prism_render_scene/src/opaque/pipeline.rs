use bevy_asset::{embedded_asset, load_embedded_asset, Handle};
use bevy_core_pipeline::core_3d::CORE_3D_DEPTH_FORMAT;
use bevy_ecs::prelude::*;
use bevy_material::descriptor::BindGroupLayoutDescriptor;
use bevy_mesh::{Mesh, MeshVertexBufferLayoutRef};
use bevy_pbr::{MeshPipeline, MeshPipelineKey, MeshPipelineViewLayoutKey};
use bevy_render::render_resource::*;
use bevy_shader::Shader;

use crate::buffers::GpuSceneBindGroup;

#[derive(Resource)]
pub(crate) struct GpuSceneOpaquePipeline {
    mesh_pipeline: MeshPipeline,
    scene_layout: BindGroupLayoutDescriptor,
    shader: Handle<Shader>,
}

pub(crate) fn embed_opaque_shader(app: &mut bevy_app::App) {
    embedded_asset!(app, "../shaders/opaque.wesl");
}

pub(crate) fn init_opaque_pipeline(
    mut commands: Commands,
    mesh_pipeline: Res<MeshPipeline>,
    scene_bindings: Res<GpuSceneBindGroup>,
    asset_server: Res<bevy_asset::AssetServer>,
) {
    commands.insert_resource(GpuSceneOpaquePipeline {
        mesh_pipeline: mesh_pipeline.clone(),
        scene_layout: scene_bindings.layout_descriptor.clone(),
        shader: load_embedded_asset!(asset_server.as_ref(), "../shaders/opaque.wesl"),
    });
}

impl SpecializedMeshPipeline for GpuSceneOpaquePipeline {
    type Key = MeshPipelineKey;

    fn specialize(
        &self,
        key: Self::Key,
        layout: &MeshVertexBufferLayoutRef,
    ) -> Result<RenderPipelineDescriptor, SpecializedMeshPipelineError> {
        let shader_defs = Vec::new();
        let vertex_layout = layout
            .0
            .get_layout(&[Mesh::ATTRIBUTE_POSITION.at_shader_location(0)])?;
        let view = self
            .mesh_pipeline
            .get_view_layout(MeshPipelineViewLayoutKey::from(key));
        Ok(RenderPipelineDescriptor {
            label: Some("prism gpu scene opaque".into()),
            layout: vec![
                view.main_layout,
                view.empty_layout,
                self.scene_layout.clone(),
            ],
            immediate_size: 8,
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
                    format: key.target_format(),
                    blend: None,
                    write_mask: ColorWrites::ALL,
                })],
                ..Default::default()
            }),
            primitive: PrimitiveState {
                topology: key.primitive_topology(),
                strip_index_format: key.strip_index_format(),
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
                count: key.msaa_samples(),
                ..Default::default()
            },
            ..Default::default()
        })
    }
}
