//! Specialized render pipeline for the transparent forward (WBOIT) draw pass.
//!
//! Draws transparent geometry from Prism's retained GPU Scene through
//! `shaders/transparent.wesl`, writing the two WBOIT MRT contributions the
//! hardware blend then folds into [`super::targets::ViewOitTargets`]:
//!
//! * `@location(0)` accumulation ([`super::targets::OIT_ACCUM_FORMAT`],
//!   `Rgba16Float`): blend `One`/`One` add on colour *and* alpha, so each
//!   fragment's weighted premultiplied colour and weighted alpha are summed;
//! * `@location(1)` revealage ([`super::targets::OIT_REVEALAGE_FORMAT`],
//!   `R16Float`): blend `Zero`/`Src` add. The shader emits `1 - alpha`, so
//!   `dst * src` turns the target into the running product of `1 - alpha`
//!   (`write_mask = RED` since it is single-channel).
//!
//! Depth testing reads the opaque depth prepared by the visibility raster
//! (reverse-Z `GreaterEqual`) so opaque geometry correctly occludes
//! transparency, but `depth_write_enabled = false` keeps transparent fragments
//! from writing depth - order-independent blending must see every fragment that
//! passes the opaque test. The pass is single-sampled, matching the rest of the
//! visibility path.
//!
//! Bind-group layout and vertex plumbing mirror [`super::super::raster`] and
//! [`crate::opaque::pipeline`]: `[view.main, view.empty, scene, material]`.

use bevy_asset::{load_embedded_asset, Handle};
use bevy_core_pipeline::core_3d::CORE_3D_DEPTH_FORMAT;
use bevy_ecs::prelude::*;
use bevy_material::descriptor::BindGroupLayoutDescriptor;
use bevy_mesh::{Mesh, MeshAttributeCompressionFlags, MeshVertexBufferLayoutRef};
use bevy_pbr::{MeshPipeline, MeshPipelineKey, MeshPipelineViewLayoutKey};
use bevy_render::render_resource::*;
use bevy_shader::Shader;

use crate::{buffers::GpuSceneBindGroup, MaterialBindGroup};

use super::targets::{OIT_ACCUM_FORMAT, OIT_REVEALAGE_FORMAT};

/// Specialization key for the transparent forward pipeline. Beyond the mesh
/// pipeline key it records which optional vertex attributes the mesh carries so
/// the shader defs and vertex layout stay in lockstep with the buffer.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub(crate) struct OitForwardPipelineKey {
    pub(crate) mesh: MeshPipelineKey,
    pub(crate) has_normals: bool,
    pub(crate) has_uvs: bool,
}

/// Render pipeline specializer for the transparent forward pass. Owns the same
/// scene + material layouts as the opaque/visibility passes and the embedded
/// `transparent.wesl` module.
#[derive(Resource)]
pub(crate) struct OitForwardPipeline {
    mesh_pipeline: MeshPipeline,
    scene_layout: BindGroupLayoutDescriptor,
    material_layout: BindGroupLayoutDescriptor,
    shader: Handle<Shader>,
}

/// `RenderStartup` initializer, ordered after `MeshPipelineSystems` so the view
/// layouts exist. Mirrors [`super::super::raster::init_visibility_raster`].
pub(crate) fn init_oit_forward_pipeline(
    mut commands: Commands,
    mesh_pipeline: Res<MeshPipeline>,
    scene: Res<GpuSceneBindGroup>,
    material: Res<MaterialBindGroup>,
    asset_server: Res<bevy_asset::AssetServer>,
) {
    commands.insert_resource(OitForwardPipeline {
        mesh_pipeline: mesh_pipeline.clone(),
        scene_layout: scene.layout_descriptor.clone(),
        material_layout: material.layout_descriptor.clone(),
        shader: load_embedded_asset!(asset_server.as_ref(), "../shaders/transparent.wesl"),
    });
}

impl SpecializedMeshPipeline for OitForwardPipeline {
    type Key = OitForwardPipelineKey;

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

        // Accumulation target: additive on colour AND alpha so both the weighted
        // premultiplied colour (rgb) and the weighted alpha (a) are summed.
        let accum_blend = BlendState {
            color: BlendComponent {
                src_factor: BlendFactor::One,
                dst_factor: BlendFactor::One,
                operation: BlendOperation::Add,
            },
            alpha: BlendComponent {
                src_factor: BlendFactor::One,
                dst_factor: BlendFactor::One,
                operation: BlendOperation::Add,
            },
        };
        // Revealage target: dst * src. The shader emits `1 - alpha`, so this
        // turns the target into the running product of `1 - alpha`.
        let revealage_blend = BlendState {
            color: BlendComponent {
                src_factor: BlendFactor::Zero,
                dst_factor: BlendFactor::Src,
                operation: BlendOperation::Add,
            },
            alpha: BlendComponent {
                src_factor: BlendFactor::Zero,
                dst_factor: BlendFactor::Src,
                operation: BlendOperation::Add,
            },
        };

        Ok(RenderPipelineDescriptor {
            label: Some("prism oit forward".into()),
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
                targets: vec![
                    Some(ColorTargetState {
                        format: OIT_ACCUM_FORMAT,
                        blend: Some(accum_blend),
                        write_mask: ColorWrites::ALL,
                    }),
                    Some(ColorTargetState {
                        format: OIT_REVEALAGE_FORMAT,
                        blend: Some(revealage_blend),
                        write_mask: ColorWrites::RED,
                    }),
                ],
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
                // Test against opaque depth (reverse-Z) so opaque geometry
                // occludes transparency, but never write - OIT must see every
                // fragment that passes the opaque test.
                depth_write_enabled: Some(false),
                depth_compare: Some(CompareFunction::GreaterEqual),
                stencil: StencilState::default(),
                bias: DepthBiasState::default(),
            }),
            // The visibility path only runs single-sampled.
            multisample: MultisampleState::default(),
            ..Default::default()
        })
    }
}
