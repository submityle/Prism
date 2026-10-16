//! Frozen selection for the Bevy device that actually owns scene buffers.
//! This is not the independently created Prism RHI device in `PrismGpu`.
use std::sync::Arc;

use bevy_ecs::prelude::*;
use bevy_render::renderer::{RenderContext, RenderDevice};
use prism_render_architecture::{
    backend::capability::{Capability, CapabilitySet},
    render_fabric::{
        builtin_catalog, builtin_resource_templates, builtin_roots, CompiledFeatureGraph,
        DeviceFabric, FrameResourceBinding, GraphLimits, PhysicalResource, RenderQualityTier,
        ResourceKey, ResourceLayout, RuntimeImport, RuntimePassBinding, WATER_SPRAY_RESOURCE,
    },
};
use prism_render_driver::{self as rhi, RenderDevice as _};

use super::rhi::{SceneRhi, SceneRhiBuffer};

#[derive(Resource)]
pub(crate) struct SceneFabric(pub(crate) DeviceFabric);

#[derive(Resource, Debug, Default)]
pub(crate) struct SceneFabricDiagnostics {
    pub(crate) compile_attempts: u64,
    pub(crate) compile_successes: u64,
    pub(crate) prepare_attempts: u64,
    pub(crate) prepare_successes: u64,
    pub(crate) frames_recorded: u64,
    pub(crate) cache_hits: u64,
    pub(crate) cache_misses: u64,
    pub(crate) compile_error: Option<String>,
    pub(crate) prepare_error: Option<String>,
    #[allow(dead_code)]
    pub(crate) record_error: Option<String>,
}

#[derive(Resource)]
pub(crate) struct SceneFabricRuntime {
    #[allow(dead_code)]
    buffers: Vec<FabricBuffer>,
    bind_groups: Arc<[(rhi::BufferId, rhi::BindGroupId)]>,
    pipeline: rhi::ComputePipelineId,
    compiled: Option<CompiledFeatureGraph>,
    imports: Arc<[RuntimeImport]>,
}

struct FabricBuffer {
    key: ResourceKey,
    frame: u64,
    lease: SceneRhiBuffer,
}

pub(crate) fn init_scene_fabric(world: &mut World) {
    let device = world.resource::<RenderDevice>();
    let mut caps = CapabilitySet::WEBGPU_BASELINE;
    let features = device.features();
    for (feature, capability) in [
        (
            wgpu_types::Features::MULTI_DRAW_INDIRECT_COUNT,
            Capability::IndirectDrawCount,
        ),
        (
            wgpu_types::Features::EXPERIMENTAL_MESH_SHADER,
            Capability::MeshShading,
        ),
        (
            wgpu_types::Features::EXPERIMENTAL_RAY_QUERY,
            Capability::RayQuery,
        ),
        (
            wgpu_types::Features::EXPERIMENTAL_RAY_TRACING_PIPELINES,
            Capability::RayTracingPipeline,
        ),
    ] {
        if features.contains(feature) {
            caps = caps.union(CapabilitySet::from_slice(&[capability]));
        }
    }
    if features.contains(
        wgpu_types::Features::TEXTURE_BINDING_ARRAY
            | wgpu_types::Features::SAMPLED_TEXTURE_AND_STORAGE_BUFFER_ARRAY_NON_UNIFORM_INDEXING,
    ) {
        caps = caps.union(CapabilitySet::from_slice(&[
            Capability::BindlessDescriptors,
        ]));
    }
    // The build profile is static, never derived from extracted/visible bodies.
    let catalog = builtin_catalog().expect("scene feature declarations");
    let roots = builtin_roots(&catalog).expect("scene build-profile roots");
    let fabric = DeviceFabric::new(catalog, roots, caps, RenderQualityTier::High)
        .expect("device cannot satisfy scene build-profile features");
    world.insert_resource(SceneFabric(fabric));
    world.init_resource::<SceneFabricDiagnostics>();
}

/// Creates only real resources in the adopted host RHI table. The small shader
/// is an adapter probe: it makes the Fabric path observable without claiming to
/// implement water/visibility semantics before their pipelines exist.
pub(crate) fn init_scene_fabric_runtime(world: &mut World) {
    let (buffers, bind_groups, pipeline, compile_result) = {
        let scene_rhi = world.resource::<SceneRhi>();
        let device = scene_rhi.device();

        let shader = device.create_shader_module(&rhi::ShaderModuleDescriptor::wgsl(
            "@group(0) @binding(0) var<storage, read_write> data: array<u32>;\n\
         @compute @workgroup_size(1) fn main(@builtin(global_invocation_id) id: vec3<u32>) {\n\
             if (id.x == 0u) { data[0] = data[0] + 1u; }\n\
         }",
        ));
        let layout = device.create_bind_group_layout(&rhi::BindGroupLayoutDescriptor {
            label: Some("scene-fabric-adapter-layout".into()),
            entries: vec![rhi::BindGroupLayoutEntry {
                binding: 0,
                visibility: rhi::ShaderStages::COMPUTE,
                ty: rhi::BindingType::Buffer {
                    ty: rhi::BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false,
                    min_binding_size: Some(4),
                },
                count: None,
            }],
        });
        let pipeline_layout = device.create_pipeline_layout(&rhi::PipelineLayoutDescriptor {
            label: Some("scene-fabric-adapter-pipeline-layout".into()),
            bind_group_layouts: vec![layout],
            push_constant_ranges: vec![],
        });
        let pipeline = device.create_compute_pipeline(&rhi::ComputePipelineDescriptor {
            label: Some("scene-fabric-adapter-pipeline".into()),
            layout: Some(pipeline_layout),
            module: shader,
            entry_point: "main".into(),
        });

        let mut buffers = Vec::new();
        for template in builtin_resource_templates() {
            let ResourceLayout::Buffer { bytes, usage } = template.layout else {
                panic!("built-in Fabric adapter requires buffer resources")
            };
            let lease = scene_rhi.create_buffer(&rhi::BufferDescriptor::new(bytes, usage));
            buffers.push(FabricBuffer {
                key: template.key,
                frame: 1,
                lease,
            });
        }
        // The water declaration has one real history edge. Keep a distinct host
        // allocation for frame zero; importing the current buffer as history would
        // hide a lifetime/provenance error.
        let history = builtin_resource_templates()
            .iter()
            .find(|template| template.key == WATER_SPRAY_RESOURCE)
            .expect("water history template");
        let ResourceLayout::Buffer { bytes, usage } = history.layout else {
            unreachable!("water history is a buffer")
        };
        buffers.push(FabricBuffer {
            key: history.key,
            frame: 0,
            lease: scene_rhi.create_buffer(&rhi::BufferDescriptor::new(bytes, usage)),
        });

        let bind_groups = buffers
            .iter()
            .map(|buffer| {
                let bind_group = device.create_bind_group(&rhi::BindGroupDescriptor {
                    label: Some("scene-fabric-adapter-bind-group".into()),
                    layout: Some(layout),
                    entries: vec![rhi::BindGroupEntry {
                        binding: 0,
                        resource: rhi::BindingResource::Buffer {
                            buffer: buffer.lease.id(),
                            offset: 0,
                            size: Some(4),
                        },
                    }],
                });
                (buffer.lease.id(), bind_group)
            })
            .collect::<Vec<_>>()
            .into();

        let bindings = buffers
            .iter()
            .map(|buffer| FrameResourceBinding {
                key: buffer.key,
                frame: buffer.frame,
                version: 0,
                physical: PhysicalResource::Buffer {
                    id: buffer.lease.id(),
                    offset: 0,
                },
                layout: builtin_resource_templates()
                    .iter()
                    .find(|template| template.key == buffer.key)
                    .expect("buffer template")
                    .layout,
                alignment: 16,
            })
            .collect::<Vec<_>>();
        let compile_result = {
            let fabric = world.resource::<SceneFabric>();
            fabric.0.compile(
                builtin_resource_templates(),
                1,
                &bindings,
                &[],
                GraphLimits::default(),
            )
        };
        device.destroy_shader_module(shader);
        device.destroy_bind_group_layout(layout);
        device.destroy_pipeline_layout(pipeline_layout);
        (buffers, bind_groups, pipeline, compile_result)
    };

    let mut diagnostics = world.resource_mut::<SceneFabricDiagnostics>();
    diagnostics.compile_attempts += 1;
    let compiled = match compile_result {
        Ok(compiled) => {
            diagnostics.compile_successes += 1;
            Some(compiled)
        }
        Err(error) => {
            diagnostics.compile_error = Some(format!("{error:?}"));
            None
        }
    };
    let imports = compiled
        .as_ref()
        .map(|compiled| {
            compiled
                .resources()
                .iter()
                .filter_map(|resource| {
                    let binding = resource.external?;
                    let PhysicalResource::Buffer { id, .. } = binding.physical else {
                        return None;
                    };
                    Some(RuntimeImport::Buffer {
                        resource: resource.id,
                        imported: prism_render_graph::ImportedBuffer {
                            buffer: id,
                            entry_state: rhi::BufferState::initial(),
                        },
                    })
                })
                .collect::<Vec<_>>()
        })
        .unwrap_or_default()
        .into();

    world.insert_resource(SceneFabricRuntime {
        buffers,
        bind_groups,
        pipeline,
        compiled,
        imports,
    });
}

pub(crate) fn record_scene_fabric(
    mut context: RenderContext,
    runtime: Res<SceneFabricRuntime>,
    mut diagnostics: ResMut<SceneFabricDiagnostics>,
    scene_rhi: Res<SceneRhi>,
) {
    let Some(compiled) = runtime.compiled.as_ref().cloned() else {
        diagnostics.cache_misses += 1;
        return;
    };
    diagnostics.cache_hits += 1;
    diagnostics.prepare_attempts += 1;
    let bind_groups = Arc::clone(&runtime.bind_groups);
    let pipeline = runtime.pipeline;
    let recorders = compiled
        .passes()
        .iter()
        .map(|pass| {
            let bind_groups = Arc::clone(&bind_groups);
            RuntimePassBinding::new(pass.key(), move |pass_context| {
                let (buffer, _, _) = pass_context
                    .buffer(0)
                    .expect("Fabric adapter pass must expose a real buffer access");
                let (_, bind_group) = bind_groups
                    .iter()
                    .find(|(id, _)| *id == buffer)
                    .copied()
                    .expect("Fabric adapter buffer must have a real bind group");
                pass_context
                    .commands()
                    .set_compute_pipeline(pipeline)
                    .set_compute_bind_group(0, bind_group, Vec::new())
                    .dispatch(1, 1, 1);
            })
        })
        .collect();
    let prepared =
        match compiled.prepare_runtime(rhi::Extent3d::new_2d(1, 1), &runtime.imports, recorders) {
            Ok(prepared) => {
                diagnostics.prepare_successes += 1;
                prepared
            }
            Err(error) => {
                diagnostics.prepare_error = Some(format!("{error:?}"));
                return;
            }
        };
    let recorded = prepared.record(scene_rhi.device().as_ref());
    let command_buffer = scene_rhi.queue().encode(recorded.commands());
    context.add_command_buffer(command_buffer);
    diagnostics.frames_recorded += 1;
}
