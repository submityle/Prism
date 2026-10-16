use prism_render_graph::frame_graph::{
    AccessKind, GpuFrameGraphBuilder, PassDescriptor, QueueClass, ResourceAccess,
    ResourceDescriptor, ResourceLifetime,
};

use super::composition::{CompositionPlane, COMPOSITION_PASS_NAME};
use super::post_graph::append_post_process_passes;

pub(crate) fn shading_frame_graph() -> GpuFrameGraphBuilder {
    let mut graph = GpuFrameGraphBuilder::default();
    let scene = resource(&mut graph, "gpu_scene", ResourceLifetime::Imported);
    let geometry = resource(&mut graph, "geometry", ResourceLifetime::Imported);
    let materials = resource(&mut graph, "materials", ResourceLifetime::Imported);
    let early_commands = resource(&mut graph, "early_commands", ResourceLifetime::Imported);
    let late_commands = resource(&mut graph, "late_commands", ResourceLifetime::Imported);
    let depth = resource(&mut graph, "main_depth", ResourceLifetime::Imported);
    let visibility_ids = resource(&mut graph, "visibility_ids", ResourceLifetime::Transient);
    let visibility_metadata = resource(
        &mut graph,
        "visibility_metadata",
        ResourceLifetime::Transient,
    );
    let pixel_classification = resource(
        &mut graph,
        "material_pixel_classification",
        ResourceLifetime::Transient,
    );
    let shading_work = resource(
        &mut graph,
        "material_shading_work",
        ResourceLifetime::Transient,
    );
    // The classification scatter produces the bounded set of texel-space
    // jobs for this frame.  The page table and atlas outlive the frame graph:
    // a completed shade generation remains sampleable until the next
    // completion-gated publication replaces it.
    let shade_jobs = resource(
        &mut graph,
        "texel_space_shade_jobs",
        ResourceLifetime::Transient,
    );
    let shade_page_table = resource(
        &mut graph,
        "texel_space_shade_page_table",
        ResourceLifetime::Persistent,
    );
    let shade_atlas = resource(
        &mut graph,
        "texel_space_shade_atlas",
        ResourceLifetime::Persistent,
    );
    let class_counts = resource(
        &mut graph,
        "material_class_counts",
        ResourceLifetime::Transient,
    );
    let class_offsets = resource(
        &mut graph,
        "material_class_offsets",
        ResourceLifetime::Transient,
    );
    let class_cursors = resource(
        &mut graph,
        "material_class_cursors",
        ResourceLifetime::Transient,
    );
    let classification_diagnostics = resource(
        &mut graph,
        "material_classification_diagnostics",
        ResourceLifetime::Transient,
    );
    let indirect_dispatch = resource(
        &mut graph,
        "shading_indirect_dispatch",
        ResourceLifetime::Transient,
    );
    let scene_color = resource(&mut graph, "hdr_scene_color", ResourceLifetime::Imported);

    let raster = graph.add_pass(PassDescriptor {
        name: "visibility_raster".into(),
        queue: QueueClass::Graphics,
        accesses: vec![
            access(scene, AccessKind::StorageRead),
            access(geometry, AccessKind::StorageRead),
            access(materials, AccessKind::StorageRead),
            access(early_commands, AccessKind::IndirectRead),
            access(late_commands, AccessKind::IndirectRead),
            access(depth, AccessKind::DepthAttachment),
            access(visibility_ids, AccessKind::ColorAttachment),
            access(visibility_metadata, AccessKind::ColorAttachment),
        ],
        depends_on: vec![],
    });
    let classify = graph.add_pass(PassDescriptor {
        name: "material_classification_count".into(),
        queue: QueueClass::Compute,
        accesses: vec![
            access(visibility_ids, AccessKind::SampledRead),
            access(visibility_metadata, AccessKind::SampledRead),
            access(materials, AccessKind::StorageRead),
            access(pixel_classification, AccessKind::StorageWrite),
            access(class_counts, AccessKind::StorageWrite),
            access(classification_diagnostics, AccessKind::StorageWrite),
        ],
        depends_on: vec![raster],
    });
    let prefix = graph.add_pass(PassDescriptor {
        name: "material_classification_prefix".into(),
        queue: QueueClass::Compute,
        accesses: vec![
            access(class_counts, AccessKind::StorageRead),
            access(class_offsets, AccessKind::StorageWrite),
            access(class_cursors, AccessKind::StorageWrite),
            access(indirect_dispatch, AccessKind::StorageWrite),
        ],
        depends_on: vec![classify],
    });
    let scatter = graph.add_pass(PassDescriptor {
        name: "material_classification_scatter".into(),
        queue: QueueClass::Compute,
        accesses: vec![
            // Scatter also writes each accepted pixel's compact work index.
            access(pixel_classification, AccessKind::StorageWrite),
            access(class_offsets, AccessKind::StorageRead),
            access(class_cursors, AccessKind::StorageWrite),
            access(shading_work, AccessKind::StorageWrite),
            access(shade_jobs, AccessKind::StorageWrite),
            access(classification_diagnostics, AccessKind::StorageWrite),
        ],
        depends_on: vec![prefix],
    });
    let shade = graph.add_pass(PassDescriptor {
        name: "texel_space_shade_cache".into(),
        queue: QueueClass::Compute,
        accesses: vec![
            access(shade_jobs, AccessKind::StorageRead),
            access(shade_page_table, AccessKind::StorageRead),
            access(shade_atlas, AccessKind::StorageWrite),
        ],
        depends_on: vec![scatter],
    });
    // Producers own these per-view GPU allocations. They are imported, not
    // transient aliases or frame-lag history; the resident endpoint admits the
    // actual extent/64B/16B capacities before allocation. No scene flag changes
    // this topology. Until split producers are connected, absence of a fresh
    // publication makes only the execution a no-op.
    let left = resource(
        &mut graph,
        CompositionPlane::Left.graph_name(),
        ResourceLifetime::Imported,
    );
    let right = resource(
        &mut graph,
        CompositionPlane::Right.graph_name(),
        ResourceLifetime::Imported,
    );
    let controls = resource(
        &mut graph,
        CompositionPlane::Controls.graph_name(),
        ResourceLifetime::Imported,
    );
    let auxiliary = resource(
        &mut graph,
        CompositionPlane::Auxiliary.graph_name(),
        ResourceLifetime::Imported,
    );
    let water_coverage = resource(
        &mut graph,
        CompositionPlane::WaterCoverage.graph_name(),
        ResourceLifetime::Imported,
    );
    // These exports are written by shading_resolve, then sampled by the SSR
    // or spec-GI composite. They are not outputs of the SSR trace group.
    let env_specular = resource(
        &mut graph,
        "post_ssr_env_specular",
        ResourceLifetime::Transient,
    );
    let spec_weight = resource(
        &mut graph,
        "post_ssr_spec_weight",
        ResourceLifetime::Transient,
    );
    let resolve = graph.add_pass(PassDescriptor {
        name: "shading_resolve".into(),
        queue: QueueClass::Compute,
        accesses: vec![
            access(visibility_ids, AccessKind::StorageRead),
            access(visibility_metadata, AccessKind::StorageRead),
            access(depth, AccessKind::SampledRead),
            access(scene, AccessKind::StorageRead),
            access(geometry, AccessKind::StorageRead),
            access(materials, AccessKind::StorageRead),
            access(shading_work, AccessKind::StorageRead),
            access(indirect_dispatch, AccessKind::IndirectRead),
            access(shade_page_table, AccessKind::StorageRead),
            access(shade_atlas, AccessKind::SampledRead),
            access(left, AccessKind::StorageWrite),
            access(scene_color, AccessKind::StorageWrite),
            access(env_specular, AccessKind::StorageWrite),
            access(spec_weight, AccessKind::StorageWrite),
        ],
        depends_on: vec![shade],
    });
    // Water is a real response producer, not a consumer of a completed
    // composition. It rasterizes after MainPass in Core3d, reads the resolved
    // scene color/depth for refraction and depth testing, and overwrites the
    // response planes for the pixels it covers. The resident composition pass
    // must therefore follow this producer.
    let water = graph.add_pass(PassDescriptor {
        name: "water_surface_raster".into(),
        queue: QueueClass::Graphics,
        accesses: vec![
            access(depth, AccessKind::DepthAttachment),
            access(scene_color, AccessKind::SampledRead),
            access(left, AccessKind::StorageWrite),
            access(right, AccessKind::StorageWrite),
            access(controls, AccessKind::StorageWrite),
            access(water_coverage, AccessKind::StorageWrite),
        ],
        depends_on: vec![resolve],
    });
    graph.add_pass(PassDescriptor {
        name: COMPOSITION_PASS_NAME.into(),
        queue: QueueClass::Compute,
        accesses: vec![
            access(left, AccessKind::StorageWrite),
            access(right, AccessKind::StorageRead),
            access(controls, AccessKind::StorageRead),
            access(auxiliary, AccessKind::StorageWrite),
            access(water_coverage, AccessKind::StorageRead),
            access(visibility_ids, AccessKind::SampledRead),
            access(visibility_metadata, AccessKind::SampledRead),
            access(scene_color, AccessKind::StorageWrite),
        ],
        depends_on: vec![water],
    });
    append_post_process_passes(&mut graph)
        .expect("generated shared post-process declarations must be valid");
    graph
}

fn resource(
    graph: &mut GpuFrameGraphBuilder,
    name: &'static str,
    lifetime: ResourceLifetime,
) -> prism_render_graph::frame_graph::ResourceId {
    graph.add_resource(ResourceDescriptor {
        name: name.into(),
        lifetime,
        size: 0,
        alignment: 16,
    })
}

#[cfg(test)]
#[path = "../../text/unit/shading/composition_graph.rs"]
mod composition_tests;

#[cfg(test)]
fn logical_resource_index(graph: &GpuFrameGraphBuilder, resource_index: usize) -> (bool, usize) {
    let is_texture = |index: usize| {
        graph.passes().iter().any(|pass| {
            pass.accesses.iter().any(|access| {
                access.resource.0 as usize == index
                    && matches!(
                        access.kind,
                        AccessKind::SampledRead
                            | AccessKind::ColorAttachment
                            | AccessKind::DepthAttachment
                    )
            })
        })
    };
    let texture = is_texture(resource_index);
    let local_index = graph
        .resources()
        .iter()
        .enumerate()
        .take(resource_index + 1)
        .filter(|(index, _)| is_texture(*index) == texture)
        .count()
        - 1;
    (texture, local_index)
}

fn access(
    resource: prism_render_graph::frame_graph::ResourceId,
    kind: AccessKind,
) -> ResourceAccess {
    ResourceAccess { resource, kind }
}

#[cfg(test)]
mod tests {
    use super::*;
    use prism_render_graph::{ExecutionPlan, LogicalResource};

    fn assert_specular_raw(
        graph: &GpuFrameGraphBuilder,
        compiled: &ExecutionPlan,
        resource_name: &str,
        producer: &str,
        consumer: &str,
    ) {
        let pass_index = |name: &str| {
            graph
                .passes()
                .iter()
                .position(|pass| pass.name == name)
                .unwrap()
        };
        let source = pass_index(producer);
        let destination = pass_index(consumer);
        let resource_index = graph
            .resources()
            .iter()
            .position(|resource| resource.name == resource_name)
            .unwrap();
        let (texture, local_index) = logical_resource_index(graph, resource_index);
        assert!(
            compiled.logical_barriers.iter().any(|barrier| {
                let matches_resource = match (texture, barrier.resource) {
                    (true, LogicalResource::Texture(id)) | (false, LogicalResource::Buffer(id)) => {
                        id.get() as usize == local_index
                    }
                    _ => false,
                };
                if !matches_resource
                    || barrier.source != source
                    || barrier.destination != destination
                {
                    return false;
                }
                let before = compiled.accesses[source][barrier.before];
                let after = compiled.accesses[destination][barrier.after];
                before.output_version == Some(after.input_version)
                    && after.input_version > 0
                    && after.output_version.is_none()
            }),
            "missing current-frame RAW: {producer} -> {consumer} on {resource_name}"
        );
    }

    #[test]
    fn specular_pipeline_graph_orders_current_frame_producers_and_consumers() {
        let graph = shading_frame_graph();
        let compiled = graph.compile().unwrap();
        let positions = [
            "shading_resolve",
            "water_surface_raster",
            COMPOSITION_PASS_NAME,
            "post_ssr",
            "post_spec_denoise",
            "post_ssgi",
            "post_taa",
        ]
        .map(|name| {
            let index = graph
                .passes()
                .iter()
                .position(|pass| pass.name == name)
                .unwrap();
            compiled.position_by_pass[index].expect("production pass must be live")
        });
        assert!(
            positions.windows(2).all(|pair| pair[0] < pair[1]),
            "{positions:?}"
        );
    }

    #[test]
    fn specular_pipeline_graph_tracks_real_raw_provenance() {
        let graph = shading_frame_graph();
        let compiled = graph.compile().unwrap();
        for name in [
            "post_ssr_depth",
            "post_ssr_normal_roughness",
            "post_ssr_trace",
            "post_ssr_hit_distance",
            "post_ssr_color_mips",
        ] {
            assert_specular_raw(&graph, &compiled, name, "post_ssr", "post_spec_denoise");
        }
        for name in ["post_ssr_env_specular", "post_ssr_spec_weight"] {
            assert_specular_raw(&graph, &compiled, name, "shading_resolve", "post_ssr");
        }
        assert_specular_raw(
            &graph,
            &compiled,
            "post_ssr_env_specular",
            "shading_resolve",
            "post_spec_denoise",
        );
        // SSGI must snapshot the specular fold, not the pre-fold colour pyramid.
        assert_specular_raw(
            &graph,
            &compiled,
            "hdr_scene_color",
            "post_spec_denoise",
            "post_ssgi",
        );
        assert_specular_raw(
            &graph,
            &compiled,
            "hdr_scene_color",
            "post_ssgi",
            "post_taa",
        );
    }

    #[test]
    fn declared_pass_count_matches_runtime_assertion() {
        let graph = shading_frame_graph();
        let compiled = graph.compile().unwrap();
        assert_eq!(compiled.order.len(), graph.passes().len());
        assert_eq!(compiled.declared_pass_count, graph.passes().len());
        assert_eq!(graph.passes()[0].name, "visibility_raster");
        assert_eq!(graph.passes()[4].name, "texel_space_shade_cache");
        assert_eq!(graph.passes()[5].name, "shading_resolve");
        assert_eq!(
            graph
                .resources()
                .iter()
                .filter(|resource| resource.name.starts_with("visibility_"))
                .count(),
            2
        );
        assert_eq!(
            graph.passes()[1].depends_on,
            [prism_render_graph::frame_graph::PassId(0)]
        );
        assert_eq!(
            graph.passes()[2].depends_on,
            [prism_render_graph::frame_graph::PassId(1)]
        );
        assert_eq!(
            graph.passes()[3].depends_on,
            [prism_render_graph::frame_graph::PassId(2)]
        );
        assert_eq!(
            graph.passes()[4].depends_on,
            [prism_render_graph::frame_graph::PassId(3)]
        );
        assert_eq!(
            graph.passes()[5].depends_on,
            [prism_render_graph::frame_graph::PassId(4)]
        );
        assert!(compiled
            .logical_barriers
            .iter()
            .any(|barrier| barrier.queue_transfer));
        let resource = |name: &str| {
            prism_render_graph::frame_graph::ResourceId(
                graph
                    .resources()
                    .iter()
                    .position(|resource| resource.name == name)
                    .unwrap() as u32,
            )
        };
        let has_barrier = |name: &str, source: u32, destination: u32| {
            let resource_index = resource(name).0 as usize;
            let (is_texture, local_index) = logical_resource_index(&graph, resource_index);
            compiled.logical_barriers.iter().any(|barrier| {
                let resource_matches = match (is_texture, barrier.resource) {
                    (true, prism_render_graph::LogicalResource::Texture(id))
                    | (false, prism_render_graph::LogicalResource::Buffer(id)) => {
                        id.get() as usize == local_index
                    }
                    _ => false,
                };
                resource_matches
                    && barrier.source == source as usize
                    && barrier.destination == destination as usize
            })
        };
        assert!(has_barrier("visibility_ids", 0, 1));
        assert!(has_barrier("material_class_counts", 1, 2));
        assert!(has_barrier("material_pixel_classification", 1, 3));
        assert!(has_barrier("material_class_offsets", 2, 3));
        assert!(has_barrier("material_class_cursors", 2, 3));
        assert!(has_barrier("material_shading_work", 3, 5));
        assert!(has_barrier("texel_space_shade_jobs", 3, 4));
        assert!(has_barrier("texel_space_shade_atlas", 4, 5));
        assert!(has_barrier("shading_indirect_dispatch", 2, 5));
        assert!(has_barrier("response_left", 5, 6));
        assert!(has_barrier("response_left", 6, 7));
        assert!(has_barrier("water_coverage", 6, 7));
        assert_eq!(graph.passes()[6].name, "water_surface_raster");
        assert_eq!(graph.passes()[7].name, COMPOSITION_PASS_NAME);
        assert_eq!(graph.passes()[8].name, "post_exposure");
        assert_eq!(graph.passes().last().unwrap().name, "post_upscale");
    }
}
