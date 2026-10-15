//! Generated declaration table for the shared post-processing chain.
//!
//! The runtime systems remain responsible for recording backend commands, but
//! topology, resource ownership, ABI identity, bind-group shape and pipeline
//! entry metadata are all emitted from this one table.  Every stylized and
//! physically based response therefore consumes the same HDR post base.

use prism_render_graph::frame_graph::{
    declare_passes, AccessDeclaration, AccessKind, BindGroupContract, CodegenError, GeneratedPass,
    GpuFrameGraphBuilder, PassSkeletonDeclaration, PipelineContract, QueueClass,
    ResourceDeclaration, ResourceLifetime, SkeletonAbi,
};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct PostExecutionBinding {
    pub(crate) declaration: &'static str,
    pub(crate) executors: &'static [&'static str],
}

const POST_RESOURCES: &[ResourceDeclaration] = &[ResourceDeclaration {
    name: "post_process_params",
    lifetime: ResourceLifetime::Persistent,
    size: 256,
    alignment: 16,
}];

const POST_ACCESSES: &[AccessDeclaration] = &[
    AccessDeclaration {
        resource: "hdr_scene_color",
        kind: AccessKind::SampledRead,
    },
    AccessDeclaration {
        resource: "post_process_params",
        kind: AccessKind::StorageRead,
    },
    AccessDeclaration {
        resource: "hdr_scene_color",
        kind: AccessKind::StorageWrite,
    },
];

const POST_BINDINGS: &[u8] = &[2, 1];

macro_rules! post_passes {
    ($(($name:literal, $hash:expr, $previous:expr)),+ $(,)?) => {
        &[
            $(PassSkeletonDeclaration {
                name: $name,
                queue: QueueClass::Compute,
                abi: SkeletonAbi { version: 1, hash: $hash },
                bind_groups: BindGroupContract {
                    group_count: 2,
                    bindings_per_group: POST_BINDINGS,
                },
                pipeline: PipelineContract {
                    entry_point: concat!("prism_post_", $name),
                    workgroup_size: [8, 8, 1],
                },
                resources: POST_RESOURCES,
                accesses: POST_ACCESSES,
                depends_on: &[$previous],
            }),+
        ]
    };
}

/// The complete fixed-topology post base.  Optional effects are data-gated;
/// disabling an effect produces a no-op dispatch without changing the graph.
pub(crate) const POST_PROCESS_DECLARATIONS: &[PassSkeletonDeclaration] = post_passes![
    ("post_exposure", 0x01, "response_composition"),
    ("post_bloom", 0x02, "post_exposure"),
    ("post_dof", 0x03, "post_bloom"),
    ("post_motion_blur", 0x04, "post_dof"),
    ("post_ssr", 0x05, "post_motion_blur"),
    ("post_ssgi", 0x06, "post_ssr"),
    ("post_spec_denoise", 0x07, "post_ssgi"),
    ("post_taa", 0x08, "post_spec_denoise"),
    ("post_posterize", 0x09, "post_taa"),
    ("post_outline", 0x0a, "post_posterize"),
    ("post_hatching", 0x0b, "post_outline"),
    ("post_halftone", 0x0c, "post_hatching"),
    ("post_kuwahara", 0x0d, "post_halftone"),
    ("post_color_grade", 0x0e, "post_kuwahara"),
    ("post_chromatic_aberration", 0x0f, "post_color_grade"),
    ("post_vignette", 0x10, "post_chromatic_aberration"),
    ("post_film_grain", 0x11, "post_vignette"),
    ("post_cas", 0x12, "post_film_grain"),
    ("post_gamut_map", 0x13, "post_cas"),
    ("post_ordered_dither", 0x14, "post_gamut_map"),
    ("post_lens_flare", 0x15, "post_ordered_dither"),
    ("post_tonemap", 0x16, "post_lens_flare"),
    ("post_upscale", 0x17, "post_tonemap"),
];

const EXEC_EXPOSURE: &[&str] = &["exposure_histogram_pass", "exposure_average_pass"];
const EXEC_BLOOM: &[&str] = &["bloom_pass"];
const EXEC_DOF: &[&str] = &["dof_pass"];
const EXEC_MOTION_BLUR: &[&str] = &["motion_blur_pass"];
const EXEC_SSR: &[&str] = &[
    "ssr_prepass_pass",
    "ssr_hzb_pass",
    "ssr_repack_pass",
    "ssr_color_mips_pass",
    "ssr_trace_pass",
    "ssr_reconstruct_pass",
    "ssr_temporal_pass",
    "ssr_composite_pass",
];
const EXEC_SSGI: &[&str] = &[
    "ssgi_trace_pass",
    "ssgi_denoise_pass",
    "ssgi_composite_pass",
];
const EXEC_SPEC_DENOISE: &[&str] = &[
    "spec_denoise_reproject_pass",
    "spec_denoise_history_clamp_pass",
    "spec_denoise_spatial_pass",
];
const EXEC_TAA: &[&str] = &["taa_resolve_pass"];

pub(crate) const POST_EXECUTION_BINDINGS: &[PostExecutionBinding] = &[
    PostExecutionBinding {
        declaration: "post_exposure",
        executors: EXEC_EXPOSURE,
    },
    PostExecutionBinding {
        declaration: "post_bloom",
        executors: EXEC_BLOOM,
    },
    PostExecutionBinding {
        declaration: "post_dof",
        executors: EXEC_DOF,
    },
    PostExecutionBinding {
        declaration: "post_motion_blur",
        executors: EXEC_MOTION_BLUR,
    },
    PostExecutionBinding {
        declaration: "post_ssr",
        executors: EXEC_SSR,
    },
    PostExecutionBinding {
        declaration: "post_ssgi",
        executors: EXEC_SSGI,
    },
    PostExecutionBinding {
        declaration: "post_spec_denoise",
        executors: EXEC_SPEC_DENOISE,
    },
    PostExecutionBinding {
        declaration: "post_taa",
        executors: EXEC_TAA,
    },
    PostExecutionBinding {
        declaration: "post_posterize",
        executors: &["posterize_pass"],
    },
    PostExecutionBinding {
        declaration: "post_outline",
        executors: &["outline_pass"],
    },
    PostExecutionBinding {
        declaration: "post_hatching",
        executors: &["hatching_pass"],
    },
    PostExecutionBinding {
        declaration: "post_halftone",
        executors: &["halftone_pass"],
    },
    PostExecutionBinding {
        declaration: "post_kuwahara",
        executors: &["kuwahara_pass"],
    },
    PostExecutionBinding {
        declaration: "post_color_grade",
        executors: &["color_grade_pass"],
    },
    PostExecutionBinding {
        declaration: "post_chromatic_aberration",
        executors: &["chromatic_aberration_pass"],
    },
    PostExecutionBinding {
        declaration: "post_vignette",
        executors: &["vignette_pass"],
    },
    PostExecutionBinding {
        declaration: "post_film_grain",
        executors: &["film_grain_pass"],
    },
    PostExecutionBinding {
        declaration: "post_cas",
        executors: &["cas_pass"],
    },
    PostExecutionBinding {
        declaration: "post_gamut_map",
        executors: &["gamut_map_pass"],
    },
    PostExecutionBinding {
        declaration: "post_ordered_dither",
        executors: &["ordered_dither_pass"],
    },
    PostExecutionBinding {
        declaration: "post_lens_flare",
        executors: &["lens_flare_pass"],
    },
    PostExecutionBinding {
        declaration: "post_tonemap",
        executors: &["tonemap_pass"],
    },
    PostExecutionBinding {
        declaration: "post_upscale",
        executors: &["upscale_pass"],
    },
];

pub(crate) fn validate_post_execution_bindings(graph: &GpuFrameGraphBuilder) -> Result<(), String> {
    let declarations = POST_PROCESS_DECLARATIONS
        .iter()
        .map(|pass| pass.name)
        .collect::<Vec<_>>();
    let bindings = POST_EXECUTION_BINDINGS
        .iter()
        .map(|binding| binding.declaration)
        .collect::<Vec<_>>();
    if declarations != bindings {
        return Err(format!(
            "post declaration/binding order mismatch: {declarations:?} != {bindings:?}"
        ));
    }
    if POST_EXECUTION_BINDINGS
        .iter()
        .any(|binding| binding.executors.is_empty())
    {
        return Err("post execution ledger contains an empty executor chain".into());
    }
    let unique = POST_EXECUTION_BINDINGS
        .iter()
        .flat_map(|binding| binding.executors.iter())
        .collect::<std::collections::BTreeSet<_>>()
        .len();
    let total = POST_EXECUTION_BINDINGS
        .iter()
        .map(|binding| binding.executors.len())
        .sum::<usize>();
    if unique != total {
        return Err("post execution ledger reuses an executor".into());
    }
    let graph_suffix = graph
        .passes()
        .iter()
        .skip_while(|pass| pass.name != "post_exposure")
        .map(|pass| pass.name.as_ref())
        .collect::<Vec<_>>();
    if graph_suffix != declarations {
        return Err(format!(
            "post graph suffix mismatch: {graph_suffix:?} != {declarations:?}"
        ));
    }
    Ok(())
}

/// Emits every shared post-process pass into the authoritative graph.
pub(crate) fn append_post_process_passes(
    graph: &mut GpuFrameGraphBuilder,
) -> Result<Vec<GeneratedPass>, CodegenError> {
    declare_passes(graph, POST_PROCESS_DECLARATIONS)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_shared_post_pass_is_generated_from_one_table() {
        assert!(POST_PROCESS_DECLARATIONS.len() >= 20);
        assert!(POST_PROCESS_DECLARATIONS
            .windows(2)
            .all(|pair| pair[1].depends_on == [pair[0].name]));
        assert!(POST_PROCESS_DECLARATIONS
            .iter()
            .all(|pass| pass.abi.version != 0
                && pass.bind_groups.group_count as usize
                    == pass.bind_groups.bindings_per_group.len()
                && pass.pipeline.entry_point.starts_with("prism_post_")));
    }

    #[test]
    fn every_declared_post_pass_has_one_unique_executor_chain() {
        let graph = crate::shading::graph::shading_frame_graph();
        validate_post_execution_bindings(&graph).unwrap();
    }
}
