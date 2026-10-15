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

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub(crate) enum PostExecutorKind {
    ExposureHistogram,
    ExposureAverage,
    Bloom,
    Dof,
    MotionBlur,
    SsrPrepass,
    SsrHzb,
    SsrRepack,
    SsrColorMips,
    SsrTrace,
    SsrReconstruct,
    SsrTemporal,
    SsrComposite,
    SsgiTrace,
    SsgiDenoise,
    SsgiComposite,
    SpecDenoiseReproject,
    SpecDenoiseHistoryClamp,
    SpecDenoiseSpatial,
    TaaResolve,
    Posterize,
    Outline,
    Hatching,
    Halftone,
    Kuwahara,
    ColorGrade,
    ChromaticAberration,
    Vignette,
    FilmGrain,
    Cas,
    GamutMap,
    OrderedDither,
    LensFlare,
    Tonemap,
    Upscale,
}

impl PostExecutorKind {
    pub(crate) const fn name(self) -> &'static str {
        match self {
            Self::ExposureHistogram => "exposure_histogram_pass",
            Self::ExposureAverage => "exposure_average_pass",
            Self::Bloom => "bloom_pass",
            Self::Dof => "dof_pass",
            Self::MotionBlur => "motion_blur_pass",
            Self::SsrPrepass => "ssr_prepass_pass",
            Self::SsrHzb => "ssr_hzb_pass",
            Self::SsrRepack => "ssr_repack_pass",
            Self::SsrColorMips => "ssr_color_mips_pass",
            Self::SsrTrace => "ssr_trace_pass",
            Self::SsrReconstruct => "ssr_reconstruct_pass",
            Self::SsrTemporal => "ssr_temporal_pass",
            Self::SsrComposite => "ssr_composite_pass",
            Self::SsgiTrace => "ssgi_trace_pass",
            Self::SsgiDenoise => "ssgi_denoise_pass",
            Self::SsgiComposite => "ssgi_composite_pass",
            Self::SpecDenoiseReproject => "spec_denoise_reproject_pass",
            Self::SpecDenoiseHistoryClamp => "spec_denoise_history_clamp_pass",
            Self::SpecDenoiseSpatial => "spec_denoise_spatial_pass",
            Self::TaaResolve => "taa_resolve_pass",
            Self::Posterize => "posterize_pass",
            Self::Outline => "outline_pass",
            Self::Hatching => "hatching_pass",
            Self::Halftone => "halftone_pass",
            Self::Kuwahara => "kuwahara_pass",
            Self::ColorGrade => "color_grade_pass",
            Self::ChromaticAberration => "chromatic_aberration_pass",
            Self::Vignette => "vignette_pass",
            Self::FilmGrain => "film_grain_pass",
            Self::Cas => "cas_pass",
            Self::GamutMap => "gamut_map_pass",
            Self::OrderedDither => "ordered_dither_pass",
            Self::LensFlare => "lens_flare_pass",
            Self::Tonemap => "tonemap_pass",
            Self::Upscale => "upscale_pass",
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct PostExecutionBinding {
    /// The concrete `shading` module that owns the effect.
    pub(crate) module: &'static str,
    pub(crate) declaration: &'static str,
    /// Tokens that must occur in the production plugin's registration source.
    /// The test-only repository audit checks these against `plugin.rs`; the
    /// runtime gate checks the typed executor chain below.
    pub(crate) registration_markers: &'static [&'static str],
    pub(crate) executors: &'static [PostExecutorKind],
}

// The declaration frontend has one graph-wide resource table, while each pass
// owns its precise access set below.  Extent-dependent textures intentionally
// use a zero logical footprint: their real dimensions are admitted by the
// per-view resource owners, not guessed at startup by this topology graph.
const POST_RESOURCES: &[ResourceDeclaration] = &[
    ResourceDeclaration {
        name: "post_process_params",
        lifetime: ResourceLifetime::Persistent,
        size: 256,
        alignment: 16,
    },
    ResourceDeclaration {
        name: "post_exposure_histogram",
        lifetime: ResourceLifetime::Persistent,
        size: 256,
        alignment: 16,
    },
    ResourceDeclaration {
        name: "post_exposure_state",
        lifetime: ResourceLifetime::Persistent,
        size: 64,
        alignment: 16,
    },
    ResourceDeclaration {
        name: "post_bloom_base",
        lifetime: ResourceLifetime::Transient,
        size: 0,
        alignment: 16,
    },
    ResourceDeclaration {
        name: "post_bloom_down",
        lifetime: ResourceLifetime::Transient,
        size: 0,
        alignment: 16,
    },
    ResourceDeclaration {
        name: "post_bloom_up",
        lifetime: ResourceLifetime::Transient,
        size: 0,
        alignment: 16,
    },
    ResourceDeclaration {
        name: "post_dof_coc",
        lifetime: ResourceLifetime::Transient,
        size: 0,
        alignment: 16,
    },
    ResourceDeclaration {
        name: "post_dof_blurred",
        lifetime: ResourceLifetime::Transient,
        size: 0,
        alignment: 16,
    },
    ResourceDeclaration {
        name: "post_dof_output",
        lifetime: ResourceLifetime::Transient,
        size: 0,
        alignment: 16,
    },
    ResourceDeclaration {
        name: "post_motion_tile_max",
        lifetime: ResourceLifetime::Transient,
        size: 0,
        alignment: 16,
    },
    ResourceDeclaration {
        name: "post_motion_neighbor_max",
        lifetime: ResourceLifetime::Transient,
        size: 0,
        alignment: 16,
    },
    ResourceDeclaration {
        name: "post_motion_output",
        lifetime: ResourceLifetime::Transient,
        size: 0,
        alignment: 16,
    },
    ResourceDeclaration {
        name: "post_ssr_depth",
        lifetime: ResourceLifetime::Transient,
        size: 0,
        alignment: 16,
    },
    ResourceDeclaration {
        name: "post_ssr_normal_roughness",
        lifetime: ResourceLifetime::Transient,
        size: 0,
        alignment: 16,
    },
    ResourceDeclaration {
        name: "post_ssr_hzb",
        lifetime: ResourceLifetime::Transient,
        size: 0,
        alignment: 16,
    },
    ResourceDeclaration {
        name: "post_ssr_color_mips",
        lifetime: ResourceLifetime::Transient,
        size: 0,
        alignment: 16,
    },
    ResourceDeclaration {
        name: "post_ssr_trace",
        lifetime: ResourceLifetime::Transient,
        size: 0,
        alignment: 16,
    },
    ResourceDeclaration {
        name: "post_ssr_resolved",
        lifetime: ResourceLifetime::Transient,
        size: 0,
        alignment: 16,
    },
    ResourceDeclaration {
        name: "post_ssr_hit_distance",
        lifetime: ResourceLifetime::Transient,
        size: 0,
        alignment: 16,
    },
    ResourceDeclaration {
        name: "post_ssr_env_specular",
        lifetime: ResourceLifetime::Transient,
        size: 0,
        alignment: 16,
    },
    ResourceDeclaration {
        name: "post_ssr_spec_weight",
        lifetime: ResourceLifetime::Transient,
        size: 0,
        alignment: 16,
    },
    ResourceDeclaration {
        name: "post_ssgi_output",
        lifetime: ResourceLifetime::Transient,
        size: 0,
        alignment: 16,
    },
    ResourceDeclaration {
        name: "post_ssgi_denoised",
        lifetime: ResourceLifetime::Transient,
        size: 0,
        alignment: 16,
    },
    ResourceDeclaration {
        name: "post_spec_history",
        lifetime: ResourceLifetime::Persistent,
        size: 0,
        alignment: 16,
    },
    ResourceDeclaration {
        name: "post_spec_reprojected",
        lifetime: ResourceLifetime::Transient,
        size: 0,
        alignment: 16,
    },
    ResourceDeclaration {
        name: "post_spec_denoised",
        lifetime: ResourceLifetime::Transient,
        size: 0,
        alignment: 16,
    },
    ResourceDeclaration {
        name: "post_taa_history",
        lifetime: ResourceLifetime::Persistent,
        size: 0,
        alignment: 16,
    },
    ResourceDeclaration {
        name: "post_taa_output",
        lifetime: ResourceLifetime::Transient,
        size: 0,
        alignment: 16,
    },
    ResourceDeclaration {
        name: "post_tonemap_output",
        lifetime: ResourceLifetime::Transient,
        size: 0,
        alignment: 16,
    },
    ResourceDeclaration {
        name: "post_upscale_color",
        lifetime: ResourceLifetime::Transient,
        size: 0,
        alignment: 16,
    },
    ResourceDeclaration {
        name: "post_upscale_output",
        lifetime: ResourceLifetime::Transient,
        size: 0,
        alignment: 16,
    },
];

const ACCESS_EXPOSURE: &[AccessDeclaration] = &[
    AccessDeclaration {
        resource: "hdr_scene_color",
        kind: AccessKind::SampledRead,
    },
    AccessDeclaration {
        resource: "post_process_params",
        kind: AccessKind::StorageRead,
    },
    AccessDeclaration {
        resource: "post_exposure_histogram",
        kind: AccessKind::StorageWrite,
    },
    AccessDeclaration {
        resource: "post_exposure_histogram",
        kind: AccessKind::StorageRead,
    },
    AccessDeclaration {
        resource: "post_exposure_state",
        kind: AccessKind::StorageWrite,
    },
];
const ACCESS_BLOOM: &[AccessDeclaration] = &[
    AccessDeclaration {
        resource: "hdr_scene_color",
        kind: AccessKind::SampledRead,
    },
    AccessDeclaration {
        resource: "post_process_params",
        kind: AccessKind::StorageRead,
    },
    AccessDeclaration {
        resource: "post_bloom_base",
        kind: AccessKind::StorageWrite,
    },
    AccessDeclaration {
        resource: "post_bloom_down",
        kind: AccessKind::StorageWrite,
    },
    AccessDeclaration {
        resource: "post_bloom_down",
        kind: AccessKind::StorageRead,
    },
    AccessDeclaration {
        resource: "post_bloom_up",
        kind: AccessKind::StorageWrite,
    },
    AccessDeclaration {
        resource: "post_bloom_up",
        kind: AccessKind::StorageRead,
    },
    AccessDeclaration {
        resource: "hdr_scene_color",
        kind: AccessKind::StorageWrite,
    },
];
const ACCESS_DOF: &[AccessDeclaration] = &[
    AccessDeclaration {
        resource: "main_depth",
        kind: AccessKind::SampledRead,
    },
    AccessDeclaration {
        resource: "hdr_scene_color",
        kind: AccessKind::SampledRead,
    },
    AccessDeclaration {
        resource: "post_process_params",
        kind: AccessKind::StorageRead,
    },
    AccessDeclaration {
        resource: "post_dof_coc",
        kind: AccessKind::StorageWrite,
    },
    AccessDeclaration {
        resource: "post_dof_coc",
        kind: AccessKind::SampledRead,
    },
    AccessDeclaration {
        resource: "post_dof_blurred",
        kind: AccessKind::StorageWrite,
    },
    AccessDeclaration {
        resource: "post_dof_blurred",
        kind: AccessKind::SampledRead,
    },
    AccessDeclaration {
        resource: "post_dof_output",
        kind: AccessKind::StorageWrite,
    },
    AccessDeclaration {
        resource: "hdr_scene_color",
        kind: AccessKind::TransferWrite,
    },
];
const ACCESS_MOTION_BLUR: &[AccessDeclaration] = &[
    AccessDeclaration {
        resource: "visibility_metadata",
        kind: AccessKind::SampledRead,
    },
    AccessDeclaration {
        resource: "post_ssr_depth",
        kind: AccessKind::SampledRead,
    },
    AccessDeclaration {
        resource: "hdr_scene_color",
        kind: AccessKind::SampledRead,
    },
    AccessDeclaration {
        resource: "post_process_params",
        kind: AccessKind::StorageRead,
    },
    AccessDeclaration {
        resource: "post_motion_tile_max",
        kind: AccessKind::StorageWrite,
    },
    AccessDeclaration {
        resource: "post_motion_tile_max",
        kind: AccessKind::SampledRead,
    },
    AccessDeclaration {
        resource: "post_motion_neighbor_max",
        kind: AccessKind::StorageWrite,
    },
    AccessDeclaration {
        resource: "post_motion_neighbor_max",
        kind: AccessKind::SampledRead,
    },
    AccessDeclaration {
        resource: "post_motion_output",
        kind: AccessKind::StorageWrite,
    },
    AccessDeclaration {
        resource: "hdr_scene_color",
        kind: AccessKind::TransferWrite,
    },
];
const ACCESS_SSR: &[AccessDeclaration] = &[
    AccessDeclaration {
        resource: "main_depth",
        kind: AccessKind::SampledRead,
    },
    AccessDeclaration {
        resource: "visibility_metadata",
        kind: AccessKind::SampledRead,
    },
    AccessDeclaration {
        resource: "hdr_scene_color",
        kind: AccessKind::SampledRead,
    },
    AccessDeclaration {
        resource: "post_process_params",
        kind: AccessKind::StorageRead,
    },
    AccessDeclaration {
        resource: "post_ssr_depth",
        kind: AccessKind::StorageWrite,
    },
    AccessDeclaration {
        resource: "post_ssr_normal_roughness",
        kind: AccessKind::StorageWrite,
    },
    AccessDeclaration {
        resource: "post_ssr_hzb",
        kind: AccessKind::StorageWrite,
    },
    AccessDeclaration {
        resource: "post_ssr_hzb",
        kind: AccessKind::SampledRead,
    },
    AccessDeclaration {
        resource: "post_ssr_color_mips",
        kind: AccessKind::StorageWrite,
    },
    AccessDeclaration {
        resource: "post_ssr_color_mips",
        kind: AccessKind::SampledRead,
    },
    AccessDeclaration {
        resource: "post_ssr_trace",
        kind: AccessKind::StorageWrite,
    },
    AccessDeclaration {
        resource: "post_ssr_trace",
        kind: AccessKind::SampledRead,
    },
    AccessDeclaration {
        resource: "post_ssr_resolved",
        kind: AccessKind::StorageWrite,
    },
    AccessDeclaration {
        resource: "post_ssr_resolved",
        kind: AccessKind::SampledRead,
    },
    AccessDeclaration {
        resource: "post_ssr_hit_distance",
        kind: AccessKind::StorageWrite,
    },
    AccessDeclaration {
        resource: "post_ssr_hit_distance",
        kind: AccessKind::SampledRead,
    },
    AccessDeclaration {
        resource: "post_ssr_env_specular",
        kind: AccessKind::StorageWrite,
    },
    AccessDeclaration {
        resource: "post_ssr_spec_weight",
        kind: AccessKind::StorageWrite,
    },
    AccessDeclaration {
        resource: "hdr_scene_color",
        kind: AccessKind::StorageWrite,
    },
];
const ACCESS_SSGI: &[AccessDeclaration] = &[
    AccessDeclaration {
        resource: "post_ssr_hzb",
        kind: AccessKind::SampledRead,
    },
    AccessDeclaration {
        resource: "post_ssr_depth",
        kind: AccessKind::SampledRead,
    },
    AccessDeclaration {
        resource: "post_ssr_normal_roughness",
        kind: AccessKind::SampledRead,
    },
    AccessDeclaration {
        resource: "post_ssr_color_mips",
        kind: AccessKind::SampledRead,
    },
    AccessDeclaration {
        resource: "post_ssgi_output",
        kind: AccessKind::StorageWrite,
    },
    AccessDeclaration {
        resource: "post_ssgi_output",
        kind: AccessKind::SampledRead,
    },
    AccessDeclaration {
        resource: "post_ssgi_denoised",
        kind: AccessKind::StorageWrite,
    },
    AccessDeclaration {
        resource: "post_ssgi_denoised",
        kind: AccessKind::SampledRead,
    },
    AccessDeclaration {
        resource: "hdr_scene_color",
        kind: AccessKind::StorageWrite,
    },
];
const ACCESS_SPEC_DENOISE: &[AccessDeclaration] = &[
    AccessDeclaration {
        resource: "post_ssr_trace",
        kind: AccessKind::SampledRead,
    },
    AccessDeclaration {
        resource: "post_ssr_resolved",
        kind: AccessKind::SampledRead,
    },
    AccessDeclaration {
        resource: "post_ssr_hit_distance",
        kind: AccessKind::SampledRead,
    },
    AccessDeclaration {
        resource: "post_spec_history",
        kind: AccessKind::StorageRead,
    },
    AccessDeclaration {
        resource: "post_spec_history",
        kind: AccessKind::StorageWrite,
    },
    AccessDeclaration {
        resource: "post_spec_reprojected",
        kind: AccessKind::StorageWrite,
    },
    AccessDeclaration {
        resource: "post_spec_reprojected",
        kind: AccessKind::SampledRead,
    },
    AccessDeclaration {
        resource: "post_spec_denoised",
        kind: AccessKind::StorageWrite,
    },
    AccessDeclaration {
        resource: "post_spec_denoised",
        kind: AccessKind::SampledRead,
    },
];
const ACCESS_TAA: &[AccessDeclaration] = &[
    AccessDeclaration {
        resource: "hdr_scene_color",
        kind: AccessKind::SampledRead,
    },
    AccessDeclaration {
        resource: "visibility_metadata",
        kind: AccessKind::SampledRead,
    },
    AccessDeclaration {
        resource: "post_taa_history",
        kind: AccessKind::StorageRead,
    },
    AccessDeclaration {
        resource: "post_taa_history",
        kind: AccessKind::StorageWrite,
    },
    AccessDeclaration {
        resource: "post_taa_output",
        kind: AccessKind::StorageWrite,
    },
];
const ACCESS_SIMPLE: &[AccessDeclaration] = &[
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
const ACCESS_OUTLINE: &[AccessDeclaration] = &[
    AccessDeclaration {
        resource: "visibility_ids",
        kind: AccessKind::SampledRead,
    },
    AccessDeclaration {
        resource: "main_depth",
        kind: AccessKind::SampledRead,
    },
    AccessDeclaration {
        resource: "post_process_params",
        kind: AccessKind::StorageRead,
    },
    AccessDeclaration {
        resource: "hdr_scene_color",
        kind: AccessKind::SampledRead,
    },
    AccessDeclaration {
        resource: "hdr_scene_color",
        kind: AccessKind::StorageWrite,
    },
];
const ACCESS_TONEMAP: &[AccessDeclaration] = &[
    AccessDeclaration {
        resource: "hdr_scene_color",
        kind: AccessKind::SampledRead,
    },
    AccessDeclaration {
        resource: "post_process_params",
        kind: AccessKind::StorageRead,
    },
    AccessDeclaration {
        resource: "post_tonemap_output",
        kind: AccessKind::StorageWrite,
    },
    AccessDeclaration {
        resource: "hdr_scene_color",
        kind: AccessKind::TransferWrite,
    },
];
const ACCESS_UPSCALE: &[AccessDeclaration] = &[
    AccessDeclaration {
        resource: "hdr_scene_color",
        kind: AccessKind::SampledRead,
    },
    AccessDeclaration {
        resource: "post_taa_history",
        kind: AccessKind::StorageRead,
    },
    AccessDeclaration {
        resource: "post_taa_history",
        kind: AccessKind::StorageWrite,
    },
    AccessDeclaration {
        resource: "post_upscale_color",
        kind: AccessKind::StorageWrite,
    },
    AccessDeclaration {
        resource: "post_upscale_color",
        kind: AccessKind::SampledRead,
    },
    AccessDeclaration {
        resource: "post_upscale_output",
        kind: AccessKind::StorageWrite,
    },
];

const POST_BINDINGS: &[u8] = &[2, 1];

macro_rules! post_passes {
    ($(($name:literal, $hash:expr, $previous:expr, $accesses:expr)),+ $(,)?) => {
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
                accesses: $accesses,
                depends_on: &[$previous],
            }),+
        ]
    };
}

/// The complete fixed-topology post base.  Optional effects are data-gated;
/// disabling an effect produces a no-op dispatch without changing the graph.
pub(crate) const POST_PROCESS_DECLARATIONS: &[PassSkeletonDeclaration] = post_passes![
    (
        "post_exposure",
        0x01,
        "response_composition",
        ACCESS_EXPOSURE
    ),
    ("post_bloom", 0x02, "post_exposure", ACCESS_BLOOM),
    ("post_dof", 0x03, "post_bloom", ACCESS_DOF),
    ("post_motion_blur", 0x04, "post_dof", ACCESS_MOTION_BLUR),
    ("post_ssr", 0x05, "post_motion_blur", ACCESS_SSR),
    ("post_ssgi", 0x06, "post_ssr", ACCESS_SSGI),
    ("post_spec_denoise", 0x07, "post_ssgi", ACCESS_SPEC_DENOISE),
    ("post_taa", 0x08, "post_spec_denoise", ACCESS_TAA),
    ("post_posterize", 0x09, "post_taa", ACCESS_SIMPLE),
    ("post_outline", 0x0a, "post_posterize", ACCESS_OUTLINE),
    ("post_hatching", 0x0b, "post_outline", ACCESS_SIMPLE),
    ("post_halftone", 0x0c, "post_hatching", ACCESS_SIMPLE),
    ("post_kuwahara", 0x0d, "post_halftone", ACCESS_SIMPLE),
    ("post_color_grade", 0x0e, "post_kuwahara", ACCESS_SIMPLE),
    (
        "post_chromatic_aberration",
        0x0f,
        "post_color_grade",
        ACCESS_SIMPLE
    ),
    (
        "post_vignette",
        0x10,
        "post_chromatic_aberration",
        ACCESS_SIMPLE
    ),
    ("post_film_grain", 0x11, "post_vignette", ACCESS_SIMPLE),
    ("post_cas", 0x12, "post_film_grain", ACCESS_SIMPLE),
    ("post_gamut_map", 0x13, "post_cas", ACCESS_SIMPLE),
    ("post_ordered_dither", 0x14, "post_gamut_map", ACCESS_SIMPLE),
    (
        "post_lens_flare",
        0x15,
        "post_ordered_dither",
        ACCESS_SIMPLE
    ),
    ("post_tonemap", 0x16, "post_lens_flare", ACCESS_TONEMAP),
    ("post_upscale", 0x17, "post_tonemap", ACCESS_UPSCALE),
];

const EXEC_EXPOSURE: &[PostExecutorKind] = &[
    PostExecutorKind::ExposureHistogram,
    PostExecutorKind::ExposureAverage,
];
const EXEC_BLOOM: &[PostExecutorKind] = &[PostExecutorKind::Bloom];
const EXEC_DOF: &[PostExecutorKind] = &[PostExecutorKind::Dof];
const EXEC_MOTION_BLUR: &[PostExecutorKind] = &[PostExecutorKind::MotionBlur];
const EXEC_SSR: &[PostExecutorKind] = &[
    PostExecutorKind::SsrPrepass,
    PostExecutorKind::SsrHzb,
    PostExecutorKind::SsrRepack,
    PostExecutorKind::SsrColorMips,
    PostExecutorKind::SsrTrace,
    PostExecutorKind::SsrReconstruct,
    PostExecutorKind::SsrTemporal,
    PostExecutorKind::SsrComposite,
];
const EXEC_SSGI: &[PostExecutorKind] = &[
    PostExecutorKind::SsgiTrace,
    PostExecutorKind::SsgiDenoise,
    PostExecutorKind::SsgiComposite,
];
const EXEC_SPEC_DENOISE: &[PostExecutorKind] = &[
    PostExecutorKind::SpecDenoiseReproject,
    PostExecutorKind::SpecDenoiseHistoryClamp,
    PostExecutorKind::SpecDenoiseSpatial,
];
const EXEC_TAA: &[PostExecutorKind] = &[PostExecutorKind::TaaResolve];

const REG_EXPOSURE: &[&str] = &[
    "init_exposure_histogram_pipeline",
    "init_exposure_average_pipeline",
    "prepare_exposure_buffers",
    "prepare_exposure_histogram_bind_groups",
    "prepare_exposure_average_bind_groups",
];
const REG_BLOOM: &[&str] = &[
    "init_bloom_pipelines",
    "prepare_bloom_textures",
    "prepare_bloom_bind_groups",
];
const REG_DOF: &[&str] = &[
    "init_dof_pipeline",
    "prepare_dof_textures",
    "prepare_dof_bind_groups",
];
const REG_MOTION_BLUR: &[&str] = &[
    "init_motion_blur_pipeline",
    "prepare_motion_blur_textures",
    "prepare_motion_blur_bind_groups",
];
const REG_SSR: &[&str] = &[
    "init_ssr_prepass_pipeline",
    "init_ssr_hzb_pipeline",
    "init_ssr_repack_pipeline",
    "init_ssr_color_mips_pipeline",
    "init_ssr_trace_pipeline",
    "init_ssr_reconstruct_pipeline",
    "init_ssr_temporal_pipeline",
    "init_ssr_composite_pipeline",
    "prepare_ssr_textures",
    "prepare_ssr_temporal_textures",
    "prepare_ssr_prepass_bind_groups",
    "prepare_ssr_hzb_bind_groups",
    "prepare_ssr_repack_bind_groups",
    "prepare_ssr_color_mips_bind_groups",
    "prepare_ssr_trace_bind_groups",
    "prepare_ssr_reconstruct_bind_groups",
    "prepare_ssr_temporal_bind_groups",
    "prepare_ssr_composite_bind_groups",
];
const REG_SSGI: &[&str] = &[
    "init_ssgi_trace_pipeline",
    "init_ssgi_denoise_pipeline",
    "init_ssgi_composite_pipeline",
    "prepare_ssgi_textures",
    "prepare_ssgi_trace_bind_groups",
    "prepare_ssgi_denoise_bind_groups",
    "prepare_ssgi_composite_bind_groups",
];
const REG_SPEC_DENOISE: &[&str] = &[
    "init_spec_denoise_spatial_pipeline",
    "init_spec_denoise_reproject_pipeline",
    "init_spec_denoise_history_clamp_pipeline",
    "prepare_spec_denoise_resources",
    "prepare_spec_denoise_temporal_resources",
    "prepare_spec_denoise_bind_groups",
    "prepare_spec_denoise_reproject_bind_groups",
    "prepare_spec_denoise_history_clamp_bind_groups",
];
const REG_TAA: &[&str] = &[
    "init_taa_resolve_pipeline",
    "prepare_taa_textures",
    "prepare_taa_bind_groups",
    "prepare_taa_jitter",
];
const REG_POSTERIZE: &[&str] = &[
    "init_posterize_pipeline",
    "prepare_posterize_textures",
    "prepare_posterize_bind_groups",
];
const REG_OUTLINE: &[&str] = &[
    "init_outline_pipeline",
    "prepare_outline_textures",
    "prepare_outline_bind_groups",
];
const REG_HATCHING: &[&str] = &[
    "init_hatching_pipeline",
    "prepare_hatching_textures",
    "prepare_hatching_bind_groups",
];
const REG_HALFTONE: &[&str] = &[
    "init_halftone_pipeline",
    "prepare_halftone_textures",
    "prepare_halftone_bind_groups",
];
const REG_KUWAHARA: &[&str] = &[
    "init_kuwahara_pipeline",
    "prepare_kuwahara_textures",
    "prepare_kuwahara_bind_groups",
];
const REG_COLOR_GRADE: &[&str] = &[
    "init_color_grade_pipeline",
    "prepare_color_grade_textures",
    "prepare_color_grade_bind_groups",
];
const REG_CHROMATIC_ABERRATION: &[&str] = &[
    "init_chromatic_aberration_pipeline",
    "prepare_chromatic_aberration_textures",
    "prepare_chromatic_aberration_bind_groups",
];
const REG_VIGNETTE: &[&str] = &[
    "init_vignette_pipeline",
    "prepare_vignette_textures",
    "prepare_vignette_bind_groups",
];
const REG_FILM_GRAIN: &[&str] = &[
    "init_film_grain_pipeline",
    "prepare_film_grain_textures",
    "prepare_film_grain_bind_groups",
];
const REG_CAS: &[&str] = &[
    "init_cas_pipeline",
    "prepare_cas_textures",
    "prepare_cas_bind_groups",
];
const REG_GAMUT_MAP: &[&str] = &[
    "init_gamut_map_pipeline",
    "prepare_gamut_map_textures",
    "prepare_gamut_map_bind_groups",
];
const REG_ORDERED_DITHER: &[&str] = &[
    "init_ordered_dither_pipeline",
    "prepare_ordered_dither_textures",
    "prepare_ordered_dither_bind_groups",
];
const REG_LENS_FLARE: &[&str] = &[
    "init_lens_flare_pipeline",
    "prepare_lens_flare_textures",
    "prepare_lens_flare_bind_groups",
];
const REG_TONEMAP: &[&str] = &[
    "init_tonemap_pipeline",
    "prepare_tonemap_textures",
    "prepare_tonemap_bind_groups",
];
const REG_UPSCALE: &[&str] = &[
    "init_upscale_pipeline",
    "prepare_upscale_textures",
    "prepare_upscale_bind_groups",
];

pub(crate) const POST_EXECUTION_BINDINGS: &[PostExecutionBinding] = &[
    PostExecutionBinding {
        module: "exposure",
        declaration: "post_exposure",
        registration_markers: REG_EXPOSURE,
        executors: EXEC_EXPOSURE,
    },
    PostExecutionBinding {
        module: "bloom",
        declaration: "post_bloom",
        registration_markers: REG_BLOOM,
        executors: EXEC_BLOOM,
    },
    PostExecutionBinding {
        module: "dof",
        declaration: "post_dof",
        registration_markers: REG_DOF,
        executors: EXEC_DOF,
    },
    PostExecutionBinding {
        module: "motion_blur",
        declaration: "post_motion_blur",
        registration_markers: REG_MOTION_BLUR,
        executors: EXEC_MOTION_BLUR,
    },
    PostExecutionBinding {
        module: "ssr",
        declaration: "post_ssr",
        registration_markers: REG_SSR,
        executors: EXEC_SSR,
    },
    PostExecutionBinding {
        module: "ssgi",
        declaration: "post_ssgi",
        registration_markers: REG_SSGI,
        executors: EXEC_SSGI,
    },
    PostExecutionBinding {
        module: "spec_denoise",
        declaration: "post_spec_denoise",
        registration_markers: REG_SPEC_DENOISE,
        executors: EXEC_SPEC_DENOISE,
    },
    PostExecutionBinding {
        module: "taa",
        declaration: "post_taa",
        registration_markers: REG_TAA,
        executors: EXEC_TAA,
    },
    PostExecutionBinding {
        module: "posterize",
        declaration: "post_posterize",
        registration_markers: REG_POSTERIZE,
        executors: &[PostExecutorKind::Posterize],
    },
    PostExecutionBinding {
        module: "outline",
        declaration: "post_outline",
        registration_markers: REG_OUTLINE,
        executors: &[PostExecutorKind::Outline],
    },
    PostExecutionBinding {
        module: "hatching",
        declaration: "post_hatching",
        registration_markers: REG_HATCHING,
        executors: &[PostExecutorKind::Hatching],
    },
    PostExecutionBinding {
        module: "halftone",
        declaration: "post_halftone",
        registration_markers: REG_HALFTONE,
        executors: &[PostExecutorKind::Halftone],
    },
    PostExecutionBinding {
        module: "kuwahara",
        declaration: "post_kuwahara",
        registration_markers: REG_KUWAHARA,
        executors: &[PostExecutorKind::Kuwahara],
    },
    PostExecutionBinding {
        module: "color_grade",
        declaration: "post_color_grade",
        registration_markers: REG_COLOR_GRADE,
        executors: &[PostExecutorKind::ColorGrade],
    },
    PostExecutionBinding {
        module: "chromatic_aberration",
        declaration: "post_chromatic_aberration",
        registration_markers: REG_CHROMATIC_ABERRATION,
        executors: &[PostExecutorKind::ChromaticAberration],
    },
    PostExecutionBinding {
        module: "vignette",
        declaration: "post_vignette",
        registration_markers: REG_VIGNETTE,
        executors: &[PostExecutorKind::Vignette],
    },
    PostExecutionBinding {
        module: "film_grain",
        declaration: "post_film_grain",
        registration_markers: REG_FILM_GRAIN,
        executors: &[PostExecutorKind::FilmGrain],
    },
    PostExecutionBinding {
        module: "cas",
        declaration: "post_cas",
        registration_markers: REG_CAS,
        executors: &[PostExecutorKind::Cas],
    },
    PostExecutionBinding {
        module: "gamut_map",
        declaration: "post_gamut_map",
        registration_markers: REG_GAMUT_MAP,
        executors: &[PostExecutorKind::GamutMap],
    },
    PostExecutionBinding {
        module: "ordered_dither",
        declaration: "post_ordered_dither",
        registration_markers: REG_ORDERED_DITHER,
        executors: &[PostExecutorKind::OrderedDither],
    },
    PostExecutionBinding {
        module: "lens_flare",
        declaration: "post_lens_flare",
        registration_markers: REG_LENS_FLARE,
        executors: &[PostExecutorKind::LensFlare],
    },
    PostExecutionBinding {
        module: "tonemap",
        declaration: "post_tonemap",
        registration_markers: REG_TONEMAP,
        executors: &[PostExecutorKind::Tonemap],
    },
    PostExecutionBinding {
        module: "upscale",
        declaration: "post_upscale",
        registration_markers: REG_UPSCALE,
        executors: &[PostExecutorKind::Upscale],
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
    let unique_modules = POST_EXECUTION_BINDINGS
        .iter()
        .map(|binding| binding.module)
        .collect::<std::collections::BTreeSet<_>>()
        .len();
    if unique_modules != POST_EXECUTION_BINDINGS.len() {
        return Err("post module inventory contains a duplicate module".into());
    }
    if POST_EXECUTION_BINDINGS.iter().any(|binding| {
        binding.declaration != format!("post_{}", binding.module)
            || binding.registration_markers.is_empty()
    }) {
        return Err(
            "post module inventory has an invalid declaration/module pair or no registration markers"
                .into(),
        );
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
        let names = POST_EXECUTION_BINDINGS
            .iter()
            .flat_map(|binding| binding.executors.iter().map(|executor| executor.name()))
            .collect::<Vec<_>>();
        return Err(format!(
            "post execution ledger reuses an executor: {names:?}"
        ));
    }
    if POST_EXECUTION_BINDINGS
        .iter()
        .flat_map(|binding| binding.registration_markers.iter())
        .any(|marker| marker.is_empty())
    {
        return Err("post module inventory contains an empty registration marker".into());
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
    use std::collections::BTreeSet;

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

    #[test]
    fn repository_inventory_names_every_production_post_module() {
        // This is deliberately a repository-source audit, not a miniature
        // graph assembled by the test. The production module root and plugin
        // source are included here so a module cannot disappear from either
        // the real Rust module tree or its real registration path while the
        // declaration/executor table remains green.
        let shading_source = include_str!("mod.rs");
        let plugin_source = include_str!("plugin.rs");
        let modules = POST_EXECUTION_BINDINGS
            .iter()
            .map(|binding| binding.module)
            .collect::<BTreeSet<_>>();
        assert_eq!(POST_EXECUTION_BINDINGS.len(), 23);
        assert_eq!(modules.len(), POST_EXECUTION_BINDINGS.len());

        for binding in POST_EXECUTION_BINDINGS {
            assert!(
                shading_source.contains(&format!("mod {};", binding.module)),
                "post module {} is not declared by shading/mod.rs",
                binding.module
            );
            assert!(
                plugin_source.contains(&format!("{}::", binding.module)),
                "post module {} is not imported by shading/plugin.rs",
                binding.module
            );
            for marker in binding.registration_markers {
                assert!(
                    plugin_source.contains(marker),
                    "post module {} is missing production registration marker {}",
                    binding.module,
                    marker
                );
            }
            for executor in binding.executors {
                assert!(
                    plugin_source.contains(executor.name()),
                    "post module {} is missing executor {} from production plugin",
                    binding.module,
                    executor.name()
                );
            }
        }

        let documented_post_modules = [
            "bloom",
            "cas",
            "chromatic_aberration",
            "color_grade",
            "dof",
            "exposure",
            "film_grain",
            "gamut_map",
            "halftone",
            "hatching",
            "kuwahara",
            "lens_flare",
            "motion_blur",
            "ordered_dither",
            "outline",
            "posterize",
            "tonemap",
            "vignette",
        ];
        for module in documented_post_modules {
            assert!(
                modules.contains(module),
                "documented production post module {module} is absent from inventory"
            );
        }
    }

    #[test]
    fn post_effects_publish_distinct_resource_contracts() {
        let find = |name: &str| {
            POST_PROCESS_DECLARATIONS
                .iter()
                .find(|pass| pass.name == name)
                .unwrap_or_else(|| panic!("missing declaration {name}"))
        };
        let has = |pass: &PassSkeletonDeclaration, resource: &str, kind: AccessKind| {
            pass.accesses
                .iter()
                .any(|access| access.resource == resource && access.kind == kind)
        };

        let exposure = find("post_exposure");
        assert!(has(
            exposure,
            "post_exposure_histogram",
            AccessKind::StorageWrite
        ));
        assert!(has(
            exposure,
            "post_exposure_state",
            AccessKind::StorageWrite
        ));

        let ssr = find("post_ssr");
        assert!(has(ssr, "post_ssr_hzb", AccessKind::StorageWrite));
        assert!(has(ssr, "post_ssr_trace", AccessKind::StorageWrite));
        assert!(has(ssr, "post_ssr_hit_distance", AccessKind::SampledRead));

        let stylized = find("post_posterize");
        assert!(has(stylized, "hdr_scene_color", AccessKind::StorageWrite));
        assert!(!has(stylized, "post_ssr_hzb", AccessKind::SampledRead));

        assert_ne!(exposure.accesses, ssr.accesses);
        assert_ne!(ssr.accesses, stylized.accesses);
    }

    #[test]
    fn post_resource_lifetimes_keep_history_persistent_and_scratch_transient() {
        let resource = |name: &str| {
            POST_RESOURCES
                .iter()
                .find(|resource| resource.name == name)
                .unwrap_or_else(|| panic!("missing resource {name}"))
        };
        assert_eq!(
            resource("post_taa_history").lifetime,
            ResourceLifetime::Persistent
        );
        assert_eq!(
            resource("post_spec_history").lifetime,
            ResourceLifetime::Persistent
        );
        assert_eq!(
            resource("post_ssr_trace").lifetime,
            ResourceLifetime::Transient
        );
        assert_eq!(
            resource("post_bloom_down").lifetime,
            ResourceLifetime::Transient
        );
    }
}
