//! Semantic render-state mapping for material queueing and pipeline selection.
//!
//! This is deliberately separate from `prism_render_app::PipelineState`, which
//! is the complete descriptor-derived fixed-function cache identity. The
//! semantic state here is the stable contract that maps the material axes to
//! a phase and the state family a renderer must specialize.

use crate::{MaterialFeatureFlags, MaterialRecord, MaterialRenderClass};
use prism_render_architecture::render_fabric::RenderDomain;

/// The frame-graph phase selected by a material's render-domain contract.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum MaterialRenderPhase {
    /// Depth-tested geometry that writes the opaque depth target.
    Opaque,
    /// Color-composited geometry that leaves the depth target unchanged.
    Transparent,
    /// Domain-owned work that is not part of the surface opaque/transparent queues.
    Custom,
}

/// The color attachment family selected by a material.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum MaterialRenderTarget {
    /// The normal opaque color target.
    Opaque,
    /// The HDR target used by transparent composites.
    HdrTransparent,
}

/// The blend family selected by a material.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum MaterialBlendMode {
    /// The source replaces the destination.
    Opaque,
    /// Straight-alpha source color.
    Alpha,
    /// Source color is already premultiplied by alpha.
    PremultipliedAlpha,
    /// Additive source color.
    Additive,
}

/// The logical depth comparison. Backends may translate this for reverse-Z.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum MaterialDepthTest {
    /// Pass when the fragment is nearer than or equal to stored depth.
    LessEqual,
    /// Always pass the depth test.
    Always,
}

/// Face-culling policy selected by a material.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum MaterialCullMode {
    /// Cull back-facing triangles.
    Back,
    /// Rasterize both windings.
    None,
}

/// Complete semantic state consumed by queueing and pipeline specialization.
///
/// This state is intentionally small and domain-neutral. It is not a GPU
/// descriptor and must be lowered into the backend's full fixed-function
/// descriptor before entering the pipeline cache.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct MaterialPipelineState {
    /// Runtime render domain.
    pub domain: RenderDomain,
    /// Seven-value material blend/raster class.
    pub render_class: MaterialRenderClass,
    /// Queue phase.
    pub phase: MaterialRenderPhase,
    /// Color target family.
    pub target: MaterialRenderTarget,
    /// Color blend mode.
    pub blend: MaterialBlendMode,
    /// Logical depth comparison.
    pub depth_test: MaterialDepthTest,
    /// Whether this draw writes depth.
    pub depth_write: bool,
    /// Triangle culling policy.
    pub cull_mode: MaterialCullMode,
}

impl MaterialPipelineState {
    /// Returns whether this state composites with an existing color target.
    #[must_use]
    pub const fn is_transparent(self) -> bool {
        !matches!(self.blend, MaterialBlendMode::Opaque)
    }
}

/// Resolves the single semantic mapping from domain, render class and features.
///
/// The water branch is intentionally checked before the ordinary surface
/// classes: water is a surface-domain feature with a fixed transparent HDR
/// contract, even when an authoring path supplied a stale opaque class.
#[must_use]
pub const fn pipeline_state_for_material(
    domain: RenderDomain,
    render_class: MaterialRenderClass,
    features: MaterialFeatureFlags,
) -> MaterialPipelineState {
    match domain {
        RenderDomain::Surface => {
            if features.contains(MaterialFeatureFlags::WATER) {
                return MaterialPipelineState {
                    domain,
                    render_class,
                    phase: MaterialRenderPhase::Transparent,
                    target: MaterialRenderTarget::HdrTransparent,
                    blend: MaterialBlendMode::PremultipliedAlpha,
                    depth_test: MaterialDepthTest::LessEqual,
                    depth_write: false,
                    cull_mode: MaterialCullMode::None,
                };
            }
            surface_state(domain, render_class, features)
        }
        RenderDomain::Particle | RenderDomain::Hair => MaterialPipelineState {
            domain,
            render_class,
            phase: MaterialRenderPhase::Opaque,
            target: MaterialRenderTarget::Opaque,
            blend: MaterialBlendMode::Opaque,
            depth_test: MaterialDepthTest::LessEqual,
            depth_write: true,
            cull_mode: if features.contains(MaterialFeatureFlags::DOUBLE_SIDED) {
                MaterialCullMode::None
            } else {
                MaterialCullMode::Back
            },
        },
        RenderDomain::Volume => MaterialPipelineState {
            domain,
            render_class,
            phase: MaterialRenderPhase::Custom,
            target: MaterialRenderTarget::HdrTransparent,
            blend: MaterialBlendMode::PremultipliedAlpha,
            depth_test: MaterialDepthTest::LessEqual,
            depth_write: false,
            cull_mode: MaterialCullMode::None,
        },
        RenderDomain::Decal => MaterialPipelineState {
            domain,
            render_class,
            phase: MaterialRenderPhase::Transparent,
            target: MaterialRenderTarget::HdrTransparent,
            blend: MaterialBlendMode::Alpha,
            depth_test: MaterialDepthTest::LessEqual,
            depth_write: false,
            cull_mode: MaterialCullMode::Back,
        },
    }
}

/// Resolves the state carried by a material record.
#[must_use]
pub const fn material_pipeline_state(record: &MaterialRecord) -> MaterialPipelineState {
    pipeline_state_for_material(record.render_domain, record.render_class, record.features)
}

const fn surface_state(
    domain: RenderDomain,
    render_class: MaterialRenderClass,
    features: MaterialFeatureFlags,
) -> MaterialPipelineState {
    let (phase, target, blend, depth_write) = match render_class {
        MaterialRenderClass::Opaque
        | MaterialRenderClass::OpaqueTwoSided
        | MaterialRenderClass::Masked
        | MaterialRenderClass::MaskedTwoSided => (
            MaterialRenderPhase::Opaque,
            MaterialRenderTarget::Opaque,
            MaterialBlendMode::Opaque,
            true,
        ),
        MaterialRenderClass::Transmissive | MaterialRenderClass::Transparent => (
            MaterialRenderPhase::Transparent,
            MaterialRenderTarget::HdrTransparent,
            MaterialBlendMode::PremultipliedAlpha,
            false,
        ),
        MaterialRenderClass::Additive => (
            MaterialRenderPhase::Transparent,
            MaterialRenderTarget::HdrTransparent,
            MaterialBlendMode::Additive,
            false,
        ),
    };
    let two_sided = matches!(
        render_class,
        MaterialRenderClass::OpaqueTwoSided | MaterialRenderClass::MaskedTwoSided
    ) || features.contains(MaterialFeatureFlags::DOUBLE_SIDED);
    MaterialPipelineState {
        domain,
        render_class,
        phase,
        target,
        blend,
        depth_test: MaterialDepthTest::LessEqual,
        depth_write,
        cull_mode: if two_sided {
            MaterialCullMode::None
        } else {
            MaterialCullMode::Back
        },
    }
}
