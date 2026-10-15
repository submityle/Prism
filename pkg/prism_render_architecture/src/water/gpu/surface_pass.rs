//! The water-surface raster draw-pass contract: the "shaded surface → frame
//! buffer" step that turns the solved water fields into visible pixels.
//!
//! Every other water `WESL` kernel in this subsystem is a `@compute` pass:
//! they evolve the spectrum, step the `SWE`/`PBF`/`FLIP` solvers, advect foam,
//! and reconstruct the surface — all writing into device-resident buffers (see
//! [`super::pipeline`] and [`super::super::kernels`]). None of them draw
//! anything. This module owns the missing half: the `@vertex`/`@fragment`
//! raster pass that reads the solved displacement/normal/foam fields, shades
//! them through the §5 response fork, and composites the result into the frame
//! buffer. It mirrors the forward transparent water pass every shipping
//! renderer runs — `UE5` Single Layer Water, `Crest`, and `WaveWorks` all draw
//! their ocean this way, after the opaque pass and before post-processing.
//!
//! Like the rest of [`super`], this is **pure integer bookkeeping**: no `GPU`
//! handles, no floats, no wall clock. It fixes the blend mode, depth state,
//! render target and the stable `WESL` entry-point names
//! the scene crate's draw node binds against, so the whole raster contract is
//! deterministic and `CPU`-testable. The `WESL` source behind each entry name
//! is authored in `water_surface.wesl`; only the contract lives here, exactly
//! as [`super::super::kernels::WaterKernel::wesl_entry_point`] contracts the
//! compute entry names.
//!
//! The geometry (vertex) stage is identical across all response plans — the
//! displaced surface mesh is response-agnostic — while the lighting response
//! (fragment) stage forks per response program (design §5). That split is expressed
//! here as a shared [`SurfaceDrawDescriptor::vertex_entry`] and a
//! response-plan-specialized [`SurfaceDrawDescriptor::fragment_entry`].

use super::super::{ResponseProgram, WaterShadingPlan};

/// Depth-test comparison the surface draw uses against the already-populated
/// scene depth buffer.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum DepthTest {
    /// Pass when the fragment depth is less than or equal to the stored depth.
    /// The transparent water surface tests against opaque geometry so submerged
    /// objects correctly occlude it, but shares the plane with coplanar detail.
    LessEqual,
    /// Pass only when strictly nearer than the stored depth.
    Less,
    /// Always pass (used for full-screen underwater overlays that ignore depth).
    Always,
}

/// Color blend mode the surface draw composites with.
///
/// The default transparent water pass pre-multiplies its refraction/reflection
/// result into the color before blending, matching the `UE5` Single Layer Water
/// and `Crest` compositing path; opaque walkable shallow water writes directly.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum SurfaceBlend {
    /// Pre-multiplied alpha: `dst = src + dst * (1 - src.a)`. Default for the
    /// transparent refraction/reflection composite.
    PremultipliedAlpha,
    /// Straight alpha: `dst = src * src.a + dst * (1 - src.a)`.
    AlphaBlend,
    /// No blending; the source replaces the destination.
    Opaque,
}

impl SurfaceBlend {
    /// `true` when the mode writes without reading the destination.
    #[must_use]
    pub fn is_opaque(self) -> bool {
        matches!(self, SurfaceBlend::Opaque)
    }
}

/// Depth-buffer interaction for the surface draw: how it tests and whether it
/// writes back.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct SurfaceDepth {
    /// Comparison against the stored scene depth.
    pub test: DepthTest,
    /// Whether the pass writes its own depth. A transparent water surface tests
    /// against opaque depth but leaves it unwritten so later transparent draws
    /// still sort against the opaque scene, not against the water.
    pub write: bool,
}

/// Render target the surface draw composites into.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum WaterRenderTarget {
    /// The main `HDR` color target, drawn after the opaque pass and before
    /// post-processing. The default for transparent ocean/surface water.
    HdrTransparent,
    /// The main `HDR` color target in the opaque phase, writing depth. Used by
    /// shallow walkable water that participates in opaque sorting.
    HdrOpaque,
}

/// Validation failures for a water-surface draw descriptor.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum SurfacePassContractError {
    /// A transparent target must use the water premultiplied composite.
    TransparentTargetState,
    /// An opaque target must replace color and write depth.
    OpaqueTargetState,
}

/// The full contract for one water-surface raster draw.
///
/// Carries the response/composition plan, blend mode, depth state, and render
/// target. The
/// `WESL` entry-point accessors give the stable names the scene crate's draw
/// node keys its render pipeline against.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct SurfaceDrawDescriptor {
    /// Which §5 response/composition plan shades this surface.
    pub plan: WaterShadingPlan,
    /// Color blend mode.
    pub blend: SurfaceBlend,
    /// Depth test / write state.
    pub depth: SurfaceDepth,
    /// Render target and phase.
    pub target: WaterRenderTarget,
}

impl SurfaceDrawDescriptor {
    /// Checks that target, blend and depth state describe one coherent pass.
    ///
    /// Keeping this check beside the CPU contract gives render backends and
    /// tests one invariant to call before lowering the descriptor to an RHI
    /// pipeline. In particular, a transparent water draw can never silently
    /// become an opaque depth-writing pass.
    pub fn validate(self) -> Result<(), SurfacePassContractError> {
        match self.target {
            WaterRenderTarget::HdrTransparent
                if self.blend == SurfaceBlend::PremultipliedAlpha
                    && self.depth.test == DepthTest::LessEqual
                    && !self.depth.write =>
            {
                Ok(())
            }
            WaterRenderTarget::HdrOpaque
                if self.blend == SurfaceBlend::Opaque
                    && self.depth.test == DepthTest::LessEqual
                    && self.depth.write =>
            {
                Ok(())
            }
            WaterRenderTarget::HdrTransparent => {
                Err(SurfacePassContractError::TransparentTargetState)
            }
            WaterRenderTarget::HdrOpaque => Err(SurfacePassContractError::OpaqueTargetState),
        }
    }

    /// The stable `WESL` vertex entry-point name.
    ///
    /// Shared across all response plans: the displaced surface mesh geometry is
    /// frontend-agnostic (design §5), so a single vertex stage feeds every
    /// lighting response. The `WESL` source lives in `water_surface.wesl`.
    #[must_use]
    pub fn vertex_entry(self) -> &'static str {
        "water_surface_vs"
    }

    /// The stable `WESL` fragment entry-point name for this response plan.
    ///
    /// Unlike the vertex stage, the fragment stage forks per response program
    /// because response programs diverge exactly in lighting response (design §5). Each
    /// name is distinct so the pipeline cache keys the correct specialization.
    #[must_use]
    pub fn fragment_entry(self) -> &'static str {
        match self.plan {
            WaterShadingPlan::Single { program } => match program {
                ResponseProgram::Pbr => "water_surface_fs_pbr",
                ResponseProgram::Npr => "water_surface_fs_npr",
                ResponseProgram::Custom => "water_surface_fs_custom",
            },
            // MaskBlend is the resident response producer.  Its name describes
            // the semantic operation rather than the retired compatibility
            // frontend, so the pipeline identity cannot accidentally revive the
            // old standalone entry.
            WaterShadingPlan::MaskBlend { .. } => "water_surface_fs_mask_blend",
        }
    }

    /// `true` when this draw blends against the destination (a transparent
    /// composite) rather than overwriting it.
    #[must_use]
    pub fn is_transparent(self) -> bool {
        !self.blend.is_opaque()
    }
}

/// Plan the default surface draw for one response program.
///
/// Uses the catalog-derived shared service closure, pre-multiplied
/// alpha compositing for the refraction/reflection result, a `LessEqual` depth
/// test against opaque geometry with depth-write disabled (the transparent
/// surface must not occlude later transparent draws), and the main `HDR`
/// transparent target. Deterministic: identical response program in, identical
/// descriptor out, so the `GPU` pipeline key is stable across frames.
#[must_use]
pub fn plan_surface_draw(frontend: ResponseProgram) -> SurfaceDrawDescriptor {
    plan_surface_plan(WaterShadingPlan::single(frontend))
}

/// Plan a surface draw from the complete response/composition plan.
#[must_use]
pub fn plan_surface_plan(plan: WaterShadingPlan) -> SurfaceDrawDescriptor {
    let descriptor = SurfaceDrawDescriptor {
        plan,
        blend: SurfaceBlend::PremultipliedAlpha,
        depth: SurfaceDepth {
            test: DepthTest::LessEqual,
            write: false,
        },
        target: WaterRenderTarget::HdrTransparent,
    };
    debug_assert!(descriptor.validate().is_ok());
    descriptor
}

#[cfg(test)]
mod tests {
    use super::*;

    const FRONTENDS: [ResponseProgram; 3] = [
        ResponseProgram::Pbr,
        ResponseProgram::Npr,
        ResponseProgram::Custom,
    ];

    #[test]
    fn every_frontend_uses_the_shared_response_contract() {
        for frontend in FRONTENDS {
            assert!(!plan_surface_draw(frontend).fragment_entry().is_empty());
        }
    }

    #[test]
    fn plan_is_deterministic() {
        for frontend in FRONTENDS {
            assert_eq!(plan_surface_draw(frontend), plan_surface_draw(frontend));
        }
    }

    #[test]
    fn default_draw_is_a_transparent_premultiplied_composite() {
        for frontend in FRONTENDS {
            let draw = plan_surface_draw(frontend);
            assert_eq!(draw.blend, SurfaceBlend::PremultipliedAlpha);
            assert!(draw.is_transparent());
            assert!(!draw.blend.is_opaque());
            assert_eq!(draw.target, WaterRenderTarget::HdrTransparent);
        }
    }

    #[test]
    fn transparent_surface_tests_opaque_depth_without_writing() {
        for frontend in FRONTENDS {
            let draw = plan_surface_draw(frontend);
            assert_eq!(draw.depth.test, DepthTest::LessEqual);
            assert!(
                !draw.depth.write,
                "transparent water must not write depth and occlude later draws"
            );
        }
    }

    #[test]
    fn vertex_entry_is_shared_and_stable() {
        let shared = plan_surface_draw(ResponseProgram::Pbr).vertex_entry();
        assert_eq!(shared, "water_surface_vs");
        for frontend in FRONTENDS {
            // The displaced-mesh geometry stage is frontend-agnostic (§5).
            assert_eq!(plan_surface_draw(frontend).vertex_entry(), shared);
        }
    }

    #[test]
    fn fragment_entry_forks_per_response_and_is_a_raster_stage() {
        let mut seen: [&'static str; 3] = [""; 3];
        for (slot, frontend) in FRONTENDS.iter().enumerate() {
            let draw = plan_surface_draw(*frontend);
            let fs = draw.fragment_entry();
            // A real raster pass: non-empty vertex + fragment entries that
            // differ from each other (proves this is not a compute kernel).
            assert!(!fs.is_empty());
            assert!(!draw.vertex_entry().is_empty());
            assert_ne!(fs, draw.vertex_entry());
            seen[slot] = fs;
        }
        // All four fragment specializations are distinct.
        for i in 0..seen.len() {
            for j in (i + 1)..seen.len() {
                assert_ne!(seen[i], seen[j], "fragment entries must be unique");
            }
        }
    }

    #[test]
    fn mask_blend_uses_the_resident_composition_entry() {
        let draw = plan_surface_plan(WaterShadingPlan::hybrid());
        assert!(matches!(draw.plan, WaterShadingPlan::MaskBlend { .. }));
        assert_eq!(draw.fragment_entry(), "water_surface_fs_mask_blend");
    }

    #[test]
    fn opaque_blend_reports_opaque() {
        assert!(SurfaceBlend::Opaque.is_opaque());
        assert!(!SurfaceBlend::PremultipliedAlpha.is_opaque());
        assert!(!SurfaceBlend::AlphaBlend.is_opaque());
    }

    #[test]
    fn target_blend_and_depth_form_one_contract() {
        for frontend in FRONTENDS {
            assert_eq!(
                plan_surface_draw(frontend).validate(),
                Ok(()),
                "every water response must use the transparent target contract"
            );
        }
        let invalid = SurfaceDrawDescriptor {
            plan: WaterShadingPlan::single(ResponseProgram::Pbr),
            blend: SurfaceBlend::Opaque,
            depth: SurfaceDepth {
                test: DepthTest::LessEqual,
                write: true,
            },
            target: WaterRenderTarget::HdrTransparent,
        };
        assert_eq!(
            invalid.validate(),
            Err(SurfacePassContractError::TransparentTargetState)
        );
    }
}
