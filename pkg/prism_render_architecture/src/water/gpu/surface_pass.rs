//! The water-surface raster draw-pass contract: the "shaded surface → frame
//! buffer" step that turns the solved water fields into visible pixels.
//!
//! Every other water `WESL` kernel in this subsystem is a `@compute` pass:
//! they evolve the spectrum, step the `SWE`/`PBF`/`FLIP` solvers, advect foam,
//! and reconstruct the surface — all writing into device-resident buffers (see
//! [`super::pipeline`] and [`super::super::kernels`]). None of them draw
//! anything. This module owns the missing half: the `@vertex`/`@fragment`
//! raster pass that reads the solved displacement/normal/foam fields, shades
//! them through the §5 frontend fork, and composites the result into the frame
//! buffer. It mirrors the forward transparent water pass every shipping
//! renderer runs — `UE5` Single Layer Water, `Crest`, and `WaveWorks` all draw
//! their ocean this way, after the opaque pass and before post-processing.
//!
//! Like the rest of [`super`], this is **pure integer bookkeeping**: no `GPU`
//! handles, no floats, no wall clock. It fixes the blend mode, depth state,
//! render target, bound shared base, and the stable `WESL` entry-point names
//! the scene crate's draw node binds against, so the whole raster contract is
//! deterministic and `CPU`-testable. The `WESL` source behind each entry name
//! is authored in `water_surface.wesl`; only the contract lives here, exactly
//! as [`super::super::kernels::WaterKernel::wesl_entry_point`] contracts the
//! compute entry names.
//!
//! The geometry (vertex) stage is identical across all four frontends — the
//! displaced surface mesh is frontend-agnostic — while the lighting response
//! (fragment) stage forks per frontend (design §5). That split is expressed
//! here as a shared [`SurfaceDrawDescriptor::vertex_entry`] and a
//! frontend-specialized [`SurfaceDrawDescriptor::fragment_entry`].

use super::super::{ShadingFrontend, SharedBaseServices};

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

/// The full contract for one water-surface raster draw.
///
/// Carries the shading frontend, the shared advanced base it binds (always the
/// full base — §5b), the blend mode, depth state, and render target. The
/// `WESL` entry-point accessors give the stable names the scene crate's draw
/// node keys its render pipeline against.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct SurfaceDrawDescriptor {
    /// Which §5 lighting-response frontend shades this surface.
    pub frontend: ShadingFrontend,
    /// The shared advanced base services bound for this draw. Every frontend
    /// binds the full base; `NPR` is not a reduced path (§5b).
    pub bound_base: SharedBaseServices,
    /// Color blend mode.
    pub blend: SurfaceBlend,
    /// Depth test / write state.
    pub depth: SurfaceDepth,
    /// Render target and phase.
    pub target: WaterRenderTarget,
}

impl SurfaceDrawDescriptor {
    /// The stable `WESL` vertex entry-point name.
    ///
    /// Shared across all four frontends: the displaced surface mesh geometry is
    /// frontend-agnostic (design §5), so a single vertex stage feeds every
    /// lighting response. The `WESL` source lives in `water_surface.wesl`.
    #[must_use]
    pub fn vertex_entry(self) -> &'static str {
        "water_surface_vs"
    }

    /// The stable `WESL` fragment entry-point name for this frontend.
    ///
    /// Unlike the vertex stage, the fragment stage forks per frontend because
    /// the frontends diverge exactly in the lighting response (design §5). Each
    /// name is distinct so the pipeline cache keys the correct specialization.
    #[must_use]
    pub fn fragment_entry(self) -> &'static str {
        match self.frontend {
            ShadingFrontend::Pbr => "water_surface_fs_pbr",
            ShadingFrontend::Npr => "water_surface_fs_npr",
            ShadingFrontend::Custom => "water_surface_fs_custom",
            ShadingFrontend::Hybrid => "water_surface_fs_hybrid",
        }
    }

    /// `true` when this draw blends against the destination (a transparent
    /// composite) rather than overwriting it.
    #[must_use]
    pub fn is_transparent(self) -> bool {
        !self.blend.is_opaque()
    }
}

/// Plan the default surface draw for a frontend.
///
/// Binds the full shared advanced base (§5b, every frontend), pre-multiplied
/// alpha compositing for the refraction/reflection result, a `LessEqual` depth
/// test against opaque geometry with depth-write disabled (the transparent
/// surface must not occlude later transparent draws), and the main `HDR`
/// transparent target. Deterministic: identical frontend in, identical
/// descriptor out, so the `GPU` pipeline key is stable across frames.
#[must_use]
pub fn plan_surface_draw(frontend: ShadingFrontend) -> SurfaceDrawDescriptor {
    SurfaceDrawDescriptor {
        frontend,
        bound_base: frontend.shared_base(),
        blend: SurfaceBlend::PremultipliedAlpha,
        depth: SurfaceDepth {
            test: DepthTest::LessEqual,
            write: false,
        },
        target: WaterRenderTarget::HdrTransparent,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const FRONTENDS: [ShadingFrontend; 4] = [
        ShadingFrontend::Pbr,
        ShadingFrontend::Npr,
        ShadingFrontend::Custom,
        ShadingFrontend::Hybrid,
    ];

    #[test]
    fn every_frontend_binds_the_full_shared_base() {
        for frontend in FRONTENDS {
            let draw = plan_surface_draw(frontend);
            assert!(
                draw.bound_base.contains(SharedBaseServices::ALL),
                "{frontend:?} must bind the full shared advanced base (§5b)"
            );
            assert_eq!(draw.bound_base.len(), SharedBaseServices::COUNT);
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
        let shared = plan_surface_draw(ShadingFrontend::Pbr).vertex_entry();
        assert_eq!(shared, "water_surface_vs");
        for frontend in FRONTENDS {
            // The displaced-mesh geometry stage is frontend-agnostic (§5).
            assert_eq!(plan_surface_draw(frontend).vertex_entry(), shared);
        }
    }

    #[test]
    fn fragment_entry_forks_per_frontend_and_is_a_raster_stage() {
        let mut seen: [&'static str; 4] = [""; 4];
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
    fn opaque_blend_reports_opaque() {
        assert!(SurfaceBlend::Opaque.is_opaque());
        assert!(!SurfaceBlend::PremultipliedAlpha.is_opaque());
        assert!(!SurfaceBlend::AlphaBlend.is_opaque());
    }
}
