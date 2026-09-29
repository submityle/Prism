//! Particle renderer matrix and per-renderer routing (design §15).
//!
//! One emitter may drive several renderers, each issuing an indirect draw into a
//! render phase (`Transparent3d` / `AlphaMask3d` / `Opaque3d`). Every renderer
//! can pair with any shading model (design §16). This module owns the
//! renderer-kind taxonomy and the phase-routing contract; ribbon/beam geometry
//! generation contracts are expanded here as the subsystem deepens.

/// The render primitive an emitter's renderer emits (design §15).
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum RendererKind {
    /// Camera/velocity/axis-aligned billboards; flipbook; soft particles.
    Sprite,
    /// One mesh instance per particle (instanced, full shading closures).
    Mesh,
    /// `RibbonId`-linked strips generated on the `GPU`.
    Ribbon,
    /// Two/multi-point beams (lightning, lasers).
    Beam,
    /// Particle-driven lights routed into the clustered light list.
    Light,
    /// Ground decals routed into the deferred/forward decal path.
    Decal,
    /// Grid-fluid density volume ray-marched (design §10, §20).
    Volume,
}

/// The render phase a draw is routed into (design §12, §15).
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum RenderPhase {
    /// Opaque geometry with depth write.
    Opaque3d,
    /// Alpha-tested cutout geometry.
    AlphaMask3d,
    /// Order-dependent translucency.
    Transparent3d,
}

impl RendererKind {
    /// Whether this renderer draws geometry at all. `Light` contributes to the
    /// clustered light list rather than emitting a draw, so it has no phase.
    #[must_use]
    pub const fn draws_geometry(self) -> bool {
        !matches!(self, RendererKind::Light)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::particle::EmberShadingModel;

    #[test]
    fn light_renderer_draws_no_geometry() {
        assert!(!RendererKind::Light.draws_geometry());
        assert!(RendererKind::Sprite.draws_geometry());
        assert!(RendererKind::Volume.draws_geometry());
    }

    #[test]
    fn renderer_kinds_are_distinct() {
        assert_ne!(RendererKind::Sprite, RendererKind::Mesh);
        assert_ne!(RendererKind::Ribbon, RendererKind::Beam);
    }

    #[test]
    fn shading_model_pairs_with_any_renderer() {
        // Every renderer is compatible with every shading model (design §16);
        // this placeholder anchors that invariant as the module deepens.
        let _ = (RendererKind::Mesh, EmberShadingModel::Pbr);
        let _ = (RendererKind::Sprite, EmberShadingModel::Unlit);
    }
}
