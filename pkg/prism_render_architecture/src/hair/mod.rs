//! Hair (groom) subsystem contracts.
//!
//! Hair is a full strand-based groom engine and a subsystem in its own right,
//! separate from cloth: the primitive is a *strand*, not a triangle. The
//! pipeline mirrors production hair engines (UE5 Groom, AMD `TressFX`, NVIDIA
//! `HairWorks`, and film-grade Chiang/Marschner shading) at the algorithm level,
//! without reusing any of their code:
//!
//! 1. **Import & interpolation** — a small set of simulated *guide* strands is
//!    interpolated into many *render* strands, so simulation cost stays bounded
//!    while visual density scales.
//! 2. **Strand dynamics** — guides are advanced by an XPBD-style solver
//!    (edge-length plus local/global shape constraints). The per-frame vertex
//!    work is arbitrated by [`crate::deformation::schedule`]; this subsystem
//!    only emits requests, it does not own the budget.
//! 3. **LOD** — a ladder from full strands to decimated strands to camera
//!    cards to a static mesh shell, selected by screen coverage; see [`lod`].
//! 4. **Rasterization** — thin strands are drawn in a compute/visibility pass
//!    (sub-pixel software raster) rather than the hardware triangle path.
//! 5. **Shading** — a physically based hair BSDF (Chiang / Marschner) with a
//!    dual-scattering multiple-scattering approximation, expressed through the
//!    material system's `HairPbr` closure rather than reimplemented here.
//! 6. **Transmittance & shadows** — deep opacity maps / order-independent
//!    transparency for self-shadowing and blending, routed through the
//!    transparency subsystem's `HairVisibility` path.
//!
//! This module owns the geometry, LOD, and simulation-binding contracts. It
//! references, never reimplements, the shared deformation budget, the material
//! closures, and the transparency routing.

pub mod lod;

use crate::deformation::DeformationHandle;

/// Identifies one hair group: a groom bound to a skinned mesh.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct HairGroupHandle(pub u32);

/// Discrete hair level-of-detail tiers, coarsening with distance/coverage.
///
/// The first two tiers stay strand-based and therefore simulate; `Cards` and
/// `Mesh` are static proxies that skip strand dynamics entirely.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum HairLodTier {
    /// Full guide-interpolated render strands at authored density.
    Strands,
    /// Decimated strands with fewer control points; still strand-based.
    ReducedStrands,
    /// Camera-facing textured cards; no per-strand geometry.
    Cards,
    /// Single static mesh shell for far or off-screen fallback.
    Mesh,
}

impl HairLodTier {
    /// Returns `true` when this tier renders individual strands and therefore
    /// drives strand dynamics through the deformation budget.
    #[must_use]
    pub fn is_strand_based(self) -> bool {
        matches!(self, HairLodTier::Strands | HairLodTier::ReducedStrands)
    }
}

/// Authoring description of one hair group at its finest LOD.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct HairGroup {
    /// Stable identity of this groom.
    pub handle: HairGroupHandle,
    /// Simulated guide strands; the interpolation source and the sim cost.
    pub guide_strand_count: u32,
    /// Render strands interpolated from guides at the finest tier.
    pub max_render_strands: u32,
    /// Control points per strand at the finest tier.
    pub segments_per_strand: u32,
    /// Deformation-cache entry that strand dynamics writes into.
    pub deformation: DeformationHandle,
}
