//! Per-surface transparency path selection and draw bucketing.
//!
//! Transparent surfaces cannot all be composited the same way: sortable glass
//! wants back-to-front alpha blending, overlapping foliage needs an
//! order-independent (OIT) resolve, water and hair have bespoke passes, and
//! participating media are marched as volumes. This module turns a surface's
//! content properties into a concrete [`TransparencyPath`] and then fans a
//! frame's transparent draws into one bucket per path, mirroring the geometry
//! raster's [`RasterBins`](crate::virtual_geometry) and the material resolve
//! buckets.
//!
//! Path selection is orthogonal to a surface's [`MaterialExecutionPath`](
//! crate::material::MaterialExecutionPath): a PBR glass and an NPR "anime"
//! rim-lit glass both route to the same transparency path and differ only in
//! their resolve shader. That keeps transparency a shared capability of every
//! material family rather than a feature bolted onto one of them.

use alloc::vec::Vec;

use super::{TransparencyOutputs, TransparencyPath};

/// The broad content class of a transparent surface.
///
/// The class captures the intent the artist assigned to the surface; the
/// finer-grained [`TransparencyPath`] is derived from it together with the
/// surface's blending needs and the backend's capabilities.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum TransparentKind {
    /// Ordinary blended surface (particles, decals, UI-in-world, foliage).
    General,
    /// A water body with its own single-layer refraction/reflection pass.
    Water,
    /// Hair or fur resolved through the dedicated visibility pass.
    Hair,
    /// Refractive glass; thickness decides single- vs multi-layer handling.
    Glass,
    /// Participating media marched as a volume.
    Volume,
}

/// Content properties that drive transparency path selection.
#[derive(Clone, Copy, Debug)]
pub struct TransparentSurface {
    /// Broad content class of the surface.
    pub kind: TransparentKind,
    /// `true` when fragments overlap unpredictably and cannot be depth-sorted,
    /// forcing an order-independent resolve.
    pub order_independent: bool,
    /// `true` when the surface wants the higher-fidelity moment-based OIT over
    /// the cheaper weighted-blended approximation.
    pub high_fidelity: bool,
    /// Number of stacked refractive layers for [`TransparentKind::Glass`].
    pub layer_count: u32,
}

impl TransparentSurface {
    /// A plain blended surface with no special requirements.
    #[must_use]
    pub const fn general() -> Self {
        Self {
            kind: TransparentKind::General,
            order_independent: false,
            high_fidelity: false,
            layer_count: 1,
        }
    }
}

/// Backend transparency capabilities that gate path selection.
#[derive(Clone, Copy, Debug, Default)]
pub struct TransparencyCapability {
    /// `true` when the backend can run the moment-based OIT resolve.
    pub moment_oit: bool,
}

/// Selects the transparency path for a surface given backend capabilities.
///
/// Water, hair, and volumes always take their dedicated paths. Glass sorts when
/// it is a single layer and uses the layered resolve otherwise. A general
/// surface that can be depth-sorted takes the cheap sorted path; when it cannot,
/// it uses moment OIT if the surface asks for fidelity and the backend supports
/// it, and otherwise falls back to weighted-blended OIT.
#[must_use]
pub fn select_transparency_path(
    surface: TransparentSurface,
    capability: TransparencyCapability,
) -> TransparencyPath {
    match surface.kind {
        TransparentKind::Water => TransparencyPath::SingleLayerWater,
        TransparentKind::Hair => TransparencyPath::HairVisibility,
        TransparentKind::Volume => TransparencyPath::Volumetric,
        TransparentKind::Glass => {
            if surface.layer_count > 1 {
                TransparencyPath::LayeredGlass
            } else {
                TransparencyPath::Sorted
            }
        }
        TransparentKind::General => {
            if surface.order_independent {
                if surface.high_fidelity && capability.moment_oit {
                    TransparencyPath::MomentOit
                } else {
                    TransparencyPath::WeightedOit
                }
            } else {
                TransparencyPath::Sorted
            }
        }
    }
}

/// Describes what a transparency path writes into shared frame targets.
///
/// The frame graph reads this to wire dependencies: which paths feed the TAA
/// reactive mask, which emit motion vectors, and which contribute to the ray
/// tracing scene for refraction/reflection.
#[must_use]
pub fn outputs_for(path: TransparencyPath) -> TransparencyOutputs {
    // Every transparent path feeds the TAA reactive mask. Paths differ in
    // whether they emit clean motion vectors (order-dependent passes do) and
    // whether they contribute to the ray scene for refraction/reflection.
    match path {
        // Order-dependent blends that stay opaque-like for motion vectors.
        TransparencyPath::Sorted | TransparencyPath::HairVisibility => TransparencyOutputs {
            writes_reactive_mask: true,
            writes_motion: true,
            contributes_to_ray_scene: false,
        },
        // Order-independent resolves cannot produce coherent motion vectors.
        TransparencyPath::WeightedOit | TransparencyPath::MomentOit => TransparencyOutputs {
            writes_reactive_mask: true,
            writes_motion: false,
            contributes_to_ray_scene: false,
        },
        // Refractive/reflective surfaces feed the ray scene and move.
        TransparencyPath::LayeredGlass | TransparencyPath::SingleLayerWater => {
            TransparencyOutputs {
                writes_reactive_mask: true,
                writes_motion: true,
                contributes_to_ray_scene: true,
            }
        }
        // Marched media feed the ray scene but have no surface motion.
        TransparencyPath::Volumetric => TransparencyOutputs {
            writes_reactive_mask: true,
            writes_motion: false,
            contributes_to_ray_scene: true,
        },
    }
}

/// Transparent draws partitioned by the compositing path each one takes.
///
/// Each bucket holds draw indices in first-seen order so submission stays
/// deterministic. The backend runs each non-empty path as its own resolve,
/// reading [`outputs_for`] to schedule shared-target writes.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct TransparencyBins {
    /// Back-to-front alpha-blended draws.
    pub sorted: Vec<u32>,
    /// Weighted-blended order-independent draws.
    pub weighted_oit: Vec<u32>,
    /// Moment-based order-independent draws.
    pub moment_oit: Vec<u32>,
    /// Multi-layer refractive glass draws.
    pub layered_glass: Vec<u32>,
    /// Single-layer water draws.
    pub single_layer_water: Vec<u32>,
    /// Hair/fur visibility draws.
    pub hair_visibility: Vec<u32>,
    /// Participating-media volume draws.
    pub volumetric: Vec<u32>,
}

impl TransparencyBins {
    /// Total number of draws across every bucket.
    #[must_use]
    pub fn total(&self) -> usize {
        self.sorted.len()
            + self.weighted_oit.len()
            + self.moment_oit.len()
            + self.layered_glass.len()
            + self.single_layer_water.len()
            + self.hair_visibility.len()
            + self.volumetric.len()
    }

    /// Returns `true` when no draw landed in any bucket.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.total() == 0
    }

    /// Immutable view of the bucket backing a given path.
    #[must_use]
    pub fn bucket(&self, path: TransparencyPath) -> &[u32] {
        match path {
            TransparencyPath::Sorted => &self.sorted,
            TransparencyPath::WeightedOit => &self.weighted_oit,
            TransparencyPath::MomentOit => &self.moment_oit,
            TransparencyPath::LayeredGlass => &self.layered_glass,
            TransparencyPath::SingleLayerWater => &self.single_layer_water,
            TransparencyPath::HairVisibility => &self.hair_visibility,
            TransparencyPath::Volumetric => &self.volumetric,
        }
    }

    /// Appends a draw index to the bucket backing `path`.
    pub fn push(&mut self, path: TransparencyPath, draw: u32) {
        match path {
            TransparencyPath::Sorted => self.sorted.push(draw),
            TransparencyPath::WeightedOit => self.weighted_oit.push(draw),
            TransparencyPath::MomentOit => self.moment_oit.push(draw),
            TransparencyPath::LayeredGlass => self.layered_glass.push(draw),
            TransparencyPath::SingleLayerWater => self.single_layer_water.push(draw),
            TransparencyPath::HairVisibility => self.hair_visibility.push(draw),
            TransparencyPath::Volumetric => self.volumetric.push(draw),
        }
    }
}

/// Partitions transparent draws into per-path buckets.
///
/// `surfaces[i]` describes draw `i`; the two slices are parallel. A draw whose
/// index exceeds `surfaces` is skipped rather than panicking, so a stale draw
/// list cannot crash submission.
#[must_use]
pub fn bin_transparent_draws(
    draws: &[u32],
    surfaces: &[TransparentSurface],
    capability: TransparencyCapability,
) -> TransparencyBins {
    let mut bins = TransparencyBins::default();
    for (slot, &draw) in draws.iter().enumerate() {
        let Some(&surface) = surfaces.get(slot) else {
            continue;
        };
        bins.push(select_transparency_path(surface, capability), draw);
    }
    bins
}

#[cfg(test)]
mod tests {
    use super::*;

    const NO_MOMENT: TransparencyCapability = TransparencyCapability { moment_oit: false };
    const WITH_MOMENT: TransparencyCapability = TransparencyCapability { moment_oit: true };

    fn kind(kind: TransparentKind) -> TransparentSurface {
        TransparentSurface {
            kind,
            ..TransparentSurface::general()
        }
    }

    #[test]
    fn dedicated_kinds_take_their_paths() {
        assert_eq!(
            select_transparency_path(kind(TransparentKind::Water), NO_MOMENT),
            TransparencyPath::SingleLayerWater
        );
        assert_eq!(
            select_transparency_path(kind(TransparentKind::Hair), NO_MOMENT),
            TransparencyPath::HairVisibility
        );
        assert_eq!(
            select_transparency_path(kind(TransparentKind::Volume), NO_MOMENT),
            TransparencyPath::Volumetric
        );
    }

    #[test]
    fn glass_layer_count_picks_sorted_or_layered() {
        let single = TransparentSurface {
            kind: TransparentKind::Glass,
            layer_count: 1,
            ..TransparentSurface::general()
        };
        let stacked = TransparentSurface {
            layer_count: 3,
            ..single
        };
        assert_eq!(
            select_transparency_path(single, NO_MOMENT),
            TransparencyPath::Sorted
        );
        assert_eq!(
            select_transparency_path(stacked, NO_MOMENT),
            TransparencyPath::LayeredGlass
        );
    }

    #[test]
    fn general_surface_sorts_unless_order_independent() {
        assert_eq!(
            select_transparency_path(TransparentSurface::general(), NO_MOMENT),
            TransparencyPath::Sorted
        );
        let oit = TransparentSurface {
            order_independent: true,
            ..TransparentSurface::general()
        };
        assert_eq!(
            select_transparency_path(oit, NO_MOMENT),
            TransparencyPath::WeightedOit
        );
    }

    #[test]
    fn moment_oit_needs_both_request_and_capability() {
        let hi = TransparentSurface {
            order_independent: true,
            high_fidelity: true,
            ..TransparentSurface::general()
        };
        // Requested but unsupported -> weighted fallback.
        assert_eq!(
            select_transparency_path(hi, NO_MOMENT),
            TransparencyPath::WeightedOit
        );
        // Requested and supported -> moment OIT.
        assert_eq!(
            select_transparency_path(hi, WITH_MOMENT),
            TransparencyPath::MomentOit
        );
    }

    #[test]
    fn ray_scene_contributors_are_glass_water_and_volume() {
        assert!(outputs_for(TransparencyPath::LayeredGlass).contributes_to_ray_scene);
        assert!(outputs_for(TransparencyPath::SingleLayerWater).contributes_to_ray_scene);
        assert!(outputs_for(TransparencyPath::Volumetric).contributes_to_ray_scene);
        assert!(!outputs_for(TransparencyPath::Sorted).contributes_to_ray_scene);
        assert!(!outputs_for(TransparencyPath::WeightedOit).contributes_to_ray_scene);
    }

    #[test]
    fn oit_paths_do_not_write_motion() {
        assert!(!outputs_for(TransparencyPath::WeightedOit).writes_motion);
        assert!(!outputs_for(TransparencyPath::MomentOit).writes_motion);
        assert!(outputs_for(TransparencyPath::Sorted).writes_motion);
    }

    #[test]
    fn bin_routes_and_preserves_order() {
        let surfaces = [
            TransparentSurface::general(),
            kind(TransparentKind::Water),
            TransparentSurface::general(),
        ];
        let draws = [10, 11, 12];
        let bins = bin_transparent_draws(&draws, &surfaces, NO_MOMENT);
        assert_eq!(bins.sorted, [10, 12]);
        assert_eq!(bins.single_layer_water, [11]);
        assert_eq!(bins.total(), 3);
    }

    #[test]
    fn bin_skips_draws_without_a_surface() {
        let surfaces = [TransparentSurface::general()];
        let draws = [0, 1];
        let bins = bin_transparent_draws(&draws, &surfaces, NO_MOMENT);
        assert_eq!(bins.total(), 1);
        assert_eq!(bins.sorted, [0]);
    }

    #[test]
    fn empty_is_empty() {
        let bins = bin_transparent_draws(&[], &[], NO_MOMENT);
        assert!(bins.is_empty());
    }
}
