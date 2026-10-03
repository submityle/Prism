//! Tunable parameters that steer approximate convex decomposition.
//!
//! The defaults mirror the practical presets shipped by production cookers
//! (the V-HACD family used by `PhysX`, Jolt and Chaos): a moderate voxel
//! resolution, a concavity floor expressed as a fraction of the whole mesh's
//! volume, and a hard cap on the number of output hulls so a pathological
//! concave input can never explode the collider budget.
//!
//! # Provenance
//!
//! These are plain tuning knobs for a textbook voxel decomposition and contain
//! **no Unreal Engine source or derived code**.

/// Controls the fidelity and cost of [`convex_decompose`](super::convex_decompose).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DecompositionParams {
    /// Target number of voxels along the mesh's longest axis.
    ///
    /// The solid voxelization grid is sized so its longest side holds this many
    /// cells; the other two axes get proportionally fewer. Higher values sharpen
    /// the concavity estimate and the split planes at a cubic memory/time cost.
    pub resolution: u32,

    /// Hard upper bound on the number of convex hulls produced.
    ///
    /// Once this many parts exist the recursion stops splitting even if some
    /// parts are still concave, guaranteeing a bounded collider budget.
    pub max_convex_hulls: u32,

    /// Concavity floor, as a fraction of the whole input mesh's solid volume.
    ///
    /// A part is left intact once its concavity (the volume its convex hull adds
    /// beyond the part's own solid volume) drops below this fraction of the
    /// total mesh volume. Smaller values chase tighter hulls; larger values
    /// accept coarser approximations.
    pub max_concavity: f32,

    /// Smallest part worth splitting, as a fraction of the total mesh volume.
    ///
    /// Parts below this fraction are accepted as-is regardless of concavity, so
    /// slivers never trigger further (fruitless) subdivision.
    pub min_volume_fraction: f32,

    /// Maximum binary-split recursion depth.
    ///
    /// A safety ceiling independent of [`Self::max_convex_hulls`]; prevents
    /// unbounded recursion on degenerate inputs.
    pub max_recursion_depth: u32,
}

impl Default for DecompositionParams {
    fn default() -> Self {
        Self {
            resolution: 48,
            max_convex_hulls: 32,
            max_concavity: 0.01,
            min_volume_fraction: 0.001,
            max_recursion_depth: 10,
        }
    }
}

impl DecompositionParams {
    /// A fast, coarse preset: low resolution and few hulls, for previews or
    /// gameplay props where a rough collider is acceptable.
    #[must_use]
    pub fn coarse() -> Self {
        Self {
            resolution: 24,
            max_convex_hulls: 8,
            max_concavity: 0.05,
            min_volume_fraction: 0.01,
            max_recursion_depth: 6,
        }
    }

    /// A high-fidelity preset: dense voxels and a generous hull budget, for
    /// hero assets where the collider must hug the surface tightly.
    #[must_use]
    pub fn fine() -> Self {
        Self {
            resolution: 80,
            max_convex_hulls: 64,
            max_concavity: 0.0025,
            min_volume_fraction: 0.0005,
            max_recursion_depth: 14,
        }
    }

    /// Clamps the parameters into safe, non-degenerate ranges.
    ///
    /// Resolution is floored at 2 (a single split needs at least two cells per
    /// axis), the hull cap at 1, the recursion depth at 1, and every fractional
    /// threshold into `[0, 1]`. Returns the sanitized copy.
    #[must_use]
    pub fn sanitized(self) -> Self {
        Self {
            resolution: self.resolution.max(2),
            max_convex_hulls: self.max_convex_hulls.max(1),
            max_concavity: self.max_concavity.clamp(0.0, 1.0),
            min_volume_fraction: self.min_volume_fraction.clamp(0.0, 1.0),
            max_recursion_depth: self.max_recursion_depth.max(1),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn defaults_are_sane() {
        let p = DecompositionParams::default();
        assert_eq!(p, p.sanitized());
        assert!(p.resolution >= 2);
        assert!(p.max_convex_hulls >= 1);
    }

    #[test]
    fn presets_order_by_fidelity() {
        assert!(
            DecompositionParams::coarse().resolution < DecompositionParams::default().resolution
        );
        assert!(DecompositionParams::fine().resolution > DecompositionParams::default().resolution);
        assert!(
            DecompositionParams::fine().max_concavity < DecompositionParams::coarse().max_concavity
        );
    }

    #[test]
    fn sanitize_floors_degenerate_values() {
        let p = DecompositionParams {
            resolution: 0,
            max_convex_hulls: 0,
            max_concavity: -1.0,
            min_volume_fraction: 2.0,
            max_recursion_depth: 0,
        }
        .sanitized();
        assert_eq!(p.resolution, 2);
        assert_eq!(p.max_convex_hulls, 1);
        assert_eq!(p.max_concavity, 0.0);
        assert_eq!(p.min_volume_fraction, 1.0);
        assert_eq!(p.max_recursion_depth, 1);
    }
}
