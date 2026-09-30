//! Tunables for the `GPU` Voronoi fragment-assignment classifier.
//!
//! The classifier bins a large point cloud (debris particles, surface samples,
//! or voxel centres) into the Voronoi cells of a set of fracture *seed sites*,
//! reporting for each point which cell (fragment) owns it and how far the point
//! sits from the nearest cell wall. This is the mass-parallel companion to the
//! branchy convex carving in [`prism_physics_core::fracture`]: the carving
//! decides the fragment *shapes* on the `CPU`, while this classifier assigns
//! millions of sample points to those fragments on the `GPU`.
//!
//! # Provenance
//!
//! This module contains **no Unreal Engine source or derived code**. Nearest-
//! site Voronoi membership and perpendicular-bisector cell walls are standard,
//! publicly documented computational-geometry results.

/// The sentinel cell index reported for a point when no seed sites were
/// supplied, so callers can filter unassigned points without a separate flag.
///
/// It is [`u32::MAX`], matching the classifier's `WGSL` sentinel exactly.
pub const NO_CELL: u32 = u32::MAX;

/// Parameters controlling a batched Voronoi assignment dispatch.
///
/// The defaults suit debris binning where the seed cloud is well separated; the
/// only knob is the degeneracy tolerance used when two sites nearly coincide.
#[derive(Clone, Copy, PartialEq, Debug)]
pub struct VoronoiAssignConfig {
    /// Squared-length threshold below which a bisector between two nearly
    /// coincident sites is treated as degenerate and skipped, mirroring the
    /// zero-normal guard in [`prism_physics_core::fracture::Plane`].
    pub degenerate_eps: f32,
}

impl VoronoiAssignConfig {
    /// Default degeneracy tolerance (`1e-12`), matching the effective guard the
    /// `CPU` golden inherits from `glam`'s `normalize_or_zero`.
    pub const DEFAULT_DEGENERATE_EPS: f32 = 1.0e-12;
}

impl Default for VoronoiAssignConfig {
    fn default() -> VoronoiAssignConfig {
        VoronoiAssignConfig {
            degenerate_eps: Self::DEFAULT_DEGENERATE_EPS,
        }
    }
}
