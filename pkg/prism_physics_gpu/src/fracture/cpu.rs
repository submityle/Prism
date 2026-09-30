//! `CPU` golden twin for the `GPU` Voronoi fragment-assignment classifier.
//!
//! [`cpu_assign_cells`] runs the identical arithmetic the `WGSL` kernel does,
//! one query point at a time, so a passing real-device parity test is direct
//! evidence that the ported kernel bins points into the same fragments as the
//! reference — not merely that the shader compiled.
//!
//! # Membership
//!
//! A point belongs to the Voronoi cell of the site nearest to it in Euclidean
//! distance; ties are broken toward the lower site index. Because the argmin is
//! taken over squared distances with a strict "closer than the current best"
//! comparison, both this twin and the kernel keep the first (lowest-index) site
//! on an exact tie, so the reported cell index is integer-exact across devices.
//!
//! # Clearance
//!
//! The reported clearance is the distance from the point to the nearest wall of
//! its owning cell. Every wall is the perpendicular bisector between the owning
//! site and a rival site (its interior half-space, `n · x <= offset`, contains
//! the owner), so an interior point has a non-positive signed distance to every
//! wall. The clearance is therefore `-max(signed_distance)` over the rivals —
//! the smallest gap to any wall. Bisector construction normalises the wall
//! normal, so the clearance carries a square root and is verified within a
//! tight tolerance rather than bit-for-bit.
//!
//! # Provenance
//!
//! This module contains **no Unreal Engine source or derived code**. It reuses
//! [`prism_physics_core::fracture::Plane::bisector`] and its signed-distance
//! algebra as the single source of truth for the cell walls.

use glam::Vec3;
use prism_physics_core::fracture::Plane;

use super::config::{VoronoiAssignConfig, NO_CELL};

/// The Voronoi cell a query point was binned into, plus its wall clearance.
#[derive(Clone, Copy, PartialEq, Debug)]
pub struct CellAssignment {
    /// Index of the owning seed site, or [`NO_CELL`] when no sites were given.
    pub cell: u32,
    /// Distance from the point to the nearest wall of its owning cell. It is
    /// [`f32::INFINITY`] when the cell has no walls (fewer than two sites).
    pub clearance: f32,
}

/// Bins every point in `points` into the Voronoi cell of the nearest site in
/// `sites`, returning one [`CellAssignment`] per point in input order.
///
/// With no sites every point is reported as [`NO_CELL`] with an infinite
/// clearance. With a single site every point belongs to cell `0` with infinite
/// clearance, since a lone cell has no walls.
#[must_use]
pub fn cpu_assign_cells(
    sites: &[Vec3],
    points: &[Vec3],
    config: &VoronoiAssignConfig,
) -> Vec<CellAssignment> {
    points
        .iter()
        .map(|&p| assign_one(sites, p, config))
        .collect()
}

/// Assigns a single query point, factored out so the twin reads as one point's
/// worth of work — exactly the body the `WGSL` kernel runs per invocation.
fn assign_one(sites: &[Vec3], p: Vec3, config: &VoronoiAssignConfig) -> CellAssignment {
    if sites.is_empty() {
        return CellAssignment {
            cell: NO_CELL,
            clearance: f32::INFINITY,
        };
    }

    // Nearest site by squared distance, strict comparison to keep the lowest
    // index on a tie.
    let mut best = 0_usize;
    let mut best_sq = length_squared(p - sites[0]);
    for (i, &s) in sites.iter().enumerate().skip(1) {
        let sq = length_squared(p - s);
        if sq < best_sq {
            best_sq = sq;
            best = i;
        }
    }

    // Clearance to the nearest wall: the least-negative bisector signed
    // distance over all rivals, negated so a positive value means "inside".
    let owner = sites[best];
    let mut worst_sd = f32::NEG_INFINITY;
    for (i, &s) in sites.iter().enumerate() {
        if i == best {
            continue;
        }
        // Skip a near-coincident rival whose bisector is degenerate, matching
        // the kernel's zero-normal guard so neither side invents a wall.
        if length_squared(s - owner) <= config.degenerate_eps {
            continue;
        }
        let sd = Plane::bisector(owner, s).signed_distance(p);
        if sd > worst_sd {
            worst_sd = sd;
        }
    }

    let clearance = if worst_sd == f32::NEG_INFINITY {
        f32::INFINITY
    } else {
        -worst_sd
    };

    CellAssignment {
        cell: best as u32,
        clearance,
    }
}

/// Squared length written as an explicit dot product so the summation order
/// matches the `WGSL` kernel's `d.x*d.x + d.y*d.y + d.z*d.z` byte-for-byte.
fn length_squared(d: Vec3) -> f32 {
    d.x * d.x + d.y * d.y + d.z * d.z
}
