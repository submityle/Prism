//! Voronoi decomposition of a convex solid into fracture fragments.
//!
//! Each seed [`site`](crate::fracture::pattern) owns the region of space closer
//! to it than to any other site. Inside a convex bounding shape, that region is
//! itself convex: it is the intersection of the bounding shape's half-spaces
//! with, for every other site, the perpendicular-bisector half-space that keeps
//! the near side (see [`Plane::bisector`]). Carving one convex cell per site and
//! filtering away sub-threshold slivers yields a watertight partition of the
//! solid into convex [`Fragment`]s ready to become dynamic rigid bodies.
//!
//! The public entry points build progressively: [`fracture_convex`] carves an
//! arbitrary convex bound, [`fracture_aabb`] wraps an axis-aligned box, and
//! [`shatter_box`] / [`shatter_box_impact`] additionally scatter the seed sites
//! from a [`FractureConfig`] so a caller can shatter a box in one call.
//!
//! # Provenance
//!
//! This module contains **no Unreal Engine source or derived code**. Bounded
//! Voronoi decomposition by per-site half-space intersection is a standard,
//! publicly documented computational-geometry construction.

use glam::Vec3;

use crate::fracture::config::FractureConfig;
use crate::fracture::fragment::Fragment;
use crate::fracture::pattern::{scatter_impact, scatter_uniform};
use crate::fracture::plane::Plane;
use crate::fracture::polyhedron::ConvexPolyhedron;

/// Decomposes the convex `bounds` into one convex [`Fragment`] per site.
///
/// For each site the cell is the intersection of the `bounds` face planes with
/// the bisector half-space to every other site. Cells that are empty,
/// degenerate, or below `config.min_fragment_volume` are dropped, so the
/// returned fragments tile `bounds` up to those sliver rejections.
///
/// With zero or one site the whole `bounds` is returned as a single fragment
/// (there is nothing to cut against).
#[must_use]
pub fn fracture_convex(
    bounds: &ConvexPolyhedron,
    sites: &[Vec3],
    config: &FractureConfig,
) -> Vec<Fragment> {
    if sites.len() <= 1 {
        let site = sites.first().copied().unwrap_or_else(|| bounds.centroid());
        return vec![Fragment::new(bounds.clone(), site)];
    }

    let base_planes = bounds.face_planes();
    let mut fragments = Vec::with_capacity(sites.len());

    for (i, &site) in sites.iter().enumerate() {
        let mut planes = base_planes.clone();
        for (j, &other) in sites.iter().enumerate() {
            if i == j {
                continue;
            }
            planes.push(Plane::bisector(site, other));
        }

        let Some(cell) = ConvexPolyhedron::from_halfspaces(&planes) else {
            continue;
        };
        if cell.volume() < config.min_fragment_volume {
            continue;
        }
        fragments.push(Fragment::new(cell, site));
    }

    fragments
}

/// Decomposes the axis-aligned box `min..=max` into convex fragments for the
/// given `sites`.
#[must_use]
pub fn fracture_aabb(
    min: Vec3,
    max: Vec3,
    sites: &[Vec3],
    config: &FractureConfig,
) -> Vec<Fragment> {
    let bounds = ConvexPolyhedron::box_aabb(min, max);
    fracture_convex(&bounds, sites, config)
}

/// Shatters the axis-aligned box `min..=max` with uniformly scattered seeds.
///
/// This is the one-call convenience path: it scatters `config.seed_count` sites
/// uniformly (see [`scatter_uniform`]) and fractures the box against them.
#[must_use]
pub fn shatter_box(min: Vec3, max: Vec3, config: &FractureConfig) -> Vec<Fragment> {
    let sites = scatter_uniform(min, max, config);
    fracture_aabb(min, max, &sites, config)
}

/// Shatters the axis-aligned box `min..=max` with seeds clustered around
/// `impact`, producing many small fragments near the contact and larger ones
/// away from it (see [`scatter_impact`]).
#[must_use]
pub fn shatter_box_impact(
    min: Vec3,
    max: Vec3,
    impact: Vec3,
    config: &FractureConfig,
) -> Vec<Fragment> {
    let sites = scatter_impact(min, max, impact, config);
    fracture_aabb(min, max, &sites, config)
}
