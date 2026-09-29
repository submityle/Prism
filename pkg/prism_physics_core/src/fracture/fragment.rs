//! A single convex fragment produced by fracturing a solid.
//!
//! A [`Fragment`] pairs the convex [`ConvexPolyhedron`] cell carved out of the
//! source shape with the Voronoi *site* that generated it. The cell is the
//! authoritative geometry (used for meshing, volume, and mass); the site is
//! retained because it is a stable, meaningful interior point (useful for
//! spawning the fragment's rigid body, seeding debris velocity from an impact
//! direction, or debugging the pattern).
//!
//! # Provenance
//!
//! This module contains **no Unreal Engine source or derived code**. A fragment
//! is a plain pairing of a convex polytope with its generating point.

use glam::Vec3;

use crate::fracture::mass::MassProperties;
use crate::fracture::polyhedron::ConvexPolyhedron;
use crate::math::scalar::Real;

/// One convex piece of a fractured solid: the carved cell plus its seed site.
#[derive(Clone, PartialEq, Debug)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub struct Fragment {
    cell: ConvexPolyhedron,
    site: Vec3,
}

impl Fragment {
    /// Builds a fragment from its convex `cell` and the `site` that generated
    /// it.
    #[must_use]
    pub fn new(cell: ConvexPolyhedron, site: Vec3) -> Fragment {
        Fragment { cell, site }
    }

    /// Returns the convex cell geometry of the fragment.
    #[must_use]
    pub fn cell(&self) -> &ConvexPolyhedron {
        &self.cell
    }

    /// Returns the Voronoi site that generated the fragment.
    #[must_use]
    pub fn site(&self) -> Vec3 {
        self.site
    }

    /// Returns the enclosed volume of the fragment cell.
    #[must_use]
    pub fn volume(&self) -> Real {
        self.cell.volume()
    }

    /// Returns the volumetric centroid of the fragment cell.
    ///
    /// Note this is the true centre of mass of the solid cell, which in general
    /// differs from the generating [`Fragment::site`].
    #[must_use]
    pub fn centroid(&self) -> Vec3 {
        self.cell.centroid()
    }

    /// Computes the full rigid-body mass properties of the fragment filled at
    /// uniform `density`.
    #[must_use]
    pub fn mass_properties(&self, density: Real) -> MassProperties {
        MassProperties::from_polyhedron(&self.cell, density)
    }
}
