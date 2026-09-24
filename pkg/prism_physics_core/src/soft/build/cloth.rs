//! Rectangular cloth builder.
//!
//! [`ClothGrid`] authors a rectangular sheet of particles laid out on the world
//! XZ plane and wires the three constraint families that give cloth its
//! behaviour:
//!
//! - **Structural** distance constraints along every grid edge (horizontal and
//!   vertical) resist stretching.
//! - **Shear** distance constraints along both diagonals of every cell resist
//!   in-plane shearing.
//! - **Bending** constraints along every row and column triple resist
//!   out-of-plane folding.
//!
//! The result is returned as a [`Cloth`], which keeps the [`SoftBody`] together
//! with the grid of particle handles so callers can pin corners, read the
//! draped shape, or attach the sheet to moving anchors.
//!
//! # Provenance
//!
//! This module contains **no Unreal Engine source or derived code**. The
//! structural / shear / bending mass-spring layout of a cloth grid is a
//! standard, publicly documented cloth-simulation topology.

use glam::Vec3;

use crate::math::scalar::Real;
use crate::soft::body::SoftBody;
use crate::soft::constraint::{BendingConstraint, DistanceConstraint};
use crate::soft::particle::ParticleHandle;
use crate::soft::solver::SoftSolverConfig;

/// Description of a rectangular cloth sheet to build.
///
/// Particles are placed in a `columns` by `rows` grid on the XZ plane, spaced
/// `spacing` metres apart, with the `(0, 0)` corner at `origin`. Column index
/// increases along `+X` and row index increases along `+Z`.
#[derive(Clone, Copy, PartialEq, Debug)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub struct ClothGrid {
    /// Number of particles along the `X` axis (must be at least 2 to form an
    /// edge).
    pub columns: u32,
    /// Number of particles along the `Z` axis (must be at least 2 to form an
    /// edge).
    pub rows: u32,
    /// Rest spacing between adjacent particles, in metres.
    pub spacing: Real,
    /// Mass of each particle, in kilograms.
    pub particle_mass: Real,
    /// Compliance of the structural and shear distance constraints.
    pub compliance: Real,
    /// Compliance of the bending constraints (typically larger/softer than
    /// [`compliance`](Self::compliance)).
    pub bending_compliance: Real,
    /// World position of the `(row = 0, column = 0)` corner.
    pub origin: Vec3,
}

impl Default for ClothGrid {
    fn default() -> Self {
        ClothGrid {
            columns: 16,
            rows: 16,
            spacing: 0.1,
            particle_mass: 0.1,
            compliance: 0.0,
            bending_compliance: 0.02,
            origin: Vec3::ZERO,
        }
    }
}

impl ClothGrid {
    /// Builds the cloth into a [`Cloth`] using the given solver configuration.
    ///
    /// Grids smaller than `2 x 2` still produce every valid particle and any
    /// constraints that fit; degenerate cases simply yield fewer constraints.
    #[must_use]
    pub fn build(&self, config: SoftSolverConfig) -> Cloth {
        let columns = self.columns.max(1);
        let rows = self.rows.max(1);
        let mut body = SoftBody::new(config);
        let mut handles = Vec::with_capacity((columns * rows) as usize);

        for row in 0..rows {
            for column in 0..columns {
                let position = self.origin
                    + Vec3::new(
                        column as Real * self.spacing,
                        0.0,
                        row as Real * self.spacing,
                    );
                handles.push(body.spawn(position, self.particle_mass));
            }
        }

        let index = |row: u32, column: u32| (row * columns + column) as usize;
        let positions = |body: &SoftBody, h: ParticleHandle| body.particles.positions()[h.index()];

        // Structural + shear distance constraints.
        for row in 0..rows {
            for column in 0..columns {
                let here = handles[index(row, column)];
                if column + 1 < columns {
                    let right = handles[index(row, column + 1)];
                    let rest = (positions(&body, here) - positions(&body, right)).length();
                    body.add_distance(DistanceConstraint::new(here, right, rest, self.compliance));
                }
                if row + 1 < rows {
                    let down = handles[index(row + 1, column)];
                    let rest = (positions(&body, here) - positions(&body, down)).length();
                    body.add_distance(DistanceConstraint::new(here, down, rest, self.compliance));
                }
                if column + 1 < columns && row + 1 < rows {
                    let here_h = handles[index(row, column)];
                    let right = handles[index(row, column + 1)];
                    let down = handles[index(row + 1, column)];
                    let diag = handles[index(row + 1, column + 1)];
                    let rest_main = (positions(&body, here_h) - positions(&body, diag)).length();
                    body.add_distance(DistanceConstraint::new(
                        here_h,
                        diag,
                        rest_main,
                        self.compliance,
                    ));
                    let rest_anti = (positions(&body, right) - positions(&body, down)).length();
                    body.add_distance(DistanceConstraint::new(
                        right,
                        down,
                        rest_anti,
                        self.compliance,
                    ));
                }
            }
        }

        // Bending constraints along each row and column triple.
        for row in 0..rows {
            for column in 0..columns {
                if column + 2 < columns {
                    let a = handles[index(row, column)];
                    let center = handles[index(row, column + 1)];
                    let b = handles[index(row, column + 2)];
                    if let Some(c) = BendingConstraint::from_positions(
                        a,
                        center,
                        b,
                        body.particles.positions(),
                        self.bending_compliance,
                    ) {
                        body.add_bending(c);
                    }
                }
                if row + 2 < rows {
                    let a = handles[index(row, column)];
                    let center = handles[index(row + 1, column)];
                    let b = handles[index(row + 2, column)];
                    if let Some(c) = BendingConstraint::from_positions(
                        a,
                        center,
                        b,
                        body.particles.positions(),
                        self.bending_compliance,
                    ) {
                        body.add_bending(c);
                    }
                }
            }
        }

        Cloth {
            body,
            handles,
            columns,
            rows,
        }
    }

    /// Builds the cloth with the default solver configuration.
    #[must_use]
    pub fn build_default(&self) -> Cloth {
        self.build(SoftSolverConfig::default())
    }
}

/// A built cloth sheet: its [`SoftBody`] plus the grid of particle handles.
#[derive(Clone, PartialEq, Debug)]
pub struct Cloth {
    /// The simulated body. Step it with [`SoftBody::step`].
    pub body: SoftBody,
    /// Row-major grid of particle handles, length `columns * rows`.
    handles: Vec<ParticleHandle>,
    /// Number of particles along the `X` axis.
    columns: u32,
    /// Number of particles along the `Z` axis.
    rows: u32,
}

impl Cloth {
    /// Returns the number of particle columns (`X` extent).
    #[must_use]
    pub fn columns(&self) -> u32 {
        self.columns
    }

    /// Returns the number of particle rows (`Z` extent).
    #[must_use]
    pub fn rows(&self) -> u32 {
        self.rows
    }

    /// Returns the handle of the particle at `(row, column)`, or `None` if the
    /// indices are out of range.
    #[must_use]
    pub fn handle(&self, row: u32, column: u32) -> Option<ParticleHandle> {
        if row >= self.rows || column >= self.columns {
            return None;
        }
        self.handles
            .get((row * self.columns + column) as usize)
            .copied()
    }

    /// Pins the particle at `(row, column)` so it becomes an immovable anchor.
    /// Returns `false` if the indices are out of range.
    pub fn pin(&mut self, row: u32, column: u32) -> bool {
        match self.handle(row, column) {
            Some(h) => {
                self.body.particles.pin(h);
                true
            }
            None => false,
        }
    }

    /// Advances the cloth by `dt` seconds.
    pub fn step(&mut self, dt: Real) {
        self.body.step(dt);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn grid_spawns_expected_particle_count() {
        let cloth = ClothGrid {
            columns: 4,
            rows: 3,
            ..ClothGrid::default()
        }
        .build_default();
        assert_eq!(cloth.body.particles.len(), 12);
        assert_eq!(cloth.columns(), 4);
        assert_eq!(cloth.rows(), 3);
    }

    #[test]
    fn grid_wires_structural_shear_and_bending_counts() {
        // 3x3 grid: horizontal edges = rows*(cols-1)=3*2=6, vertical = 6,
        // shear = 2 per cell * (2*2 cells) = 8 -> distance = 20.
        // bending: row triples = rows*(cols-2)=3*1=3, col triples = 3 -> 6.
        let cloth = ClothGrid {
            columns: 3,
            rows: 3,
            ..ClothGrid::default()
        }
        .build_default();
        assert_eq!(cloth.body.constraints.distance.len(), 20);
        assert_eq!(cloth.body.constraints.bending.len(), 6);
    }

    #[test]
    fn handle_lookup_respects_bounds() {
        let cloth = ClothGrid {
            columns: 3,
            rows: 2,
            ..ClothGrid::default()
        }
        .build_default();
        assert!(cloth.handle(0, 0).is_some());
        assert!(cloth.handle(1, 2).is_some());
        assert!(cloth.handle(2, 0).is_none());
        assert!(cloth.handle(0, 3).is_none());
    }

    #[test]
    fn pin_marks_particle_immovable() {
        let mut cloth = ClothGrid {
            columns: 3,
            rows: 3,
            ..ClothGrid::default()
        }
        .build_default();
        assert!(cloth.pin(0, 0));
        let h = cloth.handle(0, 0).unwrap();
        assert!(cloth.body.particles.is_pinned(h));
        assert!(!cloth.pin(9, 9));
    }

    #[test]
    fn pinned_corners_keep_edge_lengths_and_settle() {
        // Interactive cloth deliverable: pin two corners of one edge, drape
        // under gravity, and confirm edges stay near their rest length.
        let grid = ClothGrid {
            columns: 8,
            rows: 8,
            spacing: 0.1,
            particle_mass: 0.05,
            compliance: 0.0,
            bending_compliance: 0.02,
            origin: Vec3::ZERO,
        };
        let mut cloth = grid.build_default();
        cloth.pin(0, 0);
        cloth.pin(0, grid.columns - 1);

        for _ in 0..300 {
            cloth.step(1.0 / 60.0);
        }

        // Every position must stay finite (no blow-up).
        for &p in cloth.body.particles.positions() {
            assert!(p.is_finite(), "non-finite particle {p:?}");
        }

        // Structural edges should stay near the 0.1 m rest length.
        let mut max_ratio: Real = 0.0;
        for c in &cloth.body.constraints.distance {
            if c.rest_length <= 0.0 {
                continue;
            }
            let pa = cloth.body.particles.position(c.a).unwrap();
            let pb = cloth.body.particles.position(c.b).unwrap();
            let ratio = (pa - pb).length() / c.rest_length;
            max_ratio = max_ratio.max(ratio);
        }
        assert!(max_ratio < 1.5, "edge stretched {max_ratio}x rest length");

        // Pinned corners must not have moved.
        assert_eq!(
            cloth.body.particles.position(cloth.handle(0, 0).unwrap()),
            Some(Vec3::ZERO)
        );
    }
}
