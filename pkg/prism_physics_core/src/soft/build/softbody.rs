//! Volumetric soft-body builder (tetrahedral box lattice).
//!
//! [`SoftBoxGrid`] authors a solid rectangular block of particles arranged on
//! a regular 3D grid and fills it with tetrahedra so the interior resists both
//! stretching (distance constraints along every unique tetrahedron edge) and
//! change of volume (a [`TetraVolumeConstraint`] per tetrahedron).
//!
//! Each grid cell (a cube of eight corner particles) is split into six
//! tetrahedra using the standard *Kuhn / Freudenthal* decomposition, in which
//! all six tetrahedra share the cube's main diagonal from corner `0` to corner
//! `7`. Corner indices encode their axis offsets as bits: `bit0 = x`,
//! `bit1 = y`, `bit2 = z`, so corner `0` is `(0, 0, 0)` and corner `7` is
//! `(1, 1, 1)`.
//!
//! The result is a [`SoftBox`], which keeps the [`SoftBody`] together with the
//! grid of particle handles so callers can pin faces, attach anchors, or read
//! the deformed shape.
//!
//! # Provenance
//!
//! This module contains **no Unreal Engine source or derived code**. The
//! six-tetrahedron Kuhn decomposition of a cube and volume-preserving
//! tetrahedral constraints are standard, publicly documented finite-element /
//! position-based-dynamics techniques.

use glam::Vec3;

use crate::math::scalar::Real;
use crate::soft::body::SoftBody;
use crate::soft::constraint::{DistanceConstraint, TetraVolumeConstraint};
use crate::soft::particle::ParticleHandle;
use crate::soft::solver::SoftSolverConfig;

/// The six tetrahedra of a cube's Kuhn decomposition, expressed as local corner
/// indices (see the module docs for the bit encoding). All six share the main
/// diagonal `0 -> 7`.
const KUHN_TETRAHEDRA: [[u32; 4]; 6] = [
    [0, 1, 3, 7],
    [0, 1, 5, 7],
    [0, 2, 3, 7],
    [0, 2, 6, 7],
    [0, 4, 5, 7],
    [0, 4, 6, 7],
];

/// The six edges of a tetrahedron as index pairs into its four corners.
const TETRAHEDRON_EDGES: [[usize; 2]; 6] = [[0, 1], [0, 2], [0, 3], [1, 2], [1, 3], [2, 3]];

/// Description of a solid rectangular soft body to build.
///
/// Particles occupy a regular grid with `cells_x + 1` particles along `X`
/// (and likewise for `Y` and `Z`), spaced `spacing` metres apart, with the
/// `(0, 0, 0)` corner at `origin`.
#[derive(Clone, Copy, PartialEq, Debug)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub struct SoftBoxGrid {
    /// Number of cells along the `X` axis (particles along `X` is this plus one).
    pub cells_x: u32,
    /// Number of cells along the `Y` axis (particles along `Y` is this plus one).
    pub cells_y: u32,
    /// Number of cells along the `Z` axis (particles along `Z` is this plus one).
    pub cells_z: u32,
    /// Rest spacing between adjacent particles, in metres.
    pub spacing: Real,
    /// Mass of each particle, in kilograms.
    pub particle_mass: Real,
    /// Compliance of the structural edge (distance) constraints.
    pub compliance: Real,
    /// Compliance of the tetrahedral volume constraints.
    pub volume_compliance: Real,
    /// World position of the `(0, 0, 0)` corner.
    pub origin: Vec3,
}

impl Default for SoftBoxGrid {
    fn default() -> Self {
        SoftBoxGrid {
            cells_x: 4,
            cells_y: 4,
            cells_z: 4,
            spacing: 0.1,
            particle_mass: 0.1,
            compliance: 0.0,
            volume_compliance: 0.0,
            origin: Vec3::ZERO,
        }
    }
}

impl SoftBoxGrid {
    /// Builds the soft body into a [`SoftBox`] using the given solver config.
    #[must_use]
    pub fn build(&self, config: SoftSolverConfig) -> SoftBox {
        let nx = self.cells_x.saturating_add(1).max(1);
        let ny = self.cells_y.saturating_add(1).max(1);
        let nz = self.cells_z.saturating_add(1).max(1);

        let mut body = SoftBody::new(config);
        let mut handles = Vec::with_capacity((nx * ny * nz) as usize);

        for z in 0..nz {
            for y in 0..ny {
                for x in 0..nx {
                    let position = self.origin
                        + Vec3::new(
                            x as Real * self.spacing,
                            y as Real * self.spacing,
                            z as Real * self.spacing,
                        );
                    handles.push(body.spawn(position, self.particle_mass));
                }
            }
        }

        let index = |x: u32, y: u32, z: u32| ((z * ny + y) * nx + x) as usize;

        // Collect unique lattice edges as sorted particle-index pairs so the
        // same shared edge is not constrained twice. A plain sorted/deduped Vec
        // keeps the output deterministic (no hash-set ordering).
        let mut edges: Vec<(u32, u32)> = Vec::new();

        for cz in 0..self.cells_z {
            for cy in 0..self.cells_y {
                for cx in 0..self.cells_x {
                    // Resolve the eight corner handles of this cell.
                    let mut corner = [ParticleHandle::INVALID; 8];
                    for (local, slot) in corner.iter_mut().enumerate() {
                        let dx = (local & 1) as u32;
                        let dy = ((local >> 1) & 1) as u32;
                        let dz = ((local >> 2) & 1) as u32;
                        *slot = handles[index(cx + dx, cy + dy, cz + dz)];
                    }

                    for tet in &KUHN_TETRAHEDRA {
                        let particles = [
                            corner[tet[0] as usize],
                            corner[tet[1] as usize],
                            corner[tet[2] as usize],
                            corner[tet[3] as usize],
                        ];
                        if let Some(c) = TetraVolumeConstraint::from_positions(
                            particles,
                            body.particles.positions(),
                            self.volume_compliance,
                        ) {
                            body.add_volume(c);
                        }

                        for edge in &TETRAHEDRON_EDGES {
                            let ra = particles[edge[0]].raw();
                            let rb = particles[edge[1]].raw();
                            edges.push((ra.min(rb), ra.max(rb)));
                        }
                    }
                }
            }
        }

        edges.sort_unstable();
        edges.dedup();

        for (ra, rb) in edges {
            let a = ParticleHandle::from_index(ra);
            let b = ParticleHandle::from_index(rb);
            if let Some(c) = DistanceConstraint::from_positions(
                a,
                b,
                body.particles.positions(),
                self.compliance,
            ) {
                body.add_distance(c);
            }
        }

        SoftBox {
            body,
            handles,
            nx,
            ny,
            nz,
        }
    }

    /// Builds the soft body with the default solver configuration.
    #[must_use]
    pub fn build_default(&self) -> SoftBox {
        self.build(SoftSolverConfig::default())
    }
}

/// A built volumetric soft body: its [`SoftBody`] plus the particle grid.
#[derive(Clone, PartialEq, Debug)]
pub struct SoftBox {
    /// The simulated body. Step it with [`SoftBody::step`].
    pub body: SoftBody,
    /// Grid-ordered particle handles, length `nx * ny * nz`.
    handles: Vec<ParticleHandle>,
    /// Number of particles along `X`.
    nx: u32,
    /// Number of particles along `Y`.
    ny: u32,
    /// Number of particles along `Z`.
    nz: u32,
}

impl SoftBox {
    /// Returns the number of particles along `X`.
    #[must_use]
    pub fn particles_x(&self) -> u32 {
        self.nx
    }

    /// Returns the number of particles along `Y`.
    #[must_use]
    pub fn particles_y(&self) -> u32 {
        self.ny
    }

    /// Returns the number of particles along `Z`.
    #[must_use]
    pub fn particles_z(&self) -> u32 {
        self.nz
    }

    /// Returns the handle of the particle at grid coordinate `(x, y, z)`, or
    /// `None` if any index is out of range.
    #[must_use]
    pub fn handle(&self, x: u32, y: u32, z: u32) -> Option<ParticleHandle> {
        if x >= self.nx || y >= self.ny || z >= self.nz {
            return None;
        }
        let i = ((z * self.ny + y) * self.nx + x) as usize;
        self.handles.get(i).copied()
    }

    /// Pins the particle at grid coordinate `(x, y, z)` so it becomes an
    /// immovable anchor. Returns `false` if any index is out of range.
    pub fn pin(&mut self, x: u32, y: u32, z: u32) -> bool {
        match self.handle(x, y, z) {
            Some(h) => {
                self.body.particles.pin(h);
                true
            }
            None => false,
        }
    }

    /// Advances the soft body by `dt` seconds.
    pub fn step(&mut self, dt: Real) {
        self.body.step(dt);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn single_cell_has_six_tetrahedra_and_all_cube_edges() {
        // One cell -> 8 particles, 6 volume constraints. The unique edges of the
        // six Kuhn tetrahedra are the 12 cube edges, 6 face diagonals, and the
        // single shared main diagonal = 19 distance constraints.
        let solid = SoftBoxGrid {
            cells_x: 1,
            cells_y: 1,
            cells_z: 1,
            ..SoftBoxGrid::default()
        }
        .build_default();
        assert_eq!(solid.body.particles.len(), 8);
        assert_eq!(solid.body.constraints.volume.len(), 6);
        assert_eq!(solid.body.constraints.distance.len(), 19);
    }

    #[test]
    fn particle_count_matches_grid() {
        let solid = SoftBoxGrid {
            cells_x: 2,
            cells_y: 3,
            cells_z: 1,
            ..SoftBoxGrid::default()
        }
        .build_default();
        // (2+1)*(3+1)*(1+1) = 24 particles.
        assert_eq!(solid.body.particles.len(), 24);
        assert_eq!(solid.particles_x(), 3);
        assert_eq!(solid.particles_y(), 4);
        assert_eq!(solid.particles_z(), 2);
    }

    #[test]
    fn shared_edges_are_not_duplicated() {
        // Two stacked cells share a face; every shared edge must appear once.
        let solid = SoftBoxGrid {
            cells_x: 2,
            cells_y: 1,
            cells_z: 1,
            ..SoftBoxGrid::default()
        }
        .build_default();
        // 12 volume constraints (6 per cell), and no duplicate distance edges.
        assert_eq!(solid.body.constraints.volume.len(), 12);
        let mut seen: Vec<(u32, u32)> = solid
            .body
            .constraints
            .distance
            .iter()
            .map(|c| {
                let ra = c.a.raw();
                let rb = c.b.raw();
                (ra.min(rb), ra.max(rb))
            })
            .collect();
        let total = seen.len();
        seen.sort_unstable();
        seen.dedup();
        assert_eq!(seen.len(), total, "duplicate distance edges present");
    }

    #[test]
    fn handle_lookup_respects_bounds() {
        let solid = SoftBoxGrid {
            cells_x: 1,
            cells_y: 1,
            cells_z: 1,
            ..SoftBoxGrid::default()
        }
        .build_default();
        assert!(solid.handle(0, 0, 0).is_some());
        assert!(solid.handle(1, 1, 1).is_some());
        assert!(solid.handle(2, 0, 0).is_none());
    }

    #[test]
    fn pinned_base_keeps_volume_and_stays_finite() {
        // Pin the whole bottom face, drop under gravity, and confirm the block
        // neither blows up nor collapses: total tetra volume stays positive and
        // close to its rest value.
        let grid = SoftBoxGrid {
            cells_x: 2,
            cells_y: 2,
            cells_z: 2,
            spacing: 0.1,
            particle_mass: 0.05,
            compliance: 0.0,
            volume_compliance: 0.0,
            origin: Vec3::ZERO,
        };
        let mut solid = grid.build_default();
        for z in 0..solid.particles_z() {
            for x in 0..solid.particles_x() {
                solid.pin(x, 0, z);
            }
        }

        let rest_volume = tetra_volume_sum(&solid);
        for _ in 0..240 {
            solid.step(1.0 / 60.0);
        }
        for &p in solid.body.particles.positions() {
            assert!(p.is_finite(), "non-finite particle {p:?}");
        }
        let volume = tetra_volume_sum(&solid);
        assert!(volume > 0.0, "block inverted (volume {volume})");
        let ratio = volume / rest_volume;
        assert!(
            (0.5..1.5).contains(&ratio),
            "volume drifted to {ratio}x rest",
        );
    }

    /// Sums the absolute volume of every tetrahedron in the soft body.
    fn tetra_volume_sum(solid: &SoftBox) -> Real {
        let positions = solid.body.particles.positions();
        let mut sum = 0.0;
        for c in &solid.body.constraints.volume {
            let p0 = positions[c.particles[0].index()];
            let p1 = positions[c.particles[1].index()];
            let p2 = positions[c.particles[2].index()];
            let p3 = positions[c.particles[3].index()];
            let v = (p1 - p0).dot((p2 - p0).cross(p3 - p0)) / 6.0;
            sum += v.abs();
        }
        sum
    }
}
