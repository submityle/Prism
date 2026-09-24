//! Rope / hair builder.
//!
//! [`RopeGrid`] authors a one-dimensional chain of particles hanging along
//! the world `-Y` axis from an anchor at the top. Neighbouring particles are
//! coupled with structural distance constraints (the links), and an optional
//! layer of point-to-midpoint bending constraints along every interior triple
//! gives the rope stiffness so it resists kinking.
//!
//! The result is a [`Rope`], which keeps the [`SoftBody`] together with the
//! ordered list of particle handles so callers can pin either end, attach the
//! rope to a moving anchor, or read its hanging shape.
//!
//! # Provenance
//!
//! This module contains **no Unreal Engine source or derived code**. A chain of
//! distance constraints with optional bending is a standard, publicly
//! documented rope / hair simulation topology.

use glam::Vec3;

use crate::math::scalar::Real;
use crate::soft::body::SoftBody;
use crate::soft::constraint::{BendingConstraint, DistanceConstraint};
use crate::soft::particle::ParticleHandle;
use crate::soft::solver::SoftSolverConfig;

/// Description of a rope / hair chain to build.
///
/// Particles are placed in a straight line starting at `origin` and stepping
/// `spacing` metres along `-Y` for each successive particle. A rope with
/// `segments` links therefore has `segments + 1` particles.
#[derive(Clone, Copy, PartialEq, Debug)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub struct RopeGrid {
    /// Number of links (distance constraints) in the chain. The chain has
    /// `segments + 1` particles.
    pub segments: u32,
    /// Rest spacing between adjacent particles, in metres.
    pub spacing: Real,
    /// Mass of each particle, in kilograms.
    pub particle_mass: Real,
    /// Compliance of the structural link (distance) constraints.
    pub compliance: Real,
    /// Compliance of the bending constraints. A negative value disables bending
    /// entirely; `0` makes the rope perfectly stiff against kinking.
    pub bending_compliance: Real,
    /// World position of the first (top) particle.
    pub origin: Vec3,
}

impl Default for RopeGrid {
    fn default() -> Self {
        RopeGrid {
            segments: 16,
            spacing: 0.1,
            particle_mass: 0.1,
            compliance: 0.0,
            bending_compliance: 0.05,
            origin: Vec3::ZERO,
        }
    }
}

impl RopeGrid {
    /// Builds the rope into a [`Rope`] using the given solver configuration.
    #[must_use]
    pub fn build(&self, config: SoftSolverConfig) -> Rope {
        let particle_count = self.segments.saturating_add(1).max(1);
        let mut body = SoftBody::new(config);
        let mut handles = Vec::with_capacity(particle_count as usize);

        for i in 0..particle_count {
            let position = self.origin + Vec3::new(0.0, -(i as Real) * self.spacing, 0.0);
            handles.push(body.spawn(position, self.particle_mass));
        }

        let positions = |body: &SoftBody, h: ParticleHandle| body.particles.positions()[h.index()];

        // Structural links between consecutive particles.
        for i in 0..handles.len().saturating_sub(1) {
            let a = handles[i];
            let b = handles[i + 1];
            let rest = (positions(&body, a) - positions(&body, b)).length();
            body.add_distance(DistanceConstraint::new(a, b, rest, self.compliance));
        }

        // Optional bending along every interior triple.
        if self.bending_compliance >= 0.0 {
            for i in 1..handles.len().saturating_sub(1) {
                let prev = handles[i - 1];
                let center = handles[i];
                let next = handles[i + 1];
                if let Some(c) = BendingConstraint::from_positions(
                    prev,
                    center,
                    next,
                    body.particles.positions(),
                    self.bending_compliance,
                ) {
                    body.add_bending(c);
                }
            }
        }

        Rope { body, handles }
    }

    /// Builds the rope with the default solver configuration.
    #[must_use]
    pub fn build_default(&self) -> Rope {
        self.build(SoftSolverConfig::default())
    }
}

/// A built rope / hair chain: its [`SoftBody`] plus the ordered particle handles.
#[derive(Clone, PartialEq, Debug)]
pub struct Rope {
    /// The simulated body. Step it with [`SoftBody::step`].
    pub body: SoftBody,
    /// Particle handles ordered from the top (index `0`) to the free end.
    handles: Vec<ParticleHandle>,
}

impl Rope {
    /// Returns the number of particles in the rope.
    #[must_use]
    pub fn len(&self) -> usize {
        self.handles.len()
    }

    /// Returns `true` if the rope has no particles.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.handles.is_empty()
    }

    /// Returns the handle of the particle at chain index `i`, or `None` if out
    /// of range. Index `0` is the top of the rope.
    #[must_use]
    pub fn handle(&self, i: usize) -> Option<ParticleHandle> {
        self.handles.get(i).copied()
    }

    /// Pins the particle at chain index `i` so it becomes an immovable anchor.
    /// Returns `false` if the index is out of range.
    pub fn pin(&mut self, i: usize) -> bool {
        match self.handle(i) {
            Some(h) => {
                self.body.particles.pin(h);
                true
            }
            None => false,
        }
    }

    /// Advances the rope by `dt` seconds.
    pub fn step(&mut self, dt: Real) {
        self.body.step(dt);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn chain_spawns_segments_plus_one_particles() {
        let rope = RopeGrid {
            segments: 5,
            ..RopeGrid::default()
        }
        .build_default();
        assert_eq!(rope.len(), 6);
        assert_eq!(rope.body.particles.len(), 6);
    }

    #[test]
    fn chain_wires_link_and_bending_counts() {
        // 5 segments -> 6 particles -> 5 links, interior triples = 4 bendings.
        let rope = RopeGrid {
            segments: 5,
            ..RopeGrid::default()
        }
        .build_default();
        assert_eq!(rope.body.constraints.distance.len(), 5);
        assert_eq!(rope.body.constraints.bending.len(), 4);
    }

    #[test]
    fn negative_bending_compliance_disables_bending() {
        let rope = RopeGrid {
            segments: 5,
            bending_compliance: -1.0,
            ..RopeGrid::default()
        }
        .build_default();
        assert_eq!(rope.body.constraints.distance.len(), 5);
        assert!(rope.body.constraints.bending.is_empty());
    }

    #[test]
    fn handle_lookup_respects_bounds() {
        let rope = RopeGrid {
            segments: 3,
            ..RopeGrid::default()
        }
        .build_default();
        assert!(rope.handle(0).is_some());
        assert!(rope.handle(3).is_some());
        assert!(rope.handle(4).is_none());
    }

    #[test]
    fn pinned_top_hangs_and_keeps_link_lengths() {
        // Pin the top particle, let the rope hang, and confirm links stay near
        // their rest length while the free end drops below the anchor.
        let grid = RopeGrid {
            segments: 10,
            spacing: 0.1,
            particle_mass: 0.05,
            compliance: 0.0,
            bending_compliance: 0.05,
            origin: Vec3::ZERO,
        };
        let mut rope = grid.build_default();
        assert!(rope.pin(0));

        for _ in 0..300 {
            rope.step(1.0 / 60.0);
        }

        for &p in rope.body.particles.positions() {
            assert!(p.is_finite(), "non-finite particle {p:?}");
        }

        let mut max_ratio: Real = 0.0;
        for c in &rope.body.constraints.distance {
            if c.rest_length <= 0.0 {
                continue;
            }
            let pa = rope.body.particles.position(c.a).unwrap();
            let pb = rope.body.particles.position(c.b).unwrap();
            let ratio = (pa - pb).length() / c.rest_length;
            max_ratio = max_ratio.max(ratio);
        }
        assert!(max_ratio < 1.5, "link stretched {max_ratio}x rest length");

        // Anchor immovable; free end hangs below it.
        let top = rope
            .body
            .particles
            .position(rope.handle(0).unwrap())
            .unwrap();
        let end = rope
            .body
            .particles
            .position(rope.handle(rope.len() - 1).unwrap())
            .unwrap();
        assert_eq!(top, Vec3::ZERO);
        assert!(end.y < top.y, "free end did not hang below the anchor");
    }
}
