//! Integration tests for the M7 reduced-order (modal subspace) soft body.
//!
//! These exercise the public surface end to end: build a spring-lattice
//! [`ReducedModel`] from a [`ParticleStorage`], then drive a [`ReducedState`]
//! under gravity and confirm the expected macroscopic behaviour.
//!
//! The lattice is a *braced* plate rather than a single axial chain: a chain of
//! collinear springs has no linear transverse (bending) stiffness, so its
//! low-frequency modes are near-zero-energy and gravity would drive them without
//! bound. Adding vertical and diagonal springs gives the structure
//! positive-definite stiffness in every direction, which is what a real
//! cantilever plate has, so it sags a finite amount and settles.

use glam::Vec3;
use prism_physics_core::{
    ParticleStorage, Real, ReducedConfig, ReducedModel, ReducedState, SpringElement, SpringSet,
};

/// A braced rectangular plate in the X-Y plane with `cols` columns along `+X`
/// and `rows` rows along `+Y`, one metre spacing. The entire left column
/// (`x == 0`) is anchored; every other vertex is free. Horizontal, vertical,
/// and both diagonal springs brace each cell.
struct Plate {
    rest: Vec<Vec3>,
    masses: Vec<Real>,
    fixed: Vec<bool>,
    springs: SpringSet,
    cols: usize,
    rows: usize,
}

fn build_plate(cols: usize, rows: usize, stiffness: Real) -> Plate {
    let mut storage = ParticleStorage::default();
    let index = |c: usize, r: usize| c * rows + r;
    let count = cols * rows;
    let mut handles = Vec::with_capacity(count);
    let mut rest = Vec::with_capacity(count);
    let mut masses = Vec::with_capacity(count);
    let mut fixed = Vec::with_capacity(count);
    for c in 0..cols {
        for r in 0..rows {
            let position = Vec3::new(c as Real, r as Real, 0.0);
            handles.push(storage.spawn(position, 1.0));
            rest.push(position);
            masses.push(1.0);
            fixed.push(c == 0);
        }
    }

    let mut springs = SpringSet::new();
    let mut connect = |a: usize, b: usize| {
        let len = (rest[a] - rest[b]).length();
        springs.push(SpringElement::new(handles[a], handles[b], len, stiffness));
    };
    for c in 0..cols {
        for r in 0..rows {
            let here = index(c, r);
            if c + 1 < cols {
                connect(here, index(c + 1, r));
            }
            if r + 1 < rows {
                connect(here, index(c, r + 1));
            }
            if c + 1 < cols && r + 1 < rows {
                connect(here, index(c + 1, r + 1));
            }
            if c + 1 < cols && r >= 1 {
                connect(here, index(c + 1, r - 1));
            }
        }
    }

    Plate {
        rest,
        masses,
        fixed,
        springs,
        cols,
        rows,
    }
}

impl Plate {
    fn model(&self, num_modes: usize) -> ReducedModel {
        ReducedModel::from_springs(
            &self.rest,
            &self.masses,
            &self.fixed,
            &self.springs,
            num_modes,
        )
    }

    /// Mean height of the vertices in column `c`.
    fn column_mean_y(&self, positions: &[Vec3], c: usize) -> Real {
        let start = c * self.rows;
        let sum: Real = positions[start..start + self.rows]
            .iter()
            .map(|p| p.y)
            .sum();
        sum / self.rows as Real
    }
}

#[test]
fn cantilever_plate_sags_under_gravity() {
    let plate = build_plate(4, 2, 400.0);
    let model = plate.model(12);
    assert!(model.num_modes() > 0);
    assert_eq!(model.num_vertices(), 8);

    let mut state = ReducedState::for_model(&model);
    let config = ReducedConfig {
        gravity: Vec3::new(0.0, -9.81, 0.0),
        num_modes: model.num_modes(),
        rayleigh_alpha: 2.0,
        rayleigh_beta: 0.03,
        substeps: 4,
    };
    for _ in 0..3000 {
        state.step(&model, &config, 1.0 / 60.0, &[]);
    }

    let positions = state.positions(&model);
    // The anchored column never moves.
    for (p, rest) in positions.iter().zip(plate.rest.iter()).take(plate.rows) {
        assert_eq!(*p, *rest);
    }
    // Free columns sag, and the further-out column sags more than the first.
    let first_free = plate.column_mean_y(&positions, 1);
    let tip = plate.column_mean_y(&positions, plate.cols - 1);
    // The tip sags well below rest, and each column further from the anchor sags
    // below the previous one (monotone cantilever deflection).
    assert!(tip < -1e-2, "tip column did not sag: {tip}");
    assert!(
        tip < first_free,
        "tip {tip} should sag below first free {first_free}"
    );
    let mut prev = plate.column_mean_y(&positions, 1);
    for c in 2..plate.cols {
        let y = plate.column_mean_y(&positions, c);
        assert!(
            y <= prev + 1e-4,
            "column {c} y={y} not monotone below {prev}"
        );
        prev = y;
    }
}

#[test]
fn cantilever_plate_reaches_static_equilibrium() {
    let plate = build_plate(3, 2, 500.0);
    let model = plate.model(12);

    let mut state = ReducedState::for_model(&model);
    let config = ReducedConfig {
        gravity: Vec3::new(0.0, -9.81, 0.0),
        num_modes: model.num_modes(),
        rayleigh_alpha: 4.0,
        rayleigh_beta: 0.08,
        substeps: 4,
    };
    for _ in 0..6000 {
        state.step(&model, &config, 1.0 / 60.0, &[]);
    }
    let before = state.positions(&model);
    state.step(&model, &config, 1.0 / 60.0, &[]);
    let after = state.positions(&model);
    for (b, a) in before.iter().zip(after.iter()) {
        assert!((*a - *b).length() < 1e-4, "not settled: {b} -> {a}");
    }
    assert!(state.velocities().iter().all(|v| v.abs() < 1e-2));
}

#[test]
fn integration_is_deterministic() {
    let plate = build_plate(3, 2, 250.0);
    let model = plate.model(10);
    let config = ReducedConfig {
        gravity: Vec3::new(0.0, -9.81, 0.0),
        num_modes: model.num_modes(),
        rayleigh_alpha: 1.0,
        rayleigh_beta: 0.02,
        substeps: 2,
    };
    let run = || {
        let mut state = ReducedState::for_model(&model);
        for _ in 0..300 {
            state.step(&model, &config, 1.0 / 60.0, &[]);
        }
        state.positions(&model)
    };
    assert_eq!(run(), run());
}
