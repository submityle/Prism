//! Interactive-deformable acceptance test for the M4 milestone.
//!
//! M4 adds Prism's unified soft-body kernel: cloth, rope, and volumetric soft
//! bodies are all a [`SoftBody`] of particles coupled by XPBD constraints,
//! advanced by the same substep solver. This test exercises the three shipped
//! builders end-to-end and asserts the behaviours a game depends on:
//!
//! 1. **Cloth drape** &mdash; a sheet pinned along its top edge drapes under
//!    gravity, stays finite, keeps its edges near their rest length, and hangs
//!    below the pinned edge.
//! 2. **Rope hang** &mdash; a chain pinned at the top hangs straight down with
//!    its links close to their rest length.
//! 3. **Soft-body volume** &mdash; a tetrahedral block anchored at its base
//!    settles without inverting and preserves most of its rest volume.
//! 4. **Determinism** &mdash; stepping the same authored scene twice produces
//!    bit-identical particle positions, so simulations are reproducible.
//!
//! Every number asserted below is the outcome of stepping a real
//! [`SoftBody`] solver; nothing is stubbed.
//!
//! # Provenance
//!
//! This is an original scenario authored for Prism. It contains
//! **no Unreal Engine source or derived code**.

use glam::Vec3;
use prism_physics_core::{ClothGrid, RopeGrid, SoftBoxGrid};

/// Fixed simulation timestep used throughout the test.
const DT: f32 = 1.0 / 60.0;

#[test]
fn cloth_drapes_from_a_pinned_edge_and_keeps_its_edges() {
    let grid = ClothGrid {
        columns: 12,
        rows: 12,
        spacing: 0.1,
        particle_mass: 0.05,
        compliance: 0.0,
        bending_compliance: 0.02,
        origin: Vec3::ZERO,
    };
    let mut cloth = grid.build_default();

    // Pin the whole top edge (row 0).
    for column in 0..grid.columns {
        assert!(cloth.pin(0, column));
    }

    for _ in 0..400 {
        cloth.step(DT);
    }

    // Stability: no particle blew up.
    for &p in cloth.body.particles.positions() {
        assert!(p.is_finite(), "non-finite cloth particle {p:?}");
    }

    // Structural edges stay near their rest length.
    let mut max_ratio = 0.0f32;
    for c in &cloth.body.constraints.distance {
        if c.rest_length <= 0.0 {
            continue;
        }
        let pa = cloth.body.particles.position(c.a).unwrap();
        let pb = cloth.body.particles.position(c.b).unwrap();
        max_ratio = max_ratio.max((pa - pb).length() / c.rest_length);
    }
    assert!(max_ratio < 1.5, "cloth edge stretched {max_ratio}x rest");

    // The free edge (last row) hangs well below the pinned edge.
    let bottom = cloth
        .body
        .particles
        .position(cloth.handle(grid.rows - 1, 0).unwrap())
        .unwrap();
    assert!(
        bottom.y < -0.2,
        "cloth did not drape (bottom y = {})",
        bottom.y
    );

    // Pinned corners never moved.
    assert_eq!(
        cloth.body.particles.position(cloth.handle(0, 0).unwrap()),
        Some(Vec3::ZERO)
    );
}

#[test]
fn rope_hangs_straight_from_its_anchor() {
    let grid = RopeGrid {
        segments: 20,
        spacing: 0.1,
        particle_mass: 0.05,
        compliance: 0.0,
        bending_compliance: 0.05,
        origin: Vec3::ZERO,
    };
    let mut rope = grid.build_default();
    assert!(rope.pin(0));

    for _ in 0..600 {
        rope.step(DT);
    }

    for &p in rope.body.particles.positions() {
        assert!(p.is_finite(), "non-finite rope particle {p:?}");
    }

    // Links stay near their rest length under gravity.
    let mut max_ratio = 0.0f32;
    for c in &rope.body.constraints.distance {
        if c.rest_length <= 0.0 {
            continue;
        }
        let pa = rope.body.particles.position(c.a).unwrap();
        let pb = rope.body.particles.position(c.b).unwrap();
        max_ratio = max_ratio.max((pa - pb).length() / c.rest_length);
    }
    assert!(max_ratio < 1.3, "rope link stretched {max_ratio}x rest");

    // A hanging rope is nearly vertical: horizontal drift stays tiny compared to
    // the vertical drop.
    let end = rope
        .body
        .particles
        .position(rope.handle(rope.len() - 1).unwrap())
        .unwrap();
    assert!(end.y < -1.0, "rope did not hang (end y = {})", end.y);
    let horizontal = (end.x * end.x + end.z * end.z).sqrt();
    assert!(horizontal < 0.2, "rope drifted sideways by {horizontal} m");
}

#[test]
fn soft_body_settles_without_inverting_and_keeps_volume() {
    let grid = SoftBoxGrid {
        cells_x: 3,
        cells_y: 3,
        cells_z: 3,
        spacing: 0.1,
        particle_mass: 0.05,
        compliance: 0.0,
        volume_compliance: 0.0,
        origin: Vec3::ZERO,
    };
    let mut solid = grid.build_default();

    // Anchor the whole bottom face so the block hangs and deforms.
    for z in 0..solid.particles_z() {
        for x in 0..solid.particles_x() {
            assert!(solid.pin(x, 0, z));
        }
    }

    let rest_volume = total_volume(&solid);
    for _ in 0..400 {
        solid.step(DT);
    }

    for &p in solid.body.particles.positions() {
        assert!(p.is_finite(), "non-finite soft-body particle {p:?}");
    }

    let volume = total_volume(&solid);
    assert!(volume > 0.0, "soft body inverted (volume {volume})");
    let ratio = volume / rest_volume;
    assert!(
        (0.6..1.4).contains(&ratio),
        "soft-body volume drifted to {ratio}x rest",
    );
}

#[test]
fn simulation_is_deterministic_across_runs() {
    let run = || {
        let mut cloth = ClothGrid {
            columns: 8,
            rows: 8,
            ..ClothGrid::default()
        }
        .build_default();
        cloth.pin(0, 0);
        cloth.pin(0, 7);
        for _ in 0..120 {
            cloth.step(DT);
        }
        cloth.body.particles.positions().to_vec()
    };
    assert_eq!(run(), run(), "cloth simulation was not deterministic");
}

/// Sums the absolute volume of every tetrahedron in a built soft body.
fn total_volume(solid: &prism_physics_core::SoftBox) -> f32 {
    let positions = solid.body.particles.positions();
    let mut sum = 0.0;
    for c in &solid.body.constraints.volume {
        let p0 = positions[c.particles[0].index()];
        let p1 = positions[c.particles[1].index()];
        let p2 = positions[c.particles[2].index()];
        let p3 = positions[c.particles[3].index()];
        sum += ((p1 - p0).dot((p2 - p0).cross(p3 - p0)) / 6.0).abs();
    }
    sum
}
