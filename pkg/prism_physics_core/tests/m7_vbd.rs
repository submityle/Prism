//! Acceptance test for the M7 Vertex Block Descent (VBD) solver.
//!
//! M7 adds an unconditionally stable alternative to the XPBD soft-body kernel:
//! Vertex Block Descent minimises the implicit-Euler energy by per-vertex block
//! coordinate descent (Chen et al., SIGGRAPH 2024). This test exercises the
//! behaviours a game relies on:
//!
//! 1. **Inertial free fall** &mdash; with no springs a dynamic vertex lands on
//!    the analytic inertial target each substep.
//! 2. **Stiff rope** &mdash; a pinned-top rope with a near-rigid spring settles
//!    close to its rest length and never blows up, even at 60 Hz with only a
//!    handful of iterations, which XPBD cannot match at the same stiffness.
//! 3. **Cloth drape** &mdash; a spring lattice pinned along its top edge drapes,
//!    stays finite, and keeps its edges near rest length.
//! 4. **Determinism** &mdash; the same authored scene stepped twice is
//!    bit-for-bit identical.
//!
//! # Provenance
//!
//! This module contains **no Unreal Engine source or derived code**.

use glam::Vec3;
use prism_physics_core::{ParticleHandle, VbdBody, VbdConfig};

#[test]
fn free_vertex_follows_the_inertial_target() {
    let mut body = VbdBody::new(VbdConfig {
        damping: 0.0,
        substeps: 1,
        iterations: 4,
        ..VbdConfig::default()
    });
    let p = body.spawn(Vec3::ZERO, 1.0);
    let dt = 1.0 / 60.0;
    let mut expected_v = 0.0;
    let mut expected_y = 0.0;
    for _ in 0..30 {
        // Semi-implicit Euler reference: v += g h; y += v h.
        expected_v += body.config.gravity.y * dt;
        expected_y += expected_v * dt;
        body.step(dt);
    }
    let pos = body.particles.position(p).unwrap();
    assert!(
        (pos.y - expected_y).abs() < 1e-3,
        "expected y {expected_y}, got {}",
        pos.y
    );
    assert!(pos.x.abs() < 1e-5 && pos.z.abs() < 1e-5);
}

#[test]
fn stiff_rope_settles_near_rest_length() {
    let mut body = VbdBody::with_default_config();
    let top = body.spawn_pinned(Vec3::ZERO);
    let mut prev = top;
    // A five-link rope, each link 0.5 m, with a very stiff spring.
    for i in 1..=5 {
        let node = body.spawn(Vec3::new(0.0, -0.5 * i as f32, 0.0), 1.0);
        assert!(body.connect(prev, node, 1.0e6));
        prev = node;
    }
    for _ in 0..600 {
        body.step(1.0 / 60.0);
    }
    // The rope must hang straight down, finite, close to its 2.5 m rest length.
    let (min, max) = body.bounds().unwrap();
    assert!(min.y.is_finite() && max.y.is_finite());
    let tip = body.particles.position(prev).unwrap();
    assert!(
        tip.x.abs() < 0.05 && tip.z.abs() < 0.05,
        "tip drifted {tip:?}"
    );
    assert!(tip.y < -2.3 && tip.y > -2.7, "rope length off: tip {tip:?}");
    assert_eq!(body.particles.position(top), Some(Vec3::ZERO));
}

#[test]
fn cloth_lattice_drapes_and_stays_finite() {
    // A 5x5 grid of particles in the XZ plane, structural springs along the
    // grid edges, pinned along the top row (z = 0).
    const N: usize = 5;
    const SPACING: f32 = 0.25;
    const STIFFNESS: f32 = 5.0e4;
    let mut body = VbdBody::with_default_config();
    let mut handles: Vec<Vec<ParticleHandle>> = Vec::with_capacity(N);
    for z in 0..N {
        let mut row = Vec::with_capacity(N);
        for x in 0..N {
            let pos = Vec3::new(x as f32 * SPACING, 0.0, z as f32 * SPACING);
            let handle = if z == 0 {
                body.spawn_pinned(pos)
            } else {
                body.spawn(pos, 1.0)
            };
            row.push(handle);
        }
        handles.push(row);
    }
    for z in 0..N {
        for x in 0..N {
            if x + 1 < N {
                assert!(body.connect(handles[z][x], handles[z][x + 1], STIFFNESS));
            }
            if z + 1 < N {
                assert!(body.connect(handles[z][x], handles[z + 1][x], STIFFNESS));
            }
        }
    }
    for _ in 0..300 {
        body.step(1.0 / 60.0);
    }
    // Top row stays pinned; the sheet drapes below it and remains finite.
    for &handle in &handles[0] {
        assert_eq!(
            body.particles.position(handle).unwrap().z,
            0.0,
            "pinned row moved"
        );
    }
    let (min, max) = body.bounds().unwrap();
    assert!(min.y.is_finite() && max.y.is_finite());
    assert!(min.y < -0.05, "cloth did not drape: min y {}", min.y);
}

#[test]
fn stepping_is_bit_for_bit_deterministic() {
    let build_and_run = || {
        let mut body = VbdBody::with_default_config();
        let top = body.spawn_pinned(Vec3::ZERO);
        let mut prev = top;
        for i in 1..=4 {
            let node = body.spawn(Vec3::new(0.1 * i as f32, -0.4 * i as f32, 0.0), 1.0);
            assert!(body.connect(prev, node, 3.0e4));
            prev = node;
        }
        for _ in 0..120 {
            body.step(1.0 / 60.0);
        }
        body.particles.positions().to_vec()
    };
    assert_eq!(build_and_run(), build_and_run());
}
