//! Offline throughput benchmark for a large draping cloth (M4 scale goal).
//!
//! Like the M3 scale benchmark this is intentionally dependency-free: it uses a
//! `harness = false` plain `main` and [`std::time::Instant`] rather than an
//! external harness, so it builds and runs fully offline with no extra crates.
//! Run it with:
//!
//! ```text
//! cargo bench -p prism_physics_core --bench cloth_scale
//! ```
//!
//! It builds a large cloth sheet pinned along its top edge, drapes it under
//! gravity, and reports the average per-step time. This is the cost of the
//! unified XPBD kernel projecting tens of thousands of constraints per step.
//!
//! # Provenance
//!
//! This is an original benchmark authored for Prism. It contains
//! **no Unreal Engine source or derived code**.
#![expect(
    clippy::print_stdout,
    reason = "a benchmark binary reports its timing results to stdout"
)]

use glam::Vec3;
use prism_physics_core::{Cloth, ClothGrid};
use std::time::Instant;

/// Cloth resolution along each axis; the sheet holds `GRID * GRID` particles.
const GRID: u32 = 96;
/// Fixed simulation timestep.
const DT: f32 = 1.0 / 60.0;

/// Builds the pinned-edge cloth sheet.
fn build_cloth() -> Cloth {
    let grid = ClothGrid {
        columns: GRID,
        rows: GRID,
        spacing: 0.05,
        particle_mass: 0.02,
        compliance: 0.0,
        bending_compliance: 0.02,
        origin: Vec3::ZERO,
    };
    let mut cloth = grid.build_default();
    for column in 0..GRID {
        cloth.pin(0, column);
    }
    cloth
}

/// Steps the cloth `count` times and returns the average per-step time in
/// milliseconds.
fn time_steps(cloth: &mut Cloth, count: u32) -> f64 {
    let start = Instant::now();
    for _ in 0..count {
        cloth.step(DT);
    }
    let elapsed = start.elapsed();
    elapsed.as_secs_f64() * 1000.0 / f64::from(count)
}

fn main() {
    let mut cloth = build_cloth();
    let particles = cloth.body.particles.len();
    let distance = cloth.body.constraints.distance.len();
    let bending = cloth.body.constraints.bending.len();

    // Warm up so the first-step allocation/settling cost is not measured.
    let _ = time_steps(&mut cloth, 10);
    let avg_ms = time_steps(&mut cloth, 60);

    println!("prism_physics_core cloth_scale benchmark");
    println!("  particles:            {particles}");
    println!("  distance constraints: {distance}");
    println!("  bending constraints:  {bending}");
    println!("  avg step:             {avg_ms:.3} ms/step");
}
