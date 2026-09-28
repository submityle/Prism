//! Offline replay throughput benchmark: bake once, play back for free (M4.5).
//!
//! Like the other physics benchmarks this is intentionally dependency-free: it
//! uses a `harness = false` plain `main` and [`std::time::Instant`], so it runs
//! fully offline with no extra crates. Run it with:
//!
//! ```text
//! cargo bench -p prism_physics_core --bench cache_replay
//! ```
//!
//! It bakes a draping cloth into a [`PhysicsCache`], then compares the cost of
//! sampling the cache (pure interpolation, no solving) against the cost of the
//! live XPBD solver, and reports the trajectory's compression ratio. The point
//! of offline mode is that playback cost is independent of scene complexity.
//!
//! # Provenance
//!
//! This is an original benchmark authored for Prism. It contains
//! **no Unreal Engine source or derived code**.
#![expect(
    clippy::print_stdout,
    reason = "a benchmark binary reports its timing results to stdout"
)]

use std::time::Instant;

use prism_physics_core::{BakeConfig, Baker, ClothGrid, PlaybackConfig, Player};

fn main() {
    let columns = 48;
    let rows = 48;
    let frames = 240;

    let mut cloth = ClothGrid {
        columns,
        rows,
        spacing: 0.05,
        ..ClothGrid::default()
    }
    .build_default();
    for column in 0..cloth.columns() {
        cloth.pin(0, column);
    }

    let config = BakeConfig {
        frames,
        ..BakeConfig::default()
    };
    let baker = Baker::new(config);

    // --- Bake (offline, paid once) ---------------------------------------
    let bake_start = Instant::now();
    let cache = baker.bake_body(&mut cloth.body);
    let bake_time = bake_start.elapsed();

    // --- Live solve cost per step ----------------------------------------
    let mut live = ClothGrid {
        columns,
        rows,
        spacing: 0.05,
        ..ClothGrid::default()
    }
    .build_default();
    for column in 0..live.columns() {
        live.pin(0, column);
    }
    let solve_iters = 240;
    let solve_start = Instant::now();
    for _ in 0..solve_iters {
        live.step(config.dt);
    }
    let solve_per_step = solve_start.elapsed().as_secs_f64() / f64::from(solve_iters);

    // --- Playback cost per sample ----------------------------------------
    let player = Player::new(PlaybackConfig::default());
    let sample_iters = 5_000;
    let sample_start = Instant::now();
    let mut sink = 0usize;
    for i in 0..sample_iters {
        let mut probe = player;
        let t = f64::from(i) / f64::from(sample_iters) * f64::from(cache.duration());
        probe.set_time(t as f32);
        if let Some(positions) = probe.sample(&cache, 0) {
            sink = sink.wrapping_add(positions.len());
        }
    }
    let sample_per_call = sample_start.elapsed().as_secs_f64() / f64::from(sample_iters);

    // --- Compression ------------------------------------------------------
    let raw_components = cache.raw_component_count();

    println!("cache_replay benchmark ({columns}x{rows} cloth, {frames} frames)");
    println!(
        "  bake total:        {:>10.3} ms",
        bake_time.as_secs_f64() * 1e3
    );
    println!("  live solve/step:   {:>10.3} us", solve_per_step * 1e6);
    println!("  playback/sample:   {:>10.3} us", sample_per_call * 1e6);
    if sample_per_call > 0.0 {
        println!(
            "  speedup vs solve:  {:>10.1}x",
            solve_per_step / sample_per_call
        );
    }
    println!("  raw components:    {raw_components}");
    println!("  (sink {sink})");
}
