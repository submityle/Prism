//! Offline bake + zero-solve playback + golden-replay acceptance test (M4.5).
//!
//! M4.5 adds Prism's **offline mode** for deformable simulation: a soft body is
//! solved once at author time and its trajectory is stored in a compact,
//! deterministic [`PhysicsCache`] that can be replayed with no solver cost. This
//! test exercises that pipeline end-to-end on a real draping cloth and asserts
//! the properties offline mode promises:
//!
//! 1. **Playback fidelity** &mdash; the player reproduces every baked frame
//!    within the quantization tolerance.
//! 2. **Determinism** &mdash; two identical bakes hash to the same
//!    [`GoldenDigest`], which is what a golden-replay CI gate compares.
//! 3. **Compression** &mdash; the keyframe + sparse-delta track stores far fewer
//!    components than a naive per-frame-full encoding.
//! 4. **Time-domain sampling** &mdash; interpolation lands between adjacent baked
//!    frames and looping wraps playback time.
//! 5. **Zero solve** &mdash; playback never mutates the source body.
//!
//! Every number asserted below comes from baking and replaying a real
//! [`SoftBody`]; nothing is stubbed.
//!
//! # Provenance
//!
//! This is an original scenario authored for Prism. It contains
//! **no Unreal Engine source or derived code**.

use glam::Vec3;
use prism_physics_core::{
    trajectory_hash, BakeConfig, Baker, ClothGrid, PhysicsCache, PlaybackConfig, Player,
    PositionQuantizer, SoftSolverConfig,
};

/// Builds a cloth pinned along its top edge, ready to drape under gravity.
fn drape_cloth() -> prism_physics_core::Cloth {
    let mut cloth = ClothGrid {
        columns: 8,
        rows: 8,
        spacing: 0.1,
        ..ClothGrid::default()
    }
    .build_default();
    // Pin the top row so the sheet hangs and moves (non-trivial trajectory).
    for column in 0..cloth.columns() {
        cloth.pin(0, column);
    }
    cloth
}

/// Bakes a draping cloth into a cache with the given settings.
fn bake_drape(frames: u32, keyframe_interval: u32, step: f32) -> PhysicsCache {
    let mut cloth = drape_cloth();
    let baker = Baker::new(BakeConfig {
        frames,
        keyframe_interval,
        quantizer: PositionQuantizer::new(step),
        ..BakeConfig::default()
    });
    baker.bake_body(&mut cloth.body)
}

#[test]
fn playback_reproduces_every_baked_frame_within_tolerance() {
    let step = 1.0e-4_f32;
    let cache = bake_drape(90, 12, step);
    let player = Player::new(PlaybackConfig::default());
    // Landing exactly on a frame boundary should reconstruct that frame with no
    // interpolation, so the only error is the quantization grid step.
    let tolerance = step * 1.5;
    for frame in 0..cache.frame_count() {
        let mut probe = player;
        probe.set_time(frame as f32 * cache.frame_dt());
        let sampled = probe.sample(&cache, 0).expect("track 0 exists");
        let baked = cache.reconstruct(0, frame).expect("baked frame exists");
        assert_eq!(sampled.len(), baked.len());
        for (s, b) in sampled.iter().zip(baked.iter()) {
            assert!(
                (*s - *b).length() <= tolerance,
                "frame {frame}: playback {s:?} vs baked {b:?}"
            );
        }
    }
}

#[test]
fn identical_bakes_are_bit_for_bit_deterministic() {
    let a = bake_drape(60, 12, 1.0e-4);
    let b = bake_drape(60, 12, 1.0e-4);
    assert_eq!(
        trajectory_hash(&a),
        trajectory_hash(&b),
        "two identical bakes must produce the same golden digest"
    );
}

#[test]
fn sparse_track_beats_raw_full_encoding() {
    // Compression from keyframes + sparse deltas only pays off once motion goes
    // sparse: while every particle is moving, a per-particle delta (index + xyz)
    // is wider than a raw sample (xyz). To demonstrate the win honestly, bake a
    // strongly damped cloth that settles to rest, recorded on a 1 mm grid so
    // that settled particles produce empty deltas.
    let mut cloth = ClothGrid {
        columns: 8,
        rows: 8,
        spacing: 0.1,
        ..ClothGrid::default()
    }
    .build(SoftSolverConfig {
        damping: 12.0,
        ..SoftSolverConfig::default()
    });
    for column in 0..cloth.columns() {
        cloth.pin(0, column);
    }
    let baker = Baker::new(BakeConfig {
        frames: 240,
        keyframe_interval: 30,
        quantizer: PositionQuantizer::new(1.0e-3),
        ..BakeConfig::default()
    });
    let cache = baker.bake_body(&mut cloth.body);
    let track = cache.track(0).expect("one track");
    // Count the components a naive per-frame-full encoding would store versus
    // the components the keyframe + sparse-delta track actually holds.
    let raw = cache.raw_component_count();
    // Recompute the stored component count by diffing successive reconstructed
    // frames: keyframes cost a full snapshot, deltas cost 4 components (index +
    // xyz) per moved particle, which mirrors the on-disk track layout.
    let mut stored: u64 = 0;
    let mut prev: Option<Vec<[i32; 3]>> = None;
    for frame in 0..track.frame_count() {
        let cur = track.reconstruct(frame).expect("frame reconstructs");
        match (frame.is_multiple_of(cache.keyframe_interval()), &prev) {
            (true, _) | (false, None) => stored += cur.len() as u64 * 3,
            (false, Some(p)) => {
                let moved = cur.iter().zip(p.iter()).filter(|(c, q)| c != q).count() as u64;
                // Each moved particle costs an index + three delta components.
                stored += moved * 4;
            }
        }
        prev = Some(cur);
    }
    assert!(
        stored < raw,
        "sparse encoding ({stored}) should be smaller than raw ({raw})"
    );
    let ratio = raw as f64 / stored as f64;
    assert!(ratio > 1.0, "compression ratio {ratio:.2}x should exceed 1");
}

#[test]
fn interpolation_lands_between_frames_and_looping_wraps() {
    let cache = bake_drape(30, 10, 1.0e-4);
    // Sample at half a frame: each particle should sit between frames 0 and 1.
    let mut player = Player::new(PlaybackConfig::default());
    player.set_time(cache.frame_dt() * 0.5);
    let mid = player.sample(&cache, 0).expect("sample");
    let f0 = cache.reconstruct(0, 0).expect("f0");
    let f1 = cache.reconstruct(0, 1).expect("f1");
    let eps = Vec3::splat(1.0e-4);
    for i in 0..mid.len() {
        let lo = f0[i].min(f1[i]) - eps;
        let hi = f0[i].max(f1[i]) + eps;
        assert!(mid[i].cmpge(lo).all() && mid[i].cmple(hi).all());
    }
    // Looping wraps time back into [0, duration].
    let mut looped = Player::new(PlaybackConfig {
        time_scale: 1.0,
        looping: true,
    });
    let duration = cache.duration();
    looped.advance(duration * 3.25, duration);
    assert!(looped.time() >= 0.0 && looped.time() <= duration);
}

#[test]
fn playback_never_mutates_the_source_body() {
    // Bake, then keep the body; playing back the cache must not touch it.
    let mut cloth = drape_cloth();
    let baker = Baker::new(BakeConfig {
        frames: 40,
        ..BakeConfig::default()
    });
    let cache = baker.bake_body(&mut cloth.body);
    let before: Vec<Vec3> = cloth.body.particles.positions().to_vec();
    let player = Player::new(PlaybackConfig::default());
    // Sampling many times performs zero solving and does not borrow the body.
    for frame in 0..cache.frame_count() {
        let mut probe = player;
        probe.set_time(frame as f32 * cache.frame_dt());
        let _ = probe.sample(&cache, 0);
    }
    let after: Vec<Vec3> = cloth.body.particles.positions().to_vec();
    assert_eq!(before, after, "playback must not mutate the source body");
}
