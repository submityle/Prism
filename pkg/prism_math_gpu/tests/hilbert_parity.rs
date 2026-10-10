//! Real-device parity for the §24.1 / §24.7 2D Hilbert-curve shader mirror.
//!
//! Each test encodes a batch of integer lattice points into 2D Hilbert sort
//! keys on a real `GPU` from the single-sourced
//! [`WGSL_HILBERT2`](prism_math::shader_mirror::WGSL_HILBERT2) fragment and
//! diffs them against the CPU reference
//! [`prism_math::spatial::hilbert_encode2`]. The 2D Hilbert index is pure
//! integer arithmetic, so unlike the floating-point twins this parity is
//! **bit-exact**, not a tolerance. WGSL has no 64-bit integer type, so the
//! kernel runs the 16-bit-per-axis order; for inputs masked to that width the
//! GPU key equals the CPU `hilbert_encode2` output bit-for-bit, and the decode
//! is its exact inverse. The suite skips gracefully when no adapter is
//! available.

use prism_math::spatial::{hilbert_decode2, hilbert_encode2};
use prism_math_gpu::hilbert::MAX_COORD_2D;
use prism_math_gpu::GpuContext;
use prism_math_gpu::GpuHilbert;

/// Acquires a device, or prints a skip note and returns `None` on hosts without
/// a usable adapter.
#[expect(
    clippy::print_stderr,
    reason = "test-only skip note when no GPU adapter is present"
)]
fn with_gpu() -> Option<GpuContext> {
    match GpuContext::try_headless() {
        Some(ctx) => Some(ctx),
        None => {
            eprintln!("skipping: no usable GPU adapter on this host");
            None
        }
    }
}

/// A small deterministic linear-congruential sequence for pseudo-random coords.
fn lcg(seed: &mut u32) -> u32 {
    *seed = seed.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
    *seed
}

#[test]
fn encode2_matches_cpu_reference() {
    let Some(ctx) = with_gpu() else {
        return;
    };
    let kernel = GpuHilbert::new(&ctx);

    let mut coords: Vec<[u32; 2]> = vec![
        [0, 0],
        [MAX_COORD_2D, MAX_COORD_2D],
        [MAX_COORD_2D, 0],
        [0, MAX_COORD_2D],
        [1, 2],
    ];
    let mut seed = 0x1234_5678u32;
    for _ in 0..500 {
        coords.push([lcg(&mut seed) & MAX_COORD_2D, lcg(&mut seed) & MAX_COORD_2D]);
    }

    let gpu = kernel.encode2(&ctx, &coords);
    assert_eq!(gpu.len(), coords.len());
    for (c, &g) in coords.iter().zip(gpu.iter()) {
        let cpu = hilbert_encode2(c[0], c[1]) as u32;
        assert_eq!(g, cpu, "encode2 drift at ({}, {})", c[0], c[1]);
    }
}

#[test]
fn decode2_inverts_encode2() {
    let Some(ctx) = with_gpu() else {
        return;
    };
    let kernel = GpuHilbert::new(&ctx);

    let mut coords: Vec<[u32; 2]> = vec![[0, 0], [MAX_COORD_2D, MAX_COORD_2D], [7, 11]];
    let mut seed = 0x0bad_f00du32;
    for _ in 0..500 {
        coords.push([lcg(&mut seed) & MAX_COORD_2D, lcg(&mut seed) & MAX_COORD_2D]);
    }

    let keys = kernel.encode2(&ctx, &coords);
    let back = kernel.decode2(&ctx, &keys);
    assert_eq!(back.len(), coords.len());
    for (orig, got) in coords.iter().zip(back.iter()) {
        assert_eq!(got, orig, "decode2 round-trip drift");
    }
    // Cross-check each key against the CPU decode too.
    for (orig, &k) in coords.iter().zip(keys.iter()) {
        let (cx, cy) = hilbert_decode2(u64::from(k));
        assert_eq!([cx, cy], *orig, "cpu decode2 mismatch");
    }
}

#[test]
fn consecutive_keys_are_lattice_neighbours() {
    let Some(ctx) = with_gpu() else {
        return;
    };
    let kernel = GpuHilbert::new(&ctx);

    // Decode a run of consecutive Hilbert indices and confirm the GPU keeps the
    // defining locality property: successive points differ by one unit along a
    // single axis (Manhattan distance 1).
    let keys: Vec<u32> = (0u32..4096).collect();
    let pts = kernel.decode2(&ctx, &keys);
    assert_eq!(pts.len(), keys.len());
    for pair in pts.windows(2) {
        let dx = pair[0][0].abs_diff(pair[1][0]);
        let dy = pair[0][1].abs_diff(pair[1][1]);
        assert_eq!(
            dx + dy,
            1,
            "non-neighbour step {:?} -> {:?}",
            pair[0],
            pair[1]
        );
    }
}

#[test]
fn empty_batch_is_empty() {
    let Some(ctx) = with_gpu() else {
        return;
    };
    let kernel = GpuHilbert::new(&ctx);
    assert!(kernel.encode2(&ctx, &[]).is_empty());
    assert!(kernel.decode2(&ctx, &[]).is_empty());
}
