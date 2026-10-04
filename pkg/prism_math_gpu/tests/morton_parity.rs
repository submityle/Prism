//! Real-device parity for the §24.1 / §24.7 Morton (Z-order) spatial-key
//! shader mirror.
//!
//! Each test encodes a batch of integer lattice points into Morton sort keys
//! on a real `GPU` from the single-sourced
//! [`WGSL_MORTON`](prism_math::shader_mirror::WGSL_MORTON) fragment and diffs
//! them against the CPU reference
//! ([`prism_math::spatial::morton_encode2`] / [`morton_encode3`]). Morton keys
//! are pure integer bit interleaving, so unlike the floating-point twins this
//! parity is **bit-exact**, not a tolerance. WGSL has no 64-bit integer type,
//! so the kernels run at the GPU-representable key widths (16 bits per axis in
//! 2D, 10 in 3D); for inputs masked to those widths the GPU key equals the low
//! bits of the wider CPU encoder bit-for-bit, and the decode is its exact
//! inverse. The suite skips gracefully when no adapter is available.

use prism_math::spatial::{morton_decode2, morton_decode3, morton_encode2, morton_encode3};
use prism_math_gpu::GpuContext;
use prism_math_gpu::GpuMorton;
use prism_math_gpu::morton::{MAX_COORD_2D, MAX_COORD_3D};

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
    let kernel = GpuMorton::new(&ctx);

    let mut coords: Vec<[u32; 2]> = vec![
        [0, 0],
        [MAX_COORD_2D, MAX_COORD_2D],
        [MAX_COORD_2D, 0],
        [0, MAX_COORD_2D],
        [1, 2],
    ];
    let mut seed = 0x1234_5678u32;
    for _ in 0..200 {
        coords.push([lcg(&mut seed) & MAX_COORD_2D, lcg(&mut seed) & MAX_COORD_2D]);
    }

    let gpu = kernel.encode2(&ctx, &coords);
    assert_eq!(gpu.len(), coords.len());
    for (c, &g) in coords.iter().zip(gpu.iter()) {
        let cpu = morton_encode2(c[0], c[1]) as u32;
        assert_eq!(g, cpu, "encode2 drift at ({}, {})", c[0], c[1]);
    }
}

#[test]
fn decode2_inverts_encode2() {
    let Some(ctx) = with_gpu() else {
        return;
    };
    let kernel = GpuMorton::new(&ctx);

    let mut coords: Vec<[u32; 2]> = vec![[0, 0], [MAX_COORD_2D, MAX_COORD_2D], [7, 11]];
    let mut seed = 0x0bad_f00du32;
    for _ in 0..200 {
        coords.push([lcg(&mut seed) & MAX_COORD_2D, lcg(&mut seed) & MAX_COORD_2D]);
    }

    let keys = kernel.encode2(&ctx, &coords);
    let back = kernel.decode2(&ctx, &keys);
    assert_eq!(back.len(), coords.len());
    for (orig, got) in coords.iter().zip(back.iter()) {
        assert_eq!(got, orig, "decode2 round-trip drift");
        // Cross-check the key against the CPU decode too.
        let i = coords.iter().position(|c| c == orig).unwrap();
        let (cx, cy) = morton_decode2(u64::from(keys[i]));
        assert_eq!([cx, cy], *orig, "cpu decode2 mismatch");
    }
}

#[test]
fn encode3_matches_cpu_reference() {
    let Some(ctx) = with_gpu() else {
        return;
    };
    let kernel = GpuMorton::new(&ctx);

    let mut coords: Vec<[u32; 3]> = vec![
        [0, 0, 0],
        [MAX_COORD_3D, MAX_COORD_3D, MAX_COORD_3D],
        [MAX_COORD_3D, 0, 0],
        [0, MAX_COORD_3D, 0],
        [0, 0, MAX_COORD_3D],
        [1, 2, 3],
    ];
    let mut seed = 0x9e37_79b9u32;
    for _ in 0..200 {
        coords.push([
            lcg(&mut seed) & MAX_COORD_3D,
            lcg(&mut seed) & MAX_COORD_3D,
            lcg(&mut seed) & MAX_COORD_3D,
        ]);
    }

    let gpu = kernel.encode3(&ctx, &coords);
    assert_eq!(gpu.len(), coords.len());
    for (c, &g) in coords.iter().zip(gpu.iter()) {
        let cpu = morton_encode3(c[0], c[1], c[2]) as u32;
        assert_eq!(g, cpu, "encode3 drift at ({}, {}, {})", c[0], c[1], c[2]);
    }
}

#[test]
fn decode3_inverts_encode3() {
    let Some(ctx) = with_gpu() else {
        return;
    };
    let kernel = GpuMorton::new(&ctx);

    let mut coords: Vec<[u32; 3]> =
        vec![[0, 0, 0], [MAX_COORD_3D, MAX_COORD_3D, MAX_COORD_3D], [5, 9, 13]];
    let mut seed = 0xdead_beefu32;
    for _ in 0..200 {
        coords.push([
            lcg(&mut seed) & MAX_COORD_3D,
            lcg(&mut seed) & MAX_COORD_3D,
            lcg(&mut seed) & MAX_COORD_3D,
        ]);
    }

    let keys = kernel.encode3(&ctx, &coords);
    let back = kernel.decode3(&ctx, &keys);
    assert_eq!(back.len(), coords.len());
    for (orig, got) in coords.iter().zip(back.iter()) {
        assert_eq!(got, orig, "decode3 round-trip drift");
    }
    for (orig, &k) in coords.iter().zip(keys.iter()) {
        let (cx, cy, cz) = morton_decode3(u64::from(k));
        assert_eq!([cx, cy, cz], *orig, "cpu decode3 mismatch");
    }
}

#[test]
fn empty_batch_is_empty() {
    let Some(ctx) = with_gpu() else {
        return;
    };
    let kernel = GpuMorton::new(&ctx);
    assert!(kernel.encode2(&ctx, &[]).is_empty());
    assert!(kernel.decode2(&ctx, &[]).is_empty());
    assert!(kernel.encode3(&ctx, &[]).is_empty());
    assert!(kernel.decode3(&ctx, &[]).is_empty());
}
