//! Real-device parity for the §24.1 octahedral normal-codec shader mirror.
//!
//! The CPU codec [`prism_math::octahedral`] encodes a unit direction to two
//! numbers (or a snorm-packed `u32`) and decodes it back; this suite runs the
//! same codec on a real `GPU` from the single-sourced
//! [`WGSL_OCTAHEDRAL`](prism_math::shader_mirror::WGSL_OCTAHEDRAL) fragment and
//! checks parity on a deterministic spread of directions over the sphere.
//!
//! The encode L1-normalize divide is fast-math-sensitive on Metal, so the
//! full-precision parity is a tolerance round-trip and the snorm-pack parity is
//! defined on the reconstructed direction plus a +/-1 tolerance on each 16-bit
//! quantized code. The suite skips gracefully when no adapter is available.

use prism_math::octahedral;
use prism_math::{Vec2, Vec3};
use prism_math_gpu::GpuContext;
use prism_math_gpu::GpuOctahedral;

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

/// A deterministic spread of unit directions over the sphere, built from a
/// transcendental-free integer lattice so it covers every octant and both
/// hemispheres (exercising the octahedral fold branches). Includes the six
/// axis directions plus every normalized nonzero integer point in
/// `[-3, 3]^3`.
fn sample_dirs() -> Vec<Vec3> {
    let mut v = vec![
        Vec3::X,
        -Vec3::X,
        Vec3::Y,
        -Vec3::Y,
        Vec3::Z,
        -Vec3::Z,
    ];
    for x in -3..=3 {
        for y in -3..=3 {
            for z in -3..=3 {
                if x == 0 && y == 0 && z == 0 {
                    continue;
                }
                v.push(Vec3::new(x as f32, y as f32, z as f32).normalize());
            }
        }
    }
    v
}

fn dot4(a: [f32; 4], b: Vec3) -> f32 {
    a[0] * b.x + a[1] * b.y + a[2] * b.z
}

#[test]
fn empty_batches_are_empty() {
    let Some(ctx) = with_gpu() else {
        return;
    };
    let oct = GpuOctahedral::new(&ctx);
    assert!(oct.encode(&ctx, &[]).is_empty());
    assert!(oct.decode(&ctx, &[]).is_empty());
    assert!(oct.pack(&ctx, &[]).is_empty());
    assert!(oct.unpack(&ctx, &[]).is_empty());
}

#[test]
fn encode_matches_cpu() {
    let Some(ctx) = with_gpu() else {
        return;
    };
    let oct = GpuOctahedral::new(&ctx);
    let dirs = sample_dirs();
    let input: Vec<[f32; 4]> = dirs.iter().map(|n| [n.x, n.y, n.z, 0.0]).collect();

    let gpu = oct.encode(&ctx, &input);
    assert_eq!(gpu.len(), dirs.len());
    for (i, n) in dirs.iter().enumerate() {
        let cpu = octahedral::encode(*n);
        let dx = (gpu[i][0] - cpu.x).abs();
        let dy = (gpu[i][1] - cpu.y).abs();
        assert!(dx < 1.0e-5 && dy < 1.0e-5, "encode {i}: gpu={:?} cpu={cpu:?}", gpu[i]);
    }
}

#[test]
fn decode_matches_cpu() {
    let Some(ctx) = with_gpu() else {
        return;
    };
    let oct = GpuOctahedral::new(&ctx);
    let dirs = sample_dirs();
    // Feed CPU-encoded coordinates so both sides decode the identical input.
    let coords: Vec<[f32; 2]> = dirs
        .iter()
        .map(|n| {
            let e = octahedral::encode(*n);
            [e.x, e.y]
        })
        .collect();

    let gpu = oct.decode(&ctx, &coords);
    assert_eq!(gpu.len(), dirs.len());
    for (i, c) in coords.iter().enumerate() {
        let cpu = octahedral::decode(Vec2::new(c[0], c[1]));
        // GPU vs CPU decode of the same input agree to a tight tolerance, and
        // the result reconstructs the original direction.
        assert!(dot4(gpu[i], cpu) > 1.0 - 1.0e-5, "decode {i} vs cpu: {:?}", gpu[i]);
        assert!(dot4(gpu[i], dirs[i]) > 1.0 - 1.0e-5, "decode {i} vs orig: {:?}", gpu[i]);
    }
}

#[test]
fn pack_matches_cpu() {
    let Some(ctx) = with_gpu() else {
        return;
    };
    let oct = GpuOctahedral::new(&ctx);
    let dirs = sample_dirs();
    let input: Vec<[f32; 4]> = dirs.iter().map(|n| [n.x, n.y, n.z, 0.0]).collect();

    let gpu = oct.pack(&ctx, &input);
    assert_eq!(gpu.len(), dirs.len());
    for (i, n) in dirs.iter().enumerate() {
        let cpu = octahedral::pack_snorm(*n);
        // Each 16-bit snorm channel agrees within +/-1 (a fast-math round may
        // cross a quantization boundary); the reconstructed direction is the
        // real `GBuffer`-correctness criterion.
        let (gx, gy) = ((gpu[i] & 0xFFFF) as i32, ((gpu[i] >> 16) & 0xFFFF) as i32);
        let (cx, cy) = ((cpu & 0xFFFF) as i32, ((cpu >> 16) & 0xFFFF) as i32);
        let dx = ((gx as i16) as i32 - (cx as i16) as i32).abs();
        let dy = ((gy as i16) as i32 - (cy as i16) as i32).abs();
        assert!(dx <= 1 && dy <= 1, "pack {i}: gpu={:#010x} cpu={cpu:#010x}", gpu[i]);
        let d = octahedral::unpack_snorm(gpu[i]);
        assert!(n.dot(d) > 1.0 - 1.0e-3, "pack {i} direction: dot={}", n.dot(d));
    }
}

#[test]
fn unpack_matches_cpu() {
    let Some(ctx) = with_gpu() else {
        return;
    };
    let oct = GpuOctahedral::new(&ctx);
    let dirs = sample_dirs();
    // Feed CPU-packed codes so both sides unpack the identical input.
    let bits: Vec<u32> = dirs.iter().map(|n| octahedral::pack_snorm(*n)).collect();

    let gpu = oct.unpack(&ctx, &bits);
    assert_eq!(gpu.len(), dirs.len());
    for (i, b) in bits.iter().enumerate() {
        let cpu = octahedral::unpack_snorm(*b);
        assert!(dot4(gpu[i], cpu) > 1.0 - 1.0e-5, "unpack {i} vs cpu: {:?}", gpu[i]);
    }
}
