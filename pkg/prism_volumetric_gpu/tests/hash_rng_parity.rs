//! Real-device parity for the stateless hash / RNG twin: [`GpuHashRng`] must
//! reproduce the `CPU` goldens
//! [`hash_u32`](prism_render_architecture::volumetric::reference::hash_u32) and
//! [`rng_unit`](prism_render_architecture::volumetric::reference::rng_unit)
//! across a deterministic set of seeds, including `0`, small counters, powers of
//! two and boundary values near `u32::MAX` (which exercise the wrapping
//! multiply/add and the shift/xor mixing).
//!
//! The test skips (with a printed notice) when the host has no `wgpu` adapter,
//! so the suite stays green everywhere while still exercising the full
//! dispatch-and-readback on any real device. The kernel is portable
//! core-`WGSL`, so it needs no optional device feature.
//!
//! # Parity criterion
//!
//! `WGSL` `u32` arithmetic wraps exactly like the `CPU` golden's `wrapping_*`,
//! so the hash is asserted bit-exactly and the unit float exactly. The test
//! also checks that the unit float stays in `[0, 1)`.
//!
//! Provenance: standard `Wang`-style integer hash; no Unreal Engine source or
//! derived code.

use prism_render_architecture::volumetric::reference::{hash_u32, rng_unit};
use prism_volumetric_gpu::{GpuContext, GpuHashRng};

#[test]
#[expect(
    clippy::print_stderr,
    reason = "the skip notice must reach the test log on hosts without a wgpu adapter"
)]
fn gpu_hash_rng_matches_cpu_golden() {
    let Some(ctx) = GpuContext::try_headless() else {
        eprintln!("skipping hash-rng parity: no wgpu adapter on this host");
        return;
    };
    let gpu_kernel = GpuHashRng::new(&ctx);

    // Seeds: zero, a dense low run, powers of two, golden-gamma multiples and
    // boundary values near u32::MAX to exercise wrapping arithmetic.
    let mut seeds: Vec<u32> = (0u32..64).collect();
    seeds.extend([
        1u32 << 8,
        1u32 << 16,
        1u32 << 24,
        1u32 << 31,
        0x9E37_79B9,
        0x27D4_EB2D,
        0xDEAD_BEEF,
        0x8000_0000,
        u32::MAX - 1,
        u32::MAX,
    ]);
    assert!(!seeds.is_empty(), "the parity set must be non-empty");

    let gpu = gpu_kernel.eval(&ctx, &seeds);
    assert_eq!(gpu.len(), seeds.len());

    for (&x, got) in seeds.iter().zip(gpu.iter()) {
        let want_hash = hash_u32(x);
        let want_unit = rng_unit(x);
        assert_eq!(
            got.hash, want_hash,
            "hash mismatch for seed {x}: gpu={} cpu={want_hash}",
            got.hash
        );
        assert_eq!(
            got.unit, want_unit,
            "unit mismatch for seed {x}: gpu={} cpu={want_unit}",
            got.unit
        );
        assert!(
            (0.0..1.0).contains(&got.unit),
            "unit float out of [0,1) for seed {x}: {}",
            got.unit
        );
    }
}
