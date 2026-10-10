//! Real-device parity: the `GPU` SER radix sort reproduces the
//! `prism_render_architecture` `radix_order` permutation bit-for-bit.
//!
//! Reordering rays is a pure permutation, so an exact match between the device
//! payload and the `CPU` golden on randomised and structured inputs is direct
//! evidence of a faithful kernel port — not merely that the shaders compiled.
//! Every test acquires a headless device best-effort and skips cleanly when no
//! adapter is available, so the suite passes on `CI` images without a `GPU` and
//! does real work on an Apple `M`-series device.

use prism_ray_reorder_gpu::{GpuContext, GpuRayReorder};
use prism_render_architecture::ray_scene::reorder::{
    radix_order, CoherenceKey, CoherenceKeyLayout,
};

/// Small xorshift so the property inputs stay dependency-free and reproducible.
struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        x
    }
}

#[expect(
    clippy::print_stderr,
    reason = "the skip notice must reach the test log on hosts without a wgpu adapter"
)]
fn with_gpu(body: impl FnOnce(&GpuContext, &GpuRayReorder)) {
    let Some(ctx) = GpuContext::try_headless() else {
        eprintln!("no GPU adapter available; skipping SER radix parity test");
        return;
    };
    let sorter = GpuRayReorder::new(&ctx);
    body(&ctx, &sorter);
}

#[test]
fn empty_input_is_empty_without_touching_the_device() {
    with_gpu(|ctx, sorter| {
        assert!(sorter.reorder(ctx, &[]).is_empty());
    });
}

#[test]
fn gpu_matches_cpu_golden_on_random_raw_keys() {
    with_gpu(|ctx, sorter| {
        let mut rng = Rng(0x1234_5678_9ABC_DEF1);
        for case in 0..24 {
            // Mix full-width keys with heavily-tied low-byte-only keys so the
            // stable multi-pass ordering is exercised across block boundaries.
            let n = 1 + (rng.next() % 2048) as usize;
            let mask = if case % 3 == 0 { 0xFF } else { u64::MAX };
            let keys: Vec<CoherenceKey> = (0..n).map(|_| CoherenceKey(rng.next() & mask)).collect();
            let gpu = sorter.reorder(ctx, &keys);
            let cpu = radix_order(&keys);
            assert_eq!(gpu, cpu, "case {case}, n={n}, mask={mask:#x}");
        }
    });
}

#[test]
fn gpu_matches_cpu_golden_on_encoded_coherence_keys() {
    with_gpu(|ctx, sorter| {
        let layout = CoherenceKeyLayout::balanced([-1.0; 3], [1.0; 3]).unwrap();
        let mut rng = Rng(0xDEAD_BEEF_0BAD_F00D);
        let n = 4096usize;
        let keys: Vec<CoherenceKey> = (0..n)
            .map(|_| {
                let m = (rng.next() % 8) as u32;
                let dir = [
                    (rng.next() as i64 as f32) / (i64::MAX as f32),
                    (rng.next() as i64 as f32) / (i64::MAX as f32),
                    (rng.next() as i64 as f32) / (i64::MAX as f32),
                ];
                let org = [
                    (rng.next() % 1000) as f32 / 1000.0 * 2.0 - 1.0,
                    (rng.next() % 1000) as f32 / 1000.0 * 2.0 - 1.0,
                    (rng.next() % 1000) as f32 / 1000.0 * 2.0 - 1.0,
                ];
                layout.encode(m, dir, org)
            })
            .collect();
        let gpu = sorter.reorder(ctx, &keys);
        let cpu = radix_order(&keys);
        assert_eq!(gpu, cpu, "encoded coherence keys, n={n}");
    });
}

#[test]
fn gpu_is_stable_on_all_equal_keys() {
    with_gpu(|ctx, sorter| {
        let keys = vec![CoherenceKey(0x00AA_00BB_00CC_00DD); 777];
        let gpu = sorter.reorder(ctx, &keys);
        let identity: Vec<u32> = (0..777).collect();
        assert_eq!(gpu, identity, "equal keys must keep ascending source order");
    });
}
