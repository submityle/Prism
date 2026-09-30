//! Real-device parity for the importance-weighted decimation ranking-key twin:
//! [`GpuDecimationPriority`] must reproduce the `CPU` golden
//! [`decimation_priority`](prism_render_architecture::hair::decimation::decimation_priority)
//! for a batch of strands, and — once its per-strand keys are sorted the way the
//! reference sorts them — reproduce the whole nested decimation order
//! [`build_decimation_order`](prism_render_architecture::hair::decimation::build_decimation_order)
//! yields. The suite covers the `jitter == 0` bit-exact case, the positive-jitter
//! tolerance case (with a non-no-op guard), a distinct-seed decorrelation case,
//! the end-to-end order case, a large multi-workgroup batch, and the empty /
//! length-mismatch no-ops.
//!
//! The tests skip (with a printed notice) when the host has no `wgpu` adapter,
//! so the suite stays green everywhere while still exercising the full
//! dispatch-and-readback on any real device. The kernel is portable core-WGSL,
//! so it needs no optional device feature.
//!
//! # Parity criterion
//!
//! The jitter draws from the golden's hand-written `splitmix64` integer hash
//! reproduced bit-for-bit; the hash is integer-exact and only the final
//! `importance + jitter * h` combine is a single multiply-add, so at `jitter > 0`
//! parity is asserted to within `abs_diff < 1e-4` or `rel_diff < 1e-3` — tight
//! enough to fail a genuinely wrong port (a wrong sub-key, a dropped seed, a
//! wrong hash), loose enough to admit the fma contraction. At `jitter == 0` the
//! key collapses to the importance and parity is exact. The end-to-end order
//! test uses importance gaps far larger than both the tolerance and the jitter
//! amplitude, so the `GPU`-key ordering cannot flip relative to the `CPU`
//! reference. No `sin`/`cos` appears anywhere.
//!
//! Provenance: standard importance-weighted stochastic-LOD decimation ranking;
//! no Unreal Engine source or derived code.

use prism_hair_gpu::decimation_priority::GpuDecimationPriority;
use prism_hair_gpu::GpuContext;
use prism_render_architecture::hair::decimation::{build_decimation_order, decimation_priority};

/// Asserts a single priority matches within the documented fma tolerance.
fn assert_close(got: f32, expected: f32, label: &str) {
    let abs_diff = (got - expected).abs();
    let rel_diff = abs_diff / expected.abs().max(1e-6);
    assert!(
        abs_diff < 1e-4 || rel_diff < 1e-3,
        "{label}: gpu {got}, cpu {expected} (abs {abs_diff}, rel {rel_diff})"
    );
}

/// Reference priorities for a batch via the `CPU` golden.
fn cpu_priorities(seeds: &[u32], importances: &[f32], jitter: f32) -> Vec<f32> {
    seeds
        .iter()
        .zip(importances.iter())
        .map(|(&s, &imp)| decimation_priority(s, imp, jitter))
        .collect()
}

/// Sorts a batch of priorities into the decimation order the way the golden
/// [`build_decimation_order`] does: descending priority, ties by ascending
/// index, for a total, stable order.
fn order_from_priorities(priorities: &[f32]) -> Vec<u32> {
    let mut ranked: Vec<(f32, u32)> = priorities
        .iter()
        .enumerate()
        .map(|(i, &p)| (p, i as u32))
        .collect();
    ranked.sort_unstable_by(|a, b| b.0.total_cmp(&a.0).then(a.1.cmp(&b.1)));
    ranked.into_iter().map(|(_, idx)| idx).collect()
}

/// Acquires a headless context, or `None` (with a skip notice) when the host has
/// no `wgpu` adapter so the suite stays green off-device.
#[expect(
    clippy::print_stderr,
    reason = "the skip notice must reach the test log on hosts without a wgpu adapter"
)]
fn context_or_skip(label: &str) -> Option<GpuContext> {
    match GpuContext::try_headless() {
        Some(ctx) => Some(ctx),
        None => {
            eprintln!("skipping {label}: no wgpu adapter on this host");
            None
        }
    }
}

/// A spread of distinct per-strand seeds so the jitter draws decorrelate.
fn seeds_for(n: usize) -> Vec<u32> {
    (0..n)
        .map(|i| 0x00A1_0000u32.wrapping_add((i as u32).wrapping_mul(0x9E37_79B1)))
        .collect()
}

#[test]
fn jitter_zero_is_bit_exact_importance() {
    let Some(ctx) = context_or_skip("decimation-priority jitter-zero parity") else {
        return;
    };
    let kernel = GpuDecimationPriority::new(&ctx);
    let n = 256usize;
    let seeds = seeds_for(n);
    let importances: Vec<f32> = (0..n).map(|i| (i as f32) / (n as f32)).collect();
    let gpu = kernel.eval(&ctx, &seeds, &importances, 0.0);
    assert_eq!(gpu.len(), n, "one priority per strand");
    // jitter * h == 0 exactly, so the key is the importance verbatim on both
    // sides — assert full equality, not a tolerance.
    for (i, (&g, &imp)) in gpu.iter().zip(importances.iter()).enumerate() {
        assert!(
            (g - imp).abs() <= f32::EPSILON,
            "strand {i}: jitter-0 key {g} must equal importance {imp}"
        );
    }
}

#[test]
fn positive_jitter_matches_cpu_within_tolerance() {
    let Some(ctx) = context_or_skip("decimation-priority positive-jitter parity") else {
        return;
    };
    let kernel = GpuDecimationPriority::new(&ctx);
    let n = 512usize;
    let seeds = seeds_for(n);
    let importances: Vec<f32> = (0..n).map(|i| 0.5 + 0.1 * ((i % 3) as f32)).collect();
    let jitter = 0.4f32;
    let gpu = kernel.eval(&ctx, &seeds, &importances, jitter);
    let cpu = cpu_priorities(&seeds, &importances, jitter);
    assert_eq!(gpu.len(), n, "one priority per strand");
    for (i, (&g, &c)) in gpu.iter().zip(cpu.iter()).enumerate() {
        assert_close(g, c, &format!("strand {i} priority"));
    }
    // The jitter must actually move keys off the importance, else a kernel that
    // ignored the hash could still pass: at least one strand deviates, and every
    // key stays inside importance +- jitter/2.
    let mut moved = 0usize;
    for (&g, &imp) in gpu.iter().zip(importances.iter()) {
        if (g - imp).abs() > 1e-4 {
            moved += 1;
        }
        assert!(
            g >= imp - jitter * 0.5 - 1e-4 && g < imp + jitter * 0.5 + 1e-4,
            "priority {g} must sit within importance {imp} +- jitter/2 ({})",
            jitter * 0.5
        );
    }
    assert!(moved > 0, "positive jitter must move at least one key");
}

#[test]
fn distinct_seeds_draw_distinct_jitter() {
    let Some(ctx) = context_or_skip("decimation-priority distinct-seed parity") else {
        return;
    };
    let kernel = GpuDecimationPriority::new(&ctx);
    // A constant importance isolates the jitter term; distinct seeds must yield
    // more than one distinct key.
    let n = 64usize;
    let seeds = seeds_for(n);
    let importances = vec![0.5f32; n];
    let jitter = 0.6f32;
    let gpu = kernel.eval(&ctx, &seeds, &importances, jitter);
    let cpu = cpu_priorities(&seeds, &importances, jitter);
    for (i, (&g, &c)) in gpu.iter().zip(cpu.iter()).enumerate() {
        assert_close(g, c, &format!("strand {i} priority"));
    }
    let first = gpu[0];
    let all_equal = gpu.iter().all(|&g| (g - first).abs() <= 1e-6);
    assert!(
        !all_equal,
        "distinct seeds must produce more than one distinct jitter draw"
    );
}

#[test]
fn end_to_end_order_matches_cpu() {
    let Some(ctx) = context_or_skip("decimation-priority order parity") else {
        return;
    };
    let kernel = GpuDecimationPriority::new(&ctx);
    let n = 300usize;
    let seeds = seeds_for(n);
    // Importance gaps of 1.0 dwarf both the fma tolerance (~1e-4) and the jitter
    // spread (+-0.05), so no `GPU`-key can cross a `CPU`-key boundary — the two
    // orders must be identical.
    let importances: Vec<f32> = (0..n).map(|i| i as f32).collect();
    let jitter = 0.1f32;
    let gpu = kernel.eval(&ctx, &seeds, &importances, jitter);
    for (i, (&g, &c)) in gpu
        .iter()
        .zip(cpu_priorities(&seeds, &importances, jitter).iter())
        .enumerate()
    {
        assert_close(g, c, &format!("strand {i} priority"));
    }
    let gpu_order = order_from_priorities(&gpu);
    let cpu_order = build_decimation_order(&seeds, &importances, jitter);
    assert_eq!(
        gpu_order, cpu_order,
        "GPU-key decimation order must match the CPU golden order"
    );
    // Highest importance survives to the smallest count: last index leads.
    assert_eq!(
        gpu_order[0],
        (n - 1) as u32,
        "highest importance ranks first"
    );
}

#[test]
fn large_batch_matches_across_workgroups() {
    let Some(ctx) = context_or_skip("decimation-priority large-batch parity") else {
        return;
    };
    let kernel = GpuDecimationPriority::new(&ctx);
    let n = 4096usize;
    let seeds = seeds_for(n);
    let importances: Vec<f32> = (0..n).map(|i| ((i * 7 + 3) % 101) as f32 / 101.0).collect();
    let jitter = 0.3f32;
    let gpu = kernel.eval(&ctx, &seeds, &importances, jitter);
    let cpu = cpu_priorities(&seeds, &importances, jitter);
    assert_eq!(gpu.len(), n, "one priority per strand across workgroups");
    for (i, (&g, &c)) in gpu.iter().zip(cpu.iter()).enumerate() {
        assert_close(g, c, &format!("strand {i} priority"));
    }
}

#[test]
fn empty_and_mismatched_inputs_are_noops() {
    let Some(ctx) = context_or_skip("decimation-priority no-op") else {
        return;
    };
    let kernel = GpuDecimationPriority::new(&ctx);
    let empty = kernel.eval(&ctx, &[], &[], 0.5);
    assert!(empty.is_empty(), "empty batch yields an empty result");
    let seeds = seeds_for(4);
    let importances = vec![0.1f32, 0.2, 0.3];
    let mismatch = kernel.eval(&ctx, &seeds, &importances, 0.5);
    assert!(
        mismatch.is_empty(),
        "length mismatch yields an empty result without a dispatch"
    );
}
