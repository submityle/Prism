//! Real-device parity for the 2D `Hilbert`-curve `u32` twin:
//! [`GpuHilbertCurve`](prism_volumetric_gpu::hilbert_curve::GpuHilbertCurve)
//! must reproduce the `CPU` golden
//! [`hilbert_curve`](prism_render_architecture::particle::hilbert_curve)
//! element for element across the forward map, the inverse map and the sort
//! key.
//!
//! The fixtures cover several orders (`order = 1`, a few small orders and the
//! `order = 16` full-range boundary), a full-grid enumeration for the small
//! orders, a deterministic `LCG`-generated large batch, and the round trip
//! `xy -> d -> xy` back to the originating cell.
//!
//! The tests skip (returning early) when the host has no `wgpu` adapter, so the
//! suite stays green everywhere while still exercising the full
//! dispatch-and-readback on any real device such as an Apple `M`-series `GPU`.
//! The kernels are portable core-`WGSL`, so they need no optional device
//! feature.
//!
//! # Parity criterion
//!
//! Every transform is pure integer bit arithmetic with no rounding anywhere, so
//! `CPU` and `GPU` must agree bit for bit. The comparison is an exact `==` on
//! every forward index, every inverse `(x, y)` pair and every key, with no
//! tolerance: any mismatch is a genuine port bug. `WGSL` has no `u64`, so the
//! 3D `u64` reference variants are out of scope here.
//!
//! Provenance: 孪生自本仓 `prism_render_architecture::particle::hilbert_curve`；无第三方引擎源码或衍生代码。仅移植 2D u32 变体；3D u64 变体不移植。

use prism_render_architecture::particle::hilbert_curve::{
    hilbert_key, hilbert_to_xy, xy_to_hilbert, MAX_ORDER,
};
use prism_volumetric_gpu::hilbert_curve::GpuHilbertCurve;
use prism_volumetric_gpu::GpuContext;

/// A deterministic linear-congruential generator so the randomized fixtures are
/// reproducible bit for bit across runs and platforms (the same constants the
/// `CPU` golden tests use).
fn lcg_next(state: &mut u64) -> u64 {
    *state = state
        .wrapping_mul(6_364_136_223_846_793_005)
        .wrapping_add(1_442_695_040_888_963_407);
    *state
}

/// Enumerates every `(x, y)` cell of a `2^order × 2^order` grid, row by row.
fn full_grid(order: u32) -> (Vec<u32>, Vec<u32>) {
    let n = 1u32 << order;
    let mut xs = Vec::with_capacity((n * n) as usize);
    let mut ys = Vec::with_capacity((n * n) as usize);
    for x in 0..n {
        for y in 0..n {
            xs.push(x);
            ys.push(y);
        }
    }
    (xs, ys)
}

/// A deterministic batch of `(x, y, order)` triples spanning small and large
/// orders, including the `order = 16` full-range boundary and out-of-range
/// coordinates that the kernel folds back into the grid.
fn random_batch() -> (Vec<u32>, Vec<u32>, Vec<u32>) {
    let mut xs = Vec::new();
    let mut ys = Vec::new();
    let mut orders = Vec::new();
    let mut state = 0x0123_4567_89AB_CDEF_u64;
    for &order in &[1u32, 2, 5, 8, 12, 16] {
        for _ in 0..1024 {
            xs.push((lcg_next(&mut state) >> 32) as u32);
            ys.push((lcg_next(&mut state) >> 32) as u32);
            orders.push(order);
        }
    }
    (xs, ys, orders)
}

#[test]
fn forward_matches_reference_full_grid() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuHilbertCurve::new(&ctx);
    for order in 0..=6u32 {
        let (xs, ys) = full_grid(order);
        let got = gpu.xy_to_hilbert(&ctx, order, &xs, &ys);
        assert_eq!(got.len(), xs.len());
        for idx in 0..xs.len() {
            assert_eq!(
                got[idx],
                xy_to_hilbert(order, xs[idx], ys[idx]),
                "forward mismatch at order {order} for ({}, {})",
                xs[idx],
                ys[idx]
            );
        }
    }
}

#[test]
fn inverse_matches_reference_full_grid() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuHilbertCurve::new(&ctx);
    for order in 0..=6u32 {
        let span = 1u32 << (2 * order);
        let ds: Vec<u32> = (0..span).collect();
        let got = gpu.hilbert_to_xy(&ctx, order, &ds);
        assert_eq!(got.len(), ds.len());
        for (idx, &d) in ds.iter().enumerate() {
            assert_eq!(
                got[idx],
                hilbert_to_xy(order, d),
                "inverse mismatch at order {order} for d {d}"
            );
        }
    }
}

#[test]
fn forward_matches_reference_random_batch() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuHilbertCurve::new(&ctx);
    let (xs, ys, orders) = random_batch();
    // The kernel takes one order per dispatch, so group by order.
    for &order in &[1u32, 2, 5, 8, 12, 16] {
        let sel_x: Vec<u32> = xs
            .iter()
            .zip(&orders)
            .filter_map(|(&v, &o)| (o == order).then_some(v))
            .collect();
        let sel_y: Vec<u32> = ys
            .iter()
            .zip(&orders)
            .filter_map(|(&v, &o)| (o == order).then_some(v))
            .collect();
        let got = gpu.xy_to_hilbert(&ctx, order, &sel_x, &sel_y);
        assert_eq!(got.len(), sel_x.len());
        for idx in 0..sel_x.len() {
            assert_eq!(
                got[idx],
                xy_to_hilbert(order, sel_x[idx], sel_y[idx]),
                "forward mismatch at order {order} for ({}, {})",
                sel_x[idx],
                sel_y[idx]
            );
        }
    }
}

#[test]
fn hilbert_key_matches_reference() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuHilbertCurve::new(&ctx);
    for order in [0u32, 1, 4, 7, 16] {
        let (xs, ys) = if order <= 7 {
            full_grid(order.min(5))
        } else {
            let mut xs = Vec::new();
            let mut ys = Vec::new();
            let mut state = 0xDEAD_BEEF_0000_0001_u64;
            for _ in 0..512 {
                xs.push((lcg_next(&mut state) >> 32) as u32);
                ys.push((lcg_next(&mut state) >> 32) as u32);
            }
            (xs, ys)
        };
        let got = gpu.hilbert_key(&ctx, order, &xs, &ys);
        assert_eq!(got.len(), xs.len());
        for idx in 0..xs.len() {
            assert_eq!(
                got[idx],
                hilbert_key(order, xs[idx], ys[idx]),
                "key mismatch at order {order} for ({}, {})",
                xs[idx],
                ys[idx]
            );
            // The key is the forward map by definition.
            assert_eq!(got[idx], xy_to_hilbert(order, xs[idx], ys[idx]));
        }
    }
}

#[test]
fn round_trip_xy_to_d_to_xy_is_identity() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuHilbertCurve::new(&ctx);
    for order in 0..=6u32 {
        let (xs, ys) = full_grid(order);
        let ds = gpu.xy_to_hilbert(&ctx, order, &xs, &ys);
        let back = gpu.hilbert_to_xy(&ctx, order, &ds);
        assert_eq!(back.len(), xs.len());
        for idx in 0..xs.len() {
            assert_eq!(
                back[idx],
                (xs[idx], ys[idx]),
                "round trip mismatch at order {order} for cell ({}, {})",
                xs[idx],
                ys[idx]
            );
        }
    }
}

#[test]
fn order_sixteen_boundary_round_trip() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuHilbertCurve::new(&ctx);
    // order == MAX_ORDER exercises the total_bits == 32 inverse branch and the
    // full 2^32 index range.
    let order = MAX_ORDER;
    // Coordinates live in `[0, 2^order)`; both the kernel and the reference fold
    // out-of-range inputs into that square, so the round trip can only recover a
    // folded coordinate. Mask the random draws to the order's bit width up front
    // so the inverse comparison is well-defined while still spanning the full
    // 16-bit extent (and thus the full 2^32 index range).
    let coord_mask = (1u32 << order) - 1;
    let mut xs = Vec::new();
    let mut ys = Vec::new();
    let mut state = 0x00C0_FFEE_0000_0001_u64;
    for _ in 0..2048 {
        xs.push(((lcg_next(&mut state) >> 32) as u32) & coord_mask);
        ys.push(((lcg_next(&mut state) >> 32) as u32) & coord_mask);
    }
    let ds = gpu.xy_to_hilbert(&ctx, order, &xs, &ys);
    for idx in 0..xs.len() {
        assert_eq!(ds[idx], xy_to_hilbert(order, xs[idx], ys[idx]));
    }
    let back = gpu.hilbert_to_xy(&ctx, order, &ds);
    for idx in 0..xs.len() {
        assert_eq!(back[idx], (xs[idx], ys[idx]));
    }
}

#[test]
fn empty_input_short_circuits() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuHilbertCurve::new(&ctx);
    // No dispatch is issued and every entry point returns an empty vector.
    assert!(gpu.xy_to_hilbert(&ctx, 8, &[], &[]).is_empty());
    assert!(gpu.hilbert_to_xy(&ctx, 8, &[]).is_empty());
    assert!(gpu.hilbert_key(&ctx, 8, &[], &[]).is_empty());
}
