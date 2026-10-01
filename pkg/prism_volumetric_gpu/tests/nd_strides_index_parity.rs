//! Real-device parity for the generic `N-D` strides-index twin:
//! [`GpuNdStridesIndex`](prism_volumetric_gpu::nd_strides_index::GpuNdStridesIndex)
//! must reproduce the `CPU` golden
//! [`nd_strides_index`](prism_render_architecture::particle::nd_strides_index)
//! element for element across all six functions.
//!
//! The fixtures cover the empty scalar shape, single-element shapes, `1D` /
//! `2D` / `3D` / `4D` shapes, both `row-major` and `col-major` strides, the
//! `linear <-> coords` round trip, bounds detection (valid, out of range, wrong
//! rank, empty), a full `MAX_RANK` shape, and the explicit reference values the
//! golden's own unit tests pin.
//!
//! The tests skip (returning early) when the host has no `wgpu` adapter, so the
//! suite stays green everywhere while still exercising the full
//! dispatch-and-readback on any real device such as an Apple `M`-series `GPU`.
//! The kernels are portable core-`WGSL`, so they need no optional device
//! feature.
//!
//! # Parity criterion
//!
//! Every transform is pure unsigned integer arithmetic with no rounding
//! anywhere, so `CPU` and `GPU` must agree exactly. The comparison is a precise
//! `==` on every stride, coordinate, offset, element count and bounds flag, with
//! no tolerance: any mismatch is a genuine port bug.
//!
//! Provenance: 孪生自本仓 `prism_render_architecture::particle::nd_strides_index`；无第三方引擎源码或衍生代码。

use prism_render_architecture::particle::nd_strides_index::{
    col_major_strides, coords_from_linear, coords_in_bounds, linear_from_coords, row_major_strides,
    total_elements,
};
use prism_volumetric_gpu::nd_strides_index::{GpuNdStridesIndex, NdStridesQuery, MAX_RANK};
use prism_volumetric_gpu::GpuContext;

/// Widens a `u32` extent/coordinate slice to the golden's `usize` slice.
fn to_usize(values: &[u32]) -> Vec<usize> {
    values.iter().map(|&v| v as usize).collect()
}

/// Narrows a golden `usize` result tuple to the twin's `u32` domain. The twinned
/// problem sizes stay far below `2^32`, so this is value-preserving.
fn to_u32(values: &[usize]) -> Vec<u32> {
    values.iter().map(|&v| v as u32).collect()
}

/// The shape fixtures spanning the empty scalar, single elements, and `1D`
/// through `4D`, plus a full `MAX_RANK` shape. Every extent is non-zero so the
/// same list is reusable for the division-based inverse mapping.
fn nonzero_shapes() -> Vec<Vec<u32>> {
    vec![
        vec![],
        vec![1],
        vec![5],
        vec![3, 5],
        vec![6, 7],
        vec![2, 3, 4],
        vec![2, 3, 4, 5],
        vec![2, 2, 2, 2, 2, 2, 2, 2],
    ]
}

#[test]
fn row_major_strides_matches_reference() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuNdStridesIndex::new(&ctx);
    let shapes = nonzero_shapes();
    let queries: Vec<NdStridesQuery> = shapes.iter().map(|s| NdStridesQuery::from_shape(s)).collect();
    let got = gpu.row_major_strides(&ctx, &queries);
    assert_eq!(got.len(), shapes.len());
    for (idx, shape) in shapes.iter().enumerate() {
        let want = to_u32(&row_major_strides(&to_usize(shape)));
        assert_eq!(got[idx], want, "row-major strides mismatch for {shape:?}");
    }
    // Explicit golden reference values.
    assert_eq!(got[5], vec![12, 4, 1]);
}

#[test]
fn col_major_strides_matches_reference() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuNdStridesIndex::new(&ctx);
    let shapes = nonzero_shapes();
    let queries: Vec<NdStridesQuery> = shapes.iter().map(|s| NdStridesQuery::from_shape(s)).collect();
    let got = gpu.col_major_strides(&ctx, &queries);
    assert_eq!(got.len(), shapes.len());
    for (idx, shape) in shapes.iter().enumerate() {
        let want = to_u32(&col_major_strides(&to_usize(shape)));
        assert_eq!(got[idx], want, "col-major strides mismatch for {shape:?}");
    }
    // Explicit golden reference values.
    assert_eq!(got[5], vec![1, 2, 6]);
}

#[test]
fn total_elements_matches_reference() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuNdStridesIndex::new(&ctx);
    // Includes zero-extent shapes, which total_elements handles with no division.
    let shapes: Vec<Vec<u32>> = vec![
        vec![],
        vec![0],
        vec![5],
        vec![1, 1, 1],
        vec![2, 3, 4],
        vec![2, 3, 4, 5],
        vec![2, 2, 2, 2, 2, 2, 2, 2],
    ];
    let queries: Vec<NdStridesQuery> = shapes.iter().map(|s| NdStridesQuery::from_shape(s)).collect();
    let got = gpu.total_elements(&ctx, &queries);
    assert_eq!(got.len(), shapes.len());
    for (idx, shape) in shapes.iter().enumerate() {
        let want = total_elements(&to_usize(shape)) as u32;
        assert_eq!(got[idx], want, "total_elements mismatch for {shape:?}");
    }
    assert_eq!(got[0], 1, "empty shape is the scalar");
    assert_eq!(got[1], 0, "zero extent yields zero");
}

#[test]
fn linear_from_coords_matches_reference() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuNdStridesIndex::new(&ctx);
    // (coords, strides) pairs: origin, interior cells, 1D, the empty tuple, and
    // the two golden reference triples, for both major orders.
    let rm = row_major_strides(&[2, 3, 4]);
    let cm = col_major_strides(&[2, 3, 4]);
    let cases: Vec<(Vec<u32>, Vec<u32>)> = vec![
        (vec![], vec![]),
        (vec![5], vec![1]),
        (vec![0, 0, 0], to_u32(&rm)),
        (vec![1, 2, 3], to_u32(&rm)),
        (vec![1, 2, 0], to_u32(&cm)),
    ];
    let queries: Vec<NdStridesQuery> = cases
        .iter()
        .map(|(coords, strides)| NdStridesQuery::from_coords_strides(coords, strides))
        .collect();
    let got = gpu.linear_from_coords(&ctx, &queries);
    assert_eq!(got.len(), cases.len());
    for (idx, (coords, strides)) in cases.iter().enumerate() {
        let want = linear_from_coords(&to_usize(coords), &to_usize(strides)) as u32;
        assert_eq!(got[idx], want, "linear_from_coords mismatch for case {idx}");
    }
    // Explicit golden reference offsets.
    assert_eq!(got[3], 23);
    assert_eq!(got[4], 5);
}

#[test]
fn coords_from_linear_matches_reference_both_orders() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuNdStridesIndex::new(&ctx);
    for &row_major in &[true, false] {
        let mut queries = Vec::new();
        let mut expected = Vec::new();
        for shape in nonzero_shapes() {
            let total = total_elements(&to_usize(&shape)) as u32;
            for linear in 0..total {
                queries.push(NdStridesQuery::from_linear_shape(linear, &shape, row_major));
                expected.push(to_u32(&coords_from_linear(
                    linear as usize,
                    &to_usize(&shape),
                    row_major,
                )));
            }
        }
        let got = gpu.coords_from_linear(&ctx, &queries);
        assert_eq!(got.len(), expected.len());
        for (idx, (have, want)) in got.iter().zip(&expected).enumerate() {
            assert_eq!(
                have, want,
                "coords_from_linear mismatch (row_major={row_major}) at query {idx}"
            );
        }
    }
}

#[test]
fn round_trip_linear_coords_linear_is_identity() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuNdStridesIndex::new(&ctx);
    for &row_major in &[true, false] {
        for shape in nonzero_shapes() {
            if shape.is_empty() {
                continue;
            }
            let strides = if row_major {
                row_major_strides(&to_usize(&shape))
            } else {
                col_major_strides(&to_usize(&shape))
            };
            let strides_u32 = to_u32(&strides);
            let total = total_elements(&to_usize(&shape)) as u32;

            let inverse_queries: Vec<NdStridesQuery> = (0..total)
                .map(|linear| NdStridesQuery::from_linear_shape(linear, &shape, row_major))
                .collect();
            let coords = gpu.coords_from_linear(&ctx, &inverse_queries);

            let forward_queries: Vec<NdStridesQuery> = coords
                .iter()
                .map(|c| NdStridesQuery::from_coords_strides(c, &strides_u32))
                .collect();
            let back = gpu.linear_from_coords(&ctx, &forward_queries);

            assert_eq!(back.len(), total as usize);
            for linear in 0..total {
                assert_eq!(
                    back[linear as usize], linear,
                    "round trip mismatch (row_major={row_major}) for {shape:?} at {linear}"
                );
            }
        }
    }
}

#[test]
fn coords_in_bounds_matches_reference() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuNdStridesIndex::new(&ctx);
    // valid interior, upper-edge valid, out-of-range axis, wrong rank, empty.
    let cases: Vec<(Vec<u32>, Vec<u32>)> = vec![
        (vec![1, 2, 3], vec![2, 3, 4]),
        (vec![0, 0, 0], vec![2, 3, 4]),
        (vec![2, 0, 0], vec![2, 3, 4]),
        (vec![1, 2], vec![2, 3, 4]),
        (vec![], vec![]),
        (vec![1, 1, 1, 1, 1, 1, 1, 1], vec![2, 2, 2, 2, 2, 2, 2, 2]),
    ];
    let queries: Vec<NdStridesQuery> = cases
        .iter()
        .map(|(coords, shape)| NdStridesQuery::from_coords_shape(coords, shape))
        .collect();
    let got = gpu.coords_in_bounds(&ctx, &queries);
    assert_eq!(got.len(), cases.len());
    for (idx, (coords, shape)) in cases.iter().enumerate() {
        let want = coords_in_bounds(&to_usize(coords), &to_usize(shape));
        assert_eq!(got[idx], want, "coords_in_bounds mismatch for case {idx}");
    }
    assert!(got[0], "interior cell is in bounds");
    assert!(!got[2], "out-of-range axis is rejected");
    assert!(!got[3], "wrong rank is rejected");
    assert!(got[4], "empty coords in empty shape is the scalar");
}

#[test]
fn full_max_rank_round_trip() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuNdStridesIndex::new(&ctx);
    // A full MAX_RANK shape: 2^8 = 256 cells enumerated end to end.
    assert_eq!(MAX_RANK, 8);
    let shape = vec![2u32; MAX_RANK];
    let total = total_elements(&to_usize(&shape)) as u32;
    assert_eq!(total, 256);
    let strides = to_u32(&row_major_strides(&to_usize(&shape)));

    let inverse_queries: Vec<NdStridesQuery> = (0..total)
        .map(|linear| NdStridesQuery::from_linear_shape(linear, &shape, true))
        .collect();
    let coords = gpu.coords_from_linear(&ctx, &inverse_queries);
    for linear in 0..total {
        let want = to_u32(&coords_from_linear(linear as usize, &to_usize(&shape), true));
        assert_eq!(coords[linear as usize], want, "full-rank coords at {linear}");
    }

    let forward_queries: Vec<NdStridesQuery> = coords
        .iter()
        .map(|c| NdStridesQuery::from_coords_strides(c, &strides))
        .collect();
    let back = gpu.linear_from_coords(&ctx, &forward_queries);
    for linear in 0..total {
        assert_eq!(back[linear as usize], linear, "full-rank round trip at {linear}");
    }
}

#[test]
fn empty_batch_short_circuits() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuNdStridesIndex::new(&ctx);
    // No dispatch is issued and every entry point returns an empty vector.
    assert!(gpu.row_major_strides(&ctx, &[]).is_empty());
    assert!(gpu.col_major_strides(&ctx, &[]).is_empty());
    assert!(gpu.linear_from_coords(&ctx, &[]).is_empty());
    assert!(gpu.coords_from_linear(&ctx, &[]).is_empty());
    assert!(gpu.total_elements(&ctx, &[]).is_empty());
    assert!(gpu.coords_in_bounds(&ctx, &[]).is_empty());
}
