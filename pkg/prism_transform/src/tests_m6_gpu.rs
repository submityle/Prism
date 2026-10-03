//! M6 GPU-upload tests: byte layout matches a hand-computed packing, the
//! incremental uploader emits exactly the dirty byte ranges (and nothing on a
//! static frame), and consecutive dirty entries merge into one span.

use crate::gpu_upload::{GpuColumnBuffer, MatrixLayout, UploadRange};
use crate::{GlobalTransform, Transform, TransformGraph};
use prism_math::{Vec3, vec3};

/// Little-endian bytes of a `f32` slice, the independent "expected" packing.
fn le_bytes(floats: &[f32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(floats.len() * 4);
    for f in floats {
        out.extend_from_slice(&f.to_le_bytes());
    }
    out
}

#[test]
fn pack_3x4_matches_hand_computed_row_major_bytes() {
    // Pure scale + translation (no rotation) keeps the basis exact in f32.
    let t = Transform::from_translation(vec3(1.0, 2.0, 3.0)).with_scale(vec3(2.0, 3.0, 4.0));
    let g = GlobalTransform::from_transform(&t);

    let mut buf = GpuColumnBuffer::new(MatrixLayout::RowMajor3x4);
    let plan = buf.pack_all(core::slice::from_ref(&g));

    // Row-major 3x4: row r = [basis_col0[r], basis_col1[r], basis_col2[r], t[r]].
    let expected_floats = [
        2.0, 0.0, 0.0, 1.0, //
        0.0, 3.0, 0.0, 2.0, //
        0.0, 0.0, 4.0, 3.0, //
    ];
    assert_eq!(buf.stride(), 48);
    assert_eq!(buf.entry_count(), 1);
    assert_eq!(buf.as_bytes(), le_bytes(&expected_floats).as_slice());
    assert_eq!(plan, vec![UploadRange { offset: 0, len: 48 }]);
}

#[test]
fn pack_4x4_appends_homogeneous_bottom_row() {
    let t = Transform::from_translation(vec3(5.0, 6.0, 7.0));
    let g = GlobalTransform::from_transform(&t);

    let mut buf = GpuColumnBuffer::new(MatrixLayout::RowMajor4x4);
    buf.pack_all(core::slice::from_ref(&g));

    let expected_floats = [
        1.0, 0.0, 0.0, 5.0, //
        0.0, 1.0, 0.0, 6.0, //
        0.0, 0.0, 1.0, 7.0, //
        0.0, 0.0, 0.0, 1.0, //
    ];
    assert_eq!(buf.stride(), 64);
    assert_eq!(buf.as_bytes(), le_bytes(&expected_floats).as_slice());
}

#[test]
fn incremental_emits_exactly_the_dirty_subtree_range() {
    // root(0) -> child(1) -> grandchild(2); plus a lone root(3).
    let mut g = TransformGraph::new();
    let r0 = g.spawn_root(Transform::from_xyz(0.0, 0.0, 0.0));
    let c1 = g.spawn_child(r0, Transform::from_xyz(1.0, 0.0, 0.0));
    let _g2 = g.spawn_child(c1, Transform::from_xyz(1.0, 0.0, 0.0));
    let _r3 = g.spawn_root(Transform::from_xyz(9.0, 9.0, 9.0));

    // Baseline: full propagate + full pack.
    g.propagate_incremental_stats();
    let mut buf = GpuColumnBuffer::new(MatrixLayout::RowMajor3x4);
    buf.pack_all(g.globals());
    let stride = buf.stride();

    // Edit only node 1; its world and its descendant node 2 recompute.
    g.set_local(c1, Transform::from_xyz(1.0, 5.0, 0.0));
    g.propagate_incremental_stats();

    let dirty: Vec<u32> = g
        .recomputed_entries()
        .iter()
        .map(|n| n.index() as u32)
        .collect();
    let mut dirty_sorted = dirty.clone();
    dirty_sorted.sort_unstable();
    assert_eq!(dirty_sorted, vec![1, 2], "subtree of node 1 is {{1, 2}}");

    let plan = buf.pack_dirty(g.globals(), &dirty);
    // Consecutive indices 1,2 collapse to a single span starting at entry 1.
    assert_eq!(
        plan,
        vec![UploadRange { offset: stride, len: 2 * stride }],
    );

    // The repacked entry-1 bytes equal a fresh pack of its new world matrix.
    let mut one = GpuColumnBuffer::new(MatrixLayout::RowMajor3x4);
    one.pack_all(core::slice::from_ref(&g.global(c1)));
    assert_eq!(&buf.as_bytes()[stride..2 * stride], one.as_bytes());

    // Untouched entries 0 and 3 are byte-identical to a full repack of them.
    let mut all = GpuColumnBuffer::new(MatrixLayout::RowMajor3x4);
    all.pack_all(g.globals());
    assert_eq!(&buf.as_bytes()[0..stride], &all.as_bytes()[0..stride]);
    assert_eq!(&buf.as_bytes()[3 * stride..4 * stride], &all.as_bytes()[3 * stride..4 * stride]);
}

#[test]
fn static_frame_uploads_nothing() {
    let mut g = TransformGraph::new();
    let r = g.spawn_root(Transform::IDENTITY);
    g.spawn_child(r, Transform::from_xyz(1.0, 1.0, 1.0));
    g.propagate_incremental_stats();

    let mut buf = GpuColumnBuffer::new(MatrixLayout::RowMajor3x4);
    buf.pack_all(g.globals());
    let before = buf.as_bytes().to_vec();

    // No edits -> no recomputed entries -> empty plan, buffer unchanged.
    g.propagate_incremental_stats();
    assert!(g.recomputed_entries().is_empty());
    let dirty: Vec<u32> = g.recomputed_entries().iter().map(|n| n.index() as u32).collect();
    let plan = buf.pack_dirty(g.globals(), &dirty);
    assert!(plan.is_empty());
    assert_eq!(buf.as_bytes(), before.as_slice());
}

#[test]
fn pack_dirty_merges_runs_and_sorts_input() {
    let globals: Vec<GlobalTransform> = (0..6)
        .map(|i| GlobalTransform::from_transform(&Transform::from_translation(Vec3::splat(i as f32))))
        .collect();
    let mut buf = GpuColumnBuffer::new(MatrixLayout::RowMajor4x4);
    buf.pack_all(&globals);
    let stride = buf.stride();

    // Unsorted, with a duplicate and a gap: {5,1,3,2,2} -> {1,2,3} and {5}.
    let plan = buf.pack_dirty(&globals, &[5, 1, 3, 2, 2]);
    assert_eq!(
        plan,
        vec![
            UploadRange { offset: stride, len: 3 * stride },
            UploadRange { offset: 5 * stride, len: stride },
        ],
    );
}

#[test]
fn out_of_range_dirty_indices_are_ignored() {
    let globals: Vec<GlobalTransform> = (0..3)
        .map(|_| GlobalTransform::IDENTITY)
        .collect();
    let mut buf = GpuColumnBuffer::new(MatrixLayout::RowMajor3x4);
    buf.pack_all(&globals);
    // Index 7 does not exist; only entry 0 is valid here.
    let plan = buf.pack_dirty(&globals, &[7, 0]);
    assert_eq!(plan, vec![UploadRange { offset: 0, len: buf.stride() }]);
}

#[test]
fn entry_range_is_stride_aligned() {
    let buf = GpuColumnBuffer::new(MatrixLayout::RowMajor3x4);
    assert_eq!(buf.entry_range(2), UploadRange { offset: 96, len: 48 });
}
