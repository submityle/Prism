//! §24.4 tests: hot / cold field-separated `SoA` ([`HotCold`]) and the `SoA`
//! auto-layout maths ([`LayoutPlan`] / [`GroupLayout`] / [`ColumnShape`]).

use crate::layout::{
    align_up, ColumnShape, ColumnShapes, GroupLayout, HotCold, LayoutPlan, Temperature, CACHE_LINE,
};

// --- Container: push / index access ----------------------------------------

#[test]
fn hotcold_push_and_indexed_access() {
    // Hot = (position, velocity); Cold = (name_id, spawn_tick).
    let mut w: HotCold<([f32; 3], [f32; 3]), (u64, u64)> = HotCold::new();
    assert!(w.is_empty());
    assert_eq!(w.len(), 0);

    w.push(([0.0, 1.0, 2.0], [1.0, 0.0, 0.0]), (7, 100));
    w.push(([3.0, 4.0, 5.0], [0.0, 1.0, 0.0]), (9, 101));

    assert_eq!(w.len(), 2);
    assert!(!w.is_empty());

    // Hot access.
    assert_eq!(w.get_hot(0), Some((&[0.0, 1.0, 2.0], &[1.0, 0.0, 0.0])));
    assert_eq!(w.get_hot(1), Some((&[3.0, 4.0, 5.0], &[0.0, 1.0, 0.0])));
    // Cold access.
    assert_eq!(w.get_cold(0), Some((&7, &100)));
    assert_eq!(w.get_cold(1), Some((&9, &101)));
    // Combined.
    assert_eq!(
        w.get(1),
        Some(((&[3.0, 4.0, 5.0], &[0.0, 1.0, 0.0]), (&9, &101)))
    );
    // Out of bounds.
    assert_eq!(w.get_hot(2), None);
    assert_eq!(w.get_cold(2), None);
    assert_eq!(w.get(2), None);
}

#[test]
fn hotcold_columns_are_dense_and_independent() {
    let mut w: HotCold<(u32,), (u8,)> = HotCold::with_capacity(4);
    for i in 0..4u32 {
        w.push((i,), (i as u8,));
    }

    // Hot column is one contiguous slice; cold column is a *separate* buffer.
    let (hot_ids,) = w.hot_columns();
    assert_eq!(hot_ids.as_slice(), &[0, 1, 2, 3]);
    let (cold_tags,) = w.cold_columns();
    assert_eq!(cold_tags.as_slice(), &[0u8, 1, 2, 3]);
}

// --- Container: mutation through either half --------------------------------

#[test]
fn hotcold_independent_mutation_of_hot_and_cold() {
    let mut w: HotCold<(i32,), (i32,)> = HotCold::new();
    w.push((1,), (10,));
    w.push((2,), (20,));

    // Mutate only the hot half.
    if let Some((h,)) = w.get_hot_mut(0) {
        *h += 100;
    }
    // Mutate only the cold half.
    if let Some((c,)) = w.get_cold_mut(1) {
        *c += 5;
    }

    assert_eq!(w.get_hot(0), Some((&101,)));
    assert_eq!(w.get_cold(0), Some((&10,))); // untouched
    assert_eq!(w.get_hot(1), Some((&2,))); // untouched
    assert_eq!(w.get_cold(1), Some((&25,)));
}

// --- Container: swap_remove keeps both halves in lockstep -------------------

#[test]
fn hotcold_swap_remove_keeps_halves_consistent() {
    let mut w: HotCold<(char,), (u32,)> = HotCold::new();
    for (i, ch) in ['a', 'b', 'c', 'd'].into_iter().enumerate() {
        w.push((ch,), (i as u32,));
    }

    // Remove the middle row; the last row ('d', 3) swaps into index 1.
    let removed = w.swap_remove(1);
    assert_eq!(removed, Some((('b',), (1,))));
    assert_eq!(w.len(), 3);

    // Both halves must have swapped the SAME row in, staying paired.
    assert_eq!(w.get(0), Some(((&'a',), (&0,))));
    assert_eq!(w.get(1), Some(((&'d',), (&3,))));
    assert_eq!(w.get(2), Some(((&'c',), (&2,))));

    // Out-of-bounds remove is a no-op.
    assert_eq!(w.swap_remove(99), None);
    assert_eq!(w.len(), 3);
}

#[test]
fn hotcold_clear_empties_both_halves() {
    let mut w: HotCold<(u8,), (u8,)> = HotCold::new();
    w.push((1,), (2,));
    w.push((3,), (4,));
    w.clear();
    assert!(w.is_empty());
    assert_eq!(w.len(), 0);
    let (hot,) = w.hot_columns();
    let (cold,) = w.cold_columns();
    assert!(hot.is_empty());
    assert!(cold.is_empty());
    // Still usable after clear.
    w.push((9,), (8,));
    assert_eq!(w.get(0), Some(((&9,), (&8,))));
}

// --- Container: iteration order ---------------------------------------------

#[test]
fn hotcold_iteration_preserves_insertion_order() {
    let mut w: HotCold<(u32,), (&'static str,)> = HotCold::new();
    w.push((10,), ("ten",));
    w.push((20,), ("twenty",));
    w.push((30,), ("thirty",));

    let hot: Vec<u32> = w.iter_hot().map(|(id,)| *id).collect();
    assert_eq!(hot, [10, 20, 30]);

    let cold: Vec<&str> = w.iter_cold().map(|(name,)| *name).collect();
    assert_eq!(cold, ["ten", "twenty", "thirty"]);

    // Zipping hot and cold recovers whole rows in order.
    let rows: Vec<(u32, &str)> = w
        .iter_hot()
        .zip(w.iter_cold())
        .map(|((id,), (name,))| (*id, *name))
        .collect();
    assert_eq!(rows, [(10, "ten"), (20, "twenty"), (30, "thirty")]);
}

// --- Container: boundaries (empty / single element) ------------------------

#[test]
fn hotcold_empty_boundaries() {
    let w: HotCold<(u64,), (u64,)> = HotCold::new();
    assert_eq!(w.len(), 0);
    assert!(w.is_empty());
    assert_eq!(w.get(0), None);
    assert_eq!(w.iter_hot().count(), 0);
    assert_eq!(w.iter_cold().count(), 0);

    // Default == new.
    let d: HotCold<(u64,), (u64,)> = HotCold::default();
    assert!(d.is_empty());
}

#[test]
fn hotcold_single_element_boundary() {
    let mut w: HotCold<(i16,), (i64,)> = HotCold::new();
    w.push((-1,), (-2,));
    assert_eq!(w.len(), 1);
    assert_eq!(w.get(0), Some(((&-1,), (&-2,))));
    assert_eq!(w.iter_hot().count(), 1);

    // Removing the only element empties the container.
    assert_eq!(w.swap_remove(0), Some(((-1,), (-2,))));
    assert!(w.is_empty());
    assert_eq!(w.get(0), None);
}

// --- Layout: alignment / stride maths ---------------------------------------

#[test]
fn align_up_rounds_to_power_of_two() {
    assert_eq!(align_up(0, 16), 0);
    assert_eq!(align_up(1, 16), 16);
    assert_eq!(align_up(16, 16), 16);
    assert_eq!(align_up(17, 16), 32);
    assert_eq!(align_up(31, 8), 32);
    assert_eq!(align_up(100, 1), 100); // align 1 is identity
}

#[test]
fn column_shape_size_align_and_stride() {
    let u32s = ColumnShape::of::<u32>();
    assert_eq!(u32s.size, 4);
    assert_eq!(u32s.align, 4);
    assert_eq!(u32s.stride(), 4);
    // Forcing a 16-byte SIMD lane pads the element stride up.
    assert_eq!(u32s.stride_for(16), 16);

    // [f32; 3] is 12 bytes, 4-byte aligned.
    let v3 = ColumnShape::of::<[f32; 3]>();
    assert_eq!(v3.size, 12);
    assert_eq!(v3.align, 4);
    assert_eq!(v3.stride(), 12);
    assert_eq!(v3.stride_for(16), 16); // padded to a 16-byte lane
}

#[test]
fn group_layout_from_shapes_computes_align_and_width() {
    // A hot group of (u32, [f32; 3]) and a u64.
    let shapes = [
        ColumnShape::of::<u32>(),
        ColumnShape::of::<[f32; 3]>(),
        ColumnShape::of::<u64>(),
    ];
    let g = GroupLayout::from_shapes(Temperature::Hot, &shapes);
    assert_eq!(g.temperature, Temperature::Hot);
    assert_eq!(g.len(), 3);
    assert!(!g.is_empty());
    // group_align = max(4, 4, 8) = 8.
    assert_eq!(g.group_align, 8);
    // lane_bytes = 4 + 12 + 8 = 24.
    assert_eq!(g.lane_bytes(), 24);
    // Field indices are tuple order.
    assert_eq!(g.columns[0].field_index, 0);
    assert_eq!(g.columns[2].field_index, 2);
    // 24 bytes fits a 64-byte cache line.
    assert!(g.fits_cache_line());
    // SIMD-lane-forced width: three 16-byte lanes = 48.
    assert_eq!(g.lane_bytes_for(16), 16 + 16 + 16);
}

#[test]
fn empty_group_layout_is_well_defined() {
    let g = GroupLayout::from_shapes(Temperature::Cold, &[]);
    assert!(g.is_empty());
    assert_eq!(g.len(), 0);
    assert_eq!(g.lane_bytes(), 0);
    assert_eq!(g.group_align, 1); // minimum sane alignment
    assert!(g.fits_cache_line());
}

#[test]
fn layout_plan_splits_hot_and_cold() {
    // Hot = (position, velocity); Cold = (name_id,).
    let plan = LayoutPlan::of::<([f32; 3], [f32; 3]), (u64,)>();
    assert_eq!(plan.hot.temperature, Temperature::Hot);
    assert_eq!(plan.cold.temperature, Temperature::Cold);
    assert_eq!(plan.hot.len(), 2);
    assert_eq!(plan.cold.len(), 1);
    assert_eq!(plan.hot.lane_bytes(), 24); // 12 + 12
    assert_eq!(plan.cold.lane_bytes(), 8);
    assert_eq!(plan.hot.group_align, 4);
    assert_eq!(plan.cold.group_align, 8);

    // from_shapes agrees with of::<_, _>.
    let hot = <([f32; 3], [f32; 3]) as ColumnShapes>::column_shapes();
    let cold = <(u64,) as ColumnShapes>::column_shapes();
    assert_eq!(LayoutPlan::from_shapes(&hot, &cold), plan);
}

#[test]
fn hotcold_exposes_its_layout_plan() {
    type World = HotCold<([f32; 3], [f32; 3]), (u64, u32)>;
    let plan = World::layout_plan();
    assert_eq!(plan.hot.len(), 2);
    assert_eq!(plan.cold.len(), 2);
    assert_eq!(plan.hot.lane_bytes(), 24);
    // Cold = (u64, u32) => 8 + 4 = 12 bytes, 8-byte aligned.
    assert_eq!(plan.cold.lane_bytes(), 12);
    assert_eq!(plan.cold.group_align, 8);
    assert!(plan.hot.fits_cache_line());
    assert!(CACHE_LINE >= plan.hot.lane_bytes());
}

#[test]
fn column_shapes_arity_matches_tuple() {
    assert_eq!(<(u8,) as ColumnShapes>::ARITY, 1);
    assert_eq!(<(u8, u16, u32, u64) as ColumnShapes>::ARITY, 4);
    assert_eq!(
        <(u8, u8, u8, u8, u8, u8, u8, u8) as ColumnShapes>::ARITY,
        8
    );
    assert_eq!(
        <(u8, u16, u32, u64, u8, u16, u32, u64) as ColumnShapes>::column_shapes().len(),
        8
    );
}
