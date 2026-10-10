//! Unit tests for the [`AoSoa`](crate::layout::AoSoa) tiled columnar container.

use crate::layout::AoSoa;

#[test]
fn empty_container_is_well_defined() {
    let v: AoSoa<(u32, f32), 4> = AoSoa::new();
    assert_eq!(v.len(), 0);
    assert!(v.is_empty());
    assert_eq!(v.block_count(), 0);
    assert_eq!(v.lane_width(), 4);
    assert_eq!(v.get(0), None);
    assert_eq!(v.iter().count(), 0);
}

#[test]
fn push_opens_blocks_at_lane_boundaries() {
    let mut v: AoSoa<(u32,), 4> = AoSoa::new();
    for i in 0..9u32 {
        v.push((i,));
    }
    assert_eq!(v.len(), 9);
    // 9 rows / 4 per block => 3 blocks of 4, 4, 1.
    assert_eq!(v.block_count(), 3);
    assert_eq!(v.blocks()[0].len(), 4);
    assert_eq!(v.blocks()[1].len(), 4);
    assert_eq!(v.blocks()[2].len(), 1);
}

#[test]
fn exact_multiple_does_not_open_trailing_block() {
    let mut v: AoSoa<(u32,), 4> = AoSoa::new();
    for i in 0..8u32 {
        v.push((i,));
    }
    assert_eq!(v.block_count(), 2);
    v.push((8,));
    assert_eq!(v.block_count(), 3);
}

#[test]
fn get_resolves_block_and_offset() {
    let mut v: AoSoa<(u32, f32), 4> = AoSoa::new();
    for i in 0..10u32 {
        v.push((i, i as f32 * 0.5));
    }
    for i in 0..10 {
        assert_eq!(v.get(i), Some((&(i as u32), &(i as f32 * 0.5))));
    }
    assert_eq!(v.get(10), None);
}

#[test]
fn iter_visits_rows_in_logical_order() {
    let mut v: AoSoa<(u32,), 4> = AoSoa::new();
    for i in 0..10u32 {
        v.push((i,));
    }
    let seen: Vec<u32> = v.iter().map(|(id,)| *id).collect();
    assert_eq!(seen, (0..10).collect::<Vec<_>>());
}

#[test]
fn block_columns_are_dense_lanes() {
    let mut v: AoSoa<(u32, f32), 4> = AoSoa::new();
    for i in 0..10u32 {
        v.push((i, i as f32));
    }
    let mut flat = Vec::new();
    for block in v.blocks() {
        let (ids, _weights) = block.columns();
        flat.extend_from_slice(ids);
    }
    assert_eq!(flat, (0..10).collect::<Vec<_>>());
}

#[test]
fn get_mut_edits_in_place() {
    let mut v: AoSoa<(u32,), 4> = AoSoa::new();
    for i in 0..6u32 {
        v.push((i,));
    }
    if let Some((id,)) = v.get_mut(5) {
        *id = 99;
    }
    assert_eq!(v.get(5), Some((&99,)));
}

#[test]
fn blocks_mut_batch_pass() {
    let mut v: AoSoa<(u32,), 4> = AoSoa::new();
    for i in 0..10u32 {
        v.push((i,));
    }
    for block in v.blocks_mut() {
        let (ids,) = block.columns_mut();
        for id in ids {
            *id *= 2;
        }
    }
    let seen: Vec<u32> = v.iter().map(|(id,)| *id).collect();
    assert_eq!(seen, (0..10).map(|i| i * 2).collect::<Vec<_>>());
}

#[test]
fn swap_remove_last_just_pops() {
    let mut v: AoSoa<(u32,), 4> = AoSoa::new();
    for i in 0..9u32 {
        v.push((i,));
    }
    // Removing the global last (index 8) empties and drops the 3rd block.
    assert_eq!(v.swap_remove(8), Some((8,)));
    assert_eq!(v.len(), 8);
    assert_eq!(v.block_count(), 2);
    assert_eq!(v.get(8), None);
}

#[test]
fn swap_remove_middle_moves_last_into_place() {
    let mut v: AoSoa<(u32,), 4> = AoSoa::new();
    for i in 0..9u32 {
        v.push((i,));
    }
    // Remove index 2; the last row (8) swaps into slot 2.
    assert_eq!(v.swap_remove(2), Some((2,)));
    assert_eq!(v.len(), 8);
    assert_eq!(v.get(2), Some((&8,)));
    // The rest stays reachable; collect and check as a set.
    let mut seen: Vec<u32> = v.iter().map(|(id,)| *id).collect();
    seen.sort_unstable();
    assert_eq!(seen, vec![0, 1, 3, 4, 5, 6, 7, 8]);
}

#[test]
fn swap_remove_out_of_bounds_is_none() {
    let mut v: AoSoa<(u32,), 4> = AoSoa::new();
    v.push((1,));
    assert_eq!(v.swap_remove(1), None);
    assert_eq!(v.len(), 1);
}

#[test]
fn drain_everything_via_swap_remove() {
    let mut v: AoSoa<(u32,), 4> = AoSoa::new();
    for i in 0..10u32 {
        v.push((i,));
    }
    let mut collected = Vec::new();
    while !v.is_empty() {
        collected.push(v.swap_remove(0).unwrap().0);
    }
    assert_eq!(v.len(), 0);
    assert_eq!(v.block_count(), 0);
    collected.sort_unstable();
    assert_eq!(collected, (0..10).collect::<Vec<_>>());
}

#[test]
fn clear_resets_length_and_blocks() {
    let mut v: AoSoa<(u32,), 4> = AoSoa::new();
    for i in 0..7u32 {
        v.push((i,));
    }
    v.clear();
    assert_eq!(v.len(), 0);
    assert_eq!(v.block_count(), 0);
    assert!(v.is_empty());
    // Reusable after clear.
    v.push((42,));
    assert_eq!(v.get(0), Some((&42,)));
}

#[test]
fn with_capacity_presizes_block_directory() {
    let v: AoSoa<(u32,), 4> = AoSoa::with_capacity(10);
    // 10 rows / 4 per block => directory reserved for 3 blocks.
    assert!(v.block_capacity() >= 3);
    assert_eq!(v.len(), 0);
}

#[test]
fn lane_width_one_is_pure_aos() {
    let mut v: AoSoa<(u32,), 1> = AoSoa::new();
    for i in 0..5u32 {
        v.push((i,));
    }
    assert_eq!(v.block_count(), 5);
    assert_eq!(v.get(3), Some((&3,)));
}

#[test]
fn block_bytes_sizes_the_lane() {
    // (u32, u32) => two 4-byte columns, lane_align 4 => 8 bytes/row * 8 rows.
    assert_eq!(AoSoa::<(u32, u32), 8>::block_bytes(4), 8 * 8);
    // Forcing 16-byte lanes rounds each 4-byte column up to 16.
    assert_eq!(AoSoa::<(u32, u32), 8>::block_bytes(16), 32 * 8);
}

#[test]
fn default_matches_new() {
    let v: AoSoa<(u8, u16), 4> = AoSoa::default();
    assert_eq!(v.len(), 0);
    assert_eq!(v.lane_width(), 4);
}
