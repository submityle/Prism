//! Behavioural tests for `QueryState::simd_iter` / `simd_iter_mut` (design §7).

use alloc::vec::Vec;

use crate::component::{Component, StorageType};
use crate::query::{Changed, With};
use crate::world::World;

/// Table-backed value used as the slice payload.
#[derive(Debug, PartialEq, Clone, Copy)]
struct Value(f32);
impl Component for Value {}

/// Table-backed marker, for archetype filters.
#[derive(Debug, PartialEq, Clone, Copy)]
struct Tag;
impl Component for Tag {}

/// Sparse-backed component, to exercise the sliceability rejection.
#[derive(Debug, PartialEq, Clone, Copy)]
struct Sparse(f32);
impl Component for Sparse {
    const STORAGE: StorageType = StorageType::SparseSet;
}

/// A component whose column is exactly `f32`, so a yielded slice *is* the input
/// type of the `simd` bulk kernels (there is no repacking step in between).
#[cfg(feature = "simd")]
impl Component for f32 {}

#[test]
fn slices_are_chunk_aligned_and_cover_every_row_once() {
    let mut world = World::new();
    for i in 0..10 {
        world.spawn(Value(i as f32));
    }

    let state = world.query::<&Value>();
    let slices: Vec<Vec<f32>> = state
        .simd_iter::<4>(&world)
        .map(|slice| slice.iter().map(|v| v.0).collect())
        .collect();

    // 10 rows at 4 per slice: 4, 4, 2 — the tail chunk is shorter, never dropped.
    assert_eq!(
        slices,
        [
            alloc::vec![0.0, 1.0, 2.0, 3.0],
            alloc::vec![4.0, 5.0, 6.0, 7.0],
            alloc::vec![8.0, 9.0],
        ]
    );
}

#[test]
fn slice_size_of_one_is_row_by_row() {
    let mut world = World::new();
    for i in 0..3 {
        world.spawn(Value(i as f32));
    }
    let state = world.query::<&Value>();
    let lengths: Vec<usize> = state
        .simd_iter::<1>(&world)
        .map(|slice| slice.len())
        .collect();
    assert_eq!(lengths, [1, 1, 1]);
}

#[test]
fn mutable_slices_write_through_and_stamp_change_detection() {
    let mut world = World::new();
    for i in 0..7 {
        world.spawn(Value(i as f32));
    }
    // Advance a frame so the spawn tick is in the past.
    let now = world.change_tick();
    world.set_last_change_tick(now);
    world.increment_change_tick();

    let state = world.query::<&mut Value>();
    for slice in state.simd_iter_mut::<3>(&mut world) {
        for value in slice {
            value.0 += 10.0;
        }
    }

    // Values were written, and the whole handed-out range is marked changed
    // (a bare slice loses per-row write information).
    let read = world.query::<&Value>();
    let all: Vec<f32> = read.iter(&world).map(|v| v.0).collect();
    assert_eq!(all, [10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0]);

    let changed = world.query_filtered::<&Value, Changed<Value>>();
    // `Changed` yields the value itself; count the marked rows.
    assert!(
        changed.iter(&world).count() >= 1,
        "writes must be visible to change detection"
    );
}

#[test]
fn archetype_filters_are_supported() {
    let mut world = World::new();
    world.spawn((Value(1.0), Tag));
    world.spawn(Value(2.0));
    world.spawn((Value(3.0), Tag));

    let state = world.query_filtered::<&Value, With<Tag>>();
    let mut seen: Vec<f32> = Vec::new();
    for slice in state.simd_iter::<8>(&world) {
        seen.extend(slice.iter().map(|v| v.0));
    }
    assert_eq!(seen, [1.0, 3.0]);
}

#[test]
#[should_panic(expected = "requires table-backed components")]
fn sparse_data_is_rejected_up_front() {
    let mut world = World::new();
    world.spawn(Sparse(1.0));
    let state = world.query::<&Sparse>();
    let _ = state.simd_iter::<4>(&world).count();
}

#[test]
#[should_panic(expected = "must be archetype-uniform")]
fn sparse_filter_is_rejected_up_front() {
    let mut world = World::new();
    world.spawn((Value(1.0), Sparse(1.0)));
    // `With<Sparse>` is a per-entity test, so it cannot gate a column slice.
    let state = world.query_filtered::<&Value, With<Sparse>>();
    let _ = state.simd_iter::<4>(&world).count();
}

#[cfg(feature = "simd")]
#[test]
fn slices_feed_the_simd_bulk_kernels() {
    let mut world = World::new();
    for i in 0..9 {
        world.spawn(i as f32);
    }

    let state = world.query::<&mut f32>();
    for slice in state.simd_iter_mut::<4>(&mut world) {
        crate::query::scale_assign(slice, 2.0);
    }

    let read = world.query::<&f32>();
    let all: Vec<f32> = read.iter(&world).copied().collect();
    assert_eq!(all, [0.0, 2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0]);
}
