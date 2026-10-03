//! Behavioural tests for the dirty-chunk accessor
//! ([`QueryState::dirty_chunks`](crate::query::QueryState::dirty_chunks)), the
//! coarse, chunk-granular half of double-layer change detection (design §7/§10).
//!
//! The accessor must be a correct *superset* of the per-row
//! [`Changed<T>`](crate::query::Changed) scan: every row a `Changed<T>` filter
//! would yield lies inside some reported window, and no window is reported for a
//! chunk whose rows are all older than `last_run`.

use alloc::vec::Vec;

use crate::change::Tick;
use crate::component::Component;
use crate::entity::Entity;
use crate::query::{Changed, DirtyChunk};
use crate::storage::rows_per_chunk;
use crate::world::World;

/// A component large enough that only a handful of rows fit per logical chunk,
/// so a modest spawn count spans several chunks (keeps the test fast + legible).
/// 4 KiB per row => `rows_per_chunk == 16 KiB / 4 KiB == 4`.
#[derive(Clone, Copy)]
struct Big {
    tag: u64,
    _pad: [u64; 511],
}
impl Big {
    fn new(tag: u64) -> Self {
        Self {
            tag,
            _pad: [0; 511],
        }
    }
}
impl Component for Big {}

/// Advance the world as if a system had just finished a run: the next window
/// starts where this one ended, and the frame tick advances by one.
fn advance_frame(w: &mut World) {
    w.set_last_change_tick(w.change_tick());
    w.increment_change_tick();
}

/// Whether `row` of `archetype`-equivalent window set is covered by any reported
/// dirty chunk (archetype equality is checked by the caller when needed).
fn row_is_covered(dirty: &[DirtyChunk], row: usize) -> bool {
    dirty.iter().any(|d| row >= d.start && row < d.end)
}

#[test]
fn rpc_is_four_for_a_four_kib_row() {
    // Sanity-check the fixture math the rest of the tests rely on.
    assert_eq!(size_of::<Big>(), 4096);
    assert_eq!(rows_per_chunk(size_of::<Big>()), 4);
}

#[test]
fn only_touched_chunks_are_reported() {
    let mut w = World::new();
    // 10 rows => chunks [0,4), [4,8), [8,10) at rpc = 4.
    let entities: Vec<Entity> = (0..10).map(|i| w.spawn(Big::new(i))).collect();

    // Move past the spawn frame so the spawn-time writes fall out of the window.
    advance_frame(&mut w);

    // Touch one row in chunk 0 and one row in chunk 2; leave chunk 1 untouched.
    let last_run = w.last_change_tick();
    let this_run = w.change_tick();
    w.get_mut::<Big>(entities[1]).unwrap().tag += 100; // chunk 0
    w.get_mut::<Big>(entities[9]).unwrap().tag += 100; // chunk 2

    let state = w.query::<&mut Big>();
    let dirty = state.dirty_chunks(&w, last_run, this_run);

    // Exactly the two touched windows, nothing from chunk 1.
    assert_eq!(dirty.len(), 2, "only the two touched chunks are reported");
    assert!(
        dirty.iter().any(|d| d.start == 0 && d.end == 4),
        "chunk 0 window [0,4) reported: {dirty:?}"
    );
    assert!(
        dirty.iter().any(|d| d.start == 8 && d.end == 10),
        "last partial chunk window [8,10) reported: {dirty:?}"
    );
    assert!(
        !dirty.iter().any(|d| d.start == 4),
        "untouched middle chunk [4,8) is NOT reported: {dirty:?}"
    );
}

#[test]
fn dirty_chunks_superset_of_per_row_changed() {
    let mut w = World::new();
    let entities: Vec<Entity> = (0..10).map(|i| w.spawn(Big::new(i))).collect();
    advance_frame(&mut w);

    let last_run = w.last_change_tick();
    let this_run = w.change_tick();
    // Touch a scattered subset spanning two chunks.
    for &idx in &[0usize, 2, 9] {
        w.get_mut::<Big>(entities[idx]).unwrap().tag += 1;
    }

    // Rows yielded by the fine, per-row Changed filter.
    let changed_state = w.query_filtered::<Entity, Changed<Big>>();
    let changed: Vec<Entity> = changed_state.iter(&w).collect();
    assert_eq!(changed.len(), 3, "three rows were written this window");

    let state = w.query::<&Big>();
    let dirty = state.dirty_chunks(&w, last_run, this_run);

    // Superset: every per-row Changed entity's row lies inside a reported window
    // of its own archetype.
    for e in &changed {
        let loc = w.entities().location(*e).unwrap();
        let covered = dirty
            .iter()
            .any(|d| d.archetype == loc.archetype_id && row_is_covered(&[*d], loc.row as usize));
        assert!(covered, "changed entity {e:?} at row {} not covered", loc.row);
    }
}

#[test]
fn spawn_frame_marks_every_chunk_dirty() {
    let mut w = World::new();
    for i in 0..10 {
        w.spawn(Big::new(i));
    }
    // Observe from the very first frame: last_run = ZERO, this_run = spawn tick.
    let last_run = Tick::ZERO;
    let this_run = w.change_tick();

    let state = w.query::<&Big>();
    let dirty = state.dirty_chunks(&w, last_run, this_run);

    // All three chunks (two full + one partial) are freshly written.
    assert_eq!(dirty.len(), 3, "every chunk is dirty on the spawn frame: {dirty:?}");
    let total: usize = dirty.iter().map(DirtyChunk::len).sum();
    assert_eq!(total, 10, "windows cover all 10 rows exactly");
    assert!(dirty.iter().all(|d| !d.is_empty()));
}

#[test]
fn no_writes_reports_no_dirty_chunks() {
    let mut w = World::new();
    for i in 0..10 {
        w.spawn(Big::new(i));
    }
    // Advance so the spawn writes age out of the window, then observe a frame
    // with no writes at all.
    advance_frame(&mut w);
    let last_run = w.last_change_tick();
    let this_run = w.change_tick();

    let state = w.query::<&Big>();
    let dirty = state.dirty_chunks(&w, last_run, this_run);
    assert!(dirty.is_empty(), "a quiescent frame reports no dirty chunks: {dirty:?}");
}

#[test]
fn write_set_component_marks_chunk_dirty_for_read_query() {
    // A query reading A over an archetype {A, B}: writing B must still flag the
    // chunk, because dirty_chunks unions the query's read AND write sets, and B
    // shares the chunk window with A.
    #[derive(Clone, Copy)]
    struct Marker(u32);
    impl Component for Marker {}

    let mut w = World::new();
    let entities: Vec<Entity> = (0..6).map(|i| w.spawn((Big::new(i), Marker(0)))).collect();
    advance_frame(&mut w);

    let last_run = w.last_change_tick();
    let this_run = w.change_tick();
    // Write only Marker on one row of chunk 0.
    w.get_mut::<Marker>(entities[0]).unwrap().0 = 7;

    // Query reads Big only; its access set includes Big (read). Marker's write
    // bumps Marker's chunk version, not Big's, so a Big-only read query should
    // NOT see it — confirming per-column granularity.
    let big_only = w.query::<&Big>();
    let dirty_big = big_only.dirty_chunks(&w, last_run, this_run);
    assert!(
        dirty_big.is_empty(),
        "writing Marker does not dirty a Big-only query: {dirty_big:?}"
    );

    // A query that reads Marker does see it.
    let marker_only = w.query::<&Marker>();
    let dirty_marker = marker_only.dirty_chunks(&w, last_run, this_run);
    assert_eq!(dirty_marker.len(), 1, "Marker write flags its chunk: {dirty_marker:?}");
    // {Big, Marker} share one rpc from the summed row bytes
    // (4096 + 4 = 4100 => 16 KiB / 4100 = 3 rows per chunk), so chunk 0 is [0,3).
    assert_eq!((dirty_marker[0].start, dirty_marker[0].end), (0, 3));
}
