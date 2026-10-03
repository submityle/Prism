//! Behavioural tests for per-row change detection in queries: the `Added<T>` /
//! `Changed<T>` filters, the `Ref<T>` read-only change-detecting term, and the
//! `Mut<T>` item yielded by `&mut T` (design §10).
//!
//! These one-shot `iter`/`iter_mut` entry points derive their change-detection
//! window from the world's `[last_change_tick, change_tick)` pair, so the tests
//! drive `World::set_last_change_tick` / `World::increment_change_tick` between
//! phases to simulate successive system runs (the executor threads a per-system
//! window in M2 commit C2).

use crate::change::{Mut, Ref};
use crate::component::Component;
use crate::query::{Added, Changed, Or};
use crate::world::World;

#[derive(Debug, PartialEq, Clone, Copy)]
struct Position(i32);
impl Component for Position {}

#[derive(Debug, PartialEq, Clone, Copy)]
struct Velocity(i32);
impl Component for Velocity {}

/// Advance the world as if a system had just finished a run: the next window
/// starts where this one ended, and the frame tick advances by one.
fn advance_frame(w: &mut World) {
    w.set_last_change_tick(w.change_tick());
    w.increment_change_tick();
}

#[test]
fn added_matches_the_spawn_frame_then_stops() {
    let mut w = World::new();
    let e = w.spawn(Position(1));

    // Spawn frame: the entity is freshly added relative to last_run = ZERO.
    let state = w.query_filtered::<crate::entity::Entity, Added<Position>>();
    let seen: alloc::vec::Vec<_> = state.iter(&w).collect();
    assert_eq!(seen, [e], "newly spawned entity matches Added<Position>");

    // Next frame with no further writes: no longer "added".
    advance_frame(&mut w);
    let state = w.query_filtered::<crate::entity::Entity, Added<Position>>();
    assert_eq!(
        state.iter(&w).count(),
        0,
        "an entity added in a previous frame is not Added anymore"
    );
}

#[test]
fn changed_fires_on_mut_deref_then_goes_stale() {
    let mut w = World::new();
    let e = w.spawn(Position(1));

    // Move past the spawn frame so "added" no longer dominates the window.
    advance_frame(&mut w); // last=1, change=2

    // Mutate through the `Mut` item: DerefMut stamps changed = this_run (2).
    let state = w.query::<&mut Position>();
    for mut p in state.iter_mut(&mut w) {
        p.0 += 10;
    }
    assert_eq!(w.get::<Position>(e), Some(&Position(11)));

    // Same window still sees the write as changed.
    let state = w.query_filtered::<crate::entity::Entity, Changed<Position>>();
    assert_eq!(state.iter(&w).count(), 1, "the just-written row is Changed");

    // After the frame advances with no new write, it is stale.
    advance_frame(&mut w);
    let state = w.query_filtered::<crate::entity::Entity, Changed<Position>>();
    assert_eq!(
        state.iter(&w).count(),
        0,
        "a row not written this frame is not Changed"
    );
}

#[test]
fn unwritten_row_never_matches_changed() {
    let mut w = World::new();
    w.spawn(Position(1));
    advance_frame(&mut w); // leave the spawn frame

    // Read-only iteration must not stamp any change tick.
    let state = w.query::<&Position>();
    let _total: i32 = state.iter(&w).map(|p| p.0).sum();

    let state = w.query_filtered::<crate::entity::Entity, Changed<Position>>();
    assert_eq!(
        state.iter(&w).count(),
        0,
        "pure reads do not make a row Changed"
    );
}

#[test]
fn ref_term_reports_added_and_changed() {
    let mut w = World::new();
    w.spawn(Position(7));

    // Spawn frame: Ref reports both added and changed.
    let state = w.query::<Ref<Position>>();
    for r in state.iter(&w) {
        assert!(r.is_added(), "fresh spawn is added");
        assert!(r.is_changed(), "fresh spawn is changed");
        assert_eq!(r.0, 7);
    }

    // Later frame with no writes: neither added nor changed.
    advance_frame(&mut w);
    let state = w.query::<Ref<Position>>();
    for r in state.iter(&w) {
        assert!(!r.is_added(), "no longer added in a later frame");
        assert!(!r.is_changed(), "no longer changed without a write");
    }
}

#[test]
fn ref_is_changed_after_write() {
    let mut w = World::new();
    let e = w.spawn(Position(1));
    advance_frame(&mut w);

    w.get_mut::<Position>(e).unwrap().0 = 42;

    let state = w.query::<Ref<Position>>();
    let mut hits = 0;
    for r in state.iter(&w) {
        assert!(r.is_changed(), "write makes the Ref changed");
        assert!(!r.is_added(), "write is not an add");
        hits += 1;
    }
    assert_eq!(hits, 1);
}

#[test]
fn mut_bypass_change_detection_does_not_stamp() {
    let mut w = World::new();
    let e = w.spawn(Position(1));
    advance_frame(&mut w);

    let state = w.query::<&mut Position>();
    for mut p in state.iter_mut(&mut w) {
        // Silent write: must not bump the changed tick.
        p.bypass_change_detection().0 = 99;
    }
    assert_eq!(w.get::<Position>(e), Some(&Position(99)));

    let state = w.query_filtered::<crate::entity::Entity, Changed<Position>>();
    assert_eq!(
        state.iter(&w).count(),
        0,
        "bypass_change_detection must not mark the row Changed"
    );
}

#[test]
fn mut_set_changed_marks_without_write() {
    let mut w = World::new();
    w.spawn(Position(1));
    advance_frame(&mut w);

    let state = w.query::<&mut Position>();
    for mut p in state.iter_mut(&mut w) {
        p.set_changed();
    }

    let state = w.query_filtered::<crate::entity::Entity, Changed<Position>>();
    assert_eq!(state.iter(&w).count(), 1, "set_changed marks the row Changed");
}

#[test]
fn mut_and_changed_filter_compose_without_conflict() {
    // `Query<&mut A, Changed<A>>` reads and writes the same component. The
    // filter read must be registered as a *filter* read (not a hard read) so
    // the access set does not reject the query as self-aliasing.
    let mut w = World::new();
    w.spawn(Position(1));
    w.spawn(Position(2));
    advance_frame(&mut w);

    // Must not panic when building the state.
    let state = w.query_filtered::<&mut Position, Changed<Position>>();
    // Nothing changed since the spawn frame, so no rows.
    assert_eq!(state.iter_mut(&mut w).count(), 0);
}

#[test]
fn added_or_changed_matches_either() {
    let mut w = World::new();
    let a = w.spawn(Position(1));
    let b = w.spawn(Velocity(1));
    advance_frame(&mut w);

    // Touch only `a`'s Position; `b` has no Position at all.
    w.get_mut::<Position>(a).unwrap().0 = 5;

    // Spawn a brand-new Velocity entity this frame -> Added<Velocity>.
    let c = w.spawn(Velocity(9));

    let state = w.query_filtered::<crate::entity::Entity, Or<(Changed<Position>, Added<Velocity>)>>();
    let mut seen: alloc::vec::Vec<_> = state.iter(&w).collect();
    seen.sort();
    let mut expected = alloc::vec![a, c];
    expected.sort();
    assert_eq!(seen, expected, "Or matches rows satisfying either branch");
    let _ = b;
}

#[test]
fn mut_item_type_is_change_detecting() {
    // Compile-time assurance that `&mut T` yields `Mut<T>`.
    fn _assert(mut w: World) {
        let state = w.query::<&mut Position>();
        for p in state.iter_mut(&mut w) {
            let _: Mut<'_, Position> = p;
        }
    }
}
