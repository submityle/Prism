//! Behavioural regression tests for **SharedComponent**-backed components
//! across the full world + query surface (design §6 存储模型 四态
//! "SharedComponent"，§15 GPU 批次键).
//!
//! A shared component's *value* is de-duplicated into one interned copy and
//! used as a per-archetype *batch key*: every entity carrying the same value is
//! grouped into the same archetype variant, and distinct values fragment the
//! archetype graph. The value is immutable through queries (it is a key, not
//! per-entity data), so only `&T` / `Option<&T>` reads and `With`/`Without`
//! membership are supported; `&mut T`, `Ref`, `Added`, `Changed` are rejected
//! at query construction.
//!
//! These tests pin, end-to-end: value de-duplication and per-value archetype
//! split, correct per-archetype reads (pure and mixed with table/sparse),
//! transparent `With`/`Without` membership, structural moves when the bound
//! value changes (and the no-move case when it does not), removal semantics
//! (dropping a shared binding vs. preserving it when a *table* component is
//! removed), the single-value getter contract (`get` yields the value while the
//! change-detecting getters honestly report `None`), snapshot round-tripping of
//! shared values (including a shared-only entity), and the construction-time
//! rejection of mutable / change-detecting access.

use alloc::vec::Vec;

use crate::archetype::ArchetypeId;
use crate::component::{Component, ComponentId, Components, StorageType};
use crate::entity::Entity;
use crate::query::{Added, Changed, With, Without};
use crate::storage::SharedValueId;
use crate::world::World;

/// Table-backed (default) anchor component.
#[derive(Debug, PartialEq, Eq, Hash, Clone, Copy)]
struct Position(i32);
impl Component for Position {}

/// A second table component, used to exercise "remove a *table* component but
/// keep the shared binding".
#[derive(Debug, PartialEq, Clone, Copy)]
struct Velocity(i32);
impl Component for Velocity {}

/// Sparse-backed component, for the mixed table+sparse+shared spawn.
#[derive(Debug, PartialEq, Clone, Copy)]
struct Charge(i32);
impl Component for Charge {
    const STORAGE: StorageType = StorageType::SparseSet;
}

/// Shared-backed batch key — the (`mesh`, `material`) analogue from design §6.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct BatchKey(u32);
impl Component for BatchKey {
    const STORAGE: StorageType = StorageType::Shared;
    fn install_storage_glue(components: &mut Components, id: ComponentId) {
        components.set_shared_box(id, crate::component::shared_box_of::<Self>());
    }
}

/// A second shared component so a world can bind two independent batch keys.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct Layer(u8);
impl Component for Layer {
    const STORAGE: StorageType = StorageType::Shared;
    fn install_storage_glue(components: &mut Components, id: ComponentId) {
        components.set_shared_box(id, crate::component::shared_box_of::<Self>());
    }
}

/// The archetype an entity currently resides in (test helper; child modules may
/// read the world's private `entities`/`archetypes`).
fn archetype_of(w: &World, e: Entity) -> ArchetypeId {
    w.entities.location(e).expect("entity is live").archetype_id
}

/// Advance the world as if a system had just finished a run.
fn advance_frame(w: &mut World) {
    w.set_last_change_tick(w.change_tick());
    w.increment_change_tick();
}

/// The [`ComponentId`] of a (already-registered) component `T`. Every lifecycle
/// test spawns `T` first, so its id is always registered by the time this runs.
fn cid<T: Component>(w: &World) -> ComponentId {
    w.components
        .id_of::<T>()
        .expect("component must be registered before querying its id")
}

/// The [`SharedValueId`] that `e`'s current archetype binds for component `c`.
fn shared_sid(w: &World, e: Entity, c: ComponentId) -> SharedValueId {
    w.archetypes
        .get(archetype_of(w, e))
        .expect("live archetype")
        .shared_binding(c)
        .expect("entity binds the shared component")
}

/// Live reference count of `(c, sid)` in the world's shared-value pool.
fn refcount(w: &World, c: ComponentId, sid: SharedValueId) -> u32 {
    w.shared_components.refcount(c, sid)
}

/// Number of distinct values currently interned for shared component `c`.
fn pool_len(w: &World, c: ComponentId) -> usize {
    w.shared_components.pool(c).map_or(0, |p| p.len())
}

#[test]
fn same_value_shares_archetype_distinct_value_splits() {
    let mut w = World::new();
    // Same table set {Position} + same shared value -> SAME archetype.
    let a = w.spawn((Position(1), BatchKey(10)));
    let b = w.spawn((Position(2), BatchKey(10)));
    // Same table set, DIFFERENT shared value -> DIFFERENT archetype.
    let c = w.spawn((Position(3), BatchKey(20)));

    assert_eq!(
        archetype_of(&w, a),
        archetype_of(&w, b),
        "equal shared values dedup into one archetype"
    );
    assert_ne!(
        archetype_of(&w, a),
        archetype_of(&w, c),
        "distinct shared values split the archetype graph (batch key)"
    );
}

#[test]
fn query_reads_correct_per_archetype_shared_value() {
    let mut w = World::new();
    let a = w.spawn((Position(1), BatchKey(10)));
    let b = w.spawn((Position(2), BatchKey(10)));
    let c = w.spawn((Position(3), BatchKey(20)));

    let state = w.query::<(Entity, &BatchKey)>();
    let mut seen: Vec<(Entity, u32)> = state.iter(&w).map(|(e, k)| (e, k.0)).collect();
    seen.sort_by_key(|(e, _)| e.index());
    let mut want = alloc::vec![(a, 10u32), (b, 10), (c, 20)];
    want.sort_by_key(|(e, _)| e.index());
    assert_eq!(
        seen, want,
        "each entity reads the value bound to its archetype"
    );
}

#[test]
fn query_mixes_table_and_shared() {
    let mut w = World::new();
    let a = w.spawn((Position(1), BatchKey(10)));
    let c = w.spawn((Position(3), BatchKey(20)));

    let state = w.query::<(Entity, &Position, &BatchKey)>();
    let mut seen: Vec<(Entity, i32, u32)> = state.iter(&w).map(|(e, p, k)| (e, p.0, k.0)).collect();
    seen.sort_by_key(|(e, ..)| e.index());
    let mut want = alloc::vec![(a, 1i32, 10u32), (c, 3, 20)];
    want.sort_by_key(|(e, ..)| e.index());
    assert_eq!(seen, want);
}

#[test]
fn with_without_shared_are_transparent() {
    let mut w = World::new();
    let keyed = w.spawn((Position(1), BatchKey(10)));
    let bare = w.spawn(Position(2));

    let with = w.query_filtered::<Entity, With<BatchKey>>();
    let with_seen: Vec<Entity> = with.iter(&w).collect();
    assert_eq!(
        with_seen,
        [keyed],
        "With<Shared> matches only bound entities"
    );

    let without = w.query_filtered::<Entity, Without<BatchKey>>();
    let without_seen: Vec<Entity> = without.iter(&w).collect();
    assert_eq!(
        without_seen,
        [bare],
        "Without<Shared> matches only unbound entities"
    );
}

#[test]
fn insert_changing_shared_value_moves_archetype_same_value_does_not() {
    let mut w = World::new();
    let e = w.spawn((Position(1), BatchKey(10)));
    let before = archetype_of(&w, e);

    // Re-insert the SAME shared value: no structural move.
    assert!(w.insert(e, BatchKey(10)));
    assert_eq!(
        archetype_of(&w, e),
        before,
        "re-binding the same shared value is not a move"
    );

    // Insert a DIFFERENT shared value: structural move to the new variant.
    assert!(w.insert(e, BatchKey(20)));
    assert_ne!(
        archetype_of(&w, e),
        before,
        "changing the shared value relocates the entity"
    );
    assert_eq!(w.get::<BatchKey>(e), Some(&BatchKey(20)));
    // Table data survives the move.
    assert_eq!(w.get::<Position>(e), Some(&Position(1)));
}

#[test]
fn remove_shared_drops_binding() {
    let mut w = World::new();
    let e = w.spawn((Position(1), BatchKey(10)));
    assert!(w.has::<BatchKey>(e));

    assert!(
        w.remove::<BatchKey>(e),
        "removing a bound shared comp reports true"
    );
    assert!(!w.has::<BatchKey>(e), "binding is gone after remove");
    assert_eq!(w.get::<BatchKey>(e), None);
    // The entity still holds its table component and no longer matches a
    // shared query.
    assert_eq!(w.get::<Position>(e), Some(&Position(1)));
    let state = w.query::<(Entity, &BatchKey)>();
    assert_eq!(state.iter(&w).count(), 0);
    // Removing again is a no-op.
    assert!(!w.remove::<BatchKey>(e));
}

#[test]
fn remove_table_component_preserves_shared_binding() {
    let mut w = World::new();
    let e = w.spawn((Position(1), Velocity(2), BatchKey(10)));
    assert!(w.has::<BatchKey>(e));

    // Remove a *table* component: the shared binding must be preserved and the
    // entity re-keyed into the archetype that still binds BatchKey(10).
    assert!(w.remove::<Velocity>(e));
    assert!(!w.has::<Velocity>(e));
    assert!(
        w.has::<BatchKey>(e),
        "shared binding survives a table removal"
    );
    assert_eq!(w.get::<BatchKey>(e), Some(&BatchKey(10)));
    // The shared query still reads it.
    let state = w.query::<(Entity, &BatchKey)>();
    let seen: Vec<(Entity, u32)> = state.iter(&w).map(|(e, k)| (e, k.0)).collect();
    assert_eq!(seen, [(e, 10)]);
}

#[test]
fn getters_contract_on_shared() {
    let mut w = World::new();
    let e = w.spawn((Position(1), BatchKey(42)));

    // `get` returns the archetype-wide value...
    assert_eq!(w.get::<BatchKey>(e), Some(&BatchKey(42)));
    // ...and `has` is true...
    assert!(w.has::<BatchKey>(e));
    // ...but the change-detecting getters honestly report None (a shared value
    // has no per-entity ticks).
    assert!(
        w.get_ref::<BatchKey>(e).is_none(),
        "get_ref is None for shared"
    );
    assert!(
        w.get_ticks::<BatchKey>(e).is_none(),
        "get_ticks is None for shared"
    );
    assert!(
        w.get_mut::<BatchKey>(e).is_none(),
        "get_mut is None for shared"
    );
}

#[test]
fn spawn_mixes_table_sparse_and_shared() {
    let mut w = World::new();
    let e = w.spawn((Position(1), Velocity(2), Charge(3), BatchKey(7)));

    assert_eq!(w.get::<Position>(e), Some(&Position(1)));
    assert_eq!(w.get::<Velocity>(e), Some(&Velocity(2)));
    assert_eq!(w.get::<Charge>(e), Some(&Charge(3)));
    assert_eq!(w.get::<BatchKey>(e), Some(&BatchKey(7)));
    assert!(w.has::<Charge>(e) && w.has::<BatchKey>(e));

    // A query over the shared key plus a table and sparse term resolves all.
    let state = w.query::<(Entity, &Position, &BatchKey)>();
    let seen: Vec<(Entity, i32, u32)> = state.iter(&w).map(|(e, p, k)| (e, p.0, k.0)).collect();
    assert_eq!(seen, [(e, 1, 7)]);
}

#[test]
fn two_independent_shared_keys_coexist() {
    let mut w = World::new();
    // Same Position + same BatchKey but different Layer must split; same both
    // must dedup.
    let a = w.spawn((Position(1), BatchKey(10), Layer(0)));
    let b = w.spawn((Position(2), BatchKey(10), Layer(0)));
    let c = w.spawn((Position(3), BatchKey(10), Layer(1)));

    assert_eq!(archetype_of(&w, a), archetype_of(&w, b));
    assert_ne!(
        archetype_of(&w, a),
        archetype_of(&w, c),
        "a second shared key fragments independently"
    );
    assert_eq!(w.get::<Layer>(a), Some(&Layer(0)));
    assert_eq!(w.get::<Layer>(c), Some(&Layer(1)));
    assert_eq!(w.get::<BatchKey>(c), Some(&BatchKey(10)));
}

#[test]
fn snapshot_roundtrips_shared_values() {
    let mut w = World::new();
    w.register_snapshot_component_hashable::<Position>();
    w.register_snapshot_component_hashable::<BatchKey>();

    let _a = w.spawn((Position(1), BatchKey(10)));
    let _b = w.spawn((Position(2), BatchKey(10)));
    let _c = w.spawn((Position(3), BatchKey(20)));
    // A shared-only entity: no table columns, just a batch-key binding.
    let d = w.spawn(BatchKey(99));

    let snap = w.snapshot();

    // Diverge, then restore.
    assert!(w.insert(d, BatchKey(10)));
    let _late = w.spawn((Position(7), BatchKey(20)));
    w.restore(&snap);

    let after = w.snapshot();
    assert!(
        snap.structurally_eq(&after),
        "restore reproduces shared bindings and dedup exactly"
    );
    assert_eq!(snap.state_hash(), after.state_hash());
    // The shared-only entity is readable again at its original value.
    assert_eq!(w.get::<BatchKey>(d), Some(&BatchKey(99)));
}

#[test]
#[should_panic(expected = "shared component (design §6) is immutable and archetype-wide")]
fn mut_query_on_shared_is_rejected_at_construction() {
    let mut w = World::new();
    let _ = w.spawn((Position(1), BatchKey(10)));
    // Constructing a `&mut Shared` query must panic at init_state.
    let _state = w.query::<&mut BatchKey>();
}

#[test]
#[should_panic(expected = "change-detection filters do not support shared storage")]
fn changed_filter_on_shared_is_rejected_at_construction() {
    let mut w = World::new();
    let _ = w.spawn((Position(1), BatchKey(10)));
    let _state = w.query_filtered::<Entity, Changed<BatchKey>>();
}

#[test]
#[should_panic(expected = "change-detection filters do not support shared storage")]
fn added_filter_on_shared_is_rejected_at_construction() {
    let mut w = World::new();
    let _ = w.spawn((Position(1), BatchKey(10)));
    let _state = w.query_filtered::<Entity, Added<BatchKey>>();
}

#[test]
fn structural_move_does_not_leak_or_duplicate_entities() {
    let mut w = World::new();
    let a = w.spawn((Position(1), BatchKey(10)));
    let b = w.spawn((Position(2), BatchKey(10)));
    advance_frame(&mut w);

    // Move `a` to a new batch; `b` must stay put and remain readable.
    assert!(w.insert(a, BatchKey(20)));
    assert_eq!(w.get::<BatchKey>(a), Some(&BatchKey(20)));
    assert_eq!(w.get::<BatchKey>(b), Some(&BatchKey(10)));
    assert_eq!(w.get::<Position>(b), Some(&Position(2)));

    let state = w.query::<(Entity, &BatchKey)>();
    assert_eq!(state.iter(&w).count(), 2, "no entity lost or duplicated");
}

// --- SharedComponent refcount lifecycle (design §6 生命周期) --------------------
//
// The tests above pin *behaviour*; the ones below pin the *reference-counted
// lifecycle* of interned shared values across despawn / value-change / remove /
// snapshot-restore, and the archetype-eviction that keeps a recycled
// `SharedValueId` from aliasing a stale empty archetype. Each proves one leak
// class that the pre-fix world layer silently carried (it only ever interned,
// never released).

#[test]
fn despawn_releases_shared_reference_and_recycles_slot() {
    let mut w = World::new();
    let a = w.spawn((Position(1), BatchKey(10)));
    let b = w.spawn((Position(2), BatchKey(10)));
    let c = cid::<BatchKey>(&w);
    let sid10 = shared_sid(&w, a, c);
    assert_eq!(refcount(&w, c, sid10), 2, "two live holders of value 10");

    assert!(w.despawn(a));
    assert_eq!(
        refcount(&w, c, sid10),
        1,
        "despawning one holder releases exactly one reference"
    );

    assert!(w.despawn(b));
    assert_eq!(
        refcount(&w, c, sid10),
        0,
        "despawning the last holder frees the value"
    );
    assert_eq!(
        pool_len(&w, c),
        0,
        "the freed slot is recycled; no value stays interned"
    );

    // Spawning a *different* value must recycle the freed slot index AND read
    // the new value — proving the stale empty archetype keyed on the old
    // `(cid, sid)` was evicted (no aliasing of the old value's `Arc`).
    let d = w.spawn((Position(3), BatchKey(20)));
    let sid20 = shared_sid(&w, d, c);
    assert_eq!(
        sid20.index(),
        sid10.index(),
        "the freed dense slot index is recycled for the next value"
    );
    assert_eq!(refcount(&w, c, sid20), 1);
    assert_eq!(
        w.get::<BatchKey>(d),
        Some(&BatchKey(20)),
        "the recycled id reads the NEW value, never the evicted stale 10"
    );
}

#[test]
fn changing_shared_value_rebalances_refcounts() {
    let mut w = World::new();
    let a = w.spawn((Position(1), BatchKey(10)));
    let b = w.spawn((Position(2), BatchKey(10)));
    let c = cid::<BatchKey>(&w);
    let sid10 = shared_sid(&w, a, c);
    assert_eq!(refcount(&w, c, sid10), 2);

    // Move `a` to value 20: value 10 loses a's ref (+ gains it on 20). Before
    // the fix the old binding was never released — 10 would stay at 2.
    assert!(w.insert(a, BatchKey(20)));
    let sid20 = shared_sid(&w, a, c);
    assert_ne!(sid20, sid10, "a new value gets a distinct id");
    assert_eq!(
        refcount(&w, c, sid10),
        1,
        "the old value drops a's reference"
    );
    assert_eq!(
        refcount(&w, c, sid20),
        1,
        "the new value gains a's reference"
    );
    assert_eq!(
        w.get::<BatchKey>(b),
        Some(&BatchKey(10)),
        "the untouched holder still reads the old value"
    );

    // Move `b` to 20 as well: value 10's last holder leaves, so it frees.
    assert!(w.insert(b, BatchKey(20)));
    assert_eq!(
        refcount(&w, c, sid10),
        0,
        "value 10 frees when its last holder leaves"
    );
    assert_eq!(refcount(&w, c, sid20), 2, "both holders now hold value 20");
    assert_eq!(
        archetype_of(&w, a),
        archetype_of(&w, b),
        "equal shared value re-converges the two into one archetype"
    );
}

#[test]
fn same_value_insert_overwrite_does_not_leak_refcount() {
    let mut w = World::new();
    let a = w.spawn((Position(1), BatchKey(10)));
    let c = cid::<BatchKey>(&w);
    let sid = shared_sid(&w, a, c);
    let arch = archetype_of(&w, a);
    assert_eq!(refcount(&w, c, sid), 1);

    // Overwrite with the identical value: no archetype move, and the re-intern
    // (+1) must be balanced by releasing the prior binding (-1). Before the fix
    // this no-op overwrite leaked +1 on every call.
    assert!(w.insert(a, BatchKey(10)));
    assert_eq!(archetype_of(&w, a), arch, "same value is not a move");
    assert_eq!(
        refcount(&w, c, sid),
        1,
        "re-intern then release of an unchanged value nets zero"
    );
    assert_eq!(pool_len(&w, c), 1, "exactly one value stays interned");

    // Repeated no-op overwrites must stay flat, not accumulate.
    assert!(w.insert(a, BatchKey(10)));
    assert!(w.insert(a, BatchKey(10)));
    assert_eq!(
        refcount(&w, c, sid),
        1,
        "repeated no-op overwrites do not leak"
    );
}

#[test]
fn removing_shared_component_releases_reference() {
    let mut w = World::new();
    let a = w.spawn((Position(1), BatchKey(10)));
    let b = w.spawn((Position(2), BatchKey(10)));
    let c = cid::<BatchKey>(&w);
    let sid = shared_sid(&w, a, c);
    assert_eq!(refcount(&w, c, sid), 2);

    // Removing the shared component from one entity releases its reference...
    assert!(w.remove::<BatchKey>(a));
    assert_eq!(
        refcount(&w, c, sid),
        1,
        "remove releases the removed entity's shared reference"
    );
    assert_eq!(w.get::<BatchKey>(b), Some(&BatchKey(10)), "b still bound");

    // ...and removing the last holder frees the value entirely.
    assert!(w.remove::<BatchKey>(b));
    assert_eq!(refcount(&w, c, sid), 0, "the last removal frees the value");
    assert_eq!(
        pool_len(&w, c),
        0,
        "no value stays interned after full removal"
    );
}

#[test]
fn snapshot_restore_does_not_compound_refcounts() {
    let mut w = World::new();
    w.register_snapshot_component_hashable::<Position>();
    w.register_snapshot_component_hashable::<BatchKey>();

    let a = w.spawn((Position(1), BatchKey(10)));
    let _b = w.spawn((Position(2), BatchKey(10)));
    let _c = w.spawn((Position(3), BatchKey(20)));
    let c = cid::<BatchKey>(&w);
    let sid10 = shared_sid(&w, a, c);
    assert_eq!(
        refcount(&w, c, sid10),
        2,
        "baseline: two holders of value 10"
    );

    let snap = w.snapshot();

    // Diverge: add holders / change values so the live pool differs from snap.
    let late = w.spawn((Position(7), BatchKey(10)));
    assert_eq!(refcount(&w, c, sid10), 3);
    assert!(w.despawn(late));

    // Restore repeatedly: each restore must *rebuild* the pool, not add onto a
    // stale one. Before the fix `restore` left `shared_components` untouched, so
    // refcounts compounded with every restore.
    for _ in 0..3 {
        w.restore(&snap);
        let sid10_after = shared_sid(&w, a, c);
        assert_eq!(
            refcount(&w, c, sid10_after),
            2,
            "restore rebuilds refcounts to match live holders, never compounds"
        );
        // Round-trips byte-for-byte.
        assert!(snap.structurally_eq(&w.snapshot()));
    }
}
