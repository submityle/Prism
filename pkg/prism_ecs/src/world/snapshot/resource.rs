//! Opt-in resource capture for world snapshots (design §14 / §16.5).
//!
//! A [`WorldSnapshot`](super::WorldSnapshot) captures entity + component state;
//! this module adds the parallel, **strictly opt-in** capture of
//! [`Resources`](crate::resource::Resources). Where an entity's resident
//! component *blocks* a capture if it lacks clone glue, a resource is only ever
//! captured when it was explicitly registered via
//! [`register_snapshot_resource`](crate::world::World::register_snapshot_resource).
//!
//! The asymmetry is deliberate and load-bearing for rollback fidelity. Most
//! resources are infrastructure singletons — a task pool, a GPU device handle,
//! an asset server, a network socket — that are not meaningfully [`Clone`] and
//! must never be rolled back. Gameplay resources, by contrast — the score, a
//! round timer, the deterministic RNG cursor — *must* travel with a snapshot or
//! a rollback silently diverges. Opt-in registration draws exactly that line:
//! unregistered resources are left untouched by `snapshot`/`restore`, while
//! registered ones round-trip value-for-value.
//!
//! # Honesty boundary
//! A resource value is type-erased behind `Box<dyn Any>` and carries no raw
//! column bytes and no `PartialEq`, so — unlike components —
//! [`structurally_eq`](super::WorldSnapshot::structurally_eq) cannot byte-compare
//! two captured resources. It therefore compares resource *presence* and type
//! for every registered resource, and additionally compares a deterministic
//! value hash for resources registered as *hashable*
//! ([`register_snapshot_resource_hashable`](crate::world::World::register_snapshot_resource_hashable)).
//! Clone-only resources thus contribute presence/type to equality and the state
//! hash, but not their value bytes.

use alloc::boxed::Box;
use alloc::vec::Vec;
use core::any::{Any, TypeId};

use crate::resource::{ResourceCloneFn, ResourceId};
use crate::world::World;

use super::FnvHasher;

/// One captured resource: its id + dynamic type, an owned clone of its value,
/// an optional deterministic value hash (present iff the resource was
/// registered hashable), and the clone glue needed to re-materialise it on
/// `restore` or when a [`SnapshotDelta`](super::SnapshotDelta) copies it.
pub(super) struct SnapshotResource {
    id: ResourceId,
    type_id: TypeId,
    value: Box<dyn Any + Send + Sync>,
    value_hash: Option<u64>,
    clone: ResourceCloneFn,
}

impl SnapshotResource {
    /// The captured resource's id.
    #[inline]
    pub(super) fn id(&self) -> ResourceId {
        self.id
    }

    /// The captured resource's dynamic type.
    #[inline]
    pub(super) fn type_id(&self) -> TypeId {
        self.type_id
    }

    /// The captured deterministic value hash, if the resource is hashable.
    #[inline]
    pub(super) fn value_hash(&self) -> Option<u64> {
        self.value_hash
    }

    /// Clone this entry into an independent copy (fresh owned value via the
    /// stored clone glue). Used by [`SnapshotDelta`](super::SnapshotDelta),
    /// whose `apply` must hand back an owned snapshot.
    pub(super) fn clone_entry(&self) -> SnapshotResource {
        SnapshotResource {
            id: self.id,
            type_id: self.type_id,
            // SAFETY-free: `clone` is the glue registered for this resource's
            // concrete type and `value` holds exactly that type, so the glue's
            // internal downcast always succeeds.
            value: (self.clone)(&*self.value),
            value_hash: self.value_hash,
            clone: self.clone,
        }
    }
}

/// Clone a captured resource list element-for-element (deep copy of each value).
pub(super) fn clone_resources(src: &[SnapshotResource]) -> Vec<SnapshotResource> {
    src.iter().map(SnapshotResource::clone_entry).collect()
}

/// Capture every snapshot-registered resource that currently holds a value,
/// ascending by [`ResourceId`] so the order is deterministic. Registered
/// resources with an empty value slot are omitted (their absence is itself part
/// of the captured state and is reproduced on restore).
pub(super) fn capture_resources(world: &World) -> Vec<SnapshotResource> {
    let resources = &world.resources;
    let mut out = Vec::new();
    for id in resources.snapshot_registered_ids() {
        let Some(value) = resources.clone_value_boxed(id) else {
            // Registered for snapshotting, but no value is currently inserted.
            continue;
        };
        let type_id = resources
            .type_id_of(id)
            .expect("snapshot-registered id has a recorded type");
        let clone = resources
            .clone_fn_of(id)
            .expect("snapshot-registered id has clone glue");
        let value_hash = {
            let mut hasher = FnvHasher::new();
            if resources.hash_value_into(id, &mut hasher) {
                Some(core::hash::Hasher::finish(&hasher))
            } else {
                None
            }
        };
        out.push(SnapshotResource {
            id,
            type_id,
            value,
            value_hash,
            clone,
        });
    }
    out
}

/// Restore the snapshot-registered resources of `world` from `captured`.
///
/// For every resource currently registered for snapshotting: if the snapshot
/// captured a value for it, overwrite the live slot with a fresh clone of the
/// captured value; otherwise clear the live slot (the resource held no value at
/// capture time). Resources that were never registered for snapshotting are
/// left completely untouched — the opt-in contract. The clone keeps `captured`
/// intact so the same snapshot can drive repeated replays.
pub(super) fn restore_resources(world: &mut World, captured: &[SnapshotResource]) {
    for id in world.resources.snapshot_registered_ids() {
        match captured.iter().find(|r| r.id == id) {
            Some(sr) => {
                let fresh = (sr.clone)(&*sr.value);
                world.resources.insert_boxed(id, fresh);
            }
            None => world.resources.clear_value(id),
        }
    }
}
