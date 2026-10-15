//! Deferred structural changes: the single-threaded [`CommandQueue`] and the
//! ergonomic [`Commands`] / [`EntityCommands`] builders layered on top of it,
//! plus the M2 parallel [`ParallelCommandBuffers`] with deterministic
//! sort-replay (design §9).
//!
//! Immediate structural operations on a [`World`](crate::world::World) (spawn /
//! insert / remove / despawn) require `&mut World`, which a running system does
//! not hold while it is iterating queries. The builders in this module record
//! typed insert/remove commands and opaque operations, and hand back [`Entity`](crate::entity::Entity)
//! handles *immediately* (via the lock-free
//! [`Entities::reserve_entity`](crate::entity::Entities::reserve_entity)
//! reservation path), so a system can queue spawns and edits without taking an
//! exclusive world borrow. The recorded commands are drained and applied later
//! at a synchronization point.
//!
//! # Two layers
//!
//! * [`CommandQueue`] (in [`queue`]) is the single-threaded, insertion-ordered
//!   buffer. It is the simplest producer and the migration-compatible surface
//!   mirroring `bevy_ecs`'s `Commands` (design §18).
//! * [`ParallelCommandBuffers`] (in [`parallel`]) is the M2 refinement: one
//!   independent buffer per producer / worker with **no shared locking on the
//!   hot path**. Each recorded command carries a deterministic
//!   [`CommandKey`] so the sync-point merge replays every command in a stable
//!   total order that is independent of thread scheduling — the §9
//!   "sync point 按确定键排序合并回放" contract.
//!
//! Both layers share the same type-erased [`Command`] representation and the
//! same reserved-entity discipline: reserved handles are materialised by
//! [`World::flush_reserved`](crate::world::World::flush_reserved) before any
//! deferred spawn lands.
//!
//! Adjacent edits to the same entity share one final archetype migration when
//! no lifecycle hooks, observers, required constructors, owning groups, shared
//! edits or component destructors can observe intermediate states. Other edits
//! replay immediately. Arbitrary [`Commands::queue`] / [`ParallelCommands::queue`]
//! callbacks, spawns, despawns and entity changes are replay barriers.

pub(crate) mod coalescing;
mod parallel;
mod queue;

#[cfg(test)]
mod coalescing_tests;

use coalescing::Command;

pub use parallel::{CommandKey, ParallelCommandBuffers, ParallelCommands, ParallelEntityCommands};
pub use queue::{ChildBuilder, CommandQueue, Commands, EntityCommands};
