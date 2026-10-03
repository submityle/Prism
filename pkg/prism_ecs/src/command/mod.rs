//! Deferred structural changes: the single-threaded [`CommandQueue`] and the
//! ergonomic [`Commands`] / [`EntityCommands`] builders layered on top of it,
//! plus the M2 parallel [`ParallelCommandBuffers`] with deterministic
//! sort-replay (design §9).
//!
//! Immediate structural operations on a [`World`](crate::world::World) (spawn /
//! insert / remove / despawn) require `&mut World`, which a running system does
//! not hold while it is iterating queries. The builders in this module record
//! those same operations as closures and hand back [`Entity`](crate::entity::Entity)
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

use alloc::boxed::Box;

use crate::world::World;

mod parallel;
mod queue;

/// A boxed, type-erased structural mutation applied to a [`World`].
///
/// Shared by both the single-threaded [`CommandQueue`] and the parallel
/// [`ParallelCommandBuffers`]. The `Send + Sync` bound lets a buffer (and the
/// commands it holds) be recorded on one worker and replayed on another.
pub(crate) type Command = Box<dyn FnOnce(&mut World) + Send + Sync>;

pub use parallel::{CommandKey, ParallelCommandBuffers, ParallelCommands, ParallelEntityCommands};
pub use queue::{CommandQueue, Commands, EntityCommands};
