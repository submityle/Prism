//! Durable cell streaming owned by the scene, not the ECS kernel.
//!
//! [`PersistentPartition::advance`] is the production entry point: it consumes
//! the real `WorldPartition` plan, writes a cell atomically **before** invoking
//! `StreamDriver::evict_cell`, loads fresh entities and finally settles loads.
//! Missing archives are errors, not silently created empty cells. Author new
//! cells with `register` / `create_empty_cell` and `save_cell` first.
//!
//! Archives contain concrete reflected components, internal prefab ancestry,
//! empty entities, persistent identities and ordinary outgoing relations.
//! Ordinary relation kinds must be registered with identical names and policy
//! on the destination World. Cross-cell references should use
//! [`PersistentEntityRef`], never raw runtime handle bits. An explicit archive
//! remapper is available for application-owned intra-cell Entity fields.
//!
//! Each relation is persisted by its **source** cell. While its target is
//! absent it is pending, and rebinds when both endpoints are resident. Edits to
//! pending relations must use `link` / `unlink`; absence from the live ECS
//! relation index alone does not mean deletion of an unloaded reference.
//!
//! Safety boundaries are explicit: resources remain World-owned; unreflected
//! components, shared storage, external prefab ancestry and arbitrary World
//! lifecycle callbacks are rejected rather than silently lost. As with the
//! reflection archive, panicking destructors or impure user FromReflect/remap
//! implementations are not transactional. Atomic files are per-cell commits,
//! not a multi-cell crash-consistent global checkpoint. World replacement or
//! snapshot restore invalidates an owner; create a new owner and register its
//! current residents rather than resolve old runtime handles into that World.

mod codec;
mod owner;
mod store;

pub use owner::{
    CellFailure, CellOperation, PartitionError, PartitionTick, PersistentEntityRef,
    PersistentPartition,
};
pub use store::{CellStore, FileCellStore, MAX_CELL_BYTES};
