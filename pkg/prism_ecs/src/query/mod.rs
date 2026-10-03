//! The query system: typed, filtered iteration over entities grouped by
//! archetype (design §7).
//!
//! A query is described by two type parameters:
//! - `D: `[`QueryData`] — the data read from each row (`&T`, `&mut T`,
//!   [`Entity`](crate::entity::Entity), `Option<&T>`, tuples thereof).
//! - `F: `[`QueryFilter`] — predicates ([`With`], [`Without`], [`Or`], the
//!   change-detection filters [`Added`] / [`Changed`], and tuples) that narrow
//!   which rows are visited without yielding data.
//!
//! [`World::query`](crate::world::World::query) and
//! [`World::query_filtered`](crate::world::World::query_filtered) build a
//! reusable [`QueryState`]; iterating it ([`QueryState::iter`] /
//! [`QueryState::iter_mut`]) yields a [`QueryIter`].
//!
//! The change-detection filters [`Added`] / [`Changed`] match per row against
//! the querying observer's `[last_run, this_run)` tick window (design §10),
//! driven by the per-column ticks on [`Column`](crate::storage::Column).

mod access;
mod dirty;
mod fetch;
mod filter;
mod iter;
mod state;

pub use access::Access;
pub use dirty::DirtyChunk;
pub use fetch::{QueryData, ReadOnlyQueryData};
pub use filter::{Added, Changed, Or, QueryFilter, With, Without};
pub use iter::QueryIter;
pub use state::QueryState;

#[cfg(test)]
mod change_detection_tests;
#[cfg(test)]
mod dirty_tests;
#[cfg(test)]
mod tests;
