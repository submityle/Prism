//! The query system: typed, filtered iteration over entities grouped by
//! archetype (design §7).
//!
//! A query is described by two type parameters:
//! - `D: `[`QueryData`] — the data read from each row (`&T`, `&mut T`,
//!   [`Entity`](crate::entity::Entity), `Option<&T>`, tuples thereof).
//! - `F: `[`QueryFilter`] — archetype-level predicates ([`With`], [`Without`],
//!   [`Or`], and tuples) that narrow which archetypes are visited without
//!   yielding data.
//!
//! [`World::query`](crate::world::World::query) and
//! [`World::query_filtered`](crate::world::World::query_filtered) build a
//! reusable [`QueryState`]; iterating it ([`QueryState::iter`] /
//! [`QueryState::iter_mut`]) yields a [`QueryIter`].
//!
//! The M0 surface deliberately omits the change-detection filters
//! `Added<T>` / `Changed<T>`: they depend on the double-layer tick and
//! chunk-version machinery scheduled for M2 (design §10), and shipping them as
//! always-matching stubs would be a fake implementation.

mod access;
mod fetch;
mod filter;
mod iter;
mod state;

pub use access::Access;
pub use fetch::{QueryData, ReadOnlyQueryData};
pub use filter::{Or, QueryFilter, With, Without};
pub use iter::QueryIter;
pub use state::QueryState;

#[cfg(test)]
mod tests;
