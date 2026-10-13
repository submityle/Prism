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
mod combinations;
mod dirty;
mod fetch;
mod filter;
mod iter;
mod many;
mod matched;
#[cfg(feature = "multi_thread")]
mod par;
#[cfg(feature = "simd")]
mod simd;
mod slice;
mod slice_iter;
mod state;

pub use access::Access;
pub use combinations::QueryCombinationsIter;
pub use dirty::DirtyChunk;
pub use fetch::{AnyOf, Has, QueryData, ReadOnlyQueryData};
pub use filter::{Added, Changed, Or, QueryFilter, With, Without};
pub use iter::QueryIter;
pub use many::{QueryManyIter, QueryManyMatchedIter};
#[cfg(feature = "simd")]
pub use simd::{active, add_assign, axpy_assign, scale_assign, sum, Backend};
pub use slice::{ArchetypalFilter, ColumnSliceData};
pub use slice_iter::QuerySliceIter;
pub use state::{QueryEntityError, QuerySingleError, QueryState};

#[cfg(test)]
mod any_of_tests;
#[cfg(all(test, feature = "std"))]
mod cache_concurrency_tests;
#[cfg(test)]
mod change_detection_tests;
#[cfg(test)]
mod combinations_tests;
#[cfg(test)]
mod dirty_tests;
#[cfg(test)]
mod get_tests;
#[cfg(test)]
mod has_tests;
#[cfg(test)]
mod identity_tests;
#[cfg(test)]
mod many_tests;
#[cfg(all(test, feature = "multi_thread"))]
mod par_tests;
#[cfg(test)]
mod single_tests;
#[cfg(test)]
mod slice_iter_tests;
#[cfg(all(test, feature = "multi_thread"))]
mod slice_tests;
#[cfg(test)]
mod sparse_tests;
#[cfg(test)]
mod tests;
