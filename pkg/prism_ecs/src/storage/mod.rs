//! Physical storage for component data.
//!
//! The M0 storage model is columnar (Structure-of-Arrays): each archetype owns
//! a [`Table`] holding one type-erased [`BlobVec`] column per component type,
//! kept in lockstep with a parallel entity vector. This is the cache-friendly
//! layout that later milestones extend with fixed-size chunks, per-chunk
//! change-versions, SparseSet columns, and SIMD iteration (design §6).

mod blob_vec;
mod chunk;
mod sparse;
mod table;

pub use blob_vec::BlobVec;
pub use chunk::{rows_per_chunk, ChunkVersions, TARGET_CHUNK_BYTES};
pub use sparse::ComponentSparseSet;
pub use table::{Column, Table};
