//! Physical storage for component data (design §6 存储模型).
//!
//! The primary model is columnar (Structure-of-Arrays): each archetype owns a
//! [`Table`] of type-erased [`BlobVec`] columns, one per component type, split
//! into fixed-size chunks carrying per-chunk [`ChunkVersions`] so change
//! detection and iteration can skip untouched chunks (design §6, §10).
//!
//! Of the design's four storage states this module implements:
//!
//! * **Table** — the default columnar/chunked layout above ([`Table`],
//!   [`Column`], and the private `chunk` splitting).
//! * **SparseSet** — out-of-band per-entity columns ([`ComponentSparseSet`],
//!   [`SparseSets`]) that toggle without an archetype move.
//! * **OwningGroup** — EnTT-style perfectly-packed membership for a super-hot
//!   query ([`OwningGroup`]); the packing structure and its O(1)
//!   pack/unpack transitions are complete and tested here.
//! * **SharedComponent** — Unity-style value de-duplication (design §6, §15 GPU
//!   批次键). The de-duplicating, reference-counted *value store*
//!   ([`SharedComponents`], [`SharedValuePool`], [`SharedValueId`]) that maps
//!   each distinct shared value to a stable dense id is complete and tested
//!   here. The archetype split that routes entities by that id — splitting an
//!   archetype into one variant per distinct shared-value binding — is still
//!   pending; the store is the substrate it builds on.

mod blob_vec;
mod chunk;
mod owning_group;
mod shared;
mod sparse;
mod sparse_sets;
mod table;

pub use blob_vec::BlobVec;
pub use chunk::{rows_per_chunk, ChunkVersions, TARGET_CHUNK_BYTES};
pub use owning_group::{OwningGroup, OwningGroupId, OwningGroupIter};
pub use shared::{SharedComponents, SharedValue, SharedValueId, SharedValuePool};
pub use sparse::ComponentSparseSet;
pub use sparse_sets::SparseSets;
pub use table::{Column, Table};
