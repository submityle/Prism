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
//!
//! The fourth state, Unity-style *SharedComponent* value-clustering (design
//! §6), is not yet present; it is honestly absent rather than stubbed.

mod blob_vec;
mod chunk;
mod owning_group;
mod sparse;
mod sparse_sets;
mod table;

pub use blob_vec::BlobVec;
pub use chunk::{rows_per_chunk, ChunkVersions, TARGET_CHUNK_BYTES};
pub use owning_group::{OwningGroup, OwningGroupId, OwningGroupIter};
pub use sparse::ComponentSparseSet;
pub use sparse_sets::SparseSets;
pub use table::{Column, Table};
