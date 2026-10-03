//! Hashing primitives: a fast hasher, the map/set facade, and stable/content
//! hashing for cross-run identity.

pub mod hashers;
pub mod map;
pub mod stable;

pub use hashers::{FxBuildHasher, FxHasher};
pub use map::{new_map, new_set, HashMap, HashSet};
pub use stable::{
    stable_hash, stable_hash_bytes, stable_hash_str, ContentHash, StableBuildHasher, StableHasher,
};
