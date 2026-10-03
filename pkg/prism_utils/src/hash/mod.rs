//! Hashing primitives: a fast hasher and the map/set facade.

pub mod hashers;
pub mod map;

pub use hashers::{FxBuildHasher, FxHasher};
pub use map::{new_map, new_set, HashMap, HashSet};
