//! SwissTable-style map/set facade.
//!
//! M0 provides type aliases over the standard-library hash containers keyed by
//! the fast [`FxBuildHasher`](crate::hash::FxBuildHasher). The public names are
//! stable so a self-owned SwissTable backend can be swapped in later without
//! touching call sites.


use crate::hash::hashers::FxBuildHasher;

/// A hash map using the fast [`FxBuildHasher`].
pub type HashMap<K, V> = std::collections::HashMap<K, V, FxBuildHasher>;

/// A hash set using the fast [`FxBuildHasher`].
pub type HashSet<T> = std::collections::HashSet<T, FxBuildHasher>;

/// Construct an empty [`HashMap`].
pub fn new_map<K, V>() -> HashMap<K, V> {
    HashMap::default()
}

/// Construct an empty [`HashSet`].
pub fn new_set<T>() -> HashSet<T> {
    HashSet::default()
}
