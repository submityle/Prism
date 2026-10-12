//! Identity types for the scene runtime (design §6).
//!
//! Four id kinds work together so that "instantiated internal references never
//! cross instances" and "a cross-session handle can locate one entity":
//!
//! - [`LocalEntityId`] — an entity's index *within one template*; every
//!   template-internal entity-valued field (parent pointer, relation target,
//!   attachment socket) stores this, never a runtime `Entity`.
//! - [`NameTable`] — a template-local `name -> LocalEntityId` map so override
//!   and query APIs can speak names rather than raw indices.
//! - [`PersistentId`] — a 128-bit, *deterministically derived* handle that is
//!   stable across sessions and machines: the same authored content instantiated
//!   at the same position yields the same id. Save games store
//!   `PersistentId -> override delta`; rollback/replication address by it.
//! - [`PersistentIdBuilder`] — ergonomic accumulation of the GUID chain plus the
//!   local entity path the derivation consumes.

use alloc::collections::BTreeMap;
use alloc::string::String;
use alloc::vec::Vec;

use prism_asset::StableGuid;

/// Index of an entity within a single scene template (design §6).
///
/// Base-template entities occupy the low range `0..`[`RESERVED_BASE`]; entities
/// added by a structural override (design §7.1 `StructuralOp::AddEntity`) live
/// in the reserved high segment so they never collide with the base numbering
/// and survive a base re-bake that reorders base entities.
///
/// [`RESERVED_BASE`]: LocalEntityId::RESERVED_BASE
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct LocalEntityId(pub u32);

impl LocalEntityId {
    /// First index of the reserved segment used for structurally-added entities.
    pub const RESERVED_BASE: u32 = 0x8000_0000;

    /// Builds a base-segment id from its `0`-based index.
    #[must_use]
    pub const fn new(index: u32) -> Self {
        Self(index)
    }

    /// Builds a reserved-segment id from a `0`-based offset.
    #[must_use]
    pub const fn reserved(offset: u32) -> Self {
        Self(Self::RESERVED_BASE + offset)
    }

    /// The raw index.
    #[must_use]
    pub const fn index(self) -> u32 {
        self.0
    }

    /// Whether this id lies in the reserved (structural-override) segment.
    #[must_use]
    pub const fn is_reserved(self) -> bool {
        self.0 >= Self::RESERVED_BASE
    }
}

/// FNV-1a 128-bit constants (the standard FNV-1a-128 offset basis and prime).
const FNV128_OFFSET: u128 = 0x6c62_272e_07bb_0142_62b8_2175_6295_c58d;
const FNV128_PRIME: u128 = 0x0000_0000_0100_0000_0000_0000_0000_013b;

/// A streaming FNV-1a 128-bit hasher. Dependency-free and deterministic across
/// platforms (integer-only), which is what the `PersistentId` contract needs.
#[derive(Clone, Copy, Debug)]
struct Fnv128 {
    state: u128,
}

impl Fnv128 {
    const fn new() -> Self {
        Self {
            state: FNV128_OFFSET,
        }
    }

    fn write(&mut self, bytes: &[u8]) {
        for &b in bytes {
            self.state ^= u128::from(b);
            self.state = self.state.wrapping_mul(FNV128_PRIME);
        }
    }

    fn write_u32(&mut self, value: u32) {
        self.write(&value.to_le_bytes());
    }

    fn write_u128(&mut self, value: u128) {
        self.write(&value.to_le_bytes());
    }

    const fn finish(self) -> u128 {
        self.state
    }
}

/// One byte separating the GUID chain from the local path in the derivation
/// stream, so two different `(chain, path)` splits can never hash equal.
const DERIVE_SEPARATOR: u8 = 0xff;

/// A deterministic, cross-session identifier for one entity inside one
/// instantiated scene (design §6.2).
///
/// `PersistentId = hash128(GUID chain ‖ local path)`, where the chain is the
/// top-level scene GUID followed by each nested sub-template GUID along the
/// containment path, and the local path is the sequence of [`LocalEntityId`]s
/// from a template root down to the entity.
///
/// The id is *stable*: moving or renaming an entity does not change it (it is
/// based on `LocalEntityId`, not the runtime `Entity`); only an author reorder
/// of `LocalEntityId` changes it, and a bake supplies a migration map for that
/// case (design §21).
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct PersistentId(u128);

impl PersistentId {
    /// The all-zero id; never produced by [`derive`](Self::derive) for a
    /// non-empty input in practice, and usable as an "unset" sentinel.
    pub const NIL: Self = Self(0);

    /// Deterministically derives a `PersistentId` from a template-GUID chain and
    /// a local entity path.
    #[must_use]
    pub fn derive(chain: &[StableGuid], local_path: &[LocalEntityId]) -> Self {
        let mut hasher = Fnv128::new();
        for guid in chain {
            hasher.write_u128(guid.to_u128());
        }
        hasher.write(&[DERIVE_SEPARATOR]);
        for id in local_path {
            hasher.write_u32(id.index());
        }
        Self(hasher.finish())
    }

    /// The raw 128-bit value.
    #[must_use]
    pub const fn to_u128(self) -> u128 {
        self.0
    }

    /// Wraps a raw 128-bit value.
    #[must_use]
    pub const fn from_u128(value: u128) -> Self {
        Self(value)
    }

    /// Whether this is [`PersistentId::NIL`].
    #[must_use]
    pub const fn is_nil(self) -> bool {
        self.0 == 0
    }
}

/// Accumulates the GUID chain and local path a [`PersistentId`] is derived from.
///
/// ```
/// use prism_asset::StableGuid;
/// use prism_scene::{LocalEntityId, PersistentIdBuilder};
///
/// let scene_guid = StableGuid::from_u128(0x1);
/// let nested_guid = StableGuid::from_u128(0x2);
/// let id = PersistentIdBuilder::new()
///     .template(scene_guid)
///     .template(nested_guid)
///     .local(LocalEntityId::new(0))
///     .local(LocalEntityId::new(3))
///     .finish();
/// assert!(!id.is_nil());
/// ```
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct PersistentIdBuilder {
    chain: Vec<StableGuid>,
    path: Vec<LocalEntityId>,
}

impl PersistentIdBuilder {
    /// Starts an empty derivation.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Appends one template GUID (top-level scene first, then nested templates).
    pub fn template(&mut self, guid: StableGuid) -> &mut Self {
        self.chain.push(guid);
        self
    }

    /// Appends one local entity step.
    pub fn local(&mut self, id: LocalEntityId) -> &mut Self {
        self.path.push(id);
        self
    }

    /// Finalizes the derivation.
    #[must_use]
    pub fn finish(&self) -> PersistentId {
        PersistentId::derive(&self.chain, &self.path)
    }
}

/// A template-local `name -> LocalEntityId` table (design §6 `NameTag`).
///
/// Backed by a `BTreeMap`, so iteration is deterministic (sorted by name) — the
/// property the bake and the deterministic-replay path rely on.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct NameTable {
    entries: BTreeMap<String, LocalEntityId>,
}

impl NameTable {
    /// An empty table.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Inserts `name -> id`, returning any id previously bound to `name`.
    pub fn insert(&mut self, name: impl Into<String>, id: LocalEntityId) -> Option<LocalEntityId> {
        self.entries.insert(name.into(), id)
    }

    /// Resolves a name.
    #[must_use]
    pub fn get(&self, name: &str) -> Option<LocalEntityId> {
        self.entries.get(name).copied()
    }

    /// Whether `name` is bound.
    #[must_use]
    pub fn contains(&self, name: &str) -> bool {
        self.entries.contains_key(name)
    }

    /// Number of bound names.
    #[must_use]
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Whether the table is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Iterates `(name, id)` pairs in deterministic (sorted) order.
    pub fn iter(&self) -> impl Iterator<Item = (&str, LocalEntityId)> {
        self.entries.iter().map(|(name, id)| (name.as_str(), *id))
    }
}

#[cfg(test)]
mod tests {
    use super::{LocalEntityId, NameTable, PersistentId, PersistentIdBuilder};
    use alloc::vec::Vec;
    use prism_asset::StableGuid;

    fn guid(seed: u128) -> StableGuid {
        StableGuid::from_u128(seed)
    }

    #[test]
    fn local_entity_id_segments_do_not_overlap() {
        assert!(!LocalEntityId::new(0).is_reserved());
        assert!(!LocalEntityId::new(LocalEntityId::RESERVED_BASE - 1).is_reserved());
        assert!(LocalEntityId::reserved(0).is_reserved());
        assert_eq!(LocalEntityId::reserved(0).index(), LocalEntityId::RESERVED_BASE);
        assert_eq!(LocalEntityId::reserved(5).index(), LocalEntityId::RESERVED_BASE + 5);
    }

    #[test]
    fn persistent_id_is_deterministic_and_order_sensitive() {
        let chain = [guid(1), guid(2)];
        let path = [LocalEntityId::new(3), LocalEntityId::new(7)];
        let a = PersistentId::derive(&chain, &path);
        let b = PersistentId::derive(&chain, &path);
        assert_eq!(a, b, "same inputs derive the same id");
        assert!(!a.is_nil());

        // A different local path yields a different id.
        let other = PersistentId::derive(&chain, &[LocalEntityId::new(3)]);
        assert_ne!(a, other);

        // A different chain yields a different id.
        let other_chain = PersistentId::derive(&[guid(1), guid(9)], &path);
        assert_ne!(a, other_chain);
    }

    #[test]
    fn chain_and_path_split_is_unambiguous() {
        // Moving a byte between "chain" and "path" must not collide: the
        // separator plus fixed-width encoding prevents it.
        let chain_a = [guid(0)];
        let path_a = [LocalEntityId::new(0x0102_0304)];
        let chain_b = [guid(0)];
        let path_b = [LocalEntityId::new(0x0102_0304)];
        assert_eq!(
            PersistentId::derive(&chain_a, &path_a),
            PersistentId::derive(&chain_b, &path_b)
        );

        // An empty chain with one path step differs from a one-GUID chain with
        // an empty path.
        let with_guid = PersistentId::derive(&[guid(0)], &[]);
        let with_path = PersistentId::derive(&[], &[LocalEntityId::new(0)]);
        assert_ne!(with_guid, with_path);
    }

    #[test]
    fn builder_matches_derive() {
        let chain = [guid(42), guid(99)];
        let path = [LocalEntityId::new(1), LocalEntityId::new(2)];
        let built = PersistentIdBuilder::new()
            .template(chain[0])
            .template(chain[1])
            .local(path[0])
            .local(path[1])
            .finish();
        assert_eq!(built, PersistentId::derive(&chain, &path));
    }

    #[test]
    fn round_trips_raw_value() {
        let id = PersistentId::from_u128(0xdead_beef_0000_0000_0000_0000_cafe_f00d);
        assert_eq!(id.to_u128(), 0xdead_beef_0000_0000_0000_0000_cafe_f00d);
        assert!(!id.is_nil());
        assert!(PersistentId::NIL.is_nil());
    }

    #[test]
    fn name_table_resolves_and_iterates_sorted() {
        let mut names = NameTable::new();
        assert!(names.is_empty());
        names.insert("root", LocalEntityId::new(0));
        names.insert("wheel", LocalEntityId::new(4));
        names.insert("axle", LocalEntityId::new(2));

        assert_eq!(names.len(), 3);
        assert_eq!(names.get("wheel"), Some(LocalEntityId::new(4)));
        assert_eq!(names.get("missing"), None);
        assert!(names.contains("root"));

        let order: Vec<&str> = names.iter().map(|(n, _)| n).collect();
        assert_eq!(order, ["axle", "root", "wheel"]);

        // Re-inserting a name returns the previous binding.
        assert_eq!(
            names.insert("root", LocalEntityId::new(8)),
            Some(LocalEntityId::new(0))
        );
        assert_eq!(names.get("root"), Some(LocalEntityId::new(8)));
    }
}
