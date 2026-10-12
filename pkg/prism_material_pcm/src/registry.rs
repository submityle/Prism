//! The registry: one table for every extension point, with a
//! namespace-partitioned slot allocator, the promotion state machine and the
//! §4.5 field predicates made machine-checkable.

use crate::manifest::{ExtensionManifest, ManifestError};
use crate::namespace::{Kind, Namespace, Region, Stability};
use crate::{CapabilitySet, HEADER_DIGEST_SLOTS};
use alloc::string::String;
use alloc::vec::Vec;
use core::fmt;

/// A namespace's slot region is full.
///
/// This is an *address-space* exhaustion, deliberately loud: it is reported as
/// an error rather than silently folding a new extension into an existing bit.
/// The content model (authored layers, lobes, channels) stays unbounded; only
/// the fixed-width header digest has finite room.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct RegistryPoolExhausted {
    /// The namespace whose region is full.
    pub origin: Namespace,
    /// The region that ran out.
    pub region: Region,
    /// The region's slot capacity.
    pub capacity: u32,
}

impl fmt::Display for RegistryPoolExhausted {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "the {} registry region for {} is full ({} slots)",
            self.region.name(),
            self.origin,
            self.capacity
        )
    }
}

impl core::error::Error for RegistryPoolExhausted {}

/// Everything that can go wrong registering or promoting an extension.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum RegistryError {
    /// The manifest violated a §5.4 identity predicate.
    Manifest(ManifestError),
    /// Another extension already owns this `uid`.
    DuplicateUid(String),
    /// Another extension already owns this `registry_slot`.
    SlotInUse(u32),
    /// No extension with this `uid` is registered.
    UnknownUid(String),
    /// The origin namespace's region is exhausted.
    PoolExhausted(RegistryPoolExhausted),
}

impl fmt::Display for RegistryError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Manifest(error) => write!(f, "{error}"),
            Self::DuplicateUid(uid) => write!(f, "extension uid {uid:?} is already registered"),
            Self::SlotInUse(slot) => write!(f, "registry slot {slot} is already in use"),
            Self::UnknownUid(uid) => write!(f, "no extension registered with uid {uid:?}"),
            Self::PoolExhausted(exhausted) => write!(f, "{exhausted}"),
        }
    }
}

impl core::error::Error for RegistryError {}

impl From<ManifestError> for RegistryError {
    fn from(error: ManifestError) -> Self {
        Self::Manifest(error)
    }
}

/// The single extension registry (§5).
#[derive(Clone, Debug)]
pub struct ExtensionRegistry {
    slots: [Option<ExtensionManifest>; HEADER_DIGEST_SLOTS as usize],
    /// Extensions that resolve at bake/pass time and hold no header-digest
    /// address (§5.2). They share every other governance machine with the
    /// slot-resident rows — identity, promotion, capability needs — and are kept
    /// in registration order.
    identity_only: Vec<ExtensionManifest>,
    /// Per-region first-fit cursor, so a fresh extension takes the lowest free
    /// slot of its region and slots are never recycled.
    cursors: [u32; Region::ALL.len()],
}

impl Default for ExtensionRegistry {
    fn default() -> Self {
        Self::new()
    }
}

impl ExtensionRegistry {
    /// An empty registry.
    pub const fn new() -> Self {
        Self {
            slots: [None; HEADER_DIGEST_SLOTS as usize],
            identity_only: Vec::new(),
            cursors: [0; Region::ALL.len()],
        }
    }

    /// Register an extension whose identity and slot come from a declaration
    /// (the specification core comes from the schema, which pins its slots; a
    /// bake/pass-time extension arrives identity-only).
    ///
    /// # Errors
    ///
    /// Returns [`RegistryError::Manifest`] for a §5.4 predicate violation,
    /// [`RegistryError::DuplicateUid`] / [`RegistryError::SlotInUse`] for an
    /// identity collision.
    pub fn register(&mut self, manifest: ExtensionManifest) -> Result<(), RegistryError> {
        manifest.validate()?;
        if self.manifest_by_uid(manifest.uid()).is_some() {
            return Err(RegistryError::DuplicateUid(String::from(manifest.uid())));
        }
        let Some(registry_slot) = manifest.registry_slot() else {
            self.identity_only.push(manifest);
            return Ok(());
        };
        let slot = registry_slot as usize;
        if self.slots[slot].is_some() {
            return Err(RegistryError::SlotInUse(registry_slot));
        }
        self.slots[slot] = Some(manifest);
        // Keep the region's first-fit cursor monotone past a *pinned* slot, so a
        // later `register_new` can never hand out an address the schema (or the
        // catalogue) has already spoken for — even when the pinned slots are not
        // dense from the region base.
        let region = manifest.origin().region();
        let index = region as usize;
        let local = registry_slot - region.base() + 1;
        if local > self.cursors[index] {
            self.cursors[index] = local;
        }
        Ok(())
    }

    /// Allocate the lowest free slot in `origin`'s region and register a new
    /// **header-resident** extension there, returning the pinned manifest.
    ///
    /// # Errors
    ///
    /// Returns [`RegistryError::PoolExhausted`] when the region is full — an
    /// explicit, diagnosable failure instead of a silent fold-in — and
    /// [`RegistryError::Manifest`] when `kind` is not header resident (that kind
    /// registers identity-only, see [`Self::register_identity_only`]).
    pub fn register_new(
        &mut self,
        uid: &'static str,
        origin: Namespace,
        kind: Kind,
    ) -> Result<ExtensionManifest, RegistryError> {
        let slot = self.allocate_slot(origin)?;
        let manifest = ExtensionManifest::new(uid, slot, origin, kind);
        self.register(manifest)?;
        Ok(manifest)
    }

    /// Register a new **identity-only** extension: one that resolves at
    /// bake/pass time and therefore holds no header-digest address (§5.2).
    ///
    /// No slot is allocated, so this can never exhaust a region.
    ///
    /// # Errors
    ///
    /// Returns [`RegistryError::Manifest`] when `kind` is header resident (such
    /// a kind must pin a slot, see [`Self::register_new`]) or
    /// [`RegistryError::DuplicateUid`] on an identity collision.
    pub fn register_identity_only(
        &mut self,
        uid: &'static str,
        origin: Namespace,
        kind: Kind,
    ) -> Result<ExtensionManifest, RegistryError> {
        let manifest = ExtensionManifest::new_identity_only(uid, origin, kind);
        self.register(manifest)?;
        Ok(manifest)
    }

    /// Reserve the lowest free slot in `origin`'s region.
    ///
    /// # Errors
    ///
    /// Returns [`RegistryError::PoolExhausted`] when the region has no free
    /// slot left.
    pub fn allocate_slot(&mut self, origin: Namespace) -> Result<u32, RegistryError> {
        let region = origin.region();
        let index = region as usize;
        let mut candidate = self.cursors[index];
        while candidate < Region::SLOTS_PER_REGION {
            let slot = region.base() + candidate;
            if self.slots[slot as usize].is_none() {
                self.cursors[index] = candidate + 1;
                return Ok(slot);
            }
            candidate += 1;
        }
        Err(RegistryError::PoolExhausted(RegistryPoolExhausted {
            origin,
            region,
            capacity: Region::SLOTS_PER_REGION,
        }))
    }

    /// Promote a registered extension's attribution (§5.4).
    ///
    /// The `uid`, `registry_slot` and `origin` are untouched, so nothing that
    /// binds on identity changes.
    ///
    /// # Errors
    ///
    /// Returns [`RegistryError::UnknownUid`] when `uid` is not registered, or
    /// the [`ManifestError`]s of [`ExtensionManifest::promote`].
    pub fn promote(
        &mut self,
        uid: &str,
        to: Namespace,
        vendor: &'static str,
        stability: Stability,
    ) -> Result<ExtensionManifest, RegistryError> {
        let slot = self
            .slot_of_uid(uid)
            .ok_or_else(|| RegistryError::UnknownUid(String::from(uid)))?;
        let current = self.slots[slot as usize]
            .as_ref()
            .expect("slot_of_uid returned an occupied slot");
        let promoted = current.promote(to, vendor, stability)?;
        self.slots[slot as usize] = Some(promoted);
        Ok(promoted)
    }

    /// The extension registered at `slot`, if any.
    pub fn manifest_at(&self, slot: u32) -> Option<&ExtensionManifest> {
        self.slots.get(slot as usize)?.as_ref()
    }

    /// The extension registered under `uid`, if any.
    pub fn manifest_by_uid(&self, uid: &str) -> Option<&ExtensionManifest> {
        self.iter().find(|manifest| manifest.uid() == uid)
    }

    /// The slot `uid` was allocated, if it is registered *and* header resident.
    pub fn slot_of_uid(&self, uid: &str) -> Option<u32> {
        self.manifest_by_uid(uid)?.registry_slot()
    }

    /// Whether `slot` is registered.
    pub fn is_registered(&self, slot: u32) -> bool {
        self.manifest_at(slot).is_some()
    }

    /// Number of registered extensions, header resident and identity-only alike.
    pub fn len(&self) -> usize {
        self.slots.iter().flatten().count() + self.identity_only.len()
    }

    /// Whether the registry holds no extension.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Iterate the registered extensions: lowest slot first, then the
    /// identity-only rows in registration order.
    pub fn iter(&self) -> impl Iterator<Item = &ExtensionManifest> {
        self.slots
            .iter()
            .flatten()
            .chain(self.identity_only.iter())
    }

    /// The bake/pass-time extensions that hold no header-digest address.
    pub fn identity_only(&self) -> &[ExtensionManifest] {
        &self.identity_only
    }

    /// The capabilities required by every extension in `slots` that is
    /// registered — the load-time input to §5.3 negotiation.
    pub fn required_caps_of(&self, slots: impl IntoIterator<Item = u32>) -> CapabilitySet {
        let mut caps = CapabilitySet::EMPTY;
        for slot in slots {
            if let Some(manifest) = self.manifest_at(slot) {
                caps = caps.union(manifest.requires_caps());
            }
        }
        caps
    }

    /// The capabilities required by the extensions registered under `uids` — the
    /// **bake/pass-level** counterpart of [`Self::required_caps_of`].
    ///
    /// Identity-only extensions resolve before the material reaches the runtime,
    /// so they are negotiated by the baker or the pass-graph builder that
    /// consumes them, by name rather than by slot. An unknown name is skipped
    /// rather than assumed present: a requirement nobody can satisfy must not
    /// evaporate silently.
    pub fn required_caps_of_uids<'a>(
        &self,
        uids: impl IntoIterator<Item = &'a str>,
    ) -> CapabilitySet {
        let mut caps = CapabilitySet::EMPTY;
        for uid in uids {
            if let Some(manifest) = self.manifest_by_uid(uid) {
                caps = caps.union(manifest.requires_caps());
            }
        }
        caps
    }
}

/// The §4.5 constraint-1 predicate: is this a specification-core item?
///
/// Only `KHR`-attributed items may be compared against the `OpenPBR` reference
/// implementation; the conformance gate enumerates the registry with this
/// predicate.
pub fn is_specification_core(manifest: &ExtensionManifest) -> bool {
    manifest.namespace() == Namespace::Khr
}

/// The §4.5 constraint-2 predicate: does `slot` lie in the pure-specification
/// bit region?
///
/// A mask whose set bits all satisfy this is a *pure `OpenPBR`* mask, so
/// projecting it to the specification region is the identity.
pub fn is_specification_region(slot: u32) -> bool {
    Region::of(slot) == Some(Region::Specification)
}
