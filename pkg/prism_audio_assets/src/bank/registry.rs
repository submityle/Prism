//! The [`BankRegistry`]: a set of loaded banks with dependency bookkeeping.
//!
//! # Provenance
//! Original work; no Unreal Engine, Unity, Godot, Wwise, FMOD, Steam Audio, or
//! Google Resonance Audio source or derived code; no AI/ML.
//!
//! # Relationship
//! Implements the bank load/unload lifecycle of design section 20. The registry
//! tracks which banks are resident, enforces that a bank's declared
//! dependencies are loaded first, and refuses to unload a bank that another
//! resident bank still depends on. Media is resolved across banks by
//! `(BankId, EntryId)` or by name, and the registry reports an aggregate memory
//! footprint for budgeting.

use alloc::collections::BTreeMap;
use alloc::vec::Vec;

use crate::bank::entry::EntryId;
use crate::bank::handle::{BankLoadError, LoadedBank, LoadedMedia};
use crate::bank::manifest::{BankId, BankManifest};
use crate::codec::registry::DecoderRegistry;

/// Errors surfaced while loading or unloading a bank in the registry.
#[derive(Debug, Clone, PartialEq, Eq)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
#[non_exhaustive]
pub enum BankRegistryError {
    /// A bank with the same id is already loaded.
    AlreadyLoaded(BankId),
    /// A declared dependency is not loaded.
    MissingDependency {
        /// The bank being loaded.
        bank: BankId,
        /// The dependency that was not resident.
        dependency: BankId,
    },
    /// The bank's manifest/blob failed to load.
    Load(BankLoadError),
    /// The bank to unload is not loaded.
    NotLoaded(BankId),
    /// The bank cannot be unloaded because another resident bank depends on it.
    StillDepended {
        /// The bank that was asked to unload.
        bank: BankId,
        /// A resident bank that still depends on it.
        dependent: BankId,
    },
}

/// A collection of resident banks keyed by [`BankId`].
#[derive(Default)]
pub struct BankRegistry {
    banks: BTreeMap<BankId, LoadedBank>,
}

impl BankRegistry {
    /// Creates an empty registry.
    #[must_use]
    pub fn new() -> Self {
        Self {
            banks: BTreeMap::new(),
        }
    }

    /// Returns `true` when a bank with `id` is resident.
    #[must_use]
    pub fn is_loaded(&self, id: BankId) -> bool {
        self.banks.contains_key(&id)
    }

    /// Returns the number of resident banks.
    #[must_use]
    pub fn len(&self) -> usize {
        self.banks.len()
    }

    /// Returns `true` when no bank is resident.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.banks.is_empty()
    }

    /// Loads a bank from its `manifest`, `blob`, and decoder `registry`.
    ///
    /// Fails when the id is already loaded, when any declared dependency is not
    /// resident, or when the manifest/blob fails to load.
    pub fn load(
        &mut self,
        manifest: BankManifest,
        blob: &[u8],
        registry: &DecoderRegistry,
    ) -> Result<BankId, BankRegistryError> {
        let id = manifest.id;
        if self.banks.contains_key(&id) {
            return Err(BankRegistryError::AlreadyLoaded(id));
        }
        for dependency in &manifest.dependencies {
            if !self.banks.contains_key(dependency) {
                return Err(BankRegistryError::MissingDependency {
                    bank: id,
                    dependency: *dependency,
                });
            }
        }
        let bank = LoadedBank::load(manifest, blob, registry).map_err(BankRegistryError::Load)?;
        self.banks.insert(id, bank);
        Ok(id)
    }

    /// Unloads a resident bank.
    ///
    /// Refuses with [`BankRegistryError::StillDepended`] when another resident
    /// bank still declares `id` as a dependency.
    pub fn unload(&mut self, id: BankId) -> Result<(), BankRegistryError> {
        if !self.banks.contains_key(&id) {
            return Err(BankRegistryError::NotLoaded(id));
        }
        for (other_id, bank) in &self.banks {
            if *other_id == id {
                continue;
            }
            if bank.manifest().dependencies.contains(&id) {
                return Err(BankRegistryError::StillDepended {
                    bank: id,
                    dependent: *other_id,
                });
            }
        }
        self.banks.remove(&id);
        Ok(())
    }

    /// Returns a resident bank by id.
    #[must_use]
    pub fn bank(&self, id: BankId) -> Option<&LoadedBank> {
        self.banks.get(&id)
    }

    /// Resolves media across banks by `(BankId, EntryId)`.
    #[must_use]
    pub fn media(&self, bank: BankId, entry: EntryId) -> Option<&LoadedMedia> {
        self.banks.get(&bank)?.media_by_id(entry)
    }

    /// Resolves media by bank id and media name.
    #[must_use]
    pub fn media_by_name(&self, bank: BankId, name: &str) -> Option<&LoadedMedia> {
        self.banks.get(&bank)?.media_by_name(name)
    }

    /// Returns the ids of all resident banks in ascending order.
    #[must_use]
    pub fn loaded_ids(&self) -> Vec<BankId> {
        self.banks.keys().copied().collect()
    }

    /// Returns the total resident PCM plus encoded bytes across all banks.
    #[must_use]
    pub fn total_memory_bytes(&self) -> u64 {
        self.banks
            .values()
            .map(|bank| bank.memory_usage().total())
            .fold(0u64, u64::saturating_add)
    }
}
