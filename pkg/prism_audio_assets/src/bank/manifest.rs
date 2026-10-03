//! The [`BankManifest`]: the metadata half of a loadable asset bank.
//!
//! # Provenance
//! Original work; no Unreal Engine, Unity, Godot, Wwise, FMOD, Steam Audio, or
//! Google Resonance Audio source or derived code; no AI/ML.
//!
//! # Relationship
//! Implements the metadata model of design section 20: a bank groups a set of
//! events, containers, patches, and media entries into a single loadable and
//! unloadable unit, and declares the other banks it depends on. The manifest is
//! pure data (no I/O) so it is `no_std`-friendly and can be authored, validated,
//! and budgeted entirely off the real-time thread.

#[cfg(not(feature = "std"))]
use alloc::string::String;
#[cfg(not(feature = "std"))]
use alloc::vec::Vec;

use crate::bank::entry::{EntryId, MediaEntry};

/// A stable identifier for a bank within a registry.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub struct BankId(pub u32);

/// A reference to an authored event packaged inside a bank.
///
/// The asset crate stores only the identity and name of an event; the full
/// event graph lives in the event/runtime crate and is linked by id.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub struct EventRef {
    /// Bank-local identifier of the event.
    pub id: EntryId,
    /// Human-readable event name.
    pub name: String,
}

/// A reference to an authored container (random/sequence/blend) in a bank.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub struct ContainerRef {
    /// Bank-local identifier of the container.
    pub id: EntryId,
    /// Human-readable container name.
    pub name: String,
}

/// A reference to an authored patch (parameter / mix snapshot) in a bank.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub struct PatchRef {
    /// Bank-local identifier of the patch.
    pub id: EntryId,
    /// Human-readable patch name.
    pub name: String,
}

/// Errors surfaced while validating a manifest.
#[derive(Debug, Clone, PartialEq, Eq)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
#[non_exhaustive]
pub enum ManifestError {
    /// Two media entries share the same [`EntryId`].
    DuplicateMediaId(EntryId),
    /// A media entry declares a byte range outside the bank blob.
    MediaRangeOutOfBounds(EntryId),
    /// A bank declares itself as one of its own dependencies.
    SelfDependency(BankId),
}

/// The metadata half of a loadable bank.
///
/// A manifest pairs with a byte blob (the concatenated encoded media) to form a
/// complete bank. It carries the lists of events, containers, patches, and
/// media entries, plus the ids of other banks that must be resident first.
#[derive(Debug, Clone, PartialEq, Eq)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub struct BankManifest {
    /// Stable bank identifier.
    pub id: BankId,
    /// Human-readable bank name.
    pub name: String,
    /// Monotonic content version used for cache invalidation.
    pub version: u32,
    /// Media entries packaged in the bank.
    pub media: Vec<MediaEntry>,
    /// Event references packaged in the bank.
    pub events: Vec<EventRef>,
    /// Container references packaged in the bank.
    pub containers: Vec<ContainerRef>,
    /// Patch references packaged in the bank.
    pub patches: Vec<PatchRef>,
    /// Banks that must be loaded before this one.
    pub dependencies: Vec<BankId>,
}

impl BankManifest {
    /// Creates a new, empty manifest with the given identity.
    #[must_use]
    pub fn new(id: BankId, name: String, version: u32) -> Self {
        Self {
            id,
            name,
            version,
            media: Vec::new(),
            events: Vec::new(),
            containers: Vec::new(),
            patches: Vec::new(),
            dependencies: Vec::new(),
        }
    }

    /// Adds a media entry and returns the manifest for chaining.
    #[must_use]
    pub fn with_media(mut self, entry: MediaEntry) -> Self {
        self.media.push(entry);
        self
    }

    /// Declares a dependency on another bank and returns the manifest.
    #[must_use]
    pub fn with_dependency(mut self, dependency: BankId) -> Self {
        self.dependencies.push(dependency);
        self
    }

    /// Finds a media entry by its bank-local id.
    #[must_use]
    pub fn media_by_id(&self, id: EntryId) -> Option<&MediaEntry> {
        self.media.iter().find(|entry| entry.id == id)
    }

    /// Finds a media entry by name.
    #[must_use]
    pub fn media_by_name(&self, name: &str) -> Option<&MediaEntry> {
        self.media.iter().find(|entry| entry.name == name)
    }

    /// Returns the total number of encoded media bytes the manifest declares.
    #[must_use]
    pub fn declared_encoded_bytes(&self) -> u64 {
        self.media.iter().map(|entry| entry.byte_len).sum()
    }

    /// Returns the total number of bytes a fully resident, decoded copy of the
    /// memory-resident entries would occupy, when their frame counts are known.
    #[must_use]
    pub fn declared_resident_bytes(&self) -> u64 {
        self.media
            .iter()
            .filter(|entry| !entry.is_streaming())
            .filter_map(|entry| entry.format.decoded_bytes())
            .sum()
    }

    /// Validates the manifest against the `blob_len` bytes it will be paired
    /// with.
    ///
    /// Checks for duplicate media ids, byte ranges that fall outside the blob,
    /// and self-dependencies. Returns the first error found.
    pub fn validate(&self, blob_len: u64) -> Result<(), ManifestError> {
        for (index, entry) in self.media.iter().enumerate() {
            for other in &self.media[index + 1..] {
                if other.id == entry.id {
                    return Err(ManifestError::DuplicateMediaId(entry.id));
                }
            }
            let end = entry.byte_offset.saturating_add(entry.byte_len);
            if end > blob_len {
                return Err(ManifestError::MediaRangeOutOfBounds(entry.id));
            }
        }
        if self.dependencies.contains(&self.id) {
            return Err(ManifestError::SelfDependency(self.id));
        }
        Ok(())
    }
}
