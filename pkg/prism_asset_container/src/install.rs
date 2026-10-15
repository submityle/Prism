//! Transactional installation of runtime container overlays.
//!
//! Installation is staged against a clone of the current [`OverlaySet`].
//! Container readers are already strictly validated when created; this layer
//! validates the manifest-to-overlay binding before publishing the new
//! snapshot. A failed install therefore cannot expose a partially mounted
//! package to readers.

use alloc::string::String;
use alloc::vec::Vec;

use prism_asset::StableGuid;

use crate::{ContainerReader, ManifestReadError, ManifestReader, OverlaySet, RuntimeManifest};

/// One validated container to add to an install transaction.
#[derive(Clone, Debug)]
pub struct InstallMount {
    /// Manifest container name.
    pub name: String,
    /// Overlay priority; higher values win.
    pub priority: i32,
    /// Strictly validated container reader.
    pub reader: ContainerReader,
}

impl InstallMount {
    /// Creates an install mount from a validated reader.
    #[must_use]
    pub fn new(name: impl Into<String>, priority: i32, reader: ContainerReader) -> Self {
        Self {
            name: name.into(),
            priority,
            reader,
        }
    }
}

/// A patch tombstone to install atomically with its package containers.
#[derive(Clone, Debug)]
pub struct InstallTombstones {
    /// Tombstone mount name.
    pub name: String,
    /// Overlay priority; higher values win.
    pub priority: i32,
    /// GUIDs hidden by this patch.
    pub guids: Vec<StableGuid>,
}

impl InstallTombstones {
    /// Creates a tombstone mount specification.
    #[must_use]
    pub fn new(
        name: impl Into<String>,
        priority: i32,
        guids: impl IntoIterator<Item = StableGuid>,
    ) -> Self {
        Self {
            name: name.into(),
            priority,
            guids: guids.into_iter().collect(),
        }
    }
}

/// Errors raised while staging or validating an overlay install.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum InstallError {
    /// The manifest could not be bound to the staged overlay.
    Manifest(String),
}

impl core::fmt::Display for InstallError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::Manifest(error) => {
                write!(f, "runtime package manifest validation failed: {error}")
            }
        }
    }
}

impl std::error::Error for InstallError {}

impl From<ManifestReadError> for InstallError {
    fn from(error: ManifestReadError) -> Self {
        Self::Manifest(error.to_string())
    }
}

/// Installs a package into an overlay with all-or-nothing publication.
///
/// The supplied manifest is bound against the staged overlay before the live
/// overlay is replaced. Existing mounts remain visible if any route is
/// missing, has the wrong priority, or points at a container with a different
/// identity.
pub fn install_runtime_package(
    live: &mut OverlaySet,
    manifest: &RuntimeManifest,
    mounts: impl IntoIterator<Item = InstallMount>,
    tombstones: impl IntoIterator<Item = InstallTombstones>,
) -> Result<(), InstallError> {
    let mut staged = live.clone();
    for mount in mounts {
        staged.mount(mount.name, mount.priority, mount.reader);
    }
    for tombstone in tombstones {
        staged.mount_tombstones(tombstone.name, tombstone.priority, tombstone.guids);
    }

    // Binding performs every route, container identity, priority and payload
    // GUID check without mutating either the staged or live overlay.
    let _bound: ManifestReader<'_, '_> = manifest.bind(&staged)?;
    *live = staged;
    Ok(())
}
