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

use crate::{
    Codec, ContainerReader, ManifestReadError, ManifestReader, OverlaySet, RuntimeManifest,
};

/// Runtime admission policy applied before an install is published.
///
/// The policy is deliberately value-based and deterministic: it does not
/// inspect the host filesystem or perform I/O. Callers can therefore build a
/// policy from platform capabilities once and reuse it for every package.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RuntimeInstallPolicy {
    /// Capability bits supplied by the target runtime.
    pub available_capabilities: u64,
    /// Highest initial residency tier accepted by the target runtime.
    pub max_initial_residency: u8,
    /// Highest streaming tier accepted by the target runtime.
    pub max_streaming_level: u8,
    /// Codec bitset: bit `Codec as u8` enables that codec.
    pub allowed_codecs: u8,
}

impl Default for RuntimeInstallPolicy {
    fn default() -> Self {
        Self {
            available_capabilities: u64::MAX,
            max_initial_residency: u8::MAX,
            max_streaming_level: u8::MAX,
            allowed_codecs: (1 << Codec::Raw as u8) | (1 << Codec::Deflate as u8),
        }
    }
}

impl RuntimeInstallPolicy {
    /// Creates a policy accepting every capability and codec implemented by
    /// this crate.
    #[must_use]
    pub const fn permissive() -> Self {
        Self {
            available_capabilities: u64::MAX,
            max_initial_residency: u8::MAX,
            max_streaming_level: u8::MAX,
            allowed_codecs: (1 << Codec::Raw as u8) | (1 << Codec::Deflate as u8),
        }
    }

    /// Returns whether a manifest codec is accepted by this policy.
    #[must_use]
    pub const fn allows_codec(self, codec: Codec) -> bool {
        self.allowed_codecs & (1 << codec as u8) != 0
    }
}

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
    /// A route's persisted delivery metadata is incompatible with the target.
    Policy {
        /// Logical route rejected by the policy.
        guid: StableGuid,
        /// Human-readable policy violation.
        reason: String,
    },
}

impl core::fmt::Display for InstallError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::Manifest(error) => {
                write!(f, "runtime package manifest validation failed: {error}")
            }
            Self::Policy { guid, reason } => {
                write!(
                    f,
                    "runtime package route {guid:?} rejected by install policy: {reason}"
                )
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
    install_runtime_package_with_policy(
        live,
        manifest,
        mounts,
        tombstones,
        RuntimeInstallPolicy::permissive(),
    )
}

/// Installs a package atomically after applying a target runtime policy.
pub fn install_runtime_package_with_policy(
    live: &mut OverlaySet,
    manifest: &RuntimeManifest,
    mounts: impl IntoIterator<Item = InstallMount>,
    tombstones: impl IntoIterator<Item = InstallTombstones>,
    policy: RuntimeInstallPolicy,
) -> Result<(), InstallError> {
    let mut staged = live.clone();
    for mount in mounts {
        staged.mount(mount.name, mount.priority, mount.reader);
    }
    for tombstone in tombstones {
        staged.mount_tombstones(tombstone.name, tombstone.priority, tombstone.guids);
    }

    for route in manifest.routes() {
        let metadata = manifest
            .metadata(route.guid)
            .ok_or_else(|| InstallError::Manifest("route metadata is missing".into()))?;
        if metadata.install_capabilities & !policy.available_capabilities != 0 {
            return Err(InstallError::Policy {
                guid: route.guid,
                reason: format!(
                    "required capabilities 0x{:016x} exceed available 0x{:016x}",
                    metadata.install_capabilities, policy.available_capabilities
                ),
            });
        }
        if metadata.initial_residency > policy.max_initial_residency {
            return Err(InstallError::Policy {
                guid: route.guid,
                reason: format!(
                    "initial residency {} exceeds maximum {}",
                    metadata.initial_residency, policy.max_initial_residency
                ),
            });
        }
        if metadata.streaming_level > policy.max_streaming_level {
            return Err(InstallError::Policy {
                guid: route.guid,
                reason: format!(
                    "streaming level {} exceeds maximum {}",
                    metadata.streaming_level, policy.max_streaming_level
                ),
            });
        }
        if !policy.allows_codec(metadata.codec) {
            return Err(InstallError::Policy {
                guid: route.guid,
                reason: format!("codec {:?} is not enabled", metadata.codec),
            });
        }
    }

    // Binding performs every route, container identity, priority and payload
    // GUID check without mutating either the staged or live overlay.
    let _bound: ManifestReader<'_, '_> = manifest.bind(&staged)?;
    *live = staged;
    Ok(())
}
