//! Transactional installation of runtime container overlays.
//!
//! Installation is staged against a clone of the current [`OverlaySet`].
//! Container readers are already strictly validated when created; this layer
//! validates the manifest-to-overlay binding before publishing the new
//! snapshot. A failed install therefore cannot expose a partially mounted
//! package to readers.

use alloc::string::String;
use alloc::vec::Vec;
use core::fmt;

use prism_asset::StableGuid;

use crate::{
    Codec, ContainerIdentity, ContainerReader, ManifestReadError, ManifestReader, OverlaySet,
    RuntimeManifest,
};

/// Maximum opaque provenance input accepted by a recoverable install.
pub const MAX_RUNTIME_PROVENANCE_BYTES: usize = 64 * 1024;
/// Maximum encoded recovery record size.
pub const MAX_RUNTIME_RECOVERY_BYTES: usize = 4 * 1024 * 1024;
const RECOVERY_MAGIC: [u8; 8] = *b"PRSRBACK";
const RECOVERY_VERSION: u16 = 1;

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
            allowed_codecs: (1 << Codec::Raw as u8)
                | (1 << Codec::Deflate as u8)
                | (1 << Codec::BlockDeflate as u8),
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
            allowed_codecs: (1 << Codec::Raw as u8)
                | (1 << Codec::Deflate as u8)
                | (1 << Codec::BlockDeflate as u8),
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
    /// The recoverable package provenance is empty or exceeds its bound.
    ProvenanceOutOfBounds,
    /// The canonical manifest or recovery binding could not be constructed.
    Binding(String),
}

impl fmt::Display for InstallError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
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
            Self::ProvenanceOutOfBounds => {
                f.write_str("runtime package provenance is empty or exceeds its bound")
            }
            Self::Binding(error) => write!(f, "runtime package binding failed: {error}"),
        }
    }
}

impl std::error::Error for InstallError {}

impl From<ManifestReadError> for InstallError {
    fn from(error: ManifestReadError) -> Self {
        Self::Manifest(error.to_string())
    }
}

/// Opaque caller-owned provenance bound to a runtime package transaction.
///
/// This type stores only a fixed-size digest. The container crate does not
/// interpret, sign, encrypt, publish, or authenticate provenance bytes; the
/// producer remains responsible for their meaning and trust policy.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RuntimePackageProvenance {
    digest: [u8; 32],
}

impl RuntimePackageProvenance {
    /// Hashes bounded, non-empty canonical provenance bytes.
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, InstallError> {
        if bytes.is_empty() || bytes.len() > MAX_RUNTIME_PROVENANCE_BYTES {
            return Err(InstallError::ProvenanceOutOfBounds);
        }
        Ok(Self {
            digest: *blake3::hash(bytes).as_bytes(),
        })
    }

    /// Creates a provenance binding from a caller-owned digest.
    #[must_use]
    pub const fn from_digest(digest: [u8; 32]) -> Self {
        Self { digest }
    }

    /// Returns the provenance digest stored in the package binding.
    #[must_use]
    pub const fn digest(self) -> [u8; 32] {
        self.digest
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct RuntimePackageBinding {
    manifest_hash: [u8; 32],
    containers: Vec<(String, ContainerIdentity)>,
    provenance_hash: [u8; 32],
}

impl RuntimePackageBinding {
    fn new(
        manifest: &RuntimeManifest,
        provenance: RuntimePackageProvenance,
    ) -> Result<Self, InstallError> {
        let manifest_bytes = manifest
            .encode()
            .map_err(|error| InstallError::Binding(error.to_string()))?;
        let containers = manifest
            .routes()
            .map(|route| route.container_name.as_str())
            .collect::<alloc::collections::BTreeSet<_>>()
            .into_iter()
            .map(|name| {
                manifest
                    .container_identity(name)
                    .map(|identity| (name.to_owned(), identity))
                    .ok_or_else(|| InstallError::Binding(format!("missing identity for {name:?}")))
            })
            .collect::<Result<Vec<_>, _>>()?;
        Ok(Self {
            manifest_hash: *blake3::hash(&manifest_bytes).as_bytes(),
            containers,
            provenance_hash: provenance.digest(),
        })
    }

    fn verify_manifest_and_overlay(
        &self,
        manifest: &RuntimeManifest,
        overlays: &OverlaySet,
    ) -> Result<(), InstallError> {
        let actual = Self::new(
            manifest,
            RuntimePackageProvenance::from_digest(self.provenance_hash),
        )?;
        if actual.manifest_hash != self.manifest_hash || actual.containers != self.containers {
            return Err(InstallError::Binding(
                "manifest/container identity binding mismatch".into(),
            ));
        }
        manifest.bind(overlays)?;
        Ok(())
    }
}

/// A bounded, caller-persistable record for a pending package rollback.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RuntimeInstallRecovery {
    binding: RuntimePackageBinding,
    previous_overlay_fingerprint: [u8; 32],
    installed_overlay_fingerprint: [u8; 32],
}

impl RuntimeInstallRecovery {
    /// Returns the manifest content hash bound to this recovery record.
    #[must_use]
    pub const fn manifest_hash(&self) -> [u8; 32] {
        self.binding.manifest_hash
    }

    /// Returns the opaque provenance digest bound to this recovery record.
    #[must_use]
    pub const fn provenance_hash(&self) -> [u8; 32] {
        self.binding.provenance_hash
    }

    /// Encodes a canonical, bounded local recovery record.
    pub fn encode(&self) -> Result<Vec<u8>, RecoveryError> {
        let count = u32::try_from(self.binding.containers.len())
            .map_err(|_| RecoveryError::InvalidFormat)?;
        let mut bytes = Vec::new();
        bytes.extend_from_slice(&RECOVERY_MAGIC);
        bytes.extend_from_slice(&RECOVERY_VERSION.to_le_bytes());
        bytes.extend_from_slice(&0_u16.to_le_bytes());
        bytes.extend_from_slice(&self.binding.manifest_hash);
        bytes.extend_from_slice(&self.binding.provenance_hash);
        bytes.extend_from_slice(&self.previous_overlay_fingerprint);
        bytes.extend_from_slice(&self.installed_overlay_fingerprint);
        bytes.extend_from_slice(&count.to_le_bytes());
        for (name, identity) in &self.binding.containers {
            let name_len = u16::try_from(name.len()).map_err(|_| RecoveryError::InvalidFormat)?;
            bytes.extend_from_slice(&name_len.to_le_bytes());
            bytes.extend_from_slice(&identity.byte_len.to_le_bytes());
            bytes.extend_from_slice(&identity.content_hash);
            bytes.extend_from_slice(&identity.generation.to_le_bytes());
            bytes.extend_from_slice(name.as_bytes());
        }
        let digest = blake3::hash(&bytes);
        bytes.extend_from_slice(digest.as_bytes());
        if bytes.len() > MAX_RUNTIME_RECOVERY_BYTES {
            return Err(RecoveryError::TooLarge);
        }
        Ok(bytes)
    }

    /// Decodes and strictly validates a local recovery record.
    pub fn decode(bytes: &[u8]) -> Result<Self, RecoveryError> {
        const FIXED: usize = 8 + 2 + 2 + (32 * 4) + 4;
        if bytes.len() > MAX_RUNTIME_RECOVERY_BYTES || bytes.len() < FIXED + 32 {
            return Err(RecoveryError::Truncated);
        }
        if bytes[..8] != RECOVERY_MAGIC {
            return Err(RecoveryError::BadMagic);
        }
        if u16::from_le_bytes(bytes[8..10].try_into().unwrap()) != RECOVERY_VERSION
            || bytes[10..12] != [0, 0]
        {
            return Err(RecoveryError::InvalidFormat);
        }
        let body_end = bytes.len() - 32;
        if blake3::hash(&bytes[..body_end]).as_bytes() != &bytes[body_end..] {
            return Err(RecoveryError::HashMismatch);
        }
        let count = u32::from_le_bytes(bytes[FIXED - 4..FIXED].try_into().unwrap()) as usize;
        let mut cursor = FIXED;
        let mut containers = Vec::with_capacity(count);
        let mut previous_name: Option<String> = None;
        for _ in 0..count {
            let fixed_end = cursor
                .checked_add(2 + 8 + 32 + 8)
                .ok_or(RecoveryError::InvalidFormat)?;
            if fixed_end > body_end {
                return Err(RecoveryError::Truncated);
            }
            let name_len =
                u16::from_le_bytes(bytes[cursor..cursor + 2].try_into().unwrap()) as usize;
            let byte_len = u64::from_le_bytes(bytes[cursor + 2..cursor + 10].try_into().unwrap());
            let content_hash = bytes[cursor + 10..cursor + 42]
                .try_into()
                .map_err(|_| RecoveryError::InvalidFormat)?;
            let generation =
                u64::from_le_bytes(bytes[cursor + 42..cursor + 50].try_into().unwrap());
            cursor = fixed_end;
            let name_end = cursor
                .checked_add(name_len)
                .ok_or(RecoveryError::InvalidFormat)?;
            if name_end > body_end {
                return Err(RecoveryError::Truncated);
            }
            let name = core::str::from_utf8(&bytes[cursor..name_end])
                .map_err(|_| RecoveryError::InvalidFormat)?
                .to_owned();
            if previous_name
                .as_ref()
                .is_some_and(|previous| previous >= &name)
            {
                return Err(RecoveryError::InvalidFormat);
            }
            previous_name = Some(name.clone());
            containers.push((
                name,
                ContainerIdentity {
                    byte_len,
                    content_hash,
                    generation,
                },
            ));
            cursor = name_end;
        }
        if cursor != body_end {
            return Err(RecoveryError::InvalidFormat);
        }
        Ok(Self {
            binding: RuntimePackageBinding {
                manifest_hash: bytes[12..44].try_into().unwrap(),
                provenance_hash: bytes[44..76].try_into().unwrap(),
                containers,
            },
            previous_overlay_fingerprint: bytes[76..108].try_into().unwrap(),
            installed_overlay_fingerprint: bytes[108..140].try_into().unwrap(),
        })
    }

    /// Recovers a pending install after restart.
    ///
    /// The caller must reopen the previous overlay and provide the exact
    /// manifest and provenance bytes. Any mismatch leaves `live` untouched.
    pub fn recover_rollback(
        &self,
        live: &mut OverlaySet,
        previous: OverlaySet,
        manifest: &RuntimeManifest,
        provenance: &[u8],
    ) -> Result<(), RecoveryError> {
        let provenance = RuntimePackageProvenance::from_bytes(provenance)
            .map_err(|_| RecoveryError::ProvenanceMismatch)?;
        let expected = RuntimePackageBinding::new(manifest, provenance)
            .map_err(|_| RecoveryError::BindingMismatch)?;
        if expected != self.binding
            || live.install_fingerprint() != self.installed_overlay_fingerprint
            || previous.install_fingerprint() != self.previous_overlay_fingerprint
        {
            return Err(RecoveryError::BindingMismatch);
        }
        manifest
            .bind(live)
            .map_err(|_| RecoveryError::BindingMismatch)?;
        *live = previous;
        Ok(())
    }
}

/// Errors raised while encoding or applying a restart recovery record.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum RecoveryError {
    /// The record is too large for the bounded local format.
    TooLarge,
    /// The record ended before a complete field was available.
    Truncated,
    /// The record discriminator is invalid.
    BadMagic,
    /// The record fields or canonical ordering are invalid.
    InvalidFormat,
    /// The record digest does not match its body.
    HashMismatch,
    /// The caller supplied different manifest/container/provenance facts.
    BindingMismatch,
    /// The caller supplied invalid provenance bytes.
    ProvenanceMismatch,
}

impl fmt::Display for RecoveryError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::TooLarge => "runtime install recovery record is too large",
            Self::Truncated => "runtime install recovery record is truncated",
            Self::BadMagic => "runtime install recovery record has invalid magic",
            Self::InvalidFormat => "runtime install recovery record has invalid format",
            Self::HashMismatch => "runtime install recovery record hash mismatch",
            Self::BindingMismatch => "runtime install recovery binding mismatch",
            Self::ProvenanceMismatch => "runtime install recovery provenance mismatch",
        })
    }
}

impl std::error::Error for RecoveryError {}

/// A successful package install retaining the previous overlay for rollback.
#[derive(Debug)]
pub struct RuntimeInstallHandle {
    recovery: RuntimeInstallRecovery,
    manifest: RuntimeManifest,
    previous: Option<OverlaySet>,
}

impl RuntimeInstallHandle {
    /// Returns a cloneable record that can be persisted before acknowledging
    /// the installation as committed.
    #[must_use]
    pub fn recovery_record(&self) -> RuntimeInstallRecovery {
        self.recovery.clone()
    }

    /// Rolls back the complete package installation exactly once.
    pub fn rollback(&mut self, live: &mut OverlaySet) -> Result<(), RecoveryError> {
        if live.install_fingerprint() != self.recovery.installed_overlay_fingerprint {
            return Err(RecoveryError::BindingMismatch);
        }
        let previous = self.previous.take().ok_or(RecoveryError::BindingMismatch)?;
        if previous.install_fingerprint() != self.recovery.previous_overlay_fingerprint {
            self.previous = Some(previous);
            return Err(RecoveryError::BindingMismatch);
        }
        self.manifest
            .bind(live)
            .map_err(|_| RecoveryError::BindingMismatch)?;
        *live = previous;
        Ok(())
    }

    /// Commits the install and releases its in-memory rollback snapshot.
    pub fn commit(self) {}
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
    install_runtime_package_transaction_with_policy(
        live,
        manifest,
        mounts,
        tombstones,
        policy,
        RuntimePackageProvenance::from_bytes(b"legacy-runtime-install")?,
    )?
    .commit();
    Ok(())
}

/// Installs a package and returns a handle that can rollback the whole package.
pub fn install_runtime_package_transaction(
    live: &mut OverlaySet,
    manifest: &RuntimeManifest,
    mounts: impl IntoIterator<Item = InstallMount>,
    tombstones: impl IntoIterator<Item = InstallTombstones>,
    provenance: &[u8],
) -> Result<RuntimeInstallHandle, InstallError> {
    install_runtime_package_transaction_with_policy(
        live,
        manifest,
        mounts,
        tombstones,
        RuntimeInstallPolicy::permissive(),
        RuntimePackageProvenance::from_bytes(provenance)?,
    )
}

/// Policy-aware package install returning a rollback/restart recovery handle.
pub fn install_runtime_package_transaction_with_policy(
    live: &mut OverlaySet,
    manifest: &RuntimeManifest,
    mounts: impl IntoIterator<Item = InstallMount>,
    tombstones: impl IntoIterator<Item = InstallTombstones>,
    policy: RuntimeInstallPolicy,
    provenance: RuntimePackageProvenance,
) -> Result<RuntimeInstallHandle, InstallError> {
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
    let binding = RuntimePackageBinding::new(manifest, provenance)?;
    binding.verify_manifest_and_overlay(manifest, &staged)?;
    let previous = live.clone();
    let recovery = RuntimeInstallRecovery {
        binding,
        previous_overlay_fingerprint: live.install_fingerprint(),
        installed_overlay_fingerprint: staged.install_fingerprint(),
    };
    *live = staged;
    Ok(RuntimeInstallHandle {
        recovery,
        manifest: manifest.clone(),
        previous: Some(previous),
    })
}
