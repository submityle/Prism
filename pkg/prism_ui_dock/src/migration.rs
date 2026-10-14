//! Explicit, deterministic schema migration for persisted dock bytes.

use crate::{
    decode_layout, decode_workspace, encode_layout, encode_workspace, DecodeOptions, DockLayout,
    PersistError, WorkspaceDecodeOptions, WorkspaceManager, WorkspacePersistError,
};
use alloc::vec::Vec;

/// Canonical layout wire version emitted by this crate.
pub const CURRENT_LAYOUT_VERSION: u16 = 1;
/// Canonical workspace wire version emitted by this crate.
pub const CURRENT_WORKSPACE_VERSION: u16 = 1;

/// Failure returned by the explicit migration pipeline.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum MigrationError {
    /// The input is too short to contain the dock header.
    TruncatedHeader,
    /// The input does not contain the dock magic.
    BadMagic,
    /// No migration step was registered for this version edge.
    MissingStep {
        /// Version from which migration was requested.
        from: u16,
        /// Target canonical version.
        to: u16,
    },
    /// A migration function produced bytes that are not valid canonical data.
    InvalidOutput(PersistError),
    /// A workspace migration function produced invalid canonical data.
    InvalidWorkspaceOutput(WorkspacePersistError),
    /// The migration graph contained a non-forward or duplicate edge.
    InvalidStep {
        /// Source version of the invalid edge.
        from: u16,
        /// Destination version of the invalid edge.
        to: u16,
    },
}

/// A single explicit version edge.
#[derive(Clone, Copy)]
pub struct MigrationStep {
    /// Version accepted by the step.
    pub from: u16,
    /// Version emitted by the step.
    pub to: u16,
    /// Pure bytes-to-bytes conversion.
    pub migrate: fn(&[u8]) -> Result<Vec<u8>, MigrationError>,
}

/// Ordered migration registry.
#[derive(Clone, Default)]
pub struct MigrationRegistry {
    steps: Vec<MigrationStep>,
}

impl MigrationRegistry {
    /// Creates an empty registry. Version 1 needs no migration step.
    pub const fn new() -> Self {
        Self { steps: Vec::new() }
    }

    /// Adds one forward version edge, rejecting duplicates and self-edges.
    pub fn register(&mut self, step: MigrationStep) -> Result<(), MigrationError> {
        if step.from >= step.to
            || self
                .steps
                .iter()
                .any(|registered| registered.from == step.from)
        {
            return Err(MigrationError::InvalidStep {
                from: step.from,
                to: step.to,
            });
        }
        self.steps.push(step);
        self.steps.sort_by_key(|registered| registered.from);
        Ok(())
    }

    /// Returns the number of registered edges.
    pub fn len(&self) -> usize {
        self.steps.len()
    }

    /// Returns whether no migration edges are registered.
    pub fn is_empty(&self) -> bool {
        self.steps.is_empty()
    }

    /// Migrates bytes to the current version and validates the result.
    pub fn migrate(&self, bytes: &[u8]) -> Result<Vec<u8>, MigrationError> {
        let mut version = read_version(bytes)?;
        if version > CURRENT_LAYOUT_VERSION {
            return Err(MigrationError::MissingStep {
                from: version,
                to: CURRENT_LAYOUT_VERSION,
            });
        }
        let mut current = bytes.to_vec();
        while version < CURRENT_LAYOUT_VERSION {
            let step = self
                .steps
                .iter()
                .find(|candidate| candidate.from == version)
                .ok_or(MigrationError::MissingStep {
                    from: version,
                    to: CURRENT_LAYOUT_VERSION,
                })?;
            current = (step.migrate)(&current)?;
            let migrated_version = read_version(&current)?;
            if migrated_version != step.to {
                return Err(MigrationError::InvalidOutput(
                    PersistError::UnsupportedVersion(migrated_version),
                ));
            }
            version = step.to;
        }
        decode_layout(&current, DecodeOptions::default())
            .map_err(MigrationError::InvalidOutput)
            .map(|_| current)
    }

    /// Migrates a typed layout through a caller-supplied version path.
    ///
    /// This is useful for tools that hold a decoded legacy model and want the
    /// same canonical validation and encoder as file loading.
    pub fn encode_current(&self, layout: &DockLayout) -> Result<Vec<u8>, MigrationError> {
        encode_layout(layout).map_err(MigrationError::InvalidOutput)
    }
}

/// Explicit migration registry for `PRWS` Workspace snapshots.
#[derive(Clone, Default)]
pub struct WorkspaceMigrationRegistry {
    steps: Vec<MigrationStep>,
}

impl WorkspaceMigrationRegistry {
    /// Creates an empty registry.
    pub const fn new() -> Self {
        Self { steps: Vec::new() }
    }

    /// Registers one forward workspace version edge.
    pub fn register(&mut self, step: MigrationStep) -> Result<(), MigrationError> {
        if step.from >= step.to
            || self
                .steps
                .iter()
                .any(|registered| registered.from == step.from)
        {
            return Err(MigrationError::InvalidStep {
                from: step.from,
                to: step.to,
            });
        }
        self.steps.push(step);
        self.steps.sort_by_key(|registered| registered.from);
        Ok(())
    }

    /// Migrates and validates a workspace snapshot to the canonical version.
    pub fn migrate(&self, bytes: &[u8]) -> Result<Vec<u8>, MigrationError> {
        let mut version = read_magic_version(bytes, b"PRWS")?;
        let mut current = bytes.to_vec();
        while version < CURRENT_WORKSPACE_VERSION {
            let step = self
                .steps
                .iter()
                .find(|candidate| candidate.from == version)
                .ok_or(MigrationError::MissingStep {
                    from: version,
                    to: CURRENT_WORKSPACE_VERSION,
                })?;
            current = (step.migrate)(&current)?;
            version = read_magic_version(&current, b"PRWS")?;
            if version != step.to {
                return Err(MigrationError::InvalidWorkspaceOutput(
                    WorkspacePersistError::UnsupportedVersion(version),
                ));
            }
        }
        decode_workspace(&current, WorkspaceDecodeOptions::default())
            .map_err(MigrationError::InvalidWorkspaceOutput)
            .map(|_| current)
    }

    /// Encodes a validated manager with the current workspace codec.
    pub fn encode_current(&self, manager: &WorkspaceManager) -> Result<Vec<u8>, MigrationError> {
        encode_workspace(manager).map_err(MigrationError::InvalidWorkspaceOutput)
    }
}

fn read_version(bytes: &[u8]) -> Result<u16, MigrationError> {
    read_magic_version(bytes, b"PRDK")
}

fn read_magic_version(bytes: &[u8], magic: &[u8; 4]) -> Result<u16, MigrationError> {
    if bytes.len() < 6 {
        return Err(MigrationError::TruncatedHeader);
    }
    if &bytes[..4] != magic {
        return Err(MigrationError::BadMagic);
    }
    Ok(u16::from_le_bytes([bytes[4], bytes[5]]))
}

/// A helper for a migration that only changes the two-byte version field.
///
/// The caller remains responsible for ensuring the payload is semantically
/// compatible; the final canonical decode is always performed by the registry.
pub fn rewrite_version(bytes: &[u8], version: u16) -> Result<Vec<u8>, MigrationError> {
    rewrite_magic_version(bytes, b"PRDK", version)
}

/// Rewrites a Workspace header version for a registered migration step.
pub fn rewrite_workspace_version(bytes: &[u8], version: u16) -> Result<Vec<u8>, MigrationError> {
    rewrite_magic_version(bytes, b"PRWS", version)
}

fn rewrite_magic_version(
    bytes: &[u8],
    magic: &[u8; 4],
    version: u16,
) -> Result<Vec<u8>, MigrationError> {
    let mut output = bytes.to_vec();
    if output.len() < 6 || &output[..4] != magic {
        return Err(MigrationError::BadMagic);
    }
    output[4..6].copy_from_slice(&version.to_le_bytes());
    Ok(output)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{encode_layout, encode_workspace, DockLayout, DockNode};

    fn v0_to_v1(bytes: &[u8]) -> Result<Vec<u8>, MigrationError> {
        rewrite_version(bytes, CURRENT_LAYOUT_VERSION)
    }

    #[test]
    fn current_bytes_are_valid_without_steps() {
        let bytes = encode_layout(&DockLayout {
            root: DockNode::panel("panel"),
            ..DockLayout::default()
        })
        .unwrap();
        assert_eq!(MigrationRegistry::new().migrate(&bytes).unwrap(), bytes);
    }

    #[test]
    fn legacy_step_is_explicit_and_validated() {
        let canonical = encode_layout(&DockLayout {
            root: DockNode::panel("panel"),
            ..DockLayout::default()
        })
        .unwrap();
        let legacy = rewrite_version(&canonical, 0).unwrap();
        let mut registry = MigrationRegistry::new();
        registry
            .register(MigrationStep {
                from: 0,
                to: 1,
                migrate: v0_to_v1,
            })
            .unwrap();
        assert_eq!(registry.migrate(&legacy).unwrap(), canonical);
    }

    #[test]
    fn missing_legacy_edge_is_reported() {
        let canonical = encode_layout(&DockLayout::default()).unwrap();
        let legacy = rewrite_version(&canonical, 0).unwrap();
        assert_eq!(
            MigrationRegistry::new().migrate(&legacy),
            Err(MigrationError::MissingStep { from: 0, to: 1 })
        );
    }

    #[test]
    fn workspace_legacy_step_is_explicit_and_validated() {
        let mut manager = WorkspaceManager::new();
        manager.add_workspace("main").unwrap();
        manager
            .add_perspective(
                "default",
                &DockLayout {
                    root: DockNode::panel("panel"),
                    ..DockLayout::default()
                },
            )
            .unwrap();
        let canonical = encode_workspace(&manager).unwrap();
        let legacy = rewrite_workspace_version(&canonical, 0).unwrap();
        let mut registry = WorkspaceMigrationRegistry::new();
        registry
            .register(MigrationStep {
                from: 0,
                to: CURRENT_WORKSPACE_VERSION,
                migrate: |bytes| rewrite_workspace_version(bytes, CURRENT_WORKSPACE_VERSION),
            })
            .unwrap();
        assert_eq!(registry.migrate(&legacy).unwrap(), canonical);
    }
}
