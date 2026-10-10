//! Ordered schema migration steps and the errors they raise (design §10, §22).
//!
//! A [`Migration`] upgrades a decoded [`DynamicStruct`] payload from one schema
//! version to the next (`vN -> vN+1`). Steps are registered in a
//! [`SchemaRegistry`](crate::schema::SchemaRegistry) and composed into a chain
//! so an old serialized payload can be walked forward, one version at a time,
//! to the current shape before [`FromReflect`](crate::FromReflect) rebuilds the
//! concrete value. Each step mutates the dynamic struct in place: adding new
//! fields with defaults, renaming/dropping fields, or transforming values.

use crate::dynamic::DynamicStruct;
use crate::ser::{DeserializeError, SerializeError};
use alloc::boxed::Box;
use alloc::string::String;

/// The transform run by a single migration step.
///
/// It receives the decoded payload for its *source* version and must leave it
/// shaped like the *target* version (so the next step, or final
/// reconstruction, can consume it).
pub type MigrationFn = Box<dyn Fn(&mut DynamicStruct) -> Result<(), MigrateError> + Send + Sync>;

/// A single ordered schema migration step (`from_version -> to_version`).
///
/// `to_version` must equal `from_version + 1`; multi-version upgrades are
/// expressed as a chain of consecutive steps, not one step that skips versions.
pub struct Migration {
    from_version: u32,
    to_version: u32,
    step: MigrationFn,
}

impl Migration {
    /// Build a step from `from_version` to `from_version + 1`.
    ///
    /// # Panics
    /// Panics if `to_version != from_version + 1`; migrations are strictly
    /// single-version so a chain is unambiguous.
    #[must_use]
    pub fn new<F>(from_version: u32, to_version: u32, step: F) -> Self
    where
        F: Fn(&mut DynamicStruct) -> Result<(), MigrateError> + Send + Sync + 'static,
    {
        assert!(
            to_version == from_version + 1,
            "a migration must advance exactly one version (from {from_version} to {to_version})"
        );
        Self {
            from_version,
            to_version,
            step: Box::new(step),
        }
    }

    /// The version this step upgrades *from*.
    #[must_use]
    pub fn from_version(&self) -> u32 {
        self.from_version
    }

    /// The version this step upgrades *to*.
    #[must_use]
    pub fn to_version(&self) -> u32 {
        self.to_version
    }

    /// Run this step against a decoded payload.
    ///
    /// # Errors
    /// Returns whatever [`MigrateError`] the step reports (for example a
    /// missing source field it needed to transform).
    pub fn apply(&self, value: &mut DynamicStruct) -> Result<(), MigrateError> {
        (self.step)(value)
    }
}

impl ::core::fmt::Debug for Migration {
    fn fmt(&self, f: &mut ::core::fmt::Formatter<'_>) -> ::core::fmt::Result {
        f.debug_struct("Migration")
            .field("from_version", &self.from_version)
            .field("to_version", &self.to_version)
            .finish_non_exhaustive()
    }
}

/// An error raised while registering, composing, or running a migration chain,
/// including the serializer/deserializer failures surfaced by the versioned
/// (de)serialization helpers.
#[derive(Debug)]
#[non_exhaustive]
pub enum MigrateError {
    /// The logical type name has no schema registered.
    UnknownType(String),
    /// A migration was registered that does not advance exactly one version.
    NonContiguous {
        /// The step's `from_version`.
        from: u32,
        /// The step's `to_version`.
        to: u32,
    },
    /// No migration step covers version `from` on the way to `current`.
    NoMigrationPath {
        /// The first version that lacks an outgoing migration.
        from: u32,
        /// The current (target) version.
        current: u32,
    },
    /// The payload's recorded version is newer than the current schema.
    VersionTooNew {
        /// The version read from the payload.
        found: u32,
        /// The current (newest registered) version.
        current: u32,
    },
    /// No shape is registered for the payload's recorded version.
    UnknownVersion {
        /// The version read from the payload.
        found: u32,
    },
    /// The decoded payload was not a struct, so it cannot be migrated.
    NotAStruct,
    /// A migration step needed a field that was absent from the payload.
    MissingField(String),
    /// A migration step failed for a reason it describes.
    Step(String),
    /// The underlying reflection serializer failed.
    Serialize(SerializeError),
    /// The underlying reflection deserializer failed.
    Deserialize(DeserializeError),
    /// The versioned envelope header was malformed.
    BadEnvelope,
}

impl ::core::fmt::Display for MigrateError {
    fn fmt(&self, f: &mut ::core::fmt::Formatter<'_>) -> ::core::fmt::Result {
        match self {
            MigrateError::UnknownType(name) => {
                write!(f, "no schema registered for type `{name}`")
            }
            MigrateError::NonContiguous { from, to } => write!(
                f,
                "migration from v{from} to v{to} is not contiguous (must advance one version)"
            ),
            MigrateError::NoMigrationPath { from, current } => {
                write!(
                    f,
                    "no migration step from v{from} toward current v{current}"
                )
            }
            MigrateError::VersionTooNew { found, current } => write!(
                f,
                "payload version v{found} is newer than current schema v{current}"
            ),
            MigrateError::UnknownVersion { found } => {
                write!(f, "no schema shape registered for payload version v{found}")
            }
            MigrateError::NotAStruct => {
                write!(f, "only struct payloads can be migrated")
            }
            MigrateError::MissingField(name) => {
                write!(f, "migration needed absent field `{name}`")
            }
            MigrateError::Step(msg) => write!(f, "migration step failed: {msg}"),
            MigrateError::Serialize(err) => write!(f, "serialize error: {err}"),
            MigrateError::Deserialize(err) => write!(f, "deserialize error: {err}"),
            MigrateError::BadEnvelope => write!(f, "malformed versioned payload envelope"),
        }
    }
}

impl ::std::error::Error for MigrateError {}

impl From<SerializeError> for MigrateError {
    fn from(err: SerializeError) -> Self {
        MigrateError::Serialize(err)
    }
}

impl From<DeserializeError> for MigrateError {
    fn from(err: DeserializeError) -> Self {
        MigrateError::Deserialize(err)
    }
}
