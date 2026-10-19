//! Pure, consecutive document schema migrations.

use crate::SettingsDocument;
use prism_settings::{SettingKey, Value};
use std::{collections::BTreeMap, error::Error, fmt, sync::Arc};

/// User-provided value conversion hook for one persisted key.
pub type ValueConverter = Arc<dyn Fn(Value) -> Result<Value, String> + Send + Sync + 'static>;

/// One schema migration operation.
pub enum MigrationOperation {
    /// Renames a persisted key without changing its value.
    Rename {
        /// Previous persisted key.
        from: SettingKey,
        /// Replacement persisted key.
        to: SettingKey,
    },
    /// Drops a persisted key from the document.
    Drop {
        /// Persisted key to remove.
        key: SettingKey,
    },
    /// Converts a persisted core value using a caller-owned hook.
    Convert {
        /// Persisted key to convert.
        key: SettingKey,
        /// Pure conversion function.
        converter: ValueConverter,
    },
}

impl MigrationOperation {
    /// Creates a rename operation.
    pub fn rename(from: impl Into<SettingKey>, to: impl Into<SettingKey>) -> Self {
        Self::Rename {
            from: from.into(),
            to: to.into(),
        }
    }

    /// Creates a drop operation.
    pub fn drop(key: impl Into<SettingKey>) -> Self {
        Self::Drop { key: key.into() }
    }

    /// Creates a conversion operation from a closure.
    pub fn convert<F>(key: impl Into<SettingKey>, converter: F) -> Self
    where
        F: Fn(Value) -> Result<Value, String> + Send + Sync + 'static,
    {
        Self::Convert {
            key: key.into(),
            converter: Arc::new(converter),
        }
    }
}

impl fmt::Debug for MigrationOperation {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Rename { from, to } => formatter
                .debug_struct("Rename")
                .field("from", from)
                .field("to", to)
                .finish(),
            Self::Drop { key } => formatter.debug_struct("Drop").field("key", key).finish(),
            Self::Convert { key, .. } => formatter
                .debug_struct("Convert")
                .field("key", key)
                .field("converter", &"<hook>")
                .finish(),
        }
    }
}

/// A migration step from one schema version to the next version.
#[derive(Debug)]
struct MigrationStep {
    operations: Vec<MigrationOperation>,
}

/// A deterministic consecutive migration chain.
#[derive(Debug)]
pub struct MigrationChain {
    current_version: u32,
    steps: BTreeMap<u32, MigrationStep>,
}

impl MigrationChain {
    /// Creates a chain whose latest supported version is `current_version`.
    pub fn new(current_version: u32) -> Self {
        Self {
            current_version,
            steps: BTreeMap::new(),
        }
    }

    /// Returns the latest schema version supported by this chain.
    pub fn current_version(&self) -> u32 {
        self.current_version
    }

    /// Adds a single `from -> from + 1` migration step.
    pub fn add_step(
        &mut self,
        from: u32,
        operations: impl IntoIterator<Item = MigrationOperation>,
    ) -> Result<(), MigrationRegistrationError> {
        let to = from
            .checked_add(1)
            .ok_or(MigrationRegistrationError::VersionOverflow(from))?;
        if to > self.current_version {
            return Err(MigrationRegistrationError::BeyondCurrent { from, to });
        }
        if self
            .steps
            .insert(
                from,
                MigrationStep {
                    operations: operations.into_iter().collect(),
                },
            )
            .is_some()
        {
            return Err(MigrationRegistrationError::DuplicateStep(from));
        }
        Ok(())
    }

    /// Migrates `document` forward to the chain's current version.
    pub fn migrate(
        &self,
        document: &mut SettingsDocument,
    ) -> Result<MigrationReport, MigrationError> {
        if document.schema_version() > self.current_version {
            return Err(MigrationError::FutureVersion {
                document: document.schema_version(),
                current: self.current_version,
            });
        }

        let mut report = MigrationReport::default();
        while document.schema_version() < self.current_version {
            let from = document.schema_version();
            let step = self
                .steps
                .get(&from)
                .ok_or(MigrationError::MissingStep { from, to: from + 1 })?;
            for operation in &step.operations {
                apply_operation(document, operation, from, &mut report)?;
            }
            document.set_schema_version(from + 1);
            report.applied_steps += 1;
        }
        Ok(report)
    }
}

/// Counts the work performed by a successful migration.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct MigrationReport {
    /// Number of consecutive version steps applied.
    pub applied_steps: usize,
    /// Number of keys renamed.
    pub renamed: usize,
    /// Number of keys dropped.
    pub dropped: usize,
    /// Number of values converted.
    pub converted: usize,
}

/// Errors found while registering a migration chain.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum MigrationRegistrationError {
    /// The step would advance beyond the chain's current version.
    BeyondCurrent {
        /// Source schema version.
        from: u32,
        /// Destination schema version.
        to: u32,
    },
    /// A step for the same source version was already registered.
    DuplicateStep(u32),
    /// The source version could not be incremented.
    VersionOverflow(u32),
}

impl fmt::Display for MigrationRegistrationError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::BeyondCurrent { from, to } => {
                write!(formatter, "migration {from}->{to} exceeds current version")
            }
            Self::DuplicateStep(from) => {
                write!(formatter, "duplicate migration step from version {from}")
            }
            Self::VersionOverflow(from) => {
                write!(formatter, "migration source version {from} overflows")
            }
        }
    }
}

impl Error for MigrationRegistrationError {}

/// Errors returned when applying a migration chain.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum MigrationError {
    /// The document is newer than this binary understands.
    FutureVersion {
        /// Version found in the document.
        document: u32,
        /// Highest version supported by the chain.
        current: u32,
    },
    /// A consecutive migration step was not registered.
    MissingStep {
        /// Source schema version.
        from: u32,
        /// Required next schema version.
        to: u32,
    },
    /// An operation would overwrite another persisted key.
    KeyConflict {
        /// Migration step version.
        version: u32,
        /// Conflicting destination key.
        key: String,
    },
    /// A conversion hook rejected its input value.
    Conversion {
        /// Migration step version.
        version: u32,
        /// Converted key.
        key: String,
        /// Conversion failure explanation.
        reason: String,
    },
    /// A conversion hook targeted an unsupported preserved TOML value.
    UnsupportedPreservedValue {
        /// Migration step version.
        version: u32,
        /// Preserved key which could not be decoded for conversion.
        key: String,
    },
}

impl fmt::Display for MigrationError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::FutureVersion { document, current } => {
                write!(
                    formatter,
                    "document schema {document} is newer than current {current}"
                )
            }
            Self::MissingStep { from, to } => {
                write!(formatter, "missing migration step {from}->{to}")
            }
            Self::KeyConflict { version, key } => {
                write!(formatter, "migration {version} conflicts on `{key}`")
            }
            Self::Conversion {
                version,
                key,
                reason,
            } => write!(
                formatter,
                "migration {version} could not convert `{key}`: {reason}"
            ),
            Self::UnsupportedPreservedValue { version, key } => write!(
                formatter,
                "migration {version} cannot convert preserved TOML value `{key}`"
            ),
        }
    }
}

impl Error for MigrationError {}

fn apply_operation(
    document: &mut SettingsDocument,
    operation: &MigrationOperation,
    version: u32,
    report: &mut MigrationReport,
) -> Result<(), MigrationError> {
    match operation {
        MigrationOperation::Rename { from, to } => {
            if !document.contains_key(from.as_str()) {
                return Ok(());
            }
            if document.contains_key(to.as_str()) {
                return Err(MigrationError::KeyConflict {
                    version,
                    key: to.as_str().to_owned(),
                });
            }
            if let Some(value) = document.remove_candidate(from.as_str()) {
                document.insert_value(to.as_str().to_owned(), value);
            } else if let Some(value) = document.remove_unknown(from.as_str()) {
                document.insert_raw(to.as_str().to_owned(), value);
            }
            report.renamed += 1;
        }
        MigrationOperation::Drop { key } => {
            let removed = document.remove_candidate(key.as_str()).is_some()
                || document.remove_unknown(key.as_str()).is_some();
            report.dropped += usize::from(removed);
        }
        MigrationOperation::Convert { key, converter } => {
            if let Some(value) = document.remove_candidate(key.as_str()) {
                let converted = converter(value).map_err(|reason| MigrationError::Conversion {
                    version,
                    key: key.as_str().to_owned(),
                    reason,
                })?;
                document.insert_value(key.as_str().to_owned(), converted);
                report.converted += 1;
            } else if let Some(raw) = document.remove_unknown(key.as_str()) {
                let value = crate::document::decode_value_for_migration(&raw, key.as_str())
                    .ok_or_else(|| MigrationError::UnsupportedPreservedValue {
                        version,
                        key: key.as_str().to_owned(),
                    })?;
                let converted = converter(value).map_err(|reason| MigrationError::Conversion {
                    version,
                    key: key.as_str().to_owned(),
                    reason,
                })?;
                document.insert_value(key.as_str().to_owned(), converted);
                report.converted += 1;
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use prism_settings::Value;

    #[test]
    fn consecutive_migrations_rename_convert_and_drop() {
        let mut document = SettingsDocument::new(0);
        document.insert("audio.volume", Value::Int(50));
        document.insert("obsolete.setting", Value::Bool(true));
        let mut chain = MigrationChain::new(3);
        chain
            .add_step(
                1,
                [MigrationOperation::rename(
                    "audio.volume",
                    "audio.master_volume",
                )],
            )
            .unwrap();
        chain
            .add_step(
                2,
                [MigrationOperation::convert(
                    "audio.master_volume",
                    |value| match value {
                        Value::Int(value) => Ok(Value::Float(value as f64 / 100.0)),
                        other => Err(format!("unexpected value: {other:?}")),
                    },
                )],
            )
            .unwrap();
        chain
            .add_step(0, [MigrationOperation::drop("obsolete.setting")])
            .unwrap();
        let report = chain.migrate(&mut document).unwrap();
        assert_eq!(report.applied_steps, 3);
        assert_eq!(report.renamed, 1);
        assert_eq!(report.converted, 1);
        assert_eq!(
            document
                .candidates()
                .get(&SettingKey::from("audio.master_volume")),
            Some(&Value::Float(0.5))
        );
        assert!(!document.contains_key("obsolete.setting"));
    }

    #[test]
    fn migration_conflicts_are_transactional_at_operation_boundary() {
        let mut document = SettingsDocument::new(1);
        document.insert("old.key", Value::Bool(true));
        document.insert("new.key", Value::Bool(false));
        let mut chain = MigrationChain::new(2);
        chain
            .add_step(1, [MigrationOperation::rename("old.key", "new.key")])
            .unwrap();
        assert!(matches!(
            chain.migrate(&mut document),
            Err(MigrationError::KeyConflict { .. })
        ));
        assert!(document.contains_key("old.key"));
        assert_eq!(document.schema_version(), 1);
    }
}
