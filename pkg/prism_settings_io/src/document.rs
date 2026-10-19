//! The versioned TOML document model and codec.

use prism_settings::{SettingKey, SettingsRegistry, Value};
use std::{collections::BTreeMap, error::Error, fmt};

const SCHEMA_KEY: &str = "prism_schema";
const VALUES_KEY: &str = "values";
const TAG_KEY: &str = "__prism_value";

/// A versioned collection of persisted setting candidates.
///
/// `candidates` contains values understood by the core value model. `unknown`
/// contains TOML values for keys absent from the active registry, including
/// TOML types which the core does not represent. Both maps are ordered so the
/// encoded document is deterministic.
#[derive(Clone, Debug, PartialEq)]
pub struct SettingsDocument {
    schema_version: u32,
    candidates: BTreeMap<SettingKey, Value>,
    unknown: BTreeMap<String, toml::Value>,
}

impl SettingsDocument {
    /// Creates an empty document at `schema_version`.
    pub fn new(schema_version: u32) -> Self {
        Self {
            schema_version,
            candidates: BTreeMap::new(),
            unknown: BTreeMap::new(),
        }
    }

    /// Returns the persisted schema version.
    pub fn schema_version(&self) -> u32 {
        self.schema_version
    }

    /// Changes the persisted schema version after a successful migration step.
    pub(crate) fn set_schema_version(&mut self, version: u32) {
        self.schema_version = version;
    }

    /// Returns all core-compatible candidates in canonical key order.
    pub fn candidates(&self) -> &BTreeMap<SettingKey, Value> {
        &self.candidates
    }

    /// Returns all preserved unknown keys in canonical key order.
    pub fn unknown(&self) -> &BTreeMap<String, toml::Value> {
        &self.unknown
    }

    /// Inserts or replaces a core-compatible candidate.
    ///
    /// Inserting a key removes a previously preserved unknown value with the
    /// same spelling, so a document cannot encode the key twice.
    pub fn insert(&mut self, key: impl Into<SettingKey>, value: Value) {
        let key = key.into();
        self.unknown.remove(key.as_str());
        self.candidates.insert(key, value);
    }

    /// Preserves an unknown TOML key/value pair.
    ///
    /// A key already present as a candidate is replaced by the preserved value.
    pub fn preserve_unknown(&mut self, key: impl Into<String>, value: toml::Value) {
        let key = key.into();
        self.candidates.remove(&SettingKey::new(key.clone()));
        self.unknown.insert(key, value);
    }

    /// Encodes this document as canonical TOML.
    pub fn encode(&self) -> Result<String, CodecError> {
        let mut values = toml::Table::new();
        for (key, value) in &self.candidates {
            if self.unknown.contains_key(key.as_str()) {
                return Err(CodecError::DuplicateKey(key.as_str().to_owned()));
            }
            values.insert(key.as_str().to_owned(), encode_value(value));
        }
        for (key, value) in &self.unknown {
            if self.candidates.contains_key(&SettingKey::new(key.clone())) {
                return Err(CodecError::DuplicateKey(key.clone()));
            }
            values.insert(key.clone(), value.clone());
        }

        let mut root = toml::Table::new();
        root.insert(
            SCHEMA_KEY.to_owned(),
            toml::Value::Integer(self.schema_version as i64),
        );
        root.insert(VALUES_KEY.to_owned(), toml::Value::Table(values));
        toml::to_string(&root).map_err(|error| CodecError::Toml(error.to_string()))
    }

    /// Decodes canonical or hand-written TOML without applying a registry.
    ///
    /// Core-compatible values become candidates. Unsupported TOML values are
    /// preserved as unknown values so an older binary can still round-trip
    /// data introduced by a newer binary.
    pub fn decode(input: &str) -> Result<Self, CodecError> {
        Self::decode_with_registry(input, None)
    }

    /// Decodes TOML and classifies keys against `registry`.
    ///
    /// Keys absent from the registry are retained verbatim in [`Self::unknown`]
    /// and do not participate in core validation. Known keys must decode to a
    /// value accepted by the registry.
    pub fn decode_with_registry(
        input: &str,
        registry: Option<&SettingsRegistry>,
    ) -> Result<Self, CodecError> {
        let root: toml::Table =
            toml::from_str(input).map_err(|error| CodecError::Toml(error.to_string()))?;
        let schema_version = root
            .get(SCHEMA_KEY)
            .and_then(toml::Value::as_integer)
            .ok_or(CodecError::MissingSchemaVersion)?;
        let schema_version = u32::try_from(schema_version)
            .map_err(|_| CodecError::InvalidSchemaVersion(schema_version))?;
        let values = root
            .get(VALUES_KEY)
            .and_then(toml::Value::as_table)
            .ok_or(CodecError::ValuesNotTable)?;

        let mut document = Self::new(schema_version);
        for (key, raw) in values {
            let known = registry.and_then(|registry| registry.get(&SettingKey::new(key.clone())));
            match decode_value(raw, key) {
                Ok(value) if known.is_some() || registry.is_none() => {
                    if let Some((id, _)) = known
                        && let Some(registry) = registry
                    {
                        registry
                            .validate_value(id, value.clone())
                            .map_err(|error| CodecError::InvalidKnownValue {
                                key: key.clone(),
                                reason: error.to_string(),
                            })?;
                    }
                    document.insert(key.clone(), value);
                }
                Ok(_) => document.preserve_unknown(key.clone(), raw.clone()),
                Err(_) if known.is_none() => document.preserve_unknown(key.clone(), raw.clone()),
                Err(reason) => {
                    return Err(CodecError::InvalidValue {
                        key: key.clone(),
                        reason,
                    });
                }
            }
        }
        Ok(document)
    }

    /// Removes and returns a candidate, if one exists.
    pub(crate) fn remove_candidate(&mut self, key: &str) -> Option<Value> {
        self.candidates.remove(&SettingKey::new(key.to_owned()))
    }

    /// Removes and returns a preserved TOML value, if one exists.
    pub(crate) fn remove_unknown(&mut self, key: &str) -> Option<toml::Value> {
        self.unknown.remove(key)
    }

    /// Returns whether either representation contains `key`.
    pub(crate) fn contains_key(&self, key: &str) -> bool {
        self.candidates
            .contains_key(&SettingKey::new(key.to_owned()))
            || self.unknown.contains_key(key)
    }

    /// Re-inserts a value after a migration operation.
    pub(crate) fn insert_value(&mut self, key: String, value: Value) {
        self.insert(key, value);
    }

    /// Re-inserts a preserved TOML value after a migration operation.
    pub(crate) fn insert_raw(&mut self, key: String, value: toml::Value) {
        self.preserve_unknown(key, value);
    }
}

/// Errors produced by the versioned TOML codec.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum CodecError {
    /// The TOML parser rejected the input.
    Toml(String),
    /// The root TOML value was not a table.
    RootNotTable,
    /// The required `prism_schema` field was absent or not an integer.
    MissingSchemaVersion,
    /// The schema version could not fit in `u32`.
    InvalidSchemaVersion(i64),
    /// The required `values` table was absent or had another type.
    ValuesNotTable,
    /// A core value could not be encoded or decoded for the named key.
    InvalidValue {
        /// Persisted key which failed decoding.
        key: String,
        /// Decoder explanation.
        reason: String,
    },
    /// A known key decoded but violated its registry definition.
    InvalidKnownValue {
        /// Known schema key which failed validation.
        key: String,
        /// Registry validation explanation.
        reason: String,
    },
    /// A document was asked to encode one key twice.
    DuplicateKey(String),
}

impl fmt::Display for CodecError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Toml(error) => write!(formatter, "TOML parse error: {error}"),
            Self::RootNotTable => {
                formatter.write_str("settings document root must be a TOML table")
            }
            Self::MissingSchemaVersion => formatter.write_str("missing integer prism_schema"),
            Self::InvalidSchemaVersion(version) => {
                write!(formatter, "invalid schema version {version}")
            }
            Self::ValuesNotTable => {
                formatter.write_str("settings document values must be a TOML table")
            }
            Self::InvalidValue { key, reason } => {
                write!(formatter, "invalid value for `{key}`: {reason}")
            }
            Self::InvalidKnownValue { key, reason } => {
                write!(formatter, "known value `{key}` violates schema: {reason}")
            }
            Self::DuplicateKey(key) => write!(formatter, "duplicate persisted key `{key}`"),
        }
    }
}

impl Error for CodecError {}

fn encode_value(value: &Value) -> toml::Value {
    match value {
        Value::Bool(value) => toml::Value::Boolean(*value),
        Value::Int(value) => toml::Value::Integer(*value),
        Value::Float(value) => toml::Value::Float(*value),
        Value::String(value) => toml::Value::String(value.clone()),
        Value::Enum(value) => tagged("enum", toml::Value::Integer(i64::from(value.0))),
        Value::Bytes(value) => tagged("bytes", toml::Value::Integer(value.0 as i64)),
        Value::Secret(value) => tagged("secret", toml::Value::Integer(value.0 as i64)),
        Value::Array(values) => toml::Value::Array(values.iter().map(encode_value).collect()),
        Value::Table(values) => toml::Value::Table(
            values
                .iter()
                .map(|(key, value)| (key.clone(), encode_value(value)))
                .collect(),
        ),
        _ => toml::Value::String(format!("unsupported-value:{value:?}")),
    }
}

fn tagged(kind: &str, value: toml::Value) -> toml::Value {
    let mut table = toml::Table::new();
    table.insert(TAG_KEY.to_owned(), toml::Value::String(kind.to_owned()));
    table.insert("value".to_owned(), value);
    toml::Value::Table(table)
}

fn decode_value(value: &toml::Value, key: &str) -> Result<Value, String> {
    match value {
        toml::Value::Boolean(value) => Ok(Value::Bool(*value)),
        toml::Value::Integer(value) => Ok(Value::Int(*value)),
        toml::Value::Float(value) => Ok(Value::Float(*value)),
        toml::Value::String(value) => Ok(Value::String(value.clone())),
        toml::Value::Array(values) => values
            .iter()
            .map(|value| decode_value(value, key))
            .collect::<Result<Vec<_>, _>>()
            .map(Value::Array),
        toml::Value::Table(values) => {
            if let Some(kind) = values.get(TAG_KEY).and_then(toml::Value::as_str) {
                let raw = values
                    .get("value")
                    .ok_or_else(|| format!("tagged `{kind}` value is missing"))?;
                return match kind {
                    "enum" => {
                        decode_u32(raw, key).map(|value| Value::Enum(prism_settings::EnumId(value)))
                    }
                    "bytes" => decode_u64(raw, key)
                        .map(|value| Value::Bytes(prism_settings::BytesRef(value))),
                    "secret" => decode_u64(raw, key)
                        .map(|value| Value::Secret(prism_settings::SecretRef(value))),
                    other => Err(format!("unknown Prism value tag `{other}`")),
                };
            }
            values
                .iter()
                .map(|(key, value)| decode_value(value, key).map(|value| (key.clone(), value)))
                .collect::<Result<BTreeMap<_, _>, _>>()
                .map(Value::Table)
        }
        toml::Value::Datetime(_) => Err("TOML datetime is not a core setting value".to_owned()),
    }
}

fn decode_u32(value: &toml::Value, key: &str) -> Result<u32, String> {
    let value = value
        .as_integer()
        .ok_or_else(|| format!("tagged integer for `{key}` is not an integer"))?;
    u32::try_from(value).map_err(|_| format!("tagged integer for `{key}` is outside u32"))
}

fn decode_u64(value: &toml::Value, key: &str) -> Result<u64, String> {
    let value = value
        .as_integer()
        .ok_or_else(|| format!("tagged integer for `{key}` is not an integer"))?;
    u64::try_from(value).map_err(|_| format!("tagged integer for `{key}` is outside u64"))
}

pub(crate) fn decode_value_for_migration(value: &toml::Value, key: &str) -> Option<Value> {
    decode_value(value, key).ok()
}

#[cfg(test)]
mod tests {
    use super::*;
    use prism_settings::{RegistryBuilder, SettingDef};

    #[test]
    fn roundtrip_is_deterministic_and_preserves_nested_values() {
        let mut document = SettingsDocument::new(3);
        document.insert("audio.master_volume", Value::Float(0.75));
        document.insert("render.enabled", Value::Bool(true));
        document.preserve_unknown("future.setting", toml::Value::String("kept".to_owned()));
        let first = document.encode().unwrap();
        let second = SettingsDocument::decode(&first).unwrap().encode().unwrap();
        assert_eq!(first, second);
        let decoded = SettingsDocument::decode(&first).unwrap();
        assert_eq!(decoded.schema_version(), document.schema_version());
        assert_eq!(decoded.encode().unwrap(), first);
    }

    #[test]
    fn registry_classifies_unknown_and_rejects_invalid_known_values() {
        let mut builder = RegistryBuilder::new();
        builder.namespace("audio", "test").unwrap();
        let mut definition = SettingDef::new("audio.master_volume", "audio", Value::Float(0.5));
        definition.constraints.min = Some(0.0);
        definition.constraints.max = Some(1.0);
        builder.register(definition);
        let registry = builder.finish().unwrap();
        let document = SettingsDocument::decode_with_registry(
            "prism_schema = 1\nvalues = { \"audio.master_volume\" = 0.8, \"future.value\" = true }\n",
            Some(&registry),
        )
        .unwrap();
        assert!(document
            .candidates()
            .contains_key(&SettingKey::from("audio.master_volume")));
        assert!(document.unknown().contains_key("future.value"));
        assert!(SettingsDocument::decode_with_registry(
            "prism_schema = 1\nvalues = { \"audio.master_volume\" = 2.0 }\n",
            Some(&registry),
        )
        .is_err());
    }
}
