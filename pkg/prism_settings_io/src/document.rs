//! The versioned TOML document model and codec.

use prism_settings::{EnumOption, EnumSchema, SettingKey, SettingsRegistry, Value};
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
    enum_options: BTreeMap<SettingKey, Vec<EnumOption>>,
    enum_schemas: BTreeMap<SettingKey, Vec<EnumSchema>>,
}

impl SettingsDocument {
    /// Creates an empty document at `schema_version`.
    pub fn new(schema_version: u32) -> Self {
        Self {
            schema_version,
            candidates: BTreeMap::new(),
            unknown: BTreeMap::new(),
            enum_options: BTreeMap::new(),
            enum_schemas: BTreeMap::new(),
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
        if !matches!(
            value,
            Value::Enum(_)
                | Value::EnumPayload { .. }
                | Value::EnumTyped { .. }
                | Value::EnumPayloadTyped { .. }
        ) {
            self.enum_options.remove(&key);
            self.enum_schemas.remove(&key);
        }
        self.candidates.insert(key, value);
    }

    /// Associates registered stable enum names with candidate enum identities.
    ///
    /// This is a schema-time operation. It clones only the enum option tables
    /// needed to encode enum candidates later; it adds no per-frame work.
    pub fn bind_registry(&mut self, registry: &SettingsRegistry) {
        self.enum_options.clear();
        self.enum_schemas.clear();
        for key in self.candidates.keys() {
            let Some((_, definition)) = registry.get(key) else {
                continue;
            };
            if !definition.enum_options.is_empty() {
                self.enum_options
                    .insert(key.clone(), definition.enum_options.clone());
            }
            if !definition.enum_schemas.is_empty() {
                self.enum_schemas
                    .insert(key.clone(), definition.enum_schemas.clone());
            }
        }
    }

    /// Preserves an unknown TOML key/value pair.
    ///
    /// A key already present as a candidate is replaced by the preserved value.
    pub fn preserve_unknown(&mut self, key: impl Into<String>, value: toml::Value) {
        let key = key.into();
        self.candidates.remove(&SettingKey::new(key.clone()));
        self.enum_options.remove(&SettingKey::new(key.clone()));
        self.enum_schemas.remove(&SettingKey::new(key.clone()));
        self.unknown.insert(key, value);
    }

    /// Encodes this document as canonical TOML.
    ///
    /// Enum candidates require a registry binding from [`Self::bind_registry`]
    /// so the codec can persist stable option names rather than runtime IDs.
    pub fn encode(&self) -> Result<String, CodecError> {
        let mut values = toml::Table::new();
        for (key, value) in &self.candidates {
            if self.unknown.contains_key(key.as_str()) {
                return Err(CodecError::DuplicateKey(key.as_str().to_owned()));
            }
            values.insert(
                key.as_str().to_owned(),
                encode_value(
                    value,
                    key.as_str(),
                    self.enum_options.get(key).map(Vec::as_slice),
                    self.enum_schemas.get(key).map(Vec::as_slice),
                )?,
            );
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
            let enum_options = known
                .map(|(_, definition)| definition.enum_options.as_slice())
                .filter(|options| !options.is_empty());
            let enum_schemas = known
                .map(|(_, definition)| definition.enum_schemas.as_slice())
                .filter(|schemas| !schemas.is_empty());
            match decode_value(raw, key, enum_options, enum_schemas) {
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
                    if let Some((_, definition)) = known
                        && !definition.enum_options.is_empty()
                    {
                        document.enum_options.insert(
                            SettingKey::new(key.clone()),
                            definition.enum_options.clone(),
                        );
                    }
                    if let Some((_, definition)) = known
                        && !definition.enum_schemas.is_empty()
                    {
                        document.enum_schemas.insert(
                            SettingKey::new(key.clone()),
                            definition.enum_schemas.clone(),
                        );
                    }
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

    /// Reclassifies preserved values after schema migrations have run.
    ///
    /// Known values must decode and validate against the active registry;
    /// unresolved known values are errors instead of silently falling back to
    /// defaults. Keys absent from the registry remain losslessly preserved.
    pub(crate) fn classify_with_registry(
        &mut self,
        registry: &SettingsRegistry,
    ) -> Result<(), CodecError> {
        let known_keys: Vec<_> = self
            .unknown
            .keys()
            .filter_map(|key| {
                registry
                    .get(&SettingKey::new(key.clone()))
                    .map(|_| key.clone())
            })
            .collect();
        for key in known_keys {
            let Some(raw) = self.unknown.get(&key) else {
                continue;
            };
            let Some((id, definition)) = registry.get(&SettingKey::new(key.clone())) else {
                continue;
            };
            let options =
                (!definition.enum_options.is_empty()).then_some(definition.enum_options.as_slice());
            let schemas =
                (!definition.enum_schemas.is_empty()).then_some(definition.enum_schemas.as_slice());
            let value = decode_value(&raw, &key, options, schemas).map_err(|reason| {
                CodecError::InvalidValue {
                    key: key.clone(),
                    reason,
                }
            })?;
            registry
                .validate_value(id, value.clone())
                .map_err(|error| CodecError::InvalidKnownValue {
                    key: key.clone(),
                    reason: error.to_string(),
                })?;
            self.unknown.remove(&key);
            self.candidates.insert(SettingKey::new(key.clone()), value);
            if !definition.enum_options.is_empty() {
                self.enum_options.insert(
                    SettingKey::new(key.clone()),
                    definition.enum_options.clone(),
                );
            }
            if !definition.enum_schemas.is_empty() {
                self.enum_schemas
                    .insert(SettingKey::new(key), definition.enum_schemas.clone());
            }
        }
        Ok(())
    }

    /// Moves values for keys absent from the active schema into the lossless
    /// unknown map after a schema-free migration decode.
    pub(crate) fn preserve_unregistered(
        &mut self,
        registry: &SettingsRegistry,
    ) -> Result<(), CodecError> {
        let keys: Vec<_> = self
            .candidates
            .keys()
            .filter(|key| registry.get(key).is_none())
            .cloned()
            .collect();
        for key in keys {
            let value = self
                .candidates
                .get(&key)
                .ok_or_else(|| CodecError::InvalidValue {
                    key: key.to_string(),
                    reason: "candidate disappeared during schema classification".to_owned(),
                })?;
            let raw = encode_value(
                value,
                key.as_str(),
                self.enum_options.get(&key).map(Vec::as_slice),
                self.enum_schemas.get(&key).map(Vec::as_slice),
            )?;
            self.candidates.remove(&key);
            self.enum_options.remove(&key);
            self.enum_schemas.remove(&key);
            self.unknown.insert(key.to_string(), raw);
        }
        Ok(())
    }

    /// Removes and returns a candidate, if one exists.
    pub(crate) fn remove_candidate(&mut self, key: &str) -> Option<Value> {
        let key = SettingKey::new(key.to_owned());
        self.enum_options.remove(&key);
        self.enum_schemas.remove(&key);
        self.candidates.remove(&key)
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

fn encode_value(
    value: &Value,
    key: &str,
    enum_options: Option<&[EnumOption]>,
    enum_schemas: Option<&[EnumSchema]>,
) -> Result<toml::Value, CodecError> {
    let invalid = |reason: String| CodecError::InvalidValue {
        key: key.to_owned(),
        reason,
    };
    match value {
        Value::Bool(value) => Ok(toml::Value::Boolean(*value)),
        Value::Int(value) => Ok(toml::Value::Integer(*value)),
        Value::Float(value) if value.is_finite() => Ok(toml::Value::Float(*value)),
        Value::Float(_) => Err(invalid("non-finite floats cannot be persisted".to_owned())),
        Value::String(value) => Ok(toml::Value::String(value.clone())),
        Value::Enum(value) => Ok(tagged(
            "enum",
            toml::Value::String(enum_stable_name(*value, enum_options, &invalid)?),
        )),
        Value::EnumPayload { id, payload } => {
            let variant = enum_stable_name(*id, enum_options, &invalid)?;
            let raw_payload = match payload {
                prism_settings::EnumPayload::Tuple(values) => values
                    .iter()
                    .enumerate()
                    .map(|(index, value)| {
                        encode_value(
                            value,
                            &format!("{key}.payload[{index}]"),
                            enum_options,
                            enum_schemas,
                        )
                    })
                    .collect::<Result<Vec<_>, _>>()
                    .map(toml::Value::Array)?,
                prism_settings::EnumPayload::Struct(values) => values
                    .iter()
                    .map(|(field, value)| {
                        encode_value(
                            value,
                            &format!("{key}.payload.{field}"),
                            enum_options,
                            enum_schemas,
                        )
                        .map(|value| (field.clone(), value))
                    })
                    .collect::<Result<toml::Table, _>>()
                    .map(toml::Value::Table)?,
            };
            let mut tagged_value = toml::Table::new();
            tagged_value.insert(
                TAG_KEY.to_owned(),
                toml::Value::String("enum_payload".to_owned()),
            );
            tagged_value.insert("variant".to_owned(), toml::Value::String(variant));
            tagged_value.insert("payload".to_owned(), raw_payload);
            Ok(toml::Value::Table(tagged_value))
        }
        Value::EnumTyped { enum_type, id } => {
            let (type_name, variant_name) =
                typed_enum_names(*enum_type, *id, enum_schemas, &invalid)?;
            let mut tagged_value = toml::Table::new();
            tagged_value.insert(
                TAG_KEY.to_owned(),
                toml::Value::String("enum_typed".to_owned()),
            );
            tagged_value.insert("enum_type".to_owned(), toml::Value::String(type_name));
            tagged_value.insert("variant".to_owned(), toml::Value::String(variant_name));
            Ok(toml::Value::Table(tagged_value))
        }
        Value::EnumPayloadTyped {
            enum_type,
            id,
            payload,
        } => {
            let (type_name, variant_name) =
                typed_enum_names(*enum_type, *id, enum_schemas, &invalid)?;
            let raw_payload = match payload {
                prism_settings::EnumPayload::Tuple(values) => values
                    .iter()
                    .enumerate()
                    .map(|(index, value)| {
                        encode_value(
                            value,
                            &format!("{key}.payload[{index}]"),
                            enum_options,
                            enum_schemas,
                        )
                    })
                    .collect::<Result<Vec<_>, _>>()
                    .map(toml::Value::Array)?,
                prism_settings::EnumPayload::Struct(values) => values
                    .iter()
                    .map(|(field, value)| {
                        encode_value(
                            value,
                            &format!("{key}.payload.{field}"),
                            enum_options,
                            enum_schemas,
                        )
                        .map(|value| (field.clone(), value))
                    })
                    .collect::<Result<toml::Table, _>>()
                    .map(toml::Value::Table)?,
            };
            let mut tagged_value = toml::Table::new();
            tagged_value.insert(
                TAG_KEY.to_owned(),
                toml::Value::String("enum_typed_payload".to_owned()),
            );
            tagged_value.insert("enum_type".to_owned(), toml::Value::String(type_name));
            tagged_value.insert("variant".to_owned(), toml::Value::String(variant_name));
            tagged_value.insert("payload".to_owned(), raw_payload);
            Ok(toml::Value::Table(tagged_value))
        }
        // TOML integers are signed; decimal strings preserve the full u64 domain.
        Value::Bytes(value) => Ok(tagged("bytes", toml::Value::String(value.0.to_string()))),
        Value::Secret(value) => Ok(tagged("secret", toml::Value::String(value.0.to_string()))),
        Value::Array(values) => values
            .iter()
            .enumerate()
            .map(|(index, value)| {
                encode_value(
                    value,
                    &format!("{key}[{index}]"),
                    enum_options,
                    enum_schemas,
                )
            })
            .collect::<Result<Vec<_>, _>>()
            .map(toml::Value::Array),
        Value::Table(values) => values
            .iter()
            .map(|(child_key, value)| {
                encode_value(
                    value,
                    &format!("{key}.{child_key}"),
                    enum_options,
                    enum_schemas,
                )
                .map(|value| (child_key.clone(), value))
            })
            .collect::<Result<toml::Table, _>>()
            .map(toml::Value::Table),
        _ => Err(invalid("unsupported settings value variant".to_owned())),
    }
}

fn typed_enum_names(
    enum_type: prism_settings::EnumTypeId,
    variant: prism_settings::EnumId,
    schemas: Option<&[EnumSchema]>,
    invalid: &impl Fn(String) -> CodecError,
) -> Result<(String, String), CodecError> {
    let schema = schemas
        .and_then(|schemas| schemas.iter().find(|schema| schema.id == enum_type))
        .ok_or_else(|| invalid("typed enum value has no registered enum schema".to_owned()))?;
    let variant = schema
        .variants
        .iter()
        .find(|candidate| candidate.id == variant)
        .ok_or_else(|| invalid("typed enum value has no registered variant".to_owned()))?;
    Ok((schema.stable_name.clone(), variant.stable_name.clone()))
}

fn enum_stable_name(
    id: prism_settings::EnumId,
    enum_options: Option<&[EnumOption]>,
    invalid: &impl Fn(String) -> CodecError,
) -> Result<String, CodecError> {
    enum_options
        .and_then(|options| options.iter().find(|option| option.id == id))
        .map(|option| option.stable_name.clone())
        .ok_or_else(|| invalid("enum value has no registered stable name".to_owned()))
}

fn tagged(kind: &str, value: toml::Value) -> toml::Value {
    let mut table = toml::Table::new();
    table.insert(TAG_KEY.to_owned(), toml::Value::String(kind.to_owned()));
    table.insert("value".to_owned(), value);
    toml::Value::Table(table)
}

fn decode_value(
    value: &toml::Value,
    key: &str,
    enum_options: Option<&[EnumOption]>,
    enum_schemas: Option<&[EnumSchema]>,
) -> Result<Value, String> {
    match value {
        toml::Value::Boolean(value) => Ok(Value::Bool(*value)),
        toml::Value::Integer(value) => Ok(Value::Int(*value)),
        toml::Value::Float(value) => Ok(Value::Float(*value)),
        toml::Value::String(value) => Ok(Value::String(value.clone())),
        toml::Value::Array(values) => values
            .iter()
            .map(|value| decode_value(value, key, enum_options, enum_schemas))
            .collect::<Result<Vec<_>, _>>()
            .map(Value::Array),
        toml::Value::Table(values) => {
            if let Some(kind) = values.get(TAG_KEY).and_then(toml::Value::as_str) {
                if kind == "enum_payload" {
                    if values.len() != 3 {
                        return Err(format!(
                            "tagged enum payload for `{key}` has unexpected fields"
                        ));
                    }
                    let variant = values
                        .get("variant")
                        .and_then(toml::Value::as_str)
                        .ok_or_else(|| {
                            format!("tagged enum payload for `{key}` is missing string variant")
                        })?;
                    let option = enum_options
                        .and_then(|options| {
                            options.iter().find(|option| option.stable_name == variant)
                        })
                        .ok_or_else(|| format!("unknown stable enum name for `{key}`"))?;
                    let raw = values.get("payload").ok_or_else(|| {
                        format!("tagged enum payload for `{key}` is missing payload")
                    })?;
                    let payload = match raw {
                        toml::Value::Array(values) => prism_settings::EnumPayload::Tuple(
                            values
                                .iter()
                                .enumerate()
                                .map(|(index, value)| {
                                    decode_value(
                                        value,
                                        &format!("{key}.payload[{index}]"),
                                        enum_options,
                                        enum_schemas,
                                    )
                                })
                                .collect::<Result<Vec<_>, _>>()?,
                        ),
                        toml::Value::Table(values) => prism_settings::EnumPayload::Struct(
                            values
                                .iter()
                                .map(|(field, value)| {
                                    decode_value(
                                        value,
                                        &format!("{key}.payload.{field}"),
                                        enum_options,
                                        enum_schemas,
                                    )
                                    .map(|value| (field.clone(), value))
                                })
                                .collect::<Result<BTreeMap<_, _>, _>>()?,
                        ),
                        _ => {
                            return Err(format!(
                                "enum payload for `{key}` must be a tuple array or struct table"
                            ))
                        }
                    };
                    return Ok(Value::EnumPayload {
                        id: option.id,
                        payload,
                    });
                }
                if kind == "enum_typed" || kind == "enum_typed_payload" {
                    let expected = if kind == "enum_typed" { 3 } else { 4 };
                    if values.len() != expected {
                        return Err(format!(
                            "tagged typed enum for `{key}` has unexpected fields"
                        ));
                    }
                    let type_name = values
                        .get("enum_type")
                        .and_then(toml::Value::as_str)
                        .ok_or_else(|| {
                            format!("tagged typed enum for `{key}` is missing enum_type")
                        })?;
                    let variant_name = values
                        .get("variant")
                        .and_then(toml::Value::as_str)
                        .ok_or_else(|| {
                            format!("tagged typed enum for `{key}` is missing variant")
                        })?;
                    let schema = enum_schemas
                        .and_then(|schemas| {
                            schemas
                                .iter()
                                .find(|schema| schema.stable_name == type_name)
                        })
                        .ok_or_else(|| format!("unknown typed enum type for `{key}`"))?;
                    let variant = schema
                        .variants
                        .iter()
                        .find(|variant| variant.stable_name == variant_name)
                        .ok_or_else(|| format!("unknown typed enum variant for `{key}`"))?;
                    if kind == "enum_typed" {
                        return Ok(Value::EnumTyped {
                            enum_type: schema.id,
                            id: variant.id,
                        });
                    }
                    let raw = values.get("payload").ok_or_else(|| {
                        format!("typed enum payload for `{key}` is missing payload")
                    })?;
                    let payload = match raw {
                        toml::Value::Array(items) => prism_settings::EnumPayload::Tuple(
                            items
                                .iter()
                                .enumerate()
                                .map(|(index, item)| {
                                    decode_value(
                                        item,
                                        &format!("{key}.payload[{index}]"),
                                        enum_options,
                                        enum_schemas,
                                    )
                                })
                                .collect::<Result<Vec<_>, _>>()?,
                        ),
                        toml::Value::Table(fields) => prism_settings::EnumPayload::Struct(
                            fields
                                .iter()
                                .map(|(field, item)| {
                                    decode_value(
                                        item,
                                        &format!("{key}.payload.{field}"),
                                        enum_options,
                                        enum_schemas,
                                    )
                                    .map(|value| (field.clone(), value))
                                })
                                .collect::<Result<BTreeMap<_, _>, _>>()?,
                        ),
                        _ => {
                            return Err(format!(
                                "typed enum payload for `{key}` must be an array or table"
                            ))
                        }
                    };
                    return Ok(Value::EnumPayloadTyped {
                        enum_type: schema.id,
                        id: variant.id,
                        payload,
                    });
                }
                if values.len() != 2 {
                    return Err(format!(
                        "tagged `{kind}` value for `{key}` has unexpected fields"
                    ));
                }
                let raw = values
                    .get("value")
                    .ok_or_else(|| format!("tagged `{kind}` value is missing"))?;
                return match kind {
                    "enum" => match raw {
                        toml::Value::String(name) => enum_options
                            .and_then(|options| {
                                options.iter().find(|option| option.stable_name == *name)
                            })
                            .map(|option| Value::Enum(option.id))
                            .ok_or_else(|| format!("unknown stable enum name for `{key}`")),
                        toml::Value::Integer(_) => enum_options
                            .and_then(|options| {
                                decode_u32(raw, key).ok().and_then(|raw_id| {
                                    options.iter().find(|option| option.id.0 == raw_id)
                                })
                            })
                            .map(|option| Value::Enum(option.id))
                            .ok_or_else(|| format!("unknown legacy enum ID for `{key}`")),
                        _ => Err(format!("tagged enum value for `{key}` must be a string")),
                    },
                    "bytes" => decode_u64(raw, key)
                        .map(|value| Value::Bytes(prism_settings::BytesRef(value))),
                    "secret" => decode_u64(raw, key)
                        .map(|value| Value::Secret(prism_settings::SecretRef(value))),
                    other => Err(format!("unknown Prism value tag `{other}`")),
                };
            }
            values
                .iter()
                .map(|(key, value)| {
                    decode_value(value, key, enum_options, enum_schemas)
                        .map(|value| (key.clone(), value))
                })
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
    match value {
        toml::Value::String(value) => value
            .parse()
            .map_err(|_| format!("tagged decimal reference for `{key}` is not a u64")),
        toml::Value::Integer(value) => {
            u64::try_from(*value).map_err(|_| format!("tagged integer for `{key}` is outside u64"))
        }
        _ => Err(format!(
            "tagged reference for `{key}` is not an integer or decimal string"
        )),
    }
}

pub(crate) fn decode_value_for_migration(value: &toml::Value, key: &str) -> Option<Value> {
    // Migrations run before the active registry is bound. Stable enum names
    // still carry enough identity to produce the same canonical runtime ID;
    // the destination registry will validate the converted value afterward.
    match value {
        toml::Value::Array(values) => values
            .iter()
            .enumerate()
            .map(|(index, value)| decode_value_for_migration(value, &format!("{key}[{index}]")))
            .collect::<Option<Vec<_>>>()
            .map(Value::Array),
        toml::Value::Table(values)
            if values.get(TAG_KEY).and_then(toml::Value::as_str) == Some("enum") =>
        {
            match values.get("value")? {
                toml::Value::String(name) => {
                    Some(Value::Enum(prism_settings::EnumId::from_stable_name(name)))
                }
                toml::Value::Integer(_) => decode_u32(values.get("value")?, key)
                    .ok()
                    .map(|id| Value::Enum(prism_settings::EnumId(id))),
                _ => None,
            }
        }
        toml::Value::Table(values)
            if values.get(TAG_KEY).and_then(toml::Value::as_str) == Some("enum_payload") =>
        {
            if values.len() != 3 {
                return None;
            }
            let id = match values.get("variant")? {
                toml::Value::String(name) => prism_settings::EnumId::from_stable_name(name),
                _ => return None,
            };
            let payload = match values.get("payload")? {
                toml::Value::Array(items) => prism_settings::EnumPayload::Tuple(
                    items
                        .iter()
                        .enumerate()
                        .map(|(index, item)| {
                            decode_value_for_migration(item, &format!("{key}.payload[{index}]"))
                        })
                        .collect::<Option<Vec<_>>>()?,
                ),
                toml::Value::Table(fields) => prism_settings::EnumPayload::Struct(
                    fields
                        .iter()
                        .map(|(field, item)| {
                            decode_value_for_migration(item, &format!("{key}.payload.{field}"))
                                .map(|value| (field.clone(), value))
                        })
                        .collect::<Option<BTreeMap<_, _>>>()?,
                ),
                _ => return None,
            };
            Some(Value::EnumPayload { id, payload })
        }
        toml::Value::Table(values)
            if values.get(TAG_KEY).and_then(toml::Value::as_str) == Some("enum_typed") =>
        {
            if values.len() != 3 {
                return None;
            }
            let enum_type = match values.get("enum_type")? {
                toml::Value::String(name) => prism_settings::EnumTypeId::from_stable_name(name),
                _ => return None,
            };
            let id = match values.get("variant")? {
                toml::Value::String(name) => prism_settings::EnumId::from_stable_name(name),
                _ => return None,
            };
            Some(Value::EnumTyped { enum_type, id })
        }
        toml::Value::Table(values)
            if values.get(TAG_KEY).and_then(toml::Value::as_str) == Some("enum_typed_payload") =>
        {
            if values.len() != 4 {
                return None;
            }
            let enum_type = match values.get("enum_type")? {
                toml::Value::String(name) => prism_settings::EnumTypeId::from_stable_name(name),
                _ => return None,
            };
            let id = match values.get("variant")? {
                toml::Value::String(name) => prism_settings::EnumId::from_stable_name(name),
                _ => return None,
            };
            let payload = match values.get("payload")? {
                toml::Value::Array(items) => prism_settings::EnumPayload::Tuple(
                    items
                        .iter()
                        .enumerate()
                        .map(|(index, item)| {
                            decode_value_for_migration(item, &format!("{key}.payload[{index}]"))
                        })
                        .collect::<Option<Vec<_>>>()?,
                ),
                toml::Value::Table(fields) => prism_settings::EnumPayload::Struct(
                    fields
                        .iter()
                        .map(|(field, item)| {
                            decode_value_for_migration(item, &format!("{key}.payload.{field}"))
                                .map(|value| (field.clone(), value))
                        })
                        .collect::<Option<BTreeMap<_, _>>>()?,
                ),
                _ => return None,
            };
            Some(Value::EnumPayloadTyped {
                enum_type,
                id,
                payload,
            })
        }
        toml::Value::Table(values) => values
            .iter()
            .map(|(child_key, value)| {
                decode_value_for_migration(value, &format!("{key}.{child_key}"))
                    .map(|value| (child_key.clone(), value))
            })
            .collect::<Option<BTreeMap<_, _>>>()
            .map(Value::Table),
        _ => decode_value(value, key, None, None).ok(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use prism_settings::{RegistryBuilder, SettingDef, ValueShape, ValueType};

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

    #[test]
    fn registry_codec_roundtrips_valid_arrays_and_rejects_invalid_elements() {
        let mut builder = RegistryBuilder::new();
        builder.namespace("render", "test").unwrap();
        let definition = SettingDef::new(
            "render.samples",
            "render",
            Value::Array(vec![Value::Int(1), Value::Int(2)]),
        )
        .with_shape(ValueShape::bounded_array(
            ValueShape::scalar(ValueType::Int),
            8,
        ));
        builder.register(definition);
        let registry = builder.finish().unwrap();

        let mut document = SettingsDocument::new(1);
        document.insert(
            "render.samples",
            Value::Array(vec![Value::Int(3), Value::Int(5)]),
        );
        let encoded = document.encode().unwrap();
        let decoded = SettingsDocument::decode_with_registry(&encoded, Some(&registry)).unwrap();
        assert_eq!(decoded, document);

        let invalid = "prism_schema = 1\nvalues = { \"render.samples\" = [3, true] }\n";
        assert!(SettingsDocument::decode_with_registry(invalid, Some(&registry)).is_err());
    }

    #[test]
    fn byte_and_secret_references_roundtrip_full_u64_range() {
        let mut document = SettingsDocument::new(1);
        document.insert(
            "refs.bytes",
            Value::Bytes(prism_settings::BytesRef(u64::MAX)),
        );
        document.insert(
            "refs.secret",
            Value::Secret(prism_settings::SecretRef(u64::MAX)),
        );

        let encoded = document.encode().unwrap();
        let decoded = SettingsDocument::decode(&encoded).unwrap();
        assert_eq!(decoded, document);
        assert!(encoded.contains("18446744073709551615"));
    }

    #[test]
    fn enums_persist_by_stable_name_and_decode_legacy_ids() {
        let low = EnumOption::from_stable_name("Low");
        let high = EnumOption::from_stable_name("High");
        let mut definition = SettingDef::new("render.quality", "render", Value::Enum(low.id));
        definition.enum_options = vec![low.clone(), high.clone()];
        let mut builder = RegistryBuilder::new();
        builder.namespace("render", "test").unwrap();
        builder.register(definition);
        let registry = builder.finish().unwrap();

        let mut document = SettingsDocument::new(1);
        document.insert("render.quality", Value::Enum(high.id));
        document.bind_registry(&registry);
        let encoded = document.encode().unwrap();
        assert!(encoded.contains("value = \"High\""));
        assert!(!encoded.contains(&high.id.0.to_string()));

        let decoded = SettingsDocument::decode_with_registry(&encoded, Some(&registry)).unwrap();
        assert_eq!(decoded, document);

        let legacy = format!(
            "prism_schema = 1\n[values.\"render.quality\"]\n__prism_value = \"enum\"\nvalue = {}\n",
            high.id.0
        );
        let migrated = SettingsDocument::decode_with_registry(&legacy, Some(&registry)).unwrap();
        assert_eq!(
            migrated
                .candidates()
                .get(&SettingKey::from("render.quality")),
            Some(&Value::Enum(high.id))
        );
        assert!(migrated.encode().unwrap().contains("value = \"High\""));

        let unknown = "prism_schema = 1\n[values.\"render.quality\"]\n__prism_value = \"enum\"\nvalue = \"Ultra\"\n";
        assert!(SettingsDocument::decode_with_registry(unknown, Some(&registry)).is_err());
    }

    #[test]
    fn typed_enum_codec_roundtrips_type_identity_and_nested_payloads() {
        use prism_settings::{EnumSchema, EnumVariantSchema};
        let first_type = prism_settings::EnumTypeId::from_stable_name("settings::First");
        let second_type = prism_settings::EnumTypeId::from_stable_name("settings::Second");
        let first_variant_name = "settings::First::Value";
        let second_variant_name = "settings::Second::Value";
        let first_variant = EnumVariantSchema {
            id: prism_settings::EnumId::from_stable_name(first_variant_name),
            stable_name: first_variant_name.into(),
            payload: Some(prism_settings::EnumPayloadShape::Tuple(vec![
                ValueShape::scalar(ValueType::Int),
            ])),
        };
        let second_variant = EnumVariantSchema {
            id: prism_settings::EnumId::from_stable_name(second_variant_name),
            stable_name: second_variant_name.into(),
            payload: None,
        };
        let mut definition = SettingDef::new(
            "render.typed",
            "render",
            Value::Table(BTreeMap::from([
                (
                    "first".into(),
                    Value::EnumPayloadTyped {
                        enum_type: first_type,
                        id: first_variant.id,
                        payload: prism_settings::EnumPayload::Tuple(vec![Value::Int(3)]),
                    },
                ),
                (
                    "second".into(),
                    Value::EnumTyped {
                        enum_type: second_type,
                        id: second_variant.id,
                    },
                ),
            ])),
        );
        definition.set_shape(ValueShape::table(BTreeMap::from([
            ("first".into(), ValueShape::Enum(first_type)),
            ("second".into(), ValueShape::Enum(second_type)),
        ])));
        definition.enum_schemas = vec![
            EnumSchema {
                id: first_type,
                stable_name: "settings::First".into(),
                variants: vec![first_variant],
            },
            EnumSchema {
                id: second_type,
                stable_name: "settings::Second".into(),
                variants: vec![second_variant],
            },
        ];
        let mut builder = RegistryBuilder::new();
        builder.namespace("render", "test").unwrap();
        builder.register(definition);
        let registry = builder.finish().unwrap();

        let mut document = SettingsDocument::new(1);
        let value = registry
            .def(registry.id(&"render.typed".into()).unwrap())
            .unwrap()
            .default
            .clone();
        document.insert("render.typed", value.clone());
        document.bind_registry(&registry);
        let encoded = document.encode().unwrap();
        assert!(encoded.contains("enum_typed_payload"));
        assert!(encoded.contains("settings::First"));
        assert!(encoded.contains("settings::Second"));
        let decoded = SettingsDocument::decode_with_registry(&encoded, Some(&registry)).unwrap();
        assert_eq!(
            decoded.candidates().get(&SettingKey::from("render.typed")),
            Some(&value)
        );
    }

    #[test]
    fn nested_enum_arrays_persist_and_migrate_by_stable_name() {
        let low = EnumOption::from_stable_name("Low");
        let high = EnumOption::from_stable_name("High");
        let mut definition = SettingDef::new(
            "render.quality_history",
            "render",
            Value::Array(vec![Value::Array(vec![Value::Enum(low.id)])]),
        )
        .with_shape(ValueShape::bounded_array(
            ValueShape::bounded_array(ValueShape::scalar(ValueType::Enum), 4),
            4,
        ));
        definition.enum_options = vec![low.clone(), high.clone()];
        let mut builder = RegistryBuilder::new();
        builder.namespace("render", "test").unwrap();
        builder.register(definition);
        let registry = builder.finish().unwrap();

        let mut document = SettingsDocument::new(1);
        document.insert(
            "render.quality_history",
            Value::Array(vec![Value::Array(vec![Value::Enum(high.id)])]),
        );
        document.bind_registry(&registry);
        let encoded = document.encode().unwrap();
        assert!(encoded.contains("value = \"High\""));
        let decoded = SettingsDocument::decode_with_registry(&encoded, Some(&registry)).unwrap();
        assert_eq!(decoded, document);

        let migrated = decode_value_for_migration(
            &toml::Value::Array(vec![toml::Value::Array(vec![tagged(
                "enum",
                toml::Value::String("High".to_owned()),
            )])]),
            "render.quality_history",
        )
        .unwrap();
        assert_eq!(
            migrated,
            Value::Array(vec![Value::Array(vec![Value::Enum(high.id)])])
        );
    }

    #[test]
    fn nested_table_and_map_enum_values_roundtrip_and_migrate_by_stable_name() {
        let low = EnumOption::from_stable_name("Low");
        let high = EnumOption::from_stable_name("High");
        let nested_shape = ValueShape::table(BTreeMap::from([(
            "quality".to_owned(),
            ValueShape::scalar(ValueType::Enum),
        )]));
        let mut table_definition = SettingDef::new(
            "render.profile",
            "render",
            Value::Table(BTreeMap::from([(
                "quality".to_owned(),
                Value::Enum(low.id),
            )])),
        )
        .with_shape(nested_shape);
        table_definition.enum_options = vec![low.clone(), high.clone()];

        let mut map_definition = SettingDef::new(
            "render.quality_by_device",
            "render",
            Value::Table(BTreeMap::from([("gpu".to_owned(), Value::Enum(low.id))])),
        )
        .with_shape(ValueShape::map(ValueShape::scalar(ValueType::Enum), 8));
        map_definition.enum_options = vec![low.clone(), high.clone()];

        let mut builder = RegistryBuilder::new();
        builder.namespace("render", "test").unwrap();
        builder.register(table_definition);
        builder.register(map_definition);
        let registry = builder.finish().unwrap();

        let mut document = SettingsDocument::new(1);
        document.insert(
            "render.profile",
            Value::Table(BTreeMap::from([(
                "quality".to_owned(),
                Value::Enum(high.id),
            )])),
        );
        document.insert(
            "render.quality_by_device",
            Value::Table(BTreeMap::from([
                ("cpu".to_owned(), Value::Enum(low.id)),
                ("gpu".to_owned(), Value::Enum(high.id)),
            ])),
        );
        document.bind_registry(&registry);

        let encoded = document.encode().unwrap();
        assert!(encoded.contains("value = \"High\""));
        let decoded = SettingsDocument::decode_with_registry(&encoded, Some(&registry)).unwrap();
        assert_eq!(decoded, document);

        let migration_input: toml::Value = toml::from_str(&encoded).unwrap();
        let decoded_table = decode_value_for_migration(
            migration_input
                .get("values")
                .and_then(toml::Value::as_table)
                .and_then(|values| values.get("render.profile"))
                .unwrap(),
            "render.profile",
        )
        .unwrap();
        assert_eq!(
            decoded_table,
            Value::Table(BTreeMap::from([(
                "quality".to_owned(),
                Value::Enum(high.id)
            )]))
        );
        let decoded_map = decode_value_for_migration(
            migration_input
                .get("values")
                .and_then(toml::Value::as_table)
                .and_then(|values| values.get("render.quality_by_device"))
                .unwrap(),
            "render.quality_by_device",
        )
        .unwrap();
        assert_eq!(
            decoded_map,
            Value::Table(BTreeMap::from([
                ("cpu".to_owned(), Value::Enum(low.id)),
                ("gpu".to_owned(), Value::Enum(high.id)),
            ]))
        );
    }

    #[test]
    fn stable_enum_values_can_be_converted_during_schema_migration() {
        let old_option = EnumOption::from_stable_name("render::Quality::High");
        let new_option = EnumOption::from_stable_name("render::Quality::Ultra");
        let mut document = SettingsDocument::new(1);
        document.preserve_unknown(
            "render.quality",
            tagged("enum", toml::Value::String(old_option.stable_name.clone())),
        );

        let mut chain = crate::MigrationChain::new(2);
        chain
            .add_step(
                1,
                [crate::MigrationOperation::convert(
                    "render.quality",
                    move |value| match value {
                        Value::Enum(id) if id == old_option.id => Ok(Value::Enum(new_option.id)),
                        other => Err(format!("unexpected value: {other:?}")),
                    },
                )],
            )
            .unwrap();

        let report = chain.migrate(&mut document).unwrap();
        assert_eq!(report.converted, 1);
        assert_eq!(document.schema_version(), 2);
        assert_eq!(
            document
                .candidates()
                .get(&SettingKey::from("render.quality")),
            Some(&Value::Enum(new_option.id))
        );

        let mut builder = RegistryBuilder::new();
        builder.namespace("render", "test").unwrap();
        let mut definition =
            SettingDef::new("render.quality", "render", Value::Enum(new_option.id));
        definition.enum_options = vec![new_option.clone()];
        builder.register(definition);
        let registry = builder.finish().unwrap();
        document.bind_registry(&registry);
        assert!(document
            .encode()
            .unwrap()
            .contains("value = \"render::Quality::Ultra\""));
    }

    #[test]
    fn enum_payload_roundtrips_with_stable_variant_names_and_migrates_recursively() {
        use prism_settings::{EnumOptionPayload, EnumPayload, EnumPayloadShape};
        let plain = EnumOption::from_stable_name("render::Mode::Plain");
        let configured = EnumOption::from_stable_name("render::Mode::Configured");
        let nested = EnumOption::from_stable_name("render::Filter::Radius");
        let default = Value::EnumPayload {
            id: configured.id,
            payload: EnumPayload::Struct(BTreeMap::from([
                ("quality".to_owned(), Value::Enum(plain.id)),
                (
                    "filter".to_owned(),
                    Value::EnumPayload {
                        id: nested.id,
                        payload: EnumPayload::Tuple(vec![Value::Int(4)]),
                    },
                ),
            ])),
        };
        let mut definition = SettingDef::new("render.mode", "render", default.clone());
        definition.enum_options = vec![plain.clone(), configured.clone(), nested.clone()];
        definition.enum_payloads = vec![
            EnumOptionPayload {
                id: configured.id,
                shape: EnumPayloadShape::Struct(BTreeMap::from([
                    ("quality".to_owned(), ValueShape::scalar(ValueType::Enum)),
                    ("filter".to_owned(), ValueShape::scalar(ValueType::Enum)),
                ])),
            },
            EnumOptionPayload {
                id: nested.id,
                shape: EnumPayloadShape::Tuple(vec![ValueShape::scalar(ValueType::Int)]),
            },
        ];
        let mut builder = RegistryBuilder::new();
        builder.namespace("render", "test").unwrap();
        builder.register(definition);
        let registry = builder.finish().unwrap();

        let mut document = SettingsDocument::new(1);
        document.insert("render.mode", default.clone());
        document.bind_registry(&registry);
        let encoded = document.encode().unwrap();
        assert!(encoded.contains("__prism_value = \"enum_payload\""));
        assert!(encoded.contains("render::Mode::Configured"));
        assert!(encoded.contains("render::Filter::Radius"));
        assert!(!encoded.contains(&configured.id.0.to_string()));
        let decoded = SettingsDocument::decode_with_registry(&encoded, Some(&registry)).unwrap();
        assert_eq!(
            decoded.candidates().get(&SettingKey::from("render.mode")),
            Some(&default)
        );

        let root: toml::Value = toml::from_str(&encoded).unwrap();
        let raw = root
            .get("values")
            .and_then(toml::Value::as_table)
            .and_then(|values| values.get("render.mode"))
            .unwrap();
        assert_eq!(
            decode_value_for_migration(raw, "render.mode"),
            Some(default)
        );
    }

    #[test]
    fn enum_payload_codec_rejects_malformed_or_ambiguous_tags() {
        let option = EnumOption::from_stable_name("render::Mode::Configured");
        let mut definition = SettingDef::new("render.mode", "render", Value::Enum(option.id));
        definition.enum_options = vec![option];
        let mut builder = RegistryBuilder::new();
        builder.namespace("render", "test").unwrap();
        builder.register(definition);
        let registry = builder.finish().unwrap();
        for payload in [
            "{ __prism_value = \"enum_payload\", variant = 4, payload = [] }",
            "{ __prism_value = \"enum_payload\", variant = \"render::Mode::Configured\", payload = 1 }",
            "{ __prism_value = \"enum_payload\", variant = \"render::Mode::Missing\", payload = [] }",
            "{ __prism_value = \"enum_payload\", variant = \"render::Mode::Configured\", payload = [], extra = true }",
        ] {
            let input = format!("prism_schema = 1\nvalues = {{ \"render.mode\" = {payload} }}\n");
            assert!(SettingsDocument::decode_with_registry(&input, Some(&registry)).is_err(), "accepted {payload}");
        }
    }
}
