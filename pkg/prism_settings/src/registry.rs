//! Deterministic schema registration and explicit freeze validation.
use crate::{
    CrossRule, Domain, EnumId, LayerKind, SettingDef, SettingId, SettingKey, Value, ValueType,
};
use alloc::{
    collections::{BTreeMap, BTreeSet},
    string::{String, ToString},
    vec::Vec,
};
use core::fmt;

/// Failure category from registry construction or value resolution.
#[derive(Clone, Debug, PartialEq)]
pub enum SettingsError {
    /// Canonical key or alias already belongs to another definition.
    DuplicateKey(SettingKey),
    /// Input source named a key absent from the frozen schema.
    UnknownKey(SettingKey),
    /// Invalid key syntax or length.
    InvalidKey(SettingKey),
    /// Namespace is missing or does not own the key prefix.
    NamespaceOwnership {
        /// Definition whose prefix is invalid.
        key: SettingKey,
        /// Declared namespace prefix.
        namespace: String,
    },
    /// Namespace was declared more than once.
    DuplicateNamespace(String),
    /// Invalid schema default or constraint configuration.
    InvalidDefinition {
        /// Invalid definition key.
        key: SettingKey,
        /// Validation explanation.
        reason: String,
    },
    /// Cross-setting reference does not resolve.
    UnknownReference {
        /// Definition that contains the reference.
        key: SettingKey,
        /// Referenced but unregistered key.
        referenced: SettingKey,
    },
    /// Value does not match definition type.
    TypeMismatch {
        /// Setting key.
        key: SettingKey,
        /// Required schema type.
        expected: ValueType,
        /// Actual value type.
        actual: ValueType,
    },
    /// Invalid numeric value or bound violation.
    OutOfRange {
        /// Setting key.
        key: SettingKey,
        /// Rejected value.
        value: Value,
    },
    /// Enum identity is not allowed by the schema.
    InvalidEnum {
        /// Setting key.
        key: SettingKey,
        /// Unknown enum ID.
        value: EnumId,
    },
    /// String or array exceeds schema limit.
    TooLong {
        /// Setting key.
        key: SettingKey,
    },
    /// An array is shorter than the schema's minimum length.
    TooShort {
        /// Setting key.
        key: SettingKey,
    },
    /// Value set was sourced from a non-Trusted layer for an execution-class setting.
    UntrustedExecutionSetting {
        /// Protected setting key.
        key: SettingKey,
        /// Trust level of the rejected source.
        trust: crate::TrustLevel,
    },
    /// Setting is not visible in the requested domain.
    WrongDomain {
        /// Setting key.
        key: SettingKey,
        /// Requested domain.
        domain: Domain,
    },
    /// Layer is not legal for the requested domain.
    WrongLayer {
        /// Layer supplied by caller.
        layer: LayerKind,
        /// Domain whose stack was checked.
        domain: Domain,
    },
    /// A legal layer is excluded by the definition's precedence profile.
    PrecedenceExcluded {
        /// Setting key.
        key: SettingKey,
        /// Candidate's layer.
        layer: LayerKind,
        /// Definition policy.
        profile: crate::PrecedenceProfile,
    },
    /// Two input records claim the same source identity in one layer.
    DuplicateSource {
        /// Layer containing the duplicate identity.
        layer: LayerKind,
        /// Repeated stable source identity.
        source_id: String,
    },
    /// Mandatory policy was supplied by a source that is not trusted.
    UntrustedMandatory {
        /// Stable source identity.
        source_id: String,
        /// Supplied trust classification.
        trust: crate::TrustLevel,
    },
    /// Effective cross-setting invariant was violated.
    CrossConstraint {
        /// Definition whose rule failed.
        key: SettingKey,
        /// Related definition key.
        referenced: SettingKey,
    },
    /// Read-only setting was assigned from a non-Mandatory source.
    ReadOnly(SettingKey),
    /// Project or project-local source attempted to override a protected definition.
    ProjectOverrideDenied(SettingKey),
    /// ID is outside the frozen registry.
    InvalidId(SettingId),
}

impl fmt::Display for SettingsError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::DuplicateKey(k) => write!(f, "duplicate setting key or alias: {k}"),
            Self::UnknownKey(k) => write!(f, "unknown setting key: {k}"),
            Self::InvalidKey(k) => write!(f, "invalid setting key: {k}"),
            Self::NamespaceOwnership { key, namespace } => {
                write!(f, "namespace `{namespace}` does not own `{key}`")
            }
            Self::DuplicateNamespace(n) => write!(f, "duplicate namespace: {n}"),
            Self::InvalidDefinition { key, reason } => {
                write!(f, "invalid definition `{key}`: {reason}")
            }
            Self::UnknownReference { key, referenced } => {
                write!(f, "`{key}` references unknown setting `{referenced}`")
            }
            Self::TypeMismatch {
                key,
                expected,
                actual,
            } => write!(f, "`{key}` expects {expected:?}, got {actual:?}"),
            Self::OutOfRange { key, value } => {
                write!(f, "value {value:?} violates bounds for `{key}`")
            }
            Self::InvalidEnum { key, value } => {
                write!(f, "enum id {value:?} is not allowed for `{key}`")
            }
            Self::TooLong { key } => write!(f, "value exceeds configured length for `{key}`"),
            Self::TooShort { key } => {
                write!(f, "value is shorter than configured length for `{key}`")
            }
            Self::UntrustedExecutionSetting { key, trust } => {
                write!(f, "{trust:?} source cannot set execution-class key `{key}`")
            }
            Self::WrongDomain { key, domain } => write!(f, "`{key}` is not visible in {domain:?}"),
            Self::WrongLayer { layer, domain } => {
                write!(f, "layer {layer:?} is not valid in {domain:?}")
            }
            Self::PrecedenceExcluded {
                key,
                layer,
                profile,
            } => {
                write!(
                    f,
                    "{layer:?} candidate for `{key}` is excluded by {profile:?}"
                )
            }
            Self::DuplicateSource { layer, source_id } => {
                write!(f, "source `{source_id}` occurs more than once in {layer:?}")
            }
            Self::UntrustedMandatory { source_id, trust } => {
                write!(
                    f,
                    "mandatory source `{source_id}` must be Trusted, got {trust:?}"
                )
            }
            Self::CrossConstraint { key, referenced } => write!(
                f,
                "cross-setting rule for `{key}` conflicts with `{referenced}`"
            ),
            Self::ReadOnly(k) => write!(f, "setting `{k}` is read-only"),
            Self::ProjectOverrideDenied(k) => write!(f, "project sources cannot override `{k}`"),
            Self::InvalidId(id) => write!(f, "invalid setting id {}", id.0),
        }
    }
}

#[cfg(feature = "std")]
impl std::error::Error for SettingsError {}

/// Stable namespace record.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Namespace {
    /// Interned namespace id.
    pub id: crate::NamespaceId,
    /// Namespace prefix.
    pub name: String,
    /// Owning crate/plugin identity.
    pub owner: String,
}

/// Mutable, explicit registry construction phase.
#[derive(Default)]
pub struct RegistryBuilder {
    namespaces: BTreeMap<String, String>,
    defs: Vec<SettingDef>,
}

impl RegistryBuilder {
    /// Creates an empty builder.
    pub fn new() -> Self {
        Self::default()
    }
    /// Declares a namespace and its unique owner.
    pub fn namespace(
        &mut self,
        name: impl Into<String>,
        owner: impl Into<String>,
    ) -> Result<(), SettingsError> {
        let name = name.into();
        if self.namespaces.contains_key(&name) {
            return Err(SettingsError::DuplicateNamespace(name));
        }
        self.namespaces.insert(name, owner.into());
        Ok(())
    }
    /// Registers a schema definition. Definitions are sorted by key at freeze, never by call order.
    pub fn register(&mut self, def: SettingDef) {
        self.defs.push(def);
    }
    /// Validates every definition/reference and freezes deterministic dense IDs.
    pub fn finish(mut self) -> Result<SettingsRegistry, Vec<SettingsError>> {
        self.defs.sort_by(|a, b| a.key.cmp(&b.key));
        let mut errors = Vec::new();
        let mut key_to_id = BTreeMap::new();
        for (idx, def) in self.defs.iter().enumerate() {
            if !valid_key(def.key.as_str()) {
                errors.push(SettingsError::InvalidKey(def.key.clone()));
            }
            match self.namespaces.get(&def.namespace) {
                None => errors.push(SettingsError::NamespaceOwnership {
                    key: def.key.clone(),
                    namespace: def.namespace.clone(),
                }),
                Some(_)
                    if !(def.key.as_str() == def.namespace
                        || def
                            .key
                            .as_str()
                            .strip_prefix(&def.namespace)
                            .is_some_and(|tail| tail.starts_with('.'))) =>
                {
                    errors.push(SettingsError::NamespaceOwnership {
                        key: def.key.clone(),
                        namespace: def.namespace.clone(),
                    })
                }
                _ => {}
            }
            if def.domains.is_empty() {
                errors.push(SettingsError::InvalidDefinition {
                    key: def.key.clone(),
                    reason: "domains cannot be empty".to_string(),
                });
            }
            validate_def(def, &mut errors);
            if key_to_id
                .insert(def.key.clone(), SettingId(idx as u32))
                .is_some()
            {
                errors.push(SettingsError::DuplicateKey(def.key.clone()));
            }
        }
        let mut aliases: BTreeMap<SettingKey, SettingKey> = BTreeMap::new();
        for def in &self.defs {
            for alias in &def.aliases {
                if !valid_key(alias.as_str()) {
                    errors.push(SettingsError::InvalidKey(alias.clone()));
                }
                if key_to_id.contains_key(alias)
                    || aliases.insert(alias.clone(), def.key.clone()).is_some()
                {
                    errors.push(SettingsError::DuplicateKey(alias.clone()));
                }
            }
            for rule in &def.cross_rules {
                let referenced = match rule {
                    CrossRule::Requires { key, .. } | CrossRule::Conflicts { key, .. } => key,
                };
                if !key_to_id.contains_key(referenced) {
                    errors.push(SettingsError::UnknownReference {
                        key: def.key.clone(),
                        referenced: referenced.clone(),
                    });
                } else if def.ty != ValueType::Bool
                    || self.defs[key_to_id[referenced].0 as usize].ty != ValueType::Bool
                {
                    errors.push(SettingsError::InvalidDefinition {
                        key: def.key.clone(),
                        reason: "cross-setting boolean rules require boolean definitions"
                            .to_string(),
                    });
                }
            }
        }
        if !errors.is_empty() {
            return Err(errors);
        }
        let namespaces = self
            .namespaces
            .into_iter()
            .enumerate()
            .map(|(i, (name, owner))| Namespace {
                id: crate::NamespaceId(i as u32),
                name,
                owner,
            })
            .collect::<Vec<_>>();
        Ok(SettingsRegistry {
            defs: self.defs,
            key_to_id,
            aliases,
            namespaces,
        })
    }
}

/// Frozen, deterministic registry. Its schema can no longer be mutated.
#[derive(Clone, Debug)]
pub struct SettingsRegistry {
    defs: Vec<SettingDef>,
    key_to_id: BTreeMap<SettingKey, SettingId>,
    aliases: BTreeMap<SettingKey, SettingKey>,
    namespaces: Vec<Namespace>,
}

impl SettingsRegistry {
    /// Number of definitions.
    pub fn len(&self) -> usize {
        self.defs.len()
    }
    /// Whether no definitions were registered.
    pub fn is_empty(&self) -> bool {
        self.defs.is_empty()
    }
    /// Definitions in canonical key order.
    pub fn defs(&self) -> &[SettingDef] {
        &self.defs
    }
    /// Namespace records in canonical name order.
    pub fn namespaces(&self) -> &[Namespace] {
        &self.namespaces
    }
    /// Looks up a canonical key or alias.
    pub fn id(&self, key: &SettingKey) -> Option<SettingId> {
        self.key_to_id.get(key).copied().or_else(|| {
            self.aliases
                .get(key)
                .and_then(|canonical| self.key_to_id.get(canonical).copied())
        })
    }
    /// Looks up a definition by dense ID.
    pub fn def(&self, id: SettingId) -> Option<&SettingDef> {
        self.defs.get(id.0 as usize)
    }
    /// Looks up a definition by canonical key or alias.
    pub fn get(&self, key: &SettingKey) -> Option<(SettingId, &SettingDef)> {
        let id = self.id(key)?;
        Some((id, self.def(id)?))
    }
    /// Iterates definitions owned by a namespace in stable key order.
    pub fn iter_namespace<'a>(
        &'a self,
        namespace: &'a str,
    ) -> impl Iterator<Item = (SettingId, &'a SettingDef)> + 'a {
        self.defs
            .iter()
            .enumerate()
            .filter(move |(_, d)| d.namespace == namespace)
            .map(|(i, d)| (SettingId(i as u32), d))
    }
    /// Validates and normalizes a value according to type and scalar/collection constraints.
    pub fn validate_value(&self, id: SettingId, value: Value) -> Result<Value, SettingsError> {
        let def = self.def(id).ok_or(SettingsError::InvalidId(id))?;
        if value.value_type() != def.ty {
            return Err(SettingsError::TypeMismatch {
                key: def.key.clone(),
                expected: def.ty,
                actual: value.value_type(),
            });
        }
        validate_value(def, value)
    }
    /// Returns the alias map as canonical alias-to-key pairs.
    pub fn aliases(&self) -> &BTreeMap<SettingKey, SettingKey> {
        &self.aliases
    }
}

fn valid_key(key: &str) -> bool {
    // The design's examples include two-component keys (for example
    // `audio.master_volume`) despite its later three-component grammar.
    key.len() <= 64
        && key.split('.').count() >= 2
        && key.split('.').all(|part| {
            !part.is_empty()
                && part
                    .bytes()
                    .all(|b| b.is_ascii_lowercase() || b.is_ascii_digit() || b == b'_' || b == b'-')
        })
}

fn validate_def(def: &SettingDef, errors: &mut Vec<SettingsError>) {
    let invalid = |reason: &str| SettingsError::InvalidDefinition {
        key: def.key.clone(),
        reason: reason.to_string(),
    };
    if def.default.value_type() != def.ty {
        errors.push(SettingsError::TypeMismatch {
            key: def.key.clone(),
            expected: def.ty,
            actual: def.default.value_type(),
        });
    }
    if !def.shape.is_valid() {
        errors.push(invalid(
            "value shape is unsupported or has inverted array bounds",
        ));
    }
    if def.shape.value_type() != def.ty {
        errors.push(SettingsError::TypeMismatch {
            key: def.key.clone(),
            expected: def.ty,
            actual: def.shape.value_type(),
        });
    }
    if def.constraints.min.is_some_and(|v| !v.is_finite())
        || def.constraints.max.is_some_and(|v| !v.is_finite())
        || def
            .constraints
            .min
            .zip(def.constraints.max)
            .is_some_and(|(a, b)| a > b)
    {
        errors.push(invalid("numeric bounds must be finite and min <= max"));
    }
    let numeric_leaf_type = shape_numeric_leaf_type(&def.shape);
    if numeric_leaf_type == Some(ValueType::Int) {
        let valid_integer_bound = |bound: f64| {
            bound >= i64::MIN as f64 && bound < i64::MAX as f64 && (bound as i64) as f64 == bound
        };
        if def.constraints.min.is_some_and(|v| !valid_integer_bound(v))
            || def.constraints.max.is_some_and(|v| !valid_integer_bound(v))
        {
            errors.push(invalid(
                "integer bounds must be exactly representable i64 values",
            ));
        }
        if def
            .constraints
            .min
            .zip(def.constraints.max)
            .is_some_and(|(a, b)| a > b)
        {
            errors.push(invalid("integer bounds contain no representable value"));
        }
    }
    if let Value::Float(v) = def.default
        && !v.is_finite()
    {
        errors.push(invalid("float default must be finite"));
    }
    if def.flags.bootstrap
        && !matches!(
            def.ty,
            ValueType::Bool | ValueType::Int | ValueType::Float | ValueType::Enum
        )
    {
        errors.push(invalid("bootstrap values must be scalar"));
    }
    let shape_has_enum = shape_contains_type(&def.shape, ValueType::Enum);
    if !def.constraints.enum_values.is_empty() && !shape_has_enum {
        errors.push(invalid("enum_values require Enum type"));
    }
    if !def.enum_options.is_empty() && !shape_has_enum {
        errors.push(invalid("enum_options require Enum type"));
    }
    let mut option_ids = BTreeSet::new();
    let mut option_names = BTreeSet::new();
    for option in &def.enum_options {
        if option.stable_name.is_empty() {
            errors.push(invalid("enum option stable names must not be empty"));
        }
        if option.id != EnumId::from_stable_name(&option.stable_name) {
            errors.push(invalid("enum option ID does not match its stable name"));
        }
        if !option_ids.insert(option.id) {
            errors.push(invalid("enum option IDs collide"));
        }
        if !option_names.insert(option.stable_name.as_str()) {
            errors.push(invalid("enum option stable names must be unique"));
        }
        if !def.constraints.enum_values.is_empty()
            && !def.constraints.enum_values.contains(&option.id)
        {
            errors.push(invalid(
                "enum option conflicts with the enum_values constraint",
            ));
        }
    }
    if let Value::Enum(e) = def.default
        && ((!def.constraints.enum_values.is_empty() && !def.constraints.enum_values.contains(&e))
            || (!def.enum_options.is_empty() && !option_ids.contains(&e)))
    {
        errors.push(invalid("default enum is not allowed"));
    }
    match validate_value(def, def.default.clone()) {
        Ok(normalized) if normalized != def.default => {
            errors.push(invalid("default requires clamping"))
        }
        Ok(_) => {}
        Err(error) => errors.push(error),
    }
}

fn validate_value(def: &SettingDef, value: Value) -> Result<Value, SettingsError> {
    let key = def.key.clone();
    if !def.shape.is_valid() {
        return Err(SettingsError::InvalidDefinition {
            key,
            reason: String::from("value shape is unsupported or exceeds the maximum nesting depth"),
        });
    }
    let mut value = value;
    validate_shape(def, &key, &def.shape, &mut value)?;
    if let Value::Float(v) = value {
        if !v.is_finite() {
            return Err(SettingsError::OutOfRange {
                key,
                value: Value::Float(v),
            });
        }
        value = Value::Float(v);
    }
    if let Value::Int(n) = value {
        let minimum = def.constraints.min.map(|v| v as i64);
        let maximum = def.constraints.max.map(|v| v as i64);
        let low = minimum.is_some_and(|bound| n < bound);
        let high = maximum.is_some_and(|bound| n > bound);
        if low || high {
            if def.on_violation == crate::ViolationPolicy::Reject {
                return Err(SettingsError::OutOfRange {
                    key,
                    value: Value::Int(n),
                });
            }
            value = Value::Int(
                n.max(minimum.unwrap_or(i64::MIN))
                    .min(maximum.unwrap_or(i64::MAX)),
            );
        } else {
            value = Value::Int(n);
        }
    } else if let Value::Float(n) = value {
        let low = def.constraints.min.is_some_and(|bound| n < bound);
        let high = def.constraints.max.is_some_and(|bound| n > bound);
        if low || high {
            if def.on_violation == crate::ViolationPolicy::Reject {
                return Err(SettingsError::OutOfRange {
                    key,
                    value: Value::Float(n),
                });
            }
            value = Value::Float(
                n.max(def.constraints.min.unwrap_or(f64::NEG_INFINITY))
                    .min(def.constraints.max.unwrap_or(f64::INFINITY)),
            );
        } else {
            value = Value::Float(n);
        }
    }
    match &value {
        Value::String(s)
            if def
                .constraints
                .max_string_bytes
                .is_some_and(|max| s.len() > max) =>
        {
            return Err(SettingsError::TooLong { key })
        }
        Value::Array(a)
            if def
                .constraints
                .max_array_len
                .is_some_and(|max| a.len() > max) =>
        {
            return Err(SettingsError::TooLong { key })
        }
        Value::Enum(e)
            if (!def.constraints.enum_values.is_empty()
                && !def.constraints.enum_values.contains(e))
                || (!def.enum_options.is_empty()
                    && !def.enum_options.iter().any(|option| option.id == *e)) =>
        {
            return Err(SettingsError::InvalidEnum { key, value: *e })
        }
        _ => {}
    }
    Ok(value)
}

fn validate_shape(
    def: &SettingDef,
    key: &SettingKey,
    shape: &crate::ValueShape,
    value: &mut Value,
) -> Result<(), SettingsError> {
    match shape {
        crate::ValueShape::Scalar(expected) => {
            if value.value_type() != *expected {
                return Err(SettingsError::TypeMismatch {
                    key: key.clone(),
                    expected: *expected,
                    actual: value.value_type(),
                });
            }
            match value {
                Value::Float(number) if !number.is_finite() => {
                    return Err(SettingsError::OutOfRange {
                        key: key.clone(),
                        value: value.clone(),
                    });
                }
                Value::Int(number) => {
                    let minimum = def.constraints.min.map(|bound| bound as i64);
                    let maximum = def.constraints.max.map(|bound| bound as i64);
                    let out_of_range = minimum.is_some_and(|bound| *number < bound)
                        || maximum.is_some_and(|bound| *number > bound);
                    if out_of_range {
                        if def.on_violation == crate::ViolationPolicy::Reject {
                            return Err(SettingsError::OutOfRange {
                                key: key.clone(),
                                value: value.clone(),
                            });
                        }
                        *number = (*number)
                            .max(minimum.unwrap_or(i64::MIN))
                            .min(maximum.unwrap_or(i64::MAX));
                    }
                }
                Value::Float(number) => {
                    let out_of_range = def.constraints.min.is_some_and(|bound| *number < bound)
                        || def.constraints.max.is_some_and(|bound| *number > bound);
                    if out_of_range {
                        if def.on_violation == crate::ViolationPolicy::Reject {
                            return Err(SettingsError::OutOfRange {
                                key: key.clone(),
                                value: value.clone(),
                            });
                        }
                        *number = number
                            .max(def.constraints.min.unwrap_or(f64::NEG_INFINITY))
                            .min(def.constraints.max.unwrap_or(f64::INFINITY));
                    }
                }
                Value::String(text)
                    if def
                        .constraints
                        .max_string_bytes
                        .is_some_and(|max| text.len() > max) =>
                {
                    return Err(SettingsError::TooLong { key: key.clone() });
                }
                Value::Enum(id)
                    if (!def.constraints.enum_values.is_empty()
                        && !def.constraints.enum_values.contains(id))
                        || (!def.enum_options.is_empty()
                            && !def.enum_options.iter().any(|option| option.id == *id)) =>
                {
                    return Err(SettingsError::InvalidEnum {
                        key: key.clone(),
                        value: *id,
                    });
                }
                _ => {}
            }
        }
        crate::ValueShape::Array {
            element,
            min_len,
            max_len,
        } => {
            let Value::Array(values) = value else {
                return Err(SettingsError::TypeMismatch {
                    key: key.clone(),
                    expected: ValueType::Array,
                    actual: value.value_type(),
                });
            };
            if values.len() < *min_len {
                return Err(SettingsError::TooShort { key: key.clone() });
            }
            if values.len() > *max_len {
                return Err(SettingsError::TooLong { key: key.clone() });
            }
            for value in values {
                validate_shape(def, key, element, value)?;
            }
        }
    }
    Ok(())
}

fn shape_numeric_leaf_type(shape: &crate::ValueShape) -> Option<ValueType> {
    match shape {
        crate::ValueShape::Scalar(ValueType::Int | ValueType::Float) => Some(shape.value_type()),
        crate::ValueShape::Scalar(_) => None,
        crate::ValueShape::Array { element, .. } => shape_numeric_leaf_type(element),
    }
}

fn shape_contains_type(shape: &crate::ValueShape, expected: ValueType) -> bool {
    match shape {
        crate::ValueShape::Scalar(actual) => *actual == expected,
        crate::ValueShape::Array { element, .. } => shape_contains_type(element, expected),
    }
}

#[cfg(test)]
mod recursive_leaf_validation_tests {
    use super::*;
    use crate::EnumOption;

    #[test]
    fn nested_float_and_enum_leaves_obey_scalar_validation() {
        let mut builder = RegistryBuilder::new();
        builder.namespace("render", "test").unwrap();

        let floats = SettingDef::new(
            "render.weights",
            "render",
            Value::Array(vec![Value::Array(vec![Value::Float(0.5)])]),
        );
        builder.register(floats);

        let low = EnumOption::from_stable_name("quality.low");
        let high = EnumOption::from_stable_name("quality.high");
        let mut enums = SettingDef::new(
            "render.quality_history",
            "render",
            Value::Array(vec![Value::Enum(low.id)]),
        );
        enums.enum_options = vec![low.clone(), high.clone()];
        builder.register(enums);
        let registry = builder.finish().unwrap();

        let floats_id = registry.id(&"render.weights".into()).unwrap();
        assert!(matches!(
            registry.validate_value(
                floats_id,
                Value::Array(vec![Value::Array(vec![Value::Float(f64::INFINITY)])]),
            ),
            Err(SettingsError::OutOfRange { .. })
        ));

        let enums_id = registry.id(&"render.quality_history".into()).unwrap();
        assert!(registry
            .validate_value(enums_id, Value::Array(vec![Value::Enum(high.id)]))
            .is_ok());
        assert!(matches!(
            registry.validate_value(
                enums_id,
                Value::Array(vec![Value::Enum(EnumId::from_stable_name(
                    "quality.unknown"
                ))]),
            ),
            Err(SettingsError::InvalidEnum { .. })
        ));
    }

    #[test]
    fn nested_numeric_leaves_obey_reject_and_clamp_policies() {
        let mut reject = SettingDef::new(
            "render.int_history",
            "render",
            Value::Array(vec![Value::Array(vec![Value::Int(4)])]),
        );
        reject.constraints.min = Some(0.0);
        reject.constraints.max = Some(10.0);
        reject.shape = crate::ValueShape::array(
            crate::ValueShape::array(crate::ValueShape::scalar(ValueType::Int), 1, 4),
            1,
            4,
        );

        let mut clamp = SettingDef::new(
            "render.float_history",
            "render",
            Value::Array(vec![Value::Array(vec![Value::Float(0.5)])]),
        );
        clamp.constraints.min = Some(0.0);
        clamp.constraints.max = Some(1.0);
        clamp.on_violation = crate::ViolationPolicy::Clamp;
        clamp.shape = crate::ValueShape::array(
            crate::ValueShape::array(crate::ValueShape::scalar(ValueType::Float), 1, 4),
            1,
            4,
        );

        let mut builder = RegistryBuilder::new();
        builder.namespace("render", "test-owner").unwrap();
        builder.register(reject);
        builder.register(clamp);
        let registry = builder.finish().unwrap();

        let int_id = registry.id(&"render.int_history".into()).unwrap();
        assert!(matches!(
            registry.validate_value(
                int_id,
                Value::Array(vec![Value::Array(vec![Value::Int(-1)])]),
            ),
            Err(SettingsError::OutOfRange { .. })
        ));

        let float_id = registry.id(&"render.float_history".into()).unwrap();
        assert_eq!(
            registry
                .validate_value(
                    float_id,
                    Value::Array(vec![Value::Array(vec![Value::Float(2.0)])]),
                )
                .unwrap(),
            Value::Array(vec![Value::Array(vec![Value::Float(1.0)])]),
        );
    }
}
