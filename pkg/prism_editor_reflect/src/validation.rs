//! Reflection metadata validation with paths attached to every issue.

use crate::FieldPath;
use prism_reflect::{Access, Reflect, ReflectRef, TypeMetadata, TypeRegistry};
use std::fmt;

/// One validation problem associated with a reflected field path.
#[derive(Clone, Debug, PartialEq)]
pub struct ValidationIssue {
    /// The path of the invalid field, or the nearest addressable parent.
    pub path: FieldPath,
    /// The concrete validation failure.
    pub kind: ValidationIssueKind,
}

/// Validation failures produced from reflection metadata.
#[derive(Clone, Debug, PartialEq)]
pub enum ValidationIssueKind {
    /// A required field is absent from a dynamic struct.
    MissingRequiredField { field: String },
    /// A numeric field is outside its inclusive metadata range.
    OutOfRange { value: f64, min: f64, max: f64 },
    /// A range was attached to a non-numeric field.
    NotNumeric,
    /// A metadata range is malformed and cannot be enforced.
    InvalidRange { min: f64, max: f64 },
}

impl fmt::Display for ValidationIssueKind {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::MissingRequiredField { field } => {
                write!(formatter, "required field `{field}` is missing")
            }
            Self::OutOfRange { value, min, max } => {
                write!(formatter, "value {value} is outside range {min}..={max}")
            }
            Self::NotNumeric => formatter.write_str("range-constrained field is not numeric"),
            Self::InvalidRange { min, max } => {
                write!(formatter, "metadata range {min}..={max} is invalid")
            }
        }
    }
}

/// All validation issues found in one reflected root.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct ValidationReport {
    issues: Vec<ValidationIssue>,
}

impl ValidationReport {
    /// Build a report from collected issues.
    #[must_use]
    pub fn new(issues: Vec<ValidationIssue>) -> Self {
        Self { issues }
    }

    /// Return every issue in deterministic traversal order.
    #[must_use]
    pub fn issues(&self) -> &[ValidationIssue] {
        &self.issues
    }

    /// Whether the report contains no issues.
    #[must_use]
    pub fn is_valid(&self) -> bool {
        self.issues.is_empty()
    }

    /// Number of issues in the report.
    #[must_use]
    pub fn len(&self) -> usize {
        self.issues.len()
    }

    /// Whether the report contains no issues.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.issues.is_empty()
    }
}

impl fmt::Display for ValidationReport {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.issues.is_empty() {
            return formatter.write_str("validation succeeded");
        }
        for (index, issue) in self.issues.iter().enumerate() {
            if index != 0 {
                formatter.write_str("; ")?;
            }
            write!(formatter, "{}: {}", issue.path, issue.kind)?;
        }
        Ok(())
    }
}

/// Validate a reflected root using all registered `TypeMetadata` records.
#[must_use]
pub fn validate(root: &dyn Reflect, registry: &TypeRegistry) -> ValidationReport {
    let mut issues = Vec::new();
    visit(root, &FieldPath::root(), registry, &mut issues);
    ValidationReport::new(issues)
}

fn visit(
    value: &dyn Reflect,
    path: &FieldPath,
    registry: &TypeRegistry,
    issues: &mut Vec<ValidationIssue>,
) {
    if let Some(metadata) = metadata(value, registry) {
        validate_struct_metadata(value, path, metadata, issues);
    }

    match value.reflect_ref() {
        ReflectRef::Struct(structure) => {
            for index in 0..structure.field_count() {
                let (Some(name), Some(field)) =
                    (structure.name_at(index), structure.field_at(index))
                else {
                    continue;
                };
                let child_path = path.with_access(Access::Field(name.to_string())).ok();
                visit(field, child_path.as_ref().unwrap_or(path), registry, issues);
            }
        }
        ReflectRef::TupleStruct(tuple) => {
            for index in 0..tuple.field_count() {
                if let Some(field) = tuple.field(index) {
                    let child_path = path.with_access(Access::TupleIndex(index)).ok();
                    visit(field, child_path.as_ref().unwrap_or(path), registry, issues);
                }
            }
        }
        ReflectRef::Enum(enum_value) => {
            for index in 0..enum_value.field_count() {
                if let Some(field) = enum_value.field_at(index) {
                    let child_path = path.with_access(Access::TupleIndex(index)).ok();
                    visit(field, child_path.as_ref().unwrap_or(path), registry, issues);
                }
            }
        }
        ReflectRef::List(list) => {
            for index in 0..list.len() {
                if let Some(field) = list.get(index) {
                    let child_path = path.with_access(Access::ListIndex(index)).ok();
                    visit(field, child_path.as_ref().unwrap_or(path), registry, issues);
                }
            }
        }
        ReflectRef::Array(array) => {
            for index in 0..array.len() {
                if let Some(field) = array.get(index) {
                    let child_path = path.with_access(Access::ListIndex(index)).ok();
                    visit(field, child_path.as_ref().unwrap_or(path), registry, issues);
                }
            }
        }
        ReflectRef::Map(map) => {
            for (key, field) in map.iter_reflect() {
                let child_path = key
                    .downcast_ref::<String>()
                    .and_then(|key| path.with_access(Access::Key(key.clone())).ok());
                visit(field, child_path.as_ref().unwrap_or(path), registry, issues);
            }
        }
        ReflectRef::Set(set) => {
            for field in set.iter_reflect() {
                visit(field, path, registry, issues);
            }
        }
        ReflectRef::Value(_) => {}
        _ => {}
    }
}

fn validate_struct_metadata(
    value: &dyn Reflect,
    path: &FieldPath,
    metadata: &TypeMetadata,
    issues: &mut Vec<ValidationIssue>,
) {
    let ReflectRef::Struct(structure) = value.reflect_ref() else {
        return;
    };
    for field_metadata in metadata.fields() {
        let field_path = path
            .with_access(Access::Field(field_metadata.name().to_string()))
            .unwrap_or_else(|_| path.clone());
        let Some(field) = structure.field(field_metadata.name()) else {
            if field_metadata.is_required() {
                issues.push(ValidationIssue {
                    path: field_path,
                    kind: ValidationIssueKind::MissingRequiredField {
                        field: field_metadata.name().to_string(),
                    },
                });
            }
            continue;
        };

        let Some((min, max)) = field_metadata.range() else {
            continue;
        };
        if min.partial_cmp(&max).is_none_or(core::cmp::Ordering::is_gt) {
            issues.push(ValidationIssue {
                path: field_path,
                kind: ValidationIssueKind::InvalidRange { min, max },
            });
            continue;
        }
        match numeric_value(field) {
            Some(value) if value < min || value > max => issues.push(ValidationIssue {
                path: field_path,
                kind: ValidationIssueKind::OutOfRange { value, min, max },
            }),
            Some(_) => {}
            None => issues.push(ValidationIssue {
                path: field_path,
                kind: ValidationIssueKind::NotNumeric,
            }),
        }
    }
}

fn metadata<'a>(value: &dyn Reflect, registry: &'a TypeRegistry) -> Option<&'a TypeMetadata> {
    registry
        .get_with_name(value.type_name())
        .and_then(|registration| registration.data::<TypeMetadata>())
}

fn numeric_value(value: &dyn Reflect) -> Option<f64> {
    let any = value.as_any();
    macro_rules! read {
        ($($ty:ty),* $(,)?) => {
            $(if let Some(value) = any.downcast_ref::<$ty>() {
                return Some((*value) as f64);
            })*
        };
    }
    read!(i8, i16, i32, i64, i128, isize, u8, u16, u32, u64, u128, usize, f32, f64);
    None
}
