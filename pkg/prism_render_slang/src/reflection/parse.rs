//! Parsing Slang reflection JSON into the [`AbiModel`].
//!
//! Slang's reflection schema is deeply nested and somewhat irregular, so this
//! walks a [`serde_json::Value`] tree rather than deriving a rigid
//! `Deserialize`. Every uniform struct encountered anywhere in the parameter
//! tree is collected (deduplicated by name), which is what codegen consumes.

use serde_json::Value;

use super::model::{AbiModel, Field, FieldType, Scalar, StructLayout};
use crate::error::{SlangError, SlangResult};

/// Parse reflection JSON text into an [`AbiModel`].
pub fn parse_reflection(json: &str) -> SlangResult<AbiModel> {
    let root: Value = serde_json::from_str(json).map_err(|err| SlangError::ReflectionParse {
        detail: format!("invalid JSON: {err}"),
    })?;
    parse_value(&root)
}

/// Parse an already-deserialized reflection [`Value`].
pub fn parse_value(root: &Value) -> SlangResult<AbiModel> {
    let reflection_version = root
        .get("version")
        .and_then(Value::as_str)
        .unwrap_or("unknown")
        .to_string();

    let mut collector = StructCollector::default();

    if let Some(params) = root.get("parameters").and_then(Value::as_array) {
        for param in params {
            if let Some(ty) = param.get("type") {
                collector.walk_type(ty)?;
            }
        }
    }

    // Also honor a top-level `entryPoints[].parameters[]` shape if present.
    if let Some(entries) = root.get("entryPoints").and_then(Value::as_array) {
        for entry in entries {
            if let Some(params) = entry.get("parameters").and_then(Value::as_array) {
                for param in params {
                    if let Some(ty) = param.get("type") {
                        collector.walk_type(ty)?;
                    }
                }
            }
        }
    }

    Ok(AbiModel {
        reflection_version,
        structs: collector.into_sorted(),
    })
}

#[derive(Default)]
struct StructCollector {
    structs: Vec<StructLayout>,
}

impl StructCollector {
    fn into_sorted(mut self) -> Vec<StructLayout> {
        self.structs.sort_by(|a, b| a.name.cmp(&b.name));
        self.structs
    }

    fn already_have(&self, name: &str) -> bool {
        self.structs.iter().any(|s| s.name == name)
    }

    /// Descend through container types (constant/structured buffers, arrays)
    /// to reach and record any `struct` element types.
    fn walk_type(&mut self, ty: &Value) -> SlangResult<()> {
        let kind = ty.get("kind").and_then(Value::as_str).unwrap_or("");
        match kind {
            "struct" => self.record_struct(ty),
            "constantBuffer" | "parameterBlock" | "structuredBuffer"
            | "array" | "textureBuffer" => {
                if let Some(elem) = ty.get("elementType") {
                    self.walk_type(elem)?;
                }
                Ok(())
            }
            _ => Ok(()),
        }
    }

    fn record_struct(&mut self, ty: &Value) -> SlangResult<()> {
        let name = ty
            .get("name")
            .and_then(Value::as_str)
            .unwrap_or("Anonymous")
            .to_string();

        if self.already_have(&name) {
            return Ok(());
        }

        let (size, alignment) = uniform_size_alignment(ty);

        let mut fields = Vec::new();
        if let Some(field_arr) = ty.get("fields").and_then(Value::as_array) {
            for field in field_arr {
                // Recurse first so nested structs are recorded too.
                if let Some(fty) = field.get("type") {
                    self.walk_type(fty)?;
                }
                if let Some(parsed) = parse_field(field)? {
                    fields.push(parsed);
                }
            }
        }

        fields.sort_by_key(|f| f.offset);

        // Reserve the name slot up front would risk infinite recursion on
        // self-referential types; structs cannot contain themselves by value,
        // so recording after field recursion is safe and dedups correctly.
        if !self.already_have(&name) {
            self.structs.push(StructLayout {
                name,
                size,
                alignment,
                fields,
            });
        }
        Ok(())
    }
}

/// Extract `(size, alignment)` from a type's `sizes` array, preferring the
/// `uniform` layout entry.
fn uniform_size_alignment(ty: &Value) -> (u32, u32) {
    let Some(sizes) = ty.get("sizes").and_then(Value::as_array) else {
        return (0, 0);
    };
    let uniform = sizes
        .iter()
        .find(|s| s.get("kind").and_then(Value::as_str) == Some("uniform"))
        .or_else(|| sizes.first());
    match uniform {
        Some(entry) => {
            let size = entry.get("value").and_then(Value::as_u64).unwrap_or(0) as u32;
            let alignment = entry.get("alignment").and_then(Value::as_u64).unwrap_or(0) as u32;
            (size, alignment)
        }
        None => (0, 0),
    }
}

fn parse_field(field: &Value) -> SlangResult<Option<Field>> {
    let Some(name) = field.get("name").and_then(Value::as_str) else {
        return Ok(None);
    };
    let name = name.to_string();
    let Some(ty_value) = field.get("type") else {
        return Ok(None);
    };
    let ty = parse_field_type(ty_value)?;

    // Uniform fields carry their offset/size under `binding`.
    let binding = field.get("binding");
    let offset = binding
        .and_then(|b| b.get("offset"))
        .and_then(Value::as_u64)
        .unwrap_or(0) as u32;
    let size = binding
        .and_then(|b| b.get("size"))
        .and_then(Value::as_u64)
        .or_else(|| type_uniform_size(ty_value))
        .unwrap_or(0) as u32;

    // Non-uniform bindings (textures, samplers) have no uniform offset/size;
    // skip them so generated structs contain only the uniform block.
    let is_uniform = binding
        .and_then(|b| b.get("kind"))
        .and_then(Value::as_str)
        .map(|k| k == "uniform")
        .unwrap_or(false);
    if !is_uniform {
        return Ok(None);
    }

    Ok(Some(Field {
        name,
        ty,
        offset,
        size,
    }))
}

fn type_uniform_size(ty: &Value) -> Option<u64> {
    ty.get("sizes")
        .and_then(Value::as_array)
        .and_then(|sizes| {
            sizes
                .iter()
                .find(|s| s.get("kind").and_then(Value::as_str) == Some("uniform"))
                .or_else(|| sizes.first())
        })
        .and_then(|entry| entry.get("value").and_then(Value::as_u64))
}

fn parse_field_type(ty: &Value) -> SlangResult<FieldType> {
    let kind = ty.get("kind").and_then(Value::as_str).unwrap_or("");
    match kind {
        "scalar" => {
            let token = ty.get("scalarType").and_then(Value::as_str).unwrap_or("");
            match Scalar::from_slang(token) {
                Some(scalar) => Ok(FieldType::Scalar(scalar)),
                None => Ok(FieldType::Opaque {
                    kind: format!("scalar:{token}"),
                }),
            }
        }
        "vector" => {
            let count = ty
                .get("elementCount")
                .and_then(Value::as_u64)
                .unwrap_or(0) as u32;
            let elem = ty
                .get("elementType")
                .and_then(|e| e.get("scalarType"))
                .and_then(Value::as_str)
                .and_then(Scalar::from_slang)
                .unwrap_or(Scalar::F32);
            Ok(FieldType::Vector { elem, count })
        }
        "matrix" => {
            let rows = ty.get("rowCount").and_then(Value::as_u64).unwrap_or(0) as u32;
            let cols = ty.get("columnCount").and_then(Value::as_u64).unwrap_or(0) as u32;
            let elem = ty
                .get("elementType")
                .and_then(|e| e.get("scalarType"))
                .and_then(Value::as_str)
                .and_then(Scalar::from_slang)
                .unwrap_or(Scalar::F32);
            Ok(FieldType::Matrix { elem, rows, cols })
        }
        "array" => {
            let count = ty
                .get("elementCount")
                .and_then(Value::as_u64)
                .unwrap_or(0) as u32;
            let element = match ty.get("elementType") {
                Some(elem) => Box::new(parse_field_type(elem)?),
                None => Box::new(FieldType::Opaque {
                    kind: "array-element".to_string(),
                }),
            };
            Ok(FieldType::Array { element, count })
        }
        "struct" => {
            let name = ty
                .get("name")
                .and_then(Value::as_str)
                .unwrap_or("Anonymous")
                .to_string();
            Ok(FieldType::Struct { name })
        }
        other => Ok(FieldType::Opaque {
            kind: other.to_string(),
        }),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const SIMPLE: &str = r#"{
        "version": "1.1",
        "parameters": [
          {
            "name": "gParams",
            "type": {
              "kind": "constantBuffer",
              "elementType": {
                "kind": "struct",
                "name": "Params",
                "fields": [
                  { "name": "baseColor",
                    "type": {"kind":"vector","elementCount":3,
                             "elementType":{"kind":"scalar","scalarType":"float32"}},
                    "binding": {"kind":"uniform","offset":0,"size":12} },
                  { "name": "roughness",
                    "type": {"kind":"scalar","scalarType":"float32"},
                    "binding": {"kind":"uniform","offset":12,"size":4} }
                ],
                "sizes": [ {"kind":"uniform","value":16,"alignment":16} ]
              }
            }
          }
        ]
    }"#;

    #[test]
    fn parses_struct_and_fields() {
        let model = parse_reflection(SIMPLE).unwrap();
        assert_eq!(model.reflection_version, "1.1");
        let params = model.struct_by_name("Params").expect("Params present");
        assert_eq!(params.size, 16);
        assert_eq!(params.alignment, 16);
        assert_eq!(params.fields.len(), 2);
        assert_eq!(params.fields[0].name, "baseColor");
        assert_eq!(params.fields[0].offset, 0);
        assert_eq!(params.fields[0].size, 12);
        assert_eq!(
            params.fields[0].ty,
            FieldType::Vector {
                elem: Scalar::F32,
                count: 3
            }
        );
        assert_eq!(params.fields[1].name, "roughness");
        assert_eq!(params.fields[1].offset, 12);
        assert_eq!(params.fields[1].ty, FieldType::Scalar(Scalar::F32));
    }

    #[test]
    fn abi_version_changes_with_layout() {
        let model = parse_reflection(SIMPLE).unwrap();
        let v1 = model.abi_version();
        let shifted = SIMPLE.replace("\"offset\":12", "\"offset\":16");
        let model2 = parse_reflection(&shifted).unwrap();
        assert_ne!(v1, model2.abi_version());
    }

    #[test]
    fn non_uniform_bindings_are_skipped() {
        let json = r#"{
            "version":"1.1",
            "parameters":[
              {"name":"gTex","type":{"kind":"struct","name":"Mixed",
               "fields":[
                 {"name":"tex","type":{"kind":"resource"},
                  "binding":{"kind":"descriptorTableSlot","index":0}},
                 {"name":"scale","type":{"kind":"scalar","scalarType":"float32"},
                  "binding":{"kind":"uniform","offset":0,"size":4}}
               ],
               "sizes":[{"kind":"uniform","value":4,"alignment":4}]}}
            ]
        }"#;
        let model = parse_reflection(json).unwrap();
        let mixed = model.struct_by_name("Mixed").unwrap();
        assert_eq!(mixed.fields.len(), 1);
        assert_eq!(mixed.fields[0].name, "scale");
    }
}
