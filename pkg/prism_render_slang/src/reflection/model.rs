//! The parsed, backend-neutral ABI model extracted from Slang reflection JSON.
//!
//! This is the single source of truth that both the GPU shader and the
//! generated Rust `#[repr(C)]` bindings are derived from, so their layouts
//! cannot drift.

use crate::hash::Fnv1a;

/// A scalar element kind, mapped from Slang's `scalarType`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Scalar {
    /// 32-bit float.
    F32,
    /// 16-bit float.
    F16,
    /// 64-bit float.
    F64,
    /// 32-bit signed integer.
    I32,
    /// 32-bit unsigned integer.
    U32,
    /// 16-bit signed integer.
    I16,
    /// 16-bit unsigned integer.
    U16,
    /// 8-bit signed integer.
    I8,
    /// 8-bit unsigned integer.
    U8,
    /// 64-bit signed integer.
    I64,
    /// 64-bit unsigned integer.
    U64,
    /// Boolean (32-bit on GPU).
    Bool,
}

impl Scalar {
    /// Parse a Slang `scalarType` token.
    pub fn from_slang(token: &str) -> Option<Self> {
        Some(match token {
            "float32" | "float" => Scalar::F32,
            "float16" | "half" => Scalar::F16,
            "float64" | "double" => Scalar::F64,
            "int32" | "int" => Scalar::I32,
            "uint32" | "uint" => Scalar::U32,
            "int16" => Scalar::I16,
            "uint16" => Scalar::U16,
            "int8" => Scalar::I8,
            "uint8" => Scalar::U8,
            "int64" => Scalar::I64,
            "uint64" => Scalar::U64,
            "bool" => Scalar::Bool,
            _ => return None,
        })
    }

    /// The corresponding Rust primitive type name for generated bindings.
    ///
    /// Booleans become `u32` because that is their GPU representation.
    pub fn rust_primitive(self) -> &'static str {
        match self {
            Scalar::F32 => "f32",
            // No stable f16 primitive; f16 is stored as raw bits like u16.
            Scalar::F16 | Scalar::U16 => "u16",
            Scalar::F64 => "f64",
            Scalar::I32 => "i32",
            Scalar::U32 | Scalar::Bool => "u32",
            Scalar::I16 => "i16",
            Scalar::I8 => "i8",
            Scalar::U8 => "u8",
            Scalar::I64 => "i64",
            Scalar::U64 => "u64",
        }
    }

    /// A short, stable token used when hashing the ABI.
    pub fn stable_token(self) -> &'static str {
        match self {
            Scalar::F32 => "f32",
            Scalar::F16 => "f16",
            Scalar::F64 => "f64",
            Scalar::I32 => "i32",
            Scalar::U32 => "u32",
            Scalar::I16 => "i16",
            Scalar::U16 => "u16",
            Scalar::I8 => "i8",
            Scalar::U8 => "u8",
            Scalar::I64 => "i64",
            Scalar::U64 => "u64",
            Scalar::Bool => "bool",
        }
    }
}

/// The logical type of a struct field, independent of its byte layout.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FieldType {
    /// A single scalar.
    Scalar(Scalar),
    /// A fixed-length vector (for example `float3`).
    Vector {
        /// Element scalar.
        elem: Scalar,
        /// Component count (2, 3, or 4).
        count: u32,
    },
    /// A column-major matrix (for example `float4x4`).
    Matrix {
        /// Element scalar.
        elem: Scalar,
        /// Row count.
        rows: u32,
        /// Column count.
        cols: u32,
    },
    /// A nested named struct.
    Struct {
        /// The nested struct's name.
        name: String,
    },
    /// A fixed-size array of another field type.
    Array {
        /// Element type.
        element: Box<FieldType>,
        /// Number of elements.
        count: u32,
    },
    /// A type the parser recognized structurally but does not model precisely;
    /// layout is still honored via explicit padding in codegen.
    Opaque {
        /// The Slang `kind` token, for diagnostics.
        kind: String,
    },
}

impl FieldType {
    /// A stable textual token used for ABI hashing.
    pub fn stable_token(&self) -> String {
        match self {
            FieldType::Scalar(s) => s.stable_token().to_string(),
            FieldType::Vector { elem, count } => {
                format!("vec{count}<{}>", elem.stable_token())
            }
            FieldType::Matrix { elem, rows, cols } => {
                format!("mat{rows}x{cols}<{}>", elem.stable_token())
            }
            FieldType::Struct { name } => format!("struct:{name}"),
            FieldType::Array { element, count } => {
                format!("array<{},{count}>", element.stable_token())
            }
            FieldType::Opaque { kind } => format!("opaque:{kind}"),
        }
    }
}

/// A single field within a uniform struct, with its byte-exact placement.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Field {
    /// Field name as declared in Slang.
    pub name: String,
    /// Logical field type.
    pub ty: FieldType,
    /// Byte offset within the containing struct.
    pub offset: u32,
    /// Byte size occupied by this field.
    pub size: u32,
}

/// The byte-exact layout of a uniform struct.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StructLayout {
    /// Struct name as declared in Slang.
    pub name: String,
    /// Total size in bytes (including tail padding).
    pub size: u32,
    /// Required alignment in bytes.
    pub alignment: u32,
    /// Fields in declaration order, sorted by offset.
    pub fields: Vec<Field>,
}

/// The complete ABI model reflected from a compiled Slang module.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AbiModel {
    /// Reflection format version reported by Slang.
    pub reflection_version: String,
    /// All uniform struct layouts discovered, deduplicated by name.
    pub structs: Vec<StructLayout>,
}

impl AbiModel {
    /// Look up a struct layout by name.
    pub fn struct_by_name(&self, name: &str) -> Option<&StructLayout> {
        self.structs.iter().find(|s| s.name == name)
    }

    /// A deterministic, content-derived ABI version.
    ///
    /// Any change to struct names, field names, field types, offsets, sizes,
    /// or overall struct size/alignment changes this value, so a mismatch
    /// between GPU and CPU bindings is detectable at load time.
    pub fn abi_version(&self) -> u64 {
        let mut h = Fnv1a::new();
        h.write_framed("prism.slang.abi.v1");
        // Sort by name for a canonical order independent of parse order.
        let mut names: Vec<&StructLayout> = self.structs.iter().collect();
        names.sort_by(|a, b| a.name.cmp(&b.name));
        for layout in names {
            h.write_framed(&layout.name);
            h.write_u32(layout.size);
            h.write_u32(layout.alignment);
            for field in &layout.fields {
                h.write_framed(&field.name);
                h.write_framed(&field.ty.stable_token());
                h.write_u32(field.offset);
                h.write_u32(field.size);
            }
        }
        h.finish()
    }
}
