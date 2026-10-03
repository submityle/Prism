//! Error types for reflection-driven serialization and deserialization.
//!
//! Serialization only fails when a leaf (`Value`-kind) type is encountered that
//! the format layer does not know how to encode. Deserialization has a wider
//! surface because it reads untrusted bytes/text and reconstructs a typed value
//! against a [`TypeRegistry`](crate::TypeRegistry): the stream header, every
//! tag, and every leaf payload are validated against the target schema so that
//! corruption or a schema mismatch is reported rather than silently accepted.

use std::string::String;

/// An error raised while serializing a `&dyn Reflect` value.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum SerializeError {
    /// A leaf value had a concrete type the serializer cannot encode.
    ///
    /// The reflected subset the serializer supports is the built-in leaf set
    /// (`bool`, `char`, every fixed/pointer-width integer, `f32`/`f64`, and
    /// `String`). A `Value`-kind type outside that set (an opaque leaf) cannot
    /// be written to the self-describing stream.
    UnsupportedLeaf {
        /// The offending leaf's fully-qualified type name.
        type_name: &'static str,
    },
}

impl ::core::fmt::Display for SerializeError {
    fn fmt(&self, f: &mut ::core::fmt::Formatter<'_>) -> ::core::fmt::Result {
        match self {
            SerializeError::UnsupportedLeaf { type_name } => write!(
                f,
                "cannot serialize unsupported leaf value of type `{type_name}`"
            ),
        }
    }
}

impl ::std::error::Error for SerializeError {}

/// An error raised while deserializing a reflected value from a stream.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum DeserializeError {
    /// The input ended before a required byte/character could be read.
    UnexpectedEof,
    /// The binary header did not start with the expected magic bytes.
    BadMagic,
    /// The binary header carried a format version this build cannot read.
    UnsupportedVersion(u8),
    /// The stream's root [`StableTypeId`](crate::StableTypeId) did not match
    /// the id of the target type requested by the caller.
    StableIdMismatch {
        /// The id derived from the caller's target type.
        expected: u64,
        /// The id read from the stream.
        found: u64,
    },
    /// A leaf primitive tag byte was outside the known range.
    UnknownPrimitiveTag(u8),
    /// A node tag byte was outside the known range.
    UnknownNodeTag(u8),
    /// A node's kind tag disagreed with the kind the schema expects here.
    KindMismatch {
        /// The kind the schema expected.
        expected: &'static str,
        /// The kind the stream actually encoded.
        found: &'static str,
    },
    /// A type name referenced by the schema was not found in the registry.
    UnregisteredType(String),
    /// A leaf value's primitive tag disagreed with the schema's leaf type.
    LeafTypeMismatch {
        /// The leaf type name the schema expected.
        expected: &'static str,
    },
    /// A composite node declared a field count the schema cannot satisfy.
    FieldCountMismatch,
    /// An enum node referenced a variant index/name the schema does not define.
    UnknownVariant,
    /// A struct/struct-variant node referenced an unknown field name.
    UnknownField(String),
    /// A `char` leaf held a value that is not a Unicode scalar value.
    InvalidChar(u32),
    /// A `String` leaf held bytes that are not valid UTF-8.
    InvalidUtf8,
    /// Bytes/characters remained after the root value was fully decoded.
    TrailingData,
    /// The RON text was malformed; the string describes what was expected.
    RonSyntax(String),
}

impl ::core::fmt::Display for DeserializeError {
    fn fmt(&self, f: &mut ::core::fmt::Formatter<'_>) -> ::core::fmt::Result {
        match self {
            DeserializeError::UnexpectedEof => write!(f, "unexpected end of input"),
            DeserializeError::BadMagic => write!(f, "bad binary magic header"),
            DeserializeError::UnsupportedVersion(v) => {
                write!(f, "unsupported binary format version {v}")
            }
            DeserializeError::StableIdMismatch { expected, found } => write!(
                f,
                "stable type id mismatch: expected {expected:#018x}, found {found:#018x}"
            ),
            DeserializeError::UnknownPrimitiveTag(tag) => {
                write!(f, "unknown leaf primitive tag {tag}")
            }
            DeserializeError::UnknownNodeTag(tag) => write!(f, "unknown node tag {tag}"),
            DeserializeError::KindMismatch { expected, found } => {
                write!(f, "kind mismatch: expected {expected}, found {found}")
            }
            DeserializeError::UnregisteredType(name) => {
                write!(f, "type `{name}` is not registered in the type registry")
            }
            DeserializeError::LeafTypeMismatch { expected } => {
                write!(f, "leaf value does not match expected type `{expected}`")
            }
            DeserializeError::FieldCountMismatch => write!(f, "composite field count mismatch"),
            DeserializeError::UnknownVariant => write!(f, "unknown enum variant"),
            DeserializeError::UnknownField(name) => write!(f, "unknown field `{name}`"),
            DeserializeError::InvalidChar(code) => {
                write!(f, "invalid Unicode scalar value {code:#x}")
            }
            DeserializeError::InvalidUtf8 => write!(f, "invalid UTF-8 in string leaf"),
            DeserializeError::TrailingData => write!(f, "trailing data after root value"),
            DeserializeError::RonSyntax(msg) => write!(f, "RON syntax error: {msg}"),
        }
    }
}

impl ::std::error::Error for DeserializeError {}
