//! A self-contained RON (Rusty Object Notation) text back-end.
//!
//! The writer is a thin [`Encoder`] over a [`String`] with a delimiter stack:
//! structural `begin_*`/`end_*` callbacks push and close the matching RON
//! delimiters (`(` `)` for structs/tuples/variants, `[` `]` for list/array/set,
//! `{` `}` for maps) while the `before_*` boundary callbacks emit the field
//! separators (`,`), the field labels (`name:`), and the map `key: value`
//! colon. Because reconstruction is schema-guided (design §10, §22), the
//! emitted RON is *anonymous*: struct/variant type names are not written, only
//! the shape and the leaf values, which keeps the text compact and lossless for
//! the reflected subset.
//!
//! The reader is a schema-guided recursive-descent parser over the characters
//! of the input. The target [`TypeInfo`] (resolved for nested children through
//! the [`TypeRegistry`]) decides, at every position, which RON production to
//! expect, so the parser never has to guess a value's type from its syntax
//! alone — it only has to validate that the text matches the schema and
//! rebuild the matching [`Dynamic*`](crate::dynamic) node.

use crate::kinds::VariantType;
use crate::reflect::Reflect;
use crate::ser::de::{resolve, root_schema, Schema};
use crate::ser::encode::{serialize_value, Encoder};
use crate::ser::error::{DeserializeError, SerializeError};
use crate::ser::primitive::Primitive;
use crate::type_info::{TypeInfo, VariantKind};
use crate::{
    DynamicArray, DynamicEnum, DynamicList, DynamicMap, DynamicSet, DynamicStruct,
    DynamicTupleStruct, DynamicVariant, TypeRegistry,
};
use alloc::boxed::Box;
use alloc::string::{String, ToString};
use alloc::vec::Vec;

/// A single open composite scope in the RON writer.
///
/// `first` tracks whether the next child is the scope's first element (so the
/// separating comma is suppressed before it); `delimited` records whether the
/// scope opened a bracket that `end_*` must close (unit enum variants open no
/// bracket).
struct Frame {
    first: bool,
    delimited: bool,
}

/// An [`Encoder`] that appends anonymous, compact RON text to an owned string.
struct RonEncoder {
    out: String,
    stack: Vec<Frame>,
}

impl RonEncoder {
    fn new() -> Self {
        Self {
            out: String::new(),
            stack: Vec::new(),
        }
    }

    /// Open a composite scope with the given opening delimiter.
    fn open(&mut self, delimiter: char) {
        self.out.push(delimiter);
        self.stack.push(Frame {
            first: true,
            delimited: true,
        });
    }

    /// Close the current composite scope with the given closing delimiter.
    fn close(&mut self, delimiter: char) {
        if let Some(frame) = self.stack.pop()
            && frame.delimited
        {
            self.out.push(delimiter);
        }
    }

    /// Emit the inter-element separator unless this is the scope's first child.
    fn separator(&mut self) {
        if let Some(frame) = self.stack.last_mut() {
            if frame.first {
                frame.first = false;
            } else {
                self.out.push(',');
            }
        }
    }

    /// Append a floating-point literal, forcing a fractional form for integers.
    fn push_float(&mut self, text: &str) {
        self.out.push_str(text);
        let fractional =
            text.contains(['.', 'e', 'E']) || text.contains("inf") || text.contains("NaN");
        if !fractional {
            self.out.push_str(".0");
        }
    }

    /// Append a RON string literal with the supported escapes.
    fn push_string(&mut self, value: &str) {
        self.out.push('"');
        for ch in value.chars() {
            self.push_escaped(ch);
        }
        self.out.push('"');
    }

    /// Append a RON char literal with the supported escapes.
    fn push_char(&mut self, value: char) {
        self.out.push('\'');
        if value == '\'' {
            self.out.push_str("\\'");
        } else {
            self.push_escaped(value);
        }
        self.out.push('\'');
    }

    /// Append one character, escaping the control/backslash/quote set.
    fn push_escaped(&mut self, ch: char) {
        match ch {
            '"' => self.out.push_str("\\\""),
            '\\' => self.out.push_str("\\\\"),
            '\n' => self.out.push_str("\\n"),
            '\r' => self.out.push_str("\\r"),
            '\t' => self.out.push_str("\\t"),
            other => self.out.push(other),
        }
    }
}

impl Encoder for RonEncoder {
    fn encode_bool(&mut self, value: bool) {
        self.out.push_str(if value { "true" } else { "false" });
    }
    fn encode_char(&mut self, value: char) {
        self.push_char(value);
    }
    fn encode_i8(&mut self, value: i8) {
        self.out.push_str(&value.to_string());
    }
    fn encode_i16(&mut self, value: i16) {
        self.out.push_str(&value.to_string());
    }
    fn encode_i32(&mut self, value: i32) {
        self.out.push_str(&value.to_string());
    }
    fn encode_i64(&mut self, value: i64) {
        self.out.push_str(&value.to_string());
    }
    fn encode_i128(&mut self, value: i128) {
        self.out.push_str(&value.to_string());
    }
    fn encode_isize(&mut self, value: isize) {
        self.out.push_str(&value.to_string());
    }
    fn encode_u8(&mut self, value: u8) {
        self.out.push_str(&value.to_string());
    }
    fn encode_u16(&mut self, value: u16) {
        self.out.push_str(&value.to_string());
    }
    fn encode_u32(&mut self, value: u32) {
        self.out.push_str(&value.to_string());
    }
    fn encode_u64(&mut self, value: u64) {
        self.out.push_str(&value.to_string());
    }
    fn encode_u128(&mut self, value: u128) {
        self.out.push_str(&value.to_string());
    }
    fn encode_usize(&mut self, value: usize) {
        self.out.push_str(&value.to_string());
    }
    fn encode_f32(&mut self, value: f32) {
        let text = value.to_string();
        self.push_float(&text);
    }
    fn encode_f64(&mut self, value: f64) {
        let text = value.to_string();
        self.push_float(&text);
    }
    fn encode_str(&mut self, value: &str) {
        self.push_string(value);
    }

    fn begin_struct(&mut self, _count: usize) {
        self.open('(');
    }
    fn before_struct_field(&mut self, name: &str, _index: usize) {
        self.separator();
        self.out.push_str(name);
        self.out.push(':');
    }
    fn end_struct(&mut self) {
        self.close(')');
    }

    fn begin_tuple_struct(&mut self, _count: usize) {
        self.open('(');
    }
    fn before_tuple_struct_field(&mut self, _index: usize) {
        self.separator();
    }
    fn end_tuple_struct(&mut self) {
        self.close(')');
    }

    fn begin_enum(
        &mut self,
        _variant_index: usize,
        variant_name: &str,
        variant_type: VariantType,
        _count: usize,
    ) {
        self.out.push_str(variant_name);
        if matches!(variant_type, VariantType::Unit) {
            // A unit variant has no payload bracket; push a non-delimited frame
            // so the matching `end_enum` is still balanced.
            self.stack.push(Frame {
                first: true,
                delimited: false,
            });
        } else {
            self.open('(');
        }
    }
    fn before_enum_tuple_field(&mut self, _index: usize) {
        self.separator();
    }
    fn before_enum_struct_field(&mut self, name: &str, _index: usize) {
        self.separator();
        self.out.push_str(name);
        self.out.push(':');
    }
    fn end_enum(&mut self) {
        self.close(')');
    }

    fn begin_list(&mut self, _len: usize) {
        self.open('[');
    }
    fn before_list_element(&mut self, _index: usize) {
        self.separator();
    }
    fn end_list(&mut self) {
        self.close(']');
    }

    fn begin_array(&mut self, _len: usize) {
        self.open('[');
    }
    fn before_array_element(&mut self, _index: usize) {
        self.separator();
    }
    fn end_array(&mut self) {
        self.close(']');
    }

    fn begin_set(&mut self, _len: usize) {
        self.open('[');
    }
    fn before_set_element(&mut self, _index: usize) {
        self.separator();
    }
    fn end_set(&mut self) {
        self.close(']');
    }

    fn begin_map(&mut self, _len: usize) {
        self.open('{');
    }
    fn before_map_key(&mut self, _index: usize) {
        self.separator();
    }
    fn before_map_value(&mut self, _index: usize) {
        self.out.push(':');
    }
    fn end_map(&mut self) {
        self.close('}');
    }
}

/// Serialize any reflected value to compact RON text.
///
/// The output is anonymous (no type names), carrying only the value's shape and
/// leaves; reconstruction with [`from_ron`] is driven by the target
/// [`TypeInfo`].
///
/// # Errors
/// Returns [`SerializeError::UnsupportedLeaf`] when the value (or a nested
/// element) is a `Value`-kind type outside the built-in leaf set.
pub fn to_ron(value: &dyn Reflect) -> Result<String, SerializeError> {
    let mut encoder = RonEncoder::new();
    serialize_value(value, &mut encoder)?;
    Ok(encoder.out)
}

/// Deserialize a reflected value from RON text against a target type.
///
/// The target [`TypeInfo`] (plus the `registry` for nested types) guides
/// reconstruction into a [`Dynamic*`](crate::dynamic) tree whose represented
/// type names are stamped from the schema, so the result round-trips back to
/// the original concrete value via [`FromReflect`](crate::FromReflect). Struct
/// and struct-variant fields may appear in any order; they are matched by name.
///
/// # Errors
/// Returns a [`DeserializeError`] for malformed RON ([`DeserializeError::RonSyntax`]),
/// an unregistered nested type, a leaf-type/shape mismatch, an unknown field or
/// variant, or trailing text after the root value.
pub fn from_ron(
    text: &str,
    registry: &TypeRegistry,
    target: &TypeInfo,
) -> Result<Box<dyn Reflect>, DeserializeError> {
    let mut parser = RonParser::new(text, registry);
    let schema = root_schema(target);
    let value = parser.parse_value(&schema)?;
    parser.skip_whitespace();
    if parser.at_end() {
        Ok(value)
    } else {
        Err(DeserializeError::TrailingData)
    }
}

/// A schema-guided recursive-descent reader over the RON input characters.
struct RonParser<'a> {
    chars: Vec<char>,
    pos: usize,
    registry: &'a TypeRegistry,
}

impl<'a> RonParser<'a> {
    fn new(text: &str, registry: &'a TypeRegistry) -> Self {
        Self {
            chars: text.chars().collect(),
            pos: 0,
            registry,
        }
    }

    /// Whether all input characters have been consumed.
    fn at_end(&self) -> bool {
        self.pos >= self.chars.len()
    }

    /// Peek at the current character without consuming it.
    fn peek(&self) -> Option<char> {
        self.chars.get(self.pos).copied()
    }

    /// Consume and return the current character.
    fn bump(&mut self) -> Option<char> {
        let ch = self.chars.get(self.pos).copied();
        if ch.is_some() {
            self.pos += 1;
        }
        ch
    }

    /// Skip ASCII whitespace between tokens.
    fn skip_whitespace(&mut self) {
        while let Some(ch) = self.peek() {
            if ch.is_whitespace() {
                self.pos += 1;
            } else {
                break;
            }
        }
    }

    /// Consume an expected delimiter, erroring with context otherwise.
    fn expect(&mut self, expected: char) -> Result<(), DeserializeError> {
        self.skip_whitespace();
        match self.peek() {
            Some(ch) if ch == expected => {
                self.pos += 1;
                Ok(())
            }
            Some(ch) => Err(DeserializeError::RonSyntax(alloc::format!(
                "expected `{expected}`, found `{ch}`"
            ))),
            None => Err(DeserializeError::RonSyntax(alloc::format!(
                "expected `{expected}`, found end of input"
            ))),
        }
    }

    /// Parse an identifier (`[A-Za-z_][A-Za-z0-9_]*`).
    fn parse_ident(&mut self) -> Result<String, DeserializeError> {
        self.skip_whitespace();
        let start = self.pos;
        if let Some(ch) = self.peek() {
            if ch.is_ascii_alphabetic() || ch == '_' {
                self.pos += 1;
            } else {
                return Err(DeserializeError::RonSyntax(alloc::format!(
                    "expected identifier, found `{ch}`"
                )));
            }
        } else {
            return Err(DeserializeError::RonSyntax(
                "expected identifier, found end of input".to_string(),
            ));
        }
        while let Some(ch) = self.peek() {
            if ch.is_ascii_alphanumeric() || ch == '_' {
                self.pos += 1;
            } else {
                break;
            }
        }
        Ok(self.chars[start..self.pos].iter().collect())
    }

    /// Collect a scalar literal token, stopping at structural delimiters.
    fn parse_scalar_token(&mut self) -> String {
        self.skip_whitespace();
        let start = self.pos;
        while let Some(ch) = self.peek() {
            if ch.is_whitespace() || matches!(ch, ',' | ')' | ']' | '}' | ':') {
                break;
            }
            self.pos += 1;
        }
        self.chars[start..self.pos].iter().collect()
    }

    /// Parse one schema-directed value.
    fn parse_value(&mut self, schema: &Schema<'_>) -> Result<Box<dyn Reflect>, DeserializeError> {
        match schema {
            Schema::Primitive(primitive) => self.parse_primitive(*primitive),
            Schema::Info(info) => self.parse_info(info),
        }
    }

    /// Parse a composite (or registered-leaf) value whose shape is `info`.
    fn parse_info(&mut self, info: &TypeInfo) -> Result<Box<dyn Reflect>, DeserializeError> {
        match info {
            TypeInfo::Struct(struct_info) => {
                self.expect('(')?;
                let mut dynamic = DynamicStruct::new();
                dynamic.set_represented_type_name(struct_info.type_name());
                self.skip_whitespace();
                if self.peek() == Some(')') {
                    self.pos += 1;
                    return Ok(Box::new(dynamic));
                }
                loop {
                    let name = self.parse_ident()?;
                    let field = struct_info
                        .field(&name)
                        .ok_or(DeserializeError::UnknownField(name.clone()))?;
                    self.expect(':')?;
                    let child_schema = resolve(self.registry, field.type_name())?;
                    let child = self.parse_value(&child_schema)?;
                    dynamic.insert_boxed(field.name(), child);
                    if !self.consume_list_separator(')')? {
                        break;
                    }
                }
                Ok(Box::new(dynamic))
            }
            TypeInfo::TupleStruct(tuple_info) => {
                self.expect('(')?;
                let mut dynamic = DynamicTupleStruct::new();
                dynamic.set_represented_type_name(tuple_info.type_name());
                for (index, field) in tuple_info.fields().iter().enumerate() {
                    if index > 0 {
                        self.expect(',')?;
                    }
                    let child_schema = resolve(self.registry, field.type_name())?;
                    let child = self.parse_value(&child_schema)?;
                    dynamic.insert_boxed(child);
                }
                self.expect(')')?;
                Ok(Box::new(dynamic))
            }
            TypeInfo::Enum(enum_info) => {
                let variant_name = self.parse_ident()?;
                let variant = enum_info
                    .variant(&variant_name)
                    .ok_or(DeserializeError::UnknownVariant)?;
                let dynamic_variant = match variant.kind() {
                    VariantKind::Unit => DynamicVariant::Unit,
                    VariantKind::Tuple(fields) => {
                        self.expect('(')?;
                        let mut values = Vec::with_capacity(fields.len());
                        for (index, field) in fields.iter().enumerate() {
                            if index > 0 {
                                self.expect(',')?;
                            }
                            let child_schema = resolve(self.registry, field.type_name())?;
                            values.push(self.parse_value(&child_schema)?);
                        }
                        self.expect(')')?;
                        DynamicVariant::Tuple(values)
                    }
                    VariantKind::Struct(fields) => {
                        self.expect('(')?;
                        let mut values = Vec::with_capacity(fields.len());
                        self.skip_whitespace();
                        if self.peek() == Some(')') {
                            self.pos += 1;
                        } else {
                            loop {
                                let name = self.parse_ident()?;
                                let field = fields
                                    .iter()
                                    .find(|f| f.name() == name)
                                    .ok_or(DeserializeError::UnknownField(name.clone()))?;
                                self.expect(':')?;
                                let child_schema = resolve(self.registry, field.type_name())?;
                                let child = self.parse_value(&child_schema)?;
                                values.push((field.name(), child));
                                if !self.consume_list_separator(')')? {
                                    break;
                                }
                            }
                        }
                        DynamicVariant::Struct(values)
                    }
                };
                let mut dynamic =
                    DynamicEnum::new(variant.index(), variant.name(), dynamic_variant);
                dynamic.set_represented_type_name(enum_info.type_name());
                Ok(Box::new(dynamic))
            }
            TypeInfo::List(list_info) => {
                let item_schema = resolve(self.registry, list_info.item_type_name())?;
                let mut dynamic = DynamicList::new();
                dynamic.set_represented_type_name(list_info.type_name());
                self.parse_sequence(']', |parser| {
                    let child = parser.parse_value(&item_schema)?;
                    dynamic.push_boxed(child);
                    Ok(())
                })?;
                Ok(Box::new(dynamic))
            }
            TypeInfo::Array(array_info) => {
                let item_schema = resolve(self.registry, array_info.item_type_name())?;
                let mut dynamic = DynamicArray::new();
                dynamic.set_represented_type_name(array_info.type_name());
                self.parse_sequence(']', |parser| {
                    let child = parser.parse_value(&item_schema)?;
                    dynamic.push_boxed(child);
                    Ok(())
                })?;
                Ok(Box::new(dynamic))
            }
            TypeInfo::Set(set_info) => {
                let item_schema = resolve(self.registry, set_info.value_type_name())?;
                let mut dynamic = DynamicSet::new();
                dynamic.set_represented_type_name(set_info.type_name());
                self.parse_sequence(']', |parser| {
                    let child = parser.parse_value(&item_schema)?;
                    dynamic.push_boxed(child);
                    Ok(())
                })?;
                Ok(Box::new(dynamic))
            }
            TypeInfo::Map(map_info) => {
                let key_schema = resolve(self.registry, map_info.key_type_name())?;
                let value_schema = resolve(self.registry, map_info.value_type_name())?;
                let mut dynamic = DynamicMap::new();
                dynamic.set_represented_type_name(map_info.type_name());
                self.expect('{')?;
                self.skip_whitespace();
                if self.peek() == Some('}') {
                    self.pos += 1;
                    return Ok(Box::new(dynamic));
                }
                loop {
                    let key = self.parse_value(&key_schema)?;
                    self.expect(':')?;
                    let mapped = self.parse_value(&value_schema)?;
                    dynamic.insert_boxed(key, mapped);
                    if !self.consume_list_separator('}')? {
                        break;
                    }
                }
                Ok(Box::new(dynamic))
            }
            TypeInfo::Value(value_info) => {
                match crate::ser::primitive::leaf_primitive(value_info.type_name()) {
                    Some(primitive) => self.parse_primitive(primitive),
                    None => Err(DeserializeError::LeafTypeMismatch {
                        expected: value_info.type_name(),
                    }),
                }
            }
        }
    }

    /// Parse an opening bracket, then repeatedly invoke `each` for the
    /// comma-separated elements up to the matching `close` bracket.
    fn parse_sequence(
        &mut self,
        close: char,
        mut each: impl FnMut(&mut Self) -> Result<(), DeserializeError>,
    ) -> Result<(), DeserializeError> {
        self.expect('[')?;
        self.skip_whitespace();
        if self.peek() == Some(close) {
            self.pos += 1;
            return Ok(());
        }
        loop {
            each(self)?;
            if !self.consume_list_separator(close)? {
                break;
            }
        }
        Ok(())
    }

    /// After an element, consume a separating comma (and report whether another
    /// element follows) or the closing delimiter. A trailing comma before the
    /// close is accepted.
    ///
    /// Returns `true` when another element should be parsed, `false` when the
    /// scope has closed.
    fn consume_list_separator(&mut self, close: char) -> Result<bool, DeserializeError> {
        self.skip_whitespace();
        match self.peek() {
            Some(ch) if ch == close => {
                self.pos += 1;
                Ok(false)
            }
            Some(',') => {
                self.pos += 1;
                self.skip_whitespace();
                if self.peek() == Some(close) {
                    self.pos += 1;
                    Ok(false)
                } else {
                    Ok(true)
                }
            }
            Some(ch) => Err(DeserializeError::RonSyntax(alloc::format!(
                "expected `,` or `{close}`, found `{ch}`"
            ))),
            None => Err(DeserializeError::RonSyntax(alloc::format!(
                "expected `,` or `{close}`, found end of input"
            ))),
        }
    }

    /// Parse a leaf primitive of the schema-dictated type.
    fn parse_primitive(
        &mut self,
        primitive: Primitive,
    ) -> Result<Box<dyn Reflect>, DeserializeError> {
        match primitive {
            Primitive::Bool => {
                let token = self.parse_ident()?;
                match token.as_str() {
                    "true" => Ok(Box::new(true)),
                    "false" => Ok(Box::new(false)),
                    other => Err(DeserializeError::RonSyntax(alloc::format!(
                        "expected `true` or `false`, found `{other}`"
                    ))),
                }
            }
            Primitive::Char => {
                let ch = self.parse_char_literal()?;
                Ok(Box::new(ch))
            }
            Primitive::String => {
                let text = self.parse_string_literal()?;
                Ok(Box::new(text))
            }
            Primitive::I8 => self.parse_int::<i8>(),
            Primitive::I16 => self.parse_int::<i16>(),
            Primitive::I32 => self.parse_int::<i32>(),
            Primitive::I64 => self.parse_int::<i64>(),
            Primitive::I128 => self.parse_int::<i128>(),
            Primitive::Isize => self.parse_int::<isize>(),
            Primitive::U8 => self.parse_int::<u8>(),
            Primitive::U16 => self.parse_int::<u16>(),
            Primitive::U32 => self.parse_int::<u32>(),
            Primitive::U64 => self.parse_int::<u64>(),
            Primitive::U128 => self.parse_int::<u128>(),
            Primitive::Usize => self.parse_int::<usize>(),
            Primitive::F32 => {
                let token = self.parse_scalar_token();
                let value = token.parse::<f32>().map_err(|_| {
                    DeserializeError::RonSyntax(alloc::format!("invalid f32 literal `{token}`"))
                })?;
                Ok(Box::new(value))
            }
            Primitive::F64 => {
                let token = self.parse_scalar_token();
                let value = token.parse::<f64>().map_err(|_| {
                    DeserializeError::RonSyntax(alloc::format!("invalid f64 literal `{token}`"))
                })?;
                Ok(Box::new(value))
            }
        }
    }

    /// Parse an integer literal into the concrete integer type `T`.
    fn parse_int<T>(&mut self) -> Result<Box<dyn Reflect>, DeserializeError>
    where
        T: core::str::FromStr + Reflect,
    {
        let token = self.parse_scalar_token();
        let value = token.parse::<T>().map_err(|_| {
            DeserializeError::RonSyntax(alloc::format!("invalid integer literal `{token}`"))
        })?;
        Ok(Box::new(value))
    }

    /// Parse a quoted RON string literal with the supported escapes.
    fn parse_string_literal(&mut self) -> Result<String, DeserializeError> {
        self.expect('"')?;
        let mut text = String::new();
        loop {
            match self.bump() {
                Some('"') => break,
                Some('\\') => text.push(self.parse_escape()?),
                Some(other) => text.push(other),
                None => {
                    return Err(DeserializeError::RonSyntax(
                        "unterminated string literal".to_string(),
                    ));
                }
            }
        }
        Ok(text)
    }

    /// Parse a quoted RON char literal with the supported escapes.
    fn parse_char_literal(&mut self) -> Result<char, DeserializeError> {
        self.expect('\'')?;
        let ch = match self.bump() {
            Some('\\') => self.parse_escape()?,
            Some('\'') => {
                return Err(DeserializeError::RonSyntax(
                    "empty char literal".to_string(),
                ));
            }
            Some(other) => other,
            None => {
                return Err(DeserializeError::RonSyntax(
                    "unterminated char literal".to_string(),
                ));
            }
        };
        self.expect('\'')?;
        Ok(ch)
    }

    /// Decode the character following a `\\` escape introducer.
    fn parse_escape(&mut self) -> Result<char, DeserializeError> {
        match self.bump() {
            Some('"') => Ok('"'),
            Some('\'') => Ok('\''),
            Some('\\') => Ok('\\'),
            Some('n') => Ok('\n'),
            Some('r') => Ok('\r'),
            Some('t') => Ok('\t'),
            Some(other) => Err(DeserializeError::RonSyntax(alloc::format!(
                "unsupported escape `\\{other}`"
            ))),
            None => Err(DeserializeError::RonSyntax(
                "unterminated escape sequence".to_string(),
            )),
        }
    }
}
