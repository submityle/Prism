//! A minimal, self-contained XML writer and reader for the ADM `axml` payload.
//!
//! The ADM metadata that rides inside a BW64 container is an XML document (the
//! `audioFormatExtended` tree). Rather than depend on a general XML library,
//! this module carries a tiny deterministic element model plus a focused writer
//! and recursive-descent reader that cover exactly the ADM subset emitted by
//! [`crate::adm::model`]. The writer produces stable indentation and attribute
//! ordering so that a document round trip (`document -> bytes -> document`) and
//! the byte round trip of a re-serialised document are both exact.
//!
//! # Provenance
//!
//! Original work; no Unreal Engine, Unity, Godot, Wwise, FMOD, Steam Audio,
//! Google Resonance Audio, Dolby, or MPEG source or derived code; no AI/ML.
//! The element and attribute names follow the publicly published ADM structure
//! (ITU-R BS.2076); the XML writer and reader are hand-written here.
//!
//! # Relationship
//!
//! Serialises and parses the [`crate::adm::model::AdmDocument`] graph. The
//! resulting bytes are stored in the `axml` chunk of the
//! [`crate::adm::bw64`] container, paired with the [`crate::adm::chna`] track
//! map.

use alloc::format;
use alloc::string::String;
use alloc::vec::Vec;

use prism_audio_core::math::Sample;

use crate::adm::model::{
    AdmDocument, AdmPosition, AudioBlockFormat, AudioChannelFormat, AudioContent, AudioObject,
    AudioPackFormat, AudioProgramme, AudioTrackUid, ObjectSize, TypeDefinition,
};

/// The XML declaration prepended to every serialised document.
const DECLARATION: &str = "<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n";

/// A failure while parsing an `axml` payload.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum XmlError {
    /// The input ended before a complete element was read.
    UnexpectedEnd,
    /// A structural rule (tag match, quoting) was violated.
    Malformed,
    /// A byte outside the printable ASCII range was encountered.
    NonAscii,
    /// A numeric attribute or element value failed to parse.
    BadNumber,
    /// A required element or attribute was missing while interpreting the tree.
    MissingField,
}

/// A parsed XML node: either a child element or a run of character data.
#[derive(Clone, Debug, PartialEq)]
pub enum XmlNode {
    /// A nested element.
    Element(XmlElement),
    /// Character data (already unescaped).
    Text(String),
}

/// A single XML element: a name, ordered attributes, and ordered children.
#[derive(Clone, Debug, PartialEq)]
pub struct XmlElement {
    /// The element (tag) name.
    pub name: String,
    /// Attributes in document order as `(name, value)` pairs.
    pub attrs: Vec<(String, String)>,
    /// Child nodes in document order.
    pub children: Vec<XmlNode>,
}

impl XmlElement {
    /// Builds an empty element with the given name.
    #[must_use]
    pub fn new(name: &str) -> Self {
        Self {
            name: String::from(name),
            attrs: Vec::new(),
            children: Vec::new(),
        }
    }

    /// Appends an attribute, returning `self` for chaining.
    #[must_use]
    pub fn with_attr(mut self, name: &str, value: &str) -> Self {
        self.attrs.push((String::from(name), String::from(value)));
        self
    }

    /// Appends a child element.
    pub fn push_element(&mut self, child: XmlElement) {
        self.children.push(XmlNode::Element(child));
    }

    /// Looks up an attribute value by name.
    #[must_use]
    pub fn attr(&self, name: &str) -> Option<&str> {
        self.attrs
            .iter()
            .find(|(key, _)| key == name)
            .map(|(_, value)| value.as_str())
    }

    /// The concatenated, trimmed character data directly inside this element.
    #[must_use]
    pub fn text(&self) -> String {
        let mut joined = String::new();
        for child in &self.children {
            if let XmlNode::Text(value) = child {
                joined.push_str(value);
            }
        }
        String::from(joined.trim())
    }

    /// Iterates over child elements with the given tag name.
    pub fn children_named<'a>(&'a self, name: &'a str) -> impl Iterator<Item = &'a XmlElement> {
        self.children.iter().filter_map(move |node| match node {
            XmlNode::Element(element) if element.name == name => Some(element),
            _ => None,
        })
    }

    /// Whether this element has any child elements (as opposed to text only).
    #[must_use]
    pub fn has_element_children(&self) -> bool {
        self.children
            .iter()
            .any(|node| matches!(node, XmlNode::Element(_)))
    }

    /// Renders this element into `out` at the given indentation depth.
    fn render(&self, out: &mut String, depth: usize) {
        indent(out, depth);
        out.push('<');
        out.push_str(&self.name);
        for (key, value) in &self.attrs {
            out.push(' ');
            out.push_str(key);
            out.push_str("=\"");
            escape_into(out, value);
            out.push('"');
        }
        if self.has_element_children() {
            out.push_str(">\n");
            for child in &self.children {
                if let XmlNode::Element(element) = child {
                    element.render(out, depth + 1);
                }
            }
            indent(out, depth);
            out.push_str("</");
            out.push_str(&self.name);
            out.push_str(">\n");
            return;
        }
        let body = self.text();
        if body.is_empty() {
            out.push_str("/>\n");
        } else {
            out.push('>');
            escape_into(out, &body);
            out.push_str("</");
            out.push_str(&self.name);
            out.push_str(">\n");
        }
    }
}

/// Writes `depth` levels of two-space indentation into `out`.
fn indent(out: &mut String, depth: usize) {
    for _ in 0..depth {
        out.push_str("  ");
    }
}

/// Appends `text` to `out`, escaping the five XML metacharacters.
fn escape_into(out: &mut String, text: &str) {
    for ch in text.chars() {
        match ch {
            '&' => out.push_str("&amp;"),
            '<' => out.push_str("&lt;"),
            '>' => out.push_str("&gt;"),
            '"' => out.push_str("&quot;"),
            '\'' => out.push_str("&apos;"),
            other => out.push(other),
        }
    }
}

/// Formats a [`Sample`] deterministically for XML output.
fn fmt_sample(value: Sample) -> String {
    format!("{value}")
}

/// Serialises an [`AdmDocument`] into deterministic `axml` bytes.
#[must_use]
pub fn to_axml_bytes(document: &AdmDocument) -> Vec<u8> {
    let root = build_tree(document);
    let mut text = String::from(DECLARATION);
    root.render(&mut text, 0);
    text.into_bytes()
}

/// Builds the `audioFormatExtended` element tree from an [`AdmDocument`].
fn build_tree(document: &AdmDocument) -> XmlElement {
    let mut root = XmlElement::new("audioFormatExtended");
    for programme in &document.programmes {
        root.push_element(build_programme(programme));
    }
    for content in &document.contents {
        root.push_element(build_content(content));
    }
    for object in &document.objects {
        root.push_element(build_object(object));
    }
    for pack in &document.pack_formats {
        root.push_element(build_pack(pack));
    }
    for channel in &document.channel_formats {
        root.push_element(build_channel(channel));
    }
    for track in &document.track_uids {
        root.push_element(build_track(track));
    }
    root
}

/// Builds a `<ref>`-style element carrying a single reference string.
fn ref_element(name: &str, value: &str) -> XmlElement {
    let mut element = XmlElement::new(name);
    element.children.push(XmlNode::Text(String::from(value)));
    element
}

/// Builds a `<name>text</name>` style value element.
fn value_element(name: &str, value: &str) -> XmlElement {
    ref_element(name, value)
}

fn build_programme(programme: &AudioProgramme) -> XmlElement {
    let mut element = XmlElement::new("audioProgramme")
        .with_attr("audioProgrammeID", &programme.id)
        .with_attr("audioProgrammeName", &programme.name);
    for reference in &programme.content_refs {
        element.push_element(ref_element("audioContentIDRef", reference));
    }
    element
}

fn build_content(content: &AudioContent) -> XmlElement {
    let mut element = XmlElement::new("audioContent")
        .with_attr("audioContentID", &content.id)
        .with_attr("audioContentName", &content.name);
    for reference in &content.object_refs {
        element.push_element(ref_element("audioObjectIDRef", reference));
    }
    element
}

fn build_object(object: &AudioObject) -> XmlElement {
    let mut element = XmlElement::new("audioObject")
        .with_attr("audioObjectID", &object.id)
        .with_attr("audioObjectName", &object.name)
        .with_attr("importance", &format!("{}", object.importance));
    for reference in &object.pack_format_refs {
        element.push_element(ref_element("audioPackFormatIDRef", reference));
    }
    for reference in &object.track_uid_refs {
        element.push_element(ref_element("audioTrackUIDRef", reference));
    }
    element
}

fn build_pack(pack: &AudioPackFormat) -> XmlElement {
    let mut element = XmlElement::new("audioPackFormat")
        .with_attr("audioPackFormatID", &pack.id)
        .with_attr("audioPackFormatName", &pack.name)
        .with_attr("typeLabel", pack.type_def.type_label())
        .with_attr("typeDefinition", type_name(pack.type_def));
    for reference in &pack.channel_format_refs {
        element.push_element(ref_element("audioChannelFormatIDRef", reference));
    }
    element
}

fn build_channel(channel: &AudioChannelFormat) -> XmlElement {
    let mut element = XmlElement::new("audioChannelFormat")
        .with_attr("audioChannelFormatID", &channel.id)
        .with_attr("audioChannelFormatName", &channel.name)
        .with_attr("typeLabel", channel.type_def.type_label())
        .with_attr("typeDefinition", type_name(channel.type_def));
    for block in &channel.block_formats {
        element.push_element(build_block(block));
    }
    element
}

fn build_block(block: &AudioBlockFormat) -> XmlElement {
    let mut element = XmlElement::new("audioBlockFormat")
        .with_attr("audioBlockFormatID", &block.id)
        .with_attr("rtime", &fmt_sample(block.rtime))
        .with_attr("duration", &fmt_sample(block.duration));
    element.push_element(
        value_element("position", &fmt_sample(block.position.azimuth))
            .with_attr("coordinate", "azimuth"),
    );
    element.push_element(
        value_element("position", &fmt_sample(block.position.elevation))
            .with_attr("coordinate", "elevation"),
    );
    element.push_element(
        value_element("position", &fmt_sample(block.position.distance))
            .with_attr("coordinate", "distance"),
    );
    element.push_element(value_element("gain", &fmt_sample(block.gain)));
    element.push_element(value_element("width", &fmt_sample(block.size.width)));
    element.push_element(value_element("height", &fmt_sample(block.size.height)));
    element.push_element(value_element("depth", &fmt_sample(block.size.depth)));
    element.push_element(value_element("importance", &format!("{}", block.importance)));
    element
}

fn build_track(track: &AudioTrackUid) -> XmlElement {
    let mut element = XmlElement::new("audioTrackUID")
        .with_attr("UID", &track.id)
        .with_attr("trackIndex", &format!("{}", track.track_index));
    element.push_element(ref_element(
        "audioChannelFormatIDRef",
        &track.channel_format_ref,
    ));
    element.push_element(ref_element("audioPackFormatIDRef", &track.pack_format_ref));
    element
}

/// The cosmetic `typeDefinition` name string for a [`TypeDefinition`].
fn type_name(type_def: TypeDefinition) -> &'static str {
    match type_def {
        TypeDefinition::DirectSpeakers => "DirectSpeakers",
        TypeDefinition::Matrix => "Matrix",
        TypeDefinition::Objects => "Objects",
        TypeDefinition::Hoa => "HOA",
        TypeDefinition::Binaural => "Binaural",
    }
}

/// Parses an `axml` payload back into an [`AdmDocument`].
///
/// # Errors
///
/// Returns an [`XmlError`] describing the first structural, encoding, numeric,
/// or missing-field problem encountered.
pub fn from_axml_bytes(bytes: &[u8]) -> Result<AdmDocument, XmlError> {
    if !bytes.is_ascii() {
        return Err(XmlError::NonAscii);
    }
    let text = core::str::from_utf8(bytes).map_err(|_| XmlError::NonAscii)?;
    let root = parse_document(text)?;
    if root.name != "audioFormatExtended" {
        return Err(XmlError::Malformed);
    }
    interpret_document(&root)
}

/// Parses the top-level element out of a full XML document string.
fn parse_document(text: &str) -> Result<XmlElement, XmlError> {
    let bytes = text.as_bytes();
    let mut cursor = 0usize;
    skip_prolog(bytes, &mut cursor);
    let element = parse_element(bytes, &mut cursor)?;
    Ok(element)
}

/// Skips leading whitespace, the XML declaration, and comments.
fn skip_prolog(bytes: &[u8], cursor: &mut usize) {
    loop {
        skip_whitespace(bytes, cursor);
        if starts_with(bytes, *cursor, b"<?")
            && let Some(end) = find(bytes, *cursor, b"?>")
        {
            *cursor = end + 2;
            continue;
        }
        if starts_with(bytes, *cursor, b"<!--")
            && let Some(end) = find(bytes, *cursor, b"-->")
        {
            *cursor = end + 3;
            continue;
        }
        break;
    }
}

/// Advances `cursor` past ASCII whitespace.
fn skip_whitespace(bytes: &[u8], cursor: &mut usize) {
    while *cursor < bytes.len() && matches!(bytes[*cursor], b' ' | b'\t' | b'\r' | b'\n') {
        *cursor += 1;
    }
}

/// Whether `bytes[cursor..]` begins with `needle`.
fn starts_with(bytes: &[u8], cursor: usize, needle: &[u8]) -> bool {
    cursor + needle.len() <= bytes.len() && &bytes[cursor..cursor + needle.len()] == needle
}

/// Finds the next occurrence of `needle` at or after `cursor`.
fn find(bytes: &[u8], cursor: usize, needle: &[u8]) -> Option<usize> {
    if needle.is_empty() || cursor >= bytes.len() {
        return None;
    }
    let mut index = cursor;
    while index + needle.len() <= bytes.len() {
        if &bytes[index..index + needle.len()] == needle {
            return Some(index);
        }
        index += 1;
    }
    None
}

/// Parses one element (and its subtree) starting at `cursor`.
fn parse_element(bytes: &[u8], cursor: &mut usize) -> Result<XmlElement, XmlError> {
    if !starts_with(bytes, *cursor, b"<") {
        return Err(XmlError::Malformed);
    }
    *cursor += 1;
    let name = read_name(bytes, cursor)?;
    let mut element = XmlElement::new(&name);
    loop {
        skip_whitespace(bytes, cursor);
        match peek(bytes, *cursor)? {
            b'/' => {
                if !starts_with(bytes, *cursor, b"/>") {
                    return Err(XmlError::Malformed);
                }
                *cursor += 2;
                return Ok(element);
            }
            b'>' => {
                *cursor += 1;
                break;
            }
            _ => {
                let (key, value) = read_attribute(bytes, cursor)?;
                element.attrs.push((key, value));
            }
        }
    }
    parse_children(bytes, cursor, &mut element)?;
    Ok(element)
}

/// Parses the children and closing tag of an already-opened element.
fn parse_children(
    bytes: &[u8],
    cursor: &mut usize,
    element: &mut XmlElement,
) -> Result<(), XmlError> {
    loop {
        if *cursor >= bytes.len() {
            return Err(XmlError::UnexpectedEnd);
        }
        if starts_with(bytes, *cursor, b"</") {
            *cursor += 2;
            let close = read_name(bytes, cursor)?;
            if close != element.name {
                return Err(XmlError::Malformed);
            }
            skip_whitespace(bytes, cursor);
            if peek(bytes, *cursor)? != b'>' {
                return Err(XmlError::Malformed);
            }
            *cursor += 1;
            return Ok(());
        }
        if starts_with(bytes, *cursor, b"<!--") {
            match find(bytes, *cursor, b"-->") {
                Some(end) => {
                    *cursor = end + 3;
                    continue;
                }
                None => return Err(XmlError::UnexpectedEnd),
            }
        }
        if starts_with(bytes, *cursor, b"<") {
            let child = parse_element(bytes, cursor)?;
            element.push_element(child);
            continue;
        }
        let raw = read_until(bytes, cursor, b'<')?;
        let decoded = unescape(&raw)?;
        if !decoded.trim().is_empty() {
            element.children.push(XmlNode::Text(decoded));
        }
    }
}

/// Reads an element or attribute name (letters, digits, `_`, `-`, `:`).
fn read_name(bytes: &[u8], cursor: &mut usize) -> Result<String, XmlError> {
    let start = *cursor;
    while *cursor < bytes.len() {
        let byte = bytes[*cursor];
        let is_name = byte.is_ascii_alphanumeric() || matches!(byte, b'_' | b'-' | b':' | b'.');
        if !is_name {
            break;
        }
        *cursor += 1;
    }
    if *cursor == start {
        return Err(XmlError::Malformed);
    }
    ascii_string(&bytes[start..*cursor])
}

/// Reads a single `name="value"` attribute.
fn read_attribute(bytes: &[u8], cursor: &mut usize) -> Result<(String, String), XmlError> {
    let key = read_name(bytes, cursor)?;
    skip_whitespace(bytes, cursor);
    if peek(bytes, *cursor)? != b'=' {
        return Err(XmlError::Malformed);
    }
    *cursor += 1;
    skip_whitespace(bytes, cursor);
    if peek(bytes, *cursor)? != b'"' {
        return Err(XmlError::Malformed);
    }
    *cursor += 1;
    let raw = read_until(bytes, cursor, b'"')?;
    // Consume the closing quote.
    *cursor += 1;
    let value = unescape(&raw)?;
    Ok((key, value))
}

/// Reads raw bytes up to (not including) the delimiter.
fn read_until(bytes: &[u8], cursor: &mut usize, delimiter: u8) -> Result<Vec<u8>, XmlError> {
    let start = *cursor;
    while *cursor < bytes.len() && bytes[*cursor] != delimiter {
        *cursor += 1;
    }
    if *cursor >= bytes.len() {
        return Err(XmlError::UnexpectedEnd);
    }
    Ok(bytes[start..*cursor].to_vec())
}

/// Returns the byte at `cursor` or an end-of-input error.
fn peek(bytes: &[u8], cursor: usize) -> Result<u8, XmlError> {
    bytes.get(cursor).copied().ok_or(XmlError::UnexpectedEnd)
}

/// Converts an ASCII byte slice into a [`String`].
fn ascii_string(bytes: &[u8]) -> Result<String, XmlError> {
    if !bytes.is_ascii() {
        return Err(XmlError::NonAscii);
    }
    let mut text = String::with_capacity(bytes.len());
    for &byte in bytes {
        text.push(byte as char);
    }
    Ok(text)
}

/// Reverses [`escape_into`] on a raw byte run.
fn unescape(bytes: &[u8]) -> Result<String, XmlError> {
    let text = ascii_string(bytes)?;
    let mut out = String::with_capacity(text.len());
    let mut rest = text.as_str();
    while let Some(amp) = rest.find('&') {
        out.push_str(&rest[..amp]);
        let tail = &rest[amp..];
        if let Some(semi) = tail.find(';') {
            let entity = &tail[1..semi];
            match entity {
                "amp" => out.push('&'),
                "lt" => out.push('<'),
                "gt" => out.push('>'),
                "quot" => out.push('"'),
                "apos" => out.push('\''),
                _ => return Err(XmlError::Malformed),
            }
            rest = &tail[semi + 1..];
        } else {
            return Err(XmlError::Malformed);
        }
    }
    out.push_str(rest);
    Ok(out)
}

/// Interprets the parsed tree into an [`AdmDocument`].
fn interpret_document(root: &XmlElement) -> Result<AdmDocument, XmlError> {
    let mut document = AdmDocument::new();
    for element in root.children_named("audioProgramme") {
        document.programmes.push(read_programme(element)?);
    }
    for element in root.children_named("audioContent") {
        document.contents.push(read_content(element)?);
    }
    for element in root.children_named("audioObject") {
        document.objects.push(read_object(element)?);
    }
    for element in root.children_named("audioPackFormat") {
        document.pack_formats.push(read_pack(element)?);
    }
    for element in root.children_named("audioChannelFormat") {
        document.channel_formats.push(read_channel(element)?);
    }
    for element in root.children_named("audioTrackUID") {
        document.track_uids.push(read_track(element)?);
    }
    Ok(document)
}

fn read_programme(element: &XmlElement) -> Result<AudioProgramme, XmlError> {
    let mut programme = AudioProgramme::new(
        require_attr(element, "audioProgrammeID")?,
        element.attr("audioProgrammeName").unwrap_or(""),
    );
    for reference in element.children_named("audioContentIDRef") {
        programme.content_refs.push(reference.text());
    }
    Ok(programme)
}

fn read_content(element: &XmlElement) -> Result<AudioContent, XmlError> {
    let mut content = AudioContent::new(
        require_attr(element, "audioContentID")?,
        element.attr("audioContentName").unwrap_or(""),
    );
    for reference in element.children_named("audioObjectIDRef") {
        content.object_refs.push(reference.text());
    }
    Ok(content)
}

fn read_object(element: &XmlElement) -> Result<AudioObject, XmlError> {
    let mut object = AudioObject::new(
        require_attr(element, "audioObjectID")?,
        element.attr("audioObjectName").unwrap_or(""),
    );
    object.importance = parse_u8(element.attr("importance").unwrap_or("10"))?;
    for reference in element.children_named("audioPackFormatIDRef") {
        object.pack_format_refs.push(reference.text());
    }
    for reference in element.children_named("audioTrackUIDRef") {
        object.track_uid_refs.push(reference.text());
    }
    Ok(object)
}

fn read_pack(element: &XmlElement) -> Result<AudioPackFormat, XmlError> {
    let type_def = read_type(element)?;
    let mut pack = AudioPackFormat::new(
        require_attr(element, "audioPackFormatID")?,
        element.attr("audioPackFormatName").unwrap_or(""),
        type_def,
    );
    for reference in element.children_named("audioChannelFormatIDRef") {
        pack.channel_format_refs.push(reference.text());
    }
    Ok(pack)
}

fn read_channel(element: &XmlElement) -> Result<AudioChannelFormat, XmlError> {
    let type_def = read_type(element)?;
    let mut channel = AudioChannelFormat::new(
        require_attr(element, "audioChannelFormatID")?,
        element.attr("audioChannelFormatName").unwrap_or(""),
        type_def,
    );
    for block in element.children_named("audioBlockFormat") {
        channel.block_formats.push(read_block(block)?);
    }
    Ok(channel)
}

fn read_block(element: &XmlElement) -> Result<AudioBlockFormat, XmlError> {
    let mut azimuth = 0.0;
    let mut elevation = 0.0;
    let mut distance = 1.0;
    for position in element.children_named("position") {
        let value = parse_sample(&position.text())?;
        match position.attr("coordinate") {
            Some("azimuth") => azimuth = value,
            Some("elevation") => elevation = value,
            Some("distance") => distance = value,
            _ => return Err(XmlError::Malformed),
        }
    }
    let size = ObjectSize::new(
        parse_child_sample(element, "width", 0.0)?,
        parse_child_sample(element, "height", 0.0)?,
        parse_child_sample(element, "depth", 0.0)?,
    );
    let mut block = AudioBlockFormat::new(
        require_attr(element, "audioBlockFormatID")?,
        AdmPosition::new(azimuth, elevation, distance),
    );
    block.rtime = parse_sample(element.attr("rtime").unwrap_or("0"))?;
    block.duration = parse_sample(element.attr("duration").unwrap_or("0"))?;
    block.gain = parse_child_sample(element, "gain", 1.0)?;
    block.size = size;
    block.importance = parse_u8(&child_text(element, "importance").unwrap_or_default())
        .ok()
        .unwrap_or(10);
    Ok(block)
}

fn read_track(element: &XmlElement) -> Result<AudioTrackUid, XmlError> {
    let channel_ref = element
        .children_named("audioChannelFormatIDRef")
        .next()
        .map(XmlElement::text)
        .unwrap_or_default();
    let pack_ref = element
        .children_named("audioPackFormatIDRef")
        .next()
        .map(XmlElement::text)
        .unwrap_or_default();
    Ok(AudioTrackUid::new(
        require_attr(element, "UID")?,
        parse_u16(element.attr("trackIndex").unwrap_or("0"))?,
        &channel_ref,
        &pack_ref,
    ))
}

/// Reads the `typeLabel` attribute and resolves it to a [`TypeDefinition`].
fn read_type(element: &XmlElement) -> Result<TypeDefinition, XmlError> {
    let label = require_attr(element, "typeLabel")?;
    TypeDefinition::from_type_label(label).ok_or(XmlError::MissingField)
}

/// Returns an attribute value or [`XmlError::MissingField`].
fn require_attr<'a>(element: &'a XmlElement, name: &str) -> Result<&'a str, XmlError> {
    element.attr(name).ok_or(XmlError::MissingField)
}

/// The trimmed text of the first child element with the given name.
fn child_text(element: &XmlElement, name: &str) -> Option<String> {
    element.children_named(name).next().map(XmlElement::text)
}

/// Parses the text of a named child as a [`Sample`], or a default if absent.
fn parse_child_sample(
    element: &XmlElement,
    name: &str,
    default: Sample,
) -> Result<Sample, XmlError> {
    match child_text(element, name) {
        Some(text) if !text.is_empty() => parse_sample(&text),
        _ => Ok(default),
    }
}

/// Parses a [`Sample`] from text.
fn parse_sample(text: &str) -> Result<Sample, XmlError> {
    text.trim().parse::<Sample>().map_err(|_| XmlError::BadNumber)
}

/// Parses a `u8` from text.
fn parse_u8(text: &str) -> Result<u8, XmlError> {
    text.trim().parse::<u8>().map_err(|_| XmlError::BadNumber)
}

/// Parses a `u16` from text.
fn parse_u16(text: &str) -> Result<u16, XmlError> {
    text.trim().parse::<u16>().map_err(|_| XmlError::BadNumber)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn document() -> AdmDocument {
        let mut doc = AdmDocument::new();
        let mut channel =
            AudioChannelFormat::new("AC_00031001", "Front Left", TypeDefinition::Objects);
        let mut block =
            AudioBlockFormat::new("AB_00031001_00000001", AdmPosition::new(30.0, 10.0, 1.0));
        block.gain = 0.5;
        block.size = ObjectSize::new(0.25, 0.0, 0.0);
        channel.block_formats.push(block);
        doc.channel_formats.push(channel);

        let mut pack = AudioPackFormat::new("AP_00031001", "Mono & Wide", TypeDefinition::Objects);
        pack.channel_format_refs.push(String::from("AC_00031001"));
        doc.pack_formats.push(pack);

        doc.track_uids
            .push(AudioTrackUid::new("ATU_00000001", 1, "AC_00031001", "AP_00031001"));

        let mut object = AudioObject::new("AO_1001", "Dialogue <lead>");
        object.pack_format_refs.push(String::from("AP_00031001"));
        object.track_uid_refs.push(String::from("ATU_00000001"));
        doc.objects.push(object);

        let mut content = AudioContent::new("ACO_1001", "Dialogue");
        content.object_refs.push(String::from("AO_1001"));
        doc.contents.push(content);

        let mut programme = AudioProgramme::new("APR_1001", "Main");
        programme.content_refs.push(String::from("ACO_1001"));
        doc.programmes.push(programme);
        doc
    }

    #[test]
    fn document_round_trips() {
        let doc = document();
        let bytes = to_axml_bytes(&doc);
        let parsed = from_axml_bytes(&bytes).expect("parse");
        assert_eq!(parsed, doc);
    }

    #[test]
    fn re_serialisation_is_byte_identical() {
        let doc = document();
        let bytes = to_axml_bytes(&doc);
        let parsed = from_axml_bytes(&bytes).expect("parse");
        assert_eq!(to_axml_bytes(&parsed), bytes);
    }

    #[test]
    fn metacharacters_are_escaped() {
        let doc = document();
        let bytes = to_axml_bytes(&doc);
        let text = core::str::from_utf8(&bytes).expect("utf8");
        assert!(text.contains("Dialogue &lt;lead&gt;"));
        assert!(text.contains("Mono &amp; Wide"));
    }

    #[test]
    fn non_ascii_input_is_rejected() {
        let bytes = [0x3c, 0xc3, 0xa9, 0x3e];
        assert_eq!(from_axml_bytes(&bytes), Err(XmlError::NonAscii));
    }

    #[test]
    fn malformed_input_is_rejected() {
        let bytes = b"<audioFormatExtended><audioObject></audioFormatExtended>";
        assert!(from_axml_bytes(bytes).is_err());
    }
}
