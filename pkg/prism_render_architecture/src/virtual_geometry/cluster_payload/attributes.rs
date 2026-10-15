//! CLST v3 lossless attribute footer; geometry offsets remain identical to v2.

use alloc::collections::BTreeSet;
use alloc::string::{String, ToString};
use alloc::vec::Vec;
use prism_render_driver::VertexFormat;

use super::{encode_cluster_payload, ClusterPayloadError, STATIC_CLUSTER_PAYLOAD_VERSION};
use crate::gpu_scene::SceneBounds;

/// Maximum total vertex attributes, including the quantized position.
/// This is a payload schema limit, not a device vertex-input capability claim.
pub const MAX_CLUSTER_ATTRIBUTES: usize = 32;
/// Maximum UTF-8 semantic length in bytes in the v3 schema.
pub const MAX_CLUSTER_ATTRIBUTE_NAME_BYTES: usize = 255;

/// One lossless, tightly packed non-position attribute in a cluster.
#[derive(Clone, Debug, PartialEq)]
pub struct ClusterVertexAttribute {
    /// Persisted authored semantic identity.
    pub semantic: String,
    /// Authored pipeline shader location.
    pub shader_location: u32,
    /// RHI vertex fetch format, encoded by a stable explicit wire tag.
    pub format: VertexFormat,
    /// Little-endian values in local vertex order, without per-vertex padding.
    pub data: Vec<u8>,
}

/// Encodes CLST v3 with quantized positions and lossless vertex attributes.
///
/// After the v2 geometry bytes, zero padding aligns the footer to four bytes.
/// The footer is `position_location:u32, count:u16, reserved:u16`, followed by
/// entries sorted by shader location. Each entry is
/// `name_len:u16, format_tag:u16, location:u32, data_len:u32`, UTF-8 name,
/// zero padding to four bytes, then tightly packed little-endian vertex data.
/// All twelve current RHI formats are supported. Unknown formats, non-finite
/// floats, duplicate names/locations, or malformed byte counts are rejected.
pub fn encode_cluster_payload_with_attributes(
    bounds: SceneBounds,
    positions: &[[f32; 3]],
    indices: &[u16],
    position_shader_location: u32,
    attributes: &[ClusterVertexAttribute],
) -> Result<Vec<u8>, ClusterPayloadError> {
    validate_attributes(positions.len(), position_shader_location, attributes)?;
    let mut bytes = encode_cluster_payload(bounds, positions, indices)?;
    bytes[4..6].copy_from_slice(&STATIC_CLUSTER_PAYLOAD_VERSION.to_le_bytes());
    pad_to_word(&mut bytes);
    bytes.extend_from_slice(&position_shader_location.to_le_bytes());
    bytes.extend_from_slice(&(attributes.len() as u16).to_le_bytes());
    bytes.extend_from_slice(&0u16.to_le_bytes());
    let mut ordered: Vec<_> = attributes.iter().collect();
    ordered.sort_by_key(|attribute| attribute.shader_location);
    for attribute in ordered {
        bytes.extend_from_slice(&(attribute.semantic.len() as u16).to_le_bytes());
        bytes.extend_from_slice(&format_tag(attribute.format)?.to_le_bytes());
        bytes.extend_from_slice(&attribute.shader_location.to_le_bytes());
        bytes.extend_from_slice(&(attribute.data.len() as u32).to_le_bytes());
        bytes.extend_from_slice(attribute.semantic.as_bytes());
        pad_to_word(&mut bytes);
        bytes.extend_from_slice(&attribute.data);
    }
    Ok(bytes)
}

pub(super) fn decode_attributes(
    vertex_count: usize,
    bytes: &[u8],
    geometry_end: usize,
) -> Result<(u32, Vec<ClusterVertexAttribute>), ClusterPayloadError> {
    let mut reader = Reader {
        bytes,
        cursor: geometry_end,
    };
    reader.align()?;
    let position_shader_location = reader.u32()?;
    let count = reader.u16()? as usize;
    if reader.u16()? != 0 {
        return Err(ClusterPayloadError::InvalidFraming);
    }
    if count >= MAX_CLUSTER_ATTRIBUTES {
        return Err(ClusterPayloadError::InvalidAttribute);
    }
    let mut attributes: Vec<ClusterVertexAttribute> = Vec::with_capacity(count);
    for _ in 0..count {
        let name_len = reader.u16()? as usize;
        let format = tag_format(reader.u16()?)?;
        let shader_location = reader.u32()?;
        let data_len = reader.u32()? as usize;
        if name_len == 0
            || name_len > MAX_CLUSTER_ATTRIBUTE_NAME_BYTES
            || data_len != vertex_count * format.size() as usize
        {
            return Err(ClusterPayloadError::InvalidAttribute);
        }
        if attributes
            .last()
            .is_some_and(|previous| previous.shader_location >= shader_location)
        {
            return Err(ClusterPayloadError::InvalidFraming);
        }
        let name = core::str::from_utf8(reader.take(name_len)?)
            .map_err(|_| ClusterPayloadError::InvalidAttribute)?;
        reader.align()?;
        let data = reader.take(data_len)?;
        attributes.push(ClusterVertexAttribute {
            semantic: name.to_string(),
            shader_location,
            format,
            data: data.to_vec(),
        });
    }
    if reader.cursor != bytes.len() {
        return Err(ClusterPayloadError::InvalidFraming);
    }
    validate_attributes(vertex_count, position_shader_location, &attributes)?;
    Ok((position_shader_location, attributes))
}

fn validate_attributes(
    vertex_count: usize,
    position_shader_location: u32,
    attributes: &[ClusterVertexAttribute],
) -> Result<(), ClusterPayloadError> {
    if vertex_count == 0 || vertex_count > 64 {
        return Err(ClusterPayloadError::InvalidCounts);
    }
    if attributes.len() >= MAX_CLUSTER_ATTRIBUTES {
        return Err(ClusterPayloadError::InvalidAttribute);
    }
    let mut names = BTreeSet::new();
    let mut locations = BTreeSet::new();
    locations.insert(position_shader_location);
    names.insert("Vertex_Position");
    for attribute in attributes {
        if attribute.semantic.is_empty()
            || attribute.semantic.len() > MAX_CLUSTER_ATTRIBUTE_NAME_BYTES
            || attribute.semantic.chars().any(char::is_control)
            || attribute.data.len() != vertex_count * attribute.format.size() as usize
        {
            return Err(ClusterPayloadError::InvalidAttribute);
        }
        format_tag(attribute.format)?;
        if !names.insert(attribute.semantic.as_str())
            || !locations.insert(attribute.shader_location)
        {
            return Err(ClusterPayloadError::DuplicateAttribute);
        }
        let finite = match attribute.format {
            VertexFormat::Float32
            | VertexFormat::Float32x2
            | VertexFormat::Float32x3
            | VertexFormat::Float32x4 => attribute
                .data
                .chunks_exact(4)
                .all(|word| f32::from_le_bytes(word.try_into().unwrap()).is_finite()),
            VertexFormat::Float16x2 | VertexFormat::Float16x4 => attribute
                .data
                .chunks_exact(2)
                .all(|word| u16::from_le_bytes(word.try_into().unwrap()) & 0x7c00 != 0x7c00),
            _ => true,
        };
        if !finite {
            return Err(ClusterPayloadError::NonFinite);
        }
    }
    Ok(())
}

// These values are persisted, independent of Rust enum discriminants.
fn format_tag(format: VertexFormat) -> Result<u16, ClusterPayloadError> {
    Ok(match format {
        VertexFormat::Float32 => 0,
        VertexFormat::Float32x2 => 1,
        VertexFormat::Float32x3 => 2,
        VertexFormat::Float32x4 => 3,
        VertexFormat::Float16x2 => 4,
        VertexFormat::Float16x4 => 5,
        VertexFormat::Uint32 => 6,
        VertexFormat::Uint32x2 => 7,
        VertexFormat::Uint32x4 => 8,
        VertexFormat::Sint32 => 9,
        VertexFormat::Unorm8x4 => 10,
        VertexFormat::Snorm8x4 => 11,
        _ => return Err(ClusterPayloadError::InvalidAttribute),
    })
}

fn tag_format(tag: u16) -> Result<VertexFormat, ClusterPayloadError> {
    Ok(match tag {
        0 => VertexFormat::Float32,
        1 => VertexFormat::Float32x2,
        2 => VertexFormat::Float32x3,
        3 => VertexFormat::Float32x4,
        4 => VertexFormat::Float16x2,
        5 => VertexFormat::Float16x4,
        6 => VertexFormat::Uint32,
        7 => VertexFormat::Uint32x2,
        8 => VertexFormat::Uint32x4,
        9 => VertexFormat::Sint32,
        10 => VertexFormat::Unorm8x4,
        11 => VertexFormat::Snorm8x4,
        _ => return Err(ClusterPayloadError::InvalidAttribute),
    })
}

fn pad_to_word(bytes: &mut Vec<u8>) {
    bytes.resize((bytes.len() + 3) & !3, 0);
}

struct Reader<'a> {
    bytes: &'a [u8],
    cursor: usize,
}

impl<'a> Reader<'a> {
    fn take(&mut self, count: usize) -> Result<&'a [u8], ClusterPayloadError> {
        let end = self
            .cursor
            .checked_add(count)
            .ok_or(ClusterPayloadError::Truncated)?;
        let slice = self
            .bytes
            .get(self.cursor..end)
            .ok_or(ClusterPayloadError::Truncated)?;
        self.cursor = end;
        Ok(slice)
    }
    fn u16(&mut self) -> Result<u16, ClusterPayloadError> {
        Ok(u16::from_le_bytes(self.take(2)?.try_into().unwrap()))
    }
    fn u32(&mut self) -> Result<u32, ClusterPayloadError> {
        Ok(u32::from_le_bytes(self.take(4)?.try_into().unwrap()))
    }
    fn align(&mut self) -> Result<(), ClusterPayloadError> {
        let padding = (4 - self.cursor % 4) % 4;
        if self.take(padding)?.iter().any(|byte| *byte != 0) {
            return Err(ClusterPayloadError::InvalidFraming);
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests;
