use super::*;
use crate::virtual_geometry::cluster_payload::{
    decode_cluster_payload, POSITION_ONLY_CLUSTER_PAYLOAD_VERSION,
};

fn bounds() -> SceneBounds {
    SceneBounds {
        center: [0.0; 3],
        half_extents: [1.0; 3],
        radius: 2.0,
        _padding: 0.0,
    }
}

fn attribute(format: VertexFormat, location: u32, data: Vec<u8>) -> ClusterVertexAttribute {
    ClusterVertexAttribute {
        semantic: alloc::format!("Custom_{location}"),
        shader_location: location,
        format,
        data,
    }
}

fn encode(attributes: &[ClusterVertexAttribute]) -> Result<Vec<u8>, ClusterPayloadError> {
    encode_cluster_payload_with_attributes(bounds(), &[[0.0; 3]], &[0, 0, 0], 19, attributes)
}

#[test]
fn all_formats_preserve_wire_tag_semantic_location_and_exact_bytes() {
    let formats = [
        VertexFormat::Float32,
        VertexFormat::Float32x2,
        VertexFormat::Float32x3,
        VertexFormat::Float32x4,
        VertexFormat::Float16x2,
        VertexFormat::Float16x4,
        VertexFormat::Uint32,
        VertexFormat::Uint32x2,
        VertexFormat::Uint32x4,
        VertexFormat::Sint32,
        VertexFormat::Unorm8x4,
        VertexFormat::Snorm8x4,
    ];
    let attributes: Vec<_> = formats
        .into_iter()
        .enumerate()
        .map(|(index, format)| {
            assert_eq!(format_tag(format).unwrap(), index as u16);
            let pattern: &[u8] = match format {
                VertexFormat::Float16x2 | VertexFormat::Float16x4 => &[0, 0xbc, 0, 0x40],
                // -2.0 f32; other formats preserve these bits, without reinterpretation.
                _ => &[0, 0, 0, 0xc0],
            };
            attribute(
                format,
                index as u32,
                pattern
                    .iter()
                    .copied()
                    .cycle()
                    .take(format.size() as usize)
                    .collect(),
            )
        })
        .collect();
    let bytes = encode(&attributes).unwrap();
    assert_eq!(&bytes[4..6], &3u16.to_le_bytes());
    assert_eq!(bytes.len() % 4, 0);
    let decoded = decode_cluster_payload(bounds(), &bytes).unwrap();
    assert_eq!(decoded.position_shader_location, 19);
    assert_eq!(decoded.attributes, attributes);
    let mut reversed = attributes;
    reversed.reverse();
    assert_eq!(
        encode(&reversed).unwrap(),
        bytes,
        "input iteration order cannot change wire bytes"
    );
}

#[test]
fn legacy_bytes_and_current_position_only_payload_are_both_readable() {
    let legacy = encode_cluster_payload(bounds(), &[[0.0; 3]], &[0, 0, 0]).unwrap();
    assert_eq!(
        &legacy[4..6],
        &POSITION_ONLY_CLUSTER_PAYLOAD_VERSION.to_le_bytes()
    );
    let old = decode_cluster_payload(bounds(), &legacy).unwrap();
    assert_eq!(old.position_shader_location, 0);
    assert!(old.attributes.is_empty());
    let current = decode_cluster_payload(bounds(), &encode(&[]).unwrap()).unwrap();
    assert_eq!(current.position_shader_location, 19);
    assert_eq!(current.positions, old.positions);
    assert_eq!(current.indices, old.indices);
}

#[test]
fn every_truncation_and_trailing_byte_is_rejected() {
    let bytes = encode(&[attribute(VertexFormat::Float32x2, 1, vec![0; 8])]).unwrap();
    for end in 0..bytes.len() {
        assert!(
            decode_cluster_payload(bounds(), &bytes[..end]).is_err(),
            "accepted prefix length {end}"
        );
    }
    let mut extra = bytes.clone();
    extra.push(0);
    assert_eq!(
        decode_cluster_payload(bounds(), &extra),
        Err(ClusterPayloadError::InvalidFraming)
    );
    let mut legacy = encode_cluster_payload(bounds(), &[[0.0; 3]], &[0, 0, 0]).unwrap();
    legacy.push(0);
    assert_eq!(
        decode_cluster_payload(bounds(), &legacy),
        Err(ClusterPayloadError::InvalidFraming)
    );
}

#[test]
fn malformed_descriptors_reserved_bytes_and_non_finite_data_are_rejected() {
    // Geometry ends at 46, footer starts at 48, first entry starts at 56.
    let mut source = attribute(VertexFormat::Float32, 1, vec![0; 4]);
    source.semantic = "Custom_01".into();
    let bytes = encode(&[source]).unwrap();
    for (offset, mutation, error) in [
        (46, 1, ClusterPayloadError::InvalidFraming), // geometry padding
        (54, 1, ClusterPayloadError::InvalidFraming), // reserved
        (52, 32, ClusterPayloadError::InvalidAttribute), // total attribute count
        (56, 0, ClusterPayloadError::InvalidAttribute), // empty name
        (58, 255, ClusterPayloadError::InvalidAttribute), // unknown format
        (64, 3, ClusterPayloadError::InvalidAttribute), // incorrect data length
        (68, 255, ClusterPayloadError::InvalidAttribute), // invalid UTF-8 name
        (68, 0, ClusterPayloadError::InvalidAttribute), // control character
        (78, 1, ClusterPayloadError::InvalidFraming), // name padding
        (60, 19, ClusterPayloadError::DuplicateAttribute), // position location
    ] {
        let mut corrupt = bytes.clone();
        corrupt[offset] = mutation;
        assert_eq!(
            decode_cluster_payload(bounds(), &corrupt),
            Err(error),
            "offset {offset}"
        );
    }
    for (format, bits) in [
        (VertexFormat::Float32, f32::NAN.to_le_bytes().to_vec()),
        (
            VertexFormat::Float32,
            f32::NEG_INFINITY.to_le_bytes().to_vec(),
        ),
        (VertexFormat::Float16x2, vec![0, 0x7c, 0, 0]),
        (VertexFormat::Float16x2, vec![1, 0xfc, 0, 0]),
    ] {
        assert_eq!(
            encode(&[attribute(format, 1, bits.clone())]),
            Err(ClusterPayloadError::NonFinite)
        );
        let mut corrupt = encode(&[attribute(format, 1, vec![0; format.size() as usize])]).unwrap();
        let start = corrupt.len() - bits.len();
        corrupt[start..].copy_from_slice(&bits);
        assert_eq!(
            decode_cluster_payload(bounds(), &corrupt),
            Err(ClusterPayloadError::NonFinite)
        );
    }
}

#[test]
fn duplicate_identity_size_and_schema_caps_are_enforced() {
    let a = attribute(VertexFormat::Uint32, 1, vec![0; 4]);
    assert_eq!(
        encode(&[a.clone(), a.clone()]),
        Err(ClusterPayloadError::DuplicateAttribute)
    );
    let mut b = a.clone();
    b.shader_location = 2;
    assert_eq!(
        encode(&[a.clone(), b]),
        Err(ClusterPayloadError::DuplicateAttribute)
    );
    let mut b = a.clone();
    b.semantic = "Vertex_Position".into();
    assert_eq!(encode(&[b]), Err(ClusterPayloadError::DuplicateAttribute));
    for name in [String::new(), "x".repeat(256), "bad\nname".into()] {
        let mut b = a.clone();
        b.semantic = name;
        assert_eq!(encode(&[b]), Err(ClusterPayloadError::InvalidAttribute));
    }
    let mut b = a;
    b.data.pop();
    assert_eq!(encode(&[b]), Err(ClusterPayloadError::InvalidAttribute));
    let attributes: Vec<_> = (0..31)
        .map(|location| attribute(VertexFormat::Uint32, location, vec![0; 4]))
        .collect();
    let bytes =
        encode_cluster_payload_with_attributes(bounds(), &[[0.0; 3]], &[0, 0, 0], 100, &attributes)
            .unwrap();
    assert_eq!(
        decode_cluster_payload(bounds(), &bytes)
            .unwrap()
            .attributes
            .len(),
        31
    );
    let mut too_many = attributes;
    too_many.push(attribute(VertexFormat::Uint32, 31, vec![0; 4]));
    assert_eq!(
        encode_cluster_payload_with_attributes(bounds(), &[[0.0; 3]], &[0, 0, 0], 100, &too_many),
        Err(ClusterPayloadError::InvalidAttribute)
    );
}

#[test]
fn wire_entry_order_cannot_alias_or_reorder_shader_locations() {
    let bytes = encode(&[
        attribute(VertexFormat::Uint32, 1, vec![0; 4]),
        attribute(VertexFormat::Uint32, 2, vec![0; 4]),
    ])
    .unwrap();
    // Each descriptor occupies 24 bytes (12 + 8 name + 4 data).
    for location in [0u32, 1] {
        let mut corrupt = bytes.clone();
        corrupt[84..88].copy_from_slice(&location.to_le_bytes());
        assert_eq!(
            decode_cluster_payload(bounds(), &corrupt),
            Err(ClusterPayloadError::InvalidFraming)
        );
    }
}
