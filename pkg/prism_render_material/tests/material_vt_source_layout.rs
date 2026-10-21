//! Backend-neutral source layout and snapshot metadata compatibility.

use prism_render_architecture::texture_streaming::TexturePageKey;
use prism_render_material::{
    MaterialVtError, MaterialVtFormat, MaterialVtSourceDescriptor, MaterialVtSourceLayout,
    MaterialVtSourceSnapshot, PageRequest, NO_VIRTUAL_TEXTURE,
};

fn descriptor(
    width: u32,
    height: u32,
    layers: u16,
    page_size: u32,
    format: MaterialVtFormat,
) -> MaterialVtSourceDescriptor {
    MaterialVtSourceDescriptor::new(
        41,
        width,
        height,
        layers,
        page_size,
        (width.max(height).ilog2() + 1) as u8,
        format,
    )
    .unwrap()
}

fn key(mip: u8, layer: u16, x: u16, y: u16) -> TexturePageKey {
    TexturePageKey {
        texture: 41,
        mip,
        layer,
        x,
        y,
    }
}

#[test]
fn layout_and_snapshot_metadata_match_across_formats_dimensions_and_layers() {
    for format in [
        MaterialVtFormat::R8Unorm,
        MaterialVtFormat::Rg8Unorm,
        MaterialVtFormat::Rgba8Unorm,
        MaterialVtFormat::Rgba8Srgb,
        MaterialVtFormat::Rgba16Float,
        MaterialVtFormat::Rgba32Float,
    ] {
        for (width, height) in [(1, 1), (7, 5), (2, 9), (8, 8)] {
            for layers in [1, 3] {
                for page_size in [1, 3, 16] {
                    let descriptor = descriptor(width, height, layers, page_size, format);
                    let layout = MaterialVtSourceLayout::new(descriptor).unwrap();
                    let bytes = vec![0x5a; usize::try_from(layout.byte_len()).unwrap()];
                    let snapshot = MaterialVtSourceSnapshot::new(descriptor, bytes).unwrap();
                    assert_eq!(snapshot.layout(), &layout);
                    assert_eq!(layout.descriptor(), descriptor);
                    assert_eq!(layout.descriptor(), snapshot.descriptor());
                    assert_eq!(layout.mips(), snapshot.mips());
                    assert_eq!(snapshot.layout().mips().as_ptr(), snapshot.mips().as_ptr());
                    assert_eq!(layout.page_count(), snapshot.page_count());
                    assert_eq!(layout.byte_len(), snapshot.bytes().len() as u64);
                    assert_eq!(layout.byte_len(), snapshot.byte_len());
                    assert_eq!(layout.virtual_texture(), snapshot.virtual_texture());
                    assert_eq!(
                        layout.texture_row(9, 12, 3, 7),
                        snapshot.texture_row(9, 12, 3, 7)
                    );

                    let keys: Vec<_> = layout.page_keys().collect();
                    assert_eq!(keys, snapshot.page_keys().collect::<Vec<_>>());
                    assert_eq!(keys.len() as u64, layout.page_count());
                    assert!(keys.windows(2).all(|pair| pair[0] < pair[1]));
                    let metadata: Vec<_> = layout.page_metadata_iter().collect();
                    assert_eq!(metadata, snapshot.page_metadata_iter().collect::<Vec<_>>());
                    let mut extracted_bytes = 0;
                    for (key, expected) in keys.into_iter().zip(metadata) {
                        assert_eq!(layout.page_metadata(key).unwrap(), expected);
                        assert_eq!(snapshot.page_metadata(key).unwrap(), expected);
                        let request = PageRequest {
                            mip: u32::from(key.mip),
                            page_x: u32::from(key.x),
                            page_y: u32::from(key.y),
                        };
                        assert_eq!(layout.key_for_request(key.layer, request), Ok(key));
                        assert_eq!(snapshot.key_for_request(key.layer, request), Ok(key));
                        let page = snapshot.page(key).unwrap();
                        assert_eq!(page.metadata(), expected);
                        assert_eq!(
                            page.rows().map(|row| row.bytes.len() as u64).sum::<u64>(),
                            expected.extracted_byte_len
                        );
                        extracted_bytes += expected.extracted_byte_len;
                    }
                    assert_eq!(extracted_bytes, layout.byte_len());
                }
            }
        }
    }
}

#[test]
fn npot_floor_mips_and_layer_offsets_have_exact_metadata() {
    let layout =
        MaterialVtSourceLayout::new(descriptor(7, 5, 3, 3, MaterialVtFormat::Rg8Unorm)).unwrap();
    let mips = layout.mips();
    assert_eq!(
        mips.iter()
            .map(|mip| (mip.width, mip.height))
            .collect::<Vec<_>>(),
        [(7, 5), (3, 2), (1, 1)]
    );
    assert_eq!(
        mips.iter().map(|mip| mip.offset_bytes).collect::<Vec<_>>(),
        [0, 210, 246]
    );
    assert_eq!(mips[0].layer_stride_bytes, 70);
    assert_eq!(mips[0].level_stride_bytes, 210);
    assert_eq!(layout.byte_len(), 252);
    assert_eq!(layout.page_count(), 24);

    let edge = layout.page_metadata(key(0, 2, 2, 1)).unwrap();
    assert_eq!((edge.origin_x, edge.origin_y), (6, 3));
    assert_eq!((edge.extent_width, edge.extent_height), (1, 2));
    assert_eq!(edge.source_row_pitch_bytes, 14);
    assert_eq!(edge.row_bytes, 2);
    assert_eq!(edge.first_row_offset_bytes, 194);
    assert_eq!(edge.row_count, 2);
    assert_eq!(edge.extracted_byte_len, 4);
    assert_eq!(
        layout
            .page_metadata(key(1, 2, 0, 0))
            .unwrap()
            .first_row_offset_bytes,
        234
    );
    assert_eq!(
        layout
            .page_metadata(key(2, 2, 0, 0))
            .unwrap()
            .first_row_offset_bytes,
        250
    );
}

#[test]
fn huge_layout_needs_only_mip_metadata_and_no_texel_or_page_buffer() {
    let layout = MaterialVtSourceLayout::new(descriptor(
        1 << 30,
        1 << 30,
        1,
        1 << 14,
        MaterialVtFormat::R8Unorm,
    ))
    .unwrap();
    assert_eq!(layout.mips().len(), 31);
    assert_eq!(layout.byte_len(), ((1u64 << 62) - 1) / 3);
    assert!(layout.page_count() > u64::from(u32::MAX));
    assert_eq!(
        (layout.mips()[0].pages_x, layout.mips()[0].pages_y),
        (65_536, 65_536)
    );
    let edge_key = key(0, 0, u16::MAX, u16::MAX);
    assert_eq!(
        layout.key_for_request(
            0,
            PageRequest {
                mip: 0,
                page_x: 65_535,
                page_y: 65_535
            }
        ),
        Ok(edge_key)
    );
    let edge = layout.page_metadata(edge_key).unwrap();
    assert_eq!(
        (edge.origin_x, edge.origin_y),
        ((1 << 30) - (1 << 14), (1 << 30) - (1 << 14))
    );
    assert_eq!(
        edge.first_row_offset_bytes
            + u64::from(edge.row_count - 1) * u64::from(edge.source_row_pitch_bytes)
            + u64::from(edge.row_bytes),
        1u64 << 60
    );
    let last = layout.page_metadata(key(30, 0, 0, 0)).unwrap();
    assert_eq!(
        last.first_row_offset_bytes + last.extracted_byte_len,
        layout.byte_len()
    );
    assert_eq!(layout.page_keys().next(), Some(key(0, 0, 0, 0)));
    assert_eq!(
        layout.page_metadata_iter().next().unwrap().key,
        key(0, 0, 0, 0)
    );
}

#[test]
fn maximum_layer_count_iterator_terminates_without_wrapping() {
    let layout = MaterialVtSourceLayout::new(descriptor(
        1,
        1,
        u16::MAX,
        u32::MAX,
        MaterialVtFormat::Rgba32Float,
    ))
    .unwrap();
    assert_eq!(layout.page_count(), u64::from(u16::MAX));
    assert_eq!(layout.byte_len(), u64::from(u16::MAX) * 16);
    let mut keys = layout.page_keys();
    assert_eq!(
        keys.nth(usize::from(u16::MAX) - 1),
        Some(key(0, u16::MAX - 1, 0, 0))
    );
    assert_eq!(keys.next(), None);
    assert_eq!(keys.next(), None);
    assert_eq!(
        layout.page_metadata(key(0, u16::MAX, 0, 0)),
        Err(MaterialVtError::LayerOutOfRange { layer: u16::MAX })
    );
}

#[test]
fn layout_and_snapshot_reject_all_invalid_key_fields_identically() {
    let descriptor = descriptor(7, 5, 2, 3, MaterialVtFormat::R8Unorm);
    let layout = MaterialVtSourceLayout::new(descriptor).unwrap();
    let snapshot =
        MaterialVtSourceSnapshot::new(descriptor, vec![0; layout.byte_len() as usize]).unwrap();
    for (invalid, error) in [
        (
            TexturePageKey {
                texture: 42,
                ..key(0, 0, 0, 0)
            },
            MaterialVtError::TextureIdMismatch {
                expected: 41,
                actual: 42,
            },
        ),
        (
            key(0, 2, 0, 0),
            MaterialVtError::LayerOutOfRange { layer: 2 },
        ),
        (key(3, 0, 0, 0), MaterialVtError::MipOutOfRange { mip: 3 }),
        (
            key(0, 0, 3, 0),
            MaterialVtError::PageXOutOfRange { mip: 0, page_x: 3 },
        ),
        (
            key(0, 0, 0, 2),
            MaterialVtError::PageYOutOfRange { mip: 0, page_y: 2 },
        ),
        (
            key(2, 0, 1, 0),
            MaterialVtError::PageXOutOfRange { mip: 2, page_x: 1 },
        ),
    ] {
        assert_eq!(layout.page_metadata(invalid), Err(error));
        assert_eq!(snapshot.page_metadata(invalid), Err(error));
        assert_eq!(snapshot.page(invalid).unwrap_err(), error);
    }
    for (layer, request, error) in [
        (
            0,
            PageRequest {
                mip: 256,
                page_x: 0,
                page_y: 0,
            },
            MaterialVtError::MipOutOfRange { mip: 256 },
        ),
        (
            0,
            PageRequest {
                mip: 3,
                page_x: 0,
                page_y: 0,
            },
            MaterialVtError::MipOutOfRange { mip: 3 },
        ),
        (
            0,
            PageRequest {
                mip: 0,
                page_x: 65_536,
                page_y: 0,
            },
            MaterialVtError::PageXOutOfRange {
                mip: 0,
                page_x: 65_536,
            },
        ),
        (
            0,
            PageRequest {
                mip: 0,
                page_x: 0,
                page_y: u32::MAX,
            },
            MaterialVtError::PageYOutOfRange {
                mip: 0,
                page_y: u32::MAX,
            },
        ),
        (
            2,
            PageRequest {
                mip: 0,
                page_x: 0,
                page_y: 0,
            },
            MaterialVtError::LayerOutOfRange { layer: 2 },
        ),
    ] {
        assert_eq!(layout.key_for_request(layer, request), Err(error));
        assert_eq!(snapshot.key_for_request(layer, request), Err(error));
    }
}

#[test]
fn payload_overflow_is_checked_for_both_level_multiplication_and_chain_addition() {
    // Sixteen layers overflow at mip zero; thirteen fit at mip zero but
    // overflow while adding the remaining mips. Neither case needs texels.
    for layers in [16, 13] {
        let descriptor = descriptor(1 << 30, 1 << 30, layers, 1 << 14, MaterialVtFormat::R8Unorm);
        assert_eq!(
            MaterialVtSourceLayout::new(descriptor).unwrap_err(),
            MaterialVtError::PayloadSizeOverflow
        );
        assert_eq!(
            MaterialVtSourceSnapshot::new(descriptor, Vec::<u8>::new()).unwrap_err(),
            MaterialVtError::PayloadSizeOverflow
        );
    }
}

#[test]
fn descriptor_validation_still_requires_complete_chain_and_representable_keys() {
    for (texture, width, height, layers, page_size, mip_count, format, error) in [
        (
            NO_VIRTUAL_TEXTURE,
            1,
            1,
            1,
            1,
            1,
            MaterialVtFormat::R8Unorm,
            MaterialVtError::ReservedTextureId,
        ),
        (
            41,
            0,
            1,
            1,
            1,
            1,
            MaterialVtFormat::R8Unorm,
            MaterialVtError::ZeroDimension,
        ),
        (
            41,
            1,
            1,
            0,
            1,
            1,
            MaterialVtFormat::R8Unorm,
            MaterialVtError::ZeroLayerCount,
        ),
        (
            41,
            1,
            1,
            1,
            0,
            1,
            MaterialVtFormat::R8Unorm,
            MaterialVtError::ZeroPageSize,
        ),
        (
            41,
            7,
            5,
            1,
            3,
            2,
            MaterialVtFormat::R8Unorm,
            MaterialVtError::MipCountMismatch {
                expected: 3,
                actual: 2,
            },
        ),
        (
            41,
            7,
            5,
            1,
            3,
            4,
            MaterialVtFormat::R8Unorm,
            MaterialVtError::MipCountMismatch {
                expected: 3,
                actual: 4,
            },
        ),
        (
            41,
            65_537,
            1,
            1,
            1,
            17,
            MaterialVtFormat::R8Unorm,
            MaterialVtError::PageGridTooLarge {
                mip: 0,
                pages_x: 65_537,
                pages_y: 1,
            },
        ),
        (
            41,
            1,
            65_537,
            1,
            1,
            17,
            MaterialVtFormat::R8Unorm,
            MaterialVtError::PageGridTooLarge {
                mip: 0,
                pages_x: 1,
                pages_y: 65_537,
            },
        ),
        (
            41,
            u32::MAX,
            1,
            1,
            u32::MAX,
            32,
            MaterialVtFormat::Rgba32Float,
            MaterialVtError::RowPitchOverflow { mip: 0 },
        ),
    ] {
        assert_eq!(
            MaterialVtSourceDescriptor::new(
                texture, width, height, layers, page_size, mip_count, format
            ),
            Err(error)
        );
    }
}

#[test]
fn snapshot_keeps_exact_payload_length_contract() {
    let descriptor = descriptor(7, 5, 3, 3, MaterialVtFormat::Rg8Unorm);
    for actual in [0usize, 251, 253] {
        assert_eq!(
            MaterialVtSourceSnapshot::new(descriptor, vec![0; actual]).unwrap_err(),
            MaterialVtError::PayloadLengthMismatch {
                expected: 252,
                actual: actual as u64
            }
        );
    }
}
