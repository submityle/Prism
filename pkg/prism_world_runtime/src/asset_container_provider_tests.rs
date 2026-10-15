use super::*;

use prism_asset::StableGuid;
use prism_asset_container::{
    ContainerReader, ContainerWriter, ManifestReadError, OverlaySet, RuntimeManifest,
};
use prism_world_stream::{AabbI64, CellManifest, GlobalPos, SectorId};

use crate::{CellProvider, CellReadRequest, ChunkProvider, ManifestHeader, CELL_HEADER_SIZE};

fn cell_manifest(scene_key: u128, io_bytes: usize) -> CellManifest {
    CellManifest {
        sector: SectorId::new(0, 3, 4, 5),
        bounds: AabbI64::new(GlobalPos::new(0, 0, 0), GlobalPos::new(1, 1, 1))
            .expect("valid bounds"),
        scene_key,
        content_version: 9,
        derived_hash: 11,
        io_bytes: io_bytes as u64,
    }
}

fn mounted(
    scene_manifest: CellManifest,
    payload: &[u8],
    compressed: bool,
) -> (RuntimeManifest, OverlaySet, Vec<u8>) {
    let mut cell = ManifestHeader::from_manifest(scene_manifest).encode();
    cell.extend_from_slice(payload);
    let guid = StableGuid::from_u128(scene_manifest.scene_key);
    let mut writer = ContainerWriter::new();
    if compressed {
        writer
            .add_deflate(guid, cell.clone())
            .expect("deflate entry");
    } else {
        writer.add(guid, cell.clone()).expect("raw entry");
    }
    let bytes = writer.finish().expect("container bytes");
    let reader = ContainerReader::new(bytes).expect("validated container");
    let mut overlays = OverlaySet::new();
    overlays.mount("world", 7, reader);
    let manifest = overlays.runtime_manifest().expect("runtime manifest");
    (manifest, overlays, cell)
}

#[test]
fn reads_raw_and_deflate_cells_through_cell_and_chunk_seams() {
    for compressed in [false, true] {
        let expected_manifest = cell_manifest(0x1234, CELL_HEADER_SIZE + 5);
        let (runtime_manifest, overlays, expected) =
            mounted(expected_manifest, b"hello", compressed);
        let provider = AssetContainerCellProvider::with_max_io_bytes(
            runtime_manifest,
            overlays,
            expected.len() as u64,
        )
        .expect("provider");
        let request = CellReadRequest::new(expected_manifest, 3);
        assert_eq!(
            provider.read_cell(&request, CancellationToken::new(false)),
            Ok(expected.clone())
        );
        assert_eq!(
            provider.read_chunk(&request, CancellationToken::new(false)),
            Ok(expected)
        );
    }
}

#[test]
fn checks_entry_length_before_reading_and_rejects_stale_declaration() {
    let expected_manifest = cell_manifest(0x2345, CELL_HEADER_SIZE + 5);
    let (runtime_manifest, overlays, _) = mounted(expected_manifest, b"hello", false);
    let provider = AssetContainerCellProvider::new(runtime_manifest, overlays);
    let wrong = cell_manifest(
        expected_manifest.scene_key,
        usize::try_from(expected_manifest.io_bytes)
            .expect("test length")
            .checked_add(1)
            .expect("test length increment"),
    );
    let error = provider
        .read_cell(
            &CellReadRequest::new(wrong, 0),
            CancellationToken::new(false),
        )
        .expect_err("stale declaration must fail");
    assert!(matches!(
        error,
        AssetContainerProviderError::PayloadLengthMismatch {
            expected_io_bytes,
            actual_payload_bytes,
            ..
        } if expected_io_bytes == wrong.io_bytes && actual_payload_bytes == expected_manifest.io_bytes
    ));
}

#[test]
fn validates_header_after_container_read() {
    let expected_manifest = cell_manifest(0x3456, CELL_HEADER_SIZE + 5);
    let mut cell = ManifestHeader::from_manifest(expected_manifest).encode();
    cell[20..28].copy_from_slice(&10_u64.to_le_bytes());
    cell.extend_from_slice(b"hello");
    let guid = StableGuid::from_u128(expected_manifest.scene_key);
    let mut writer = ContainerWriter::new();
    writer.add(guid, cell).expect("entry");
    let reader = ContainerReader::new(writer.finish().expect("container")).expect("reader");
    let mut overlays = OverlaySet::new();
    overlays.mount("world", 1, reader);
    let runtime_manifest = overlays.runtime_manifest().expect("manifest");
    let provider = AssetContainerCellProvider::new(runtime_manifest, overlays);
    let error = provider
        .read_cell(
            &CellReadRequest::new(expected_manifest, 0),
            CancellationToken::new(false),
        )
        .expect_err("header mismatch");
    assert!(matches!(
        error,
        AssetContainerProviderError::Header(HeaderError::ContentVersionMismatch { .. })
    ));
}

#[test]
fn cancellation_and_bounds_are_checked_before_container_read() {
    let expected_manifest = cell_manifest(0x4567, CELL_HEADER_SIZE + 5);
    let (runtime_manifest, overlays, _) = mounted(expected_manifest, b"hello", false);
    let provider = AssetContainerCellProvider::with_max_io_bytes(
        runtime_manifest,
        overlays,
        expected_manifest.io_bytes,
    )
    .expect("provider");
    let request = CellReadRequest::new(expected_manifest, 0);
    assert_eq!(
        provider.read_cell(&request, CancellationToken::new(true)),
        Err(AssetContainerProviderError::Cancelled)
    );

    let too_small = AssetContainerCellProvider::with_max_io_bytes(
        provider.manifest().clone(),
        provider.overlays().clone(),
        CELL_HEADER_SIZE as u64,
    )
    .expect("header-sized bound");
    let error = too_small
        .read_cell(&request, CancellationToken::new(false))
        .expect_err("request exceeds bound");
    assert!(matches!(
        error,
        AssetContainerProviderError::RequestTooLarge { .. }
    ));

    let invalid = cell_manifest(expected_manifest.scene_key, CELL_HEADER_SIZE - 1);
    let error = provider
        .read_cell(
            &CellReadRequest::new(invalid, 0),
            CancellationToken::new(false),
        )
        .expect_err("header-less declaration");
    assert!(matches!(
        error,
        AssetContainerProviderError::InvalidDeclaredLength { .. }
    ));
}

#[test]
fn invalid_provider_bound_and_missing_route_are_explicit() {
    let error =
        AssetContainerCellProvider::with_max_io_bytes(RuntimeManifest::new(), OverlaySet::new(), 0)
            .expect_err("zero bound");
    assert!(matches!(
        error,
        AssetContainerProviderError::InvalidMaxIoBytes { .. }
    ));

    let cell = cell_manifest(0x5678, CELL_HEADER_SIZE);
    let provider = AssetContainerCellProvider::new(RuntimeManifest::new(), OverlaySet::new());
    let error = provider
        .read_cell(
            &CellReadRequest::new(cell, 0),
            CancellationToken::new(false),
        )
        .expect_err("missing logical route");
    assert!(matches!(
        error,
        AssetContainerProviderError::Manifest(error)
            if matches!(&*error, ManifestReadError::MissingRoute(guid)
                if *guid == StableGuid::from_u128(cell.scene_key))
    ));
}
