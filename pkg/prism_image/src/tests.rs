//! Behavioural tests for the image asset.

use alloc::vec;
use alloc::vec::Vec;

use prism_asset::{AssetPath, AssetServer, Assets, LoadState, StableGuid};
use prism_asset_container::{
    ContainerIdentity, ContainerReader, ContainerWriter, ManifestAssetReader, ManifestRoute,
    OverlaySet, RuntimeManifest,
};
use prism_render_driver::{AddressMode, Extent3d, FilterMode, SamplerDescriptor, TextureFormat};
use std::sync::Arc;

use crate::{
    Image, ImageArtifact, ImageArtifactError, ImageError, RuntimeImageLoadError, RuntimeImageLoader,
};

#[test]
fn an_image_validates_its_data_against_its_extents_and_format() {
    let extent = Extent3d::new_2d(4, 2);
    // 4×2 texels × 4 bytes (Rgba8Unorm).
    let image = Image::new(extent, TextureFormat::Rgba8Unorm, vec![0; 32]).expect("valid");
    assert_eq!(image.extent(), extent);
    assert_eq!(image.format(), TextureFormat::Rgba8Unorm);
    assert_eq!(image.data().len(), 32);
    assert_eq!(image.bytes_per_row(), 16, "4 texels × 4 bytes");

    assert_eq!(
        Image::new(extent, TextureFormat::Rgba8Unorm, vec![0; 31]),
        Err(ImageError::SizeMismatch {
            expected: 32,
            found: 31
        }),
        "a short buffer would read past its own end"
    );
    assert_eq!(
        Image::new(extent, TextureFormat::Rgba8Unorm, vec![0; 40]),
        Err(ImageError::SizeMismatch {
            expected: 32,
            found: 40
        }),
        "a long buffer would upload junk"
    );
}

#[test]
fn a_format_with_a_different_texel_size_changes_what_is_required() {
    let extent = Extent3d::new_2d(3, 3);
    assert_eq!(
        Image::new(extent, TextureFormat::R8Unorm, vec![0; 9]).map(|image| image.bytes_per_row()),
        Ok(3)
    );
    assert_eq!(
        Image::new(extent, TextureFormat::Rgba32Float, vec![0; 9 * 16])
            .map(|image| image.bytes_per_row()),
        Ok(48)
    );
    assert_eq!(
        Image::new(extent, TextureFormat::R8Unorm, vec![0; 8]),
        Err(ImageError::SizeMismatch {
            expected: 9,
            found: 8
        })
    );
}

#[test]
fn degenerate_and_depth_images_are_refused() {
    assert_eq!(
        Image::new(
            Extent3d::new_2d(0, 4),
            TextureFormat::Rgba8Unorm,
            Vec::new()
        ),
        Err(ImageError::EmptyExtent)
    );
    assert_eq!(
        Image::new(
            Extent3d::new_2d(4, 0),
            TextureFormat::Rgba8Unorm,
            Vec::new()
        ),
        Err(ImageError::EmptyExtent)
    );
    assert_eq!(
        Image::new(
            Extent3d::new_2d(1, 1),
            TextureFormat::Depth32Float,
            vec![0; 4]
        ),
        Err(ImageError::NotColor {
            depth_stencil: false
        }),
        "a depth format is not sampleable content"
    );
    assert_eq!(
        Image::new(
            Extent3d::new_2d(1, 1),
            TextureFormat::Depth24PlusStencil8,
            vec![0; 4]
        ),
        Err(ImageError::NotColor {
            depth_stencil: true
        })
    );
}

#[test]
fn an_image_carries_the_sampler_it_wants() {
    let sampler = SamplerDescriptor {
        address_mode_u: AddressMode::Repeat,
        min_filter: FilterMode::Linear,
        ..Default::default()
    };
    let image = Image::with_sampler(
        Extent3d::new_2d(1, 1),
        TextureFormat::Rgba8Unorm,
        vec![0; 4],
        sampler.clone(),
    )
    .expect("valid");
    assert_eq!(image.sampler(), &sampler);

    let default_sampler = Image::new(
        Extent3d::new_2d(1, 1),
        TextureFormat::Rgba8Unorm,
        vec![0; 4],
    )
    .expect("valid");
    assert_eq!(default_sampler.sampler(), &SamplerDescriptor::default());
}

#[test]
fn an_image_is_an_asset() {
    assert_eq!(
        <Image as prism_asset::Asset>::TYPE_NAME,
        "prism_image::Image"
    );
}

#[test]
fn canonical_artifact_decodes_into_the_runtime_image_asset() {
    let artifact = ImageArtifact {
        width: 2,
        height: 1,
        rgba8: vec![1, 2, 3, 255, 4, 5, 6, 255],
    };
    let image = ImageArtifact::decode(&artifact.encode().unwrap())
        .unwrap()
        .to_image()
        .unwrap();
    assert_eq!(image.extent(), Extent3d::new_2d(2, 1));
    assert_eq!(image.format(), TextureFormat::Rgba8Unorm);
    assert_eq!(image.data(), artifact.rgba8);
}

fn mounted_runtime_image() -> (RuntimeManifest, OverlaySet, StableGuid) {
    let guid = StableGuid::from_path("textures/test.prismimg");
    let artifact = ImageArtifact {
        width: 2,
        height: 1,
        rgba8: vec![10, 20, 30, 255, 40, 50, 60, 255],
    }
    .encode()
    .unwrap();
    let mut writer = ContainerWriter::new();
    writer.add(guid, artifact).unwrap();
    let bytes = writer.finish().unwrap();
    let reader = ContainerReader::new(bytes).unwrap();
    let info = reader.info();
    let mut overlays = OverlaySet::new();
    overlays.mount("textures", 7, reader);
    let mut manifest = RuntimeManifest::new();
    manifest
        .insert(ManifestRoute {
            guid,
            container_name: "textures".into(),
            priority: 7,
            container_guid: guid,
        })
        .unwrap();
    manifest
        .bind_container(
            "textures",
            ContainerIdentity {
                byte_len: info.total_len,
                content_hash: info.content_hash,
                generation: info.generation,
            },
        )
        .unwrap();
    (manifest, overlays, guid)
}

#[test]
fn runtime_loader_reads_container_artifact_and_mints_typed_runtime_id() {
    let (manifest, overlays, guid) = mounted_runtime_image();
    let loader = RuntimeImageLoader::new(manifest, overlays);
    let mut assets = Assets::<Image>::new();
    let (id, handle) = loader.load_with_id(guid, &mut assets).unwrap();
    let image = assets.get(id).expect("runtime image is resident");
    assert_eq!(handle.id(), id);
    assert_eq!(image.extent(), Extent3d::new_2d(2, 1));
    assert_eq!(image.data(), &[10, 20, 30, 255, 40, 50, 60, 255]);
}

#[test]
fn runtime_loader_rejects_missing_routes_and_malformed_artifacts() {
    let (manifest, overlays, guid) = mounted_runtime_image();
    let loader = RuntimeImageLoader::new(manifest, overlays);
    let missing = StableGuid::from_u128(0x9999);
    assert!(matches!(
        loader.load(missing),
        Err(RuntimeImageLoadError::Read(
            prism_asset_container::ManifestReadError::MissingRoute(actual)
        )) if actual == missing
    ));

    let mut writer = ContainerWriter::new();
    writer.add(guid, b"not-an-image").unwrap();
    let bytes = writer.finish().unwrap();
    let reader = ContainerReader::new(bytes).unwrap();
    let info = reader.info();
    let mut overlays = OverlaySet::new();
    overlays.mount("bad", 1, reader);
    let mut manifest = RuntimeManifest::new();
    manifest
        .insert(ManifestRoute {
            guid,
            container_name: "bad".into(),
            priority: 1,
            container_guid: guid,
        })
        .unwrap();
    manifest
        .bind_container(
            "bad",
            ContainerIdentity {
                byte_len: info.total_len,
                content_hash: info.content_hash,
                generation: info.generation,
            },
        )
        .unwrap();
    let bad_loader = RuntimeImageLoader::new(manifest, overlays);
    assert!(matches!(
        bad_loader.load(guid),
        Err(RuntimeImageLoadError::Artifact(
            ImageArtifactError::InvalidHeader,
        ))
    ));
}

#[test]
fn runtime_loader_fails_closed_when_a_mounted_container_is_replaced() {
    let (manifest, overlays, guid) = mounted_runtime_image();
    let mut replacement_writer = ContainerWriter::new();
    replacement_writer.add(guid, b"replacement").unwrap();
    let replacement = ContainerReader::new(replacement_writer.finish().unwrap()).unwrap();
    let mut replacement_overlays = overlays;
    replacement_overlays.mount("textures", 7, replacement);
    let replaced = RuntimeImageLoader::new(manifest, replacement_overlays);
    assert!(matches!(
        replaced.load(guid),
        Err(RuntimeImageLoadError::Read(
            prism_asset_container::ManifestReadError::ContainerIdentityMismatch { .. }
        ))
    ));
}

#[test]
fn runtime_loader_integrates_with_asset_server_and_manifest_vfs() {
    let (manifest, overlays, guid) = mounted_runtime_image();
    let source = ManifestAssetReader::new(manifest.clone(), overlays.clone());
    let server = AssetServer::new();
    server.mount("runtime", Arc::new(source.clone()), 0);
    server.register_loader(RuntimeImageLoader::new(manifest, overlays.clone()));

    let handle = server.load::<Image>(AssetPath::parse("runtime://textures/test.prismimg"));
    assert_eq!(server.load_state(handle.id().untyped()), LoadState::Loaded);
    assert_eq!(
        server.with_asset(handle.id(), |image| {
            image.extent() == Extent3d::new_2d(2, 1) && image.data()[..4] == [10, 20, 30, 255]
        }),
        Some(true)
    );

    let direct = RuntimeImageLoader::new(source.manifest().clone(), overlays)
        .load(guid)
        .unwrap();
    assert_eq!(direct.data(), &[10, 20, 30, 255, 40, 50, 60, 255]);
}
