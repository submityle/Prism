//! Behavioural tests for the image asset.

use alloc::vec;
use alloc::vec::Vec;

use prism_asset::{AssetPath, AssetServer, Assets, LoadState, StableGuid};
use prism_asset_container::{
    ContainerIdentity, ContainerReader, ContainerWriter, ManifestAssetReader, ManifestRoute,
    OverlaySet, RuntimeManifest,
};
use prism_render_driver::{
    AddressMode, CommandBuffer, Extent3d, FilterMode, ImageDataLayout, RenderQueue,
    SamplerDescriptor, TextureFormat, TextureId, TextureWrite,
};
use std::sync::Arc;

use crate::upload::checked_row_range;
use crate::{
    AlphaCoveragePolicy, Image, ImageArtifact, ImageArtifactError, ImageError, ImageMipArtifact,
    ImageMipChain, ImageUploadPlan, MipGenerationSettings, RuntimeImageLoadError,
    RuntimeImageLoader, TextureSemantic, IMAGE_UPLOAD_ROW_ALIGNMENT,
};

#[derive(Default)]
struct CapturingQueue {
    writes: std::sync::Mutex<Vec<(TextureWrite, Vec<u8>, ImageDataLayout)>>,
}

impl RenderQueue for CapturingQueue {
    fn write_buffer(&self, _: prism_render_driver::BufferId, _: u64, _: &[u8]) {}

    fn write_texture(&self, destination: TextureWrite, data: &[u8], layout: ImageDataLayout) {
        self.writes
            .lock()
            .unwrap()
            .push((destination, data.to_vec(), layout));
    }

    fn submit(&self, _: &[CommandBuffer]) {}
}

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

#[test]
fn mip_artifact_roundtrips_metadata_and_encoding_is_deterministic() {
    let rgba8 = (0..(3 * 5 * 4))
        .map(|value| value as u8)
        .collect::<Vec<_>>();
    let chain = ImageMipChain::generate(
        3,
        5,
        &rgba8,
        MipGenerationSettings {
            semantic: TextureSemantic::NormalMap,
            alpha: AlphaCoveragePolicy::Coverage { threshold: 127 },
            ..MipGenerationSettings::default()
        },
    )
    .unwrap();
    let artifact = ImageMipArtifact::from_chain(&chain).unwrap();
    let first = artifact.encode().unwrap();
    let second = artifact.encode().unwrap();
    assert_eq!(first, second);

    let decoded = ImageMipArtifact::decode(&first).unwrap();
    assert_eq!(decoded, artifact);
    let image = decoded.to_image().unwrap();
    assert_eq!(image.mip_level_count(), 4);
    assert_eq!(image.mip_extent(1), Some(Extent3d::new_2d(2, 3)));
}

#[test]
fn mip_artifact_rejects_truncation_trailing_bytes_and_invalid_dimensions() {
    let artifact = ImageArtifact {
        width: 2,
        height: 2,
        rgba8: vec![0; 16],
    }
    .encode_with_mips(MipGenerationSettings::default())
    .unwrap();

    assert_eq!(
        ImageMipArtifact::decode(&artifact[..artifact.len() - 1]),
        Err(ImageArtifactError::Truncated)
    );

    let mut trailing = artifact.clone();
    trailing.push(0);
    assert_eq!(
        ImageMipArtifact::decode(&trailing),
        Err(ImageArtifactError::TrailingBytes)
    );

    let mut invalid_dimensions = artifact;
    invalid_dimensions[28..32].copy_from_slice(&3u32.to_le_bytes());
    assert_eq!(
        ImageMipArtifact::decode(&invalid_dimensions),
        Err(ImageArtifactError::InvalidMipChain)
    );
}

#[test]
fn mip_artifact_rejects_budget_overflow_before_allocating_payload() {
    let mut bytes = ImageArtifact {
        width: 1,
        height: 1,
        rgba8: vec![0; 4],
    }
    .encode_with_mips(MipGenerationSettings::default())
    .unwrap();
    // The first level payload length begins immediately after its 16-byte
    // level header's width and height fields. u64::MAX must fail closed in
    // checked offset arithmetic without allocating or copying payload bytes.
    bytes[36..44].copy_from_slice(&u64::MAX.to_le_bytes());
    assert_eq!(
        ImageMipArtifact::decode(&bytes),
        Err(ImageArtifactError::BudgetExceeded)
    );
}

#[test]
fn upload_plan_pads_each_row_but_preserves_tight_image_bytes() {
    let image = Image::new(
        Extent3d::new_2d(2, 2),
        TextureFormat::Rgba8Unorm,
        vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16],
    )
    .unwrap();
    let plan = ImageUploadPlan::for_image(&image).unwrap();

    assert_eq!(plan.extent(), image.extent());
    assert_eq!(plan.format(), image.format());
    assert_eq!(plan.layout().offset, 0);
    assert_eq!(
        plan.layout().bytes_per_row,
        Some(IMAGE_UPLOAD_ROW_ALIGNMENT)
    );
    assert_eq!(plan.layout().rows_per_image, Some(2));
    assert_eq!(plan.len(), (IMAGE_UPLOAD_ROW_ALIGNMENT * 2) as usize);
    assert_eq!(&plan.data()[..8], &image.data()[..8]);
    assert_eq!(&plan.data()[256..264], &image.data()[8..16]);
    assert!(plan.data()[8..256].iter().all(|byte| *byte == 0));
    assert!(plan.data()[264..].iter().all(|byte| *byte == 0));
}

#[test]
fn upload_plan_writes_only_through_the_supplied_real_texture_id() {
    let image = Image::new(
        Extent3d::new_2d(1, 2),
        TextureFormat::R8Unorm,
        vec![0x11, 0x22],
    )
    .unwrap();
    let plan = ImageUploadPlan::for_image(&image).unwrap();
    let queue = CapturingQueue::default();
    let texture = TextureId::from_parts(9, 4);

    plan.write_to(&queue, texture);

    let writes = queue.writes.lock().unwrap();
    assert_eq!(writes.len(), 1);
    let (destination, data, layout) = &writes[0];
    assert_eq!(destination.texture, texture);
    assert_eq!(destination.size, image.extent());
    assert_eq!(*layout, plan.layout());
    assert_eq!(data, plan.data());
}

#[test]
fn upload_plan_writes_each_mip_with_its_own_extent_and_padding() {
    let rgba8 = (0..(3 * 5 * 4))
        .map(|value| value as u8)
        .collect::<Vec<_>>();
    let chain = ImageMipChain::generate(3, 5, &rgba8, MipGenerationSettings::default())
        .expect("odd dimensions generate a complete chain");
    let image = Image::from_mip_chain(
        &chain,
        TextureFormat::Rgba8Unorm,
        SamplerDescriptor::default(),
    )
    .unwrap();
    let plan = ImageUploadPlan::for_image(&image).unwrap();
    assert_eq!(plan.mip_count(), 4);
    assert_eq!(plan.mip(0).unwrap().extent(), Extent3d::new_2d(3, 5));
    assert_eq!(plan.mip(1).unwrap().extent(), Extent3d::new_2d(2, 3));
    assert_eq!(plan.mip(2).unwrap().extent(), Extent3d::new_2d(1, 2));
    assert_eq!(plan.mip(3).unwrap().extent(), Extent3d::new_2d(1, 1));

    let queue = CapturingQueue::default();
    let texture = TextureId::from_parts(12, 7);
    plan.write_to(&queue, texture);
    let writes = queue.writes.lock().unwrap();
    assert_eq!(writes.len(), 4);
    for (level, (destination, data, layout)) in writes.iter().enumerate() {
        assert_eq!(destination.texture, texture);
        assert_eq!(destination.mip_level, level as u32);
        assert_eq!(destination.size, plan.mip(level as u32).unwrap().extent());
        assert_eq!(*layout, plan.mip(level as u32).unwrap().layout());
        assert_eq!(data, plan.mip(level as u32).unwrap().data());
        assert_eq!(
            layout.bytes_per_row.unwrap() % IMAGE_UPLOAD_ROW_ALIGNMENT,
            0
        );
    }
}

#[test]
fn upload_plan_preserves_layer_boundaries_in_rows_per_image_layout() {
    let extent = Extent3d {
        width: 1,
        height: 2,
        depth_or_array_layers: 2,
    };
    let image = Image::new(extent, TextureFormat::R8Unorm, vec![1, 2, 3, 4]).unwrap();
    let plan = ImageUploadPlan::for_image(&image).unwrap();

    assert_eq!(
        plan.layout().bytes_per_row,
        Some(IMAGE_UPLOAD_ROW_ALIGNMENT)
    );
    assert_eq!(plan.layout().rows_per_image, Some(2));
    assert_eq!(plan.len(), (IMAGE_UPLOAD_ROW_ALIGNMENT * 4) as usize);
    assert_eq!(plan.data()[0], 1);
    assert_eq!(plan.data()[256], 2);
    assert_eq!(plan.data()[512], 3);
    assert_eq!(plan.data()[768], 4);
}

#[test]
fn upload_row_index_arithmetic_fails_closed_on_usize_overflow() {
    assert_eq!(
        checked_row_range(usize::MAX, 2, 1, usize::MAX),
        Err(crate::ImageUploadPlanError::StagingSizeOverflow)
    );
    assert_eq!(
        checked_row_range(usize::MAX - 1, 1, 2, usize::MAX),
        Err(crate::ImageUploadPlanError::StagingSizeOverflow)
    );
    assert_eq!(checked_row_range(3, 8, 4, 32), Ok(24..28));
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
