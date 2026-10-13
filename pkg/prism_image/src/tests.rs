//! Behavioural tests for the image asset.

use alloc::vec;
use alloc::vec::Vec;

use prism_render_driver::{AddressMode, Extent3d, FilterMode, SamplerDescriptor, TextureFormat};

use crate::{Image, ImageArtifact, ImageError};

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
