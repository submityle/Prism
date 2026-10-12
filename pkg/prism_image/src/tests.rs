//! Verification for the image container, its formats, the generators, and the
//! RHI upload path.
//!
//! The CPU tests pin the stride math, the round-trip constructor, and generator
//! determinism. The GPU test uploads a real texture and reads it back through
//! the wgpu backend, asserting the pixels survive verbatim; without an adapter
//! it returns early, so the suite still passes on a GPU-less host.

use prism_render_driver::{Extent3d, TextureFormat, TextureUsages};
use prism_render_driver_wgpu::{WgpuDevice, WgpuInstance, WgpuQueue};

use crate::format::ImageFormat;
use crate::generators::{checkerboard, gradient_horizontal, gradient_vertical};
use crate::gpu::UPLOAD_USAGE;
use crate::image::{Image, ImageError};

/// The two checkerboard colours, chosen so a channel swap is observable.
const RED: [u8; 4] = [255, 0, 0, 255];
const BLUE: [u8; 4] = [0, 0, 255, 255];

/// Whether a missing adapter must fail the test instead of skipping it.
///
/// Set `PRISM_REQUIRE_GPU` to opt into strictness, so a run meant to exercise
/// hardware cannot silently degrade into a no-op.
fn require_gpu() -> bool {
    std::env::var_os("PRISM_REQUIRE_GPU").is_some()
}

/// Brings up a wgpu device/queue pair, or `None` when no GPU is available.
fn device_or_skip() -> Option<(WgpuDevice, WgpuQueue)> {
    WgpuInstance::new().request_default_device().ok()
}

#[test]
fn extent_and_stride_are_tightly_packed() {
    let image = Image::from_rgba8(4, 3, vec![0; 48]).expect("48 is exactly 4 * 3 * 4 bytes");

    assert_eq!(image.extent(), Extent3d::new_2d(4, 3));
    assert_eq!(image.width(), 4);
    assert_eq!(image.height(), 3);
    assert_eq!(image.format(), ImageFormat::Rgba8Unorm);
    assert_eq!(image.row_stride(), 16, "four texels of four bytes each");
    assert_eq!(image.len(), 48);
    assert!(!image.is_empty());

    let layout = image.data_layout();
    assert_eq!(layout.offset, 0);
    assert_eq!(layout.bytes_per_row, Some(16));
    assert_eq!(layout.rows_per_image, Some(3));
}

#[test]
fn from_rgba8_round_trips_external_bytes() {
    let bytes: Vec<u8> = (0..16).collect();
    let image = Image::from_rgba8(2, 2, bytes.clone()).expect("16 is exactly 2 * 2 * 4 bytes");

    assert_eq!(image.data(), bytes.as_slice(), "bytes are stored verbatim");
    assert_eq!(image.texel(0, 0), [0, 1, 2, 3]);
    assert_eq!(image.texel(1, 0), [4, 5, 6, 7]);
    assert_eq!(
        image.texel(0, 1),
        [8, 9, 10, 11],
        "row 1 starts after row_stride"
    );
    assert_eq!(image.texel(1, 1), [12, 13, 14, 15]);
}

#[test]
fn construction_rejects_wrong_lengths_and_zero_extents() {
    assert_eq!(
        Image::from_rgba8(2, 2, vec![0; 15]),
        Err(ImageError::LengthMismatch {
            expected: 16,
            actual: 15,
        }),
    );
    assert_eq!(
        Image::new(Extent3d::new_2d(0, 4), ImageFormat::Rgba8Unorm, Vec::new()),
        Err(ImageError::ZeroExtent),
    );
}

#[test]
fn generators_are_deterministic() {
    assert_eq!(
        checkerboard(8, 8, 2, RED, BLUE),
        checkerboard(8, 8, 2, RED, BLUE),
        "the same checkerboard request yields identical bytes",
    );
    assert_eq!(
        gradient_horizontal(5, 3),
        gradient_horizontal(5, 3),
        "the same horizontal gradient request yields identical bytes",
    );
    assert_eq!(
        gradient_vertical(5, 3),
        gradient_vertical(5, 3),
        "the same vertical gradient request yields identical bytes",
    );
}

#[test]
fn checkerboard_alternates_in_cells() {
    let image = checkerboard(4, 4, 2, RED, BLUE);

    assert_eq!(image.len(), 64);
    assert_eq!(image.format(), ImageFormat::Rgba8Unorm);
    assert_eq!(image.texel(0, 0), RED, "top-left cell is color_a");
    assert_eq!(image.texel(1, 1), RED, "the whole first cell is color_a");
    assert_eq!(image.texel(2, 0), BLUE, "the next cell right flips");
    assert_eq!(image.texel(0, 2), BLUE, "the next cell down flips");
    assert_eq!(image.texel(3, 3), RED, "the diagonal cell flips back");
}

#[test]
fn gradients_hit_both_endpoints() {
    let horizontal = gradient_horizontal(3, 1);
    assert_eq!(horizontal.texel(0, 0), [0, 0, 0, 255], "left is black");
    assert_eq!(
        horizontal.texel(1, 0),
        [127, 127, 127, 255],
        "middle steps evenly",
    );
    assert_eq!(
        horizontal.texel(2, 0),
        [255, 255, 255, 255],
        "right is white"
    );

    let vertical = gradient_vertical(1, 3);
    assert_eq!(vertical.texel(0, 0), [0, 0, 0, 255], "top is black");
    assert_eq!(
        vertical.texel(0, 2),
        [255, 255, 255, 255],
        "bottom is white"
    );

    let degenerate = gradient_horizontal(1, 1);
    assert_eq!(
        degenerate.texel(0, 0),
        [0, 0, 0, 255],
        "a single column is all black rather than dividing by zero",
    );
}

#[test]
fn image_formats_map_to_rhi_texture_formats() {
    let table = [
        (ImageFormat::Rgba8Unorm, TextureFormat::Rgba8Unorm, false),
        (
            ImageFormat::Rgba8UnormSrgb,
            TextureFormat::Rgba8UnormSrgb,
            true,
        ),
        (ImageFormat::Bgra8Unorm, TextureFormat::Bgra8Unorm, false),
        (
            ImageFormat::Bgra8UnormSrgb,
            TextureFormat::Bgra8UnormSrgb,
            true,
        ),
    ];

    for (image_format, texture_format, srgb) in table {
        assert_eq!(image_format.texture_format(), texture_format);
        assert_eq!(image_format.is_srgb(), srgb);
        assert_eq!(image_format.bytes_per_texel(), 4);
        assert_eq!(
            texture_format.bytes_per_texel(),
            image_format.bytes_per_texel(),
            "the image's stride and the RHI texture's texel size must agree",
        );
    }
}

#[test]
fn upload_usage_permits_binding_a_copy_in_and_a_copy_out() {
    assert!(UPLOAD_USAGE.contains(TextureUsages::TEXTURE_BINDING));
    assert!(UPLOAD_USAGE.contains(TextureUsages::COPY_DST));
    assert!(UPLOAD_USAGE.contains(TextureUsages::COPY_SRC));
}

#[test]
fn gpu_upload_reads_back_the_source_bytes_exactly() {
    let Some((device, queue)) = device_or_skip() else {
        assert!(
            !require_gpu(),
            "PRISM_REQUIRE_GPU is set but no adapter was found",
        );
        return;
    };

    let source = checkerboard(16, 16, 4, RED, BLUE);
    let gpu = source.upload(&device, &queue);

    assert_eq!(gpu.format, TextureFormat::Rgba8Unorm);
    assert_eq!(gpu.size, Extent3d::new_2d(16, 16));

    let readback = queue.read_texture_rgba8(gpu.texture, 16, 16);
    assert_eq!(
        readback,
        source.data(),
        "the uploaded checkerboard must read back byte-for-byte",
    );

    gpu.destroy(&device);
}
