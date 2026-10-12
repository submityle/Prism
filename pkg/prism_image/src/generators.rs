//! Deterministic procedural image generators for tests and demos.
//!
//! Every generator is a pure function of its arguments: the same input yields
//! byte-identical output, so a generated image can serve as a golden fixture
//! without being stored. Nothing here is random or reads the clock.

use prism_render_driver::Extent3d;

use crate::format::ImageFormat;
use crate::image::Image;

/// Which axis a gradient sweeps along.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Axis {
    Horizontal,
    Vertical,
}

/// Builds a checkerboard: a `width` x `height` [`ImageFormat::Rgba8Unorm`]
/// image of `cell`-texel squares alternating between `color_a` and `color_b`.
///
/// Texel `(x, y)` takes `color_a` when `(x / cell + y / cell)` is even and
/// `color_b` otherwise, so the top-left cell (`x < cell`, `y < cell`) is
/// `color_a`. A `cell` of zero is treated as one so the checker stays defined.
#[must_use]
pub fn checkerboard(
    width: u32,
    height: u32,
    cell: u32,
    color_a: [u8; 4],
    color_b: [u8; 4],
) -> Image {
    let cell = cell.max(1);
    let mut data = Vec::with_capacity(buffer_len(width, height));
    for y in 0..height {
        for x in 0..width {
            let even = (x / cell + y / cell).is_multiple_of(2);
            data.extend_from_slice(if even { &color_a } else { &color_b });
        }
    }
    Image::from_parts(
        Extent3d::new_2d(width, height),
        ImageFormat::Rgba8Unorm,
        data,
    )
}

/// Builds a horizontal, black-to-white, opaque [`ImageFormat::Rgba8Unorm`]
/// gradient of `width` x `height` texels.
///
/// The leftmost column is black, the rightmost white, and interior columns step
/// evenly between them (`index * 255 / (width - 1)`, integer division), so the
/// ramp is deterministic and includes both endpoints. A single-column image is
/// all black.
#[must_use]
pub fn gradient_horizontal(width: u32, height: u32) -> Image {
    gradient(width, height, Axis::Horizontal)
}

/// Builds a vertical, black-to-white, opaque [`ImageFormat::Rgba8Unorm`]
/// gradient of `width` x `height` texels.
///
/// The top row is black, the bottom white, and interior rows step evenly between
/// them, mirroring [`gradient_horizontal`] with the roles of the axes swapped.
/// A single-row image is all black.
#[must_use]
pub fn gradient_vertical(width: u32, height: u32) -> Image {
    gradient(width, height, Axis::Vertical)
}

/// Shared implementation of the two gradient directions.
fn gradient(width: u32, height: u32, axis: Axis) -> Image {
    let mut data = Vec::with_capacity(buffer_len(width, height));
    for y in 0..height {
        for x in 0..width {
            let level = match axis {
                Axis::Horizontal => ramp(x, width),
                Axis::Vertical => ramp(y, height),
            };
            data.extend_from_slice(&[level, level, level, 255]);
        }
    }
    Image::from_parts(
        Extent3d::new_2d(width, height),
        ImageFormat::Rgba8Unorm,
        data,
    )
}

/// The `0..=255` level for position `index` of `length` samples, inclusive of
/// both endpoints. A degenerate length of one is level zero.
fn ramp(index: u32, length: u32) -> u8 {
    if length <= 1 {
        return 0;
    }
    ((u64::from(index) * 255) / u64::from(length - 1)) as u8
}

/// The tightly packed buffer size for a `width` x `height` `RGBA8` image.
fn buffer_len(width: u32, height: u32) -> usize {
    u64::from(width)
        .saturating_mul(u64::from(height))
        .saturating_mul(4)
        .try_into()
        .unwrap_or(usize::MAX)
}
