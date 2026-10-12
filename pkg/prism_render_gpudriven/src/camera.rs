//! The orthographic camera the scene draw and the naive baseline share.
//!
//! The camera is deliberately axis-aligned and pixel-exact: world XY maps
//! linearly to target pixels, so the CPU reference can predict the exact pixel
//! an instance's centre lands on without reproducing a projection matrix.

use crate::quad::QUAD_HALF_EXTENT;

/// The byte size of the frame uniform: three `vec2<f32>` plus one `vec2<f32>`
/// of geometry parameters.
pub const FRAME_UNIFORM_BYTES: u64 = 32;

/// An axis-aligned orthographic view onto a square or rectangular target.
#[derive(Clone, Copy, PartialEq, Debug)]
pub struct OrthoCamera {
    /// The world-space XY of the view's top-left corner.
    pub world_min: [f32; 2],
    /// The world-space width and height the view covers.
    pub world_size: [f32; 2],
    /// The target's size in pixels.
    pub resolution: [f32; 2],
}

impl OrthoCamera {
    /// A camera framing `world_size` from `world_min` on a `resolution`-pixel target.
    #[must_use]
    pub fn new(world_min: [f32; 2], world_size: [f32; 2], resolution: [f32; 2]) -> Self {
        assert!(
            world_size[0] > 0.0 && world_size[1] > 0.0,
            "the view needs a positive size"
        );
        assert!(
            resolution[0] > 0.0 && resolution[1] > 0.0,
            "the target needs a positive size"
        );
        Self {
            world_min,
            world_size,
            resolution,
        }
    }

    /// A camera centred on the origin, framing `world_size`.
    #[must_use]
    pub fn centered(world_size: [f32; 2], resolution: [f32; 2]) -> Self {
        Self::new(
            [-world_size[0] * 0.5, -world_size[1] * 0.5],
            world_size,
            resolution,
        )
    }

    /// The fractional pixel `(x, y)` a world point lands on, measured from the
    /// target's top-left corner.
    #[must_use]
    pub fn pixel(&self, world: [f32; 2]) -> [f64; 2] {
        [
            f64::from(world[0] - self.world_min[0]) / f64::from(self.world_size[0])
                * f64::from(self.resolution[0]),
            f64::from(world[1] - self.world_min[1]) / f64::from(self.world_size[1])
                * f64::from(self.resolution[1]),
        ]
    }

    /// The integer pixel index containing `world`, or `None` when it falls
    /// outside the target.
    #[must_use]
    pub fn pixel_index(&self, world: [f32; 2]) -> Option<[u32; 2]> {
        let pixel = self.pixel(world);
        let x = pixel[0].floor();
        let y = pixel[1].floor();
        if x < 0.0
            || y < 0.0
            || x >= f64::from(self.resolution[0])
            || y >= f64::from(self.resolution[1])
        {
            return None;
        }
        Some([x as u32, y as u32])
    }
}

/// The uniform the scene and baseline vertex shaders read.
///
/// It bundles the camera with the two geometry constants the shaders would
/// otherwise have to hardcode ([`QUAD_HALF_EXTENT`] and
/// [`crate::draw::INDICES_PER_INSTANCE`]), so a change to either is a host-side
/// edit rather than a silent shader drift.
#[derive(Clone, Copy, PartialEq, Debug)]
pub struct FrameUniform {
    /// The orthographic view.
    pub camera: OrthoCamera,
    /// The quad's half-extent in world units.
    pub quad_half_extent: f32,
    /// The number of indices one instance draws.
    pub indices_per_instance: u32,
}

impl FrameUniform {
    /// The uniform for `camera` with the crate's quad geometry.
    #[must_use]
    pub const fn new(camera: OrthoCamera, indices_per_instance: u32) -> Self {
        Self {
            camera,
            quad_half_extent: QUAD_HALF_EXTENT,
            indices_per_instance,
        }
    }

    /// The uniform packed at [`FRAME_UNIFORM_BYTES`].
    #[must_use]
    pub fn to_bytes(self) -> [u8; FRAME_UNIFORM_BYTES as usize] {
        let mut bytes = [0u8; FRAME_UNIFORM_BYTES as usize];
        let mut write_f32 = |offset: usize, value: f32| {
            bytes[offset..offset + 4].copy_from_slice(&value.to_le_bytes());
        };
        write_f32(0, self.camera.resolution[0]);
        write_f32(4, self.camera.resolution[1]);
        write_f32(8, self.camera.world_min[0]);
        write_f32(12, self.camera.world_min[1]);
        write_f32(16, self.camera.world_size[0]);
        write_f32(20, self.camera.world_size[1]);
        write_f32(24, self.quad_half_extent);
        let indices = self.indices_per_instance as f32;
        write_f32(28, indices);
        bytes
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn world_maps_linearly_to_pixels() {
        let camera = OrthoCamera::centered([2.0, 2.0], [64.0, 64.0]);
        assert_eq!(camera.pixel([-1.0, -1.0]), [0.0, 0.0]);
        assert_eq!(camera.pixel([1.0, 1.0]), [64.0, 64.0]);
        assert_eq!(camera.pixel([0.0, 0.0]), [32.0, 32.0]);
        assert_eq!(camera.pixel_index([0.0, 0.0]), Some([32, 32]));
        assert_eq!(camera.pixel_index([1.5, 0.0]), None);
        assert_eq!(camera.pixel_index([0.0, -1.5]), None);
    }

    #[test]
    fn frame_uniform_packs_the_geometry_params() {
        let uniform = FrameUniform::new(OrthoCamera::centered([1.0, 1.0], [8.0, 8.0]), 4);
        let bytes = uniform.to_bytes();
        assert_eq!(bytes.len() as u64, FRAME_UNIFORM_BYTES);
        assert_eq!(
            f32::from_le_bytes([bytes[24], bytes[25], bytes[26], bytes[27]]),
            QUAD_HALF_EXTENT
        );
        assert_eq!(
            f32::from_le_bytes([bytes[28], bytes[29], bytes[30], bytes[31]]),
            4.0
        );
    }
}
