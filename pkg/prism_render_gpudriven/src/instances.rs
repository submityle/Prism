//! Per-instance transform and color: the data the scene draw reads from a
//! storage buffer indexed by the surviving instance id.
//!
//! Unlike [`crate::cull`], which only needs an AABB per instance, the draw needs
//! the instance's world placement and its color. Both live in one padded record
//! so a single storage buffer holds the whole scene and the shader indexes it
//! by the id the cull pass emitted.

use alloc::vec::Vec;

/// The storage stride of one instance record: `center.xyz` + pad, `color.rgb` + pad.
pub const INSTANCE_STRIDE: u64 = 32;

/// One instance's world-space placement and linear color.
#[derive(Clone, Copy, PartialEq, Debug)]
pub struct Instance {
    /// The world-space centre of the instance's quad.
    pub center: [f32; 3],
    /// The instance's linear RGB color, in `0.0..=1.0`.
    pub color: [f32; 3],
}

impl Instance {
    /// A record from its centre and color.
    #[must_use]
    pub const fn new(center: [f32; 3], color: [f32; 3]) -> Self {
        Self { center, color }
    }

    /// The record packed at [`INSTANCE_STRIDE`], ready for an upload.
    #[must_use]
    pub fn to_bytes(self) -> [u8; INSTANCE_STRIDE as usize] {
        let mut bytes = [0u8; INSTANCE_STRIDE as usize];
        for (slot, component) in bytes[0..12].chunks_exact_mut(4).zip(self.center) {
            slot.copy_from_slice(&component.to_le_bytes());
        }
        for (slot, component) in bytes[16..28].chunks_exact_mut(4).zip(self.color) {
            slot.copy_from_slice(&component.to_le_bytes());
        }
        bytes
    }
}

/// The instance records packed at [`INSTANCE_STRIDE`] each.
#[must_use]
pub fn instance_storage_bytes(instances: &[Instance]) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(instances.len() * INSTANCE_STRIDE as usize);
    for instance in instances {
        bytes.extend_from_slice(&instance.to_bytes());
    }
    bytes
}

/// The color that uniquely identifies instance `id` on screen.
///
/// The id's two low bytes become the red and green channels (normalized) and
/// blue is pinned to `1.0`, so every instance is distinct and none of them is
/// the black background. [`id_color_bytes`] is the exact byte triple a
/// correctly rendered instance produces.
#[must_use]
pub fn id_color(id: u32) -> [f32; 3] {
    let bytes = id_color_bytes(id);
    [
        f32::from(bytes[0]) / 255.0,
        f32::from(bytes[1]) / 255.0,
        f32::from(bytes[2]) / 255.0,
    ]
}

/// The exact 8-bit RGB a correctly rendered instance reads back as.
///
/// An `Rgba8Unorm` target stores `round(value * 255)`, so a shader writing
/// `id_color(id)` produces these bytes verbatim — which makes the pixel check
/// an exact equality rather than a tolerance comparison.
#[must_use]
pub fn id_color_bytes(id: u32) -> [u8; 3] {
    [(id & 0xFF) as u8, ((id >> 8) & 0xFF) as u8, 0xFF]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn instance_records_are_padded_to_the_shader_stride() {
        let bytes = Instance::new([1.0, 2.0, 3.0], [0.25, 0.5, 0.75]).to_bytes();
        assert_eq!(bytes.len() as u64, INSTANCE_STRIDE);
        assert_eq!(
            f32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]),
            1.0
        );
        assert_eq!(
            f32::from_le_bytes([bytes[8], bytes[9], bytes[10], bytes[11]]),
            3.0
        );
        assert_eq!(
            f32::from_le_bytes([bytes[12], bytes[13], bytes[14], bytes[15]]),
            0.0,
            "the float3 is padded, not packed"
        );
        assert_eq!(
            f32::from_le_bytes([bytes[16], bytes[17], bytes[18], bytes[19]]),
            0.25
        );
        assert_eq!(
            f32::from_le_bytes([bytes[28], bytes[29], bytes[30], bytes[31]]),
            0.0
        );
    }

    #[test]
    fn id_colors_round_trip_through_eight_bits() {
        for id in [0u32, 1, 255, 256, 9999] {
            let color = id_color(id);
            let bytes = id_color_bytes(id);
            for (channel, byte) in color.iter().zip(bytes) {
                assert_eq!((channel * 255.0).round() as u8, byte);
            }
            assert_ne!(bytes, [0, 0, 0], "no instance may equal the background");
        }
        assert_ne!(id_color_bytes(1), id_color_bytes(256));
    }
}
