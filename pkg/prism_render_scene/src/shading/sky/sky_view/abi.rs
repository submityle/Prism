//! Host ABI for the physical-sky sky-view LUT compute shader.
use bytemuck::{Pod, Zeroable};
/// Must match `@workgroup_size(8, 8, 1)` in `sky_view_lut.wesl`.
pub(crate) const WORKGROUP_SIZE: u32 = 8;
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable, PartialEq)]
pub(crate) struct GpuSkyViewParams {
    pub width: u32,
    pub height: u32,
    pub samples: u32,
    pub sun_samples: u32,
}
impl GpuSkyViewParams {
    pub(crate) fn new(width: u32, height: u32, samples: u32, sun_samples: u32) -> Self {
        Self {
            width,
            height,
            samples: samples.max(1),
            sun_samples: sun_samples.max(1),
        }
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn immediate_layout_is_16_byte_aligned() {
        assert_eq!(size_of::<GpuSkyViewParams>(), 16);
        assert_eq!(align_of::<GpuSkyViewParams>(), 4);
    }
}
