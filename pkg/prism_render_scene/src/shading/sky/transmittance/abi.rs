//! Host ABI for the physical-sky transmittance LUT compute shader.
use bytemuck::{Pod, Zeroable};
/// Must match `@workgroup_size(8, 8, 1)` in `sky_transmittance_lut.wesl`.
pub(crate) const WORKGROUP_SIZE: u32 = 8;
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable, PartialEq)]
pub(crate) struct GpuSkyTransmittanceParams {
    pub width: u32,
    pub height: u32,
    pub samples: u32,
    pub pad: u32,
}
impl GpuSkyTransmittanceParams {
    pub(crate) fn new(width: u32, height: u32, samples: u32) -> Self {
        Self {
            width,
            height,
            samples: samples.max(1),
            pad: 0,
        }
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn immediate_layout_is_16_byte_aligned() {
        assert_eq!(size_of::<GpuSkyTransmittanceParams>(), 16);
        assert_eq!(align_of::<GpuSkyTransmittanceParams>(), 4);
    }
}
