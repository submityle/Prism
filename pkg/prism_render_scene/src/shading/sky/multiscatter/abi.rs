//! Host ABI for the physical-sky LUT compute shader.
use bytemuck::{Pod, Zeroable};
/// Must match `@workgroup_size(8, 8, 1)` in `sky_multiscatter_lut.wesl`.
pub(crate) const WORKGROUP_SIZE: u32 = 8;
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable, PartialEq)]
pub(crate) struct GpuSkyMultiscatterParams {
    pub width: u32,
    pub height: u32,
    pub dir_samples: u32,
    pub march_samples: u32,
}
impl GpuSkyMultiscatterParams {
    pub(crate) fn new(width: u32, height: u32, dir_samples: u32, march_samples: u32) -> Self {
        Self {
            width,
            height,
            dir_samples: dir_samples.max(1),
            march_samples: march_samples.max(1),
        }
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn immediate_layout_is_16_byte_aligned() {
        assert_eq!(size_of::<GpuSkyMultiscatterParams>(), 16);
        assert_eq!(align_of::<GpuSkyMultiscatterParams>(), 4);
    }
}
