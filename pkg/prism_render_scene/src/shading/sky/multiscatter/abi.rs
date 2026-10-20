//! Host ABI for the physical-sky LUT compute shader.
use bytemuck::{Pod, Zeroable};
use prism_render_shading::LobeSampleBudget;
/// Must match `@workgroup_size(8, 8, 1)` in `sky_multiscatter_lut.wesl`.
pub(crate) const WORKGROUP_SIZE: u32 = 8;

/// The only lobe-budget fields consumed by the sky multiple-scattering pass.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(crate) struct SkyMultiscatterSampleBudget {
    pub(crate) dir_samples: u32,
    pub(crate) march_samples: u32,
}

impl SkyMultiscatterSampleBudget {
    pub(crate) const fn new(dir_samples: u32, march_samples: u32) -> Self {
        Self {
            dir_samples: clamp_min(dir_samples, LobeSampleBudget::MIN_MULTISCATTER_DIR),
            march_samples: clamp_min(march_samples, LobeSampleBudget::MIN_MULTISCATTER_MARCH),
        }
    }
}

const fn clamp_min(value: u32, floor: u32) -> u32 {
    if value < floor {
        floor
    } else {
        value
    }
}

impl From<LobeSampleBudget> for SkyMultiscatterSampleBudget {
    fn from(budget: LobeSampleBudget) -> Self {
        Self::new(
            budget.multiscatter_dir_samples,
            budget.multiscatter_march_samples,
        )
    }
}

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
        Self::for_budget(
            width,
            height,
            SkyMultiscatterSampleBudget::new(dir_samples, march_samples),
        )
    }

    pub(crate) const fn for_budget(
        width: u32,
        height: u32,
        budget: SkyMultiscatterSampleBudget,
    ) -> Self {
        Self {
            width,
            height,
            dir_samples: budget.dir_samples,
            march_samples: budget.march_samples,
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

    #[test]
    fn sample_counts_clamp_to_the_shared_model_floors() {
        let params = GpuSkyMultiscatterParams::new(256, 128, 0, 1);
        assert_eq!(params.dir_samples, LobeSampleBudget::MIN_MULTISCATTER_DIR);
        assert_eq!(
            params.march_samples,
            LobeSampleBudget::MIN_MULTISCATTER_MARCH
        );
    }

    #[test]
    fn high_budget_maps_to_the_historical_gpu_output() {
        let budget =
            SkyMultiscatterSampleBudget::from(prism_render_shading::ShadingQuality::High.budget());
        let params = GpuSkyMultiscatterParams::for_budget(256, 128, budget);
        assert_eq!((params.dir_samples, params.march_samples), (64, 16));
    }
}
