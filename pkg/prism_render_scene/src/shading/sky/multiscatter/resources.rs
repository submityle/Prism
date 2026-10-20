//! Global 2D altitude/sun-zenith LUT allocation.
use bevy_ecs::prelude::*;
use bevy_render::{render_resource::*, renderer::RenderDevice};

use super::abi::SkyMultiscatterSampleBudget;
pub(crate) const LUT_FORMAT: TextureFormat = TextureFormat::Rgba16Float;
pub(crate) const LUT_WIDTH: u32 = 256;
pub(crate) const LUT_HEIGHT: u32 = 128;
#[derive(Resource)]
#[expect(dead_code, reason = "owns the GPU allocation backing the view")]
pub(crate) struct SkyMultiscatterLut {
    texture: Texture,
    view: TextureView,
    pub(crate) width: u32,
    pub(crate) height: u32,
}

/// Successful bake state shared with dependent LUT passes.
#[derive(Resource, Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct SkyMultiscatterGeneration {
    budget: Option<SkyMultiscatterSampleBudget>,
    revision: u64,
}

impl SkyMultiscatterGeneration {
    pub(crate) fn needs_rebuild(&self, budget: SkyMultiscatterSampleBudget) -> bool {
        self.budget != Some(budget)
    }

    pub(crate) fn mark_generated(&mut self, budget: SkyMultiscatterSampleBudget) {
        if self.needs_rebuild(budget) {
            self.budget = Some(budget);
            self.revision = self.revision.checked_add(1).unwrap_or(1);
        }
    }

    pub(crate) fn revision(&self) -> Option<u64> {
        self.budget.map(|_| self.revision)
    }
}
impl SkyMultiscatterLut {
    pub(crate) fn view(&self) -> &TextureView {
        &self.view
    }
}
pub(crate) fn init_sky_multiscatter_lut(mut commands: Commands, device: Res<RenderDevice>) {
    let texture = device.create_texture(&TextureDescriptor {
        label: Some("prism sky multiple-scattering LUT"),
        size: Extent3d {
            width: LUT_WIDTH,
            height: LUT_HEIGHT,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: TextureDimension::D2,
        format: LUT_FORMAT,
        usage: TextureUsages::STORAGE_BINDING | TextureUsages::TEXTURE_BINDING,
        view_formats: &[],
    });
    let view = texture.create_view(&TextureViewDescriptor::default());
    commands.insert_resource(SkyMultiscatterLut {
        texture,
        view,
        width: LUT_WIDTH,
        height: LUT_HEIGHT,
    });
    commands.init_resource::<SkyMultiscatterGeneration>();
}

#[cfg(test)]
mod tests {
    use super::*;
    use prism_render_shading::LobeSampleBudget;

    #[test]
    fn generation_lifecycle_rebuilds_only_for_effective_budget_changes() {
        let high = SkyMultiscatterSampleBudget::new(64, 16);
        let low = SkyMultiscatterSampleBudget::new(16, 8);
        let mut generation = SkyMultiscatterGeneration::default();

        assert_eq!(generation.revision(), None);
        assert!(generation.needs_rebuild(high));

        generation.mark_generated(high);
        assert_eq!(generation.revision(), Some(1));
        assert!(!generation.needs_rebuild(high));

        generation.mark_generated(high);
        assert_eq!(generation.revision(), Some(1));

        assert!(generation.needs_rebuild(low));
        generation.mark_generated(low);
        assert_eq!(generation.revision(), Some(2));
        assert!(!generation.needs_rebuild(low));
    }

    #[test]
    fn clamped_equivalent_budgets_do_not_rebuild_twice() {
        let mut generation = SkyMultiscatterGeneration::default();
        let zero = SkyMultiscatterSampleBudget::new(0, 0);
        let floors = SkyMultiscatterSampleBudget::new(
            LobeSampleBudget::MIN_MULTISCATTER_DIR,
            LobeSampleBudget::MIN_MULTISCATTER_MARCH,
        );
        assert_eq!(zero, floors);
        generation.mark_generated(zero);
        assert!(!generation.needs_rebuild(floors));
        assert_eq!(generation.revision(), Some(1));
    }
}
