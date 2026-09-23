use alloc::vec::Vec;
use prism_render_architecture::gpu_scene::GeometryHandle;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GeometryLod {
    pub level: u16,
    pub screen_error: f32,
    pub resident: bool,
    pub fallback: bool,
}

#[derive(Clone, Debug, PartialEq)]
pub struct GeometryLodChain {
    pub geometry: GeometryHandle,
    pub lods: Vec<GeometryLod>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct LodSelection {
    pub level: u16,
    pub used_fallback: bool,
}

impl GeometryLodChain {
    pub fn select(
        &self,
        projected_radius: f32,
        lod_scale: f32,
        previous: Option<u16>,
    ) -> Option<LodSelection> {
        let target = 1.0 / (projected_radius.max(1.0e-4) * lod_scale.max(1.0e-4));
        let mut selected = self
            .lods
            .iter()
            .filter(|lod| lod.resident && lod.screen_error <= target)
            .min_by(|a, b| a.screen_error.total_cmp(&b.screen_error));
        if let Some(previous) = previous
            && let Some(previous_lod) = self
                .lods
                .iter()
                .find(|lod| lod.level == previous && lod.resident)
            && selected.is_some_and(|next| {
                (next.screen_error - previous_lod.screen_error).abs() < target * 0.15
            })
        {
            selected = Some(previous_lod);
        }
        selected
            .or_else(|| {
                self.lods
                    .iter()
                    .filter(|lod| lod.resident && lod.fallback)
                    .min_by_key(|lod| lod.level)
            })
            .map(|lod| LodSelection {
                level: lod.level,
                used_fallback: lod.fallback,
            })
    }
}
