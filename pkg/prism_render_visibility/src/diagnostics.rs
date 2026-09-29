#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct VisibilityDiagnostics {
    pub input_instances: u32,
    pub visible_instances: u32,
    pub stale_handles: u32,
    pub layer_rejected: u32,
    pub frustum_rejected: u32,
    pub occlusion_rejected: u32,
    pub missing_geometry: u32,
    pub missing_material: u32,
    pub lod_fallbacks: u32,
    pub shadow_casters: u32,
    pub ray_scene_instances: u32,
    pub material_bins: u32,
    pub overflowed: bool,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_diagnostics_start_fully_zeroed() {
        let stats = VisibilityDiagnostics::default();
        assert_eq!(stats.input_instances, 0);
        assert_eq!(stats.visible_instances, 0);
        assert_eq!(stats.stale_handles, 0);
        assert_eq!(stats.layer_rejected, 0);
        assert_eq!(stats.frustum_rejected, 0);
        assert_eq!(stats.occlusion_rejected, 0);
        assert_eq!(stats.missing_geometry, 0);
        assert_eq!(stats.missing_material, 0);
        assert_eq!(stats.lod_fallbacks, 0);
        assert_eq!(stats.shadow_casters, 0);
        assert_eq!(stats.ray_scene_instances, 0);
        assert_eq!(stats.material_bins, 0);
        assert!(!stats.overflowed);
    }

    #[test]
    fn rejection_counters_accumulate_independently() {
        // Model the accumulation pattern used by `cull_view`: every rejection
        // reason bumps its own counter without disturbing the others.
        let mut stats = VisibilityDiagnostics {
            input_instances: 6,
            ..Default::default()
        };
        stats.stale_handles += 1;
        stats.layer_rejected += 1;
        stats.frustum_rejected += 2;
        stats.occlusion_rejected += 1;
        stats.visible_instances += 1;

        assert_eq!(stats.input_instances, 6);
        assert_eq!(stats.stale_handles, 1);
        assert_eq!(stats.layer_rejected, 1);
        assert_eq!(stats.frustum_rejected, 2);
        assert_eq!(stats.occlusion_rejected, 1);
        assert_eq!(stats.visible_instances, 1);
        // A rejection reason that never fired stays at zero.
        assert_eq!(stats.missing_material, 0);
        // The tallied outcomes account for every input instance.
        let accounted = stats.visible_instances
            + stats.stale_handles
            + stats.layer_rejected
            + stats.frustum_rejected
            + stats.occlusion_rejected;
        assert_eq!(accounted, stats.input_instances);
    }

    #[test]
    fn equality_and_copy_reflect_field_state() {
        let base = VisibilityDiagnostics {
            visible_instances: 3,
            material_bins: 2,
            ..Default::default()
        };
        let copy = base;
        assert_eq!(base, copy);

        let mut mutated = base;
        mutated.overflowed = true;
        assert_ne!(base, mutated);
        // The overflow flag is the only differing field.
        mutated.overflowed = false;
        assert_eq!(base, mutated);
    }
}
