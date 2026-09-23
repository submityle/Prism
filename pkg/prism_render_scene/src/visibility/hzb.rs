use bevy_core_pipeline::mip_generation::experimental::depth::ViewDepthPyramid;
use bevy_ecs::prelude::*;
use bevy_platform::collections::{HashMap, HashSet};
use bevy_render::{render_resource::TextureView, view::RetainedViewEntity};

use super::runtime::UnifiedVisibilityState;

/// Prism's temporal ownership metadata for Bevy's public depth-pyramid view.
/// The texture itself remains backend-owned; consumers use this component to
/// distinguish last-frame history from the current frame written later in the
/// Core3d schedule.
#[derive(Component)]
pub(crate) struct PrismViewHzbHistory {
    previous: TextureView,
    current: TextureView,
    mip_count: u32,
    history_epoch: u64,
    previous_valid: bool,
}

pub(crate) fn inspect_hzb_history(
    histories: Query<&PrismViewHzbHistory>,
    mut diagnostics: ResMut<super::runtime::PrismVisibilityDiagnostics>,
) {
    diagnostics.hzb_views = 0;
    diagnostics.hzb_valid_histories = 0;
    for history in &histories {
        diagnostics.hzb_views += 1;
        diagnostics.hzb_valid_histories += u32::from(history.previous_valid);
        // Keep both texture identities and their epoch/mip contract live for
        // the upcoming early/late bind groups.
        let _ = (&history.previous, &history.current, history.mip_count, history.history_epoch);
    }
}

#[derive(Default)]
pub(crate) struct HzbHistoryCache {
    views: HashMap<RetainedViewEntity, CachedHzb>,
}

struct CachedHzb {
    texture: TextureView,
    mip_count: u32,
    history_epoch: u64,
}

pub(crate) fn prepare_hzb_history(
    mut commands: Commands,
    state: Res<UnifiedVisibilityState>,
    pyramids: Query<(Entity, &bevy_render::view::ExtractedView, &ViewDepthPyramid)>,
    mut cache: Local<HzbHistoryCache>,
) {
    let mut retained = HashSet::<RetainedViewEntity>::new();
    for (entity, view, pyramid) in &pyramids {
        let retained_view = view.retained_view_entity;
        retained.insert(retained_view);
        let current_epoch = state
            .handle_for_retained(retained_view)
            .and_then(|handle| state.view_record(handle))
            .map_or(0, |record| record.history_epoch);
        let cached = cache.views.remove(&retained_view);
        let flags = state
            .handle_for_retained(retained_view)
            .and_then(|handle| state.view_record(handle))
            .map_or(prism_render_visibility::ViewFlags::default(), |record| record.flags);
        let previous_valid = cached.as_ref().is_some_and(|previous| {
            history_is_valid(
                previous.history_epoch,
                current_epoch,
                previous.mip_count,
                pyramid.mip_count,
                flags,
            )
        });
        let previous = cached
            .map(|previous| previous.texture)
            .unwrap_or_else(|| pyramid.all_mips.clone());
        commands.entity(entity).insert(PrismViewHzbHistory {
            previous,
            current: pyramid.all_mips.clone(),
            mip_count: pyramid.mip_count,
            history_epoch: current_epoch,
            previous_valid,
        });
        cache.views.insert(
            retained_view,
            CachedHzb {
                texture: pyramid.all_mips.clone(),
                mip_count: pyramid.mip_count,
                history_epoch: current_epoch,
            },
        );
    }
    cache.views.retain(|view, _| retained.contains(view));
}

fn history_is_valid(
    previous_epoch: u64,
    current_epoch: u64,
    previous_mips: u32,
    current_mips: u32,
    flags: prism_render_visibility::ViewFlags,
) -> bool {
    previous_epoch == current_epoch
        && previous_mips == current_mips
        && flags.contains(prism_render_visibility::ViewFlags::REVERSE_Z)
        && !flags.contains(prism_render_visibility::ViewFlags::CAMERA_CUT)
}

#[cfg(test)]
mod tests {
    use prism_render_visibility::ViewFlags;

    use super::history_is_valid;

    #[test]
    fn hzb_history_invalidates_on_epoch_size_and_camera_cut() {
        assert!(history_is_valid(3, 3, 8, 8, ViewFlags::REVERSE_Z));
        assert!(!history_is_valid(2, 3, 8, 8, ViewFlags::REVERSE_Z));
        assert!(!history_is_valid(3, 3, 7, 8, ViewFlags::REVERSE_Z));
        assert!(!history_is_valid(
            3,
            3,
            8,
            8,
            ViewFlags::REVERSE_Z | ViewFlags::CAMERA_CUT
        ));
    }
}
