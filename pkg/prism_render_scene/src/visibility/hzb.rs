use bevy_core_pipeline::mip_generation::experimental::depth::ViewDepthPyramid;
use bevy_ecs::prelude::*;
use bevy_platform::collections::{HashMap, HashSet};
use bevy_render::view::RetainedViewEntity;

use super::runtime::UnifiedVisibilityState;

/// Prism's temporal validity metadata for Bevy's persistent depth-pyramid
/// texture. Bevy updates that same texture twice in the Core3d schedule: its
/// contents are previous-frame HZB before early downsample and current-frame
/// HZB afterward. Prism therefore tracks epoch/mip validity, not a cloned
/// `TextureView` (which would alias the same texture rather than snapshot it).
#[derive(Component)]
pub(crate) struct PrismViewHzbHistory {
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
        let _ = (history.mip_count, history.history_epoch);
    }
}

#[derive(Default)]
pub(crate) struct HzbHistoryCache {
    views: HashMap<RetainedViewEntity, CachedHzb>,
}

struct CachedHzb {
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
        commands.entity(entity).insert(PrismViewHzbHistory {
            mip_count: pyramid.mip_count,
            history_epoch: current_epoch,
            previous_valid,
        });
        cache.views.insert(
            retained_view,
            CachedHzb {
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
