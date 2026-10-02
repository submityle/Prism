use bevy_app::SubApp;
use bevy_core_pipeline::mip_generation::experimental::depth::ViewDepthPyramid;
use bevy_core_pipeline::{
    mip_generation::experimental::depth::early_downsample_depth,
    prepass::node::{early_prepass, late_prepass},
    schedule::Core3d,
};
use bevy_ecs::prelude::*;
use bevy_platform::collections::{HashMap, HashSet};
use bevy_render::view::RetainedViewEntity;
use bevy_render::{
    render_resource::{ComputePassDescriptor, PipelineCache},
    renderer::RenderContext,
};

use super::runtime::UnifiedVisibilityState;

#[derive(SystemSet, Debug, Clone, PartialEq, Eq, Hash)]
enum PrismHzbSystems {
    PreviousClassify,
    EarlyCompact,
    CurrentClassify,
    LateCompact,
}

/// Schedules Prism's previous-HZB test before the early prepass and the
/// current-HZB retest after the pyramid has been rebuilt from early depth.
pub(crate) fn install_hzb_schedule(app: &mut SubApp) {
    app.configure_sets(
        Core3d,
        (
            PrismHzbSystems::PreviousClassify,
            PrismHzbSystems::EarlyCompact,
        )
            .chain()
            .before(early_prepass),
    );
    app.add_systems(
        Core3d,
        (
            dispatch_previous_hzb.in_set(PrismHzbSystems::PreviousClassify),
            super::systems::dispatch_unified_visibility_for_view
                .in_set(PrismHzbSystems::EarlyCompact),
            dispatch_current_hzb
                .in_set(PrismHzbSystems::CurrentClassify)
                .after(early_downsample_depth)
                .before(late_prepass),
            super::hzb_late::dispatch_hzb_late_compact
                .in_set(PrismHzbSystems::LateCompact)
                .after(PrismHzbSystems::CurrentClassify)
                .before(late_prepass),
        ),
    );
}

/// Runs the previous-frame HZB classifier before early depth. Only valid
/// history is consumed, and the entire path remains graduation-gated.
fn dispatch_previous_hzb(
    current_view: bevy_render::renderer::ViewQuery<&bevy_render::view::ExtractedView>,
    enabled: Res<super::runtime::UnifiedVisibilityEnabled>,
    history: Option<bevy_render::renderer::ViewQuery<&PrismViewHzbHistory>>,
    bindings: Option<bevy_render::renderer::ViewQuery<&super::hzb_gpu::HzbVisibilityBindGroup>>,
    settings: Res<super::runtime::UnifiedVisibilitySettings>,
    pipeline: Res<super::hzb_gpu::HzbVisibilityPipeline>,
    pipeline_cache: Res<PipelineCache>,
    buffers: Res<super::hzb_gpu::HzbVisibilityBuffers>,
    mut ctx: RenderContext,
    mut diagnostics: ResMut<super::runtime::PrismVisibilityDiagnostics>,
) {
    let retained = current_view.into_inner().retained_view_entity;
    let Some((output_start, candidate_count)) = buffers.view_range(retained) else {
        return;
    };
    let active_count = buffers.active_count_in_range(output_start, candidate_count);
    let _conservative_policy = (
        settings.hzb_depth_bias.max(0.0),
        settings.hzb_fast_motion_threshold.max(0.0),
    );
    let Some(history) = history else {
        return;
    };
    let history = history.into_inner();
    if !super::runtime::hzb_runtime_gate(*enabled, &settings) || !history.previous_valid {
        return;
    }
    let Some(bind_group) = bindings.and_then(|bindings| bindings.into_inner().bind_group.as_ref())
    else {
        return;
    };
    diagnostics.hzb_previous_ready_views += 1;
    let Some(compute_pipeline) = pipeline_cache.get_compute_pipeline(pipeline.pipeline) else {
        return;
    };
    let immediates = hzb_immediates(
        candidate_count,
        history.mip_count,
        output_start,
        0,
        true,
        settings.hzb_depth_bias,
        settings.hzb_fast_motion_threshold,
    );
    let mut pass = ctx
        .command_encoder()
        .begin_compute_pass(&ComputePassDescriptor {
            label: Some("prism previous hzb visibility"),
            timestamp_writes: None,
        });
    pass.set_pipeline(compute_pipeline);
    pass.set_bind_group(0, bind_group, &[]);
    pass.set_immediates(0, bytemuck::bytes_of(&immediates));
    pass.dispatch_workgroups(candidate_count.div_ceil(64), 1, 1);
    diagnostics.hzb_previous_dispatches += 1;
    diagnostics.hzb_candidates += active_count;
}

/// Runs the current-frame HZB retest after early depth has rebuilt the pyramid.
/// Missing resources keep all deferred candidates visible.
fn dispatch_current_hzb(
    current_view: bevy_render::renderer::ViewQuery<&bevy_render::view::ExtractedView>,
    enabled: Res<super::runtime::UnifiedVisibilityEnabled>,
    history: Option<bevy_render::renderer::ViewQuery<&PrismViewHzbHistory>>,
    bindings: Option<bevy_render::renderer::ViewQuery<&super::hzb_gpu::HzbVisibilityBindGroup>>,
    settings: Res<super::runtime::UnifiedVisibilitySettings>,
    pipeline: Res<super::hzb_gpu::HzbVisibilityPipeline>,
    pipeline_cache: Res<PipelineCache>,
    buffers: Res<super::hzb_gpu::HzbVisibilityBuffers>,
    mut ctx: RenderContext,
    mut diagnostics: ResMut<super::runtime::PrismVisibilityDiagnostics>,
) {
    let retained = current_view.into_inner().retained_view_entity;
    let Some((output_start, candidate_count)) = buffers.view_range(retained) else {
        return;
    };
    let active_count = buffers.active_count_in_range(output_start, candidate_count);
    if !super::runtime::hzb_runtime_gate(*enabled, &settings) {
        return;
    }
    let (Some(history), Some(bind_group), Some(compute_pipeline)) = (
        history,
        bindings.and_then(|bindings| bindings.into_inner().bind_group.as_ref()),
        pipeline_cache.get_compute_pipeline(pipeline.pipeline),
    ) else {
        return;
    };
    diagnostics.hzb_current_ready_views += 1;
    let immediates = hzb_immediates(
        candidate_count,
        history.into_inner().mip_count,
        output_start,
        1,
        true,
        settings.hzb_depth_bias,
        settings.hzb_fast_motion_threshold,
    );
    let mut pass = ctx
        .command_encoder()
        .begin_compute_pass(&ComputePassDescriptor {
            label: Some("prism current hzb visibility"),
            timestamp_writes: None,
        });
    pass.set_pipeline(compute_pipeline);
    pass.set_bind_group(0, bind_group, &[]);
    pass.set_immediates(0, bytemuck::bytes_of(&immediates));
    pass.dispatch_workgroups(candidate_count.div_ceil(64), 1, 1);
    diagnostics.hzb_current_dispatches += 1;
    diagnostics.hzb_late_retests += active_count;
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct HzbDispatch {
    candidate_count: u32,
    mip_count: u32,
    output_start: u32,
    phase: u32,
    history_valid: u32,
    camera_cut: u32,
    depth_bias: f32,
    fast_motion_threshold: f32,
}

fn hzb_immediates(
    candidate_count: u32,
    mip_count: u32,
    output_start: u32,
    phase: u32,
    history_valid: bool,
    depth_bias: f32,
    fast_motion_threshold: f32,
) -> HzbDispatch {
    HzbDispatch {
        candidate_count,
        mip_count: mip_count.max(1),
        output_start,
        phase,
        history_valid: u32::from(history_valid),
        camera_cut: 0,
        depth_bias,
        fast_motion_threshold,
    }
}

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
            .map_or(prism_render_visibility::ViewFlags::default(), |record| {
                record.flags
            });
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
