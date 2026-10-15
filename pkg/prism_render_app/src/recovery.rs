//! Device replacement and render-world resource replay before GPU work resumes.

use alloc::string::{String, ToString};
use prism_ecs::prelude::Resource;
use prism_ecs::world::World;
use prism_render_driver::{FrameSync, Instance};
use prism_render_runtime::RecoveryReport;
use std::panic::{catch_unwind, AssertUnwindSafe};

use crate::device_loss::DeviceLossSignal;
use crate::material::MaterialParameterBuffers;
use crate::mesh_pipeline::MeshPipelineSpecializer;
use crate::mesh_pipelines::MeshPipelines;
use crate::queue::RenderQueues;
use crate::render_asset::{
    invalidate_registered_render_assets, recovered_assets_pending, PrepareRecoveredAssets,
};
use crate::runtime_resource::RenderRuntimeResource;
use crate::scene::VisibilityWorkList;
use crate::startup::{ensure_render_startup, RenderStartupState};
use crate::target::RenderTarget;
use crate::texture_cache::TextureCache;

#[derive(Default, Resource)]
struct RecoveryState {
    active: bool,
    attempts: u64,
    replacement: Option<RecoveryReport>,
    core_rebuilt: bool,
}

pub(crate) fn is_recovering(world: &World) -> bool {
    world
        .get_resource::<RecoveryState>()
        .is_some_and(|state| state.active)
}

/// Returns true while a loss/recovery frame must skip the normal Render schedule.
pub(crate) fn recover_before_render<I>(world: &mut World) -> bool
where
    I: Instance + Send + Sync + 'static,
    I::Adapter: Send + Sync + 'static,
    I::Device: Send + Sync + 'static,
    I::Queue: Send + Sync + FrameSync + 'static,
    I::Surface: Send + Sync + 'static,
{
    world.init_resource::<RecoveryState>();
    let reason = world
        .resource::<RenderRuntimeResource<I>>()
        .runtime()
        .device_lost();
    if !world.resource::<RecoveryState>().active {
        let Some(reason) = reason else { return false };
        world.resource_mut::<RecoveryState>().active = true;
        world.resource::<DeviceLossSignal>().report(reason);
        world.resource_mut::<RenderQueues>().clear();
        world.resource_mut::<VisibilityWorkList>().clear();
        // Give the main App a frame to observe DeviceLost before any restore.
        return true;
    }

    let mut state = world
        .remove_resource::<RecoveryState>()
        .expect("recovery state");
    state.attempts = state.attempts.saturating_add(1);
    // If the replacement itself was lost during startup, replace it again.
    if state.replacement.is_some() && reason.is_some() {
        state.replacement = None;
        state.core_rebuilt = false;
    }
    let result = catch_unwind(AssertUnwindSafe(|| -> Result<bool, String> {
        let core_was_rebuilt = state.core_rebuilt;
        world
            .resource::<MeshPipelines>()
            .check_recovery()
            .map_err(|e| e.to_string())?;
        if state.replacement.is_none() || !state.core_rebuilt {
            let mut runtime = world
                .remove_resource::<RenderRuntimeResource<I>>()
                .expect("render runtime");
            // Keep the runtime outside the unwind boundary so it is always
            // returned to the world, including a backend/replay recipe panic.
            let rebuilt = catch_unwind(AssertUnwindSafe(|| -> Result<(), String> {
                if state.replacement.is_none() {
                    state.replacement = runtime
                        .runtime_mut()
                        .recover_if_lost()
                        .map_err(|e| e.to_string())?;
                    if state.replacement.is_none() {
                        return Err("recovery has no replacement device".into());
                    }
                    world.resource_mut::<MeshPipelines>().begin_recovery();
                }
                let runtime = runtime.runtime();
                let extent = world.resource::<RenderTarget>().viewport().extent();
                world.insert_resource(RenderTarget::new(runtime.device(), extent));
                world
                    .resource_mut::<MeshPipelines>()
                    .recover(runtime.device())
                    .map_err(|e| e.to_string())?;
                world.insert_resource(MeshPipelineSpecializer::with_worker(
                    runtime.device_handle(),
                ));
                world
                    .resource_mut::<MaterialParameterBuffers>()
                    .recover(runtime.device(), runtime.queue());
                world.insert_resource(TextureCache::new());
                invalidate_registered_render_assets(world);
                world.resource_mut::<RenderQueues>().clear();
                world.resource_mut::<VisibilityWorkList>().clear();
                world.init_resource::<RenderStartupState>();
                world.resource_mut::<RenderStartupState>().invalidate();
                state.core_rebuilt = true;
                Ok(())
            }));
            world.insert_resource(runtime);
            match rebuilt {
                Ok(result) => result?,
                Err(payload) => return Err(panic_message(payload)),
            }
        }
        ensure_render_startup(world);
        // Startup can execute provider-owned systems that submit work to the
        // replacement device. Observe a loss here before an asset-recovery
        // deferral can turn this frame into an apparently clean retry; the
        // caller must retain the precise replacement-loss diagnostic.
        if world
            .resource::<RenderRuntimeResource<I>>()
            .runtime()
            .device_lost()
            .is_some()
        {
            return Err("replacement device was lost during startup".into());
        }
        if !core_was_rebuilt && recovered_assets_pending(world) {
            // The next one-way extract must first copy unchanged source assets.
            return Ok(false);
        }
        world.run_schedule(PrepareRecoveredAssets);
        crate::transforms::prepare_transform_buffer::<I>(world)?;
        if recovered_assets_pending(world) {
            return Err(
                "render asset recovery is awaiting source data or successful preparation".into(),
            );
        }
        if world
            .resource::<RenderRuntimeResource<I>>()
            .runtime()
            .device_lost()
            .is_some()
        {
            return Err("replacement device was lost during resource recovery".into());
        }
        Ok(true)
    }));
    match result {
        Ok(Ok(true)) => {
            world.resource::<DeviceLossSignal>().restored(
                state.attempts,
                state.replacement.expect("successful recovery has a report"),
            );
            state = RecoveryState::default();
        }
        Ok(Ok(false)) => {}
        Ok(Err(error)) => world
            .resource::<DeviceLossSignal>()
            .failed(state.attempts, error),
        Err(payload) => world
            .resource::<DeviceLossSignal>()
            .failed(state.attempts, panic_message(payload)),
    }
    world.insert_resource(state);
    true
}

fn panic_message(payload: Box<dyn core::any::Any + Send>) -> String {
    let detail = payload
        .downcast_ref::<String>()
        .map(String::as_str)
        .or_else(|| payload.downcast_ref::<&str>().copied())
        .unwrap_or("non-string panic payload");
    format!("device recovery panicked: {detail}")
}

#[cfg(test)]
mod hardware_tests;
#[cfg(test)]
mod tests;
