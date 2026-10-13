//! Render-world resource and lifecycle systems for the production shade pass.
//!
//! Preparation is separated from command recording.  Preparation owns CPU
//! validation and queue uploads; the Core3d node only records the already
//! prepared compute pass.  A recorded batch remains in-flight until the
//! shared queue completion tracker proves its submission complete.  Completion
//! is observable here, but this module intentionally does not publish a page
//! table: publication remains the shade-store producer's responsibility.

use bevy_ecs::prelude::*;
use bevy_render::{
    render_resource::CommandEncoder,
    renderer::{RenderContext, RenderDevice, RenderQueue},
    Extract,
};
use prism_render_architecture::gpu_scene::GpuCompletionValue;

use crate::{
    completion::GpuCompletionTracker, geometry::RenderShadingGeometryRegistry,
    lighting::ExtractedLights,
};

use super::{
    build_shade_dispatch_batch, PreparedShadeDispatch, ShadeDispatch, ShadeDispatchBuildError,
    ShadeDispatchConfig, ShadeDispatchInputs,
};

/// Default fixed allocation limits for an explicitly enabled writer.
pub const DEFAULT_SHADE_DISPATCH_CONFIG: ShadeDispatchConfig = ShadeDispatchConfig {
    atlas_extent: [256, 256],
    max_jobs: 256,
    max_pages: 256,
    max_vertices: 16_384,
};

/// Render-world opt-in for production texel-space shade dispatch.
///
/// It is disabled by default because no object/material shade-page producer is
/// installed yet.  Enabling it only allocates the writer; a compute pass is
/// still impossible until [`ShadeDispatchInputs`] contains real page/chart
/// requests and the render world has matching geometry and lights.
#[derive(Resource, Clone, Copy, Debug, PartialEq, Eq)]
pub struct ShadeDispatchProductionSettings {
    /// Whether the production resource may allocate and prepare work.
    pub enabled: bool,
    /// Fixed GPU capacities for the writer.
    pub config: ShadeDispatchConfig,
}

impl Default for ShadeDispatchProductionSettings {
    fn default() -> Self {
        Self {
            enabled: false,
            config: DEFAULT_SHADE_DISPATCH_CONFIG,
        }
    }
}

/// Public lifecycle state for diagnostics and integration tests.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ShadeDispatchProductionStatus {
    /// Settings or input policy intentionally prevented any GPU work.
    Disabled,
    /// The opt-in is available but no complete input frame was supplied.
    NoInput,
    /// A validated batch has been uploaded and awaits the Core3d node.
    Prepared { job_count: u32 },
    /// The compute pass was recorded, but its writes are not completion-safe.
    WaitingForCompletion { completion: u64, job_count: u32 },
    /// Queue completion was observed. This is not a page-publication signal.
    Completed { completion: u64, job_count: u32 },
    /// Input/resource validation failed; no pass was recorded for this batch.
    Error,
}

struct InFlightShadeDispatch {
    completion: GpuCompletionValue,
    job_count: u32,
}

/// Owns the production writer and its explicit prepare → record → complete
/// boundary.
#[derive(Resource)]
pub struct ShadeDispatchProductionGpu {
    dispatch: Option<ShadeDispatch>,
    prepared: Option<PreparedShadeDispatch>,
    in_flight: Option<InFlightShadeDispatch>,
    status: ShadeDispatchProductionStatus,
}

impl Default for ShadeDispatchProductionGpu {
    fn default() -> Self {
        Self {
            dispatch: None,
            prepared: None,
            in_flight: None,
            status: ShadeDispatchProductionStatus::Disabled,
        }
    }
}

impl ShadeDispatchProductionGpu {
    /// Last lifecycle state observed by the render-world integration.
    #[must_use]
    pub const fn status(&self) -> ShadeDispatchProductionStatus {
        self.status
    }

    fn ensure_dispatch(&mut self, device: &RenderDevice, config: ShadeDispatchConfig) {
        if self.dispatch.is_some() {
            return;
        }
        match ShadeDispatch::new(device, config) {
            Ok(dispatch) => {
                self.dispatch = Some(dispatch);
                self.status = ShadeDispatchProductionStatus::NoInput;
            }
            Err(_) => self.status = ShadeDispatchProductionStatus::Error,
        }
    }

    fn poll_completion(&mut self, completed: GpuCompletionValue) {
        let Some(in_flight) = self.in_flight.take() else {
            return;
        };
        if completed.0 >= in_flight.completion.0 {
            self.status = ShadeDispatchProductionStatus::Completed {
                completion: in_flight.completion.0,
                job_count: in_flight.job_count,
            };
        } else {
            self.in_flight = Some(in_flight);
        }
    }

    fn cancel_prepared(&mut self) {
        if let (Some(dispatch), Some(prepared)) = (self.dispatch.as_mut(), self.prepared.take()) {
            let _ = dispatch.cancel(prepared);
        }
    }

    fn prepare(
        &mut self,
        queue: &RenderQueue,
        inputs: &ShadeDispatchInputs,
        geometry: &RenderShadingGeometryRegistry,
        lights: &ExtractedLights,
    ) {
        if self.in_flight.is_some() || self.prepared.is_some() {
            return;
        }
        let Some(dispatch) = self.dispatch.as_mut() else {
            self.status = ShadeDispatchProductionStatus::Error;
            return;
        };
        let batch = match build_shade_dispatch_batch(inputs, geometry, lights) {
            Ok(batch) => batch,
            Err(ShadeDispatchBuildError::NoInput) => {
                self.status = ShadeDispatchProductionStatus::NoInput;
                return;
            }
            Err(_) => {
                self.status = ShadeDispatchProductionStatus::Error;
                return;
            }
        };
        match dispatch.prepare(queue, &batch.jobs, &batch.page_table, &batch.vertices) {
            Ok(prepared) => {
                self.status = ShadeDispatchProductionStatus::Prepared {
                    job_count: prepared.job_count(),
                };
                self.prepared = Some(prepared);
            }
            Err(_) => self.status = ShadeDispatchProductionStatus::Error,
        }
    }

    fn record(&mut self, encoder: &mut CommandEncoder, tracker: &GpuCompletionTracker) {
        let Some(prepared) = self.prepared else {
            return;
        };
        let Some(dispatch) = self.dispatch.as_mut() else {
            self.status = ShadeDispatchProductionStatus::Error;
            return;
        };
        match dispatch.record(encoder, prepared) {
            Ok(stats) => {
                self.prepared = None;
                let completion = tracker.next_submission_value();
                self.status = ShadeDispatchProductionStatus::WaitingForCompletion {
                    completion: completion.0,
                    job_count: stats.job_count,
                };
                self.in_flight = Some(InFlightShadeDispatch {
                    completion,
                    job_count: stats.job_count,
                });
            }
            Err(_) => self.status = ShadeDispatchProductionStatus::Error,
        }
    }
}

/// Copies the main-world producer's explicit frame input into the render
/// world. An empty copy is meaningful: it disables the pass for that frame.
pub(crate) fn extract_shade_dispatch_inputs(
    source: Extract<Res<ShadeDispatchInputs>>,
    mut destination: ResMut<ShadeDispatchInputs>,
) {
    *destination = (*source).clone();
}

/// Allocates the writer when explicitly enabled, observes completion, and
/// prepares only against real render-world geometry/light inputs.
pub(crate) fn prepare_shade_dispatch(
    settings: Res<ShadeDispatchProductionSettings>,
    inputs: Res<ShadeDispatchInputs>,
    geometry: Res<RenderShadingGeometryRegistry>,
    lights: Res<ExtractedLights>,
    device: Res<RenderDevice>,
    queue: Res<RenderQueue>,
    tracker: Res<GpuCompletionTracker>,
    mut production: ResMut<ShadeDispatchProductionGpu>,
) {
    production.poll_completion(tracker.completed_value());
    if !settings.enabled {
        if production.in_flight.is_none() {
            production.cancel_prepared();
            production.status = ShadeDispatchProductionStatus::Disabled;
        }
        return;
    }
    production.ensure_dispatch(&device, settings.config);
    production.prepare(&queue, &inputs, &geometry, &lights);
}

/// Core3d node that records the already prepared writer pass.
pub(crate) fn shade_dispatch_pass(
    mut ctx: RenderContext,
    tracker: Res<GpuCompletionTracker>,
    mut production: ResMut<ShadeDispatchProductionGpu>,
) {
    production.record(ctx.command_encoder(), &tracker);
}
