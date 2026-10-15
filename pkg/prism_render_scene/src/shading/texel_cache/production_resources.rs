//! Render-world resource and lifecycle systems for the production shade pass.
//!
//! Preparation is separated from command recording.  Preparation owns CPU
//! validation and queue uploads; the Core3d node only records the already
//! prepared compute pass.  A recorded batch remains in-flight until the
//! shared queue completion tracker proves its submission complete.  Completion
//! is observable here, but this module intentionally does not publish a page
//! table: publication remains the shade-store producer's responsibility.

use std::collections::{BTreeMap, BTreeSet};

use bevy_ecs::prelude::*;
use bevy_render::{
    render_resource::{Buffer, CommandEncoder, TextureView},
    renderer::{RenderContext, RenderDevice, RenderQueue},
    Extract,
};
use prism_render_architecture::gpu_scene::GpuCompletionValue;

use crate::{
    completion::GpuCompletionTracker, geometry::RenderShadingGeometryRegistry,
    lighting::ExtractedLights,
};

use super::{
    build_shade_dispatch_batch, build_shade_dispatch_batch_mixed_epochs, FaceChart,
    PreparedShadeDispatch, ShadeAtlasLayout, ShadeDemandRegistry, ShadeDispatch,
    ShadeDispatchBuildError, ShadeDispatchConfig, ShadeDispatchInputs, ShadePageHandle,
    ShadePageKey, ShadePageTableRecord, SHADE_PAGE_SIZE,
};

/// Default fixed allocation limits for an explicitly enabled writer.
pub const DEFAULT_SHADE_DISPATCH_CONFIG: ShadeDispatchConfig = ShadeDispatchConfig {
    atlas_extent: [256, 256],
    max_jobs: 256,
    max_pages: 256,
    max_vertices: 16_384,
};

/// Render-world control for production texel-space shade dispatch.
#[derive(Resource, Clone, Copy, Debug, PartialEq, Eq)]
pub struct ShadeDispatchProductionSettings {
    /// Whether the production resource may allocate and prepare work.
    pub enabled: bool,
    /// Fixed GPU capacities for the writer.
    pub config: ShadeDispatchConfig,
    /// Explicit directional-light policy used by the automatic producer.
    pub directional_light: u32,
}

impl Default for ShadeDispatchProductionSettings {
    fn default() -> Self {
        Self {
            enabled: true,
            config: DEFAULT_SHADE_DISPATCH_CONFIG,
            directional_light: 0,
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

#[derive(Clone, Debug)]
struct AutoShadePublication {
    records: BTreeMap<ShadePageKey, ShadePageTableRecord>,
    charts: BTreeMap<ShadePageKey, FaceChart>,
    jobs: BTreeMap<ShadePageKey, super::ShadeDispatchJobInput>,
}

/// Owns the production writer and its explicit prepare → record → complete
/// boundary.
#[derive(Resource)]
pub struct ShadeDispatchProductionGpu {
    dispatch: Option<ShadeDispatch>,
    prepared: Option<PreparedShadeDispatch>,
    in_flight: Option<InFlightShadeDispatch>,
    status: ShadeDispatchProductionStatus,
    active_page_count: u32,
    active_chart_count: u32,
    next_slot: u32,
    next_generation: u64,
    active_records: BTreeMap<ShadePageKey, ShadePageTableRecord>,
    active_charts: BTreeMap<ShadePageKey, FaceChart>,
    active_jobs: BTreeMap<ShadePageKey, super::ShadeDispatchJobInput>,
    pending_publication: Option<AutoShadePublication>,
}

impl Default for ShadeDispatchProductionGpu {
    fn default() -> Self {
        Self {
            dispatch: None,
            prepared: None,
            in_flight: None,
            status: ShadeDispatchProductionStatus::Disabled,
            active_page_count: 0,
            active_chart_count: 0,
            next_slot: 0,
            next_generation: 1,
            active_records: BTreeMap::new(),
            active_charts: BTreeMap::new(),
            active_jobs: BTreeMap::new(),
            pending_publication: None,
        }
    }
}

impl ShadeDispatchProductionGpu {
    /// Last lifecycle state observed by the render-world integration.
    #[must_use]
    pub const fn status(&self) -> ShadeDispatchProductionStatus {
        self.status
    }

    /// GPU resources for the last validated page-table snapshot. The page
    /// table is uploaded before the writer pass; the render graph must order
    /// the writer before any consumer that samples the atlas.
    pub(crate) fn binding(&self) -> Option<ShadeDispatchBinding> {
        let dispatch = self.dispatch.as_ref()?;
        (self.active_page_count != 0).then(|| ShadeDispatchBinding {
            page_table: dispatch.page_table_buffer().clone(),
            page_lookup: dispatch.page_lookup_buffer().clone(),
            resolve_pages: dispatch.resolve_page_buffer().clone(),
            chart_infos: dispatch.chart_info_buffer().clone(),
            atlas: dispatch.atlas_view().clone(),
            page_count: self.active_page_count,
            chart_count: self.active_chart_count,
            atlas_extent: dispatch.config().atlas_extent,
        })
    }

    fn deactivate_snapshot(&mut self) {
        self.active_page_count = 0;
        self.active_chart_count = 0;
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
            if let Some(publication) = self.pending_publication.take() {
                self.active_records = publication.records;
                self.active_charts = publication.charts;
                self.active_jobs = publication.jobs;
            }
            self.status = ShadeDispatchProductionStatus::Completed {
                completion: in_flight.completion.0,
                job_count: in_flight.job_count,
            };
        } else {
            self.in_flight = Some(in_flight);
        }
    }

    fn cancel_prepared(&mut self) {
        self.pending_publication = None;
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
        auto_publication: Option<AutoShadePublication>,
        mixed_epochs: bool,
    ) {
        if self.in_flight.is_some() || self.prepared.is_some() {
            return;
        }
        let Some(dispatch) = self.dispatch.as_mut() else {
            self.deactivate_snapshot();
            self.status = ShadeDispatchProductionStatus::Error;
            return;
        };
        let batch = match if mixed_epochs {
            build_shade_dispatch_batch_mixed_epochs(inputs, geometry, lights)
        } else {
            build_shade_dispatch_batch(inputs, geometry, lights)
        } {
            Ok(batch) => batch,
            Err(ShadeDispatchBuildError::NoInput) => {
                self.deactivate_snapshot();
                self.status = ShadeDispatchProductionStatus::NoInput;
                return;
            }
            Err(_) => {
                self.deactivate_snapshot();
                self.pending_publication = None;
                self.status = ShadeDispatchProductionStatus::Error;
                return;
            }
        };
        match dispatch.prepare(
            queue,
            &batch.jobs,
            &batch.page_table,
            &batch.resolve_pages,
            &batch.chart_infos,
            &batch.vertices,
        ) {
            Ok(prepared) => {
                self.active_page_count = batch.page_table.len() as u32;
                self.active_chart_count = batch.chart_infos.len() as u32;
                self.pending_publication = auto_publication;
                self.status = ShadeDispatchProductionStatus::Prepared {
                    job_count: prepared.job_count(),
                };
                self.prepared = Some(prepared);
            }
            Err(_) => {
                self.deactivate_snapshot();
                self.pending_publication = None;
                self.status = ShadeDispatchProductionStatus::Error;
            }
        }
    }

    fn record(&mut self, encoder: &mut CommandEncoder, tracker: &GpuCompletionTracker) {
        let Some(prepared) = self.prepared else {
            return;
        };
        let Some(dispatch) = self.dispatch.as_mut() else {
            self.deactivate_snapshot();
            self.pending_publication = None;
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
            Err(_) => {
                self.deactivate_snapshot();
                self.pending_publication = None;
                self.status = ShadeDispatchProductionStatus::Error;
            }
        }
    }

    fn build_auto_inputs(
        &mut self,
        demands: &ShadeDemandRegistry,
        config: ShadeDispatchConfig,
        directional_light: u32,
    ) -> Result<Option<(ShadeDispatchInputs, AutoShadePublication, bool)>, ()> {
        let batches = demands.epoch_batches().map_err(|_| ())?;
        let mut requested = BTreeMap::<ShadePageKey, super::ShadeDemandRecord>::new();
        for batch in batches {
            for record in batch.records {
                requested.insert(record.request.key, record);
            }
        }
        if requested.is_empty() {
            return Ok(None);
        }

        let layout =
            ShadeAtlasLayout::new(config.atlas_extent, [SHADE_PAGE_SIZE; 2]).map_err(|_| ())?;
        let capacity = config.max_pages.min(layout.capacity()) as usize;
        let mut pending_records = self.active_records.clone();
        let mut pending_charts = self.active_charts.clone();
        let mut pending_jobs = self.active_jobs.clone();
        let used_slots = pending_records
            .values()
            .map(|record| record.handle.slot)
            .collect::<BTreeSet<_>>();

        let mut candidates = requested.into_values().collect::<Vec<_>>();
        candidates.sort_unstable_by(|a, b| {
            b.request
                .priority
                .cmp(&a.request.priority)
                .then_with(|| a.request.key.cmp(&b.request.key))
        });

        let mut next_slot = self.next_slot;
        for demand in candidates {
            if pending_jobs.len() >= config.max_jobs as usize {
                break;
            }
            if pending_records.len() >= capacity
                && !pending_records.contains_key(&demand.request.key)
            {
                continue;
            }
            if let Some(previous) = pending_charts.get(&demand.request.key)
                && *previous != demand.chart
            {
                return Err(());
            }

            let slot = if let Some(previous) = pending_records.get(&demand.request.key) {
                previous.handle.slot
            } else {
                while next_slot < layout.capacity() && used_slots.contains(&next_slot) {
                    next_slot = next_slot.saturating_add(1);
                }
                if next_slot >= layout.capacity() {
                    continue;
                }
                next_slot
            };
            let generation = self.next_generation;
            self.next_generation = self.next_generation.checked_add(1).ok_or(())?;
            let handle = ShadePageHandle {
                slot,
                generation,
                epoch: demand.request.key.material_epoch,
            };
            let [atlas_x, atlas_y] = layout.origin(slot).ok_or(())?;
            pending_records.insert(
                demand.request.key,
                ShadePageTableRecord {
                    key: demand.request.key,
                    handle,
                    atlas_x,
                    atlas_y,
                },
            );
            pending_charts.insert(demand.request.key, demand.chart);
            pending_jobs.insert(
                demand.request.key,
                super::ShadeDispatchJobInput {
                    geometry_index: demand.geometry_index,
                    chart: demand.chart,
                    page_key: demand.request.key,
                    directional_light,
                },
            );
            if !used_slots.contains(&slot) {
                next_slot = next_slot.saturating_add(1);
            }
        }
        if pending_jobs.is_empty() {
            return Ok(None);
        }

        let jobs = pending_jobs.values().copied().collect::<Vec<_>>();

        let pages = pending_records.values().copied().collect::<Vec<_>>();
        let epoch = pages
            .first()
            .map(|record| record.key.material_epoch)
            .unwrap_or(0);
        let mixed_epochs = pages
            .iter()
            .any(|record| record.key.material_epoch != epoch);
        self.next_slot = next_slot;
        Ok(Some((
            ShadeDispatchInputs {
                epoch,
                page_size: SHADE_PAGE_SIZE,
                pages,
                jobs,
            },
            AutoShadePublication {
                records: pending_records,
                charts: pending_charts,
                jobs: pending_jobs,
            },
            mixed_epochs,
        )))
    }
}

/// Bindable page-table/atlas resources owned by the production writer.
#[derive(Clone)]
pub(crate) struct ShadeDispatchBinding {
    pub(crate) page_table: Buffer,
    pub(crate) page_lookup: Buffer,
    pub(crate) resolve_pages: Buffer,
    pub(crate) chart_infos: Buffer,
    pub(crate) atlas: TextureView,
    pub(crate) page_count: u32,
    pub(crate) chart_count: u32,
    pub(crate) atlas_extent: [u32; 2],
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
    demands: Res<ShadeDemandRegistry>,
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
            production.deactivate_snapshot();
            production.status = ShadeDispatchProductionStatus::Disabled;
        }
        return;
    }
    production.ensure_dispatch(&device, settings.config);
    let (effective_inputs, auto_publication, mixed_epochs) = if inputs.is_empty() {
        match production.build_auto_inputs(&demands, settings.config, settings.directional_light) {
            Ok(Some((generated, publication, mixed))) => (generated, Some(publication), mixed),
            Ok(None) => {
                if production.in_flight.is_none() {
                    production.status = ShadeDispatchProductionStatus::NoInput;
                }
                return;
            }
            Err(()) => {
                production.deactivate_snapshot();
                production.status = ShadeDispatchProductionStatus::Error;
                return;
            }
        }
    } else {
        (inputs.clone(), None, false)
    };
    production.prepare(
        &queue,
        &effective_inputs,
        &geometry,
        &lights,
        auto_publication,
        mixed_epochs,
    );
}

/// Core3d node that records the already prepared writer pass.
pub(crate) fn shade_dispatch_pass(
    mut ctx: RenderContext,
    tracker: Res<GpuCompletionTracker>,
    mut production: ResMut<ShadeDispatchProductionGpu>,
) {
    production.record(ctx.command_encoder(), &tracker);
}
