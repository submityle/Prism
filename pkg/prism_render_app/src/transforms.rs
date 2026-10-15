//! Simulation publication → immutable extraction → mesh instance-stream consumption.
//!
//! Bind a simulation entity to its generational scene handle with
//! [`RenderTransformBinding`]. Opt-in resident affine columns bind physical ECS
//! rows directly; otherwise the plugin propagates locals and publishes the
//! complete double-buffered pose in main `Last`; local/hierarchy writers must
//! finish before that phase. Extraction only clones an immutable CPU snapshot.
//! GPU writes run after the device-loss gate, and recovery uploads the current
//! snapshot before reporting restoration. The spine applies the affine to its
//! clip-space positions; camera projection remains the pipeline provider's job.

use std::collections::BTreeMap;
use std::sync::Arc;

use prism_ecs::{component::Component, entity::Entity, resource::Resource, world::World};
use prism_render_architecture::gpu_scene::SceneHandle;
use prism_render_driver::{
    BufferDescriptor, BufferId, BufferUsages, FrameSync, Instance, RenderDevice, RenderQueue,
    VertexBufferLayout, VertexFormat, VertexStepMode,
};
use prism_transform::{
    double_buffer::MultiBuffer,
    gpu_upload::{GpuColumnBuffer, MatrixLayout},
    GlobalTransform,
};

use crate::ecs_columns::{has_resident_transform, ResidentTransformRows};
use crate::runtime_resource::RenderRuntimeResource;
use crate::{DrawFunctionId, DrawFunctions, DrawRequest, MeshPipelineProvider, MeshPipelines};

const STRIDE: usize = 48;

/// Associates an entity's world pose with one render scene instance.
/// Handles include the generation; reused scene slots never inherit old poses.
/// A declared GPU transform column supplies its affine directly; other entities
/// use the propagated `GlobalTransform` publication path.
#[derive(Clone, Copy, Debug)]
pub struct RenderTransformBinding(pub SceneHandle);
impl Component for RenderTransformBinding {}

#[derive(Default)]
struct Snapshot {
    version: u64,
    rows: BTreeMap<SceneHandle, usize>,
    bytes: Arc<[u8]>,
    error: Option<String>,
}

/// Main-world CPU publication. No device, queue or GPU handle crosses this seam.
pub struct PublishedTransforms {
    buffers: MultiBuffer<GlobalTransform>,
    packed: GpuColumnBuffer,
    snapshot: Arc<Snapshot>,
}
impl Resource for PublishedTransforms {}

impl Default for PublishedTransforms {
    fn default() -> Self {
        Self {
            buffers: MultiBuffer::double(0, GlobalTransform::IDENTITY),
            packed: GpuColumnBuffer::new(MatrixLayout::RowMajor3x4),
            snapshot: Arc::default(),
        }
    }
}

impl PublishedTransforms {
    /// Monotonic publication sequence, including failed/empty publications.
    pub fn version(&self) -> u64 {
        self.snapshot.version
    }

    /// Malformed hierarchy, duplicate scene bindings or missing global pose.
    pub fn last_error(&self) -> Option<&str> {
        self.snapshot.error.as_deref()
    }

    /// Packed row-major 3×4 poses from the most recent complete publication.
    pub fn bytes(&self) -> &[u8] {
        &self.snapshot.bytes
    }

    fn publish(
        &mut self,
        rows: BTreeMap<SceneHandle, usize>,
        globals: &[GlobalTransform],
        error: Option<String>,
    ) {
        let version = self.snapshot.version.saturating_add(1);
        if error.is_some() {
            // Preserve binding identities so a broken bound instance is rejected
            // instead of silently drawing its untransformed geometry.
            self.snapshot = Arc::new(Snapshot {
                version,
                rows,
                error,
                ..Default::default()
            });
            return;
        }
        let structural = self.buffers.len() != globals.len()
            || self.snapshot.rows != rows
            || self.snapshot.error.is_some();
        let dirty: Vec<u32> = if structural {
            Vec::new()
        } else {
            self.buffers
                .read()
                .iter()
                .zip(globals)
                .enumerate()
                .filter_map(|(i, (old, new))| (old != new).then_some(i as u32))
                .collect()
        };
        if self.buffers.len() != globals.len() {
            self.buffers = MultiBuffer::double(globals.len(), GlobalTransform::IDENTITY);
        }
        self.buffers.publish_from(globals);
        let ranges = if structural {
            self.packed.pack_all(self.buffers.read())
        } else {
            self.packed.pack_dirty(self.buffers.read(), &dirty)
        };
        let bytes = if structural || !ranges.is_empty() {
            Arc::from(self.packed.as_bytes())
        } else {
            self.snapshot.bytes.clone()
        };
        self.snapshot = Arc::new(Snapshot {
            version,
            rows,
            bytes,
            error: None,
        });
    }
}

/// Immutable render-world CPU snapshot. It survives subsequent main publishes.
#[derive(Default)]
pub struct ExtractedTransforms(Arc<Snapshot>);
impl Resource for ExtractedTransforms {}
impl ExtractedTransforms {
    pub fn version(&self) -> u64 {
        self.0.version
    }
    pub fn bytes(&self) -> &[u8] {
        &self.0.bytes
    }
    pub fn row(&self, scene: SceneHandle) -> Option<usize> {
        self.0.rows.get(&scene).copied()
    }
}

pub(crate) fn publish_transforms(world: &mut World) {
    let mut query = world.query::<(Entity, &RenderTransformBinding)>();
    let mut bindings: Vec<_> = query
        .iter(world)
        .map(|(entity, binding)| (binding.0, entity))
        .collect();
    bindings.sort_unstable();
    let mut rows = BTreeMap::new();
    let mut error = None;
    for pair in bindings.windows(2) {
        if pair[0].0 == pair[1].0 {
            let handle = pair[0].0;
            error = Some(format!("duplicate transform scene binding: {handle:?}"));
        }
    }
    bindings.retain(|&(_, entity)| !has_resident_transform(world, entity));
    for (i, &(handle, _)) in bindings.iter().enumerate() {
        rows.insert(handle, i);
    }
    if !bindings.is_empty() && error.is_none() {
        if let Err(reason) = prism_transform::ecs::propagate_transforms(world) {
            error = Some(format!("transform propagation failed: {reason:?}"));
        }
    }
    let mut globals = Vec::with_capacity(bindings.len());
    if error.is_none() {
        for &(handle, entity) in &bindings {
            match world.get::<GlobalTransform>(entity) {
                Some(global) => globals.push(*global),
                None => {
                    error = Some(format!("missing global transform for {handle:?}"));
                    break;
                }
            }
        }
    }
    world.init_resource::<PublishedTransforms>();
    world
        .resource_mut::<PublishedTransforms>()
        .publish(rows, &globals, error);
}

pub(crate) fn extract_transforms(main: &mut World, render: &mut World) {
    let snapshot = main
        .get_resource::<PublishedTransforms>()
        .map_or_else(Arc::default, |source| source.snapshot.clone());
    render.insert_resource(ExtractedTransforms(snapshot));
}

struct Allocation {
    device: Arc<dyn RenderDevice + Send + Sync>,
    id: BufferId,
    capacity: usize,
}
impl Drop for Allocation {
    fn drop(&mut self) {
        self.device.destroy_buffer(self.id);
    }
}

/// Work and rejected draws for the latest render preparation/queue cycle.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct TransformUploadStats {
    pub allocations: u64,
    pub upload_calls: u64,
    pub uploaded_bytes: u64,
    pub draws_rejected: u64,
}

/// Render-world device mirror, also used by the mesh queue and draw recorder.
#[derive(Default)]
pub struct RenderTransforms {
    source: Arc<Snapshot>,
    resident: ResidentTransformRows,
    // Failed preparation retains both the old submitted views and the new
    // binding identities, so newly bound scenes cannot fall back to bare meshes.
    pending_resident: ResidentTransformRows,
    allocation: Option<Arc<Allocation>>,
    uploaded: Vec<u8>,
    last_upload: Option<Arc<[u8]>>,
    draw: Option<DrawFunctionId>,
    stats: TransformUploadStats,
    error: Option<String>,
    prepared: bool,
}
impl Resource for RenderTransforms {}
impl RenderTransforms {
    pub fn stats(&self) -> TransformUploadStats {
        self.stats
    }
    pub fn last_error(&self) -> Option<&str> {
        self.error.as_deref()
    }
    pub fn buffer(&self) -> Option<BufferId> {
        self.allocation
            .as_ref()
            .filter(|_| self.prepared && !self.source.bytes.is_empty())
            .map(|a| a.id)
    }
    pub fn version(&self) -> u64 {
        self.source.version
    }

    fn needs_retirement(&self, device: &Arc<dyn RenderDevice + Send + Sync>, bytes: usize) -> bool {
        self.allocation
            .as_ref()
            .is_some_and(|a| Arc::ptr_eq(&a.device, device) && bytes > a.capacity)
    }

    fn reject_preparation(
        &mut self,
        source: Arc<Snapshot>,
        resident: ResidentTransformRows,
        error: String,
    ) {
        self.source = source;
        self.pending_resident = resident;
        self.stats = TransformUploadStats::default();
        self.error = Some(error);
        self.prepared = false;
    }

    fn sync(
        &mut self,
        source: Arc<Snapshot>,
        device: Arc<dyn RenderDevice + Send + Sync>,
        queue: &dyn RenderQueue,
    ) -> Result<(), String> {
        self.stats = TransformUploadStats::default();
        self.error = None;
        self.prepared = false;
        let unchanged = self
            .last_upload
            .as_ref()
            .is_some_and(|bytes| Arc::ptr_eq(bytes, &source.bytes));
        self.source = source;
        if let Some(error) = &self.source.error {
            return Err(error.clone());
        }
        let bytes = &self.source.bytes;
        if bytes.is_empty() {
            self.uploaded.clear();
            self.last_upload = Some(bytes.clone());
            self.prepared = true;
            return Ok(());
        }
        let replace = self
            .allocation
            .as_ref()
            .is_none_or(|a| !Arc::ptr_eq(&a.device, &device) || bytes.len() > a.capacity);
        if replace {
            self.allocation = Some(Arc::new(Allocation {
                id: device.create_buffer(&BufferDescriptor::new(
                    bytes.len() as u64,
                    BufferUsages::VERTEX | BufferUsages::COPY_DST,
                )),
                device,
                capacity: bytes.len(),
            }));
            self.uploaded.clear();
            self.stats.allocations = 1;
        }
        if !replace && unchanged {
            self.prepared = true;
            return Ok(());
        }
        let buffer = self.allocation.as_ref().expect("nonempty pose buffer").id;
        // Diff against the last consumed bytes, not the source frame's dirty
        // delta. Extraction may skip arbitrarily many publications/recovery frames.
        let mut start = None;
        for row in 0..=bytes.len() / STRIDE {
            let offset = row * STRIDE;
            let dirty = offset < bytes.len()
                && self.uploaded.get(offset..offset + STRIDE)
                    != Some(&bytes[offset..offset + STRIDE]);
            if dirty {
                start.get_or_insert(offset);
            } else if let Some(first) = start.take() {
                queue.write_buffer(buffer, first as u64, &bytes[first..offset]);
                self.stats.upload_calls += 1;
                self.stats.uploaded_bytes += (offset - first) as u64;
            }
        }
        self.uploaded.clear();
        self.uploaded.extend_from_slice(bytes);
        self.last_upload = Some(bytes.clone());
        self.prepared = true;
        Ok(())
    }

    fn install_draw(&mut self, functions: &mut DrawFunctions) {
        let allocation = self.allocation.clone();
        let resident = self.resident.clone();
        if allocation.is_none() && resident.0.is_empty() {
            // Clear a previous closure's retained views even after all bindings
            // disappear; its stable id may still be referenced by recorded work.
            if let Some(id) = self.draw {
                functions.replace(id, |_, _| {});
            }
            return;
        }
        let rows = self.source.rows.clone();
        let draw = move |item: &crate::PhaseItem, ctx: &mut dyn crate::MeshRecorder| {
            let (buffer, offset) = match resident.0.get(&item.scene) {
                Some(Ok(binding)) => {
                    let Some(buffer) = binding.column.buffer_id() else {
                        return;
                    };
                    (buffer, (binding.row * STRIDE) as u64)
                }
                Some(Err(_)) => return,
                None => {
                    let (Some(&row), Some(allocation)) = (rows.get(&item.scene), &allocation)
                    else {
                        return;
                    };
                    (allocation.id, (row * STRIDE) as u64)
                }
            };
            if item.indirect.is_some() || item.vertices.last() != Some(&buffer) {
                return;
            }
            ctx.set_pipeline(item.pipeline);
            for (slot, &buffer) in item.vertices.iter().enumerate() {
                let offset = if slot + 1 == item.vertices.len() {
                    offset
                } else {
                    0
                };
                ctx.set_vertex_buffer(slot as u32, buffer, offset);
            }
            if let Some((indices, format)) = item.indices {
                ctx.set_index_buffer(indices, format, 0);
                ctx.draw_indexed(item.first..item.first + item.count, item.base_vertex, 0..1);
            } else {
                ctx.draw(item.first..item.first + item.count, 0..1);
            }
        };
        match self.draw {
            Some(id) => functions.replace(id, draw),
            None => self.draw = Some(functions.register(draw)),
        }
    }

    pub(crate) fn decorate(
        &mut self,
        request: &mut DrawRequest,
        provider: &MeshPipelines,
    ) -> Result<bool, String> {
        if !self.source.rows.contains_key(&request.scene)
            && !self.resident.0.contains_key(&request.scene)
            && !self.pending_resident.0.contains_key(&request.scene)
        {
            return Ok(false);
        }
        let fail = if !self.prepared {
            Some(
                self.error
                    .clone()
                    .unwrap_or_else(|| "transform buffer has not been prepared".into()),
            )
        } else if self.source.error.is_some() {
            self.source.error.clone()
        } else if let Some(Err(error)) = self.resident.0.get(&request.scene) {
            Some(error.clone())
        } else if provider.transformed_descriptor(&request.variant).is_none() {
            Some("mesh provider does not support the affine transform stream".into())
        } else if crate::mesh_pipeline::full_lobe_fallback(request.variant)
            .is_some_and(|variant| provider.transformed_descriptor(&variant).is_none())
        {
            Some("mesh fallback does not support the affine transform stream".into())
        } else if request.draw != crate::draw::canonical::MESH {
            Some("transform stream requires the direct mesh draw contract".into())
        } else if request
            .layout
            .iter()
            .flat_map(|l| &l.attributes)
            .any(|a| a.shader_location >= 13)
        {
            Some("mesh attributes overlap transform locations 13..15".into())
        } else {
            None
        };
        if let Some(error) = fail {
            self.stats.draws_rejected += 1;
            self.error = Some(error.clone());
            return Err(error);
        }
        let buffer = match self.resident.0.get(&request.scene) {
            Some(Ok(binding)) => binding.column.buffer_id(),
            Some(Err(_)) => None,
            None => self.allocation.as_ref().map(|a| a.id),
        };
        let (Some(buffer), Some(draw)) = (buffer, self.draw) else {
            let error = "transform buffer has not been prepared".to_owned();
            self.error = Some(error.clone());
            self.stats.draws_rejected += 1;
            return Err(error);
        };
        request.layout.push(VertexBufferLayout::packed(
            VertexStepMode::Instance,
            13,
            &[VertexFormat::Float32x4; 3],
        ));
        request.vertices.push(buffer);
        request.draw = draw;
        Ok(true)
    }
}

pub(crate) fn prepare_transform_resources<I>(world: &mut World)
where
    I: Instance + Send + Sync + 'static,
    I::Adapter: Send + Sync + 'static,
    I::Device: Send + Sync + 'static,
    I::Queue: Send + Sync + FrameSync + 'static,
    I::Surface: Send + Sync + 'static,
{
    let _ = prepare_transform_buffer::<I>(world);
}

pub(crate) fn prepare_transform_buffer<I>(world: &mut World) -> Result<(), String>
where
    I: Instance + Send + Sync + 'static,
    I::Adapter: Send + Sync + 'static,
    I::Device: Send + Sync + 'static,
    I::Queue: Send + Sync + FrameSync + 'static,
    I::Surface: Send + Sync + 'static,
{
    world.init_resource::<RenderTransforms>();
    let source = world
        .get_resource::<ExtractedTransforms>()
        .map_or_else(Arc::default, |e| e.0.clone());
    let resident = world
        .get_resource::<ResidentTransformRows>()
        .cloned()
        .unwrap_or_default();
    let runtime = world.resource::<RenderRuntimeResource<I>>().runtime();
    let device: Arc<dyn RenderDevice + Send + Sync> = runtime.device_handle();
    // Retire an old healthy allocation only after its submitted frames finish.
    // A replacement device owns a different queue and needs no old-device wait.
    let error = if runtime.device_lost().is_some() {
        Some("transform upload paused while the device is lost".to_owned())
    } else if (world
        .resource::<RenderTransforms>()
        .needs_retirement(&device, source.bytes.len())
        || world
            .resource::<RenderTransforms>()
            .resident
            .releases_on_device(&resident, &device))
        && !runtime.wait_idle()
    {
        Some("transform buffer retirement could not wait for submitted frames".to_owned())
    } else {
        None
    };
    if let Some(error) = error {
        let state = world.resource_mut::<RenderTransforms>();
        // Keep prior views until a later successful retirement barrier. The old
        // draw closure can still own buffers read by submitted frames.
        state.reject_preparation(source, resident, error.clone());
        return Err(error);
    }
    let mut state = world
        .remove_resource::<RenderTransforms>()
        .expect("transform state");
    state.resident = resident;
    state.pending_resident = ResidentTransformRows::default();
    // A backend may panic during allocation/upload. Restore the resource before
    // the recovery gate handles the panic, so diagnostics and retries keep the
    // same CPU baseline instead of losing it with the unwinding stack.
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        state.sync(
            source,
            device,
            world
                .resource::<RenderRuntimeResource<I>>()
                .runtime()
                .queue(),
        )
    }));
    let result = match result {
        Ok(result) => result,
        Err(payload) => {
            state.prepared = false;
            state.error = Some("transform allocation or upload panicked".into());
            // A partially completed upload must be retried in full.
            state.last_upload = None;
            state.uploaded.clear();
            world.insert_resource(state);
            std::panic::resume_unwind(payload);
        }
    };
    if let Err(error) = &result {
        state.error = Some(error.clone());
    } else {
        state.install_draw(world.resource_mut::<DrawFunctions>());
    }
    world.insert_resource(state);
    result
}

#[cfg(test)]
mod tests;

#[cfg(test)]
mod resident_tests;
