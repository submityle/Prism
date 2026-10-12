//! The material stage: the render world's material registry, and the system
//! that keeps the ABI buffers in step with it.

use alloc::vec::Vec;

use prism_ecs::prelude::{Res, ResMut, Resource};
use prism_render_driver::{FrameSync, Instance};
use prism_render_architecture::abi::GenerationalHandle;
use prism_render_material::{
    MaterialRecord, MaterialRegistry, MaterialRegistryError, MaterialSnapshot,
    fallback_material_record,
};

/// The index of the permanent fallback material (slot zero).
pub const FALLBACK_MATERIAL_INDEX: u32 = 0;

use crate::runtime_resource::RenderRuntimeResource;

use super::buffers::{MaterialParameterBuffers, MaterialSyncReport};

/// The render world's material registry.
///
/// The authoritative records live here, in the render world, because that is
/// where they are needed: the ABI buffers are built from them and a shader reads
/// them every frame. The simulation publishes into this registry through the
/// extract step (the same one-way seam as every other render input), which is
/// why the wrapper — and not a bare `MaterialRegistry` — is the resource: the
/// registry type belongs to `prism_render_material`, so the resource marker has
/// to be implemented on a type local to this crate.
pub struct RenderMaterials {
    registry: MaterialRegistry,
}

impl RenderMaterials {
    /// A registry with room for `capacity` materials.
    #[must_use]
    pub fn new(capacity: u32) -> Self {
        Self {
            registry: MaterialRegistry::new(capacity),
        }
    }

    /// Publishes (or replaces) a material record, marking its row dirty.
    ///
    /// # Errors
    /// Propagates the registry's own errors: a handle outside the capacity, or a
    /// revision that would move backwards.
    pub fn publish(&mut self, record: MaterialRecord) -> Result<(), MaterialRegistryError> {
        self.registry.publish(record)
    }

    /// The registry's current epoch and live-material count.
    #[must_use]
    pub fn snapshot(&self) -> MaterialSnapshot {
        self.registry.snapshot()
    }

    /// The registry itself, for systems that only read it.
    #[must_use]
    pub fn registry(&self) -> &MaterialRegistry {
        &self.registry
    }

    /// The registry itself, mutably.
    pub fn registry_mut(&mut self) -> &mut MaterialRegistry {
        &mut self.registry
    }
}

impl Resource for RenderMaterials {}

/// The most recent upload pass's report.
#[derive(Resource, Default, Debug)]
pub struct MaterialUploadStats {
    /// What the last pass did.
    pub last: MaterialSyncReport,
}

/// Packs the registry's dirty materials into the ABI buffers.
///
/// Runs in [`RenderSystems::PrepareResources`](crate::RenderSystems::PrepareResources),
/// before anything reads the buffers or binds them. Only dirty rows are touched,
/// so a frame where no material changed uploads nothing; a dirty index with no
/// record is a retirement, and its row is rewritten as an inactive placeholder
/// while its word run goes back to the heap.
pub(crate) fn sync_material_buffers<I>(
    mut buffers: ResMut<MaterialParameterBuffers>,
    mut materials: ResMut<RenderMaterials>,
    mut stats: ResMut<MaterialUploadStats>,
    runtime: Res<RenderRuntimeResource<I>>,
) where
    I: Instance + Send + Sync + 'static,
    I::Adapter: Send + Sync + 'static,
    I::Device: Send + Sync + 'static,
    I::Queue: Send + Sync + FrameSync + 'static,
    I::Surface: Send + Sync + 'static,
{
    let dirty = materials.registry_mut().take_dirty();
    if dirty.is_empty() {
        return;
    }
    let epoch = materials.registry().snapshot().epoch;
    let device = runtime.runtime().device();
    let queue = runtime.runtime().queue();

    let mut records = Vec::<(u32, MaterialRecord)>::new();
    let mut retired = Vec::<u32>::new();
    for index in dirty {
        // Slot zero is the permanent fallback, not a scene material: it has no
        // published record, and a shader that reads a stale handle must find a
        // valid lit surface there rather than a retired row.
        if index == FALLBACK_MATERIAL_INDEX {
            let handle = GenerationalHandle::new(FALLBACK_MATERIAL_INDEX, 0);
            records.push((index, fallback_material_record(handle, epoch)));
            continue;
        }
        match materials.registry().record_at(index) {
            Some(record) => records.push((index, record.clone())),
            None => retired.push(index),
        }
    }

    let mut report = buffers.sync(device, queue, records, epoch);
    for index in retired {
        let generation = materials.registry().generation_at(index).unwrap_or(0);
        buffers.remove(queue, index, generation);
        report.headers_written += 1;
    }
    stats.last = report;
}

