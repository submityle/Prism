//! Persistent, row-addressed GPU mirrors of table-backed ECS columns (§15).
//!
//! `T: Pod` is the explicit CPU/GPU byte-layout contract: use a shader-compatible
//! `repr(C)`/transparent type without padding or pointers. Nonzero strides must
//! be multiples of four. This bridge does not translate Rust layouts to WGSL.
//! It borrows physical chunk slices; it never packs entities into a CPU mirror.
//!
//! Synchronize after simulation writers before submitting GPU consumers on the
//! same device's queue. Independent column revisions detect tracked writes even
//! across multiple syncs in one tick or writes stamped with historical ticks.
//! Snapshot restore is detected through the world's archetype generation,
//! including equal restored tick/epochs.
//! Call [`EcsResidentColumns::reset`] for external invalidation not reflected
//! in that generation. Ordinary structural edits are detected automatically,
//! including moves that preserve component ticks.

use std::any::TypeId;
use std::collections::{BTreeMap, BTreeSet};
use std::marker::PhantomData;
use std::sync::Arc;

use bytemuck::Pod;
use prism_app::App;
use prism_ecs::archetype::{ArchetypeId, ColumnChangeCursor};
use prism_ecs::component::{Component, ComponentId, StorageType};
use prism_ecs::entity::Entity;
use prism_ecs::resource::Resource;
use prism_ecs::world::{World, WorldId};
use prism_render_architecture::gpu_scene::SceneHandle;
use prism_render_driver::{
    BufferDescriptor, BufferId, BufferUsages, FrameSync, Instance, RenderDevice, RenderQueue,
};

use crate::{RenderApp, RenderRuntimeResource};

type SyncColumn =
    fn(&mut World, &mut World, Arc<dyn RenderDevice + Send + Sync>, &dyn RenderQueue, bool);

#[derive(Default)]
struct GpuColumnDeclarations(Vec<(TypeId, SyncColumn)>);
impl Resource for GpuColumnDeclarations {}

struct TransformColumnDeclaration {
    id: TypeId,
    contains: fn(&World, Entity) -> bool,
    extract: fn(&mut World, &World, &mut ResidentTransformRows),
}

#[derive(Default)]
struct TransformColumnDeclarations(Vec<TransformColumnDeclaration>);
impl Resource for TransformColumnDeclarations {}

/// A retained physical column row, not a second packed transform allocation.
#[derive(Clone)]
pub(crate) struct ResidentTransformRow {
    pub column: ResidentEcsColumn,
    pub row: usize,
}

#[derive(Clone, Default)]
pub(crate) struct ResidentTransformRows(
    pub BTreeMap<SceneHandle, Result<ResidentTransformRow, String>>,
);
impl Resource for ResidentTransformRows {}

impl ResidentTransformRows {
    /// A draw closure owns these views only until the next preparation. Drain
    /// submitted reads before dropping the last view of a replaced allocation.
    /// In-place updates use the same queue, so unchanged allocations need no wait.
    pub(crate) fn releases_on_device(
        &self,
        next: &Self,
        device: &Arc<dyn RenderDevice + Send + Sync>,
    ) -> bool {
        if self.0.len() == next.0.len()
            && self
                .0
                .iter()
                .zip(&next.0)
                .all(|((old_handle, old), (new_handle, new))| {
                    old_handle == new_handle
                        && match (old, new) {
                            (Ok(old), Ok(new)) => match (&old.column.buffer, &new.column.buffer) {
                                (Some(old), Some(new)) => Arc::ptr_eq(old, new),
                                (None, None) => true,
                                _ => false,
                            },
                            (Err(_), _) => true,
                            _ => false,
                        }
                })
        {
            return false;
        }
        let retained: BTreeSet<_> = next
            .0
            .values()
            .filter_map(|row| row.as_ref().ok())
            .filter_map(|row| row.column.buffer.as_ref())
            .filter(|buffer| Arc::ptr_eq(&buffer.device, device))
            .map(|buffer| buffer.id)
            .collect();
        self.0
            .values()
            .filter_map(|row| row.as_ref().ok())
            .filter_map(|row| row.column.buffer.as_ref())
            .any(|buffer| Arc::ptr_eq(&buffer.device, device) && !retained.contains(&buffer.id))
    }
}

pub(crate) fn has_resident_transform(world: &World, entity: Entity) -> bool {
    world
        .get_resource::<TransformColumnDeclarations>()
        .is_some_and(|declarations| declarations.0.iter().any(|d| (d.contains)(world, entity)))
}

fn extract_transform_rows<T: Component + Pod>(
    main: &mut World,
    render: &World,
    output: &mut ResidentTransformRows,
) {
    let mut query = main.query::<(Entity, &crate::RenderTransformBinding, &T)>();
    for (entity, binding, _) in query.iter(main) {
        let row = (|| {
            let report = render
                .get_resource::<GpuColumnSyncReport<T>>()
                .ok_or_else(|| "resident transform column has not been synchronized".to_owned())?;
            report
                .result
                .map_err(|error| format!("resident transform upload: {error:?}"))?;
            let location = main
                .entities()
                .location(entity)
                .expect("live transform entity");
            let column = render
                .resource::<EcsResidentColumns<T>>()
                .column(location.archetype_id)
                .filter(|column| {
                    column.buffer_id().is_some() && (location.row as usize) < column.rows
                })
                .ok_or_else(|| "resident transform row is not uploaded".to_owned())?;
            Ok(ResidentTransformRow {
                column,
                row: location.row as usize,
            })
        })();
        match output.0.entry(binding.0) {
            std::collections::btree_map::Entry::Vacant(entry) => {
                entry.insert(row);
            }
            std::collections::btree_map::Entry::Occupied(mut entry) => {
                let _ = entry.insert(Err(format!(
                    "duplicate resident transform scene binding: {:?}",
                    binding.0
                )));
            }
        }
    }
}

/// Declare a table component for automatic main-world → render-world upload.
///
/// Call before or after adding [`crate::PrismRenderPlugin`]. Repeated declarations
/// are idempotent. The plugin synchronizes after main-world writers during
/// extraction, before the Render schedule. Consumers read
/// `EcsResidentColumns<T>` and `GpuColumnSyncReport<T>` in the render world,
/// use source-world archetype/row addresses, and retain returned column views
/// with their recorded work. A declaration never creates a main-world GPU resource.
pub fn register_gpu_column<T: Component + Pod>(app: &mut App) -> Result<(), EcsColumnUploadError> {
    validate_layout::<T>()?;
    app.world_mut().register_component::<T>();
    app.world_mut().init_resource::<GpuColumnDeclarations>();
    let declarations = app.world_mut().resource_mut::<GpuColumnDeclarations>();
    if !declarations
        .0
        .iter()
        .any(|(id, _)| *id == TypeId::of::<T>())
    {
        declarations.0.push((TypeId::of::<T>(), sync_column::<T>));
    }
    Ok(())
}

/// App extension for declaring resident GPU columns independently of the backend.
pub trait GpuColumnAppExt {
    /// See [`register_gpu_column`]. Returns the app for chained declarations.
    fn register_gpu_column<T: Component + Pod>(
        &mut self,
    ) -> Result<&mut Self, EcsColumnUploadError>;

    /// Use a resident column directly as the production mesh affine instance stream.
    ///
    /// `T` must contain exactly three row-major `[f32; 4]` affine rows (48 bytes).
    /// This is an explicit shader-layout contract, not an automatic conversion.
    /// Attach [`crate::RenderTransformBinding`] to select the scene instance.
    /// These entities bypass the legacy `GlobalTransform` packing/upload path;
    /// the application owns their world-space affine values. The mesh provider
    /// must support the existing transform locations 13..15. Indirect/bin draws
    /// are not supported by this direct-mesh contract and are rejected.
    fn register_gpu_transform_column<T: Component + Pod>(
        &mut self,
    ) -> Result<&mut Self, EcsColumnUploadError>;
}

impl GpuColumnAppExt for App {
    fn register_gpu_column<T: Component + Pod>(
        &mut self,
    ) -> Result<&mut Self, EcsColumnUploadError> {
        register_gpu_column::<T>(self)?;
        Ok(self)
    }

    fn register_gpu_transform_column<T: Component + Pod>(
        &mut self,
    ) -> Result<&mut Self, EcsColumnUploadError> {
        validate_layout::<T>()?;
        if size_of::<T>() != 48 {
            return Err(EcsColumnUploadError::InvalidTransformStride {
                stride: size_of::<T>(),
            });
        }
        register_gpu_column::<T>(self)?;
        self.world_mut()
            .init_resource::<TransformColumnDeclarations>();
        let declarations = self
            .world_mut()
            .resource_mut::<TransformColumnDeclarations>();
        if !declarations.0.iter().any(|d| d.id == TypeId::of::<T>()) {
            declarations.0.push(TransformColumnDeclaration {
                id: TypeId::of::<T>(),
                contains: |world, entity| world.has::<T>(entity),
                extract: extract_transform_rows::<T>,
            });
        }
        Ok(self)
    }
}

/// Result of the most recent automatic extraction for one declared component.
///
/// Present after its first extract. On failure the cache is reset so consumers
/// cannot accidentally bind stale rows; inspect this result before recording.
/// Device loss publishes `DeviceLost` even on clean frames, until a healthy
/// replacement device is available. Previously retained views must be retired
/// by consumers; they belong to the device that created them.
pub struct GpuColumnSyncReport<T: Component + Pod> {
    /// Work queued this frame, or the validation error preventing synchronization.
    pub result: Result<EcsColumnUploadStats, EcsColumnUploadError>,
    marker: PhantomData<T>,
}
impl<T: Component + Pod> Resource for GpuColumnSyncReport<T> {}

fn validate_layout<T: Component + Pod>() -> Result<(), EcsColumnUploadError> {
    if T::STORAGE != StorageType::Table {
        return Err(EcsColumnUploadError::NotTableStorage);
    }
    let stride = size_of::<T>();
    if !stride.is_multiple_of(4) {
        return Err(EcsColumnUploadError::UnalignedStride { stride });
    }
    Ok(())
}

fn sync_column<T: Component + Pod>(
    main: &mut World,
    render: &mut World,
    device: Arc<dyn RenderDevice + Send + Sync>,
    queue: &dyn RenderQueue,
    upload_allowed: bool,
) {
    if !upload_allowed {
        if let Some(cache) = render.get_resource_mut::<EcsResidentColumns<T>>() {
            cache.reset();
        }
        render.insert_resource(GpuColumnSyncReport::<T> {
            result: Err(EcsColumnUploadError::DeviceLost),
            marker: PhantomData,
        });
        return;
    }
    let mut cache = render
        .remove_resource::<EcsResidentColumns<T>>()
        .filter(|cache| cache.world_id == main.id())
        .unwrap_or_else(|| {
            EcsResidentColumns::new(device.clone(), main).expect("validated GPU column declaration")
        });
    if !Arc::ptr_eq(&cache.device, &device) {
        cache.reset_device(device);
    }
    let result = cache.sync(queue, main);
    if result.is_err() {
        cache.reset();
    }
    render.insert_resource(cache);
    render.insert_resource(GpuColumnSyncReport::<T> {
        result,
        marker: PhantomData,
    });
}

pub(crate) fn install_gpu_column_extraction<I>(app: &mut App)
where
    I: Instance + Send + Sync + 'static,
    I::Adapter: Send + Sync + 'static,
    I::Device: Send + Sync + 'static,
    I::Queue: Send + Sync + FrameSync + 'static,
    I::Surface: Send + Sync + 'static,
{
    app.add_extract(RenderApp, |main, render| {
        // Replace every extraction, including a replacement main world with no
        // declarations, so stale scene bindings cannot outlive their source.
        render.insert_resource(ResidentTransformRows::default());
        let Some(declarations) = main.get_resource::<GpuColumnDeclarations>() else {
            return;
        };
        let callbacks: Vec<_> = declarations.0.iter().map(|(_, sync)| *sync).collect();
        // Temporarily own the runtime so typed callbacks can insert render-world
        // resources while borrowing its matching device and queue. GPU state
        // stays in the render sub-app throughout the extraction boundary.
        let runtime = render
            .remove_resource::<RenderRuntimeResource<I>>()
            .expect("GPU column extraction requires the render runtime");
        // Extraction precedes recovery's render-stage gate. Poll the runtime's
        // retained loss record, not the backend's destructive take method, so
        // the recovery driver still observes the same loss later this frame.
        let upload_allowed = runtime.runtime().device_lost().is_none();
        for sync in callbacks {
            sync(
                main,
                render,
                runtime.runtime().device_handle(),
                runtime.runtime().queue(),
                upload_allowed,
            );
        }
        render.insert_resource(runtime);
        let callbacks: Vec<_> = main
            .get_resource::<TransformColumnDeclarations>()
            .map(|declarations| declarations.0.iter().map(|d| d.extract).collect())
            .unwrap_or_default();
        let mut rows = ResidentTransformRows::default();
        for extract in callbacks {
            extract(main, render, &mut rows);
        }
        render.insert_resource(rows);
    });
}

/// Validation failures are reported before consuming this frame's dirty state.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum EcsColumnUploadError {
    /// Sparse/shared storage cannot be mirrored as an archetype's row column.
    NotTableStorage,
    /// Queue writes require offsets and lengths aligned to four bytes.
    UnalignedStride {
        /// CPU element size rejected by the queue alignment contract.
        stride: usize,
    },
    /// Component/archetype ids are local to one world.
    WrongWorld,
    /// Overflow or a capacity exceeding the RHI storage-buffer binding limit.
    CapacityExceeded,
    /// Automatic extraction observed device loss and invalidated its cache.
    /// No column writes or dirty-state acknowledgements were issued. A healthy
    /// replacement device triggers full initialization on a later extraction.
    DeviceLost,
    /// The direct mesh affine stream consists of exactly three float4 rows.
    InvalidTransformStride {
        /// Rejected CPU/GPU element size.
        stride: usize,
    },
}

/// Actual work queued by one synchronization (clean frames report zero copies).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct EcsColumnUploadStats {
    /// Bytes passed to the queue, excluding unused allocation capacity.
    pub uploaded_bytes: u64,
    /// Queue writes; contiguous logical spans may occupy separate CPU chunks.
    pub upload_calls: usize,
    /// New GPU allocations, including replacements during growth.
    pub allocations: usize,
}

/// Source-side work of the last successful sync, separate from upload bytes.
/// Full baseline/restore/log-gap and allocation growth are intentionally NOT
/// strict change-volume paths. Normal discovery costs O(events since cursor),
/// with ordered sparse sets adding O(log affected archetypes/chunks) bookkeeping.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct EcsColumnDiscoveryStats {
    /// All source journal events read, including unrelated component writes.
    pub events_read: usize,
    /// Discovery table lookups/iteration visits; payload addressing is separate.
    pub archetypes_visited: usize,
    /// Indexed chunk candidates checked, including removed/empty candidates.
    pub chunks_visited: usize,
    /// Physical typed slices fetched for upload (never full-query iteration).
    pub payload_chunks_read: usize,
    /// Full registry rebuilds: first sync, reset, restore or retained-log gap.
    pub full_rebuilds: usize,
}

struct OwnedBuffer {
    device: Arc<dyn RenderDevice + Send + Sync>,
    id: BufferId,
    capacity_bytes: u64,
    generation: u64,
}

impl Drop for OwnedBuffer {
    fn drop(&mut self) {
        self.device.destroy_buffer(self.id);
    }
}

/// A retained GPU allocation and a snapshot of its live row extent.
///
/// Clones keep the allocation alive across growth, reset, or cache destruction.
/// Retain this value alongside bind groups/recorded work until they are retired.
/// In-place uploads change its bytes; row/length metadata is for the sync that
/// produced this view. Refresh the view each frame, and bind only `live_bytes()`.
/// Empty and zero-sized columns have no bindable live range. Tail capacity after
/// shrink is deliberately unspecified and must not be read by consumers.
#[derive(Clone)]
pub struct ResidentEcsColumn {
    buffer: Option<Arc<OwnedBuffer>>,
    /// Number of live rows at the synchronization that produced this view.
    pub rows: usize,
    /// Exact CPU/GPU element size, with no padding added by this bridge.
    pub stride: usize,
}

impl ResidentEcsColumn {
    /// Allocation handle, absent for zero-sized or never-populated columns.
    pub fn buffer_id(&self) -> Option<BufferId> {
        self.buffer.as_ref().map(|buffer| buffer.id)
    }

    /// Bindable live extent; a zero extent must not be bound.
    pub fn live_bytes(&self) -> u64 {
        (self.rows * self.stride) as u64
    }

    /// Reserved GPU bytes, retaining the high-water mark after shrink.
    pub fn capacity_bytes(&self) -> u64 {
        self.buffer
            .as_ref()
            .map_or(0, |buffer| buffer.capacity_bytes)
    }

    /// Changes whenever this cache replaces an allocation, including on reset.
    pub fn generation(&self) -> u64 {
        self.buffer.as_ref().map_or(0, |buffer| buffer.generation)
    }
}

struct ColumnState {
    rows: usize,
    capacity_blocks: usize,
    buffer: Option<Arc<OwnedBuffer>>,
}

/// Mirrors one nominated component type, with one buffer per source archetype.
///
/// World identity and archetype identity together scope row addresses; a single
/// component-id keyed buffer would alias unrelated archetypes. The render world
/// can own this cache while `sync` borrows the authoritative source world during
/// extraction. [`register_gpu_column`] installs automatic plugin synchronization.
/// A non-destructive revision journal discovers writes AND structural edits
/// without scanning clean archetypes/chunks. Independent cursors see same-tick
/// writes. Structural events also invalidate recycled physical block addresses.
/// Restore, reset and journal gaps rebuild a full baseline; growth uploads every
/// live chunk of the affected newly allocated buffer.
pub struct EcsResidentColumns<T: Component + Pod> {
    device: Arc<dyn RenderDevice + Send + Sync>,
    world_id: WorldId,
    world_generation: u64,
    component: ComponentId,
    cursor: Option<ColumnChangeCursor>,
    columns: BTreeMap<ArchetypeId, ColumnState>,
    generation: u64,
    discovery: EcsColumnDiscoveryStats,
    marker: PhantomData<T>,
}

impl<T: Component + Pod> Resource for EcsResidentColumns<T> {}

impl<T: Component + Pod> EcsResidentColumns<T> {
    /// Register the nominated component and bind this cache to one source world.
    pub fn new(
        device: Arc<dyn RenderDevice + Send + Sync>,
        world: &mut World,
    ) -> Result<Self, EcsColumnUploadError> {
        validate_layout::<T>()?;
        world.register_component::<T>();
        let component = world.components().id_of::<T>().expect("registered T");
        Ok(Self {
            device,
            world_id: world.id(),
            world_generation: world.archetype_generation(),
            component,
            cursor: None,
            columns: BTreeMap::new(),
            generation: 0,
            discovery: EcsColumnDiscoveryStats::default(),
            marker: PhantomData,
        })
    }

    /// Retain the allocation and current row extent for one source archetype.
    pub fn column(&self, archetype: ArchetypeId) -> Option<ResidentEcsColumn> {
        self.columns
            .get(&archetype)
            .map(|column| ResidentEcsColumn {
                buffer: column.buffer.clone(),
                rows: column.rows,
                stride: size_of::<T>(),
            })
    }

    /// Discard upload history and release the cache's allocation references.
    /// The next sync fully initializes all live columns, including tick-zero
    /// data. Previously exported views retain their own allocations.
    pub fn reset(&mut self) {
        self.columns.clear();
        self.cursor = None;
    }

    /// Discovery and addressed-fetch operation counts for the last successful
    /// sync. An idle successful frame reports all zeros, not merely zero uploads.
    pub fn discovery_stats(&self) -> EcsColumnDiscoveryStats {
        self.discovery
    }

    /// Reset after device replacement. Pass the replacement device's queue to
    /// the next sync. Old views remain tied to the old device and must be retired
    /// by their consumers; they cannot be submitted on the replacement device.
    pub fn reset_device(&mut self, device: Arc<dyn RenderDevice + Send + Sync>) {
        self.reset();
        self.device = device;
    }

    /// Queue only dirty physical chunks, plus chunks whose row membership or
    /// live extent changed. Write revisions, not observer tick windows, detect
    /// same-tick and explicitly backdated writes. Growth fully initializes a new
    /// allocation; shrink retains its high-water capacity. Submission belongs to
    /// the render driver.
    ///
    /// The queue must belong to the device passed at construction/reset; the RHI
    /// does not expose queue identity or recoverable upload errors. A host must
    /// handle device loss through `reset_device`, not acknowledge failed work.
    pub fn sync(
        &mut self,
        queue: &dyn RenderQueue,
        world: &World,
    ) -> Result<EcsColumnUploadStats, EcsColumnUploadError> {
        if world.id() != self.world_id {
            return Err(EcsColumnUploadError::WrongWorld);
        }
        let stride = size_of::<T>();
        let replaced = self.world_generation != world.archetype_generation();
        let limit = self
            .device
            .capabilities()
            .limits
            .max_storage_buffer_binding_size;
        let changes =
            world
                .archetypes()
                .column_changes_since(if replaced { None } else { self.cursor });
        let mut discovery = EcsColumnDiscoveryStats {
            events_read: changes.events.len(),
            full_rebuilds: usize::from(changes.gap),
            ..Default::default()
        };
        let mut affected: BTreeMap<ArchetypeId, BTreeSet<usize>> = BTreeMap::new();
        if changes.gap {
            for archetype in world.archetypes().iter() {
                discovery.archetypes_visited += 1;
                if archetype.table().column(self.component).is_some() {
                    affected.insert(archetype.id(), BTreeSet::new());
                }
            }
        } else {
            for event in &changes.events {
                if event.component.is_none() || event.component == Some(self.component) {
                    let chunks = affected.entry(event.archetype).or_default();
                    if let Some(chunk) = event.chunk {
                        chunks.insert(chunk);
                    }
                }
            }
        }

        // Validate ALL affected capacities before mutating state/acknowledging
        // the cursor. An error is retryable without silently losing the delta.
        let mut capacities = BTreeMap::new();
        for &id in affected.keys() {
            discovery.archetypes_visited += 1;
            let Some(archetype) = world.archetypes().get(id) else {
                continue;
            };
            let table = archetype.table();
            if table.column(self.component).is_none() {
                continue;
            }
            // Prewarm can reserve arbitrarily many empty CPU blocks. Only live
            // row windows need GPU capacity (or validation against its limit).
            let required = if stride == 0 {
                0
            } else {
                table.len().div_ceil(table.rows_per_chunk())
            };
            let previous = if replaced {
                0
            } else {
                self.columns
                    .get(&id)
                    .map_or(0, |column| column.capacity_blocks)
            };
            let capacity = if required > previous {
                required
                    .checked_next_power_of_two()
                    .ok_or(EcsColumnUploadError::CapacityExceeded)?
            } else {
                previous
            };
            let bytes = capacity
                .checked_mul(table.rows_per_chunk())
                .and_then(|rows| rows.checked_mul(stride))
                .ok_or(EcsColumnUploadError::CapacityExceeded)?;
            if bytes as u64 > limit {
                return Err(EcsColumnUploadError::CapacityExceeded);
            }
            capacities.insert(id, capacity);
        }

        if replaced {
            self.reset();
            self.world_generation = world.archetype_generation();
        }
        let mut stats = EcsColumnUploadStats::default();
        for (id, chunks) in affected {
            discovery.archetypes_visited += 1;
            let Some(archetype) = world.archetypes().get(id) else {
                continue;
            };
            let table = archetype.table();
            let Some(&capacity) = capacities.get(&id) else {
                continue;
            };
            let initialize = changes.gap || !self.columns.contains_key(&id);
            let state = self.columns.entry(id).or_insert_with(|| ColumnState {
                rows: 0,
                capacity_blocks: 0,
                buffer: None,
            });
            state.rows = table.len();
            if stride == 0 {
                continue; // markers have rows, but never allocate or upload
            }
            let reallocate = capacity > state.capacity_blocks;
            if reallocate {
                self.generation += 1;
                let capacity_bytes = (capacity * table.rows_per_chunk() * stride) as u64;
                let mut descriptor = BufferDescriptor::new(
                    capacity_bytes,
                    BufferUsages::STORAGE
                        | BufferUsages::COPY_DST
                        | BufferUsages::COPY_SRC
                        | BufferUsages::VERTEX,
                );
                descriptor.label = Some(format!(
                    "ecs-column-{}-{}",
                    self.component.index(),
                    archetype.id().index()
                ));
                let id = self.device.create_buffer(&descriptor);
                state.buffer = Some(Arc::new(OwnedBuffer {
                    device: self.device.clone(),
                    id,
                    capacity_bytes,
                    generation: self.generation,
                }));
                stats.allocations += 1;
                state.capacity_blocks = capacity;
            }
            // Explicit baseline/growth visits live windows directly, without
            // materializing a dense set or traversing prewarmed empty blocks.
            // The normal path only checks deduplicated journal candidates,
            // including removed tails so their live row extent stays current.
            let full = initialize || reallocate;
            let live_chunks = if full {
                table.len().div_ceil(table.rows_per_chunk())
            } else {
                0
            };
            for chunk_index in (0..live_chunks).chain(chunks.into_iter().filter(|_| !full)) {
                discovery.chunks_visited += 1;
                let Some(values) = world.column_chunk::<T>(id, chunk_index) else {
                    continue;
                };
                discovery.payload_chunks_read += 1;
                let bytes = bytemuck::cast_slice(values);
                let buffer = state.buffer.as_ref().expect("allocated upload");
                queue.write_buffer(
                    buffer.id,
                    (chunk_index * table.rows_per_chunk() * stride) as u64,
                    bytes,
                );
                stats.upload_calls += 1;
                stats.uploaded_bytes += bytes.len() as u64;
            }
        }
        self.cursor = Some(changes.cursor);
        self.discovery = discovery;
        Ok(stats)
    }
}
