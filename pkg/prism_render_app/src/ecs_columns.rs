//! Persistent, row-addressed GPU mirrors of table-backed ECS columns (§15).
//!
//! `T: Pod` is the explicit CPU/GPU byte-layout contract: use a shader-compatible
//! `repr(C)`/transparent type without padding or pointers. Nonzero strides must
//! be multiples of four. This bridge does not translate Rust layouts to WGSL.
//! It borrows physical chunk slices; it never packs entities into a CPU mirror.
//!
//! Synchronize after all simulation writers, once per change tick, before
//! submitting GPU consumers on the same device's queue. Advance the world's
//! tick before subsequent value writes. Snapshot restore is detected through
//! the world's archetype generation, including equal restored tick/epochs.
//! Call [`EcsResidentColumns::reset`] for external invalidation not reflected
//! in that generation. Ordinary structural edits are detected automatically,
//! including moves that preserve component ticks.

use std::any::TypeId;
use std::collections::BTreeMap;
use std::marker::PhantomData;
use std::sync::Arc;

use bytemuck::Pod;
use prism_app::App;
use prism_ecs::archetype::ArchetypeId;
use prism_ecs::change::Tick;
use prism_ecs::component::{Component, ComponentId, StorageType};
use prism_ecs::entity::Entity;
use prism_ecs::gpu_resident::{GpuResidentColumn, GpuUpload};
use prism_ecs::query::QueryState;
use prism_ecs::resource::Resource;
use prism_ecs::storage::{ColumnUploadSpan, TARGET_CHUNK_BYTES};
use prism_ecs::world::{World, WorldId};
use prism_render_driver::{
    BufferDescriptor, BufferId, BufferUsages, FrameSync, Instance, RenderDevice, RenderQueue,
};

use crate::{RenderApp, RenderRuntimeResource};

type SyncColumn =
    fn(&mut World, &mut World, Arc<dyn RenderDevice + Send + Sync>, &dyn RenderQueue, bool);

#[derive(Default)]
struct GpuColumnDeclarations(Vec<(TypeId, SyncColumn)>);
impl Resource for GpuColumnDeclarations {}

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
}

impl GpuColumnAppExt for App {
    fn register_gpu_column<T: Component + Pod>(
        &mut self,
    ) -> Result<&mut Self, EcsColumnUploadError> {
        register_gpu_column::<T>(self)?;
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
    resident: GpuResidentColumn,
    chunks: Vec<ChunkState>,
    buffer: Option<Arc<OwnedBuffer>>,
}

#[derive(Clone, Copy, PartialEq, Eq)]
struct ChunkState {
    incarnation: u64,
    epoch: u64,
    rows: usize,
}

/// Mirrors one nominated component type, with one buffer per source archetype.
///
/// World identity and archetype identity together scope row addresses; a single
/// component-id keyed buffer would alias unrelated archetypes. The render world
/// can own this cache while `sync` borrows the authoritative source world during
/// extraction. [`register_gpu_column`] installs automatic plugin synchronization.
/// GPU copies are dirty-only. Structural validation inspects one structural
/// epoch and live row count per chunk, without scanning entity identities or
/// repacking component bytes. A world archetype-generation change resets the
/// cache before this per-chunk validation, so recreated physical blocks cannot
/// inherit upload history.
pub struct EcsResidentColumns<T: Component + Pod> {
    device: Arc<dyn RenderDevice + Send + Sync>,
    world_id: WorldId,
    world_generation: u64,
    component: ComponentId,
    query: QueryState<(Entity, &'static T)>,
    columns: BTreeMap<ArchetypeId, ColumnState>,
    last_upload: Tick,
    generation: u64,
}

impl<T: Component + Pod> Resource for EcsResidentColumns<T> {}

impl<T: Component + Pod> EcsResidentColumns<T> {
    /// Register the nominated component and bind this cache to one source world.
    pub fn new(
        device: Arc<dyn RenderDevice + Send + Sync>,
        world: &mut World,
    ) -> Result<Self, EcsColumnUploadError> {
        validate_layout::<T>()?;
        let query = world.query::<(Entity, &T)>();
        let component = world.components().id_of::<T>().expect("query registered T");
        Ok(Self {
            device,
            world_id: world.id(),
            world_generation: world.archetype_generation(),
            component,
            query,
            columns: BTreeMap::new(),
            last_upload: Tick::ZERO,
            generation: 0,
        })
    }

    /// Retain the allocation and current row extent for one source archetype.
    pub fn column(&self, archetype: ArchetypeId) -> Option<ResidentEcsColumn> {
        self.columns
            .get(&archetype)
            .map(|column| ResidentEcsColumn {
                buffer: column.buffer.clone(),
                rows: column.resident.len(),
                stride: size_of::<T>(),
            })
    }

    /// Discard upload history and release the cache's allocation references.
    /// The next sync fully initializes all live columns, including tick-zero
    /// data. Previously exported views retain their own allocations.
    pub fn reset(&mut self) {
        self.columns.clear();
        self.last_upload = Tick::ZERO;
    }

    /// Reset after device replacement. Pass the replacement device's queue to
    /// the next sync. Old views remain tied to the old device and must be retired
    /// by their consumers; they cannot be submitted on the replacement device.
    pub fn reset_device(&mut self, device: Arc<dyn RenderDevice + Send + Sync>) {
        self.reset();
        self.device = device;
    }

    /// Queue only dirty physical chunks, plus chunks whose row membership or
    /// live extent changed. Growth fully initializes a new allocation; shrink
    /// retains its high-water capacity. Submission belongs to the render driver.
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
        // Validate all capacities before acknowledging any column's dirty bits.
        for archetype in world.archetypes().iter() {
            let table = archetype.table();
            if table.column(self.component).is_none() || stride == 0 {
                continue;
            }
            let required = table.len().div_ceil(table.rows_per_chunk());
            let previous = if replaced {
                0
            } else {
                self.columns
                    .get(&archetype.id())
                    .map_or(0, |column| column.resident.gpu_capacity_blocks())
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
        }

        if replaced {
            self.reset();
            self.world_generation = world.archetype_generation();
        }
        let now = world.change_tick();
        // Backward/ambiguous tick windows need a full refresh. Ordinary u32
        // wraparound within MAX_CHANGE_AGE remains an incremental window.
        let refresh = self.last_upload.age_since(now) > Tick::MAX_CHANGE_AGE;
        let mut plans: BTreeMap<ArchetypeId, GpuUpload> = BTreeMap::new();
        let mut spans: Vec<ColumnUploadSpan> = Vec::new();
        let mut stats = EcsColumnUploadStats::default();
        self.columns.retain(|id, _| {
            world
                .archetypes()
                .get(*id)
                .is_some_and(|archetype| archetype.table().column(self.component).is_some())
        });
        for archetype in world.archetypes().iter() {
            let table = archetype.table();
            if table.column(self.component).is_none() {
                continue;
            }
            let state = self
                .columns
                .entry(archetype.id())
                .or_insert_with(|| ColumnState {
                    resident: GpuResidentColumn::new(stride, table.rows_per_chunk()),
                    chunks: Vec::new(),
                    buffer: None,
                });
            state.resident.truncate(table.len());
            state.resident.ensure_len(table.len());
            if stride == 0 {
                continue; // markers have rows, but never allocate or upload
            }
            spans.clear();
            table.column_upload_spans(self.component, self.last_upload, now, &mut spans);
            for span in &spans {
                state
                    .resident
                    .mark_range(span.buffer_offset / stride, span.len / stride);
            }
            for (chunk_index, chunk) in table.iter_chunks().enumerate() {
                if chunk.rows == 0 {
                    continue; // reserved physical blocks need not contain rows
                }
                let current = ChunkState {
                    incarnation: chunk.incarnation,
                    epoch: chunk.meta.structural_epoch(),
                    rows: chunk.rows,
                };
                let changed = state.chunks.get(chunk_index) != Some(&current);
                if refresh || changed {
                    state.resident.mark_range(chunk.first_row, chunk.rows);
                }
                if chunk_index == state.chunks.len() {
                    state.chunks.push(current);
                } else if changed {
                    state.chunks[chunk_index] = current;
                }
            }
            state.chunks.truncate(table.chunk_count());
            let plan = state.resident.take_upload_plan();
            if let Some(allocation) = plan.reallocate {
                self.generation += 1;
                let capacity_bytes = allocation.capacity_bytes as u64;
                let mut descriptor = BufferDescriptor::new(
                    capacity_bytes,
                    BufferUsages::STORAGE | BufferUsages::COPY_DST | BufferUsages::COPY_SRC,
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
            }
            if !plan.spans.is_empty() {
                plans.insert(archetype.id(), plan);
            }
        }

        if !plans.is_empty() {
            // K exceeds every nonzero component's physical chunk row count.
            // Entity's first row identifies each slice without relying on query
            // archetype ordering. `cast_slice` is safe because T is Pod. Raw
            // ColumnUploadSpan pointers are used only as dirty-range metadata.
            for (entities, values) in self.query.simd_iter::<TARGET_CHUNK_BYTES>(world) {
                let location = world
                    .entities()
                    .location(entities[0])
                    .expect("live slice entity");
                let Some(plan) = plans.get(&location.archetype_id) else {
                    continue;
                };
                let start = location.row as usize * stride;
                let bytes = bytemuck::cast_slice(values);
                let end = start + bytes.len();
                let first = plan
                    .spans
                    .partition_point(|span| span.byte_offset + span.byte_len <= start);
                for span in &plan.spans[first..] {
                    if span.byte_offset >= end {
                        break;
                    }
                    let from = start.max(span.byte_offset);
                    let to = end.min(span.byte_offset + span.byte_len);
                    let buffer = self.columns[&location.archetype_id]
                        .buffer
                        .as_ref()
                        .expect("allocated upload");
                    queue.write_buffer(buffer.id, from as u64, &bytes[from - start..to - start]);
                    stats.upload_calls += 1;
                    stats.uploaded_bytes += (to - from) as u64;
                }
            }
        }
        self.last_upload = now;
        Ok(stats)
    }
}
