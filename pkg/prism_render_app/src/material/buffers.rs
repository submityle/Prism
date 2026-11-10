//! The material ABI in GPU memory: header rows and the parameter word heap.
//!
//! Two buffers hold every material: a row per material in the header buffer, and
//! a shared word heap holding each material's packed surface block at the
//! offset its row names. A shader reads the row for the material it is shading
//! and decodes `parameter_size / 4` words from the heap with `lobe_mask`, which
//! is why the heap is *variable* length — a plain dielectric costs 12 words, a
//! fully-loaded surface up to 36 — and why it needs a real sub-allocator
//! ([`ParameterHeap`]) rather than a fixed stride.
//!
//! This layer places and uploads the ABI; what a header contains and how a
//! surface packs stays in `prism_render_material`.

use core::any::Any;
use core::mem::size_of;
use core::sync::atomic::{AtomicBool, AtomicU32, Ordering};

use alloc::collections::BTreeMap;
use alloc::vec::Vec;

use alloc::sync::Arc;
use prism_render_driver::{
    BufferDescriptor, BufferId, BufferUsages, FrameSync, RenderDevice, RenderQueue,
};
use prism_render_material::EXECUTION_PLAN_ROW_WORDS;
use prism_render_material::{
    fallback_material_record, inactive_material_header, DynamicChannels, GpuMaterialHeader,
    GpuMaterialTexture, MaterialExecutionBinding, MaterialRecord, MaterialRegistry,
};

use super::virtual_texture::MaterialVtLeaseSnapshot;
use super::ParameterHeap;

/// The usages the header buffer is created with: a shader reads the rows.
pub(crate) const HEADER_USAGES: BufferUsages = BufferUsages::STORAGE.union(BufferUsages::COPY_DST);
/// The usages the parameter word heap is created with: a shader reads the words.
pub(crate) const PARAMETER_USAGES: BufferUsages =
    BufferUsages::STORAGE.union(BufferUsages::COPY_DST);
/// The usages the texture row buffer is created with: a shader reads the rows.
pub(crate) const TEXTURE_USAGES: BufferUsages = BufferUsages::STORAGE.union(BufferUsages::COPY_DST);

/// Immutable, completion-published execution-table buffers.
///
/// Graphs retain this object as an external owner. Its buffers are retired only
/// after both graph recording/submission owners and queue completion owners have
/// released it, and are always destroyed through the device that created them.
struct ExecutionTableBufferOwner {
    device: Arc<dyn RenderDevice + Send + Sync>,
    plans: BufferId,
    programs: BufferId,
}

impl Drop for ExecutionTableBufferOwner {
    fn drop(&mut self) {
        self.device.destroy_buffer(self.plans);
        self.device.destroy_buffer(self.programs);
    }
}

struct ExecutionTableSnapshotInner {
    version: u32,
    plan_rows: Vec<[u32; EXECUTION_PLAN_ROW_WORDS]>,
    program_rows: Vec<[u32; 5]>,
    registered_response_wgsl: Arc<str>,
    buffers: ExecutionTableBufferOwner,
    ready: AtomicBool,
}

/// A single immutable execution-table generation suitable for binding to a
/// frame graph. Retaining `external_owner()` pins both buffers to submission
/// completion even if a newer registry version is prepared meanwhile.
#[derive(Clone)]
pub struct NativeExecutionTableSnapshot(Arc<ExecutionTableSnapshotInner>);

impl NativeExecutionTableSnapshot {
    /// Registry version represented by this completed snapshot.
    #[must_use]
    pub fn version(&self) -> u32 {
        self.0.version
    }

    /// Immutable GPU plan table.
    #[must_use]
    pub fn plan_buffer(&self) -> BufferId {
        self.0.buffers.plans
    }

    /// Immutable GPU program table.
    #[must_use]
    pub fn program_buffer(&self) -> BufferId {
        self.0.buffers.programs
    }

    /// CPU mirror of the exact plan rows uploaded to this snapshot.
    #[must_use]
    pub fn plan_rows(&self) -> &[[u32; EXECUTION_PLAN_ROW_WORDS]] {
        &self.0.plan_rows
    }

    /// CPU mirror of the exact program rows uploaded to this snapshot.
    #[must_use]
    pub fn program_rows(&self) -> &[[u32; 5]] {
        &self.0.program_rows
    }

    /// Exact registered response link section for this execution generation.
    #[must_use]
    pub fn registered_response_wgsl(&self) -> &str {
        &self.0.registered_response_wgsl
    }

    /// Owner token for `RenderGraph::retain_external_owner`.
    #[must_use]
    pub fn external_owner(&self) -> Arc<dyn Any + Send + Sync> {
        self.0.clone()
    }

    pub(crate) fn same_device(&self, material: &MaterialBufferSnapshot) -> bool {
        Arc::ptr_eq(&self.0.buffers.device, &material.allocation.device)
    }
}

/// GPU mirror for the registry's verified execution plan and program tables.
///
/// CPU rows remain authoritative across device recovery. Every registry change
/// uploads into fresh immutable buffers; the previously completed generation
/// remains available until the replacement's queue completion proves it ready.
#[derive(prism_ecs::prelude::Resource)]
pub struct NativeExecutionTableBuffers {
    active: Option<Arc<ExecutionTableSnapshotInner>>,
    pending: Option<Arc<ExecutionTableSnapshotInner>>,
    plan_rows: Vec<[u32; EXECUTION_PLAN_ROW_WORDS]>,
    program_rows: Vec<[u32; 5]>,
    registered_response_wgsl: String,
    requested_version: u32,
    completion_generation: Arc<AtomicU32>,
}

/// Why an execution-table mirror refused a sync.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum ExecutionTableSyncError {
    /// The registry has not installed verified execution tables.
    TablesUnavailable,
}

impl NativeExecutionTableBuffers {
    /// Creates an empty mirror. Buffers are allocated on the first successful sync.
    #[must_use]
    pub fn new() -> Self {
        Self {
            active: None,
            pending: None,
            plan_rows: Vec::new(),
            program_rows: Vec::new(),
            requested_version: 0,
            completion_generation: Arc::new(AtomicU32::new(0)),
            registered_response_wgsl: String::new(),
        }
    }

    /// Plan storage buffer, available only for a completion-confirmed version.
    #[must_use]
    pub fn plan_buffer(&self) -> Option<BufferId> {
        self.snapshot().map(|snapshot| snapshot.plan_buffer())
    }

    /// Program storage buffer, available only for a completion-confirmed version.
    #[must_use]
    pub fn program_buffer(&self) -> Option<BufferId> {
        self.snapshot().map(|snapshot| snapshot.program_buffer())
    }

    /// Number of valid plan rows in the mirror (a one-row sentinel for empty tables).
    #[must_use]
    pub fn plan_count(&self) -> u32 {
        self.ready_inner()
            .map_or(self.plan_rows.len(), |snapshot| snapshot.plan_rows.len()) as u32
    }

    /// Number of valid program rows in the mirror (a one-row sentinel for empty tables).
    #[must_use]
    pub fn program_count(&self) -> u32 {
        self.ready_inner()
            .map_or(self.program_rows.len(), |snapshot| {
                snapshot.program_rows.len()
            }) as u32
    }

    fn ready_inner(&self) -> Option<&Arc<ExecutionTableSnapshotInner>> {
        self.pending
            .as_ref()
            .filter(|snapshot| snapshot.ready.load(Ordering::Acquire))
            .or(self.active.as_ref())
            .filter(|snapshot| snapshot.ready.load(Ordering::Acquire))
    }

    /// Registry execution-table version currently confirmed on the GPU.
    #[must_use]
    pub fn execution_table_version(&self) -> Option<u32> {
        self.snapshot().map(|snapshot| snapshot.version())
    }

    /// Latest completed immutable snapshot. An unfinished replacement never
    /// hides the last published generation from graphs being recorded.
    #[must_use]
    pub fn snapshot(&self) -> Option<NativeExecutionTableSnapshot> {
        self.ready_inner()
            .cloned()
            .map(NativeExecutionTableSnapshot)
    }

    /// CPU mirror of plan rows, including the sentinel row used for empty tables.
    #[must_use]
    pub fn plan_rows(&self) -> &[[u32; EXECUTION_PLAN_ROW_WORDS]] {
        self.ready_inner()
            .map(|snapshot| snapshot.plan_rows.as_slice())
            .unwrap_or(&self.plan_rows)
    }

    /// CPU mirror of program rows, including the sentinel row used for empty tables.
    #[must_use]
    pub fn program_rows(&self) -> &[[u32; 5]] {
        self.ready_inner()
            .map(|snapshot| snapshot.program_rows.as_slice())
            .unwrap_or(&self.program_rows)
    }

    /// Copies the registry's authoritative GPU rows and queues a full upload.
    /// A fresh allocation is used for each version; no queue-wide idle is
    /// required, and an incomplete upload cannot replace the active snapshot.
    pub fn sync_registry<D: RenderDevice + Send + Sync + 'static, Q: RenderQueue + FrameSync>(
        &mut self,
        device: Arc<D>,
        queue: &Q,
        registry: &MaterialRegistry,
    ) -> Result<bool, ExecutionTableSyncError> {
        let Some(tables) = registry.execution_tables() else {
            self.pending = None;
            self.active = None;
            self.registered_response_wgsl.clear();
            self.invalidate();
            return Err(ExecutionTableSyncError::TablesUnavailable);
        };
        let version = registry.execution_table_version();
        if self
            .pending
            .as_ref()
            .is_some_and(|snapshot| snapshot.ready.load(Ordering::Acquire))
        {
            self.active = self.pending.take();
        }
        if self.requested_version == version
            && (self
                .active
                .as_ref()
                .is_some_and(|snapshot| snapshot.version == version)
                || self
                    .pending
                    .as_ref()
                    .is_some_and(|snapshot| snapshot.version == version))
        {
            return Ok(false);
        }

        let mut plans = tables.gpu_plan_rows();
        let mut programs = tables.gpu_program_rows();
        self.registered_response_wgsl = tables.registered_response_wgsl();
        if plans.is_empty() {
            plans.push([u32::MAX; EXECUTION_PLAN_ROW_WORDS]);
        }
        if programs.is_empty() {
            programs.push([0; 5]);
        }
        self.plan_rows = plans;
        self.program_rows = programs;
        let plan_capacity = (self.plan_rows.len() as u32).next_power_of_two().max(1);
        let program_capacity = (self.program_rows.len() as u32).next_power_of_two().max(1);
        let plans_id = device.create_buffer(&BufferDescriptor {
            label: Some("prism native material execution plans".into()),
            size: u64::from(plan_capacity) * (EXECUTION_PLAN_ROW_WORDS * 4) as u64,
            usage: HEADER_USAGES,
            mapped_at_creation: false,
        });
        let programs_id = device.create_buffer(&BufferDescriptor {
            label: Some("prism native material execution programs".into()),
            size: u64::from(program_capacity) * 20,
            usage: HEADER_USAGES,
            mapped_at_creation: false,
        });
        queue.write_buffer(plans_id, 0, bytemuck::cast_slice(&self.plan_rows));
        queue.write_buffer(programs_id, 0, bytemuck::cast_slice(&self.program_rows));
        self.requested_version = version;
        let device: Arc<dyn RenderDevice + Send + Sync> = device;
        let snapshot = Arc::new(ExecutionTableSnapshotInner {
            version,
            plan_rows: self.plan_rows.clone(),
            program_rows: self.program_rows.clone(),
            registered_response_wgsl: self.registered_response_wgsl.clone().into(),
            buffers: ExecutionTableBufferOwner {
                device,
                plans: plans_id,
                programs: programs_id,
            },
            ready: AtomicBool::new(false),
        });
        self.pending = Some(Arc::clone(&snapshot));
        let generation = self
            .completion_generation
            .fetch_add(1, Ordering::AcqRel)
            .wrapping_add(1);
        let completion_generation = Arc::clone(&self.completion_generation);
        queue.on_submitted_work_done(Box::new(move || {
            if completion_generation.load(Ordering::Acquire) == generation {
                snapshot.ready.store(true, Ordering::Release);
            }
        }));
        Ok(true)
    }

    /// Recreates buffers after device loss and replays the authoritative CPU rows.
    /// Old IDs are forgotten, not destroyed on the replacement device.
    pub fn recover<D: RenderDevice + Send + Sync + 'static, Q: RenderQueue + FrameSync>(
        &mut self,
        device: Arc<D>,
        queue: &Q,
    ) -> Result<(), ExecutionTableSyncError> {
        self.active = None;
        self.pending = None;
        let version = self.requested_version;
        if version == 0 {
            self.invalidate();
            return Ok(());
        }
        self.invalidate();
        let plans_id = device.create_buffer(&BufferDescriptor {
            label: Some("prism native material execution plans".into()),
            size: (self.plan_rows.len() as u64).next_power_of_two().max(1)
                * (EXECUTION_PLAN_ROW_WORDS * 4) as u64,
            usage: HEADER_USAGES,
            mapped_at_creation: false,
        });
        let programs_id = device.create_buffer(&BufferDescriptor {
            label: Some("prism native material execution programs".into()),
            size: (self.program_rows.len() as u64).next_power_of_two().max(1) * 20,
            usage: HEADER_USAGES,
            mapped_at_creation: false,
        });
        queue.write_buffer(plans_id, 0, bytemuck::cast_slice(&self.plan_rows));
        queue.write_buffer(programs_id, 0, bytemuck::cast_slice(&self.program_rows));
        // `invalidate` clears the published/requested version. Restore the
        // authoritative CPU mirror's version before scheduling completion so
        // the callback can publish this replayed allocation.
        self.requested_version = version;
        let device: Arc<dyn RenderDevice + Send + Sync> = device;
        let snapshot = Arc::new(ExecutionTableSnapshotInner {
            version,
            plan_rows: self.plan_rows.clone(),
            program_rows: self.program_rows.clone(),
            registered_response_wgsl: self.registered_response_wgsl.clone().into(),
            buffers: ExecutionTableBufferOwner {
                device,
                plans: plans_id,
                programs: programs_id,
            },
            ready: AtomicBool::new(false),
        });
        self.pending = Some(Arc::clone(&snapshot));
        let generation = self
            .completion_generation
            .fetch_add(1, Ordering::AcqRel)
            .wrapping_add(1);
        let completion_generation = Arc::clone(&self.completion_generation);
        queue.on_submitted_work_done(Box::new(move || {
            if completion_generation.load(Ordering::Acquire) == generation {
                snapshot.ready.store(true, Ordering::Release);
            }
        }));
        Ok(())
    }

    /// Releases current-device buffers and invalidates publication.
    pub fn release(&mut self) {
        self.active = None;
        self.pending = None;
        self.invalidate();
    }

    fn invalidate(&mut self) {
        self.requested_version = 0;
        self.completion_generation.fetch_add(1, Ordering::AcqRel);
    }
}

impl Default for NativeExecutionTableBuffers {
    fn default() -> Self {
        Self::new()
    }
}

/// One material's run in the word heap.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
struct Placement {
    /// The word offset of the packed surface block.
    offset: u32,
    /// The block's length in words.
    words: u32,
}

/// What one sync did, so a caller can tell a quiet frame from a rebuild.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
pub struct MaterialSyncReport {
    /// Materials whose header row was written this pass.
    pub headers_written: u32,
    /// Materials whose packed parameters were written this pass.
    pub parameters_written: u32,
    /// Materials whose texture rows were written this pass.
    pub textures_written: u32,
    /// Materials that gave up their previous run because their block changed
    /// size.
    pub relocated: u32,
    /// Whether a buffer was reallocated (and therefore fully re-uploaded).
    pub grew: bool,
}

/// The material ABI's two buffers, with a CPU mirror of each.
///
/// A render-world resource: the render world owns the GPU buffers, and the
/// prepare stage is the only thing that writes them.
///
/// The mirror is not a cache: it is how the buffers can *grow*. A reallocated
/// buffer keeps nothing, so the new one has to be filled from something that
/// survives the reallocation — the mirror — without asking the caller to hand
/// back every material it ever synced. It is also what keeps a quiet frame quiet:
/// only the rows and words that changed are uploaded.
#[derive(prism_ecs::prelude::Resource)]
pub struct MaterialParameterBuffers {
    headers: Option<BufferId>,
    header_capacity: u32,
    parameters: Option<BufferId>,
    parameter_capacity_words: u32,
    heap: ParameterHeap,
    textures: Option<BufferId>,
    texture_capacity_rows: u32,
    /// The texture table is the same kind of address space as the parameter heap
    /// — a material owns a variable-length run of rows — so it uses the same
    /// allocator, with rows for words.
    texture_heap: ParameterHeap,
    /// Row mirror, one entry per header slot.
    rows: Vec<GpuMaterialHeader>,
    /// Word mirror, one entry per parameter word slot.
    words: Vec<u32>,
    /// Texture row mirror, one entry per texture row slot.
    texture_rows: Vec<GpuMaterialTexture>,
    placements: BTreeMap<u32, Placement>,
    /// Each material's run in the texture table.
    texture_placements: BTreeMap<u32, Placement>,
    /// Extra usages every created buffer carries — `COPY_SRC` when a caller wants
    /// a diagnostic readback; production passes none.
    extra_usage: BufferUsages,
    /// Device-owned allocation generation. Frame graphs retain snapshots of
    /// this Arc so a growth/recovery replacement cannot destroy buffers still
    /// referenced by an in-flight submission.
    allocation_owner: Option<Arc<MaterialBufferAllocation>>,
    /// Present on the production path; test-only borrowed constructors retain
    /// the historical explicit-release behavior.
    allocation_device: Option<Arc<dyn RenderDevice + Send + Sync>>,
    /// Whether the current GPU allocations have been initialized from the CPU
    /// mirrors.  A freshly-created backend buffer is not required to contain
    /// zeroes, while a recovered allocation is filled from the mirrors.  Keeping
    /// this bit lets the first normal sync initialize the whole allocation once,
    /// so recovery has deterministic bytes to replay (including slack).
    gpu_initialized: bool,
    /// Plan/program registry version corresponding to the CPU material mirror.
    /// Captured by value in each snapshot generation.
    execution_table_version: u32,
    /// VT publication generation corresponding to the CPU material rows.
    /// Captured by value so a frame cannot mix rows with another atlas lease.
    vt_publication_generation: u64,
    /// Registry epoch whose exact header rows are present in this allocation.
    material_epoch: u64,
}

/// One physical generation of the three canonical material ABI buffers.
struct MaterialBufferAllocation {
    device: Arc<dyn RenderDevice + Send + Sync>,
    headers: BufferId,
    parameters: BufferId,
    textures: BufferId,
    abandoned: AtomicBool,
}

impl Drop for MaterialBufferAllocation {
    fn drop(&mut self) {
        if self.abandoned.load(Ordering::Acquire) {
            return;
        }
        self.device.destroy_buffer(self.headers);
        self.device.destroy_buffer(self.parameters);
        self.device.destroy_buffer(self.textures);
    }
}

/// Immutable byte-generation snapshot of the current material ABI buffers.
/// While a snapshot is retained, any subsequent dirty upload or retirement
/// detaches the producer onto a fresh allocation before writing. The physical
/// buffers and their bytes therefore remain stable through graph submission.
#[derive(Clone)]
pub struct MaterialBufferSnapshot {
    allocation: Arc<MaterialBufferAllocation>,
    header_capacity: u32,
    parameter_capacity_words: u32,
    texture_capacity_rows: u32,
    execution_table_version: u32,
    vt_publication_generation: u64,
    material_epoch: u64,
}

/// The immutable logical material frame consumed by a native render graph.
///
/// Material headers/parameters and execution plans are two physical resource
/// families, but they are one logical ABI: header rows contain dense indices
/// into the execution tables. Keeping the pair in a typed bundle prevents a
/// caller from accidentally binding two independently captured generations.
#[derive(Clone)]
pub struct MaterialFrameSnapshotBundle {
    material: MaterialBufferSnapshot,
    execution: NativeExecutionTableSnapshot,
    vt: Option<MaterialVtLeaseSnapshot>,
}

/// Why two otherwise valid physical snapshots cannot form one material frame.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum MaterialFrameSnapshotError {
    /// Zero is reserved for “not published”; it can never authorize a draw.
    UnpublishedVersion,
    /// Headers and dense execution rows were captured from different registry
    /// publications and therefore cannot safely be consumed together.
    VersionMismatch { material: u32, execution: u32 },
    /// The physical resources came from different creator devices.
    DeviceMismatch,
    /// Material rows and a VT lease belong to different publications.
    VirtualTextureGenerationMismatch { material: u64, vt: u64 },
}

impl MaterialFrameSnapshotBundle {
    /// Combines two completion-published snapshots only when their logical
    /// registry publication is identical and non-zero.
    pub fn new(
        material: MaterialBufferSnapshot,
        execution: NativeExecutionTableSnapshot,
    ) -> Result<Self, MaterialFrameSnapshotError> {
        let material_version = material.execution_table_version();
        let execution_version = execution.version();
        if material_version == 0 || execution_version == 0 {
            return Err(MaterialFrameSnapshotError::UnpublishedVersion);
        }
        if material_version != execution_version {
            return Err(MaterialFrameSnapshotError::VersionMismatch {
                material: material_version,
                execution: execution_version,
            });
        }
        if !execution.same_device(&material) {
            return Err(MaterialFrameSnapshotError::DeviceMismatch);
        }
        Ok(Self {
            material,
            execution,
            vt: None,
        })
    }

    /// Combines all material resources consumed by a frame, including the
    /// exact VT lease whose page table and atlas back the material rows.
    pub fn new_with_vt(
        material: MaterialBufferSnapshot,
        execution: NativeExecutionTableSnapshot,
        vt: MaterialVtLeaseSnapshot,
    ) -> Result<Self, MaterialFrameSnapshotError> {
        let mut bundle = Self::new(material, execution)?;
        if bundle.material.vt_publication_generation != vt.publication_generation()
            && vt.publication_generation() != 0
        {
            return Err(
                MaterialFrameSnapshotError::VirtualTextureGenerationMismatch {
                    material: bundle.material.vt_publication_generation,
                    vt: vt.publication_generation(),
                },
            );
        }
        if !vt.same_device(&bundle.material) {
            return Err(MaterialFrameSnapshotError::DeviceMismatch);
        }
        bundle.vt = Some(vt);
        Ok(bundle)
    }

    #[must_use]
    pub const fn material(&self) -> &MaterialBufferSnapshot {
        &self.material
    }

    #[must_use]
    pub const fn execution(&self) -> &NativeExecutionTableSnapshot {
        &self.execution
    }

    #[must_use]
    pub const fn vt(&self) -> Option<&MaterialVtLeaseSnapshot> {
        self.vt.as_ref()
    }

    #[must_use]
    pub const fn version(&self) -> u32 {
        self.material.execution_table_version()
    }

    /// Retains both creator-device allocations through graph submission and
    /// completion. The bundle itself is the single owner token for this frame.
    #[must_use]
    pub fn external_owner(self: &Arc<Self>) -> Arc<dyn Any + Send + Sync> {
        self.clone()
    }
}

impl MaterialBufferSnapshot {
    #[must_use]
    pub fn headers(&self) -> BufferId {
        self.allocation.headers
    }

    #[must_use]
    pub fn parameters(&self) -> BufferId {
        self.allocation.parameters
    }

    #[must_use]
    pub fn textures(&self) -> BufferId {
        self.allocation.textures
    }

    #[must_use]
    pub const fn header_capacity(&self) -> u32 {
        self.header_capacity
    }

    #[must_use]
    pub const fn parameter_capacity_words(&self) -> u32 {
        self.parameter_capacity_words
    }

    #[must_use]
    pub const fn texture_capacity_rows(&self) -> u32 {
        self.texture_capacity_rows
    }

    /// Execution-table registry version used when these material rows were
    /// last synchronized. Consumers must pair this snapshot only with the
    /// matching immutable execution-table snapshot.
    #[must_use]
    pub const fn execution_table_version(&self) -> u32 {
        self.execution_table_version
    }

    #[must_use]
    pub fn external_owner(&self) -> Arc<dyn Any + Send + Sync> {
        self.allocation.clone()
    }

    pub(crate) fn device_identity(&self) -> Arc<dyn RenderDevice + Send + Sync> {
        Arc::clone(&self.allocation.device)
    }

    #[must_use]
    pub const fn vt_publication_generation(&self) -> u64 {
        self.vt_publication_generation
    }

    /// Registry epoch stamped into the immutable header rows of this snapshot.
    #[must_use]
    pub const fn material_epoch(&self) -> u64 {
        self.material_epoch
    }
}

impl MaterialParameterBuffers {
    /// Allocates the ABI buffers, sized for `header_capacity` materials and
    /// `parameter_words` of surface data.
    #[must_use]
    pub fn new<D: RenderDevice>(
        device: &D,
        header_capacity: u32,
        parameter_words: u32,
        texture_rows: u32,
        extra_usage: BufferUsages,
    ) -> Self {
        let mut buffers = Self {
            headers: None,
            header_capacity: 0,
            parameters: None,
            parameter_capacity_words: 0,
            heap: ParameterHeap::new(),
            textures: None,
            texture_capacity_rows: 0,
            texture_heap: ParameterHeap::new(),
            rows: Vec::new(),
            words: Vec::new(),
            texture_rows: Vec::new(),
            placements: BTreeMap::new(),
            texture_placements: BTreeMap::new(),
            extra_usage,
            allocation_owner: None,
            allocation_device: None,
            gpu_initialized: false,
            execution_table_version: 0,
            vt_publication_generation: 0,
            material_epoch: 0,
        };
        buffers.allocate(
            device,
            header_capacity.max(1),
            parameter_words.max(1),
            texture_rows.max(1),
            extra_usage,
        );
        buffers
    }

    /// Production constructor retaining the creating device so allocation
    /// generations can be retired only after their graph/submission owners
    /// release them.
    #[must_use]
    pub fn new_owned<D: RenderDevice + Send + Sync + 'static>(
        device: Arc<D>,
        header_capacity: u32,
        parameter_words: u32,
        texture_rows: u32,
        extra_usage: BufferUsages,
    ) -> Self {
        let mut buffers = Self::new(
            device.as_ref(),
            header_capacity,
            parameter_words,
            texture_rows,
            extra_usage,
        );
        let device: Arc<dyn RenderDevice + Send + Sync> = device;
        buffers.allocation_device = Some(Arc::clone(&device));
        buffers.allocation_owner = Some(Arc::new(MaterialBufferAllocation {
            device,
            headers: buffers.headers.expect("constructor allocated headers"),
            parameters: buffers
                .parameters
                .expect("constructor allocated parameters"),
            textures: buffers
                .textures
                .expect("constructor allocated texture rows"),
            abandoned: AtomicBool::new(false),
        }));
        buffers
    }

    /// Captures the exact physical allocation generation and a graph-retainable
    /// owner. Returns `None` only after explicit release.
    #[must_use]
    pub fn snapshot(&self) -> Option<MaterialBufferSnapshot> {
        Some(MaterialBufferSnapshot {
            allocation: Arc::clone(self.allocation_owner.as_ref()?),
            header_capacity: self.header_capacity,
            parameter_capacity_words: self.parameter_capacity_words,
            texture_capacity_rows: self.texture_capacity_rows,
            execution_table_version: self.execution_table_version,
            vt_publication_generation: self.vt_publication_generation,
            material_epoch: self.material_epoch,
        })
    }

    /// Records the active VT publication that the next material snapshot names.
    pub(crate) fn set_vt_publication_generation(&mut self, generation: u64) {
        self.vt_publication_generation = generation;
    }

    /// Execution-table registry version paired with the current material
    /// mirror. Graph consumers use [`Self::snapshot`]; this crate-visible
    /// accessor is also useful for headless upload admission diagnostics.
    #[must_use]
    pub(crate) const fn execution_table_version(&self) -> u32 {
        self.execution_table_version
    }

    /// The header buffer (one [`GpuMaterialHeader`] per material slot).
    #[must_use]
    pub const fn headers(&self) -> Option<BufferId> {
        self.headers
    }

    /// The parameter word heap.
    #[must_use]
    pub const fn parameters(&self) -> Option<BufferId> {
        self.parameters
    }

    /// The texture row buffer.
    #[must_use]
    pub const fn textures(&self) -> Option<BufferId> {
        self.textures
    }

    /// How many header slots the buffer holds.
    #[must_use]
    pub const fn header_capacity(&self) -> u32 {
        self.header_capacity
    }

    /// How many parameter words the heap buffer holds.
    #[must_use]
    pub const fn parameter_capacity_words(&self) -> u32 {
        self.parameter_capacity_words
    }

    /// How many words the heap currently uses (one past the last allocated
    /// word), i.e. how far into the buffer a shader may read.
    #[must_use]
    pub fn used_words(&self) -> u32 {
        self.heap.word_capacity()
    }

    /// How many texture rows the table currently uses.
    #[must_use]
    pub fn used_texture_rows(&self) -> u32 {
        self.texture_heap.word_capacity()
    }

    /// How many texture rows the buffer holds.
    #[must_use]
    pub const fn texture_capacity_rows(&self) -> u32 {
        self.texture_capacity_rows
    }

    /// The texture rows of `index`, as last uploaded.
    #[must_use]
    pub fn textures_of(&self, index: u32) -> Option<&[GpuMaterialTexture]> {
        let placement = self.texture_placements.get(&index)?;
        let start = placement.offset as usize;
        self.texture_rows
            .get(start..start + placement.words as usize)
    }

    /// The header row of `index`, as last uploaded.
    #[must_use]
    pub fn header(&self, index: u32) -> Option<&GpuMaterialHeader> {
        self.rows.get(index as usize)
    }

    /// The packed word run of `index`, as last uploaded.
    #[must_use]
    pub fn parameters_of(&self, index: u32) -> Option<&[u32]> {
        let placement = self.placements.get(&index)?;
        let start = placement.offset as usize;
        self.words.get(start..start + placement.words as usize)
    }

    /// Uploads dirty live rows from the authoritative registry, including its
    /// sanitized dynamic channels and validated execution bindings. Retirements
    /// are handled by the caller through `remove`; slot zero remains fallback.
    /// Records, revisions and generations cannot be supplied independently of
    /// the registry's runtime state. Unchanged heap contents are not uploaded
    /// unless an allocation grows or requires initialization.
    pub fn sync_registry<D: RenderDevice, Q: RenderQueue>(
        &mut self,
        device: &D,
        queue: &Q,
        registry: &MaterialRegistry,
        dirty: impl IntoIterator<Item = u32>,
    ) -> MaterialSyncReport {
        let epoch = registry.snapshot().epoch;
        let execution_table_version = registry.execution_table_version();
        let records = dirty.into_iter().filter_map(|index| {
            if index == super::stage::FALLBACK_MATERIAL_INDEX {
                let handle = prism_render_architecture::abi::GenerationalHandle::new(index, 0);
                Some((
                    index,
                    fallback_material_record(handle, epoch),
                    DynamicChannels::NONE,
                    None,
                ))
            } else {
                Some((
                    index,
                    registry.record_at(index)?.clone(),
                    registry.dynamic_channels_at(index),
                    registry.execution_binding_at(index),
                ))
            }
        });
        let report = self.sync_uploads(device, queue, records, epoch);
        self.execution_table_version = execution_table_version;
        self.material_epoch = epoch;
        report
    }

    fn sync_uploads<D: RenderDevice, Q: RenderQueue>(
        &mut self,
        device: &D,
        queue: &Q,
        records: impl IntoIterator<
            Item = (
                u32,
                MaterialRecord,
                DynamicChannels,
                Option<MaterialExecutionBinding>,
            ),
        >,
        epoch: u64,
    ) -> MaterialSyncReport {
        let mut report = MaterialSyncReport::default();

        let mut dirty_headers = Vec::new();
        let mut dirty_words: Vec<(u32, u32)> = Vec::new();
        let mut dirty_textures: Vec<(u32, u32)> = Vec::new();

        for (index, record, dynamic, execution) in records {
            // A material past the current row count grows the header buffer; the
            // mirrors carry what was already there into the new allocation.
            if self.ensure_header(device, index + 1) {
                report.grew = true;
            }

            let packed = record.packed_parameters();
            let block = packed.pack();
            let len = block.len() as u32;

            let placement = match self.placements.get(&index) {
                // Same width: the run already fits. A lobe *value* changes far
                // more often than the lobe *set*, so this is the common case.
                Some(placement) if placement.words == len => *placement,
                previous => {
                    if let Some(previous) = previous {
                        self.heap.free(previous.offset, previous.words);
                        report.relocated += 1;
                    }
                    let placement = Placement {
                        offset: self.heap.alloc(len),
                        words: len,
                    };
                    self.placements.insert(index, placement);
                    placement
                }
            };

            // The heap may have bumped its high-water mark past what the buffer
            // covers, which is what grows the word heap.
            if self.ensure_words(device, self.heap.word_capacity()) {
                report.grew = true;
            }

            let start = placement.offset as usize;
            let slot = &mut self.words[start..start + len as usize];
            if slot != block.as_slice() {
                slot.copy_from_slice(&block);
                dirty_words.push((placement.offset, placement.words));
                report.parameters_written += 1;
            }

            // The material's texture entries own their own variable-length run
            // in the texture table, addressed by the header's `texture_offset` /
            // `texture_count` exactly as parameters are by `parameter_offset`.
            let textures = &record.textures;
            let texture_len = textures.len() as u32;
            let texture_placement = match self.texture_placements.get(&index) {
                Some(placement) if placement.words == texture_len => *placement,
                previous => {
                    if let Some(previous) = previous {
                        self.texture_heap.free(previous.offset, previous.words);
                    }
                    let placement = Placement {
                        offset: self.texture_heap.alloc(texture_len),
                        words: texture_len,
                    };
                    self.texture_placements.insert(index, placement);
                    placement
                }
            };
            if self.ensure_texture_rows(device, self.texture_heap.word_capacity()) {
                report.grew = true;
            }
            if texture_len != 0 {
                let start = texture_placement.offset as usize;
                let slot = &mut self.texture_rows[start..start + texture_len as usize];
                if slot != textures.as_slice() {
                    slot.copy_from_slice(textures);
                    dirty_textures.push((texture_placement.offset, texture_len));
                    report.textures_written += 1;
                }
            }

            // Construct the row only after heap placement: registry GPU tables
            // use a different parameter address space and cannot be copied here.
            let mut row = record.header_with_execution(
                placement.offset,
                texture_placement.offset,
                epoch,
                execution,
            );
            let [wetness, snow, damage, dissolve] = dynamic.to_words();
            row.dynamic_wetness = wetness;
            row.dynamic_snow = snow;
            row.dynamic_damage = damage;
            row.dynamic_dissolve = dissolve;
            let slot = &mut self.rows[index as usize];
            if *slot != row {
                *slot = row;
                dirty_headers.push(index);
                report.headers_written += 1;
            }
        }

        let has_changes = report.headers_written != 0
            || report.parameters_written != 0
            || report.textures_written != 0;
        let detached = has_changes && self.detach_if_pinned();
        if report.grew || detached || !self.gpu_initialized {
            self.upload_all(queue);
        } else {
            self.upload_rows(queue, &dirty_headers);
            self.upload_words(queue, &dirty_words);
            self.upload_textures(queue, &dirty_textures);
        }
        report
    }

    /// Retires a material: its run goes back to the heap and its row becomes an
    /// inactive tombstone carrying the slot's generation, so a shader holding
    /// a stale handle rejects the row instead of reading freed words.
    /// Returns false without changing mirrors, allocations or GPU bytes when
    /// the row is absent, inactive, or belongs to another generation.
    pub fn remove<Q: RenderQueue>(&mut self, queue: &Q, index: u32, generation: u32) -> bool {
        if self
            .rows
            .get(index as usize)
            .is_none_or(|row| row.generation != generation || row.is_active == 0)
        {
            return false;
        }
        let Some(placement) = self.placements.remove(&index) else {
            return false;
        };
        self.heap.free(placement.offset, placement.words);
        if let Some(textures) = self.texture_placements.remove(&index) {
            self.texture_heap.free(textures.offset, textures.words);
        }
        let slot = &mut self.rows[index as usize];
        let row = inactive_material_header(generation);
        if *slot != row {
            *slot = row;
            if self.detach_if_pinned() {
                self.upload_all(queue);
            } else {
                self.upload_rows(queue, &[index]);
            }
        }
        true
    }

    /// If a graph/submission still owns the current allocation, move the
    /// producer to fresh buffers before its next write. Unpinned generations
    /// retain the normal dirty-range path; COW cost is paid only when a real
    /// mutation overlaps an outstanding snapshot.
    fn detach_if_pinned(&mut self) -> bool {
        let Some(owner) = self.allocation_owner.as_ref() else {
            return false;
        };
        if Arc::strong_count(owner) == 1 {
            return false;
        }
        let Some(device) = self.allocation_device.clone() else {
            return false;
        };
        self.allocate(
            device.as_ref(),
            self.header_capacity.max(1),
            self.parameter_capacity_words.max(1),
            self.texture_capacity_rows.max(1),
            self.extra_usage,
        );
        true
    }

    /// Releases every buffer.
    pub fn release<D: RenderDevice>(&mut self, device: &D) {
        if self.allocation_device.is_some() {
            self.allocation_owner = None;
            self.headers = None;
            self.parameters = None;
            self.textures = None;
            self.gpu_initialized = false;
            return;
        }
        if let (Some(headers), Some(parameters), Some(textures)) =
            (self.headers, self.parameters, self.textures)
        {
            device.destroy_buffer(headers);
            device.destroy_buffer(parameters);
            device.destroy_buffer(textures);
        }
        self.headers = None;
        self.parameters = None;
        self.textures = None;
        self.gpu_initialized = false;
    }

    /// Abandons allocations from a lost device without issuing destroy calls.
    /// Any graph-retained snapshot shares the same gate, so it also becomes a
    /// no-op when its last owner is dropped after device loss.
    pub(crate) fn abandon_device(&mut self) {
        if let Some(owner) = self.allocation_owner.as_ref() {
            owner.abandoned.store(true, Ordering::Release);
        }
        self.allocation_owner = None;
        self.allocation_device = None;
        self.headers = None;
        self.parameters = None;
        self.textures = None;
        self.gpu_initialized = false;
    }

    /// Recreates all three buffers on a replacement device and restores their
    /// contents from the CPU mirrors.
    ///
    /// Old ids are deliberately forgotten without being destroyed: after
    /// device loss they belong to the dead device generation and must never be
    /// passed to the replacement device.
    pub(crate) fn recover<D: RenderDevice, Q: RenderQueue>(&mut self, device: &D, queue: &Q) {
        self.abandon_device();
        self.allocate(
            device,
            self.header_capacity.max(1),
            self.parameter_capacity_words.max(1),
            self.texture_capacity_rows.max(1),
            self.extra_usage,
        );
        self.upload_all(queue);
    }

    /// Recreates allocations on a replacement device while snapshots of the
    /// lost generation continue to retire through their original device.
    pub(crate) fn recover_owned<D, Q>(&mut self, device: Arc<D>, queue: &Q)
    where
        D: RenderDevice + Send + Sync + 'static,
        Q: RenderQueue,
    {
        let device_owner: Arc<dyn RenderDevice + Send + Sync> = device.clone();
        self.abandon_device();
        self.allocation_device = Some(device_owner);
        self.headers = None;
        self.parameters = None;
        self.textures = None;
        self.gpu_initialized = false;
        self.allocate(
            device.as_ref(),
            self.header_capacity.max(1),
            self.parameter_capacity_words.max(1),
            self.texture_capacity_rows.max(1),
            self.extra_usage,
        );
        self.upload_all(queue);
    }

    /// Grows the header buffer so it has at least `needed` rows, preserving the
    /// mirror. Returns whether it reallocated.
    fn ensure_header<D: RenderDevice>(&mut self, device: &D, needed: u32) -> bool {
        if needed <= self.header_capacity {
            return false;
        }
        let capacity = needed
            .next_power_of_two()
            .max(self.header_capacity.saturating_mul(2));
        self.allocate(
            device,
            capacity,
            self.parameter_capacity_words,
            self.texture_capacity_rows,
            self.extra_usage,
        );
        true
    }

    /// Grows the word heap buffer so it covers `needed` words, preserving the
    /// mirror. Returns whether it reallocated.
    fn ensure_words<D: RenderDevice>(&mut self, device: &D, needed: u32) -> bool {
        if needed <= self.parameter_capacity_words {
            return false;
        }
        let capacity = needed
            .next_power_of_two()
            .max(self.parameter_capacity_words.saturating_mul(2));
        self.allocate(
            device,
            self.header_capacity,
            capacity,
            self.texture_capacity_rows,
            self.extra_usage,
        );
        true
    }

    /// (Re)allocates both buffers, refilling them from the mirrors.
    fn allocate<D: RenderDevice + ?Sized>(
        &mut self,
        device: &D,
        header_capacity: u32,
        words: u32,
        texture_rows: u32,
        extra_usage: BufferUsages,
    ) {
        if self.allocation_device.is_none() {
            if let (Some(headers), Some(parameters), Some(textures)) =
                (self.headers, self.parameters, self.textures)
            {
                device.destroy_buffer(headers);
                device.destroy_buffer(parameters);
                device.destroy_buffer(textures);
            }
        }
        self.header_capacity = header_capacity;
        self.parameter_capacity_words = words;
        self.texture_capacity_rows = texture_rows;
        self.headers = Some(device.create_buffer(&BufferDescriptor {
            label: Some("prism material headers".into()),
            size: u64::from(header_capacity) * header_row_bytes(),
            usage: HEADER_USAGES.union(extra_usage),
            mapped_at_creation: false,
        }));
        self.parameters = Some(device.create_buffer(&BufferDescriptor {
            label: Some("prism material parameters".into()),
            size: u64::from(words) * 4,
            usage: PARAMETER_USAGES.union(extra_usage),
            mapped_at_creation: false,
        }));
        self.textures = Some(device.create_buffer(&BufferDescriptor {
            label: Some("prism material textures".into()),
            size: u64::from(texture_rows) * texture_row_bytes(),
            usage: TEXTURE_USAGES.union(extra_usage),
            mapped_at_creation: false,
        }));
        if let Some(owner_device) = self.allocation_device.as_ref() {
            self.allocation_owner = Some(Arc::new(MaterialBufferAllocation {
                device: Arc::clone(owner_device),
                headers: self.headers.expect("allocation created headers"),
                parameters: self.parameters.expect("allocation created parameters"),
                textures: self.textures.expect("allocation created texture rows"),
                abandoned: AtomicBool::new(false),
            }));
        }
        // The mirrors are the buffers' contents, so growth *extends* them: the
        // rows and words already written stay, and only the slack a reallocated
        // buffer gained starts as explicit values rather than whatever the driver
        // handed back.
        self.rows
            .resize(header_capacity as usize, inactive_material_header(0));
        self.words.resize(words as usize, 0);
        self.texture_rows
            .resize(texture_rows as usize, GpuMaterialTexture::default());
        self.gpu_initialized = false;
    }

    /// Uploads the whole mirror (after a growth).
    fn upload_all<Q: RenderQueue>(&mut self, queue: &Q) {
        let (Some(headers), Some(parameters), Some(textures)) =
            (self.headers, self.parameters, self.textures)
        else {
            return;
        };
        queue.write_buffer(headers, 0, &header_bytes(&self.rows));
        queue.write_buffer(parameters, 0, &word_bytes(&self.words));
        queue.write_buffer(textures, 0, &texture_bytes(&self.texture_rows));
        self.gpu_initialized = true;
    }

    /// Uploads the given texture runs, batched into the span they cover.
    fn upload_textures<Q: RenderQueue>(&self, queue: &Q, dirty: &[(u32, u32)]) {
        let Some(first) = dirty.iter().map(|(offset, _)| *offset).min() else {
            return;
        };
        let last = dirty
            .iter()
            .map(|(offset, rows)| offset + rows)
            .max()
            .expect("a dirty list is non-empty");
        let Some(textures) = self.textures else {
            return;
        };
        queue.write_buffer(
            textures,
            u64::from(first) * texture_row_bytes(),
            &texture_bytes(&self.texture_rows[first as usize..last as usize]),
        );
    }

    /// Grows the texture row buffer so it holds `needed` rows, preserving the
    /// mirror. Returns whether it reallocated.
    fn ensure_texture_rows<D: RenderDevice>(&mut self, device: &D, needed: u32) -> bool {
        if needed <= self.texture_capacity_rows {
            return false;
        }
        let capacity = needed
            .next_power_of_two()
            .max(self.texture_capacity_rows.saturating_mul(2));
        self.allocate(
            device,
            self.header_capacity,
            self.parameter_capacity_words,
            capacity,
            self.extra_usage,
        );
        true
    }

    /// Uploads the given header rows, batched into the span they cover.
    fn upload_rows<Q: RenderQueue>(&self, queue: &Q, dirty: &[u32]) {
        let Some((first, last)) = span(dirty) else {
            return;
        };
        let Some(headers) = self.headers else {
            return;
        };
        let rows = &self.rows[first as usize..=last as usize];
        queue.write_buffer(
            headers,
            u64::from(first) * header_row_bytes(),
            &header_bytes(rows),
        );
    }

    /// Uploads the given word runs, batched into the span they cover.
    fn upload_words<Q: RenderQueue>(&self, queue: &Q, dirty: &[(u32, u32)]) {
        let Some(first) = dirty.iter().map(|(offset, _)| *offset).min() else {
            return;
        };
        let last = dirty
            .iter()
            .map(|(offset, words)| offset + words)
            .max()
            .expect("a dirty list is non-empty");
        let Some(parameters) = self.parameters else {
            return;
        };
        queue.write_buffer(
            parameters,
            u64::from(first) * 4,
            &word_bytes(&self.words[first as usize..last as usize]),
        );
    }
}

#[cfg(test)]
mod ownership_tests {
    use super::*;
    use prism_render_driver::{DeviceDescriptor, Instance, RequestAdapterOptions};
    use prism_render_driver_null::NullInstance;

    #[test]
    fn allocation_snapshot_pins_old_buffers_across_growth() {
        let instance = NullInstance::new();
        let adapter = instance
            .request_adapter(&RequestAdapterOptions::default())
            .expect("null adapter exists");
        let (device, _queue) = instance
            .create_device(&adapter, &DeviceDescriptor::default())
            .expect("null device is available");
        let device = Arc::new(device);
        let mut buffers =
            MaterialParameterBuffers::new_owned(Arc::clone(&device), 1, 8, 2, BufferUsages::NONE);
        let old = buffers.snapshot().expect("initial allocation snapshot");
        let old_ids = (old.headers(), old.parameters(), old.textures());
        assert_eq!(device.live_resource_count(), 3);

        buffers.allocate(&*device, 2, 8, 2, BufferUsages::NONE);
        assert_eq!(device.live_resource_count(), 6);
        assert_eq!(
            (old.headers(), old.parameters(), old.textures()),
            old_ids,
            "the retained snapshot names the original physical generation"
        );
        assert_ne!(buffers.headers(), Some(old.headers()));

        drop(old);
        assert_eq!(device.live_resource_count(), 3);
        buffers.release(device.as_ref());
        assert_eq!(device.live_resource_count(), 0);
    }

    #[test]
    fn frame_snapshot_bundle_rejects_unpublished_and_mismatched_versions() {
        let instance = NullInstance::new();
        let adapter = instance
            .request_adapter(&RequestAdapterOptions::default())
            .expect("null adapter exists");
        let (device, _queue) = instance
            .create_device(&adapter, &DeviceDescriptor::default())
            .expect("null device is available");
        let device = Arc::new(device);
        let buffers =
            MaterialParameterBuffers::new_owned(Arc::clone(&device), 1, 8, 2, BufferUsages::NONE);
        let material = buffers.snapshot().expect("material allocation snapshot");
        let plans = device.create_buffer(&BufferDescriptor::new(16, HEADER_USAGES));
        let programs = device.create_buffer(&BufferDescriptor::new(20, HEADER_USAGES));
        let execution = NativeExecutionTableSnapshot(Arc::new(ExecutionTableSnapshotInner {
            version: 1,
            plan_rows: vec![[0; EXECUTION_PLAN_ROW_WORDS]],
            program_rows: vec![[0; 5]],
            registered_response_wgsl: Arc::<str>::from(""),
            buffers: ExecutionTableBufferOwner {
                device: Arc::clone(&device) as Arc<dyn RenderDevice + Send + Sync>,
                plans,
                programs,
            },
            ready: AtomicBool::new(true),
        }));

        assert!(matches!(
            MaterialFrameSnapshotBundle::new(material.clone(), execution.clone()),
            Err(MaterialFrameSnapshotError::UnpublishedVersion)
        ));

        let mismatched = MaterialBufferSnapshot {
            execution_table_version: 2,
            ..material
        };
        assert!(matches!(
            MaterialFrameSnapshotBundle::new(mismatched, execution),
            Err(MaterialFrameSnapshotError::VersionMismatch {
                material: 2,
                execution: 1,
            })
        ));
    }

    #[test]
    fn frame_bundle_keeps_vt_lease_and_rejects_a_different_publication() {
        let instance = NullInstance::new();
        let adapter = instance
            .request_adapter(&RequestAdapterOptions::default())
            .expect("null adapter exists");
        let (device, _queue) = instance
            .create_device(&adapter, &DeviceDescriptor::default())
            .expect("null device is available");
        let device = Arc::new(device);
        let mut buffers =
            MaterialParameterBuffers::new_owned(Arc::clone(&device), 1, 8, 2, BufferUsages::NONE);
        buffers.set_vt_publication_generation(7);
        buffers.execution_table_version = 1;
        let material = buffers.snapshot().expect("material allocation snapshot");
        let plans = device.create_buffer(&BufferDescriptor::new(16, HEADER_USAGES));
        let programs = device.create_buffer(&BufferDescriptor::new(20, HEADER_USAGES));
        let execution = NativeExecutionTableSnapshot(Arc::new(ExecutionTableSnapshotInner {
            version: 1,
            plan_rows: vec![[0; EXECUTION_PLAN_ROW_WORDS]],
            program_rows: vec![[0; 5]],
            registered_response_wgsl: Arc::<str>::from(""),
            buffers: ExecutionTableBufferOwner {
                device: Arc::clone(&device) as Arc<dyn RenderDevice + Send + Sync>,
                plans,
                programs,
            },
            ready: AtomicBool::new(true),
        }));
        let vt = MaterialVtLeaseSnapshot::for_test(
            Arc::new(()),
            Arc::clone(&device) as Arc<dyn RenderDevice + Send + Sync>,
            plans,
            programs,
            device.create_texture_view(
                device.create_texture(&prism_render_driver::TextureDescriptor::new_2d(
                    prism_render_driver::Extent3d::new_2d(1, 1),
                    prism_render_driver::TextureFormat::Rgba8Unorm,
                    prism_render_driver::TextureUsages::TEXTURE_BINDING,
                )),
                &prism_render_driver::TextureViewDescriptor::default(),
            ),
            device.create_sampler(&prism_render_driver::SamplerDescriptor::default()),
            0,
            7,
        );
        let bundle = MaterialFrameSnapshotBundle::new_with_vt(
            material.clone(),
            execution.clone(),
            vt.clone(),
        )
        .expect("same material/execution/VT generation is admissible");
        assert_eq!(bundle.vt().unwrap().publication_generation(), 7);

        let stale = MaterialVtLeaseSnapshot::for_test(
            Arc::new(()),
            Arc::clone(&device) as Arc<dyn RenderDevice + Send + Sync>,
            vt.page_table(),
            vt.config(),
            vt.atlas_view(),
            vt.sampler(),
            vt.device_generation(),
            8,
        );
        assert!(matches!(
            MaterialFrameSnapshotBundle::new_with_vt(material, execution, stale),
            Err(
                MaterialFrameSnapshotError::VirtualTextureGenerationMismatch { material: 7, vt: 8 }
            )
        ));
    }

    #[test]
    fn retained_snapshot_is_immutable_across_dirty_rows_and_retirement() {
        let instance = NullInstance::new();
        let adapter = instance
            .request_adapter(&RequestAdapterOptions::default())
            .expect("null adapter exists");
        let (device, queue) = instance
            .create_device(&adapter, &DeviceDescriptor::default())
            .expect("null device is available");
        let device = Arc::new(device);
        let mut registry = MaterialRegistry::new(2);
        registry.take_dirty();
        let handle = registry.allocate().expect("material slot is available");
        registry
            .publish(fallback_material_record(handle, 1))
            .expect("material publishes");
        let mut buffers =
            MaterialParameterBuffers::new_owned(Arc::clone(&device), 2, 64, 8, BufferUsages::NONE);
        let dirty = registry.take_dirty();
        buffers.sync_registry(&*device, &queue, &registry, dirty);

        let first = buffers.snapshot().expect("first material generation");
        let first_ids = (first.headers(), first.parameters(), first.textures());
        registry
            .set_dynamic_channels(
                handle,
                DynamicChannels {
                    wetness: 0.75,
                    ..DynamicChannels::NONE
                },
            )
            .expect("runtime channel update is valid");
        let dirty = registry.take_dirty();
        buffers.sync_registry(&*device, &queue, &registry, dirty);

        let second = buffers.snapshot().expect("second material generation");
        let second_ids = (second.headers(), second.parameters(), second.textures());
        assert_ne!(
            first_ids, second_ids,
            "a pinned generation is never overwritten"
        );
        assert_eq!(
            device.live_resource_count(),
            6,
            "both generations remain alive"
        );
        drop(first);
        assert_eq!(
            device.live_resource_count(),
            3,
            "the old generation retires on last release"
        );

        let second_generation = second.headers();
        assert!(buffers.remove(&queue, handle.index, handle.generation));
        let third = buffers
            .snapshot()
            .expect("retirement publishes a fresh allocation");
        assert_ne!(third.headers(), second_generation);
        assert_eq!(device.live_resource_count(), 6);
        drop(second);
        assert_eq!(device.live_resource_count(), 3);
        drop(third);
        buffers.release(device.as_ref());
        assert_eq!(device.live_resource_count(), 0);
    }

    #[test]
    fn material_snapshot_tracks_execution_version_without_material_dirty_rows() {
        let instance = NullInstance::new();
        let adapter = instance
            .request_adapter(&RequestAdapterOptions::default())
            .expect("null adapter exists");
        let (device, queue) = instance
            .create_device(&adapter, &DeviceDescriptor::default())
            .expect("null device is available");
        let device = Arc::new(device);
        let mut registry = MaterialRegistry::new(2);
        registry.take_dirty();
        let mut buffers =
            MaterialParameterBuffers::new_owned(Arc::clone(&device), 2, 64, 8, BufferUsages::NONE);

        let first_version = registry.execution_table_version();
        buffers.sync_registry(&*device, &queue, &registry, core::iter::empty());
        let first = buffers.snapshot().expect("initial snapshot");
        assert_eq!(first.execution_table_version(), first_version);
        assert_eq!(first.material_epoch(), registry.snapshot().epoch);

        registry
            .install_execution_tables(&prism_render_material::MaterialExecutionTables::new())
            .unwrap();
        let second_version = registry.execution_table_version();
        assert_ne!(first_version, second_version);
        buffers.sync_registry(&*device, &queue, &registry, core::iter::empty());
        let second = buffers.snapshot().expect("versioned snapshot");
        assert_eq!(second.execution_table_version(), second_version);
        assert_eq!(second.material_epoch(), registry.snapshot().epoch);
        assert_eq!(first.execution_table_version(), first_version);
        assert_eq!(first.material_epoch(), 0);
        assert_eq!(first.headers(), second.headers());
    }
}

/// The byte size of one header row.
#[must_use]
pub(crate) fn header_row_bytes() -> u64 {
    size_of::<GpuMaterialHeader>() as u64
}

/// The byte size of one texture row.
#[must_use]
pub(crate) fn texture_row_bytes() -> u64 {
    size_of::<GpuMaterialTexture>() as u64
}

/// Texture rows as bytes, in the layout the ABI defines.
fn texture_bytes(rows: &[GpuMaterialTexture]) -> Vec<u8> {
    bytemuck::cast_slice(rows).to_vec()
}

/// Rows as bytes, in the layout the ABI defines.
fn header_bytes(rows: &[GpuMaterialHeader]) -> Vec<u8> {
    bytemuck::cast_slice(rows).to_vec()
}

/// Words as little-endian bytes.
fn word_bytes(words: &[u32]) -> Vec<u8> {
    bytemuck::cast_slice(words).to_vec()
}

/// The inclusive `(first, last)` indices a dirty list covers.
fn span(dirty: &[u32]) -> Option<(u32, u32)> {
    Some((dirty.iter().copied().min()?, dirty.iter().copied().max()?))
}
