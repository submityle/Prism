//! Per-frame `GPU` simulation-pipeline orchestration: the ordered compute
//! sub-graph an emitter walks every frame (design §9).
//!
//! This module is the §9 *per-frame sub-graph orchestration layer*. It turns
//! one emitter's live counters into the ordered, conditionally-included list of
//! compute / draw passes that advance and render it, and it derives the small
//! scheduling facts a real backend needs: how many indirect dispatches each pass
//! takes, where the minimal set of pipeline barriers must sit, which passes may
//! overlap the previous frame's graphics on an asynchronous compute queue, and
//! how a persistent-thread (grid-stride) kernel divides its work.
//!
//! The design mirrors production `GPU`-driven `VFX` pipelines — Unreal
//! `Niagara`'s `GPU` simulation and `Frostbite`'s indirect compute stack — at
//! the algorithm level, without reusing their code. The ten ordered steps are:
//! `EmitterUpdate`, `Spawn`, `SimulationStages`, `Event Scatter`, `Compaction`
//! (optional), `Bounds`, `Cull`, `Sort` (conditional), `Fill Draw Args`, and
//! `Render Draw`.
//!
//! Division of responsibility: the raw dispatch arithmetic (workgroup count,
//! per-dispatch split, persistent-thread flag) is **not** re-derived here — it
//! reuses [`super::dual_backend::GpuDispatchPlan`] and
//! [`super::dual_backend::BackendTuning`], and this layer only orchestrates one
//! [`super::dual_backend::GpuDispatchPlan`] per pass. The in-domain numerical
//! solve for each simulation stage lives in [`super::stages`]; this module never
//! re-implements it, it only sequences the passes around it.
//!
//! Only ordinary integer arithmetic (with round-up via `div_ceil`) and a single
//! fragmentation ratio division are used — no transcendental functions — so the
//! contract stays reproducible against a future `GPU` kernel.

use alloc::vec::Vec;

use super::dual_backend::{BackendCapabilities, BackendTuning, GpuDispatchPlan};
use super::IterationDomain;

/// Absolute tolerance for comparing two fragmentation ratios (or other derived
/// `f32` scalars) in tests and internal equality checks.
///
/// `f32` fields are never compared with a bare `==`; this epsilon absorbs the
/// last-bit noise of a single division while still catching real divergence.
pub const CMP_EPS: f32 = 1.0e-6;

/// The default dead-slot fragmentation ratio at or above which the optional
/// `Compaction` pass (§9 step 5) is scheduled.
///
/// The ratio is the fraction of a compacted alive-list's slots that are dead
/// holes awaiting reclamation; below this the holes are cheap to skip, above it
/// a prefix-sum rebuild of the alive list pays for itself (design §5.2, §11).
pub const DEFAULT_COMPACTION_FRAGMENTATION_THRESHOLD: f32 = 0.5;

/// A logical `GPU` buffer the per-frame pipeline reads from or writes to.
///
/// These are the coarse resources the barrier analysis reasons about; each maps
/// to a bit in a [`ResourceSet`]. The concrete device buffers are owned by the
/// backend and are out of scope for this contract layer.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum PipelineResource {
    /// The indirect dispatch / draw argument buffer (workgroup or instance
    /// counts produced on-device).
    IndirectArgs,
    /// The per-emitter spawn count and atomic pool counters.
    SpawnCounter,
    /// The pool free-list slots handed out to newly spawned particles.
    FreeList,
    /// The compacted alive-index list the simulation and render walk.
    AliveList,
    /// The Structure-of-Arrays (`SoA`) particle attribute pool.
    ParticlePool,
    /// The event / `Niagara`-style data-channel ring buffer.
    EventRing,
    /// The reduced axis-aligned bounding box (`AABB`) buffer.
    BoundsBuffer,
    /// The post-cull visible-instance index list.
    VisibleList,
    /// The sort-key / sorted-index buffer.
    SortKeys,
    /// The indirect draw argument buffer (per-instance counts for the draw).
    DrawArgs,
    /// The previous-frame transform buffer feeding motion vectors (`MV`).
    PrevTransform,
    /// The color / depth render target the draw pass writes.
    RenderTarget,
}

impl PipelineResource {
    /// Returns the single-bit mask this resource occupies inside a
    /// [`ResourceSet`].
    #[must_use]
    pub const fn bit(self) -> u32 {
        let index = match self {
            PipelineResource::IndirectArgs => 0,
            PipelineResource::SpawnCounter => 1,
            PipelineResource::FreeList => 2,
            PipelineResource::AliveList => 3,
            PipelineResource::ParticlePool => 4,
            PipelineResource::EventRing => 5,
            PipelineResource::BoundsBuffer => 6,
            PipelineResource::VisibleList => 7,
            PipelineResource::SortKeys => 8,
            PipelineResource::DrawArgs => 9,
            PipelineResource::PrevTransform => 10,
            PipelineResource::RenderTarget => 11,
        };
        1u32 << index
    }
}

/// A small immutable set of [`PipelineResource`]s stored as a bitmask.
///
/// A bitmask keeps the barrier hazard test a single bitwise `AND`, which is both
/// deterministic and allocation-free.
#[derive(Clone, Copy, Debug, Default, Eq, Hash, PartialEq)]
pub struct ResourceSet {
    /// The packed resource bits (see [`PipelineResource::bit`]).
    pub bits: u32,
}

impl ResourceSet {
    /// The empty set.
    pub const EMPTY: Self = Self { bits: 0 };

    /// Builds a set containing exactly `resource`.
    #[must_use]
    pub const fn from_resource(resource: PipelineResource) -> Self {
        Self {
            bits: resource.bit(),
        }
    }

    /// Returns a copy of this set with `resource` added.
    #[must_use]
    pub const fn with(self, resource: PipelineResource) -> Self {
        Self {
            bits: self.bits | resource.bit(),
        }
    }

    /// Returns `true` when `resource` is a member.
    #[must_use]
    pub const fn contains(self, resource: PipelineResource) -> bool {
        (self.bits & resource.bit()) != 0
    }

    /// Returns the union of two sets.
    #[must_use]
    pub const fn union(self, other: Self) -> Self {
        Self {
            bits: self.bits | other.bits,
        }
    }

    /// Returns `true` when the two sets share at least one resource.
    #[must_use]
    pub const fn intersects(self, other: Self) -> bool {
        (self.bits & other.bits) != 0
    }

    /// Returns `true` when the set has no members.
    #[must_use]
    pub const fn is_empty(self) -> bool {
        self.bits == 0
    }
}

/// Whether a pass runs on the compute path or the graphics (draw) path.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum PassKind {
    /// A `GPU` compute dispatch (indirect or direct).
    Compute,
    /// A `GPU` indirect draw.
    Draw,
}

impl PassKind {
    /// Returns `true` when this is a compute pass.
    #[must_use]
    pub fn is_compute(self) -> bool {
        matches!(self, PassKind::Compute)
    }
}

/// The read and write resource sets a pass touches, used for barrier analysis.
#[derive(Clone, Copy, Debug, Default, Eq, Hash, PartialEq)]
pub struct PassAccess {
    /// Resources the pass reads.
    pub reads: ResourceSet,
    /// Resources the pass writes.
    pub writes: ResourceSet,
}

impl PassAccess {
    /// Builds an access record from explicit read and write sets.
    #[must_use]
    pub const fn new(reads: ResourceSet, writes: ResourceSet) -> Self {
        Self { reads, writes }
    }
}

/// One of the ten ordered per-frame passes an emitter walks (design §9).
///
/// The declaration order matches the pipeline order; [`FramePass::order_index`]
/// exposes it as a stable integer and [`FramePass::ALL`] lists every pass once
/// in that order.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum FramePass {
    /// Step 1 — one invocation per emitter computes the spawn count and writes
    /// the indirect dispatch arguments for the spawn pass.
    EmitterUpdate,
    /// Step 2 — dispatches `spawn_count` invocations that pull free-list slots
    /// and initialize new particles.
    Spawn,
    /// Step 3 — the ordered simulation stages (forces, integration, aging) in
    /// the §7 order; the in-domain solve lives in [`super::stages`].
    SimulationStages,
    /// Step 4 — scatters `GPU` events into the ring buffer / data channel.
    EventScatter,
    /// Step 5 — optional prefix-sum compaction that rebuilds the alive list when
    /// the dead-slot fragmentation ratio is high.
    Compaction,
    /// Step 6 — a `GPU` reduction that computes the emitter's axis-aligned
    /// bounding box (`AABB`).
    Bounds,
    /// Step 7 — frustum / distance / `HZB` culling that fills the visible-instance
    /// list.
    Cull,
    /// Step 8 — conditional depth sort of the visible list (only for translucent
    /// instances).
    Sort,
    /// Step 9 — writes the indirect draw instance count and the previous-frame
    /// transform used for motion vectors (`MV`).
    FillDrawArgs,
    /// Step 10 — the indirect draw that shades the visible instances.
    RenderDraw,
}

impl FramePass {
    /// Every pass once, in pipeline order.
    pub const ALL: [FramePass; 10] = [
        FramePass::EmitterUpdate,
        FramePass::Spawn,
        FramePass::SimulationStages,
        FramePass::EventScatter,
        FramePass::Compaction,
        FramePass::Bounds,
        FramePass::Cull,
        FramePass::Sort,
        FramePass::FillDrawArgs,
        FramePass::RenderDraw,
    ];

    /// Returns the pass's zero-based position in the pipeline order.
    #[must_use]
    pub fn order_index(self) -> u32 {
        match self {
            FramePass::EmitterUpdate => 0,
            FramePass::Spawn => 1,
            FramePass::SimulationStages => 2,
            FramePass::EventScatter => 3,
            FramePass::Compaction => 4,
            FramePass::Bounds => 5,
            FramePass::Cull => 6,
            FramePass::Sort => 7,
            FramePass::FillDrawArgs => 8,
            FramePass::RenderDraw => 9,
        }
    }

    /// Returns whether the pass is compute or draw.
    #[must_use]
    pub fn kind(self) -> PassKind {
        match self {
            FramePass::RenderDraw => PassKind::Draw,
            _ => PassKind::Compute,
        }
    }

    /// Returns `true` when the pass is only included under a runtime condition
    /// (`Compaction` when fragmentation is high, `Sort` when translucent
    /// instances are visible).
    #[must_use]
    pub fn is_conditional(self) -> bool {
        matches!(self, FramePass::Compaction | FramePass::Sort)
    }

    /// Returns `true` when the pass may overlap the *previous* frame's
    /// post-processing / shadow work on a dedicated asynchronous compute queue.
    ///
    /// The simulation front (`EmitterUpdate` through `Bounds`) has no dependency
    /// on the current frame's graphics and is the classic `Frostbite`-style
    /// async-compute overlap window. `Cull`, `Sort`, and `FillDrawArgs` sit on
    /// the critical path immediately before the draw, and the draw itself is a
    /// graphics-queue pass, so none of those overlap.
    #[must_use]
    pub fn overlaps_async_compute(self) -> bool {
        matches!(
            self,
            FramePass::EmitterUpdate
                | FramePass::Spawn
                | FramePass::SimulationStages
                | FramePass::EventScatter
                | FramePass::Compaction
                | FramePass::Bounds
        )
    }

    /// Returns the iteration domain a compute pass dispatches over, or `None`
    /// for the draw pass (which is an indirect draw, not a dispatch).
    #[must_use]
    pub fn iteration_domain(self) -> Option<IterationDomain> {
        match self {
            FramePass::EmitterUpdate | FramePass::FillDrawArgs => Some(IterationDomain::Custom(1)),
            FramePass::EventScatter => Some(IterationDomain::PerEvent),
            FramePass::SimulationStages
            | FramePass::Spawn
            | FramePass::Compaction
            | FramePass::Bounds
            | FramePass::Cull
            | FramePass::Sort => Some(IterationDomain::PerParticle),
            FramePass::RenderDraw => None,
        }
    }

    /// Returns the pass's default read / write resource sets.
    ///
    /// A real backend may refine these once shader bindings are known; the
    /// defaults encode the §9 data-flow and are what the barrier analysis reasons
    /// about.
    #[must_use]
    pub fn default_access(self) -> PassAccess {
        use PipelineResource as R;
        match self {
            FramePass::EmitterUpdate => PassAccess::new(
                ResourceSet::from_resource(R::SpawnCounter).with(R::AliveList),
                ResourceSet::from_resource(R::SpawnCounter).with(R::IndirectArgs),
            ),
            FramePass::Spawn => PassAccess::new(
                ResourceSet::from_resource(R::SpawnCounter)
                    .with(R::IndirectArgs)
                    .with(R::FreeList),
                ResourceSet::from_resource(R::FreeList)
                    .with(R::ParticlePool)
                    .with(R::AliveList)
                    .with(R::SpawnCounter),
            ),
            FramePass::SimulationStages => PassAccess::new(
                ResourceSet::from_resource(R::ParticlePool).with(R::AliveList),
                ResourceSet::from_resource(R::ParticlePool),
            ),
            FramePass::EventScatter => PassAccess::new(
                ResourceSet::from_resource(R::ParticlePool).with(R::AliveList),
                ResourceSet::from_resource(R::EventRing),
            ),
            FramePass::Compaction => PassAccess::new(
                ResourceSet::from_resource(R::AliveList).with(R::ParticlePool),
                ResourceSet::from_resource(R::AliveList)
                    .with(R::FreeList)
                    .with(R::SpawnCounter),
            ),
            FramePass::Bounds => PassAccess::new(
                ResourceSet::from_resource(R::ParticlePool).with(R::AliveList),
                ResourceSet::from_resource(R::BoundsBuffer),
            ),
            FramePass::Cull => PassAccess::new(
                ResourceSet::from_resource(R::AliveList)
                    .with(R::ParticlePool)
                    .with(R::BoundsBuffer),
                ResourceSet::from_resource(R::VisibleList),
            ),
            FramePass::Sort => PassAccess::new(
                ResourceSet::from_resource(R::VisibleList).with(R::ParticlePool),
                ResourceSet::from_resource(R::SortKeys).with(R::VisibleList),
            ),
            FramePass::FillDrawArgs => PassAccess::new(
                ResourceSet::from_resource(R::VisibleList).with(R::SortKeys),
                ResourceSet::from_resource(R::DrawArgs).with(R::PrevTransform),
            ),
            FramePass::RenderDraw => PassAccess::new(
                ResourceSet::from_resource(R::DrawArgs)
                    .with(R::VisibleList)
                    .with(R::ParticlePool)
                    .with(R::SortKeys)
                    .with(R::PrevTransform),
                ResourceSet::from_resource(R::RenderTarget),
            ),
        }
    }
}

/// One emitter's live per-frame counters — the demand side of the plan.
///
/// All fields are plain counts, so the state is exactly comparable; the derived
/// fragmentation ratio is computed on demand rather than stored.
#[derive(Clone, Copy, Debug, Default, Eq, Hash, PartialEq)]
pub struct EmitterFrameState {
    /// The number of particles to spawn this frame (drives step 2).
    pub spawn_count: u32,
    /// The number of particles alive at the start of the frame.
    pub alive_count: u32,
    /// Dead slots still present in the compacted alive list, awaiting
    /// reclamation (the numerator of the fragmentation ratio).
    pub stale_slots: u32,
    /// The emitter pool's fixed capacity.
    pub capacity: u32,
    /// The number of free pool slots available to the spawn pass.
    pub free_count: u32,
    /// Visible translucent instances after culling (drives the sort condition).
    pub visible_translucent_count: u32,
    /// Visible opaque instances after culling.
    pub visible_opaque_count: u32,
}

impl EmitterFrameState {
    /// The number of particles the simulation stages iterate this frame: those
    /// already alive plus those spawned this frame (saturating).
    #[must_use]
    pub fn simulated_count(self) -> u32 {
        self.alive_count.saturating_add(self.spawn_count)
    }

    /// The current length of the compacted alive list, including dead holes.
    #[must_use]
    pub fn list_slot_count(self) -> u32 {
        self.alive_count.saturating_add(self.stale_slots)
    }

    /// The total number of visible instances (opaque plus translucent).
    #[must_use]
    pub fn visible_count(self) -> u32 {
        self.visible_opaque_count
            .saturating_add(self.visible_translucent_count)
    }

    /// The dead-slot fragmentation ratio of the alive list, in `0.0..=1.0`.
    ///
    /// This is `stale_slots / (alive_count + stale_slots)`; an empty list yields
    /// `0.0` so the value is always defined without a bare-`==` guard downstream.
    #[must_use]
    pub fn fragmentation_ratio(self) -> f32 {
        let slots = self.list_slot_count();
        if slots == 0 {
            return 0.0;
        }
        (self.stale_slots as f32) / (slots as f32)
    }
}

/// Why the optional `Compaction` pass was or was not scheduled.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum CompactionReason {
    /// The alive list is empty, so there is nothing to compact.
    EmptyList,
    /// The fragmentation ratio is below the threshold; skipping is cheaper.
    BelowThreshold,
    /// The fragmentation ratio meets or exceeds the threshold; a prefix-sum
    /// rebuild of the alive list pays for itself.
    FragmentationHigh,
}

/// The decision for §9 step 5: whether to run the compaction pass, with the
/// ratio that drove it.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CompactionDecision {
    /// Whether the compaction pass is scheduled this frame.
    pub execute: bool,
    /// The reason behind the decision.
    pub reason: CompactionReason,
    /// The dead-slot fragmentation ratio that was evaluated.
    pub fragmentation_ratio: f32,
}

impl CompactionDecision {
    /// Decides whether to run compaction for `state` against `threshold`.
    ///
    /// An empty alive list never compacts; otherwise the pass runs exactly when
    /// the fragmentation ratio meets or exceeds `threshold`.
    #[must_use]
    pub fn decide(state: EmitterFrameState, threshold: f32) -> Self {
        let ratio = state.fragmentation_ratio();
        if state.list_slot_count() == 0 {
            return Self {
                execute: false,
                reason: CompactionReason::EmptyList,
                fragmentation_ratio: ratio,
            };
        }
        if ratio >= threshold {
            Self {
                execute: true,
                reason: CompactionReason::FragmentationHigh,
                fragmentation_ratio: ratio,
            }
        } else {
            Self {
                execute: false,
                reason: CompactionReason::BelowThreshold,
                fragmentation_ratio: ratio,
            }
        }
    }

    /// Decides using [`DEFAULT_COMPACTION_FRAGMENTATION_THRESHOLD`].
    #[must_use]
    pub fn decide_default(state: EmitterFrameState) -> Self {
        Self::decide(state, DEFAULT_COMPACTION_FRAGMENTATION_THRESHOLD)
    }
}

/// Why the conditional `Sort` pass was or was not scheduled.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum SortReason {
    /// No translucent instances are visible, so no depth sort is needed.
    NoTranslucent,
    /// Translucent instances are visible and must be drawn back-to-front.
    TranslucentVisible,
}

/// The decision for §9 step 8: whether to depth-sort the visible list.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct SortDecision {
    /// Whether the sort pass is scheduled this frame.
    pub execute: bool,
    /// The reason behind the decision.
    pub reason: SortReason,
    /// The number of visible translucent instances evaluated.
    pub translucent_count: u32,
}

impl SortDecision {
    /// Decides whether to sort for `state`: the pass runs exactly when at least
    /// one translucent instance is visible.
    #[must_use]
    pub fn decide(state: EmitterFrameState) -> Self {
        if state.visible_translucent_count > 0 {
            Self {
                execute: true,
                reason: SortReason::TranslucentVisible,
                translucent_count: state.visible_translucent_count,
            }
        } else {
            Self {
                execute: false,
                reason: SortReason::NoTranslucent,
                translucent_count: 0,
            }
        }
    }
}

/// The number of grid-stride iterations each thread of a persistent-thread
/// kernel performs to cover `element_count` elements.
///
/// A persistent-thread kernel launches a fixed number of workgroups whose total
/// thread count is `workgroup_count * workgroup_size`; each thread then loops in
/// strides over the element array. The iteration count is
/// `ceil(element_count / total_threads)`, guarded against a zero thread count.
#[must_use]
pub fn grid_stride_iterations(
    workgroup_count: u32,
    workgroup_size: u32,
    element_count: u32,
) -> u32 {
    let total_threads = workgroup_count.saturating_mul(workgroup_size);
    if total_threads == 0 {
        return 0;
    }
    element_count.div_ceil(total_threads)
}

/// Derives the per-pass element count for `pass` from the emitter `state`.
///
/// Returns `None` for the draw pass, whose instance count is produced on-device
/// into the indirect draw arguments rather than dispatched by count.
#[must_use]
pub fn pass_element_count(pass: FramePass, state: EmitterFrameState) -> Option<u32> {
    match pass {
        FramePass::EmitterUpdate | FramePass::FillDrawArgs => Some(1),
        FramePass::Spawn => Some(state.spawn_count),
        FramePass::SimulationStages
        | FramePass::EventScatter
        | FramePass::Bounds
        | FramePass::Cull => Some(state.simulated_count()),
        FramePass::Compaction => Some(state.list_slot_count()),
        FramePass::Sort => Some(state.visible_translucent_count),
        FramePass::RenderDraw => None,
    }
}

/// Derives the indirect dispatch plan for a compute `pass`.
///
/// The dispatch arithmetic is delegated wholesale to
/// [`GpuDispatchPlan::derive`]: this function only picks the element count and
/// iteration domain for the pass. Returns `None` for the draw pass.
#[must_use]
pub fn derive_pass_dispatch(
    pass: FramePass,
    state: EmitterFrameState,
    caps: BackendCapabilities,
    tuning: BackendTuning,
) -> Option<GpuDispatchPlan> {
    let domain = pass.iteration_domain()?;
    let element_count = pass_element_count(pass, state)?;
    Some(GpuDispatchPlan::derive(caps, tuning, domain, element_count))
}

/// Computes the minimal set of barrier insertion points for a pass sequence.
///
/// A barrier must precede pass `i` when any resource written by an earlier pass
/// since the last barrier is read by pass `i` (read-after-write, "写后读") or
/// written by pass `i` (write-after-write, "写后写"). Pure read-after-read and
/// write-after-read pairs need no barrier, so independent adjacent compute
/// passes are implicitly merged into one barrier-free run.
///
/// Pending writes are accumulated across passes and flushed at each inserted
/// barrier, so a hazard separated by an unrelated pass is still caught. The
/// returned indices are strictly increasing and one-based against the pass that
/// the barrier precedes (index `0` never appears — nothing precedes the first
/// pass).
#[must_use]
pub fn minimal_barrier_points(accesses: &[PassAccess]) -> Vec<usize> {
    let mut points = Vec::new();
    let mut pending_writes = ResourceSet::EMPTY;
    for (index, access) in accesses.iter().enumerate() {
        if index == 0 {
            pending_writes = access.writes;
            continue;
        }
        let touched = access.reads.union(access.writes);
        if pending_writes.intersects(touched) {
            points.push(index);
            // The barrier flushes all prior writes; only this pass's writes
            // remain outstanding afterward.
            pending_writes = access.writes;
        } else {
            pending_writes = pending_writes.union(access.writes);
        }
    }
    points
}

/// A run of consecutive compute passes with no barrier between them, which a
/// backend may fuse into a single command list segment.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct MergeRun {
    /// The index of the first pass in the run.
    pub start_index: u32,
    /// The number of passes in the run.
    pub pass_count: u32,
}

/// Groups a pass sequence into runs of adjacent compute passes that share no
/// barrier, so same-kind neighbors are coalesced.
///
/// A run is broken by a required barrier, by a draw pass, or by the end of the
/// sequence. Draw passes are never placed in a compute run; each is reported as
/// its own single-pass run so the returned runs tile the whole sequence.
#[must_use]
pub fn coalesced_compute_runs(kinds: &[PassKind], barrier_points: &[usize]) -> Vec<MergeRun> {
    let mut runs = Vec::new();
    let mut cursor = 0usize;
    while cursor < kinds.len() {
        let start = cursor;
        if kinds[start].is_compute() {
            cursor += 1;
            while cursor < kinds.len()
                && kinds[cursor].is_compute()
                && !barrier_points.contains(&cursor)
            {
                cursor += 1;
            }
        } else {
            // A draw pass stands alone.
            cursor += 1;
        }
        runs.push(MergeRun {
            start_index: start as u32,
            pass_count: (cursor - start) as u32,
        });
    }
    runs
}

/// One scheduled pass in a built frame plan.
///
/// Draw passes carry no [`GpuDispatchPlan`]; their instance count is produced
/// on-device into the indirect draw arguments.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct FrameStep {
    /// Which pass this step runs.
    pub pass: FramePass,
    /// The pass's read / write resource sets (used for barrier analysis).
    pub access: PassAccess,
    /// The compute dispatch plan, or `None` for the draw pass.
    pub dispatch: Option<GpuDispatchPlan>,
    /// Whether a pipeline barrier must be inserted immediately before this step.
    pub needs_barrier_before: bool,
    /// Whether this step may overlap the previous frame's graphics on a
    /// dedicated asynchronous compute queue (only set when the device exposes
    /// one).
    pub async_overlap: bool,
}

/// A fully built per-frame pipeline plan for one emitter (design §9).
///
/// The `steps` are in pipeline order with the conditional passes already
/// resolved; the compaction and sort decisions are retained for observability,
/// and `barrier_count` is the number of `needs_barrier_before` steps.
#[derive(Clone, Debug, PartialEq)]
pub struct FramePlan {
    /// The ordered, conditionally-included steps.
    pub steps: Vec<FrameStep>,
    /// The step-5 compaction decision.
    pub compaction: CompactionDecision,
    /// The step-8 sort decision.
    pub sort: SortDecision,
    /// The number of pipeline barriers the plan inserts.
    pub barrier_count: u32,
}

impl FramePlan {
    /// Returns the number of scheduled steps.
    #[must_use]
    pub fn len(&self) -> usize {
        self.steps.len()
    }

    /// Returns `true` when the plan schedules no steps.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.steps.is_empty()
    }

    /// Returns `true` when a step for `pass` is present in the plan.
    #[must_use]
    pub fn contains(&self, pass: FramePass) -> bool {
        self.steps.iter().any(|step| step.pass == pass)
    }
}

/// Builds the ordered, conditional per-frame pipeline plan for one emitter
/// (design §9).
///
/// The ten passes are walked in order; the two conditional passes are included
/// only when their decision says so (`Compaction` on high fragmentation against
/// `compaction_threshold`, `Sort` on visible translucent instances). Each
/// compute step gets an indirect dispatch plan from
/// [`derive_pass_dispatch`], barriers are placed by [`minimal_barrier_points`],
/// and the asynchronous-compute overlap flag is set only where the pass allows
/// it and the device exposes a dedicated compute queue.
#[must_use]
pub fn build_frame_plan(
    state: EmitterFrameState,
    caps: BackendCapabilities,
    tuning: BackendTuning,
    compaction_threshold: f32,
) -> FramePlan {
    let compaction = CompactionDecision::decide(state, compaction_threshold);
    let sort = SortDecision::decide(state);

    // Gather the included passes in order.
    let mut passes = Vec::new();
    for &pass in FramePass::ALL.iter() {
        let include = match pass {
            FramePass::Compaction => compaction.execute,
            FramePass::Sort => sort.execute,
            _ => true,
        };
        if include {
            passes.push(pass);
        }
    }

    // Barrier analysis over the included accesses.
    let accesses: Vec<PassAccess> = passes.iter().map(|&pass| pass.default_access()).collect();
    let barrier_points = minimal_barrier_points(&accesses);

    let async_available = caps.has_dedicated_compute_queue;
    let mut steps = Vec::with_capacity(passes.len());
    for (index, &pass) in passes.iter().enumerate() {
        let async_overlap =
            async_available && pass.kind().is_compute() && pass.overlaps_async_compute();
        steps.push(FrameStep {
            pass,
            access: accesses[index],
            dispatch: derive_pass_dispatch(pass, state, caps, tuning),
            needs_barrier_before: barrier_points.contains(&index),
            async_overlap,
        });
    }

    FramePlan {
        steps,
        compaction,
        sort,
        barrier_count: barrier_points.len() as u32,
    }
}

/// Builds a frame plan using [`DEFAULT_COMPACTION_FRAGMENTATION_THRESHOLD`].
#[must_use]
pub fn build_frame_plan_default(
    state: EmitterFrameState,
    caps: BackendCapabilities,
    tuning: BackendTuning,
) -> FramePlan {
    build_frame_plan(
        state,
        caps,
        tuning,
        DEFAULT_COMPACTION_FRAGMENTATION_THRESHOLD,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::particle::dual_backend::BackendCapabilities;

    fn desktop() -> BackendCapabilities {
        BackendCapabilities::desktop_gpu()
    }

    fn base_state() -> EmitterFrameState {
        EmitterFrameState {
            spawn_count: 128,
            alive_count: 1_000,
            stale_slots: 0,
            capacity: 4_096,
            free_count: 3_096,
            visible_translucent_count: 0,
            visible_opaque_count: 500,
        }
    }

    #[test]
    fn resource_set_membership_and_ops() {
        let set = ResourceSet::from_resource(PipelineResource::AliveList)
            .with(PipelineResource::ParticlePool);
        assert!(set.contains(PipelineResource::AliveList));
        assert!(set.contains(PipelineResource::ParticlePool));
        assert!(!set.contains(PipelineResource::SortKeys));
        assert!(!set.is_empty());
        assert!(ResourceSet::EMPTY.is_empty());

        let other = ResourceSet::from_resource(PipelineResource::ParticlePool);
        assert!(set.intersects(other));
        let disjoint = ResourceSet::from_resource(PipelineResource::SortKeys);
        assert!(!set.intersects(disjoint));
        assert_eq!(
            set.union(disjoint).bits,
            set.bits | PipelineResource::SortKeys.bit()
        );
    }

    #[test]
    fn all_passes_are_ordered_and_unique() {
        assert_eq!(FramePass::ALL.len(), 10);
        for (index, &pass) in FramePass::ALL.iter().enumerate() {
            assert_eq!(pass.order_index() as usize, index);
        }
        // Every order index appears exactly once.
        for i in 0..FramePass::ALL.len() {
            let count = FramePass::ALL
                .iter()
                .filter(|p| p.order_index() as usize == i)
                .count();
            assert_eq!(count, 1);
        }
    }

    #[test]
    fn only_render_draw_is_a_draw_pass() {
        for &pass in FramePass::ALL.iter() {
            match pass {
                FramePass::RenderDraw => assert_eq!(pass.kind(), PassKind::Draw),
                _ => assert_eq!(pass.kind(), PassKind::Compute),
            }
        }
    }

    #[test]
    fn conditional_passes_are_compaction_and_sort() {
        for &pass in FramePass::ALL.iter() {
            let expected = matches!(pass, FramePass::Compaction | FramePass::Sort);
            assert_eq!(pass.is_conditional(), expected);
        }
    }

    #[test]
    fn render_draw_has_no_iteration_domain() {
        assert_eq!(FramePass::RenderDraw.iteration_domain(), None);
        for &pass in FramePass::ALL.iter() {
            if pass != FramePass::RenderDraw {
                assert!(pass.iteration_domain().is_some());
            }
        }
    }

    #[test]
    fn async_overlap_covers_simulation_front_only() {
        let overlapping = [
            FramePass::EmitterUpdate,
            FramePass::Spawn,
            FramePass::SimulationStages,
            FramePass::EventScatter,
            FramePass::Compaction,
            FramePass::Bounds,
        ];
        for &pass in FramePass::ALL.iter() {
            let expected = overlapping.contains(&pass);
            assert_eq!(pass.overlaps_async_compute(), expected);
        }
    }

    #[test]
    fn fragmentation_ratio_is_defined_and_bounded() {
        let empty = EmitterFrameState::default();
        assert!((empty.fragmentation_ratio() - 0.0).abs() < CMP_EPS);

        let half = EmitterFrameState {
            alive_count: 100,
            stale_slots: 100,
            ..EmitterFrameState::default()
        };
        assert!((half.fragmentation_ratio() - 0.5).abs() < CMP_EPS);

        let heavy = EmitterFrameState {
            alive_count: 25,
            stale_slots: 75,
            ..EmitterFrameState::default()
        };
        assert!((heavy.fragmentation_ratio() - 0.75).abs() < CMP_EPS);
    }

    #[test]
    fn compaction_skipped_on_empty_list() {
        let state = EmitterFrameState::default();
        let decision = CompactionDecision::decide_default(state);
        assert!(!decision.execute);
        assert_eq!(decision.reason, CompactionReason::EmptyList);
    }

    #[test]
    fn compaction_skipped_below_threshold() {
        let state = EmitterFrameState {
            alive_count: 900,
            stale_slots: 100,
            ..EmitterFrameState::default()
        };
        let decision = CompactionDecision::decide_default(state);
        assert!(!decision.execute);
        assert_eq!(decision.reason, CompactionReason::BelowThreshold);
        assert!((decision.fragmentation_ratio - 0.1).abs() < CMP_EPS);
    }

    #[test]
    fn compaction_runs_above_threshold() {
        let state = EmitterFrameState {
            alive_count: 400,
            stale_slots: 600,
            ..EmitterFrameState::default()
        };
        let decision = CompactionDecision::decide_default(state);
        assert!(decision.execute);
        assert_eq!(decision.reason, CompactionReason::FragmentationHigh);
        assert!((decision.fragmentation_ratio - 0.6).abs() < CMP_EPS);
    }

    #[test]
    fn compaction_runs_exactly_at_threshold() {
        let state = EmitterFrameState {
            alive_count: 100,
            stale_slots: 100,
            ..EmitterFrameState::default()
        };
        // Ratio is exactly 0.5, the default threshold, so it runs.
        let decision = CompactionDecision::decide_default(state);
        assert!(decision.execute);
    }

    #[test]
    fn sort_decision_tracks_translucent_visibility() {
        let none = EmitterFrameState {
            visible_translucent_count: 0,
            ..base_state()
        };
        let d0 = SortDecision::decide(none);
        assert!(!d0.execute);
        assert_eq!(d0.reason, SortReason::NoTranslucent);

        let some = EmitterFrameState {
            visible_translucent_count: 42,
            ..base_state()
        };
        let d1 = SortDecision::decide(some);
        assert!(d1.execute);
        assert_eq!(d1.reason, SortReason::TranslucentVisible);
        assert_eq!(d1.translucent_count, 42);
    }

    #[test]
    fn grid_stride_iterations_round_up_and_guard() {
        // 10 000 elements over 4 workgroups of 64 threads = 256 threads.
        assert_eq!(grid_stride_iterations(4, 64, 10_000), 40);
        // Exact multiple.
        assert_eq!(grid_stride_iterations(4, 64, 256), 1);
        // One extra element rolls to a second iteration.
        assert_eq!(grid_stride_iterations(4, 64, 257), 2);
        // Zero threads is guarded.
        assert_eq!(grid_stride_iterations(0, 64, 100), 0);
        assert_eq!(grid_stride_iterations(4, 0, 100), 0);
        // No work.
        assert_eq!(grid_stride_iterations(4, 64, 0), 0);
    }

    #[test]
    fn pass_element_counts_follow_state() {
        let state = base_state();
        assert_eq!(pass_element_count(FramePass::EmitterUpdate, state), Some(1));
        assert_eq!(pass_element_count(FramePass::Spawn, state), Some(128));
        assert_eq!(
            pass_element_count(FramePass::SimulationStages, state),
            Some(1_128)
        );
        assert_eq!(pass_element_count(FramePass::Bounds, state), Some(1_128));
        assert_eq!(pass_element_count(FramePass::Sort, state), Some(0));
        assert_eq!(pass_element_count(FramePass::FillDrawArgs, state), Some(1));
        assert_eq!(pass_element_count(FramePass::RenderDraw, state), None);
    }

    #[test]
    fn emitter_update_dispatch_is_single_thread() {
        let plan = derive_pass_dispatch(
            FramePass::EmitterUpdate,
            base_state(),
            desktop(),
            BackendTuning::default(),
        )
        .expect("compute pass has a dispatch");
        assert_eq!(plan.element_count, 1);
        assert_eq!(plan.workgroup_count, 1);
        assert!(plan.covers_all_elements());
    }

    #[test]
    fn draw_pass_has_no_dispatch() {
        let plan = derive_pass_dispatch(
            FramePass::RenderDraw,
            base_state(),
            desktop(),
            BackendTuning::default(),
        );
        assert!(plan.is_none());
    }

    #[test]
    fn simulation_dispatch_splits_when_limit_is_small() {
        // A tiny per-dispatch workgroup limit forces a multi-dispatch split.
        let caps = BackendCapabilities {
            max_workgroups_per_dispatch: 2,
            ..BackendCapabilities::desktop_gpu()
        };
        let tuning = BackendTuning {
            gpu_workgroup_size: 64,
            gpu_persistent_threads: false,
            ..BackendTuning::default()
        };
        let state = EmitterFrameState {
            spawn_count: 0,
            alive_count: 64 * 5, // 5 workgroups of 64.
            ..EmitterFrameState::default()
        };
        let plan = derive_pass_dispatch(FramePass::SimulationStages, state, caps, tuning)
            .expect("compute pass has a dispatch");
        assert_eq!(plan.workgroup_count, 5);
        assert_eq!(plan.workgroups_per_dispatch, 2);
        assert_eq!(plan.dispatch_count, 3);
        assert!(plan.covers_all_elements());
    }

    #[test]
    fn barrier_detects_read_after_write() {
        let write_pool = PassAccess::new(
            ResourceSet::EMPTY,
            ResourceSet::from_resource(PipelineResource::ParticlePool),
        );
        let read_pool = PassAccess::new(
            ResourceSet::from_resource(PipelineResource::ParticlePool),
            ResourceSet::EMPTY,
        );
        let points = minimal_barrier_points(&[write_pool, read_pool]);
        assert_eq!(points, [1]);
    }

    #[test]
    fn barrier_detects_write_after_write() {
        let write_a = PassAccess::new(
            ResourceSet::EMPTY,
            ResourceSet::from_resource(PipelineResource::AliveList),
        );
        let write_b = PassAccess::new(
            ResourceSet::EMPTY,
            ResourceSet::from_resource(PipelineResource::AliveList),
        );
        let points = minimal_barrier_points(&[write_a, write_b]);
        assert_eq!(points, [1]);
    }

    #[test]
    fn barrier_ignores_read_after_read_and_write_after_read() {
        // Two reads of the same buffer: no barrier.
        let read_a = PassAccess::new(
            ResourceSet::from_resource(PipelineResource::ParticlePool),
            ResourceSet::EMPTY,
        );
        let read_b = read_a;
        assert!(minimal_barrier_points(&[read_a, read_b]).is_empty());

        // Read then a disjoint write: write-after-read on different resources,
        // no hazard.
        let read_pool = PassAccess::new(
            ResourceSet::from_resource(PipelineResource::ParticlePool),
            ResourceSet::EMPTY,
        );
        let write_other = PassAccess::new(
            ResourceSet::EMPTY,
            ResourceSet::from_resource(PipelineResource::SortKeys),
        );
        assert!(minimal_barrier_points(&[read_pool, write_other]).is_empty());
    }

    #[test]
    fn barrier_catches_gapped_hazard_across_unrelated_pass() {
        let write_pool = PassAccess::new(
            ResourceSet::EMPTY,
            ResourceSet::from_resource(PipelineResource::ParticlePool),
        );
        let unrelated = PassAccess::new(
            ResourceSet::from_resource(PipelineResource::SortKeys),
            ResourceSet::from_resource(PipelineResource::VisibleList),
        );
        let read_pool = PassAccess::new(
            ResourceSet::from_resource(PipelineResource::ParticlePool),
            ResourceSet::EMPTY,
        );
        // The write is at 0, the dependent read at 2; the barrier lands before 2.
        let points = minimal_barrier_points(&[write_pool, unrelated, read_pool]);
        assert_eq!(points, [2]);
    }

    #[test]
    fn coalesced_runs_merge_barrier_free_compute_and_isolate_draw() {
        let kinds = [
            PassKind::Compute,
            PassKind::Compute,
            PassKind::Compute,
            PassKind::Draw,
        ];
        // A barrier before index 2 splits the first run.
        let runs = coalesced_compute_runs(&kinds, &[2]);
        assert_eq!(
            runs,
            [
                MergeRun {
                    start_index: 0,
                    pass_count: 2
                },
                MergeRun {
                    start_index: 2,
                    pass_count: 1
                },
                MergeRun {
                    start_index: 3,
                    pass_count: 1
                },
            ]
        );
    }

    #[test]
    fn frame_plan_orders_passes_and_ends_with_draw() {
        let plan = build_frame_plan_default(base_state(), desktop(), BackendTuning::default());
        assert!(!plan.is_empty());
        // Steps are in strictly increasing pipeline order.
        for window in plan.steps.windows(2) {
            assert!(window[0].pass.order_index() < window[1].pass.order_index());
        }
        // The final step is always the draw.
        assert_eq!(
            plan.steps.last().map(|s| s.pass),
            Some(FramePass::RenderDraw)
        );
    }

    #[test]
    fn frame_plan_excludes_compaction_and_sort_when_not_needed() {
        // Low fragmentation, no translucent instances.
        let state = EmitterFrameState {
            spawn_count: 64,
            alive_count: 1_000,
            stale_slots: 10,
            capacity: 4_096,
            free_count: 3_086,
            visible_translucent_count: 0,
            visible_opaque_count: 800,
        };
        let plan = build_frame_plan_default(state, desktop(), BackendTuning::default());
        assert!(!plan.contains(FramePass::Compaction));
        assert!(!plan.contains(FramePass::Sort));
        // The eight unconditional passes remain.
        assert_eq!(plan.len(), 8);
    }

    #[test]
    fn frame_plan_includes_compaction_and_sort_when_needed() {
        let state = EmitterFrameState {
            spawn_count: 64,
            alive_count: 300,
            stale_slots: 700, // 70% fragmentation.
            capacity: 4_096,
            free_count: 3_096,
            visible_translucent_count: 250,
            visible_opaque_count: 50,
        };
        let plan = build_frame_plan_default(state, desktop(), BackendTuning::default());
        assert!(plan.contains(FramePass::Compaction));
        assert!(plan.contains(FramePass::Sort));
        assert_eq!(plan.len(), 10);
        assert!(plan.compaction.execute);
        assert!(plan.sort.execute);
    }

    #[test]
    fn frame_plan_inserts_at_least_one_barrier_and_counts_match() {
        let plan = build_frame_plan_default(base_state(), desktop(), BackendTuning::default());
        let counted = plan
            .steps
            .iter()
            .filter(|step| step.needs_barrier_before)
            .count() as u32;
        assert_eq!(counted, plan.barrier_count);
        assert!(plan.barrier_count > 0);
    }

    #[test]
    fn async_overlap_requires_a_dedicated_compute_queue() {
        // Desktop has a dedicated queue: the simulation front overlaps.
        let with_queue =
            build_frame_plan_default(base_state(), desktop(), BackendTuning::default());
        assert!(with_queue.steps.iter().any(|s| s.async_overlap));

        // Without a dedicated queue, no step overlaps.
        let caps = BackendCapabilities {
            has_dedicated_compute_queue: false,
            ..BackendCapabilities::desktop_gpu()
        };
        let no_queue = build_frame_plan_default(base_state(), caps, BackendTuning::default());
        assert!(no_queue.steps.iter().all(|s| !s.async_overlap));
    }

    #[test]
    fn draw_step_never_overlaps_and_has_no_dispatch() {
        let plan = build_frame_plan_default(base_state(), desktop(), BackendTuning::default());
        let draw = plan
            .steps
            .iter()
            .find(|s| s.pass == FramePass::RenderDraw)
            .expect("draw step present");
        assert!(!draw.async_overlap);
        assert!(draw.dispatch.is_none());
    }
}
