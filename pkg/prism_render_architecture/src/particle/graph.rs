//! Node-graph to `IR` compilation contract (design §6) and the `Simulation`
//! `Stage` scheduling analysis (design §7).
//!
//! An authored particle graph (`.ember`) compiles through the pipeline
//!
//! ```text
//! Graph -> semantic validation -> IR (SSA-like) -> optimization -> WESL codegen -> specialization key
//! ```
//!
//! Ember follows the same "graph to shader codegen" philosophy as Unity
//! `VFX Graph` (graph to `HLSL`) and Unreal `Niagara`, but targets a
//! `Graph -> IR -> WESL` kernel per stage instead of a runtime `VM`. The `WESL`
//! codegen itself needs the `GPU` backend and is out of scope for this
//! `CPU`-verifiable contract layer. What this module owns is the deterministic
//! *analysis* the codegen and the `pipeline_cache` consume:
//!
//! * [`AttributeDemand`] — the per-stage read/write set, unioned across stages;
//! * [`validate`] — type/read-write consistency, dependency-`DAG` cycle
//!   detection, and shading-model compatibility, reported as [`CompileError`];
//! * [`GraphIr`]/[`StageIr`] — the intermediate representation and its stage
//!   ordering ([`schedule_order`]);
//! * attribute liveness so unused buffers are never allocated
//!   ([`prunable_stages`], [`plan_layout`]), ping-pong decisions
//!   ([`ping_pong_semantics`]), and dispatch unrolling ([`total_dispatches`]);
//! * barrier minimization ([`plan_barriers`]) so adjacent, same-domain,
//!   non-conflicting stages need no barrier; and
//! * a deterministic [`SpecializationKey`] for the `pipeline_cache`.

use alloc::string::String;
use alloc::vec::Vec;

use super::attributes::{
    shading_input_attributes, AttributeAccess, AttributeFormat, AttributeLayoutPlan,
    AttributeSemantic, AttributeUsage,
};
use super::{EmberShadingModel, IterationDomain, ShadingBasis};

/// The 64-bit `FNV`-1a offset basis, used to seed the deterministic content
/// hashes that back the [`SpecializationKey`].
const FNV_OFFSET_BASIS: u64 = 0xcbf2_9ce4_8422_2325;
/// The 64-bit `FNV`-1a prime.
const FNV_PRIME: u64 = 0x0000_0100_0000_01b3;

/// Folds one 64-bit value into an `FNV`-1a accumulator, little-endian byte by
/// byte. `wrapping_mul` keeps the hash total across the 64-bit ring without an
/// overflow panic.
#[must_use]
fn fnv_u64(mut hash: u64, value: u64) -> u64 {
    for byte in value.to_le_bytes() {
        hash ^= u64::from(byte);
        hash = hash.wrapping_mul(FNV_PRIME);
    }
    hash
}

/// The aggregated attribute demand a compiled graph (or a single stage) places
/// on the pool.
///
/// The compiler unions every stage's per-attribute access so the layout planner
/// allocates exactly the buffers that are touched (design §5.1, §6). Insertion
/// order is preserved for first-touch determinism; the layout planner re-sorts
/// by ordinal, so the final plan is insertion-order independent.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct AttributeDemand {
    usages: Vec<AttributeUsage>,
}

impl AttributeDemand {
    /// An empty demand.
    #[must_use]
    pub fn new() -> Self {
        Self { usages: Vec::new() }
    }

    /// Records that a stage touches `usage.semantic`, unioning the access with
    /// any previously recorded access for that semantic.
    pub fn touch(&mut self, usage: AttributeUsage) {
        if let Some(existing) = self
            .usages
            .iter_mut()
            .find(|u| u.semantic == usage.semantic)
        {
            existing.access = existing.access.union(usage.access);
        } else {
            self.usages.push(usage);
        }
    }

    /// Builder form of [`AttributeDemand::touch`].
    #[must_use]
    pub fn with(mut self, usage: AttributeUsage) -> Self {
        self.touch(usage);
        self
    }

    /// Unions every usage of `other` into `self`.
    pub fn union_with(&mut self, other: &Self) {
        for usage in &other.usages {
            self.touch(*usage);
        }
    }

    /// The merged usages, in first-touch order.
    #[must_use]
    pub fn usages(&self) -> &[AttributeUsage] {
        &self.usages
    }

    /// Whether a semantic is demanded with a non-empty access.
    #[must_use]
    pub fn demands(&self, semantic: AttributeSemantic) -> bool {
        self.usages
            .iter()
            .any(|u| u.semantic == semantic && !u.access.is_empty())
    }

    /// Whether nothing is demanded (no non-empty access recorded).
    #[must_use]
    pub fn is_empty(&self) -> bool {
        !self.usages.iter().any(|u| !u.access.is_empty())
    }

    /// The set of semantics touched with a non-empty access, in first-touch
    /// order.
    #[must_use]
    pub fn semantics(&self) -> Vec<AttributeSemantic> {
        let mut out = Vec::new();
        for usage in &self.usages {
            if !usage.access.is_empty() {
                out.push(usage.semantic);
            }
        }
        out
    }

    /// Whether the two demands touch at least one common semantic.
    #[must_use]
    pub fn intersects(&self, other: &Self) -> bool {
        self.usages
            .iter()
            .any(|u| !u.access.is_empty() && other.demands(u.semantic))
    }
}

/// Builds an [`AttributeUsage`] for `semantic` with the given access, choosing
/// the semantic's natural format (falling back to [`AttributeFormat::F32`] for
/// a `Custom` channel that has no intrinsic format).
#[must_use]
fn usage_with(semantic: AttributeSemantic, access: AttributeAccess) -> AttributeUsage {
    let format = semantic.default_format().unwrap_or(AttributeFormat::F32);
    AttributeUsage::new(semantic, format, access)
}

/// A current-frame read usage of `semantic` with its default format.
#[must_use]
pub fn read(semantic: AttributeSemantic) -> AttributeUsage {
    usage_with(semantic, AttributeAccess::READ)
}

/// A write usage of `semantic` with its default format.
#[must_use]
pub fn write(semantic: AttributeSemantic) -> AttributeUsage {
    usage_with(semantic, AttributeAccess::WRITE)
}

/// A previous-frame read usage of `semantic`; combined with a write elsewhere
/// this forces a ping-pong buffer.
#[must_use]
pub fn read_prev(semantic: AttributeSemantic) -> AttributeUsage {
    usage_with(semantic, AttributeAccess::READ_PREV)
}

/// A read-write usage of `semantic` with its default format.
#[must_use]
pub fn read_write(semantic: AttributeSemantic) -> AttributeUsage {
    usage_with(
        semantic,
        AttributeAccess::READ.union(AttributeAccess::WRITE),
    )
}

/// One stage of the compiled intermediate representation (design §7).
///
/// A stage declares its iteration domain, how many solver iterations it unrolls
/// into dispatches, its read/write attribute sets, and the indices of the
/// stages it depends on (edges of the scheduling `DAG`).
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct StageIr {
    /// A human-readable stage name (`spawn`, `integrate`, `pressure_solve`).
    pub name: String,
    /// The dispatch domain the scheduler sizes this stage against.
    pub domain: IterationDomain,
    /// The iteration count unrolled into that many dispatches (design §7,
    /// Jacobi pressure / `XPBD`); must be at least `1`.
    pub iterations: u32,
    /// Attributes this stage reads (current frame or, via `READ_PREV`, the
    /// previous frame).
    pub reads: AttributeDemand,
    /// Attributes this stage writes.
    pub writes: AttributeDemand,
    /// Indices of the stages that must run before this one.
    pub deps: Vec<usize>,
}

impl StageIr {
    /// A stage with empty read/write sets and no dependencies.
    #[must_use]
    pub fn new(name: impl Into<String>, domain: IterationDomain, iterations: u32) -> Self {
        Self {
            name: name.into(),
            domain,
            iterations,
            reads: AttributeDemand::new(),
            writes: AttributeDemand::new(),
            deps: Vec::new(),
        }
    }

    /// Builder: records a read usage.
    #[must_use]
    pub fn reading(mut self, usage: AttributeUsage) -> Self {
        self.reads.touch(usage);
        self
    }

    /// Builder: records a write usage.
    #[must_use]
    pub fn writing(mut self, usage: AttributeUsage) -> Self {
        self.writes.touch(usage);
        self
    }

    /// Builder: records a dependency on the stage at `stage`.
    #[must_use]
    pub fn depends_on(mut self, stage: usize) -> Self {
        self.deps.push(stage);
        self
    }

    /// The union of this stage's read and write demands.
    #[must_use]
    pub fn demand(&self) -> AttributeDemand {
        let mut demand = self.reads.clone();
        demand.union_with(&self.writes);
        demand
    }
}

/// The target platform profile that participates in the [`SpecializationKey`]
/// (design §6, §28). Distinct profiles compile distinct kernels.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum PlatformProfile {
    /// A high-end desktop / workstation profile.
    Desktop,
    /// A mobile / integrated-`GPU` profile.
    Mobile,
    /// A fixed console profile.
    Console,
    /// A `WebGPU` browser profile.
    Web,
}

impl PlatformProfile {
    /// A stable numeric code for hashing and round-tripping.
    #[must_use]
    pub const fn code(self) -> u32 {
        match self {
            PlatformProfile::Desktop => 0,
            PlatformProfile::Mobile => 1,
            PlatformProfile::Console => 2,
            PlatformProfile::Web => 3,
        }
    }

    /// The inverse of [`PlatformProfile::code`]; `None` for an unknown code.
    #[must_use]
    pub const fn from_code(code: u32) -> Option<Self> {
        let profile = match code {
            0 => PlatformProfile::Desktop,
            1 => PlatformProfile::Mobile,
            2 => PlatformProfile::Console,
            3 => PlatformProfile::Web,
            _ => return None,
        };
        Some(profile)
    }
}

/// The complete compiled intermediate representation of a particle graph.
///
/// It carries the pool capacity, the emitter shading model, the target
/// platform, the ordered stages, and the attribute semantics consumed downstream
/// by renderers (`outputs`). Liveness analysis seeds from `outputs` plus the
/// shading model's required inputs.
#[derive(Clone, Debug, PartialEq)]
pub struct GraphIr {
    /// The particle pool capacity every attribute buffer is sized for.
    pub capacity: u32,
    /// The emitter shading model (drives required shading inputs, design §16).
    pub shading_model: EmberShadingModel,
    /// The target platform profile.
    pub platform: PlatformProfile,
    /// The stages, in authored order.
    pub stages: Vec<StageIr>,
    /// Attribute semantics consumed by renderers after simulation; these keep
    /// their producing stages live even when no other stage reads them.
    pub outputs: Vec<AttributeSemantic>,
}

impl GraphIr {
    /// An empty graph for `capacity` particles.
    #[must_use]
    pub fn new(capacity: u32, shading_model: EmberShadingModel, platform: PlatformProfile) -> Self {
        Self {
            capacity,
            shading_model,
            platform,
            stages: Vec::new(),
            outputs: Vec::new(),
        }
    }

    /// Builder: appends a stage.
    #[must_use]
    pub fn with_stage(mut self, stage: StageIr) -> Self {
        self.stages.push(stage);
        self
    }

    /// Builder: marks `semantic` as a renderer-consumed output.
    #[must_use]
    pub fn with_output(mut self, semantic: AttributeSemantic) -> Self {
        self.outputs.push(semantic);
        self
    }

    /// Runs semantic validation on this graph.
    ///
    /// # Errors
    ///
    /// Returns the first [`CompileError`] discovered.
    pub fn validate(&self) -> Result<(), CompileError> {
        validate(self)
    }

    /// The deterministic specialization key for the `pipeline_cache`.
    #[must_use]
    pub fn specialization_key(&self) -> SpecializationKey {
        SpecializationKey::from_graph(self)
    }
}

/// Why a [`GraphIr`] failed semantic validation (design §6.2).
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CompileError {
    /// A stage has an empty name.
    EmptyStageName {
        /// Index of the offending stage.
        stage: usize,
    },
    /// A stage declares zero iterations (a stage must dispatch at least once).
    ZeroIterations {
        /// Index of the offending stage.
        stage: usize,
    },
    /// A stage depends on a stage index that does not exist.
    DependencyOutOfRange {
        /// Index of the stage carrying the bad dependency.
        stage: usize,
        /// The out-of-range dependency index.
        dependency: usize,
    },
    /// The stage dependency graph is not a `DAG`: a cycle was detected.
    DependencyCycle {
        /// A stage that participates in the cycle (smallest index).
        stage: usize,
    },
    /// The same semantic is used with two incompatible formats.
    FormatConflict {
        /// The conflicting semantic.
        semantic: AttributeSemantic,
        /// The first format seen for it.
        first: AttributeFormat,
        /// The conflicting format seen later.
        second: AttributeFormat,
    },
    /// The shading model requires a shading input that no stage writes.
    ShadingInputUnfulfilled {
        /// The required-but-unproduced shading semantic.
        semantic: AttributeSemantic,
    },
}

/// Validates a [`GraphIr`] (design §6.2): structural checks, dependency-`DAG`
/// cycle detection, attribute format consistency, and shading-model
/// compatibility.
///
/// Checks run in a fixed order so a graph with several problems reports the
/// earliest, most fundamental one first: per-stage structure, then dependency
/// ranges and cycles, then format consistency, then shading fulfillment.
///
/// # Errors
///
/// Returns the first [`CompileError`] discovered.
pub fn validate(ir: &GraphIr) -> Result<(), CompileError> {
    let stage_count = ir.stages.len();
    for (index, stage) in ir.stages.iter().enumerate() {
        if stage.name.is_empty() {
            return Err(CompileError::EmptyStageName { stage: index });
        }
        if stage.iterations == 0 {
            return Err(CompileError::ZeroIterations { stage: index });
        }
        for &dep in &stage.deps {
            if dep >= stage_count {
                return Err(CompileError::DependencyOutOfRange {
                    stage: index,
                    dependency: dep,
                });
            }
        }
    }

    // Dependency ranges are valid here, so the only error `schedule_order` can
    // surface is a cycle.
    schedule_order(ir)?;

    // Attribute format consistency: a semantic must use one format everywhere.
    let mut seen: Vec<(AttributeSemantic, AttributeFormat)> = Vec::new();
    for stage in &ir.stages {
        for usage in stage.reads.usages().iter().chain(stage.writes.usages()) {
            if let Some(&(_, first)) = seen.iter().find(|(s, _)| *s == usage.semantic) {
                if first != usage.format {
                    return Err(CompileError::FormatConflict {
                        semantic: usage.semantic,
                        first,
                        second: usage.format,
                    });
                }
            } else {
                seen.push((usage.semantic, usage.format));
            }
        }
    }

    // Shading-model compatibility: every required shading input must be written.
    for semantic in shading_input_attributes(ir.shading_model) {
        let produced = ir.stages.iter().any(|s| s.writes.demands(semantic));
        if !produced {
            return Err(CompileError::ShadingInputUnfulfilled { semantic });
        }
    }

    Ok(())
}

/// Computes a deterministic topological order of the stages (design §7).
///
/// Uses Kahn's algorithm, always selecting the smallest ready stage index so
/// the schedule is reproducible. A stage's `deps` are the stages that must run
/// before it.
///
/// # Errors
///
/// Returns [`CompileError::DependencyOutOfRange`] if a dependency index is
/// invalid, or [`CompileError::DependencyCycle`] if the graph is not a `DAG`.
pub fn schedule_order(ir: &GraphIr) -> Result<Vec<usize>, CompileError> {
    let stage_count = ir.stages.len();

    let mut indegree: Vec<u32> = alloc::vec![0; stage_count];
    let mut successors: Vec<Vec<usize>> = Vec::new();
    successors.resize_with(stage_count, Vec::new);

    for (stage_index, stage) in ir.stages.iter().enumerate() {
        for &dep in &stage.deps {
            if dep >= stage_count {
                return Err(CompileError::DependencyOutOfRange {
                    stage: stage_index,
                    dependency: dep,
                });
            }
            indegree[stage_index] += 1;
            successors[dep].push(stage_index);
        }
    }

    let mut done: Vec<bool> = Vec::new();
    done.resize(stage_count, false);
    let mut order: Vec<usize> = Vec::new();

    while order.len() < stage_count {
        let mut ready: Option<usize> = None;
        for (index, &finished) in done.iter().enumerate() {
            if !finished && indegree[index] == 0 {
                ready = Some(index);
                break;
            }
        }
        match ready {
            Some(node) => {
                done[node] = true;
                order.push(node);
                for &successor in &successors[node] {
                    indegree[successor] -= 1;
                }
            }
            None => {
                // No stage is ready but work remains: a cycle. Report the
                // smallest unfinished stage for a deterministic diagnosis.
                let stage = done
                    .iter()
                    .position(|&finished| !finished)
                    .unwrap_or_default();
                return Err(CompileError::DependencyCycle { stage });
            }
        }
    }

    Ok(order)
}

/// Marks which stages are live under attribute liveness analysis (design §6.2,
/// dead-code elimination).
///
/// Seeds the live-semantic set from the graph's renderer `outputs` and the
/// shading model's required inputs, then reaches backwards to a fixed point: a
/// stage is live if it writes a live semantic, and a live stage's reads become
/// live semantics. Stages that never become live produce nothing observable.
#[must_use]
pub fn live_stage_flags(ir: &GraphIr) -> Vec<bool> {
    let stage_count = ir.stages.len();

    let mut live_semantics: Vec<AttributeSemantic> = Vec::new();
    for &semantic in &ir.outputs {
        let _ = push_unique(&mut live_semantics, semantic);
    }
    for semantic in shading_input_attributes(ir.shading_model) {
        let _ = push_unique(&mut live_semantics, semantic);
    }

    let mut live_stage: Vec<bool> = Vec::new();
    live_stage.resize(stage_count, false);

    loop {
        let mut changed = false;
        #[expect(
            clippy::needless_range_loop,
            reason = "index addresses both `live_stage` and `ir.stages` while mutating `live_stage`"
        )]
        for index in 0..stage_count {
            if live_stage[index] {
                continue;
            }
            let writes_live = ir.stages[index]
                .writes
                .semantics()
                .iter()
                .any(|s| live_semantics.contains(s));
            if writes_live {
                live_stage[index] = true;
                changed = true;
                for semantic in ir.stages[index].reads.semantics() {
                    if push_unique(&mut live_semantics, semantic) {
                        changed = true;
                    }
                }
            }
        }
        if !changed {
            break;
        }
    }

    live_stage
}

/// Pushes `semantic` into `set` if absent; returns whether it was newly added.
#[must_use]
fn push_unique(set: &mut Vec<AttributeSemantic>, semantic: AttributeSemantic) -> bool {
    if set.contains(&semantic) {
        false
    } else {
        set.push(semantic);
        true
    }
}

/// The indices of stages that dead-code elimination can prune (design §6.2).
///
/// A stage is prunable when it is not live per [`live_stage_flags`]: nothing it
/// writes is read by another live stage, consumed by a renderer output, or
/// required by the shading model.
#[must_use]
pub fn prunable_stages(ir: &GraphIr) -> Vec<usize> {
    let live = live_stage_flags(ir);
    let mut prunable = Vec::new();
    for (index, &alive) in live.iter().enumerate() {
        if !alive {
            prunable.push(index);
        }
    }
    prunable
}

/// The merged attribute demand of a graph's *live* stages plus its renderer
/// outputs and shading inputs (design §5.1, §6).
///
/// This is the exact set the layout planner allocates for: pruned (dead) stages
/// contribute nothing, so their write-only scratch buffers cost zero.
#[must_use]
pub fn merged_demand(ir: &GraphIr) -> AttributeDemand {
    let live = live_stage_flags(ir);
    let mut demand = AttributeDemand::new();
    for (index, stage) in ir.stages.iter().enumerate() {
        if !live[index] {
            continue;
        }
        demand.union_with(&stage.reads);
        demand.union_with(&stage.writes);
    }
    for &semantic in &ir.outputs {
        demand.touch(usage_with(semantic, AttributeAccess::READ));
    }
    for semantic in shading_input_attributes(ir.shading_model) {
        demand.touch(usage_with(semantic, AttributeAccess::READ));
    }
    demand
}

/// The live attribute usages that feed [`AttributeLayoutPlan::build`].
#[must_use]
pub fn live_usages(ir: &GraphIr) -> Vec<AttributeUsage> {
    merged_demand(ir).usages().to_vec()
}

/// Builds the `SoA` layout plan for a graph, allocating only live attributes.
#[must_use]
pub fn plan_layout(ir: &GraphIr) -> AttributeLayoutPlan {
    AttributeLayoutPlan::build(ir.capacity, &live_usages(ir))
}

/// The semantics that must be double-buffered (ping-pong): their merged access
/// pairs a previous-frame read with a write this frame (design §5.1, §6.2).
///
/// The result is sorted by [`AttributeSemantic::ordinal`] for determinism.
#[must_use]
pub fn ping_pong_semantics(ir: &GraphIr) -> Vec<AttributeSemantic> {
    let demand = merged_demand(ir);
    let mut out = Vec::new();
    for usage in demand.usages() {
        if usage.access.needs_ping_pong() {
            out.push(usage.semantic);
        }
    }
    out.sort_by_key(|s| s.ordinal());
    out
}

/// The total number of compute dispatches the live stages unroll into: each
/// stage's `iterations` becomes that many dispatches (design §7).
///
/// `saturating_add` keeps the count well-defined even for a pathologically deep
/// schedule.
#[must_use]
pub fn total_dispatches(ir: &GraphIr) -> u32 {
    let live = live_stage_flags(ir);
    let mut total = 0u32;
    for (index, stage) in ir.stages.iter().enumerate() {
        if live[index] {
            total = total.saturating_add(stage.iterations);
        }
    }
    total
}

/// Why a barrier was inserted between two adjacent stages.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum BarrierReason {
    /// The stages have a read/write data hazard (`RAW`, `WAW`, or `WAR`).
    DataHazard,
    /// The stages iterate different domains, so the dispatch topology changes.
    DomainChange,
}

/// A memory/execution barrier the scheduler must insert between two adjacent
/// stages (design §7).
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Barrier {
    /// Index of the earlier stage.
    pub before: usize,
    /// Index of the later stage.
    pub after: usize,
    /// Why the barrier is required.
    pub reason: BarrierReason,
}

/// Whether two stages have a read/write data hazard.
///
/// Covers read-after-write (`RAW`), write-after-write (`WAW`), and
/// write-after-read (`WAR`) on any shared semantic.
#[must_use]
fn stages_conflict(earlier: &StageIr, later: &StageIr) -> bool {
    earlier.writes.intersects(&later.reads)
        || earlier.writes.intersects(&later.writes)
        || earlier.reads.intersects(&later.writes)
}

/// Plans the minimal set of barriers between consecutive stages (design §7).
///
/// A barrier is inserted between adjacent stages only when their read/write sets
/// conflict (a data hazard) or when the iteration domain changes. Adjacent,
/// same-domain stages with disjoint read/write sets need no barrier and can be
/// merged, matching the "insert the fewest barriers" rule of the design.
#[must_use]
pub fn plan_barriers(stages: &[StageIr]) -> Vec<Barrier> {
    let mut barriers = Vec::new();
    for (index, pair) in stages.windows(2).enumerate() {
        let earlier = &pair[0];
        let later = &pair[1];
        let reason = if stages_conflict(earlier, later) {
            Some(BarrierReason::DataHazard)
        } else if earlier.domain != later.domain {
            Some(BarrierReason::DomainChange)
        } else {
            None
        };
        if let Some(reason) = reason {
            barriers.push(Barrier {
                before: index,
                after: index + 1,
                reason,
            });
        }
    }
    barriers
}

/// A deterministic specialization key handed to the `pipeline_cache` so a graph
/// compiles once per distinct (layout, shading, platform) tuple (design §6.2).
///
/// The key stores integer content hashes only, so it derives `Eq`/`Hash`
/// exactly and round-trips losslessly through [`SpecializationKey::to_words`]
/// and [`SpecializationKey::from_words`]. The attribute-layout hash is computed
/// from the ordinal-sorted plan, so it is independent of stage authoring order.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct SpecializationKey {
    /// A content hash of the `SoA` attribute layout plan.
    pub layout_hash: u64,
    /// A deterministic encoding of the emitter shading model.
    pub shading_code: u64,
    /// The target platform profile.
    pub platform: PlatformProfile,
}

impl SpecializationKey {
    /// Constructs the key from a compiled graph.
    #[must_use]
    pub fn from_graph(ir: &GraphIr) -> Self {
        Self {
            layout_hash: hash_layout(&plan_layout(ir)),
            shading_code: encode_shading_model(ir.shading_model),
            platform: ir.platform,
        }
    }

    /// Packs the key into three 64-bit words for storage / transport.
    #[must_use]
    pub fn to_words(self) -> [u64; 3] {
        [
            self.layout_hash,
            self.shading_code,
            u64::from(self.platform.code()),
        ]
    }

    /// Reconstructs a key from [`SpecializationKey::to_words`]; `None` if the
    /// platform word is not a valid [`PlatformProfile`] code.
    #[must_use]
    pub fn from_words(words: [u64; 3]) -> Option<Self> {
        let platform_code = u32::try_from(words[2]).ok()?;
        let platform = PlatformProfile::from_code(platform_code)?;
        Some(Self {
            layout_hash: words[0],
            shading_code: words[1],
            platform,
        })
    }
}

/// A stable numeric code for one shading basis lobe.
#[must_use]
const fn encode_basis(basis: ShadingBasis) -> u64 {
    match basis {
        ShadingBasis::Unlit => 1,
        ShadingBasis::Pbr => 2,
        ShadingBasis::Npr => 3,
        ShadingBasis::Custom(id) => (4u64 << 32) | (id as u64),
    }
}

/// A deterministic `FNV`-1a encoding of an [`EmberShadingModel`].
///
/// The `Hybrid` blend weight is folded in through its exact `f32` bit pattern
/// (`to_bits`), so no floating-point comparison is ever performed on it.
#[must_use]
fn encode_shading_model(model: EmberShadingModel) -> u64 {
    let mut hash = FNV_OFFSET_BASIS;
    match model {
        EmberShadingModel::Unlit => hash = fnv_u64(hash, 1),
        EmberShadingModel::Pbr => hash = fnv_u64(hash, 2),
        EmberShadingModel::Npr => hash = fnv_u64(hash, 3),
        EmberShadingModel::Custom(id) => {
            hash = fnv_u64(hash, 4);
            hash = fnv_u64(hash, u64::from(id));
        }
        EmberShadingModel::Hybrid {
            base,
            overlay,
            weight,
        } => {
            hash = fnv_u64(hash, 5);
            hash = fnv_u64(hash, encode_basis(base));
            hash = fnv_u64(hash, encode_basis(overlay));
            hash = fnv_u64(hash, u64::from(weight.to_bits()));
        }
    }
    hash
}

/// A stable numeric code for an [`AttributeFormat`].
#[must_use]
const fn format_code(format: AttributeFormat) -> u32 {
    match format {
        AttributeFormat::F32 => 0,
        AttributeFormat::U32 => 1,
        AttributeFormat::Vec2 => 2,
        AttributeFormat::Vec3 => 3,
        AttributeFormat::Vec4 => 4,
    }
}

/// A deterministic `FNV`-1a content hash of a layout plan, folding capacity and
/// every planned attribute's ordinal, format, stride, copy count, and byte span.
#[must_use]
fn hash_layout(plan: &AttributeLayoutPlan) -> u64 {
    let mut hash = FNV_OFFSET_BASIS;
    hash = fnv_u64(hash, u64::from(plan.capacity));
    for attribute in &plan.attributes {
        hash = fnv_u64(hash, u64::from(attribute.semantic.ordinal()));
        hash = fnv_u64(hash, u64::from(format_code(attribute.format)));
        hash = fnv_u64(hash, u64::from(attribute.stride));
        hash = fnv_u64(hash, u64::from(attribute.copies));
        hash = fnv_u64(hash, attribute.byte_offset);
        hash = fnv_u64(hash, attribute.byte_size);
    }
    hash
}

#[cfg(test)]
mod tests {
    use super::*;

    fn linear_update() -> GraphIr {
        GraphIr::new(64, EmberShadingModel::Unlit, PlatformProfile::Desktop)
            .with_stage(
                StageIr::new("emit", IterationDomain::PerParticle, 1)
                    .writing(write(AttributeSemantic::Position))
                    .writing(write(AttributeSemantic::Velocity)),
            )
            .with_stage(
                StageIr::new("integrate", IterationDomain::PerParticle, 1)
                    .reading(read(AttributeSemantic::Velocity))
                    .writing(write(AttributeSemantic::Position))
                    .depends_on(0),
            )
            .with_output(AttributeSemantic::Position)
    }

    #[test]
    fn touch_unions_access_for_repeated_semantics() {
        let mut demand = AttributeDemand::new();
        demand.touch(AttributeUsage::new(
            AttributeSemantic::Position,
            AttributeFormat::Vec3,
            AttributeAccess::READ,
        ));
        demand.touch(AttributeUsage::new(
            AttributeSemantic::Position,
            AttributeFormat::Vec3,
            AttributeAccess::WRITE,
        ));
        assert_eq!(demand.usages().len(), 1);
        let usage = demand.usages()[0];
        assert!(usage.access.contains(AttributeAccess::READ));
        assert!(usage.access.contains(AttributeAccess::WRITE));
    }

    #[test]
    fn demands_reports_touched_semantics() {
        let mut demand = AttributeDemand::new();
        demand.touch(read_write(AttributeSemantic::Velocity));
        assert!(demand.demands(AttributeSemantic::Velocity));
        assert!(!demand.demands(AttributeSemantic::Color));
    }

    #[test]
    fn empty_demand_and_intersection() {
        let mut a = AttributeDemand::new();
        assert!(a.is_empty());
        a.touch(read(AttributeSemantic::Position));
        assert!(!a.is_empty());
        let mut b = AttributeDemand::new();
        b.touch(write(AttributeSemantic::Position));
        assert!(a.intersects(&b));
        let mut c = AttributeDemand::new();
        c.touch(write(AttributeSemantic::Color));
        assert!(!a.intersects(&c));
    }

    #[test]
    fn stage_demand_unions_reads_and_writes() {
        let stage = StageIr::new("s", IterationDomain::PerParticle, 1)
            .reading(read(AttributeSemantic::Velocity))
            .writing(write(AttributeSemantic::Position));
        let demand = stage.demand();
        assert!(demand.demands(AttributeSemantic::Velocity));
        assert!(demand.demands(AttributeSemantic::Position));
    }

    #[test]
    fn linear_graph_validates_and_orders() {
        let ir = linear_update();
        assert!(ir.validate().is_ok());
        assert_eq!(schedule_order(&ir).unwrap(), alloc::vec![0, 1]);
    }

    #[test]
    fn cycle_is_detected() {
        let ir = GraphIr::new(8, EmberShadingModel::Unlit, PlatformProfile::Desktop)
            .with_stage(StageIr::new("a", IterationDomain::PerParticle, 1).depends_on(1))
            .with_stage(StageIr::new("b", IterationDomain::PerParticle, 1).depends_on(0));
        assert!(matches!(
            ir.validate(),
            Err(CompileError::DependencyCycle { .. })
        ));
        assert!(matches!(
            schedule_order(&ir),
            Err(CompileError::DependencyCycle { .. })
        ));
    }

    #[test]
    fn self_dependency_is_a_cycle() {
        let ir = GraphIr::new(8, EmberShadingModel::Unlit, PlatformProfile::Desktop)
            .with_stage(StageIr::new("a", IterationDomain::PerParticle, 1).depends_on(0));
        assert!(matches!(
            ir.validate(),
            Err(CompileError::DependencyCycle { stage: 0 })
        ));
    }

    #[test]
    fn diamond_schedule_is_deterministic() {
        let ir = GraphIr::new(8, EmberShadingModel::Unlit, PlatformProfile::Desktop)
            .with_stage(StageIr::new("root", IterationDomain::PerParticle, 1))
            .with_stage(StageIr::new("left", IterationDomain::PerParticle, 1).depends_on(0))
            .with_stage(StageIr::new("right", IterationDomain::PerParticle, 1).depends_on(0))
            .with_stage(
                StageIr::new("join", IterationDomain::PerParticle, 1)
                    .depends_on(1)
                    .depends_on(2),
            );
        assert_eq!(schedule_order(&ir).unwrap(), alloc::vec![0, 1, 2, 3]);
    }

    #[test]
    fn dependency_out_of_range_is_rejected() {
        let ir = GraphIr::new(8, EmberShadingModel::Unlit, PlatformProfile::Desktop)
            .with_stage(StageIr::new("a", IterationDomain::PerParticle, 1).depends_on(5));
        assert!(matches!(
            ir.validate(),
            Err(CompileError::DependencyOutOfRange {
                stage: 0,
                dependency: 5
            })
        ));
    }

    #[test]
    fn zero_iterations_is_rejected() {
        let ir = GraphIr::new(8, EmberShadingModel::Unlit, PlatformProfile::Desktop)
            .with_stage(StageIr::new("a", IterationDomain::PerParticle, 0));
        assert!(matches!(
            ir.validate(),
            Err(CompileError::ZeroIterations { stage: 0 })
        ));
    }

    #[test]
    fn empty_stage_name_is_rejected() {
        let ir = GraphIr::new(8, EmberShadingModel::Unlit, PlatformProfile::Desktop)
            .with_stage(StageIr::new("", IterationDomain::PerParticle, 1));
        assert!(matches!(
            ir.validate(),
            Err(CompileError::EmptyStageName { stage: 0 })
        ));
    }

    #[test]
    fn format_conflict_is_detected() {
        let ir = GraphIr::new(8, EmberShadingModel::Unlit, PlatformProfile::Desktop)
            .with_stage(
                StageIr::new("a", IterationDomain::PerParticle, 1)
                    .reading(read(AttributeSemantic::Position)),
            )
            .with_stage(StageIr::new("b", IterationDomain::PerParticle, 1).writing(
                AttributeUsage::new(
                    AttributeSemantic::Position,
                    AttributeFormat::Vec4,
                    AttributeAccess::WRITE,
                ),
            ));
        assert!(matches!(
            ir.validate(),
            Err(CompileError::FormatConflict {
                semantic: AttributeSemantic::Position,
                ..
            })
        ));
    }

    #[test]
    fn shading_input_unfulfilled_is_detected() {
        let ir = GraphIr::new(8, EmberShadingModel::Pbr, PlatformProfile::Desktop).with_stage(
            StageIr::new("update", IterationDomain::PerParticle, 1)
                .writing(write(AttributeSemantic::Position)),
        );
        assert!(matches!(
            ir.validate(),
            Err(CompileError::ShadingInputUnfulfilled { .. })
        ));
    }

    #[test]
    fn shading_inputs_fulfilled_validates() {
        let ir = GraphIr::new(8, EmberShadingModel::Pbr, PlatformProfile::Desktop).with_stage(
            StageIr::new("shade_setup", IterationDomain::PerParticle, 1)
                .writing(write(AttributeSemantic::Normal))
                .writing(write(AttributeSemantic::Roughness))
                .writing(write(AttributeSemantic::Metallic))
                .writing(write(AttributeSemantic::MaterialId)),
        );
        assert!(ir.validate().is_ok());
    }

    #[test]
    fn dead_stage_is_prunable_and_not_allocated() {
        let ir = GraphIr::new(32, EmberShadingModel::Unlit, PlatformProfile::Desktop)
            .with_stage(
                StageIr::new("emit", IterationDomain::PerParticle, 1)
                    .writing(write(AttributeSemantic::Velocity)),
            )
            .with_stage(
                StageIr::new("integrate", IterationDomain::PerParticle, 1)
                    .reading(read(AttributeSemantic::Velocity))
                    .writing(write(AttributeSemantic::Position))
                    .depends_on(0),
            )
            .with_stage(
                StageIr::new("dead", IterationDomain::PerParticle, 3)
                    .writing(write(AttributeSemantic::Color)),
            )
            .with_output(AttributeSemantic::Position);

        assert_eq!(prunable_stages(&ir), alloc::vec![2]);
        let plan = plan_layout(&ir);
        assert!(plan.contains(AttributeSemantic::Position));
        assert!(plan.contains(AttributeSemantic::Velocity));
        assert!(!plan.contains(AttributeSemantic::Color));
        // The dead stage's iterations do not count toward dispatch work.
        assert_eq!(total_dispatches(&ir), 2);
    }

    #[test]
    fn transitive_dead_code_is_pruned() {
        // `a` feeds only `b`, and `b` is itself dead, so both prune.
        let ir = GraphIr::new(16, EmberShadingModel::Unlit, PlatformProfile::Desktop)
            .with_stage(
                StageIr::new("a", IterationDomain::PerParticle, 1)
                    .writing(write(AttributeSemantic::Custom(1))),
            )
            .with_stage(
                StageIr::new("b", IterationDomain::PerParticle, 1)
                    .reading(read(AttributeSemantic::Custom(1)))
                    .writing(write(AttributeSemantic::Custom(2)))
                    .depends_on(0),
            );
        assert_eq!(prunable_stages(&ir), alloc::vec![0, 1]);
    }

    #[test]
    fn ping_pong_decision_reads_prev_and_writes() {
        let ir = GraphIr::new(16, EmberShadingModel::Unlit, PlatformProfile::Desktop)
            .with_stage(
                StageIr::new("advect", IterationDomain::PerParticle, 1)
                    .reading(read_prev(AttributeSemantic::Position))
                    .writing(write(AttributeSemantic::Position)),
            )
            .with_output(AttributeSemantic::Position);
        assert_eq!(
            ping_pong_semantics(&ir),
            alloc::vec![AttributeSemantic::Position]
        );
        assert_eq!(plan_layout(&ir).ping_pong_count(), 1);
    }

    #[test]
    fn read_write_in_place_is_not_ping_pong() {
        let ir = GraphIr::new(16, EmberShadingModel::Unlit, PlatformProfile::Desktop)
            .with_stage(
                StageIr::new("integrate", IterationDomain::PerParticle, 1)
                    .reading(read(AttributeSemantic::Position))
                    .writing(write(AttributeSemantic::Position)),
            )
            .with_output(AttributeSemantic::Position);
        assert!(ping_pong_semantics(&ir).is_empty());
    }

    #[test]
    fn barrier_inserted_on_read_after_write() {
        let stages = alloc::vec![
            StageIr::new("write", IterationDomain::PerParticle, 1)
                .writing(write(AttributeSemantic::Position)),
            StageIr::new("read", IterationDomain::PerParticle, 1)
                .reading(read(AttributeSemantic::Position)),
        ];
        let barriers = plan_barriers(&stages);
        assert_eq!(barriers.len(), 1);
        assert_eq!(barriers[0].before, 0);
        assert_eq!(barriers[0].after, 1);
        assert_eq!(barriers[0].reason, BarrierReason::DataHazard);
    }

    #[test]
    fn no_barrier_when_sets_are_disjoint_same_domain() {
        let stages = alloc::vec![
            StageIr::new("a", IterationDomain::PerParticle, 1)
                .writing(write(AttributeSemantic::Velocity)),
            StageIr::new("b", IterationDomain::PerParticle, 1)
                .writing(write(AttributeSemantic::Color)),
        ];
        assert!(plan_barriers(&stages).is_empty());
    }

    #[test]
    fn barrier_on_domain_change_even_without_hazard() {
        let stages = alloc::vec![
            StageIr::new("particles", IterationDomain::PerParticle, 1)
                .writing(write(AttributeSemantic::Velocity)),
            StageIr::new("voxels", IterationDomain::PerGridVoxel, 1)
                .writing(write(AttributeSemantic::Color)),
        ];
        let barriers = plan_barriers(&stages);
        assert_eq!(barriers.len(), 1);
        assert_eq!(barriers[0].reason, BarrierReason::DomainChange);
    }

    #[test]
    fn write_after_read_is_a_hazard() {
        let stages = alloc::vec![
            StageIr::new("reader", IterationDomain::PerParticle, 1)
                .reading(read(AttributeSemantic::Position)),
            StageIr::new("writer", IterationDomain::PerParticle, 1)
                .writing(write(AttributeSemantic::Position)),
        ];
        let barriers = plan_barriers(&stages);
        assert_eq!(barriers.len(), 1);
        assert_eq!(barriers[0].reason, BarrierReason::DataHazard);
    }

    #[test]
    fn specialization_key_round_trips() {
        let ir = linear_update();
        let key = ir.specialization_key();
        let words = key.to_words();
        assert_eq!(SpecializationKey::from_words(words), Some(key));
    }

    #[test]
    fn specialization_key_rejects_bad_platform_word() {
        assert_eq!(SpecializationKey::from_words([1, 2, 999]), None);
    }

    #[test]
    fn specialization_key_is_insertion_order_independent() {
        let forward = GraphIr::new(64, EmberShadingModel::Unlit, PlatformProfile::Desktop)
            .with_stage(
                StageIr::new("s", IterationDomain::PerParticle, 1)
                    .writing(write(AttributeSemantic::Position))
                    .writing(write(AttributeSemantic::Velocity)),
            )
            .with_output(AttributeSemantic::Position)
            .with_output(AttributeSemantic::Velocity);
        let reversed = GraphIr::new(64, EmberShadingModel::Unlit, PlatformProfile::Desktop)
            .with_stage(
                StageIr::new("s", IterationDomain::PerParticle, 1)
                    .writing(write(AttributeSemantic::Velocity))
                    .writing(write(AttributeSemantic::Position)),
            )
            .with_output(AttributeSemantic::Position)
            .with_output(AttributeSemantic::Velocity);
        assert_eq!(forward.specialization_key(), reversed.specialization_key());
    }

    #[test]
    fn specialization_key_differs_by_platform_and_model() {
        let base = linear_update();
        let mut mobile = base.clone();
        mobile.platform = PlatformProfile::Mobile;
        assert_ne!(base.specialization_key(), mobile.specialization_key());

        let mut pbr = base.clone();
        pbr.shading_model = EmberShadingModel::Pbr;
        assert_ne!(
            base.specialization_key().shading_code,
            pbr.specialization_key().shading_code
        );
    }

    #[test]
    fn hybrid_shading_encoding_is_deterministic() {
        let model = EmberShadingModel::Hybrid {
            base: ShadingBasis::Pbr,
            overlay: ShadingBasis::Npr,
            weight: 0.25,
        };
        assert_eq!(encode_shading_model(model), encode_shading_model(model));
        let other = EmberShadingModel::Hybrid {
            base: ShadingBasis::Pbr,
            overlay: ShadingBasis::Npr,
            weight: 0.75,
        };
        assert_ne!(encode_shading_model(model), encode_shading_model(other));
    }

    #[test]
    fn empty_graph_is_well_defined() {
        let ir = GraphIr::new(0, EmberShadingModel::Unlit, PlatformProfile::Desktop);
        assert!(ir.validate().is_ok());
        assert!(schedule_order(&ir).unwrap().is_empty());
        assert!(plan_layout(&ir).attributes.is_empty());
        assert!(plan_barriers(&ir.stages).is_empty());
        assert!(prunable_stages(&ir).is_empty());
        assert!(ping_pong_semantics(&ir).is_empty());
        assert_eq!(total_dispatches(&ir), 0);
    }

    #[test]
    fn platform_code_round_trips() {
        for profile in [
            PlatformProfile::Desktop,
            PlatformProfile::Mobile,
            PlatformProfile::Console,
            PlatformProfile::Web,
        ] {
            assert_eq!(PlatformProfile::from_code(profile.code()), Some(profile));
        }
        assert_eq!(PlatformProfile::from_code(42), None);
    }

    #[test]
    fn total_dispatches_unrolls_iterations() {
        let ir = GraphIr::new(16, EmberShadingModel::Unlit, PlatformProfile::Desktop)
            .with_stage(
                StageIr::new("pressure", IterationDomain::PerGridVoxel, 8)
                    .writing(write(AttributeSemantic::Position)),
            )
            .with_output(AttributeSemantic::Position);
        assert_eq!(total_dispatches(&ir), 8);
    }
}
