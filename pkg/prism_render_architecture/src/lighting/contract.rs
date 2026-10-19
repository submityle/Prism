//! Unified lighting/GI contracts and deterministic planning.
//!
//! This module is the CPU-side authority for the lighting vertical slice. It
//! deliberately carries policy and accounting only: GPU implementations may
//! consume the resulting plan, but they cannot invent a second recipe list,
//! error model, or invalidation path.

use alloc::string::String;
use alloc::vec::Vec;
use core::fmt;

use super::debug::DebugMeta;

/// Monotonic epochs for fields that independently invalidate lighting work.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct FieldEpoch {
    pub geometry: u32,
    pub material: u32,
    pub lighting: u32,
    pub environment: u32,
    pub layout: u32,
}

impl FieldEpoch {
    /// Returns the next epoch, saturating instead of wrapping stale data into
    /// a seemingly newer generation.
    #[must_use]
    pub fn advanced(self, fields: ChangeFields) -> Self {
        Self {
            geometry: bump_if(self.geometry, fields.contains(ChangeFields::GEOMETRY)),
            material: bump_if(self.material, fields.contains(ChangeFields::MATERIAL)),
            lighting: bump_if(self.lighting, fields.contains(ChangeFields::LIGHTING)),
            environment: bump_if(self.environment, fields.contains(ChangeFields::ENVIRONMENT)),
            layout: bump_if(self.layout, fields.contains(ChangeFields::LAYOUT)),
        }
    }
}

fn bump_if(value: u32, enabled: bool) -> u32 {
    if enabled {
        value.saturating_add(1)
    } else {
        value
    }
}

/// Fields that may invalidate a lighting cache.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct ChangeFields(u8);

impl ChangeFields {
    pub const GEOMETRY: Self = Self(1 << 0);
    pub const MATERIAL: Self = Self(1 << 1);
    pub const LIGHTING: Self = Self(1 << 2);
    pub const ENVIRONMENT: Self = Self(1 << 3);
    pub const LAYOUT: Self = Self(1 << 4);

    #[must_use]
    pub const fn contains(self, other: Self) -> bool {
        self.0 & other.0 == other.0
    }

    pub fn insert(&mut self, other: Self) {
        self.0 |= other.0;
    }

    #[must_use]
    pub const fn is_empty(self) -> bool {
        self.0 == 0
    }
}

/// A write-ahead change record. The journal is appended before the scene
/// state is published, which makes replay and device-loss reconstruction
/// deterministic.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ChangeRecord {
    pub sequence: u64,
    pub frame: u64,
    pub fields: ChangeFields,
    pub epoch: FieldEpoch,
}

/// Incremental invalidation journal with explicit checkpoints.
#[derive(Clone, Debug, Default)]
pub struct ChangeJournal {
    epoch: FieldEpoch,
    next_sequence: u64,
    records: Vec<ChangeRecord>,
}

impl ChangeJournal {
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Appends one record and returns the resulting field epoch.
    pub fn append(&mut self, frame: u64, fields: ChangeFields) -> FieldEpoch {
        if fields.is_empty() {
            return self.epoch;
        }
        self.epoch = self.epoch.advanced(fields);
        self.next_sequence = self.next_sequence.saturating_add(1);
        self.records.push(ChangeRecord {
            sequence: self.next_sequence,
            frame,
            fields,
            epoch: self.epoch,
        });
        self.epoch
    }

    #[must_use]
    pub fn epoch(&self) -> FieldEpoch {
        self.epoch
    }

    #[must_use]
    pub fn records_since(&self, sequence: u64) -> &[ChangeRecord] {
        let index = self
            .records
            .partition_point(|record| record.sequence <= sequence);
        &self.records[index..]
    }

    /// Drops records covered by a durable checkpoint while retaining the
    /// current epoch as the authoritative state version.
    pub fn checkpoint(&mut self) -> JournalSnapshot {
        let snapshot = JournalSnapshot {
            sequence: self.next_sequence,
            epoch: self.epoch,
        };
        self.records.clear();
        snapshot
    }

    #[must_use]
    pub fn snapshot(&self) -> JournalSnapshot {
        JournalSnapshot {
            sequence: self.next_sequence,
            epoch: self.epoch,
        }
    }
}

/// Explicitly serializable journal checkpoint.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct JournalSnapshot {
    pub sequence: u64,
    pub epoch: FieldEpoch,
}

/// The trace workload consumed by a backend.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum TraceKind {
    Shadow,
    Diffuse,
    Specular,
    Volumetric,
    Occlusion,
}

/// Backend execution model.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ExecModel {
    Sync,
    Async { depth: u16 },
    Persistent { threads: u16 },
}

/// Maximum result latency accepted by the caller.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum LatencyClass {
    SameFrame,
    Plus1Frame,
    PlusNFrame(u16),
}

impl LatencyClass {
    #[must_use]
    pub const fn allowed_frames(self) -> u64 {
        match self {
            Self::SameFrame => 0,
            Self::Plus1Frame => 1,
            Self::PlusNFrame(n) => n as u64,
        }
    }
}

/// Static backend capabilities.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BackendCaps {
    pub kinds: u8,
    pub max_t_max: f32,
    pub alpha: bool,
    pub is_hardware: bool,
    pub exec: ExecModel,
    pub latency: LatencyClass,
    pub max_batch: u32,
}

impl BackendCaps {
    #[must_use]
    pub const fn supports(self, kind: TraceKind) -> bool {
        let bit = match kind {
            TraceKind::Shadow => 1 << 0,
            TraceKind::Diffuse => 1 << 1,
            TraceKind::Specular => 1 << 2,
            TraceKind::Volumetric => 1 << 3,
            TraceKind::Occlusion => 1 << 4,
        };
        self.kinds & bit != 0
    }
}

/// A trace query is intentionally backend-neutral.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TraceQuery {
    pub kind: TraceKind,
    pub origin: [f32; 3],
    pub direction: [f32; 3],
    pub t_max: f32,
}

/// A trace result carrying systematic and statistical error separately.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RayHit {
    pub radiance: [f32; 3],
    pub depth: f32,
    pub material: u32,
    pub bias_bound: f32,
    pub variance: f32,
}

/// Submission identity includes the backend epoch, so device recovery cannot
/// accidentally consume results from the old device.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct SubmissionToken {
    pub backend_epoch: u64,
    pub id: u64,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct TraceSubmission {
    pub token: SubmissionToken,
    pub slot_base: u32,
    pub n: u32,
    pub generation: u64,
    pub submitted_frame: u64,
    pub deadline_frame: u64,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum BackendError {
    DeviceLost,
    OutOfMemory,
    KindUnavailable(TraceKind),
    InvalidBatch,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PollState {
    Pending,
    Partial,
    Ready,
    Cancelled { acked: bool },
    Expired,
    Failed(BackendError),
    Stale,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CancelAck {
    Pending,
    Acked,
}

/// The only trace entry point. Implementations must preserve slot order.
pub trait RayBackend {
    fn capabilities(&self) -> BackendCaps;
    fn runtime(&self) -> BackendRuntime;
    fn submit(
        &mut self,
        queries: &[TraceQuery],
        slot_base: u32,
        generation: u64,
        deadline_frame: u64,
    ) -> Result<TraceSubmission, BackendError>;
    fn poll(
        &mut self,
        token: SubmissionToken,
        generation: u64,
        frame: u64,
        out: &mut [Option<RayHit>],
    ) -> PollState;
    fn cancel(&mut self, token: SubmissionToken, generation: u64) -> CancelAck;
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum BackendRuntimeError {
    Ok,
    DeviceLost,
    OutOfMemory,
    KindUnavailable,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct BackendRuntime {
    pub inflight: u32,
    pub queue_depth: u32,
    pub last_completion_frame: u64,
    pub error: BackendRuntimeError,
    pub resident_bytes: u64,
}

/// The common integration target of every lighting/GI contribution.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct TargetId(pub u16);

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum WeightKind {
    Mis { pdf: u32 },
    Ratio,
    AdHoc,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Contrib {
    pub value: [f32; 3],
    pub variance: f32,
    pub bias_bound: f32,
    pub target: TargetId,
    pub weight_semantics: WeightKind,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum BlendPolicy {
    Unbiased,
    Conservative,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BlendOut {
    pub value: [f32; 3],
    pub variance: f32,
    pub bias_bound: f32,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum BlendError {
    Empty,
    TargetMismatch,
    MissingPdf,
    BiasRequiresConservative,
}

/// Blends contributions only after checking target and estimator semantics.
pub fn blend(contributions: &[Contrib], policy: BlendPolicy) -> Result<BlendOut, BlendError> {
    let Some(first) = contributions.first() else {
        return Err(BlendError::Empty);
    };
    if contributions.iter().any(|item| item.target != first.target) {
        return Err(BlendError::TargetMismatch);
    }
    if matches!(policy, BlendPolicy::Unbiased)
        && contributions.iter().any(|item| item.bias_bound > 0.0)
    {
        return Err(BlendError::BiasRequiresConservative);
    }
    let mut weights = Vec::with_capacity(contributions.len());
    for item in contributions {
        let weight = match (policy, item.weight_semantics) {
            (BlendPolicy::Unbiased, WeightKind::Mis { pdf }) if pdf != 0 => 1.0 / pdf as f32,
            (BlendPolicy::Unbiased, _) => return Err(BlendError::MissingPdf),
            (BlendPolicy::Conservative, _) => {
                1.0 / (1.0 + item.variance.max(0.0) + item.bias_bound.max(0.0))
            }
        };
        weights.push(weight);
    }
    let sum = weights.iter().sum::<f32>().max(f32::MIN_POSITIVE);
    let mut out = BlendOut {
        value: [0.0; 3],
        variance: 0.0,
        bias_bound: 0.0,
    };
    for (item, weight) in contributions.iter().zip(weights) {
        let w = weight / sum;
        for channel in 0..3 {
            out.value[channel] += item.value[channel] * w;
        }
        out.variance += item.variance.max(0.0) * w * w;
        out.bias_bound += item.bias_bound.max(0.0) * w;
    }
    Ok(out)
}

/// The complete recipe white-list from the architecture proposal.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum RecipeId {
    DirectStochasticRt,
    DirectStochasticVsm,
    DirectDeterministicCsm,
    SpecializedDeterministic,
    DdgiProbes,
    ScreenProbeGi,
    SurfaceCacheSurfel,
    WorldRestir,
    SsgiScreenSpace,
    SpecularRestir,
    SparseSdfGi,
    RadianceCascades,
    PathTracedOffline,
}

/// The resource domain a recipe contributes to in the frame graph.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum RecipeDomain {
    Direct,
    Shadow,
    Indirect,
    Reflection,
    Offline,
}

/// The stable execution declaration emitted for an admitted recipe.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RecipeExecution {
    pub recipe: RecipeId,
    pub domain: RecipeDomain,
    pub cache_effect: CacheEffect,
    pub exec_policy: ExecPolicy,
}

impl RecipeId {
    pub const ALL: [Self; 13] = [
        Self::DirectStochasticRt,
        Self::DirectStochasticVsm,
        Self::DirectDeterministicCsm,
        Self::SpecializedDeterministic,
        Self::DdgiProbes,
        Self::ScreenProbeGi,
        Self::SurfaceCacheSurfel,
        Self::WorldRestir,
        Self::SsgiScreenSpace,
        Self::SpecularRestir,
        Self::SparseSdfGi,
        Self::RadianceCascades,
        Self::PathTracedOffline,
    ];

    #[must_use]
    pub const fn domain(self) -> RecipeDomain {
        match self {
            Self::DirectStochasticRt | Self::SpecializedDeterministic => RecipeDomain::Direct,
            Self::DirectStochasticVsm | Self::DirectDeterministicCsm => RecipeDomain::Shadow,
            Self::DdgiProbes
            | Self::ScreenProbeGi
            | Self::SurfaceCacheSurfel
            | Self::WorldRestir
            | Self::SparseSdfGi
            | Self::RadianceCascades => RecipeDomain::Indirect,
            Self::SsgiScreenSpace | Self::SpecularRestir => RecipeDomain::Reflection,
            Self::PathTracedOffline => RecipeDomain::Offline,
        }
    }

    /// Stable compact identifier used by GPU diagnostics and telemetry.
    #[must_use]
    pub const fn ordinal(self) -> u32 {
        match self {
            Self::DirectStochasticRt => 0,
            Self::DirectStochasticVsm => 1,
            Self::DirectDeterministicCsm => 2,
            Self::SpecializedDeterministic => 3,
            Self::DdgiProbes => 4,
            Self::ScreenProbeGi => 5,
            Self::SurfaceCacheSurfel => 6,
            Self::WorldRestir => 7,
            Self::SsgiScreenSpace => 8,
            Self::SpecularRestir => 9,
            Self::SparseSdfGi => 10,
            Self::RadianceCascades => 11,
            Self::PathTracedOffline => 12,
        }
    }

    /// Whether the current render host has a real execution path for this
    /// recipe. A recipe stays in the architectural whitelist while admission
    /// remains closed until its GPU/CPU twin and graph pass are registered.
    #[must_use]
    pub const fn has_execution_backend(self) -> bool {
        matches!(self, Self::SpecializedDeterministic | Self::WorldRestir)
    }

    #[must_use]
    pub const fn execution(self, pass: u32, budget_ms: f32) -> RecipeExecution {
        let async_policy = match self {
            Self::PathTracedOffline | Self::DdgiProbes | Self::SurfaceCacheSurfel => {
                AsyncPolicy::Split
            }
            Self::WorldRestir | Self::SpecularRestir | Self::RadianceCascades => AsyncPolicy::Join,
            _ => AsyncPolicy::Inline,
        };
        RecipeExecution {
            recipe: self,
            domain: self.domain(),
            cache_effect: CacheEffect {
                pass,
                destroys: match self.domain() {
                    RecipeDomain::Shadow => 1 << 0,
                    RecipeDomain::Indirect => 1 << 1,
                    RecipeDomain::Reflection => 1 << 2,
                    RecipeDomain::Offline => 1 << 3,
                    RecipeDomain::Direct => 0,
                },
                preserves: 1 << 4,
            },
            exec_policy: ExecPolicy {
                pass,
                async_policy,
                persistent: !matches!(self, Self::DirectDeterministicCsm),
                shading_rate: 1.0,
                budget_ms,
            },
        }
    }

    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Self::DirectStochasticRt => "DirectStochasticRT",
            Self::DirectStochasticVsm => "DirectStochasticVSM",
            Self::DirectDeterministicCsm => "DirectDeterministicCSM",
            Self::SpecializedDeterministic => "SpecializedDeterministic",
            Self::DdgiProbes => "DdgiProbes",
            Self::ScreenProbeGi => "ScreenProbeGi",
            Self::SurfaceCacheSurfel => "SurfaceCacheSurfel",
            Self::WorldRestir => "WorldRestir",
            Self::SsgiScreenSpace => "SsgiScreenSpace",
            Self::SpecularRestir => "SpecularRestir",
            Self::SparseSdfGi => "SparseSdfGi",
            Self::RadianceCascades => "RadianceCascades",
            Self::PathTracedOffline => "PathTracedOffline",
        }
    }

    /// Static conservative cost used by the planner before runtime telemetry
    /// calibrates a backend-specific coefficient.
    #[must_use]
    pub const fn cost(self) -> RecipeCost {
        match self {
            Self::DirectStochasticRt => RecipeCost::new(2.4, 24 * 1024 * 1024, 48 * 1024 * 1024, 0),
            Self::DirectStochasticVsm => {
                RecipeCost::new(1.6, 32 * 1024 * 1024, 64 * 1024 * 1024, 128)
            }
            Self::DirectDeterministicCsm => {
                RecipeCost::new(2.1, 16 * 1024 * 1024, 32 * 1024 * 1024, 64)
            }
            Self::SpecializedDeterministic => {
                RecipeCost::new(1.0, 8 * 1024 * 1024, 8 * 1024 * 1024, 0)
            }
            Self::DdgiProbes => RecipeCost::new(2.2, 48 * 1024 * 1024, 96 * 1024 * 1024, 512),
            Self::ScreenProbeGi => RecipeCost::new(1.2, 32 * 1024 * 1024, 32 * 1024 * 1024, 256),
            Self::SurfaceCacheSurfel => {
                RecipeCost::new(1.8, 64 * 1024 * 1024, 80 * 1024 * 1024, 256)
            }
            Self::WorldRestir => RecipeCost::new(1.7, 48 * 1024 * 1024, 64 * 1024 * 1024, 128),
            Self::SsgiScreenSpace => RecipeCost::new(0.8, 24 * 1024 * 1024, 16 * 1024 * 1024, 0),
            Self::SpecularRestir => RecipeCost::new(2.0, 48 * 1024 * 1024, 64 * 1024 * 1024, 128),
            Self::SparseSdfGi => RecipeCost::new(1.4, 32 * 1024 * 1024, 48 * 1024 * 1024, 384),
            Self::RadianceCascades => RecipeCost::new(1.1, 32 * 1024 * 1024, 40 * 1024 * 1024, 320),
            Self::PathTracedOffline => {
                RecipeCost::new(20.0, 128 * 1024 * 1024, 512 * 1024 * 1024, 0)
            }
        }
    }
}

/// Static resource cost used for admission control.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct RecipeCost {
    pub gpu_ms: f32,
    pub bandwidth: u64,
    pub resident_bytes: u64,
    pub io_ops: u32,
}

impl RecipeCost {
    const fn new(gpu_ms: f32, bandwidth: u64, resident_bytes: u64, io_ops: u32) -> Self {
        Self {
            gpu_ms,
            bandwidth,
            resident_bytes,
            io_ops,
        }
    }
}

/// Scene statistics used by the budget compiler.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct SceneStats {
    pub lights: u32,
    pub punctual_max: u32,
    pub npr_materials: u32,
    pub proxy_page_ratio: f32,
    pub triangle_density: f32,
}

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct FrameResourceBudget {
    pub gpu_ms: f32,
    pub bandwidth: u64,
    pub resident_bytes: u64,
    pub io_ops: u32,
    pub shading_rate_budget: f32,
}

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct BudgetPartition {
    pub direct_ms: f32,
    pub indirect_ms: f32,
    pub shadow_ms: f32,
    pub streaming_ms: f32,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Tier {
    Preview,
    Balanced,
    Quality,
    Offline,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct PlanCaps {
    pub ray_tracing: bool,
    pub virtual_shadow: bool,
    pub offline: bool,
    pub deterministic_npr: bool,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Rejection {
    Unsupported(RecipeId),
    Unavailable(RecipeId),
    Budget(RecipeId),
    Determinism(RecipeId),
    VariantGate {
        recipe: RecipeId,
        selected_variants: u32,
        max_per_recipe: u32,
        max_total: u32,
    },
}

/// The machine-readable reason a recipe was omitted from a compiled plan.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum RejectionCode {
    Unsupported,
    Unavailable,
    Budget,
    Determinism,
    VariantGate,
}

/// Structured explanation for one rejected recipe.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct RejectionExplanation {
    pub recipe: RecipeId,
    pub code: RejectionCode,
    pub detail: &'static str,
    pub requested_variants: u32,
    pub variant_gate: Option<VariantGateEvidence>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct VariantGateEvidence {
    pub selected_variants: u32,
    pub max_per_recipe: u32,
    pub max_total: u32,
}

impl Rejection {
    #[must_use]
    pub const fn explanation(self) -> RejectionExplanation {
        let (recipe, code, detail, variant_gate) = match self {
            Self::Unsupported(recipe) => (recipe, RejectionCode::Unsupported, "capability", None),
            Self::Unavailable(recipe) => (recipe, RejectionCode::Unavailable, "backend", None),
            Self::Budget(recipe) => (recipe, RejectionCode::Budget, "resource_budget", None),
            Self::Determinism(recipe) => (recipe, RejectionCode::Determinism, "determinism", None),
            Self::VariantGate {
                recipe,
                selected_variants,
                max_per_recipe,
                max_total,
            } => (
                recipe,
                RejectionCode::VariantGate,
                "variant_gate",
                Some(VariantGateEvidence {
                    selected_variants,
                    max_per_recipe,
                    max_total,
                }),
            ),
        };
        RejectionExplanation {
            recipe,
            code,
            detail,
            requested_variants: recipe.variant_count(),
            variant_gate,
        }
    }
}

/// Admission limits for compiled shader variants.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct VariantGate {
    pub max_per_recipe: u32,
    pub max_total: u32,
}

impl VariantGate {
    pub const DEFAULT: Self = Self {
        max_per_recipe: 4,
        max_total: 8,
    };

    #[must_use]
    pub const fn admits(self, recipe: RecipeId, selected_variants: u32) -> bool {
        recipe.variant_count() <= self.max_per_recipe
            && selected_variants.saturating_add(recipe.variant_count()) <= self.max_total
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct LightingPlan {
    pub tier: Tier,
    pub recipes: Vec<RecipeId>,
    pub partition: BudgetPartition,
    pub budget: FrameResourceBudget,
    pub rejected: Vec<Rejection>,
}

/// Deterministic planner. It is intentionally a small compiler, not a mutable
/// collection of feature toggles: every output is derived from inputs and the
/// rejection list explains every omitted recipe.
pub struct BudgetCompiler;

impl BudgetCompiler {
    #[must_use]
    pub fn compile(
        stats: SceneStats,
        budget: FrameResourceBudget,
        caps: PlanCaps,
        tier: Tier,
    ) -> LightingPlan {
        Self::compile_with_variant_gate(stats, budget, caps, tier, VariantGate::DEFAULT)
    }

    #[must_use]
    pub fn compile_with_variant_gate(
        stats: SceneStats,
        budget: FrameResourceBudget,
        caps: PlanCaps,
        tier: Tier,
        variant_gate: VariantGate,
    ) -> LightingPlan {
        let partition = BudgetPartition {
            direct_ms: budget.gpu_ms * 0.30,
            indirect_ms: budget.gpu_ms * 0.36,
            shadow_ms: budget.gpu_ms * 0.22,
            streaming_ms: budget.gpu_ms * 0.12,
        };
        let mut recipes: Vec<RecipeId> = Vec::new();
        let mut rejected = Vec::new();
        let mut selected_variants = 0;
        for recipe in RecipeId::ALL {
            let unsupported = match recipe {
                RecipeId::DirectStochasticRt | RecipeId::SpecularRestir => !caps.ray_tracing,
                RecipeId::DirectStochasticVsm => !caps.virtual_shadow,
                RecipeId::PathTracedOffline => !caps.offline || !matches!(tier, Tier::Offline),
                _ => false,
            };
            if unsupported {
                rejected.push(Rejection::Unsupported(recipe));
                continue;
            }
            if caps.deterministic_npr
                && matches!(
                    recipe,
                    RecipeId::DirectStochasticRt
                        | RecipeId::DirectStochasticVsm
                        | RecipeId::WorldRestir
                        | RecipeId::SpecularRestir
                )
            {
                rejected.push(Rejection::Determinism(recipe));
                continue;
            }
            if !recipe.has_execution_backend() {
                rejected.push(Rejection::Unavailable(recipe));
                continue;
            }
            if !variant_gate.admits(recipe, selected_variants) {
                rejected.push(Rejection::VariantGate {
                    recipe,
                    selected_variants,
                    max_per_recipe: variant_gate.max_per_recipe,
                    max_total: variant_gate.max_total,
                });
                continue;
            }
            if stats.lights > stats.punctual_max.saturating_mul(8)
                && matches!(recipe, RecipeId::DirectDeterministicCsm)
            {
                rejected.push(Rejection::Budget(recipe));
                continue;
            }
            let used_gpu_ms = recipes
                .iter()
                .map(|selected| selected.cost().gpu_ms)
                .sum::<f32>();
            let used_resident_bytes = recipes
                .iter()
                .map(|selected| selected.cost().resident_bytes)
                .sum::<u64>();
            let used_bandwidth = recipes
                .iter()
                .map(|selected| selected.cost().bandwidth)
                .sum::<u64>();
            let used_io_ops = recipes
                .iter()
                .map(|selected| selected.cost().io_ops)
                .sum::<u32>();
            let cost = recipe.cost();
            if used_gpu_ms + cost.gpu_ms > budget.gpu_ms.max(0.1)
                || used_resident_bytes.saturating_add(cost.resident_bytes) > budget.resident_bytes
                || used_bandwidth.saturating_add(cost.bandwidth) > budget.bandwidth
                || used_io_ops.saturating_add(cost.io_ops) > budget.io_ops
            {
                rejected.push(Rejection::Budget(recipe));
                continue;
            }
            recipes.push(recipe);
            selected_variants = selected_variants.saturating_add(recipe.variant_count());
        }
        if recipes.is_empty() {
            recipes.push(RecipeId::SpecializedDeterministic);
        }
        LightingPlan {
            tier,
            recipes,
            partition,
            budget,
            rejected,
        }
    }
}

impl LightingPlan {
    #[must_use]
    pub fn executions(&self) -> Vec<RecipeExecution> {
        self.recipes
            .iter()
            .enumerate()
            .map(|(index, recipe)| recipe.execution(index as u32, recipe.cost().gpu_ms))
            .collect()
    }

    #[must_use]
    pub fn explain(&self) -> String {
        let selected = self
            .recipes
            .iter()
            .map(|recipe| recipe.name())
            .collect::<Vec<_>>()
            .join(", ");
        let rejected = self
            .rejected
            .iter()
            .map(|item| match item {
                Rejection::Unsupported(recipe) => format_name("unsupported", *recipe),
                Rejection::Unavailable(recipe) => format_name("unavailable", *recipe),
                Rejection::Budget(recipe) => format_name("budget", *recipe),
                Rejection::Determinism(recipe) => format_name("determinism", *recipe),
                Rejection::VariantGate { recipe, .. } => format_name("variant_gate", *recipe),
            })
            .collect::<Vec<_>>()
            .join(", ");
        format!(
            "tier={:?}; gpu_ms={:.3}; selected=[{}]; rejected=[{}]",
            self.tier, self.budget.gpu_ms, selected, rejected
        )
    }

    /// Returns deterministic, machine-readable explanations for every
    /// rejected recipe. The plan remains the source of truth; this method does
    /// not recompile or infer additional GPU capability.
    #[must_use]
    pub fn rejection_explanations(&self) -> Vec<RejectionExplanation> {
        self.rejected
            .iter()
            .map(|rejection| rejection.explanation())
            .collect()
    }

    /// Returns the display metadata for every selected recipe in plan order.
    #[must_use]
    pub fn debug_metadata(&self) -> Vec<(RecipeId, DebugMeta)> {
        self.recipes
            .iter()
            .copied()
            .map(|recipe| (recipe, recipe.debug_meta()))
            .collect()
    }
}

fn format_name(reason: &str, recipe: RecipeId) -> String {
    format!("{}:{}", reason, recipe.name())
}

/// Explicit cache effects keep invalidation visible to graph compilers.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct CacheEffect {
    pub pass: u32,
    pub destroys: u32,
    pub preserves: u32,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum AsyncPolicy {
    Inline,
    Split,
    Join,
}

/// Execution concerns kept separate from graph resource declarations.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ExecPolicy {
    pub pass: u32,
    pub async_policy: AsyncPolicy,
    pub persistent: bool,
    pub shading_rate: f32,
    pub budget_ms: f32,
}

impl fmt::Display for RecipeId {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(self.name())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn journal_is_wal_ordered_and_checkpointable() {
        let mut journal = ChangeJournal::new();
        let epoch = journal.append(4, ChangeFields::GEOMETRY);
        assert_eq!(epoch.geometry, 1);
        journal.append(5, ChangeFields::LIGHTING);
        assert_eq!(journal.records_since(0).len(), 2);
        let snapshot = journal.checkpoint();
        assert_eq!(snapshot.sequence, 2);
        assert!(journal.records_since(0).is_empty());
        assert_eq!(journal.epoch().lighting, 1);
    }

    #[test]
    fn blend_rejects_mixed_targets_and_unbiased_bias() {
        let base = Contrib {
            value: [1.0; 3],
            variance: 0.1,
            bias_bound: 0.0,
            target: TargetId(1),
            weight_semantics: WeightKind::Mis { pdf: 1 },
        };
        let mut other = base;
        other.target = TargetId(2);
        assert_eq!(
            blend(&[base, other], BlendPolicy::Unbiased),
            Err(BlendError::TargetMismatch)
        );
        let mut biased = base;
        biased.bias_bound = 0.2;
        assert_eq!(
            blend(&[biased], BlendPolicy::Unbiased),
            Err(BlendError::BiasRequiresConservative)
        );
    }

    #[test]
    fn conservative_blend_preserves_error_channels() {
        let output = blend(
            &[
                Contrib {
                    value: [1.0, 0.0, 0.0],
                    variance: 0.0,
                    bias_bound: 0.2,
                    target: TargetId(0),
                    weight_semantics: WeightKind::Ratio,
                },
                Contrib {
                    value: [0.0, 1.0, 0.0],
                    variance: 0.0,
                    bias_bound: 0.0,
                    target: TargetId(0),
                    weight_semantics: WeightKind::Ratio,
                },
            ],
            BlendPolicy::Conservative,
        )
        .unwrap();
        assert!(output.bias_bound > 0.0 && output.bias_bound < 0.2);
        assert!(output.value[1] > output.value[0]);
    }

    #[test]
    fn planner_explains_every_recipe_rejection() {
        let plan = BudgetCompiler::compile(
            SceneStats {
                lights: 128,
                punctual_max: 8,
                ..SceneStats::default()
            },
            FrameResourceBudget {
                gpu_ms: 16.6,
                ..FrameResourceBudget::default()
            },
            PlanCaps {
                ray_tracing: false,
                virtual_shadow: false,
                offline: false,
                deterministic_npr: true,
            },
            Tier::Balanced,
        );
        assert!(plan.recipes.contains(&RecipeId::SpecializedDeterministic));
        assert!(plan.explain().contains("unsupported:DirectStochasticRT"));
        assert!(plan.explain().contains("unavailable:DdgiProbes"));
        assert!(plan.explain().contains("determinism:WorldRestir"));
    }

    #[test]
    fn planner_admits_only_within_bandwidth_budget() {
        let plan = BudgetCompiler::compile(
            SceneStats::default(),
            FrameResourceBudget {
                gpu_ms: 100.0,
                bandwidth: 1,
                resident_bytes: u64::MAX,
                io_ops: u32::MAX,
                shading_rate_budget: 1.0,
            },
            PlanCaps {
                ray_tracing: false,
                virtual_shadow: false,
                offline: false,
                deterministic_npr: false,
            },
            Tier::Balanced,
        );
        assert_eq!(plan.recipes, vec![RecipeId::SpecializedDeterministic]);
        assert!(plan.rejected.iter().any(|rejection| matches!(
            rejection,
            Rejection::Unavailable(RecipeId::SsgiScreenSpace)
        )));
    }

    #[test]
    fn planner_admits_world_restir_when_backend_and_budget_are_available() {
        let plan = BudgetCompiler::compile(
            SceneStats::default(),
            FrameResourceBudget {
                gpu_ms: 16.666,
                bandwidth: 512 * 1024 * 1024,
                resident_bytes: 512 * 1024 * 1024,
                io_ops: 4096,
                shading_rate_budget: 1.0,
            },
            PlanCaps {
                ray_tracing: false,
                virtual_shadow: false,
                offline: false,
                deterministic_npr: false,
            },
            Tier::Balanced,
        );
        assert!(plan.recipes.contains(&RecipeId::SpecializedDeterministic));
        assert!(plan.recipes.contains(&RecipeId::WorldRestir));
        assert!(!plan
            .rejected
            .iter()
            .any(|rejection| matches!(rejection, Rejection::Unavailable(RecipeId::WorldRestir))));
    }

    #[test]
    fn planner_rejects_world_restir_for_deterministic_npr() {
        let plan = BudgetCompiler::compile(
            SceneStats::default(),
            FrameResourceBudget::default(),
            PlanCaps {
                ray_tracing: false,
                virtual_shadow: false,
                offline: false,
                deterministic_npr: true,
            },
            Tier::Balanced,
        );
        assert!(!plan.recipes.contains(&RecipeId::WorldRestir));
        assert!(plan
            .rejected
            .iter()
            .any(|rejection| matches!(rejection, Rejection::Determinism(RecipeId::WorldRestir))));
    }

    #[test]
    fn variant_gate_rejects_recipe_with_structured_evidence() {
        let plan = BudgetCompiler::compile_with_variant_gate(
            SceneStats::default(),
            FrameResourceBudget {
                gpu_ms: 16.666,
                bandwidth: 512 * 1024 * 1024,
                resident_bytes: 512 * 1024 * 1024,
                io_ops: 4096,
                shading_rate_budget: 1.0,
            },
            PlanCaps {
                ray_tracing: false,
                virtual_shadow: false,
                offline: false,
                deterministic_npr: false,
            },
            Tier::Balanced,
            VariantGate {
                max_per_recipe: 4,
                max_total: 2,
            },
        );
        let explanation = plan
            .rejection_explanations()
            .into_iter()
            .find(|item| item.code == RejectionCode::VariantGate)
            .expect("the second admitted recipe must hit the total variant gate");
        assert_eq!(explanation.recipe, RecipeId::WorldRestir);
        assert_eq!(explanation.requested_variants, 2);
        assert_eq!(
            explanation.variant_gate,
            Some(VariantGateEvidence {
                selected_variants: 1,
                max_per_recipe: 4,
                max_total: 2,
            })
        );
        assert!(plan.explain().contains("variant_gate:WorldRestir"));
    }

    #[test]
    fn selected_recipe_debug_metadata_is_queryable_without_recompilation() {
        let plan = BudgetCompiler::compile(
            SceneStats::default(),
            FrameResourceBudget {
                gpu_ms: 16.666,
                bandwidth: 512 * 1024 * 1024,
                resident_bytes: 512 * 1024 * 1024,
                io_ops: 4096,
                shading_rate_budget: 1.0,
            },
            PlanCaps {
                ray_tracing: false,
                virtual_shadow: false,
                offline: false,
                deterministic_npr: false,
            },
            Tier::Balanced,
        );
        let metadata = plan.debug_metadata();
        assert_eq!(metadata.len(), plan.recipes.len());
        assert!(metadata.iter().any(|(recipe, meta)| {
            *recipe == RecipeId::SpecializedDeterministic
                && meta.channel == super::super::debug::ChannelSemantics::Radiance
        }));
    }

    struct MockBackend {
        submission: Option<TraceSubmission>,
        next_id: u64,
    }

    impl MockBackend {
        fn new() -> Self {
            Self {
                submission: None,
                next_id: 0,
            }
        }
    }

    impl RayBackend for MockBackend {
        fn capabilities(&self) -> BackendCaps {
            BackendCaps {
                kinds: 0b1_1111,
                max_t_max: 100.0,
                alpha: true,
                is_hardware: false,
                exec: ExecModel::Sync,
                latency: LatencyClass::SameFrame,
                max_batch: 64,
            }
        }

        fn runtime(&self) -> BackendRuntime {
            BackendRuntime {
                inflight: if self.submission.is_some() { 1 } else { 0 },
                queue_depth: 0,
                last_completion_frame: 0,
                error: BackendRuntimeError::Ok,
                resident_bytes: 0,
            }
        }

        fn submit(
            &mut self,
            queries: &[TraceQuery],
            slot_base: u32,
            generation: u64,
            deadline_frame: u64,
        ) -> Result<TraceSubmission, BackendError> {
            if queries.is_empty() || queries.len() > 64 || self.submission.is_some() {
                return Err(BackendError::InvalidBatch);
            }
            self.next_id += 1;
            let submission = TraceSubmission {
                token: SubmissionToken {
                    backend_epoch: 1,
                    id: self.next_id,
                },
                slot_base,
                n: queries.len() as u32,
                generation,
                submitted_frame: 0,
                deadline_frame,
            };
            self.submission = Some(submission);
            Ok(submission)
        }

        fn poll(
            &mut self,
            token: SubmissionToken,
            generation: u64,
            frame: u64,
            out: &mut [Option<RayHit>],
        ) -> PollState {
            let Some(submission) = self.submission else {
                return PollState::Stale;
            };
            if token != submission.token || generation != submission.generation {
                return PollState::Stale;
            }
            if frame > submission.deadline_frame {
                self.submission = None;
                return PollState::Expired;
            }
            for (index, slot) in out.iter_mut().enumerate().take(submission.n as usize) {
                *slot = Some(RayHit {
                    radiance: [index as f32, 0.0, 0.0],
                    depth: 1.0,
                    material: index as u32,
                    bias_bound: 0.0,
                    variance: 0.0,
                });
            }
            self.submission = None;
            PollState::Ready
        }

        fn cancel(&mut self, token: SubmissionToken, generation: u64) -> CancelAck {
            if self.submission.is_some_and(|submission| {
                submission.token == token && submission.generation == generation
            }) {
                self.submission = None;
                CancelAck::Acked
            } else {
                CancelAck::Pending
            }
        }
    }

    #[test]
    fn trace_contract_preserves_slots_and_rejects_stale_generations() {
        let query = TraceQuery {
            kind: TraceKind::Diffuse,
            origin: [0.0; 3],
            direction: [0.0, 1.0, 0.0],
            t_max: 10.0,
        };
        let mut backend = MockBackend::new();
        let submission = backend.submit(&[query, query], 7, 12, 1).unwrap();
        let mut out = [None, None, None];
        assert_eq!(
            backend.poll(submission.token, 11, 0, &mut out),
            PollState::Stale
        );
        assert_eq!(
            backend.poll(submission.token, 12, 0, &mut out),
            PollState::Ready
        );
        assert_eq!(out[0].unwrap().material, 0);
        assert_eq!(out[1].unwrap().material, 1);
        assert!(
            out[2].is_none(),
            "the caller's slot range is never compacted"
        );
    }

    #[test]
    fn trace_contract_expires_after_deadline_and_cancel_ack_releases_submission() {
        let query = TraceQuery {
            kind: TraceKind::Shadow,
            origin: [0.0; 3],
            direction: [0.0, 0.0, 1.0],
            t_max: 4.0,
        };
        let mut backend = MockBackend::new();
        let submission = backend.submit(&[query], 0, 1, 2).unwrap();
        let mut out = [None];
        assert_eq!(
            backend.poll(submission.token, 1, 3, &mut out),
            PollState::Expired
        );
        let submission = backend.submit(&[query], 0, 2, 2).unwrap();
        assert_eq!(backend.cancel(submission.token, 2), CancelAck::Acked);
        assert_eq!(backend.runtime().inflight, 0);
    }
}
