//! Performance budget arbitration and effect-acceptance contracts for the
//! particle subsystem (design §32).
//!
//! Where [`super::lod`] owns the *quality-tier particle-count ladder*
//! ([`super::lod::budget_for_quality`] scales a spawn cap by quality) and
//! [`super::feedback`] owns the *`GPU`->`CPU` counter readback*
//! ([`super::feedback::StatsCounters`] / [`super::feedback::StatsBudget`]),
//! this module is the *frame-budget and acceptance* layer sitting on top of
//! both. It answers three questions that neither sibling owns:
//!
//! 1. **Frame-time budget** — do the per-stage `GPU` costs (simulation, grid
//!    fluid, sort, cull, volumetric shading, draw) fit the single-frame target
//!    of §32? See [`FrameTimeBudget`] and [`FrameTimeReport`].
//! 2. **Triple budget** — the §28 particle-count / `VRAM` / `GPU`-time budgets
//!    accumulated into one [`BudgetLedger`] that reports headroom and overspend
//!    per dimension, consuming (never redefining) the [`super::feedback`]
//!    counters through [`BudgetLedger::charge_stats`].
//! 3. **Effect acceptance** — the §32 acceptance items (`PBR` / `NPR` / hybrid
//!    / temporal / stability) expressed as programmable thresholds that
//!    [`aggregate_acceptance`] folds into an overall pass/fail plus a failure
//!    list.
//!
//! # Design targets, not measurements
//! Every frame-time and `VRAM` number here is a **design target, not a
//! measurement** — the current sandbox has no `GPU`, so real numbers must be
//! calibrated on target hardware through the §33 benchmark loop. The evaluation
//! functions are pure `CPU`-verifiable arithmetic (add / subtract / multiply /
//! divide only, no transcendental functions) so they can be unit-tested today
//! and re-fed with measured samples later.
//!
//! # Signal, not action
//! [`arbitrate`] converts an overspend into a scalar *pressure* signal for the
//! platform degradation layer (§28) to consume. This module deliberately emits
//! only the signal; it never performs a degradation action (dropping spawns,
//! disabling sort/`OIT`, lowering resolution, and so on) itself.

use alloc::vec::Vec;

use super::feedback::StatsCounters;

/// Absolute tolerance guarding `f32` millisecond and error comparisons, so a
/// value exactly on a ceiling is treated as within budget rather than over it,
/// and so divisions never test a raw zero.
pub const BUDGET_EPS: f32 = 1e-6;

// ---------------------------------------------------------------------------
// Per-stage frame-time cost model (§32 desktop-Ultra table)
// ---------------------------------------------------------------------------

/// One accountable `GPU` stage of the single-frame pipeline (design §32).
///
/// The discriminant order matches the §32 desktop-Ultra table rows and is used
/// as the array index for [`FrameTimeBudget`] / [`FrameTimeSample`].
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
#[repr(u8)]
pub enum FrameStage {
    /// Per-particle simulation of the full pool (indirect + persistent-thread).
    Simulation = 0,
    /// Grid-fluid solve for smoke/fire, including pressure projection.
    GridFluid = 1,
    /// Sort of the visible translucent set (one-sweep radix, visible only).
    Sort = 2,
    /// Frustum / distance / `HZB` culling plus bounds reduction.
    CullBounds = 3,
    /// Volumetric six-way shading interpolation (not per-sample ray marching).
    VolumetricShading = 4,
    /// Indirect draw including the `PBR` / `NPR` closure and `OIT` compositing.
    Draw = 5,
}

impl FrameStage {
    /// The number of accountable stages.
    pub const COUNT: usize = 6;

    /// Every stage in table order, for iteration.
    pub const ALL: [FrameStage; Self::COUNT] = [
        FrameStage::Simulation,
        FrameStage::GridFluid,
        FrameStage::Sort,
        FrameStage::CullBounds,
        FrameStage::VolumetricShading,
        FrameStage::Draw,
    ];

    /// The array index for this stage.
    #[must_use]
    pub const fn index(self) -> usize {
        self as usize
    }

    /// A short human-readable label for diagnostics.
    #[must_use]
    pub const fn label(self) -> &'static str {
        match self {
            FrameStage::Simulation => "simulation",
            FrameStage::GridFluid => "grid_fluid",
            FrameStage::Sort => "sort",
            FrameStage::CullBounds => "cull_bounds",
            FrameStage::VolumetricShading => "volumetric_shading",
            FrameStage::Draw => "draw",
        }
    }
}

/// The per-stage frame-time budget in milliseconds (design §32).
///
/// The numbers are **design targets, not measurements** (no `GPU` in the
/// sandbox). [`FrameTimeBudget::total_target_ms`] is stored separately from the
/// per-stage sum because §32 allows a small slack over the raw sum
/// (`~5.5 ms` vs a `5.4 ms` sum): the net increase drops further once the work
/// overlaps the main render asynchronously.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct FrameTimeBudget {
    /// Target milliseconds for each [`FrameStage`], indexed by
    /// [`FrameStage::index`].
    pub stage_targets_ms: [f32; FrameStage::COUNT],
    /// Target milliseconds for the whole frame (the §32 combined ceiling).
    pub total_target_ms: f32,
}

impl FrameTimeBudget {
    /// The desktop-Ultra targets straight from the §32 table.
    pub const ULTRA_DESKTOP: Self = Self {
        stage_targets_ms: [1.0, 1.5, 0.4, 0.2, 0.8, 1.5],
        total_target_ms: 5.5,
    };

    /// The target for one stage, in milliseconds.
    #[must_use]
    pub fn stage_target_ms(&self, stage: FrameStage) -> f32 {
        self.stage_targets_ms[stage.index()]
    }

    /// The sum of the per-stage targets, in milliseconds. This is `<=`
    /// [`FrameTimeBudget::total_target_ms`]; the difference is the async-overlap
    /// slack described in §32.
    #[must_use]
    pub fn stage_sum_ms(&self) -> f32 {
        let mut sum = 0.0;
        for target in self.stage_targets_ms {
            sum += target;
        }
        sum
    }

    /// Evaluates a per-stage timing sample against this budget, reporting
    /// per-stage and total overspend.
    #[must_use]
    pub fn evaluate(&self, sample: &FrameTimeSample) -> FrameTimeReport {
        let mut stage_overspend_ms = [0.0; FrameStage::COUNT];
        for (over, (measured, target)) in stage_overspend_ms
            .iter_mut()
            .zip(sample.stage_ms.iter().zip(self.stage_targets_ms.iter()))
        {
            *over = (measured - target).max(0.0);
        }
        let total_measured_ms = sample.total_ms();
        let total_overspend_ms = (total_measured_ms - self.total_target_ms).max(0.0);
        FrameTimeReport {
            stage_overspend_ms,
            total_measured_ms,
            total_target_ms: self.total_target_ms,
            total_overspend_ms,
        }
    }
}

/// A per-stage frame-time sample in milliseconds — measured on target hardware,
/// or estimated for design alignment when no `GPU` is available.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct FrameTimeSample {
    /// Milliseconds spent in each [`FrameStage`], indexed by
    /// [`FrameStage::index`].
    pub stage_ms: [f32; FrameStage::COUNT],
}

impl FrameTimeSample {
    /// An all-zero sample.
    pub const ZERO: Self = Self {
        stage_ms: [0.0; FrameStage::COUNT],
    };

    /// Builds a sample from an explicit per-stage array.
    #[must_use]
    pub const fn from_stages(stage_ms: [f32; FrameStage::COUNT]) -> Self {
        Self { stage_ms }
    }

    /// Returns a copy with one stage's time replaced (builder-style).
    #[must_use]
    pub fn with_stage(mut self, stage: FrameStage, ms: f32) -> Self {
        self.stage_ms[stage.index()] = ms;
        self
    }

    /// The time recorded for one stage, in milliseconds.
    #[must_use]
    pub fn stage_ms(&self, stage: FrameStage) -> f32 {
        self.stage_ms[stage.index()]
    }

    /// The sum of all stage times, in milliseconds.
    #[must_use]
    pub fn total_ms(&self) -> f32 {
        let mut sum = 0.0;
        for ms in self.stage_ms {
            sum += ms;
        }
        sum
    }
}

/// The verdict of comparing a [`FrameTimeSample`] against a [`FrameTimeBudget`].
///
/// Overspend fields are clamped at `0.0`: a stage or frame that fits its target
/// reports `0.0`, never a negative "underspend".
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct FrameTimeReport {
    /// Milliseconds each stage exceeded its target by (`0.0` if within target).
    pub stage_overspend_ms: [f32; FrameStage::COUNT],
    /// The measured total frame time, in milliseconds.
    pub total_measured_ms: f32,
    /// The total target this was compared against, in milliseconds.
    pub total_target_ms: f32,
    /// Milliseconds the total exceeded [`FrameTimeReport::total_target_ms`] by.
    pub total_overspend_ms: f32,
}

impl FrameTimeReport {
    /// The overspend of one stage, in milliseconds.
    #[must_use]
    pub fn stage_overspend_ms(&self, stage: FrameStage) -> f32 {
        self.stage_overspend_ms[stage.index()]
    }

    /// Whether the whole frame stayed within its total target.
    #[must_use]
    pub fn is_within_total(&self) -> bool {
        self.total_overspend_ms <= BUDGET_EPS
    }

    /// Whether any individual stage exceeded its own target, even if the frame
    /// total stayed within budget (one stage may borrow slack from another).
    #[must_use]
    pub fn any_stage_over(&self) -> bool {
        self.stage_overspend_ms
            .iter()
            .any(|over| *over > BUDGET_EPS)
    }

    /// The total overspend as a fraction of the target (`0.0` when within
    /// budget). A (near) zero target yields `0.0` rather than dividing by zero.
    #[must_use]
    pub fn overspend_ratio(&self) -> f32 {
        if self.total_target_ms <= BUDGET_EPS {
            return 0.0;
        }
        self.total_overspend_ms / self.total_target_ms
    }
}

// ---------------------------------------------------------------------------
// VRAM estimation (§27 attribute quantization)
// ---------------------------------------------------------------------------

/// A per-particle attribute whose enabled set drives the on-demand `VRAM`
/// estimate (design §5 attribute-on-demand, §27 quantization).
///
/// The byte cost of each is the *quantized* footprint from §27, not the
/// unpacked `f32` size: only the attributes a graph actually reads are
/// allocated, and each uses the narrowest encoding that preserves its meaning.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
#[repr(u8)]
pub enum ParticleAttribute {
    /// World/local position, `fp16` x3 (relative-quantized), 6 bytes.
    Position = 0,
    /// Linear velocity, `fp16` x3, 6 bytes.
    Velocity = 1,
    /// Color, `RGBA8`, 4 bytes.
    Color = 2,
    /// Shading normal, oct-encoded, 4 bytes (allocated only for lit models).
    Normal = 3,
    /// Normalized age, `fp16`, 2 bytes.
    Age = 4,
    /// Size/scale, `fp16`, 2 bytes.
    Size = 5,
    /// Rotation angle, `fp16`, 2 bytes.
    Rotation = 6,
    /// A user-authored 32-bit custom attribute, 4 bytes.
    Custom32 = 7,
}

impl ParticleAttribute {
    /// The quantized byte cost of this attribute per particle (design §27).
    #[must_use]
    pub const fn quantized_bytes(self) -> u32 {
        match self {
            ParticleAttribute::Position | ParticleAttribute::Velocity => 6,
            ParticleAttribute::Color | ParticleAttribute::Normal | ParticleAttribute::Custom32 => 4,
            ParticleAttribute::Age | ParticleAttribute::Size | ParticleAttribute::Rotation => 2,
        }
    }
}

/// Estimates the `VRAM` footprint (bytes) of `particle_count` particles that
/// each carry a fixed `bytes_per_particle` payload. Saturating multiply keeps a
/// pathological input from wrapping.
#[must_use]
pub fn estimate_vram_bytes(particle_count: u32, bytes_per_particle: u32) -> u64 {
    u64::from(particle_count).saturating_mul(u64::from(bytes_per_particle))
}

/// Estimates the on-demand, quantized `VRAM` footprint (bytes) of
/// `particle_count` particles carrying exactly the given enabled `attributes`
/// (design §5 + §27). Only listed attributes are charged.
#[must_use]
pub fn estimate_attribute_vram_bytes(particle_count: u32, attributes: &[ParticleAttribute]) -> u64 {
    let mut per_particle: u64 = 0;
    for attribute in attributes {
        per_particle = per_particle.saturating_add(u64::from(attribute.quantized_bytes()));
    }
    per_particle.saturating_mul(u64::from(particle_count))
}

/// Converts a microsecond count (as reported by
/// [`super::feedback::StatsCounters::simulation_micros`]) to milliseconds.
#[must_use]
pub fn micros_to_ms(micros: u32) -> f32 {
    micros as f32 / 1000.0
}

// ---------------------------------------------------------------------------
// Triple budget: particle count / VRAM / GPU time (§28)
// ---------------------------------------------------------------------------

/// The §28 triple budget: a live-particle ceiling, a `VRAM` ceiling in bytes,
/// and a `GPU`-time ceiling in milliseconds. All three are **design targets**.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TripleBudget {
    /// Maximum tolerated live particle count.
    pub max_particles: u32,
    /// Maximum tolerated `VRAM` footprint, in bytes.
    pub max_vram_bytes: u64,
    /// Maximum tolerated `GPU` time per frame, in milliseconds.
    pub max_gpu_ms: f32,
}

impl TripleBudget {
    /// Builds a triple budget whose `GPU`-time ceiling is taken from a
    /// [`FrameTimeBudget`]'s total target, pairing it with count and `VRAM`
    /// ceilings.
    #[must_use]
    pub fn from_frame_time_budget(
        max_particles: u32,
        max_vram_bytes: u64,
        frame_budget: &FrameTimeBudget,
    ) -> Self {
        Self {
            max_particles,
            max_vram_bytes,
            max_gpu_ms: frame_budget.total_target_ms,
        }
    }
}

/// One integer-dimension budget line (used for the particle-count dimension).
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct CountBudgetLine {
    /// How much of the budget is used.
    pub used: u32,
    /// The budget ceiling.
    pub limit: u32,
    /// Remaining headroom (`limit - used`, clamped at `0`).
    pub headroom: u32,
    /// Amount over the ceiling (`used - limit`, clamped at `0`).
    pub overspend: u32,
}

impl CountBudgetLine {
    /// Whether this dimension is over its ceiling.
    #[must_use]
    pub const fn is_over(self) -> bool {
        self.overspend > 0
    }
}

/// One byte-dimension budget line (used for the `VRAM` dimension).
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct BytesBudgetLine {
    /// How many bytes are used.
    pub used: u64,
    /// The byte ceiling.
    pub limit: u64,
    /// Remaining headroom in bytes (`limit - used`, clamped at `0`).
    pub headroom: u64,
    /// Bytes over the ceiling (`used - limit`, clamped at `0`).
    pub overspend: u64,
}

impl BytesBudgetLine {
    /// Whether this dimension is over its ceiling.
    #[must_use]
    pub const fn is_over(self) -> bool {
        self.overspend > 0
    }
}

/// One time-dimension budget line, in milliseconds (used for the `GPU`-time
/// dimension).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TimeBudgetLine {
    /// Milliseconds used.
    pub used_ms: f32,
    /// The millisecond ceiling.
    pub limit_ms: f32,
    /// Remaining headroom in milliseconds (clamped at `0.0`).
    pub headroom_ms: f32,
    /// Milliseconds over the ceiling (clamped at `0.0`).
    pub overspend_ms: f32,
}

impl TimeBudgetLine {
    /// Whether this dimension is over its ceiling.
    #[must_use]
    pub fn is_over(self) -> bool {
        self.overspend_ms > BUDGET_EPS
    }
}

/// A full report of all three [`TripleBudget`] dimensions.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BudgetReport {
    /// The particle-count dimension.
    pub particles: CountBudgetLine,
    /// The `VRAM` dimension.
    pub vram: BytesBudgetLine,
    /// The `GPU`-time dimension.
    pub gpu_time: TimeBudgetLine,
}

impl BudgetReport {
    /// Whether any of the three dimensions is over its ceiling.
    #[must_use]
    pub fn any_over(&self) -> bool {
        self.particles.is_over() || self.vram.is_over() || self.gpu_time.is_over()
    }
}

/// Accumulates the §28 triple budget (particles / `VRAM` / `GPU` time) across
/// contributions and reports per-dimension headroom and overspend.
///
/// Integer dimensions accumulate with saturating arithmetic so a long or
/// pathological accumulation can never wrap. The ledger *consumes* the
/// [`super::feedback`] counters through [`BudgetLedger::charge_stats`] rather
/// than redefining them.
#[derive(Clone, Debug, PartialEq)]
pub struct BudgetLedger {
    budget: TripleBudget,
    particles: u32,
    vram_bytes: u64,
    gpu_ms: f32,
}

impl BudgetLedger {
    /// Creates an empty ledger bound to a triple budget.
    #[must_use]
    pub fn new(budget: TripleBudget) -> Self {
        Self {
            budget,
            particles: 0,
            vram_bytes: 0,
            gpu_ms: 0.0,
        }
    }

    /// Charges live particles against the count budget (saturating).
    pub fn charge_particles(&mut self, count: u32) {
        self.particles = self.particles.saturating_add(count);
    }

    /// Charges bytes against the `VRAM` budget (saturating).
    pub fn charge_vram(&mut self, bytes: u64) {
        self.vram_bytes = self.vram_bytes.saturating_add(bytes);
    }

    /// Charges milliseconds against the `GPU`-time budget. Negative inputs are
    /// ignored so the accumulator never runs backwards.
    pub fn charge_gpu_ms(&mut self, ms: f32) {
        self.gpu_ms += ms.max(0.0);
    }

    /// Charges a [`super::feedback::StatsCounters`] snapshot: its live count
    /// against the particle budget and its simulation time (converted from
    /// microseconds) against the `GPU`-time budget.
    pub fn charge_stats(&mut self, stats: &StatsCounters) {
        self.charge_particles(stats.alive);
        self.charge_gpu_ms(micros_to_ms(stats.simulation_micros));
    }

    /// The particles charged so far.
    #[must_use]
    pub const fn particles(&self) -> u32 {
        self.particles
    }

    /// The `VRAM` bytes charged so far.
    #[must_use]
    pub const fn vram_bytes(&self) -> u64 {
        self.vram_bytes
    }

    /// The `GPU` milliseconds charged so far.
    #[must_use]
    pub const fn gpu_ms(&self) -> f32 {
        self.gpu_ms
    }

    /// Clears all accumulated charges, keeping the configured budget.
    pub fn reset(&mut self) {
        self.particles = 0;
        self.vram_bytes = 0;
        self.gpu_ms = 0.0;
    }

    /// Produces a per-dimension headroom/overspend report.
    #[must_use]
    pub fn report(&self) -> BudgetReport {
        let particles = CountBudgetLine {
            used: self.particles,
            limit: self.budget.max_particles,
            headroom: self.budget.max_particles.saturating_sub(self.particles),
            overspend: self.particles.saturating_sub(self.budget.max_particles),
        };
        let vram = BytesBudgetLine {
            used: self.vram_bytes,
            limit: self.budget.max_vram_bytes,
            headroom: self.budget.max_vram_bytes.saturating_sub(self.vram_bytes),
            overspend: self.vram_bytes.saturating_sub(self.budget.max_vram_bytes),
        };
        let gpu_time = TimeBudgetLine {
            used_ms: self.gpu_ms,
            limit_ms: self.budget.max_gpu_ms,
            headroom_ms: (self.budget.max_gpu_ms - self.gpu_ms).max(0.0),
            overspend_ms: (self.gpu_ms - self.budget.max_gpu_ms).max(0.0),
        };
        BudgetReport {
            particles,
            vram,
            gpu_time,
        }
    }
}

// ---------------------------------------------------------------------------
// Budget arbitration: overspend -> scalar pressure signal (§28)
// ---------------------------------------------------------------------------

/// A per-dimension overspend severity, expressed as the fraction by which a
/// dimension exceeds its ceiling (`0.0` = within budget, `0.5` = 50% over).
///
/// This is a **signal only**: the platform degradation layer (§28) consumes it
/// to decide how hard to degrade. This module performs no degradation action.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BudgetPressure {
    /// Overspend fraction of the particle-count budget.
    pub particle_severity: f32,
    /// Overspend fraction of the `VRAM` budget.
    pub vram_severity: f32,
    /// Overspend fraction of the `GPU`-time budget.
    pub gpu_time_severity: f32,
    /// Overspend fraction of the frame-time total target.
    pub frame_time_severity: f32,
}

impl BudgetPressure {
    /// A no-pressure signal (all dimensions within budget).
    pub const NONE: Self = Self {
        particle_severity: 0.0,
        vram_severity: 0.0,
        gpu_time_severity: 0.0,
        frame_time_severity: 0.0,
    };

    /// The worst severity across all dimensions — the scalar the degradation
    /// layer typically drives its staircase from.
    #[must_use]
    pub fn max_severity(self) -> f32 {
        self.particle_severity
            .max(self.vram_severity)
            .max(self.gpu_time_severity)
            .max(self.frame_time_severity)
    }

    /// Whether any dimension is meaningfully over budget.
    #[must_use]
    pub fn is_over_budget(self) -> bool {
        self.max_severity() > BUDGET_EPS
    }
}

/// Computes the overspend severity of a `u32` dimension as a fraction of its
/// limit. A zero limit reports full pressure (`1.0`) when anything is used.
fn count_severity(overspend: u32, limit: u32) -> f32 {
    if limit == 0 {
        return if overspend > 0 { 1.0 } else { 0.0 };
    }
    overspend as f32 / limit as f32
}

/// Computes the overspend severity of a `u64` dimension as a fraction of its
/// limit. A zero limit reports full pressure (`1.0`) when anything is used.
fn bytes_severity(overspend: u64, limit: u64) -> f32 {
    if limit == 0 {
        return if overspend > 0 { 1.0 } else { 0.0 };
    }
    overspend as f32 / limit as f32
}

/// Computes the overspend severity of a millisecond dimension as a fraction of
/// its limit. A (near) zero limit reports full pressure (`1.0`) when the
/// overspend is meaningful.
fn time_severity(overspend_ms: f32, limit_ms: f32) -> f32 {
    if limit_ms <= BUDGET_EPS {
        return if overspend_ms > BUDGET_EPS { 1.0 } else { 0.0 };
    }
    overspend_ms / limit_ms
}

/// Arbitrates a [`BudgetReport`] and a [`FrameTimeReport`] into a scalar
/// [`BudgetPressure`] signal for the §28 degradation layer.
///
/// This only *measures* overspend and emits the signal; it never chooses or
/// applies a degradation action.
#[must_use]
pub fn arbitrate(report: &BudgetReport, frame: &FrameTimeReport) -> BudgetPressure {
    BudgetPressure {
        particle_severity: count_severity(report.particles.overspend, report.particles.limit),
        vram_severity: bytes_severity(report.vram.overspend, report.vram.limit),
        gpu_time_severity: time_severity(report.gpu_time.overspend_ms, report.gpu_time.limit_ms),
        frame_time_severity: frame.overspend_ratio(),
    }
}

// ---------------------------------------------------------------------------
// Effect acceptance thresholds (§32 acceptance items)
// ---------------------------------------------------------------------------

/// One §32 effect-acceptance category.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
#[repr(u8)]
pub enum AcceptanceCategory {
    /// Physically based smoke/fire: lighting response, edge scatter, self
    /// shadow, black-body ramp, refraction/heat-haze correctness.
    Pbr = 0,
    /// Stylized: clean anti-aliased outlines, banded ramps, resolution-
    /// independent energy `SDF` shapes, stylized shadow/`GI` intake.
    Npr = 1,
    /// Emitter- and particle-level blends: smooth weight-driven transitions and
    /// regressible custom closures.
    Hybrid = 2,
    /// Temporal: fast particles stay crisp under `TAA`/upsampling, static
    /// particles do not jitter (motion vector + reactive mask).
    Temporal = 3,
    /// Stability: capacity overflow, free-list races, event back-pressure, and
    /// compaction correctness never crash; deterministic mode reproduces.
    Stability = 4,
}

impl AcceptanceCategory {
    /// Every acceptance category, for iteration.
    pub const ALL: [AcceptanceCategory; 5] = [
        AcceptanceCategory::Pbr,
        AcceptanceCategory::Npr,
        AcceptanceCategory::Hybrid,
        AcceptanceCategory::Temporal,
        AcceptanceCategory::Stability,
    ];

    /// A short human-readable label for diagnostics.
    #[must_use]
    pub const fn label(self) -> &'static str {
        match self {
            AcceptanceCategory::Pbr => "pbr",
            AcceptanceCategory::Npr => "npr",
            AcceptanceCategory::Hybrid => "hybrid",
            AcceptanceCategory::Temporal => "temporal",
            AcceptanceCategory::Stability => "stability",
        }
    }
}

/// The programmable pass thresholds applied to an [`AcceptanceCheck`].
///
/// `max_numerical_error` bounds a numerical-consistency metric (for example the
/// `CPU`/`GPU` reference divergence), `max_golden_error` bounds a golden-image
/// difference metric, and `require_stability` demands the stability flag pass.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct AcceptanceThresholds {
    /// Maximum tolerated numerical-consistency error.
    pub max_numerical_error: f32,
    /// Maximum tolerated golden-image difference.
    pub max_golden_error: f32,
    /// Whether the stability flag must be `true` to pass.
    pub require_stability: bool,
}

impl AcceptanceThresholds {
    /// A strict default: tight numerical and golden tolerances with stability
    /// required.
    pub const STRICT: Self = Self {
        max_numerical_error: 1e-4,
        max_golden_error: 1e-2,
        require_stability: true,
    };
}

/// One measured acceptance result for a category (design §32).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct AcceptanceCheck {
    /// Which category this result belongs to.
    pub category: AcceptanceCategory,
    /// The measured numerical-consistency error.
    pub numerical_error: f32,
    /// The measured golden-image difference.
    pub golden_error: f32,
    /// Whether the stability sub-checks passed.
    pub stability_pass: bool,
}

impl AcceptanceCheck {
    /// Evaluates this result against the given thresholds, returning `true`
    /// when every applicable threshold is satisfied.
    #[must_use]
    pub fn passes(&self, thresholds: &AcceptanceThresholds) -> bool {
        let numerical_ok = self.numerical_error <= thresholds.max_numerical_error + BUDGET_EPS;
        let golden_ok = self.golden_error <= thresholds.max_golden_error + BUDGET_EPS;
        let stability_ok = !thresholds.require_stability || self.stability_pass;
        numerical_ok && golden_ok && stability_ok
    }
}

/// Which threshold an [`AcceptanceCheck`] failed.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
#[repr(u8)]
pub enum AcceptanceFailureKind {
    /// The numerical-consistency error exceeded its tolerance.
    Numerical = 0,
    /// The golden-image difference exceeded its tolerance.
    Golden = 1,
    /// The stability flag was required but did not pass.
    Stability = 2,
}

/// A single acceptance failure: which category failed which threshold.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct AcceptanceFailure {
    /// The category that failed.
    pub category: AcceptanceCategory,
    /// The threshold that was violated.
    pub kind: AcceptanceFailureKind,
}

/// The aggregate verdict over a set of [`AcceptanceCheck`]s.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct AcceptanceReport {
    /// Whether every check passed (no failures).
    pub passed: bool,
    /// The number of checks evaluated.
    pub checked: u32,
    /// Every individual threshold violation, in evaluation order.
    pub failures: Vec<AcceptanceFailure>,
}

/// Aggregates many [`AcceptanceCheck`]s under one [`AcceptanceThresholds`] into
/// an overall pass/fail plus the list of every violation (design §32).
///
/// An empty check set passes vacuously with an empty failure list.
#[must_use]
pub fn aggregate_acceptance(
    checks: &[AcceptanceCheck],
    thresholds: &AcceptanceThresholds,
) -> AcceptanceReport {
    let mut failures = Vec::new();
    for check in checks {
        if check.numerical_error > thresholds.max_numerical_error + BUDGET_EPS {
            failures.push(AcceptanceFailure {
                category: check.category,
                kind: AcceptanceFailureKind::Numerical,
            });
        }
        if check.golden_error > thresholds.max_golden_error + BUDGET_EPS {
            failures.push(AcceptanceFailure {
                category: check.category,
                kind: AcceptanceFailureKind::Golden,
            });
        }
        if thresholds.require_stability && !check.stability_pass {
            failures.push(AcceptanceFailure {
                category: check.category,
                kind: AcceptanceFailureKind::Stability,
            });
        }
    }
    AcceptanceReport {
        passed: failures.is_empty(),
        checked: checks.len() as u32,
        failures,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Absolute tolerance for comparing `f32` test expectations.
    const TEST_EPS: f32 = 1e-4;

    fn close(a: f32, b: f32) -> bool {
        (a - b).abs() <= TEST_EPS
    }

    // ----- frame-time cost model -------------------------------------------

    #[test]
    fn ultra_desktop_stage_sum_and_total_target() {
        let budget = FrameTimeBudget::ULTRA_DESKTOP;
        assert!(close(budget.stage_sum_ms(), 5.4));
        assert!(close(budget.total_target_ms, 5.5));
        // The stored total leaves async-overlap slack over the raw sum.
        assert!(budget.total_target_ms >= budget.stage_sum_ms());
        assert!(close(budget.stage_target_ms(FrameStage::Simulation), 1.0));
        assert!(close(budget.stage_target_ms(FrameStage::Draw), 1.5));
    }

    #[test]
    fn frame_sample_within_budget_reports_no_overspend() {
        let budget = FrameTimeBudget::ULTRA_DESKTOP;
        let sample = FrameTimeSample::from_stages([0.9, 1.4, 0.3, 0.1, 0.7, 1.4]);
        let report = budget.evaluate(&sample);
        assert!(report.is_within_total());
        assert!(!report.any_stage_over());
        assert!(close(report.total_overspend_ms, 0.0));
        assert!(close(report.overspend_ratio(), 0.0));
    }

    #[test]
    fn frame_sample_over_total_reports_overspend() {
        let budget = FrameTimeBudget::ULTRA_DESKTOP;
        // Every stage a bit hot: total 6.5 ms vs 5.5 ms target.
        let sample = FrameTimeSample::from_stages([1.2, 1.8, 0.5, 0.3, 1.0, 1.7]);
        let report = budget.evaluate(&sample);
        assert!(!report.is_within_total());
        assert!(close(report.total_measured_ms, 6.5));
        assert!(close(report.total_overspend_ms, 1.0));
        assert!(close(report.overspend_ratio(), 1.0 / 5.5));
    }

    #[test]
    fn single_stage_over_detected_even_when_total_fits() {
        let budget = FrameTimeBudget::ULTRA_DESKTOP;
        // Draw borrows slack from cheap stages: total stays under 5.5 ms.
        let sample = FrameTimeSample::ZERO.with_stage(FrameStage::Draw, 2.5);
        let report = budget.evaluate(&sample);
        assert!(report.is_within_total());
        assert!(report.any_stage_over());
        assert!(close(report.stage_overspend_ms(FrameStage::Draw), 1.0));
        assert!(close(report.stage_overspend_ms(FrameStage::Sort), 0.0));
    }

    #[test]
    fn overspend_ratio_guards_zero_target() {
        let report = FrameTimeReport {
            stage_overspend_ms: [0.0; FrameStage::COUNT],
            total_measured_ms: 3.0,
            total_target_ms: 0.0,
            total_overspend_ms: 3.0,
        };
        assert!(close(report.overspend_ratio(), 0.0));
    }

    // ----- VRAM estimation --------------------------------------------------

    #[test]
    fn attribute_vram_estimate_sums_quantized_bytes() {
        // position(6) + color(4) + age(2) = 12 bytes/particle.
        let bytes = estimate_attribute_vram_bytes(
            1_000_000,
            &[
                ParticleAttribute::Position,
                ParticleAttribute::Color,
                ParticleAttribute::Age,
            ],
        );
        assert_eq!(bytes, 12_000_000);
    }

    #[test]
    fn attribute_vram_estimate_empty_set_is_zero() {
        assert_eq!(estimate_attribute_vram_bytes(1_000_000, &[]), 0);
    }

    #[test]
    fn raw_vram_estimate_saturates() {
        let bytes = estimate_vram_bytes(u32::MAX, u32::MAX);
        assert_eq!(bytes, u64::from(u32::MAX) * u64::from(u32::MAX));
    }

    #[test]
    fn micros_convert_to_ms() {
        assert!(close(micros_to_ms(1_500), 1.5));
        assert!(close(micros_to_ms(0), 0.0));
    }

    // ----- triple budget ledger --------------------------------------------

    fn ledger() -> BudgetLedger {
        BudgetLedger::new(TripleBudget {
            max_particles: 1_000_000,
            max_vram_bytes: 32 * 1024 * 1024,
            max_gpu_ms: 5.5,
        })
    }

    #[test]
    fn ledger_reports_headroom_when_within_budget() {
        let mut ledger = ledger();
        ledger.charge_particles(400_000);
        ledger.charge_vram(8 * 1024 * 1024);
        ledger.charge_gpu_ms(3.0);
        let report = ledger.report();
        assert!(!report.any_over());
        assert_eq!(report.particles.headroom, 600_000);
        assert_eq!(report.particles.overspend, 0);
        assert_eq!(report.vram.headroom, 24 * 1024 * 1024);
        assert!(close(report.gpu_time.headroom_ms, 2.5));
        assert!(!report.gpu_time.is_over());
    }

    #[test]
    fn ledger_reports_overspend_per_dimension() {
        let mut ledger = ledger();
        ledger.charge_particles(1_200_000);
        ledger.charge_vram(40 * 1024 * 1024);
        ledger.charge_gpu_ms(7.0);
        let report = ledger.report();
        assert!(report.any_over());
        assert_eq!(report.particles.overspend, 200_000);
        assert_eq!(report.particles.headroom, 0);
        assert_eq!(report.vram.overspend, 8 * 1024 * 1024);
        assert!(close(report.gpu_time.overspend_ms, 1.5));
        assert!(report.gpu_time.is_over());
    }

    #[test]
    fn ledger_charges_stats_counters() {
        let mut ledger = ledger();
        ledger.charge_stats(&StatsCounters {
            alive: 500_000,
            spawned: 10_000,
            killed: 5_000,
            overflow: 0,
            simulation_micros: 2_500,
        });
        assert_eq!(ledger.particles(), 500_000);
        assert!(close(ledger.gpu_ms(), 2.5));
    }

    #[test]
    fn ledger_accumulates_and_saturates() {
        let mut ledger = BudgetLedger::new(TripleBudget {
            max_particles: 10,
            max_vram_bytes: 10,
            max_gpu_ms: 1.0,
        });
        ledger.charge_particles(u32::MAX);
        ledger.charge_particles(100);
        assert_eq!(ledger.particles(), u32::MAX);
        ledger.charge_gpu_ms(-5.0);
        assert!(close(ledger.gpu_ms(), 0.0));
    }

    #[test]
    fn ledger_reset_clears_charges() {
        let mut ledger = ledger();
        ledger.charge_particles(100);
        ledger.charge_vram(100);
        ledger.charge_gpu_ms(1.0);
        ledger.reset();
        assert_eq!(ledger.particles(), 0);
        assert_eq!(ledger.vram_bytes(), 0);
        assert!(close(ledger.gpu_ms(), 0.0));
    }

    #[test]
    fn triple_budget_from_frame_budget_uses_total_target() {
        let budget = TripleBudget::from_frame_time_budget(
            500_000,
            16 * 1024 * 1024,
            &FrameTimeBudget::ULTRA_DESKTOP,
        );
        assert_eq!(budget.max_particles, 500_000);
        assert!(close(budget.max_gpu_ms, 5.5));
    }

    // ----- arbitration ------------------------------------------------------

    #[test]
    fn arbitrate_within_budget_has_no_pressure() {
        let mut ledger = ledger();
        ledger.charge_particles(100_000);
        ledger.charge_gpu_ms(2.0);
        let report = ledger.report();
        let frame = FrameTimeBudget::ULTRA_DESKTOP.evaluate(&FrameTimeSample::ZERO);
        let pressure = arbitrate(&report, &frame);
        assert!(!pressure.is_over_budget());
        assert!(close(pressure.max_severity(), 0.0));
    }

    #[test]
    fn arbitrate_reports_particle_severity() {
        let mut ledger = ledger();
        ledger.charge_particles(1_500_000); // 50% over a 1,000,000 ceiling.
        let report = ledger.report();
        let frame = FrameTimeBudget::ULTRA_DESKTOP.evaluate(&FrameTimeSample::ZERO);
        let pressure = arbitrate(&report, &frame);
        assert!(pressure.is_over_budget());
        assert!(close(pressure.particle_severity, 0.5));
        assert!(close(pressure.max_severity(), 0.5));
    }

    #[test]
    fn arbitrate_takes_worst_of_time_and_frame() {
        let mut ledger = ledger();
        ledger.charge_gpu_ms(11.0); // 100% over the 5.5 ms ceiling.
        let report = ledger.report();
        let sample = FrameTimeSample::from_stages([1.2, 1.8, 0.5, 0.3, 1.0, 1.7]); // 6.5 ms.
        let frame = FrameTimeBudget::ULTRA_DESKTOP.evaluate(&sample);
        let pressure = arbitrate(&report, &frame);
        assert!(close(pressure.gpu_time_severity, 1.0));
        assert!(close(pressure.frame_time_severity, 1.0 / 5.5));
        assert!(close(pressure.max_severity(), 1.0));
    }

    #[test]
    fn arbitrate_zero_limit_reports_full_pressure() {
        assert!(close(count_severity(1, 0), 1.0));
        assert!(close(count_severity(0, 0), 0.0));
        assert!(close(bytes_severity(1, 0), 1.0));
        assert!(close(time_severity(1.0, 0.0), 1.0));
        assert!(close(time_severity(0.0, 0.0), 0.0));
    }

    // ----- acceptance thresholds -------------------------------------------

    fn passing_check(category: AcceptanceCategory) -> AcceptanceCheck {
        AcceptanceCheck {
            category,
            numerical_error: 1e-5,
            golden_error: 1e-3,
            stability_pass: true,
        }
    }

    #[test]
    fn acceptance_all_pass_yields_empty_failures() {
        let checks: Vec<AcceptanceCheck> = AcceptanceCategory::ALL
            .iter()
            .map(|c| passing_check(*c))
            .collect();
        let report = aggregate_acceptance(&checks, &AcceptanceThresholds::STRICT);
        assert!(report.passed);
        assert_eq!(report.checked, 5);
        assert!(report.failures.is_empty());
    }

    #[test]
    fn acceptance_numerical_failure_is_listed() {
        let mut check = passing_check(AcceptanceCategory::Pbr);
        check.numerical_error = 1.0;
        let report = aggregate_acceptance(&[check], &AcceptanceThresholds::STRICT);
        assert!(!report.passed);
        assert_eq!(
            report.failures,
            alloc::vec![AcceptanceFailure {
                category: AcceptanceCategory::Pbr,
                kind: AcceptanceFailureKind::Numerical,
            }]
        );
        assert!(!check.passes(&AcceptanceThresholds::STRICT));
    }

    #[test]
    fn acceptance_collects_multiple_failures_per_check() {
        let check = AcceptanceCheck {
            category: AcceptanceCategory::Stability,
            numerical_error: 1e-5,
            golden_error: 1.0,
            stability_pass: false,
        };
        let report = aggregate_acceptance(&[check], &AcceptanceThresholds::STRICT);
        assert!(!report.passed);
        assert_eq!(report.failures.len(), 2);
        assert!(report.failures.contains(&AcceptanceFailure {
            category: AcceptanceCategory::Stability,
            kind: AcceptanceFailureKind::Golden,
        }));
        assert!(report.failures.contains(&AcceptanceFailure {
            category: AcceptanceCategory::Stability,
            kind: AcceptanceFailureKind::Stability,
        }));
    }

    #[test]
    fn acceptance_stability_optional_when_not_required() {
        let thresholds = AcceptanceThresholds {
            require_stability: false,
            ..AcceptanceThresholds::STRICT
        };
        let check = AcceptanceCheck {
            category: AcceptanceCategory::Temporal,
            numerical_error: 1e-5,
            golden_error: 1e-3,
            stability_pass: false,
        };
        assert!(check.passes(&thresholds));
        let report = aggregate_acceptance(&[check], &thresholds);
        assert!(report.passed);
    }

    #[test]
    fn acceptance_empty_set_passes_vacuously() {
        let report = aggregate_acceptance(&[], &AcceptanceThresholds::STRICT);
        assert!(report.passed);
        assert_eq!(report.checked, 0);
        assert!(report.failures.is_empty());
    }

    #[test]
    fn acceptance_at_tolerance_boundary_passes() {
        let check = AcceptanceCheck {
            category: AcceptanceCategory::Npr,
            numerical_error: AcceptanceThresholds::STRICT.max_numerical_error,
            golden_error: AcceptanceThresholds::STRICT.max_golden_error,
            stability_pass: true,
        };
        assert!(check.passes(&AcceptanceThresholds::STRICT));
    }
}
