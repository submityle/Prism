//! Dual-backend simulation contract: choosing and planning the `CPU` or `GPU`
//! path from a single graph `IR` (design §24).
//!
//! A compiled particle graph is backend-agnostic: the same iteration domains and
//! stage order must produce the same effect whether the per-frame update runs as
//! a `GPU` compute dispatch or as a `CPU` job graph. This module owns three
//! contracts that make that dual path well-defined without pulling in a device:
//!
//! 1. **Selection** — [`choose_backend`] turns measured [`BackendCapabilities`]
//!    and a [`SimulationRequirement`] into a [`SimBackend`] through an explicit,
//!    ordered decision matrix. `GPU` is the default for large offline-style
//!    workloads; the `CPU` path is the fallback when there is no compute support
//!    and the interactive choice when a small workload needs per-frame `CPU`
//!    gameplay callbacks (pickup queries, hit reactions).
//! 2. **Planning** — [`BackendPlan`] describes *how* the chosen backend schedules
//!    the work: the `GPU` variant derives a workgroup count and an indirect /
//!    persistent-thread dispatch strategy; the `CPU` variant derives a
//!    Structure-of-Arrays (`SoA`) chunk count for a `bevy_tasks`-style parallel
//!    sweep. Both are pure integer derivations with explicit divide-by-zero and
//!    round-up guards.
//! 3. **Parity** — [`SemanticParity`] declares the invariants both backends must
//!    honor (same seed, same `dt`, same stage order imply the same result, per
//!    design §29) and provides a tolerance-based consistency check over per-frame
//!    output digests. The real deterministic `RNG` lives in §29; this module only
//!    owns the contract and the comparison tooling.
//!
//! Only ordinary integer / floating-point arithmetic (and, transitively through
//! [`super::Vec3`], `sqrt`) is used — no transcendental functions — so the
//! contract stays reproducible against a future `GPU` kernel.

use super::{IterationDomain, Vec3};

/// Absolute tolerance, in world units, below which two position samples from the
/// `CPU` and `GPU` backends are treated as identical for a parity check.
///
/// A small non-zero value absorbs the last-bit differences between a scalar
/// `CPU` reference and a vectorized `GPU` kernel while still catching real
/// divergence (design §29).
pub const DEFAULT_PARITY_POSITION_EPS: f32 = 1.0e-4;

/// Absolute tolerance, in world units per second, for a velocity parity sample.
pub const DEFAULT_PARITY_VELOCITY_EPS: f32 = 1.0e-4;

/// Absolute tolerance for a generic scalar parity sample (age, size, and other
/// per-particle scalars summarized into a [`FrameDigest`]).
pub const DEFAULT_PARITY_SCALAR_EPS: f32 = 1.0e-4;

/// The largest particle count that still counts as a *small* workload for the
/// interactive `CPU`-selection rule in [`choose_backend`].
///
/// At or below this many live particles the round-trip cost of reading results
/// back for `CPU` gameplay logic is cheaper than driving a `GPU` dispatch, so an
/// interactive emitter is scheduled on the `CPU` path. Above it the `GPU` path
/// wins even when interaction is requested, and the interaction is serviced
/// through an asynchronous read-back instead.
pub const SMALL_INTERACTIVE_PARTICLE_MAX: u32 = 4_096;

/// Which simulation backend advances an emitter's per-frame update (design §24).
#[derive(Clone, Copy, Debug, Default, Eq, Hash, PartialEq)]
pub enum SimBackend {
    /// The `GPU` compute path: the scalable default for large emitters.
    #[default]
    Gpu,
    /// The `CPU` job-graph path: the fallback when compute is unavailable and
    /// the preferred path for small interactive emitters.
    Cpu,
}

impl SimBackend {
    /// Returns `true` when this is the `GPU` compute backend.
    #[must_use]
    pub fn is_gpu(self) -> bool {
        matches!(self, SimBackend::Gpu)
    }

    /// Returns `true` when this is the `CPU` job-graph backend.
    #[must_use]
    pub fn is_cpu(self) -> bool {
        matches!(self, SimBackend::Cpu)
    }
}

/// The class of device an effect is being compiled for (design §24, §28).
///
/// The platform does not select the backend on its own — measured
/// [`BackendCapabilities`] do — but it drives the capability presets in
/// [`BackendCapabilities::for_platform`] and records the intended target on a
/// [`SimulationRequirement`].
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum TargetPlatform {
    /// A discrete-`GPU` desktop with full compute and indirect dispatch.
    Desktop,
    /// A fixed-hardware console: full compute, often a dedicated compute queue.
    Console,
    /// A mobile tile-based `GPU`: compute is present but limited.
    Mobile,
    /// A `WebGPU` target: compute is present but conservative on limits.
    Web,
    /// A software / headless target with no compute device at all.
    Headless,
}

/// The compute capabilities of the device the effect will run on (design §24).
///
/// These are treated as *measured* limits: [`choose_backend`] and
/// [`BackendPlan`] read them rather than assuming a platform. A missing compute
/// device (`supports_compute == false`) forces the `CPU` path.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct BackendCapabilities {
    /// Whether the device can run compute shaders at all. When `false` the only
    /// viable backend is [`SimBackend::Cpu`].
    pub supports_compute: bool,
    /// Whether indirect dispatch (`dispatch_workgroups_indirect`) is available,
    /// letting the workgroup count be produced on-device from a live particle
    /// count instead of being fixed on the `CPU`.
    pub supports_indirect_dispatch: bool,
    /// Whether a dedicated asynchronous compute queue exists, so simulation can
    /// overlap graphics work rather than serialize behind it.
    pub has_dedicated_compute_queue: bool,
    /// The maximum invocations per workgroup along the primary dimension (the
    /// `GPU`'s per-workgroup thread budget).
    pub max_workgroup_size: u32,
    /// The maximum number of workgroups a single dispatch may launch along the
    /// primary dimension; larger totals must be split across dispatches or
    /// covered by a persistent-thread grid-stride loop.
    pub max_workgroups_per_dispatch: u32,
}

impl BackendCapabilities {
    /// A conservative "no compute device" capability set that forces the `CPU`
    /// path.
    #[must_use]
    pub const fn cpu_only() -> Self {
        Self {
            supports_compute: false,
            supports_indirect_dispatch: false,
            has_dedicated_compute_queue: false,
            max_workgroup_size: 0,
            max_workgroups_per_dispatch: 0,
        }
    }

    /// A full-featured discrete-`GPU` capability set (indirect dispatch, a
    /// dedicated compute queue, and generous limits).
    #[must_use]
    pub const fn desktop_gpu() -> Self {
        Self {
            supports_compute: true,
            supports_indirect_dispatch: true,
            has_dedicated_compute_queue: true,
            max_workgroup_size: 256,
            max_workgroups_per_dispatch: 65_535,
        }
    }

    /// The default capability preset for a [`TargetPlatform`].
    ///
    /// These are starting points a real backend refines with queried device
    /// limits; the selection and planning logic never assumes them, it reads the
    /// concrete values.
    #[must_use]
    pub const fn for_platform(platform: TargetPlatform) -> Self {
        match platform {
            TargetPlatform::Desktop | TargetPlatform::Console => Self {
                supports_compute: true,
                supports_indirect_dispatch: true,
                has_dedicated_compute_queue: true,
                max_workgroup_size: 256,
                max_workgroups_per_dispatch: 65_535,
            },
            TargetPlatform::Mobile => Self {
                supports_compute: true,
                supports_indirect_dispatch: false,
                has_dedicated_compute_queue: false,
                max_workgroup_size: 128,
                max_workgroups_per_dispatch: 65_535,
            },
            TargetPlatform::Web => Self {
                supports_compute: true,
                supports_indirect_dispatch: true,
                has_dedicated_compute_queue: false,
                max_workgroup_size: 128,
                max_workgroups_per_dispatch: 65_535,
            },
            TargetPlatform::Headless => Self::cpu_only(),
        }
    }
}

/// What one emitter needs from its simulation this frame (design §24, §28).
///
/// This is the demand side of the selection decision: how many particles, and
/// whether the effect needs `CPU`-side interaction or bit-reproducible results.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct SimulationRequirement {
    /// The live particle count driving the update (the workload "scale").
    pub particle_count: u32,
    /// Whether per-frame `CPU` gameplay logic must read or mutate particles
    /// (pickup queries, hit callbacks, spawn-from-collision). This is the
    /// interaction that biases a small workload toward the `CPU` path.
    pub needs_cpu_interaction: bool,
    /// Whether the effect requires bit-reproducible results (replays, netcode).
    ///
    /// Because both backends draw from the same stateless hash `RNG` (design
    /// §29), determinism does *not* force the `CPU` path; it is recorded so a
    /// [`SemanticParity`] check can be demanded downstream.
    pub needs_determinism: bool,
    /// The platform the effect is compiled for (informational for selection;
    /// see [`BackendCapabilities::for_platform`]).
    pub target_platform: TargetPlatform,
}

impl SimulationRequirement {
    /// Builds a requirement for a non-interactive, non-deterministic emitter of
    /// `particle_count` particles on `target_platform` — the common bulk-`VFX`
    /// case.
    #[must_use]
    pub const fn bulk(particle_count: u32, target_platform: TargetPlatform) -> Self {
        Self {
            particle_count,
            needs_cpu_interaction: false,
            needs_determinism: false,
            target_platform,
        }
    }
}

/// The reason [`choose_backend_explained`] arrived at its decision.
///
/// Exposing the reason keeps the decision matrix auditable: a tool can show why
/// an emitter fell back to the `CPU` path instead of only *that* it did.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum BackendSelectionReason {
    /// The device reported no compute support, so the `CPU` path is forced.
    NoComputeSupport,
    /// A small workload requested `CPU` interaction, so the `CPU` path wins.
    SmallInteractiveWorkload,
    /// The default scalable choice: the `GPU` compute path.
    LargeScaleGpu,
}

/// A backend decision paired with the rule that produced it.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct BackendDecision {
    /// The chosen backend.
    pub backend: SimBackend,
    /// The rule that selected it.
    pub reason: BackendSelectionReason,
}

/// Chooses a backend from device capabilities and an emitter's requirement,
/// also reporting the deciding rule (design §24).
///
/// The decision matrix is evaluated in strict priority order:
///
/// 1. **No compute** — if the device cannot run compute shaders, the `CPU` path
///    is the only option ([`BackendSelectionReason::NoComputeSupport`]).
/// 2. **Small + interactive** — if the effect needs per-frame `CPU` interaction
///    and its particle count is at or below `small_interactive_max`, the `CPU`
///    path avoids a read-back round-trip
///    ([`BackendSelectionReason::SmallInteractiveWorkload`]).
/// 3. **Otherwise** — the scalable `GPU` compute path
///    ([`BackendSelectionReason::LargeScaleGpu`]).
///
/// Determinism never forces the `CPU` path: both backends agree by construction
/// (design §29), so a deterministic large emitter still runs on the `GPU`.
#[must_use]
pub fn choose_backend_with_threshold(
    caps: BackendCapabilities,
    requirement: SimulationRequirement,
    small_interactive_max: u32,
) -> BackendDecision {
    if !caps.supports_compute {
        return BackendDecision {
            backend: SimBackend::Cpu,
            reason: BackendSelectionReason::NoComputeSupport,
        };
    }
    if requirement.needs_cpu_interaction && requirement.particle_count <= small_interactive_max {
        return BackendDecision {
            backend: SimBackend::Cpu,
            reason: BackendSelectionReason::SmallInteractiveWorkload,
        };
    }
    BackendDecision {
        backend: SimBackend::Gpu,
        reason: BackendSelectionReason::LargeScaleGpu,
    }
}

/// Chooses a backend using the default [`SMALL_INTERACTIVE_PARTICLE_MAX`]
/// threshold, reporting the deciding rule (design §24).
#[must_use]
pub fn choose_backend_explained(
    caps: BackendCapabilities,
    requirement: SimulationRequirement,
) -> BackendDecision {
    choose_backend_with_threshold(caps, requirement, SMALL_INTERACTIVE_PARTICLE_MAX)
}

/// Chooses a backend using the default threshold, returning just the backend
/// (design §24).
///
/// Use [`choose_backend_explained`] when the deciding rule is also needed.
#[must_use]
pub fn choose_backend(caps: BackendCapabilities, requirement: SimulationRequirement) -> SimBackend {
    choose_backend_explained(caps, requirement).backend
}

/// Round `numerator` up to the next multiple of `divisor`, guarding a zero
/// divisor.
///
/// Returns `0` when `divisor == 0` (a degenerate configuration such as a zero
/// workgroup size or chunk size) so a plan collapses to a well-defined no-op
/// instead of dividing by zero. Uses [`u32::div_ceil`] to round up without the
/// overflow a manual `(n + d - 1) / d` would risk near [`u32::MAX`].
#[must_use]
fn ceil_div_guarded(numerator: u32, divisor: u32) -> u32 {
    if divisor == 0 {
        0
    } else {
        numerator.div_ceil(divisor)
    }
}

/// Tunable knobs shared by both backend plans (design §24).
///
/// These are authoring / profile defaults a real backend may override; the plan
/// derivation clamps them against the measured [`BackendCapabilities`].
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct BackendTuning {
    /// The requested invocations per `GPU` workgroup along the primary axis
    /// (clamped down to [`BackendCapabilities::max_workgroup_size`]).
    pub gpu_workgroup_size: u32,
    /// Whether the `GPU` path prefers indirect dispatch (honored only when the
    /// device supports it).
    pub gpu_use_indirect: bool,
    /// Whether the `GPU` path uses a persistent-thread grid-stride kernel: a
    /// single capped dispatch whose workgroups loop over all elements, instead
    /// of launching one workgroup per element slice.
    pub gpu_persistent_threads: bool,
    /// The number of particles one `CPU` chunk (job) processes over the `SoA`
    /// arrays.
    pub cpu_chunk_size: u32,
    /// Whether the `CPU` path fans chunks out across a `bevy_tasks`-style pool
    /// (only meaningful when there is more than one chunk).
    pub cpu_parallel: bool,
}

impl Default for BackendTuning {
    fn default() -> Self {
        Self {
            gpu_workgroup_size: 64,
            gpu_use_indirect: true,
            gpu_persistent_threads: false,
            cpu_chunk_size: 256,
            cpu_parallel: true,
        }
    }
}

/// How the `GPU` compute path dispatches an iteration domain (design §24).
///
/// All counts are derived by rounding the element count up against the
/// workgroup size and the per-dispatch workgroup limit. When the total exceeds
/// one dispatch's limit the work is split across [`dispatch_count`] dispatches,
/// unless [`persistent_threads`] is set, in which case a single capped dispatch
/// covers everything through a grid-stride loop.
///
/// [`dispatch_count`]: GpuDispatchPlan::dispatch_count
/// [`persistent_threads`]: GpuDispatchPlan::persistent_threads
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct GpuDispatchPlan {
    /// The iteration domain this dispatch covers.
    pub domain: IterationDomain,
    /// The number of elements to process (particles, cells, voxels, ...).
    pub element_count: u32,
    /// The invocations per workgroup actually used (clamped to device limits).
    pub workgroup_size: u32,
    /// The logical number of workgroups needed to cover every element,
    /// `ceil(element_count / workgroup_size)`.
    pub workgroup_count: u32,
    /// The workgroups launched per dispatch (capped at the device limit).
    pub workgroups_per_dispatch: u32,
    /// How many dispatches cover the whole domain (`1` for a persistent-thread
    /// kernel, or when the logical count fits one dispatch).
    pub dispatch_count: u32,
    /// Whether the workgroup count is fed to the device indirectly.
    pub use_indirect: bool,
    /// Whether a single capped persistent-thread dispatch grid-strides over all
    /// elements.
    pub persistent_threads: bool,
}

impl GpuDispatchPlan {
    /// Derives a dispatch plan for `element_count` elements of `domain` from the
    /// device `caps` and the authoring `tuning`.
    ///
    /// The workgroup size is clamped to [`BackendCapabilities::max_workgroup_size`]
    /// and indirect dispatch is enabled only when the device supports it. A zero
    /// element count (or a zero effective workgroup size) yields a no-op plan.
    #[must_use]
    pub fn derive(
        caps: BackendCapabilities,
        tuning: BackendTuning,
        domain: IterationDomain,
        element_count: u32,
    ) -> Self {
        let workgroup_size = tuning.gpu_workgroup_size.min(caps.max_workgroup_size);
        let use_indirect = tuning.gpu_use_indirect && caps.supports_indirect_dispatch;
        // A zero device limit is degenerate; fall back to one workgroup per
        // dispatch so the split arithmetic stays well-defined.
        let max_per_dispatch = caps.max_workgroups_per_dispatch.max(1);
        let workgroup_count = ceil_div_guarded(element_count, workgroup_size);

        let (workgroups_per_dispatch, dispatch_count) = if workgroup_count == 0 {
            (0, 0)
        } else if tuning.gpu_persistent_threads {
            (workgroup_count.min(max_per_dispatch), 1)
        } else {
            (
                workgroup_count.min(max_per_dispatch),
                ceil_div_guarded(workgroup_count, max_per_dispatch),
            )
        };

        Self {
            domain,
            element_count,
            workgroup_size,
            workgroup_count,
            workgroups_per_dispatch,
            dispatch_count,
            use_indirect,
            persistent_threads: tuning.gpu_persistent_threads,
        }
    }

    /// Returns `true` when the plan schedules no work (no elements, or a
    /// degenerate zero workgroup size).
    #[must_use]
    pub fn is_no_op(&self) -> bool {
        self.workgroup_count == 0 || self.dispatch_count == 0
    }

    /// Returns `true` when the launched workgroups reach every element.
    ///
    /// For a fixed-grid plan this checks that the dispatched workgroups times the
    /// workgroup size covers the element count; a persistent-thread plan always
    /// covers its domain by construction (its grid-stride loop revisits slices).
    #[must_use]
    pub fn covers_all_elements(&self) -> bool {
        if self.is_no_op() {
            return self.element_count == 0;
        }
        if self.persistent_threads {
            return true;
        }
        let covered = self
            .dispatch_count
            .saturating_mul(self.workgroups_per_dispatch)
            .saturating_mul(self.workgroup_size);
        covered >= self.element_count
    }
}

/// How the `CPU` job-graph path chunks an iteration domain (design §24).
///
/// The `SoA` particle arrays are swept in fixed-size chunks so a
/// `bevy_tasks`-style pool can process disjoint chunks in parallel. The chunk
/// count is `ceil(element_count / chunk_size)`, guarded against a zero chunk
/// size.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct CpuChunkPlan {
    /// The iteration domain this sweep covers.
    pub domain: IterationDomain,
    /// The number of elements to process.
    pub element_count: u32,
    /// The number of elements one chunk (job) processes.
    pub chunk_size: u32,
    /// The number of chunks, `ceil(element_count / chunk_size)`.
    pub chunk_count: u32,
    /// Whether the chunks are fanned out in parallel (only when more than one
    /// chunk exists and parallelism was requested).
    pub parallel: bool,
}

impl CpuChunkPlan {
    /// Derives a chunk plan for `element_count` elements of `domain` from the
    /// authoring `tuning`.
    ///
    /// A zero element count (or a zero chunk size) yields a no-op plan, and
    /// parallel fan-out is disabled unless there is more than one chunk.
    #[must_use]
    pub fn derive(tuning: BackendTuning, domain: IterationDomain, element_count: u32) -> Self {
        let chunk_size = tuning.cpu_chunk_size;
        let chunk_count = ceil_div_guarded(element_count, chunk_size);
        let parallel = tuning.cpu_parallel && chunk_count > 1;
        Self {
            domain,
            element_count,
            chunk_size,
            chunk_count,
            parallel,
        }
    }

    /// Returns `true` when the plan schedules no work.
    #[must_use]
    pub fn is_no_op(&self) -> bool {
        self.chunk_count == 0
    }

    /// Returns `true` when the chunks cover every element.
    #[must_use]
    pub fn covers_all_elements(&self) -> bool {
        if self.is_no_op() {
            return self.element_count == 0;
        }
        self.chunk_count.saturating_mul(self.chunk_size) >= self.element_count
    }
}

/// The scheduling plan for one iteration domain on the chosen backend
/// (design §24).
///
/// The same graph `IR` produces either variant: [`BackendPlan::derive`] routes
/// on the [`SimBackend`] so a caller derives a plan without matching by hand.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum BackendPlan {
    /// A `GPU` compute dispatch plan.
    Gpu(GpuDispatchPlan),
    /// A `CPU` chunked-sweep plan.
    Cpu(CpuChunkPlan),
}

impl BackendPlan {
    /// Derives the plan for `backend` from the device `caps`, authoring
    /// `tuning`, iteration `domain`, and `element_count` (design §24).
    #[must_use]
    pub fn derive(
        backend: SimBackend,
        caps: BackendCapabilities,
        tuning: BackendTuning,
        domain: IterationDomain,
        element_count: u32,
    ) -> Self {
        match backend {
            SimBackend::Gpu => {
                BackendPlan::Gpu(GpuDispatchPlan::derive(caps, tuning, domain, element_count))
            }
            SimBackend::Cpu => {
                BackendPlan::Cpu(CpuChunkPlan::derive(tuning, domain, element_count))
            }
        }
    }

    /// Returns `true` when the plan schedules no work on either backend.
    #[must_use]
    pub fn is_no_op(&self) -> bool {
        match self {
            BackendPlan::Gpu(plan) => plan.is_no_op(),
            BackendPlan::Cpu(plan) => plan.is_no_op(),
        }
    }

    /// Returns the backend this plan targets.
    #[must_use]
    pub fn backend(&self) -> SimBackend {
        match self {
            BackendPlan::Gpu(_) => SimBackend::Gpu,
            BackendPlan::Cpu(_) => SimBackend::Cpu,
        }
    }
}

/// The absolute tolerances a parity check allows between the two backends'
/// per-frame output summaries (design §29).
///
/// Contains floating-point fields, so it derives [`PartialEq`] rather than
/// [`Eq`]; parity comparisons themselves use these epsilons, never `==`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ParityTolerance {
    /// Position tolerance in world units.
    pub position_eps: f32,
    /// Velocity tolerance in world units per second.
    pub velocity_eps: f32,
    /// Generic scalar tolerance.
    pub scalar_eps: f32,
}

impl Default for ParityTolerance {
    fn default() -> Self {
        Self {
            position_eps: DEFAULT_PARITY_POSITION_EPS,
            velocity_eps: DEFAULT_PARITY_VELOCITY_EPS,
            scalar_eps: DEFAULT_PARITY_SCALAR_EPS,
        }
    }
}

/// A compact, order-independent summary of one backend's output for a single
/// frame, used to compare the `CPU` and `GPU` paths (design §29).
///
/// The summary is intentionally small — a live count, centroids, and bounds —
/// so a parity hook can compare backends every frame without reading back full
/// buffers. Centroids and bounds are order-independent reductions, so they do
/// not depend on the (possibly different) per-thread ordering of the backends.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct FrameDigest {
    /// The number of live particles summarized.
    pub live_count: u32,
    /// The mean position of the live particles ([`Vec3::ZERO`] when none).
    pub position_centroid: Vec3,
    /// The mean velocity of the live particles ([`Vec3::ZERO`] when none).
    pub velocity_centroid: Vec3,
    /// The component-wise minimum position (bounds lower corner).
    pub bounds_min: Vec3,
    /// The component-wise maximum position (bounds upper corner).
    pub bounds_max: Vec3,
}

impl FrameDigest {
    /// An empty digest describing a frame with no live particles.
    #[must_use]
    pub const fn empty() -> Self {
        Self {
            live_count: 0,
            position_centroid: Vec3::ZERO,
            velocity_centroid: Vec3::ZERO,
            bounds_min: Vec3::ZERO,
            bounds_max: Vec3::ZERO,
        }
    }

    /// Summarizes paired position / velocity `SoA` slices into a digest.
    ///
    /// The two slices are expected to have the same length (paired per
    /// particle); the shorter length is used defensively so a mismatched pair
    /// never reads out of range. An empty input yields [`FrameDigest::empty`].
    #[must_use]
    pub fn summarize(positions: &[Vec3], velocities: &[Vec3]) -> Self {
        let live_count = positions.len().min(velocities.len());
        if live_count == 0 {
            return Self::empty();
        }
        let mut position_sum = Vec3::ZERO;
        let mut velocity_sum = Vec3::ZERO;
        let mut bounds_min = positions[0];
        let mut bounds_max = positions[0];
        for i in 0..live_count {
            let p = positions[i];
            position_sum = position_sum.add(p);
            velocity_sum = velocity_sum.add(velocities[i]);
            bounds_min = bounds_min.min(p);
            bounds_max = bounds_max.max(p);
        }
        // `live_count` is a non-zero slice length, well within `f32` exact range
        // for any realistic particle count.
        let inv = 1.0 / live_count as f32;
        Self {
            // `live_count` is bounded by a slice length; the effect budgets keep
            // it well under `u32::MAX`.
            live_count: live_count as u32,
            position_centroid: position_sum.scale(inv),
            velocity_centroid: velocity_sum.scale(inv),
            bounds_min,
            bounds_max,
        }
    }
}

/// The result of comparing two [`FrameDigest`]s under a [`ParityTolerance`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ParityReport {
    /// Whether both backends reported the same live count.
    pub live_count_matches: bool,
    /// Whether the position centroid and bounds agree within tolerance.
    pub position_within_tolerance: bool,
    /// Whether the velocity centroid agrees within tolerance.
    pub velocity_within_tolerance: bool,
    /// The largest position deviation observed (centroid or bounds corner).
    pub max_position_error: f32,
    /// The largest velocity deviation observed.
    pub max_velocity_error: f32,
}

impl ParityReport {
    /// Returns `true` when every checked invariant holds: matching live counts
    /// and both position and velocity within tolerance.
    #[must_use]
    pub fn is_consistent(&self) -> bool {
        self.live_count_matches && self.position_within_tolerance && self.velocity_within_tolerance
    }
}

/// The largest absolute per-component difference between two vectors.
#[must_use]
fn max_component_delta(a: Vec3, b: Vec3) -> f32 {
    let dx = (a.x - b.x).abs();
    let dy = (a.y - b.y).abs();
    let dz = (a.z - b.z).abs();
    dx.max(dy).max(dz)
}

/// The cross-backend semantic-equivalence contract for an emitter (design §29).
///
/// It declares the invariants that must hold for the `CPU` and `GPU` paths to
/// agree — the same seed, the same `dt`, and the same stage order — and carries
/// the [`ParityTolerance`] a consistency check uses. The deterministic `RNG`
/// that makes the invariants achievable is owned by design §29; this type only
/// states the contract and compares frame digests.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SemanticParity {
    /// Both backends must be seeded identically.
    pub require_same_seed: bool,
    /// Both backends must advance by the same time step.
    pub require_same_dt: bool,
    /// Both backends must execute stages in the same order.
    pub require_same_stage_order: bool,
    /// The tolerance a frame-digest comparison allows.
    pub tolerance: ParityTolerance,
}

impl Default for SemanticParity {
    fn default() -> Self {
        Self {
            require_same_seed: true,
            require_same_dt: true,
            require_same_stage_order: true,
            tolerance: ParityTolerance::default(),
        }
    }
}

impl SemanticParity {
    /// Checks whether the declared preconditions actually hold for a run, given
    /// observed equality of seed, `dt`, and stage order.
    ///
    /// A required invariant that is violated fails the check; an invariant that
    /// was not required is ignored. When every required invariant holds the two
    /// backends are contractually obligated to produce digests that pass
    /// [`SemanticParity::check`].
    #[must_use]
    pub fn preconditions_hold(
        &self,
        seed_equal: bool,
        dt_equal: bool,
        stage_order_equal: bool,
    ) -> bool {
        (!self.require_same_seed || seed_equal)
            && (!self.require_same_dt || dt_equal)
            && (!self.require_same_stage_order || stage_order_equal)
    }

    /// Compares a `CPU` digest against a `GPU` digest under this parity's
    /// tolerance, producing a [`ParityReport`] (design §29).
    ///
    /// Live counts must match exactly; centroids and bounds must agree within
    /// the position / velocity epsilons. This is the per-frame consistency hook
    /// a validation harness calls; it never uses exact floating-point equality.
    #[must_use]
    pub fn check(&self, cpu: &FrameDigest, gpu: &FrameDigest) -> ParityReport {
        let live_count_matches = cpu.live_count == gpu.live_count;

        let centroid_pos_err = max_component_delta(cpu.position_centroid, gpu.position_centroid);
        let min_err = max_component_delta(cpu.bounds_min, gpu.bounds_min);
        let max_err = max_component_delta(cpu.bounds_max, gpu.bounds_max);
        let max_position_error = centroid_pos_err.max(min_err).max(max_err);

        let max_velocity_error = max_component_delta(cpu.velocity_centroid, gpu.velocity_centroid);

        ParityReport {
            live_count_matches,
            position_within_tolerance: max_position_error <= self.tolerance.position_eps,
            velocity_within_tolerance: max_velocity_error <= self.tolerance.velocity_eps,
            max_position_error,
            max_velocity_error,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const CMP_EPS: f32 = 1.0e-6;

    // --- Selection matrix -------------------------------------------------

    #[test]
    fn no_compute_forces_cpu_even_at_large_scale() {
        let caps = BackendCapabilities::cpu_only();
        let req = SimulationRequirement::bulk(10_000_000, TargetPlatform::Headless);
        let decision = choose_backend_explained(caps, req);
        assert_eq!(decision.backend, SimBackend::Cpu);
        assert_eq!(decision.reason, BackendSelectionReason::NoComputeSupport);
    }

    #[test]
    fn small_interactive_selects_cpu() {
        let caps = BackendCapabilities::desktop_gpu();
        let req = SimulationRequirement {
            particle_count: 512,
            needs_cpu_interaction: true,
            needs_determinism: false,
            target_platform: TargetPlatform::Desktop,
        };
        let decision = choose_backend_explained(caps, req);
        assert_eq!(decision.backend, SimBackend::Cpu);
        assert_eq!(
            decision.reason,
            BackendSelectionReason::SmallInteractiveWorkload
        );
    }

    #[test]
    fn large_interactive_still_selects_gpu() {
        let caps = BackendCapabilities::desktop_gpu();
        let req = SimulationRequirement {
            particle_count: SMALL_INTERACTIVE_PARTICLE_MAX + 1,
            needs_cpu_interaction: true,
            needs_determinism: false,
            target_platform: TargetPlatform::Desktop,
        };
        let decision = choose_backend_explained(caps, req);
        assert_eq!(decision.backend, SimBackend::Gpu);
        assert_eq!(decision.reason, BackendSelectionReason::LargeScaleGpu);
    }

    #[test]
    fn small_non_interactive_selects_gpu() {
        let caps = BackendCapabilities::desktop_gpu();
        // Small, but no interaction requested: interaction — not scale — is the
        // trigger for the CPU path.
        let req = SimulationRequirement::bulk(16, TargetPlatform::Desktop);
        assert_eq!(choose_backend(caps, req), SimBackend::Gpu);
    }

    #[test]
    fn large_scale_selects_gpu() {
        let caps = BackendCapabilities::desktop_gpu();
        let req = SimulationRequirement::bulk(2_000_000, TargetPlatform::Desktop);
        let decision = choose_backend_explained(caps, req);
        assert_eq!(decision.backend, SimBackend::Gpu);
        assert_eq!(decision.reason, BackendSelectionReason::LargeScaleGpu);
    }

    #[test]
    fn determinism_does_not_force_cpu() {
        let caps = BackendCapabilities::desktop_gpu();
        let req = SimulationRequirement {
            particle_count: 1_000_000,
            needs_cpu_interaction: false,
            needs_determinism: true,
            target_platform: TargetPlatform::Console,
        };
        // Both backends agree by construction, so determinism keeps the GPU path.
        assert_eq!(choose_backend(caps, req), SimBackend::Gpu);
    }

    #[test]
    fn small_interactive_threshold_boundary_is_inclusive() {
        let caps = BackendCapabilities::desktop_gpu();
        let at_threshold = SimulationRequirement {
            particle_count: SMALL_INTERACTIVE_PARTICLE_MAX,
            needs_cpu_interaction: true,
            needs_determinism: false,
            target_platform: TargetPlatform::Desktop,
        };
        assert_eq!(choose_backend(caps, at_threshold), SimBackend::Cpu);

        let above = SimulationRequirement {
            particle_count: SMALL_INTERACTIVE_PARTICLE_MAX + 1,
            ..at_threshold
        };
        assert_eq!(choose_backend(caps, above), SimBackend::Gpu);
    }

    #[test]
    fn custom_threshold_is_honored() {
        let caps = BackendCapabilities::desktop_gpu();
        let req = SimulationRequirement {
            particle_count: 100,
            needs_cpu_interaction: true,
            needs_determinism: false,
            target_platform: TargetPlatform::Desktop,
        };
        // With a threshold below the count, the small-interactive rule no longer
        // fires and the GPU path wins.
        assert_eq!(
            choose_backend_with_threshold(caps, req, 50).backend,
            SimBackend::Gpu
        );
        assert_eq!(
            choose_backend_with_threshold(caps, req, 200).backend,
            SimBackend::Cpu
        );
    }

    #[test]
    fn sim_backend_predicates() {
        assert!(SimBackend::default().is_gpu());
        assert!(SimBackend::Gpu.is_gpu());
        assert!(!SimBackend::Gpu.is_cpu());
        assert!(SimBackend::Cpu.is_cpu());
        assert!(!SimBackend::Cpu.is_gpu());
    }

    #[test]
    fn platform_presets_reflect_compute_support() {
        assert!(BackendCapabilities::for_platform(TargetPlatform::Desktop).supports_compute);
        assert!(BackendCapabilities::for_platform(TargetPlatform::Console).supports_compute);
        assert!(BackendCapabilities::for_platform(TargetPlatform::Mobile).supports_compute);
        assert!(BackendCapabilities::for_platform(TargetPlatform::Web).supports_compute);
        assert!(!BackendCapabilities::for_platform(TargetPlatform::Headless).supports_compute);
        // Mobile lacks indirect dispatch in the preset.
        assert!(
            !BackendCapabilities::for_platform(TargetPlatform::Mobile).supports_indirect_dispatch
        );
    }

    // --- ceil_div guard ---------------------------------------------------

    #[test]
    fn ceil_div_zero_count_is_zero() {
        assert_eq!(ceil_div_guarded(0, 64), 0);
    }

    #[test]
    fn ceil_div_single_element_rounds_to_one() {
        assert_eq!(ceil_div_guarded(1, 64), 1);
    }

    #[test]
    fn ceil_div_exact_multiple() {
        assert_eq!(ceil_div_guarded(128, 64), 2);
    }

    #[test]
    fn ceil_div_with_remainder_rounds_up() {
        assert_eq!(ceil_div_guarded(130, 64), 3);
    }

    #[test]
    fn ceil_div_zero_divisor_is_guarded() {
        assert_eq!(ceil_div_guarded(100, 0), 0);
    }

    #[test]
    fn ceil_div_does_not_overflow_near_max() {
        // A manual (n + d - 1) / d would overflow here; div_ceil must not.
        assert_eq!(ceil_div_guarded(u32::MAX, 1), u32::MAX);
        assert_eq!(ceil_div_guarded(u32::MAX, 2), (u32::MAX / 2) + 1);
    }

    // --- GPU dispatch plan -----------------------------------------------

    #[test]
    fn gpu_plan_basic_workgroup_count() {
        let caps = BackendCapabilities::desktop_gpu();
        let tuning = BackendTuning::default();
        let plan = GpuDispatchPlan::derive(caps, tuning, IterationDomain::PerParticle, 130);
        assert_eq!(plan.workgroup_size, 64);
        assert_eq!(plan.workgroup_count, 3);
        assert_eq!(plan.dispatch_count, 1);
        assert!(plan.use_indirect);
        assert!(!plan.persistent_threads);
        assert!(!plan.is_no_op());
        assert!(plan.covers_all_elements());
    }

    #[test]
    fn gpu_plan_clamps_workgroup_size_to_device_limit() {
        let caps = BackendCapabilities {
            max_workgroup_size: 32,
            ..BackendCapabilities::desktop_gpu()
        };
        let tuning = BackendTuning {
            gpu_workgroup_size: 256,
            ..BackendTuning::default()
        };
        let plan = GpuDispatchPlan::derive(caps, tuning, IterationDomain::PerParticle, 100);
        assert_eq!(plan.workgroup_size, 32);
        assert_eq!(plan.workgroup_count, 4); // ceil(100 / 32)
    }

    #[test]
    fn gpu_plan_disables_indirect_when_unsupported() {
        let caps = BackendCapabilities {
            supports_indirect_dispatch: false,
            ..BackendCapabilities::desktop_gpu()
        };
        let tuning = BackendTuning::default();
        let plan = GpuDispatchPlan::derive(caps, tuning, IterationDomain::PerParticle, 100);
        assert!(!plan.use_indirect);
    }

    #[test]
    fn gpu_plan_splits_across_dispatches_when_over_limit() {
        let caps = BackendCapabilities {
            max_workgroup_size: 1,
            max_workgroups_per_dispatch: 8,
            ..BackendCapabilities::desktop_gpu()
        };
        let tuning = BackendTuning {
            gpu_workgroup_size: 1,
            gpu_persistent_threads: false,
            ..BackendTuning::default()
        };
        let plan = GpuDispatchPlan::derive(caps, tuning, IterationDomain::PerParticle, 20);
        assert_eq!(plan.workgroup_count, 20);
        assert_eq!(plan.workgroups_per_dispatch, 8);
        assert_eq!(plan.dispatch_count, 3); // ceil(20 / 8)
        assert!(plan.covers_all_elements());
    }

    #[test]
    fn gpu_plan_persistent_threads_use_single_capped_dispatch() {
        let caps = BackendCapabilities {
            max_workgroup_size: 1,
            max_workgroups_per_dispatch: 8,
            ..BackendCapabilities::desktop_gpu()
        };
        let tuning = BackendTuning {
            gpu_workgroup_size: 1,
            gpu_persistent_threads: true,
            ..BackendTuning::default()
        };
        let plan = GpuDispatchPlan::derive(caps, tuning, IterationDomain::PerParticle, 20);
        assert_eq!(plan.workgroup_count, 20);
        assert_eq!(plan.workgroups_per_dispatch, 8);
        assert_eq!(plan.dispatch_count, 1);
        assert!(plan.persistent_threads);
        assert!(plan.covers_all_elements());
    }

    #[test]
    fn gpu_plan_zero_elements_is_no_op() {
        let caps = BackendCapabilities::desktop_gpu();
        let plan = GpuDispatchPlan::derive(
            caps,
            BackendTuning::default(),
            IterationDomain::PerParticle,
            0,
        );
        assert!(plan.is_no_op());
        assert_eq!(plan.workgroup_count, 0);
        assert_eq!(plan.dispatch_count, 0);
        assert!(plan.covers_all_elements()); // zero elements are trivially covered
    }

    #[test]
    fn gpu_plan_zero_workgroup_size_is_guarded_no_op() {
        // A device reporting a zero max workgroup size is degenerate; the plan
        // must collapse to a no-op rather than divide by zero.
        let caps = BackendCapabilities {
            max_workgroup_size: 0,
            ..BackendCapabilities::desktop_gpu()
        };
        let plan = GpuDispatchPlan::derive(
            caps,
            BackendTuning::default(),
            IterationDomain::PerParticle,
            100,
        );
        assert_eq!(plan.workgroup_size, 0);
        assert!(plan.is_no_op());
    }

    #[test]
    fn gpu_plan_exact_multiple_single_dispatch() {
        let caps = BackendCapabilities::desktop_gpu();
        let plan = GpuDispatchPlan::derive(
            caps,
            BackendTuning::default(),
            IterationDomain::PerParticle,
            128,
        );
        assert_eq!(plan.workgroup_count, 2);
        assert_eq!(plan.dispatch_count, 1);
    }

    // --- CPU chunk plan ---------------------------------------------------

    #[test]
    fn cpu_plan_chunk_count_rounds_up() {
        let tuning = BackendTuning {
            cpu_chunk_size: 256,
            ..BackendTuning::default()
        };
        let plan = CpuChunkPlan::derive(tuning, IterationDomain::PerParticle, 300);
        assert_eq!(plan.chunk_count, 2); // ceil(300 / 256)
        assert!(plan.parallel);
        assert!(!plan.is_no_op());
        assert!(plan.covers_all_elements());
    }

    #[test]
    fn cpu_plan_single_chunk_is_not_parallel() {
        let tuning = BackendTuning {
            cpu_chunk_size: 256,
            ..BackendTuning::default()
        };
        let plan = CpuChunkPlan::derive(tuning, IterationDomain::PerParticle, 200);
        assert_eq!(plan.chunk_count, 1);
        assert!(!plan.parallel);
    }

    #[test]
    fn cpu_plan_exact_multiple() {
        let tuning = BackendTuning {
            cpu_chunk_size: 100,
            ..BackendTuning::default()
        };
        let plan = CpuChunkPlan::derive(tuning, IterationDomain::PerParticle, 400);
        assert_eq!(plan.chunk_count, 4);
        assert!(plan.covers_all_elements());
    }

    #[test]
    fn cpu_plan_zero_elements_is_no_op() {
        let plan = CpuChunkPlan::derive(BackendTuning::default(), IterationDomain::PerParticle, 0);
        assert!(plan.is_no_op());
        assert_eq!(plan.chunk_count, 0);
        assert!(!plan.parallel);
        assert!(plan.covers_all_elements());
    }

    #[test]
    fn cpu_plan_zero_chunk_size_is_guarded_no_op() {
        let tuning = BackendTuning {
            cpu_chunk_size: 0,
            ..BackendTuning::default()
        };
        let plan = CpuChunkPlan::derive(tuning, IterationDomain::PerParticle, 500);
        assert!(plan.is_no_op());
        assert_eq!(plan.chunk_count, 0);
    }

    #[test]
    fn cpu_plan_parallel_disabled_by_tuning() {
        let tuning = BackendTuning {
            cpu_chunk_size: 64,
            cpu_parallel: false,
            ..BackendTuning::default()
        };
        let plan = CpuChunkPlan::derive(tuning, IterationDomain::PerParticle, 1_000);
        assert!(plan.chunk_count > 1);
        assert!(!plan.parallel);
    }

    // --- BackendPlan routing ---------------------------------------------

    #[test]
    fn backend_plan_routes_to_gpu() {
        let caps = BackendCapabilities::desktop_gpu();
        let plan = BackendPlan::derive(
            SimBackend::Gpu,
            caps,
            BackendTuning::default(),
            IterationDomain::PerParticle,
            256,
        );
        assert_eq!(plan.backend(), SimBackend::Gpu);
        assert!(matches!(plan, BackendPlan::Gpu(_)));
        assert!(!plan.is_no_op());
    }

    #[test]
    fn backend_plan_routes_to_cpu() {
        let caps = BackendCapabilities::cpu_only();
        let plan = BackendPlan::derive(
            SimBackend::Cpu,
            caps,
            BackendTuning::default(),
            IterationDomain::PerParticle,
            256,
        );
        assert_eq!(plan.backend(), SimBackend::Cpu);
        assert!(matches!(plan, BackendPlan::Cpu(_)));
    }

    #[test]
    fn backend_plan_no_op_on_empty_domain() {
        let caps = BackendCapabilities::desktop_gpu();
        let gpu = BackendPlan::derive(
            SimBackend::Gpu,
            caps,
            BackendTuning::default(),
            IterationDomain::PerGridVoxel,
            0,
        );
        let cpu = BackendPlan::derive(
            SimBackend::Cpu,
            caps,
            BackendTuning::default(),
            IterationDomain::PerGridVoxel,
            0,
        );
        assert!(gpu.is_no_op());
        assert!(cpu.is_no_op());
    }

    // --- Frame digest & parity -------------------------------------------

    #[test]
    fn frame_digest_empty_slices() {
        let digest = FrameDigest::summarize(&[], &[]);
        assert_eq!(digest, FrameDigest::empty());
        assert_eq!(digest.live_count, 0);
    }

    #[test]
    fn frame_digest_centroid_and_bounds() {
        let positions = [
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(2.0, 0.0, 0.0),
            Vec3::new(4.0, 6.0, 0.0),
        ];
        let velocities = [
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(1.0, 3.0, 0.0),
        ];
        let digest = FrameDigest::summarize(&positions, &velocities);
        assert_eq!(digest.live_count, 3);
        assert!((digest.position_centroid.x - 2.0).abs() < CMP_EPS);
        assert!((digest.position_centroid.y - 2.0).abs() < CMP_EPS);
        assert!((digest.velocity_centroid.y - 1.0).abs() < CMP_EPS);
        assert_eq!(digest.bounds_min, Vec3::new(0.0, 0.0, 0.0));
        assert_eq!(digest.bounds_max, Vec3::new(4.0, 6.0, 0.0));
    }

    #[test]
    fn frame_digest_uses_shorter_paired_length() {
        let positions = [Vec3::new(1.0, 1.0, 1.0), Vec3::new(2.0, 2.0, 2.0)];
        let velocities = [Vec3::new(0.0, 0.0, 0.0)];
        let digest = FrameDigest::summarize(&positions, &velocities);
        // Only the first paired particle is summarized.
        assert_eq!(digest.live_count, 1);
        assert_eq!(digest.bounds_max, Vec3::new(1.0, 1.0, 1.0));
    }

    #[test]
    fn parity_consistent_within_tolerance() {
        let parity = SemanticParity::default();
        let cpu = FrameDigest {
            live_count: 1_000,
            position_centroid: Vec3::new(1.0, 2.0, 3.0),
            velocity_centroid: Vec3::new(0.5, 0.0, 0.0),
            bounds_min: Vec3::new(-1.0, -1.0, -1.0),
            bounds_max: Vec3::new(3.0, 5.0, 7.0),
        };
        // GPU differs by less than the default 1e-4 tolerance.
        let gpu = FrameDigest {
            position_centroid: Vec3::new(1.000_02, 2.0, 3.0),
            velocity_centroid: Vec3::new(0.500_03, 0.0, 0.0),
            ..cpu
        };
        let report = parity.check(&cpu, &gpu);
        assert!(report.is_consistent());
        assert!(report.live_count_matches);
        assert!(report.max_position_error < parity.tolerance.position_eps);
    }

    #[test]
    fn parity_fails_when_position_exceeds_tolerance() {
        let parity = SemanticParity::default();
        let cpu = FrameDigest {
            live_count: 500,
            position_centroid: Vec3::new(0.0, 0.0, 0.0),
            velocity_centroid: Vec3::ZERO,
            bounds_min: Vec3::ZERO,
            bounds_max: Vec3::ZERO,
        };
        let gpu = FrameDigest {
            // 0.5 world units of drift — far beyond 1e-4.
            bounds_max: Vec3::new(0.5, 0.0, 0.0),
            ..cpu
        };
        let report = parity.check(&cpu, &gpu);
        assert!(!report.position_within_tolerance);
        assert!(!report.is_consistent());
        assert!((report.max_position_error - 0.5).abs() < CMP_EPS);
    }

    #[test]
    fn parity_fails_when_velocity_exceeds_tolerance() {
        let parity = SemanticParity::default();
        let cpu = FrameDigest::empty();
        let gpu = FrameDigest {
            velocity_centroid: Vec3::new(0.0, 1.0, 0.0),
            ..cpu
        };
        let report = parity.check(&cpu, &gpu);
        assert!(!report.velocity_within_tolerance);
        assert!(!report.is_consistent());
    }

    #[test]
    fn parity_fails_when_live_count_differs() {
        let parity = SemanticParity::default();
        let cpu = FrameDigest {
            live_count: 100,
            ..FrameDigest::empty()
        };
        let gpu = FrameDigest {
            live_count: 101,
            ..FrameDigest::empty()
        };
        let report = parity.check(&cpu, &gpu);
        assert!(!report.live_count_matches);
        assert!(!report.is_consistent());
    }

    #[test]
    fn parity_identical_digests_are_consistent() {
        let parity = SemanticParity::default();
        let digest = FrameDigest {
            live_count: 42,
            position_centroid: Vec3::new(1.0, 1.0, 1.0),
            velocity_centroid: Vec3::new(2.0, 2.0, 2.0),
            bounds_min: Vec3::new(-5.0, -5.0, -5.0),
            bounds_max: Vec3::new(5.0, 5.0, 5.0),
        };
        let report = parity.check(&digest, &digest);
        assert!(report.is_consistent());
        assert!((report.max_position_error - 0.0).abs() < CMP_EPS);
        assert!((report.max_velocity_error - 0.0).abs() < CMP_EPS);
    }

    #[test]
    fn parity_preconditions_respect_requirements() {
        let parity = SemanticParity::default();
        // All invariants hold.
        assert!(parity.preconditions_hold(true, true, true));
        // A violated required invariant fails.
        assert!(!parity.preconditions_hold(false, true, true));
        assert!(!parity.preconditions_hold(true, false, true));
        assert!(!parity.preconditions_hold(true, true, false));
    }

    #[test]
    fn parity_ignores_non_required_invariants() {
        let parity = SemanticParity {
            require_same_seed: false,
            require_same_dt: true,
            require_same_stage_order: false,
            tolerance: ParityTolerance::default(),
        };
        // Seed and stage-order mismatches are ignored; only dt is required.
        assert!(parity.preconditions_hold(false, true, false));
        assert!(!parity.preconditions_hold(true, false, true));
    }

    #[test]
    fn parity_custom_tolerance_is_used() {
        let parity = SemanticParity {
            tolerance: ParityTolerance {
                position_eps: 1.0,
                velocity_eps: 1.0,
                scalar_eps: 1.0,
            },
            ..SemanticParity::default()
        };
        let cpu = FrameDigest::empty();
        let gpu = FrameDigest {
            // 0.5 drift now passes under the loosened 1.0 tolerance.
            bounds_max: Vec3::new(0.5, 0.0, 0.0),
            ..cpu
        };
        let report = parity.check(&cpu, &gpu);
        assert!(report.is_consistent());
    }
}
