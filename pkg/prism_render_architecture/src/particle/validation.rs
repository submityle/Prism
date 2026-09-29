//! Quality and trust infrastructure: the computable contract layer (design
//! §33).
//!
//! Where the sibling modules *produce* particle state, this module *judges* it.
//! It is the cross-cutting "validation / assertion / consistency" capability
//! that turns the design's trust goals into small, pure, `CPU`-verifiable
//! predicates a test harness or a runtime debug build can call. Nothing here
//! owns simulation state; every function takes already-extracted samples and
//! returns a *structured* verdict, so the same check works against a `CPU`
//! reference, a `GPU` readback, or a recorded golden capture.
//!
//! # Numerical invariant guards
//! [`first_non_finite_scalar`] and [`first_non_finite_vector`] locate the first
//! `NaN` or infinity in a sample buffer (position, component, and class), so a
//! blow-up is reported *where* it happened rather than as a bare boolean.
//! [`BoundsGuard`] flags the first particle whose speed or position leaves the
//! authored envelope, and [`total_momentum`] / [`total_mass`] feed
//! [`check_momentum_conserved`] / [`check_mass_conserved`], which apply an
//! absolute-plus-relative tolerance so "conserved" is a defined, testable
//! predicate rather than an exact-equality wish.
//!
//! # `GPU` vs `CPU` numerical consistency
//! [`compare_scalar_samples`] and [`compare_vector_samples`] take the *same*
//! graph's output from two backends and, per attribute, report the first
//! over-tolerance sample and the maximum error. This is the fine-grained,
//! per-sample counterpart to [`super::dual_backend`]'s frame-digest
//! [`super::dual_backend::SemanticParity`]; the two are complementary and this
//! module does not re-derive the digest. Only the deterministic *compute*
//! buckets are portable enough for a `CPU` golden; `RT` (ray-traced) traversal
//! is intentionally *not* `CPU`-golden-able (driver `BVH` builds and hit-order
//! differ), so those attributes should be validated against a recorded `GPU`
//! reference, not a `CPU` reference — see [`ErrorTolerance`].
//!
//! # Golden image regression contract
//! [`evaluate_golden`] does **not** decode images. It consumes already-extracted
//! per-pixel error magnitudes and a [`GoldenTolerance`] (per-pixel epsilon,
//! failed-pixel budget, mean-error limit) and aggregates them into a
//! pass/fail [`GoldenReport`] with an over-tolerance count — the numeric
//! contract that a real image differ would feed.
//!
//! # Stress / boundary assertion helpers
//! [`check_capacity`], [`check_free_list`], [`check_event_backpressure`],
//! [`check_compaction`], and [`check_oit_layers`] each model one failure mode
//! (pool capacity overflow, `free-list` double-alloc / double-free races over a
//! deterministic op sequence, event ring back-pressure, prefix-sum compaction
//! correctness, and `OIT` per-pixel layer overflow) and return a structured
//! violation report instead of panicking, so a test can assert on the *reason*.
//!
//! Only ordinary arithmetic (`+ - * /`) and `sqrt` (transitively, through
//! [`super::Vec3`]) are used — no transcendental functions — and `f32`
//! equality is never tested with `==`; non-finite detection uses
//! [`f32::is_nan`] / [`f32::is_finite`] and every threshold is an epsilon or a
//! relative error, keeping this layer reproducible against a future `GPU`
//! kernel.

use alloc::vec::Vec;

use super::Vec3;

/// The class of a non-finite `f32`.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum NonFiniteKind {
    /// A `NaN` (not-a-number) value, detected with [`f32::is_nan`].
    Nan,
    /// A positive or negative infinity.
    Infinite,
}

/// Which component of a [`super::Vec3`] a non-finite value was found in.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum VectorComponent {
    /// The `x` component.
    X,
    /// The `y` component.
    Y,
    /// The `z` component.
    Z,
}

/// The location and class of the first non-finite scalar in a sample buffer.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct NonFiniteScalar {
    /// Index of the offending sample.
    pub index: usize,
    /// Whether the value is `NaN` or an infinity.
    pub kind: NonFiniteKind,
}

/// The location and class of the first non-finite [`super::Vec3`] component.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct NonFiniteVector {
    /// Index of the offending sample.
    pub index: usize,
    /// Which component carried the non-finite value.
    pub component: VectorComponent,
    /// Whether the value is `NaN` or an infinity.
    pub kind: NonFiniteKind,
}

/// Classifies a scalar as `NaN`, infinite, or finite (returning `None` when
/// finite), using only the allowed `is_nan` / `is_finite` predicates.
#[must_use]
pub fn classify_non_finite(value: f32) -> Option<NonFiniteKind> {
    if value.is_nan() {
        Some(NonFiniteKind::Nan)
    } else if !value.is_finite() {
        Some(NonFiniteKind::Infinite)
    } else {
        None
    }
}

/// Locates the first non-finite scalar in `samples`, or `None` if all are
/// finite.
#[must_use]
pub fn first_non_finite_scalar(samples: &[f32]) -> Option<NonFiniteScalar> {
    for (index, &value) in samples.iter().enumerate() {
        if let Some(kind) = classify_non_finite(value) {
            return Some(NonFiniteScalar { index, kind });
        }
    }
    None
}

/// Locates the first non-finite [`super::Vec3`] component in `samples`
/// (scanning `x`, then `y`, then `z` per sample), or `None` if all are finite.
#[must_use]
pub fn first_non_finite_vector(samples: &[Vec3]) -> Option<NonFiniteVector> {
    for (index, sample) in samples.iter().enumerate() {
        if let Some(kind) = classify_non_finite(sample.x) {
            return Some(NonFiniteVector {
                index,
                component: VectorComponent::X,
                kind,
            });
        }
        if let Some(kind) = classify_non_finite(sample.y) {
            return Some(NonFiniteVector {
                index,
                component: VectorComponent::Y,
                kind,
            });
        }
        if let Some(kind) = classify_non_finite(sample.z) {
            return Some(NonFiniteVector {
                index,
                component: VectorComponent::Z,
                kind,
            });
        }
    }
    None
}

/// A speed violation: a particle whose velocity magnitude exceeds the bound.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SpeedViolation {
    /// Index of the offending particle.
    pub index: usize,
    /// The offending speed (velocity magnitude).
    pub magnitude: f32,
    /// The limit that was exceeded.
    pub limit: f32,
}

/// A position violation: a particle whose position leaves the authored extent.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PositionViolation {
    /// Index of the offending particle.
    pub index: usize,
    /// Which component left the extent.
    pub component: VectorComponent,
    /// The offending component value.
    pub value: f32,
    /// The symmetric extent limit (`-limit..=limit`) that was exceeded.
    pub limit: f32,
}

/// An authored physical envelope: a maximum speed and a symmetric per-axis
/// position extent, used to catch runaway particles before they poison a frame.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BoundsGuard {
    /// Maximum allowed velocity magnitude, in world units per second.
    pub max_speed: f32,
    /// Maximum allowed absolute value of any position component, in world
    /// units (the domain is `-max_position_extent..=max_position_extent`).
    pub max_position_extent: f32,
}

impl BoundsGuard {
    /// Builds a guard from a speed cap and a symmetric position extent.
    #[must_use]
    pub const fn new(max_speed: f32, max_position_extent: f32) -> Self {
        Self {
            max_speed,
            max_position_extent,
        }
    }

    /// Returns the first velocity whose magnitude exceeds [`Self::max_speed`],
    /// or `None` if every sample is within the cap.
    #[must_use]
    pub fn first_speed_violation(&self, velocities: &[Vec3]) -> Option<SpeedViolation> {
        for (index, velocity) in velocities.iter().enumerate() {
            let magnitude = velocity.length();
            if magnitude > self.max_speed {
                return Some(SpeedViolation {
                    index,
                    magnitude,
                    limit: self.max_speed,
                });
            }
        }
        None
    }

    /// Returns the first position component that leaves the symmetric extent,
    /// or `None` if every sample is inside the domain.
    #[must_use]
    pub fn first_position_violation(&self, positions: &[Vec3]) -> Option<PositionViolation> {
        let limit = self.max_position_extent;
        for (index, position) in positions.iter().enumerate() {
            if position.x.abs() > limit {
                return Some(PositionViolation {
                    index,
                    component: VectorComponent::X,
                    value: position.x,
                    limit,
                });
            }
            if position.y.abs() > limit {
                return Some(PositionViolation {
                    index,
                    component: VectorComponent::Y,
                    value: position.y,
                    limit,
                });
            }
            if position.z.abs() > limit {
                return Some(PositionViolation {
                    index,
                    component: VectorComponent::Z,
                    value: position.z,
                    limit,
                });
            }
        }
        None
    }
}

/// A combined absolute-plus-relative error budget, the trust layer's unit of
/// "close enough". The accepted error at a given `reference` magnitude is
/// `absolute + relative * reference`, which lets tiny values pass on the
/// absolute floor while large values scale with a relative fraction.
///
/// For portable *compute* buckets a `CPU` golden is valid, so a tight budget is
/// appropriate. `RT` traversal is not `CPU`-golden-able, so those attributes
/// should be compared `GPU`-to-`GPU` with a looser budget rather than against a
/// `CPU` reference.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ErrorTolerance {
    /// Absolute error floor, in the attribute's own units.
    pub absolute: f32,
    /// Relative error fraction applied to the reference magnitude.
    pub relative: f32,
}

impl ErrorTolerance {
    /// Builds a tolerance from an absolute floor and a relative fraction.
    #[must_use]
    pub const fn new(absolute: f32, relative: f32) -> Self {
        Self { absolute, relative }
    }

    /// The maximum error accepted at the given reference magnitude,
    /// `absolute + relative * |reference|`.
    #[must_use]
    pub fn allowed(self, reference: f32) -> f32 {
        self.absolute + self.relative * reference.abs()
    }

    /// Returns `true` when an already-computed non-negative `error` is within
    /// the budget for the given `scale` magnitude.
    #[must_use]
    pub fn accepts_magnitude(self, error: f32, scale: f32) -> bool {
        error <= self.allowed(scale)
    }

    /// Returns `true` when two scalars agree within the budget, scaling the
    /// relative term by the larger of the two magnitudes.
    #[must_use]
    pub fn accepts(self, a: f32, b: f32) -> bool {
        let error = (a - b).abs();
        self.accepts_magnitude(error, a.abs().max(b.abs()))
    }
}

/// The verdict of a conservation check (momentum or mass).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ConservationReport {
    /// The measured drift between the before and after quantities (a magnitude
    /// for vectors, an absolute difference for scalars).
    pub error: f32,
    /// The error the tolerance allowed at this scale.
    pub allowed: f32,
    /// Whether the quantity was conserved within tolerance.
    pub conserved: bool,
}

/// Total linear momentum `sum(mass[i] * velocity[i])`, pairing masses and
/// velocities by index (extra entries in the longer slice are ignored).
#[must_use]
pub fn total_momentum(masses: &[f32], velocities: &[Vec3]) -> Vec3 {
    let mut sum = Vec3::ZERO;
    for (&mass, &velocity) in masses.iter().zip(velocities.iter()) {
        sum = sum.add(velocity.scale(mass));
    }
    sum
}

/// Total mass `sum(mass[i])`.
#[must_use]
pub fn total_mass(masses: &[f32]) -> f32 {
    let mut sum = 0.0;
    for &mass in masses {
        sum += mass;
    }
    sum
}

/// Checks that total momentum is conserved between two states within the given
/// tolerance. The relative term scales with the larger total-momentum
/// magnitude, so a fast-moving system is not held to the same absolute drift as
/// a nearly-still one. Use only when no external impulse acted this step.
#[must_use]
pub fn check_momentum_conserved(
    before: Vec3,
    after: Vec3,
    tolerance: ErrorTolerance,
) -> ConservationReport {
    let error = after.distance(before);
    let scale = before.length().max(after.length());
    let allowed = tolerance.allowed(scale);
    ConservationReport {
        error,
        allowed,
        conserved: error <= allowed,
    }
}

/// Checks that total mass is conserved between two states within the given
/// tolerance (for example, that emission and death bookkeeping balanced).
#[must_use]
pub fn check_mass_conserved(
    before: f32,
    after: f32,
    tolerance: ErrorTolerance,
) -> ConservationReport {
    let error = (after - before).abs();
    let allowed = tolerance.allowed(before.abs().max(after.abs()));
    ConservationReport {
        error,
        allowed,
        conserved: error <= allowed,
    }
}

/// The element-wise verdict of a two-backend comparison.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ConsistencyReport {
    /// Number of sample pairs compared.
    pub compared: usize,
    /// Index of the first over-tolerance sample, if any.
    pub first_violation: Option<usize>,
    /// The largest per-sample error observed (a `NaN` operand yields no finite
    /// error and is instead surfaced through `first_violation`).
    pub max_error: f32,
    /// Index at which [`Self::max_error`] occurred.
    pub max_error_index: usize,
    /// Whether every compared sample was within tolerance.
    pub consistent: bool,
}

/// The outcome of comparing two backends' output for one attribute.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum ConsistencyOutcome {
    /// Both buffers had equal length and were compared element-wise.
    Compared(ConsistencyReport),
    /// The two backends produced a different number of samples, which is a
    /// structural divergence no per-element tolerance can absorb.
    LengthMismatch {
        /// Number of `CPU`-side samples.
        cpu_len: usize,
        /// Number of `GPU`-side samples.
        gpu_len: usize,
    },
}

/// Compares one scalar attribute (for example, per-particle age) produced by
/// the `CPU` and `GPU` backends, reporting the first over-tolerance sample and
/// the maximum error. This is the portable *compute* bucket path; do not use it
/// for `RT`-derived attributes against a `CPU` reference.
#[must_use]
pub fn compare_scalar_samples(
    cpu: &[f32],
    gpu: &[f32],
    tolerance: ErrorTolerance,
) -> ConsistencyOutcome {
    if cpu.len() != gpu.len() {
        return ConsistencyOutcome::LengthMismatch {
            cpu_len: cpu.len(),
            gpu_len: gpu.len(),
        };
    }
    let mut report = ConsistencyReport {
        compared: cpu.len(),
        first_violation: None,
        max_error: 0.0,
        max_error_index: 0,
        consistent: true,
    };
    for (index, (&a, &b)) in cpu.iter().zip(gpu.iter()).enumerate() {
        let error = (a - b).abs();
        if error > report.max_error {
            report.max_error = error;
            report.max_error_index = index;
        }
        if !tolerance.accepts(a, b) && report.first_violation.is_none() {
            report.first_violation = Some(index);
            report.consistent = false;
        }
    }
    ConsistencyOutcome::Compared(report)
}

/// Compares one vector attribute (for example, position or velocity) produced
/// by the `CPU` and `GPU` backends. The per-sample error is the Euclidean
/// distance between the two vectors, scaled by the larger vector magnitude for
/// the relative term.
#[must_use]
pub fn compare_vector_samples(
    cpu: &[Vec3],
    gpu: &[Vec3],
    tolerance: ErrorTolerance,
) -> ConsistencyOutcome {
    if cpu.len() != gpu.len() {
        return ConsistencyOutcome::LengthMismatch {
            cpu_len: cpu.len(),
            gpu_len: gpu.len(),
        };
    }
    let mut report = ConsistencyReport {
        compared: cpu.len(),
        first_violation: None,
        max_error: 0.0,
        max_error_index: 0,
        consistent: true,
    };
    for (index, (&a, &b)) in cpu.iter().zip(gpu.iter()).enumerate() {
        let error = a.distance(b);
        if error > report.max_error {
            report.max_error = error;
            report.max_error_index = index;
        }
        let scale = a.length().max(b.length());
        if !tolerance.accepts_magnitude(error, scale) && report.first_violation.is_none() {
            report.first_violation = Some(index);
            report.consistent = false;
        }
    }
    ConsistencyOutcome::Compared(report)
}

/// The numeric tolerance for a golden image regression, expressed over
/// *already-extracted* error statistics rather than raw pixels.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GoldenTolerance {
    /// A pixel counts as failed when its error magnitude exceeds this epsilon.
    pub per_pixel_epsilon: f32,
    /// The maximum number of failed pixels the comparison still passes with.
    pub max_failed_pixels: u32,
    /// The maximum mean error, over all pixels, the comparison passes with.
    pub mean_error_limit: f32,
}

/// The aggregated verdict of a golden image regression comparison.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GoldenReport {
    /// Number of pixel errors aggregated.
    pub pixels: u32,
    /// Number of pixels over [`GoldenTolerance::per_pixel_epsilon`].
    pub failed_pixels: u32,
    /// The largest single-pixel error observed.
    pub max_error: f32,
    /// The mean error over all pixels (zero when there are no pixels).
    pub mean_error: f32,
    /// Whether the comparison passed both the failed-pixel and mean limits.
    pub passed: bool,
}

/// Aggregates a buffer of per-pixel error magnitudes into a pass/fail
/// [`GoldenReport`]. No image decoding happens here: `pixel_errors` are the
/// already-computed per-pixel deltas a real image differ would produce.
#[must_use]
pub fn evaluate_golden(pixel_errors: &[f32], tolerance: GoldenTolerance) -> GoldenReport {
    let pixels = pixel_errors.len();
    let mut failed_pixels: u32 = 0;
    let mut max_error = 0.0f32;
    let mut sum_error = 0.0f32;
    for &error in pixel_errors {
        if error > tolerance.per_pixel_epsilon {
            failed_pixels += 1;
        }
        if error > max_error {
            max_error = error;
        }
        sum_error += error;
    }
    let mean_error = if pixels == 0 {
        0.0
    } else {
        sum_error / pixels as f32
    };
    let passed =
        failed_pixels <= tolerance.max_failed_pixels && mean_error <= tolerance.mean_error_limit;
    GoldenReport {
        pixels: pixels as u32,
        failed_pixels,
        max_error,
        mean_error,
        passed,
    }
}

/// A pool capacity overflow: more slots were requested than the pool holds.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct CapacityViolation {
    /// The pool's fixed capacity.
    pub capacity: u32,
    /// The number of slots requested.
    pub requested: u32,
    /// How many requests could not be satisfied (`requested - capacity`).
    pub overflow: u32,
}

/// Checks a single-shot capacity request against a fixed pool capacity,
/// returning the overflow amount when the request does not fit.
#[must_use]
pub fn check_capacity(capacity: u32, requested: u32) -> Option<CapacityViolation> {
    let overflow = requested.saturating_sub(capacity);
    if overflow > 0 {
        Some(CapacityViolation {
            capacity,
            requested,
            overflow,
        })
    } else {
        None
    }
}

/// One step in a deterministic `free-list` op sequence used to model
/// allocator races without threads: replaying a fixed interleaving is enough to
/// expose the invariant breaks a real data race would cause.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum FreeListOp {
    /// Hand out the given slot index to a caller.
    Allocate(u32),
    /// Return the given slot index to the free list.
    Free(u32),
}

/// The kind of `free-list` invariant that was broken.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum FreeListFault {
    /// A slot was allocated while already live (a lost free / double alloc).
    DoubleAllocate,
    /// A slot was freed while not live (a double free).
    DoubleFree,
    /// A slot index was outside the pool capacity.
    OutOfRange,
}

/// A structured `free-list` violation, naming the step and slot at fault.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct FreeListViolation {
    /// Zero-based index into the op sequence where the fault occurred.
    pub step: usize,
    /// The slot the faulting op referenced.
    pub slot: u32,
    /// Which invariant was broken.
    pub fault: FreeListFault,
}

/// Replays a deterministic `free-list` op sequence against a pool of the given
/// capacity and returns the first invariant break (double allocate, double
/// free, or out-of-range slot), or `None` if the whole sequence is consistent.
#[must_use]
pub fn check_free_list(capacity: u32, ops: &[FreeListOp]) -> Option<FreeListViolation> {
    let mut live: Vec<bool> = Vec::new();
    live.resize(capacity as usize, false);
    for (step, op) in ops.iter().enumerate() {
        let (slot, allocate) = match op {
            FreeListOp::Allocate(slot) => (*slot, true),
            FreeListOp::Free(slot) => (*slot, false),
        };
        let Some(entry) = live.get_mut(slot as usize) else {
            return Some(FreeListViolation {
                step,
                slot,
                fault: FreeListFault::OutOfRange,
            });
        };
        if allocate {
            if *entry {
                return Some(FreeListViolation {
                    step,
                    slot,
                    fault: FreeListFault::DoubleAllocate,
                });
            }
            *entry = true;
        } else {
            if !*entry {
                return Some(FreeListViolation {
                    step,
                    slot,
                    fault: FreeListFault::DoubleFree,
                });
            }
            *entry = false;
        }
    }
    None
}

/// The aggregated verdict of an event-ring back-pressure model.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct BackpressureReport {
    /// The ring capacity that was modeled.
    pub capacity: u32,
    /// Total events dropped across all steps due to overflow.
    pub total_dropped: u32,
    /// The first step at which any event was dropped, if any.
    pub first_drop_step: Option<usize>,
    /// The peak occupancy reached during the run.
    pub peak_occupancy: u32,
}

/// Models a bounded event ring over a deterministic timeline: at each step
/// `produced[i]` events are enqueued (overflow beyond `capacity` is dropped and
/// counted) and then `consumed[i]` events are drained. The timelines are paired
/// by index; a shorter `consumed` slice drains zero on the extra steps. This is
/// a stress-assertion aid — the live ring lives in [`super::feedback`] — that
/// lets a test assert drops happen only when occupancy genuinely overflows.
#[must_use]
pub fn check_event_backpressure(
    capacity: u32,
    produced: &[u32],
    consumed: &[u32],
) -> BackpressureReport {
    let mut occupancy: u32 = 0;
    let mut total_dropped: u32 = 0;
    let mut first_drop_step: Option<usize> = None;
    let mut peak_occupancy: u32 = 0;
    for (step, &incoming) in produced.iter().enumerate() {
        occupancy = occupancy.saturating_add(incoming);
        let dropped = occupancy.saturating_sub(capacity);
        if dropped > 0 {
            total_dropped = total_dropped.saturating_add(dropped);
            if first_drop_step.is_none() {
                first_drop_step = Some(step);
            }
            occupancy = capacity;
        }
        if occupancy > peak_occupancy {
            peak_occupancy = occupancy;
        }
        let drained = consumed.get(step).copied().unwrap_or(0);
        occupancy = occupancy.saturating_sub(drained);
    }
    BackpressureReport {
        capacity,
        total_dropped,
        first_drop_step,
        peak_occupancy,
    }
}

/// The kind of prefix-sum compaction invariant that was broken.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum CompactionFault {
    /// A compacted entry points at a slot that was not live (a dead survivor).
    DeadSurvivor,
    /// The compacted order does not preserve the original slot order (stable
    /// compaction is required so per-particle history stays aligned).
    OrderBroken,
    /// The compacted output has more entries than there were live slots.
    ExtraSurvivor,
    /// A live slot is missing from the compacted output.
    MissingSurvivor,
}

/// A structured compaction violation, naming the output position at fault.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct CompactionViolation {
    /// Position in the compacted output where the fault was detected.
    pub position: usize,
    /// Which invariant was broken.
    pub fault: CompactionFault,
}

/// Validates a stable prefix-sum compaction: `compacted` must list exactly the
/// indices where `live_mask` is `true`, in ascending original order, with no
/// gaps, duplicates, or dead survivors. Returns the first violation or `None`.
#[must_use]
pub fn check_compaction(live_mask: &[bool], compacted: &[usize]) -> Option<CompactionViolation> {
    let mut expected: Vec<usize> = Vec::new();
    for (index, &alive) in live_mask.iter().enumerate() {
        if alive {
            expected.push(index);
        }
    }
    for (position, &slot) in compacted.iter().enumerate() {
        let alive = live_mask.get(slot).copied().unwrap_or(false);
        if !alive {
            return Some(CompactionViolation {
                position,
                fault: CompactionFault::DeadSurvivor,
            });
        }
        match expected.get(position) {
            Some(&want) if want == slot => {}
            Some(_) => {
                return Some(CompactionViolation {
                    position,
                    fault: CompactionFault::OrderBroken,
                });
            }
            None => {
                return Some(CompactionViolation {
                    position,
                    fault: CompactionFault::ExtraSurvivor,
                });
            }
        }
    }
    if compacted.len() < expected.len() {
        return Some(CompactionViolation {
            position: compacted.len(),
            fault: CompactionFault::MissingSurvivor,
        });
    }
    None
}

/// The aggregated verdict of an `OIT` per-pixel layer-budget check.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct OitLayerReport {
    /// Number of pixels inspected.
    pub pixels: usize,
    /// The per-pixel transparent-layer capacity that was checked.
    pub capacity: u32,
    /// Number of pixels whose fragment count exceeded the capacity.
    pub overflowed_pixels: u32,
    /// The largest fragment count seen at any pixel.
    pub max_layers: u32,
    /// The first pixel index to overflow, if any.
    pub first_overflow: Option<usize>,
}

impl OitLayerReport {
    /// Whether every pixel fit within the `OIT` layer capacity.
    #[must_use]
    pub fn within_budget(&self) -> bool {
        self.overflowed_pixels == 0
    }
}

/// Checks per-pixel transparent-fragment counts against an `OIT` layer capacity
/// (the `K` in a `K`-buffer / per-pixel linked-list scheme). Pixels over the
/// capacity would drop or merge layers, so they are counted and the first one
/// is reported.
#[must_use]
pub fn check_oit_layers(fragment_counts: &[u32], capacity: u32) -> OitLayerReport {
    let mut overflowed_pixels: u32 = 0;
    let mut max_layers: u32 = 0;
    let mut first_overflow: Option<usize> = None;
    for (index, &count) in fragment_counts.iter().enumerate() {
        if count > max_layers {
            max_layers = count;
        }
        if count > capacity {
            overflowed_pixels += 1;
            if first_overflow.is_none() {
                first_overflow = Some(index);
            }
        }
    }
    OitLayerReport {
        pixels: fragment_counts.len(),
        capacity,
        overflowed_pixels,
        max_layers,
        first_overflow,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const F32_EPS: f32 = 1e-6;

    fn approx(a: f32, b: f32) -> bool {
        (a - b).abs() < F32_EPS
    }

    #[test]
    fn classify_non_finite_distinguishes_nan_and_infinity() {
        assert_eq!(classify_non_finite(1.0), None);
        assert_eq!(classify_non_finite(f32::NAN), Some(NonFiniteKind::Nan));
        assert_eq!(
            classify_non_finite(f32::INFINITY),
            Some(NonFiniteKind::Infinite)
        );
        assert_eq!(
            classify_non_finite(f32::NEG_INFINITY),
            Some(NonFiniteKind::Infinite)
        );
    }

    #[test]
    fn first_non_finite_scalar_locates_the_offender() {
        let samples = [0.0, 1.0, f32::INFINITY, 2.0];
        assert_eq!(
            first_non_finite_scalar(&samples),
            Some(NonFiniteScalar {
                index: 2,
                kind: NonFiniteKind::Infinite,
            })
        );
        assert_eq!(first_non_finite_scalar(&[0.0, 1.0, 2.0]), None);
    }

    #[test]
    fn first_non_finite_vector_reports_component() {
        let samples = [
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(1.0, f32::NAN, 3.0),
            Vec3::new(4.0, 5.0, 6.0),
        ];
        assert_eq!(
            first_non_finite_vector(&samples),
            Some(NonFiniteVector {
                index: 1,
                component: VectorComponent::Y,
                kind: NonFiniteKind::Nan,
            })
        );
        assert_eq!(first_non_finite_vector(&[Vec3::new(1.0, 2.0, 3.0)]), None);
    }

    #[test]
    fn bounds_guard_flags_first_over_speed() {
        let guard = BoundsGuard::new(10.0, 100.0);
        let velocities = [
            Vec3::new(3.0, 4.0, 0.0),  // magnitude 5, ok
            Vec3::new(9.0, 12.0, 0.0), // magnitude 15, too fast
        ];
        let violation = guard.first_speed_violation(&velocities).unwrap();
        assert_eq!(violation.index, 1);
        assert!(approx(violation.magnitude, 15.0));
        assert!(approx(violation.limit, 10.0));
        assert!(guard.first_speed_violation(&velocities[..1]).is_none());
    }

    #[test]
    fn bounds_guard_flags_position_extent() {
        let guard = BoundsGuard::new(10.0, 5.0);
        let positions = [Vec3::new(1.0, 2.0, 3.0), Vec3::new(0.0, -6.0, 0.0)];
        let violation = guard.first_position_violation(&positions).unwrap();
        assert_eq!(violation.index, 1);
        assert_eq!(violation.component, VectorComponent::Y);
        assert!(approx(violation.value, -6.0));
        assert!(approx(violation.limit, 5.0));
    }

    #[test]
    fn error_tolerance_scales_with_magnitude() {
        let tol = ErrorTolerance::new(0.01, 0.1);
        assert!(approx(tol.allowed(0.0), 0.01));
        assert!(approx(tol.allowed(10.0), 0.01 + 1.0));
        assert!(tol.accepts(100.0, 100.5));
        assert!(!tol.accepts(1.0, 1.5));
    }

    #[test]
    fn momentum_and_mass_totals() {
        let masses = [1.0, 2.0];
        let velocities = [Vec3::new(1.0, 0.0, 0.0), Vec3::new(0.0, 1.0, 0.0)];
        let momentum = total_momentum(&masses, &velocities);
        assert!(approx(momentum.x, 1.0));
        assert!(approx(momentum.y, 2.0));
        assert!(approx(momentum.z, 0.0));
        assert!(approx(total_mass(&masses), 3.0));
    }

    #[test]
    fn momentum_conservation_within_tolerance() {
        let tol = ErrorTolerance::new(1e-3, 1e-4);
        let before = Vec3::new(5.0, 0.0, 0.0);
        let close = Vec3::new(5.0004, 0.0, 0.0);
        let report = check_momentum_conserved(before, close, tol);
        assert!(report.conserved);
        assert!(report.error <= report.allowed);

        let far = Vec3::new(6.0, 0.0, 0.0);
        let broken = check_momentum_conserved(before, far, tol);
        assert!(!broken.conserved);
    }

    #[test]
    fn mass_conservation_detects_drift() {
        let tol = ErrorTolerance::new(1e-3, 0.0);
        assert!(check_mass_conserved(100.0, 100.0005, tol).conserved);
        assert!(!check_mass_conserved(100.0, 101.0, tol).conserved);
    }

    #[test]
    fn scalar_consistency_reports_first_violation_and_max_error() {
        let tol = ErrorTolerance::new(0.01, 0.0);
        let cpu = [1.0, 2.0, 3.0, 4.0];
        let gpu = [1.005, 2.5, 3.0, 5.0];
        match compare_scalar_samples(&cpu, &gpu, tol) {
            ConsistencyOutcome::Compared(report) => {
                assert_eq!(report.compared, 4);
                assert_eq!(report.first_violation, Some(1));
                assert!(!report.consistent);
                assert_eq!(report.max_error_index, 3);
                assert!(approx(report.max_error, 1.0));
            }
            ConsistencyOutcome::LengthMismatch { .. } => panic!("expected comparison"),
        }
    }

    #[test]
    fn scalar_consistency_reports_length_mismatch() {
        let tol = ErrorTolerance::new(0.01, 0.0);
        assert_eq!(
            compare_scalar_samples(&[1.0, 2.0], &[1.0], tol),
            ConsistencyOutcome::LengthMismatch {
                cpu_len: 2,
                gpu_len: 1,
            }
        );
    }

    #[test]
    fn scalar_consistency_all_within_tolerance() {
        let tol = ErrorTolerance::new(0.01, 0.0);
        let cpu = [1.0, 2.0, 3.0];
        let gpu = [1.001, 1.999, 3.0];
        match compare_scalar_samples(&cpu, &gpu, tol) {
            ConsistencyOutcome::Compared(report) => {
                assert!(report.consistent);
                assert_eq!(report.first_violation, None);
            }
            ConsistencyOutcome::LengthMismatch { .. } => panic!("expected comparison"),
        }
    }

    #[test]
    fn vector_consistency_uses_distance() {
        let tol = ErrorTolerance::new(0.05, 0.0);
        let cpu = [Vec3::new(0.0, 0.0, 0.0), Vec3::new(1.0, 0.0, 0.0)];
        let gpu = [Vec3::new(0.0, 0.0, 0.0), Vec3::new(1.0, 0.3, 0.0)];
        match compare_vector_samples(&cpu, &gpu, tol) {
            ConsistencyOutcome::Compared(report) => {
                assert_eq!(report.first_violation, Some(1));
                assert!(approx(report.max_error, 0.3));
            }
            ConsistencyOutcome::LengthMismatch { .. } => panic!("expected comparison"),
        }
    }

    #[test]
    fn golden_aggregation_passes_and_fails() {
        let tol = GoldenTolerance {
            per_pixel_epsilon: 0.1,
            max_failed_pixels: 1,
            mean_error_limit: 0.1,
        };
        let clean = [0.01, 0.02, 0.2, 0.0];
        let report = evaluate_golden(&clean, tol);
        assert_eq!(report.pixels, 4);
        assert_eq!(report.failed_pixels, 1);
        assert!(approx(report.max_error, 0.2));
        assert!(report.passed);

        let noisy = [0.5, 0.5, 0.5, 0.5];
        let bad = evaluate_golden(&noisy, tol);
        assert_eq!(bad.failed_pixels, 4);
        assert!(!bad.passed);
    }

    #[test]
    fn golden_handles_empty_input() {
        let tol = GoldenTolerance {
            per_pixel_epsilon: 0.1,
            max_failed_pixels: 0,
            mean_error_limit: 0.1,
        };
        let report = evaluate_golden(&[], tol);
        assert_eq!(report.pixels, 0);
        assert!(approx(report.mean_error, 0.0));
        assert!(report.passed);
    }

    #[test]
    fn capacity_overflow_is_reported() {
        assert_eq!(check_capacity(64, 40), None);
        assert_eq!(
            check_capacity(64, 100),
            Some(CapacityViolation {
                capacity: 64,
                requested: 100,
                overflow: 36,
            })
        );
    }

    #[test]
    fn free_list_accepts_a_valid_sequence() {
        let ops = [
            FreeListOp::Allocate(0),
            FreeListOp::Allocate(1),
            FreeListOp::Free(0),
            FreeListOp::Allocate(0),
            FreeListOp::Free(1),
        ];
        assert_eq!(check_free_list(4, &ops), None);
    }

    #[test]
    fn free_list_detects_double_allocate() {
        let ops = [FreeListOp::Allocate(2), FreeListOp::Allocate(2)];
        assert_eq!(
            check_free_list(4, &ops),
            Some(FreeListViolation {
                step: 1,
                slot: 2,
                fault: FreeListFault::DoubleAllocate,
            })
        );
    }

    #[test]
    fn free_list_detects_double_free_and_range() {
        let double_free = [FreeListOp::Free(0)];
        assert_eq!(
            check_free_list(4, &double_free),
            Some(FreeListViolation {
                step: 0,
                slot: 0,
                fault: FreeListFault::DoubleFree,
            })
        );
        let out_of_range = [FreeListOp::Allocate(9)];
        assert_eq!(
            check_free_list(4, &out_of_range),
            Some(FreeListViolation {
                step: 0,
                slot: 9,
                fault: FreeListFault::OutOfRange,
            })
        );
    }

    #[test]
    fn backpressure_drops_only_on_overflow() {
        let report = check_event_backpressure(4, &[2, 3, 1], &[0, 1, 0]);
        // step0: occ 2, no drop. step1: occ 5 -> drop 1, occ 4, drain 1 -> 3.
        // step2: occ 4, no drop.
        assert_eq!(report.total_dropped, 1);
        assert_eq!(report.first_drop_step, Some(1));
        assert_eq!(report.peak_occupancy, 4);
    }

    #[test]
    fn backpressure_stays_clean_when_drained() {
        let report = check_event_backpressure(8, &[4, 4, 4], &[4, 4, 4]);
        assert_eq!(report.total_dropped, 0);
        assert_eq!(report.first_drop_step, None);
    }

    #[test]
    fn compaction_accepts_stable_output() {
        let mask = [true, false, true, true, false];
        assert_eq!(check_compaction(&mask, &[0, 2, 3]), None);
    }

    #[test]
    fn compaction_detects_dead_survivor() {
        let mask = [true, false, true];
        assert_eq!(
            check_compaction(&mask, &[0, 1]),
            Some(CompactionViolation {
                position: 1,
                fault: CompactionFault::DeadSurvivor,
            })
        );
    }

    #[test]
    fn compaction_detects_order_and_count_faults() {
        let mask = [true, true, true];
        assert_eq!(
            check_compaction(&mask, &[0, 2, 1]),
            Some(CompactionViolation {
                position: 1,
                fault: CompactionFault::OrderBroken,
            })
        );
        assert_eq!(
            check_compaction(&mask, &[0, 1]),
            Some(CompactionViolation {
                position: 2,
                fault: CompactionFault::MissingSurvivor,
            })
        );
        assert_eq!(
            check_compaction(&mask, &[0, 1, 2, 2]),
            Some(CompactionViolation {
                position: 3,
                fault: CompactionFault::ExtraSurvivor,
            })
        );
    }

    #[test]
    fn oit_layers_counts_overflow() {
        let report = check_oit_layers(&[2, 8, 4, 9], 8);
        assert_eq!(report.pixels, 4);
        assert_eq!(report.capacity, 8);
        assert_eq!(report.overflowed_pixels, 1);
        assert_eq!(report.max_layers, 9);
        assert_eq!(report.first_overflow, Some(3));
        assert!(!report.within_budget());
    }

    #[test]
    fn oit_layers_within_budget() {
        let report = check_oit_layers(&[1, 2, 3], 4);
        assert!(report.within_budget());
        assert_eq!(report.first_overflow, None);
    }
}
