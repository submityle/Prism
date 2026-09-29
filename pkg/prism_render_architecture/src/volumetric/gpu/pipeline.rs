//! The Extract -> Prepare -> Queue frame plan for the `GPU`-driven volumetric
//! cloud/atmosphere solve (design section 17, milestone M8).
//!
//! This mirrors the three-stage shape production renderers expose (and the
//! cloth subsystem's [`crate::cloth::gpu::pipeline`]): an `Extract` that
//! snapshots one cloud domain's `GPU`-relevant sizes and the pass cadence for
//! the frame, a `Prepare` that expands the fixed producer -> consumer chain
//! into a flat, ordered list of concrete compute dispatches, and a `Queue` that
//! aggregates the per-domain plans into the per-frame dispatch and
//! resident-byte totals the scheduler arbitrates against a budget.
//!
//! Unlike the cloth solve (a substep loop over graph-colored constraints), the
//! volumetric frame is a single fixed dependency chain — weather advect ->
//! noise bake -> density modelling -> multiple-scatter `LUT` bake -> view
//! ray-march -> octave-scatter resolve -> cloud-shadow march -> temporal
//! upsample — where each stage may be gated off for the frame by its own
//! cadence (the weather map, the noise/density bakes, the `LUT` and the shadow
//! map are refreshed less often than the per-frame ray-march and upsample).
//! [`prepare`] walks [`super::kernels::VolumetricKernel::ALL`], which is stored
//! in exactly that chain order, so the recorded schedule always preserves the
//! producer-before-consumer ordering the passes depend on.
//!
//! Everything here is integer bookkeeping over the [`super::kernels`] contract
//! and the [`super::buffers`] sizing — no floats, no `GPU` handles, no wall
//! clock — so a whole frame's dispatch schedule can be asserted
//! deterministically in `CPU` tests and diffed across builds.

use alloc::vec::Vec;

use super::buffers::{BufferCounts, PersistentBufferSet};
use super::kernels::{linear_group_count, VolumetricKernel};

/// A bit set over the eight [`VolumetricKernel`] passes recording which run
/// this frame.
///
/// The per-frame cadence gates the expensive bakes (weather, noise, density,
/// multiple-scatter `LUT`, cloud shadow) so they refresh less often than the
/// per-frame ray-march / scatter / upsample. A disabled pass is simply not
/// recorded; its previous resident result is reused. Bit `i` corresponds to
/// `VolumetricKernel::ALL[i]`.
#[derive(Clone, Copy, Debug, Default, Eq, Hash, PartialEq)]
pub struct PassMask(u8);

impl PassMask {
    /// The empty mask: no pass runs.
    pub const NONE: Self = Self(0);

    /// The full mask: every pass runs this frame.
    pub const ALL: Self = Self(u8::MAX);

    /// The bit index of a kernel within [`VolumetricKernel::ALL`].
    #[must_use]
    fn bit(kernel: VolumetricKernel) -> u8 {
        match kernel {
            VolumetricKernel::WeatherAdvect => 0,
            VolumetricKernel::NoiseBake => 1,
            VolumetricKernel::Modeling => 2,
            VolumetricKernel::MultiscatterLutBake => 3,
            VolumetricKernel::Raymarch => 4,
            VolumetricKernel::ScatterResolve => 5,
            VolumetricKernel::ShadowMarch => 6,
            VolumetricKernel::Upsample => 7,
        }
    }

    /// Returns the mask with `kernel` enabled.
    #[must_use]
    pub fn with(self, kernel: VolumetricKernel) -> Self {
        Self(self.0 | (1 << Self::bit(kernel)))
    }

    /// Returns the mask with `kernel` disabled.
    #[must_use]
    pub fn without(self, kernel: VolumetricKernel) -> Self {
        Self(self.0 & !(1 << Self::bit(kernel)))
    }

    /// `true` when `kernel` runs this frame.
    #[must_use]
    pub fn enabled(self, kernel: VolumetricKernel) -> bool {
        (self.0 & (1 << Self::bit(kernel))) != 0
    }

    /// Number of passes enabled.
    #[must_use]
    pub fn count(self) -> u32 {
        self.0.count_ones()
    }

    /// `true` when no pass runs.
    #[must_use]
    pub fn is_empty(self) -> bool {
        self.0 == 0
    }
}

/// A snapshot of one cloud domain's `GPU`-relevant sizes and pass cadence for a
/// single frame.
///
/// Produced by [`extract`] from the domain's resident buffer counts and the
/// frame's cadence decision. It is deliberately flat and float-free so the
/// prepare stage is a pure function of it.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct VolumetricGpuExtract {
    /// The resident element counts (also the dispatch extents per stage).
    pub counts: BufferCounts,
    /// Which passes run this frame.
    pub passes: PassMask,
}

/// One fully sized compute dispatch in the recorded schedule.
///
/// `groups` is the number of workgroups to launch, already divided from the
/// stage's domain extent by the kernel's workgroup tile. `barrier_after`
/// records whether the pass writes a resource a later pass in the same frame
/// reads, so the renderer must insert a barrier before recording the next
/// dispatch.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct PlannedDispatch {
    /// The kernel being dispatched.
    pub kernel: VolumetricKernel,
    /// The number of workgroups to launch.
    pub groups: u32,
    /// Whether a barrier must follow this dispatch before its consumer runs.
    pub barrier_after: bool,
}

/// The expanded, ordered dispatch schedule for one cloud domain.
///
/// Built by [`prepare`]. The dispatches are in exact producer-before-consumer
/// chain order: weather -> noise -> modelling -> multiple-scatter `LUT` ->
/// ray-march -> scatter resolve -> cloud shadow -> upsample, skipping any
/// gated-off pass and any pass whose domain extent is zero.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct VolumetricGpuPrepare {
    /// The resident buffer set sizing for the domain.
    pub buffers: PersistentBufferSet,
    /// The ordered dispatch schedule.
    pub dispatches: Vec<PlannedDispatch>,
}

impl VolumetricGpuPrepare {
    /// The total number of workgroups launched across every dispatch in the
    /// schedule, saturating.
    #[must_use]
    pub fn total_groups(&self) -> u64 {
        self.dispatches
            .iter()
            .fold(0u64, |acc, d| acc.saturating_add(u64::from(d.groups)))
    }
}

/// The per-frame aggregate across every cloud domain's plan.
///
/// Produced by [`queue`]; the scheduler reads these totals to arbitrate the
/// volumetric solve against the frame's compute and memory budget.
#[derive(Clone, Copy, Debug, Default, Eq, Hash, PartialEq)]
pub struct VolumetricGpuQueue {
    /// Number of cloud domains scheduled this frame.
    pub domains: u32,
    /// Total number of compute dispatches across every domain.
    pub dispatches: u64,
    /// Total number of workgroups across every dispatch.
    pub groups: u64,
    /// Total resident bytes across every domain's persistent buffer set.
    pub resident_bytes: u64,
}

/// The whole frame plan: every domain's extract and prepare, plus the aggregate
/// queue.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct VolumetricGpuFramePlan {
    /// The per-domain snapshots.
    pub extracts: Vec<VolumetricGpuExtract>,
    /// The per-domain expanded schedules.
    pub prepares: Vec<VolumetricGpuPrepare>,
    /// The per-frame aggregate.
    pub queue: VolumetricGpuQueue,
}

/// Snapshots one cloud domain into a [`VolumetricGpuExtract`].
#[must_use]
pub fn extract(counts: BufferCounts, passes: PassMask) -> VolumetricGpuExtract {
    VolumetricGpuExtract { counts, passes }
}

/// The dispatch extent (element count over which the launch is sized) for a
/// kernel, read from the domain's resident counts so the plan and the buffer
/// sizing can never drift apart.
#[must_use]
fn kernel_extent(kernel: VolumetricKernel, counts: BufferCounts) -> u32 {
    match kernel {
        VolumetricKernel::WeatherAdvect => counts.weather_texels,
        VolumetricKernel::NoiseBake | VolumetricKernel::Modeling => counts.density_voxels,
        VolumetricKernel::MultiscatterLutBake => counts.multiscatter_cells,
        VolumetricKernel::Raymarch | VolumetricKernel::ScatterResolve => counts.raymarch_tiles,
        VolumetricKernel::ShadowMarch => counts.shadow_texels,
        VolumetricKernel::Upsample => counts.history_pixels,
    }
}

/// Expands a [`VolumetricGpuExtract`] into the ordered dispatch schedule.
///
/// Walks [`VolumetricKernel::ALL`] (stored in producer-before-consumer chain
/// order), emitting one dispatch per enabled pass whose domain extent is
/// non-zero. The workgroup tile is read from each kernel's descriptor so the
/// plan and the dispatch contract can never drift apart; a zero-extent or
/// gated-off pass is skipped so the schedule never records a zero-group
/// dispatch.
#[must_use]
pub fn prepare(extract: &VolumetricGpuExtract) -> VolumetricGpuPrepare {
    let buffers = PersistentBufferSet::new(extract.counts);
    let mut dispatches: Vec<PlannedDispatch> = Vec::new();

    for kernel in VolumetricKernel::ALL {
        if !extract.passes.enabled(kernel) {
            continue;
        }
        let extent = kernel_extent(kernel, extract.counts);
        let group = kernel.descriptor().workgroup.invocations_per_group();
        let groups = linear_group_count(extent, group);
        if groups == 0 {
            continue;
        }
        dispatches.push(PlannedDispatch {
            kernel,
            groups,
            barrier_after: kernel.produces_for_later_pass(),
        });
    }

    VolumetricGpuPrepare {
        buffers,
        dispatches,
    }
}

/// Aggregates a set of prepared domain plans into the per-frame
/// [`VolumetricGpuQueue`] totals, all saturating.
#[must_use]
pub fn queue(prepares: &[VolumetricGpuPrepare]) -> VolumetricGpuQueue {
    let mut out = VolumetricGpuQueue {
        domains: prepares.len() as u32,
        dispatches: 0,
        groups: 0,
        resident_bytes: 0,
    };
    for prepare in prepares {
        out.dispatches = out
            .dispatches
            .saturating_add(prepare.dispatches.len() as u64);
        out.groups = out.groups.saturating_add(prepare.total_groups());
        out.resident_bytes = out
            .resident_bytes
            .saturating_add(u64::from(prepare.buffers.total_bytes()));
    }
    out
}

/// Runs the whole Extract -> Prepare -> Queue flow over a set of extracted
/// domains, returning the full [`VolumetricGpuFramePlan`].
#[must_use]
pub fn plan_frame(extracts: Vec<VolumetricGpuExtract>) -> VolumetricGpuFramePlan {
    let prepares: Vec<VolumetricGpuPrepare> = extracts.iter().map(prepare).collect();
    let queue = queue(&prepares);
    VolumetricGpuFramePlan {
        extracts,
        prepares,
        queue,
    }
}

#[cfg(test)]
mod tests {
    use super::{extract, plan_frame, prepare, queue, PassMask, PlannedDispatch};
    use super::{BufferCounts, VolumetricGpuQueue, VolumetricKernel};
    use alloc::vec;
    use alloc::vec::Vec;

    fn sample_counts() -> BufferCounts {
        BufferCounts {
            density_voxels: 4096,
            weather_texels: 256,
            raymarch_tiles: 480,
            history_pixels: 2048,
            shadow_texels: 512,
            multiscatter_cells: 64,
        }
    }

    #[test]
    fn pass_mask_sets_clears_and_counts() {
        let m = PassMask::NONE
            .with(VolumetricKernel::Raymarch)
            .with(VolumetricKernel::Upsample);
        assert!(m.enabled(VolumetricKernel::Raymarch));
        assert!(m.enabled(VolumetricKernel::Upsample));
        assert!(!m.enabled(VolumetricKernel::NoiseBake));
        assert_eq!(m.count(), 2);
        let cleared = m.without(VolumetricKernel::Raymarch);
        assert!(!cleared.enabled(VolumetricKernel::Raymarch));
        assert_eq!(cleared.count(), 1);
    }

    #[test]
    fn pass_mask_all_enables_every_kernel() {
        for kernel in VolumetricKernel::ALL {
            assert!(PassMask::ALL.enabled(kernel));
        }
        assert_eq!(PassMask::ALL.count(), VolumetricKernel::ALL.len() as u32);
        assert!(PassMask::NONE.is_empty());
    }

    #[test]
    fn full_schedule_is_in_dependency_chain_order() {
        let e = extract(sample_counts(), PassMask::ALL);
        let plan = prepare(&e);
        let order: Vec<VolumetricKernel> = plan.dispatches.iter().map(|d| d.kernel).collect();
        assert_eq!(order, VolumetricKernel::ALL.to_vec());
    }

    #[test]
    fn gated_off_passes_are_skipped() {
        let passes = PassMask::NONE
            .with(VolumetricKernel::Raymarch)
            .with(VolumetricKernel::ScatterResolve)
            .with(VolumetricKernel::Upsample);
        let plan = prepare(&extract(sample_counts(), passes));
        assert_eq!(plan.dispatches.len(), 3);
        assert!(!plan
            .dispatches
            .iter()
            .any(|d| d.kernel == VolumetricKernel::NoiseBake));
    }

    #[test]
    fn zero_extent_passes_are_skipped() {
        let counts = BufferCounts {
            shadow_texels: 0,
            ..sample_counts()
        };
        let plan = prepare(&extract(counts, PassMask::ALL));
        assert!(!plan
            .dispatches
            .iter()
            .any(|d| d.kernel == VolumetricKernel::ShadowMarch));
    }

    #[test]
    fn only_producers_request_a_following_barrier() {
        let plan = prepare(&extract(sample_counts(), PassMask::ALL));
        for d in &plan.dispatches {
            assert_eq!(d.barrier_after, d.kernel.produces_for_later_pass());
        }
        // The two terminal resolves never request a barrier.
        let terminal_barriers = plan
            .dispatches
            .iter()
            .filter(|d| {
                matches!(
                    d.kernel,
                    VolumetricKernel::ScatterResolve | VolumetricKernel::Upsample
                )
            })
            .filter(|d| d.barrier_after)
            .count();
        assert_eq!(terminal_barriers, 0);
    }

    #[test]
    fn total_groups_matches_sum_of_dispatch_groups() {
        let plan = prepare(&extract(sample_counts(), PassMask::ALL));
        let manual: u64 = plan.dispatches.iter().map(|d| u64::from(d.groups)).sum();
        assert_eq!(plan.total_groups(), manual);
    }

    #[test]
    fn queue_aggregates_and_is_deterministic() {
        let e = extract(sample_counts(), PassMask::ALL);
        let plan_a = plan_frame(vec![e, e]);
        let plan_b = plan_frame(vec![e, e]);
        assert_eq!(plan_a, plan_b);
        assert_eq!(plan_a.queue.domains, 2);
        let single = prepare(&plan_a.extracts[0]);
        let single_q = queue(&[single]);
        assert_eq!(plan_a.queue.dispatches, single_q.dispatches * 2);
        assert_eq!(plan_a.queue.groups, single_q.groups * 2);
        assert_eq!(plan_a.queue.resident_bytes, single_q.resident_bytes * 2);
    }

    #[test]
    fn empty_frame_plan_is_zeroed() {
        let plan = plan_frame(Vec::new());
        assert_eq!(plan.queue, VolumetricGpuQueue::default());
    }

    #[test]
    fn a_domain_with_no_passes_records_nothing() {
        let plan = prepare(&extract(sample_counts(), PassMask::NONE));
        assert!(plan.dispatches.is_empty());
        assert_eq!(plan.total_groups(), 0);
    }

    #[test]
    fn planned_dispatch_groups_are_nonzero_when_recorded() {
        let plan = prepare(&extract(sample_counts(), PassMask::ALL));
        for d in &plan.dispatches {
            assert!(
                d.groups > 0,
                "{:?} recorded a zero-group dispatch",
                d.kernel
            );
        }
        let _ = PlannedDispatch {
            kernel: VolumetricKernel::Raymarch,
            groups: 1,
            barrier_after: false,
        };
    }
}
