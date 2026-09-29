//! The Extract → Prepare → Queue frame plan for the `GPU`-driven cloth solve.
//!
//! This mirrors the three-stage shape production renderers expose (and the
//! water subsystem's [`crate::water::pipeline`]): an `Extract` that snapshots
//! one cloth piece's `GPU`-relevant sizes for the frame, a `Prepare` that
//! expands the substep loop and graph-color batching into a flat, ordered list
//! of concrete compute dispatches, and a `Queue` that aggregates the per-piece
//! plans into the per-frame dispatch and resident-byte totals the scheduler
//! arbitrates against a budget.
//!
//! Everything here is integer bookkeeping over the [`super::kernels`] contract
//! and the [`super::buffers`] sizing — no floats, no `GPU` handles, no wall
//! clock — so a whole frame's dispatch schedule can be asserted deterministically
//! in `CPU` tests and diffed across builds. The projection passes are emitted
//! once per graph color, in color order, so the recorded schedule preserves the
//! parallel-within-a-color / serial-across-colors Gauss-Seidel ordering the
//! solver depends on.

use alloc::vec::Vec;

use super::buffers::{BufferCounts, PersistentBufferSet};
use super::kernels::{linear_group_count, ClothKernel, DispatchDomain};

/// A snapshot of one cloth piece's `GPU`-relevant sizes for a single frame.
///
/// Produced by [`extract`] from the piece's resident buffer counts and its
/// colored constraint graph. It is deliberately flat and float-free so the
/// prepare stage is a pure function of it.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ClothGpuExtract {
    /// The resident element counts for the piece.
    pub counts: BufferCounts,
    /// The number of distance constraints in each graph color, in color order.
    pub distance_colors: Vec<u32>,
    /// The number of bending constraints in each graph color, in color order.
    pub bending_colors: Vec<u32>,
    /// The number of long-range attachment constraints in each color.
    pub long_range_colors: Vec<u32>,
    /// Substeps per frame (clamped to at least one by [`extract`]).
    pub substeps: u32,
    /// Constraint-projection iterations per substep (clamped to at least one).
    pub iterations: u32,
    /// Whether self-collision (hash build + resolve) runs this frame.
    pub self_collision: bool,
    /// Whether the render-mesh skin embedding runs this frame.
    pub embed: bool,
    /// Whether the painted-backstop projection runs this frame.
    pub backstop: bool,
}

/// One fully sized compute dispatch in the recorded schedule.
///
/// `groups` is the number of workgroups to launch, already divided from the
/// domain extent by the kernel's workgroup tile. `color` records which graph
/// color a projection dispatch belongs to (`None` for the per-particle passes),
/// so the schedule reads back in solver order.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct PlannedDispatch {
    /// The kernel being dispatched.
    pub kernel: ClothKernel,
    /// The number of workgroups to launch.
    pub groups: u32,
    /// The graph color this dispatch belongs to, for the projection passes.
    pub color: Option<u32>,
}

/// The expanded, ordered dispatch schedule for one cloth piece.
///
/// Built by [`prepare`]. The dispatches are in exact record order: for every
/// substep, predict → distance colors → bending colors → long-range colors →
/// strain limit → (optional) self-collision build+resolve → body collision →
/// (optional) backstop → velocity update; then, once per frame, the (optional)
/// skin embed. Self-collision precedes the body/backstop re-projection so the
/// collider and backstop passes hold the final positional authority, matching
/// the CPU golden `ClothPipeline::step` tail order.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ClothGpuPrepare {
    /// The resident buffer set sizing for the piece.
    pub buffers: PersistentBufferSet,
    /// The ordered dispatch schedule.
    pub dispatches: Vec<PlannedDispatch>,
}

impl ClothGpuPrepare {
    /// The total number of workgroups launched across every dispatch in the
    /// schedule, saturating.
    #[must_use]
    pub fn total_groups(&self) -> u64 {
        self.dispatches
            .iter()
            .fold(0u64, |acc, d| acc.saturating_add(u64::from(d.groups)))
    }
}

/// The per-frame aggregate across every cloth piece's plan.
///
/// Produced by [`queue`]; the scheduler reads these totals to arbitrate the
/// cloth solve against the frame's compute and memory budget.
#[derive(Clone, Copy, Debug, Default, Eq, Hash, PartialEq)]
pub struct ClothGpuQueue {
    /// Number of cloth pieces scheduled this frame.
    pub pieces: u32,
    /// Total number of compute dispatches across every piece.
    pub dispatches: u64,
    /// Total number of workgroups across every dispatch.
    pub groups: u64,
    /// Total resident bytes across every piece's persistent buffer set.
    pub resident_bytes: u64,
}

/// The whole frame plan: every piece's extract and prepare, plus the aggregate
/// queue.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ClothGpuFramePlan {
    /// The per-piece snapshots.
    pub extracts: Vec<ClothGpuExtract>,
    /// The per-piece expanded schedules.
    pub prepares: Vec<ClothGpuPrepare>,
    /// The per-frame aggregate.
    pub queue: ClothGpuQueue,
}

/// Snapshots one cloth piece into a [`ClothGpuExtract`], clamping the substep
/// and iteration counts to at least one so the prepare stage never emits an
/// empty substep loop.
#[must_use]
pub fn extract(
    counts: BufferCounts,
    distance_colors: Vec<u32>,
    bending_colors: Vec<u32>,
    long_range_colors: Vec<u32>,
    substeps: u32,
    iterations: u32,
    self_collision: bool,
    embed: bool,
    backstop: bool,
) -> ClothGpuExtract {
    ClothGpuExtract {
        counts,
        distance_colors,
        bending_colors,
        long_range_colors,
        substeps: substeps.max(1),
        iterations: iterations.max(1),
        self_collision,
        embed,
        backstop,
    }
}

/// Pushes the projection dispatches for one constraint class, one per color, in
/// color order. A color with zero constraints is skipped (no work to launch).
fn push_projection(
    dispatches: &mut Vec<PlannedDispatch>,
    kernel: ClothKernel,
    colors: &[u32],
    group_size: u32,
) {
    for (index, &count) in colors.iter().enumerate() {
        let groups = linear_group_count(count, group_size);
        if groups == 0 {
            continue;
        }
        dispatches.push(PlannedDispatch {
            kernel,
            groups,
            color: Some(index as u32),
        });
    }
}

/// Pushes one per-particle (or per-render-vertex) dispatch, skipping it when
/// the domain is empty.
fn push_particle(
    dispatches: &mut Vec<PlannedDispatch>,
    kernel: ClothKernel,
    extent: u32,
    group_size: u32,
) {
    let groups = linear_group_count(extent, group_size);
    if groups == 0 {
        return;
    }
    dispatches.push(PlannedDispatch {
        kernel,
        groups,
        color: None,
    });
}

/// Expands a [`ClothGpuExtract`] into the ordered dispatch schedule.
///
/// The substep loop and per-color projection are unrolled into a flat list in
/// exact record order. Per-particle passes are sized by the particle count;
/// projection passes by each color's constraint count; the hash-grid build by
/// the cell count; the embed by the render-vertex count. Empty domains are
/// skipped so the schedule never records a zero-group dispatch.
#[must_use]
pub fn prepare(extract: &ClothGpuExtract) -> ClothGpuPrepare {
    let buffers = PersistentBufferSet::new(extract.counts);
    let mut dispatches: Vec<PlannedDispatch> = Vec::new();
    let group = particle_group_size();
    let particles = extract.counts.particles;

    for _ in 0..extract.substeps {
        push_particle(&mut dispatches, ClothKernel::Predict, particles, group);

        for _ in 0..extract.iterations {
            push_projection(
                &mut dispatches,
                ClothKernel::ProjectDistanceBatch,
                &extract.distance_colors,
                group,
            );
            push_projection(
                &mut dispatches,
                ClothKernel::ProjectBendingBatch,
                &extract.bending_colors,
                group,
            );
            push_projection(
                &mut dispatches,
                ClothKernel::ProjectLongRangeBatch,
                &extract.long_range_colors,
                group,
            );
        }

        push_particle(&mut dispatches, ClothKernel::StrainLimit, particles, group);

        // Self-collision runs before the body/backstop re-projection so the
        // latter has the final positional authority. This mirrors the CPU
        // golden `ClothPipeline::step` tail order
        // (`resolve_self_collision` -> `resolve_body_collisions` ->
        // `resolve_backstops`), whose comment guarantees "a frame never ends
        // inside a collider": self-collision can shove a particle back into the
        // body, so the collider and backstop passes must be the last positional
        // corrections before the velocity update.
        if extract.self_collision {
            let hash = hash_grid_group_size();
            push_particle(
                &mut dispatches,
                ClothKernel::SelfCollisionHashBuild,
                extract.counts.hash_cells,
                hash,
            );
            push_particle(
                &mut dispatches,
                ClothKernel::SelfCollisionResolve,
                particles,
                group,
            );
        }

        push_particle(
            &mut dispatches,
            ClothKernel::BodyCollision,
            particles,
            group,
        );
        if extract.backstop {
            push_particle(&mut dispatches, ClothKernel::Backstop, particles, group);
        }

        push_particle(
            &mut dispatches,
            ClothKernel::VelocityUpdate,
            particles,
            group,
        );
    }

    if extract.embed {
        push_particle(
            &mut dispatches,
            ClothKernel::SkinEmbed,
            extract.counts.render_vertices,
            group,
        );
    }

    ClothGpuPrepare {
        buffers,
        dispatches,
    }
}

/// The linear workgroup size the per-particle and per-constraint passes launch
/// with, read from the [`ClothKernel::Predict`] descriptor so the plan and the
/// dispatch contract can never drift apart.
#[must_use]
fn particle_group_size() -> u32 {
    ClothKernel::Predict
        .descriptor()
        .workgroup
        .invocations_per_group()
}

/// The workgroup size the self-collision hash-grid build launches with, read
/// from its descriptor for the same reason.
#[must_use]
fn hash_grid_group_size() -> u32 {
    let d = ClothKernel::SelfCollisionHashBuild.descriptor();
    debug_assert!(matches!(d.domain, DispatchDomain::HashGrid));
    d.workgroup.invocations_per_group()
}

/// Aggregates a set of prepared piece plans into the per-frame [`ClothGpuQueue`]
/// totals, all saturating.
#[must_use]
pub fn queue(prepares: &[ClothGpuPrepare]) -> ClothGpuQueue {
    let mut out = ClothGpuQueue {
        pieces: prepares.len() as u32,
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

/// Runs the whole Extract → Prepare → Queue flow over a set of extracted
/// pieces, returning the full [`ClothGpuFramePlan`].
#[must_use]
pub fn plan_frame(extracts: Vec<ClothGpuExtract>) -> ClothGpuFramePlan {
    let prepares: Vec<ClothGpuPrepare> = extracts.iter().map(prepare).collect();
    let queue = queue(&prepares);
    ClothGpuFramePlan {
        extracts,
        prepares,
        queue,
    }
}

#[cfg(test)]
mod tests {
    use super::{extract, plan_frame, prepare, queue};
    use super::{ClothKernel, PlannedDispatch};
    use crate::cloth::gpu::buffers::BufferCounts;
    use alloc::vec;
    use alloc::vec::Vec;

    fn sample_counts() -> BufferCounts {
        BufferCounts {
            particles: 130,
            constraints: 300,
            hash_cells: 64,
            render_vertices: 260,
            backstops: 130,
        }
    }

    fn sample_extract() -> super::ClothGpuExtract {
        extract(
            sample_counts(),
            vec![100, 80],
            vec![60],
            vec![40],
            2,
            1,
            true,
            true,
            true,
        )
    }

    #[test]
    fn extract_clamps_substeps_and_iterations() {
        let e = extract(
            sample_counts(),
            vec![10],
            Vec::new(),
            Vec::new(),
            0,
            0,
            false,
            false,
            false,
        );
        assert_eq!(e.substeps, 1);
        assert_eq!(e.iterations, 1);
    }

    #[test]
    fn schedule_is_in_solver_order_per_substep() {
        let e = sample_extract();
        let plan = prepare(&e);
        // First dispatch of the frame is a predict pass.
        assert_eq!(plan.dispatches[0].kernel, ClothKernel::Predict);
        // The last dispatch of the frame is the once-per-frame embed.
        assert_eq!(
            plan.dispatches.last().unwrap().kernel,
            ClothKernel::SkinEmbed
        );
        // Exactly one embed dispatch even though there are two substeps.
        let embeds = plan
            .dispatches
            .iter()
            .filter(|d| d.kernel == ClothKernel::SkinEmbed)
            .count();
        assert_eq!(embeds, 1);
    }

    #[test]
    fn projection_passes_are_emitted_once_per_nonempty_color() {
        let e = sample_extract();
        let plan = prepare(&e);
        let distance: Vec<&PlannedDispatch> = plan
            .dispatches
            .iter()
            .filter(|d| d.kernel == ClothKernel::ProjectDistanceBatch)
            .collect();
        // Two colors * two substeps * one iteration = four distance dispatches.
        assert_eq!(distance.len(), 4);
        // Colors are recorded in order 0,1 within a substep.
        assert_eq!(distance[0].color, Some(0));
        assert_eq!(distance[1].color, Some(1));
    }

    #[test]
    fn empty_colors_are_skipped() {
        let e = extract(
            sample_counts(),
            vec![0, 0],
            Vec::new(),
            Vec::new(),
            1,
            1,
            false,
            false,
            false,
        );
        let plan = prepare(&e);
        let has_distance = plan
            .dispatches
            .iter()
            .any(|d| d.kernel == ClothKernel::ProjectDistanceBatch);
        assert!(!has_distance);
    }

    #[test]
    fn disabling_self_collision_and_embed_drops_their_passes() {
        let e = extract(
            sample_counts(),
            vec![50],
            Vec::new(),
            Vec::new(),
            1,
            1,
            false,
            false,
            false,
        );
        let plan = prepare(&e);
        assert!(!plan.dispatches.iter().any(|d| {
            matches!(
                d.kernel,
                ClothKernel::SelfCollisionHashBuild
                    | ClothKernel::SelfCollisionResolve
                    | ClothKernel::SkinEmbed
            )
        }));
    }

    #[test]
    fn backstop_pass_follows_body_collision_when_enabled() {
        let e = sample_extract();
        let plan = prepare(&e);
        // Every backstop dispatch is immediately preceded by a body-collision
        // dispatch in the same substep, matching the CPU golden post-collision
        // ordering.
        let mut saw_backstop = false;
        for pair in plan.dispatches.windows(2) {
            if pair[1].kernel == ClothKernel::Backstop {
                saw_backstop = true;
                assert_eq!(pair[0].kernel, ClothKernel::BodyCollision);
            }
        }
        assert!(saw_backstop, "backstop dispatch was not emitted");
        // Two substeps enable it, so two backstop dispatches are recorded.
        let backstops = plan
            .dispatches
            .iter()
            .filter(|d| d.kernel == ClothKernel::Backstop)
            .count();
        assert_eq!(backstops, 2);
    }

    #[test]
    fn self_collision_resolve_precedes_body_reprojection() {
        // The CPU golden `ClothPipeline::step` runs `resolve_self_collision`
        // before `resolve_body_collisions` / `resolve_backstops` so the collider
        // and backstop passes have the final positional authority and a frame
        // never ends inside a collider. The GPU schedule must preserve that
        // per-substep ordering: within each substep the self-collision resolve
        // dispatch appears before the body-collision dispatch.
        let e = sample_extract();
        let plan = prepare(&e);
        let first_self = plan
            .dispatches
            .iter()
            .position(|d| d.kernel == ClothKernel::SelfCollisionResolve)
            .expect("self-collision resolve dispatch was not emitted");
        let first_body = plan
            .dispatches
            .iter()
            .position(|d| d.kernel == ClothKernel::BodyCollision)
            .expect("body-collision dispatch was not emitted");
        assert!(
            first_self < first_body,
            "self-collision resolve ({first_self}) must precede body collision ({first_body})"
        );
        // The hash build always precedes its own resolve.
        let first_hash = plan
            .dispatches
            .iter()
            .position(|d| d.kernel == ClothKernel::SelfCollisionHashBuild)
            .expect("self-collision hash build dispatch was not emitted");
        assert!(
            first_hash < first_self,
            "hash build ({first_hash}) must precede resolve ({first_self})"
        );
    }

    #[test]
    fn disabling_backstop_drops_its_pass() {
        let e = extract(
            sample_counts(),
            vec![50],
            Vec::new(),
            Vec::new(),
            1,
            1,
            false,
            false,
            false,
        );
        let plan = prepare(&e);
        assert!(!plan
            .dispatches
            .iter()
            .any(|d| d.kernel == ClothKernel::Backstop));
    }

    #[test]
    fn iterations_multiply_projection_dispatches() {
        let e = extract(
            sample_counts(),
            vec![100],
            Vec::new(),
            Vec::new(),
            1,
            3,
            false,
            false,
            false,
        );
        let plan = prepare(&e);
        let distance = plan
            .dispatches
            .iter()
            .filter(|d| d.kernel == ClothKernel::ProjectDistanceBatch)
            .count();
        // One color * one substep * three iterations.
        assert_eq!(distance, 3);
    }

    #[test]
    fn queue_aggregates_and_is_deterministic() {
        let e = sample_extract();
        let plan_a = plan_frame(vec![e.clone(), e.clone()]);
        let plan_b = plan_frame(vec![e.clone(), e]);
        assert_eq!(plan_a, plan_b);
        assert_eq!(plan_a.queue.pieces, 2);
        // Two identical pieces: totals are exactly double one piece's.
        let single = prepare(&plan_a.extracts[0]);
        let single_q = queue(&[single]);
        assert_eq!(plan_a.queue.dispatches, single_q.dispatches * 2);
        assert_eq!(plan_a.queue.groups, single_q.groups * 2);
        assert_eq!(plan_a.queue.resident_bytes, single_q.resident_bytes * 2);
    }

    #[test]
    fn total_groups_matches_sum_of_dispatch_groups() {
        let plan = prepare(&sample_extract());
        let manual: u64 = plan.dispatches.iter().map(|d| u64::from(d.groups)).sum();
        assert_eq!(plan.total_groups(), manual);
    }

    #[test]
    fn empty_frame_plan_is_zeroed() {
        let plan = plan_frame(Vec::new());
        assert_eq!(plan.queue, super::ClothGpuQueue::default());
    }
}
