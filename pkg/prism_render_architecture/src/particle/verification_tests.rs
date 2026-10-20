//! Cross-contract stability and resource-budget gates for the particle system.
//!
//! These tests validate deterministic CPU reference behavior and structured
//! resource/dispatch ceilings. Millisecond values are design constraints only;
//! no device timing is collected here.

use alloc::vec::Vec;

use super::determinism::{replay_sample_digest, DeterminismConfig, RngKey, StreamId};
use super::events::{AppendOutcome, ChannelRecord, EventRing, EventRouter};
use super::frame_pipeline::{
    build_frame_plan_default, EmitterFrameState, FramePass, PipelineResource,
};
use super::merge_sort_stable::merge_sort_by_key;
use super::perf_budget::{
    estimate_attribute_vram_bytes, FrameStage, FrameTimeBudget, ParticleAttribute,
};
use super::pool::{CapacityPolicy, ParticlePool};
use super::simulation::{gravity, integrate, Kinematics};
use super::sort_cull::{
    apply_hzb, choose_sort_strategy, cull_sphere, sort_key, BlendMode, CullReason, Frustum, Plane,
    SortDecision,
};
use super::{IntegratorKind, Vec3};
use crate::particle::dual_backend::{BackendCapabilities, BackendTuning};

#[test]
fn capacity_overflow_and_recycling_preserve_pool_invariants() {
    let mut pool = ParticlePool::with_capacity(257);
    let mut accepted = Vec::new();
    for _ in 0..300 {
        if let Some(slot) = pool.spawn(CapacityPolicy::DiscardNewest) {
            accepted.push(slot);
        }
    }
    assert_eq!(accepted.len(), 257);
    assert_eq!(pool.alive_count(), pool.capacity());
    assert_eq!(pool.dead_count(), 0);
    assert_eq!(pool.spawn(CapacityPolicy::DiscardNewest), None);

    // Retiring arbitrary live slots must make each slot reusable exactly once.
    for slot in accepted.iter().copied().step_by(3) {
        pool.kill(slot);
        pool.kill(slot);
    }
    let freed = pool.dead_count();
    let mut reused = Vec::new();
    for _ in 0..freed {
        reused.push(pool.spawn(CapacityPolicy::DiscardNewest).unwrap());
    }
    reused.sort_unstable();
    assert_eq!(reused.len() as u32, freed);
    assert!(reused.windows(2).all(|pair| pair[0] != pair[1]));
    assert_eq!(pool.alive_count(), 257);
    assert_eq!(pool.spawn(CapacityPolicy::DiscardNewest), None);

    let mut recycled = ParticlePool::with_capacity(3);
    let first = recycled.spawn(CapacityPolicy::DiscardNewest).unwrap();
    recycled.spawn(CapacityPolicy::DiscardNewest).unwrap();
    recycled.spawn(CapacityPolicy::DiscardNewest).unwrap();
    assert_eq!(recycled.spawn(CapacityPolicy::RecycleOldest), Some(first));
    assert_eq!(recycled.alive_count(), 3);
    assert_eq!(recycled.spawn_count(), 4);
}

#[test]
fn free_list_compaction_matches_stable_alive_mask_for_fragmented_pool() {
    let mut pool = ParticlePool::with_capacity(513);
    let slots: Vec<u32> = (0..513)
        .map(|_| pool.spawn(CapacityPolicy::DiscardNewest).unwrap())
        .collect();
    let expected: Vec<u32> = slots
        .iter()
        .copied()
        .filter(|slot| slot % 5 != 1 && slot % 7 != 3)
        .collect();
    for slot in slots.iter().copied() {
        if slot % 5 == 1 || slot % 7 == 3 {
            pool.kill(slot);
        }
    }

    let compacted = pool.compact();
    assert_eq!(compacted.alive_list, expected);
    assert_eq!(compacted.len() as u32, pool.alive_count());
    for slot in 0..pool.capacity() {
        let expected_scatter = expected
            .iter()
            .position(|alive| *alive == slot)
            .map(|i| i as u32);
        assert_eq!(compacted.scatter[slot as usize], expected_scatter);
        assert_eq!(pool.is_alive(slot), expected_scatter.is_some());
    }

    // Reused free slots re-enter the same stable scan order, independent of
    // free-list pop order.
    let reclaimed: Vec<u32> = (0..pool.dead_count())
        .map(|_| pool.spawn(CapacityPolicy::DiscardNewest).unwrap())
        .collect();
    assert_eq!(reclaimed.len() as u32, 513 - expected.len() as u32);
    let after_reuse = pool.compact();
    assert!(after_reuse
        .alive_list
        .windows(2)
        .all(|pair| pair[0] < pair[1]));
    assert_eq!(after_reuse.len(), 513);
}

#[test]
fn event_backpressure_is_bounded_saturating_and_frame_scoped() {
    let mut ring = EventRing::new(5);
    let first = ring.reserve(3);
    let overflow = ring.reserve(u32::MAX);
    assert_eq!((first.granted, first.dropped), (3, 0));
    assert_eq!((overflow.granted, overflow.dropped), (2, u32::MAX - 2));
    assert_eq!(ring.len(), ring.capacity());
    assert_eq!(ring.dropped_total(), u32::MAX - 2);
    assert_eq!(ring.reserve(3).dropped, 3);
    assert_eq!(ring.dropped_total(), u32::MAX);
    ring.begin_frame();
    assert_eq!(ring.len(), 0);
    assert_eq!(ring.dropped_total(), u32::MAX);

    let mut router = EventRouter::new();
    assert!(router.register_channel(7, 1));
    assert!(!router.register_channel(7, 99));
    let record = ChannelRecord::new(Vec3::ZERO, Vec3::ZERO, 2.0, 7);
    assert_eq!(router.publish(7, record), AppendOutcome::Accepted);
    assert_eq!(router.publish(7, record), AppendOutcome::Dropped);
    assert_eq!(router.publish(8, record), AppendOutcome::NoSuchChannel);
    assert_eq!(router.total_dropped(), 1);
    assert_eq!(router.snapshot_len(7), Some(0));
    router.publish_frame();
    assert_eq!(router.snapshot(7), Some(&[record][..]));
    router.begin_frame();
    assert_eq!(router.snapshot_len(7), Some(1));
}

fn simulate_cpu_partitioned(states: &mut [Kinematics], partitions: usize, dt: f32) {
    let partition_size = states.len().div_ceil(partitions.max(1));
    for chunk in states.chunks_mut(partition_size.max(1)) {
        for state in chunk {
            *state = integrate(IntegratorKind::Rk2, *state, dt, |_, _| {
                gravity(Vec3::new(0.0, -9.81, 0.25))
            });
        }
    }
}

#[test]
fn deterministic_cpu_parity_survives_chunk_partitioning_and_replay() {
    let config = DeterminismConfig::new(1.0 / 120.0, 0xA17E_2026, true);
    let ids: Vec<u32> = (0..1025).collect();
    let streams = [
        StreamId::InitialPosition,
        StreamId::Color,
        StreamId::Custom(41),
    ];
    let digest_a = replay_sample_digest(&config, &ids, &streams, 10, 50);
    let digest_b = replay_sample_digest(&config, &ids, &streams, 10, 50);
    assert_eq!(digest_a.finish(), digest_b.finish());

    let mut serial: Vec<Kinematics> = ids
        .iter()
        .map(|&id| {
            let rng = RngKey::for_stream(id, config.global_seed, StreamId::InitialVelocity, 0);
            Kinematics::new(
                Vec3::new(
                    rng.range_f32(0, -3.0, 3.0),
                    rng.range_f32(1, -2.0, 2.0),
                    0.0,
                ),
                Vec3::new(rng.range_f32(2, -1.0, 1.0), rng.range_f32(3, 0.0, 4.0), 0.0),
            )
        })
        .collect();
    let mut partitioned = serial.clone();
    for _ in 0..120 {
        simulate_cpu_partitioned(&mut serial, 1, config.dt());
        simulate_cpu_partitioned(&mut partitioned, 17, config.dt());
    }
    assert_eq!(serial, partitioned);
    assert!(serial.iter().all(|state| {
        state.position.x.is_finite()
            && state.position.y.is_finite()
            && state.position.z.is_finite()
            && state.velocity.x.is_finite()
            && state.velocity.y.is_finite()
            && state.velocity.z.is_finite()
    }));
}

#[test]
fn stable_sort_cull_pipeline_preserves_ties_and_rejects_only_invisible_spheres() {
    let mut records = vec![
        (sort_key(80.0, 0.0, 100.0, true), 0u32),
        (sort_key(20.0, 0.0, 100.0, true), 1),
        (sort_key(80.0, 0.0, 100.0, true), 2),
        (sort_key(20.0, 0.0, 100.0, true), 3),
    ];
    merge_sort_by_key(&mut records, |record| u32::from(record.0));
    let sorted_keys: Vec<u16> = records.iter().map(|record| record.0).collect();
    assert_eq!(
        records.iter().map(|record| record.1).collect::<Vec<_>>(),
        [0, 2, 1, 3]
    );
    assert!(sorted_keys.windows(2).all(|pair| pair[0] <= pair[1]));

    let frustum = Frustum::new([
        Plane::new(Vec3::new(1.0, 0.0, 0.0), 1.0),
        Plane::new(Vec3::new(-1.0, 0.0, 0.0), 1.0),
        Plane::new(Vec3::new(0.0, 1.0, 0.0), 1.0),
        Plane::new(Vec3::new(0.0, -1.0, 0.0), 1.0),
        Plane::new(Vec3::new(0.0, 0.0, 1.0), 1.0),
        Plane::new(Vec3::new(0.0, 0.0, -1.0), 1.0),
    ]);
    assert!(cull_sphere(frustum, Vec3::ZERO, Vec3::ZERO, 0.25, 10.0).visible);
    assert_eq!(
        cull_sphere(frustum, Vec3::ZERO, Vec3::new(3.0, 0.0, 0.0), 0.1, 10.0).reason,
        CullReason::OutsideFrustum
    );
    let visible = cull_sphere(frustum, Vec3::ZERO, Vec3::ZERO, 0.25, 10.0);
    assert_eq!(apply_hzb(visible, true).reason, CullReason::HzbOccluded);
    assert_eq!(
        choose_sort_strategy(SortDecision {
            blend: BlendMode::AlphaBlend,
            particle_count: 4096,
            radix_min_count: 4096,
            prefer_shared_oit: false,
        }),
        super::SortStrategy::ViewDepthRadix
    );
}

#[test]
fn stage_barrier_plan_covers_hazards_without_serializing_independent_work() {
    let state = EmitterFrameState {
        spawn_count: 64,
        alive_count: 4096,
        stale_slots: 4096,
        capacity: 8192,
        free_count: 0,
        visible_translucent_count: 2048,
        visible_opaque_count: 1024,
    };
    let plan = build_frame_plan_default(
        state,
        BackendCapabilities::desktop_gpu(),
        BackendTuning::default(),
    );
    let passes: Vec<FramePass> = plan.steps.iter().map(|step| step.pass).collect();
    assert_eq!(
        passes,
        [
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
        ]
    );
    let barriers: Vec<usize> = plan
        .steps
        .iter()
        .enumerate()
        .filter_map(|(index, step)| step.needs_barrier_before.then_some(index))
        .collect();
    assert_eq!(barriers, [1, 2, 3, 5, 6, 7, 8, 9]);
    assert_eq!(plan.barrier_count as usize, barriers.len());
    assert!(plan.steps[3].needs_barrier_before);
    for index in barriers {
        let step = plan.steps[index];
        let prior_writes = plan.steps[..index]
            .iter()
            .fold(0u32, |bits, prior| bits | prior.access.writes.bits);
        assert_ne!(
            prior_writes & (step.access.reads.bits | step.access.writes.bits),
            0,
            "barrier before {:?} must correspond to a resource hazard",
            step.pass
        );
    }
    assert!(plan.steps[0]
        .access
        .writes
        .contains(PipelineResource::IndirectArgs));
}

#[derive(Clone, Copy)]
struct PerformanceContract {
    label: &'static str,
    particle_count: u32,
    attributes: &'static [ParticleAttribute],
    max_vram_bytes: u64,
    max_total_ms: f32,
}

const BASIC_ATTRIBUTES: &[ParticleAttribute] = &[
    ParticleAttribute::Position,
    ParticleAttribute::Velocity,
    ParticleAttribute::Color,
    ParticleAttribute::Age,
    ParticleAttribute::Size,
];
const LIT_ATTRIBUTES: &[ParticleAttribute] = &[
    ParticleAttribute::Position,
    ParticleAttribute::Velocity,
    ParticleAttribute::Color,
    ParticleAttribute::Normal,
    ParticleAttribute::Age,
    ParticleAttribute::Size,
    ParticleAttribute::Custom32,
];

const PERFORMANCE_CONTRACTS: [PerformanceContract; 3] = [
    PerformanceContract {
        label: "desktop_ultra_basic_1m",
        particle_count: 1_000_000,
        attributes: BASIC_ATTRIBUTES,
        max_vram_bytes: 32 * 1024 * 1024,
        max_total_ms: 5.5,
    },
    PerformanceContract {
        label: "desktop_ultra_lit_1m",
        particle_count: 1_000_000,
        attributes: LIT_ATTRIBUTES,
        max_vram_bytes: 32 * 1024 * 1024,
        max_total_ms: 5.5,
    },
    PerformanceContract {
        label: "desktop_ultra_lit_100k",
        particle_count: 100_000,
        attributes: LIT_ATTRIBUTES,
        max_vram_bytes: 8 * 1024 * 1024,
        max_total_ms: 5.5,
    },
];

#[test]
fn performance_design_contracts_fit_particle_memory_dispatch_and_time_envelopes() {
    let frame_budget = FrameTimeBudget::ULTRA_DESKTOP;
    assert_eq!(
        frame_budget.stage_targets_ms[FrameStage::Simulation.index()],
        1.0
    );
    assert_eq!(
        frame_budget.stage_targets_ms[FrameStage::GridFluid.index()],
        1.5
    );
    assert_eq!(frame_budget.stage_targets_ms[FrameStage::Sort.index()], 0.4);
    assert_eq!(
        frame_budget.stage_targets_ms[FrameStage::CullBounds.index()],
        0.2
    );
    assert_eq!(
        frame_budget.stage_targets_ms[FrameStage::VolumetricShading.index()],
        0.8
    );
    assert_eq!(frame_budget.stage_targets_ms[FrameStage::Draw.index()], 1.5);
    assert_eq!(frame_budget.stage_sum_ms(), 5.4);

    for contract in PERFORMANCE_CONTRACTS {
        let estimated_bytes =
            estimate_attribute_vram_bytes(contract.particle_count, contract.attributes);
        assert!(
            estimated_bytes <= contract.max_vram_bytes,
            "{} uses {estimated_bytes} bytes, ceiling {}",
            contract.label,
            contract.max_vram_bytes
        );
        assert!(contract.max_total_ms >= frame_budget.stage_sum_ms());
        assert!(contract.max_total_ms <= frame_budget.total_target_ms);
        let dispatch_groups = contract.particle_count.div_ceil(64);
        assert_eq!(
            dispatch_groups.saturating_mul(64) >= contract.particle_count,
            true
        );
        assert!(
            dispatch_groups <= 65_535,
            "{} exceeds one desktop dispatch",
            contract.label
        );
    }
}
