//! §24.4 — the four-way determinism link, exercised end to end in one harness.
//!
//! The design states the contract as **同输入 → 位等价输出**, and requires all
//! four links to hold *simultaneously* or the chain is not deterministic:
//!
//! | link | source | exercised here by |
//! |---|---|---|
//! | 系统/实体序 | ECS §8.4 / §14 | a `Schedule` run on the **parallel** executor, hashed with `WorldSnapshot::state_hash` per tick |
//! | 并行归并     | tasks §24.7 | `TaskPool::deterministic_ordered` (a worker-count-independent non-commutative fold) |
//! | 时间步长     | time §10 / §24.8 | `prism_time::TickClock` (rational `1/60 s` step, integer tick counter) |
//! | 空间传播     | transform §12 | `propagate_fixed` + `hash_globals` (integer fixed-point hierarchy, bit-exact) |
//!
//! The same scenario is replayed with **1 vs 8 workers** and every per-tick
//! value must be bit-identical. A failure names the tick, which is exactly the
//! "首个发散 tick" the design asks the audit to locate.

/// The harness itself; gated on `multi_thread` because it drives the parallel
/// executor. It lives in a module rather than behind a crate-level `cfg` so the
/// crate documentation above survives every feature combination.
#[cfg(feature = "multi_thread")]
mod link {
    use prism_ecs::prelude::*;
    use prism_math::{Fixed, FxVec3};
    use prism_time::TickClock;
    use prism_transform::determinism::{hash_globals, propagate_fixed, FxAffine3};
    use prism_transform::hierarchy::Hierarchy;

    /// Fixed steps to simulate.
    const TICKS: u64 = 120;
    /// Entities per tick, so the ECS hash covers a non-trivial archetype.
    const ENTITIES: usize = 256;
    /// Independent tasks in the ordered fold (a non-commutative accumulate).
    const TASKS: usize = 64;
    /// Nodes in the fixed-point hierarchy chain.
    const NODES: usize = 8;

    /// Table-backed position.
    #[derive(Debug, Clone, Copy, Hash, PartialEq, Eq)]
    struct Pos(i64, i64);
    impl Component for Pos {}

    /// Table-backed velocity.
    #[derive(Debug, Clone, Copy, Hash, PartialEq, Eq)]
    struct Vel(i64, i64);
    impl Component for Vel {}

    /// The tick the ECS system integrates with (design §24.4 时间步长来源).
    #[derive(Debug, Default, PartialEq, Eq)]
    struct TickRes(u64);
    impl Resource for TickRes {}

    /// One tick's four-link record.
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    struct TickRecord {
        tick: u64,
        /// ECS link: hash of the whole world after the schedule ran.
        ecs: u64,
        /// Tasks link: the ordered fold's accumulated value.
        tasks: u64,
        /// Transform link: hash of the fixed-point global transforms.
        transform: u64,
    }

    /// The ECS link's system: integrate every entity by the current tick.
    fn integrate(tick: Res<TickRes>, mut q: Query<(&Vel, &mut Pos)>) {
        let t = tick.0 as i64;
        for (v, mut p) in &mut q {
            p.0 = p.0.wrapping_add(v.0).wrapping_add(t);
            p.1 = p.1.wrapping_add(v.1).wrapping_sub(t);
        }
    }

    /// A depth-`NODES` chain, so propagation order (not just composition) matters.
    fn chain_hierarchy() -> Hierarchy {
        let mut hierarchy = Hierarchy::new();
        let mut parent = hierarchy.spawn_root();
        for _ in 1..NODES {
            parent = hierarchy.spawn_child(parent);
        }
        hierarchy
    }

    /// Replay the scenario on `threads` workers and record all four links per tick.
    fn run_scenario(threads: usize, seed: u64) -> Vec<TickRecord> {
        let pool = prism_tasks::TaskPool::with_threads(threads);

        let mut world = World::new();
        world.register_snapshot_component_hashable::<Pos>();
        world.register_snapshot_component_hashable::<Vel>();
        for i in 0..ENTITIES {
            world.spawn((Pos(i as i64, 0), Vel(3, 5)));
        }
        world.insert_resource(TickRes::default());

        let mut schedule = Schedule::new();
        schedule.add_systems(integrate);

        let hierarchy = chain_hierarchy();
        let mut locals = vec![FxAffine3::IDENTITY; NODES];
        let mut globals = vec![FxAffine3::IDENTITY; NODES];

        let mut clock = TickClock::from_hz(60);
        let mut records = Vec::with_capacity(TICKS as usize);
        for _ in 0..TICKS {
            // ── link 3: rational fixed step + integer tick ──
            clock.step_once();
            let tick = clock.tick();

            // ── link 1: stable schedule order under the parallel executor ──
            world.resource_mut::<TickRes>().0 = tick;
            schedule.run_parallel(&pool, &mut world);
            let ecs = world.snapshot().state_hash();

            // ── link 2: worker-count-independent ordered fold ──
            let tasks = pool.deterministic_ordered(
                seed,
                TASKS,
                |i| {
                    let i = i as u64;
                    (seed ^ i.wrapping_mul(0x9E37_79B9_7F4A_7C15)).rotate_left((i % 63) as u32)
                },
                0u64,
                |acc, i, v| {
                    acc.wrapping_mul(0x0000_0100_0000_01B3)
                        .wrapping_add(v ^ (i as u64).rotate_left(7))
                },
            );

            // ── link 4: bit-exact fixed-point hierarchy propagation ──
            for (i, local) in locals.iter_mut().enumerate() {
                let shift = Fixed::from_int((tick as i64).wrapping_add(i as i64));
                *local = FxAffine3::from_translation(FxVec3::new(
                    shift,
                    Fixed::from_int(1),
                    Fixed::from_int(2),
                ));
            }
            propagate_fixed(&hierarchy, &locals, &mut globals).expect("the chain is acyclic");
            let transform = hash_globals(&globals);

            records.push(TickRecord {
                tick,
                ecs,
                tasks,
                transform,
            });
        }
        records
    }

    #[test]
    fn four_way_link_is_bit_identical_across_worker_counts() {
        let one = run_scenario(1, 0xC0FF_EE00_1234_5678);
        let eight = run_scenario(8, 0xC0FF_EE00_1234_5678);

        assert_eq!(one.len(), TICKS as usize);
        assert_eq!(eight.len(), TICKS as usize);
        for (a, b) in one.iter().zip(&eight) {
            assert_eq!(
                a, b,
                "four-way link diverged at tick {} (1 vs 8 workers)",
                a.tick
            );
        }
        // The fixed step really advanced the integer tick counter (1-based).
        assert_eq!(one.last().expect("120 ticks").tick, TICKS);
    }

    /// The ECS-side auditor consumes exactly these per-tick hashes to locate the
    /// first divergence, so feeding it a worker-count pair must report none.
    #[cfg(feature = "determinism")]
    #[test]
    fn ecs_auditor_locates_no_divergence_across_worker_counts() {
        let one = run_scenario(1, 7);
        let eight = run_scenario(8, 7);

        let mut log_one = FrameHashLog::with_capacity(one.len());
        let mut log_eight = FrameHashLog::with_capacity(eight.len());
        for r in &one {
            log_one.record(r.tick, r.ecs);
        }
        for r in &eight {
            log_eight.record(r.tick, r.ecs);
        }
        assert!(log_one.first_divergence(&log_eight).is_none());
    }
} // mod link
