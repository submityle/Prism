//! Prism ECS 自研化基准基线（S-E0）。
//!
//! 这些基准为「成本 ∝ 变化量」内核演进（chunk 变更版本 S-E3、arena S-E2、
//! 命令排序回放 S-E1 等）提供**改造前的权威基线**。每个场景都对应方案
//! `docs/prism_ecs_refactor_plan_zh.md` §9 的目标度量，改造后与此基线对拍。
//!
//! 诚实边界：这些是 CPU 侧机制基线，不代表 GPU/大世界/网络规模；
//! 规模性结论需在真实设备或大场景中另行验证。

use core::hint::black_box;

use bevy_ecs::prelude::*;
use criterion::{criterion_group, BatchSize, Criterion};
use glam::*;

#[derive(Component, Copy, Clone)]
struct Transform(Mat4);

#[derive(Component, Copy, Clone)]
struct Position(Vec3);

#[derive(Component, Copy, Clone)]
struct Velocity(Vec3);

/// 稀疏脏标记：当前内核无 chunk 级版本，查询必须全表扫描才能找到少量脏实体。
/// 这是 S-E3（chunk 变更版本）要击败的基线成本。
#[derive(Component, Copy, Clone)]
struct Active(bool);

const SCALE: usize = 1_000_000;

/// 场景 A：百万实体变换传播（单线程迭代吞吐基线）。
fn transform_propagate_1m(c: &mut Criterion) {
    let mut world = World::new();
    world.spawn_batch(core::iter::repeat_n(
        (
            Transform(Mat4::from_scale(Vec3::ONE)),
            Position(Vec3::X),
            Velocity(Vec3::X),
        ),
        SCALE,
    ));
    let mut query = world.query::<(&Velocity, &mut Position)>();

    c.bench_function("prism_baseline/transform_propagate_1m", |b| {
        b.iter(|| {
            for (velocity, mut position) in query.iter_mut(&mut world) {
                position.0 += velocity.0;
            }
        });
    });
}

/// 场景 B：百万实体并行变换传播（多核扩展基线，对比 Fiber 作业图 S-E7）。
fn par_transform_propagate_1m(c: &mut Criterion) {
    let mut world = World::new();
    world.spawn_batch(core::iter::repeat_n(
        (
            Transform(Mat4::from_scale(Vec3::ONE)),
            Position(Vec3::X),
            Velocity(Vec3::X),
        ),
        SCALE,
    ));
    let mut query = world.query::<(&Velocity, &mut Position)>();

    c.bench_function("prism_baseline/par_transform_propagate_1m", |b| {
        b.iter(|| {
            query
                .par_iter_mut(&mut world)
                .for_each(|(velocity, mut position)| {
                    position.0 += velocity.0;
                });
        });
    });
}

/// 场景 C：稀疏脏（1%）全表扫描基线。
///
/// 百万实体中仅 1% 为脏，但当前内核必须扫描全部才能定位。这是「成本 ∝ 变化量」
/// 的反例基线——改造后（chunk 脏块跳过）应正比于脏比例而非总量。
fn sparse_dirty_1pct_scan(c: &mut Criterion) {
    let mut world = World::new();
    world.spawn_batch((0..SCALE).map(|i| {
        (
            Position(Vec3::X),
            // 每 100 个实体 1 个为活跃（脏）。
            Active(i % 100 == 0),
        )
    }));
    let mut query = world.query::<(&Active, &mut Position)>();

    c.bench_function("prism_baseline/sparse_dirty_1pct_scan", |b| {
        b.iter(|| {
            let mut touched = 0usize;
            for (active, mut position) in query.iter_mut(&mut world) {
                if active.0 {
                    position.0 += Vec3::ONE;
                    touched += 1;
                }
            }
            black_box(touched)
        });
    });
}

/// 场景 D：批量 spawn 结构变更基线（对比 arena S-E2 + 命令排序回放 S-E1）。
fn batch_spawn_200k(c: &mut Criterion) {
    const N: usize = 200_000;
    c.bench_function("prism_baseline/batch_spawn_200k", |b| {
        b.iter_batched(
            World::new,
            |mut world| {
                world.spawn_batch(core::iter::repeat_n(
                    (
                        Transform(Mat4::from_scale(Vec3::ONE)),
                        Position(Vec3::X),
                        Velocity(Vec3::X),
                    ),
                    N,
                ));
                black_box(world.entities().len())
            },
            BatchSize::SmallInput,
        );
    });
}

/// 场景 E：逐个 spawn 命令同步点基线（对比命令排序回放 S-E1 的原型迁移合并）。
fn command_spawn_50k(c: &mut Criterion) {
    const N: usize = 50_000;
    c.bench_function("prism_baseline/command_spawn_50k", |b| {
        b.iter_batched(
            World::new,
            |mut world| {
                let mut queue = bevy_ecs::world::CommandQueue::default();
                let mut commands = Commands::new(&mut queue, &world);
                for _ in 0..N {
                    commands.spawn((Transform(Mat4::from_scale(Vec3::ONE)), Position(Vec3::X)));
                }
                queue.apply(&mut world);
                black_box(world.entities().len())
            },
            BatchSize::SmallInput,
        );
    });
}

criterion_group!(
    benches,
    transform_propagate_1m,
    par_transform_propagate_1m,
    sparse_dirty_1pct_scan,
    batch_spawn_200k,
    command_spawn_50k,
);
