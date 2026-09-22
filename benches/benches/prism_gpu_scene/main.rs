use criterion::{criterion_group, criterion_main, BatchSize, Criterion};
use prism_render_architecture::gpu_scene::{
    CpuRenderScene, InstanceRecord, SceneHandle, SceneOperation, SceneTransaction, UploadBudget,
    UploadPlanner,
};
use std::hint::black_box;

const INSTANCE_COUNT: u32 = 100_000;
const DIRTY_COUNT: u32 = INSTANCE_COUNT / 100;

fn create_operations() -> (Vec<SceneOperation>, Vec<SceneHandle>) {
    let handles: Vec<_> = (1..=INSTANCE_COUNT)
        .map(|index| SceneHandle {
            index,
            generation: 1,
        })
        .collect();
    let operations = handles
        .iter()
        .copied()
        .map(|handle| SceneOperation::Create {
            handle,
            record: InstanceRecord::default(),
        })
        .collect();
    (operations, handles)
}

fn update_one_percent(c: &mut Criterion) {
    let (create, handles) = create_operations();
    c.bench_function("prism_gpu_scene/100k_update_1_percent", |b| {
        b.iter_batched(
            || {
                let mut scene = CpuRenderScene::default();
                let report = scene.apply(&SceneTransaction {
                    frame_epoch: 1,
                    sequence: 1,
                    producer: 1,
                    operations: create.clone(),
                });
                assert!(report.errors.is_empty());
                scene
            },
            |mut scene| {
                let operations = handles
                    .iter()
                    .step_by(100)
                    .take(DIRTY_COUNT as usize)
                    .copied()
                    .map(|handle| SceneOperation::SetTransform {
                        handle,
                        current: Default::default(),
                    })
                    .collect();
                let report = scene.apply(&SceneTransaction {
                    frame_epoch: 2,
                    sequence: 2,
                    producer: 1,
                    operations,
                });
                let plan = UploadPlanner::new(UploadBudget::default())
                    .plan(scene.capacity() as u32, &report.dirty_slots);
                black_box((report, plan));
            },
            BatchSize::LargeInput,
        );
    });
}

criterion_group!(benches, update_one_percent);
criterion_main!(benches);
