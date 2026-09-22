use super::super::{
    CpuRenderScene, InstanceRecord, SceneApplyError, SceneBounds, SceneHandle, SceneOperation,
    SceneTransaction, SceneTransactionBuilder, SceneTransform,
};

fn handle(generation: u32) -> SceneHandle {
    SceneHandle {
        index: 1,
        generation,
    }
}

fn translated(x: f32) -> SceneTransform {
    let mut value = SceneTransform::IDENTITY;
    value.rows[0][3] = x;
    value
}

#[test]
fn previous_transform_is_captured_once_per_frame() {
    let mut scene = CpuRenderScene::default();
    let mut create = SceneTransactionBuilder::new(1, 1);
    create.push(SceneOperation::Create {
        handle: handle(1),
        record: InstanceRecord::default(),
    });
    scene.apply(&create.finish());
    let mut updates = SceneTransactionBuilder::new(2, 2);
    updates.push(SceneOperation::SetTransform {
        handle: handle(1),
        current: translated(1.0),
    });
    updates.push(SceneOperation::SetTransform {
        handle: handle(1),
        current: translated(2.0),
    });
    scene.apply(&updates.finish());
    let record = scene.get(handle(1)).unwrap();
    assert_eq!(record.previous_transform, SceneTransform::IDENTITY);
    assert_eq!(record.current_transform, translated(2.0));
}

#[test]
fn failed_transaction_rolls_back_and_tombstone_rejects_stale_generation() {
    let mut scene = CpuRenderScene::default();
    scene.apply(&SceneTransaction {
        frame_epoch: 1,
        sequence: 1,
        producer: 0,
        operations: vec![SceneOperation::Create {
            handle: handle(2),
            record: InstanceRecord::default(),
        }],
    });
    let failed = scene.apply(&SceneTransaction {
        frame_epoch: 2,
        sequence: 2,
        producer: 0,
        operations: vec![SceneOperation::SetBounds {
            handle: handle(1),
            bounds: SceneBounds::default(),
        }],
    });
    assert!(matches!(
        failed.errors[0],
        SceneApplyError::StaleHandle { .. }
    ));
    scene.apply(&SceneTransaction {
        frame_epoch: 3,
        sequence: 3,
        producer: 0,
        operations: vec![SceneOperation::Destroy { handle: handle(2) }],
    });
    let stale = scene.apply(&SceneTransaction {
        frame_epoch: 4,
        sequence: 4,
        producer: 0,
        operations: vec![SceneOperation::Create {
            handle: handle(2),
            record: InstanceRecord::default(),
        }],
    });
    assert!(matches!(
        stale.errors[0],
        SceneApplyError::NonMonotonicGeneration { .. }
    ));
}
