use super::super::{
    CpuRenderScene, GpuSceneInstance, InstanceRecord, SceneApplyError, SceneBounds, SceneFieldMask,
    SceneHandle, SceneOperation, SceneTransaction, SceneTransactionBuilder, SceneTransform,
    UploadBudget, UploadPlanner, UploadStrategy,
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

#[test]
fn render_layers_update_only_the_instance_upload_and_roll_back_atomically() {
    let mut scene = CpuRenderScene::default();
    let mut create = SceneTransactionBuilder::new(1, 1);
    create.push(SceneOperation::Create {
        handle: handle(1),
        record: InstanceRecord::default(),
    });
    assert!(scene.apply(&create.finish()).errors.is_empty());
    let initial_epoch = scene.scene_epoch();
    let mut update = SceneTransactionBuilder::new(2, 2);
    update.push(SceneOperation::SetRenderLayers {
        handle: handle(1),
        render_layers: 0x8000_0001,
    });
    let report = scene.apply(&update.finish());
    assert!(report.errors.is_empty());
    assert_eq!(report.updated, 1);
    assert_eq!(report.scene_epoch, initial_epoch + 1);
    assert_eq!(report.dirty_slots.len(), 1);
    assert_eq!(report.dirty_slots[0].fields, SceneFieldMask::INSTANCE);
    let row = GpuSceneInstance::from_record(1, scene.get(handle(1)));
    assert_eq!(row.render_layers, 0x8000_0001);
    let upload = UploadPlanner::new(UploadBudget::default())
        .plan(scene.capacity() as u32, &report.dirty_slots);
    assert_ne!(upload.instances.strategy, UploadStrategy::None);
    assert_eq!(upload.current_transforms.strategy, UploadStrategy::None);
    assert_eq!(upload.previous_transforms.strategy, UploadStrategy::None);
    assert_eq!(upload.bounds.strategy, UploadStrategy::None);

    let mut invalid = SceneTransactionBuilder::new(3, 3);
    invalid.push(SceneOperation::SetRenderLayers {
        handle: handle(1),
        render_layers: 2,
    });
    invalid.push(SceneOperation::SetRenderLayers {
        handle: handle(2),
        render_layers: 4,
    });
    let failed = scene.apply(&invalid.finish());
    assert!(!failed.errors.is_empty());
    assert!(failed.dirty_slots.is_empty());
    assert_eq!(failed.updated, 0);
    assert_eq!(scene.scene_epoch(), initial_epoch + 1);
    assert_eq!(scene.get(handle(1)).unwrap().render_layers, 0x8000_0001);
}
