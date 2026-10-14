use super::super::{
    merge_transactions, InstanceRecord, MergeIssueKind, SceneBounds, SceneChangeKind, SceneHandle,
    SceneOperation, SceneTransaction, SceneTransactionBuilder, SceneTransform,
};

fn handle(index: u32) -> SceneHandle {
    SceneHandle {
        index,
        generation: 1,
    }
}
fn translated(x: f32) -> SceneTransform {
    let mut value = SceneTransform::IDENTITY;
    value.rows[0][3] = x;
    value
}

#[test]
fn create_updates_fold_and_create_destroy_cancels() {
    let mut tx = SceneTransactionBuilder::new(1, 1);
    tx.push(SceneOperation::Create {
        handle: handle(1),
        record: InstanceRecord::default(),
    });
    tx.push(SceneOperation::SetTransform {
        handle: handle(1),
        current: translated(3.0),
    });
    tx.push(SceneOperation::Create {
        handle: handle(2),
        record: InstanceRecord::default(),
    });
    tx.push(SceneOperation::Destroy { handle: handle(2) });
    let report = merge_transactions(&[tx.finish()]);
    assert_eq!(report.canceled_creates, 1);
    assert_eq!(report.transaction.operations.len(), 1);
}

#[test]
fn producer_order_is_deterministic_and_duplicate_keys_are_rejected() {
    let tx = |producer, x| SceneTransaction {
        frame_epoch: 1,
        sequence: 5,
        producer,
        operations: vec![SceneOperation::SetTransform {
            handle: handle(1),
            current: translated(x),
        }],
    };
    let forward = merge_transactions(&[tx(2, 2.0), tx(1, 1.0)]);
    let reverse = merge_transactions(&[tx(1, 1.0), tx(2, 2.0)]);
    assert_eq!(forward.transaction, reverse.transaction);
    let duplicate = merge_transactions(&[tx(1, 1.0), tx(1, 2.0)]);
    assert_eq!(
        duplicate.issues[0].kind,
        MergeIssueKind::DuplicateOrderingKey
    );
}

#[test]
fn every_persistent_instance_field_has_a_delta_kind() {
    let record = InstanceRecord {
        current_transform: translated(1.0),
        previous_transform: translated(0.0),
        bounds: SceneBounds::default(),
        geometry: handle(2),
        material: handle(3),
        render_layers: 0x8000_0001,
        flags: 0x20,
    };
    let target = handle(1);
    let deltas = [
        (
            SceneOperation::Create {
                handle: target,
                record,
            },
            SceneChangeKind::Create,
        ),
        (
            SceneOperation::Destroy { handle: target },
            SceneChangeKind::Destroy,
        ),
        (
            SceneOperation::SetTransform {
                handle: target,
                current: record.current_transform,
            },
            SceneChangeKind::Transform,
        ),
        (
            SceneOperation::SetBounds {
                handle: target,
                bounds: record.bounds,
            },
            SceneChangeKind::Bounds,
        ),
        (
            SceneOperation::SetGeometry {
                handle: target,
                geometry: record.geometry,
            },
            SceneChangeKind::Geometry,
        ),
        (
            SceneOperation::SetMaterial {
                handle: target,
                material: record.material,
            },
            SceneChangeKind::Material,
        ),
        (
            SceneOperation::SetFlags {
                handle: target,
                mask: u32::MAX,
                value: record.flags,
            },
            SceneChangeKind::Flags,
        ),
        (
            SceneOperation::SetRenderLayers {
                handle: target,
                render_layers: record.render_layers,
            },
            SceneChangeKind::RenderLayers,
        ),
    ];
    for (operation, kind) in deltas {
        assert_eq!(operation.change_kind(), kind);
        assert_eq!(operation.handle(), target);
    }
}

#[test]
fn render_layers_fold_into_creation_and_last_update_wins() {
    let mut create = SceneTransactionBuilder::new(1, 1);
    create.push(SceneOperation::Create {
        handle: handle(1),
        record: InstanceRecord::default(),
    });
    create.push(SceneOperation::SetRenderLayers {
        handle: handle(1),
        render_layers: 2,
    });
    create.push(SceneOperation::SetRenderLayers {
        handle: handle(1),
        render_layers: 0x8000_0001,
    });
    let merged = merge_transactions(&[create.finish()]);
    assert!(merged.issues.is_empty());
    assert!(matches!(
        merged.transaction.operations.as_slice(),
        [SceneOperation::Create {
            record: InstanceRecord {
                render_layers: 0x8000_0001,
                ..
            },
            ..
        }]
    ));

    let mut update = SceneTransactionBuilder::new(2, 2);
    update.push(SceneOperation::SetRenderLayers {
        handle: handle(1),
        render_layers: 2,
    });
    update.push(SceneOperation::SetRenderLayers {
        handle: handle(1),
        render_layers: 0x8000_0001,
    });
    let merged = merge_transactions(&[update.finish()]);
    assert!(merged.issues.is_empty());
    assert_eq!(
        merged.transaction.operations,
        vec![SceneOperation::SetRenderLayers {
            handle: handle(1),
            render_layers: 0x8000_0001
        }]
    );
}
