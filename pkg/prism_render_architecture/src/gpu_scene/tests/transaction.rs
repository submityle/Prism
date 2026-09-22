use super::super::{
    merge_transactions, InstanceRecord, MergeIssueKind, SceneHandle, SceneOperation,
    SceneTransaction, SceneTransactionBuilder, SceneTransform,
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
