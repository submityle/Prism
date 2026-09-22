use alloc::collections::BTreeMap;

use super::{
    GeometryHandle, InstanceRecord, SceneBounds, SceneHandle, SceneMaterialHandle, SceneTransform,
};

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum SceneOperation {
    Create {
        handle: SceneHandle,
        record: InstanceRecord,
    },
    Destroy {
        handle: SceneHandle,
    },
    SetTransform {
        handle: SceneHandle,
        current: SceneTransform,
    },
    SetBounds {
        handle: SceneHandle,
        bounds: SceneBounds,
    },
    SetGeometry {
        handle: SceneHandle,
        geometry: GeometryHandle,
    },
    SetMaterial {
        handle: SceneHandle,
        material: SceneMaterialHandle,
    },
    SetFlags {
        handle: SceneHandle,
        mask: u32,
        value: u32,
    },
    SetRenderLayers {
        handle: SceneHandle,
        render_layers: u32,
    },
}

impl SceneOperation {
    pub const fn handle(&self) -> SceneHandle {
        match *self {
            Self::Create { handle, .. }
            | Self::Destroy { handle }
            | Self::SetTransform { handle, .. }
            | Self::SetBounds { handle, .. }
            | Self::SetGeometry { handle, .. }
            | Self::SetMaterial { handle, .. }
            | Self::SetFlags { handle, .. }
            | Self::SetRenderLayers { handle, .. } => handle,
        }
    }
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct SceneTransaction {
    pub frame_epoch: u64,
    pub sequence: u64,
    pub producer: u32,
    pub operations: Vec<SceneOperation>,
}

pub struct SceneTransactionBuilder {
    transaction: SceneTransaction,
}

impl SceneTransactionBuilder {
    pub fn new(frame_epoch: u64, sequence: u64) -> Self {
        Self {
            transaction: SceneTransaction {
                frame_epoch,
                sequence,
                producer: 0,
                operations: Vec::new(),
            },
        }
    }

    pub fn for_producer(frame_epoch: u64, sequence: u64, producer: u32) -> Self {
        let mut builder = Self::new(frame_epoch, sequence);
        builder.transaction.producer = producer;
        builder
    }

    pub fn push(&mut self, operation: SceneOperation) -> &mut Self {
        self.transaction.operations.push(operation);
        self
    }

    pub fn finish(self) -> SceneTransaction {
        self.transaction
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum MergeIssueKind {
    InvalidHandle,
    OperationAfterDestroy,
    DuplicateCreate,
    CreateAfterDestroyWithSameGeneration,
    MixedFrameEpoch,
    DuplicateOrderingKey,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct MergeIssue {
    pub handle: SceneHandle,
    pub sequence: u64,
    pub operation_index: u32,
    pub kind: MergeIssueKind,
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct MergeReport {
    pub transaction: SceneTransaction,
    pub issues: Vec<MergeIssue>,
    pub canceled_creates: u32,
}

#[derive(Clone, Copy, Debug)]
enum Pending {
    Create(InstanceRecord),
    Update(UpdateRecord),
    Destroy,
    CanceledCreate,
}

#[derive(Clone, Copy, Debug, Default)]
struct UpdateRecord {
    transform: Option<SceneTransform>,
    bounds: Option<SceneBounds>,
    geometry: Option<GeometryHandle>,
    material: Option<SceneMaterialHandle>,
    flags: Option<(u32, u32)>,
    render_layers: Option<u32>,
}

/// Deterministically merges thread-local scene transactions for one frame.
pub fn merge_transactions(transactions: &[SceneTransaction]) -> MergeReport {
    if transactions.is_empty() {
        return MergeReport::default();
    }

    let mut order: Vec<_> = transactions.iter().enumerate().collect();
    order.sort_by_key(|(_, transaction)| (transaction.sequence, transaction.producer));
    let frame_epoch = transactions
        .iter()
        .map(|transaction| transaction.frame_epoch)
        .min()
        .unwrap_or_default();
    let mut ordering_key_counts = BTreeMap::new();
    for transaction in transactions {
        *ordering_key_counts
            .entry((transaction.sequence, transaction.producer))
            .or_insert(0_u32) += 1;
    }
    let duplicate_ordering_keys: Vec<_> = ordering_key_counts
        .into_iter()
        .filter_map(|(key, count)| (count > 1).then_some(key))
        .collect();
    let mut pending = BTreeMap::<SceneHandle, Pending>::new();
    let mut issues: Vec<_> = duplicate_ordering_keys
        .iter()
        .map(|(sequence, _)| MergeIssue {
            handle: SceneHandle::INVALID,
            sequence: *sequence,
            operation_index: 0,
            kind: MergeIssueKind::DuplicateOrderingKey,
        })
        .collect();
    let mut canceled_creates = 0;

    for (_, transaction) in order {
        let ordering_key = (transaction.sequence, transaction.producer);
        if duplicate_ordering_keys.binary_search(&ordering_key).is_ok() {
            continue;
        }
        if transaction.frame_epoch != frame_epoch {
            issues.push(MergeIssue {
                handle: SceneHandle::INVALID,
                sequence: transaction.sequence,
                operation_index: 0,
                kind: MergeIssueKind::MixedFrameEpoch,
            });
            continue;
        }
        for (operation_index, operation) in transaction.operations.iter().copied().enumerate() {
            let handle = operation.handle();
            if !handle.is_valid() || handle.index == 0 {
                issues.push(issue(
                    handle,
                    transaction.sequence,
                    operation_index,
                    MergeIssueKind::InvalidHandle,
                ));
                continue;
            }
            match operation {
                SceneOperation::Create { record, .. } => match pending.get(&handle) {
                    Some(Pending::Destroy) | Some(Pending::CanceledCreate) => issues.push(issue(
                        handle,
                        transaction.sequence,
                        operation_index,
                        MergeIssueKind::CreateAfterDestroyWithSameGeneration,
                    )),
                    Some(_) => issues.push(issue(
                        handle,
                        transaction.sequence,
                        operation_index,
                        MergeIssueKind::DuplicateCreate,
                    )),
                    None => {
                        pending.insert(handle, Pending::Create(record));
                    }
                },
                SceneOperation::Destroy { .. } => match pending.get(&handle) {
                    Some(Pending::Create(_)) => {
                        pending.insert(handle, Pending::CanceledCreate);
                        canceled_creates += 1;
                    }
                    Some(Pending::CanceledCreate | Pending::Destroy) => {}
                    Some(Pending::Update(_)) | None => {
                        pending.insert(handle, Pending::Destroy);
                    }
                },
                update => match pending.get_mut(&handle) {
                    Some(Pending::Create(record)) => apply_to_record(record, update),
                    Some(Pending::Update(record)) => apply_to_update(record, update),
                    Some(Pending::Destroy | Pending::CanceledCreate) => issues.push(issue(
                        handle,
                        transaction.sequence,
                        operation_index,
                        MergeIssueKind::OperationAfterDestroy,
                    )),
                    None => {
                        let mut record = UpdateRecord::default();
                        apply_to_update(&mut record, update);
                        pending.insert(handle, Pending::Update(record));
                    }
                },
            }
        }
    }

    let mut operations = Vec::new();
    for (handle, operation) in pending {
        match operation {
            Pending::Create(record) => operations.push(SceneOperation::Create { handle, record }),
            Pending::Destroy => operations.push(SceneOperation::Destroy { handle }),
            Pending::Update(update) => append_updates(&mut operations, handle, update),
            Pending::CanceledCreate => {}
        }
    }

    MergeReport {
        transaction: SceneTransaction {
            frame_epoch,
            sequence: transactions
                .iter()
                .map(|transaction| transaction.sequence)
                .max()
                .unwrap_or_default(),
            producer: 0,
            operations,
        },
        issues,
        canceled_creates,
    }
}

fn issue(
    handle: SceneHandle,
    sequence: u64,
    operation_index: usize,
    kind: MergeIssueKind,
) -> MergeIssue {
    MergeIssue {
        handle,
        sequence,
        operation_index: operation_index as u32,
        kind,
    }
}

fn apply_to_record(record: &mut InstanceRecord, operation: SceneOperation) {
    match operation {
        SceneOperation::SetTransform { current, .. } => record.current_transform = current,
        SceneOperation::SetBounds { bounds, .. } => record.bounds = bounds,
        SceneOperation::SetGeometry { geometry, .. } => record.geometry = geometry,
        SceneOperation::SetMaterial { material, .. } => record.material = material,
        SceneOperation::SetFlags { mask, value, .. } => {
            record.flags = (record.flags & !mask) | (value & mask);
        }
        SceneOperation::SetRenderLayers { render_layers, .. } => {
            record.render_layers = render_layers;
        }
        SceneOperation::Create { .. } | SceneOperation::Destroy { .. } => unreachable!(),
    }
}

fn apply_to_update(record: &mut UpdateRecord, operation: SceneOperation) {
    match operation {
        SceneOperation::SetTransform { current, .. } => record.transform = Some(current),
        SceneOperation::SetBounds { bounds, .. } => record.bounds = Some(bounds),
        SceneOperation::SetGeometry { geometry, .. } => record.geometry = Some(geometry),
        SceneOperation::SetMaterial { material, .. } => record.material = Some(material),
        SceneOperation::SetFlags { mask, value, .. } => {
            record.flags = Some(match record.flags {
                Some((old_mask, old_value)) => {
                    let combined_mask = old_mask | mask;
                    let combined_value = (old_value & !mask) | (value & mask);
                    (combined_mask, combined_value)
                }
                None => (mask, value),
            });
        }
        SceneOperation::SetRenderLayers { render_layers, .. } => {
            record.render_layers = Some(render_layers);
        }
        SceneOperation::Create { .. } | SceneOperation::Destroy { .. } => unreachable!(),
    }
}

fn append_updates(output: &mut Vec<SceneOperation>, handle: SceneHandle, update: UpdateRecord) {
    if let Some(current) = update.transform {
        output.push(SceneOperation::SetTransform { handle, current });
    }
    if let Some(bounds) = update.bounds {
        output.push(SceneOperation::SetBounds { handle, bounds });
    }
    if let Some(geometry) = update.geometry {
        output.push(SceneOperation::SetGeometry { handle, geometry });
    }
    if let Some(material) = update.material {
        output.push(SceneOperation::SetMaterial { handle, material });
    }
    if let Some((mask, value)) = update.flags {
        output.push(SceneOperation::SetFlags {
            handle,
            mask,
            value,
        });
    }
    if let Some(render_layers) = update.render_layers {
        output.push(SceneOperation::SetRenderLayers {
            handle,
            render_layers,
        });
    }
}
