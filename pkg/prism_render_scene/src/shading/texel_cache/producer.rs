//! Transactional CPU bridge from visible shade requests to published pages.
//!
//! This module deliberately accepts explicit [`ShadePageRequest`]s. It does
//! not infer visibility, geometry, or lighting. Its job is to coordinate the
//! deterministic budget planner, the residency state machine, and the
//! double-buffered page store so a future GPU producer cannot leave either
//! state machine stranded after a prepare/submit failure.

use prism_render_architecture::gpu_scene::GpuCompletionValue;

use super::{
    build_shade_plan, ShadeDemandEpochBatch, ShadeDemandRegistry, ShadePageRequest, ShadePageState,
    ShadePageStore, ShadeResidencyError, ShadeResidencyTable, ShadeStoreError, ShadeSubmission,
    ShadeWriteTicket,
};

/// Errors raised while opening or submitting a transactional shade batch.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ShadeWriteProducerError {
    /// Residency state rejected a transition.
    Residency(ShadeResidencyError),
    /// The physical page store rejected a transition.
    Store(ShadeStoreError),
    /// The registry contains no validated geometry/chart provenance.
    MissingProvenance,
    /// The caller attempted to feed one transaction with multiple material
    /// epochs, which the page-table ABI cannot represent safely.
    MixedEpoch,
    /// The same page was observed with conflicting chart provenance.
    ConflictingDuplicate,
}

/// A prepared set of residency/store writes that has not been submitted.
#[derive(Debug)]
pub struct PreparedShadeWrites {
    frame: super::ShadeFrameToken,
    tickets: Vec<ShadeWriteTicket>,
    writes: Vec<super::ShadePageWrite>,
}

/// Submitted writes awaiting one queue completion proof.
#[derive(Debug)]
pub struct SubmittedShadeWrites {
    frame: super::ShadeFrameToken,
    tickets: Vec<ShadeWriteTicket>,
    submissions: Vec<ShadeSubmission>,
}

/// Opens a transactional shade batch from explicit visible-page requests.
pub fn begin_shade_writes(
    residency: &mut ShadeResidencyTable,
    store: &mut ShadePageStore,
    requests: impl IntoIterator<Item = ShadePageRequest>,
    epoch: u64,
    byte_budget: u64,
) -> Result<PreparedShadeWrites, ShadeWriteProducerError> {
    let plan = build_shade_plan(residency, requests, epoch, byte_budget);
    let frame = store
        .begin_frame(epoch)
        .map_err(ShadeWriteProducerError::Store)?;
    let mut tickets = Vec::with_capacity(plan.selected.len());
    let mut writes = Vec::with_capacity(plan.selected.len());
    for item in plan.selected {
        let ticket = match residency.begin_write(item.key, epoch) {
            Ok(ticket) => ticket,
            Err(error) => {
                let _ = store.abort_frame(frame);
                for ticket in tickets {
                    let _ = residency.abort_write(ticket);
                }
                return Err(ShadeWriteProducerError::Residency(error));
            }
        };
        let write = match store.begin_write(frame, item.key, ticket.generation) {
            Ok(write) => write,
            Err(error) => {
                let _ = residency.abort_write(ticket);
                let _ = store.abort_frame(frame);
                for ticket in tickets {
                    let _ = residency.abort_write(ticket);
                }
                return Err(ShadeWriteProducerError::Store(error));
            }
        };
        tickets.push(ticket);
        writes.push(write);
    }
    Ok(PreparedShadeWrites {
        frame,
        tickets,
        writes,
    })
}

/// Consumes the current explicit visibility-derived demand snapshot through
/// the same transactional path as direct request callers. The registry is
/// borrowed immutably, so failed preparation never clears or mutates feedback.
pub fn begin_shade_writes_from_registry(
    residency: &mut ShadeResidencyTable,
    store: &mut ShadePageStore,
    registry: &ShadeDemandRegistry,
    epoch: u64,
    byte_budget: u64,
) -> Result<PreparedShadeWrites, ShadeWriteProducerError> {
    let batches = registry.epoch_batches().map_err(|error| match error {
        super::ShadeDemandBatchError::MissingProvenance => {
            ShadeWriteProducerError::MissingProvenance
        }
        super::ShadeDemandBatchError::ConflictingDuplicate => {
            ShadeWriteProducerError::ConflictingDuplicate
        }
    })?;
    let batch = batches
        .into_iter()
        .find(|batch| batch.epoch == epoch)
        .ok_or(ShadeWriteProducerError::MixedEpoch)?;
    begin_shade_writes_from_epoch_batch(residency, store, &batch, byte_budget)
}

/// Opens a transaction from one already normalized, epoch-homogeneous batch.
///
/// Keeping this function separate from the registry lookup makes the
/// single-epoch invariant explicit at the call site and lets a scheduler hold
/// the remaining epoch batches for later frames without losing demand.
pub fn begin_shade_writes_from_epoch_batch(
    residency: &mut ShadeResidencyTable,
    store: &mut ShadePageStore,
    batch: &ShadeDemandEpochBatch,
    byte_budget: u64,
) -> Result<PreparedShadeWrites, ShadeWriteProducerError> {
    begin_shade_writes(
        residency,
        store,
        batch.records.iter().map(|record| record.request),
        batch.epoch,
        byte_budget,
    )
}

impl PreparedShadeWrites {
    /// Aborts the CPU transaction and restores every requested/stale page.
    pub fn abort(
        self,
        residency: &mut ShadeResidencyTable,
        store: &mut ShadePageStore,
    ) -> Result<(), ShadeWriteProducerError> {
        let store_result = store
            .abort_frame(self.frame)
            .map_err(ShadeWriteProducerError::Store);
        for ticket in self.tickets {
            residency
                .abort_write(ticket)
                .map_err(ShadeWriteProducerError::Residency)?;
        }
        store_result
    }

    /// Records all prepared writes as one GPU submission value.
    pub fn submit(
        self,
        residency: &mut ShadeResidencyTable,
        store: &mut ShadePageStore,
        completion: GpuCompletionValue,
    ) -> Result<SubmittedShadeWrites, ShadeWriteProducerError> {
        let mut submissions = Vec::with_capacity(self.writes.len());
        for write in self.writes.iter().copied() {
            match store.submit_write(write, completion) {
                Ok(submission) => submissions.push(submission),
                Err(error) => {
                    let _ = store.abort_frame(self.frame);
                    for ticket in self.tickets {
                        let _ = residency.abort_write(ticket);
                    }
                    return Err(ShadeWriteProducerError::Store(error));
                }
            }
        }
        Ok(SubmittedShadeWrites {
            frame: self.frame,
            tickets: self.tickets,
            submissions,
        })
    }
}

impl SubmittedShadeWrites {
    /// Applies one observed completion proof and atomically publishes the
    /// staged store generation after every residency write commits.
    pub fn complete_and_publish(
        self,
        residency: &mut ShadeResidencyTable,
        store: &mut ShadePageStore,
        proof: super::ShadeCompletionProof,
        frame: u64,
    ) -> Result<(), ShadeWriteProducerError> {
        for submission in self.submissions {
            store
                .complete_write(submission, proof)
                .map_err(ShadeWriteProducerError::Store)?;
        }
        for ticket in self.tickets {
            residency
                .commit_write(ticket, frame)
                .map_err(ShadeWriteProducerError::Residency)?;
        }
        store
            .publish_after(self.frame, proof)
            .map_err(ShadeWriteProducerError::Store)
    }
}

/// Small helper used by diagnostics and tests to distinguish pages that are
/// still being prepared from pages safe for a snapshot.
#[must_use]
pub fn is_write_pending(state: ShadePageState) -> bool {
    matches!(state, ShadePageState::Writing { .. })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::shading::texel_cache::{
        ShadeAtlasLayout, ShadeChannel, ShadePageKey, ShadeStoreLookup,
    };

    fn key() -> ShadePageKey {
        ShadePageKey {
            object_id: 1,
            material_id: 2,
            material_epoch: 3,
            face: 0,
            mip: 0,
            channel: ShadeChannel::LitDiffuse,
            page_x: 0,
            page_y: 0,
        }
    }

    fn request() -> ShadePageRequest {
        ShadePageRequest {
            key: key(),
            priority: 10,
            byte_cost: 64,
            frame: 1,
        }
    }

    #[test]
    fn abort_restores_residency_and_does_not_publish_store_state() {
        let mut residency = ShadeResidencyTable::new();
        let mut store = ShadePageStore::new(1);
        let prepared = begin_shade_writes(&mut residency, &mut store, [request()], 3, 64)
            .expect("request should open a write");
        assert!(is_write_pending(residency.state(key())));
        prepared.abort(&mut residency, &mut store).unwrap();
        assert_eq!(residency.state(key()), ShadePageState::Requested);
        assert_eq!(store.lookup(key(), 3), ShadeStoreLookup::Fallback);
    }

    #[test]
    fn completion_proof_commits_residency_and_publishes_store() {
        let mut residency = ShadeResidencyTable::new();
        let mut store = ShadePageStore::new(1);
        let prepared = begin_shade_writes(&mut residency, &mut store, [request()], 3, 64)
            .expect("request should open a write");
        let submitted = prepared
            .submit(&mut residency, &mut store, GpuCompletionValue(1))
            .unwrap();
        submitted
            .complete_and_publish(
                &mut residency,
                &mut store,
                super::super::store::test_completion_proof(GpuCompletionValue(1)),
                1,
            )
            .unwrap();
        assert!(matches!(
            residency.state(key()),
            ShadePageState::Ready { .. }
        ));
        assert!(matches!(store.lookup(key(), 3), ShadeStoreLookup::Ready(_)));
        let layout = ShadeAtlasLayout::new([4, 4], [4, 4]).unwrap();
        assert_eq!(store.active_entries().count(), 1);
        assert_eq!(layout.origin(0), Some([0, 0]));
    }

    #[test]
    fn registry_demands_use_the_same_transactional_path() {
        let mut registry = ShadeDemandRegistry::default();
        let record = super::super::ShadeDemandRecord {
            request: request(),
            geometry_index: 1,
            chart: super::super::FaceChart {
                face: 0,
                uv: [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]],
                uv_min: [0.0, 0.0],
                uv_max: [1.0, 1.0],
                texel_extent: [8, 8],
            },
        };
        registry.replace_records([record]);
        let mut residency = ShadeResidencyTable::new();
        let mut store = ShadePageStore::new(1);
        let prepared =
            begin_shade_writes_from_registry(&mut residency, &mut store, &registry, 3, 64)
                .expect("registry demand should open a write");
        assert!(is_write_pending(residency.state(key())));
        prepared.abort(&mut residency, &mut store).unwrap();
    }

    #[test]
    fn mixed_epoch_registry_only_opens_the_requested_epoch_batch() {
        let mut registry = ShadeDemandRegistry::default();
        let first = super::super::ShadeDemandRecord {
            request: request(),
            geometry_index: 1,
            chart: super::super::FaceChart {
                face: 0,
                uv: [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]],
                uv_min: [0.0, 0.0],
                uv_max: [1.0, 1.0],
                texel_extent: [8, 8],
            },
        };
        let mut second = first;
        second.request.key.material_epoch = 4;
        registry.replace_records([first, second]);
        let mut residency = ShadeResidencyTable::new();
        let mut store = ShadePageStore::new(2);
        let prepared =
            begin_shade_writes_from_registry(&mut residency, &mut store, &registry, 3, 128)
                .expect("the requested epoch is selected as its own batch");
        prepared.abort(&mut residency, &mut store).unwrap();
        assert!(matches!(registry.epoch_batches(), Ok(batches) if batches.len() == 2));
    }
}
