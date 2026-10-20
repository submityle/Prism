use crate::model::{
    apply_changes, Change, EndpointDescriptor, Request, RequestId, Response, ResponseEnvelope,
    ResponsePayload, Revision, SyncError, TargetId, WorldSnapshot,
};
use std::collections::BTreeMap;

/// A request retained until its response is accepted.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PendingRequest {
    /// Correlation ID sent to the endpoint.
    pub request_id: RequestId,
    /// Request issued by the client.
    pub request: Request,
}

/// Small client-side request coordinator with strict response correlation.
pub struct SyncClient {
    target: TargetId,
    next_request_id: u64,
    pending: BTreeMap<RequestId, Request>,
    descriptor: Option<EndpointDescriptor>,
    revision: Revision,
}

impl SyncClient {
    /// Creates a client for one synchronization target.
    #[must_use]
    pub fn new(target: TargetId) -> Self {
        Self {
            target,
            next_request_id: 1,
            pending: BTreeMap::new(),
            descriptor: None,
            revision: Revision::ZERO,
        }
    }

    /// Returns the target this client addresses.
    #[must_use]
    pub const fn target(&self) -> TargetId {
        self.target
    }

    /// Returns the last revision accepted by this client.
    #[must_use]
    pub const fn revision(&self) -> Revision {
        self.revision
    }

    /// Returns endpoint metadata after a successful describe response.
    #[must_use]
    pub fn descriptor(&self) -> Option<&EndpointDescriptor> {
        self.descriptor.as_ref()
    }

    /// Issues a request and records it for response correlation.
    pub fn issue(&mut self, request: Request) -> PendingRequest {
        let request_id = RequestId::from_raw(self.next_request_id);
        self.next_request_id = self.next_request_id.saturating_add(1).max(1);
        self.pending.insert(request_id, request.clone());
        PendingRequest {
            request_id,
            request,
        }
    }

    /// Issues a metadata request.
    pub fn describe(&mut self) -> PendingRequest {
        self.issue(Request::Describe)
    }

    /// Issues a snapshot request for the configured target.
    pub fn read_snapshot(&mut self) -> PendingRequest {
        self.issue(Request::ReadSnapshot {
            target: self.target,
        })
    }

    /// Issues an incremental read from the client's current revision.
    pub fn read_changes(&mut self) -> PendingRequest {
        self.issue(Request::ReadChanges {
            target: self.target,
            from_revision: self.revision,
        })
    }

    /// Issues an optimistic write against the client's current revision.
    pub fn apply_changes(&mut self, changes: Vec<Change>) -> PendingRequest {
        self.issue(Request::ApplyChanges {
            target: self.target,
            base_revision: self.revision,
            changes,
        })
    }

    /// Accepts a response, verifies its request ID, and advances known state
    /// only for successful responses concerning this client's target.
    pub fn accept(&mut self, response: ResponseEnvelope) -> Result<ResponsePayload, SyncError> {
        let request =
            self.pending
                .remove(&response.request_id)
                .ok_or_else(|| SyncError::InvalidRequest {
                    code: crate::ErrorCode::InvalidRequest,
                    detail: format!(
                        "response {} has no pending request",
                        response.request_id.raw()
                    ),
                })?;
        match response.response {
            Response::Error(error) => Err(error),
            Response::Success(payload) => {
                self.observe_success(&request, &payload)?;
                Ok(payload)
            }
        }
    }

    fn observe_success(
        &mut self,
        request: &Request,
        payload: &ResponsePayload,
    ) -> Result<(), SyncError> {
        match (request, payload) {
            (Request::Describe, ResponsePayload::Descriptor(descriptor)) => {
                self.descriptor = Some(descriptor.clone());
            }
            (Request::ReadSnapshot { target }, ResponsePayload::Snapshot(snapshot)) => {
                ensure_target(*target, snapshot.target)?;
                self.revision = snapshot.revision;
            }
            (Request::ReadChanges { target: _, .. }, ResponsePayload::Changes(changes)) => {
                if changes.base_revision != self.revision {
                    return Err(SyncError::RevisionGap {
                        expected: self.revision,
                        actual: changes.base_revision,
                    });
                }
                self.revision = changes.target_revision;
            }
            (
                Request::ApplyChanges { target, .. },
                ResponsePayload::Commit {
                    target: committed_target,
                    revision,
                    ..
                },
            ) => {
                ensure_target(*target, *committed_target)?;
                if *revision <= self.revision {
                    return Err(SyncError::RevisionGap {
                        expected: self.revision.next(),
                        actual: *revision,
                    });
                }
                self.revision = *revision;
            }
            (Request::Ping, ResponsePayload::Pong) => {}
            _ => {
                return Err(SyncError::InvalidRequest {
                    code: crate::ErrorCode::InvalidRequest,
                    detail: String::from("response payload does not match request kind"),
                });
            }
        }
        Ok(())
    }
}

fn ensure_target(expected: TargetId, actual: TargetId) -> Result<(), SyncError> {
    if expected == actual {
        Ok(())
    } else {
        Err(SyncError::TargetMismatch { expected, actual })
    }
}

/// A local mirror that applies snapshots and contiguous change sets.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Replica {
    target: TargetId,
    snapshot: Option<WorldSnapshot>,
}

impl Replica {
    /// Creates an empty replica for one target.
    #[must_use]
    pub fn new(target: TargetId) -> Self {
        Self {
            target,
            snapshot: None,
        }
    }

    /// Returns the configured target.
    #[must_use]
    pub const fn target(&self) -> TargetId {
        self.target
    }

    /// Returns the currently installed snapshot, if synchronized.
    #[must_use]
    pub fn snapshot(&self) -> Option<&WorldSnapshot> {
        self.snapshot.as_ref()
    }

    /// Installs a complete snapshot and resets local history to its revision.
    pub fn install_snapshot(&mut self, snapshot: WorldSnapshot) -> Result<(), SyncError> {
        ensure_target(self.target, snapshot.target)?;
        self.snapshot = Some(snapshot);
        Ok(())
    }

    /// Applies a contiguous change set atomically.
    pub fn apply_changes(&mut self, changes: &crate::ChangeSet) -> Result<(), SyncError> {
        let snapshot = self.snapshot.as_mut().ok_or(SyncError::RevisionGap {
            expected: Revision::ZERO,
            actual: changes.base_revision,
        })?;
        if changes.base_revision != snapshot.revision {
            return Err(SyncError::RevisionGap {
                expected: snapshot.revision,
                actual: changes.base_revision,
            });
        }
        if changes.target_revision < changes.base_revision {
            return Err(SyncError::InvalidRequest {
                code: crate::ErrorCode::InvalidRevisionRange,
                detail: String::from("change set target revision precedes base revision"),
            });
        }
        let mut candidate = snapshot.clone();
        apply_changes(&mut candidate, &changes.changes)?;
        candidate.revision = changes.target_revision;
        *snapshot = candidate;
        Ok(())
    }

    /// Applies a response payload relevant to this replica.
    pub fn apply_payload(&mut self, payload: ResponsePayload) -> Result<(), SyncError> {
        match payload {
            ResponsePayload::Snapshot(snapshot) => self.install_snapshot(snapshot),
            ResponsePayload::Changes(changes) => self.apply_changes(&changes),
            ResponsePayload::Commit {
                target,
                revision,
                changes,
            } => {
                ensure_target(self.target, target)?;
                let base_revision = self
                    .snapshot
                    .as_ref()
                    .map_or(Revision::ZERO, |s| s.revision);
                self.apply_changes(&crate::ChangeSet::new(base_revision, revision, changes)?)?;
                Ok(())
            }
            _ => Err(SyncError::InvalidRequest {
                code: crate::ErrorCode::InvalidRequest,
                detail: String::from("payload does not contain replica state"),
            }),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        Capability, Change, ComponentId, Endpoint, EndpointCapabilities, EntityId, EntityRecord,
        InMemoryEndpoint, InMemoryStore, RequestEnvelope, Value,
    };

    fn endpoint() -> InMemoryEndpoint {
        let capabilities = EndpointCapabilities::from_capabilities([
            Capability::SnapshotRead,
            Capability::IncrementalRead,
            Capability::ComponentWrite,
            Capability::RevisionChecks,
        ]);
        let mut store = InMemoryStore::new("replica", capabilities, 4).expect("store");
        let mut snapshot = WorldSnapshot::empty(TargetId::from_raw(1));
        snapshot
            .entities
            .insert(EntityId::from_raw(1), EntityRecord::new());
        store.register_target(snapshot).expect("target");
        crate::SyncEndpoint::new(store)
    }

    #[test]
    fn client_and_replica_follow_snapshot_then_incremental_commit() {
        let mut endpoint = endpoint();
        let mut client = SyncClient::new(TargetId::from_raw(1));
        let mut replica = Replica::new(TargetId::from_raw(1));

        let snapshot_request = client.read_snapshot();
        let snapshot_response = endpoint.handle(RequestEnvelope {
            request_id: snapshot_request.request_id,
            request: snapshot_request.request,
        });
        let snapshot_payload = client.accept(snapshot_response).expect("snapshot");
        replica.apply_payload(snapshot_payload).expect("install");

        let write = client.apply_changes(vec![Change::SetComponent {
            entity: EntityId::from_raw(1),
            component: ComponentId::from_static("Value"),
            value: Value::Unsigned(9),
        }]);
        let response = endpoint.handle(RequestEnvelope {
            request_id: write.request_id,
            request: write.request,
        });
        let payload = client.accept(response).expect("commit");
        replica.apply_payload(payload).expect("apply commit");
        assert_eq!(client.revision(), Revision::from_raw(1));
        assert_eq!(
            replica.snapshot().expect("snapshot").revision,
            Revision::from_raw(1)
        );
    }

    #[test]
    fn client_rejects_unknown_response_ids() {
        let mut client = SyncClient::new(TargetId::from_raw(1));
        let error = client
            .accept(ResponseEnvelope {
                request_id: RequestId::from_raw(99),
                response: Response::Success(ResponsePayload::Pong),
            })
            .expect_err("unknown response");
        assert_eq!(error.code(), crate::ErrorCode::InvalidRequest);
    }
}
