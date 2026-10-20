use crate::model::{
    apply_changes, Capability, Change, EndpointCapabilities, EndpointDescriptor, Request,
    RequestEnvelope, Response, ResponseEnvelope, ResponsePayload, Revision, SyncError, TargetId,
    WorldSnapshot,
};
use std::collections::BTreeMap;

/// Transport-independent endpoint implementation boundary.
pub trait Endpoint {
    /// Returns endpoint metadata and advertised behavior.
    fn descriptor(&self) -> &EndpointDescriptor;

    /// Handles one request and echoes its request ID in the response.
    fn handle(&mut self, request: RequestEnvelope) -> ResponseEnvelope;
}

/// Storage contract used by [`SyncEndpoint`].
pub trait SyncStore {
    /// Returns endpoint metadata.
    fn descriptor(&self) -> &EndpointDescriptor;

    /// Reads a complete snapshot.
    fn read_snapshot(&self, target: TargetId) -> Result<WorldSnapshot, SyncError>;

    /// Reads retained changes strictly after `from_revision`.
    fn read_changes(
        &self,
        target: TargetId,
        from_revision: Revision,
    ) -> Result<crate::ChangeSet, SyncError>;

    /// Applies changes against an exact current revision and returns the new
    /// revision plus the accepted ordered changes.
    fn apply_changes(
        &mut self,
        target: TargetId,
        base_revision: Revision,
        changes: &[Change],
    ) -> Result<CommitResult, SyncError>;
}

/// The result of a successful store commit.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CommitResult {
    /// Target that changed.
    pub target: TargetId,
    /// New target revision.
    pub revision: Revision,
    /// Ordered changes that were committed.
    pub changes: Vec<Change>,
}

/// A complete endpoint that applies capability checks before delegating to a
/// [`SyncStore`].
pub struct SyncEndpoint<S> {
    store: S,
}

impl<S> SyncEndpoint<S> {
    /// Wraps a store in the protocol endpoint.
    #[must_use]
    pub fn new(store: S) -> Self {
        Self { store }
    }

    /// Returns the wrapped store.
    #[must_use]
    pub fn store(&self) -> &S {
        &self.store
    }

    /// Returns the wrapped store mutably.
    #[must_use]
    pub fn store_mut(&mut self) -> &mut S {
        &mut self.store
    }
}

impl<S: SyncStore> Endpoint for SyncEndpoint<S> {
    fn descriptor(&self) -> &EndpointDescriptor {
        self.store.descriptor()
    }

    fn handle(&mut self, envelope: RequestEnvelope) -> ResponseEnvelope {
        let request_id = envelope.request_id;
        let response = match envelope.request {
            Request::Describe => {
                Response::Success(ResponsePayload::Descriptor(self.store.descriptor().clone()))
            }
            Request::Ping => Response::Success(ResponsePayload::Pong),
            Request::ReadSnapshot { target } => self
                .require(Capability::SnapshotRead)
                .and_then(|()| self.store.read_snapshot(target))
                .map(ResponsePayload::Snapshot)
                .map_or_else(Response::Error, Response::Success),
            Request::ReadChanges {
                target,
                from_revision,
            } => self
                .require(Capability::IncrementalRead)
                .and_then(|()| self.store.read_changes(target, from_revision))
                .map(ResponsePayload::Changes)
                .map_or_else(Response::Error, Response::Success),
            Request::ApplyChanges {
                target,
                base_revision,
                changes,
            } => self.apply(target, base_revision, &changes),
        };
        ResponseEnvelope {
            request_id,
            response,
        }
    }
}

impl<S: SyncStore> SyncEndpoint<S> {
    fn require(&self, capability: Capability) -> Result<(), SyncError> {
        if self.descriptor().capabilities.contains(capability) {
            Ok(())
        } else {
            Err(SyncError::UnsupportedCapability { capability })
        }
    }

    fn apply(&mut self, target: TargetId, base_revision: Revision, changes: &[Change]) -> Response {
        let capability = if changes.iter().any(Change::is_property_change) {
            Capability::PropertyWrite
        } else {
            Capability::ComponentWrite
        };
        let result = self
            .require(Capability::RevisionChecks)
            .and_then(|()| self.require(capability))
            .and_then(|()| {
                if changes.is_empty() {
                    return Err(SyncError::InvalidRequest {
                        code: crate::ErrorCode::InvalidRequest,
                        detail: String::from("an apply request must contain at least one change"),
                    });
                }
                self.store.apply_changes(target, base_revision, changes)
            })
            .map(|commit| ResponsePayload::Commit {
                target: commit.target,
                revision: commit.revision,
                changes: commit.changes,
            });
        result.map_or_else(Response::Error, Response::Success)
    }
}

/// In-memory store used by tests, headless tools, and adapter bring-up.
pub struct InMemoryStore {
    descriptor: EndpointDescriptor,
    targets: BTreeMap<TargetId, TargetState>,
    history_limit: usize,
}

struct TargetState {
    snapshot: WorldSnapshot,
    history: Vec<crate::ChangeSet>,
}

impl InMemoryStore {
    /// Creates a store with a fixed history retention count.
    pub fn new(
        name: impl Into<String>,
        capabilities: EndpointCapabilities,
        history_limit: usize,
    ) -> Result<Self, SyncError> {
        if history_limit == 0 {
            return Err(SyncError::InvalidRequest {
                code: crate::ErrorCode::InvalidRequest,
                detail: String::from("history_limit must be greater than zero"),
            });
        }
        Ok(Self {
            descriptor: EndpointDescriptor::new(name, capabilities),
            targets: BTreeMap::new(),
            history_limit,
        })
    }

    /// Registers a target snapshot. Registration is rejected for duplicate IDs.
    pub fn register_target(&mut self, snapshot: WorldSnapshot) -> Result<(), SyncError> {
        if self.targets.contains_key(&snapshot.target) {
            return Err(SyncError::InvalidRequest {
                code: crate::ErrorCode::InvalidRequest,
                detail: format!("target {} is already registered", snapshot.target.raw()),
            });
        }
        self.descriptor.add_target(snapshot.target);
        self.targets.insert(
            snapshot.target,
            TargetState {
                snapshot,
                history: Vec::new(),
            },
        );
        Ok(())
    }

    /// Returns the configured retention count.
    #[must_use]
    pub const fn history_limit(&self) -> usize {
        self.history_limit
    }
}

impl SyncStore for InMemoryStore {
    fn descriptor(&self) -> &EndpointDescriptor {
        &self.descriptor
    }

    fn read_snapshot(&self, target: TargetId) -> Result<WorldSnapshot, SyncError> {
        self.targets
            .get(&target)
            .map(|state| state.snapshot.clone())
            .ok_or(SyncError::UnknownTarget { target })
    }

    fn read_changes(
        &self,
        target: TargetId,
        from_revision: Revision,
    ) -> Result<crate::ChangeSet, SyncError> {
        let state = self
            .targets
            .get(&target)
            .ok_or(SyncError::UnknownTarget { target })?;
        let current = state.snapshot.revision;
        if from_revision > current {
            return Err(SyncError::RevisionConflict {
                target,
                expected: current,
                actual: from_revision,
            });
        }
        if from_revision == current {
            return Ok(crate::ChangeSet::empty(current));
        }
        let oldest = state
            .history
            .first()
            .map_or(current, |change_set| change_set.base_revision);
        if from_revision < oldest {
            return Err(SyncError::RevisionUnavailable {
                target,
                requested: from_revision,
                oldest_available: oldest,
            });
        }
        let mut changes = Vec::new();
        let mut revision = from_revision;
        for change_set in &state.history {
            if change_set.target_revision <= from_revision {
                continue;
            }
            if change_set.base_revision != revision {
                return Err(SyncError::RevisionUnavailable {
                    target,
                    requested: from_revision,
                    oldest_available: oldest,
                });
            }
            revision = change_set.target_revision;
            changes.extend(change_set.changes.iter().cloned());
        }
        crate::ChangeSet::new(from_revision, revision, changes)
    }

    fn apply_changes(
        &mut self,
        target: TargetId,
        base_revision: Revision,
        changes: &[Change],
    ) -> Result<CommitResult, SyncError> {
        let state = self
            .targets
            .get_mut(&target)
            .ok_or(SyncError::UnknownTarget { target })?;
        let actual = state.snapshot.revision;
        if base_revision != actual {
            return Err(SyncError::RevisionConflict {
                target,
                expected: actual,
                actual: base_revision,
            });
        }
        if changes.is_empty() {
            return Err(SyncError::InvalidRequest {
                code: crate::ErrorCode::InvalidRequest,
                detail: String::from("an apply request must contain at least one change"),
            });
        }
        let mut candidate = state.snapshot.clone();
        apply_changes(&mut candidate, changes)?;
        let revision = actual.next();
        candidate.revision = revision;
        let change_set = crate::ChangeSet::new(actual, revision, changes.to_vec())?;
        state.snapshot = candidate;
        state.history.push(change_set);
        if state.history.len() > self.history_limit {
            let excess = state.history.len() - self.history_limit;
            state.history.drain(..excess);
        }
        Ok(CommitResult {
            target,
            revision,
            changes: changes.to_vec(),
        })
    }
}

/// Convenient endpoint alias for the reference in-memory implementation.
pub type InMemoryEndpoint = SyncEndpoint<InMemoryStore>;

trait ChangeKind {
    fn is_property_change(&self) -> bool;
}

impl ChangeKind for Change {
    fn is_property_change(&self) -> bool {
        matches!(
            self,
            Change::SetProperty { .. } | Change::RemoveProperty { .. }
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        Change, ComponentId, Endpoint, EntityId, EntityRecord, RequestId, Value, WorldSnapshot,
    };

    fn endpoint() -> InMemoryEndpoint {
        let capabilities = EndpointCapabilities::from_capabilities([
            Capability::SnapshotRead,
            Capability::IncrementalRead,
            Capability::ComponentWrite,
            Capability::PropertyWrite,
            Capability::RevisionChecks,
            Capability::DeterministicCodec,
        ]);
        let mut store = InMemoryStore::new("test", capabilities, 8).expect("store");
        let mut snapshot = WorldSnapshot::empty(TargetId::from_raw(7));
        snapshot
            .entities
            .insert(EntityId::from_raw(1), EntityRecord::new());
        store.register_target(snapshot).expect("target");
        SyncEndpoint::new(store)
    }

    #[test]
    fn endpoint_echoes_request_id_and_detects_stale_writes() {
        let mut endpoint = endpoint();
        let component = ComponentId::from_static("Position");
        let request = RequestEnvelope {
            request_id: RequestId::from_raw(10),
            request: Request::ApplyChanges {
                target: TargetId::from_raw(7),
                base_revision: Revision::ZERO,
                changes: vec![Change::SetComponent {
                    entity: EntityId::from_raw(1),
                    component: component.clone(),
                    value: Value::Signed(3),
                }],
            },
        };
        let response = endpoint.handle(request);
        assert_eq!(response.request_id.raw(), 10);
        assert!(matches!(response.response, Response::Success(_)));

        let stale = endpoint.handle(RequestEnvelope {
            request_id: RequestId::from_raw(11),
            request: Request::ApplyChanges {
                target: TargetId::from_raw(7),
                base_revision: Revision::ZERO,
                changes: vec![Change::SetComponent {
                    entity: EntityId::from_raw(1),
                    component,
                    value: Value::Signed(4),
                }],
            },
        });
        assert!(matches!(
            stale.response,
            Response::Error(SyncError::RevisionConflict { .. })
        ));
    }

    #[test]
    fn history_retention_reports_unavailable_revision() {
        let capabilities = EndpointCapabilities::from_capabilities([
            Capability::SnapshotRead,
            Capability::IncrementalRead,
            Capability::ComponentWrite,
            Capability::RevisionChecks,
        ]);
        let mut store = InMemoryStore::new("test", capabilities, 2).expect("store");
        let mut snapshot = WorldSnapshot::empty(TargetId::from_raw(7));
        snapshot
            .entities
            .insert(EntityId::from_raw(1), EntityRecord::new());
        store.register_target(snapshot).expect("target");
        let mut endpoint = SyncEndpoint::new(store);
        for value in 0..3 {
            let response = endpoint.handle(RequestEnvelope {
                request_id: RequestId::from_raw(value + 1),
                request: Request::ApplyChanges {
                    target: TargetId::from_raw(7),
                    base_revision: Revision::from_raw(value),
                    changes: vec![Change::SetComponent {
                        entity: EntityId::from_raw(1),
                        component: ComponentId::from_static("Value"),
                        value: Value::Signed(value as i64),
                    }],
                },
            });
            assert!(matches!(response.response, Response::Success(_)));
        }
        let response = endpoint.handle(RequestEnvelope {
            request_id: RequestId::from_raw(20),
            request: Request::ReadChanges {
                target: TargetId::from_raw(7),
                from_revision: Revision::ZERO,
            },
        });
        assert!(matches!(
            response.response,
            Response::Error(SyncError::RevisionUnavailable { .. })
        ));
    }
}
