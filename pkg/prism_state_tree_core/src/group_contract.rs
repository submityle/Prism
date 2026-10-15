//! Deterministic contracts for group trees and shared blackboard access.
//!
//! This module is host-neutral. Identifiers are opaque scalar values, payloads
//! are represented by `u64`, and every collection has a caller-selected fixed
//! capacity. A host can therefore translate these contracts to ECS relations,
//! replicated commands, or a local scheduler without importing a runtime,
//! clock, or synchronization primitive.

extern crate alloc;

use alloc::vec::Vec;
use core::cmp::Ordering;

/// A fixed frame number used for lease expiry and request ordering.
pub type FrameIndex = u64;

/// Stable identity of a group-tree node.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct GroupNodeId(pub u32);

/// Stable identity of a group member or requesting authority.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct ActorId(pub u32);

/// Stable identity of a shared blackboard key.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct BlackboardKey(pub u32);

/// Semantic alias for a shared-blackboard key.
pub type SharedBlackboardKey = BlackboardKey;

/// Stable identity of a claim lease.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct ClaimId(pub u64);

/// Stable identity of a negotiation request.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct RequestId(pub u64);

/// A group node and its parent relation.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct GroupNode {
    /// Stable node identity.
    pub id: GroupNodeId,
    /// Parent node, or `None` for the sole root.
    pub parent: Option<GroupNodeId>,
    /// Number of parent edges from the root.
    pub depth: u16,
}

/// Failure while constructing a group tree.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum GroupTreeError {
    /// The configured node capacity is zero.
    ZeroCapacity,
    /// The tree already contains its configured number of nodes.
    CapacityExceeded {
        /// Configured maximum number of nodes.
        capacity: usize,
    },
    /// A node with the requested identity already exists.
    DuplicateNode(GroupNodeId),
    /// A non-root node referenced a parent that does not exist.
    MissingParent(GroupNodeId),
    /// A second root was requested.
    MultipleRoots,
    /// Adding the child would exceed the representable depth.
    DepthOverflow(GroupNodeId),
}

/// A bounded, canonical group tree.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct GroupTree {
    capacity: usize,
    nodes: Vec<GroupNode>,
}

impl GroupTree {
    /// Creates an empty tree with a fixed node capacity.
    pub fn new(capacity: usize) -> Result<Self, GroupTreeError> {
        if capacity == 0 {
            return Err(GroupTreeError::ZeroCapacity);
        }
        Ok(Self {
            capacity,
            nodes: Vec::with_capacity(capacity),
        })
    }

    /// Returns the configured node capacity.
    #[must_use]
    pub const fn capacity(&self) -> usize {
        self.capacity
    }

    /// Returns the number of nodes currently stored.
    #[must_use]
    pub fn len(&self) -> usize {
        self.nodes.len()
    }

    /// Returns whether the tree contains no nodes.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.nodes.is_empty()
    }

    /// Adds the sole root node.
    pub fn add_root(&mut self, id: GroupNodeId) -> Result<(), GroupTreeError> {
        self.add(GroupNode {
            id,
            parent: None,
            depth: 0,
        })
    }

    /// Adds a child under an existing parent.
    pub fn add_child(
        &mut self,
        id: GroupNodeId,
        parent: GroupNodeId,
    ) -> Result<(), GroupTreeError> {
        let parent_node = self
            .node(parent)
            .ok_or(GroupTreeError::MissingParent(parent))?;
        let depth = parent_node
            .depth
            .checked_add(1)
            .ok_or(GroupTreeError::DepthOverflow(id))?;
        self.add(GroupNode {
            id,
            parent: Some(parent),
            depth,
        })
    }

    /// Returns nodes in ascending identity order.
    #[must_use]
    pub fn nodes(&self) -> &[GroupNode] {
        &self.nodes
    }

    /// Returns a node by identity.
    #[must_use]
    pub fn node(&self, id: GroupNodeId) -> Option<&GroupNode> {
        self.nodes
            .binary_search_by_key(&id, |node| node.id)
            .ok()
            .map(|index| &self.nodes[index])
    }

    /// Returns whether a node identity is present.
    #[must_use]
    pub fn contains(&self, id: GroupNodeId) -> bool {
        self.node(id).is_some()
    }

    fn add(&mut self, node: GroupNode) -> Result<(), GroupTreeError> {
        if self.node(node.id).is_some() {
            return Err(GroupTreeError::DuplicateNode(node.id));
        }
        if node.parent.is_none() && self.nodes.iter().any(|entry| entry.parent.is_none()) {
            return Err(GroupTreeError::MultipleRoots);
        }
        if self.nodes.len() == self.capacity {
            return Err(GroupTreeError::CapacityExceeded {
                capacity: self.capacity,
            });
        }
        let index = self
            .nodes
            .binary_search_by_key(&node.id, |entry| entry.id)
            .unwrap_err();
        self.nodes.insert(index, node);
        Ok(())
    }
}

/// The kind of access granted by a blackboard contract.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum AccessMode {
    /// Read a key without changing its value.
    Read,
    /// Write a key, subject to the contract's claim and conflict policy.
    Write,
    /// Receive change notifications for a key.
    Subscribe,
}

/// The deterministic policy used when multiple writes target one key in one
/// submitted frame.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum ConcurrentWritePolicy {
    /// Reject every write in a concurrent write set.
    Reject,
    /// Accept the write with the greatest request identity; lower identities
    /// are rejected as superseded.
    LastWriteWins,
    /// Accept all writes in canonical request order and fold them through the
    /// supplied [`DeterministicMerger`].
    Merge,
}

/// A single access grant in a group contract.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct AccessGrant {
    /// Blackboard key covered by the grant.
    pub key: BlackboardKey,
    /// Permission represented by the grant.
    pub mode: AccessMode,
}

/// Failure while building a group blackboard access contract.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum AccessContractError {
    /// The configured grant capacity is zero.
    ZeroCapacity,
    /// The contract already contains its configured number of grants.
    CapacityExceeded {
        /// Configured maximum number of grants.
        capacity: usize,
    },
    /// The exact key and permission were already declared.
    DuplicateGrant(AccessGrant),
}

/// A fixed-capacity read/write/subscribe contract for one group actor.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct GroupAccessContract {
    group: GroupNodeId,
    actor: ActorId,
    policy: ConcurrentWritePolicy,
    capacity: usize,
    grants: Vec<AccessGrant>,
}

impl GroupAccessContract {
    /// Creates an empty contract. A zero grant capacity is valid but cannot
    /// accept grants, which lets configuration errors remain structured.
    #[must_use]
    pub fn new(
        group: GroupNodeId,
        actor: ActorId,
        capacity: usize,
        policy: ConcurrentWritePolicy,
    ) -> Self {
        Self {
            group,
            actor,
            policy,
            capacity,
            grants: Vec::with_capacity(capacity),
        }
    }

    /// Returns the group covered by this contract.
    #[must_use]
    pub const fn group(&self) -> GroupNodeId {
        self.group
    }

    /// Returns the actor covered by this contract.
    #[must_use]
    pub const fn actor(&self) -> ActorId {
        self.actor
    }

    /// Returns the concurrent write policy.
    #[must_use]
    pub const fn write_policy(&self) -> ConcurrentWritePolicy {
        self.policy
    }

    /// Returns grants in canonical `(key, mode)` order.
    #[must_use]
    pub fn grants(&self) -> &[AccessGrant] {
        &self.grants
    }

    /// Adds a read grant.
    pub fn allow_read(&mut self, key: BlackboardKey) -> Result<(), AccessContractError> {
        self.add_grant(AccessGrant {
            key,
            mode: AccessMode::Read,
        })
    }

    /// Adds a write grant. A write grant also authorizes reads.
    pub fn allow_write(&mut self, key: BlackboardKey) -> Result<(), AccessContractError> {
        self.add_grant(AccessGrant {
            key,
            mode: AccessMode::Write,
        })
    }

    /// Adds a subscription grant.
    pub fn allow_subscribe(&mut self, key: BlackboardKey) -> Result<(), AccessContractError> {
        self.add_grant(AccessGrant {
            key,
            mode: AccessMode::Subscribe,
        })
    }

    /// Returns whether the contract grants the requested access.
    #[must_use]
    pub fn allows(&self, key: BlackboardKey, mode: AccessMode) -> bool {
        self.grants
            .binary_search_by_key(&(key, mode), |grant| (grant.key, grant.mode))
            .is_ok()
            || matches!(mode, AccessMode::Read)
                && self
                    .grants
                    .binary_search_by_key(&(key, AccessMode::Write), |grant| {
                        (grant.key, grant.mode)
                    })
                    .is_ok()
    }

    /// Returns whether this contract belongs to the supplied tree.
    #[must_use]
    pub fn is_in_tree(&self, tree: &GroupTree) -> bool {
        tree.contains(self.group)
    }

    fn add_grant(&mut self, grant: AccessGrant) -> Result<(), AccessContractError> {
        let index = match self
            .grants
            .binary_search_by_key(&(grant.key, grant.mode), |entry| (entry.key, entry.mode))
        {
            Ok(_) => return Err(AccessContractError::DuplicateGrant(grant)),
            Err(index) => index,
        };
        if self.capacity == 0 || self.grants.len() == self.capacity {
            return Err(AccessContractError::CapacityExceeded {
                capacity: self.capacity,
            });
        }
        self.grants.insert(index, grant);
        Ok(())
    }
}

/// Whether a claim is shared by readers or exclusive for a writer.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum ClaimMode {
    /// Multiple owners may hold a read claim together.
    SharedRead,
    /// Only one owner may hold the claim.
    ExclusiveWrite,
}

/// A claim lease with an exclusive expiration frame.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct ClaimLease {
    /// Stable claim identity.
    pub id: ClaimId,
    /// Lease owner.
    pub owner: ActorId,
    /// Claimed blackboard key.
    pub key: BlackboardKey,
    /// Shared or exclusive claim mode.
    pub mode: ClaimMode,
    /// Frame at which the lease was acquired or renewed.
    pub acquired_at: FrameIndex,
    /// First frame at which the lease is no longer active.
    pub expires_at: FrameIndex,
}

impl ClaimLease {
    /// Returns whether this lease is active at a fixed frame.
    #[must_use]
    pub const fn is_active_at(self, frame: FrameIndex) -> bool {
        frame < self.expires_at
    }
}

/// The operation carried by a negotiation request.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum NegotiationOperation {
    /// Request a new claim lease.
    Acquire {
        /// Claim identity to create.
        claim: ClaimId,
        /// Shared or exclusive claim mode.
        mode: ClaimMode,
        /// Number of fixed frames for which the lease is active.
        lease_frames: FrameIndex,
    },
    /// Request renewal of an existing claim.
    Renew {
        /// Claim identity to renew.
        claim: ClaimId,
        /// Number of fixed frames for which the renewed lease is active.
        lease_frames: FrameIndex,
    },
    /// Release an existing claim.
    Release {
        /// Claim identity to release.
        claim: ClaimId,
    },
    /// Read the current value of a key.
    Read,
    /// Write a value using the contract's declared conflict policy.
    Write {
        /// New opaque value.
        value: u64,
        /// Policy signed into the request contract.
        policy: ConcurrentWritePolicy,
    },
    /// Subscribe to changes for a key.
    Subscribe,
}

/// A fixed-size request submitted to the deterministic arbitration phase.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct NegotiationRequest {
    /// Globally unique request identity.
    pub id: RequestId,
    /// Requesting actor.
    pub actor: ActorId,
    /// Group under which the request is authorized.
    pub group: GroupNodeId,
    /// Target blackboard key.
    pub key: BlackboardKey,
    /// Frame in which the request was submitted.
    pub submitted_at: FrameIndex,
    /// Requested operation.
    pub operation: NegotiationOperation,
}

impl NegotiationRequest {
    /// Creates a request with explicit fixed-frame ordering metadata.
    #[must_use]
    pub const fn new(
        id: RequestId,
        actor: ActorId,
        group: GroupNodeId,
        key: BlackboardKey,
        submitted_at: FrameIndex,
        operation: NegotiationOperation,
    ) -> Self {
        Self {
            id,
            actor,
            group,
            key,
            submitted_at,
            operation,
        }
    }
}

/// Reason a submitted request was rejected during arbitration.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum NegotiationRejectReason {
    /// The target key does not exist.
    MissingKey,
    /// The request has no matching access grant.
    Unauthorized,
    /// A claim identity already exists or was not found as required.
    ClaimNotFoundOrDuplicate,
    /// Another active lease conflicts with the requested lease.
    ClaimConflict,
    /// The request owner does not own the referenced claim.
    ClaimOwnerMismatch,
    /// The lease duration was zero or overflowed the frame domain.
    InvalidLease,
    /// The signed request policy does not match the contract policy.
    PolicyMismatch,
    /// Another write won under `LastWriteWins`.
    Superseded,
    /// `Reject` found more than one concurrent writer.
    ConcurrentWrite,
    /// A merge policy was submitted without a merge operator.
    MergeUnavailable,
}

/// Result retained for one processed request.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum NegotiationDecisionKind {
    /// The request changed or successfully observed the blackboard state.
    Accepted,
    /// The request was rejected with a deterministic reason.
    Rejected(NegotiationRejectReason),
}

/// A request identity and its arbitration result.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct NegotiationDecision {
    /// Processed request identity.
    pub request: RequestId,
    /// Arbitration result.
    pub kind: NegotiationDecisionKind,
}

/// Counters for one fixed-frame arbitration pass.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct FrameReport {
    /// Frame that was processed.
    pub frame: FrameIndex,
    /// Number of leases released before request processing.
    pub expired_claims: usize,
    /// Number of requests removed from the pending queue.
    pub processed: usize,
    /// Number of accepted requests.
    pub accepted: usize,
    /// Number of rejected requests.
    pub rejected: usize,
}

/// Failure returned by blackboard registration, submission, or arbitration.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum BlackboardError {
    /// A selected fixed capacity is zero.
    ZeroCapacity,
    /// A fixed collection has no remaining slot.
    CapacityExceeded {
        /// Configured maximum for the collection that was full.
        capacity: usize,
    },
    /// A blackboard key was registered twice.
    DuplicateKey(BlackboardKey),
    /// A request identity is already pending.
    DuplicateRequest(RequestId),
    /// The pending request queue is full.
    RequestQueueFull {
        /// Configured maximum number of pending requests.
        capacity: usize,
    },
    /// The request actor/group did not match the supplied contract.
    ContractSubjectMismatch,
    /// The contract refers to a group absent from the supplied tree.
    ContractGroupMissing(GroupNodeId),
    /// The request operation is malformed for its target key or claim.
    InvalidRequest,
    /// A request could not be admitted and was returned as a structured
    /// decision instead of a fatal arbitration failure.
    Rejected(NegotiationRejectReason),
}

/// A deterministic value combiner for [`ConcurrentWritePolicy::Merge`].
pub trait DeterministicMerger {
    /// Folds `right` into `left` for a blackboard key.
    fn merge(&self, key: BlackboardKey, left: u64, right: u64) -> u64;
}

/// A fixed-capacity shared blackboard and its claim/request arbiter.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct SharedBlackboard {
    key_capacity: usize,
    lease_capacity: usize,
    request_capacity: usize,
    slots: Vec<(BlackboardKey, u64)>,
    leases: Vec<ClaimLease>,
    requests: Vec<NegotiationRequest>,
    decisions: Vec<NegotiationDecision>,
}

impl SharedBlackboard {
    /// Creates an empty blackboard with fixed capacities for keys, leases, and
    /// pending requests. Arbitration decisions use the request capacity too.
    #[must_use]
    pub fn new(key_capacity: usize, lease_capacity: usize, request_capacity: usize) -> Self {
        Self {
            key_capacity,
            lease_capacity,
            request_capacity,
            slots: Vec::with_capacity(key_capacity),
            leases: Vec::with_capacity(lease_capacity),
            requests: Vec::with_capacity(request_capacity),
            decisions: Vec::with_capacity(request_capacity),
        }
    }

    /// Returns the current number of registered keys.
    #[must_use]
    pub fn key_len(&self) -> usize {
        self.slots.len()
    }

    /// Returns the current number of active leases.
    #[must_use]
    pub fn lease_len(&self) -> usize {
        self.leases.len()
    }

    /// Returns the number of pending requests.
    #[must_use]
    pub fn pending_len(&self) -> usize {
        self.requests.len()
    }

    /// Registers a key and initial opaque value.
    pub fn register_key(&mut self, key: BlackboardKey, value: u64) -> Result<(), BlackboardError> {
        let index = match self.slots.binary_search_by_key(&key, |entry| entry.0) {
            Ok(_) => return Err(BlackboardError::DuplicateKey(key)),
            Err(index) => index,
        };
        if self.key_capacity == 0 || self.slots.len() == self.key_capacity {
            return Err(BlackboardError::CapacityExceeded {
                capacity: self.key_capacity,
            });
        }
        self.slots.insert(index, (key, value));
        Ok(())
    }

    /// Reads a value when the supplied contract allows it.
    pub fn read(
        &self,
        key: BlackboardKey,
        contract: &GroupAccessContract,
    ) -> Result<u64, BlackboardError> {
        if !contract.allows(key, AccessMode::Read) {
            return Err(BlackboardError::Rejected(
                NegotiationRejectReason::Unauthorized,
            ));
        }
        self.value(key).ok_or(BlackboardError::Rejected(
            NegotiationRejectReason::MissingKey,
        ))
    }

    /// Returns a registered value without checking a contract.
    #[must_use]
    pub fn value(&self, key: BlackboardKey) -> Option<u64> {
        self.slots
            .binary_search_by_key(&key, |entry| entry.0)
            .ok()
            .map(|index| self.slots[index].1)
    }

    /// Returns active leases in canonical claim-identity order.
    #[must_use]
    pub fn leases(&self) -> &[ClaimLease] {
        &self.leases
    }

    /// Returns the decisions from the most recent arbitration pass.
    #[must_use]
    pub fn last_decisions(&self) -> &[NegotiationDecision] {
        &self.decisions
    }

    /// Submits a request after checking its group, actor, key access, and
    /// signed write policy against the supplied contract.
    pub fn submit(
        &mut self,
        request: NegotiationRequest,
        contract: &GroupAccessContract,
        tree: &GroupTree,
    ) -> Result<(), BlackboardError> {
        if request.actor != contract.actor() || request.group != contract.group() {
            return Err(BlackboardError::ContractSubjectMismatch);
        }
        if !tree.contains(request.group) {
            return Err(BlackboardError::ContractGroupMissing(request.group));
        }
        if self.requests.iter().any(|entry| entry.id == request.id) {
            return Err(BlackboardError::DuplicateRequest(request.id));
        }
        let required_mode = match request.operation {
            NegotiationOperation::Read => AccessMode::Read,
            NegotiationOperation::Write { policy, .. } => {
                if policy != contract.write_policy() {
                    return Err(BlackboardError::Rejected(
                        NegotiationRejectReason::PolicyMismatch,
                    ));
                }
                AccessMode::Write
            }
            NegotiationOperation::Subscribe => AccessMode::Subscribe,
            NegotiationOperation::Acquire { mode, .. } => match mode {
                ClaimMode::SharedRead => AccessMode::Read,
                ClaimMode::ExclusiveWrite => AccessMode::Write,
            },
            NegotiationOperation::Renew { .. } | NegotiationOperation::Release { .. } => {
                AccessMode::Write
            }
        };
        if !contract.allows(request.key, required_mode) {
            return Err(BlackboardError::Rejected(
                NegotiationRejectReason::Unauthorized,
            ));
        }
        if self.value(request.key).is_none() {
            return Err(BlackboardError::Rejected(
                NegotiationRejectReason::MissingKey,
            ));
        }
        if self.request_capacity == 0 || self.requests.len() == self.request_capacity {
            return Err(BlackboardError::RequestQueueFull {
                capacity: self.request_capacity,
            });
        }
        let index = self
            .requests
            .binary_search_by(|entry| request_order(entry, &request))
            .unwrap_or_else(|index| index);
        self.requests.insert(index, request);
        Ok(())
    }

    /// Releases every lease whose exclusive expiry frame is at or before
    /// `frame` and returns the number released.
    pub fn release_expired(&mut self, frame: FrameIndex) -> usize {
        let mut released = 0;
        let mut index = 0;
        while index < self.leases.len() {
            if self.leases[index].is_active_at(frame) {
                index += 1;
            } else {
                self.leases.remove(index);
                released += 1;
            }
        }
        released
    }

    /// Processes all requests submitted no later than `frame` using the
    /// built-in arbitration rules. Merge writes are rejected unless processed
    /// through [`SharedBlackboard::process_frame_with_merger`].
    pub fn process_frame(&mut self, frame: FrameIndex) -> FrameReport {
        self.process_frame_inner(frame, None)
    }

    /// Processes all requests submitted no later than `frame`, folding merge
    /// writes with the supplied deterministic operator.
    pub fn process_frame_with_merger(
        &mut self,
        frame: FrameIndex,
        merger: &dyn DeterministicMerger,
    ) -> FrameReport {
        self.process_frame_inner(frame, Some(merger))
    }

    fn process_frame_inner(
        &mut self,
        frame: FrameIndex,
        merger: Option<&dyn DeterministicMerger>,
    ) -> FrameReport {
        let expired_claims = self.release_expired(frame);
        self.decisions.clear();
        let ready = self
            .requests
            .iter()
            .take_while(|request| request.submitted_at <= frame)
            .count();
        let mut accepted = 0;
        let mut rejected = 0;
        for index in 0..ready {
            let request = self.requests[index];
            let kind = self.apply_request(request, ready, frame, merger);
            if matches!(kind, NegotiationDecisionKind::Accepted) {
                accepted += 1;
            } else {
                rejected += 1;
            }
            self.decisions.push(NegotiationDecision {
                request: request.id,
                kind,
            });
        }
        self.requests.drain(..ready);
        FrameReport {
            frame,
            expired_claims,
            processed: ready,
            accepted,
            rejected,
        }
    }

    fn apply_request(
        &mut self,
        request: NegotiationRequest,
        ready: usize,
        frame: FrameIndex,
        merger: Option<&dyn DeterministicMerger>,
    ) -> NegotiationDecisionKind {
        match request.operation {
            NegotiationOperation::Acquire {
                claim,
                mode,
                lease_frames,
            } => self.acquire(request, claim, mode, lease_frames, frame),
            NegotiationOperation::Renew {
                claim,
                lease_frames,
            } => self.renew(request, claim, lease_frames, frame),
            NegotiationOperation::Release { claim } => self.release(request, claim),
            NegotiationOperation::Read | NegotiationOperation::Subscribe => {
                NegotiationDecisionKind::Accepted
            }
            NegotiationOperation::Write { value, policy } => {
                self.write(request, value, policy, ready, frame, merger)
            }
        }
    }

    fn acquire(
        &mut self,
        request: NegotiationRequest,
        claim: ClaimId,
        mode: ClaimMode,
        lease_frames: FrameIndex,
        frame: FrameIndex,
    ) -> NegotiationDecisionKind {
        if lease_frames == 0 {
            return rejected(NegotiationRejectReason::InvalidLease);
        }
        if self.leases.iter().any(|lease| lease.id == claim) {
            return rejected(NegotiationRejectReason::ClaimNotFoundOrDuplicate);
        }
        if self.claim_conflicts(request.key, mode, frame) {
            return rejected(NegotiationRejectReason::ClaimConflict);
        }
        let Some(expires_at) = frame.checked_add(lease_frames) else {
            return rejected(NegotiationRejectReason::InvalidLease);
        };
        if self.lease_capacity == 0 || self.leases.len() == self.lease_capacity {
            return rejected(NegotiationRejectReason::ClaimConflict);
        }
        let lease = ClaimLease {
            id: claim,
            owner: request.actor,
            key: request.key,
            mode,
            acquired_at: frame,
            expires_at,
        };
        let index = self
            .leases
            .binary_search_by_key(&claim, |entry| entry.id)
            .unwrap_or_else(|index| index);
        self.leases.insert(index, lease);
        NegotiationDecisionKind::Accepted
    }

    fn renew(
        &mut self,
        request: NegotiationRequest,
        claim: ClaimId,
        lease_frames: FrameIndex,
        frame: FrameIndex,
    ) -> NegotiationDecisionKind {
        if lease_frames == 0 {
            return rejected(NegotiationRejectReason::InvalidLease);
        }
        let Some(index) = self.leases.iter().position(|lease| lease.id == claim) else {
            return rejected(NegotiationRejectReason::ClaimNotFoundOrDuplicate);
        };
        let lease = self.leases[index];
        if lease.owner != request.actor || lease.key != request.key {
            return rejected(NegotiationRejectReason::ClaimOwnerMismatch);
        }
        if !lease.is_active_at(frame) {
            return rejected(NegotiationRejectReason::ClaimNotFoundOrDuplicate);
        }
        let Some(expires_at) = frame.checked_add(lease_frames) else {
            return rejected(NegotiationRejectReason::InvalidLease);
        };
        self.leases[index] = ClaimLease {
            acquired_at: frame,
            expires_at,
            ..lease
        };
        NegotiationDecisionKind::Accepted
    }

    fn release(&mut self, request: NegotiationRequest, claim: ClaimId) -> NegotiationDecisionKind {
        let Some(index) = self.leases.iter().position(|lease| lease.id == claim) else {
            return rejected(NegotiationRejectReason::ClaimNotFoundOrDuplicate);
        };
        if self.leases[index].owner != request.actor || self.leases[index].key != request.key {
            return rejected(NegotiationRejectReason::ClaimOwnerMismatch);
        }
        self.leases.remove(index);
        NegotiationDecisionKind::Accepted
    }

    fn write(
        &mut self,
        request: NegotiationRequest,
        value: u64,
        policy: ConcurrentWritePolicy,
        ready: usize,
        frame: FrameIndex,
        merger: Option<&dyn DeterministicMerger>,
    ) -> NegotiationDecisionKind {
        if !self.has_exclusive_claim(request.key, request.actor, frame) {
            return rejected(NegotiationRejectReason::ClaimConflict);
        }
        let writers = self.concurrent_writers(request, ready);
        if writers > 1 {
            if self.concurrent_policy(request, ready) != Some(policy) {
                return rejected(NegotiationRejectReason::PolicyMismatch);
            }
            match policy {
                ConcurrentWritePolicy::Reject => {
                    return rejected(NegotiationRejectReason::ConcurrentWrite);
                }
                ConcurrentWritePolicy::LastWriteWins => {
                    let winner = self
                        .requests
                        .iter()
                        .take(ready)
                        .filter(|candidate| {
                            candidate.key == request.key
                                && candidate.submitted_at == request.submitted_at
                                && matches!(
                                    candidate.operation,
                                    NegotiationOperation::Write { policy: candidate_policy, .. }
                                        if candidate_policy == policy
                                )
                        })
                        .map(|candidate| candidate.id)
                        .max();
                    if winner != Some(request.id) {
                        return rejected(NegotiationRejectReason::Superseded);
                    }
                }
                ConcurrentWritePolicy::Merge => {
                    if merger.is_none() {
                        return rejected(NegotiationRejectReason::MergeUnavailable);
                    }
                }
            }
        }
        let Some(slot) = self
            .slots
            .binary_search_by_key(&request.key, |entry| entry.0)
            .ok()
        else {
            return rejected(NegotiationRejectReason::MissingKey);
        };
        self.slots[slot].1 = match policy {
            ConcurrentWritePolicy::Merge if writers > 1 => merger
                .expect("merge operator checked above")
                .merge(request.key, self.slots[slot].1, value),
            _ => value,
        };
        NegotiationDecisionKind::Accepted
    }

    fn concurrent_writers(&self, request: NegotiationRequest, ready: usize) -> usize {
        self.requests
            .iter()
            .take(ready)
            .filter(|candidate| {
                candidate.key == request.key
                    && candidate.submitted_at == request.submitted_at
                    && matches!(candidate.operation, NegotiationOperation::Write { .. })
            })
            .count()
    }

    fn concurrent_policy(
        &self,
        request: NegotiationRequest,
        ready: usize,
    ) -> Option<ConcurrentWritePolicy> {
        self.requests.iter().take(ready).find_map(|candidate| {
            if candidate.key == request.key
                && candidate.submitted_at == request.submitted_at
                && let NegotiationOperation::Write { policy, .. } = candidate.operation
            {
                return Some(policy);
            }
            None
        })
    }

    fn claim_conflicts(&self, key: BlackboardKey, mode: ClaimMode, frame: FrameIndex) -> bool {
        self.leases.iter().any(|lease| {
            lease.key == key
                && lease.is_active_at(frame)
                && (mode == ClaimMode::ExclusiveWrite || lease.mode == ClaimMode::ExclusiveWrite)
        })
    }

    fn has_exclusive_claim(&self, key: BlackboardKey, owner: ActorId, frame: FrameIndex) -> bool {
        self.leases.iter().any(|lease| {
            lease.key == key
                && lease.owner == owner
                && lease.mode == ClaimMode::ExclusiveWrite
                && lease.is_active_at(frame)
        })
    }
}

fn request_order(left: &NegotiationRequest, right: &NegotiationRequest) -> Ordering {
    left.submitted_at
        .cmp(&right.submitted_at)
        .then_with(|| left.key.cmp(&right.key))
        .then_with(|| left.id.cmp(&right.id))
        .then_with(|| left.actor.cmp(&right.actor))
        .then_with(|| left.group.cmp(&right.group))
}

const fn rejected(reason: NegotiationRejectReason) -> NegotiationDecisionKind {
    NegotiationDecisionKind::Rejected(reason)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tree() -> GroupTree {
        let mut tree = GroupTree::new(4).unwrap();
        tree.add_root(GroupNodeId(1)).unwrap();
        tree.add_child(GroupNodeId(3), GroupNodeId(1)).unwrap();
        tree.add_child(GroupNodeId(2), GroupNodeId(1)).unwrap();
        tree
    }

    fn contract(actor: u32, policy: ConcurrentWritePolicy) -> GroupAccessContract {
        let mut contract = GroupAccessContract::new(GroupNodeId(1), ActorId(actor), 4, policy);
        contract.allow_read(BlackboardKey(7)).unwrap();
        contract.allow_write(BlackboardKey(7)).unwrap();
        contract.allow_subscribe(BlackboardKey(7)).unwrap();
        contract
    }

    fn request(
        id: u64,
        actor: u32,
        frame: u64,
        operation: NegotiationOperation,
    ) -> NegotiationRequest {
        NegotiationRequest::new(
            RequestId(id),
            ActorId(actor),
            GroupNodeId(1),
            BlackboardKey(7),
            frame,
            operation,
        )
    }

    #[test]
    fn group_and_contract_are_canonical_and_bounded() {
        let group = tree();
        assert_eq!(
            group.nodes(),
            &[
                GroupNode {
                    id: GroupNodeId(1),
                    parent: None,
                    depth: 0,
                },
                GroupNode {
                    id: GroupNodeId(2),
                    parent: Some(GroupNodeId(1)),
                    depth: 1,
                },
                GroupNode {
                    id: GroupNodeId(3),
                    parent: Some(GroupNodeId(1)),
                    depth: 1,
                },
            ]
        );
        assert_eq!(GroupTree::new(0), Err(GroupTreeError::ZeroCapacity));
        assert_eq!(group.node(GroupNodeId(9)), None,);

        let mut access =
            GroupAccessContract::new(GroupNodeId(1), ActorId(4), 1, ConcurrentWritePolicy::Reject);
        access.allow_write(BlackboardKey(7)).unwrap();
        assert!(access.allows(BlackboardKey(7), AccessMode::Read));
        assert_eq!(
            access.allow_read(BlackboardKey(8)),
            Err(AccessContractError::CapacityExceeded { capacity: 1 })
        );
    }

    #[test]
    fn last_write_wins_is_deterministic_and_uses_request_identity() {
        let group = tree();
        let mut blackboard = SharedBlackboard::new(1, 2, 8);
        blackboard.register_key(BlackboardKey(7), 10).unwrap();
        let writer_a = contract(10, ConcurrentWritePolicy::LastWriteWins);
        let writer_b = contract(11, ConcurrentWritePolicy::LastWriteWins);
        blackboard
            .submit(
                request(
                    2,
                    11,
                    5,
                    NegotiationOperation::Acquire {
                        claim: ClaimId(2),
                        mode: ClaimMode::ExclusiveWrite,
                        lease_frames: 4,
                    },
                ),
                &writer_b,
                &group,
            )
            .unwrap();
        blackboard
            .submit(
                request(
                    1,
                    10,
                    5,
                    NegotiationOperation::Acquire {
                        claim: ClaimId(1),
                        mode: ClaimMode::ExclusiveWrite,
                        lease_frames: 4,
                    },
                ),
                &writer_a,
                &group,
            )
            .unwrap();
        let report = blackboard.process_frame(5);
        assert_eq!(report.accepted, 1);
        assert_eq!(report.rejected, 1);
        assert_eq!(blackboard.lease_len(), 1);

        let winner = blackboard.leases()[0].owner;
        let winner_contract = if winner == ActorId(10) {
            &writer_a
        } else {
            &writer_b
        };
        blackboard
            .submit(
                request(
                    4,
                    winner.0,
                    6,
                    NegotiationOperation::Write {
                        value: 20,
                        policy: ConcurrentWritePolicy::LastWriteWins,
                    },
                ),
                winner_contract,
                &group,
            )
            .unwrap();
        assert_eq!(blackboard.process_frame(6).accepted, 1);
        assert_eq!(blackboard.value(BlackboardKey(7)), Some(20));
    }

    #[test]
    fn reject_policy_rejects_concurrent_writes_and_merge_is_ordered() {
        let group = tree();
        let mut blackboard = SharedBlackboard::new(1, 2, 8);
        blackboard.register_key(BlackboardKey(7), 0).unwrap();
        let access = contract(10, ConcurrentWritePolicy::Reject);
        blackboard
            .submit(
                request(
                    1,
                    10,
                    1,
                    NegotiationOperation::Acquire {
                        claim: ClaimId(1),
                        mode: ClaimMode::ExclusiveWrite,
                        lease_frames: 5,
                    },
                ),
                &access,
                &group,
            )
            .unwrap();
        assert_eq!(blackboard.process_frame(1).accepted, 1);
        blackboard
            .submit(
                request(
                    2,
                    10,
                    2,
                    NegotiationOperation::Write {
                        value: 3,
                        policy: ConcurrentWritePolicy::Reject,
                    },
                ),
                &access,
                &group,
            )
            .unwrap();
        blackboard
            .submit(
                request(
                    3,
                    10,
                    2,
                    NegotiationOperation::Write {
                        value: 4,
                        policy: ConcurrentWritePolicy::Reject,
                    },
                ),
                &access,
                &group,
            )
            .unwrap();
        let report = blackboard.process_frame(2);
        assert_eq!((report.accepted, report.rejected), (0, 2));
        assert_eq!(blackboard.value(BlackboardKey(7)), Some(0));

        struct Add;
        impl DeterministicMerger for Add {
            fn merge(&self, _key: BlackboardKey, left: u64, right: u64) -> u64 {
                left + right
            }
        }
        let mut merge_board = SharedBlackboard::new(1, 1, 8);
        merge_board.register_key(BlackboardKey(7), 1).unwrap();
        let merge_access = contract(10, ConcurrentWritePolicy::Merge);
        merge_board
            .submit(
                request(
                    1,
                    10,
                    1,
                    NegotiationOperation::Acquire {
                        claim: ClaimId(1),
                        mode: ClaimMode::ExclusiveWrite,
                        lease_frames: 5,
                    },
                ),
                &merge_access,
                &group,
            )
            .unwrap();
        merge_board.process_frame(1);
        for (id, value) in [(3, 4), (2, 3)] {
            merge_board
                .submit(
                    request(
                        id,
                        10,
                        2,
                        NegotiationOperation::Write {
                            value,
                            policy: ConcurrentWritePolicy::Merge,
                        },
                    ),
                    &merge_access,
                    &group,
                )
                .unwrap();
        }
        assert_eq!(merge_board.process_frame_with_merger(2, &Add).accepted, 2);
        assert_eq!(merge_board.value(BlackboardKey(7)), Some(8));
    }

    #[test]
    fn leases_expire_on_fixed_frame_boundaries_and_can_be_renewed() {
        let group = tree();
        let mut blackboard = SharedBlackboard::new(1, 1, 4);
        blackboard.register_key(BlackboardKey(7), 0).unwrap();
        let access = contract(10, ConcurrentWritePolicy::Reject);
        blackboard
            .submit(
                request(
                    1,
                    10,
                    4,
                    NegotiationOperation::Acquire {
                        claim: ClaimId(1),
                        mode: ClaimMode::ExclusiveWrite,
                        lease_frames: 2,
                    },
                ),
                &access,
                &group,
            )
            .unwrap();
        blackboard.process_frame(4);
        assert!(blackboard.leases()[0].is_active_at(5));
        assert_eq!(blackboard.process_frame(6).expired_claims, 1);
        assert!(blackboard.leases().is_empty());

        blackboard
            .submit(
                request(
                    2,
                    10,
                    7,
                    NegotiationOperation::Acquire {
                        claim: ClaimId(2),
                        mode: ClaimMode::ExclusiveWrite,
                        lease_frames: 1,
                    },
                ),
                &access,
                &group,
            )
            .unwrap();
        blackboard.process_frame(7);
        blackboard
            .submit(
                request(
                    3,
                    10,
                    7,
                    NegotiationOperation::Renew {
                        claim: ClaimId(2),
                        lease_frames: 3,
                    },
                ),
                &access,
                &group,
            )
            .unwrap();
        blackboard.process_frame(7);
        assert!(blackboard.leases()[0].is_active_at(9));
        assert!(!blackboard.leases()[0].is_active_at(10));
    }

    #[test]
    fn structured_submission_errors_preserve_fixed_capacity() {
        let mut tree = GroupTree::new(1).unwrap();
        tree.add_root(GroupNodeId(1)).unwrap();
        let mut blackboard = SharedBlackboard::new(0, 0, 1);
        let mut access = GroupAccessContract::new(
            GroupNodeId(1),
            ActorId(10),
            1,
            ConcurrentWritePolicy::Reject,
        );
        access.allow_read(BlackboardKey(7)).unwrap();
        let request = request(1, 10, 0, NegotiationOperation::Read);
        assert_eq!(
            blackboard.submit(request, &access, &tree),
            Err(BlackboardError::Rejected(
                NegotiationRejectReason::MissingKey
            ))
        );
        assert_eq!(blackboard.pending_len(), 0);
        assert_eq!(
            blackboard.register_key(BlackboardKey(7), 0),
            Err(BlackboardError::CapacityExceeded { capacity: 0 })
        );
    }
}
