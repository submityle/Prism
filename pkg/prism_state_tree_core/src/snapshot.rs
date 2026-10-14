//! Pure-data snapshots and atomic restoration for a running [`Instance`].
//!
//! The snapshot deliberately contains no plan, callback, or guard object. A
//! plan is immutable for the lifetime of an instance and guards are runtime
//! services; restoring a snapshot therefore keeps both installed guards and
//! the plan already owned by the instance. The byte format implemented here is
//! small, versioned, and independent of `std` or a serialization crate.

use alloc::{collections::VecDeque, vec, vec::Vec};

use crate::{
    model::{HistoryMode, NodeKind, Plan, StateId},
    runtime::{Event, Instance},
};

const MAGIC: [u8; 4] = *b"PTS1";
const LEGACY_VERSION: u8 = 1;
const VERSION: u8 = 2;

/// A pure-data representation of an [`Instance`]'s semantic state.
///
/// `active` is in deterministic parent-before-child entry order. `history`
/// has one slot per state in the associated plan; a slot is `Some` only for a
/// compound or parallel state that has remembered history. The queue is in
/// FIFO order. Guards, the immutable plan, the lifecycle log, and the current
/// transient event are intentionally not part of a snapshot.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Snapshot {
    /// Semantic identity of the plan that produced this snapshot.
    ///
    /// `Some` is emitted by [`Instance::snapshot`] and is encoded by the v2
    /// wire format. `None` represents a legacy v1 snapshot; it remains
    /// readable for save compatibility, but can only receive the structural
    /// validation available before plan identity was recorded.
    pub plan_fingerprint: Option<u128>,
    /// Whether the instance has entered its root and is eligible to run.
    pub started: bool,
    /// Active state configuration in deterministic entry order.
    pub active: Vec<StateId>,
    /// Per-state remembered history, indexed by `StateId`.
    pub history: Vec<Option<Vec<StateId>>>,
    /// Pending events in FIFO order.
    pub queued_events: Vec<Event>,
}

impl Snapshot {
    /// Encodes this snapshot into the crate's versioned, deterministic format.
    pub fn encode(&self) -> Result<Vec<u8>, SnapshotCodecError> {
        let mut output = Vec::new();
        output.extend_from_slice(&MAGIC);
        output.push(if self.plan_fingerprint.is_some() {
            VERSION
        } else {
            LEGACY_VERSION
        });
        if let Some(fingerprint) = self.plan_fingerprint {
            output.extend_from_slice(&fingerprint.to_le_bytes());
        }
        output.push(u8::from(self.started));
        write_len(&mut output, self.active.len(), "active")?;
        for state in &self.active {
            write_u32(&mut output, state.0);
        }
        write_len(&mut output, self.history.len(), "history")?;
        for entry in &self.history {
            match entry {
                None => output.push(0),
                Some(states) => {
                    output.push(1);
                    write_len(&mut output, states.len(), "history entry")?;
                    for state in states {
                        write_u32(&mut output, state.0);
                    }
                }
            }
        }
        write_len(&mut output, self.queued_events.len(), "queued events")?;
        for event in &self.queued_events {
            write_u32(&mut output, event.id.0);
        }
        Ok(output)
    }

    /// Decodes a snapshot produced by [`Snapshot::encode`].
    pub fn decode(bytes: &[u8]) -> Result<Self, SnapshotCodecError> {
        let mut reader = Reader::new(bytes);
        if reader.take(4)? != MAGIC {
            return Err(SnapshotCodecError::InvalidHeader);
        }
        let version = reader.byte()?;
        let plan_fingerprint = match version {
            LEGACY_VERSION => None,
            VERSION => Some(reader.u128()?),
            _ => return Err(SnapshotCodecError::UnsupportedVersion),
        };
        let started = match reader.byte()? {
            0 => false,
            1 => true,
            _ => return Err(SnapshotCodecError::InvalidBoolean),
        };

        let active_len = reader.count("active", 4)?;
        let mut active = Vec::with_capacity(active_len);
        for _ in 0..active_len {
            active.push(StateId(reader.u32()?));
        }

        let history_len = reader.count("history", 1)?;
        let mut history = Vec::with_capacity(history_len);
        for _ in 0..history_len {
            match reader.byte()? {
                0 => history.push(None),
                1 => {
                    let entry_len = reader.count("history entry", 4)?;
                    let mut entry = Vec::with_capacity(entry_len);
                    for _ in 0..entry_len {
                        entry.push(StateId(reader.u32()?));
                    }
                    history.push(Some(entry));
                }
                _ => return Err(SnapshotCodecError::InvalidHistoryMarker),
            }
        }

        let queue_len = reader.count("queued events", 4)?;
        let mut queued_events = Vec::with_capacity(queue_len);
        for _ in 0..queue_len {
            queued_events.push(Event::new(crate::EventId(reader.u32()?)));
        }
        if !reader.is_empty() {
            return Err(SnapshotCodecError::TrailingBytes);
        }
        Ok(Self {
            plan_fingerprint,
            started,
            active,
            history,
            queued_events,
        })
    }
}

/// Failure while validating a snapshot against an instance's plan.
///
/// Validation is completed before the instance is mutated. Consequently an
/// error from [`Instance::restore`] leaves every observable runtime field
/// unchanged, including the queue and installed guards. This atomicity covers
/// all validation errors; allocation failure is a process-level Rust abort and
/// is outside the `Result` contract of this `no_std + alloc` crate.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum RestoreError {
    /// The snapshot was produced by a different semantic plan revision.
    PlanFingerprintMismatch {
        /// Fingerprint of the plan owned by the target instance.
        expected: u128,
        /// Fingerprint recorded in the snapshot.
        actual: u128,
    },
    /// A state identity is outside the instance's plan.
    InvalidState(StateId),
    /// The history vector is not indexed one-for-one with the plan.
    HistoryLength {
        /// Number of states in the plan.
        expected: usize,
        /// Number of history slots supplied by the snapshot.
        actual: usize,
    },
    /// The snapshot contains more events than this instance can hold.
    QueueCapacityExceeded {
        /// Capacity of the target instance.
        capacity: usize,
        /// Number of queued events in the snapshot.
        queued: usize,
    },
    /// A not-started instance cannot carry an active configuration.
    UnstartedHasActiveConfiguration,
    /// A started instance must carry at least its root state.
    StartedHasNoActiveConfiguration,
    /// The root is absent from a started active configuration.
    ActiveRootMissing,
    /// The root is not the first state in the active entry order.
    ActiveRootNotFirst,
    /// A state occurs more than once in the active configuration.
    DuplicateActiveState(StateId),
    /// A child is listed without its parent being active.
    ActiveParentMissing {
        /// Child whose parent is absent.
        state: StateId,
        /// Required parent.
        parent: StateId,
    },
    /// A parent appears after its child, violating deterministic entry order.
    ActiveParentAfterChild {
        /// Child that appeared first.
        state: StateId,
        /// Parent that appeared later.
        parent: StateId,
    },
    /// The active direct children do not match the node's compound/parallel
    /// semantics.
    InvalidActiveChildren(StateId),
    /// History was supplied for a node that does not support history.
    HistoryOwnerWithoutHistory(StateId),
    /// A history entry contains the same state more than once.
    DuplicateHistoryState {
        /// State whose history slot is invalid.
        owner: StateId,
        /// Repeated state.
        state: StateId,
    },
    /// A history state is not below its history owner.
    HistoryStateOutsideOwner {
        /// History owner.
        owner: StateId,
        /// Invalid remembered state.
        state: StateId,
    },
    /// A remembered state is missing its remembered parent.
    HistoryParentMissing {
        /// History owner.
        owner: StateId,
        /// Child in the remembered configuration.
        state: StateId,
        /// Required parent.
        parent: StateId,
    },
    /// A remembered configuration has an invalid child shape.
    InvalidHistoryChildren {
        /// History owner whose slot is being checked.
        owner: StateId,
        /// Remembered state with invalid children, or the owner itself.
        state: StateId,
    },
    /// A present history slot cannot be empty because restoration would leave
    /// its owner without a valid child configuration.
    EmptyHistory(StateId),
}

/// Failure while encoding or decoding the portable snapshot format.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum SnapshotCodecError {
    /// A vector length cannot be represented by the format's `u32` length.
    LengthOverflow(&'static str),
    /// The four-byte format marker did not match.
    InvalidHeader,
    /// The byte format version is not supported.
    UnsupportedVersion,
    /// The input ended before a complete field was read.
    Truncated,
    /// A boolean field contained a value other than zero or one.
    InvalidBoolean,
    /// A history presence marker contained a value other than zero or one.
    InvalidHistoryMarker,
    /// Bytes remained after the complete snapshot was decoded.
    TrailingBytes,
    /// A declared vector length is not representable on this target.
    LengthNotRepresentable,
}

impl Instance {
    /// Captures semantic state without cloning the immutable plan or guards.
    pub fn snapshot(&self) -> Snapshot {
        Snapshot {
            plan_fingerprint: Some(plan_fingerprint(&self.plan)),
            started: self.started,
            active: self.active.clone(),
            history: self.history.clone(),
            queued_events: self.queue.iter().copied().collect(),
        }
    }

    /// Restores a snapshot atomically against this instance's existing plan.
    ///
    /// The plan, installed guards, event capacity, microstep limit, and
    /// lifecycle log allocation remain owned by the instance. A successful
    /// restore clears the transient log and current event because neither is
    /// part of the portable semantic snapshot. No lifecycle callbacks run.
    pub fn restore(&mut self, snapshot: &Snapshot) -> Result<(), RestoreError> {
        validate_snapshot(self, snapshot)?;

        let active = snapshot.active.clone();
        let history = snapshot.history.clone();
        let queue = VecDeque::from(snapshot.queued_events.clone());

        self.active = active;
        self.history = history;
        self.queue = queue;
        self.started = snapshot.started;
        self.current_event = None;
        self.log.clear();
        Ok(())
    }

    /// Alias for [`Instance::restore`] that makes snapshot intent explicit at
    /// call sites which have several restoration paths.
    pub fn restore_snapshot(&mut self, snapshot: &Snapshot) -> Result<(), RestoreError> {
        self.restore(snapshot)
    }
}

fn validate_snapshot(instance: &Instance, snapshot: &Snapshot) -> Result<(), RestoreError> {
    let plan = &instance.plan;
    let expected_fingerprint = plan_fingerprint(plan);
    if let Some(actual) = snapshot.plan_fingerprint
        && actual != expected_fingerprint
    {
        return Err(RestoreError::PlanFingerprintMismatch {
            expected: expected_fingerprint,
            actual,
        });
    }
    let count = plan.state_count();
    if snapshot.history.len() != count {
        return Err(RestoreError::HistoryLength {
            expected: count,
            actual: snapshot.history.len(),
        });
    }
    if snapshot.queued_events.len() > instance.event_capacity {
        return Err(RestoreError::QueueCapacityExceeded {
            capacity: instance.event_capacity,
            queued: snapshot.queued_events.len(),
        });
    }

    let mut active_seen = vec![false; count];
    let mut active_positions = vec![usize::MAX; count];
    for (position, &state) in snapshot.active.iter().enumerate() {
        let index = state.0 as usize;
        if index >= count {
            return Err(RestoreError::InvalidState(state));
        }
        if active_seen[index] {
            return Err(RestoreError::DuplicateActiveState(state));
        }
        active_seen[index] = true;
        active_positions[index] = position;
    }

    if snapshot.started {
        if snapshot.active.is_empty() {
            return Err(RestoreError::StartedHasNoActiveConfiguration);
        }
        if !active_seen[plan.root().0 as usize] {
            return Err(RestoreError::ActiveRootMissing);
        }
        if active_positions[plan.root().0 as usize] != 0 {
            return Err(RestoreError::ActiveRootNotFirst);
        }
    } else if !snapshot.active.is_empty() {
        return Err(RestoreError::UnstartedHasActiveConfiguration);
    }

    for (position, &state) in snapshot.active.iter().enumerate() {
        let Some(parent) = plan.parent(state) else {
            continue;
        };
        if !active_seen[parent.0 as usize] {
            return Err(RestoreError::ActiveParentMissing { state, parent });
        }
        if active_positions[parent.0 as usize] > position {
            return Err(RestoreError::ActiveParentAfterChild { state, parent });
        }
    }

    for &state in &snapshot.active {
        if !validate_children(plan, state, &active_seen) {
            return Err(RestoreError::InvalidActiveChildren(state));
        }
    }

    for (index, entry) in snapshot.history.iter().enumerate() {
        let Some(states) = entry else {
            continue;
        };
        let owner = StateId(index as u32);
        let Some(mode) = plan.node(owner).and_then(|node| node.kind.history()) else {
            return Err(RestoreError::HistoryOwnerWithoutHistory(owner));
        };
        validate_history(plan, owner, mode, states)?;
    }
    Ok(())
}

/// Computes a stable semantic identity for a compiled plan.
///
/// The fingerprint includes every plan field consulted by the runtime,
/// including child/region order and transition declaration order. It is
/// intentionally independent of allocation addresses and the `Arc` wrapper,
/// so independently compiled equivalent plans accept one another's snapshots.
fn plan_fingerprint(plan: &Plan) -> u128 {
    let mut hash = 0x6c62272e07bb014262b821756295c58du128;
    hash_u128(&mut hash, plan.root.0 as u128);
    hash_u128(&mut hash, plan.nodes.len() as u128);
    for node in &plan.nodes {
        hash_u128(&mut hash, node.id.0 as u128);
        hash_option_state(&mut hash, node.parent);
        match &node.kind {
            NodeKind::Atomic => hash_u8(&mut hash, 0),
            NodeKind::Compound {
                children,
                initial,
                history,
            } => {
                hash_u8(&mut hash, 1);
                hash_states(&mut hash, children);
                hash_u128(&mut hash, initial.0 as u128);
                hash_history_mode(&mut hash, *history);
            }
            NodeKind::Parallel {
                children,
                regions,
                history,
            } => {
                hash_u8(&mut hash, 2);
                hash_states(&mut hash, children);
                hash_states(&mut hash, regions);
                hash_history_mode(&mut hash, *history);
            }
        }
        hash_u128(&mut hash, node.transitions.len() as u128);
        for transition in &node.transitions {
            hash_u128(&mut hash, transition.source.0 as u128);
            match transition.target {
                crate::TransitionTarget::State(state) => {
                    hash_u8(&mut hash, 0);
                    hash_u128(&mut hash, state.0 as u128);
                }
                crate::TransitionTarget::History { owner, depth } => {
                    hash_u8(&mut hash, 1);
                    hash_u128(&mut hash, owner.0 as u128);
                    hash_history_mode(&mut hash, Some(depth));
                }
                crate::TransitionTarget::SelfState => hash_u8(&mut hash, 2),
            }
            match transition.trigger {
                crate::Trigger::Event(event) => {
                    hash_u8(&mut hash, 0);
                    hash_u128(&mut hash, event.0 as u128);
                }
                crate::Trigger::Eventless => hash_u8(&mut hash, 1),
            }
            hash_option_u32(&mut hash, transition.guard.map(|guard| guard.0));
            hash_u128(&mut hash, transition.priority as u128);
            hash_u8(
                &mut hash,
                match transition.kind {
                    crate::TransitionKind::External => 0,
                    crate::TransitionKind::Internal => 1,
                },
            );
            hash_u128(&mut hash, transition.declaration_order as u128);
        }
    }
    hash
}

fn hash_states(hash: &mut u128, states: &[StateId]) {
    hash_u128(hash, states.len() as u128);
    for state in states {
        hash_u128(hash, state.0 as u128);
    }
}

fn hash_option_state(hash: &mut u128, state: Option<StateId>) {
    match state {
        Some(state) => {
            hash_u8(hash, 1);
            hash_u128(hash, state.0 as u128);
        }
        None => hash_u8(hash, 0),
    }
}

fn hash_option_u32(hash: &mut u128, value: Option<u32>) {
    match value {
        Some(value) => {
            hash_u8(hash, 1);
            hash_u128(hash, value as u128);
        }
        None => hash_u8(hash, 0),
    }
}

fn hash_history_mode(hash: &mut u128, mode: Option<HistoryMode>) {
    match mode {
        Some(HistoryMode::Shallow) => hash_u8(hash, 1),
        Some(HistoryMode::Deep) => hash_u8(hash, 2),
        None => hash_u8(hash, 0),
    }
}

fn hash_u8(hash: &mut u128, value: u8) {
    *hash ^= u128::from(value);
    *hash = hash.wrapping_mul(0x1000000000000000000013b_u128);
}

fn hash_u128(hash: &mut u128, value: u128) {
    for byte in value.to_le_bytes() {
        hash_u8(hash, byte);
    }
}

fn validate_history(
    plan: &Plan,
    owner: StateId,
    mode: HistoryMode,
    states: &[StateId],
) -> Result<(), RestoreError> {
    if states.is_empty() {
        return Err(RestoreError::EmptyHistory(owner));
    }
    let mut seen = vec![false; plan.state_count()];
    for &state in states {
        let index = state.0 as usize;
        if index >= seen.len() {
            return Err(RestoreError::InvalidState(state));
        }
        if seen[index] {
            return Err(RestoreError::DuplicateHistoryState { owner, state });
        }
        if state == owner || !plan.is_descendant(state, owner) {
            return Err(RestoreError::HistoryStateOutsideOwner { owner, state });
        }
        seen[index] = true;
    }

    for &state in states {
        let parent = plan
            .parent(state)
            .expect("non-root history state has a parent");
        if parent != owner && !seen[parent.0 as usize] {
            return Err(RestoreError::HistoryParentMissing {
                owner,
                state,
                parent,
            });
        }
    }

    if mode == HistoryMode::Shallow
        && states
            .iter()
            .any(|&state| plan.parent(state) != Some(owner))
    {
        return Err(RestoreError::InvalidHistoryChildren {
            owner,
            state: owner,
        });
    }
    // A deep entry is a complete active configuration, not merely a set of
    // direct children. Validate every remembered compound/parallel node as a
    // recursive configuration so a damaged nested region cannot be restored
    // into a partially active tree.
    if !history_children_match(plan, owner, &seen) {
        return Err(RestoreError::InvalidHistoryChildren {
            owner,
            state: owner,
        });
    }
    if mode == HistoryMode::Deep
        && states
            .iter()
            .any(|&state| !validate_children(plan, state, &seen))
    {
        let state = states
            .iter()
            .copied()
            .find(|&state| !validate_children(plan, state, &seen))
            .unwrap_or(owner);
        return Err(RestoreError::InvalidHistoryChildren { owner, state });
    }
    Ok(())
}

fn history_children_match(plan: &Plan, owner: StateId, seen: &[bool]) -> bool {
    let node = plan.node(owner).expect("validated owner");
    match &node.kind {
        NodeKind::Atomic => false,
        NodeKind::Compound { children, .. } => {
            children
                .iter()
                .filter(|child| seen[child.0 as usize])
                .count()
                == 1
        }
        NodeKind::Parallel { regions, .. } => {
            let direct = node
                .kind
                .children()
                .iter()
                .filter(|child| seen[child.0 as usize])
                .count();
            direct == regions.len()
                && regions.iter().all(|region| seen[region.0 as usize])
                && node
                    .kind
                    .children()
                    .iter()
                    .filter(|child| seen[child.0 as usize])
                    .all(|child| regions.contains(child))
        }
    }
}

fn validate_children(plan: &Plan, owner: StateId, seen: &[bool]) -> bool {
    let node = plan.node(owner).expect("validated state");
    match &node.kind {
        NodeKind::Atomic => !node
            .kind
            .children()
            .iter()
            .any(|child| seen[child.0 as usize]),
        NodeKind::Compound { children, .. } => {
            children
                .iter()
                .filter(|child| seen[child.0 as usize])
                .count()
                == 1
        }
        NodeKind::Parallel { regions, .. } => {
            let direct = node
                .kind
                .children()
                .iter()
                .filter(|child| seen[child.0 as usize])
                .count();
            direct == regions.len()
                && regions.iter().all(|region| seen[region.0 as usize])
                && node
                    .kind
                    .children()
                    .iter()
                    .filter(|child| seen[child.0 as usize])
                    .all(|child| regions.contains(child))
        }
    }
}

fn write_len(
    output: &mut Vec<u8>,
    length: usize,
    field: &'static str,
) -> Result<(), SnapshotCodecError> {
    let length = u32::try_from(length).map_err(|_| SnapshotCodecError::LengthOverflow(field))?;
    write_u32(output, length);
    Ok(())
}

fn write_u32(output: &mut Vec<u8>, value: u32) {
    output.extend_from_slice(&value.to_le_bytes());
}

struct Reader<'a> {
    bytes: &'a [u8],
    position: usize,
}

impl<'a> Reader<'a> {
    const fn new(bytes: &'a [u8]) -> Self {
        Self { bytes, position: 0 }
    }

    fn take(&mut self, length: usize) -> Result<&'a [u8], SnapshotCodecError> {
        let end = self
            .position
            .checked_add(length)
            .ok_or(SnapshotCodecError::Truncated)?;
        if end > self.bytes.len() {
            return Err(SnapshotCodecError::Truncated);
        }
        let result = &self.bytes[self.position..end];
        self.position = end;
        Ok(result)
    }

    fn byte(&mut self) -> Result<u8, SnapshotCodecError> {
        Ok(*self.take(1)?.first().expect("one byte was requested"))
    }

    fn u32(&mut self) -> Result<u32, SnapshotCodecError> {
        let bytes = self.take(4)?;
        Ok(u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]))
    }

    fn u128(&mut self) -> Result<u128, SnapshotCodecError> {
        let bytes = self.take(16)?;
        Ok(u128::from_le_bytes([
            bytes[0], bytes[1], bytes[2], bytes[3], bytes[4], bytes[5], bytes[6], bytes[7],
            bytes[8], bytes[9], bytes[10], bytes[11], bytes[12], bytes[13], bytes[14], bytes[15],
        ]))
    }

    fn count(
        &mut self,
        _field: &'static str,
        minimum_item_bytes: usize,
    ) -> Result<usize, SnapshotCodecError> {
        let count =
            usize::try_from(self.u32()?).map_err(|_| SnapshotCodecError::LengthNotRepresentable)?;
        if count > usize::MAX / minimum_item_bytes
            || count > self.bytes.len().saturating_sub(self.position) / minimum_item_bytes
        {
            return Err(SnapshotCodecError::LengthNotRepresentable);
        }
        Ok(count)
    }

    const fn is_empty(&self) -> bool {
        self.position == self.bytes.len()
    }
}

#[cfg(test)]
mod tests {
    use alloc::{vec, vec::Vec};

    use crate::{
        EventId, Guard, GuardContext, GuardId, HistoryMode, MicrostepLimit, NoopLifecycle,
        StateNode, Transition, TransitionKind, TransitionTarget, TreeDef, Trigger,
    };

    use super::*;

    fn history_plan() -> Plan {
        let root = StateNode::compound(
            StateId(0),
            None,
            vec![StateId(1), StateId(4)],
            StateId(1),
            None,
        );
        let owner = StateNode::compound(
            StateId(1),
            Some(StateId(0)),
            vec![StateId(2), StateId(3)],
            StateId(2),
            Some(HistoryMode::Deep),
        );
        let leaf_a =
            StateNode::atomic(StateId(2), Some(StateId(1))).with_transition(Transition::new(
                StateId(2),
                TransitionTarget::State(StateId(4)),
                Trigger::Event(EventId(1)),
                None,
                0,
                TransitionKind::External,
            ));
        let leaf_b = StateNode::atomic(StateId(3), Some(StateId(1)));
        let outside =
            StateNode::atomic(StateId(4), Some(StateId(0))).with_transition(Transition::new(
                StateId(4),
                TransitionTarget::History {
                    owner: StateId(1),
                    depth: HistoryMode::Deep,
                },
                Trigger::Event(EventId(2)),
                None,
                0,
                TransitionKind::External,
            ));
        TreeDef {
            root: StateId(0),
            nodes: vec![root, owner, leaf_a, leaf_b, outside],
        }
        .compile()
        .unwrap()
    }

    fn legacy(snapshot: Snapshot) -> Snapshot {
        Snapshot {
            plan_fingerprint: None,
            ..snapshot
        }
    }

    fn instance(capacity: usize) -> Instance {
        Instance::new(history_plan(), capacity, MicrostepLimit(8)).unwrap()
    }

    fn start(instance: &mut Instance) {
        instance.start(&mut NoopLifecycle).unwrap();
    }

    #[test]
    fn snapshot_bytes_round_trip_and_instance_restore() {
        let mut source = instance(3);
        start(&mut source);
        source.send(Event::new(EventId(8))).unwrap();
        source.send(Event::new(EventId(9))).unwrap();
        let snapshot = source.snapshot();
        let decoded = Snapshot::decode(&snapshot.encode().unwrap()).unwrap();
        assert_eq!(decoded, snapshot);

        let mut target = instance(3);
        target.restore(&decoded).unwrap();
        assert_eq!(target.snapshot(), snapshot);
    }

    #[test]
    fn legacy_v1_snapshot_decodes_and_restores_without_fingerprint() {
        let mut source = instance(3);
        start(&mut source);
        source.send(Event::new(EventId(8))).unwrap();
        let current = source.snapshot();
        let legacy = legacy(current.clone());
        let decoded = Snapshot::decode(&legacy.encode().unwrap()).unwrap();

        assert_eq!(decoded, legacy);
        assert_eq!(decoded.plan_fingerprint, None);

        let mut target = instance(3);
        target.restore(&decoded).unwrap();
        assert_eq!(target.snapshot().started, current.started);
        assert_eq!(target.snapshot().active, current.active);
        assert_eq!(target.snapshot().history, current.history);
        assert_eq!(target.snapshot().queued_events, current.queued_events);
    }

    #[test]
    fn plan_fingerprint_rejects_semantically_different_plan_atomically() {
        let mut source = instance(2);
        start(&mut source);
        let snapshot = source.snapshot();

        let mut different_plan = history_plan();
        different_plan.nodes[2].transitions[0].trigger = Trigger::Event(EventId(99));
        let mut target = Instance::new(different_plan, 2, MicrostepLimit(8)).unwrap();
        start(&mut target);
        target.send(Event::new(EventId(7))).unwrap();
        let before = target.snapshot();

        assert_eq!(
            target.restore(&snapshot),
            Err(RestoreError::PlanFingerprintMismatch {
                expected: plan_fingerprint(target.plan()),
                actual: snapshot.plan_fingerprint.unwrap(),
            })
        );
        assert_eq!(target.snapshot(), before);
    }

    #[test]
    fn restore_rejects_unknown_state_without_mutation() {
        let mut target = instance(2);
        start(&mut target);
        target.send(Event::new(EventId(7))).unwrap();
        let before = target.snapshot();
        let mut invalid = before.clone();
        invalid.active.push(StateId(99));
        assert_eq!(
            target.restore(&invalid),
            Err(RestoreError::InvalidState(StateId(99)))
        );
        assert_eq!(target.snapshot(), before);
    }

    #[test]
    fn restore_checks_parent_and_compound_configuration() {
        let mut target = instance(2);
        let invalid = Snapshot {
            plan_fingerprint: Some(plan_fingerprint(target.plan())),
            started: true,
            active: vec![StateId(0), StateId(2)],
            history: vec![None; 5],
            queued_events: Vec::new(),
        };
        assert_eq!(
            target.restore(&invalid),
            Err(RestoreError::ActiveParentMissing {
                state: StateId(2),
                parent: StateId(1),
            })
        );
        assert_eq!(
            target.snapshot(),
            Snapshot {
                plan_fingerprint: Some(plan_fingerprint(target.plan())),
                started: false,
                active: Vec::new(),
                history: vec![None; 5],
                queued_events: Vec::new(),
            }
        );
    }

    #[test]
    fn restore_rejects_compound_with_two_active_children() {
        let mut target = instance(2);
        let invalid = Snapshot {
            plan_fingerprint: Some(plan_fingerprint(target.plan())),
            started: true,
            active: vec![StateId(0), StateId(1), StateId(2), StateId(3)],
            history: vec![None; 5],
            queued_events: Vec::new(),
        };
        assert_eq!(
            target.restore(&invalid),
            Err(RestoreError::InvalidActiveChildren(StateId(1)))
        );
    }

    #[test]
    fn restore_rejects_queue_over_capacity_atomically() {
        let mut target = instance(1);
        start(&mut target);
        let before = target.snapshot();
        let mut invalid = before.clone();
        invalid.queued_events = vec![Event::new(EventId(1)), Event::new(EventId(2))];
        assert_eq!(
            target.restore(&invalid),
            Err(RestoreError::QueueCapacityExceeded {
                capacity: 1,
                queued: 2,
            })
        );
        assert_eq!(target.snapshot(), before);
    }

    #[test]
    fn history_is_restored_and_drives_later_history_transition() {
        let mut source = instance(2);
        start(&mut source);
        source.send(Event::new(EventId(1))).unwrap();
        source.run(&mut NoopLifecycle).unwrap();
        assert_eq!(source.snapshot().history[1], Some(vec![StateId(2)]));

        let snapshot = source.snapshot();
        let mut target = instance(2);
        target.restore(&snapshot).unwrap();
        target.send(Event::new(EventId(2))).unwrap();
        target.run(&mut NoopLifecycle).unwrap();
        assert_eq!(
            target.active_states(),
            &[StateId(0), StateId(1), StateId(2)]
        );
    }

    #[test]
    fn restore_keeps_installed_guard_objects() {
        struct RejectGuard;
        impl Guard for RejectGuard {
            fn evaluate(&self, _context: GuardContext) -> bool {
                false
            }
        }

        let root = StateNode::compound(
            StateId(0),
            None,
            vec![StateId(1), StateId(2)],
            StateId(1),
            None,
        );
        let leaf =
            StateNode::atomic(StateId(1), Some(StateId(0))).with_transition(Transition::new(
                StateId(1),
                TransitionTarget::State(StateId(2)),
                Trigger::Event(EventId(3)),
                Some(GuardId(0)),
                0,
                TransitionKind::External,
            ));
        let other = StateNode::atomic(StateId(2), Some(StateId(0)));
        let plan = TreeDef {
            root: StateId(0),
            nodes: vec![root, leaf, other],
        }
        .compile()
        .unwrap();
        let mut target = Instance::new(plan, 2, MicrostepLimit(4)).unwrap();
        target.set_guard(GuardId(0), RejectGuard);
        start(&mut target);
        let snapshot = target.snapshot();
        target.restore(&snapshot).unwrap();
        target.send(Event::new(EventId(3))).unwrap();
        target.run(&mut NoopLifecycle).unwrap();
        assert_eq!(target.active_states(), &[StateId(0), StateId(1)]);
    }

    #[test]
    fn malformed_history_is_rejected_before_any_state_changes() {
        let mut target = instance(2);
        start(&mut target);
        target.send(Event::new(EventId(6))).unwrap();
        let before = target.snapshot();
        let mut invalid = before.clone();
        invalid.history[1] = Some(vec![StateId(4)]);
        assert_eq!(
            target.restore(&invalid),
            Err(RestoreError::HistoryStateOutsideOwner {
                owner: StateId(1),
                state: StateId(4),
            })
        );
        assert_eq!(target.snapshot(), before);
    }

    #[test]
    fn history_length_and_malformed_codec_are_checked() {
        let mut target = instance(2);
        let invalid = Snapshot {
            plan_fingerprint: Some(plan_fingerprint(target.plan())),
            started: false,
            active: Vec::new(),
            history: Vec::new(),
            queued_events: Vec::new(),
        };
        assert_eq!(
            target.restore(&invalid),
            Err(RestoreError::HistoryLength {
                expected: 5,
                actual: 0,
            })
        );
        assert_eq!(
            Snapshot::decode(b"nope"),
            Err(SnapshotCodecError::InvalidHeader)
        );
        let mut encoded = Snapshot {
            plan_fingerprint: Some(plan_fingerprint(target.plan())),
            started: false,
            active: Vec::new(),
            history: vec![None; 5],
            queued_events: Vec::new(),
        }
        .encode()
        .unwrap();
        encoded.push(0);
        assert_eq!(
            Snapshot::decode(&encoded),
            Err(SnapshotCodecError::TrailingBytes)
        );
    }
}
