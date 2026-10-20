use crate::clock::{ClockError, HlcClock, HlcTimestamp};
use crate::command::{
    ActorId, CommandEnvelope, CommandId, CommandLimits, CommandPayload, FieldPath, Value,
};
use core::fmt;
use std::collections::{BTreeMap, BTreeSet};

#[derive(Clone, Debug, Eq, PartialEq)]
struct Register {
    timestamp: HlcTimestamp,
    actor: ActorId,
    sequence: u64,
    value: Option<Value>,
}

impl Register {
    fn wins_over(&self, other: &Self) -> bool {
        (self.timestamp, &self.actor, self.sequence)
            > (other.timestamp, &other.actor, other.sequence)
    }
}

/// Materialized field-level LWW document state.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct DocumentState {
    registers: BTreeMap<FieldPath, Register>,
}

impl DocumentState {
    /// Creates an empty document.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Returns visible fields, excluding delete tombstones.
    #[must_use]
    pub fn fields(&self) -> BTreeMap<FieldPath, Value> {
        self.registers
            .iter()
            .filter_map(|(path, register)| {
                register.value.clone().map(|value| (path.clone(), value))
            })
            .collect()
    }

    /// Returns a visible value for a path.
    #[must_use]
    pub fn get(&self, path: &FieldPath) -> Option<&Value> {
        self.registers
            .get(path)
            .and_then(|register| register.value.as_ref())
    }

    /// Returns whether a path has a tombstone, including invisible state.
    #[must_use]
    pub fn has_tombstone(&self, path: &FieldPath) -> bool {
        self.registers
            .get(path)
            .is_some_and(|register| register.value.is_none())
    }

    fn apply(&mut self, command: &CommandEnvelope) -> ApplyOutcome {
        let (path, value) = match &command.payload {
            CommandPayload::SetField { path, value } => (path, Some(value.clone())),
            CommandPayload::DeleteField { path } => (path, None),
            CommandPayload::Opaque(_) => return ApplyOutcome::RecordedOnly,
        };
        let candidate = Register {
            timestamp: command.timestamp,
            actor: command.actor.clone(),
            sequence: command.id.sequence,
            value,
        };
        match self.registers.get(path) {
            Some(existing) if !candidate.wins_over(existing) => ApplyOutcome::IgnoredOlderWrite,
            _ => {
                self.registers.insert(path.clone(), candidate);
                ApplyOutcome::Applied
            }
        }
    }
}

/// Result of applying a command payload to a document.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ApplyOutcome {
    /// The register was replaced by this command.
    Applied,
    /// The command was valid but older than the current register.
    IgnoredOlderWrite,
    /// The command was recorded without document interpretation.
    RecordedOnly,
}

/// Immutable state plus revision used to bootstrap a replica.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Snapshot {
    /// State at the snapshot revision.
    pub state: DocumentState,
    /// Number of applied commands represented by the snapshot.
    pub revision: u64,
}

/// One retained command and its monotonically increasing local revision.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct HistoryEntry {
    /// Local application revision.
    pub revision: u64,
    /// Applied command.
    pub command: CommandEnvelope,
}

/// Current bounded-history accounting.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct HistoryInfo {
    /// Number of commands retained in memory.
    pub retained: usize,
    /// Number of older commands no longer available for incremental replay.
    pub dropped: u64,
    /// First retained revision, or the next revision when no history exists.
    pub first_retained_revision: u64,
    /// Most recent applied revision.
    pub latest_revision: u64,
}

/// Reasons an incoming command cannot be applied.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum ApplyError {
    /// The incoming command failed the replica's current command limits.
    InvalidCommand(crate::command::CommandError),
    /// The same command ID was received with different bytes/metadata.
    ConflictingDuplicate(CommandId),
    /// The public envelope fields disagree about the command actor.
    ActorMismatch(CommandId),
    /// The command has an unresolvable causal cycle.
    CausalCycle(CommandId),
    /// The replica clock could not advance.
    Clock(ClockError),
    /// A bounded replica store cannot retain another command in this class.
    LimitExceeded {
        /// The bounded store that reached its configured limit.
        kind: &'static str,
        /// Maximum number of entries allowed in that store.
        limit: usize,
    },
}

impl fmt::Display for ApplyError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "collaboration apply error: {self:?}")
    }
}

impl std::error::Error for ApplyError {}

impl From<ClockError> for ApplyError {
    fn from(error: ClockError) -> Self {
        Self::Clock(error)
    }
}

impl From<crate::command::CommandError> for ApplyError {
    fn from(error: crate::command::CommandError) -> Self {
        Self::InvalidCommand(error)
    }
}

/// Memory bounds for one collaboration replica.
///
/// The replica rejects new commands when any bound is reached. It does not
/// silently evict command IDs because doing so would break causal ordering or
/// duplicate detection; callers can install a snapshot into a fresh replica
/// when they want to compact the command metadata.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ReplicaLimits {
    /// Maximum number of received command envelopes retained for deduplication
    /// and replay.
    pub max_known_commands: usize,
    /// Maximum number of commands waiting for causal predecessors.
    pub max_pending_commands: usize,
    /// Maximum number of applied command IDs retained for causal checks.
    pub max_applied_commands: usize,
}

impl Default for ReplicaLimits {
    fn default() -> Self {
        Self {
            max_known_commands: 65_536,
            max_pending_commands: 4_096,
            max_applied_commands: 65_536,
        }
    }
}

/// Result of receiving one command and draining newly ready commands.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum ReceiveOutcome {
    /// Command IDs applied in this call, in causal-valid order.
    Applied(Vec<CommandId>),
    /// Command was buffered because one or more causal IDs are not applied.
    Buffered(Vec<CommandId>),
    /// Exact duplicate was ignored.
    Duplicate(CommandId),
}

/// History query failures.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum HistoryError {
    /// The requested cursor predates retained history.
    BeforeRetention {
        /// Cursor supplied by the caller.
        requested_after: u64,
        /// First revision still available.
        first_retained: u64,
    },
}

impl fmt::Display for HistoryError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "history query error: {self:?}")
    }
}

impl std::error::Error for HistoryError {}

/// Replay failures for a command set or a bounded history window.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum ReplayError {
    /// The requested history window was truncated.
    History(HistoryError),
    /// A command depends on an ID not available in the replay input.
    MissingDependency {
        /// Command whose dependency cannot be found.
        command: CommandId,
        /// Missing causal predecessor.
        dependency: CommandId,
    },
    /// Commands in the replay input form a cycle.
    CausalCycle(CommandId),
}

impl fmt::Display for ReplayError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "collaboration replay error: {self:?}")
    }
}

impl std::error::Error for ReplayError {}

/// A replica of the generic field-register document.
#[derive(Clone, Debug)]
pub struct Replica {
    actor: ActorId,
    clock: HlcClock,
    command_limits: CommandLimits,
    limits: ReplicaLimits,
    history_limit: usize,
    state: DocumentState,
    known: BTreeMap<CommandId, CommandEnvelope>,
    pending: BTreeMap<CommandId, CommandEnvelope>,
    applied: BTreeSet<CommandId>,
    history: Vec<HistoryEntry>,
    dropped: u64,
    revision: u64,
}

impl Replica {
    /// Creates a replica with a bounded retained history.
    pub fn new(actor: ActorId, history_limit: usize) -> Result<Self, ApplyError> {
        let clock = HlcClock::new(actor.clone());
        Ok(Self {
            actor,
            clock,
            command_limits: CommandLimits::default(),
            limits: ReplicaLimits::default(),
            history_limit,
            state: DocumentState::new(),
            known: BTreeMap::new(),
            pending: BTreeMap::new(),
            applied: BTreeSet::new(),
            history: Vec::new(),
            dropped: 0,
            revision: 0,
        })
    }

    /// Sets command validation limits before receiving commands.
    pub fn with_command_limits(mut self, limits: CommandLimits) -> Self {
        self.command_limits = limits;
        self
    }

    /// Sets explicit in-memory command bounds before receiving commands.
    #[must_use]
    pub fn with_replica_limits(mut self, limits: ReplicaLimits) -> Self {
        self.limits = limits;
        self
    }

    /// Returns the configured in-memory command bounds.
    #[must_use]
    pub const fn replica_limits(&self) -> ReplicaLimits {
        self.limits
    }

    /// Returns the local actor identity.
    #[must_use]
    pub fn actor(&self) -> &ActorId {
        &self.actor
    }

    /// Returns the materialized state.
    #[must_use]
    pub fn state(&self) -> &DocumentState {
        &self.state
    }

    /// Returns the current revision.
    #[must_use]
    pub const fn revision(&self) -> u64 {
        self.revision
    }

    /// Returns bounded history accounting.
    #[must_use]
    pub fn history_info(&self) -> HistoryInfo {
        HistoryInfo {
            retained: self.history.len(),
            dropped: self.dropped,
            first_retained_revision: self
                .history
                .first()
                .map_or(self.revision.saturating_add(1), |entry| entry.revision),
            latest_revision: self.revision,
        }
    }

    /// Returns IDs that are currently waiting on missing causal commands.
    #[must_use]
    pub fn pending_ids(&self) -> Vec<CommandId> {
        self.pending.keys().cloned().collect()
    }

    /// Receives one command, deduplicates it, and drains causal-ready work.
    pub fn receive(&mut self, command: CommandEnvelope) -> Result<ReceiveOutcome, ApplyError> {
        let command = self.validate_command(command)?;
        if let Some(existing) = self.known.get(&command.id) {
            if existing == &command {
                return Ok(ReceiveOutcome::Duplicate(command.id));
            }
            return Err(ApplyError::ConflictingDuplicate(command.id));
        }
        let mut candidate = self.clone();
        let outcome = candidate.receive_new(command)?;
        *self = candidate;
        Ok(outcome)
    }

    fn receive_new(&mut self, command: CommandEnvelope) -> Result<ReceiveOutcome, ApplyError> {
        if command.id.actor != command.actor {
            return Err(ApplyError::ActorMismatch(command.id));
        }
        if self.known.len() >= self.limits.max_known_commands {
            return Err(ApplyError::LimitExceeded {
                kind: "known commands",
                limit: self.limits.max_known_commands,
            });
        }
        let ready = command
            .causal
            .iter()
            .all(|dependency| self.applied.contains(dependency));
        if self.pending.len() >= self.limits.max_pending_commands && !ready {
            return Err(ApplyError::LimitExceeded {
                kind: "pending commands",
                limit: self.limits.max_pending_commands,
            });
        }
        self.clock
            .observe(command.timestamp, command.timestamp.physical)?;
        let command_id = command.id.clone();
        self.known.insert(command_id.clone(), command.clone());
        let applied = if ready && self.pending.len() >= self.limits.max_pending_commands {
            self.apply_one(command)?;
            let mut applied = vec![command_id.clone()];
            applied.extend(self.drain_ready()?);
            applied
        } else {
            self.pending.insert(command_id.clone(), command);
            self.drain_ready()?
        };
        if !applied.is_empty() {
            Ok(ReceiveOutcome::Applied(applied))
        } else if let Some(cycle) = self.pending_cycle() {
            self.known.remove(&command_id);
            self.pending.remove(&command_id);
            Err(ApplyError::CausalCycle(cycle))
        } else {
            Ok(ReceiveOutcome::Buffered(self.missing_for(&command_id)))
        }
    }

    fn validate_command(&self, command: CommandEnvelope) -> Result<CommandEnvelope, ApplyError> {
        let command_id = command.id.clone();
        if command_id.actor != command.actor {
            return Err(ApplyError::ActorMismatch(command_id));
        }
        CommandEnvelope::new(
            command.actor,
            command_id.sequence,
            command.timestamp,
            command.causal,
            command.affected_domains,
            command.payload,
            self.command_limits,
        )
        .map_err(ApplyError::InvalidCommand)
    }

    /// Returns retained commands after a local revision cursor.
    pub fn history_since(&self, after_revision: u64) -> Result<Vec<HistoryEntry>, HistoryError> {
        let info = self.history_info();
        if info.dropped > 0 && after_revision.saturating_add(1) < info.first_retained_revision {
            return Err(HistoryError::BeforeRetention {
                requested_after: after_revision,
                first_retained: info.first_retained_revision,
            });
        }
        Ok(self
            .history
            .iter()
            .filter(|entry| entry.revision > after_revision)
            .cloned()
            .collect())
    }

    /// Takes a materialized snapshot that can seed a later replay.
    #[must_use]
    pub fn snapshot(&self) -> Snapshot {
        Snapshot {
            state: self.state.clone(),
            revision: self.revision,
        }
    }

    /// Replays all known commands in deterministic causal order.
    pub fn replay_all(&self) -> Result<DocumentState, ReplayError> {
        replay_commands(self.known.values().cloned().collect())
    }

    /// Replays retained history on a snapshot.
    pub fn replay_from_snapshot(&self, snapshot: &Snapshot) -> Result<DocumentState, ReplayError> {
        let entries = self
            .history_since(snapshot.revision)
            .map_err(ReplayError::History)?;
        let mut state = snapshot.state.clone();
        for entry in entries {
            state.apply(&entry.command);
        }
        Ok(state)
    }

    fn missing_for(&self, id: &CommandId) -> Vec<CommandId> {
        self.pending
            .get(id)
            .map(|command| {
                command
                    .causal
                    .iter()
                    .filter(|dependency| !self.applied.contains(*dependency))
                    .cloned()
                    .collect()
            })
            .unwrap_or_default()
    }

    fn pending_cycle(&self) -> Option<CommandId> {
        let mut active: BTreeSet<_> = self.pending.keys().cloned().collect();
        loop {
            let removable: Vec<_> = active
                .iter()
                .filter(|id| {
                    self.pending.get(*id).is_some_and(|command| {
                        command.causal.iter().all(|dependency| {
                            self.applied.contains(dependency) || !active.contains(dependency)
                        })
                    })
                })
                .cloned()
                .collect();
            if removable.is_empty() {
                return active.into_iter().next();
            }
            for id in removable {
                active.remove(&id);
            }
            if active.is_empty() {
                return None;
            }
        }
    }

    fn drain_ready(&mut self) -> Result<Vec<CommandId>, ApplyError> {
        let mut applied_now = Vec::new();
        loop {
            let next = self
                .pending
                .values()
                .filter(|command| command.causal.iter().all(|id| self.applied.contains(id)))
                .min_by(|left, right| left.order_key().cmp(&right.order_key()))
                .map(|command| command.id.clone());
            let Some(id) = next else { break };
            let command = self
                .pending
                .remove(&id)
                .expect("ready command must remain in pending");
            self.apply_one(command)?;
            applied_now.push(id);
        }
        Ok(applied_now)
    }

    fn apply_one(&mut self, command: CommandEnvelope) -> Result<(), ApplyError> {
        if self.applied.len() >= self.limits.max_applied_commands {
            return Err(ApplyError::LimitExceeded {
                kind: "applied commands",
                limit: self.limits.max_applied_commands,
            });
        }
        let id = command.id.clone();
        self.state.apply(&command);
        self.applied.insert(id);
        self.revision = self.revision.saturating_add(1);
        self.history.push(HistoryEntry {
            revision: self.revision,
            command,
        });
        if self.history.len() > self.history_limit {
            let overflow = self.history.len() - self.history_limit;
            self.history.drain(..overflow);
            self.dropped = self.dropped.saturating_add(overflow as u64);
        }
        Ok(())
    }
}

fn replay_commands(commands: Vec<CommandEnvelope>) -> Result<DocumentState, ReplayError> {
    let by_id: BTreeMap<CommandId, CommandEnvelope> = commands
        .into_iter()
        .map(|command| (command.id.clone(), command))
        .collect();
    let mut applied = BTreeSet::new();
    let mut state = DocumentState::new();
    while applied.len() < by_id.len() {
        let next = by_id
            .values()
            .filter(|command| !applied.contains(&command.id))
            .filter(|command| command.causal.iter().all(|id| by_id.contains_key(id)))
            .filter(|command| command.causal.iter().all(|id| applied.contains(id)))
            .min_by(|left, right| left.order_key().cmp(&right.order_key()));
        let Some(command) = next else {
            let command = by_id
                .values()
                .find(|command| !applied.contains(&command.id))
                .expect("unresolved replay must contain a command");
            if let Some(dependency) = command.causal.iter().find(|id| !by_id.contains_key(*id)) {
                return Err(ReplayError::MissingDependency {
                    command: command.id.clone(),
                    dependency: dependency.clone(),
                });
            }
            return Err(ReplayError::CausalCycle(command.id.clone()));
        };
        state.apply(command);
        applied.insert(command.id.clone());
    }
    Ok(state)
}
