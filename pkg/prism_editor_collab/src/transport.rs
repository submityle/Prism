use crate::command::{
    ActorId, CommandEnvelope, CommandId, CommandKind, CommandLimits, CommandPayload, DomainId,
    FieldPath, PathSegment, Value,
};
use crate::document::{ApplyError, ReceiveOutcome, Replica, ReplicaLimits};
use std::collections::{BTreeMap, BTreeSet, VecDeque};
use std::fmt;
use std::io::{self, ErrorKind, Read, Write};
use std::net::{SocketAddr, TcpListener, TcpStream};
use std::sync::mpsc::{self, Receiver, SyncSender, TrySendError};
use std::sync::{Arc, Mutex};
use std::thread::{self, JoinHandle};
use std::time::Duration;

const PROTOCOL_VERSION: u8 = 1;
const MESSAGE_HELLO: u8 = 1;
const MESSAGE_WELCOME: u8 = 2;
const MESSAGE_SUBMIT: u8 = 3;
const MESSAGE_COMMAND: u8 = 4;
const MESSAGE_REJECT: u8 = 5;
const MAX_STRING_BYTES: usize = 1024 * 1024;

/// Bounded resources used by the TCP transport.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct TransportLimits {
    /// Maximum encoded frame body in bytes.
    pub max_frame_bytes: usize,
    /// Maximum number of outbound messages queued for one client.
    pub max_client_queue: usize,
    /// Maximum accepted command history retained by the server.
    pub max_history_commands: usize,
    /// Maximum simultaneously registered clients.
    pub max_clients: usize,
}

impl Default for TransportLimits {
    fn default() -> Self {
        Self {
            max_frame_bytes: 8 * 1024 * 1024,
            max_client_queue: 1_024,
            max_history_commands: 512,
            max_clients: 64,
        }
    }
}

impl TransportLimits {
    fn validate(self) -> Result<(), TransportConfigError> {
        if self.max_frame_bytes < 64 {
            return Err(TransportConfigError::InvalidLimit("max_frame_bytes"));
        }
        if self.max_client_queue == 0 {
            return Err(TransportConfigError::InvalidLimit("max_client_queue"));
        }
        if self.max_history_commands == 0 {
            return Err(TransportConfigError::InvalidLimit("max_history_commands"));
        }
        if self.max_clients == 0 {
            return Err(TransportConfigError::InvalidLimit("max_clients"));
        }
        if self.max_history_commands >= self.max_client_queue {
            return Err(TransportConfigError::HistoryExceedsQueue);
        }
        Ok(())
    }
}

/// Permission policy for one authenticated actor.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CommandCapabilities {
    /// Allows commands touching any domain when true.
    pub allow_all_domains: bool,
    /// Domains allowed when `allow_all_domains` is false.
    pub allowed_domains: BTreeSet<DomainId>,
    /// Allows field writes.
    pub allow_set_field: bool,
    /// Allows field tombstones.
    pub allow_delete_field: bool,
    /// Allows adapter-defined commands.
    pub allow_opaque: bool,
    /// Maximum payload size accepted for this actor.
    pub max_payload_bytes: usize,
}

impl CommandCapabilities {
    /// Creates a policy that permits every command within the payload bound.
    #[must_use]
    pub fn unrestricted(max_payload_bytes: usize) -> Self {
        Self {
            allow_all_domains: true,
            allowed_domains: BTreeSet::new(),
            allow_set_field: true,
            allow_delete_field: true,
            allow_opaque: true,
            max_payload_bytes,
        }
    }

    fn check(&self, command: &CommandEnvelope) -> Result<(), CapabilityError> {
        if !self.allow_all_domains
            && let Some(domain) = command
                .affected_domains
                .iter()
                .find(|domain| !self.allowed_domains.contains(*domain))
        {
            return Err(CapabilityError::DomainDenied(domain.clone()));
        }
        let allowed = match command.kind() {
            CommandKind::SetField => self.allow_set_field,
            CommandKind::DeleteField => self.allow_delete_field,
            CommandKind::Opaque => self.allow_opaque,
        };
        if !allowed {
            return Err(CapabilityError::OperationDenied(command.kind()));
        }
        if command.payload_size() > self.max_payload_bytes {
            return Err(CapabilityError::PayloadTooLarge {
                actual: command.payload_size(),
                limit: self.max_payload_bytes,
            });
        }
        Ok(())
    }
}

/// Capability policy failures.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum CapabilityError {
    /// The actor has no registered capability policy.
    UnknownActor,
    /// One affected domain is not allowed.
    DomainDenied(DomainId),
    /// The operation class is not allowed.
    OperationDenied(CommandKind),
    /// The command payload exceeds the actor bound.
    PayloadTooLarge {
        /// Actual encoded payload size.
        actual: usize,
        /// Maximum payload size allowed by the policy.
        limit: usize,
    },
}

impl fmt::Display for CapabilityError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "command capability denied: {self:?}")
    }
}

impl std::error::Error for CapabilityError {}

/// Server configuration.
#[derive(Clone, Debug)]
pub struct ServerConfig {
    /// Wire and queue bounds.
    pub transport: TransportLimits,
    /// Command envelope validation bounds.
    pub command_limits: CommandLimits,
    /// Replica metadata bounds used by the authoritative server.
    pub replica_limits: ReplicaLimits,
    /// Capability policy keyed by authenticated actor identity.
    pub capabilities: BTreeMap<ActorId, CommandCapabilities>,
}

impl Default for ServerConfig {
    fn default() -> Self {
        let transport = TransportLimits::default();
        Self {
            transport,
            command_limits: CommandLimits::default(),
            replica_limits: ReplicaLimits {
                max_known_commands: transport.max_history_commands,
                max_pending_commands: transport.max_history_commands,
                max_applied_commands: transport.max_history_commands,
            },
            capabilities: BTreeMap::new(),
        }
    }
}

impl ServerConfig {
    /// Adds or replaces an actor capability policy.
    #[must_use]
    pub fn with_capability(mut self, actor: ActorId, capability: CommandCapabilities) -> Self {
        self.capabilities.insert(actor, capability);
        self
    }

    fn validate(&self) -> Result<(), TransportConfigError> {
        self.transport.validate()?;
        if self.replica_limits.max_known_commands < self.transport.max_history_commands {
            return Err(TransportConfigError::ReplicaBoundTooSmall(
                "max_known_commands",
            ));
        }
        if self.replica_limits.max_applied_commands < self.transport.max_history_commands {
            return Err(TransportConfigError::ReplicaBoundTooSmall(
                "max_applied_commands",
            ));
        }
        if self.replica_limits.max_pending_commands == 0 {
            return Err(TransportConfigError::InvalidLimit("max_pending_commands"));
        }
        Ok(())
    }
}

/// Configuration failures detected before opening a listener.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum TransportConfigError {
    /// A positive bounded setting was zero or too small.
    InvalidLimit(&'static str),
    /// Replay history cannot fit in a bounded client queue.
    HistoryExceedsQueue,
    /// Authoritative replica metadata cannot retain configured history.
    ReplicaBoundTooSmall(&'static str),
}

impl fmt::Display for TransportConfigError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            formatter,
            "invalid collaboration transport configuration: {self:?}"
        )
    }
}

impl std::error::Error for TransportConfigError {}

/// Reasons a server rejected a submitted command.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum RejectionReason {
    /// Actor is not present in the server capability table.
    Unauthorized,
    /// Actor policy rejected the command.
    Capability(CapabilityError),
    /// Envelope validation failed.
    InvalidCommand,
    /// Replica detected a causal or metadata failure.
    Replica,
    /// Server history reached its explicit bound.
    HistoryFull,
    /// Client protocol state was invalid.
    Protocol,
}

/// Events delivered by a connected client.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum ClientEvent {
    /// One authoritative server order was applied to the local replica.
    Command {
        /// Server-assigned admission order.
        order: u64,
        /// Original command envelope.
        command: CommandEnvelope,
        /// IDs newly applied by the local causal drain.
        applied: Vec<CommandId>,
    },
    /// The server rejected a queued command; it was removed from the queue.
    Rejected {
        /// ID of the rejected command when available.
        command: Option<CommandId>,
        /// Explicit rejection reason.
        reason: RejectionReason,
    },
}

/// Client-side transport failures.
#[derive(Debug)]
pub enum ClientError {
    /// Standard library I/O failure.
    Io(io::Error),
    /// The server closed the connection; queued commands remain retained.
    Disconnected,
    /// The local pending-command queue is full.
    QueueFull,
    /// A pending command ID conflicts with an existing queued command.
    ConflictingPending(CommandId),
    /// The server order stream contains a gap.
    ServerOrderGap {
        /// Next order required by the client.
        expected: u64,
        /// Order received from the server.
        observed: u64,
    },
    /// A command could not be applied to the local replica.
    Replica(ApplyError),
    /// The peer sent an invalid wire message.
    Protocol(String),
    /// The client was rejected during handshake.
    Handshake(RejectionReason),
}

impl fmt::Display for ClientError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "collaboration client error: {self:?}")
    }
}

impl std::error::Error for ClientError {}

impl From<io::Error> for ClientError {
    fn from(error: io::Error) -> Self {
        Self::Io(error)
    }
}

/// Server-side failures.
#[derive(Debug)]
pub enum ServerError {
    /// Standard library I/O failure.
    Io(io::Error),
    /// Configuration was rejected.
    Configuration(TransportConfigError),
    /// The server's authoritative replica could not be created.
    Replica(ApplyError),
}

impl fmt::Display for ServerError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "collaboration server error: {self:?}")
    }
}

impl std::error::Error for ServerError {}

impl From<io::Error> for ServerError {
    fn from(error: io::Error) -> Self {
        Self::Io(error)
    }
}

#[derive(Clone, Debug)]
struct SequencedCommand {
    order: u64,
    command: CommandEnvelope,
}

#[derive(Clone, Debug)]
struct ServerState {
    config: ServerConfig,
    replica: Replica,
    accepted: Vec<SequencedCommand>,
    by_id: BTreeMap<CommandId, SequencedCommand>,
    sessions: BTreeMap<u64, SyncSender<WireMessage>>,
    next_order: u64,
    next_session: u64,
}

impl ServerState {
    fn new(config: ServerConfig) -> Result<Self, ServerError> {
        let server_actor = ActorId::from_static("__prism_collaboration_server__");
        let replica = Replica::new(server_actor, config.transport.max_history_commands)
            .map_err(ServerError::Replica)?
            .with_command_limits(config.command_limits)
            .with_replica_limits(config.replica_limits);
        Ok(Self {
            config,
            replica,
            accepted: Vec::new(),
            by_id: BTreeMap::new(),
            sessions: BTreeMap::new(),
            next_order: 1,
            next_session: 1,
        })
    }

    fn register(
        &mut self,
        actor: &ActorId,
        last_order: u64,
        sender: SyncSender<WireMessage>,
    ) -> Result<u64, RejectionReason> {
        if !self.config.capabilities.contains_key(actor) {
            return Err(RejectionReason::Unauthorized);
        }
        if last_order > self.next_order.saturating_sub(1) {
            return Err(RejectionReason::Protocol);
        }
        if self.sessions.len() >= self.config.transport.max_clients {
            return Err(RejectionReason::Protocol);
        }
        let replay: Vec<_> = self
            .accepted
            .iter()
            .filter(|entry| entry.order > last_order)
            .cloned()
            .collect();
        let session = self.next_session;
        self.next_session = self.next_session.saturating_add(1);
        self.sessions.insert(session, sender.clone());
        let welcome = WireMessage::Welcome {
            current_order: self.next_order.saturating_sub(1),
            replay_count: replay.len() as u64,
        };
        if sender.try_send(welcome).is_err() {
            self.sessions.remove(&session);
            return Err(RejectionReason::Protocol);
        }
        for entry in replay {
            if sender.try_send(WireMessage::Command(entry)).is_err() {
                self.sessions.remove(&session);
                return Err(RejectionReason::Protocol);
            }
        }
        Ok(session)
    }

    fn remove_session(&mut self, session: u64) {
        self.sessions.remove(&session);
    }

    fn submit(
        &mut self,
        actor: &ActorId,
        command: CommandEnvelope,
    ) -> Result<(SequencedCommand, bool), RejectionReason> {
        let capability = self
            .config
            .capabilities
            .get(actor)
            .ok_or(RejectionReason::Unauthorized)?;
        if command.actor != *actor {
            return Err(RejectionReason::Unauthorized);
        }
        let canonical = CommandEnvelope::new(
            command.actor,
            command.id.sequence,
            command.timestamp,
            command.causal,
            command.affected_domains,
            command.payload,
            self.config.command_limits,
        )
        .map_err(|_| RejectionReason::InvalidCommand)?;
        capability
            .check(&canonical)
            .map_err(RejectionReason::Capability)?;
        if let Some(existing) = self.by_id.get(&canonical.id) {
            if existing.command == canonical {
                return Ok((existing.clone(), false));
            }
            return Err(RejectionReason::InvalidCommand);
        }
        if self.accepted.len() >= self.config.transport.max_history_commands {
            return Err(RejectionReason::HistoryFull);
        }
        self.replica
            .receive(canonical.clone())
            .map_err(|_| RejectionReason::Replica)?;
        let entry = SequencedCommand {
            order: self.next_order,
            command: canonical,
        };
        self.next_order = self.next_order.saturating_add(1);
        self.by_id.insert(entry.command.id.clone(), entry.clone());
        self.accepted.push(entry.clone());
        Ok((entry, true))
    }

    fn broadcast(&mut self, entry: SequencedCommand) {
        let mut closed = Vec::new();
        for (session, sender) in &self.sessions {
            match sender.try_send(WireMessage::Command(entry.clone())) {
                Ok(()) => {}
                Err(TrySendError::Full(_)) | Err(TrySendError::Disconnected(_)) => {
                    closed.push(*session);
                }
            }
        }
        for session in closed {
            self.sessions.remove(&session);
        }
    }
}

/// A real TCP collaboration server.
pub struct TcpCollaborationServer {
    listener: TcpListener,
    state: Arc<Mutex<ServerState>>,
}

impl TcpCollaborationServer {
    /// Binds a server to a local or remote socket address.
    pub fn bind(address: SocketAddr, config: ServerConfig) -> Result<Self, ServerError> {
        config.validate().map_err(ServerError::Configuration)?;
        let listener = TcpListener::bind(address)?;
        let state = Arc::new(Mutex::new(ServerState::new(config)?));
        Ok(Self { listener, state })
    }

    /// Returns the actual listener address, including an ephemeral port.
    pub fn local_addr(&self) -> io::Result<SocketAddr> {
        self.listener.local_addr()
    }

    /// Starts accepting clients on a background thread.
    pub fn start(self) -> Result<TcpServerHandle, ServerError> {
        self.listener.set_nonblocking(true)?;
        let address = self.listener.local_addr()?;
        let state = Arc::clone(&self.state);
        let (stop_sender, stop_receiver) = mpsc::channel();
        let listener = self.listener;
        let thread = thread::Builder::new()
            .name("prism-collaboration-accept".to_owned())
            .spawn(move || accept_loop(listener, state, stop_receiver))
            .map_err(ServerError::Io)?;
        Ok(TcpServerHandle {
            address,
            stop_sender: Some(stop_sender),
            thread: Some(thread),
        })
    }
}

/// Handle used to inspect and stop a running TCP server.
pub struct TcpServerHandle {
    address: SocketAddr,
    stop_sender: Option<mpsc::Sender<()>>,
    thread: Option<JoinHandle<()>>,
}

impl TcpServerHandle {
    /// Returns the bound server address.
    #[must_use]
    pub const fn local_addr(&self) -> SocketAddr {
        self.address
    }

    /// Requests shutdown and waits for the accept loop to finish.
    pub fn shutdown(mut self) {
        if let Some(sender) = self.stop_sender.take() {
            let _ = sender.send(());
        }
        if let Some(thread) = self.thread.take() {
            let _ = thread.join();
        }
    }
}

impl Drop for TcpServerHandle {
    fn drop(&mut self) {
        if let Some(sender) = self.stop_sender.take() {
            let _ = sender.send(());
        }
    }
}

fn accept_loop(listener: TcpListener, state: Arc<Mutex<ServerState>>, stop_receiver: Receiver<()>) {
    loop {
        if stop_receiver.try_recv().is_ok() {
            return;
        }
        match listener.accept() {
            Ok((stream, _)) => {
                let state = Arc::clone(&state);
                let _ = thread::Builder::new()
                    .name("prism-collaboration-client".to_owned())
                    .spawn(move || serve_client(stream, state));
            }
            Err(error) if error.kind() == ErrorKind::WouldBlock => {
                thread::sleep(Duration::from_millis(2));
            }
            Err(_) => return,
        }
    }
}

fn serve_client(mut stream: TcpStream, state: Arc<Mutex<ServerState>>) {
    if stream.set_nonblocking(false).is_err() {
        return;
    }
    let (limits, command_limits) = match state.lock() {
        Ok(guard) => (guard.config.transport, guard.config.command_limits),
        Err(_) => return,
    };
    let hello = match read_message(&mut stream, limits, command_limits) {
        Ok(WireMessage::Hello { actor, last_order }) => (actor, last_order),
        Ok(_) | Err(_) => return,
    };
    let (sender, receiver) = mpsc::sync_channel(limits.max_client_queue);
    let session = match state.lock() {
        Ok(mut guard) => guard.register(&hello.0, hello.1, sender.clone()),
        Err(_) => Err(RejectionReason::Protocol),
    };
    let session = match session {
        Ok(session) => session,
        Err(reason) => {
            let _ = write_message(
                &mut stream,
                &WireMessage::Reject {
                    command: None,
                    reason,
                },
                limits,
            );
            return;
        }
    };
    let writer_stream = match stream.try_clone() {
        Ok(stream) => stream,
        Err(_) => {
            if let Ok(mut guard) = state.lock() {
                guard.remove_session(session);
            }
            return;
        }
    };
    let writer_state = Arc::clone(&state);
    let writer = thread::Builder::new()
        .name("prism-collaboration-writer".to_owned())
        .spawn(move || writer_loop(writer_stream, receiver, writer_state, session, limits));
    if writer.is_err() {
        if let Ok(mut guard) = state.lock() {
            guard.remove_session(session);
        }
        return;
    }
    while let Ok(WireMessage::Submit { command }) =
        read_message(&mut stream, limits, command_limits)
    {
        let result = match state.lock() {
            Ok(mut guard) => guard.submit(&hello.0, command.clone()),
            Err(_) => Err(RejectionReason::Protocol),
        };
        match result {
            Ok((entry, is_new)) => {
                if let Ok(mut guard) = state.lock() {
                    if is_new {
                        guard.broadcast(entry);
                    } else if sender.try_send(WireMessage::Command(entry)).is_err() {
                        guard.remove_session(session);
                        break;
                    }
                }
            }
            Err(reason) => {
                if sender
                    .try_send(WireMessage::Reject {
                        command: Some(command.id),
                        reason,
                    })
                    .is_err()
                {
                    break;
                }
            }
        }
    }
    if let Ok(mut guard) = state.lock() {
        guard.remove_session(session);
    }
}

fn writer_loop(
    mut stream: TcpStream,
    receiver: Receiver<WireMessage>,
    state: Arc<Mutex<ServerState>>,
    session: u64,
    limits: TransportLimits,
) {
    for message in receiver {
        if write_message(&mut stream, &message, limits).is_err() {
            if let Ok(mut guard) = state.lock() {
                guard.remove_session(session);
            }
            return;
        }
    }
}

#[derive(Clone, Debug)]
enum WireMessage {
    Hello {
        actor: ActorId,
        last_order: u64,
    },
    Welcome {
        current_order: u64,
        replay_count: u64,
    },
    Submit {
        command: CommandEnvelope,
    },
    Command(SequencedCommand),
    Reject {
        command: Option<CommandId>,
        reason: RejectionReason,
    },
}

/// A synchronous TCP client with a retained, bounded outbound queue.
pub struct TcpCollaborationClient {
    actor: ActorId,
    replica: Replica,
    queue_limit: usize,
    transport_limits: TransportLimits,
    command_limits: CommandLimits,
    address: Option<SocketAddr>,
    stream: Option<TcpStream>,
    read_buffer: Vec<u8>,
    pending: VecDeque<CommandEnvelope>,
    sent: BTreeSet<CommandId>,
    last_server_order: u64,
}

impl TcpCollaborationClient {
    /// Creates a disconnected client around an existing replica.
    pub fn new(
        actor: ActorId,
        replica: Replica,
        queue_limit: usize,
        command_limits: CommandLimits,
    ) -> Result<Self, ClientError> {
        if queue_limit == 0 {
            return Err(ClientError::Protocol(
                "queue limit must be positive".to_owned(),
            ));
        }
        Ok(Self {
            actor,
            replica,
            queue_limit,
            transport_limits: TransportLimits {
                max_client_queue: queue_limit,
                ..TransportLimits::default()
            },
            command_limits,
            address: None,
            stream: None,
            read_buffer: Vec::new(),
            pending: VecDeque::new(),
            sent: BTreeSet::new(),
            last_server_order: 0,
        })
    }

    /// Replaces the client wire bounds before connecting.
    pub fn with_transport_limits(mut self, limits: TransportLimits) -> Result<Self, ClientError> {
        limits
            .validate()
            .map_err(|error| ClientError::Protocol(error.to_string()))?;
        if limits.max_client_queue < self.queue_limit {
            return Err(ClientError::Protocol(
                "transport queue is smaller than client queue".to_owned(),
            ));
        }
        self.transport_limits = limits;
        Ok(self)
    }

    /// Returns whether the TCP connection is currently live.
    #[must_use]
    pub fn is_connected(&self) -> bool {
        self.stream.is_some()
    }

    /// Returns the local replica.
    #[must_use]
    pub fn replica(&self) -> &Replica {
        &self.replica
    }

    /// Returns the last contiguous server order applied locally.
    #[must_use]
    pub const fn last_server_order(&self) -> u64 {
        self.last_server_order
    }

    /// Returns the number of commands retained for delivery or retry.
    #[must_use]
    pub fn pending_len(&self) -> usize {
        self.pending.len()
    }

    /// Connects or reconnects and replays commands after the local cursor.
    pub fn connect(&mut self, address: SocketAddr) -> Result<Vec<ClientEvent>, ClientError> {
        self.disconnect();
        let mut stream = TcpStream::connect(address)?;
        stream.set_read_timeout(Some(Duration::from_secs(10)))?;
        stream.set_write_timeout(Some(Duration::from_secs(10)))?;
        write_message(
            &mut stream,
            &WireMessage::Hello {
                actor: self.actor.clone(),
                last_order: self.last_server_order,
            },
            self.transport_limits,
        )?;
        let limits = self.transport_limits;
        let (current_order, replay_count) =
            match read_message(&mut stream, limits, self.command_limits) {
                Ok(WireMessage::Welcome {
                    current_order,
                    replay_count,
                }) => (current_order, replay_count),
                Ok(WireMessage::Reject { reason, .. }) => {
                    return Err(ClientError::Handshake(reason))
                }
                Ok(_) => return Err(ClientError::Protocol("expected welcome".to_owned())),
                Err(error) => return Err(ClientError::Io(error)),
            };
        let mut events = Vec::new();
        for _ in 0..replay_count {
            let message = read_message(&mut stream, limits, self.command_limits)?;
            let WireMessage::Command(entry) = message else {
                return Err(ClientError::Protocol("expected replay command".to_owned()));
            };
            events.push(self.apply_server_command(entry)?);
        }
        if self.last_server_order > current_order {
            return Err(ClientError::ServerOrderGap {
                expected: self.last_server_order,
                observed: current_order,
            });
        }
        stream.set_nonblocking(true)?;
        self.address = Some(address);
        self.stream = Some(stream);
        self.flush_pending()?.into_iter().for_each(|_| {});
        Ok(events)
    }

    /// Closes the socket while retaining all pending commands for retry.
    pub fn disconnect(&mut self) {
        self.stream = None;
        self.read_buffer.clear();
        self.sent.clear();
    }

    /// Reconnects to the last successful address.
    pub fn reconnect(&mut self) -> Result<Vec<ClientEvent>, ClientError> {
        let address = self
            .address
            .ok_or_else(|| ClientError::Protocol("no previous server address".to_owned()))?;
        self.connect(address)
    }

    /// Queues a command and attempts immediate delivery when connected.
    pub fn submit(&mut self, command: CommandEnvelope) -> Result<(), ClientError> {
        let canonical = CommandEnvelope::new(
            command.actor,
            command.id.sequence,
            command.timestamp,
            command.causal,
            command.affected_domains,
            command.payload,
            self.command_limits,
        )
        .map_err(|error| ClientError::Protocol(error.to_string()))?;
        if canonical.actor != self.actor {
            return Err(ClientError::Protocol(
                "command actor does not match client".to_owned(),
            ));
        }
        if let Some(existing) = self.pending.iter().find(|item| item.id == canonical.id) {
            if existing == &canonical {
                return Ok(());
            }
            return Err(ClientError::ConflictingPending(canonical.id));
        }
        if self.pending.len() >= self.queue_limit {
            return Err(ClientError::QueueFull);
        }
        self.pending.push_back(canonical);
        self.flush_pending().map(|_| ())
    }

    /// Reads all currently available server messages without blocking.
    pub fn poll(&mut self) -> Result<Vec<ClientEvent>, ClientError> {
        let mut events = Vec::new();
        while let Some(message) = self.read_nonblocking_message()? {
            match message {
                WireMessage::Command(entry) => events.push(self.apply_server_command(entry)?),
                WireMessage::Reject { command, reason } => {
                    if let Some(id) = command.as_ref() {
                        self.pending.retain(|item| &item.id != id);
                        self.sent.remove(id);
                    }
                    events.push(ClientEvent::Rejected { command, reason });
                }
                _ => {
                    return Err(ClientError::Protocol(
                        "unexpected server message".to_owned(),
                    ))
                }
            }
        }
        Ok(events)
    }

    fn flush_pending(&mut self) -> Result<Vec<CommandId>, ClientError> {
        let Some(stream) = self.stream.as_mut() else {
            return Ok(Vec::new());
        };
        let mut sent = Vec::new();
        for command in &self.pending {
            if self.sent.contains(&command.id) {
                continue;
            }
            if write_message(
                stream,
                &WireMessage::Submit {
                    command: command.clone(),
                },
                self.transport_limits,
            )
            .is_err()
            {
                self.stream = None;
                self.sent.clear();
                return Err(ClientError::Disconnected);
            }
            self.sent.insert(command.id.clone());
            sent.push(command.id.clone());
        }
        Ok(sent)
    }

    fn apply_server_command(
        &mut self,
        entry: SequencedCommand,
    ) -> Result<ClientEvent, ClientError> {
        let expected = self.last_server_order.saturating_add(1);
        if entry.order != expected {
            if entry.order <= self.last_server_order {
                return Ok(ClientEvent::Command {
                    order: entry.order,
                    command: entry.command,
                    applied: Vec::new(),
                });
            }
            return Err(ClientError::ServerOrderGap {
                expected,
                observed: entry.order,
            });
        }
        let command = entry.command;
        let applied = self
            .replica
            .receive(command.clone())
            .map_err(ClientError::Replica)?;
        self.last_server_order = entry.order;
        self.pending.retain(|item| item.id != command.id);
        self.sent.remove(&command.id);
        let applied_ids = match applied {
            ReceiveOutcome::Applied(ids) => ids,
            ReceiveOutcome::Buffered(_) | ReceiveOutcome::Duplicate(_) => Vec::new(),
        };
        Ok(ClientEvent::Command {
            order: entry.order,
            command,
            applied: applied_ids,
        })
    }

    fn read_nonblocking_message(&mut self) -> Result<Option<WireMessage>, ClientError> {
        let Some(stream) = self.stream.as_mut() else {
            return Err(ClientError::Disconnected);
        };
        let mut chunk = [0_u8; 8192];
        loop {
            match stream.read(&mut chunk) {
                Ok(0) => {
                    self.stream = None;
                    return Err(ClientError::Disconnected);
                }
                Ok(length) => self.read_buffer.extend_from_slice(&chunk[..length]),
                Err(error) if error.kind() == ErrorKind::WouldBlock => break,
                Err(error) => {
                    self.stream = None;
                    return Err(ClientError::Io(error));
                }
            }
        }
        parse_buffer(
            &mut self.read_buffer,
            self.transport_limits,
            self.command_limits,
        )
        .map_err(ClientError::Protocol)
    }
}

fn write_message(
    stream: &mut TcpStream,
    message: &WireMessage,
    limits: TransportLimits,
) -> io::Result<()> {
    let body =
        encode_message(message).map_err(|error| io::Error::new(ErrorKind::InvalidData, error))?;
    if body.len() > limits.max_frame_bytes || body.len() > u32::MAX as usize {
        return Err(io::Error::new(
            ErrorKind::InvalidData,
            "frame exceeds limit",
        ));
    }
    stream.write_all(&(body.len() as u32).to_be_bytes())?;
    stream.write_all(&body)
}

fn read_message(
    stream: &mut TcpStream,
    limits: TransportLimits,
    command_limits: CommandLimits,
) -> io::Result<WireMessage> {
    let mut length = [0_u8; 4];
    stream.read_exact(&mut length)?;
    let length = u32::from_be_bytes(length) as usize;
    if length > limits.max_frame_bytes {
        return Err(io::Error::new(
            ErrorKind::InvalidData,
            "frame exceeds limit",
        ));
    }
    let mut body = vec![0_u8; length];
    stream.read_exact(&mut body)?;
    decode_message(&body, limits, command_limits)
        .map_err(|error| io::Error::new(ErrorKind::InvalidData, error))
}

fn parse_buffer(
    buffer: &mut Vec<u8>,
    limits: TransportLimits,
    command_limits: CommandLimits,
) -> Result<Option<WireMessage>, String> {
    if buffer.len() < 4 {
        return Ok(None);
    }
    let length = u32::from_be_bytes([buffer[0], buffer[1], buffer[2], buffer[3]]) as usize;
    if length > limits.max_frame_bytes {
        return Err("frame exceeds limit".to_owned());
    }
    if buffer.len() < length + 4 {
        return Ok(None);
    }
    let body = buffer[4..length + 4].to_vec();
    buffer.drain(..length + 4);
    decode_message(&body, limits, command_limits).map(Some)
}

fn encode_message(message: &WireMessage) -> Result<Vec<u8>, String> {
    let mut output = Vec::new();
    output.push(PROTOCOL_VERSION);
    match message {
        WireMessage::Hello { actor, last_order } => {
            output.push(MESSAGE_HELLO);
            put_string(&mut output, actor.as_str())?;
            put_u64(&mut output, *last_order);
        }
        WireMessage::Welcome {
            current_order,
            replay_count,
        } => {
            output.push(MESSAGE_WELCOME);
            put_u64(&mut output, *current_order);
            put_u64(&mut output, *replay_count);
        }
        WireMessage::Submit { command } => {
            output.push(MESSAGE_SUBMIT);
            encode_command(&mut output, command)?;
        }
        WireMessage::Command(entry) => {
            output.push(MESSAGE_COMMAND);
            put_u64(&mut output, entry.order);
            encode_command(&mut output, &entry.command)?;
        }
        WireMessage::Reject { command, reason } => {
            output.push(MESSAGE_REJECT);
            put_u8(&mut output, u8::from(command.is_some()));
            if let Some(command) = command {
                put_actor_id(&mut output, &command.actor)?;
                put_u64(&mut output, command.sequence);
            }
            encode_rejection(&mut output, reason)?;
        }
    }
    Ok(output)
}

fn decode_message(
    bytes: &[u8],
    _limits: TransportLimits,
    command_limits: CommandLimits,
) -> Result<WireMessage, String> {
    let mut cursor = Cursor::new(bytes);
    if cursor.u8()? != PROTOCOL_VERSION {
        return Err("unsupported protocol version".to_owned());
    }
    let message = match cursor.u8()? {
        MESSAGE_HELLO => WireMessage::Hello {
            actor: read_actor(&mut cursor)?,
            last_order: cursor.u64()?,
        },
        MESSAGE_WELCOME => WireMessage::Welcome {
            current_order: cursor.u64()?,
            replay_count: cursor.u64()?,
        },
        MESSAGE_SUBMIT => WireMessage::Submit {
            command: decode_command(&mut cursor, command_limits)?,
        },
        MESSAGE_COMMAND => WireMessage::Command(SequencedCommand {
            order: cursor.u64()?,
            command: decode_command(&mut cursor, command_limits)?,
        }),
        MESSAGE_REJECT => {
            let command = if cursor.u8()? == 0 {
                None
            } else {
                Some(CommandId {
                    actor: read_actor(&mut cursor)?,
                    sequence: cursor.u64()?,
                })
            };
            WireMessage::Reject {
                command,
                reason: decode_rejection(&mut cursor)?,
            }
        }
        _ => return Err("unknown message type".to_owned()),
    };
    if cursor.remaining() != 0 {
        return Err("trailing frame bytes".to_owned());
    }
    Ok(message)
}

fn encode_command(output: &mut Vec<u8>, command: &CommandEnvelope) -> Result<(), String> {
    put_actor_id(output, &command.actor)?;
    put_u64(output, command.id.sequence);
    put_u64(output, command.timestamp.physical);
    put_u32(output, command.timestamp.logical);
    put_u32(output, command.causal.len() as u32);
    for dependency in &command.causal {
        put_actor_id(output, &dependency.actor)?;
        put_u64(output, dependency.sequence);
    }
    put_u32(output, command.affected_domains.len() as u32);
    for domain in &command.affected_domains {
        put_string(output, domain.as_str())?;
    }
    match &command.payload {
        CommandPayload::SetField { path, value } => {
            put_u8(output, 1);
            encode_path(output, path)?;
            encode_value(output, value)?;
        }
        CommandPayload::DeleteField { path } => {
            put_u8(output, 2);
            encode_path(output, path)?;
        }
        CommandPayload::Opaque(bytes) => {
            put_u8(output, 3);
            put_bytes(output, bytes)?;
        }
    }
    Ok(())
}

fn decode_command(
    cursor: &mut Cursor<'_>,
    limits: CommandLimits,
) -> Result<CommandEnvelope, String> {
    let actor = read_actor(cursor)?;
    let sequence = cursor.u64()?;
    let timestamp = crate::clock::HlcTimestamp::new(cursor.u64()?, cursor.u32()?);
    let causal_len = cursor.count(limits.max_causal_dependencies)?;
    let mut causal = Vec::with_capacity(causal_len);
    for _ in 0..causal_len {
        causal.push(CommandId {
            actor: read_actor(cursor)?,
            sequence: cursor.u64()?,
        });
    }
    let domain_len = cursor.count(limits.max_domains)?;
    let mut domains = BTreeSet::new();
    for _ in 0..domain_len {
        let domain = DomainId::new(cursor.string()?).map_err(|_| "invalid domain".to_owned())?;
        if !domains.insert(domain) {
            return Err("duplicate domain".to_owned());
        }
    }
    let payload = match cursor.u8()? {
        1 => CommandPayload::SetField {
            path: decode_path(cursor)?,
            value: decode_value(cursor, limits.max_payload_bytes)?,
        },
        2 => CommandPayload::DeleteField {
            path: decode_path(cursor)?,
        },
        3 => CommandPayload::Opaque(cursor.bytes(limits.max_payload_bytes)?),
        _ => return Err("unknown command payload".to_owned()),
    };
    CommandEnvelope::new(actor, sequence, timestamp, causal, domains, payload, limits)
        .map_err(|error| error.to_string())
}

fn encode_path(output: &mut Vec<u8>, path: &FieldPath) -> Result<(), String> {
    put_u32(output, path.segments().len() as u32);
    for segment in path.segments() {
        match segment {
            PathSegment::Field(value) => {
                put_u8(output, 1);
                put_string(output, value)?;
            }
            PathSegment::Key(value) => {
                put_u8(output, 2);
                put_string(output, value)?;
            }
            PathSegment::Index(value) => {
                put_u8(output, 3);
                put_u32(output, *value);
            }
        }
    }
    Ok(())
}

fn decode_path(cursor: &mut Cursor<'_>) -> Result<FieldPath, String> {
    let count = cursor.count(cursor.remaining().min(1024))?;
    let mut segments = Vec::with_capacity(count);
    for _ in 0..count {
        segments.push(match cursor.u8()? {
            1 => PathSegment::Field(cursor.string()?),
            2 => PathSegment::Key(cursor.string()?),
            3 => PathSegment::Index(cursor.u32()?),
            _ => return Err("unknown path segment".to_owned()),
        });
    }
    FieldPath::new(segments).map_err(|error| error.to_string())
}

fn encode_value(output: &mut Vec<u8>, value: &Value) -> Result<(), String> {
    match value {
        Value::Null => put_u8(output, 0),
        Value::Bool(value) => {
            put_u8(output, 1);
            put_u8(output, u8::from(*value));
        }
        Value::Int(value) => {
            put_u8(output, 2);
            output.extend_from_slice(&value.to_be_bytes());
        }
        Value::UInt(value) => {
            put_u8(output, 3);
            put_u64(output, *value);
        }
        Value::FloatBits(value) => {
            put_u8(output, 4);
            put_u64(output, *value);
        }
        Value::Text(value) => {
            put_u8(output, 5);
            put_string(output, value)?;
        }
        Value::Bytes(value) => {
            put_u8(output, 6);
            put_bytes(output, value)?;
        }
    }
    Ok(())
}

fn decode_value(cursor: &mut Cursor<'_>, payload_limit: usize) -> Result<Value, String> {
    match cursor.u8()? {
        0 => Ok(Value::Null),
        1 => Ok(Value::Bool(cursor.u8()? != 0)),
        2 => Ok(Value::Int(i64::from_be_bytes(cursor.array8()?))),
        3 => Ok(Value::UInt(cursor.u64()?)),
        4 => Ok(Value::FloatBits(cursor.u64()?)),
        5 => Ok(Value::Text(cursor.string()?)),
        6 => Ok(Value::Bytes(cursor.bytes(payload_limit)?)),
        _ => Err("unknown value".to_owned()),
    }
}

fn encode_rejection(output: &mut Vec<u8>, reason: &RejectionReason) -> Result<(), String> {
    let code = match reason {
        RejectionReason::Unauthorized => 1,
        RejectionReason::Capability(_) => 2,
        RejectionReason::InvalidCommand => 3,
        RejectionReason::Replica => 4,
        RejectionReason::HistoryFull => 5,
        RejectionReason::Protocol => 6,
    };
    put_u8(output, code);
    if let RejectionReason::Capability(error) = reason {
        match error {
            CapabilityError::UnknownActor => put_u8(output, 0),
            CapabilityError::DomainDenied(domain) => {
                put_u8(output, 1);
                put_string(output, domain.as_str())?;
            }
            CapabilityError::OperationDenied(kind) => {
                put_u8(output, 2);
                put_u8(output, command_kind_code(*kind));
            }
            CapabilityError::PayloadTooLarge { actual, limit } => {
                put_u8(output, 3);
                put_u64(output, *actual as u64);
                put_u64(output, *limit as u64);
            }
        }
    }
    Ok(())
}

fn decode_rejection(cursor: &mut Cursor<'_>) -> Result<RejectionReason, String> {
    match cursor.u8()? {
        1 => Ok(RejectionReason::Unauthorized),
        2 => match cursor.u8()? {
            0 => Ok(RejectionReason::Capability(CapabilityError::UnknownActor)),
            1 => Ok(RejectionReason::Capability(CapabilityError::DomainDenied(
                DomainId::new(cursor.string()?).map_err(|_| "invalid domain".to_owned())?,
            ))),
            2 => Ok(RejectionReason::Capability(
                CapabilityError::OperationDenied(decode_command_kind(cursor.u8()?)?),
            )),
            3 => Ok(RejectionReason::Capability(
                CapabilityError::PayloadTooLarge {
                    actual: cursor.u64()? as usize,
                    limit: cursor.u64()? as usize,
                },
            )),
            _ => Err("unknown capability rejection".to_owned()),
        },
        3 => Ok(RejectionReason::InvalidCommand),
        4 => Ok(RejectionReason::Replica),
        5 => Ok(RejectionReason::HistoryFull),
        6 => Ok(RejectionReason::Protocol),
        _ => Err("unknown rejection".to_owned()),
    }
}

fn command_kind_code(kind: CommandKind) -> u8 {
    match kind {
        CommandKind::SetField => 1,
        CommandKind::DeleteField => 2,
        CommandKind::Opaque => 3,
    }
}

fn decode_command_kind(code: u8) -> Result<CommandKind, String> {
    match code {
        1 => Ok(CommandKind::SetField),
        2 => Ok(CommandKind::DeleteField),
        3 => Ok(CommandKind::Opaque),
        _ => Err("unknown command kind".to_owned()),
    }
}

fn put_actor_id(output: &mut Vec<u8>, actor: &ActorId) -> Result<(), String> {
    put_string(output, actor.as_str())
}

fn read_actor(cursor: &mut Cursor<'_>) -> Result<ActorId, String> {
    ActorId::new(cursor.string()?).map_err(|_| "invalid actor".to_owned())
}

fn put_string(output: &mut Vec<u8>, value: &str) -> Result<(), String> {
    if value.len() > MAX_STRING_BYTES || value.len() > u32::MAX as usize {
        return Err("string exceeds limit".to_owned());
    }
    put_u32(output, value.len() as u32);
    output.extend_from_slice(value.as_bytes());
    Ok(())
}

fn put_bytes(output: &mut Vec<u8>, value: &[u8]) -> Result<(), String> {
    if value.len() > u32::MAX as usize {
        return Err("byte vector exceeds limit".to_owned());
    }
    put_u32(output, value.len() as u32);
    output.extend_from_slice(value);
    Ok(())
}

fn put_u8(output: &mut Vec<u8>, value: u8) {
    output.push(value);
}

fn put_u32(output: &mut Vec<u8>, value: u32) {
    output.extend_from_slice(&value.to_be_bytes());
}

fn put_u64(output: &mut Vec<u8>, value: u64) {
    output.extend_from_slice(&value.to_be_bytes());
}

struct Cursor<'a> {
    bytes: &'a [u8],
    position: usize,
}

impl<'a> Cursor<'a> {
    fn new(bytes: &'a [u8]) -> Self {
        Self { bytes, position: 0 }
    }

    fn remaining(&self) -> usize {
        self.bytes.len().saturating_sub(self.position)
    }

    fn take(&mut self, length: usize) -> Result<&'a [u8], String> {
        let end = self
            .position
            .checked_add(length)
            .ok_or_else(|| "cursor overflow".to_owned())?;
        if end > self.bytes.len() {
            return Err("truncated frame".to_owned());
        }
        let result = &self.bytes[self.position..end];
        self.position = end;
        Ok(result)
    }

    fn u8(&mut self) -> Result<u8, String> {
        Ok(self.take(1)?[0])
    }

    fn u32(&mut self) -> Result<u32, String> {
        Ok(u32::from_be_bytes(
            self.take(4)?.try_into().expect("four bytes"),
        ))
    }

    fn u64(&mut self) -> Result<u64, String> {
        Ok(u64::from_be_bytes(
            self.take(8)?.try_into().expect("eight bytes"),
        ))
    }

    fn array8(&mut self) -> Result<[u8; 8], String> {
        self.take(8)?
            .try_into()
            .map_err(|_| "invalid array".to_owned())
    }

    fn count(&mut self, limit: usize) -> Result<usize, String> {
        let count = self.u32()? as usize;
        if count > limit {
            return Err("collection exceeds limit".to_owned());
        }
        Ok(count)
    }

    fn string(&mut self) -> Result<String, String> {
        let length = self.count(MAX_STRING_BYTES)?;
        String::from_utf8(self.take(length)?.to_vec()).map_err(|_| "invalid UTF-8".to_owned())
    }

    fn bytes(&mut self, limit: usize) -> Result<Vec<u8>, String> {
        let length = self.count(limit)?;
        Ok(self.take(length)?.to_vec())
    }
}
