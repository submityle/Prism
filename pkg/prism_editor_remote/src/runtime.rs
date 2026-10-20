//! Controlled runtime process lifecycle and engine-neutral runtime bridges.
//!
//! The process layer owns only process supervision and the bounded JSON-RPC
//! connection. An engine adapter supplies a launch transformation and an
//! optional startup handshake; it does not need to expose engine or ECS types
//! to the editor. Both stdio and TCP use the same length-prefixed, bounded
//! JSON-RPC client.

use crate::{
    connect_tcp, JsonRpcClient, JsonRpcExchange, JsonRpcId, JsonRpcMessage, JsonRpcRequest,
    TransportError, TransportLimits,
};
use std::collections::BTreeMap;
use std::ffi::OsString;
use std::fmt;
use std::io::{self, Read, Write};
use std::net::{SocketAddr, TcpStream};
use std::path::PathBuf;
use std::process::{Child, ChildStdin, ChildStdout, Command, Stdio};
use std::sync::atomic::{AtomicBool, AtomicU8, AtomicUsize, Ordering};
use std::sync::mpsc::{self, Receiver, RecvTimeoutError, SyncSender, TrySendError};
use std::sync::{Arc, Mutex};
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};

const CANCEL_NONE: u8 = 0;
const CANCEL_REQUESTED: u8 = 1;
const CANCEL_TIMED_OUT: u8 = 2;

/// The wire connection used by a supervised runtime.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum RuntimeTransport {
    /// Connect the child process through its stdin and stdout pipes.
    Stdio,
    /// Connect to a TCP endpoint after the child process has started.
    Tcp {
        /// Address where the child runtime accepts the JSON-RPC connection.
        address: SocketAddr,
    },
}

/// Engine-neutral command and connection description for one runtime.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct RuntimeLaunchSpec {
    /// Executable to spawn.
    pub program: PathBuf,
    /// Arguments passed to the executable in order.
    pub args: Vec<OsString>,
    /// Additional environment variables. Existing variables remain inherited.
    pub env: BTreeMap<OsString, OsString>,
    /// Optional working directory for the child process.
    pub current_dir: Option<PathBuf>,
    /// Connection used after the process is spawned.
    pub transport: RuntimeTransport,
}

impl RuntimeLaunchSpec {
    /// Creates a launch specification with no arguments or environment changes.
    #[must_use]
    pub fn new(program: impl Into<PathBuf>, transport: RuntimeTransport) -> Self {
        Self {
            program: program.into(),
            args: Vec::new(),
            env: BTreeMap::new(),
            current_dir: None,
            transport,
        }
    }
}

/// Resource, framing, and lifecycle bounds for a supervised runtime.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct RuntimeLimits {
    /// Limits for the JSON-RPC length-prefixed connection.
    pub transport: TransportLimits,
    /// Maximum time allowed for an adapter startup handshake.
    pub startup_timeout: Duration,
    /// Maximum time allowed for process termination after a stop request.
    pub shutdown_timeout: Duration,
    /// Maximum unfinished requests, including the request executing on the worker.
    pub max_pending_requests: usize,
}

impl Default for RuntimeLimits {
    fn default() -> Self {
        Self {
            transport: TransportLimits::default(),
            startup_timeout: Duration::from_secs(10),
            shutdown_timeout: Duration::from_secs(5),
            max_pending_requests: 8,
        }
    }
}

/// Lifecycle state observed by the editor.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum RuntimeState {
    /// No process has been started by this bridge.
    Created,
    /// A child is being connected and initialized.
    Starting,
    /// The child and its bounded JSON-RPC connection are usable.
    Running,
    /// A stop operation has been requested and termination is in progress.
    Stopping,
    /// The child was intentionally stopped by the bridge.
    Stopped,
    /// The child exited without a bridge stop operation.
    Exited,
    /// The connection or lifecycle operation failed.
    Failed,
}

/// Lifecycle operation used in diagnostics and timeout errors.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum RuntimeOperation {
    /// Starting and initializing a runtime.
    Start,
    /// Sending a JSON-RPC request.
    Request,
    /// Stopping a runtime.
    Stop,
    /// Waiting for a runtime to exit.
    Wait,
    /// Restarting a runtime.
    Restart,
}

/// A point-in-time runtime status snapshot.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct RuntimeStatus {
    /// Current lifecycle state.
    pub state: RuntimeState,
    /// Operating-system process ID, while known.
    pub process_id: Option<u32>,
    /// Child exit code once the process has been reaped.
    pub exit_code: Option<i32>,
    /// Monotonic bridge generation, incremented for each start.
    pub generation: u64,
    /// Last connection or lifecycle diagnostic, if one exists.
    pub last_error: Option<String>,
}

impl RuntimeStatus {
    fn created() -> Self {
        Self {
            state: RuntimeState::Created,
            process_id: None,
            exit_code: None,
            generation: 0,
            last_error: None,
        }
    }
}

/// Errors raised by runtime supervision or the bounded connection.
#[derive(Debug)]
pub enum RuntimeError {
    /// The requested configuration or lifecycle transition is invalid.
    InvalidConfiguration(String),
    /// The operation is not valid for the current lifecycle state.
    InvalidState {
        /// Operation that was attempted.
        operation: RuntimeOperation,
        /// State observed before the operation.
        state: RuntimeState,
    },
    /// An operating-system process or pipe operation failed.
    Io(io::Error),
    /// The bounded JSON-RPC transport rejected or could not complete a frame.
    Transport(TransportError),
    /// An adapter rejected launch or initialization.
    Adapter(String),
    /// An operation did not finish before its deadline.
    Timeout {
        /// Operation that reached its deadline.
        operation: RuntimeOperation,
        /// Configured deadline.
        timeout: Duration,
    },
    /// A request was explicitly cancelled.
    Cancelled {
        /// Operation that was cancelled.
        operation: RuntimeOperation,
    },
    /// The bounded request queue has no capacity.
    RequestQueueFull,
    /// The runtime worker ended before a result could be returned.
    WorkerClosed,
    /// A worker thread could not be joined after process termination.
    WorkerJoin,
}

impl fmt::Display for RuntimeError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidConfiguration(detail) => {
                write!(formatter, "invalid runtime configuration: {detail}")
            }
            Self::InvalidState { operation, state } => {
                write!(
                    formatter,
                    "cannot perform {operation:?} while runtime is {state:?}"
                )
            }
            Self::Io(error) => write!(formatter, "runtime process I/O error: {error}"),
            Self::Transport(error) => write!(formatter, "runtime transport error: {error}"),
            Self::Adapter(detail) => write!(formatter, "runtime adapter error: {detail}"),
            Self::Timeout { operation, timeout } => {
                write!(
                    formatter,
                    "runtime {operation:?} timed out after {timeout:?}"
                )
            }
            Self::Cancelled { operation } => {
                write!(formatter, "runtime {operation:?} was cancelled")
            }
            Self::RequestQueueFull => formatter.write_str("runtime request queue is full"),
            Self::WorkerClosed => formatter.write_str("runtime connection worker closed"),
            Self::WorkerJoin => {
                formatter.write_str("runtime connection worker could not be joined")
            }
        }
    }
}

impl std::error::Error for RuntimeError {}

impl From<io::Error> for RuntimeError {
    fn from(error: io::Error) -> Self {
        Self::Io(error)
    }
}

impl From<TransportError> for RuntimeError {
    fn from(error: TransportError) -> Self {
        Self::Transport(error)
    }
}

/// A cancellable request submitted to a running runtime.
pub struct RuntimeRequest {
    response: Receiver<Result<JsonRpcExchange, RuntimeError>>,
    control: Arc<RequestControl>,
}

impl RuntimeRequest {
    /// Waits without an additional deadline.
    pub fn wait(self) -> Result<JsonRpcExchange, RuntimeError> {
        self.response
            .recv()
            .map_err(|_| RuntimeError::WorkerClosed)?
    }

    /// Waits up to `timeout`; a timeout terminates the controlled runtime.
    pub fn wait_timeout(self, timeout: Duration) -> Result<JsonRpcExchange, RuntimeError> {
        match self.response.recv_timeout(timeout) {
            Ok(result) => result,
            Err(RecvTimeoutError::Timeout) => {
                self.control.cancel(CANCEL_TIMED_OUT)?;
                Err(RuntimeError::Timeout {
                    operation: RuntimeOperation::Request,
                    timeout,
                })
            }
            Err(RecvTimeoutError::Disconnected) => Err(RuntimeError::WorkerClosed),
        }
    }

    /// Cancels the request and terminates its controlled runtime process.
    pub fn cancel(&self) -> Result<(), RuntimeError> {
        self.control.cancel(CANCEL_REQUESTED)
    }
}

/// Session boundary exposed to engine adapters during startup.
pub trait RuntimeSession {
    /// Sends one request and waits for its correlated response.
    fn request(
        &mut self,
        message: JsonRpcMessage,
        timeout: Duration,
    ) -> Result<JsonRpcExchange, RuntimeError>;
}

/// Engine-specific launch and initialization boundary.
pub trait EngineRuntimeAdapter: Send + Sync {
    /// Stable engine identifier used in diagnostics and telemetry.
    fn engine_id(&self) -> &str;

    /// Converts the generic launch request into an engine-specific process
    /// and connection description.
    fn prepare_launch(
        &self,
        requested: &RuntimeLaunchSpec,
    ) -> Result<RuntimeLaunchSpec, RuntimeError>;

    /// Performs an optional protocol handshake after the connection is ready.
    fn initialize(
        &self,
        session: &mut dyn RuntimeSession,
        timeout: Duration,
    ) -> Result<(), RuntimeError>;
}

/// A no-engine-dependency adapter for local loopback runtimes and tests.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct LoopbackRuntimeAdapter {
    engine_id: String,
}

impl LoopbackRuntimeAdapter {
    /// Creates an adapter whose startup handshake uses `rpc.discover`.
    #[must_use]
    pub fn new() -> Self {
        Self {
            engine_id: String::from("loopback"),
        }
    }

    /// Creates a loopback adapter with a caller-visible engine identifier.
    #[must_use]
    pub fn with_engine_id(engine_id: impl Into<String>) -> Self {
        Self {
            engine_id: engine_id.into(),
        }
    }
}

impl Default for LoopbackRuntimeAdapter {
    fn default() -> Self {
        Self::new()
    }
}

impl EngineRuntimeAdapter for LoopbackRuntimeAdapter {
    fn engine_id(&self) -> &str {
        &self.engine_id
    }

    fn prepare_launch(
        &self,
        requested: &RuntimeLaunchSpec,
    ) -> Result<RuntimeLaunchSpec, RuntimeError> {
        Ok(requested.clone())
    }

    fn initialize(
        &self,
        session: &mut dyn RuntimeSession,
        timeout: Duration,
    ) -> Result<(), RuntimeError> {
        let exchange = session.request(
            JsonRpcMessage::Request(JsonRpcRequest {
                id: JsonRpcId::Number(1),
                method: String::from("rpc.discover"),
                params: None,
            }),
            timeout,
        )?;
        if let Some(error) = exchange.response.error {
            return Err(RuntimeError::Adapter(format!(
                "loopback discovery failed with code {}: {}",
                error.code, error.message
            )));
        }
        Ok(())
    }
}

/// Supervises one engine adapter and its child runtime process.
pub struct RuntimeBridge<A> {
    adapter: A,
    requested: RuntimeLaunchSpec,
    limits: RuntimeLimits,
    process: Option<RuntimeProcess>,
    last_status: RuntimeStatus,
}

impl<A: EngineRuntimeAdapter> RuntimeBridge<A> {
    /// Creates a stopped bridge. The process is not spawned until [`Self::start`].
    #[must_use]
    pub fn new(adapter: A, requested: RuntimeLaunchSpec, limits: RuntimeLimits) -> Self {
        Self {
            adapter,
            requested,
            limits,
            process: None,
            last_status: RuntimeStatus::created(),
        }
    }

    /// Returns the engine-specific adapter.
    #[must_use]
    pub const fn adapter(&self) -> &A {
        &self.adapter
    }

    /// Returns the bridge limits.
    #[must_use]
    pub const fn limits(&self) -> RuntimeLimits {
        self.limits
    }

    /// Starts, connects, and initializes the runtime process.
    pub fn start(&mut self) -> Result<RuntimeStatus, RuntimeError> {
        if let Some(process) = &mut self.process {
            let status = process.status();
            if matches!(
                status.state,
                RuntimeState::Starting | RuntimeState::Running | RuntimeState::Stopping
            ) {
                return Err(RuntimeError::InvalidState {
                    operation: RuntimeOperation::Start,
                    state: status.state,
                });
            }
            self.last_status = status;
            self.process = None;
        }
        let launch = self.adapter.prepare_launch(&self.requested)?;
        let generation = self.last_status.generation.saturating_add(1);
        let mut process = RuntimeProcess::spawn(launch, self.limits, generation)?;
        if let Err(error) = self
            .adapter
            .initialize(&mut process, self.limits.startup_timeout)
        {
            let _ = process.stop(self.limits.shutdown_timeout);
            self.last_status = process.status();
            self.last_status.state = RuntimeState::Failed;
            self.last_status.last_error = Some(error.to_string());
            return Err(error);
        }
        let status = process.status();
        self.last_status = status.clone();
        self.process = Some(process);
        Ok(status)
    }

    /// Stops the runtime process and returns its final status.
    pub fn stop(&mut self) -> Result<RuntimeStatus, RuntimeError> {
        let Some(mut process) = self.process.take() else {
            return Err(RuntimeError::InvalidState {
                operation: RuntimeOperation::Stop,
                state: self.last_status.state,
            });
        };
        let result = process.stop(self.limits.shutdown_timeout);
        match result {
            Ok(status) => {
                self.last_status = status.clone();
                Ok(status)
            }
            Err(error) => {
                self.last_status = process.status();
                Err(error)
            }
        }
    }

    /// Stops the current process, then starts a fresh adapter generation.
    pub fn restart(&mut self) -> Result<RuntimeStatus, RuntimeError> {
        if self.process.is_some() {
            self.stop().map_err(|error| match error {
                RuntimeError::Timeout { timeout, .. } => RuntimeError::Timeout {
                    operation: RuntimeOperation::Restart,
                    timeout,
                },
                other => other,
            })?;
        }
        self.start()
    }

    /// Returns a status snapshot and reaps an exited child when possible.
    pub fn status(&mut self) -> RuntimeStatus {
        if let Some(process) = &mut self.process {
            self.last_status = process.status();
        }
        self.last_status.clone()
    }

    /// Waits for the child to exit without killing it.
    pub fn wait(&mut self, timeout: Duration) -> Result<RuntimeStatus, RuntimeError> {
        let deadline = Instant::now() + timeout;
        loop {
            let status = self.status();
            if matches!(
                status.state,
                RuntimeState::Exited | RuntimeState::Stopped | RuntimeState::Failed
            ) {
                return Ok(status);
            }
            if Instant::now() >= deadline {
                return Err(RuntimeError::Timeout {
                    operation: RuntimeOperation::Wait,
                    timeout,
                });
            }
            thread::sleep(Duration::from_millis(2));
        }
    }

    /// Queues one request for the runtime connection.
    pub fn request(&mut self, message: JsonRpcMessage) -> Result<RuntimeRequest, RuntimeError> {
        let process = self.process.as_mut().ok_or(RuntimeError::InvalidState {
            operation: RuntimeOperation::Request,
            state: self.last_status.state,
        })?;
        process.request(message)
    }
}

impl<A> Drop for RuntimeBridge<A> {
    fn drop(&mut self) {
        if let Some(process) = &mut self.process {
            let _ = process.stop(self.limits.shutdown_timeout);
        }
    }
}

impl RuntimeSession for RuntimeProcess {
    fn request(
        &mut self,
        message: JsonRpcMessage,
        timeout: Duration,
    ) -> Result<JsonRpcExchange, RuntimeError> {
        self.request(message)?.wait_timeout(timeout)
    }
}

struct RuntimeProcess {
    control: Arc<ProcessControl>,
    sender: SyncSender<WorkerCommand>,
    worker: Option<JoinHandle<()>>,
    limits: RuntimeLimits,
    status: RuntimeStatus,
    connection_alive: Arc<AtomicBool>,
    pending_requests: Arc<AtomicUsize>,
}

impl RuntimeProcess {
    fn spawn(
        launch: RuntimeLaunchSpec,
        limits: RuntimeLimits,
        generation: u64,
    ) -> Result<Self, RuntimeError> {
        if launch.program.as_os_str().is_empty() {
            return Err(RuntimeError::InvalidConfiguration(String::from(
                "runtime executable must not be empty",
            )));
        }
        if limits.max_pending_requests == 0 {
            return Err(RuntimeError::InvalidConfiguration(String::from(
                "max_pending_requests must be greater than zero",
            )));
        }

        let mut command = Command::new(&launch.program);
        command
            .args(&launch.args)
            .envs(&launch.env)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::null());
        if let Some(current_dir) = &launch.current_dir {
            command.current_dir(current_dir);
        }
        let mut child = command.spawn()?;
        let process_id = Some(child.id());
        let connection = match launch.transport {
            RuntimeTransport::Stdio => {
                let stdin = child.stdin.take().ok_or_else(|| {
                    RuntimeError::InvalidConfiguration(String::from(
                        "spawned runtime did not provide stdin",
                    ))
                })?;
                let stdout = child.stdout.take().ok_or_else(|| {
                    RuntimeError::InvalidConfiguration(String::from(
                        "spawned runtime did not provide stdout",
                    ))
                })?;
                RuntimeIo::Stdio(StdioIo {
                    reader: stdout,
                    writer: stdin,
                })
            }
            RuntimeTransport::Tcp { address } => {
                child.stdin.take();
                child.stdout.take();
                let mut transport = limits.transport;
                if transport.connect_timeout.is_none() {
                    transport.connect_timeout = Some(limits.startup_timeout);
                }
                match connect_tcp_retry(address, transport, limits.startup_timeout) {
                    Ok(client) => RuntimeIo::Tcp(client.into_inner()),
                    Err(error) => {
                        terminate_child(&mut child);
                        return Err(RuntimeError::Transport(error));
                    }
                }
            }
        };

        let control = Arc::new(ProcessControl {
            child: Mutex::new(child),
            failure: Mutex::new(None),
        });
        let connection_alive = Arc::new(AtomicBool::new(true));
        let (sender, receiver) = mpsc::sync_channel(limits.max_pending_requests);
        let worker_control = Arc::clone(&control);
        let worker_alive = Arc::clone(&connection_alive);
        let worker = thread::Builder::new()
            .name(format!("prism-runtime-{generation}"))
            .spawn(move || {
                worker_loop(
                    connection,
                    receiver,
                    worker_control,
                    worker_alive,
                    limits.transport,
                );
            })
            .map_err(RuntimeError::Io)?;

        Ok(Self {
            control,
            sender,
            worker: Some(worker),
            limits,
            status: RuntimeStatus {
                state: RuntimeState::Running,
                process_id,
                exit_code: None,
                generation,
                last_error: None,
            },
            connection_alive,
            pending_requests: Arc::new(AtomicUsize::new(0)),
        })
    }

    fn request(&mut self, message: JsonRpcMessage) -> Result<RuntimeRequest, RuntimeError> {
        let status = self.status();
        if status.state != RuntimeState::Running {
            return Err(RuntimeError::InvalidState {
                operation: RuntimeOperation::Request,
                state: status.state,
            });
        }
        let permit =
            RequestPermit::acquire(&self.pending_requests, self.limits.max_pending_requests)?;
        let (response_sender, response) = mpsc::channel();
        let control = Arc::new(RequestControl {
            process: Arc::clone(&self.control),
            reason: AtomicU8::new(CANCEL_NONE),
        });
        let command = WorkerCommand::Request {
            message,
            control: Arc::clone(&control),
            response: response_sender,
            permit,
        };
        match self.sender.try_send(command) {
            Ok(()) => Ok(RuntimeRequest { response, control }),
            Err(TrySendError::Full(_)) => Err(RuntimeError::RequestQueueFull),
            Err(TrySendError::Disconnected(_)) => Err(RuntimeError::WorkerClosed),
        }
    }

    fn status(&mut self) -> RuntimeStatus {
        if let Ok(mut child) = self.control.child.lock() {
            match child.try_wait() {
                Ok(Some(exit)) => {
                    self.status.state = RuntimeState::Exited;
                    self.status.exit_code = exit.code();
                    self.connection_alive.store(false, Ordering::Release);
                }
                Ok(None) => {
                    if !self.connection_alive.load(Ordering::Acquire)
                        && self.status.state == RuntimeState::Running
                    {
                        self.status.state = RuntimeState::Failed;
                        self.status.last_error = self
                            .control
                            .failure
                            .lock()
                            .ok()
                            .and_then(|failure| failure.clone());
                    }
                }
                Err(error) => {
                    self.status.state = RuntimeState::Failed;
                    self.status.last_error = Some(error.to_string());
                }
            }
        } else {
            self.status.state = RuntimeState::Failed;
            self.status.last_error = Some(String::from("runtime process lock was poisoned"));
        }
        self.status.clone()
    }

    fn stop(&mut self, timeout: Duration) -> Result<RuntimeStatus, RuntimeError> {
        let current = self.status();
        if current.state == RuntimeState::Stopped {
            return Ok(current);
        }
        self.status.state = RuntimeState::Stopping;
        terminate_control(&self.control)?;
        let deadline = Instant::now() + timeout;
        loop {
            let status = self.status();
            if status.state == RuntimeState::Exited {
                break;
            }
            if Instant::now() >= deadline {
                self.status.state = RuntimeState::Stopping;
                return Err(RuntimeError::Timeout {
                    operation: RuntimeOperation::Stop,
                    timeout,
                });
            }
            thread::sleep(Duration::from_millis(2));
        }
        let _ = self.sender.try_send(WorkerCommand::Shutdown);
        if let Some(worker) = self.worker.take() {
            worker.join().map_err(|_| RuntimeError::WorkerJoin)?;
        }
        self.status.state = RuntimeState::Stopped;
        Ok(self.status.clone())
    }
}

impl Drop for RuntimeProcess {
    fn drop(&mut self) {
        let _ = self.stop(self.limits.shutdown_timeout);
    }
}

struct ProcessControl {
    child: Mutex<Child>,
    failure: Mutex<Option<String>>,
}

impl ProcessControl {
    fn terminate(&self) -> Result<(), RuntimeError> {
        let mut child = self
            .child
            .lock()
            .map_err(|_| RuntimeError::Io(io::Error::other("runtime process lock poisoned")))?;
        match child.kill() {
            Ok(()) => Ok(()),
            Err(error) if error.kind() == io::ErrorKind::NotFound => Ok(()),
            Err(error) => Err(RuntimeError::Io(error)),
        }
    }
}

fn terminate_control(control: &ProcessControl) -> Result<(), RuntimeError> {
    control.terminate()
}

struct RequestControl {
    process: Arc<ProcessControl>,
    reason: AtomicU8,
}

impl RequestControl {
    fn cancel(&self, reason: u8) -> Result<(), RuntimeError> {
        if self
            .reason
            .compare_exchange(CANCEL_NONE, reason, Ordering::AcqRel, Ordering::Acquire)
            .is_ok()
        {
            self.process.terminate()?;
        }
        Ok(())
    }

    fn error(&self) -> Option<RuntimeError> {
        match self.reason.load(Ordering::Acquire) {
            CANCEL_REQUESTED => Some(RuntimeError::Cancelled {
                operation: RuntimeOperation::Request,
            }),
            CANCEL_TIMED_OUT => Some(RuntimeError::Timeout {
                operation: RuntimeOperation::Request,
                timeout: Duration::ZERO,
            }),
            _ => None,
        }
    }
}

struct RequestPermit {
    pending: Arc<AtomicUsize>,
}

impl RequestPermit {
    fn acquire(pending: &Arc<AtomicUsize>, limit: usize) -> Result<Self, RuntimeError> {
        pending
            .fetch_update(Ordering::AcqRel, Ordering::Acquire, |count| {
                (count < limit).then(|| count + 1)
            })
            .map_err(|_| RuntimeError::RequestQueueFull)?;
        Ok(Self {
            pending: Arc::clone(pending),
        })
    }
}

impl Drop for RequestPermit {
    fn drop(&mut self) {
        self.pending.fetch_sub(1, Ordering::AcqRel);
    }
}

enum WorkerCommand {
    Request {
        message: JsonRpcMessage,
        control: Arc<RequestControl>,
        response: mpsc::Sender<Result<JsonRpcExchange, RuntimeError>>,
        permit: RequestPermit,
    },
    Shutdown,
}

fn worker_loop(
    connection: RuntimeIo,
    receiver: Receiver<WorkerCommand>,
    process: Arc<ProcessControl>,
    connection_alive: Arc<AtomicBool>,
    limits: TransportLimits,
) {
    let mut client = JsonRpcClient::new(connection, limits);
    while let Ok(command) = receiver.recv() {
        match command {
            WorkerCommand::Request {
                message,
                control,
                response,
                permit,
            } => {
                let result = client.request(message).map_err(RuntimeError::Transport);
                let result = if let Some(error) = control.error() {
                    Err(error)
                } else {
                    result
                };
                if result.is_err() {
                    connection_alive.store(false, Ordering::Release);
                    if let Err(error) = &result
                        && let Ok(mut failure) = process.failure.lock()
                    {
                        *failure = Some(error.to_string());
                    }
                }
                let should_stop = result.is_err();
                // Completion releases capacity before making the result observable.
                // Dropping a caller's request handle must not release running work.
                drop(permit);
                let _ = response.send(result);
                if should_stop {
                    break;
                }
            }
            WorkerCommand::Shutdown => break,
        }
    }
    connection_alive.store(false, Ordering::Release);
}

struct StdioIo<R, W> {
    reader: R,
    writer: W,
}

impl<R: Read, W> Read for StdioIo<R, W> {
    fn read(&mut self, buffer: &mut [u8]) -> io::Result<usize> {
        self.reader.read(buffer)
    }
}

impl<R, W: Write> Write for StdioIo<R, W> {
    fn write(&mut self, buffer: &[u8]) -> io::Result<usize> {
        self.writer.write(buffer)
    }

    fn flush(&mut self) -> io::Result<()> {
        self.writer.flush()
    }
}

enum RuntimeIo {
    Stdio(StdioIo<ChildStdout, ChildStdin>),
    Tcp(TcpStream),
}

fn connect_tcp_retry(
    address: SocketAddr,
    mut limits: TransportLimits,
    timeout: Duration,
) -> Result<JsonRpcClient<TcpStream>, TransportError> {
    let deadline = Instant::now() + timeout;
    loop {
        let remaining = deadline.saturating_duration_since(Instant::now());
        if remaining.is_zero() {
            return Err(TransportError::Io(io::Error::new(
                io::ErrorKind::TimedOut,
                "runtime TCP endpoint did not become ready before startup deadline",
            )));
        }
        limits.connect_timeout = Some(
            limits
                .connect_timeout
                .unwrap_or(remaining)
                .min(remaining)
                .min(Duration::from_millis(100)),
        );
        match connect_tcp(address, limits) {
            Ok(client) => return Ok(client),
            Err(TransportError::Io(error))
                if matches!(
                    error.kind(),
                    io::ErrorKind::ConnectionRefused
                        | io::ErrorKind::AddrNotAvailable
                        | io::ErrorKind::TimedOut
                ) =>
            {
                thread::sleep(Duration::from_millis(2));
            }
            Err(error) => return Err(error),
        }
    }
}

impl Read for RuntimeIo {
    fn read(&mut self, buffer: &mut [u8]) -> io::Result<usize> {
        match self {
            Self::Stdio(io) => io.read(buffer),
            Self::Tcp(stream) => stream.read(buffer),
        }
    }
}

impl Write for RuntimeIo {
    fn write(&mut self, buffer: &[u8]) -> io::Result<usize> {
        match self {
            Self::Stdio(io) => io.write(buffer),
            Self::Tcp(stream) => stream.write(buffer),
        }
    }

    fn flush(&mut self) -> io::Result<()> {
        match self {
            Self::Stdio(io) => io.flush(),
            Self::Tcp(stream) => stream.flush(),
        }
    }
}

fn terminate_child(child: &mut Child) {
    let _ = child.kill();
    let _ = child.wait();
}
