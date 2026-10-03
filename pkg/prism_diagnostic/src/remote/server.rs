//! Loopback socket transport for the remote observability protocol (`remote`
//! feature).
//!
//! [`RemoteServer`] binds a TCP listener (default to localhost, per design §15's
//! "loopback by default" posture), accepts panel connections, and runs each on
//! its own reader thread that decodes [`RemoteCommand`]s and applies them to the
//! shared [`RuntimeControls`]. The engine side pushes [`RemoteEvent`]s to every
//! connected panel with [`RemoteServerHandle::broadcast`]. [`RemoteClient`] is
//! the panel-side counterpart used by tests and simple tooling.
//!
//! This is the only part of M5 that uses `std::net`/`std::thread`, and it is
//! gated behind the `remote` feature so the default build neither opens sockets
//! nor spawns threads. The hot path (`broadcast`) never blocks on a slow or dead
//! client: a write error simply drops that client (design §16's "drop, don't
//! block" backpressure stance).

extern crate alloc;

use alloc::string::String;
use alloc::sync::Arc;
use alloc::vec::Vec;
use core::sync::atomic::{AtomicBool, Ordering};
use std::io;
use std::net::{Shutdown, SocketAddr, TcpListener, TcpStream, ToSocketAddrs};
use std::sync::Mutex;
use std::thread::{self, JoinHandle};

use super::command::{CommandOutcome, RemoteCommand, RuntimeControls};
use super::protocol::{self, RemoteEvent};

/// Shared server state, held behind an `Arc` so the accept thread, per-client
/// reader threads, and the owning handle all observe the same truth.
#[derive(Debug)]
struct ServerInner {
    controls: Arc<RuntimeControls>,
    clients: Mutex<Vec<TcpStream>>,
    received: Mutex<Vec<RemoteCommand>>,
    readers: Mutex<Vec<JoinHandle<()>>>,
    shutdown: AtomicBool,
}

impl ServerInner {
    fn is_shutting_down(&self) -> bool {
        self.shutdown.load(Ordering::Acquire)
    }

    fn register_client(self: &Arc<Self>, stream: TcpStream) {
        let _ = stream.set_nodelay(true);
        let Ok(reader_stream) = stream.try_clone() else {
            return;
        };
        self.clients.lock().unwrap().push(stream);
        let inner = Arc::clone(self);
        if let Ok(handle) = thread::Builder::new()
            .name(String::from("prism-remote-reader"))
            .spawn(move || reader_loop(reader_stream, &inner))
        {
            self.readers.lock().unwrap().push(handle);
        }
    }

    /// Deliver `event` to every connected client, pruning any that error out.
    fn broadcast(&self, event: &RemoteEvent) -> usize {
        let mut clients = self.clients.lock().unwrap();
        let mut delivered = 0usize;
        clients.retain_mut(|stream| match protocol::write_event(stream, event) {
            Ok(()) => {
                delivered += 1;
                true
            }
            Err(_) => false,
        });
        delivered
    }
}

fn reader_loop(mut stream: TcpStream, inner: &Arc<ServerInner>) {
    loop {
        if inner.is_shutting_down() {
            break;
        }
        match protocol::read_command(&mut stream) {
            Ok(command) => {
                inner.received.lock().unwrap().push(command);
                if matches!(inner.controls.apply(command), CommandOutcome::ShuttingDown) {
                    inner.shutdown.store(true, Ordering::Release);
                    break;
                }
            }
            // Peer closed, socket shut down on teardown, or a decode error:
            // end this client's session without disturbing others.
            Err(_) => break,
        }
    }
}

fn accept_loop(listener: &TcpListener, inner: &Arc<ServerInner>) {
    for stream in listener.incoming() {
        if inner.is_shutting_down() {
            break;
        }
        match stream {
            Ok(stream) => inner.register_client(stream),
            Err(_) => {
                if inner.is_shutting_down() {
                    break;
                }
            }
        }
    }
}

/// A bound-but-not-yet-serving remote server.
///
/// Call [`run`](RemoteServer::run) to start the accept loop and obtain a
/// [`RemoteServerHandle`] for broadcasting and shutdown.
#[derive(Debug)]
pub struct RemoteServer {
    listener: TcpListener,
    inner: Arc<ServerInner>,
}

impl RemoteServer {
    /// Bind to `addr` with fresh default [`RuntimeControls`].
    ///
    /// Pass `"127.0.0.1:0"` to bind an ephemeral loopback port and read the
    /// chosen address back with [`RemoteServerHandle::local_addr`].
    pub fn bind(addr: impl ToSocketAddrs) -> io::Result<Self> {
        Self::bind_with(addr, Arc::new(RuntimeControls::new()))
    }

    /// Bind to `addr`, applying commands to the caller-provided `controls` so
    /// the engine can observe the same tuning state the panel drives.
    pub fn bind_with(addr: impl ToSocketAddrs, controls: Arc<RuntimeControls>) -> io::Result<Self> {
        let listener = TcpListener::bind(addr)?;
        Ok(Self {
            listener,
            inner: Arc::new(ServerInner {
                controls,
                clients: Mutex::new(Vec::new()),
                received: Mutex::new(Vec::new()),
                readers: Mutex::new(Vec::new()),
                shutdown: AtomicBool::new(false),
            }),
        })
    }

    /// The shared runtime controls commands will be applied to.
    pub fn controls(&self) -> Arc<RuntimeControls> {
        Arc::clone(&self.inner.controls)
    }

    /// The bound local address.
    pub fn local_addr(&self) -> io::Result<SocketAddr> {
        self.listener.local_addr()
    }

    /// Start the background accept loop, returning a handle for broadcasting and
    /// shutdown.
    pub fn run(self) -> io::Result<RemoteServerHandle> {
        let local_addr = self.listener.local_addr()?;
        let listener = self.listener;
        let inner = self.inner;
        let accept_inner = Arc::clone(&inner);
        let accept = thread::Builder::new()
            .name(String::from("prism-remote-accept"))
            .spawn(move || accept_loop(&listener, &accept_inner))?;
        Ok(RemoteServerHandle {
            inner,
            local_addr,
            accept: Some(accept),
        })
    }
}

/// A running remote server. Dropping or calling [`shutdown`](Self::shutdown)
/// stops the accept loop and joins all worker threads.
#[derive(Debug)]
pub struct RemoteServerHandle {
    inner: Arc<ServerInner>,
    local_addr: SocketAddr,
    accept: Option<JoinHandle<()>>,
}

impl RemoteServerHandle {
    /// The bound local address.
    pub fn local_addr(&self) -> SocketAddr {
        self.local_addr
    }

    /// The shared runtime controls commands are applied to.
    pub fn controls(&self) -> Arc<RuntimeControls> {
        Arc::clone(&self.inner.controls)
    }

    /// Snapshot every command received from panels so far, in arrival order.
    pub fn received_commands(&self) -> Vec<RemoteCommand> {
        self.inner.received.lock().unwrap().clone()
    }

    /// Number of currently connected clients.
    pub fn connected_clients(&self) -> usize {
        self.inner.clients.lock().unwrap().len()
    }

    /// Broadcast `event` to every connected panel, returning how many received
    /// it. Dead clients are pruned.
    pub fn broadcast(&self, event: &RemoteEvent) -> usize {
        self.inner.broadcast(event)
    }

    /// Whether a client requested shutdown or [`shutdown`](Self::shutdown) ran.
    pub fn is_shutting_down(&self) -> bool {
        self.inner.is_shutting_down()
    }

    /// Stop the server: end the accept loop, close client sockets, and join all
    /// worker threads. Idempotent with the `Drop` teardown.
    pub fn shutdown(mut self) {
        self.teardown();
    }

    fn teardown(&mut self) {
        if self.accept.is_none() {
            return;
        }
        self.inner.shutdown.store(true, Ordering::Release);
        // Wake the blocking `accept` with a throwaway loopback connection.
        if let Ok(stream) = TcpStream::connect(self.local_addr) {
            let _ = stream.shutdown(Shutdown::Both);
        }
        // Unblock reader threads parked in `read`: shutting the shared socket
        // down makes their `read_exact` return immediately.
        for client in self.inner.clients.lock().unwrap().iter() {
            let _ = client.shutdown(Shutdown::Both);
        }
        if let Some(accept) = self.accept.take() {
            let _ = accept.join();
        }
        let readers: Vec<JoinHandle<()>> = self.inner.readers.lock().unwrap().drain(..).collect();
        for reader in readers {
            let _ = reader.join();
        }
    }
}

impl Drop for RemoteServerHandle {
    fn drop(&mut self) {
        self.teardown();
    }
}

/// The panel-side client: connects to a [`RemoteServer`], sends
/// [`RemoteCommand`]s, and reads [`RemoteEvent`]s.
#[derive(Debug)]
pub struct RemoteClient {
    stream: TcpStream,
}

impl RemoteClient {
    /// Connect to a remote server at `addr`.
    pub fn connect(addr: impl ToSocketAddrs) -> io::Result<Self> {
        let stream = TcpStream::connect(addr)?;
        stream.set_nodelay(true)?;
        Ok(Self { stream })
    }

    /// Bound local address of the client socket.
    pub fn local_addr(&self) -> io::Result<SocketAddr> {
        self.stream.local_addr()
    }

    /// Limit how long [`recv_event`](Self::recv_event) blocks; `None` blocks
    /// indefinitely.
    pub fn set_read_timeout(&self, timeout: Option<core::time::Duration>) -> io::Result<()> {
        self.stream.set_read_timeout(timeout)
    }

    /// Send one command to the server.
    pub fn send_command(&mut self, command: &RemoteCommand) -> io::Result<()> {
        protocol::write_command(&mut self.stream, command)
    }

    /// Block until the next event arrives from the server.
    pub fn recv_event(&mut self) -> io::Result<RemoteEvent> {
        protocol::read_event(&mut self.stream)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::Level;
    use crate::remote::protocol::FrameSummary;
    use core::time::Duration;
    use std::time::Instant;

    fn wait_until(mut cond: impl FnMut() -> bool) -> bool {
        let deadline = Instant::now() + Duration::from_secs(5);
        while Instant::now() < deadline {
            if cond() {
                return true;
            }
            thread::sleep(Duration::from_millis(5));
        }
        cond()
    }

    #[test]
    fn loopback_command_applies_and_event_broadcasts() {
        let server = RemoteServer::bind("127.0.0.1:0").expect("bind");
        let controls = server.controls();
        let handle = server.run().expect("run");
        let addr = handle.local_addr();

        let mut client = RemoteClient::connect(addr).expect("connect");
        client
            .set_read_timeout(Some(Duration::from_secs(5)))
            .unwrap();

        // Wait for the server to register the client before broadcasting.
        assert!(wait_until(|| handle.connected_clients() >= 1));

        // Panel → engine: toggle sinks off and change the log level.
        client
            .send_command(&RemoteCommand::ToggleSink { enabled: false })
            .unwrap();
        client
            .send_command(&RemoteCommand::SetLogLevel(Level::Error))
            .unwrap();

        assert!(wait_until(|| !controls.sinks_enabled()));
        assert!(wait_until(|| crate::filter::max_level() == Level::Error));
        assert!(wait_until(|| handle.received_commands().len() >= 2));

        // Engine → panel: broadcast a couple of events and read them back.
        let frame = RemoteEvent::Frame(FrameSummary {
            frame_index: 1,
            last_delta_ms: 16.0,
            min_ms: 15.0,
            avg_ms: 16.0,
            max_ms: 17.0,
            fps: 62.0,
        });
        assert_eq!(handle.broadcast(&frame), 1);
        assert_eq!(client.recv_event().unwrap(), frame);

        let log = RemoteEvent::Log {
            level: Level::Warn,
            target: String::from("net"),
            message: String::from("hi"),
            timestamp_nanos: 42,
        };
        handle.broadcast(&log);
        assert_eq!(client.recv_event().unwrap(), log);

        // Restore the shared global filter the command mutated.
        crate::filter::set_max_level(Level::Info);
        handle.shutdown();
    }

    #[test]
    fn shutdown_command_stops_the_server() {
        let server = RemoteServer::bind("127.0.0.1:0").expect("bind");
        let handle = server.run().expect("run");
        let addr = handle.local_addr();

        let mut client = RemoteClient::connect(addr).expect("connect");
        assert!(wait_until(|| handle.connected_clients() >= 1));
        client.send_command(&RemoteCommand::Shutdown).unwrap();
        assert!(wait_until(|| handle.is_shutting_down()));
        handle.shutdown();
    }

    #[test]
    fn broadcast_with_no_clients_delivers_zero() {
        let server = RemoteServer::bind("127.0.0.1:0").expect("bind");
        let handle = server.run().expect("run");
        let event = RemoteEvent::Metric {
            name: String::from("x"),
            value: 1.0,
            timestamp_nanos: 0,
        };
        assert_eq!(handle.broadcast(&event), 0);
        handle.shutdown();
    }
}
