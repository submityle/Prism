//! A real UDP transport (design §6 "可靠 UDP" — the datagram underlayer).
//!
//! [`UdpTransport`] is the provider-seam [`Transport`] over a real `UdpSocket`.
//! UDP is **unreliable and unordered** by design: this layer is best-effort, and
//! reliability, ordering and framing live above it (the [`crate::sequence`]
//! delivery window restores order; [`crate::frame`] splits a stream; an ARQ layer
//! would retransmit). That is exactly why the pieces below it are pure and
//! testable on one machine: the socket is the only part that needs a network.
//!
//! The socket is non-blocking, so [`recv`](Transport::recv) returns `None` when
//! nothing has arrived yet — a frame loop polls it each frame rather than
//! blocking on it.

use alloc::vec;
use alloc::vec::Vec;
use std::io;
use std::net::{SocketAddr, ToSocketAddrs, UdpSocket};
use std::time::Duration;

use crate::transport::Transport;

/// The largest UDP payload (the IPv4 limit): a datagram larger than this can
/// never arrive, so it bounds the receive buffer.
pub const MAX_DATAGRAM: usize = 65_507;

/// An unreliable UDP transport to one peer.
pub struct UdpTransport {
    socket: UdpSocket,
    peer: SocketAddr,
    connected: bool,
    /// The receive scratch buffer, reused across polls.
    buffer: Vec<u8>,
    last_error: Option<io::ErrorKind>,
    dropped: u64,
    received: u64,
}

impl UdpTransport {
    /// Bind a local address (non-blocking). Bind `127.0.0.1:0` for an ephemeral
    /// loopback port and read it back with [`local_addr`](Self::local_addr).
    ///
    /// # Errors
    /// Any bind / `set_nonblocking` error.
    pub fn bind(local: impl ToSocketAddrs) -> io::Result<Self> {
        let socket = UdpSocket::bind(local)?;
        socket.set_nonblocking(true)?;
        Ok(Self {
            socket,
            peer: SocketAddr::from(([0, 0, 0, 0], 0)),
            connected: false,
            buffer: vec![0u8; MAX_DATAGRAM],
            last_error: None,
            dropped: 0,
            received: 0,
        })
    }

    /// Target `peer` for subsequent sends.
    ///
    /// # Errors
    /// [`io::ErrorKind::InvalidInput`] when `peer` resolves to no address.
    pub fn connect_to(&mut self, peer: impl ToSocketAddrs) -> io::Result<()> {
        let peer = peer.to_socket_addrs()?.next().ok_or_else(|| {
            io::Error::new(io::ErrorKind::InvalidInput, "peer resolved to nothing")
        })?;
        self.socket.connect(peer)?;
        self.peer = peer;
        self.connected = true;
        Ok(())
    }

    /// The local address the socket is bound to.
    ///
    /// # Errors
    /// The underlying `local_addr` error.
    pub fn local_addr(&self) -> io::Result<SocketAddr> {
        self.socket.local_addr()
    }

    /// The peer this transport sends to.
    #[must_use]
    pub fn peer(&self) -> SocketAddr {
        self.peer
    }

    /// Whether [`connect_to`](Self::connect_to) completed successfully. An
    /// unconnected endpoint refuses sends instead of silently targeting the
    /// placeholder `0.0.0.0:0` address.
    #[must_use]
    pub fn is_connected(&self) -> bool {
        self.connected
    }

    /// Take the last socket error observed by send/receive polling.
    pub fn take_error_kind(&mut self) -> Option<io::ErrorKind> {
        self.last_error.take()
    }

    /// Number of datagrams rejected by the local socket or by an unconnected
    /// endpoint.
    #[must_use]
    pub fn dropped(&self) -> u64 {
        self.dropped
    }

    /// Number of datagrams received from the connected peer.
    #[must_use]
    pub fn received(&self) -> u64 {
        self.received
    }

    /// Set a receive timeout (for a blocking-style caller).
    ///
    /// # Errors
    /// The underlying `set_read_timeout` error.
    pub fn set_read_timeout(&self, timeout: Option<Duration>) -> io::Result<()> {
        self.socket.set_read_timeout(timeout)
    }
}

impl Transport for UdpTransport {
    /// Best-effort send to the connected peer. A datagram error is recorded and
    /// counted as a local drop; higher layers may retransmit it.
    fn send(&mut self, bytes: &[u8]) {
        if !self.connected {
            self.last_error = Some(io::ErrorKind::NotConnected);
            self.dropped = self.dropped.saturating_add(1);
            return;
        }
        if let Err(error) = self.socket.send(bytes) {
            self.last_error = Some(error.kind());
            self.dropped = self.dropped.saturating_add(1);
        }
    }

    /// Take the next datagram from the connected peer, or `None` when none has
    /// arrived (`WouldBlock` / timeout). Connecting the socket makes the OS
    /// discard datagrams from other peers before they reach this seam.
    fn recv(&mut self) -> Option<Vec<u8>> {
        match self.socket.recv(&mut self.buffer) {
            Ok(len) => {
                self.received = self.received.saturating_add(1);
                Some(self.buffer[..len].to_vec())
            }
            Err(ref error)
                if matches!(
                    error.kind(),
                    io::ErrorKind::WouldBlock | io::ErrorKind::TimedOut
                ) =>
            {
                None
            }
            Err(error) => {
                self.last_error = Some(error.kind());
                None
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Poll a non-blocking transport for up to ~1s (loopback UDP is immediate,
    /// but the scheduler may delay it a moment).
    fn recv_within(mut transport: impl FnMut() -> Option<Vec<u8>>) -> Option<Vec<u8>> {
        for _ in 0..1_000 {
            if let Some(bytes) = transport() {
                return Some(bytes);
            }
            std::thread::sleep(Duration::from_millis(1));
        }
        None
    }

    /// Whether a bind failure must fail the test instead of skipping it.
    fn require_net() -> bool {
        std::env::var_os("PRISM_REQUIRE_NET").is_some()
    }

    /// Bind a loopback socket, or skip when the environment forbids sockets (a
    /// sandbox denying binds). `PRISM_REQUIRE_NET=1` makes the skip a failure,
    /// so a CI run can prove the socket path really executed.
    fn bind_or_skip() -> Option<UdpTransport> {
        match UdpTransport::bind("127.0.0.1:0") {
            Ok(transport) => Some(transport),
            Err(error) => {
                assert!(
                    !require_net(),
                    "PRISM_REQUIRE_NET is set but binding a UDP socket failed: {error}"
                );
                None
            }
        }
    }

    #[test]
    fn two_udp_transports_exchange_a_datagram_over_loopback() {
        let Some(mut a) = bind_or_skip() else {
            return; // no socket permission here; the wiring is exercised elsewhere
        };
        let Some(mut b) = bind_or_skip() else {
            return;
        };
        let a_addr = a.local_addr().expect("local addr");
        let b_addr = b.local_addr().expect("local addr");
        a.connect_to(b_addr).expect("peer");
        b.connect_to(a_addr).expect("peer");
        assert!(a.is_connected());
        assert!(b.is_connected());

        a.send(b"hello over udp");
        let received = recv_within(|| b.recv());
        assert_eq!(received, Some(b"hello over udp".to_vec()));

        // Nothing else is pending.
        assert_eq!(b.recv(), None);
        assert_eq!(a.dropped(), 0);
        assert_eq!(b.received(), 1);
    }

    #[test]
    fn an_unconnected_socket_reports_send_failure() {
        let Some(mut transport) = bind_or_skip() else {
            return;
        };
        transport.send(b"not connected");
        assert_eq!(transport.dropped(), 1);
        assert_eq!(
            transport.take_error_kind(),
            Some(io::ErrorKind::NotConnected)
        );
    }
}
