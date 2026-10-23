//! Connected, non-blocking UDP adapter for complete bounded PWN1 frames.

use alloc::vec::Vec;
use core::fmt;
use std::io::{self, ErrorKind};
use std::net::{SocketAddr, UdpSocket};

use crate::ByteTransport;

const HEADER_BYTES: usize = 34;
const MAX_UDP_PAYLOAD: usize = 65_507;

/// Limits for a connected UDP byte transport.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct UdpByteTransportConfig {
    /// Maximum complete PWN1 frame bytes accepted in either direction.
    pub max_frame_bytes: usize,
}

impl UdpByteTransportConfig {
    /// Creates a validated configuration for the selected datagram ceiling.
    pub const fn new(max_frame_bytes: usize) -> Result<Self, UdpByteTransportError> {
        if max_frame_bytes < HEADER_BYTES || max_frame_bytes > MAX_UDP_PAYLOAD {
            return Err(UdpByteTransportError::InvalidFrameLimit(max_frame_bytes));
        }
        Ok(Self { max_frame_bytes })
    }
}

/// Errors from a connected UDP byte transport.
#[derive(Debug)]
pub enum UdpByteTransportError {
    /// The configured datagram ceiling is outside the PWN1/UDP range.
    InvalidFrameLimit(usize),
    /// The socket is not connected to a peer.
    UnconnectedSocket,
    /// The outgoing frame exceeds the configured datagram ceiling.
    FrameTooLarge {
        /// Supplied complete frame bytes.
        frame_bytes: usize,
        /// Configured maximum complete frame bytes.
        limit: usize,
    },
    /// A datagram exceeded the configured ceiling and was discarded.
    ReceivedDatagramTooLarge {
        /// Configured maximum complete frame bytes.
        limit: usize,
    },
    /// The UDP socket returned an I/O failure.
    Io(io::Error),
}

impl fmt::Display for UdpByteTransportError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidFrameLimit(limit) => write!(formatter, "invalid UDP frame limit: {limit}"),
            Self::UnconnectedSocket => formatter.write_str("UDP socket must be connected"),
            Self::FrameTooLarge { frame_bytes, limit } => write!(
                formatter,
                "UDP frame needs {frame_bytes} bytes, limit {limit}"
            ),
            Self::ReceivedDatagramTooLarge { limit } => {
                write!(formatter, "received UDP frame exceeds {limit} bytes")
            }
            Self::Io(error) => write!(formatter, "UDP I/O error: {error}"),
        }
    }
}

impl core::error::Error for UdpByteTransportError {
    fn source(&self) -> Option<&(dyn core::error::Error + 'static)> {
        match self {
            Self::Io(error) => Some(error),
            _ => None,
        }
    }
}

/// Connected non-blocking UDP transport for complete PWN1 datagrams.
///
/// The reliable session supplies sequence, ACK, retry and duplicate handling.
/// This adapter does not provide encryption, peer authentication or congestion
/// control; deployments must use an authenticated secure channel or another
/// ByteTransport implementation when those properties are required.
pub struct UdpByteTransport {
    socket: UdpSocket,
    peer: SocketAddr,
    config: UdpByteTransportConfig,
    receive_buffer: Vec<u8>,
}

impl UdpByteTransport {
    /// Constructs an adapter from a connected UDP socket and sets it non-blocking.
    pub fn new(
        socket: UdpSocket,
        config: UdpByteTransportConfig,
    ) -> Result<Self, UdpByteTransportError> {
        let peer = socket
            .peer_addr()
            .map_err(|_| UdpByteTransportError::UnconnectedSocket)?;
        if peer.port() == 0 {
            return Err(UdpByteTransportError::UnconnectedSocket);
        }
        socket
            .set_nonblocking(true)
            .map_err(UdpByteTransportError::Io)?;
        let mut receive_buffer = Vec::new();
        receive_buffer
            .try_reserve_exact(config.max_frame_bytes + 1)
            .map_err(|_| UdpByteTransportError::Io(io::Error::other("allocation failed")))?;
        receive_buffer.resize(config.max_frame_bytes + 1, 0);
        Ok(Self {
            socket,
            peer,
            config,
            receive_buffer,
        })
    }

    /// Returns the connected remote peer.
    pub const fn peer(&self) -> SocketAddr {
        self.peer
    }

    /// Returns the configured complete-frame byte ceiling.
    pub const fn max_frame_bytes(&self) -> usize {
        self.config.max_frame_bytes
    }

    #[cfg(all(test, feature = "crypto"))]
    pub(crate) fn inject_test_datagram(&self, bytes: &[u8]) {
        self.socket
            .send(bytes)
            .expect("test datagram send succeeds");
    }
}

impl ByteTransport for UdpByteTransport {
    type Error = UdpByteTransportError;

    fn try_send_bytes(&mut self, frame: &[u8]) -> Result<(), Self::Error> {
        if frame.len() > self.config.max_frame_bytes {
            return Err(UdpByteTransportError::FrameTooLarge {
                frame_bytes: frame.len(),
                limit: self.config.max_frame_bytes,
            });
        }
        match self.socket.send(frame) {
            Ok(sent) if sent == frame.len() => Ok(()),
            Ok(_) => Err(UdpByteTransportError::Io(io::Error::new(
                ErrorKind::WriteZero,
                "partial UDP datagram send",
            ))),
            Err(error) => Err(UdpByteTransportError::Io(error)),
        }
    }

    fn try_recv_bytes(&mut self) -> Result<Option<Vec<u8>>, Self::Error> {
        match self.socket.recv(&mut self.receive_buffer) {
            Ok(bytes) if bytes > self.config.max_frame_bytes => {
                Err(UdpByteTransportError::ReceivedDatagramTooLarge {
                    limit: self.config.max_frame_bytes,
                })
            }
            Ok(bytes) => {
                let mut frame = Vec::new();
                frame.try_reserve_exact(bytes).map_err(|_| {
                    UdpByteTransportError::Io(io::Error::other("allocation failed"))
                })?;
                frame.extend_from_slice(&self.receive_buffer[..bytes]);
                Ok(Some(frame))
            }
            Err(error) if error.kind() == ErrorKind::WouldBlock => Ok(None),
            Err(error) => Err(UdpByteTransportError::Io(error)),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec;
    use std::net::UdpSocket;
    use std::time::{Duration, Instant};

    fn pair() -> (UdpByteTransport, UdpByteTransport) {
        let left = UdpSocket::bind(("127.0.0.1", 0)).unwrap();
        let right = UdpSocket::bind(("127.0.0.1", 0)).unwrap();
        left.connect(right.local_addr().unwrap()).unwrap();
        right.connect(left.local_addr().unwrap()).unwrap();
        let config = UdpByteTransportConfig::new(1200).unwrap();
        (
            UdpByteTransport::new(left, config).unwrap(),
            UdpByteTransport::new(right, config).unwrap(),
        )
    }

    fn receive_until(
        transport: &mut UdpByteTransport,
    ) -> Result<Option<Vec<u8>>, UdpByteTransportError> {
        let deadline = Instant::now() + Duration::from_secs(1);
        loop {
            if let Some(frame) = transport.try_recv_bytes()? {
                return Ok(Some(frame));
            }
            if Instant::now() >= deadline {
                return Ok(None);
            }
            std::thread::yield_now();
        }
    }

    #[test]
    fn connected_udp_round_trips_bounded_datagrams() {
        let (mut left, mut right) = pair();
        let frame = vec![0x5a; 900];
        left.try_send_bytes(&frame).unwrap();
        assert_eq!(receive_until(&mut right).unwrap().unwrap(), frame);
    }

    #[test]
    fn udp_rejects_oversized_outbound_and_inbound_datagrams() {
        let (mut left, right) = pair();
        assert!(matches!(
            left.try_send_bytes(&vec![0; 1201]),
            Err(UdpByteTransportError::FrameTooLarge { .. })
        ));
        right.socket.send(&vec![0; 1201]).unwrap();
        assert!(matches!(
            receive_until(&mut left),
            Err(UdpByteTransportError::ReceivedDatagramTooLarge { limit: 1200 })
        ));
    }

    #[test]
    fn udp_receive_is_nonblocking_when_empty() {
        let (mut left, _right) = pair();
        assert_eq!(left.try_recv_bytes().unwrap(), None);
    }

    #[test]
    fn udp_configuration_rejects_limits_outside_datagram_range() {
        assert!(UdpByteTransportConfig::new(HEADER_BYTES - 1).is_err());
        assert!(UdpByteTransportConfig::new(MAX_UDP_PAYLOAD + 1).is_err());
    }
}
