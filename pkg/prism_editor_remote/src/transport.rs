//! Bounded length-prefixed transports for the JSON-RPC synchronization layer.
//!
//! The transport uses a four-byte big-endian payload length followed by one
//! JSON-RPC message. It is deliberately small and synchronous: standard
//! library TCP is enough for editor/runtime loopback and keeps framing,
//! timeout, EOF, and allocation behavior explicit.

use crate::{
    decode_json_rpc_with_limits, encode_json_rpc, Endpoint, JsonRpcCodecError,
    JsonRpcDispatchError, JsonRpcId, JsonRpcLimits, JsonRpcMessage, JsonRpcNotification,
    JsonRpcRequest, JsonRpcResponse, JsonRpcServer,
};
use std::fmt;
use std::io::{self, Read, Write};
use std::net::{TcpListener, TcpStream, ToSocketAddrs};
use std::time::Duration;

const FRAME_LENGTH_BYTES: usize = 4;

/// Limits and socket timing settings used by a framed transport.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct TransportLimits {
    /// Maximum JSON-RPC payload size in bytes.
    pub max_frame_bytes: usize,
    /// Read timeout applied to TCP streams. `None` means no timeout.
    pub read_timeout: Option<Duration>,
    /// Write timeout applied to TCP streams. `None` means no timeout.
    pub write_timeout: Option<Duration>,
    /// Connection timeout used by [`connect_tcp`]. `None` uses the standard
    /// library's default connection behavior.
    pub connect_timeout: Option<Duration>,
}

impl Default for TransportLimits {
    fn default() -> Self {
        Self {
            max_frame_bytes: 16 * 1024 * 1024,
            read_timeout: None,
            write_timeout: None,
            connect_timeout: None,
        }
    }
}

/// The part of a frame that ended prematurely.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum FrameStage {
    /// The four-byte payload length was incomplete.
    Header,
    /// The payload ended before its declared length.
    Payload,
}

/// Errors raised by framing, JSON-RPC dispatch, or the underlying stream.
#[derive(Debug)]
pub enum TransportError {
    /// An underlying stream operation failed.
    Io(io::Error),
    /// A frame exceeded the configured payload limit.
    FrameTooLarge {
        /// Declared or attempted payload length.
        length: usize,
        /// Configured maximum payload length.
        max: usize,
    },
    /// A frame cannot be represented by the four-byte length prefix.
    FrameLengthOverflow {
        /// Attempted payload length.
        length: usize,
    },
    /// The peer closed the stream before a complete frame was available.
    UnexpectedEof {
        /// Frame section that ended prematurely.
        stage: FrameStage,
        /// Number of bytes required for the section.
        expected: usize,
        /// Number of bytes received before EOF.
        received: usize,
    },
    /// The JSON-RPC codec rejected a payload.
    JsonRpc(JsonRpcCodecError),
    /// A server-side notification could not be dispatched.
    Dispatch(JsonRpcDispatchError),
    /// A client request waited for a response but the peer closed the stream.
    PeerClosedBeforeResponse {
        /// Request ID that was waiting for a response.
        id: JsonRpcId,
    },
    /// A response did not match the request currently in flight.
    UnexpectedResponseId {
        /// Request ID currently in flight.
        expected: JsonRpcId,
        /// Response ID received from the peer.
        received: JsonRpcId,
    },
    /// A client request was given a message that cannot receive a response.
    InvalidClientRequest,
    /// A response wait received a message that cannot complete the request.
    UnexpectedMessage {
        /// Stable diagnostic for the unsupported message shape.
        detail: &'static str,
    },
}

impl TransportError {
    /// Returns whether the error was caused by a configured I/O timeout.
    #[must_use]
    pub fn is_timeout(&self) -> bool {
        matches!(
            self,
            Self::Io(error)
                if matches!(error.kind(), io::ErrorKind::TimedOut | io::ErrorKind::WouldBlock)
        )
    }
}

impl fmt::Display for TransportError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Io(error) => write!(formatter, "transport I/O error: {error}"),
            Self::FrameTooLarge { length, max } => {
                write!(formatter, "frame length {length} exceeds limit {max}")
            }
            Self::FrameLengthOverflow { length } => {
                write!(formatter, "frame length {length} does not fit in a u32")
            }
            Self::UnexpectedEof {
                stage,
                expected,
                received,
            } => write!(
                formatter,
                "unexpected EOF in {stage:?}: expected {expected} bytes, received {received}"
            ),
            Self::JsonRpc(error) => write!(formatter, "JSON-RPC transport payload error: {error}"),
            Self::Dispatch(error) => {
                write!(formatter, "JSON-RPC dispatch error: {}", error.message)
            }
            Self::PeerClosedBeforeResponse { id } => {
                write!(formatter, "peer closed before response for request {id:?}")
            }
            Self::UnexpectedResponseId { expected, received } => write!(
                formatter,
                "unexpected response ID {received:?}; expected {expected:?}"
            ),
            Self::InvalidClientRequest => {
                formatter.write_str("client request must be a JSON-RPC request with an ID")
            }
            Self::UnexpectedMessage { detail } => formatter.write_str(detail),
        }
    }
}

impl std::error::Error for TransportError {}

impl From<io::Error> for TransportError {
    fn from(error: io::Error) -> Self {
        Self::Io(error)
    }
}

impl From<JsonRpcCodecError> for TransportError {
    fn from(error: JsonRpcCodecError) -> Self {
        Self::JsonRpc(error)
    }
}

/// A synchronous stream with bounded, length-prefixed frames.
pub struct LengthPrefixedStream<S> {
    stream: S,
    limits: TransportLimits,
}

impl<S> LengthPrefixedStream<S> {
    /// Wraps a stream with the supplied framing limits.
    #[must_use]
    pub fn new(stream: S, limits: TransportLimits) -> Self {
        Self { stream, limits }
    }

    /// Returns the configured framing limits.
    #[must_use]
    pub const fn limits(&self) -> TransportLimits {
        self.limits
    }

    /// Borrows the wrapped stream.
    #[must_use]
    pub const fn get_ref(&self) -> &S {
        &self.stream
    }

    /// Mutably borrows the wrapped stream.
    #[must_use]
    pub const fn get_mut(&mut self) -> &mut S {
        &mut self.stream
    }

    /// Unwraps the framed stream.
    #[must_use]
    pub fn into_inner(self) -> S {
        self.stream
    }
}

impl<S: Read> LengthPrefixedStream<S> {
    /// Reads one frame. Clean EOF before the next header is returned as
    /// `Ok(None)`; EOF after any header or payload byte is an error.
    pub fn read_frame(&mut self) -> Result<Option<Vec<u8>>, TransportError> {
        let mut header = [0_u8; FRAME_LENGTH_BYTES];
        if !read_exact_or_eof(&mut self.stream, &mut header, FrameStage::Header)? {
            return Ok(None);
        }
        let length = u32::from_be_bytes(header) as usize;
        if length > self.limits.max_frame_bytes {
            return Err(TransportError::FrameTooLarge {
                length,
                max: self.limits.max_frame_bytes,
            });
        }
        let mut payload = vec![0_u8; length];
        read_exact_or_eof(&mut self.stream, &mut payload, FrameStage::Payload)?;
        Ok(Some(payload))
    }

    /// Reads and decodes one JSON-RPC message using the default JSON limits.
    pub fn read_message(&mut self) -> Result<Option<JsonRpcMessage>, TransportError> {
        let frame = self.read_frame()?;
        frame
            .map(|bytes| decode_json_rpc_with_limits(&bytes, json_limits(self.limits)))
            .transpose()
            .map_err(TransportError::from)
    }
}

impl<S: Write> LengthPrefixedStream<S> {
    /// Writes one bounded frame and flushes the wrapped stream.
    pub fn write_frame(&mut self, payload: &[u8]) -> Result<(), TransportError> {
        if payload.len() > self.limits.max_frame_bytes {
            return Err(TransportError::FrameTooLarge {
                length: payload.len(),
                max: self.limits.max_frame_bytes,
            });
        }
        let length =
            u32::try_from(payload.len()).map_err(|_| TransportError::FrameLengthOverflow {
                length: payload.len(),
            })?;
        self.stream.write_all(&length.to_be_bytes())?;
        self.stream.write_all(payload)?;
        self.stream.flush()?;
        Ok(())
    }

    /// Encodes and writes one JSON-RPC message.
    pub fn write_message(&mut self, message: &JsonRpcMessage) -> Result<(), TransportError> {
        self.write_frame(&encode_json_rpc(message))
    }
}

fn read_exact_or_eof<R: Read>(
    reader: &mut R,
    buffer: &mut [u8],
    stage: FrameStage,
) -> Result<bool, TransportError> {
    if buffer.is_empty() {
        return Ok(true);
    }
    let mut received = 0;
    while received < buffer.len() {
        match reader.read(&mut buffer[received..]) {
            Ok(0) if received == 0 && stage == FrameStage::Header => return Ok(false),
            Ok(0) => {
                return Err(TransportError::UnexpectedEof {
                    stage,
                    expected: buffer.len(),
                    received,
                });
            }
            Ok(count) => received += count,
            Err(error) if error.kind() == io::ErrorKind::Interrupted => continue,
            Err(error) => return Err(TransportError::Io(error)),
        }
    }
    Ok(true)
}

fn json_limits(limits: TransportLimits) -> JsonRpcLimits {
    let mut result = JsonRpcLimits::default();
    result.max_message_bytes = result.max_message_bytes.min(limits.max_frame_bytes);
    result
}

/// Messages returned by a synchronous client request.
#[derive(Clone, Debug, PartialEq)]
pub struct JsonRpcExchange {
    /// The response matching the request ID.
    pub response: JsonRpcResponse,
    /// Notifications received before the response.
    pub notifications: Vec<JsonRpcNotification>,
}

/// A JSON-RPC client over any readable and writable framed stream.
pub struct JsonRpcClient<S> {
    transport: LengthPrefixedStream<S>,
    json_limits: JsonRpcLimits,
}

impl<S> JsonRpcClient<S> {
    /// Creates a client using the transport frame limit for JSON message size.
    #[must_use]
    pub fn new(stream: S, limits: TransportLimits) -> Self {
        Self::with_json_limits(stream, limits, json_limits(limits))
    }

    /// Creates a client with independent transport and JSON parser limits.
    #[must_use]
    pub fn with_json_limits(
        stream: S,
        limits: TransportLimits,
        json_limits: JsonRpcLimits,
    ) -> Self {
        Self {
            transport: LengthPrefixedStream::new(stream, limits),
            json_limits,
        }
    }

    /// Borrows the framed transport.
    #[must_use]
    pub const fn transport(&self) -> &LengthPrefixedStream<S> {
        &self.transport
    }

    /// Mutably borrows the framed transport.
    #[must_use]
    pub const fn transport_mut(&mut self) -> &mut LengthPrefixedStream<S> {
        &mut self.transport
    }

    /// Unwraps the client and returns its framed stream.
    #[must_use]
    pub fn into_inner(self) -> S {
        self.transport.into_inner()
    }

    /// Sends a JSON-RPC message, including notifications.
    pub fn send(&mut self, message: &JsonRpcMessage) -> Result<(), TransportError>
    where
        S: Write,
    {
        self.transport.write_message(message)
    }
}

impl<S: Read> JsonRpcClient<S> {
    /// Receives one JSON-RPC message, or `None` after clean peer EOF.
    pub fn receive(&mut self) -> Result<Option<JsonRpcMessage>, TransportError> {
        let frame = self.transport.read_frame()?;
        frame
            .map(|bytes| decode_json_rpc_with_limits(&bytes, self.json_limits))
            .transpose()
            .map_err(TransportError::from)
    }
}

impl<S: Read + Write> JsonRpcClient<S> {
    /// Sends a request and waits for its correlated response. Notifications
    /// received while waiting are retained in the returned exchange.
    pub fn request(&mut self, message: JsonRpcMessage) -> Result<JsonRpcExchange, TransportError> {
        let expected = match &message {
            JsonRpcMessage::Request(JsonRpcRequest { id, .. }) => id.clone(),
            _ => return Err(TransportError::InvalidClientRequest),
        };
        self.send(&message)?;
        let mut notifications = Vec::new();
        loop {
            let Some(message) = self.receive()? else {
                return Err(TransportError::PeerClosedBeforeResponse { id: expected });
            };
            match message {
                JsonRpcMessage::Response(response) if response.id == expected => {
                    return Ok(JsonRpcExchange {
                        response,
                        notifications,
                    });
                }
                JsonRpcMessage::Response(response) => {
                    return Err(TransportError::UnexpectedResponseId {
                        expected,
                        received: response.id,
                    });
                }
                JsonRpcMessage::Notification(notification) => notifications.push(notification),
                JsonRpcMessage::Request(_) => {
                    return Err(TransportError::UnexpectedMessage {
                        detail: "received a peer request while waiting for a response",
                    });
                }
                JsonRpcMessage::Batch(_) => {
                    return Err(TransportError::UnexpectedMessage {
                        detail: "received a JSON-RPC batch while waiting for a response",
                    });
                }
            }
        }
    }
}

/// A server session that connects a [`JsonRpcServer`] to one framed stream.
pub struct JsonRpcServerSession<E, S> {
    server: JsonRpcServer<E>,
    transport: LengthPrefixedStream<S>,
}

impl<E: Endpoint, S> JsonRpcServerSession<E, S> {
    /// Creates a session around a JSON-RPC server and stream.
    #[must_use]
    pub fn new(server: JsonRpcServer<E>, stream: S, limits: TransportLimits) -> Self {
        Self {
            server,
            transport: LengthPrefixedStream::new(stream, limits),
        }
    }

    /// Borrows the protocol server.
    #[must_use]
    pub const fn server(&self) -> &JsonRpcServer<E> {
        &self.server
    }

    /// Mutably borrows the protocol server.
    #[must_use]
    pub const fn server_mut(&mut self) -> &mut JsonRpcServer<E> {
        &mut self.server
    }

    /// Borrows the framed stream.
    #[must_use]
    pub const fn transport(&self) -> &LengthPrefixedStream<S> {
        &self.transport
    }

    /// Mutably borrows the framed stream.
    #[must_use]
    pub const fn transport_mut(&mut self) -> &mut LengthPrefixedStream<S> {
        &mut self.transport
    }
}

impl<E: Endpoint, S: Read + Write> JsonRpcServerSession<E, S> {
    /// Processes one incoming frame and writes all responses and notifications.
    /// Returns `false` for clean peer EOF before the next frame.
    pub fn serve_one(&mut self) -> Result<bool, TransportError> {
        let Some(frame) = self.transport.read_frame()? else {
            return Ok(false);
        };
        let output = self.server.handle_bytes(&frame)?;
        for message in output {
            self.transport.write_frame(&message)?;
        }
        Ok(true)
    }

    /// Serves the connection until the peer closes it or an error occurs.
    pub fn serve(&mut self) -> Result<(), TransportError> {
        while self.serve_one()? {}
        Ok(())
    }

    /// Publishes subscribed changes immediately on this connection.
    pub fn publish_changes(
        &mut self,
        target: crate::TargetId,
        changes: &crate::ChangeSet,
    ) -> Result<usize, TransportError> {
        let messages = self
            .server
            .publish_changes(target, changes)
            .map_err(TransportError::Dispatch)?;
        let count = messages.len();
        for message in messages {
            self.transport.write_message(&message)?;
        }
        Ok(count)
    }
}

/// A standard-library TCP listener that creates bounded JSON-RPC sessions.
pub struct TcpJsonRpcListener {
    listener: TcpListener,
    limits: TransportLimits,
}

impl TcpJsonRpcListener {
    /// Binds a TCP listener to the supplied address.
    pub fn bind<A: ToSocketAddrs>(
        address: A,
        limits: TransportLimits,
    ) -> Result<Self, TransportError> {
        Ok(Self {
            listener: TcpListener::bind(address)?,
            limits,
        })
    }

    /// Returns the OS-selected listening address.
    pub fn local_addr(&self) -> Result<std::net::SocketAddr, TransportError> {
        Ok(self.listener.local_addr()?)
    }

    /// Accepts one connection and attaches a fresh JSON-RPC server to it.
    pub fn accept<E: Endpoint>(
        &self,
        server: JsonRpcServer<E>,
    ) -> Result<JsonRpcServerSession<E, TcpStream>, TransportError> {
        let (stream, _) = self.listener.accept()?;
        configure_tcp_stream(&stream, self.limits)?;
        Ok(JsonRpcServerSession::new(server, stream, self.limits))
    }
}

/// Opens a bounded JSON-RPC client over a standard-library TCP stream.
pub fn connect_tcp<A: ToSocketAddrs>(
    address: A,
    limits: TransportLimits,
) -> Result<JsonRpcClient<TcpStream>, TransportError> {
    let stream = if let Some(timeout) = limits.connect_timeout {
        let mut addresses = address.to_socket_addrs()?;
        let address = addresses.next().ok_or_else(|| {
            io::Error::new(
                io::ErrorKind::InvalidInput,
                "TCP address resolved to no endpoints",
            )
        })?;
        TcpStream::connect_timeout(&address, timeout)?
    } else {
        TcpStream::connect(address)?
    };
    configure_tcp_stream(&stream, limits)?;
    Ok(JsonRpcClient::new(stream, limits))
}

fn configure_tcp_stream(stream: &TcpStream, limits: TransportLimits) -> Result<(), TransportError> {
    stream.set_read_timeout(limits.read_timeout)?;
    stream.set_write_timeout(limits.write_timeout)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        Capability, Change, ChangeSet, EndpointCapabilities, EntityId, EntityRecord, InMemoryStore,
        JsonNumber, JsonValue, Revision, SyncEndpoint, TargetId, WorldSnapshot,
    };
    use std::collections::BTreeMap;
    use std::io::Cursor;
    use std::thread;

    fn endpoint() -> SyncEndpoint<InMemoryStore> {
        let capabilities = EndpointCapabilities::from_capabilities([
            Capability::SnapshotRead,
            Capability::IncrementalRead,
            Capability::ComponentWrite,
            Capability::RevisionChecks,
        ]);
        let mut store = InMemoryStore::new("transport-test", capabilities, 8).expect("store");
        let target = TargetId::from_raw(7);
        let mut snapshot = WorldSnapshot::empty(target);
        snapshot
            .entities
            .insert(EntityId::from_raw(1), EntityRecord::new());
        store.register_target(snapshot).expect("target");
        SyncEndpoint::new(store)
    }

    fn initialize(id: u64) -> JsonRpcMessage {
        JsonRpcMessage::Request(JsonRpcRequest {
            id: JsonRpcId::Number(id),
            method: String::from("prism/initialize"),
            params: Some(JsonValue::Object(BTreeMap::from([
                (
                    String::from("protocol"),
                    JsonValue::Object(BTreeMap::from([
                        (
                            String::from("major"),
                            JsonValue::Number(JsonNumber::Unsigned(1)),
                        ),
                        (
                            String::from("minor"),
                            JsonValue::Number(JsonNumber::Unsigned(0)),
                        ),
                    ])),
                ),
                (
                    String::from("capabilities"),
                    JsonValue::Array(vec![
                        JsonValue::String(String::from("snapshotRead")),
                        JsonValue::String(String::from("incrementalRead")),
                        JsonValue::String(String::from("componentWrite")),
                        JsonValue::String(String::from("revisionChecks")),
                    ]),
                ),
            ]))),
        })
    }

    fn subscribe(id: u64) -> JsonRpcMessage {
        JsonRpcMessage::Request(JsonRpcRequest {
            id: JsonRpcId::Number(id),
            method: String::from("prism/world/subscribe"),
            params: Some(JsonValue::Object(BTreeMap::from([
                (String::from("target"), JsonValue::String(String::from("7"))),
                (
                    String::from("fromRevision"),
                    JsonValue::Number(JsonNumber::Unsigned(0)),
                ),
            ]))),
        })
    }

    #[test]
    fn length_prefixed_stream_round_trips_and_bounds_frames() {
        let limits = TransportLimits {
            max_frame_bytes: 8,
            ..TransportLimits::default()
        };
        let mut writer = LengthPrefixedStream::new(Cursor::new(Vec::new()), limits);
        writer.write_frame(b"hello").expect("write");
        let bytes = writer.into_inner().into_inner();
        assert_eq!(&bytes[..4], &[0, 0, 0, 5]);
        let mut reader = LengthPrefixedStream::new(Cursor::new(bytes), limits);
        assert_eq!(reader.read_frame().expect("read"), Some(b"hello".to_vec()));
        assert_eq!(reader.read_frame().expect("eof"), None);
        assert_eq!(reader.limits().max_frame_bytes, 8);
        let mut oversized = LengthPrefixedStream::new(Cursor::new(Vec::new()), limits);
        assert!(matches!(
            oversized.write_frame(b"123456789"),
            Err(TransportError::FrameTooLarge { length: 9, max: 8 })
        ));
    }

    #[test]
    fn partial_header_and_payload_are_rejected() {
        let limits = TransportLimits::default();
        let mut header = LengthPrefixedStream::new(Cursor::new(vec![0, 0]), limits);
        assert!(matches!(
            header.read_frame(),
            Err(TransportError::UnexpectedEof {
                stage: FrameStage::Header,
                expected: 4,
                received: 2
            })
        ));
        let mut payload = LengthPrefixedStream::new(Cursor::new(vec![0, 0, 0, 4, 1, 2]), limits);
        assert!(matches!(
            payload.read_frame(),
            Err(TransportError::UnexpectedEof {
                stage: FrameStage::Payload,
                expected: 4,
                received: 2
            })
        ));
    }

    #[test]
    fn input_limit_and_io_timeouts_are_reported_before_protocol_decode() {
        let limits = TransportLimits {
            max_frame_bytes: 8,
            ..TransportLimits::default()
        };
        let mut oversized = LengthPrefixedStream::new(Cursor::new(vec![0, 0, 0, 9]), limits);
        assert!(matches!(
            oversized.read_frame(),
            Err(TransportError::FrameTooLarge { length: 9, max: 8 })
        ));

        let mut timed_out = LengthPrefixedStream::new(TimeoutStream, TransportLimits::default());
        let error = timed_out.read_frame().expect_err("timeout");
        assert!(error.is_timeout());
        let error = timed_out.write_frame(b"payload").expect_err("timeout");
        assert!(error.is_timeout());
    }

    #[test]
    fn tcp_loopback_serves_request_response_and_notification() {
        let limits = TransportLimits {
            read_timeout: Some(Duration::from_secs(5)),
            write_timeout: Some(Duration::from_secs(5)),
            connect_timeout: Some(Duration::from_secs(5)),
            ..TransportLimits::default()
        };
        let listener = TcpJsonRpcListener::bind("127.0.0.1:0", limits).expect("bind");
        let address = listener.local_addr().expect("address");
        let server_thread = thread::spawn(move || {
            let server = JsonRpcServer::new(endpoint());
            let mut session = listener.accept(server).expect("accept");
            assert!(session.serve_one().expect("initialize"));
            assert!(session.serve_one().expect("subscribe"));
            let changes = ChangeSet::new(
                Revision::ZERO,
                Revision::from_raw(1),
                vec![Change::CreateEntity {
                    entity: EntityId::from_raw(2),
                }],
            )
            .expect("changes");
            assert_eq!(
                session
                    .publish_changes(TargetId::from_raw(7), &changes)
                    .expect("publish"),
                1
            );
            session.serve()
        });

        let mut client = connect_tcp(address, limits).expect("connect");
        let initialize_exchange = client.request(initialize(1)).expect("initialize request");
        assert!(initialize_exchange.response.error.is_none());
        let subscribe_exchange = client.request(subscribe(2)).expect("subscribe request");
        assert!(subscribe_exchange.response.error.is_none());
        let notification = client
            .receive()
            .expect("notification read")
            .expect("notification");
        assert!(matches!(
            notification,
            JsonRpcMessage::Notification(JsonRpcNotification { method, .. })
                if method == "prism/world/changed"
        ));
        drop(client);
        server_thread.join().expect("server thread").expect("serve");
    }

    struct TimeoutStream;

    impl Read for TimeoutStream {
        fn read(&mut self, _buffer: &mut [u8]) -> io::Result<usize> {
            Err(io::Error::new(io::ErrorKind::TimedOut, "read timeout"))
        }
    }

    impl Write for TimeoutStream {
        fn write(&mut self, _buffer: &[u8]) -> io::Result<usize> {
            Err(io::Error::new(io::ErrorKind::TimedOut, "write timeout"))
        }

        fn flush(&mut self) -> io::Result<()> {
            Ok(())
        }
    }
}
