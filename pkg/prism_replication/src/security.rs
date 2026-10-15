//! Authenticated session transport for replication datagrams.
//!
//! This module deliberately does not ship a home-grown cipher or a default
//! key.  [`SessionSecurityProvider`] is the cryptographic seam: an application
//! supplies a reviewed implementation for authentication, key agreement and
//! authenticated encryption.  A plain [`crate::transport::Transport`] is not
//! accepted by the replication plugins; it must be wrapped in
//! [`SecureTransport`].
//!
//! The code here owns the protocol invariants that are independent of the
//! selected crypto implementation:
//!
//! - both sides exchange nonces and authenticated handshake material;
//! - application datagrams are authenticated and bound to a direction and
//!   monotonically increasing sequence number;
//! - authenticated duplicates and packets outside the replay window are
//!   discarded before they reach multiplexing;
//! - sequence exhaustion, malformed handshakes and provider failures reject
//!   the session instead of silently falling back to plaintext.

use alloc::collections::VecDeque;
use alloc::vec::Vec;

use crate::transport::{Transport, TransportSecurity};

/// Number of bytes in each endpoint nonce.
pub const SESSION_NONCE_LEN: usize = 32;
/// Maximum authentication material carried by one handshake message.
pub const MAX_HANDSHAKE_AUTH: usize = 512;
/// Maximum encoded handshake message accepted by this seam.
pub const MAX_HANDSHAKE_FRAME: usize = 1_024;
/// Number of sequence positions retained by the anti-replay window.
pub const REPLAY_WINDOW_BITS: u8 = 64;

const WIRE_MAGIC: [u8; 2] = *b"PS";
const WIRE_VERSION: u8 = 1;
const KIND_HELLO: u8 = 0;
const KIND_ACCEPT: u8 = 1;
const KIND_DATA: u8 = 2;
const HANDSHAKE_HEADER_LEN: usize = 2 + 1 + 1 + 2 + SESSION_NONCE_LEN + 2;
const DATA_HEADER_LEN: usize = 2 + 1 + 1 + 2 + 8;
const MAX_PENDING_DATAGRAMS: usize = 256;

/// Errors raised by the security seam.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SecurityError {
    /// No provider was supplied, or an unprotected transport was passed to a
    /// replication session.
    ProviderRequired,
    /// The provider rejected credentials, a transcript, or a ciphertext.
    ProviderRejected,
    /// The provider returned an invalid or empty nonce.
    InvalidNonce,
    /// The provider id or security wire version did not match.
    VersionMismatch,
    /// A security frame was truncated, oversized, or used an unknown kind.
    MalformedFrame,
    /// A packet arrived before the authenticated session was established.
    NotEstablished,
    /// A sequence number would wrap and reuse a nonce/sequence pair.
    SequenceExhausted,
    /// The peer attempted to use a different nonce pair on an existing link.
    NonceMismatch,
    /// The bounded queue of application datagrams waiting for the handshake is
    /// full.
    HandshakeBackpressure,
}

/// A direction used as authenticated-encryption associated data.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SecurityDirection {
    /// Bytes sent by the client and opened by the server.
    ClientToServer,
    /// Bytes sent by the server and opened by the client.
    ServerToClient,
}

impl SecurityDirection {
    #[cfg(feature = "crypto")]
    const fn wire(self) -> u8 {
        match self {
            Self::ClientToServer => 0,
            Self::ServerToClient => 1,
        }
    }
}

/// A fixed-size endpoint nonce. The provider must fill it with cryptographically
/// random, never-reused bytes for every new client session.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SessionNonce(pub [u8; SESSION_NONCE_LEN]);

impl SessionNonce {
    fn is_zero(self) -> bool {
        self.0.iter().all(|byte| *byte == 0)
    }
}

/// Client's authenticated handshake offer.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ClientHello {
    /// Provider/algorithm identifier. It is part of the transcript.
    pub provider_id: u16,
    /// Fresh client nonce.
    pub nonce: SessionNonce,
    /// Provider-defined authenticated identity/credential material.
    pub authenticator: Vec<u8>,
}

/// Server's authenticated handshake response.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ServerAccept {
    /// Provider/algorithm identifier, repeated to prevent downgrade.
    pub provider_id: u16,
    /// Fresh server nonce.
    pub nonce: SessionNonce,
    /// Provider-defined proof and key-agreement material.
    pub authenticator: Vec<u8>,
}

/// The application-provided authentication and AEAD/key-agreement contract.
///
/// Implementations must use a reviewed cryptographic library. In particular,
/// `authenticator` is not a hash placeholder and `seal`/`open` must provide
/// authenticated encryption (or an equivalent integrity-protected encrypted
/// construction). The replication crate intentionally has no insecure default.
pub trait SessionSecurityProvider: Send + Sync + 'static {
    /// Stable provider/algorithm id negotiated in the cleartext handshake.
    fn provider_id(&self) -> u16;

    /// Maximum bytes added by [`Self::seal`] to one plaintext. This is used to
    /// reserve MTU and congestion budget before a datagram is queued.
    fn max_seal_overhead(&self) -> usize;

    /// Create a fresh client nonce and authenticated client offer.
    fn create_client_hello(&mut self) -> Result<ClientHello, SecurityError>;

    /// Verify the client offer and create an authenticated server response.
    /// Returning `Ok` commits the server provider to this session.
    fn accept_client_hello(&mut self, hello: &ClientHello) -> Result<ServerAccept, SecurityError>;

    /// Verify the server response and commit the client provider to this
    /// session. The implementation must authenticate the complete nonce pair
    /// and handshake transcript before returning `Ok`.
    fn finish_client(
        &mut self,
        hello: &ClientHello,
        accept: &ServerAccept,
    ) -> Result<(), SecurityError>;

    /// Encrypt and authenticate one datagram. `aad` includes the security
    /// header, while the nonce pair and sequence are supplied separately so a
    /// provider can derive a unique AEAD nonce without serializing secrets.
    fn seal(
        &mut self,
        direction: SecurityDirection,
        client_nonce: SessionNonce,
        server_nonce: SessionNonce,
        sequence: u64,
        aad: &[u8],
        plaintext: &[u8],
        out: &mut Vec<u8>,
    ) -> Result<(), SecurityError>;

    /// Verify and decrypt one datagram. Authentication must be checked before
    /// the caller accepts its sequence number into the replay window.
    fn open(
        &mut self,
        direction: SecurityDirection,
        client_nonce: SessionNonce,
        server_nonce: SessionNonce,
        sequence: u64,
        aad: &[u8],
        ciphertext: &[u8],
        out: &mut Vec<u8>,
    ) -> Result<(), SecurityError>;
}

/// Provider id for [`RingPskProvider`].
#[cfg(feature = "crypto")]
pub const RING_PSK_PROVIDER_ID: u16 = 0x5250;

/// A production-ready PSK provider backed by `ring`.
///
/// The handshake authenticates both endpoint nonces with HMAC-SHA256 and
/// derives a per-session AES-256-GCM key from the PSK and the complete nonce
/// pair. Every payload nonce is derived from the negotiated session and the
/// direction/sequence tuple, while the security header is authenticated as
/// associated data. The application owns the PSK and must rotate it by
/// creating a new provider/session; this type never embeds a default secret.
#[cfg(feature = "crypto")]
pub struct RingPskProvider {
    psk: [u8; 32],
    client_nonce: Option<SessionNonce>,
    server_nonce: Option<SessionNonce>,
    session_key: Option<[u8; 32]>,
}

#[cfg(feature = "crypto")]
impl RingPskProvider {
    /// Construct a provider from an application-owned 256-bit pre-shared key.
    #[must_use]
    pub const fn new(psk: [u8; 32]) -> Self {
        Self {
            psk,
            client_nonce: None,
            server_nonce: None,
            session_key: None,
        }
    }

    fn mac(&self, domain: &[u8], fields: &[&[u8]]) -> [u8; 32] {
        use ring::hmac;
        let key = hmac::Key::new(hmac::HMAC_SHA256, &self.psk);
        let mut context = Vec::new();
        context.extend_from_slice(domain);
        for field in fields {
            context.extend_from_slice(field);
        }
        let tag = hmac::sign(&key, &context);
        let mut output = [0; 32];
        output.copy_from_slice(tag.as_ref());
        output
    }

    fn verify_mac(&self, expected: &[u8], domain: &[u8], fields: &[&[u8]]) -> bool {
        use ring::hmac;
        let key = hmac::Key::new(hmac::HMAC_SHA256, &self.psk);
        let mut context = Vec::new();
        context.extend_from_slice(domain);
        for field in fields {
            context.extend_from_slice(field);
        }
        hmac::verify(&key, &context, expected).is_ok()
    }

    fn derive_session_key(
        &self,
        client_nonce: SessionNonce,
        server_nonce: SessionNonce,
    ) -> [u8; 32] {
        self.mac(
            b"prism/replication/session-key/v1",
            &[
                &RING_PSK_PROVIDER_ID.to_le_bytes(),
                &client_nonce.0,
                &server_nonce.0,
            ],
        )
    }

    fn payload_nonce(direction: SecurityDirection, sequence: u64) -> ring::aead::Nonce {
        let mut bytes = [0; 12];
        bytes[0] = direction.wire();
        bytes[4..].copy_from_slice(&sequence.to_be_bytes());
        ring::aead::Nonce::assume_unique_for_key(bytes)
    }

    fn key(&self) -> Result<ring::aead::LessSafeKey, SecurityError> {
        let key = self.session_key.ok_or(SecurityError::NotEstablished)?;
        let unbound = ring::aead::UnboundKey::new(&ring::aead::AES_256_GCM, &key)
            .map_err(|_| SecurityError::ProviderRejected)?;
        Ok(ring::aead::LessSafeKey::new(unbound))
    }

    fn fresh_nonce() -> Result<SessionNonce, SecurityError> {
        use ring::rand::SecureRandom;
        let mut nonce = [0; SESSION_NONCE_LEN];
        ring::rand::SystemRandom::new()
            .fill(&mut nonce)
            .map_err(|_| SecurityError::InvalidNonce)?;
        let nonce = SessionNonce(nonce);
        if nonce.is_zero() {
            return Err(SecurityError::InvalidNonce);
        }
        Ok(nonce)
    }
}

#[cfg(feature = "crypto")]
impl SessionSecurityProvider for RingPskProvider {
    fn provider_id(&self) -> u16 {
        RING_PSK_PROVIDER_ID
    }

    fn max_seal_overhead(&self) -> usize {
        ring::aead::AES_256_GCM.tag_len()
    }

    fn create_client_hello(&mut self) -> Result<ClientHello, SecurityError> {
        let nonce = Self::fresh_nonce()?;
        let provider_id = self.provider_id().to_le_bytes();
        let authenticator = self.mac(
            b"prism/replication/client-hello/v1",
            &[&provider_id, &nonce.0],
        );
        self.client_nonce = Some(nonce);
        self.server_nonce = None;
        self.session_key = None;
        Ok(ClientHello {
            provider_id: self.provider_id(),
            nonce,
            authenticator: authenticator.to_vec(),
        })
    }

    fn accept_client_hello(&mut self, hello: &ClientHello) -> Result<ServerAccept, SecurityError> {
        if hello.provider_id != self.provider_id()
            || hello.authenticator.len() != 32
            || !self.verify_mac(
                &hello.authenticator,
                b"prism/replication/client-hello/v1",
                &[&hello.provider_id.to_le_bytes(), &hello.nonce.0],
            )
        {
            return Err(SecurityError::ProviderRejected);
        }
        let server_nonce = Self::fresh_nonce()?;
        let provider_id = self.provider_id().to_le_bytes();
        let authenticator = self.mac(
            b"prism/replication/server-accept/v1",
            &[&provider_id, &hello.nonce.0, &server_nonce.0],
        );
        self.client_nonce = Some(hello.nonce);
        self.server_nonce = Some(server_nonce);
        self.session_key = Some(self.derive_session_key(hello.nonce, server_nonce));
        Ok(ServerAccept {
            provider_id: self.provider_id(),
            nonce: server_nonce,
            authenticator: authenticator.to_vec(),
        })
    }

    fn finish_client(
        &mut self,
        hello: &ClientHello,
        accept: &ServerAccept,
    ) -> Result<(), SecurityError> {
        if hello.provider_id != self.provider_id()
            || accept.provider_id != self.provider_id()
            || accept.authenticator.len() != 32
            || !self.verify_mac(
                &accept.authenticator,
                b"prism/replication/server-accept/v1",
                &[
                    &self.provider_id().to_le_bytes(),
                    &hello.nonce.0,
                    &accept.nonce.0,
                ],
            )
        {
            return Err(SecurityError::ProviderRejected);
        }
        self.client_nonce = Some(hello.nonce);
        self.server_nonce = Some(accept.nonce);
        self.session_key = Some(self.derive_session_key(hello.nonce, accept.nonce));
        Ok(())
    }

    fn seal(
        &mut self,
        direction: SecurityDirection,
        _client_nonce: SessionNonce,
        _server_nonce: SessionNonce,
        sequence: u64,
        aad: &[u8],
        plaintext: &[u8],
        out: &mut Vec<u8>,
    ) -> Result<(), SecurityError> {
        let key = self.key()?;
        out.clear();
        out.extend_from_slice(plaintext);
        key.seal_in_place_append_tag(
            Self::payload_nonce(direction, sequence),
            ring::aead::Aad::from(aad),
            out,
        )
        .map_err(|_| SecurityError::ProviderRejected)
    }

    fn open(
        &mut self,
        direction: SecurityDirection,
        _client_nonce: SessionNonce,
        _server_nonce: SessionNonce,
        sequence: u64,
        aad: &[u8],
        ciphertext: &[u8],
        out: &mut Vec<u8>,
    ) -> Result<(), SecurityError> {
        let key = self.key()?;
        out.clear();
        out.extend_from_slice(ciphertext);
        let plaintext = key
            .open_in_place(
                Self::payload_nonce(direction, sequence),
                ring::aead::Aad::from(aad),
                out,
            )
            .map_err(|_| SecurityError::ProviderRejected)?;
        let length = plaintext.len();
        out.truncate(length);
        Ok(())
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ReplayResult {
    Fresh,
    Duplicate,
    TooOld,
}

#[derive(Clone, Copy, Debug, Default)]
struct ReplayWindow {
    highest: Option<u64>,
    bits: u64,
}

impl ReplayWindow {
    fn observe(&mut self, sequence: u64) -> ReplayResult {
        let Some(highest) = self.highest else {
            self.highest = Some(sequence);
            self.bits = 1;
            return ReplayResult::Fresh;
        };
        if sequence > highest {
            let shift = sequence - highest;
            self.bits = if shift >= u64::from(REPLAY_WINDOW_BITS) {
                1
            } else {
                (self.bits << shift) | 1
            };
            self.highest = Some(sequence);
            return ReplayResult::Fresh;
        }
        let distance = highest - sequence;
        if distance >= u64::from(REPLAY_WINDOW_BITS) {
            return ReplayResult::TooOld;
        }
        let mask = 1u64 << distance;
        if self.bits & mask != 0 {
            ReplayResult::Duplicate
        } else {
            self.bits |= mask;
            ReplayResult::Fresh
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Role {
    Client,
    Server,
}

impl Role {
    const fn send_direction(self) -> SecurityDirection {
        match self {
            Self::Client => SecurityDirection::ClientToServer,
            Self::Server => SecurityDirection::ServerToClient,
        }
    }

    const fn recv_direction(self) -> SecurityDirection {
        match self {
            Self::Client => SecurityDirection::ServerToClient,
            Self::Server => SecurityDirection::ClientToServer,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum State {
    ClientHelloSent,
    ServerWaiting,
    Established,
    Rejected,
}

/// A transport wrapper that authenticates and protects every datagram sent by
/// the replication multiplex. It is generic so an application can supply its
/// own socket and crypto provider without pulling either into this crate.
pub struct SecureTransport<T, P> {
    inner: T,
    provider: P,
    role: Role,
    state: State,
    client_hello: Option<ClientHello>,
    server_accept: Option<ServerAccept>,
    client_nonce: Option<SessionNonce>,
    server_nonce: Option<SessionNonce>,
    next_send: u64,
    replay: ReplayWindow,
    pending: VecDeque<Vec<u8>>,
    inbound: VecDeque<Vec<u8>>,
    poll_ticks: u32,
    last_error: Option<SecurityError>,
    replayed: u64,
}

impl<T, P> SecureTransport<T, P>
where
    T: Transport,
    P: SessionSecurityProvider,
{
    /// Create a client endpoint. The provider's fresh nonce and authenticator
    /// are generated before the endpoint can send application bytes.
    pub fn new_client(inner: T, mut provider: P) -> Result<Self, SecurityError> {
        let hello = provider.create_client_hello()?;
        validate_hello(&hello, provider.provider_id())?;
        // Touch the transport here so a provider-backed wrapper never silently
        // accepts a wrapper around another unprotected secure layer.
        if !inner.supports_secure_session()
            && inner.security_state() != TransportSecurity::Unprotected
        {
            return Err(SecurityError::ProviderRejected);
        }
        let mut transport = Self {
            inner,
            provider,
            role: Role::Client,
            state: State::ClientHelloSent,
            client_nonce: Some(hello.nonce),
            client_hello: Some(hello),
            server_accept: None,
            server_nonce: None,
            next_send: 0,
            replay: ReplayWindow::default(),
            pending: VecDeque::new(),
            inbound: VecDeque::new(),
            poll_ticks: 0,
            last_error: None,
            replayed: 0,
        };
        transport.send_handshake();
        Ok(transport)
    }

    /// Create a server endpoint. It waits for an authenticated client offer.
    #[must_use]
    pub fn new_server(inner: T, provider: P) -> Self {
        Self {
            inner,
            provider,
            role: Role::Server,
            state: State::ServerWaiting,
            client_hello: None,
            server_accept: None,
            client_nonce: None,
            server_nonce: None,
            next_send: 0,
            replay: ReplayWindow::default(),
            pending: VecDeque::new(),
            inbound: VecDeque::new(),
            poll_ticks: 0,
            last_error: None,
            replayed: 0,
        }
    }

    /// Access the wrapped transport for socket setup or diagnostics.
    pub fn inner_mut(&mut self) -> &mut T {
        &mut self.inner
    }

    /// Number of authenticated duplicates/old packets discarded.
    #[must_use]
    pub fn replayed_datagrams(&self) -> u64 {
        self.replayed
    }

    /// Whether the authenticated handshake has completed.
    #[must_use]
    pub fn established(&self) -> bool {
        self.state == State::Established
    }

    fn set_error(&mut self, error: SecurityError) {
        self.last_error.get_or_insert(error);
        self.state = State::Rejected;
        self.pending.clear();
    }

    fn send_handshake(&mut self) {
        let Some(hello) = self.client_hello.as_ref() else {
            return;
        };
        self.inner.send(&encode_hello(hello));
    }

    fn send_accept(&mut self) {
        let Some(accept) = self.server_accept.as_ref() else {
            return;
        };
        self.inner.send(&encode_accept(accept));
    }

    fn flush_pending(&mut self) {
        if self.state != State::Established {
            return;
        }
        while let Some(bytes) = self.pending.pop_front() {
            if let Err(error) = self.send_data(&bytes) {
                self.set_error(error);
                return;
            }
        }
    }

    fn send_data(&mut self, plaintext: &[u8]) -> Result<(), SecurityError> {
        if self.state != State::Established {
            return Err(SecurityError::NotEstablished);
        }
        let sequence = self.next_send;
        self.next_send = sequence
            .checked_add(1)
            .ok_or(SecurityError::SequenceExhausted)?;
        let provider_id = self.provider.provider_id();
        let mut frame = Vec::with_capacity(DATA_HEADER_LEN + plaintext.len());
        frame.extend_from_slice(&WIRE_MAGIC);
        frame.push(WIRE_VERSION);
        frame.push(KIND_DATA);
        frame.extend_from_slice(&provider_id.to_le_bytes());
        frame.extend_from_slice(&sequence.to_le_bytes());
        let aad = frame.clone();
        let client_nonce = self.client_nonce.ok_or(SecurityError::NotEstablished)?;
        let server_nonce = self.server_nonce.ok_or(SecurityError::NotEstablished)?;
        let mut sealed = Vec::with_capacity(
            plaintext
                .len()
                .saturating_add(self.provider.max_seal_overhead()),
        );
        self.provider.seal(
            self.role.send_direction(),
            client_nonce,
            server_nonce,
            sequence,
            &aad,
            plaintext,
            &mut sealed,
        )?;
        frame.extend_from_slice(&sealed);
        self.inner.send(&frame);
        Ok(())
    }

    fn process_frame(&mut self, frame: &[u8]) {
        if frame.len() > MAX_HANDSHAKE_FRAME
            || frame.len() < 4
            || frame[..2] != WIRE_MAGIC
            || frame[2] != WIRE_VERSION
        {
            self.set_error(SecurityError::MalformedFrame);
            return;
        }
        match frame[3] {
            KIND_HELLO => self.process_hello(frame),
            KIND_ACCEPT => self.process_accept(frame),
            KIND_DATA => self.process_data(frame),
            _ => self.set_error(SecurityError::MalformedFrame),
        }
    }

    fn process_hello(&mut self, frame: &[u8]) {
        if self.role != Role::Server || self.state == State::Rejected {
            return;
        }
        let Some(hello) = decode_hello(frame) else {
            self.set_error(SecurityError::MalformedFrame);
            return;
        };
        if validate_hello(&hello, self.provider.provider_id()).is_err() {
            self.set_error(SecurityError::VersionMismatch);
            return;
        }
        if self.state == State::Established {
            if self.client_hello.as_ref() == Some(&hello) {
                self.send_accept();
            } else {
                self.set_error(SecurityError::NonceMismatch);
            }
            return;
        }
        let accept = match self.provider.accept_client_hello(&hello) {
            Ok(accept) => accept,
            Err(error) => {
                self.set_error(error);
                return;
            }
        };
        if validate_accept(&accept, self.provider.provider_id(), hello.nonce).is_err() {
            self.set_error(SecurityError::InvalidNonce);
            return;
        }
        self.client_nonce = Some(hello.nonce);
        self.server_nonce = Some(accept.nonce);
        self.client_hello = Some(hello);
        self.server_accept = Some(accept);
        self.state = State::Established;
        self.send_accept();
        self.flush_pending();
    }

    fn process_accept(&mut self, frame: &[u8]) {
        if self.role != Role::Client || self.state != State::ClientHelloSent {
            return;
        }
        let Some(accept) = decode_accept(frame) else {
            self.set_error(SecurityError::MalformedFrame);
            return;
        };
        let Some(hello) = self.client_hello.as_ref() else {
            self.set_error(SecurityError::NotEstablished);
            return;
        };
        if validate_accept(&accept, self.provider.provider_id(), hello.nonce).is_err() {
            self.set_error(SecurityError::VersionMismatch);
            return;
        }
        if let Err(error) = self.provider.finish_client(hello, &accept) {
            self.set_error(error);
            return;
        }
        self.server_nonce = Some(accept.nonce);
        self.server_accept = Some(accept);
        self.state = State::Established;
        self.flush_pending();
    }

    fn process_data(&mut self, frame: &[u8]) {
        if self.state != State::Established || frame.len() < DATA_HEADER_LEN {
            self.set_error(if self.state == State::Established {
                SecurityError::MalformedFrame
            } else {
                SecurityError::NotEstablished
            });
            return;
        }
        let provider_id = u16::from_le_bytes([frame[4], frame[5]]);
        if provider_id != self.provider.provider_id() {
            self.set_error(SecurityError::VersionMismatch);
            return;
        }
        let Ok(sequence_bytes) = <[u8; 8]>::try_from(&frame[6..14]) else {
            self.set_error(SecurityError::MalformedFrame);
            return;
        };
        let sequence = u64::from_le_bytes(sequence_bytes);
        let Some(client_nonce) = self.client_nonce else {
            self.set_error(SecurityError::NotEstablished);
            return;
        };
        let Some(server_nonce) = self.server_nonce else {
            self.set_error(SecurityError::NotEstablished);
            return;
        };
        let mut plaintext = Vec::new();
        if self
            .provider
            .open(
                self.role.recv_direction(),
                client_nonce,
                server_nonce,
                sequence,
                &frame[..DATA_HEADER_LEN],
                &frame[DATA_HEADER_LEN..],
                &mut plaintext,
            )
            .is_err()
        {
            self.set_error(SecurityError::ProviderRejected);
            return;
        }
        match self.replay.observe(sequence) {
            ReplayResult::Fresh => self.inbound.push_back(plaintext),
            ReplayResult::Duplicate | ReplayResult::TooOld => {
                self.replayed = self.replayed.saturating_add(1);
            }
        }
    }

    fn drive(&mut self) {
        self.poll_ticks = self.poll_ticks.wrapping_add(1);
        if self.state == State::ClientHelloSent && self.poll_ticks % 4 == 0 {
            self.send_handshake();
        }
        if self.state == State::Established {
            self.flush_pending();
        }
    }
}

impl<T, P> Transport for SecureTransport<T, P>
where
    T: Transport,
    P: SessionSecurityProvider,
{
    fn send(&mut self, bytes: &[u8]) {
        if self.state == State::Rejected {
            return;
        }
        if self.state != State::Established {
            if self.pending.len() >= MAX_PENDING_DATAGRAMS {
                self.set_error(SecurityError::HandshakeBackpressure);
                return;
            }
            self.pending.push_back(bytes.to_vec());
            self.send_handshake();
            return;
        }
        if let Err(error) = self.send_data(bytes) {
            self.set_error(error);
        }
    }

    fn recv(&mut self) -> Option<Vec<u8>> {
        if let Some(bytes) = self.inbound.pop_front() {
            return Some(bytes);
        }
        self.drive();
        for _ in 0..MAX_PENDING_DATAGRAMS {
            let Some(frame) = self.inner.recv() else {
                break;
            };
            self.process_frame(&frame);
            if self.state == State::Rejected {
                break;
            }
            if let Some(bytes) = self.inbound.pop_front() {
                return Some(bytes);
            }
        }
        self.inbound.pop_front()
    }

    fn supports_secure_session(&self) -> bool {
        true
    }

    fn security_state(&self) -> TransportSecurity {
        match self.state {
            State::ClientHelloSent | State::ServerWaiting => TransportSecurity::Pending,
            State::Established => TransportSecurity::Established,
            State::Rejected => TransportSecurity::Rejected,
        }
    }

    fn security_overhead(&self) -> usize {
        DATA_HEADER_LEN.saturating_add(self.provider.max_seal_overhead())
    }

    fn take_security_error(&mut self) -> Option<SecurityError> {
        self.last_error.take()
    }
}

fn validate_hello(hello: &ClientHello, provider_id: u16) -> Result<(), SecurityError> {
    if hello.provider_id != provider_id || hello.nonce.is_zero() {
        return Err(if hello.nonce.is_zero() {
            SecurityError::InvalidNonce
        } else {
            SecurityError::VersionMismatch
        });
    }
    if hello.authenticator.len() > MAX_HANDSHAKE_AUTH {
        return Err(SecurityError::MalformedFrame);
    }
    Ok(())
}

fn validate_accept(
    accept: &ServerAccept,
    provider_id: u16,
    client_nonce: SessionNonce,
) -> Result<(), SecurityError> {
    if accept.provider_id != provider_id {
        return Err(SecurityError::VersionMismatch);
    }
    if accept.nonce.is_zero() || accept.nonce == client_nonce {
        return Err(SecurityError::InvalidNonce);
    }
    if accept.authenticator.len() > MAX_HANDSHAKE_AUTH {
        return Err(SecurityError::MalformedFrame);
    }
    Ok(())
}

fn encode_hello(hello: &ClientHello) -> Vec<u8> {
    encode_handshake(
        KIND_HELLO,
        hello.provider_id,
        hello.nonce,
        &hello.authenticator,
    )
}

fn encode_accept(accept: &ServerAccept) -> Vec<u8> {
    encode_handshake(
        KIND_ACCEPT,
        accept.provider_id,
        accept.nonce,
        &accept.authenticator,
    )
}

fn encode_handshake(kind: u8, provider_id: u16, nonce: SessionNonce, auth: &[u8]) -> Vec<u8> {
    let mut frame = Vec::with_capacity(HANDSHAKE_HEADER_LEN + auth.len());
    frame.extend_from_slice(&WIRE_MAGIC);
    frame.push(WIRE_VERSION);
    frame.push(kind);
    frame.extend_from_slice(&provider_id.to_le_bytes());
    frame.extend_from_slice(&nonce.0);
    frame.extend_from_slice(&(auth.len() as u16).to_le_bytes());
    frame.extend_from_slice(auth);
    frame
}

fn decode_hello(frame: &[u8]) -> Option<ClientHello> {
    decode_handshake(frame, KIND_HELLO).map(|(provider_id, nonce, authenticator)| ClientHello {
        provider_id,
        nonce,
        authenticator,
    })
}

fn decode_accept(frame: &[u8]) -> Option<ServerAccept> {
    decode_handshake(frame, KIND_ACCEPT).map(|(provider_id, nonce, authenticator)| ServerAccept {
        provider_id,
        nonce,
        authenticator,
    })
}

fn decode_handshake(frame: &[u8], kind: u8) -> Option<(u16, SessionNonce, Vec<u8>)> {
    if frame.len() > MAX_HANDSHAKE_FRAME
        || frame.len() < HANDSHAKE_HEADER_LEN
        || frame[..2] != WIRE_MAGIC
        || frame[2] != WIRE_VERSION
        || frame[3] != kind
    {
        return None;
    }
    let provider_id = u16::from_le_bytes([frame[4], frame[5]]);
    let mut nonce = [0; SESSION_NONCE_LEN];
    nonce.copy_from_slice(&frame[6..6 + SESSION_NONCE_LEN]);
    let auth_len_at = 6 + SESSION_NONCE_LEN;
    let auth_len = usize::from(u16::from_le_bytes([
        frame[auth_len_at],
        frame[auth_len_at + 1],
    ]));
    if auth_len > MAX_HANDSHAKE_AUTH || frame.len() != HANDSHAKE_HEADER_LEN + auth_len {
        return None;
    }
    Some((
        provider_id,
        SessionNonce(nonce),
        frame[HANDSHAKE_HEADER_LEN..].to_vec(),
    ))
}

#[cfg(test)]
pub(crate) mod test_support {
    use super::*;

    /// Deterministic provider used only by this crate's transport/plugin tests.
    /// It is intentionally not exported in non-test builds and is not a
    /// production cryptographic implementation.
    pub(crate) struct TestProvider {
        pub(crate) secret: u8,
        pub(crate) nonce: u8,
        pub(crate) client_nonce: Option<SessionNonce>,
        pub(crate) server_nonce: Option<SessionNonce>,
    }

    impl TestProvider {
        pub(crate) fn new(secret: u8, nonce: u8) -> Self {
            Self {
                secret,
                nonce,
                client_nonce: None,
                server_nonce: None,
            }
        }
    }

    pub(crate) use TestProvider as Provider;
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::security::test_support::TestProvider;
    use alloc::collections::VecDeque;
    use alloc::vec;

    #[derive(Default)]
    struct MockTransport {
        outbox: VecDeque<Vec<u8>>,
        inbox: VecDeque<Vec<u8>>,
    }

    impl Transport for MockTransport {
        fn send(&mut self, bytes: &[u8]) {
            self.outbox.push_back(bytes.to_vec());
        }

        fn recv(&mut self) -> Option<Vec<u8>> {
            self.inbox.pop_front()
        }
    }

    impl SessionSecurityProvider for TestProvider {
        fn provider_id(&self) -> u16 {
            7
        }

        fn max_seal_overhead(&self) -> usize {
            1
        }

        fn create_client_hello(&mut self) -> Result<ClientHello, SecurityError> {
            let nonce = SessionNonce([self.nonce; SESSION_NONCE_LEN]);
            self.client_nonce = Some(nonce);
            Ok(ClientHello {
                provider_id: self.provider_id(),
                nonce,
                authenticator: vec![self.secret],
            })
        }

        fn accept_client_hello(
            &mut self,
            hello: &ClientHello,
        ) -> Result<ServerAccept, SecurityError> {
            if hello.authenticator != [self.secret] {
                return Err(SecurityError::ProviderRejected);
            }
            // The test provider uses distinct endpoint nonce values; a real
            // provider must use an independent CSPRNG for each endpoint.
            let nonce = SessionNonce([self.nonce.wrapping_add(1); SESSION_NONCE_LEN]);
            self.client_nonce = Some(hello.nonce);
            self.server_nonce = Some(nonce);
            Ok(ServerAccept {
                provider_id: self.provider_id(),
                nonce,
                authenticator: vec![self.secret ^ 0xA5],
            })
        }

        fn finish_client(
            &mut self,
            hello: &ClientHello,
            accept: &ServerAccept,
        ) -> Result<(), SecurityError> {
            if accept.authenticator != [self.secret ^ 0xA5] {
                return Err(SecurityError::ProviderRejected);
            }
            self.client_nonce = Some(hello.nonce);
            self.server_nonce = Some(accept.nonce);
            Ok(())
        }

        fn seal(
            &mut self,
            direction: SecurityDirection,
            client_nonce: SessionNonce,
            server_nonce: SessionNonce,
            sequence: u64,
            aad: &[u8],
            plaintext: &[u8],
            out: &mut Vec<u8>,
        ) -> Result<(), SecurityError> {
            let mut tag = self.secret ^ direction.wire();
            tag ^= client_nonce.0[0] ^ server_nonce.0[0] ^ sequence as u8;
            for byte in aad {
                tag ^= *byte;
            }
            for byte in plaintext {
                tag ^= *byte;
                out.push(*byte ^ self.secret);
            }
            out.push(tag);
            Ok(())
        }

        fn open(
            &mut self,
            direction: SecurityDirection,
            client_nonce: SessionNonce,
            server_nonce: SessionNonce,
            sequence: u64,
            aad: &[u8],
            ciphertext: &[u8],
            out: &mut Vec<u8>,
        ) -> Result<(), SecurityError> {
            let Some((&tag, encrypted)) = ciphertext.split_last() else {
                return Err(SecurityError::ProviderRejected);
            };
            let mut expected = self.secret ^ direction.wire();
            expected ^= client_nonce.0[0] ^ server_nonce.0[0] ^ sequence as u8;
            for byte in aad {
                expected ^= *byte;
            }
            for byte in encrypted {
                let plain = *byte ^ self.secret;
                expected ^= plain;
                out.push(plain);
            }
            if expected != tag {
                out.clear();
                return Err(SecurityError::ProviderRejected);
            }
            Ok(())
        }
    }

    fn connect() -> (
        SecureTransport<MockTransport, test_support::Provider>,
        SecureTransport<MockTransport, test_support::Provider>,
    ) {
        let mut client = SecureTransport::new_client(
            MockTransport::default(),
            test_support::Provider::new(0x31, 1),
        )
        .unwrap();
        let mut server = SecureTransport::new_server(
            MockTransport::default(),
            test_support::Provider::new(0x31, 9),
        );
        server.inner.inbox.extend(client.inner.outbox.drain(..));
        assert!(server.recv().is_none());
        client.inner.inbox.extend(server.inner.outbox.drain(..));
        assert!(client.recv().is_none());
        assert!(client.established());
        assert!(server.established());
        (client, server)
    }

    #[test]
    fn handshake_authenticates_and_protects_payload() {
        let (mut client, mut server) = connect();
        client.send(b"hello");
        let frame = client.inner.outbox.pop_front().unwrap();
        assert_ne!(&frame[DATA_HEADER_LEN..frame.len() - 1], b"hello");
        server.inner.inbox.push_back(frame.clone());
        assert_eq!(server.recv(), Some(b"hello".to_vec()));
        server.inner.inbox.push_back(frame);
        assert_eq!(server.recv(), None, "authenticated duplicate is discarded");
        assert_eq!(server.replayed_datagrams(), 1);
    }

    #[test]
    fn tampering_rejects_the_session_before_delivery() {
        let (mut client, mut server) = connect();
        client.send(b"hello");
        let mut frame = client.inner.outbox.pop_front().unwrap();
        let last = frame.len() - 1;
        frame[last] ^= 0x80;
        server.inner.inbox.push_back(frame);
        assert_eq!(server.recv(), None);
        assert_eq!(server.security_state(), TransportSecurity::Rejected);
        assert_eq!(
            server.take_security_error(),
            Some(SecurityError::ProviderRejected)
        );
    }

    #[test]
    fn replay_window_rejects_old_sequences_without_plaintext_fallback() {
        let (mut client, mut server) = connect();
        let first = {
            client.send(b"x");
            client.inner.outbox.pop_front().unwrap()
        };
        server.inner.inbox.push_back(first.clone());
        assert_eq!(server.recv(), Some(b"x".to_vec()));
        for _ in 0..REPLAY_WINDOW_BITS {
            client.send(b"x");
            let frame = client.inner.outbox.pop_front().unwrap();
            server.inner.inbox.push_back(frame);
            assert_eq!(server.recv(), Some(b"x".to_vec()));
        }
        server.inner.inbox.push_back(first);
        assert_eq!(
            server.recv(),
            None,
            "an authenticated packet outside the window is discarded"
        );
        assert_eq!(server.replayed_datagrams(), 1);
    }

    #[test]
    fn sequence_exhaustion_rejects_before_nonce_reuse() {
        let (mut client, mut server) = connect();
        client.next_send = u64::MAX;
        client.send(b"last");
        assert_eq!(client.security_state(), TransportSecurity::Rejected);
        assert_eq!(
            client.take_security_error(),
            Some(SecurityError::SequenceExhausted)
        );
        assert!(client.inner.outbox.is_empty());
        assert!(server.recv().is_none());
    }

    #[test]
    fn invalid_nonce_is_rejected() {
        let provider = test_support::Provider::new(1, 0);
        assert!(matches!(
            SecureTransport::new_client(MockTransport::default(), provider),
            Err(SecurityError::InvalidNonce)
        ));
    }

    #[cfg(feature = "crypto")]
    fn connect_ring() -> (
        SecureTransport<MockTransport, RingPskProvider>,
        SecureTransport<MockTransport, RingPskProvider>,
    ) {
        let psk = [0x5a; 32];
        let mut client =
            SecureTransport::new_client(MockTransport::default(), RingPskProvider::new(psk))
                .expect("ring client hello should be generated");
        let mut server =
            SecureTransport::new_server(MockTransport::default(), RingPskProvider::new(psk));
        server.inner.inbox.extend(client.inner.outbox.drain(..));
        assert!(server.recv().is_none());
        client.inner.inbox.extend(server.inner.outbox.drain(..));
        assert!(client.recv().is_none());
        assert!(client.established());
        assert!(server.established());
        (client, server)
    }

    #[cfg(feature = "crypto")]
    #[test]
    fn ring_provider_round_trips_both_directions_and_rejects_tampering() {
        let (mut client, mut server) = connect_ring();

        client.send(b"ring-client");
        let client_frame = client.inner.outbox.pop_front().unwrap();
        assert_ne!(&client_frame[DATA_HEADER_LEN..], b"ring-client");
        server.inner.inbox.push_back(client_frame.clone());
        assert_eq!(server.recv(), Some(b"ring-client".to_vec()));
        server.inner.inbox.push_back(client_frame);
        assert_eq!(server.recv(), None);
        assert_eq!(server.replayed_datagrams(), 1);

        server.send(b"ring-server");
        let server_frame = server.inner.outbox.pop_front().unwrap();
        client.inner.inbox.push_back(server_frame);
        assert_eq!(client.recv(), Some(b"ring-server".to_vec()));

        client.send(b"tamper");
        let mut tampered = client.inner.outbox.pop_front().unwrap();
        *tampered.last_mut().unwrap() ^= 0x80;
        server.inner.inbox.push_back(tampered);
        assert_eq!(server.recv(), None);
        assert_eq!(server.security_state(), TransportSecurity::Rejected);
        assert_eq!(
            server.take_security_error(),
            Some(SecurityError::ProviderRejected)
        );
    }

    #[cfg(feature = "crypto")]
    #[test]
    fn ring_provider_rejects_a_wrong_psk_during_handshake() {
        let mut client =
            SecureTransport::new_client(MockTransport::default(), RingPskProvider::new([0x11; 32]))
                .unwrap();
        let mut server =
            SecureTransport::new_server(MockTransport::default(), RingPskProvider::new([0x22; 32]));
        server.inner.inbox.extend(client.inner.outbox.drain(..));
        assert!(server.recv().is_none());
        assert_eq!(server.security_state(), TransportSecurity::Rejected);
        assert_eq!(
            server.take_security_error(),
            Some(SecurityError::ProviderRejected)
        );
        assert!(!client.established());
    }
}
