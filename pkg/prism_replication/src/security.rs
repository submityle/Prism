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

use crate::transport::{Transport, TransportMigrationCapability, TransportSecurity};

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

/// Numeric diagnostics for one authenticated session.
///
/// This snapshot intentionally contains no peer identity, key material, nonce,
/// or payload data. It is safe to export to a connection-health dashboard and
/// remains useful after a reconnect because each [`SecureTransport`] owns one
/// session's counters.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SecurityTelemetrySnapshot {
    /// Current authenticated-session state.
    pub state: TransportSecurity,
    /// Credential id selected by a rotating-key provider, if applicable.
    pub credential_id: Option<u32>,
    /// Handshake frames sent, including retransmissions.
    pub handshake_transmissions: u64,
    /// Handshake frames sent after the initial offer/accept.
    pub handshake_retries: u64,
    /// Protected application datagrams successfully sealed.
    pub protected_datagrams_sent: u64,
    /// Protected application datagrams that passed authentication.
    pub protected_datagrams_received: u64,
    /// Authenticated duplicates or packets outside the replay window.
    pub replayed_datagrams: u64,
    /// Frames rejected by the security wire decoder.
    pub malformed_frames: u64,
    /// Frames rejected by provider authentication/decryption.
    pub authentication_failures: u64,
    /// Number of completed authenticated handshakes.
    pub handshakes_completed: u64,
    /// First/most recent terminal security error, if any.
    pub last_error: Option<SecurityError>,
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
    #[cfg(any(test, feature = "crypto"))]
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

    /// Optional deployment credential id selected for this session.
    ///
    /// This is metadata only: it never exposes key material. Providers without
    /// rotating credentials keep the default `None`.
    fn credential_id(&self) -> Option<u32> {
        None
    }

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

/// Stable identifier for [`RingPskKeyringProvider`].
#[cfg(feature = "crypto")]
pub const RING_PSK_KEYRING_PROVIDER_ID: u16 = 0x524b;

/// Maximum number of previous credentials retained for reconnects after a
/// rotation. Existing sessions keep their derived session key; retained keys
/// only authorize new handshakes during the bounded overlap window.
#[cfg(feature = "crypto")]
pub const MAX_RETAINED_PSK_KEYS: usize = 3;

/// An application-owned PSK with an explicit deployment key id.
#[cfg(feature = "crypto")]
#[derive(Clone, Copy, PartialEq, Eq)]
pub struct PskCredential {
    id: u32,
    secret: [u8; 32],
}

#[cfg(feature = "crypto")]
impl PskCredential {
    /// Construct a credential. The secret is never formatted by this type.
    #[must_use]
    pub const fn new(id: u32, secret: [u8; 32]) -> Self {
        Self { id, secret }
    }

    /// Stable key id carried inside the authenticated handshake proof.
    #[must_use]
    pub const fn id(self) -> u32 {
        self.id
    }
}

/// Errors from the bounded in-process credential lifecycle.
#[cfg(feature = "crypto")]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PskKeyringError {
    /// The key id is already active or retained.
    DuplicateId(u32),
    /// The bounded overlap window is full.
    Capacity,
    /// The requested key is not retained by this keyring.
    UnknownId(u32),
    /// The active key cannot be retired while it is serving new handshakes.
    ActiveKey(u32),
}

/// Bounded active/previous PSK lifecycle for reconnects during key rotation.
///
/// `rotate` makes the new credential active and retains the old one. The
/// caller explicitly retires old credentials once every deployment has moved
/// to the new key. This is intentionally a provider-level primitive: storage,
/// access control, identity attestation and KMS integration remain owned by
/// the deployment application.
#[cfg(feature = "crypto")]
#[derive(Clone)]
pub struct PskKeyring {
    active: PskCredential,
    previous: Vec<PskCredential>,
}

#[cfg(feature = "crypto")]
impl PskKeyring {
    /// Start a keyring with one active credential.
    #[must_use]
    pub fn new(active: PskCredential) -> Self {
        Self {
            active,
            previous: Vec::new(),
        }
    }

    /// Return the credential used for new handshakes.
    #[must_use]
    pub const fn active(&self) -> PskCredential {
        self.active
    }

    /// Return a retained credential by id, including the active one.
    #[must_use]
    fn resolve(&self, id: u32) -> Option<PskCredential> {
        if self.active.id == id {
            return Some(self.active);
        }
        self.previous.iter().copied().find(|key| key.id == id)
    }

    /// Make `next` active and retain the previous active credential.
    pub fn rotate(&mut self, next: PskCredential) -> Result<(), PskKeyringError> {
        if self.resolve(next.id).is_some() {
            return Err(PskKeyringError::DuplicateId(next.id));
        }
        if self.previous.len() >= MAX_RETAINED_PSK_KEYS {
            return Err(PskKeyringError::Capacity);
        }
        let old = core::mem::replace(&mut self.active, next);
        self.previous.insert(0, old);
        Ok(())
    }

    /// Retire a previous credential so it can no longer authenticate a new
    /// session. Existing sessions are unaffected because they use a derived
    /// session key rather than consulting the keyring again.
    pub fn retire(&mut self, id: u32) -> Result<(), PskKeyringError> {
        if self.active.id == id {
            return Err(PskKeyringError::ActiveKey(id));
        }
        let Some(index) = self.previous.iter().position(|key| key.id == id) else {
            return Err(PskKeyringError::UnknownId(id));
        };
        self.previous.remove(index);
        Ok(())
    }

    /// Number of active and retained credentials.
    #[must_use]
    pub fn len(&self) -> usize {
        1 + self.previous.len()
    }

    /// Whether the keyring contains no credentials.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

/// A PSK provider with an authenticated key id and bounded rotation overlap.
///
/// New clients use the keyring's active credential. Servers accept the active
/// credential and retained previous credentials, but the selected credential
/// is committed to the session transcript and cannot change mid-session.
#[cfg(feature = "crypto")]
pub struct RingPskKeyringProvider {
    keyring: PskKeyring,
    selected_id: Option<u32>,
    client_nonce: Option<SessionNonce>,
    server_nonce: Option<SessionNonce>,
    session_key: Option<[u8; 32]>,
}

#[cfg(feature = "crypto")]
impl RingPskKeyringProvider {
    /// Construct a provider from an application-owned keyring.
    #[must_use]
    pub fn new(keyring: PskKeyring) -> Self {
        Self {
            keyring,
            selected_id: None,
            client_nonce: None,
            server_nonce: None,
            session_key: None,
        }
    }

    /// Inspect the provider's credential lifecycle state.
    #[must_use]
    pub const fn keyring(&self) -> &PskKeyring {
        &self.keyring
    }

    /// Rotate credentials for future handshakes made with this provider.
    pub fn rotate(&mut self, next: PskCredential) -> Result<(), PskKeyringError> {
        self.keyring.rotate(next)
    }

    /// Retire a previous credential for future handshakes.
    pub fn retire(&mut self, id: u32) -> Result<(), PskKeyringError> {
        self.keyring.retire(id)
    }

    fn mac(secret: &[u8; 32], domain: &[u8], fields: &[&[u8]]) -> [u8; 32] {
        use ring::hmac;
        let key = hmac::Key::new(hmac::HMAC_SHA256, secret);
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

    fn verify_mac(secret: &[u8; 32], expected: &[u8], domain: &[u8], fields: &[&[u8]]) -> bool {
        use ring::hmac;
        let key = hmac::Key::new(hmac::HMAC_SHA256, secret);
        let mut context = Vec::new();
        context.extend_from_slice(domain);
        for field in fields {
            context.extend_from_slice(field);
        }
        hmac::verify(&key, &context, expected).is_ok()
    }

    fn derive_session_key(
        secret: &[u8; 32],
        key_id: u32,
        client_nonce: SessionNonce,
        server_nonce: SessionNonce,
    ) -> [u8; 32] {
        Self::mac(
            secret,
            b"prism/replication/keyring-session-key/v1",
            &[
                &RING_PSK_KEYRING_PROVIDER_ID.to_le_bytes(),
                &key_id.to_le_bytes(),
                &client_nonce.0,
                &server_nonce.0,
            ],
        )
    }

    fn client_auth(credential: PskCredential, nonce: SessionNonce) -> Vec<u8> {
        let provider_id = RING_PSK_KEYRING_PROVIDER_ID.to_le_bytes();
        let key_id = credential.id.to_le_bytes();
        let mac = Self::mac(
            &credential.secret,
            b"prism/replication/keyring-client-hello/v1",
            &[&provider_id, &key_id, &nonce.0],
        );
        let mut auth = Vec::with_capacity(4 + mac.len());
        auth.extend_from_slice(&key_id);
        auth.extend_from_slice(&mac);
        auth
    }

    fn server_auth(
        credential: PskCredential,
        client_nonce: SessionNonce,
        server_nonce: SessionNonce,
    ) -> Vec<u8> {
        let provider_id = RING_PSK_KEYRING_PROVIDER_ID.to_le_bytes();
        let key_id = credential.id.to_le_bytes();
        let mac = Self::mac(
            &credential.secret,
            b"prism/replication/keyring-server-accept/v1",
            &[&provider_id, &key_id, &client_nonce.0, &server_nonce.0],
        );
        let mut auth = Vec::with_capacity(4 + mac.len());
        auth.extend_from_slice(&key_id);
        auth.extend_from_slice(&mac);
        auth
    }

    fn parse_keyed_auth(auth: &[u8]) -> Option<(u32, &[u8])> {
        if auth.len() != 4 + 32 {
            return None;
        }
        let id = u32::from_le_bytes(auth[..4].try_into().ok()?);
        Some((id, &auth[4..]))
    }

    fn key(&self) -> Result<ring::aead::LessSafeKey, SecurityError> {
        let key = self.session_key.ok_or(SecurityError::NotEstablished)?;
        let unbound = ring::aead::UnboundKey::new(&ring::aead::AES_256_GCM, &key)
            .map_err(|_| SecurityError::ProviderRejected)?;
        Ok(ring::aead::LessSafeKey::new(unbound))
    }

    fn fresh_nonce() -> Result<SessionNonce, SecurityError> {
        RingPskProvider::fresh_nonce()
    }
}

#[cfg(feature = "crypto")]
impl SessionSecurityProvider for RingPskKeyringProvider {
    fn provider_id(&self) -> u16 {
        RING_PSK_KEYRING_PROVIDER_ID
    }

    fn credential_id(&self) -> Option<u32> {
        self.selected_id
    }

    fn max_seal_overhead(&self) -> usize {
        ring::aead::AES_256_GCM.tag_len()
    }

    fn create_client_hello(&mut self) -> Result<ClientHello, SecurityError> {
        let nonce = Self::fresh_nonce()?;
        let credential = self.keyring.active();
        self.selected_id = Some(credential.id);
        self.client_nonce = Some(nonce);
        self.server_nonce = None;
        self.session_key = None;
        Ok(ClientHello {
            provider_id: self.provider_id(),
            nonce,
            authenticator: Self::client_auth(credential, nonce),
        })
    }

    fn accept_client_hello(&mut self, hello: &ClientHello) -> Result<ServerAccept, SecurityError> {
        if hello.provider_id != self.provider_id() {
            return Err(SecurityError::ProviderRejected);
        }
        let Some((key_id, mac)) = Self::parse_keyed_auth(&hello.authenticator) else {
            return Err(SecurityError::ProviderRejected);
        };
        let Some(credential) = self.keyring.resolve(key_id) else {
            return Err(SecurityError::ProviderRejected);
        };
        let provider_id = self.provider_id().to_le_bytes();
        let key_id_bytes = key_id.to_le_bytes();
        if !Self::verify_mac(
            &credential.secret,
            mac,
            b"prism/replication/keyring-client-hello/v1",
            &[&provider_id, &key_id_bytes, &hello.nonce.0],
        ) {
            return Err(SecurityError::ProviderRejected);
        }
        let server_nonce = Self::fresh_nonce()?;
        self.selected_id = Some(key_id);
        self.client_nonce = Some(hello.nonce);
        self.server_nonce = Some(server_nonce);
        self.session_key = Some(Self::derive_session_key(
            &credential.secret,
            key_id,
            hello.nonce,
            server_nonce,
        ));
        Ok(ServerAccept {
            provider_id: self.provider_id(),
            nonce: server_nonce,
            authenticator: Self::server_auth(credential, hello.nonce, server_nonce),
        })
    }

    fn finish_client(
        &mut self,
        hello: &ClientHello,
        accept: &ServerAccept,
    ) -> Result<(), SecurityError> {
        if hello.provider_id != self.provider_id() || accept.provider_id != self.provider_id() {
            return Err(SecurityError::ProviderRejected);
        }
        let Some((key_id, mac)) = Self::parse_keyed_auth(&accept.authenticator) else {
            return Err(SecurityError::ProviderRejected);
        };
        let Some(credential) = self.keyring.resolve(key_id) else {
            return Err(SecurityError::ProviderRejected);
        };
        let provider_id = self.provider_id().to_le_bytes();
        let key_id_bytes = key_id.to_le_bytes();
        if !Self::verify_mac(
            &credential.secret,
            mac,
            b"prism/replication/keyring-server-accept/v1",
            &[&provider_id, &key_id_bytes, &hello.nonce.0, &accept.nonce.0],
        ) {
            return Err(SecurityError::ProviderRejected);
        }
        self.selected_id = Some(key_id);
        self.client_nonce = Some(hello.nonce);
        self.server_nonce = Some(accept.nonce);
        self.session_key = Some(Self::derive_session_key(
            &credential.secret,
            key_id,
            hello.nonce,
            accept.nonce,
        ));
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
            RingPskProvider::payload_nonce(direction, sequence),
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
                RingPskProvider::payload_nonce(direction, sequence),
                ring::aead::Aad::from(aad),
                out,
            )
            .map_err(|_| SecurityError::ProviderRejected)?;
        let length = plaintext.len();
        out.truncate(length);
        Ok(())
    }
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
    poll_ticks: u64,
    last_error: Option<SecurityError>,
    replayed: u64,
    handshake_transmissions: u64,
    handshake_retries: u64,
    protected_datagrams_sent: u64,
    protected_datagrams_received: u64,
    malformed_frames: u64,
    authentication_failures: u64,
    handshakes_completed: u64,
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
            handshake_transmissions: 0,
            handshake_retries: 0,
            protected_datagrams_sent: 0,
            protected_datagrams_received: 0,
            malformed_frames: 0,
            authentication_failures: 0,
            handshakes_completed: 0,
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
            handshake_transmissions: 0,
            handshake_retries: 0,
            protected_datagrams_sent: 0,
            protected_datagrams_received: 0,
            malformed_frames: 0,
            authentication_failures: 0,
            handshakes_completed: 0,
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

    /// Copy numeric connection-health counters without exposing secrets.
    #[must_use]
    pub fn telemetry(&self) -> SecurityTelemetrySnapshot {
        SecurityTelemetrySnapshot {
            state: self.security_state(),
            credential_id: self.provider.credential_id(),
            handshake_transmissions: self.handshake_transmissions,
            handshake_retries: self.handshake_retries,
            protected_datagrams_sent: self.protected_datagrams_sent,
            protected_datagrams_received: self.protected_datagrams_received,
            replayed_datagrams: self.replayed,
            malformed_frames: self.malformed_frames,
            authentication_failures: self.authentication_failures,
            handshakes_completed: self.handshakes_completed,
            last_error: self.last_error,
        }
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
        if self.handshake_transmissions > 0 {
            self.handshake_retries = self.handshake_retries.saturating_add(1);
        }
        self.handshake_transmissions = self.handshake_transmissions.saturating_add(1);
        self.inner.send(&encode_hello(hello));
    }

    fn send_accept(&mut self) {
        let Some(accept) = self.server_accept.as_ref() else {
            return;
        };
        if self.handshake_transmissions > 0 {
            self.handshake_retries = self.handshake_retries.saturating_add(1);
        }
        self.handshake_transmissions = self.handshake_transmissions.saturating_add(1);
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
        self.protected_datagrams_sent = self.protected_datagrams_sent.saturating_add(1);
        Ok(())
    }

    fn process_frame(&mut self, frame: &[u8]) {
        if frame.len() > MAX_HANDSHAKE_FRAME
            || frame.len() < 4
            || frame[..2] != WIRE_MAGIC
            || frame[2] != WIRE_VERSION
        {
            self.malformed_frames = self.malformed_frames.saturating_add(1);
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
            self.malformed_frames = self.malformed_frames.saturating_add(1);
            self.set_error(SecurityError::MalformedFrame);
            return;
        };
        if validate_hello(&hello, self.provider.provider_id()).is_err() {
            self.authentication_failures = self.authentication_failures.saturating_add(1);
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
                self.authentication_failures = self.authentication_failures.saturating_add(1);
                self.set_error(error);
                return;
            }
        };
        if validate_accept(&accept, self.provider.provider_id(), hello.nonce).is_err() {
            self.malformed_frames = self.malformed_frames.saturating_add(1);
            self.set_error(SecurityError::InvalidNonce);
            return;
        }
        self.client_nonce = Some(hello.nonce);
        self.server_nonce = Some(accept.nonce);
        self.client_hello = Some(hello);
        self.server_accept = Some(accept);
        self.state = State::Established;
        self.handshakes_completed = self.handshakes_completed.saturating_add(1);
        self.send_accept();
        self.flush_pending();
    }

    fn process_accept(&mut self, frame: &[u8]) {
        if self.role != Role::Client || self.state != State::ClientHelloSent {
            return;
        }
        let Some(accept) = decode_accept(frame) else {
            self.malformed_frames = self.malformed_frames.saturating_add(1);
            self.set_error(SecurityError::MalformedFrame);
            return;
        };
        let Some(hello) = self.client_hello.as_ref() else {
            self.set_error(SecurityError::NotEstablished);
            return;
        };
        if validate_accept(&accept, self.provider.provider_id(), hello.nonce).is_err() {
            self.malformed_frames = self.malformed_frames.saturating_add(1);
            self.set_error(SecurityError::VersionMismatch);
            return;
        }
        if let Err(error) = self.provider.finish_client(hello, &accept) {
            self.authentication_failures = self.authentication_failures.saturating_add(1);
            self.set_error(error);
            return;
        }
        self.server_nonce = Some(accept.nonce);
        self.server_accept = Some(accept);
        self.state = State::Established;
        self.handshakes_completed = self.handshakes_completed.saturating_add(1);
        self.flush_pending();
    }

    fn process_data(&mut self, frame: &[u8]) {
        if self.state != State::Established || frame.len() < DATA_HEADER_LEN {
            self.malformed_frames = self.malformed_frames.saturating_add(1);
            self.set_error(if self.state == State::Established {
                SecurityError::MalformedFrame
            } else {
                SecurityError::NotEstablished
            });
            return;
        }
        let provider_id = u16::from_le_bytes([frame[4], frame[5]]);
        if provider_id != self.provider.provider_id() {
            self.authentication_failures = self.authentication_failures.saturating_add(1);
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
            self.authentication_failures = self.authentication_failures.saturating_add(1);
            self.set_error(SecurityError::ProviderRejected);
            return;
        }
        self.protected_datagrams_received = self.protected_datagrams_received.saturating_add(1);
        match self.replay.observe(sequence) {
            ReplayResult::Fresh => self.inbound.push_back(plaintext),
            ReplayResult::Duplicate | ReplayResult::TooOld => {
                self.replayed = self.replayed.saturating_add(1);
            }
        }
    }

    fn drive(&mut self) {
        self.poll_ticks = self.poll_ticks.saturating_add(1);
        if self.state == State::ClientHelloSent && self.poll_ticks.is_multiple_of(4) {
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

    fn migration_capability(&self) -> TransportMigrationCapability {
        self.inner.migration_capability()
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
    fn queued_datagrams_are_discarded_and_reported_if_sequence_exhausts_on_flush() {
        let mut client = SecureTransport::new_client(
            MockTransport::default(),
            test_support::Provider::new(0x31, 1),
        )
        .unwrap();
        client.send(b"queued-before-handshake");
        assert_eq!(client.pending.len(), 1);
        // The queued send emits a client hello; the test below models the
        // peer completing that handshake before the local sequence is used.
        client.inner.outbox.clear();

        // Model a session whose handshake has just completed after consuming
        // the entire outbound nonce sequence. Flushing must fail closed: no
        // queued plaintext is emitted under a reused nonce, and the terminal
        // error remains observable through the public diagnostics API.
        client.state = State::Established;
        client.next_send = u64::MAX;
        client.flush_pending();

        assert_eq!(client.security_state(), TransportSecurity::Rejected);
        assert_eq!(client.pending.len(), 0);
        assert!(client.inner.outbox.is_empty());
        assert_eq!(
            client.take_security_error(),
            Some(SecurityError::SequenceExhausted)
        );
        client.flush_pending();
        assert!(client.inner.outbox.is_empty());
        assert!(client.pending.is_empty());
    }

    #[test]
    fn handshake_retry_clock_saturates_without_rewinding_retry_phase() {
        let mut client = SecureTransport::new_client(
            MockTransport::default(),
            test_support::Provider::new(0x31, 1),
        )
        .unwrap();
        client.poll_ticks = u64::MAX - 1;
        client.drive();
        assert_eq!(client.poll_ticks, u64::MAX);
        let retries = client.handshake_retries;
        client.drive();
        assert_eq!(client.poll_ticks, u64::MAX);
        assert_eq!(client.handshake_retries, retries);
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

    #[cfg(feature = "crypto")]
    fn connect_keyring(
        client_keys: PskKeyring,
        server_keys: PskKeyring,
    ) -> (
        SecureTransport<MockTransport, RingPskKeyringProvider>,
        SecureTransport<MockTransport, RingPskKeyringProvider>,
    ) {
        let mut client = SecureTransport::new_client(
            MockTransport::default(),
            RingPskKeyringProvider::new(client_keys),
        )
        .expect("keyring client hello should be generated");
        let mut server = SecureTransport::new_server(
            MockTransport::default(),
            RingPskKeyringProvider::new(server_keys),
        );
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
    fn keyring_accepts_old_and_new_credentials_during_bounded_rotation() {
        let old = PskCredential::new(11, [0x11; 32]);
        let next = PskCredential::new(12, [0x12; 32]);
        let mut server_keys = PskKeyring::new(old);
        server_keys.rotate(next).unwrap();
        assert_eq!(server_keys.active().id(), 12);
        assert_eq!(server_keys.len(), 2);

        let (mut old_client, mut old_server) =
            connect_keyring(PskKeyring::new(old), server_keys.clone());
        old_client.send(b"old-session");
        old_server
            .inner
            .inbox
            .extend(old_client.inner.outbox.drain(..));
        assert_eq!(old_server.recv(), Some(b"old-session".to_vec()));
        assert_eq!(old_client.telemetry().credential_id, Some(11));
        assert_eq!(old_server.telemetry().credential_id, Some(11));

        let (mut new_client, mut new_server) = connect_keyring(PskKeyring::new(next), server_keys);
        new_client.send(b"new-session");
        new_server
            .inner
            .inbox
            .extend(new_client.inner.outbox.drain(..));
        assert_eq!(new_server.recv(), Some(b"new-session".to_vec()));
        assert_eq!(new_client.telemetry().credential_id, Some(12));
        assert_eq!(new_server.telemetry().credential_id, Some(12));
    }

    #[cfg(feature = "crypto")]
    #[test]
    fn ciphertext_from_an_old_session_is_rejected_after_reconnect() {
        let old = PskCredential::new(41, [0x41; 32]);
        let next = PskCredential::new(42, [0x42; 32]);
        let (mut old_client, mut old_server) =
            connect_keyring(PskKeyring::new(old), PskKeyring::new(old));
        old_client.send(b"stale");
        let stale_frame = old_client.inner.outbox.pop_front().unwrap();
        old_server.inner.inbox.push_back(stale_frame.clone());
        assert_eq!(old_server.recv(), Some(b"stale".to_vec()));

        let (new_client, mut new_server) =
            connect_keyring(PskKeyring::new(next), PskKeyring::new(next));
        assert_eq!(new_client.telemetry().credential_id, Some(42));
        assert_eq!(new_server.telemetry().credential_id, Some(42));
        new_server.inner.inbox.push_back(stale_frame);
        assert_eq!(new_server.recv(), None);
        assert_eq!(new_server.security_state(), TransportSecurity::Rejected);
        assert_eq!(
            new_server.take_security_error(),
            Some(SecurityError::ProviderRejected)
        );
    }

    #[cfg(feature = "crypto")]
    #[test]
    fn keyring_retirement_blocks_new_old_key_handshakes() {
        let old = PskCredential::new(21, [0x21; 32]);
        let next = PskCredential::new(22, [0x22; 32]);
        let mut server_keys = PskKeyring::new(old);
        server_keys.rotate(next).unwrap();
        server_keys.retire(old.id()).unwrap();
        assert_eq!(server_keys.len(), 1);

        let mut client = SecureTransport::new_client(
            MockTransport::default(),
            RingPskKeyringProvider::new(PskKeyring::new(old)),
        )
        .unwrap();
        let mut server = SecureTransport::new_server(
            MockTransport::default(),
            RingPskKeyringProvider::new(server_keys),
        );
        server.inner.inbox.extend(client.inner.outbox.drain(..));
        assert!(server.recv().is_none());
        assert_eq!(server.security_state(), TransportSecurity::Rejected);
        assert_eq!(
            server.take_security_error(),
            Some(SecurityError::ProviderRejected)
        );
    }

    #[cfg(feature = "crypto")]
    #[test]
    fn keyring_rotation_is_bounded_and_rejects_duplicate_ids() {
        let mut keyring = PskKeyring::new(PskCredential::new(30, [0x30; 32]));
        assert_eq!(
            keyring.rotate(PskCredential::new(30, [0x30; 32])),
            Err(PskKeyringError::DuplicateId(30))
        );
        for id in 31..=33 {
            keyring
                .rotate(PskCredential::new(id, [id as u8; 32]))
                .unwrap();
        }
        assert_eq!(keyring.len(), 4);
        assert_eq!(
            keyring.rotate(PskCredential::new(34, [0x34; 32])),
            Err(PskKeyringError::Capacity)
        );
    }
}
