//! Mutually authenticated PSK handshake for encrypted world-network datagrams.

use core::fmt;
use ring::{
    hmac,
    rand::{SecureRandom, SystemRandom},
};

use crate::{Aes256GcmTransport, UdpByteTransport};

const HELLO_MAGIC: [u8; 4] = *b"PWH1";
const HELLO_VERSION: u8 = 1;
const CLIENT_HELLO_BYTES: usize = 4 + 1 + 16 + 32 + 32;
const SERVER_HELLO_BYTES: usize = 4 + 1 + 16 + 32 + 32;
const TAG_BYTES: usize = 32;
const MAX_HMAC_INPUT_BYTES: usize = 256;
const DOMAIN_CLIENT: &[u8] = b"prism/world-network/psk-client/v1";
const DOMAIN_SERVER: &[u8] = b"prism/world-network/psk-server/v1";
const DOMAIN_KEY: &[u8] = b"prism/world-network/psk-session/v1";

/// Handshake or key-derivation error.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum HandshakeError {
    /// A message has an invalid fixed length, marker, or version.
    InvalidMessage,
    /// A handshake message failed HMAC authentication.
    AuthenticationFailed,
    /// The peer nonce equals this endpoint's nonce.
    NonceReuse,
    /// The system cryptographic random source failed.
    RandomUnavailable,
    /// This handshake state already consumed or rejected a message.
    InvalidState,
    /// The PSK-derived 256-bit AEAD key was rejected.
    InvalidDerivedKey,
}

impl fmt::Display for HandshakeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::InvalidMessage => "invalid PSK handshake message",
            Self::AuthenticationFailed => "PSK handshake authentication failed",
            Self::NonceReuse => "PSK handshake nonce was reused",
            Self::RandomUnavailable => "secure random source unavailable",
            Self::InvalidState => "PSK handshake state does not accept this operation",
            Self::InvalidDerivedKey => "derived PSK session key is invalid",
        })
    }
}

impl core::error::Error for HandshakeError {}

/// Opaque client hello containing a fresh nonce and PSK authentication tag.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ClientHello([u8; CLIENT_HELLO_BYTES]);

impl ClientHello {
    /// Serializes the fixed-size hello for a datagram or reliable control channel.
    pub const fn as_bytes(&self) -> &[u8; CLIENT_HELLO_BYTES] {
        &self.0
    }

    /// Parses and structurally validates a serialized client hello.
    pub fn decode(bytes: &[u8]) -> Result<Self, HandshakeError> {
        if bytes.len() != CLIENT_HELLO_BYTES
            || bytes[..4] != HELLO_MAGIC
            || bytes[4] != HELLO_VERSION
        {
            return Err(HandshakeError::InvalidMessage);
        }
        let mut value = [0; CLIENT_HELLO_BYTES];
        value.copy_from_slice(bytes);
        Ok(Self(value))
    }
}

/// Opaque authenticated server hello and proof of the complete handshake transcript.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ServerHello([u8; SERVER_HELLO_BYTES]);

impl ServerHello {
    /// Serializes the fixed-size hello for a datagram or reliable control channel.
    pub const fn as_bytes(&self) -> &[u8; SERVER_HELLO_BYTES] {
        &self.0
    }

    /// Parses and structurally validates a serialized server hello.
    pub fn decode(bytes: &[u8]) -> Result<Self, HandshakeError> {
        if bytes.len() != SERVER_HELLO_BYTES
            || bytes[..4] != HELLO_MAGIC
            || bytes[4] != HELLO_VERSION
        {
            return Err(HandshakeError::InvalidMessage);
        }
        let mut value = [0; SERVER_HELLO_BYTES];
        value.copy_from_slice(bytes);
        Ok(Self(value))
    }
}

/// Directional session keys and nonce prefixes bound to one authenticated session id.
///
/// This configuration is returned only after the role-specific PSK proof passes.
/// Client and server values are complementary.
pub struct SecureSessionConfig {
    session_id: u128,
    send_key: [u8; 32],
    receive_key: [u8; 32],
    send_prefix: [u8; 4],
    receive_prefix: [u8; 4],
    send_direction: u8,
}

impl Drop for SecureSessionConfig {
    fn drop(&mut self) {
        self.send_key.fill(0);
        self.receive_key.fill(0);
        self.send_prefix.fill(0);
        self.receive_prefix.fill(0);
    }
}

impl SecureSessionConfig {
    /// Returns the handshake-bound session identity.
    pub const fn session_id(&self) -> u128 {
        self.session_id
    }

    /// Consumes the key material and wraps a connected UDP transport with AES-GCM.
    pub fn into_transport(
        mut self,
        transport: UdpByteTransport,
        max_plaintext_frame_bytes: usize,
    ) -> Result<Aes256GcmTransport, crate::CryptoTransportError> {
        let result = Aes256GcmTransport::new(
            transport,
            self.send_key,
            self.receive_key,
            self.send_prefix,
            self.receive_prefix,
            self.send_direction,
            self.session_id,
            max_plaintext_frame_bytes,
        );
        self.send_key.fill(0);
        self.receive_key.fill(0);
        result
    }
}

/// Client side of the one-shot mutually authenticated PSK handshake.
pub struct ClientHandshake {
    psk: [u8; 32],
    session_id: u128,
    client_nonce: [u8; 32],
    hello: ClientHello,
    finished: bool,
}

impl Drop for ClientHandshake {
    fn drop(&mut self) {
        self.psk.fill(0);
        self.client_nonce.fill(0);
    }
}

impl ClientHandshake {
    /// Creates a fresh client hello using the OS cryptographic random source.
    pub fn start(psk: [u8; 32], session_id: u128) -> Result<Self, HandshakeError> {
        let mut nonce = [0; 32];
        SystemRandom::new()
            .fill(&mut nonce)
            .map_err(|_| HandshakeError::RandomUnavailable)?;
        let mut hello = [0; CLIENT_HELLO_BYTES];
        hello[..4].copy_from_slice(&HELLO_MAGIC);
        hello[4] = HELLO_VERSION;
        hello[5..21].copy_from_slice(&session_id.to_le_bytes());
        hello[21..53].copy_from_slice(&nonce);
        let tag = hmac_tag(&psk, DOMAIN_CLIENT, &[&hello[..21], &nonce]);
        hello[53..].copy_from_slice(&tag);
        Ok(Self {
            psk,
            session_id,
            client_nonce: nonce,
            hello: ClientHello(hello),
            finished: false,
        })
    }

    /// Returns the hello to send to the server.
    pub const fn hello(&self) -> &ClientHello {
        &self.hello
    }

    /// Authenticates the server proof and consumes this handshake into session keys.
    pub fn finish(mut self, server: &ServerHello) -> Result<SecureSessionConfig, HandshakeError> {
        if self.finished {
            return Err(HandshakeError::InvalidState);
        }
        let bytes = &server.0;
        let session_id = u128::from_le_bytes(bytes[5..21].try_into().unwrap());
        let server_nonce = &bytes[21..53];
        if session_id != self.session_id {
            return Err(HandshakeError::AuthenticationFailed);
        }
        if server_nonce == self.client_nonce {
            return Err(HandshakeError::NonceReuse);
        }
        if !verify_tag(
            &self.psk,
            DOMAIN_SERVER,
            &[self.hello.as_bytes(), &bytes[..53]],
            &bytes[53..],
        ) {
            self.finished = true;
            return Err(HandshakeError::AuthenticationFailed);
        }
        self.finished = true;
        derive_config(
            &self.psk,
            self.session_id,
            &self.client_nonce,
            server_nonce,
            true,
        )
    }
}

/// Server side of the one-shot mutually authenticated PSK handshake.
pub struct ServerHandshake {
    psk: [u8; 32],
}

impl Drop for ServerHandshake {
    fn drop(&mut self) {
        self.psk.fill(0);
    }
}

impl ServerHandshake {
    /// Creates a server handshake verifier for the configured pre-shared key.
    pub const fn new(psk: [u8; 32]) -> Self {
        Self { psk }
    }

    /// Verifies a client hello and returns the authenticated response plus server keys.
    pub fn accept(
        self,
        client: &ClientHello,
    ) -> Result<(ServerHello, SecureSessionConfig), HandshakeError> {
        let bytes = &client.0;
        let session_id = u128::from_le_bytes(bytes[5..21].try_into().unwrap());
        let client_nonce = &bytes[21..53];
        if !verify_tag(
            &self.psk,
            DOMAIN_CLIENT,
            &[&bytes[..21], client_nonce],
            &bytes[53..],
        ) {
            return Err(HandshakeError::AuthenticationFailed);
        }
        let mut server_nonce = [0; 32];
        SystemRandom::new()
            .fill(&mut server_nonce)
            .map_err(|_| HandshakeError::RandomUnavailable)?;
        if server_nonce.as_slice() == client_nonce {
            return Err(HandshakeError::NonceReuse);
        }
        let mut bytes_out = [0; SERVER_HELLO_BYTES];
        bytes_out[..4].copy_from_slice(&HELLO_MAGIC);
        bytes_out[4] = HELLO_VERSION;
        bytes_out[5..21].copy_from_slice(&session_id.to_le_bytes());
        bytes_out[21..53].copy_from_slice(&server_nonce);
        let proof = hmac_tag(
            &self.psk,
            DOMAIN_SERVER,
            &[client.as_bytes(), &bytes_out[..53]],
        );
        bytes_out[53..].copy_from_slice(&proof);
        let config = derive_config(&self.psk, session_id, client_nonce, &server_nonce, false)?;
        Ok((ServerHello(bytes_out), config))
    }
}

fn hmac_tag(psk: &[u8; 32], domain: &[u8], fields: &[&[u8]]) -> [u8; TAG_BYTES] {
    let key = hmac::Key::new(hmac::HMAC_SHA256, psk);
    let mut context = hmac::Context::with_key(&key);
    context.update(domain);
    for field in fields {
        context.update(&(field.len() as u64).to_le_bytes());
        context.update(field);
    }
    context.sign().as_ref().try_into().unwrap()
}

fn verify_tag(psk: &[u8; 32], domain: &[u8], fields: &[&[u8]], tag: &[u8]) -> bool {
    if tag.len() != TAG_BYTES {
        return false;
    }

    let mut message = [0; MAX_HMAC_INPUT_BYTES];
    let mut used = 0;
    if !append_hmac_component(&mut message, &mut used, domain) {
        return false;
    }
    for field in fields {
        let length = (field.len() as u64).to_le_bytes();
        if !append_hmac_component(&mut message, &mut used, &length)
            || !append_hmac_component(&mut message, &mut used, field)
        {
            return false;
        }
    }
    let key = hmac::Key::new(hmac::HMAC_SHA256, psk);
    hmac::verify(&key, &message[..used], tag).is_ok()
}

fn append_hmac_component(
    message: &mut [u8; MAX_HMAC_INPUT_BYTES],
    used: &mut usize,
    component: &[u8],
) -> bool {
    let Some(end) = used.checked_add(component.len()) else {
        return false;
    };
    let Some(destination) = message.get_mut(*used..end) else {
        return false;
    };
    destination.copy_from_slice(component);
    *used = end;
    true
}

fn derive_config(
    psk: &[u8; 32],
    session_id: u128,
    client_nonce: &[u8],
    server_nonce: &[u8],
    is_client: bool,
) -> Result<SecureSessionConfig, HandshakeError> {
    let transcript = hmac_tag(
        psk,
        DOMAIN_KEY,
        &[&session_id.to_le_bytes(), client_nonce, server_nonce],
    );
    let c2s_key = hmac_tag(psk, b"prism/world-network/key/c2s/v1", &[&transcript]);
    let s2c_key = hmac_tag(psk, b"prism/world-network/key/s2c/v1", &[&transcript]);
    let c2s_prefix_full = hmac_tag(psk, b"prism/world-network/nonce/c2s/v1", &[&transcript]);
    let s2c_prefix_full = hmac_tag(psk, b"prism/world-network/nonce/s2c/v1", &[&transcript]);
    let mut c2s_prefix = [0; 4];
    let mut s2c_prefix = [0; 4];
    c2s_prefix.copy_from_slice(&c2s_prefix_full[..4]);
    s2c_prefix.copy_from_slice(&s2c_prefix_full[..4]);
    let (send_key, receive_key, send_prefix, receive_prefix, direction) = if is_client {
        (c2s_key, s2c_key, c2s_prefix, s2c_prefix, 0)
    } else {
        (s2c_key, c2s_key, s2c_prefix, c2s_prefix, 1)
    };
    Ok(SecureSessionConfig {
        session_id,
        send_key,
        receive_key,
        send_prefix,
        receive_prefix,
        send_direction: direction,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn client_server_handshake_derives_complementary_session_keys() {
        let psk = [0x37; 32];
        let client = ClientHandshake::start(psk, 0xfeed).unwrap();
        let encoded = client.hello().as_bytes();
        let parsed = ClientHello::decode(encoded).unwrap();
        let server = ServerHandshake::new(psk);
        let (server_hello, server_config) = server.accept(&parsed).unwrap();
        let decoded_server = ServerHello::decode(server_hello.as_bytes()).unwrap();
        let client_config = client.finish(&decoded_server).unwrap();
        assert_eq!(client_config.session_id(), 0xfeed);
        assert_eq!(server_config.session_id(), 0xfeed);
        assert_eq!(client_config.send_key, server_config.receive_key);
        assert_eq!(client_config.receive_key, server_config.send_key);
        assert_eq!(client_config.send_prefix, server_config.receive_prefix);
        assert_eq!(client_config.receive_prefix, server_config.send_prefix);
        assert_eq!(client_config.send_direction, 0);
        assert_eq!(server_config.send_direction, 1);
    }

    #[test]
    fn wrong_psk_and_tampered_transcript_are_rejected() {
        let client = ClientHandshake::start([1; 32], 99).unwrap();
        let parsed = ClientHello::decode(client.hello().as_bytes()).unwrap();
        assert!(matches!(
            ServerHandshake::new([2; 32]).accept(&parsed),
            Err(HandshakeError::AuthenticationFailed)
        ));
        let (mut response, _) = ServerHandshake::new([1; 32]).accept(&parsed).unwrap();
        response.0[53] ^= 1;
        assert!(matches!(
            client.finish(&response),
            Err(HandshakeError::AuthenticationFailed)
        ));
    }

    #[test]
    fn malformed_hello_and_wrong_session_are_rejected() {
        assert_eq!(
            ClientHello::decode(&[0; 4]),
            Err(HandshakeError::InvalidMessage)
        );
        let client = ClientHandshake::start([3; 32], 7).unwrap();
        let parsed = ClientHello::decode(client.hello().as_bytes()).unwrap();
        let (mut response, _) = ServerHandshake::new([3; 32]).accept(&parsed).unwrap();
        response.0[5] ^= 0x80;
        assert!(matches!(
            client.finish(&response),
            Err(HandshakeError::AuthenticationFailed)
        ));
    }
}
