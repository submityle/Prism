//! The remote-session state machine and capability negotiation.
//!
//! A remote profiler connection moves through a small, explicit lifecycle:
//! [`SessionState::Disconnected`] until a link exists,
//! [`SessionState::Handshaking`] while both sides exchange capabilities,
//! [`SessionState::Connected`] once agreed parameters are stored, and
//! [`SessionState::Closed`] afterward. [`Capabilities::negotiate`] takes the
//! locally desired limits and the peer's advertised limits and returns the
//! conservative intersection (minimum bandwidth, minimum sampling, and write
//! access only when both sides allow it), which decides how much telemetry the
//! link carries and whether writes are even possible.
//!
//! # Provenance
//! Original work; no Unreal Engine, Unity, Godot, Wwise, FMOD, Steam Audio,
//! Dolby, or Google Resonance Audio source or derived code; no AI/ML.
//!
//! # Relationship
//! Implements the remote profiler session and capability probe of design
//! section 38. The negotiated write flag gates the command path of design
//! section 21, and the negotiated telemetry rate throttles the mirror fed from
//! design section 26.

/// The lifecycle state of a [`RemoteSession`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub enum SessionState {
    /// No link exists yet.
    Disconnected,
    /// A link exists and capabilities are being exchanged.
    Handshaking,
    /// Capabilities are agreed and the session is live.
    Connected,
    /// The session has ended and cannot be reused.
    Closed,
}

/// The parameters a side of the link advertises or that were agreed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub struct Capabilities {
    /// Maximum sustained transport bandwidth in bytes per second.
    pub max_bandwidth_bytes_per_sec: u32,
    /// Audio sample rate in hertz used when streaming metered audio.
    pub sample_rate: u32,
    /// Telemetry refresh rate in snapshots per second.
    pub telemetry_hz: u32,
    /// Whether write commands are permitted over this link.
    pub allow_write: bool,
}

impl Capabilities {
    /// Returns the conservative intersection of `self` and `other`.
    ///
    /// Numeric limits take the smaller of the two sides so neither is
    /// overrun; write access is granted only when both sides allow it.
    #[must_use]
    pub fn negotiate(&self, other: &Capabilities) -> Capabilities {
        Capabilities {
            max_bandwidth_bytes_per_sec: self
                .max_bandwidth_bytes_per_sec
                .min(other.max_bandwidth_bytes_per_sec),
            sample_rate: self.sample_rate.min(other.sample_rate),
            telemetry_hz: self.telemetry_hz.min(other.telemetry_hz),
            allow_write: self.allow_write && other.allow_write,
        }
    }
}

/// The reason a session transition was refused.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub enum SessionError {
    /// The requested transition is not valid from the current state.
    InvalidTransition {
        /// The state the session was in.
        from: SessionState,
    },
}

/// The remote-session state machine with its negotiated capabilities.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RemoteSession {
    state: SessionState,
    capabilities: Option<Capabilities>,
}

impl RemoteSession {
    /// Creates a fresh session in [`SessionState::Disconnected`].
    #[must_use]
    pub fn new() -> Self {
        Self {
            state: SessionState::Disconnected,
            capabilities: None,
        }
    }

    /// Returns the current lifecycle state.
    #[must_use]
    pub fn state(&self) -> SessionState {
        self.state
    }

    /// Returns the negotiated capabilities, available only once connected.
    #[must_use]
    pub fn capabilities(&self) -> Option<&Capabilities> {
        self.capabilities.as_ref()
    }

    /// Returns `true` when the session is live.
    #[must_use]
    pub fn is_connected(&self) -> bool {
        self.state == SessionState::Connected
    }

    /// Starts the handshake, moving from `Disconnected` to `Handshaking`.
    ///
    /// # Errors
    ///
    /// Returns [`SessionError::InvalidTransition`] from any other state.
    pub fn begin_handshake(&mut self) -> Result<(), SessionError> {
        match self.state {
            SessionState::Disconnected => {
                self.state = SessionState::Handshaking;
                Ok(())
            }
            other => Err(SessionError::InvalidTransition { from: other }),
        }
    }

    /// Completes the handshake with the `negotiated` capabilities, moving from
    /// `Handshaking` to `Connected`.
    ///
    /// # Errors
    ///
    /// Returns [`SessionError::InvalidTransition`] from any other state.
    pub fn complete_handshake(
        &mut self,
        negotiated: Capabilities,
    ) -> Result<(), SessionError> {
        match self.state {
            SessionState::Handshaking => {
                self.capabilities = Some(negotiated);
                self.state = SessionState::Connected;
                Ok(())
            }
            other => Err(SessionError::InvalidTransition { from: other }),
        }
    }

    /// Ends the session, moving to `Closed` from any open state.
    ///
    /// # Errors
    ///
    /// Returns [`SessionError::InvalidTransition`] if already closed.
    pub fn close(&mut self) -> Result<(), SessionError> {
        match self.state {
            SessionState::Closed => {
                Err(SessionError::InvalidTransition { from: SessionState::Closed })
            }
            _ => {
                self.state = SessionState::Closed;
                Ok(())
            }
        }
    }
}

impl Default for RemoteSession {
    #[inline]
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tool_caps() -> Capabilities {
        Capabilities {
            max_bandwidth_bytes_per_sec: 1_000_000,
            sample_rate: 48_000,
            telemetry_hz: 60,
            allow_write: true,
        }
    }

    fn device_caps() -> Capabilities {
        Capabilities {
            max_bandwidth_bytes_per_sec: 256_000,
            sample_rate: 44_100,
            telemetry_hz: 30,
            allow_write: false,
        }
    }

    #[test]
    fn negotiate_takes_conservative_intersection() {
        let agreed = tool_caps().negotiate(&device_caps());
        assert_eq!(agreed.max_bandwidth_bytes_per_sec, 256_000);
        assert_eq!(agreed.sample_rate, 44_100);
        assert_eq!(agreed.telemetry_hz, 30);
        assert!(!agreed.allow_write);
    }

    #[test]
    fn happy_path_lifecycle() {
        let mut session = RemoteSession::new();
        assert_eq!(session.state(), SessionState::Disconnected);
        assert!(session.capabilities().is_none());

        session.begin_handshake().unwrap();
        assert_eq!(session.state(), SessionState::Handshaking);

        let agreed = tool_caps().negotiate(&device_caps());
        session.complete_handshake(agreed).unwrap();
        assert!(session.is_connected());
        assert_eq!(session.capabilities(), Some(&agreed));

        session.close().unwrap();
        assert_eq!(session.state(), SessionState::Closed);
    }

    #[test]
    fn invalid_transitions_are_refused() {
        let mut session = RemoteSession::new();
        assert_eq!(
            session.complete_handshake(tool_caps()),
            Err(SessionError::InvalidTransition {
                from: SessionState::Disconnected
            })
        );

        session.begin_handshake().unwrap();
        assert_eq!(
            session.begin_handshake(),
            Err(SessionError::InvalidTransition {
                from: SessionState::Handshaking
            })
        );
    }

    #[test]
    fn double_close_is_refused() {
        let mut session = RemoteSession::new();
        session.close().unwrap();
        assert_eq!(
            session.close(),
            Err(SessionError::InvalidTransition {
                from: SessionState::Closed
            })
        );
    }
}
