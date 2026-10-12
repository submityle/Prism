//! The `prism_app` seam (design §25.2): drive replication sessions from the app's
//! frame loop.
//!
//! Replication is wired into the app shell as `prism_app` plugins:
//!
//! - [`ReplicationPlugin`] installs the **client** half — a [`Replication`]
//!   resource (a [`ClientSession`](crate::session::ClientSession) that predicts
//!   local input and reconciles authority) plus a `First` system that ticks its
//!   channel once per frame and drains the recovered frames into the prediction.
//! - [`ReplicationServerPlugin`] installs the **authority** half — a
//!   [`ServerReplication`] resource (a
//!   [`ServerSession`](crate::session::ServerSession) that decodes each inbound
//!   command, applies it, and broadcasts the framed delta) plus a `First` system
//!   that pumps it every frame.
//!
//! Both wrap the caller's [`Transport`](crate::transport::Transport) in a
//! [`ReliableChannel`](crate::reliable::ReliableChannel), so reliability
//! (sequence, ACK, retransmit) is *in* the frame loop: a dropped datagram is
//! retransmitted on a later frame rather than lost. Each plugin owns exactly one
//! transport — one peer per plugin — so a server hosting many clients instantiates
//! one per connection (a connection registry is the documented next step).

mod client;
mod server;

pub use client::{Replication, ReplicationPlugin, ReplicationStrategy};
pub use server::{ReplicationServerPlugin, ServerReplication, ServerStrategy};

/// The retransmit floor (in frames) both plugins use for their channel.
pub(crate) const DEFAULT_MIN_RTO: u32 = 2;
/// How many out-of-order frames a plugin's channel buffers before it refuses.
pub(crate) const DEFAULT_WINDOW: usize = 64;
