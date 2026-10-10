//! Remote / real-time observability (`remote` feature, M5).
//!
//! A running engine can open a socket so a self-hosted panel or CLI connects
//! live — streaming the log/metric/frame feed and the timeline — and send
//! runtime-tuning commands back without stopping the engine (design §15).
//!
//! ## Layout
//! - [`command`] — [`RemoteCommand`]s and the shared, lock-free
//!   [`RuntimeControls`] they mutate (log level, sink toggle, sampling ratio,
//!   capture requests), plus the [`CommandOutcome`] each produces.
//! - [`protocol`] — the [`RemoteEvent`] engine→panel feed and the hand-rolled,
//!   dependency-free binary codec (built on [`crate::wire`]) with
//!   length-prefixed framing for both events and commands.
//! - [`server`] — the loopback [`RemoteServer`]/[`RemoteServerHandle`] transport
//!   and the panel-side [`RemoteClient`]; the only part that touches
//!   `std::net`/`std::thread`.
//!
//! Everything here is behind the `remote` feature; the default build opens no
//! sockets and spawns no threads.

pub mod command;
pub mod protocol;
pub mod server;

pub use command::{CommandOutcome, RemoteCommand, RuntimeControls};
pub use protocol::{
    command_from_bytes, command_to_bytes, decode_command, encode_command, read_command, read_event,
    write_command, write_event, FrameSummary, RemoteEvent, MAX_FRAME_LEN,
};
pub use server::{RemoteClient, RemoteServer, RemoteServerHandle};
