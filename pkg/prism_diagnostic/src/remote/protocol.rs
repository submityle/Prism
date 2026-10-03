//! Remote observability wire protocol (`remote` feature).
//!
//! Defines the two message families exchanged with a remote panel and their
//! hand-rolled binary codec (built on [`wire`](crate::wire); no `serde`):
//! - [`RemoteEvent`] — engine → panel: log events, metric samples, frame
//!   summaries, timeline zones, and plot samples.
//! - [`RemoteCommand`] — panel → engine (defined in
//!   [`command`](super::command)): runtime-tuning requests.
//!
//! Each message is length-prefixed when written to a stream via
//! [`write_event`]/[`write_command`] (and read back with
//! [`read_event`]/[`read_command`]), giving self-delimiting frames over a raw
//! socket. A `u32` length cap ([`MAX_FRAME_LEN`]) bounds reader allocation so a
//! malicious or corrupt peer cannot force an unbounded buffer.

extern crate alloc;

use alloc::string::String;
use alloc::vec;
use alloc::vec::Vec;
use std::io::{self, Read, Write};

use crate::metrics::FrameStatsSnapshot;
use crate::model::{Event, FieldValue, Level};
use crate::profiler::{PlotValue, Zone};
use crate::wire::{ByteReader, ByteWriter, WireError};

use super::command::RemoteCommand;

/// Maximum accepted frame payload length (16 MiB), guarding reader allocation.
pub const MAX_FRAME_LEN: u32 = 16 * 1024 * 1024;

/// A compact frame-statistics summary carried to the panel.
///
/// This mirrors the numeric core of [`FrameStatsSnapshot`] without the
/// per-counter map, which travels as separate [`RemoteEvent::Metric`]s.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct FrameSummary {
    /// Monotonic frame index.
    pub frame_index: u64,
    /// Most recent frame delta in milliseconds.
    pub last_delta_ms: f64,
    /// Window minimum frame time in milliseconds.
    pub min_ms: f64,
    /// Window average frame time in milliseconds.
    pub avg_ms: f64,
    /// Window maximum frame time in milliseconds.
    pub max_ms: f64,
    /// Window average FPS.
    pub fps: f64,
}

impl From<&FrameStatsSnapshot> for FrameSummary {
    fn from(s: &FrameStatsSnapshot) -> Self {
        Self {
            frame_index: s.frame_index,
            last_delta_ms: s.last_delta_ms,
            min_ms: s.min_ms,
            avg_ms: s.avg_ms,
            max_ms: s.max_ms,
            fps: s.fps,
        }
    }
}

/// An event streamed from the engine to a remote panel.
#[derive(Clone, Debug, PartialEq)]
pub enum RemoteEvent {
    /// A structured log event (level + target + message + timestamp).
    Log {
        /// Severity.
        level: Level,
        /// Source subsystem/target.
        target: String,
        /// Rendered message text.
        message: String,
        /// Monotonic timestamp in nanoseconds.
        timestamp_nanos: u64,
    },
    /// A single named metric sample.
    Metric {
        /// Metric name.
        name: String,
        /// Sampled value.
        value: f64,
        /// Monotonic timestamp in nanoseconds.
        timestamp_nanos: u64,
    },
    /// A per-frame statistics summary.
    Frame(FrameSummary),
    /// One completed timeline zone.
    Zone {
        /// Zone name.
        name: String,
        /// Owning thread id.
        thread_id: u64,
        /// Start timestamp in nanoseconds.
        start_nanos: u64,
        /// Duration in nanoseconds.
        duration_nanos: u64,
        /// Nesting depth.
        depth: u32,
    },
    /// A plot/counter sample for a named series.
    Plot {
        /// Series name.
        name: String,
        /// Sampled value.
        value: PlotValue,
        /// Monotonic timestamp in nanoseconds.
        timestamp_nanos: u64,
    },
}

const EV_LOG: u8 = 1;
const EV_METRIC: u8 = 2;
const EV_FRAME: u8 = 3;
const EV_ZONE: u8 = 4;
const EV_PLOT: u8 = 5;

const CMD_SET_LOG_LEVEL: u8 = 1;
const CMD_TOGGLE_SINK: u8 = 2;
const CMD_SET_SAMPLING: u8 = 3;
const CMD_TRIGGER_CAPTURE: u8 = 4;
const CMD_REQUEST_SNAPSHOT: u8 = 5;
const CMD_SHUTDOWN: u8 = 6;

const PLOT_I64: u8 = 0;
const PLOT_U64: u8 = 1;
const PLOT_F64: u8 = 2;

fn put_level(w: &mut ByteWriter, level: Level) {
    w.put_u8(level as u8);
}

fn get_level(r: &mut ByteReader<'_>) -> Result<Level, WireError> {
    match r.get_u8()? {
        0 => Ok(Level::Trace),
        1 => Ok(Level::Debug),
        2 => Ok(Level::Info),
        3 => Ok(Level::Warn),
        4 => Ok(Level::Error),
        _ => Err(WireError::InvalidValue),
    }
}

fn put_plot_value(w: &mut ByteWriter, value: PlotValue) {
    match value {
        PlotValue::I64(v) => {
            w.put_u8(PLOT_I64);
            w.put_i64(v);
        }
        PlotValue::U64(v) => {
            w.put_u8(PLOT_U64);
            w.put_u64(v);
        }
        PlotValue::F64(v) => {
            w.put_u8(PLOT_F64);
            w.put_f64(v);
        }
    }
}

fn get_plot_value(r: &mut ByteReader<'_>) -> Result<PlotValue, WireError> {
    match r.get_u8()? {
        PLOT_I64 => Ok(PlotValue::I64(r.get_i64()?)),
        PLOT_U64 => Ok(PlotValue::U64(r.get_u64()?)),
        PLOT_F64 => Ok(PlotValue::F64(r.get_f64()?)),
        other => Err(WireError::InvalidTag(other)),
    }
}

impl RemoteEvent {
    /// Build a [`RemoteEvent::Log`] from a diagnostic [`Event`], flattening the
    /// message so the field list need not cross the wire separately.
    pub fn from_event(event: &Event) -> Self {
        let mut message = event.message.clone();
        for i in 0..event.fields.len() {
            if let Some(field) = event.fields.get(i) {
                message.push(' ');
                message.push_str(field.key);
                message.push('=');
                match &field.value {
                    FieldValue::Str(s) => message.push_str(s),
                    FieldValue::I64(v) => message.push_str(&alloc::format!("{v}")),
                    FieldValue::U64(v) => message.push_str(&alloc::format!("{v}")),
                    FieldValue::F64(v) => message.push_str(&alloc::format!("{v}")),
                    FieldValue::Bool(v) => message.push_str(&alloc::format!("{v}")),
                }
            }
        }
        RemoteEvent::Log {
            level: event.level,
            target: String::from(event.target),
            message,
            timestamp_nanos: event.timestamp_nanos,
        }
    }

    /// Build a [`RemoteEvent::Zone`] from a profiler [`Zone`].
    pub fn from_zone(zone: &Zone) -> Self {
        RemoteEvent::Zone {
            name: zone.name.clone(),
            thread_id: zone.thread_id,
            start_nanos: zone.start_nanos,
            duration_nanos: zone.duration_nanos,
            depth: zone.depth,
        }
    }

    /// Encode this event (without a length prefix) into `w`.
    pub fn encode(&self, w: &mut ByteWriter) {
        match self {
            RemoteEvent::Log {
                level,
                target,
                message,
                timestamp_nanos,
            } => {
                w.put_u8(EV_LOG);
                put_level(w, *level);
                w.put_str(target);
                w.put_str(message);
                w.put_u64(*timestamp_nanos);
            }
            RemoteEvent::Metric {
                name,
                value,
                timestamp_nanos,
            } => {
                w.put_u8(EV_METRIC);
                w.put_str(name);
                w.put_f64(*value);
                w.put_u64(*timestamp_nanos);
            }
            RemoteEvent::Frame(summary) => {
                w.put_u8(EV_FRAME);
                w.put_u64(summary.frame_index);
                w.put_f64(summary.last_delta_ms);
                w.put_f64(summary.min_ms);
                w.put_f64(summary.avg_ms);
                w.put_f64(summary.max_ms);
                w.put_f64(summary.fps);
            }
            RemoteEvent::Zone {
                name,
                thread_id,
                start_nanos,
                duration_nanos,
                depth,
            } => {
                w.put_u8(EV_ZONE);
                w.put_str(name);
                w.put_u64(*thread_id);
                w.put_u64(*start_nanos);
                w.put_u64(*duration_nanos);
                w.put_u32(*depth);
            }
            RemoteEvent::Plot {
                name,
                value,
                timestamp_nanos,
            } => {
                w.put_u8(EV_PLOT);
                w.put_str(name);
                put_plot_value(w, *value);
                w.put_u64(*timestamp_nanos);
            }
        }
    }

    /// Decode one event (without a length prefix) from `r`.
    pub fn decode(r: &mut ByteReader<'_>) -> Result<Self, WireError> {
        match r.get_u8()? {
            EV_LOG => Ok(RemoteEvent::Log {
                level: get_level(r)?,
                target: r.get_str()?,
                message: r.get_str()?,
                timestamp_nanos: r.get_u64()?,
            }),
            EV_METRIC => Ok(RemoteEvent::Metric {
                name: r.get_str()?,
                value: r.get_f64()?,
                timestamp_nanos: r.get_u64()?,
            }),
            EV_FRAME => Ok(RemoteEvent::Frame(FrameSummary {
                frame_index: r.get_u64()?,
                last_delta_ms: r.get_f64()?,
                min_ms: r.get_f64()?,
                avg_ms: r.get_f64()?,
                max_ms: r.get_f64()?,
                fps: r.get_f64()?,
            })),
            EV_ZONE => Ok(RemoteEvent::Zone {
                name: r.get_str()?,
                thread_id: r.get_u64()?,
                start_nanos: r.get_u64()?,
                duration_nanos: r.get_u64()?,
                depth: r.get_u32()?,
            }),
            EV_PLOT => Ok(RemoteEvent::Plot {
                name: r.get_str()?,
                value: get_plot_value(r)?,
                timestamp_nanos: r.get_u64()?,
            }),
            other => Err(WireError::InvalidTag(other)),
        }
    }

    /// Encode this event into a fresh byte buffer (no length prefix).
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut w = ByteWriter::new();
        self.encode(&mut w);
        w.into_vec()
    }

    /// Decode an event from a complete byte buffer (no length prefix).
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, WireError> {
        let mut r = ByteReader::new(bytes);
        Self::decode(&mut r)
    }
}

/// Encode a [`RemoteCommand`] (without a length prefix) into `w`.
pub fn encode_command(command: &RemoteCommand, w: &mut ByteWriter) {
    match command {
        RemoteCommand::SetLogLevel(level) => {
            w.put_u8(CMD_SET_LOG_LEVEL);
            put_level(w, *level);
        }
        RemoteCommand::ToggleSink { enabled } => {
            w.put_u8(CMD_TOGGLE_SINK);
            w.put_bool(*enabled);
        }
        RemoteCommand::SetSampling {
            numerator,
            denominator,
        } => {
            w.put_u8(CMD_SET_SAMPLING);
            w.put_u32(*numerator);
            w.put_u32(*denominator);
        }
        RemoteCommand::TriggerCapture => w.put_u8(CMD_TRIGGER_CAPTURE),
        RemoteCommand::RequestSnapshot => w.put_u8(CMD_REQUEST_SNAPSHOT),
        RemoteCommand::Shutdown => w.put_u8(CMD_SHUTDOWN),
    }
}

/// Decode a [`RemoteCommand`] (without a length prefix) from `r`.
pub fn decode_command(r: &mut ByteReader<'_>) -> Result<RemoteCommand, WireError> {
    match r.get_u8()? {
        CMD_SET_LOG_LEVEL => Ok(RemoteCommand::SetLogLevel(get_level(r)?)),
        CMD_TOGGLE_SINK => Ok(RemoteCommand::ToggleSink {
            enabled: r.get_bool()?,
        }),
        CMD_SET_SAMPLING => Ok(RemoteCommand::SetSampling {
            numerator: r.get_u32()?,
            denominator: r.get_u32()?,
        }),
        CMD_TRIGGER_CAPTURE => Ok(RemoteCommand::TriggerCapture),
        CMD_REQUEST_SNAPSHOT => Ok(RemoteCommand::RequestSnapshot),
        CMD_SHUTDOWN => Ok(RemoteCommand::Shutdown),
        other => Err(WireError::InvalidTag(other)),
    }
}

/// Encode a command into a fresh byte buffer (no length prefix).
pub fn command_to_bytes(command: &RemoteCommand) -> Vec<u8> {
    let mut w = ByteWriter::new();
    encode_command(command, &mut w);
    w.into_vec()
}

/// Decode a command from a complete byte buffer (no length prefix).
pub fn command_from_bytes(bytes: &[u8]) -> Result<RemoteCommand, WireError> {
    let mut r = ByteReader::new(bytes);
    decode_command(&mut r)
}

fn write_frame<W: Write>(w: &mut W, payload: &[u8]) -> io::Result<()> {
    let len = u32::try_from(payload.len())
        .map_err(|_| io::Error::new(io::ErrorKind::InvalidInput, "frame payload too large"))?;
    w.write_all(&len.to_le_bytes())?;
    w.write_all(payload)?;
    w.flush()
}

fn read_frame<R: Read>(r: &mut R) -> io::Result<Vec<u8>> {
    let mut len_bytes = [0u8; 4];
    r.read_exact(&mut len_bytes)?;
    let len = u32::from_le_bytes(len_bytes);
    if len > MAX_FRAME_LEN {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "frame length exceeds MAX_FRAME_LEN",
        ));
    }
    let mut payload = vec![0u8; len as usize];
    r.read_exact(&mut payload)?;
    Ok(payload)
}

fn wire_to_io(err: WireError) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, err)
}

/// Write a length-prefixed [`RemoteEvent`] frame to `w`.
pub fn write_event<W: Write>(w: &mut W, event: &RemoteEvent) -> io::Result<()> {
    write_frame(w, &event.to_bytes())
}

/// Read a length-prefixed [`RemoteEvent`] frame from `r`.
pub fn read_event<R: Read>(r: &mut R) -> io::Result<RemoteEvent> {
    let payload = read_frame(r)?;
    RemoteEvent::from_bytes(&payload).map_err(wire_to_io)
}

/// Write a length-prefixed [`RemoteCommand`] frame to `w`.
pub fn write_command<W: Write>(w: &mut W, command: &RemoteCommand) -> io::Result<()> {
    write_frame(w, &command_to_bytes(command))
}

/// Read a length-prefixed [`RemoteCommand`] frame from `r`.
pub fn read_command<R: Read>(r: &mut R) -> io::Result<RemoteCommand> {
    let payload = read_frame(r)?;
    command_from_bytes(&payload).map_err(wire_to_io)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::Event;

    fn sample_events() -> Vec<RemoteEvent> {
        vec![
            RemoteEvent::Log {
                level: Level::Error,
                target: String::from("render"),
                message: String::from("device lost"),
                timestamp_nanos: 123,
            },
            RemoteEvent::Metric {
                name: String::from("heap_mb"),
                value: 512.5,
                timestamp_nanos: 200,
            },
            RemoteEvent::Frame(FrameSummary {
                frame_index: 9,
                last_delta_ms: 16.6,
                min_ms: 15.0,
                avg_ms: 16.0,
                max_ms: 20.0,
                fps: 62.5,
            }),
            RemoteEvent::Zone {
                name: String::from("sim"),
                thread_id: 3,
                start_nanos: 1,
                duration_nanos: 2,
                depth: 1,
            },
            RemoteEvent::Plot {
                name: String::from("entities"),
                value: PlotValue::U64(4096),
                timestamp_nanos: 300,
            },
        ]
    }

    #[test]
    fn event_round_trips_through_bytes() {
        for event in sample_events() {
            let bytes = event.to_bytes();
            assert_eq!(RemoteEvent::from_bytes(&bytes).unwrap(), event);
        }
    }

    #[test]
    fn command_round_trips_through_bytes() {
        let commands = [
            RemoteCommand::SetLogLevel(Level::Warn),
            RemoteCommand::ToggleSink { enabled: true },
            RemoteCommand::SetSampling {
                numerator: 1,
                denominator: 8,
            },
            RemoteCommand::TriggerCapture,
            RemoteCommand::RequestSnapshot,
            RemoteCommand::Shutdown,
        ];
        for command in commands {
            let bytes = command_to_bytes(&command);
            assert_eq!(command_from_bytes(&bytes).unwrap(), command);
        }
    }

    #[test]
    fn framed_events_round_trip_over_a_buffer() {
        let events = sample_events();
        let mut buf: Vec<u8> = Vec::new();
        for event in &events {
            write_event(&mut buf, event).unwrap();
        }
        let mut cursor = io::Cursor::new(buf);
        for event in &events {
            assert_eq!(&read_event(&mut cursor).unwrap(), event);
        }
    }

    #[test]
    fn framed_commands_round_trip_over_a_buffer() {
        let command = RemoteCommand::SetSampling {
            numerator: 3,
            denominator: 4,
        };
        let mut buf: Vec<u8> = Vec::new();
        write_command(&mut buf, &command).unwrap();
        let mut cursor = io::Cursor::new(buf);
        assert_eq!(read_command(&mut cursor).unwrap(), command);
    }

    #[test]
    fn from_event_flattens_fields_into_message() {
        let event = Event::new(Level::Info, "subsys", "spawned")
            .with_field("count", FieldValue::U64(3))
            .with_field("ok", FieldValue::Bool(true));
        match RemoteEvent::from_event(&event) {
            RemoteEvent::Log { message, .. } => {
                assert_eq!(message, "spawned count=3 ok=true");
            }
            other => panic!("expected log event, got {other:?}"),
        }
    }

    #[test]
    fn oversized_frame_is_rejected() {
        let mut buf: Vec<u8> = Vec::new();
        buf.extend_from_slice(&(MAX_FRAME_LEN + 1).to_le_bytes());
        let mut cursor = io::Cursor::new(buf);
        let err = read_event(&mut cursor).unwrap_err();
        assert_eq!(err.kind(), io::ErrorKind::InvalidData);
    }

    #[test]
    fn truncated_frame_is_unexpected_eof() {
        let mut buf: Vec<u8> = Vec::new();
        // Claim 10 bytes, provide none.
        buf.extend_from_slice(&10u32.to_le_bytes());
        let mut cursor = io::Cursor::new(buf);
        let err = read_event(&mut cursor).unwrap_err();
        assert_eq!(err.kind(), io::ErrorKind::UnexpectedEof);
    }
}
