//! Self-contained Tracy-compatible profiler backend (`tracy` feature).
//!
//! Tracy is a frame profiler whose client library streams a sequence of small
//! messages — zone begin/end, frame marks, plot samples, messages, GPU zones —
//! to its UI. The `tracy-client` crate is **not** a Prism workspace dependency,
//! and the diagnostic kernel must not pull in a non-workspace crate, so this
//! module implements a dependency-free emitter that produces that same
//! conceptual message stream and encodes it with the crate's [`wire`] codec.
//!
//! [`TracyBackend`] is a [`Profiler`] backend: install it with
//! [`set_profiler`](super::set_profiler) and every zone/frame/plot/message/GPU
//! event is translated into one or more [`TracyMessage`]s. A completed
//! [`Zone`](super::Zone) becomes a [`TracyMessage::ZoneBegin`] +
//! [`TracyMessage::ZoneEnd`] pair, matching Tracy's begin/end wire semantics.
//! The accumulated stream can be inspected ([`TracyBackend::snapshot`]),
//! drained, or serialized to bytes ([`TracyBackend::encode`]) for transport to
//! a real Tracy bridge or an archive. [`decode_stream`] reverses the encoding.
//!
//! [`wire`]: crate::wire

extern crate alloc;

use alloc::string::String;
use alloc::vec::Vec;
use std::sync::Mutex;

use crate::model::Level;
use crate::wire::{ByteReader, ByteWriter, WireError};

use super::{FrameMark, GpuZone, PlotValue, Profiler, Zone};

/// One message in the Tracy-compatible event stream.
///
/// This is a faithful, dependency-free model of the event kinds a Tracy client
/// emits. It is `Clone`/`PartialEq` so tests can assert exact round-trips.
#[derive(Clone, Debug, PartialEq)]
pub enum TracyMessage {
    /// A zone (scope) opened.
    ZoneBegin {
        /// Zone name.
        name: String,
        /// Optional category/track label.
        category: Option<String>,
        /// Owning thread id.
        thread_id: u64,
        /// Begin timestamp in nanoseconds.
        timestamp_nanos: u64,
        /// Nesting depth at entry.
        depth: u32,
    },
    /// The most recently opened zone on `thread_id` closed.
    ZoneEnd {
        /// Owning thread id.
        thread_id: u64,
        /// End timestamp in nanoseconds.
        timestamp_nanos: u64,
    },
    /// A frame boundary.
    FrameMark {
        /// `None` for the continuous primary frame, else the secondary frame name.
        name: Option<String>,
        /// Timestamp in nanoseconds.
        timestamp_nanos: u64,
    },
    /// A plot/counter sample.
    Plot {
        /// Series name.
        name: String,
        /// Sampled value.
        value: PlotValue,
        /// Timestamp in nanoseconds.
        timestamp_nanos: u64,
    },
    /// A free-form message.
    Message {
        /// Severity.
        level: Level,
        /// Message text.
        text: String,
        /// Timestamp in nanoseconds.
        timestamp_nanos: u64,
    },
    /// A resolved GPU zone.
    GpuZone {
        /// Zone name.
        name: String,
        /// Backend queue id.
        queue_id: u32,
        /// Start timestamp in nanoseconds.
        start_nanos: u64,
        /// Duration in nanoseconds.
        duration_nanos: u64,
        /// Optional CPU-side correlation token.
        correlation: Option<u64>,
    },
}

const TAG_ZONE_BEGIN: u8 = 1;
const TAG_ZONE_END: u8 = 2;
const TAG_FRAME_MARK: u8 = 3;
const TAG_PLOT: u8 = 4;
const TAG_MESSAGE: u8 = 5;
const TAG_GPU_ZONE: u8 = 6;

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

fn put_opt_str(w: &mut ByteWriter, value: Option<&str>) {
    match value {
        Some(s) => {
            w.put_bool(true);
            w.put_str(s);
        }
        None => w.put_bool(false),
    }
}

fn get_opt_str(r: &mut ByteReader<'_>) -> Result<Option<String>, WireError> {
    if r.get_bool()? {
        Ok(Some(r.get_str()?))
    } else {
        Ok(None)
    }
}

fn put_opt_u64(w: &mut ByteWriter, value: Option<u64>) {
    match value {
        Some(v) => {
            w.put_bool(true);
            w.put_u64(v);
        }
        None => w.put_bool(false),
    }
}

fn get_opt_u64(r: &mut ByteReader<'_>) -> Result<Option<u64>, WireError> {
    if r.get_bool()? {
        Ok(Some(r.get_u64()?))
    } else {
        Ok(None)
    }
}

impl TracyMessage {
    /// Append this message's tagged encoding to `w`.
    pub fn encode(&self, w: &mut ByteWriter) {
        match self {
            TracyMessage::ZoneBegin {
                name,
                category,
                thread_id,
                timestamp_nanos,
                depth,
            } => {
                w.put_u8(TAG_ZONE_BEGIN);
                w.put_str(name);
                put_opt_str(w, category.as_deref());
                w.put_u64(*thread_id);
                w.put_u64(*timestamp_nanos);
                w.put_u32(*depth);
            }
            TracyMessage::ZoneEnd {
                thread_id,
                timestamp_nanos,
            } => {
                w.put_u8(TAG_ZONE_END);
                w.put_u64(*thread_id);
                w.put_u64(*timestamp_nanos);
            }
            TracyMessage::FrameMark {
                name,
                timestamp_nanos,
            } => {
                w.put_u8(TAG_FRAME_MARK);
                put_opt_str(w, name.as_deref());
                w.put_u64(*timestamp_nanos);
            }
            TracyMessage::Plot {
                name,
                value,
                timestamp_nanos,
            } => {
                w.put_u8(TAG_PLOT);
                w.put_str(name);
                put_plot_value(w, *value);
                w.put_u64(*timestamp_nanos);
            }
            TracyMessage::Message {
                level,
                text,
                timestamp_nanos,
            } => {
                w.put_u8(TAG_MESSAGE);
                put_level(w, *level);
                w.put_str(text);
                w.put_u64(*timestamp_nanos);
            }
            TracyMessage::GpuZone {
                name,
                queue_id,
                start_nanos,
                duration_nanos,
                correlation,
            } => {
                w.put_u8(TAG_GPU_ZONE);
                w.put_str(name);
                w.put_u32(*queue_id);
                w.put_u64(*start_nanos);
                w.put_u64(*duration_nanos);
                put_opt_u64(w, *correlation);
            }
        }
    }

    /// Decode one tagged message from `r`.
    pub fn decode(r: &mut ByteReader<'_>) -> Result<Self, WireError> {
        match r.get_u8()? {
            TAG_ZONE_BEGIN => Ok(TracyMessage::ZoneBegin {
                name: r.get_str()?,
                category: get_opt_str(r)?,
                thread_id: r.get_u64()?,
                timestamp_nanos: r.get_u64()?,
                depth: r.get_u32()?,
            }),
            TAG_ZONE_END => Ok(TracyMessage::ZoneEnd {
                thread_id: r.get_u64()?,
                timestamp_nanos: r.get_u64()?,
            }),
            TAG_FRAME_MARK => Ok(TracyMessage::FrameMark {
                name: get_opt_str(r)?,
                timestamp_nanos: r.get_u64()?,
            }),
            TAG_PLOT => Ok(TracyMessage::Plot {
                name: r.get_str()?,
                value: get_plot_value(r)?,
                timestamp_nanos: r.get_u64()?,
            }),
            TAG_MESSAGE => Ok(TracyMessage::Message {
                level: get_level(r)?,
                text: r.get_str()?,
                timestamp_nanos: r.get_u64()?,
            }),
            TAG_GPU_ZONE => Ok(TracyMessage::GpuZone {
                name: r.get_str()?,
                queue_id: r.get_u32()?,
                start_nanos: r.get_u64()?,
                duration_nanos: r.get_u64()?,
                correlation: get_opt_u64(r)?,
            }),
            other => Err(WireError::InvalidTag(other)),
        }
    }
}

/// Encode a message stream into a flat byte buffer (one message after another).
pub fn encode_stream(messages: &[TracyMessage]) -> Vec<u8> {
    let mut w = ByteWriter::new();
    w.put_u32(messages.len() as u32);
    for m in messages {
        m.encode(&mut w);
    }
    w.into_vec()
}

/// Decode a byte buffer produced by [`encode_stream`] back into messages.
pub fn decode_stream(bytes: &[u8]) -> Result<Vec<TracyMessage>, WireError> {
    let mut r = ByteReader::new(bytes);
    let count = r.get_u32()? as usize;
    let mut out = Vec::with_capacity(count);
    for _ in 0..count {
        out.push(TracyMessage::decode(&mut r)?);
    }
    Ok(out)
}

/// A Tracy-compatible [`Profiler`] backend that accumulates the emitted message
/// stream in memory.
///
/// Each profiler event is translated to one or more [`TracyMessage`]s; a
/// completed [`Zone`] expands to a begin/end pair. The stream can be inspected,
/// drained, or [`encode`](TracyBackend::encode)d for transport.
#[derive(Debug, Default)]
pub struct TracyBackend {
    stream: Mutex<Vec<TracyMessage>>,
}

impl TracyBackend {
    /// Create an empty backend.
    pub fn new() -> Self {
        Self::default()
    }

    fn push(&self, message: TracyMessage) {
        self.stream.lock().unwrap().push(message);
    }

    /// Clone the accumulated message stream in emission order.
    pub fn snapshot(&self) -> Vec<TracyMessage> {
        self.stream.lock().unwrap().clone()
    }

    /// Remove and return the accumulated message stream.
    pub fn drain(&self) -> Vec<TracyMessage> {
        core::mem::take(&mut *self.stream.lock().unwrap())
    }

    /// Number of accumulated messages.
    pub fn len(&self) -> usize {
        self.stream.lock().unwrap().len()
    }

    /// Whether no messages have been emitted.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Serialize the accumulated stream with [`encode_stream`].
    pub fn encode(&self) -> Vec<u8> {
        encode_stream(&self.snapshot())
    }
}

impl Profiler for TracyBackend {
    fn zone(&self, zone: &Zone) {
        self.push(TracyMessage::ZoneBegin {
            name: zone.name.clone(),
            category: zone.category.clone(),
            thread_id: zone.thread_id,
            timestamp_nanos: zone.start_nanos,
            depth: zone.depth,
        });
        self.push(TracyMessage::ZoneEnd {
            thread_id: zone.thread_id,
            timestamp_nanos: zone.end_nanos(),
        });
    }

    fn frame_mark(&self, mark: &FrameMark, timestamp_nanos: u64) {
        let name = match mark {
            FrameMark::Continuous => None,
            FrameMark::Named(n) => Some(n.clone()),
        };
        self.push(TracyMessage::FrameMark {
            name,
            timestamp_nanos,
        });
    }

    fn plot(&self, name: &str, value: PlotValue, timestamp_nanos: u64) {
        self.push(TracyMessage::Plot {
            name: String::from(name),
            value,
            timestamp_nanos,
        });
    }

    fn message(&self, level: Level, text: &str, timestamp_nanos: u64) {
        self.push(TracyMessage::Message {
            level,
            text: String::from(text),
            timestamp_nanos,
        });
    }

    fn gpu_zone(&self, zone: &GpuZone) {
        self.push(TracyMessage::GpuZone {
            name: zone.name.clone(),
            queue_id: zone.queue_id,
            start_nanos: zone.start_nanos,
            duration_nanos: zone.duration_nanos,
            correlation: zone.correlation,
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_stream() -> Vec<TracyMessage> {
        alloc::vec![
            TracyMessage::ZoneBegin {
                name: String::from("update"),
                category: Some(String::from("ecs")),
                thread_id: 2,
                timestamp_nanos: 100,
                depth: 0,
            },
            TracyMessage::ZoneEnd {
                thread_id: 2,
                timestamp_nanos: 150,
            },
            TracyMessage::FrameMark {
                name: None,
                timestamp_nanos: 160,
            },
            TracyMessage::FrameMark {
                name: Some(String::from("render")),
                timestamp_nanos: 161,
            },
            TracyMessage::Plot {
                name: String::from("drawcalls"),
                value: PlotValue::U64(42),
                timestamp_nanos: 170,
            },
            TracyMessage::Message {
                level: Level::Warn,
                text: String::from("over budget"),
                timestamp_nanos: 180,
            },
            TracyMessage::GpuZone {
                name: String::from("shadow"),
                queue_id: 1,
                start_nanos: 200,
                duration_nanos: 30,
                correlation: Some(7),
            },
        ]
    }

    #[test]
    fn stream_round_trips() {
        let messages = sample_stream();
        let bytes = encode_stream(&messages);
        let decoded = decode_stream(&bytes).unwrap();
        assert_eq!(decoded, messages);
    }

    #[test]
    fn backend_expands_zone_to_begin_end_pair() {
        let backend = TracyBackend::new();
        backend.zone(&Zone {
            name: String::from("z"),
            category: None,
            thread_id: 4,
            start_nanos: 10,
            duration_nanos: 5,
            depth: 1,
        });
        let msgs = backend.snapshot();
        assert_eq!(msgs.len(), 2);
        assert!(matches!(msgs[0], TracyMessage::ZoneBegin { depth: 1, .. }));
        assert_eq!(
            msgs[1],
            TracyMessage::ZoneEnd {
                thread_id: 4,
                timestamp_nanos: 15,
            }
        );
    }

    #[test]
    fn backend_records_all_event_kinds_and_drains() {
        let backend = TracyBackend::new();
        backend.frame_mark(&FrameMark::Continuous, 1);
        backend.plot("fps", PlotValue::F64(60.0), 2);
        backend.message(Level::Info, "hello", 3);
        backend.gpu_zone(&GpuZone {
            name: String::from("g"),
            queue_id: 0,
            start_nanos: 4,
            duration_nanos: 6,
            correlation: None,
        });
        assert_eq!(backend.len(), 4);

        // Encoding then decoding reproduces the live stream exactly.
        let decoded = decode_stream(&backend.encode()).unwrap();
        assert_eq!(decoded, backend.snapshot());

        let drained = backend.drain();
        assert_eq!(drained.len(), 4);
        assert!(backend.is_empty());
    }

    #[test]
    fn decode_rejects_bad_tag() {
        let mut w = ByteWriter::new();
        w.put_u32(1);
        w.put_u8(250);
        let err = decode_stream(&w.into_vec()).unwrap_err();
        assert_eq!(err, WireError::InvalidTag(250));
    }
}
