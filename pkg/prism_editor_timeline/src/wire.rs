//! Versioned deterministic command bytes with preallocation resource checks.
use crate::{
    Interpolation, KeyId, Keyframe, Timeline, TimelineEdit, TimelineError as Error, TimelineEvent,
    TimelineLimits, Track, TrackId,
};
use alloc::{string::String, vec::Vec};

const MAGIC: &[u8; 8] = b"PRSTLN01";
const DOCUMENT_MAGIC: &[u8; 8] = b"PRSTLD01";

/// Complete authoring data, independent from bounded command history. Stable
/// identities, channel values, tangents and event names survive save/reload.
#[derive(Clone, Debug, PartialEq)]
pub struct TimelineDocument {
    /// Integer ticks per second. Zero is rejected by both codec directions.
    pub ticks_per_second: u64,
    /// Authoritative authored tracks and events, without UI or undo state.
    pub timeline: Timeline,
}
impl TimelineDocument {
    /// Encodes a complete document with a separate file-size policy. Encoding
    /// borrows track payloads; it does not clone keys or manufacture a command.
    pub fn encode(&self, limits: TimelineLimits, max_bytes: usize) -> Result<Vec<u8>, Error> {
        if self.ticks_per_second == 0 {
            return Err(Error::EmptyInput);
        }
        check(self.timeline.track_count(), limits.max_tracks, "tracks")?;
        check(self.timeline.event_count(), limits.max_events, "events")?;
        let mut out = Writer {
            bytes: Vec::new(),
            limits: TimelineLimits {
                max_history_bytes: max_bytes,
                ..limits
            },
        };
        out.put(DOCUMENT_MAGIC)?;
        out.put(&self.ticks_per_second.to_le_bytes())?;
        out.count(self.timeline.track_count())?;
        let mut keys = 0usize;
        for track in self.timeline.tracks() {
            keys = keys
                .checked_add(track.key_count())
                .ok_or(Error::LimitExceeded("keys"))?;
            check(keys, limits.max_keys, "keys")?;
            out.track(track)?;
        }
        out.count(self.timeline.event_count())?;
        for event in self.timeline.events_in(0..=u64::MAX) {
            out.put(&event.id.to_le_bytes())?;
            out.put(&event.tick.to_le_bytes())?;
            out.text(&event.name)?;
        }
        Ok(out.bytes)
    }

    /// Decodes into an isolated candidate, rejecting invalid counts before
    /// allocation. Document track/event counts are independent of `max_edits` and
    /// `max_history_bytes`; `max_bytes` alone bounds encoded snapshot storage.
    pub fn decode(bytes: &[u8], limits: TimelineLimits, max_bytes: usize) -> Result<Self, Error> {
        check(bytes.len(), max_bytes, "document bytes")?;
        let mut input = Reader {
            bytes,
            limits,
            keys: 0,
        };
        if &input.array::<8>()? != DOCUMENT_MAGIC {
            return Err(Error::InvalidEncoding);
        }
        let ticks_per_second = u64::from_le_bytes(input.array()?);
        if ticks_per_second == 0 {
            return Err(Error::EmptyInput);
        }
        let tracks = input.count(limits.max_tracks, "tracks")?;
        check(tracks, input.bytes.len() / 29, "encoded tracks")?;
        let mut timeline = Timeline::default();
        for _ in 0..tracks {
            timeline.add_track(input.track()?)?;
        }
        let events = input.count(limits.max_events, "events")?;
        check(events, input.bytes.len() / 29, "encoded events")?;
        for _ in 0..events {
            timeline.add_event(TimelineEvent {
                id: input.id()?,
                tick: u64::from_le_bytes(input.array()?),
                name: input.text()?,
            })?;
        }
        if !input.bytes.is_empty() {
            return Err(Error::InvalidEncoding);
        }
        Ok(Self {
            ticks_per_second,
            timeline,
        })
    }
}

/// Replayable authoring command. The byte format preserves exact finite f64
/// bits and stable u128 identities, including slopes and removed track payloads.
#[derive(Clone, Debug, PartialEq)]
pub struct TimelineCommand {
    /// User-facing command label.
    pub label: String,
    /// Ordered edits, applied atomically by `TimelineEditor::execute`.
    pub edits: Vec<TimelineEdit>,
}
impl TimelineCommand {
    /// Encodes version 1 in little endian. No platform-sized integers occur on
    /// the wire. The encoded packet cannot exceed the history byte policy.
    pub fn encode(&self, limits: TimelineLimits) -> Result<Vec<u8>, Error> {
        if self.edits.is_empty() {
            return Err(Error::EmptyInput);
        }
        check(self.edits.len(), limits.max_edits, "transaction edits")?;
        let mut out = Writer {
            bytes: Vec::new(),
            limits,
        };
        out.put(MAGIC)?;
        out.text(&self.label)?;
        out.count(self.edits.len())?;
        let mut keys = 0usize;
        for edit in &self.edits {
            match edit {
                TimelineEdit::AddTrack(track) => {
                    keys = keys
                        .checked_add(track.key_count())
                        .ok_or(Error::LimitExceeded("keys"))?;
                    check(keys, limits.max_keys, "keys")?;
                    out.put(&[0])?;
                    out.track(track)?;
                }
                TimelineEdit::RemoveTrack(id) => {
                    out.put(&[1])?;
                    out.put(&id.raw().to_le_bytes())?;
                }
                TimelineEdit::PutKey { track, key } => {
                    keys = keys.checked_add(1).ok_or(Error::LimitExceeded("keys"))?;
                    check(keys, limits.max_keys, "keys")?;
                    out.put(&[2])?;
                    out.put(&track.raw().to_le_bytes())?;
                    out.key(key)?;
                }
                TimelineEdit::RemoveKey { track, key } => {
                    out.put(&[3])?;
                    out.put(&track.raw().to_le_bytes())?;
                    out.put(&key.raw().to_le_bytes())?;
                }
                TimelineEdit::AddEvent(event) => {
                    out.put(&[4])?;
                    out.put(&event.id.to_le_bytes())?;
                    out.put(&event.tick.to_le_bytes())?;
                    out.text(&event.name)?;
                }
                TimelineEdit::RemoveEvent(id) => {
                    out.put(&[5])?;
                    out.put(&id.to_le_bytes())?;
                }
            }
        }
        Ok(out.bytes)
    }
    /// Decodes a bounded packet. Counts and remaining bytes are checked before
    /// allocation; unknown tags, duplicate keys/ticks, nonfinite channels and
    /// extra trailing bytes are rejected. No document is mutated on failure.
    pub fn decode(bytes: &[u8], limits: TimelineLimits) -> Result<Self, Error> {
        check(bytes.len(), limits.max_history_bytes, "command bytes")?;
        let mut input = Reader {
            bytes,
            limits,
            keys: 0,
        };
        if &input.array::<8>()? != MAGIC {
            return Err(Error::InvalidEncoding);
        }
        let label = input.text()?;
        let count = input.count(limits.max_edits, "transaction edits")?;
        if count == 0 {
            return Err(Error::EmptyInput);
        }
        // Every edit requires a tag and at least one 128-bit identity.
        check(count, input.bytes.len() / 17, "encoded edits")?;
        let mut edits = Vec::with_capacity(count);
        for _ in 0..count {
            edits.push(match input.array::<1>()?[0] {
                0 => TimelineEdit::AddTrack(input.track()?),
                1 => TimelineEdit::RemoveTrack(TrackId::from_raw(input.id()?)),
                2 => {
                    let track = TrackId::from_raw(input.id()?);
                    TimelineEdit::PutKey {
                        track,
                        key: input.key()?,
                    }
                }
                3 => {
                    let track = TrackId::from_raw(input.id()?);
                    TimelineEdit::RemoveKey {
                        track,
                        key: KeyId::from_raw(input.id()?),
                    }
                }
                4 => TimelineEdit::AddEvent(TimelineEvent {
                    id: input.id()?,
                    tick: u64::from_le_bytes(input.array()?),
                    name: input.text()?,
                }),
                5 => TimelineEdit::RemoveEvent(input.id()?),
                _ => return Err(Error::InvalidEncoding),
            });
        }
        if !input.bytes.is_empty() {
            return Err(Error::InvalidEncoding);
        }
        Ok(Self { label, edits })
    }
}
fn check(count: usize, max: usize, name: &'static str) -> Result<(), Error> {
    if count > max {
        Err(Error::LimitExceeded(name))
    } else {
        Ok(())
    }
}
fn valid_text(text: &str, limits: TimelineLimits) -> Result<(), Error> {
    if text.trim().is_empty() {
        return Err(Error::EmptyInput);
    }
    check(text.len(), limits.max_text_bytes, "text bytes")
}
struct Writer {
    bytes: Vec<u8>,
    limits: TimelineLimits,
}
impl Writer {
    fn put(&mut self, bytes: &[u8]) -> Result<(), Error> {
        check(
            self.bytes.len().saturating_add(bytes.len()),
            self.limits.max_history_bytes,
            "command bytes",
        )?;
        self.bytes.extend_from_slice(bytes);
        Ok(())
    }
    fn count(&mut self, count: usize) -> Result<(), Error> {
        self.put(
            &u32::try_from(count)
                .map_err(|_| Error::InvalidEncoding)?
                .to_le_bytes(),
        )
    }
    fn text(&mut self, text: &str) -> Result<(), Error> {
        valid_text(text, self.limits)?;
        self.count(text.len())?;
        self.put(text.as_bytes())
    }
    fn track(&mut self, track: &Track) -> Result<(), Error> {
        check(track.dimension(), self.limits.max_dimension, "channels")?;
        self.put(&track.id().raw().to_le_bytes())?;
        self.text(track.name())?;
        self.count(track.dimension())?;
        self.count(track.key_count())?;
        for key in track.keyframes() {
            self.key(key)?;
        }
        Ok(())
    }
    fn key(&mut self, key: &Keyframe) -> Result<(), Error> {
        if key.values().is_empty() {
            return Err(Error::EmptyValue);
        }
        check(key.values().len(), self.limits.max_dimension, "channels")?;
        self.put(&key.id().raw().to_le_bytes())?;
        self.put(&key.tick().to_le_bytes())?;
        self.put(&[match key.interpolation() {
            Interpolation::Step => 0,
            Interpolation::Linear => 1,
            Interpolation::Cubic => 2,
        }])?;
        self.count(key.values().len())?;
        for vector in [key.values(), key.incoming(), key.outgoing()] {
            for value in vector {
                if !value.is_finite() {
                    return Err(Error::NonFiniteValue);
                }
                self.put(&value.to_bits().to_le_bytes())?;
            }
        }
        Ok(())
    }
}
struct Reader<'a> {
    bytes: &'a [u8],
    limits: TimelineLimits,
    keys: usize,
}
impl Reader<'_> {
    fn take(&mut self, count: usize) -> Result<&[u8], Error> {
        let (head, rest) = self
            .bytes
            .split_at_checked(count)
            .ok_or(Error::InvalidEncoding)?;
        self.bytes = rest;
        Ok(head)
    }
    fn array<const N: usize>(&mut self) -> Result<[u8; N], Error> {
        self.take(N)?.try_into().map_err(|_| Error::InvalidEncoding)
    }
    fn id(&mut self) -> Result<u128, Error> {
        Ok(u128::from_le_bytes(self.array()?))
    }
    fn track(&mut self) -> Result<Track, Error> {
        let id = TrackId::from_raw(self.id()?);
        let name = self.text()?;
        let dimension = self.count(self.limits.max_dimension, "channels")?;
        let count = self.count(self.limits.max_keys.saturating_sub(self.keys), "keys")?;
        // Even a zero-channel key requires identity, tick, mode and count.
        check(count, self.bytes.len() / 29, "encoded keys")?;
        let mut track = Track::new(id, name, dimension)?;
        for _ in 0..count {
            track.insert(self.key()?)?;
        }
        Ok(track)
    }
    fn count(&mut self, max: usize, name: &'static str) -> Result<usize, Error> {
        let count = usize::try_from(u32::from_le_bytes(self.array()?))
            .map_err(|_| Error::InvalidEncoding)?;
        check(count, max, name)?;
        Ok(count)
    }
    fn text(&mut self) -> Result<String, Error> {
        let count = self.count(self.limits.max_text_bytes, "text bytes")?;
        let limits = self.limits;
        let text = core::str::from_utf8(self.take(count)?).map_err(|_| Error::InvalidEncoding)?;
        valid_text(text, limits)?;
        Ok(String::from(text))
    }
    fn key(&mut self) -> Result<Keyframe, Error> {
        check(self.keys.saturating_add(1), self.limits.max_keys, "keys")?;
        self.keys += 1;
        let id = KeyId::from_raw(self.id()?);
        let tick = u64::from_le_bytes(self.array()?);
        let mode = match self.array::<1>()?[0] {
            0 => Interpolation::Step,
            1 => Interpolation::Linear,
            2 => Interpolation::Cubic,
            _ => return Err(Error::InvalidEncoding),
        };
        let dimension = self.count(self.limits.max_dimension, "channels")?;
        if dimension == 0 {
            return Err(Error::EmptyValue);
        }
        let byte_count = dimension.checked_mul(24).ok_or(Error::InvalidEncoding)?;
        let bytes = self.take(byte_count)?;
        let mut vectors = [
            Vec::with_capacity(dimension),
            Vec::with_capacity(dimension),
            Vec::with_capacity(dimension),
        ];
        for (vector, bytes) in vectors.iter_mut().zip(bytes.chunks_exact(dimension * 8)) {
            for bytes in bytes.chunks_exact(8) {
                let value = f64::from_bits(u64::from_le_bytes(
                    bytes.try_into().map_err(|_| Error::InvalidEncoding)?,
                ));
                if !value.is_finite() {
                    return Err(Error::NonFiniteValue);
                }
                vector.push(value);
            }
        }
        let [values, incoming, outgoing] = vectors;
        let mut key = Keyframe::new(id, tick, values, mode);
        key.set_tangents(incoming, outgoing)?;
        Ok(key)
    }
}
