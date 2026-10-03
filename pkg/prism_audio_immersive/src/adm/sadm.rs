//! Serial ADM (S-ADM, ITU-R BS.2125): frame-serialised ADM metadata.
//!
//! Where [`crate::adm::model`] describes a whole programme at once, S-ADM
//! carries the same object trajectories and bed configuration as a sequence of
//! per-frame updates suitable for live and streaming delivery. This module
//! models that frame sequence, serialises it to a deterministic little-endian
//! byte stream (round-trip byte identical), and samples a continuous object
//! trajectory by linear interpolation between frames.
//!
//! # Provenance
//!
//! Original work; no Unreal Engine, Unity, Godot, Wwise, FMOD, Steam Audio,
//! Google Resonance Audio, Dolby, or MPEG source or derived code; no AI/ML.
//! Implements the publicly published Serial ADM frame model (ITU-R BS.2125)
//! over the ADM metadata of ITU-R BS.2076.
//!
//! # Relationship
//!
//! Shares the [`crate::adm::model::AdmPosition`] polar convention with the
//! static ADM graph and feeds the same source-position truth consumed by the
//! [`crate::array`] panners for real-time rendering.

use alloc::vec::Vec;

use prism_audio_core::math::Sample;

use crate::adm::model::AdmPosition;

/// The S-ADM stream magic word.
const MAGIC: &[u8; 4] = b"SADM";
/// The serialised stream version.
const VERSION: u16 = 1;

/// Whether a frame carries a complete scene or only a divided update.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub enum FrameType {
    /// A full frame: a complete snapshot of every active object.
    Full,
    /// A divided frame: a partial update continuing the previous state.
    Divided,
}

impl FrameType {
    /// The wire tag byte for this frame type.
    #[must_use]
    fn tag(self) -> u8 {
        match self {
            FrameType::Full => 0,
            FrameType::Divided => 1,
        }
    }

    /// Parses a wire tag byte back into a [`FrameType`].
    #[must_use]
    fn from_tag(tag: u8) -> Option<Self> {
        match tag {
            0 => Some(FrameType::Full),
            1 => Some(FrameType::Divided),
            _ => None,
        }
    }
}

/// A failure while parsing an S-ADM stream.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SadmError {
    /// The byte stream ended before a field could be read.
    Truncated,
    /// The leading magic word or version was wrong.
    BadHeader,
    /// A frame-type tag was out of range.
    BadFrameType,
}

/// A frame header: its index, timing window, and type.
#[derive(Clone, Copy, Debug, PartialEq)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub struct SadmFrameFormat {
    /// Zero-based frame index.
    pub frame_index: u32,
    /// Frame start time in seconds.
    pub start: Sample,
    /// Frame duration in seconds.
    pub duration: Sample,
    /// Whether the frame is full or divided.
    pub frame_type: FrameType,
}

impl SadmFrameFormat {
    /// Builds a frame header.
    #[must_use]
    pub fn new(frame_index: u32, start: Sample, duration: Sample, frame_type: FrameType) -> Self {
        Self {
            frame_index,
            start,
            duration,
            frame_type,
        }
    }
}

/// One object's position and gain within a frame.
#[derive(Clone, Copy, Debug, PartialEq)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub struct SadmObjectUpdate {
    /// The object identifier this update applies to.
    pub object_id: u32,
    /// The object's polar position during the frame.
    pub position: AdmPosition,
    /// Linear gain during the frame.
    pub gain: Sample,
}

impl SadmObjectUpdate {
    /// Builds an object update.
    #[must_use]
    pub fn new(object_id: u32, position: AdmPosition, gain: Sample) -> Self {
        Self {
            object_id,
            position,
            gain,
        }
    }
}

/// A single S-ADM frame: a header plus the object updates it carries.
#[derive(Clone, Debug, Default, PartialEq)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub struct SadmFrame {
    /// The frame header, or a default full frame at the origin of time.
    pub format: SadmFrameFormat,
    /// The object updates in this frame.
    pub updates: Vec<SadmObjectUpdate>,
}

impl Default for SadmFrameFormat {
    #[inline]
    fn default() -> Self {
        Self {
            frame_index: 0,
            start: 0.0,
            duration: 0.0,
            frame_type: FrameType::Full,
        }
    }
}

impl SadmFrame {
    /// Builds a frame from a header.
    #[must_use]
    pub fn new(format: SadmFrameFormat) -> Self {
        Self {
            format,
            updates: Vec::new(),
        }
    }

    /// Appends an object update.
    pub fn push(&mut self, update: SadmObjectUpdate) {
        self.updates.push(update);
    }

    /// Finds the update for `object_id` within this frame.
    #[must_use]
    pub fn update_for(&self, object_id: u32) -> Option<&SadmObjectUpdate> {
        self.updates.iter().find(|u| u.object_id == object_id)
    }
}

/// A complete S-ADM sequence: ordered frames plus a nominal frame rate.
#[derive(Clone, Debug, Default, PartialEq)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub struct SadmSequence {
    /// Frames in ascending start-time order.
    pub frames: Vec<SadmFrame>,
    /// Nominal frame rate in frames per second.
    pub frame_rate: Sample,
}

impl SadmSequence {
    /// Builds a sequence with the given frame rate and no frames.
    #[must_use]
    pub fn new(frame_rate: Sample) -> Self {
        Self {
            frames: Vec::new(),
            frame_rate,
        }
    }

    /// Appends a frame.
    pub fn push(&mut self, frame: SadmFrame) {
        self.frames.push(frame);
    }

    /// Serialises the sequence to a deterministic little-endian byte stream.
    #[must_use]
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut out = Vec::new();
        out.extend_from_slice(MAGIC);
        out.extend_from_slice(&VERSION.to_le_bytes());
        out.extend_from_slice(&self.frame_rate.to_le_bytes());
        let frame_count = u32::try_from(self.frames.len()).unwrap_or(u32::MAX);
        out.extend_from_slice(&frame_count.to_le_bytes());
        for frame in &self.frames {
            out.extend_from_slice(&frame.format.frame_index.to_le_bytes());
            out.push(frame.format.frame_type.tag());
            out.extend_from_slice(&frame.format.start.to_le_bytes());
            out.extend_from_slice(&frame.format.duration.to_le_bytes());
            let update_count = u32::try_from(frame.updates.len()).unwrap_or(u32::MAX);
            out.extend_from_slice(&update_count.to_le_bytes());
            for update in &frame.updates {
                out.extend_from_slice(&update.object_id.to_le_bytes());
                out.extend_from_slice(&update.position.azimuth.to_le_bytes());
                out.extend_from_slice(&update.position.elevation.to_le_bytes());
                out.extend_from_slice(&update.position.distance.to_le_bytes());
                out.extend_from_slice(&update.gain.to_le_bytes());
            }
        }
        out
    }

    /// Parses an S-ADM sequence from a little-endian byte stream.
    ///
    /// # Errors
    ///
    /// Returns a [`SadmError`] on a truncated stream, a bad header, or an
    /// out-of-range frame-type tag.
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, SadmError> {
        let mut reader = Reader::new(bytes);
        if reader.take(4)? != MAGIC {
            return Err(SadmError::BadHeader);
        }
        if reader.u16()? != VERSION {
            return Err(SadmError::BadHeader);
        }
        let frame_rate = reader.sample()?;
        let frame_count = reader.u32()? as usize;
        let mut frames = Vec::with_capacity(frame_count);
        for _ in 0..frame_count {
            let frame_index = reader.u32()?;
            let frame_type = FrameType::from_tag(reader.u8()?).ok_or(SadmError::BadFrameType)?;
            let start = reader.sample()?;
            let duration = reader.sample()?;
            let update_count = reader.u32()? as usize;
            let mut frame = SadmFrame::new(SadmFrameFormat::new(
                frame_index,
                start,
                duration,
                frame_type,
            ));
            for _ in 0..update_count {
                let object_id = reader.u32()?;
                let azimuth = reader.sample()?;
                let elevation = reader.sample()?;
                let distance = reader.sample()?;
                let gain = reader.sample()?;
                frame.push(SadmObjectUpdate::new(
                    object_id,
                    AdmPosition::new(azimuth, elevation, distance),
                    gain,
                ));
            }
            frames.push(frame);
        }
        Ok(Self { frames, frame_rate })
    }

    /// Samples the continuous trajectory of `object_id` at `time` seconds by
    /// linear interpolation between the frames that carry that object.
    ///
    /// Returns [`None`] if no frame carries the object. Times before the first
    /// carrying frame clamp to that frame, and times after the last clamp to
    /// the last.
    #[must_use]
    pub fn sample_at(&self, object_id: u32, time: Sample) -> Option<SadmObjectUpdate> {
        let mut previous: Option<(Sample, &SadmObjectUpdate)> = None;
        for frame in &self.frames {
            let Some(update) = frame.update_for(object_id) else {
                continue;
            };
            let frame_time = frame.format.start;
            if time <= frame_time {
                return Some(match previous {
                    Some((prev_time, prev_update)) if frame_time > prev_time => {
                        let span = frame_time - prev_time;
                        let fraction = ((time - prev_time) / span).clamp(0.0, 1.0);
                        lerp_update(object_id, prev_update, update, fraction)
                    }
                    _ => *update,
                });
            }
            previous = Some((frame_time, update));
        }
        previous.map(|(_, update)| *update)
    }
}

/// Linearly interpolates two object updates by `fraction` in `[0, 1]`.
fn lerp_update(
    object_id: u32,
    start: &SadmObjectUpdate,
    end: &SadmObjectUpdate,
    fraction: Sample,
) -> SadmObjectUpdate {
    let azimuth = lerp(start.position.azimuth, end.position.azimuth, fraction);
    let elevation = lerp(start.position.elevation, end.position.elevation, fraction);
    let distance = lerp(start.position.distance, end.position.distance, fraction);
    let gain = lerp(start.gain, end.gain, fraction);
    SadmObjectUpdate::new(object_id, AdmPosition::new(azimuth, elevation, distance), gain)
}

/// Linear interpolation `a + fraction * (b - a)`.
fn lerp(a: Sample, b: Sample, fraction: Sample) -> Sample {
    a + fraction * (b - a)
}

/// A small forward-only little-endian byte reader.
struct Reader<'a> {
    bytes: &'a [u8],
    offset: usize,
}

impl<'a> Reader<'a> {
    fn new(bytes: &'a [u8]) -> Self {
        Self { bytes, offset: 0 }
    }

    fn take(&mut self, count: usize) -> Result<&'a [u8], SadmError> {
        if self.offset + count > self.bytes.len() {
            return Err(SadmError::Truncated);
        }
        let slice = &self.bytes[self.offset..self.offset + count];
        self.offset += count;
        Ok(slice)
    }

    fn u8(&mut self) -> Result<u8, SadmError> {
        Ok(self.take(1)?[0])
    }

    fn u16(&mut self) -> Result<u16, SadmError> {
        let bytes = self.take(2)?;
        Ok(u16::from_le_bytes([bytes[0], bytes[1]]))
    }

    fn u32(&mut self) -> Result<u32, SadmError> {
        let bytes = self.take(4)?;
        Ok(u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]))
    }

    fn sample(&mut self) -> Result<Sample, SadmError> {
        let bytes = self.take(4)?;
        Ok(Sample::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use bevy_math::ops;

    const EPS: Sample = 1e-5;

    fn close(a: Sample, b: Sample) -> bool {
        ops::abs(a - b) <= EPS
    }

    fn sequence() -> SadmSequence {
        let mut seq = SadmSequence::new(25.0);
        let mut frame0 = SadmFrame::new(SadmFrameFormat::new(0, 0.0, 1.0, FrameType::Full));
        frame0.push(SadmObjectUpdate::new(
            7,
            AdmPosition::new(0.0, 0.0, 1.0),
            1.0,
        ));
        let mut frame1 = SadmFrame::new(SadmFrameFormat::new(1, 1.0, 1.0, FrameType::Divided));
        frame1.push(SadmObjectUpdate::new(
            7,
            AdmPosition::new(90.0, 20.0, 0.5),
            0.0,
        ));
        seq.push(frame0);
        seq.push(frame1);
        seq
    }

    #[test]
    fn round_trip_is_byte_identical() {
        let seq = sequence();
        let bytes = seq.to_bytes();
        let parsed = SadmSequence::from_bytes(&bytes).expect("parse");
        assert_eq!(parsed, seq);
        assert_eq!(parsed.to_bytes(), bytes);
    }

    #[test]
    fn sample_midpoint_interpolates() {
        let seq = sequence();
        let mid = seq.sample_at(7, 0.5).expect("object present");
        assert!(close(mid.position.azimuth, 45.0));
        assert!(close(mid.position.elevation, 10.0));
        assert!(close(mid.position.distance, 0.75));
        assert!(close(mid.gain, 0.5));
    }

    #[test]
    fn sample_clamps_outside_range() {
        let seq = sequence();
        let before = seq.sample_at(7, -1.0).expect("present");
        assert!(close(before.position.azimuth, 0.0));
        let after = seq.sample_at(7, 10.0).expect("present");
        assert!(close(after.position.azimuth, 90.0));
    }

    #[test]
    fn missing_object_returns_none() {
        assert!(sequence().sample_at(999, 0.5).is_none());
    }

    #[test]
    fn bad_header_is_rejected() {
        assert_eq!(
            SadmSequence::from_bytes(b"XXXX"),
            Err(SadmError::BadHeader)
        );
        assert_eq!(SadmSequence::from_bytes(&[]), Err(SadmError::Truncated));
    }
}
