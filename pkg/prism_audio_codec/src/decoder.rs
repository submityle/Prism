//! The [`SymphoniaDecoder`]: a [`SourceDecoder`] backed by Symphonia.
//!
//! # Provenance
//! Original integration work; no Unreal Engine, Unity, Godot, Wwise, FMOD,
//! Steam Audio, or Google Resonance Audio source or derived code; no AI/ML.
//!
//! # Relationship
//! Adapts Symphonia's demux + decode pipeline to the engine's incremental
//! [`SourceDecoder`] contract (design sections 20 and 44.1). The decoder owns a
//! format reader and a codec decoder, pulls packets on demand, converts each
//! decoded block to interleaved `f32`, and hands whole frames to the caller a
//! bounded slice at a time so a background task can pump decode work without
//! ever allocating an unbounded scratch buffer per call.
//!
//! [`SourceDecoder`]: prism_audio_assets::codec::decoder::SourceDecoder

use std::io::Cursor;

use prism_audio_assets::codec::decoder::{DecodeError, SourceDecoder};
use prism_audio_assets::codec::metadata::{AudioStreamInfo, CodecTag, CustomCodecId};
use prism_audio_core::math::Sample;
use symphonia::core::audio::{Channels, SampleBuffer};
use symphonia::core::codecs::{
    CodecType, Decoder, DecoderOptions, CODEC_TYPE_AAC, CODEC_TYPE_FLAC, CODEC_TYPE_MP3,
    CODEC_TYPE_VORBIS,
};
use symphonia::core::errors::Error as SymphoniaError;
use symphonia::core::formats::{FormatOptions, FormatReader, SeekMode, SeekTo};
use symphonia::core::io::MediaSourceStream;
use symphonia::core::meta::MetadataOptions;
use symphonia::core::probe::Hint;

/// A [`SourceDecoder`] that decodes compressed audio with Symphonia.
///
/// Construct one with [`SymphoniaDecoder::new`] from a complete in-memory
/// encoded asset. The decoder is `Send` so it can be moved onto a background
/// decode task, but it is **not** real-time safe: `decode` and `seek` perform
/// bounded heap allocation inside Symphonia and must never be called from the
/// audio callback (section 44.1).
///
/// [`SourceDecoder`]: prism_audio_assets::codec::decoder::SourceDecoder
pub struct SymphoniaDecoder {
    /// The container demuxer that yields encoded packets.
    format: Box<dyn FormatReader>,
    /// The per-track codec decoder that turns packets into audio buffers.
    decoder: Box<dyn Decoder>,
    /// Identifier of the track being decoded; foreign-track packets are skipped.
    track_id: u32,
    /// Immutable stream description reported through [`SourceDecoder::info`].
    info: AudioStreamInfo,
    /// Interleaved channel count, cached from `info` for hot-path arithmetic.
    channels: usize,
    /// Decoded-but-undelivered interleaved samples (always whole frames).
    pending: Vec<Sample>,
    /// Read cursor into `pending`, in samples.
    pending_pos: usize,
    /// Index of the next frame [`SourceDecoder::decode`] will emit.
    position: u64,
    /// Set once the demuxer reports end of stream; combined with an empty
    /// `pending` it means the decoder is exhausted.
    eof: bool,
    /// Channel count observed in the most recently decoded buffer, used to
    /// learn the signal spec of codecs that omit it from the container header.
    observed_channels: usize,
    /// Sample rate observed in the most recently decoded buffer.
    observed_rate: u32,
}

impl SymphoniaDecoder {
    /// Builds a decoder from a complete encoded byte stream.
    ///
    /// The bytes are copied into an owned, seekable in-memory source so the
    /// decoder can seek freely. Returns [`DecodeError::MalformedHeader`] when no
    /// decodable default track is found and [`DecodeError::UnsupportedFormat`]
    /// when the container probes but its codec has no registered decoder.
    pub fn new(bytes: &[u8]) -> Result<Self, DecodeError> {
        let source = Cursor::new(bytes.to_vec());
        let mss = MediaSourceStream::new(Box::new(source), Default::default());

        let hint = Hint::new();
        let format_opts = FormatOptions {
            enable_gapless: true,
            ..FormatOptions::default()
        };
        let metadata_opts = MetadataOptions::default();

        let probed = symphonia::default::get_probe()
            .format(&hint, mss, &format_opts, &metadata_opts)
            .map_err(map_error)?;
        let format = probed.format;

        let track = format.default_track().ok_or(DecodeError::MalformedHeader)?;
        let track_id = track.id;
        let codec_params = track.codec_params.clone();

        let decoder = symphonia::default::get_codecs()
            .make(&codec_params, &DecoderOptions::default())
            .map_err(map_error)?;

        let codec_tag = map_codec_tag(codec_params.codec);
        let frame_count = codec_params.n_frames;
        let declared_channels = codec_params.channels.map(Channels::count);
        let declared_rate = codec_params.sample_rate;

        let mut this = Self {
            format,
            decoder,
            track_id,
            info: AudioStreamInfo::new(0, 0, frame_count, codec_tag.clone()),
            channels: 0,
            pending: Vec::new(),
            pending_pos: 0,
            position: 0,
            eof: false,
            observed_channels: 0,
            observed_rate: 0,
        };

        // Some codecs (notably MP3/AAC) do not report channel count or sample
        // rate in the container header; they are only known once the first
        // frame is decoded. Prime one packet so `info` is always accurate and
        // never advertises a zero channel count.
        let (channels, rate) = match (declared_channels, declared_rate) {
            (Some(channels), Some(rate)) if channels > 0 => (channels, rate),
            _ => this.prime_first_packet()?,
        };

        let channels_u16 = u16::try_from(channels).map_err(|_| DecodeError::MalformedHeader)?;
        this.channels = channels;
        this.info = AudioStreamInfo::new(channels_u16, rate, frame_count, codec_tag);
        Ok(this)
    }

    /// Decodes packets until the first non-empty audio buffer is produced and
    /// returns its `(channels, sample_rate)`, buffering the decoded samples.
    ///
    /// Used only during construction to learn the signal spec of codecs that
    /// omit it from the container header. On an immediately-empty stream it
    /// returns a mono, zero-rate fallback so construction still succeeds and
    /// [`SourceDecoder::is_exhausted`] reports `true`.
    ///
    /// [`SourceDecoder::is_exhausted`]: prism_audio_assets::codec::decoder::SourceDecoder::is_exhausted
    fn prime_first_packet(&mut self) -> Result<(usize, u32), DecodeError> {
        self.fill_pending()?;
        if self.pending.is_empty() {
            // Empty or header-only stream: honest zero-length description.
            return Ok((1, 0));
        }
        Ok((self.observed_channels, self.observed_rate))
    }

    /// Pulls and decodes packets until `pending` holds at least one frame or
    /// the stream ends. Clears any already-consumed `pending` prefix first.
    fn fill_pending(&mut self) -> Result<(), DecodeError> {
        self.pending.clear();
        self.pending_pos = 0;
        loop {
            let packet = match self.format.next_packet() {
                Ok(packet) => packet,
                Err(SymphoniaError::IoError(ref err))
                    if err.kind() == std::io::ErrorKind::UnexpectedEof =>
                {
                    self.eof = true;
                    return Ok(());
                }
                Err(SymphoniaError::ResetRequired) => {
                    // A track-list change: reset the decoder and keep pulling.
                    self.decoder.reset();
                    continue;
                }
                Err(err) => return Err(map_error(err)),
            };

            if packet.track_id() != self.track_id {
                continue;
            }

            match self.decoder.decode(&packet) {
                Ok(audio_buf) => {
                    let spec = *audio_buf.spec();
                    let frame_capacity = audio_buf.capacity() as u64;
                    let mut sample_buf = SampleBuffer::<Sample>::new(frame_capacity, spec);
                    sample_buf.copy_interleaved_ref(audio_buf);
                    let samples = sample_buf.samples();
                    if !samples.is_empty() {
                        self.observed_channels = spec.channels.count();
                        self.observed_rate = spec.rate;
                        self.pending.extend_from_slice(samples);
                        return Ok(());
                    }
                    // Zero-length buffer (e.g. priming frame); keep going.
                }
                // Decode-level errors are recoverable per Symphonia's contract:
                // skip the damaged packet and continue with the next one.
                Err(SymphoniaError::DecodeError(_)) => continue,
                Err(SymphoniaError::ResetRequired) => {
                    self.decoder.reset();
                    continue;
                }
                Err(err) => return Err(map_error(err)),
            }
        }
    }
}

impl SourceDecoder for SymphoniaDecoder {
    fn info(&self) -> AudioStreamInfo {
        self.info.clone()
    }

    fn decode(&mut self, out: &mut [Sample]) -> Result<usize, DecodeError> {
        let channels = self.channels;
        if channels == 0 || out.len() < channels {
            return Err(DecodeError::OutputTooSmall);
        }
        let max_frames = out.len() / channels;
        let mut written_frames = 0usize;

        while written_frames < max_frames {
            if self.pending_pos >= self.pending.len() {
                if self.eof {
                    break;
                }
                self.fill_pending()?;
                if self.pending_pos >= self.pending.len() {
                    // Either EOF was reached or a packet produced no frames.
                    if self.eof {
                        break;
                    }
                    continue;
                }
            }

            let available_samples = self.pending.len() - self.pending_pos;
            let available_frames = available_samples / channels;
            let needed_frames = max_frames - written_frames;
            let take_frames = needed_frames.min(available_frames);
            if take_frames == 0 {
                break;
            }

            let src_start = self.pending_pos;
            let src_end = src_start + take_frames * channels;
            let dst_start = written_frames * channels;
            let dst_end = dst_start + take_frames * channels;
            out[dst_start..dst_end].copy_from_slice(&self.pending[src_start..src_end]);

            self.pending_pos = src_end;
            written_frames += take_frames;
        }

        self.position += written_frames as u64;
        Ok(written_frames)
    }

    fn seek(&mut self, frame: u64) -> Result<(), DecodeError> {
        if let Some(frame_count) = self.info.frame_count {
            if frame > frame_count {
                return Err(DecodeError::SeekOutOfRange);
            }
            if frame == frame_count {
                // Seeking to the end leaves the decoder exhausted (trait rule).
                self.pending.clear();
                self.pending_pos = 0;
                self.position = frame;
                self.eof = true;
                return Ok(());
            }
        }

        let seeked = self
            .format
            .seek(
                SeekMode::Accurate,
                SeekTo::TimeStamp {
                    ts: frame,
                    track_id: self.track_id,
                },
            )
            .map_err(map_error)?;

        self.decoder.reset();
        self.pending.clear();
        self.pending_pos = 0;
        self.position = seeked.actual_ts;
        self.eof = false;
        Ok(())
    }

    fn position(&self) -> u64 {
        self.position
    }

    fn is_exhausted(&self) -> bool {
        self.eof && self.pending_pos >= self.pending.len()
    }
}

/// Maps a Symphonia [`CodecType`] to the engine's [`CodecTag`].
///
/// Lossless FLAC and Ogg Vorbis map to their dedicated tags; MP3 and AAC have
/// no first-class tag in the engine's enum and are reported through
/// [`CodecTag::Custom`] with a stable lowercase label so banks can still
/// describe and route them.
fn map_codec_tag(codec: CodecType) -> CodecTag {
    if codec == CODEC_TYPE_FLAC {
        CodecTag::Flac
    } else if codec == CODEC_TYPE_VORBIS {
        CodecTag::Vorbis
    } else if codec == CODEC_TYPE_MP3 {
        CodecTag::Custom(CustomCodecId("mp3".to_string()))
    } else if codec == CODEC_TYPE_AAC {
        CodecTag::Custom(CustomCodecId("aac".to_string()))
    } else {
        CodecTag::Custom(CustomCodecId(format!("symphonia-{codec}")))
    }
}

/// Maps a Symphonia error onto the engine's codec-neutral [`DecodeError`].
fn map_error(err: SymphoniaError) -> DecodeError {
    match err {
        SymphoniaError::IoError(ref inner) if inner.kind() == std::io::ErrorKind::UnexpectedEof => {
            DecodeError::UnexpectedEof
        }
        SymphoniaError::IoError(_) => DecodeError::UnexpectedEof,
        SymphoniaError::DecodeError(_)
        | SymphoniaError::LimitError(_)
        | SymphoniaError::ResetRequired => DecodeError::MalformedHeader,
        SymphoniaError::Unsupported(_) => DecodeError::UnsupportedFormat,
        SymphoniaError::SeekError(_) => DecodeError::SeekOutOfRange,
    }
}

#[cfg(test)]
mod tests {
    use super::{map_codec_tag, map_error};
    use prism_audio_assets::codec::decoder::DecodeError;
    use prism_audio_assets::codec::metadata::CodecTag;
    use symphonia::core::codecs::{
        CODEC_TYPE_AAC, CODEC_TYPE_FLAC, CODEC_TYPE_MP3, CODEC_TYPE_NULL, CODEC_TYPE_VORBIS,
    };
    use symphonia::core::errors::{Error as SymphoniaError, SeekErrorKind};

    /// Each Symphonia codec identifier must map onto the engine's codec family,
    /// with MP3/AAC surfaced as stable custom labels and unknown codecs tagged
    /// with their numeric identifier rather than silently dropped.
    #[test]
    fn codec_tags_map_to_engine_families() {
        assert!(matches!(map_codec_tag(CODEC_TYPE_FLAC), CodecTag::Flac));
        assert!(matches!(map_codec_tag(CODEC_TYPE_VORBIS), CodecTag::Vorbis));
        match map_codec_tag(CODEC_TYPE_MP3) {
            CodecTag::Custom(id) => assert_eq!(id.0, "mp3"),
            other => panic!("mp3 should map to a custom tag, got {other:?}"),
        }
        match map_codec_tag(CODEC_TYPE_AAC) {
            CodecTag::Custom(id) => assert_eq!(id.0, "aac"),
            other => panic!("aac should map to a custom tag, got {other:?}"),
        }
        match map_codec_tag(CODEC_TYPE_NULL) {
            CodecTag::Custom(id) => assert!(id.0.starts_with("symphonia-0x")),
            other => panic!("unknown codec should map to a labelled custom tag, got {other:?}"),
        }
    }

    /// Every Symphonia error variant must collapse onto a codec-neutral
    /// [`DecodeError`], with unexpected end-of-stream distinguished from other
    /// I/O faults.
    #[test]
    fn errors_map_to_codec_neutral_variants() {
        let eof = SymphoniaError::IoError(std::io::Error::from(std::io::ErrorKind::UnexpectedEof));
        assert_eq!(map_error(eof), DecodeError::UnexpectedEof);
        let other_io = SymphoniaError::IoError(std::io::Error::from(std::io::ErrorKind::Other));
        assert_eq!(map_error(other_io), DecodeError::UnexpectedEof);
        assert_eq!(
            map_error(SymphoniaError::DecodeError("x")),
            DecodeError::MalformedHeader
        );
        assert_eq!(
            map_error(SymphoniaError::LimitError("x")),
            DecodeError::MalformedHeader
        );
        assert_eq!(
            map_error(SymphoniaError::ResetRequired),
            DecodeError::MalformedHeader
        );
        assert_eq!(
            map_error(SymphoniaError::Unsupported("x")),
            DecodeError::UnsupportedFormat
        );
        assert_eq!(
            map_error(SymphoniaError::SeekError(SeekErrorKind::OutOfRange)),
            DecodeError::SeekOutOfRange
        );
    }
}
