//! The BW64 (ITU-R BS.2088) 64-bit broadcast WAV container.
//!
//! BW64 is a RIFF/WAVE variant with a `ds64` chunk that carries 64-bit sizes so
//! files may exceed four gigabytes. This module reads and writes the container
//! at the byte level: the `BW64`/`WAVE` form, the `ds64` size table, the `fmt `
//! format block, the raw `data` payload, the `chna` track map, and the `axml`
//! ADM metadata. The writer is deterministic and a write-then-read-then-write
//! cycle is byte identical.
//!
//! # Provenance
//!
//! Original work; no Unreal Engine, Unity, Godot, Wwise, FMOD, Steam Audio,
//! Google Resonance Audio, Dolby, or MPEG source or derived code; no AI/ML.
//! Implements the publicly published BW64 container layout (ITU-R BS.2088) and
//! the RIFF/WAVE chunk conventions it builds on.
//!
//! # Relationship
//!
//! Holds the `axml` bytes produced by [`crate::adm::xml`] from the
//! [`crate::adm::model`] graph together with the [`crate::adm::chna`] track
//! map, so a complete ADM master is a single self-describing file.

use alloc::vec::Vec;

use crate::adm::chna::{ChnaChunk, ChnaError};

/// The sentinel written into 32-bit size fields that are superseded by `ds64`.
const SENTINEL_32: u32 = 0xFFFF_FFFF;
/// The fixed body length of a `ds64` chunk (three `u64` plus a `u32`).
const DS64_BODY_LEN: usize = 8 + 8 + 8 + 4;
/// The fixed body length of a canonical PCM `fmt ` chunk.
const FMT_BODY_LEN: usize = 16;

/// A failure while parsing a BW64 container.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Bw64Error {
    /// The byte stream was shorter than a required chunk needs.
    Truncated,
    /// A magic word (`BW64`, `WAVE`) or required chunk was missing or wrong.
    BadMagic,
    /// A required chunk (`ds64` or `fmt `) was absent.
    MissingChunk,
    /// The embedded `chna` chunk failed to parse.
    Chna(ChnaError),
}

impl From<ChnaError> for Bw64Error {
    fn from(error: ChnaError) -> Self {
        Bw64Error::Chna(error)
    }
}

/// A canonical PCM `fmt ` block.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub struct WaveFormat {
    /// The WAVE format tag (`1` for integer PCM, `3` for float PCM).
    pub format_tag: u16,
    /// Channel count.
    pub channels: u16,
    /// Sample rate in hertz.
    pub sample_rate: u32,
    /// Bits per sample.
    pub bits_per_sample: u16,
}

impl WaveFormat {
    /// Builds a format block.
    #[must_use]
    pub fn new(format_tag: u16, channels: u16, sample_rate: u32, bits_per_sample: u16) -> Self {
        Self {
            format_tag,
            channels,
            sample_rate,
            bits_per_sample,
        }
    }

    /// The frame size in bytes (`channels * bits_per_sample / 8`).
    #[must_use]
    pub fn block_align(self) -> u32 {
        u32::from(self.channels) * (u32::from(self.bits_per_sample) / 8)
    }

    /// The byte rate (`sample_rate * block_align`).
    #[must_use]
    pub fn byte_rate(self) -> u32 {
        self.sample_rate.saturating_mul(self.block_align())
    }
}

impl Default for WaveFormat {
    /// 48 kHz, 24-bit, stereo integer PCM.
    #[inline]
    fn default() -> Self {
        Self {
            format_tag: 1,
            channels: 2,
            sample_rate: 48_000,
            bits_per_sample: 24,
        }
    }
}

/// A complete BW64 file: format, PCM payload, track map, and ADM metadata.
#[derive(Clone, Debug, Default, PartialEq)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub struct Bw64File {
    /// The `fmt ` block.
    pub format: WaveFormat,
    /// The raw interleaved PCM payload (`data` chunk contents).
    pub data: Vec<u8>,
    /// The `chna` track map.
    pub chna: ChnaChunk,
    /// The `axml` ADM metadata payload.
    pub axml: Vec<u8>,
}

impl Bw64File {
    /// Builds a container from its parts.
    #[must_use]
    pub fn new(format: WaveFormat, data: Vec<u8>, chna: ChnaChunk, axml: Vec<u8>) -> Self {
        Self {
            format,
            data,
            chna,
            axml,
        }
    }

    /// The number of PCM frames implied by the payload and format.
    #[must_use]
    pub fn frame_count(&self) -> u64 {
        let align = u64::from(self.format.block_align());
        (self.data.len() as u64).checked_div(align).unwrap_or(0)
    }

    /// Serialises the container to deterministic BW64 bytes.
    #[must_use]
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut body = Vec::new();
        body.extend_from_slice(b"WAVE");

        let data_size = self.data.len() as u64;
        let sample_count = self.frame_count();

        // ds64 chunk; riff_size is patched in once the body length is known.
        let ds64_offset = body.len();
        push_chunk_header(&mut body, b"ds64", DS64_BODY_LEN as u32);
        body.extend_from_slice(&0u64.to_le_bytes());
        body.extend_from_slice(&data_size.to_le_bytes());
        body.extend_from_slice(&sample_count.to_le_bytes());
        body.extend_from_slice(&0u32.to_le_bytes());

        // fmt chunk.
        push_chunk_header(&mut body, b"fmt ", FMT_BODY_LEN as u32);
        body.extend_from_slice(&self.format.format_tag.to_le_bytes());
        body.extend_from_slice(&self.format.channels.to_le_bytes());
        body.extend_from_slice(&self.format.sample_rate.to_le_bytes());
        body.extend_from_slice(&self.format.byte_rate().to_le_bytes());
        let block_align = u16::try_from(self.format.block_align()).unwrap_or(u16::MAX);
        body.extend_from_slice(&block_align.to_le_bytes());
        body.extend_from_slice(&self.format.bits_per_sample.to_le_bytes());

        // data chunk with a 32-bit sentinel size (the real size is in ds64).
        push_chunk_header(&mut body, b"data", SENTINEL_32);
        body.extend_from_slice(&self.data);
        pad_to_even(&mut body);

        // chna chunk.
        let chna_bytes = self.chna.to_bytes();
        push_chunk(&mut body, b"chna", &chna_bytes);

        // axml chunk.
        push_chunk(&mut body, b"axml", &self.axml);

        // Patch the ds64 riff_size field (total bytes following the leading 8).
        let riff_size = body.len() as u64;
        let patch = ds64_offset + 8;
        body[patch..patch + 8].copy_from_slice(&riff_size.to_le_bytes());

        let mut out = Vec::with_capacity(8 + body.len());
        out.extend_from_slice(b"BW64");
        out.extend_from_slice(&SENTINEL_32.to_le_bytes());
        out.extend_from_slice(&body);
        out
    }

    /// Parses a container from BW64 bytes.
    ///
    /// # Errors
    ///
    /// Returns a [`Bw64Error`] on a truncated stream, a bad magic word, a
    /// missing required chunk, or an invalid embedded `chna` chunk.
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, Bw64Error> {
        if bytes.len() < 12 {
            return Err(Bw64Error::Truncated);
        }
        if &bytes[0..4] != b"BW64" {
            return Err(Bw64Error::BadMagic);
        }
        if &bytes[8..12] != b"WAVE" {
            return Err(Bw64Error::BadMagic);
        }

        let mut cursor = 12usize;
        let mut data_size_64: Option<u64> = None;
        let mut format: Option<WaveFormat> = None;
        let mut data = Vec::new();
        let mut chna = ChnaChunk::new();
        let mut axml = Vec::new();

        while cursor + 8 <= bytes.len() {
            let id = [
                bytes[cursor],
                bytes[cursor + 1],
                bytes[cursor + 2],
                bytes[cursor + 3],
            ];
            let declared =
                u32::from_le_bytes([bytes[cursor + 4], bytes[cursor + 5], bytes[cursor + 6], bytes[cursor + 7]]);
            cursor += 8;

            let size = if &id == b"data" && declared == SENTINEL_32 {
                usize::try_from(data_size_64.ok_or(Bw64Error::MissingChunk)?)
                    .map_err(|_| Bw64Error::Truncated)?
            } else {
                declared as usize
            };
            if cursor + size > bytes.len() {
                return Err(Bw64Error::Truncated);
            }
            let payload = &bytes[cursor..cursor + size];

            match &id {
                b"ds64" => {
                    if payload.len() < DS64_BODY_LEN {
                        return Err(Bw64Error::Truncated);
                    }
                    let mut raw = [0u8; 8];
                    raw.copy_from_slice(&payload[8..16]);
                    data_size_64 = Some(u64::from_le_bytes(raw));
                }
                b"fmt " => {
                    if payload.len() < FMT_BODY_LEN {
                        return Err(Bw64Error::Truncated);
                    }
                    format = Some(WaveFormat {
                        format_tag: u16::from_le_bytes([payload[0], payload[1]]),
                        channels: u16::from_le_bytes([payload[2], payload[3]]),
                        sample_rate: u32::from_le_bytes([
                            payload[4], payload[5], payload[6], payload[7],
                        ]),
                        bits_per_sample: u16::from_le_bytes([payload[14], payload[15]]),
                    });
                }
                b"data" => data = payload.to_vec(),
                b"chna" => chna = ChnaChunk::from_bytes(payload)?,
                b"axml" => axml = payload.to_vec(),
                _ => {}
            }

            cursor += size;
            if size % 2 == 1 {
                cursor += 1;
            }
        }

        Ok(Self {
            format: format.ok_or(Bw64Error::MissingChunk)?,
            data,
            chna,
            axml,
        })
    }
}

/// Pushes a chunk header (`id` plus a little-endian `u32` size) into `out`.
fn push_chunk_header(out: &mut Vec<u8>, id: &[u8; 4], size: u32) {
    out.extend_from_slice(id);
    out.extend_from_slice(&size.to_le_bytes());
}

/// Pushes a full chunk (`id`, size, payload, even-pad) into `out`.
fn push_chunk(out: &mut Vec<u8>, id: &[u8; 4], payload: &[u8]) {
    let size = u32::try_from(payload.len()).unwrap_or(SENTINEL_32);
    push_chunk_header(out, id, size);
    out.extend_from_slice(payload);
    pad_to_even(out);
}

/// Appends a single zero byte if `out` has an odd length.
fn pad_to_even(out: &mut Vec<u8>) {
    if out.len() % 2 == 1 {
        out.push(0);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::adm::chna::AudioId;

    fn sample_file() -> Bw64File {
        let format = WaveFormat::new(1, 2, 48_000, 24);
        let data = (0..18u8).collect::<Vec<u8>>();
        let mut chna = ChnaChunk::new();
        chna.push(AudioId::new(1, "ATU_00000001", "AC_00031001", "AP_00031001"));
        let axml = b"<audioFormatExtended/>".to_vec();
        Bw64File::new(format, data, chna, axml)
    }

    #[test]
    fn round_trip_is_byte_identical() {
        let file = sample_file();
        let bytes = file.to_bytes();
        let parsed = Bw64File::from_bytes(&bytes).expect("parse");
        assert_eq!(parsed, file);
        assert_eq!(parsed.to_bytes(), bytes);
    }

    #[test]
    fn header_uses_bw64_and_wave() {
        let bytes = sample_file().to_bytes();
        assert_eq!(&bytes[0..4], b"BW64");
        assert_eq!(&bytes[8..12], b"WAVE");
    }

    #[test]
    fn frame_count_matches_payload() {
        let file = sample_file();
        // 18 bytes / (2 channels * 3 bytes) = 3 frames.
        assert_eq!(file.frame_count(), 3);
    }

    #[test]
    fn bad_magic_is_rejected() {
        let mut bytes = sample_file().to_bytes();
        bytes[0] = b'R';
        assert_eq!(Bw64File::from_bytes(&bytes), Err(Bw64Error::BadMagic));
    }

    #[test]
    fn truncated_input_is_rejected() {
        assert_eq!(Bw64File::from_bytes(&[0u8; 4]), Err(Bw64Error::Truncated));
    }
}
