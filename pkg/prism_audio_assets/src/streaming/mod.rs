//! Section 20 streaming media: lock-free ring, byte sources, and prefetch.
//!
//! # Provenance
//! Original work; no Unreal Engine, Unity, Godot, Wwise, FMOD, Steam Audio, or
//! Google Resonance Audio source or derived code; no AI/ML.
//!
//! # Relationship
//! Implements the memory-vs-streaming split of design section 20. Long assets
//! stream through a [`ring::FrameRing`] fed by a background decode pump over a
//! [`stream_source::ByteSource`]; the first segment stays resident in a
//! [`prefetch::PrefetchStream`] for zero-latency start, and underruns fall
//! through to silence rather than blocking the real-time thread.

pub mod prefetch;
pub mod ring;
pub mod stream_source;

pub use prefetch::PrefetchStream;
pub use ring::{FrameRing, ReadResult};
pub use stream_source::{ByteSource, ByteSourceError, MemoryByteSource};

#[cfg(feature = "std")]
pub use stream_source::FileByteSource;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::codec::ima_adpcm::ImaAdpcmEncoder;
    use crate::codec::ms_adpcm::{self, MsAdpcmDecoder};
    use crate::codec::pcm::PcmDecoder;
    use crate::codec::metadata::PcmSampleFormat;
    use alloc::boxed::Box;
    use alloc::vec::Vec;

    const EPSILON: f32 = 1.0e-4;

    fn pcm_decoder(frames: usize) -> PcmDecoder {
        let mut data = Vec::new();
        for n in 0..frames {
            let s = (n as i16).wrapping_mul(37);
            data.extend_from_slice(&s.to_le_bytes());
        }
        PcmDecoder::new(data, PcmSampleFormat::S16Le, 1, 48_000).unwrap()
    }

    #[test]
    fn ring_round_trips_frames() {
        let mut ring = FrameRing::new(2, 4);
        let written = ring.write_interleaved(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        assert_eq!(written, 3);
        assert_eq!(ring.available_to_read(), 3);
        let mut out = [0.0f32; 4];
        let result = ring.read_interleaved(&mut out);
        assert_eq!(result.frames_from_data, 2);
        assert_eq!(result.frames_silenced, 0);
        assert_eq!(out, [1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn ring_wraps_around_capacity() {
        let mut ring = FrameRing::new(1, 3);
        assert_eq!(ring.write_interleaved(&[1.0, 2.0, 3.0]), 3);
        assert!(ring.is_full());
        // Overflow is rejected, not overwritten.
        assert_eq!(ring.write_interleaved(&[9.0]), 0);
        let mut out = [0.0f32; 2];
        ring.read_interleaved(&mut out);
        assert_eq!(out, [1.0, 2.0]);
        // Space freed, writer wraps past the end.
        assert_eq!(ring.write_interleaved(&[4.0, 5.0]), 2);
        let mut rest = [0.0f32; 3];
        let result = ring.read_interleaved(&mut rest);
        assert_eq!(result.frames_from_data, 3);
        assert_eq!(rest, [3.0, 4.0, 5.0]);
    }

    #[test]
    fn ring_underrun_outputs_silence_and_counts() {
        let mut ring = FrameRing::new(1, 4);
        ring.write_interleaved(&[1.0, 2.0]);
        let mut out = [7.0f32; 4];
        let result = ring.read_interleaved(&mut out);
        assert_eq!(result.frames_from_data, 2);
        assert_eq!(result.frames_silenced, 2);
        assert_eq!(out, [1.0, 2.0, 0.0, 0.0]);
        assert_eq!(ring.underrun_frames(), 2);
    }

    #[test]
    fn prefetch_starts_immediately_then_streams() {
        let decoder = Box::new(pcm_decoder(2000));
        let mut stream = PrefetchStream::new(decoder, 256, 512).unwrap();
        // Resident segment is available with no pumping.
        assert_eq!(stream.resident_segment().len(), 256);
        assert!(stream.buffered_frames() >= 256);
        let mut out = [0.0f32; 128];
        let result = stream.read(&mut out);
        assert_eq!(result.frames_silenced, 0);
        // First resident sample equals frame 0 of the source (0 * 37 = 0).
        assert!((out[0] - 0.0).abs() < EPSILON);
        // Pump the rest and drain fully.
        let mut total_read = 128usize;
        let mut guard = 0;
        while !stream.is_finished() && guard < 100_000 {
            stream.pump(1024).unwrap();
            let mut block = [0.0f32; 256];
            let r = stream.read(&mut block);
            total_read += r.frames_from_data;
            guard += 1;
        }
        assert_eq!(total_read, 2000);
    }

    #[test]
    fn prefetch_restart_reuses_resident() {
        let decoder = Box::new(pcm_decoder(1000));
        let mut stream = PrefetchStream::new(decoder, 100, 300).unwrap();
        let mut out = [0.0f32; 50];
        stream.read(&mut out);
        stream.restart().unwrap();
        assert!(stream.buffered_frames() >= 100);
        let mut again = [0.0f32; 1];
        stream.read(&mut again);
        assert!((again[0] - 0.0).abs() < EPSILON);
    }

    #[test]
    fn prefetch_streams_adpcm_asset() {
        // Encode a mono ramp with IMA ADPCM, then stream-decode it.
        let samples_per_block = 505;
        let encoder = ImaAdpcmEncoder::new(samples_per_block);
        let mut pcm = Vec::new();
        for n in 0..samples_per_block {
            pcm.push((bevy_math::ops::sin(n as f32 * 0.02) * 6000.0) as i16);
        }
        let block = encoder.encode_block(&pcm);
        let decoder = crate::codec::ima_adpcm::ImaAdpcmDecoder::new(
            block,
            1,
            48_000,
            encoder.block_align(),
            samples_per_block,
        )
        .unwrap();
        let mut stream = PrefetchStream::new(Box::new(decoder), 64, 256).unwrap();
        let mut collected = Vec::new();
        let mut guard = 0;
        while !stream.is_finished() && guard < 100_000 {
            stream.pump(512).unwrap();
            let mut block = [0.0f32; 128];
            let r = stream.read(&mut block);
            collected.extend_from_slice(&block[..r.frames_from_data]);
            guard += 1;
        }
        assert_eq!(collected.len(), samples_per_block);
    }

    #[test]
    fn memory_byte_source_reads_ranges() {
        let mut source = MemoryByteSource::new(alloc::vec![10, 20, 30, 40, 50]);
        let mut out = [0u8; 3];
        let read = source.read_at(2, &mut out).unwrap();
        assert_eq!(read, 3);
        assert_eq!(out, [30, 40, 50]);
        let all = source.read_to_vec().unwrap();
        assert_eq!(all, alloc::vec![10, 20, 30, 40, 50]);
        assert_eq!(source.read_at(99, &mut out), Err(ByteSourceError::OutOfRange));
    }

    #[test]
    fn ms_adpcm_streams_through_ring() {
        let samples_per_block = 500;
        let encoder = ms_adpcm::MsAdpcmEncoder::new(samples_per_block);
        let mut pcm = Vec::new();
        for n in 0..samples_per_block {
            pcm.push((bevy_math::ops::sin(n as f32 * 0.04) * 7000.0) as i16);
        }
        let block = encoder.encode_block(&pcm);
        let decoder = MsAdpcmDecoder::new(
            block,
            1,
            48_000,
            encoder.block_align(),
            samples_per_block,
            ms_adpcm::DEFAULT_COEFFICIENTS.to_vec(),
        )
        .unwrap();
        let mut stream = PrefetchStream::new(Box::new(decoder), 32, 128).unwrap();
        let mut frames = 0usize;
        let mut guard = 0;
        while !stream.is_finished() && guard < 100_000 {
            stream.pump(256).unwrap();
            let mut block = [0.0f32; 64];
            let r = stream.read(&mut block);
            frames += r.frames_from_data;
            guard += 1;
        }
        assert_eq!(frames, samples_per_block);
    }
}
