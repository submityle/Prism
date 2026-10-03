//! Section 20 bank model: manifests, media entries, load/unload, and a
//! dependency-aware registry.
//!
//! # Provenance
//! Original work; no Unreal Engine, Unity, Godot, Wwise, FMOD, Steam Audio, or
//! Google Resonance Audio source or derived code; no AI/ML.
//!
//! # Relationship
//! Implements the loadable-unit model of design section 20. A [`BankManifest`]
//! groups events, containers, patches, and [`MediaEntry`] records into a single
//! unit; a [`LoadedBank`] pairs that manifest with decoded resident PCM (short
//! effects) and retained encoded bytes (streamed assets); a [`BankRegistry`]
//! tracks residency, enforces dependency ordering, and reports memory usage.

pub mod entry;
pub mod handle;
pub mod manifest;
pub mod registry;

pub use entry::{EntryId, MediaEntry, MediaFormat, Residency};
pub use handle::{BankLoadError, BankMemoryUsage, LoadedBank, LoadedMedia};
pub use manifest::{
    BankId, BankManifest, ContainerRef, EventRef, ManifestError, PatchRef,
};
pub use registry::{BankRegistry, BankRegistryError};

#[cfg(test)]
mod tests {
    use super::*;
    use crate::codec::metadata::CodecTag;
    use crate::codec::registry::DecoderRegistry;
    use alloc::string::String;
    use alloc::vec::Vec;

    const EPSILON: f32 = 1.0e-4;

    /// Builds a minimal canonical PCM WAV file in memory.
    fn build_wav(
        channels: u16,
        sample_rate: u32,
        bits_per_sample: u16,
        data: &[u8],
    ) -> Vec<u8> {
        let block_align = channels * (bits_per_sample / 8);
        let byte_rate = sample_rate * u32::from(block_align);
        let mut out = Vec::new();
        out.extend_from_slice(b"RIFF");
        out.extend_from_slice(&(36 + data.len() as u32).to_le_bytes());
        out.extend_from_slice(b"WAVE");
        out.extend_from_slice(b"fmt ");
        out.extend_from_slice(&16u32.to_le_bytes());
        out.extend_from_slice(&0x0001u16.to_le_bytes());
        out.extend_from_slice(&channels.to_le_bytes());
        out.extend_from_slice(&sample_rate.to_le_bytes());
        out.extend_from_slice(&byte_rate.to_le_bytes());
        out.extend_from_slice(&block_align.to_le_bytes());
        out.extend_from_slice(&bits_per_sample.to_le_bytes());
        out.extend_from_slice(b"data");
        out.extend_from_slice(&(data.len() as u32).to_le_bytes());
        out.extend_from_slice(data);
        out
    }

    /// Builds a mono 16-bit WAV holding `frames` ascending samples.
    fn mono_wav(frames: usize) -> Vec<u8> {
        let mut data = Vec::new();
        for n in 0..frames {
            let value = ((n as i32 % 2000) - 1000) as i16;
            data.extend_from_slice(&value.to_le_bytes());
        }
        build_wav(1, 48_000, 16, &data)
    }

    #[test]
    fn load_resolve_and_unload_single_bank() {
        let wav = mono_wav(16);
        let frame_count = Some(16);
        let entry = MediaEntry::new(
            EntryId(1),
            String::from("blip"),
            MediaFormat::new(CodecTag::Pcm, 1, 48_000, frame_count),
            Residency::Memory,
            0,
            wav.len() as u64,
        );
        let manifest = BankManifest::new(BankId(10), String::from("sfx"), 1).with_media(entry);
        let registry = DecoderRegistry::with_native();

        let mut banks = BankRegistry::new();
        banks.load(manifest, &wav, &registry).unwrap();
        assert!(banks.is_loaded(BankId(10)));

        let media = banks.media(BankId(10), EntryId(1)).unwrap();
        let pcm = media.resident_pcm().unwrap();
        assert_eq!(pcm.len(), 16);
        // Frame 0 is -1000 / 32768.
        assert!((pcm[0] - (-1000.0 / 32768.0)).abs() < EPSILON);

        // Name-based resolution hits the same payload.
        let by_name = banks.media_by_name(BankId(10), "blip").unwrap();
        assert_eq!(by_name.resident_pcm().unwrap().len(), 16);

        banks.unload(BankId(10)).unwrap();
        assert!(!banks.is_loaded(BankId(10)));
        assert!(banks.is_empty());
    }

    #[test]
    fn dependency_must_be_loaded_first_and_blocks_unload() {
        let wav = mono_wav(8);
        let base = BankManifest::new(BankId(1), String::from("base"), 1).with_media(
            MediaEntry::new(
                EntryId(1),
                String::from("tone"),
                MediaFormat::new(CodecTag::Pcm, 1, 48_000, Some(8)),
                Residency::Memory,
                0,
                wav.len() as u64,
            ),
        );
        let dependent = BankManifest::new(BankId(2), String::from("level"), 1)
            .with_dependency(BankId(1));
        let registry = DecoderRegistry::with_native();
        let mut banks = BankRegistry::new();

        // Loading the dependent before its dependency fails.
        let err = banks
            .load(dependent.clone(), &[], &registry)
            .unwrap_err();
        assert_eq!(
            err,
            BankRegistryError::MissingDependency {
                bank: BankId(2),
                dependency: BankId(1),
            }
        );

        banks.load(base, &wav, &registry).unwrap();
        banks.load(dependent, &[], &registry).unwrap();

        // The base bank cannot be unloaded while the dependent is resident.
        let err = banks.unload(BankId(1)).unwrap_err();
        assert_eq!(
            err,
            BankRegistryError::StillDepended {
                bank: BankId(1),
                dependent: BankId(2),
            }
        );

        // Unload in dependency order succeeds.
        banks.unload(BankId(2)).unwrap();
        banks.unload(BankId(1)).unwrap();
        assert!(banks.is_empty());
    }

    #[test]
    fn streaming_entry_retains_bytes_and_opens_stream() {
        let frames = 1200;
        let wav = mono_wav(frames);
        let entry = MediaEntry::new(
            EntryId(7),
            String::from("music"),
            MediaFormat::new(CodecTag::Pcm, 1, 48_000, Some(frames as u64)),
            Residency::Streaming {
                prefetch_frames: 128,
            },
            0,
            wav.len() as u64,
        );
        let manifest =
            BankManifest::new(BankId(20), String::from("music_bank"), 1).with_media(entry);
        let registry = DecoderRegistry::with_native();
        let bank = LoadedBank::load(manifest, &wav, &registry).unwrap();

        let media = bank.media_by_id(EntryId(7)).unwrap();
        assert!(media.is_streaming());
        assert!(media.resident_pcm().is_none());

        let mut stream = media.open_stream(&registry, 512).unwrap().unwrap();
        assert!(stream.buffered_frames() >= 128);
        let mut total = 0usize;
        let mut guard = 0;
        while !stream.is_finished() && guard < 100_000 {
            stream.pump(512).unwrap();
            let mut block = [0.0f32; 256];
            let r = stream.read(&mut block);
            total += r.frames_from_data;
            guard += 1;
        }
        assert_eq!(total, frames);
    }

    #[test]
    fn memory_accounting_tracks_resident_and_encoded() {
        let short = mono_wav(10);
        let long = mono_wav(500);
        let manifest = BankManifest::new(BankId(30), String::from("mixed"), 2)
            .with_media(MediaEntry::new(
                EntryId(1),
                String::from("short"),
                MediaFormat::new(CodecTag::Pcm, 1, 48_000, Some(10)),
                Residency::Memory,
                0,
                short.len() as u64,
            ))
            .with_media(MediaEntry::new(
                EntryId(2),
                String::from("long"),
                MediaFormat::new(CodecTag::Pcm, 1, 48_000, Some(500)),
                Residency::Streaming {
                    prefetch_frames: 64,
                },
                short.len() as u64,
                long.len() as u64,
            ));
        let mut blob = Vec::new();
        blob.extend_from_slice(&short);
        blob.extend_from_slice(&long);
        let registry = DecoderRegistry::with_native();
        let bank = LoadedBank::load(manifest, &blob, &registry).unwrap();

        let usage = bank.memory_usage();
        // Resident PCM: 10 f32 samples.
        assert_eq!(usage.resident_pcm_bytes, 10 * 4);
        // Encoded bytes: the full long WAV retained for streaming.
        assert_eq!(usage.encoded_bytes, long.len() as u64);
        assert!(usage.metadata_bytes > 0);
        assert_eq!(
            usage.total(),
            usage.resident_pcm_bytes + usage.encoded_bytes + usage.metadata_bytes
        );
    }

    #[test]
    fn duplicate_media_id_is_rejected() {
        let wav = mono_wav(4);
        let manifest = BankManifest::new(BankId(40), String::from("dup"), 1)
            .with_media(MediaEntry::new(
                EntryId(1),
                String::from("a"),
                MediaFormat::new(CodecTag::Pcm, 1, 48_000, Some(4)),
                Residency::Memory,
                0,
                wav.len() as u64,
            ))
            .with_media(MediaEntry::new(
                EntryId(1),
                String::from("b"),
                MediaFormat::new(CodecTag::Pcm, 1, 48_000, Some(4)),
                Residency::Memory,
                0,
                wav.len() as u64,
            ));
        let registry = DecoderRegistry::with_native();
        let mut banks = BankRegistry::new();
        let err = banks.load(manifest, &wav, &registry).unwrap_err();
        assert_eq!(
            err,
            BankRegistryError::Load(BankLoadError::Manifest(
                ManifestError::DuplicateMediaId(EntryId(1))
            ))
        );
    }

    #[test]
    fn declared_byte_totals_match_entries() {
        let manifest = BankManifest::new(BankId(50), String::from("totals"), 1)
            .with_media(MediaEntry::new(
                EntryId(1),
                String::from("a"),
                MediaFormat::new(CodecTag::Pcm, 2, 48_000, Some(100)),
                Residency::Memory,
                0,
                64,
            ))
            .with_media(MediaEntry::new(
                EntryId(2),
                String::from("b"),
                MediaFormat::new(CodecTag::Pcm, 1, 48_000, Some(200)),
                Residency::Streaming {
                    prefetch_frames: 16,
                },
                64,
                128,
            ));
        assert_eq!(manifest.declared_encoded_bytes(), 192);
        // Only the memory-resident stereo entry contributes resident bytes:
        // 100 frames * 2 channels * 4 bytes = 800.
        assert_eq!(manifest.declared_resident_bytes(), 800);
    }
}
