//! The pluggable [`DecoderRegistry`]: container/tag dispatch to decoders.
//!
//! # Provenance
//! Original work; no Unreal Engine, Unity, Godot, Wwise, FMOD, Steam Audio, or
//! Google Resonance Audio source or derived code; no AI/ML.
//!
//! # Relationship
//! Implements the "format plugin-able, third parties can register custom
//! decoders" requirement of design sections 20 and 44.1. The registry maps a
//! container magic signature or an explicit [`CodecTag`] to a factory that
//! builds a boxed [`SourceDecoder`]. The native WAV container (PCM/IMA/MS
//! ADPCM) is registered by [`DecoderRegistry::with_native`]; external
//! Vorbis/Opus/FLAC crates register their own factories through
//! [`DecoderRegistry::register_tag`].

use alloc::boxed::Box;
use alloc::collections::BTreeMap;
use alloc::vec::Vec;

use crate::codec::decoder::{DecodeError, SourceDecoder};
use crate::codec::metadata::CodecTag;
use crate::codec::wav::decode_wav;

/// A boxed decoder trait object that is safe to move between tasks.
pub type BoxedDecoder = Box<dyn SourceDecoder + Send>;

/// A factory that turns encoded bytes into a boxed decoder.
pub type DecoderFactory =
    Box<dyn Fn(&[u8]) -> Result<BoxedDecoder, DecodeError> + Send + Sync>;

/// A container entry: a leading-byte magic signature and its factory.
struct ContainerEntry {
    magic: Vec<u8>,
    factory: DecoderFactory,
}

/// Dispatches encoded media to a registered decoder by container magic or by
/// an explicit codec tag.
#[derive(Default)]
pub struct DecoderRegistry {
    containers: Vec<ContainerEntry>,
    by_tag: BTreeMap<CodecTag, DecoderFactory>,
}

impl DecoderRegistry {
    /// Creates an empty registry with no decoders.
    #[must_use]
    pub fn new() -> Self {
        Self {
            containers: Vec::new(),
            by_tag: BTreeMap::new(),
        }
    }

    /// Creates a registry pre-populated with this crate's native decoders.
    ///
    /// Registers the RIFF/WAVE container (`RIFF` magic) which resolves to the
    /// PCM, IMA ADPCM, or MS ADPCM decoder based on the `fmt ` tag.
    #[must_use]
    pub fn with_native() -> Self {
        let mut registry = Self::new();
        registry.register_container(b"RIFF", Box::new(decode_wav));
        registry
    }

    /// Registers a container decoder keyed by a leading-byte magic signature.
    ///
    /// Longer, more specific signatures should be registered first; probing
    /// checks containers in registration order.
    pub fn register_container(&mut self, magic: &[u8], factory: DecoderFactory) {
        self.containers.push(ContainerEntry {
            magic: magic.to_vec(),
            factory,
        });
    }

    /// Registers a decoder factory for an explicit [`CodecTag`].
    ///
    /// This is the extension point external Vorbis/Opus/FLAC decoder crates use
    /// to plug into the engine without this crate depending on them.
    pub fn register_tag(&mut self, tag: CodecTag, factory: DecoderFactory) {
        self.by_tag.insert(tag, factory);
    }

    /// Returns `true` when a factory is registered for `tag`.
    #[must_use]
    pub fn supports_tag(&self, tag: &CodecTag) -> bool {
        self.by_tag.contains_key(tag)
    }

    /// Decodes `bytes` by probing registered container signatures.
    ///
    /// Returns [`DecodeError::UnsupportedFormat`] when no container magic
    /// matches the leading bytes.
    pub fn decode_bytes(&self, bytes: &[u8]) -> Result<BoxedDecoder, DecodeError> {
        for entry in &self.containers {
            if bytes.len() >= entry.magic.len() && bytes[..entry.magic.len()] == entry.magic[..] {
                return (entry.factory)(bytes);
            }
        }
        Err(DecodeError::UnsupportedFormat)
    }

    /// Decodes `bytes` using the factory registered for `tag`.
    ///
    /// Returns [`DecodeError::UnsupportedFormat`] when no factory is registered
    /// for the tag (for example an Opus asset with no external decoder loaded).
    pub fn decode_tagged(&self, tag: &CodecTag, bytes: &[u8]) -> Result<BoxedDecoder, DecodeError> {
        match self.by_tag.get(tag) {
            Some(factory) => factory(bytes),
            None => Err(DecodeError::UnsupportedFormat),
        }
    }
}
