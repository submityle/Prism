//! ADM / BW64 / Serial ADM production interchange (design section 49.2).
//!
//! This module is the production-and-exchange interoperability layer: the
//! Audio Definition Model (ADM, ITU-R BS.2076) object graph, its `axml` XML
//! serialisation, the `chna` channel-to-track map, the BW64 (ITU-R BS.2088)
//! 64-bit broadcast WAV container that carries both, and Serial ADM
//! (ITU-R BS.2125) frame-serialised metadata for live and streaming delivery.
//! Everything is deterministic structured data plus byte-exact serialisers;
//! there is no codec and no spatialisation here.
//!
//! The submodules split the standard by concept:
//!
//! * [`model`] - the owned ADM object graph (programme / content / object /
//!   pack / channel / block / track-UID) and its validation.
//! * [`xml`] - a minimal deterministic `axml` writer and reader.
//! * [`chna`] - the fixed-width `chna` track-mapping table.
//! * [`bw64`] - the byte-level BW64 container (RIFF / `ds64` / `data` /
//!   `axml` / `chna`).
//! * [`sadm`] - the Serial ADM frame sequence with trajectory sampling.
//!
//! # Provenance
//!
//! Original work; no Unreal Engine, Unity, Godot, Wwise, FMOD, Steam Audio,
//! Google Resonance Audio, Dolby, or MPEG source or derived code; no AI/ML.
//! Implements only the public standard semantics of ADM (ITU-R BS.2076), BW64
//! (ITU-R BS.2088), and Serial ADM (ITU-R BS.2125).
//!
//! # Relationship
//!
//! Describes the same source-position truth that the [`crate::eif`] scene
//! layer and the [`crate::array`] panners consume. A deliverable ADM master is
//! a [`bw64::Bw64File`] whose `axml` is the [`xml`]-serialised
//! [`model::AdmDocument`] and whose `chna` is the [`chna::ChnaChunk`] track
//! map; live delivery uses the [`sadm::SadmSequence`] frame stream.

pub mod bw64;
pub mod chna;
pub mod model;
pub mod sadm;
pub mod xml;

pub use bw64::{Bw64Error, Bw64File, WaveFormat};
pub use chna::{AudioId, ChnaChunk, ChnaError};
pub use model::{
    AdmDocument, AdmError, AdmPosition, AudioBlockFormat, AudioChannelFormat, AudioContent,
    AudioObject, AudioPackFormat, AudioProgramme, AudioTrackUid, ObjectSize, TypeDefinition,
};
pub use sadm::{
    FrameType, SadmError, SadmFrame, SadmFrameFormat, SadmObjectUpdate, SadmSequence,
};
pub use xml::{from_axml_bytes, to_axml_bytes, XmlElement, XmlError, XmlNode};
