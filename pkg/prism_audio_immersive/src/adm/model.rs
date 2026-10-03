//! The Audio Definition Model (ADM, ITU-R BS.2076) object graph.
//!
//! This module is a plain, owned, in-memory model of the ADM metadata graph:
//! the programme / content / object / pack-format / channel-format /
//! block-format hierarchy, the track-UID leaves that bind the graph to physical
//! WAV tracks, and the per-type spatial attributes (polar position, gain,
//! object size, importance). It is pure data plus lookups and validation; the
//! XML serialisation lives in [`crate::adm::xml`] and the track mapping in
//! [`crate::adm::chna`].
//!
//! # Provenance
//!
//! Original work; no Unreal Engine, Unity, Godot, Wwise, FMOD, Steam Audio,
//! Google Resonance Audio, Dolby, or MPEG source or derived code; no AI/ML.
//! The element names and the channel/object/scene (HOA) type taxonomy are the
//! publicly published ADM structure (ITU-R BS.2076), modelled here as owned
//! Rust data.
//!
//! # Relationship
//!
//! Serialised to / parsed from `axml` bytes by [`crate::adm::xml`], paired with
//! the [`crate::adm::chna`] track map, and carried inside the
//! [`crate::adm::bw64`] container. The spatial attributes mirror the same
//! source-position truth the [`crate::eif`] and [`crate::array`] layers use.

use alloc::string::String;
use alloc::vec::Vec;

use prism_audio_core::math::Sample;

/// The ADM `typeDefinition` of a pack / channel: which rendering class the
/// audio belongs to.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub enum TypeDefinition {
    /// Channel-based audio bound to fixed loudspeaker positions.
    DirectSpeakers,
    /// Matrix (for example downmix / upmix coefficient) audio.
    Matrix,
    /// Object-based audio with freely moving positions.
    Objects,
    /// Scene-based higher-order Ambisonic audio.
    Hoa,
    /// Binaural (two-channel head-related) audio.
    Binaural,
}

impl TypeDefinition {
    /// The ADM four-hex-digit `typeLabel` for this type definition.
    #[must_use]
    pub fn type_label(self) -> &'static str {
        match self {
            TypeDefinition::DirectSpeakers => "0001",
            TypeDefinition::Matrix => "0002",
            TypeDefinition::Objects => "0003",
            TypeDefinition::Hoa => "0004",
            TypeDefinition::Binaural => "0005",
        }
    }

    /// Parses an ADM `typeLabel` back into a [`TypeDefinition`].
    #[must_use]
    pub fn from_type_label(label: &str) -> Option<Self> {
        match label {
            "0001" => Some(TypeDefinition::DirectSpeakers),
            "0002" => Some(TypeDefinition::Matrix),
            "0003" => Some(TypeDefinition::Objects),
            "0004" => Some(TypeDefinition::Hoa),
            "0005" => Some(TypeDefinition::Binaural),
            _ => None,
        }
    }
}

/// An ADM spatial position in the standard polar convention: azimuth and
/// elevation in degrees, distance normalised to `[0, 1]`.
///
/// Azimuth is positive to the left in the ADM convention; the engine's own
/// geometry uses a right-handed frame, so conversions happen at the mapping
/// boundary rather than being baked in here.
#[derive(Clone, Copy, Debug, PartialEq)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub struct AdmPosition {
    /// Azimuth in degrees (ADM convention: positive to the left).
    pub azimuth: Sample,
    /// Elevation in degrees above the horizontal plane.
    pub elevation: Sample,
    /// Normalised distance in `[0, 1]`.
    pub distance: Sample,
}

impl AdmPosition {
    /// Builds a polar position, clamping distance into `[0, 1]`.
    #[must_use]
    pub fn new(azimuth: Sample, elevation: Sample, distance: Sample) -> Self {
        Self {
            azimuth,
            elevation,
            distance: distance.clamp(0.0, 1.0),
        }
    }
}

impl Default for AdmPosition {
    /// Dead ahead at full distance.
    #[inline]
    fn default() -> Self {
        Self {
            azimuth: 0.0,
            elevation: 0.0,
            distance: 1.0,
        }
    }
}

/// The ADM object `size` triplet (angular width / height and radial depth),
/// each in `[0, 1]`.
#[derive(Clone, Copy, Debug, PartialEq)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub struct ObjectSize {
    /// Angular width in `[0, 1]`.
    pub width: Sample,
    /// Angular height in `[0, 1]`.
    pub height: Sample,
    /// Radial depth in `[0, 1]`.
    pub depth: Sample,
}

impl ObjectSize {
    /// Builds a size triplet, clamping each component into `[0, 1]`.
    #[must_use]
    pub fn new(width: Sample, height: Sample, depth: Sample) -> Self {
        Self {
            width: width.clamp(0.0, 1.0),
            height: height.clamp(0.0, 1.0),
            depth: depth.clamp(0.0, 1.0),
        }
    }
}

impl Default for ObjectSize {
    /// A point source (zero size).
    #[inline]
    fn default() -> Self {
        Self {
            width: 0.0,
            height: 0.0,
            depth: 0.0,
        }
    }
}

/// One `audioBlockFormat`: a time segment of a channel's spatial metadata.
///
/// `rtime` and `duration` are in seconds relative to the start of the parent
/// element; a block with zero `rtime` and zero `duration` is a static
/// (single-segment) channel.
#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub struct AudioBlockFormat {
    /// `audioBlockFormatID` (for example `AB_00031001_00000001`).
    pub id: String,
    /// Segment start time in seconds relative to the channel.
    pub rtime: Sample,
    /// Segment duration in seconds (`0` for a static block).
    pub duration: Sample,
    /// Polar position of the source during this segment.
    pub position: AdmPosition,
    /// Linear gain applied during this segment.
    pub gain: Sample,
    /// Object size during this segment.
    pub size: ObjectSize,
    /// Rendering importance in `[0, 10]` (ADM integer importance range).
    pub importance: u8,
}

impl AudioBlockFormat {
    /// Builds a static block at `position` with unit gain and no size.
    #[must_use]
    pub fn new(id: &str, position: AdmPosition) -> Self {
        Self {
            id: String::from(id),
            rtime: 0.0,
            duration: 0.0,
            position,
            gain: 1.0,
            size: ObjectSize::default(),
            importance: 10,
        }
    }

    /// Whether this block is static (zero duration).
    #[must_use]
    pub fn is_static(&self) -> bool {
        self.duration <= 0.0
    }
}

/// One `audioChannelFormat`: a named sequence of block formats of a given type.
#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub struct AudioChannelFormat {
    /// `audioChannelFormatID`.
    pub id: String,
    /// Human-readable name.
    pub name: String,
    /// The channel's type definition.
    pub type_def: TypeDefinition,
    /// The ordered block formats describing the channel over time.
    pub block_formats: Vec<AudioBlockFormat>,
}

impl AudioChannelFormat {
    /// Builds a channel format with no block formats yet.
    #[must_use]
    pub fn new(id: &str, name: &str, type_def: TypeDefinition) -> Self {
        Self {
            id: String::from(id),
            name: String::from(name),
            type_def,
            block_formats: Vec::new(),
        }
    }
}

/// One `audioPackFormat`: a grouping of channel formats of a single type.
#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub struct AudioPackFormat {
    /// `audioPackFormatID`.
    pub id: String,
    /// Human-readable name.
    pub name: String,
    /// The pack's type definition.
    pub type_def: TypeDefinition,
    /// `audioChannelFormatIDRef` entries.
    pub channel_format_refs: Vec<String>,
}

impl AudioPackFormat {
    /// Builds a pack format with no channel references yet.
    #[must_use]
    pub fn new(id: &str, name: &str, type_def: TypeDefinition) -> Self {
        Self {
            id: String::from(id),
            name: String::from(name),
            type_def,
            channel_format_refs: Vec::new(),
        }
    }
}

/// One `audioObject`: binds pack formats to physical track UIDs.
#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub struct AudioObject {
    /// `audioObjectID`.
    pub id: String,
    /// Human-readable name.
    pub name: String,
    /// `audioPackFormatIDRef` entries.
    pub pack_format_refs: Vec<String>,
    /// `audioTrackUIDRef` entries.
    pub track_uid_refs: Vec<String>,
    /// Rendering importance in `[0, 10]`.
    pub importance: u8,
}

impl AudioObject {
    /// Builds an audio object with no references yet.
    #[must_use]
    pub fn new(id: &str, name: &str) -> Self {
        Self {
            id: String::from(id),
            name: String::from(name),
            pack_format_refs: Vec::new(),
            track_uid_refs: Vec::new(),
            importance: 10,
        }
    }
}

/// One `audioContent`: a semantic grouping of audio objects (for example
/// "dialogue", "music").
#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub struct AudioContent {
    /// `audioContentID`.
    pub id: String,
    /// Human-readable name.
    pub name: String,
    /// `audioObjectIDRef` entries.
    pub object_refs: Vec<String>,
}

impl AudioContent {
    /// Builds an audio content element with no object references yet.
    #[must_use]
    pub fn new(id: &str, name: &str) -> Self {
        Self {
            id: String::from(id),
            name: String::from(name),
            object_refs: Vec::new(),
        }
    }
}

/// One `audioProgramme`: the top-level deliverable grouping of audio content.
#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub struct AudioProgramme {
    /// `audioProgrammeID`.
    pub id: String,
    /// Human-readable name.
    pub name: String,
    /// `audioContentIDRef` entries.
    pub content_refs: Vec<String>,
}

impl AudioProgramme {
    /// Builds an audio programme with no content references yet.
    #[must_use]
    pub fn new(id: &str, name: &str) -> Self {
        Self {
            id: String::from(id),
            name: String::from(name),
            content_refs: Vec::new(),
        }
    }
}

/// One `audioTrackUID`: the leaf binding a physical WAV track to a channel and
/// pack format.
#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub struct AudioTrackUid {
    /// `audioTrackUID` (for example `ATU_00000001`).
    pub id: String,
    /// One-based physical track index inside the WAV container.
    pub track_index: u16,
    /// `audioChannelFormatIDRef`.
    pub channel_format_ref: String,
    /// `audioPackFormatIDRef`.
    pub pack_format_ref: String,
}

impl AudioTrackUid {
    /// Builds a track UID binding.
    #[must_use]
    pub fn new(id: &str, track_index: u16, channel_format_ref: &str, pack_format_ref: &str) -> Self {
        Self {
            id: String::from(id),
            track_index,
            channel_format_ref: String::from(channel_format_ref),
            pack_format_ref: String::from(pack_format_ref),
        }
    }
}

/// A validation failure for an [`AdmDocument`].
#[derive(Clone, Debug, PartialEq, Eq)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub enum AdmError {
    /// A reference pointed at an id that does not exist in the document.
    DanglingReference {
        /// The element id that holds the dangling reference.
        from: String,
        /// The referenced id that was not found.
        to: String,
    },
}

/// The complete ADM document: every element table plus cross-reference lookups
/// and validation.
#[derive(Clone, Debug, Default, PartialEq)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub struct AdmDocument {
    /// `audioProgramme` elements.
    pub programmes: Vec<AudioProgramme>,
    /// `audioContent` elements.
    pub contents: Vec<AudioContent>,
    /// `audioObject` elements.
    pub objects: Vec<AudioObject>,
    /// `audioPackFormat` elements.
    pub pack_formats: Vec<AudioPackFormat>,
    /// `audioChannelFormat` elements.
    pub channel_formats: Vec<AudioChannelFormat>,
    /// `audioTrackUID` elements.
    pub track_uids: Vec<AudioTrackUid>,
}

impl AdmDocument {
    /// An empty document.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Looks up a channel format by id.
    #[must_use]
    pub fn channel_format(&self, id: &str) -> Option<&AudioChannelFormat> {
        self.channel_formats.iter().find(|c| c.id == id)
    }

    /// Looks up a pack format by id.
    #[must_use]
    pub fn pack_format(&self, id: &str) -> Option<&AudioPackFormat> {
        self.pack_formats.iter().find(|p| p.id == id)
    }

    /// Looks up an object by id.
    #[must_use]
    pub fn object(&self, id: &str) -> Option<&AudioObject> {
        self.objects.iter().find(|o| o.id == id)
    }

    /// Looks up a track UID by id.
    #[must_use]
    pub fn track_uid(&self, id: &str) -> Option<&AudioTrackUid> {
        self.track_uids.iter().find(|t| t.id == id)
    }

    /// Validates that every cross-reference in the document resolves to an
    /// element that exists.
    ///
    /// # Errors
    ///
    /// Returns the first [`AdmError::DanglingReference`] encountered.
    pub fn validate(&self) -> Result<(), AdmError> {
        for programme in &self.programmes {
            for reference in &programme.content_refs {
                if !self.contents.iter().any(|c| &c.id == reference) {
                    return Err(dangling(&programme.id, reference));
                }
            }
        }
        for content in &self.contents {
            for reference in &content.object_refs {
                if !self.objects.iter().any(|o| &o.id == reference) {
                    return Err(dangling(&content.id, reference));
                }
            }
        }
        for object in &self.objects {
            for reference in &object.pack_format_refs {
                if !self.pack_formats.iter().any(|p| &p.id == reference) {
                    return Err(dangling(&object.id, reference));
                }
            }
            for reference in &object.track_uid_refs {
                if !self.track_uids.iter().any(|t| &t.id == reference) {
                    return Err(dangling(&object.id, reference));
                }
            }
        }
        for pack in &self.pack_formats {
            for reference in &pack.channel_format_refs {
                if !self.channel_formats.iter().any(|c| &c.id == reference) {
                    return Err(dangling(&pack.id, reference));
                }
            }
        }
        for track in &self.track_uids {
            if !self
                .channel_formats
                .iter()
                .any(|c| c.id == track.channel_format_ref)
            {
                return Err(dangling(&track.id, &track.channel_format_ref));
            }
            if !self
                .pack_formats
                .iter()
                .any(|p| p.id == track.pack_format_ref)
            {
                return Err(dangling(&track.id, &track.pack_format_ref));
            }
        }
        Ok(())
    }
}

/// Builds a [`AdmError::DanglingReference`] from borrowed ids.
fn dangling(from: &str, to: &str) -> AdmError {
    AdmError::DanglingReference {
        from: String::from(from),
        to: String::from(to),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn minimal() -> AdmDocument {
        let mut doc = AdmDocument::new();
        let mut channel = AudioChannelFormat::new("AC_00031001", "Front Left", TypeDefinition::Objects);
        channel
            .block_formats
            .push(AudioBlockFormat::new("AB_00031001_00000001", AdmPosition::new(30.0, 0.0, 1.0)));
        doc.channel_formats.push(channel);

        let mut pack = AudioPackFormat::new("AP_00031001", "Mono Object", TypeDefinition::Objects);
        pack.channel_format_refs.push(String::from("AC_00031001"));
        doc.pack_formats.push(pack);

        doc.track_uids
            .push(AudioTrackUid::new("ATU_00000001", 1, "AC_00031001", "AP_00031001"));

        let mut object = AudioObject::new("AO_1001", "Dialogue");
        object.pack_format_refs.push(String::from("AP_00031001"));
        object.track_uid_refs.push(String::from("ATU_00000001"));
        doc.objects.push(object);

        let mut content = AudioContent::new("ACO_1001", "Dialogue");
        content.object_refs.push(String::from("AO_1001"));
        doc.contents.push(content);

        let mut programme = AudioProgramme::new("APR_1001", "Main");
        programme.content_refs.push(String::from("ACO_1001"));
        doc.programmes.push(programme);
        doc
    }

    #[test]
    fn type_label_round_trips() {
        for t in [
            TypeDefinition::DirectSpeakers,
            TypeDefinition::Matrix,
            TypeDefinition::Objects,
            TypeDefinition::Hoa,
            TypeDefinition::Binaural,
        ] {
            assert_eq!(TypeDefinition::from_type_label(t.type_label()), Some(t));
        }
        assert_eq!(TypeDefinition::from_type_label("9999"), None);
    }

    #[test]
    fn valid_document_passes() {
        assert_eq!(minimal().validate(), Ok(()));
    }

    #[test]
    fn dangling_reference_is_detected() {
        let mut doc = minimal();
        doc.objects[0].pack_format_refs[0] = String::from("AP_99999999");
        match doc.validate() {
            Err(AdmError::DanglingReference { from, to }) => {
                assert_eq!(from, "AO_1001");
                assert_eq!(to, "AP_99999999");
            }
            other => panic!("expected dangling reference, got {other:?}"),
        }
    }

    #[test]
    fn lookups_resolve() {
        let doc = minimal();
        assert!(doc.channel_format("AC_00031001").is_some());
        assert!(doc.pack_format("AP_00031001").is_some());
        assert!(doc.object("AO_1001").is_some());
        assert!(doc.track_uid("ATU_00000001").is_some());
        assert!(doc.channel_format("missing").is_none());
    }

    #[test]
    fn position_and_size_clamp() {
        let p = AdmPosition::new(30.0, 10.0, 5.0);
        assert!((p.distance - 1.0).abs() < 1e-6);
        let s = ObjectSize::new(2.0, -1.0, 0.5);
        assert!((s.width - 1.0).abs() < 1e-6);
        assert!((s.height - 0.0).abs() < 1e-6);
    }
}
