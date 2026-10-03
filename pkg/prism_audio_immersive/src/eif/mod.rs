//! MPEG-I-style Encoder Input Format (EIF): a declarative six-degree-of-freedom
//! acoustic scene model and its import/export bridges.
//!
//! This module gathers the EIF scene layer described in design section 49.1: a
//! declarative description of geometry, acoustic materials, sources, walkable
//! regions, portals, and reverberation zones that maps onto the engine's
//! shared geometry and parameter buses rather than maintaining a second
//! acoustic truth. The submodules split that model by concept:
//!
//! * [`material`] - frequency-dependent acoustic surface materials.
//! * [`geometry`] - faces, boxes, and mesh references with derived bounds.
//! * [`source`] - object / channel / HOA sources with directivity and roll-off.
//! * [`scene`] - the authoritative [`scene::EifScene`] aggregate plus
//!   validation.
//! * [`import`] - scene to runtime emitters, descriptors, and BVH geometry.
//! * [`export`] - engine object/bed scene to an EIF scene for interchange.
//!
//! # Provenance
//!
//! Original work; no Unreal Engine, Unity, Godot, Wwise, FMOD, Steam Audio,
//! Google Resonance Audio, Dolby, or MPEG source or derived code; no AI/ML.
//! Implements only the public EIF scene semantics of MPEG-I Immersive Audio
//! (ISO/IEC 23090-4, section 49.1).
//!
//! # Relationship
//!
//! Builds on `prism_audio_spatial` (materials, directivity, attenuation,
//! spatializer, geometry) and `prism_audio_object` (bed/object model). The
//! other top-level modules of this crate, [`crate::adm`] and [`crate::array`],
//! consume the same source-position truth this module imports and exports.

pub mod export;
pub mod geometry;
pub mod import;
pub mod material;
pub mod scene;
pub mod source;

pub use export::{export_bed, export_object, export_object_scene};
pub use geometry::{
    Aabb, AcousticBox, AcousticFace, AcousticMesh, GeometryPrimitive, MaterialRef, MeshRef,
};
pub use import::{import_scene, import_source, ImportedGeometry, ImportedScene, ImportedSource};
pub use material::AcousticMaterial;
pub use scene::{EifPortal, EifReverbZone, EifScene, SceneError};
pub use source::{ChannelLabel, EifSource, EifSourceId, EifSourceKind};
