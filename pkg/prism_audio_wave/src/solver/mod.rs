//! Offline wave solver: the pieces that turn a static voxel scene into
//! time-domain impulse responses on the probe grid.
//!
//! # Provenance
//! Original work; no Unreal Engine, Unity, Godot, Wwise, FMOD, Steam Audio,
//! Microsoft Project Acoustics, or Google Resonance Audio source or derived
//! code; no AI/ML.
//!
//! # Relationship
//! Groups the "Wave Bake" stages of design section 43: a voxelised scene
//! ([`scene`]), its rectangular decomposition ([`partition`]), the
//! Adaptive Rectangular Decomposition (`ARD`) solver ([`ard`]) that advances
//! each partition analytically and couples neighbours numerically, and the
//! time-domain impulse response ([`impulse`]) the solver records at each probe
//! cell. Everything here is offline only; the runtime path never touches it.

pub mod ard;
pub mod impulse;
pub mod partition;
pub mod scene;

pub use ard::{SolveConfig, WaveSolver, DEFAULT_SOUND_SPEED};
pub use impulse::ImpulseResponse;
pub use partition::{partition_scene, Partition};
pub use scene::VoxelScene;
