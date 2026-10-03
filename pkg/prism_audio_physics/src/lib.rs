//! Physics-engine coupling bridge for Prism's next-generation audio engine.
//!
//! This crate is the typed, deterministic conduit between the rigid-body
//! physics world and the procedural contact-sound synthesis of
//! [`prism_audio_procedural`]. The physics solver already computes the
//! impulses, contact points, and relative velocities that a listener hears as
//! sound; this bridge reads those facts and translates them into the
//! [`prism_audio_procedural::contact`] event stream
//! (`ImpactEvent` / `SustainEvent` / `SeparationEvent`) carrying sample-accurate
//! offsets, without building a second "acoustic collision" system.
//!
//! # Module map
//!
//! - [`body`] holds [`body::BodyAudioState`] and [`body::BodyAudioId`], the
//!   per-body kinematic/material snapshot the bridge consumes.
//! - [`cluster`] collapses distant, near-simultaneous impacts into a single
//!   representative "group impact" to bound the voice count.
//! - [`config`] holds [`config::TranslatorConfig`] and the sub-configs that
//!   tune impulse estimation, merging, and clustering.
//! - [`contact_id`] allocates stable [`contact_id::ContactKey`] identities so a
//!   persistent manifold keeps the same procedural `ContactId` across blocks.
//! - [`contact_input`] holds the engine-agnostic input views
//!   ([`contact_input::ContactManifoldView`], [`contact_input::ContactPhase`]).
//! - [`impulse`] estimates the collision impulse and its normal/tangential
//!   split from the closing velocity and reduced mass.
//! - [`kinematics`] computes the relative contact velocity and its
//!   normal/tangential decomposition.
//! - [`material`] resolves an ordered acoustic-material pair into the
//!   procedural `MaterialPairId`, with a category fallback that leaves no hole.
//! - [`offset`] maps a physics sub-step time fraction into a sample offset
//!   inside the current audio block.
//! - [`roughness`] derives the normalized strike position and surface
//!   roughness that shape the modal excitation.
//! - [`translator`] is the stateful [`translator::ContactAudioTranslator`] that
//!   ingests per-block contact facts and drains procedural events.
//!
//! The optional `physics-core` feature adds [`physics_core`], concrete adapters
//! that read [`prism_physics_core`] contact manifolds and physics events.
//!
//! # Provenance
//! Original work; no Unreal Engine, Unity, Godot, Wwise, FMOD, Steam Audio,
//! Dolby, or Google Resonance Audio source or derived code; no AI/ML.
//!
//! # Relationship
//! Implements the physics-engine coupling of design section 47 (47.1 contact
//! event bus, energy mapping, budget/merge/cluster). It bridges the rigid-body
//! facts exposed by `prism_physics_core` to the synthesis stages in
//! `prism_audio_procedural`, feeding sample-accurate events toward
//! `prism_audio_core::scheduler::EventScheduler`.
#![forbid(unsafe_code)]
#![cfg_attr(not(feature = "std"), no_std)]

extern crate alloc;

pub mod body;
pub mod cluster;
pub mod config;
pub mod contact_id;
pub mod contact_input;
pub mod impulse;
pub mod kinematics;
pub mod material;
pub mod offset;
#[cfg(feature = "physics-core")]
pub mod physics_core;
pub mod roughness;
pub mod translator;
