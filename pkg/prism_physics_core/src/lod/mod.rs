//! Adaptive time-step and spatial level-of-detail control (milestone M7).
//!
//! Two orthogonal controllers keep simulation cost proportional to visible
//! importance. The temporal controller picks a substep count each frame from a
//! CFL target so fast motion is stepped finely and slow motion cheaply. The
//! spatial controller buckets bodies into level-of-detail tiers by distance
//! from a focus point, scaling their solver iteration budget so distant
//! objects cost less without popping.
//!
//! This module is populated by milestone M7.
//!
//! # Provenance
//!
//! This module contains **no Unreal Engine source or derived code**. CFL-based
//! adaptive substepping and distance-based spatial LOD are standard, publicly
//! documented real-time-simulation techniques.
