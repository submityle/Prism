//! Fixed-step **render interpolation** (M4).
//!
//! A deterministic simulation advances on a fixed timestep, but frames are
//! presented at an unrelated (usually higher, usually variable) rate. Rendering
//! the simulation's latest state verbatim makes motion stutter whenever the
//! render cadence and the sim cadence disagree. The standard fix is to keep the
//! *previous* and *current* simulation states and present
//! `lerp(previous, current, alpha)`, where `alpha ∈ [0, 1]` is the fraction of
//! the next fixed step that the current frame falls into (the accumulator's
//! "overstep fraction").
//!
//! This module provides:
//! - [`Transform::interpolate`] / [`GlobalTransform::interpolate`] — a single
//!   SRT-correct blend (translation `lerp`, rotation shortest-path `slerp`,
//!   scale `lerp`).
//! - [`InterpolationBuffer`] — a double-buffered snapshot of every node's world
//!   transform plus a **teleport** bitset, so a node that was repositioned
//!   discontinuously snaps to its new pose instead of smearing across the jump.
//!
//! Everything here is `no_std` + `alloc`, pure math (no threads, no clock), and
//! leaves the authoritative simulation state untouched — interpolation only ever
//! produces display poses.

use alloc::vec::Vec;

use prism_math::{Quat, Vec3};

use crate::{GlobalTransform, Transform};

/// Clamp an interpolation factor into the closed unit interval.
///
/// `alpha` arrives from an accumulator's overstep fraction, which *should*
/// already be in `[0, 1]`; clamping defends against a caller that oversteps or
/// passes a tiny negative from float error, so interpolation never extrapolates.
#[inline]
#[must_use]
pub fn clamp_alpha(alpha: f32) -> f32 {
    // Written as nested min/max (not `f32::clamp`) so a NaN collapses to 0.0
    // rather than propagating; a NaN display pose would be far worse than snap.
    let a = if alpha > 1.0 { 1.0 } else { alpha };
    if a > 0.0 {
        a
    } else {
        0.0
    }
}

impl Transform {
    /// Interpolate between `self` (the previous fixed-step pose) and `target`
    /// (the current one) by `alpha ∈ [0, 1]`.
    ///
    /// Translation and scale use component-wise `lerp`; rotation uses
    /// shortest-path `slerp` (so a quaternion and its negation — the same
    /// orientation — never take the long way round). `alpha` is clamped, so the
    /// result is always a convex blend and never extrapolates past either pose.
    ///
    /// At `alpha == 0.0` the result equals `self`; at `alpha == 1.0` it equals
    /// `target` (both exactly, barring `slerp`'s renormalization of an already
    /// unit input).
    #[inline]
    #[must_use]
    pub fn interpolate(&self, target: &Transform, alpha: f32) -> Transform {
        let a = clamp_alpha(alpha);
        Transform {
            translation: self.translation.lerp(target.translation, a),
            rotation: self.rotation.slerp(target.rotation, a),
            scale: self.scale.lerp(target.scale, a),
        }
    }
}

impl GlobalTransform {
    /// Interpolate between two world transforms by `alpha ∈ [0, 1]`.
    ///
    /// World transforms are stored as [`Affine3`](prism_math::Affine3), which
    /// can carry shear when a hierarchy mixes non-uniform scale with rotation —
    /// and shear cannot be `slerp`'d. To interpolate rigidly we decompose each
    /// side into scale / rotation / translation, blend those (scale & translation
    /// `lerp`, rotation `slerp`), and recompose. For the overwhelmingly common
    /// shear-free case this is exact; when shear *is* present the blend is the
    /// best rigid approximation and the shear is dropped for the displayed frame
    /// only (the authoritative transform is unaffected). This matches how
    /// production engines interpolate skeletal/scene nodes.
    #[inline]
    #[must_use]
    pub fn interpolate(&self, target: &GlobalTransform, alpha: f32) -> GlobalTransform {
        let a = clamp_alpha(alpha);
        let (s0, r0, t0) = self.0.to_scale_rotation_translation();
        let (s1, r1, t1) = target.0.to_scale_rotation_translation();
        GlobalTransform::from_srt(s0.lerp(s1, a), r0.slerp(r1, a), t0.lerp(t1, a))
    }

    /// Build a world transform from scale / rotation / translation.
    #[inline]
    #[must_use]
    fn from_srt(scale: Vec3, rotation: Quat, translation: Vec3) -> GlobalTransform {
        GlobalTransform(prism_math::Affine3::from_scale_rotation_translation(
            scale,
            rotation,
            translation,
        ))
    }
}

/// A double-buffered store of per-node world transforms for render interpolation,
/// plus a teleport bitset that marks nodes which must snap rather than blend.
///
/// # Lifecycle (one entry per simulation step)
/// 1. [`begin_step`](Self::begin_step) — the old `current` becomes `previous`.
/// 2. [`set_current`](Self::set_current) (or [`set_current_from`](Self::set_current_from))
///    — write the freshly simulated world transforms into `current`.
/// 3. Flag any discontinuous node with [`mark_teleport`](Self::mark_teleport).
/// 4. Per rendered frame, [`sample`](Self::sample) with the accumulator's
///    overstep fraction to fill a display buffer.
/// 5. [`clear_teleports`](Self::clear_teleports) once the step's frames are done,
///    so the flag is a one-step pulse.
///
/// The teleport flags are stored as a packed `u64` word bitset (one bit per
/// node), so marking and testing are O(1) and the whole set clears in a memset.
#[derive(Clone, Debug, Default)]
pub struct InterpolationBuffer {
    previous: Vec<GlobalTransform>,
    current: Vec<GlobalTransform>,
    /// One bit per node; bit set ⇒ that node teleported this step and must snap.
    teleport: Vec<u64>,
    len: usize,
}

#[inline]
const fn words_for(len: usize) -> usize {
    len.div_ceil(64)
}

impl InterpolationBuffer {
    /// An empty buffer with no nodes.
    #[inline]
    #[must_use]
    pub const fn new() -> Self {
        Self {
            previous: Vec::new(),
            current: Vec::new(),
            teleport: Vec::new(),
            len: 0,
        }
    }

    /// A buffer sized for `len` nodes, every pose seeded to identity and no
    /// teleport flags set.
    #[inline]
    #[must_use]
    pub fn with_len(len: usize) -> Self {
        let mut b = Self::new();
        b.resize(len);
        b
    }

    /// Number of nodes the buffer tracks.
    #[inline]
    #[must_use]
    pub fn len(&self) -> usize {
        self.len
    }

    /// Whether the buffer tracks no nodes.
    #[inline]
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Resize to `len` nodes. New entries seed to identity in both buffers; the
    /// teleport bitset grows/shrinks to match and any excess high bits are
    /// cleared so a shrink-then-grow never resurrects a stale flag.
    pub fn resize(&mut self, len: usize) {
        self.previous.resize(len, GlobalTransform::IDENTITY);
        self.current.resize(len, GlobalTransform::IDENTITY);
        self.teleport.resize(words_for(len), 0);
        // Clear any bits in the final word beyond `len`.
        if let Some(last) = self.teleport.last_mut() {
            let used = len % 64;
            if used != 0 {
                *last &= (1u64 << used) - 1;
            }
        }
        self.len = len;
    }

    /// Begin a new simulation step: the current poses become the previous poses.
    ///
    /// Call this immediately before writing the newly simulated state via
    /// [`set_current`](Self::set_current).
    #[inline]
    pub fn begin_step(&mut self) {
        self.previous.copy_from_slice(&self.current);
    }

    /// Overwrite the current poses from a slice.
    ///
    /// # Panics
    /// Panics if `globals.len() != self.len()`.
    #[inline]
    pub fn set_current(&mut self, globals: &[GlobalTransform]) {
        assert_eq!(globals.len(), self.len, "set_current length mismatch");
        self.current.copy_from_slice(globals);
    }

    /// Overwrite the current pose of a single node.
    ///
    /// # Panics
    /// Panics if `node >= self.len()`.
    #[inline]
    pub fn set_current_at(&mut self, node: usize, global: GlobalTransform) {
        self.current[node] = global;
    }

    /// Read-only view of the previous-step poses.
    #[inline]
    #[must_use]
    pub fn previous(&self) -> &[GlobalTransform] {
        &self.previous
    }

    /// Read-only view of the current-step poses.
    #[inline]
    #[must_use]
    pub fn current(&self) -> &[GlobalTransform] {
        &self.current
    }

    /// Flag `node` as teleported this step: [`sample`](Self::sample) will snap it
    /// to its current pose instead of blending from its (now meaningless)
    /// previous pose.
    ///
    /// # Panics
    /// Panics if `node >= self.len()`.
    #[inline]
    pub fn mark_teleport(&mut self, node: usize) {
        assert!(node < self.len, "mark_teleport out of bounds");
        self.teleport[node / 64] |= 1u64 << (node % 64);
    }

    /// Whether `node` is currently flagged as teleported.
    ///
    /// # Panics
    /// Panics if `node >= self.len()`.
    #[inline]
    #[must_use]
    pub fn is_teleport(&self, node: usize) -> bool {
        assert!(node < self.len, "is_teleport out of bounds");
        (self.teleport[node / 64] >> (node % 64)) & 1 != 0
    }

    /// Clear every teleport flag. Call once a step's rendered frames are done so
    /// the flag behaves as a single-step pulse.
    #[inline]
    pub fn clear_teleports(&mut self) {
        for word in &mut self.teleport {
            *word = 0;
        }
    }

    /// Sample a display pose for every node into `out`.
    ///
    /// Non-teleported nodes get `previous.interpolate(current, alpha)`;
    /// teleported nodes snap directly to `current` (no blend across the jump).
    ///
    /// # Panics
    /// Panics if `out.len() != self.len()`.
    pub fn sample(&self, alpha: f32, out: &mut [GlobalTransform]) {
        assert_eq!(out.len(), self.len, "sample output length mismatch");
        let a = clamp_alpha(alpha);
        for (i, slot) in out.iter_mut().enumerate() {
            *slot = if self.is_teleport(i) {
                self.current[i]
            } else {
                self.previous[i].interpolate(&self.current[i], a)
            };
        }
    }

    /// Sample a display pose for every node into a freshly allocated `Vec`.
    #[inline]
    #[must_use]
    pub fn sample_to_vec(&self, alpha: f32) -> Vec<GlobalTransform> {
        let mut out = alloc::vec![GlobalTransform::IDENTITY; self.len];
        self.sample(alpha, &mut out);
        out
    }
}
