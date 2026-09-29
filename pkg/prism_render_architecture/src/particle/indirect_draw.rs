//! `GPU`-driven indirect draw / dispatch argument assembly (design §9, §15).
//!
//! A `GPU`-simulated emitter never tells the host how many particles survived a
//! frame; the alive count lives in device memory. The draw and dispatch commands
//! must therefore read their launch parameters from an indirect argument buffer
//! that a compute pass fills in. This module is the deterministic `CPU` reference
//! for packing those parameters — the little-endian `u32` words that `wgpu` /
//! `WebGPU` `draw_indirect`, `draw_indexed_indirect`, and
//! `dispatch_workgroups_indirect` consume — from a live alive count and the
//! per-renderer geometry contract. It mirrors how Unreal `Niagara`'s `GPU`
//! simulation writes its `DrawIndirectArgs` after the sim kernel resolves the
//! particle count.
//!
//! **Orthogonality.** This layer owns only the *packing* of already-decided
//! counts into the indirect buffer (the command end):
//! - [`super::dual_backend`] decides *scheduling strategy* — whether dispatch is
//!   indirect at all, the workgroup size, and the per-dispatch workgroup budget.
//!   [`DispatchIndirectArgs::linear_1d`] merely packs a workgroup count using a
//!   caller-supplied size and cap; it does not choose them.
//! - [`super::sort_cull`] decides *ordering and visibility* (sort key, frustum /
//!   `HZB` culling); the alive / visible count it produces is an input here.
//! - [`super::renderers`] owns the *renderer matrix* and the per-[`RendererKind`]
//!   geometry (quad, mesh, ribbon strip, beam chain); this module consumes that
//!   [`RendererKind`] taxonomy to select an argument layout.
//!
//! All arithmetic is integer and saturating: `saturating_add` / `saturating_mul`
//! for count products and prefix offsets, and a divide-by-zero-guarded ceil-div
//! `(n + d - 1) / d` for the workgroup count. There is no `f32` here, so no
//! epsilon comparison policy applies.

use alloc::vec::Vec;

use super::renderers::RendererKind;

/// Vertices (or indices) in one camera-facing sprite quad: two triangles.
///
/// A `Sprite` renderer draws this fixed six-vertex quad once per particle via
/// instancing, so the per-instance geometry is constant and the alive count
/// only drives `instance_count`.
pub const SPRITE_QUAD_VERTEX_COUNT: u32 = 6;

/// Indices in one indexed sprite quad (two triangles sharing an edge).
pub const SPRITE_QUAD_INDEX_COUNT: u32 = 6;

/// Indices generated per ribbon/beam segment: two triangles forming a quad
/// between consecutive spine points (`GPU` strip expansion, design §15).
pub const INDICES_PER_SEGMENT: u32 = 6;

/// Non-indexed indirect draw parameters, matching the `wgpu` / `WebGPU`
/// `draw_indirect` buffer layout `[vertex_count, instance_count, first_vertex,
/// first_instance]` (design §9).
#[derive(Clone, Copy, Debug, Default, Eq, Hash, PartialEq)]
pub struct DrawIndirectArgs {
    /// Vertices drawn per instance.
    pub vertex_count: u32,
    /// Instances drawn (one per alive particle for a `Sprite`).
    pub instance_count: u32,
    /// Index of the first vertex within the vertex buffer.
    pub first_vertex: u32,
    /// Index of the first instance, used to offset into a shared instance
    /// buffer when several emitters batch into one buffer.
    pub first_instance: u32,
}

impl DrawIndirectArgs {
    /// Draw args for a non-indexed `Sprite` emitter: a fixed
    /// [`SPRITE_QUAD_VERTEX_COUNT`]-vertex quad instanced once per alive
    /// particle, offset by `first_instance` into a shared instance buffer.
    #[must_use]
    pub const fn sprite(alive_count: u32, first_instance: u32) -> Self {
        Self {
            vertex_count: SPRITE_QUAD_VERTEX_COUNT,
            instance_count: alive_count,
            first_vertex: 0,
            first_instance,
        }
    }

    /// The four `u32` words in `wgpu` / `WebGPU` indirect-buffer order.
    #[must_use]
    pub const fn as_words(&self) -> [u32; 4] {
        [
            self.vertex_count,
            self.instance_count,
            self.first_vertex,
            self.first_instance,
        ]
    }
}

/// Indexed indirect draw parameters, matching the `wgpu` / `WebGPU`
/// `draw_indexed_indirect` buffer layout `[index_count, instance_count,
/// first_index, base_vertex, first_instance]` (design §9).
///
/// `base_vertex` is signed, as in `WebGPU`; its raw 32-bit two's-complement word
/// is what the indirect buffer stores.
#[derive(Clone, Copy, Debug, Default, Eq, Hash, PartialEq)]
pub struct DrawIndexedIndirectArgs {
    /// Indices drawn per instance.
    pub index_count: u32,
    /// Instances drawn.
    pub instance_count: u32,
    /// Index of the first index within the index buffer.
    pub first_index: u32,
    /// Value added to each index before indexing the vertex buffer (signed).
    pub base_vertex: i32,
    /// Index of the first instance within a shared instance buffer.
    pub first_instance: u32,
}

impl DrawIndexedIndirectArgs {
    /// Draw args for a `Mesh` emitter: one indexed mesh instance per alive
    /// particle. `index_count` is the source mesh's index count.
    #[must_use]
    pub const fn mesh(alive_count: u32, index_count: u32, first_instance: u32) -> Self {
        Self {
            index_count,
            instance_count: alive_count,
            first_index: 0,
            base_vertex: 0,
            first_instance,
        }
    }

    /// Draw args for a `Ribbon` emitter: one indexed strip instance per chain,
    /// where each of `segments_per_chain` segments expands to
    /// [`INDICES_PER_SEGMENT`] indices. The index count saturates on overflow.
    #[must_use]
    pub const fn ribbon(chain_count: u32, segments_per_chain: u32, first_instance: u32) -> Self {
        Self {
            index_count: segments_per_chain.saturating_mul(INDICES_PER_SEGMENT),
            instance_count: chain_count,
            first_index: 0,
            base_vertex: 0,
            first_instance,
        }
    }

    /// Draw args for a `Beam` emitter. Beams share the ribbon's segment-to-index
    /// expansion ([`INDICES_PER_SEGMENT`] per segment, one instance per chain);
    /// the distinct constructor keeps the renderer intent explicit.
    #[must_use]
    pub const fn beam(chain_count: u32, segments_per_chain: u32, first_instance: u32) -> Self {
        Self {
            index_count: segments_per_chain.saturating_mul(INDICES_PER_SEGMENT),
            instance_count: chain_count,
            first_index: 0,
            base_vertex: 0,
            first_instance,
        }
    }

    /// The five `u32` words in `wgpu` / `WebGPU` indirect-buffer order; the
    /// signed `base_vertex` is reinterpreted as its raw 32-bit word.
    #[must_use]
    pub const fn as_words(&self) -> [u32; 5] {
        [
            self.index_count,
            self.instance_count,
            self.first_index,
            self.base_vertex as u32,
            self.first_instance,
        ]
    }
}

/// Indirect compute dispatch parameters, matching the `wgpu` / `WebGPU`
/// `dispatch_workgroups_indirect` buffer layout `[x, y, z]` (design §9, §24).
#[derive(Clone, Copy, Debug, Default, Eq, Hash, PartialEq)]
pub struct DispatchIndirectArgs {
    /// Workgroups launched along the primary (`X`) axis.
    pub x: u32,
    /// Workgroups along `Y` (`1` for a 1D particle sweep).
    pub y: u32,
    /// Workgroups along `Z` (`1` for a 1D particle sweep).
    pub z: u32,
}

impl DispatchIndirectArgs {
    /// A dispatch with explicit workgroup counts on each axis.
    #[must_use]
    pub const fn new(x: u32, y: u32, z: u32) -> Self {
        Self { x, y, z }
    }

    /// A 1D dispatch covering `alive_count` invocations at `workgroup_size`
    /// invocations per workgroup, with `y = z = 1`.
    ///
    /// The workgroup count is `ceil(alive_count / workgroup_size)`, clamped to
    /// `max_workgroups`. A `workgroup_size` of `0` yields `x = 0` (no launch),
    /// and a `max_workgroups` of `0` likewise clamps `x` to `0`. This packs a
    /// count only; the size and cap are policy chosen by
    /// [`super::dual_backend`].
    #[must_use]
    pub const fn linear_1d(alive_count: u32, workgroup_size: u32, max_workgroups: u32) -> Self {
        let needed = ceil_div_saturating(alive_count, workgroup_size);
        let x = if needed > max_workgroups {
            max_workgroups
        } else {
            needed
        };
        Self { x, y: 1, z: 1 }
    }

    /// The three `u32` words in `wgpu` / `WebGPU` indirect-buffer order.
    #[must_use]
    pub const fn as_words(&self) -> [u32; 3] {
        [self.x, self.y, self.z]
    }
}

/// A packed indirect draw command tagged by whether it is indexed, so a batch
/// assembler can route each emitter to the matching indirect buffer (design
/// §9, §15).
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum EmitterDrawArgs {
    /// A non-indexed draw (`Sprite`).
    NonIndexed(DrawIndirectArgs),
    /// An indexed draw (`Mesh` / `Ribbon` / `Beam`).
    Indexed(DrawIndexedIndirectArgs),
}

/// The per-renderer geometry inputs needed to derive draw args, one variant per
/// supported [`RendererKind`] (design §15).
///
/// Each variant carries the alive-driven count and the per-instance geometry the
/// renderer contributes: a `Sprite` needs only its alive count, a `Mesh` its
/// index count, and a `Ribbon` / `Beam` its chain count and segments per chain.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum RendererGeometry {
    /// Non-indexed billboards, one quad instance per alive particle.
    Sprite {
        /// Alive particles this frame.
        alive_count: u32,
        /// Starting instance offset into a shared instance buffer.
        first_instance: u32,
    },
    /// Indexed mesh instances, one per alive particle.
    Mesh {
        /// Alive particles this frame.
        alive_count: u32,
        /// Indices in the source mesh.
        index_count: u32,
        /// Starting instance offset into a shared instance buffer.
        first_instance: u32,
    },
    /// Indexed ribbon strips, one instance per chain.
    Ribbon {
        /// Ribbon chains this frame.
        chain_count: u32,
        /// Segments per chain (each expands to [`INDICES_PER_SEGMENT`]).
        segments_per_chain: u32,
        /// Starting instance offset into a shared instance buffer.
        first_instance: u32,
    },
    /// Indexed beam chains, one instance per chain.
    Beam {
        /// Beam chains this frame.
        chain_count: u32,
        /// Segments per chain (each expands to [`INDICES_PER_SEGMENT`]).
        segments_per_chain: u32,
        /// Starting instance offset into a shared instance buffer.
        first_instance: u32,
    },
}

impl RendererGeometry {
    /// The [`RendererKind`] this geometry variant corresponds to, reusing the
    /// renderer taxonomy from [`super::renderers`].
    #[must_use]
    pub const fn kind(&self) -> RendererKind {
        match self {
            RendererGeometry::Sprite { .. } => RendererKind::Sprite,
            RendererGeometry::Mesh { .. } => RendererKind::Mesh,
            RendererGeometry::Ribbon { .. } => RendererKind::Ribbon,
            RendererGeometry::Beam { .. } => RendererKind::Beam,
        }
    }

    /// Derive the packed indirect draw args for this renderer geometry.
    #[must_use]
    pub const fn draw_args(&self) -> EmitterDrawArgs {
        match *self {
            RendererGeometry::Sprite {
                alive_count,
                first_instance,
            } => EmitterDrawArgs::NonIndexed(DrawIndirectArgs::sprite(alive_count, first_instance)),
            RendererGeometry::Mesh {
                alive_count,
                index_count,
                first_instance,
            } => EmitterDrawArgs::Indexed(DrawIndexedIndirectArgs::mesh(
                alive_count,
                index_count,
                first_instance,
            )),
            RendererGeometry::Ribbon {
                chain_count,
                segments_per_chain,
                first_instance,
            } => EmitterDrawArgs::Indexed(DrawIndexedIndirectArgs::ribbon(
                chain_count,
                segments_per_chain,
                first_instance,
            )),
            RendererGeometry::Beam {
                chain_count,
                segments_per_chain,
                first_instance,
            } => EmitterDrawArgs::Indexed(DrawIndexedIndirectArgs::beam(
                chain_count,
                segments_per_chain,
                first_instance,
            )),
        }
    }
}

/// Exclusive prefix sum of per-emitter alive counts, giving each emitter's
/// `first_instance` start when several emitters batch into one instance buffer
/// (design §9, §15).
///
/// The returned vector has the same length as `alive_counts`; entry `i` is the
/// sum of `alive_counts[0..i]`, saturating at [`u32::MAX`] so a runaway total
/// never wraps. The first entry is always `0`.
#[must_use]
pub fn first_instance_prefix(alive_counts: &[u32]) -> Vec<u32> {
    let mut offsets = Vec::with_capacity(alive_counts.len());
    let mut running = 0u32;
    for &count in alive_counts {
        offsets.push(running);
        running = running.saturating_add(count);
    }
    offsets
}

/// Assemble a flat `u32` indirect buffer from a slice of non-indexed draw args,
/// four words per command in `wgpu` / `WebGPU` order.
#[must_use]
pub fn pack_draw_indirect(args: &[DrawIndirectArgs]) -> Vec<u32> {
    let mut words = Vec::with_capacity(args.len().saturating_mul(4));
    for a in args {
        words.extend_from_slice(&a.as_words());
    }
    words
}

/// Assemble a flat `u32` indirect buffer from a slice of indexed draw args, five
/// words per command in `wgpu` / `WebGPU` order.
#[must_use]
pub fn pack_draw_indexed_indirect(args: &[DrawIndexedIndirectArgs]) -> Vec<u32> {
    let mut words = Vec::with_capacity(args.len().saturating_mul(5));
    for a in args {
        words.extend_from_slice(&a.as_words());
    }
    words
}

/// Assemble a flat `u32` indirect buffer from a slice of dispatch args, three
/// words per command in `wgpu` / `WebGPU` order.
#[must_use]
pub fn pack_dispatch_indirect(args: &[DispatchIndirectArgs]) -> Vec<u32> {
    let mut words = Vec::with_capacity(args.len().saturating_mul(3));
    for a in args {
        words.extend_from_slice(&a.as_words());
    }
    words
}

/// Divide-by-zero-guarded ceil-div `(n + d - 1) / d` with a saturating numerator
/// addition so the round-up never wraps near [`u32::MAX`]. A divisor of `0`
/// yields `0`.
const fn ceil_div_saturating(numerator: u32, divisor: u32) -> u32 {
    match numerator
        .saturating_add(divisor.saturating_sub(1))
        .checked_div(divisor)
    {
        Some(quotient) => quotient,
        None => 0,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sprite_args_are_instanced_quad() {
        let args = DrawIndirectArgs::sprite(1000, 0);
        assert_eq!(args.vertex_count, SPRITE_QUAD_VERTEX_COUNT);
        assert_eq!(args.instance_count, 1000);
        assert_eq!(args.first_vertex, 0);
        assert_eq!(args.first_instance, 0);
        assert_eq!(args.as_words(), [6, 1000, 0, 0]);
    }

    #[test]
    fn mesh_args_use_source_index_count() {
        let args = DrawIndexedIndirectArgs::mesh(250, 36, 7);
        assert_eq!(args.index_count, 36);
        assert_eq!(args.instance_count, 250);
        assert_eq!(args.first_index, 0);
        assert_eq!(args.base_vertex, 0);
        assert_eq!(args.first_instance, 7);
        assert_eq!(args.as_words(), [36, 250, 0, 0, 7]);
    }

    #[test]
    fn ribbon_args_derive_index_count_from_segments() {
        let args = DrawIndexedIndirectArgs::ribbon(4, 10, 0);
        // 10 segments * 6 indices = 60 indices, one instance per chain.
        assert_eq!(args.index_count, 60);
        assert_eq!(args.instance_count, 4);
    }

    #[test]
    fn beam_args_match_segment_expansion() {
        let args = DrawIndexedIndirectArgs::beam(3, 5, 2);
        assert_eq!(args.index_count, 30);
        assert_eq!(args.instance_count, 3);
        assert_eq!(args.first_instance, 2);
    }

    #[test]
    fn geometry_dispatch_matches_direct_constructors() {
        let sprite = RendererGeometry::Sprite {
            alive_count: 12,
            first_instance: 0,
        };
        assert_eq!(sprite.kind(), RendererKind::Sprite);
        assert_eq!(
            sprite.draw_args(),
            EmitterDrawArgs::NonIndexed(DrawIndirectArgs::sprite(12, 0))
        );

        let mesh = RendererGeometry::Mesh {
            alive_count: 8,
            index_count: 96,
            first_instance: 4,
        };
        assert_eq!(mesh.kind(), RendererKind::Mesh);
        assert_eq!(
            mesh.draw_args(),
            EmitterDrawArgs::Indexed(DrawIndexedIndirectArgs::mesh(8, 96, 4))
        );

        let ribbon = RendererGeometry::Ribbon {
            chain_count: 2,
            segments_per_chain: 7,
            first_instance: 0,
        };
        assert_eq!(ribbon.kind(), RendererKind::Ribbon);
        assert_eq!(
            ribbon.draw_args(),
            EmitterDrawArgs::Indexed(DrawIndexedIndirectArgs::ribbon(2, 7, 0))
        );

        let beam = RendererGeometry::Beam {
            chain_count: 5,
            segments_per_chain: 3,
            first_instance: 1,
        };
        assert_eq!(beam.kind(), RendererKind::Beam);
        assert_eq!(
            beam.draw_args(),
            EmitterDrawArgs::Indexed(DrawIndexedIndirectArgs::beam(5, 3, 1))
        );
    }

    #[test]
    fn first_instance_prefix_is_exclusive_prefix_sum() {
        assert_eq!(first_instance_prefix(&[10, 20, 5]), &[0, 10, 30]);
        assert_eq!(first_instance_prefix(&[7]), &[0]);
        assert!(first_instance_prefix(&[]).is_empty());
    }

    #[test]
    fn first_instance_prefix_saturates() {
        let offsets = first_instance_prefix(&[u32::MAX, 100, 1]);
        assert_eq!(offsets, &[0, u32::MAX, u32::MAX]);
    }

    #[test]
    fn ceil_div_boundaries() {
        // Zero numerator -> zero workgroups.
        assert_eq!(ceil_div_saturating(0, 64), 0);
        // Exact multiple.
        assert_eq!(ceil_div_saturating(256, 64), 4);
        // Remainder of one rounds up.
        assert_eq!(ceil_div_saturating(257, 64), 5);
        // Divisor of zero is guarded.
        assert_eq!(ceil_div_saturating(100, 0), 0);
    }

    #[test]
    fn ceil_div_saturates_near_max() {
        // `numerator + (divisor - 1)` would overflow; `saturating_add` caps the
        // numerator at `u32::MAX` instead of wrapping, so the round-up is lost at
        // the extreme and the result floors by one. This is safe: no real alive
        // count approaches `u32::MAX`.
        assert_eq!(ceil_div_saturating(u32::MAX, 1), u32::MAX);
        assert_eq!(ceil_div_saturating(u32::MAX, 2), u32::MAX / 2);
    }

    #[test]
    fn dispatch_linear_1d_rounds_up_and_sets_1d() {
        let d = DispatchIndirectArgs::linear_1d(257, 64, 65_535);
        assert_eq!(d.x, 5);
        assert_eq!(d.y, 1);
        assert_eq!(d.z, 1);
        assert_eq!(d.as_words(), [5, 1, 1]);
    }

    #[test]
    fn dispatch_linear_1d_clamps_to_max_workgroups() {
        // 1_000_000 / 64 = 15625, clamped to the cap.
        let d = DispatchIndirectArgs::linear_1d(1_000_000, 64, 128);
        assert_eq!(d.x, 128);

        // A zero cap clamps to zero.
        let none = DispatchIndirectArgs::linear_1d(1_000, 64, 0);
        assert_eq!(none.x, 0);

        // A zero workgroup size is guarded to a no-op launch.
        let guarded = DispatchIndirectArgs::linear_1d(1_000, 0, 128);
        assert_eq!(guarded.x, 0);
    }

    #[test]
    fn ribbon_index_count_saturates_on_overflow() {
        // segments_per_chain * 6 overflows u32 and saturates.
        let args = DrawIndexedIndirectArgs::ribbon(1, u32::MAX, 0);
        assert_eq!(args.index_count, u32::MAX);
    }

    #[test]
    fn base_vertex_word_is_raw_bits() {
        let args = DrawIndexedIndirectArgs {
            index_count: 3,
            instance_count: 1,
            first_index: 0,
            base_vertex: -1,
            first_instance: 0,
        };
        assert_eq!(args.as_words(), [3, 1, 0, u32::MAX, 0]);
    }

    #[test]
    fn pack_functions_flatten_in_buffer_order() {
        let draws = [
            DrawIndirectArgs::sprite(2, 0),
            DrawIndirectArgs::sprite(3, 2),
        ];
        assert_eq!(pack_draw_indirect(&draws), &[6, 2, 0, 0, 6, 3, 0, 2]);

        let indexed = [DrawIndexedIndirectArgs::mesh(4, 36, 0)];
        assert_eq!(pack_draw_indexed_indirect(&indexed), &[36, 4, 0, 0, 0]);

        let dispatches = [DispatchIndirectArgs::new(5, 1, 1)];
        assert_eq!(pack_dispatch_indirect(&dispatches), &[5, 1, 1]);
    }

    #[test]
    fn all_u32_args_derive_eq_and_hash() {
        use core::hash::{Hash, Hasher};

        fn hash<T: Hash>(value: &T) -> u64 {
            // A tiny FNV-style hasher to exercise the derived `Hash` impls.
            struct FnvHasher(u64);
            impl Hasher for FnvHasher {
                fn finish(&self) -> u64 {
                    self.0
                }
                fn write(&mut self, bytes: &[u8]) {
                    for &b in bytes {
                        self.0 = (self.0 ^ u64::from(b)).wrapping_mul(0x0100_0000_01b3);
                    }
                }
            }
            let mut hasher = FnvHasher(0xcbf2_9ce4_8422_2325);
            value.hash(&mut hasher);
            hasher.finish()
        }

        let a = DrawIndirectArgs::sprite(9, 0);
        let b = DrawIndirectArgs::sprite(9, 0);
        assert_eq!(a, b);
        assert_eq!(hash(&a), hash(&b));

        let g = RendererGeometry::Sprite {
            alive_count: 9,
            first_instance: 0,
        };
        assert_eq!(hash(&g), hash(&g));
    }
}
