//! The baking entry points: single-triangle bake and a deduplicating builder.

use alloc::collections::BTreeMap;
use alloc::vec::Vec;

use crate::omm::classify::{classify_micro_triangle, SampleStrategy};
use crate::omm::mask::AlphaMask;
use crate::omm::pack::pack;
use crate::omm::state::{OmmFormat, OpacityState};
use crate::omm::subdivision::{micro_triangles, SubdivisionLevel};

/// Inputs describing one base triangle to bake.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct OmmBakeInput {
    /// Texture coordinates of the base triangle's three corners.
    pub uv: [[f32; 2]; 3],
    /// Subdivision level controlling micro-triangle density.
    pub level: SubdivisionLevel,
    /// Output encoding width.
    pub format: OmmFormat,
    /// Coverage sampling strategy.
    pub strategy: SampleStrategy,
}

/// Aggregate classification counts for a baked micromap.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct OmmStats {
    /// Micro-triangles classified fully opaque.
    pub opaque: u32,
    /// Micro-triangles classified fully transparent.
    pub transparent: u32,
    /// Micro-triangles with mixed coverage (`Unknown`).
    pub unknown: u32,
}

impl OmmStats {
    /// Total micro-triangles counted.
    #[must_use]
    pub const fn total(&self) -> u32 {
        self.opaque + self.transparent + self.unknown
    }

    /// Fraction of micro-triangles that remain `Unknown` (any-hit cost), in
    /// `0..=1`. Returns `0.0` for an empty micromap.
    #[must_use]
    pub fn unknown_ratio(&self) -> f32 {
        let total = self.total();
        if total == 0 {
            0.0
        } else {
            self.unknown as f32 / total as f32
        }
    }

    fn observe(&mut self, state: OpacityState) {
        match state {
            OpacityState::Opaque => self.opaque += 1,
            OpacityState::Transparent => self.transparent += 1,
            OpacityState::UnknownOpaque | OpacityState::UnknownTransparent => self.unknown += 1,
        }
    }
}

/// A baked micromap for a single base triangle.
#[derive(Debug, Clone, PartialEq)]
pub struct BakedOmm {
    format: OmmFormat,
    level: SubdivisionLevel,
    states: Vec<OpacityState>,
    data: Vec<u8>,
    stats: OmmStats,
}

impl BakedOmm {
    /// The output encoding width.
    #[must_use]
    pub const fn format(&self) -> OmmFormat {
        self.format
    }

    /// The subdivision level.
    #[must_use]
    pub const fn level(&self) -> SubdivisionLevel {
        self.level
    }

    /// Number of micro-triangles (`4^level`).
    #[must_use]
    pub fn micro_triangle_count(&self) -> u32 {
        u32::try_from(self.states.len()).expect("micro-triangle count exceeds u32")
    }

    /// The per-micro-triangle states in canonical order, already normalised to
    /// the output format.
    #[must_use]
    pub fn states(&self) -> &[OpacityState] {
        &self.states
    }

    /// The packed `DXR`-layout micromap bytes.
    #[must_use]
    pub fn data(&self) -> &[u8] {
        &self.data
    }

    /// The aggregate classification counts.
    #[must_use]
    pub const fn stats(&self) -> OmmStats {
        self.stats
    }
}

/// Interpolates a base-triangle `UV` from integer barycentric lattice weights.
#[inline]
fn lattice_uv(base: &[[f32; 2]; 3], w: [u32; 3], inv_n: f32) -> [f32; 2] {
    let b0 = w[0] as f32 * inv_n;
    let b1 = w[1] as f32 * inv_n;
    let b2 = w[2] as f32 * inv_n;
    [
        base[0][0] * b0 + base[1][0] * b1 + base[2][0] * b2,
        base[0][1] * b0 + base[1][1] * b1 + base[2][1] * b2,
    ]
}

/// Bakes a single base triangle into a [`BakedOmm`].
#[must_use]
pub fn bake_triangle<M: AlphaMask + ?Sized>(input: &OmmBakeInput, mask: &M) -> BakedOmm {
    let level = input.level;
    let inv_n = 1.0 / level.segments() as f32;
    let micros = micro_triangles(level);

    let mut states = Vec::with_capacity(micros.len());
    let mut stats = OmmStats::default();
    for micro in &micros {
        let micro_uv = [
            lattice_uv(&input.uv, micro.vertices[0], inv_n),
            lattice_uv(&input.uv, micro.vertices[1], inv_n),
            lattice_uv(&input.uv, micro.vertices[2], inv_n),
        ];
        let raw = classify_micro_triangle(mask, &micro_uv, input.strategy);
        let state = input.format.normalize(raw);
        stats.observe(state);
        states.push(state);
    }

    let data = pack(&states, input.format);
    BakedOmm {
        format: input.format,
        level,
        states,
        data,
        stats,
    }
}

/// Output of [`OmmBuilder::finish`]: deduplicated micromaps plus a per-triangle
/// index into them.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct OmmBuilderOutput {
    /// Unique baked micromaps, in first-seen order.
    pub micromaps: Vec<BakedOmm>,
    /// For each submitted triangle, the index of its micromap in
    /// [`OmmBuilderOutput::micromaps`].
    pub indices: Vec<u32>,
}

impl OmmBuilderOutput {
    /// Number of base triangles that were submitted.
    #[must_use]
    pub fn triangle_count(&self) -> usize {
        self.indices.len()
    }

    /// Number of unique micromaps after deduplication.
    #[must_use]
    pub fn unique_count(&self) -> usize {
        self.micromaps.len()
    }
}

/// Accumulates many base triangles sharing a format and sampling strategy,
/// deduplicating byte-identical micromaps across triangles the way a
/// production `OMM` build does.
#[derive(Debug, Clone)]
pub struct OmmBuilder {
    format: OmmFormat,
    strategy: SampleStrategy,
    micromaps: Vec<BakedOmm>,
    indices: Vec<u32>,
    dedup: BTreeMap<(u8, Vec<u8>), u32>,
}

impl OmmBuilder {
    /// Creates a builder with the given output `format` and sampling
    /// `strategy`.
    #[must_use]
    pub fn new(format: OmmFormat, strategy: SampleStrategy) -> Self {
        Self {
            format,
            strategy,
            micromaps: Vec::new(),
            indices: Vec::new(),
            dedup: BTreeMap::new(),
        }
    }

    /// Bakes one base triangle and records its deduplicated micromap index.
    ///
    /// Returns the index of the (possibly shared) micromap assigned to this
    /// triangle.
    pub fn add_triangle<M: AlphaMask + ?Sized>(
        &mut self,
        uv: [[f32; 2]; 3],
        level: SubdivisionLevel,
        mask: &M,
    ) -> u32 {
        let baked = bake_triangle(
            &OmmBakeInput {
                uv,
                level,
                format: self.format,
                strategy: self.strategy,
            },
            mask,
        );
        let key = (level.get(), baked.data().to_vec());
        let index = if let Some(&existing) = self.dedup.get(&key) {
            existing
        } else {
            let new_index = u32::try_from(self.micromaps.len()).expect("micromap count exceeds u32");
            self.micromaps.push(baked);
            self.dedup.insert(key, new_index);
            new_index
        };
        self.indices.push(index);
        index
    }

    /// Consumes the builder, returning the deduplicated micromaps and indices.
    #[must_use]
    pub fn finish(self) -> OmmBuilderOutput {
        OmmBuilderOutput {
            micromaps: self.micromaps,
            indices: self.indices,
        }
    }
}
