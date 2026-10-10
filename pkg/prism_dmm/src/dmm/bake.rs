//! The baking entry points: single-triangle bake, world-space displacement,
//! and a deduplicating builder.

use alloc::collections::BTreeMap;
use alloc::vec::Vec;

use crate::dmm::encode::{pack_unorm11, unpack_unorm11};
use crate::dmm::heightmap::DisplacementMap;
use crate::dmm::quantize::DisplacementScaleBias;
use crate::dmm::subdivision::{barycentric_f32, micro_vertices, DmmSubdivisionLevel};

/// How the per-triangle [`DisplacementScaleBias`] is chosen when baking.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ScaleBiasMode {
    /// Derive the scale/bias automatically from the triangle's own sampled
    /// `[min, max]` height range (the usual choice for maximum precision).
    PerTriangle,
    /// Use a fixed, caller-provided range for every triangle (useful when many
    /// triangles must share one quantization domain).
    Fixed(DisplacementScaleBias),
}

/// Inputs describing one base triangle to bake.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DmmBakeInput {
    /// Texture coordinates of the three base-triangle corners, matching the
    /// barycentric vertex order `(0, 1, 2)`.
    pub uv: [[f32; 2]; 3],
    /// Subdivision level controlling the micro-vertex lattice density.
    pub level: DmmSubdivisionLevel,
    /// How to derive the quantization scale/bias for this triangle.
    pub scale_bias_mode: ScaleBiasMode,
}

/// A baked displacement micro-map for one base triangle.
#[derive(Debug, Clone, PartialEq)]
pub struct BakedDmm {
    level: DmmSubdivisionLevel,
    scale_bias: DisplacementScaleBias,
    codes: Vec<u16>,
    data: Vec<u8>,
}

impl BakedDmm {
    /// Returns the subdivision level used for this bake.
    #[must_use]
    pub const fn level(&self) -> DmmSubdivisionLevel {
        self.level
    }

    /// Returns the per-triangle displacement scale/bias.
    #[must_use]
    pub const fn scale_bias(&self) -> DisplacementScaleBias {
        self.scale_bias
    }

    /// Returns the per-micro-vertex `11-bit` codes in canonical order.
    #[must_use]
    pub fn codes(&self) -> &[u16] {
        &self.codes
    }

    /// Returns the raw packed `11-bit` bitstream.
    #[must_use]
    pub fn data(&self) -> &[u8] {
        &self.data
    }

    /// Number of baked micro-vertices.
    #[must_use]
    pub fn micro_vertex_count(&self) -> u32 {
        self.level.micro_vertex_count()
    }

    /// Decodes the stored codes back into per-micro-vertex heights.
    #[must_use]
    pub fn decoded_heights(&self) -> Vec<f32> {
        self.codes
            .iter()
            .map(|&code| self.scale_bias.dequantize(code))
            .collect()
    }

    /// Unpacks the raw bitstream back into the `11-bit` codes, returning
    /// [`None`] when the stored data is somehow too short.
    #[must_use]
    pub fn unpack_codes(&self) -> Option<Vec<u16>> {
        unpack_unorm11(&self.data, self.codes.len())
    }
}

/// Interpolates a base-triangle attribute by barycentric weights.
fn interp2(uv: [[f32; 2]; 3], bary: [f32; 3]) -> [f32; 2] {
    [
        uv[0][0] * bary[0] + uv[1][0] * bary[1] + uv[2][0] * bary[2],
        uv[0][1] * bary[0] + uv[1][1] * bary[1] + uv[2][1] * bary[2],
    ]
}

/// Bakes a single base triangle into a [`BakedDmm`].
///
/// The bake is two-pass: it first gathers the displacement height at every
/// micro-vertex (interpolating each micro-vertex `UV` from the base-triangle
/// corners), computes the triangle `[min, max]` range to build the
/// [`DisplacementScaleBias`], then normalizes and quantizes each height to an
/// `11-bit` code and packs the codes into the raw bitstream.
#[must_use]
pub fn bake_triangle<M: DisplacementMap + ?Sized>(input: &DmmBakeInput, map: &M) -> BakedDmm {
    let vertices = micro_vertices(input.level);

    // Pass one: gather heights at every micro-vertex.
    let mut heights = Vec::with_capacity(vertices.len());
    for &vertex in &vertices {
        let bary = barycentric_f32(input.level, vertex);
        let uv = interp2(input.uv, bary);
        heights.push(map.sample_height(uv[0], uv[1]));
    }

    // Determine the quantization scale/bias.
    let scale_bias = match input.scale_bias_mode {
        ScaleBiasMode::Fixed(sb) => sb,
        ScaleBiasMode::PerTriangle => {
            let mut min = heights[0];
            let mut max = heights[0];
            for &h in &heights[1..] {
                if h < min {
                    min = h;
                }
                if h > max {
                    max = h;
                }
            }
            DisplacementScaleBias::new(min, max)
        }
    };

    // Pass two: normalize + quantize.
    let codes: Vec<u16> = heights.iter().map(|&h| scale_bias.quantize(h)).collect();
    let data = pack_unorm11(&codes);

    BakedDmm {
        level: input.level,
        scale_bias,
        codes,
        data,
    }
}

/// Returns the world-space position of one displaced micro-vertex.
///
/// Given the three base-triangle positions, their three per-vertex
/// displacement directions, the micro-vertex barycentric weights, and the
/// decoded scalar displacement `height`, this barycentric-interpolates both
/// the base position and the displacement direction, then offsets the position
/// by `direction * height`. The direction is interpolated but not
/// renormalized, matching the usual `DMM` evaluation convention.
#[must_use]
pub fn displaced_position(
    base_positions: [[f32; 3]; 3],
    directions: [[f32; 3]; 3],
    barycentric: [f32; 3],
    height: f32,
) -> [f32; 3] {
    let mut out = [0.0f32; 3];
    for axis in 0..3 {
        let pos = base_positions[0][axis] * barycentric[0]
            + base_positions[1][axis] * barycentric[1]
            + base_positions[2][axis] * barycentric[2];
        let dir = directions[0][axis] * barycentric[0]
            + directions[1][axis] * barycentric[1]
            + directions[2][axis] * barycentric[2];
        out[axis] = pos + dir * height;
    }
    out
}

/// Output of [`DmmBuilder::finish`]: deduplicated micro-maps plus a
/// per-triangle index into them.
#[derive(Debug, Clone, PartialEq)]
pub struct DmmBuilderOutput {
    /// The unique baked micro-maps, in first-seen order.
    pub micromaps: Vec<BakedDmm>,
    /// Per source triangle, the index into
    /// [`DmmBuilderOutput::micromaps`].
    pub indices: Vec<u32>,
}

impl DmmBuilderOutput {
    /// Number of source triangles added to the builder.
    #[must_use]
    pub fn triangle_count(&self) -> usize {
        self.indices.len()
    }

    /// Number of unique micro-maps after deduplication.
    #[must_use]
    pub fn unique_count(&self) -> usize {
        self.micromaps.len()
    }
}

/// Bakes many base triangles, deduplicating byte-identical micro-maps the way
/// a real asset-build step shares one `DMM` across repeated geometry.
///
/// Two micro-maps are considered identical when they share the same
/// subdivision level and the same raw packed bytes. The scale/bias is *not*
/// part of the dedup key, so triangles with identical normalized displacement
/// but different absolute height ranges still collapse to one stored map; the
/// first-seen scale/bias is retained.
#[derive(Debug, Default)]
pub struct DmmBuilder {
    micromaps: Vec<BakedDmm>,
    indices: Vec<u32>,
    dedup: BTreeMap<(u8, Vec<u8>), u32>,
}

impl DmmBuilder {
    /// Creates an empty builder.
    #[must_use]
    pub fn new() -> Self {
        Self {
            micromaps: Vec::new(),
            indices: Vec::new(),
            dedup: BTreeMap::new(),
        }
    }

    /// Bakes one base triangle and records its deduplicated micro-map index.
    pub fn add_triangle<M: DisplacementMap + ?Sized>(&mut self, input: &DmmBakeInput, map: &M) {
        let baked = bake_triangle(input, map);
        let key = (baked.level().get(), baked.data().to_vec());
        let index = if let Some(&existing) = self.dedup.get(&key) {
            existing
        } else {
            let new_index =
                u32::try_from(self.micromaps.len()).expect("unique micro-map count exceeds u32");
            self.micromaps.push(baked);
            self.dedup.insert(key, new_index);
            new_index
        };
        self.indices.push(index);
    }

    /// Consumes the builder, returning the deduplicated micro-maps and indices.
    #[must_use]
    pub fn finish(self) -> DmmBuilderOutput {
        DmmBuilderOutput {
            micromaps: self.micromaps,
            indices: self.indices,
        }
    }
}
