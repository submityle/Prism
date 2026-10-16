//! Deterministic author-node evaluation and material-VT baking.
//!
//! This is the runtime-facing P2-b slice: a restricted procedural subgraph is
//! evaluated on the CPU into a complete RGBA8 mip chain, then returned as the
//! same immutable [`MaterialVtSourceSnapshot`] consumed by the production
//! material-VT source registry. The render path never evaluates the graph or
//! keeps its nodes; it consumes pages from the existing atlas/page-table
//! producer just like an image source.

use alloc::{collections::BTreeMap, sync::Arc, vec::Vec};
use core::fmt;

use crate::{
    MaterialGraph, MaterialNode, MaterialNodeId, MaterialValue, MaterialVtError, MaterialVtFormat,
    MaterialVtSourceDescriptor, MaterialVtSourceSnapshot,
};

/// Configuration for one procedural material-VT bake.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ProceduralVtBakeConfig {
    /// Stable texture identity used by page keys and material rows.
    pub texture_id: u32,
    /// Base-mip width in texels.
    pub width: u32,
    /// Base-mip height in texels.
    pub height: u32,
    /// Virtual page edge in texels.
    pub page_size: u32,
}

/// Maximum payload produced by the runtime procedural-bake slice.
///
/// This bound protects the render thread from accidentally treating an
/// authoring-sized graph as an unbounded runtime allocation. Larger sources
/// belong to the offline asset-build path.
pub const MAX_PROCEDURAL_VT_BAKE_BYTES: u64 = 256 * 1024 * 1024;

impl ProceduralVtBakeConfig {
    /// Creates a bake configuration for a single-layer RGBA8 source.
    #[must_use]
    pub const fn new(texture_id: u32, width: u32, height: u32, page_size: u32) -> Self {
        Self {
            texture_id,
            width,
            height,
            page_size,
        }
    }
}

/// Failure while lowering a procedural graph into a VT source snapshot.
#[derive(Clone, Debug, PartialEq)]
pub enum ProceduralVtBakeError {
    /// The caller could not allocate a stable source texture identity.
    TextureIdExhausted,
    /// The graph topology is invalid.
    Graph(crate::MaterialValidationError),
    /// The graph contains a node that is not part of the procedural bake ABI.
    UnsupportedNode(MaterialNodeId),
    /// A curve has no control points.
    EmptyCurve(MaterialNodeId),
    /// A curve's control points are not ordered by input coordinate.
    UnsortedCurve(MaterialNodeId),
    /// A curve has two control points with the same input coordinate.
    DuplicateCurveCoordinate(MaterialNodeId),
    /// A procedural input such as noise scale is not finite.
    NonFiniteInput(MaterialNodeId),
    /// The complete generated mip chain exceeds the runtime bake budget.
    BakeTooLarge { bytes: u64, max_bytes: u64 },
    /// The output value could not be represented as finite normalized bytes.
    NonFiniteOutput(MaterialNodeId),
    /// The generated source snapshot failed its shared VT contract.
    Source(MaterialVtError),
}

impl fmt::Display for ProceduralVtBakeError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::TextureIdExhausted => {
                formatter.write_str("material VT texture id space is exhausted")
            }
            Self::Graph(error) => error.fmt(formatter),
            Self::UnsupportedNode(id) => write!(formatter, "node {} is not bakeable", id.0),
            Self::EmptyCurve(id) => write!(formatter, "curve node {} has no points", id.0),
            Self::UnsortedCurve(id) => write!(formatter, "curve node {} is not sorted", id.0),
            Self::DuplicateCurveCoordinate(id) => {
                write!(
                    formatter,
                    "curve node {} has duplicate input coordinates",
                    id.0
                )
            }
            Self::NonFiniteInput(id) => write!(formatter, "node {} has a non-finite input", id.0),
            Self::BakeTooLarge { bytes, max_bytes } => write!(
                formatter,
                "procedural VT bake payload {bytes} bytes exceeds runtime limit {max_bytes} bytes"
            ),
            Self::NonFiniteOutput(id) => {
                write!(formatter, "node {} produced a non-finite value", id.0)
            }
            Self::Source(error) => error.fmt(formatter),
        }
    }
}

impl std::error::Error for ProceduralVtBakeError {}

/// Bakes a restricted author graph into the immutable source contract used by
/// the production material-VT producer.
pub fn bake_procedural_vt(
    graph: &MaterialGraph,
    config: ProceduralVtBakeConfig,
) -> Result<MaterialVtSourceSnapshot, ProceduralVtBakeError> {
    if config.width == 0 {
        return Err(ProceduralVtBakeError::Source(
            MaterialVtError::ZeroDimension,
        ));
    }
    if config.height == 0 {
        return Err(ProceduralVtBakeError::Source(
            MaterialVtError::ZeroDimension,
        ));
    }
    if config.page_size == 0 {
        return Err(ProceduralVtBakeError::Source(MaterialVtError::ZeroPageSize));
    }
    crate::validate_graph(graph).map_err(ProceduralVtBakeError::Graph)?;
    let total_bytes = generated_payload_bytes(config.width, config.height).ok_or(
        ProceduralVtBakeError::Source(MaterialVtError::PayloadSizeOverflow),
    )?;
    if total_bytes > MAX_PROCEDURAL_VT_BAKE_BYTES {
        return Err(ProceduralVtBakeError::BakeTooLarge {
            bytes: total_bytes,
            max_bytes: MAX_PROCEDURAL_VT_BAKE_BYTES,
        });
    }
    let output = graph.output.expect("validated graph has an output");
    let mut levels = Vec::new();
    let mut width = config.width;
    let mut height = config.height;
    let level_bytes = usize::try_from(
        u64::from(width)
            .checked_mul(u64::from(height))
            .and_then(|value| value.checked_mul(4))
            .ok_or(ProceduralVtBakeError::Source(
                MaterialVtError::PayloadSizeOverflow,
            ))?,
    )
    .map_err(|_| ProceduralVtBakeError::Source(MaterialVtError::PayloadSizeOverflow))?;
    let mut level = Vec::with_capacity(level_bytes);
    for y in 0..height {
        for x in 0..width {
            let uv = [
                (x as f32 + 0.5) / config.width as f32,
                (y as f32 + 0.5) / config.height as f32,
            ];
            let mut memo = BTreeMap::new();
            let value = evaluate(graph, output, uv, &mut memo)?;
            let rgba = to_rgba8(value, output)?;
            level.extend_from_slice(&rgba);
        }
    }
    levels.push(level);
    while width > 1 || height > 1 {
        let next_width = width.div_ceil(2).max(1);
        let next_height = height.div_ceil(2).max(1);
        let previous = levels.last().expect("level was pushed");
        levels.push(downsample_rgba8(
            previous,
            width,
            height,
            next_width,
            next_height,
        ));
        width = next_width;
        height = next_height;
    }

    let mip_count = levels.len() as u8;
    let descriptor = MaterialVtSourceDescriptor::new(
        config.texture_id,
        config.width,
        config.height,
        1,
        config.page_size,
        mip_count,
        MaterialVtFormat::Rgba8Unorm,
    )
    .map_err(ProceduralVtBakeError::Source)?;
    let bytes = levels.into_iter().flatten().collect::<Vec<_>>();
    MaterialVtSourceSnapshot::new(descriptor, Arc::<[u8]>::from(bytes))
        .map_err(ProceduralVtBakeError::Source)
}

fn evaluate(
    graph: &MaterialGraph,
    id: MaterialNodeId,
    uv: [f32; 2],
    memo: &mut BTreeMap<MaterialNodeId, [f32; 4]>,
) -> Result<[f32; 4], ProceduralVtBakeError> {
    if let Some(value) = memo.get(&id) {
        return Ok(*value);
    }
    let node = graph
        .nodes
        .get(&id)
        .ok_or(ProceduralVtBakeError::UnsupportedNode(id))?;
    let value = match node {
        MaterialNode::Constant(value) | MaterialNode::Parameter { default: value, .. } => {
            value_to_rgba(*value)
        }
        MaterialNode::Uv => [uv[0], uv[1], 0.0, 1.0],
        MaterialNode::Noise { scale, seed } => {
            if !scale.is_finite() {
                return Err(ProceduralVtBakeError::NonFiniteInput(id));
            }
            let x = (uv[0] * *scale).floor() as i32;
            let y = (uv[1] * *scale).floor() as i32;
            let hash = hash2(x, y, *seed);
            [hash, hash, hash, 1.0]
        }
        MaterialNode::Curve { input, points } => {
            if points.is_empty() {
                return Err(ProceduralVtBakeError::EmptyCurve(id));
            }
            if points.iter().flatten().any(|value| !value.is_finite()) {
                return Err(ProceduralVtBakeError::NonFiniteInput(id));
            }
            if points.windows(2).any(|pair| pair[0][0] > pair[1][0]) {
                return Err(ProceduralVtBakeError::UnsortedCurve(id));
            }
            if points.windows(2).any(|pair| pair[0][0] == pair[1][0]) {
                return Err(ProceduralVtBakeError::DuplicateCurveCoordinate(id));
            }
            let input = evaluate(graph, *input, uv, memo)?[0];
            let value = curve_value(points, input);
            [value, value, value, 1.0]
        }
        MaterialNode::Add(a, b) => zip(
            evaluate(graph, *a, uv, memo)?,
            evaluate(graph, *b, uv, memo)?,
            |left, right| left + right,
        ),
        MaterialNode::Multiply(a, b) => zip(
            evaluate(graph, *a, uv, memo)?,
            evaluate(graph, *b, uv, memo)?,
            |left, right| left * right,
        ),
        MaterialNode::Blend { a, b, weight } => {
            let left = evaluate(graph, *a, uv, memo)?;
            let right = evaluate(graph, *b, uv, memo)?;
            let factor = evaluate(graph, *weight, uv, memo)?[0].clamp(0.0, 1.0);
            zip(left, right, |l, r| l + (r - l) * factor)
        }
        MaterialNode::TextureSample { .. } | MaterialNode::Closure { .. } => {
            return Err(ProceduralVtBakeError::UnsupportedNode(id));
        }
    };
    memo.insert(id, value);
    Ok(value)
}

fn value_to_rgba(value: MaterialValue) -> [f32; 4] {
    match value {
        MaterialValue::Scalar(value) => [value; 4],
        MaterialValue::Vector(value) => value,
        MaterialValue::Boolean(value) => [if value { 1.0 } else { 0.0 }; 4],
    }
}

fn zip(left: [f32; 4], right: [f32; 4], f: impl Fn(f32, f32) -> f32) -> [f32; 4] {
    [
        f(left[0], right[0]),
        f(left[1], right[1]),
        f(left[2], right[2]),
        f(left[3], right[3]),
    ]
}

fn curve_value(points: &[[f32; 2]], input: f32) -> f32 {
    if input <= points[0][0] {
        return points[0][1];
    }
    for pair in points.windows(2) {
        if input <= pair[1][0] {
            let span = pair[1][0] - pair[0][0];
            let t = if span > 0.0 {
                (input - pair[0][0]) / span
            } else {
                0.0
            };
            return pair[0][1] + (pair[1][1] - pair[0][1]) * t;
        }
    }
    points.last().expect("non-empty curve")[1]
}

fn hash2(x: i32, y: i32, seed: u32) -> f32 {
    let mut value = (x as u32).wrapping_mul(0x45d9f3b)
        ^ (y as u32).wrapping_mul(0x119de1f3)
        ^ seed.wrapping_mul(0x3449_5b9f);
    value ^= value >> 16;
    value = value.wrapping_mul(0x7feb_352d);
    value ^= value >> 15;
    value = value.wrapping_mul(0x846c_a68b);
    value ^= value >> 16;
    value as f32 / u32::MAX as f32
}

fn to_rgba8(value: [f32; 4], id: MaterialNodeId) -> Result<[u8; 4], ProceduralVtBakeError> {
    if value.iter().any(|channel| !channel.is_finite()) {
        return Err(ProceduralVtBakeError::NonFiniteOutput(id));
    }
    Ok(value.map(|channel| (channel.clamp(0.0, 1.0) * 255.0).round() as u8))
}

fn downsample_rgba8(
    source: &[u8],
    width: u32,
    height: u32,
    next_width: u32,
    next_height: u32,
) -> Vec<u8> {
    let mut output = Vec::with_capacity(next_width as usize * next_height as usize * 4);
    for y in 0..next_height {
        for x in 0..next_width {
            let mut sum = [0_u32; 4];
            for oy in 0..2 {
                for ox in 0..2 {
                    let sx = (x * 2 + ox).min(width - 1);
                    let sy = (y * 2 + oy).min(height - 1);
                    let offset = ((sy * width + sx) * 4) as usize;
                    for channel in 0..4 {
                        sum[channel] += u32::from(source[offset + channel]);
                    }
                }
            }
            output.extend(sum.map(|value| (value / 4) as u8));
        }
    }
    output
}

fn generated_payload_bytes(mut width: u32, mut height: u32) -> Option<u64> {
    let mut total = 0_u64;
    loop {
        let level = u64::from(width)
            .checked_mul(u64::from(height))?
            .checked_mul(4)?;
        total = total.checked_add(level)?;
        if width == 1 && height == 1 {
            return Some(total);
        }
        width = width.div_ceil(2).max(1);
        height = height.div_ceil(2).max(1);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::collections::BTreeMap;

    #[test]
    fn procedural_graph_bakes_noise_curve_and_blend_into_complete_vt() {
        let uv = MaterialNodeId(1);
        let noise = MaterialNodeId(2);
        let curve = MaterialNodeId(3);
        let white = MaterialNodeId(4);
        let output = MaterialNodeId(5);
        let graph = MaterialGraph {
            nodes: BTreeMap::from([
                (uv, MaterialNode::Uv),
                (
                    noise,
                    MaterialNode::Noise {
                        scale: 4.0,
                        seed: 7,
                    },
                ),
                (
                    curve,
                    MaterialNode::Curve {
                        input: noise,
                        points: vec![[0.0, 0.1], [1.0, 0.9]],
                    },
                ),
                (
                    white,
                    MaterialNode::Constant(MaterialValue::Vector([1.0; 4])),
                ),
                (
                    output,
                    MaterialNode::Blend {
                        a: curve,
                        b: white,
                        weight: uv,
                    },
                ),
            ]),
            output: Some(output),
            plan: None,
        };
        let snapshot = bake_procedural_vt(&graph, ProceduralVtBakeConfig::new(7, 4, 2, 2))
            .expect("procedural graph should bake");
        assert_eq!(snapshot.descriptor().format(), MaterialVtFormat::Rgba8Unorm);
        assert_eq!(snapshot.descriptor().mip_count(), 3);
        assert_eq!(snapshot.page_count(), 4);
        assert!(snapshot.bytes().iter().any(|value| *value != 0));
    }

    #[test]
    fn procedural_bake_rejects_runtime_texture_and_closure_nodes() {
        let uv = MaterialNodeId(2);
        let graph = MaterialGraph {
            nodes: BTreeMap::from([
                (
                    MaterialNodeId(1),
                    MaterialNode::TextureSample {
                        texture_slot: 0,
                        uv,
                    },
                ),
                (uv, MaterialNode::Uv),
            ]),
            output: Some(MaterialNodeId(1)),
            plan: None,
        };
        assert!(matches!(
            bake_procedural_vt(&graph, ProceduralVtBakeConfig::new(1, 2, 2, 2)),
            Err(ProceduralVtBakeError::UnsupportedNode(MaterialNodeId(1)))
        ));
    }

    #[test]
    fn procedural_bake_rejects_ambiguous_curve_and_non_finite_noise() {
        let curve = MaterialNodeId(1);
        let curve_graph = MaterialGraph {
            nodes: BTreeMap::from([
                (
                    curve,
                    MaterialNode::Curve {
                        input: MaterialNodeId(2),
                        points: vec![[0.0, 0.0], [0.0, 1.0]],
                    },
                ),
                (MaterialNodeId(2), MaterialNode::Uv),
            ]),
            output: Some(curve),
            plan: None,
        };
        assert!(matches!(
            bake_procedural_vt(&curve_graph, ProceduralVtBakeConfig::new(1, 2, 2, 2)),
            Err(ProceduralVtBakeError::DuplicateCurveCoordinate(_))
        ));

        let noise = MaterialNodeId(3);
        let noise_graph = MaterialGraph {
            nodes: BTreeMap::from([(
                noise,
                MaterialNode::Noise {
                    scale: f32::NAN,
                    seed: 0,
                },
            )]),
            output: Some(noise),
            plan: None,
        };
        assert!(matches!(
            bake_procedural_vt(&noise_graph, ProceduralVtBakeConfig::new(1, 2, 2, 2)),
            Err(ProceduralVtBakeError::NonFiniteInput(_))
        ));
    }

    #[test]
    fn procedural_bake_rejects_payloads_over_runtime_budget() {
        let graph = MaterialGraph {
            nodes: BTreeMap::from([(
                MaterialNodeId(1),
                MaterialNode::Constant(MaterialValue::Scalar(1.0)),
            )]),
            output: Some(MaterialNodeId(1)),
            plan: None,
        };
        assert!(matches!(
            bake_procedural_vt(&graph, ProceduralVtBakeConfig::new(1, 16_384, 16_384, 128)),
            Err(ProceduralVtBakeError::BakeTooLarge { .. })
        ));
    }
}
