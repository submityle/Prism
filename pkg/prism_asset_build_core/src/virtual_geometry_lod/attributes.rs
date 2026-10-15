//! Semantic policies and exact source-vertex provenance for static LODs.

use alloc::collections::BTreeMap;
use prism_mesh::{Mesh, VertexAttributeValues, POSITION_ATTRIBUTE};
use prism_render_architecture::particle::half_float_f16::f16_bits_to_f32;
use prism_render_architecture::ray_scene::{DecimationChannel, VertexDecimationConstraints};

use super::StaticVirtualGeometryLodError as Error;

/// How an authored attribute influences collapse ordering. Every mode retains
/// the selected endpoint's original bytes; no mode performs reconstruction.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum LodAttributeMetric {
    /// Squared lane differences weighted in object-space squared units.
    /// This is an ordering heuristic, not an attribute-error bound.
    Continuous {
        /// Nonnegative finite squared-distance multiplier.
        weight: f64,
    },
    /// Only equal raw attribute values can collapse into each other.
    Categorical,
    /// Preserve values, without adding an attribute-detail penalty.
    PreserveOnly,
}

/// Explicit policy for a named semantic. Position is always geometric QEM and
/// cannot be overridden. Missing policies use continuous weight 1 for float,
/// half and normalized formats, and categorical equality for integer formats.
#[derive(Clone, Debug, PartialEq)]
pub struct LodAttributePolicy {
    /// Exact authored semantic name.
    pub name: &'static str,
    /// Collapse rule for this semantic.
    pub metric: LodAttributeMetric,
}

/// Extensible static-LOD policy. Open chart/material edges, coincident split
/// vertices and nonmanifold edges are always pinned. Tangent handedness is
/// categorical even when its direction uses a continuous metric.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct StaticLodPolicy {
    /// Optional semantic overrides; order does not affect artifact identity.
    pub attributes: Vec<LodAttributePolicy>,
}

pub(super) fn resolve(
    mesh: &Mesh,
    policy: &StaticLodPolicy,
) -> Result<BTreeMap<&'static str, LodAttributeMetric>, Error> {
    let mut resolved = BTreeMap::new();
    for (attribute, values) in mesh
        .attributes()
        .filter(|(a, _)| a.name != POSITION_ATTRIBUTE)
    {
        let metric = match values {
            VertexAttributeValues::Uint32(_)
            | VertexAttributeValues::Uint32x2(_)
            | VertexAttributeValues::Uint32x4(_)
            | VertexAttributeValues::Sint32(_) => LodAttributeMetric::Categorical,
            _ => LodAttributeMetric::Continuous { weight: 1.0 },
        };
        resolved.insert(attribute.name, metric);
    }
    let mut seen = alloc::collections::BTreeSet::new();
    for override_ in &policy.attributes {
        if !resolved.contains_key(override_.name) || !seen.insert(override_.name) {
            return Err(Error::InvalidPolicy(override_.name));
        }
        if let LodAttributeMetric::Continuous { weight } = override_.metric {
            if !weight.is_finite() || weight < 0.0 || weight > f64::from(f32::MAX).powi(2) {
                return Err(Error::InvalidPolicy(override_.name));
            }
        }
        // Canonicalize signed zero so equivalent policies have one identity.
        let metric = match override_.metric {
            LodAttributeMetric::Continuous { weight } if weight == 0.0 => {
                LodAttributeMetric::Continuous { weight: 0.0 }
            }
            metric => metric,
        };
        resolved.insert(override_.name, metric);
    }
    Ok(resolved)
}

pub(super) fn constraints(
    mesh: &Mesh,
    policies: &BTreeMap<&'static str, LodAttributeMetric>,
) -> Result<VertexDecimationConstraints, Error> {
    let positions = super::positions(mesh)?;
    let mut duplicates = BTreeMap::new();
    let mut locked_vertices = vec![false; positions.len()];
    for (vertex, position) in positions.iter().enumerate() {
        // Signed zero denotes the same point. Pin coincident copies even when
        // only one chart uses them: independent partitions cannot open cracks.
        let key = position.map(|v| if v == 0.0 { 0 } else { v.to_bits() });
        if let Some(&first) = duplicates.get(&key) {
            locked_vertices[first] = true;
            locked_vertices[vertex] = true;
        } else {
            duplicates.insert(key, vertex);
        }
    }
    let mut categorical = Vec::new();
    let mut channels = Vec::new();
    let indices: Vec<_> = (0..positions.len() as u32).collect();
    for (attribute, values) in mesh
        .attributes()
        .filter(|(a, _)| a.name != POSITION_ATTRIBUTE)
    {
        let metric = policies[attribute.name];
        let tangent = attribute.name == "Vertex_Tangent"
            && matches!(
                values,
                VertexAttributeValues::Float32x4(_)
                    | VertexAttributeValues::Float16x4(_)
                    | VertexAttributeValues::Unorm8x4(_)
                    | VertexAttributeValues::Snorm8x4(_)
            );
        if metric == LodAttributeMetric::Categorical {
            categorical.push((
                values
                    .gather_bytes(&indices)
                    .ok_or(Error::InvalidPolicy(attribute.name))?,
                values.format().size() as usize,
            ));
        } else if tangent || matches!(metric, LodAttributeMetric::Continuous { .. }) {
            let (decoded, components) = numeric_values(values);
            if tangent {
                categorical.push((decoded.iter().map(|v| u8::from(v[3] < 0.0)).collect(), 1));
            }
            if let LodAttributeMetric::Continuous { weight } = metric {
                channels.push(DecimationChannel {
                    values: decoded,
                    components: if tangent { 3 } else { components },
                    weight,
                });
            }
        }
    }
    let discrete_groups = if categorical.is_empty() {
        Vec::new()
    } else {
        let key_size = categorical.iter().map(|(_, stride)| *stride).sum();
        let mut groups = BTreeMap::<Vec<u8>, u32>::new();
        let mut ids = Vec::with_capacity(positions.len());
        let mut key = Vec::with_capacity(key_size);
        for vertex in 0..positions.len() {
            key.clear();
            for (bytes, stride) in &categorical {
                key.extend_from_slice(&bytes[vertex * stride..(vertex + 1) * stride]);
            }
            let next = groups.len() as u32;
            let group = if let Some(group) = groups.get(&key) {
                *group
            } else {
                groups.insert(key.clone(), next);
                next
            };
            ids.push(group);
        }
        ids
    };
    Ok(VertexDecimationConstraints {
        locked_vertices,
        discrete_groups,
        channels,
        preserve_boundary: true,
    })
}

fn numeric_values(values: &VertexAttributeValues) -> (Vec<[f64; 4]>, u8) {
    macro_rules! vectors {
        ($values:expr, $lanes:expr, $convert:expr) => {{
            let output = $values
                .iter()
                .map(|v| {
                    let mut output = [0.0; 4];
                    for lane in 0..$lanes {
                        output[lane] = ($convert)(v[lane]);
                    }
                    output
                })
                .collect();
            (output, $lanes as u8)
        }};
    }
    match values {
        VertexAttributeValues::Float32(v) => (
            v.iter().map(|v| [f64::from(*v), 0.0, 0.0, 0.0]).collect(),
            1,
        ),
        VertexAttributeValues::Float32x2(v) => vectors!(v, 2, f64::from),
        VertexAttributeValues::Float32x3(v) => vectors!(v, 3, f64::from),
        VertexAttributeValues::Float32x4(v) => vectors!(v, 4, f64::from),
        VertexAttributeValues::Float16x2(v) => {
            vectors!(v, 2, |bits| f64::from(f16_bits_to_f32(bits)))
        }
        VertexAttributeValues::Float16x4(v) => {
            vectors!(v, 4, |bits| f64::from(f16_bits_to_f32(bits)))
        }
        VertexAttributeValues::Uint32(v) => (
            v.iter().map(|v| [f64::from(*v), 0.0, 0.0, 0.0]).collect(),
            1,
        ),
        VertexAttributeValues::Uint32x2(v) => vectors!(v, 2, f64::from),
        VertexAttributeValues::Uint32x4(v) => vectors!(v, 4, f64::from),
        VertexAttributeValues::Sint32(v) => (
            v.iter().map(|v| [f64::from(*v), 0.0, 0.0, 0.0]).collect(),
            1,
        ),
        VertexAttributeValues::Unorm8x4(v) => vectors!(v, 4, |v| f64::from(v) / 255.0),
        VertexAttributeValues::Snorm8x4(v) => vectors!(v, 4, |v| (f64::from(v) / 127.0).max(-1.0)),
    }
}

pub(super) fn hash_policies(
    hasher: &mut blake3::Hasher,
    policies: &BTreeMap<&'static str, LodAttributeMetric>,
) {
    hasher.update(&(policies.len() as u64).to_le_bytes());
    for (name, metric) in policies {
        hasher.update(&(name.len() as u64).to_le_bytes());
        hasher.update(name.as_bytes());
        match metric {
            LodAttributeMetric::Continuous { weight } => {
                hasher.update(&[0]);
                hasher.update(&weight.to_le_bytes());
            }
            LodAttributeMetric::Categorical => {
                hasher.update(&[1]);
            }
            LodAttributeMetric::PreserveOnly => {
                hasher.update(&[2]);
            }
        }
    }
}
