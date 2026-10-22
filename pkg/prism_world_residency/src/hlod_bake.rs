use alloc::collections::BTreeSet;
use alloc::{vec, vec::Vec};
use core::fmt;

use crate::hlod::{
    HlodBounds, HlodCluster, HlodLevel, HlodMaterialParams, HlodProxyArtifact, HlodProxyGeometry,
    HlodProxyMaterial, HlodProxyVertex, HlodRepresentation,
};

const CLUSTER_BASE_BYTES: u64 = 32;
const MEMBER_BYTES: u64 = 8;
const REPRESENTATION_BYTES: u64 = 48;
const PROXY_VERTEX_COUNT: usize = 8;
const PROXY_INDEX_COUNT: usize = 36;
const PROXY_VERTEX_BYTES: u64 = 32;
const PROXY_INDEX_BYTES: u64 = 4;
const PROXY_BOUNDS_BYTES: u64 = 28;
const PROXY_GEOMETRY_HASH_BYTES: u64 = 8;
const PROXY_MATERIAL_KEY_BYTES: u64 = 8;
const PROXY_MATERIAL_PARAMS_BYTES: u64 = 24;
const PROXY_MATERIAL_HASH_BYTES: u64 = 8;
const PROXY_ARTIFACT_HEADER_BYTES: u64 = 24;

/// Engine-neutral source record consumed by the HLOD declaration baker.
///
/// The baker only reads identity and bounds metadata. It does not create mesh
/// data, renderer resources, or streaming requests. `source_hash` identifies
/// the source asset/content revision without prescribing how that content is
/// stored by the caller.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct HlodProxyRecord {
    /// Stable source-proxy identity. IDs must be unique within one bake.
    pub id: u64,
    /// Source-proxy center in the caller's world units.
    pub center: [f32; 3],
    /// Non-negative source-proxy bounding radius.
    pub radius: f32,
    /// Stable source-content revision hash.
    pub source_hash: u64,
}

/// Limits applied before any HLOD declarations are materialized.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct HlodBakeBudget {
    /// Maximum number of source proxy records accepted.
    pub max_input_proxies: u32,
    /// Maximum number of output clusters.
    pub max_clusters: u32,
    /// Maximum number of output representations.
    pub max_representations: u32,
    /// Maximum estimated bytes for declarations and proxy artifacts together.
    pub max_declaration_bytes: u64,
}

impl HlodBakeBudget {
    /// Creates a declaration budget.
    pub const fn new(
        max_input_proxies: u32,
        max_clusters: u32,
        max_representations: u32,
        max_declaration_bytes: u64,
    ) -> Self {
        Self {
            max_input_proxies,
            max_clusters,
            max_representations,
            max_declaration_bytes,
        }
    }
}

/// Deterministic HLOD declaration policy.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct HlodBakeConfig {
    /// Number of representations emitted for each cluster.
    pub levels: u32,
    /// Number of sorted source proxies grouped into one cluster.
    pub proxies_per_cluster: u32,
    /// Distance increment between adjacent representations.
    pub distance_step: u32,
    /// Half-width of every representation transition band.
    pub transition_width: u32,
    /// Seed mixed into all generated identities.
    pub identity_seed: u64,
    /// Hard output limits.
    pub budget: HlodBakeBudget,
}

impl HlodBakeConfig {
    /// Creates a validated-at-bake configuration.
    pub const fn new(
        levels: u32,
        proxies_per_cluster: u32,
        distance_step: u32,
        transition_width: u32,
        identity_seed: u64,
        budget: HlodBakeBudget,
    ) -> Self {
        Self {
            levels,
            proxies_per_cluster,
            distance_step,
            transition_width,
            identity_seed,
            budget,
        }
    }
}

/// Errors returned by the deterministic HLOD declaration baker.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum HlodBakeError {
    /// The bake configuration cannot produce a valid declaration.
    InvalidConfig,
    /// An input proxy contains a NaN or infinity.
    NonFiniteProxy {
        /// Stable ID of the invalid source proxy.
        id: u64,
    },
    /// A proxy radius is negative.
    NegativeRadius {
        /// Stable ID of the invalid source proxy.
        id: u64,
    },
    /// Two input records use the same stable proxy ID.
    DuplicateProxyId(u64),
    /// An input or output count exceeds its configured limit.
    CountBudgetExceeded {
        /// Count category that exceeded its limit.
        kind: HlodBudgetKind,
        /// Count required by the requested bake.
        required: u64,
        /// Configured maximum count.
        limit: u64,
    },
    /// The estimated declaration bytes exceed the configured limit.
    DeclarationBytesBudgetExceeded {
        /// Estimated bytes required by the requested bake.
        required: u64,
        /// Configured maximum declaration bytes.
        limit: u64,
    },
    /// A generated identity collided with an earlier generated identity.
    GeneratedIdCollision(u64),
    /// A checked size calculation overflowed.
    SizeOverflow,
    /// Source bounds cannot be represented by finite `f32` proxy geometry.
    ProxyGeometryOverflow {
        /// Stable index of the cluster whose bounds overflowed.
        cluster_index: u32,
    },
}

/// Count category reported by [`HlodBakeError::CountBudgetExceeded`].
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum HlodBudgetKind {
    /// Source input record count.
    InputProxies,
    /// Output cluster count.
    Clusters,
    /// Output representation count.
    Representations,
}

impl fmt::Display for HlodBakeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidConfig => f.write_str("invalid HLOD bake configuration"),
            Self::NonFiniteProxy { id } => write!(f, "HLOD proxy {id} contains a non-finite value"),
            Self::NegativeRadius { id } => write!(f, "HLOD proxy {id} has a negative radius"),
            Self::DuplicateProxyId(id) => write!(f, "duplicate HLOD proxy ID {id}"),
            Self::CountBudgetExceeded {
                kind,
                required,
                limit,
            } => write!(
                f,
                "HLOD {kind:?} budget exceeded: required {required}, limit {limit}"
            ),
            Self::DeclarationBytesBudgetExceeded { required, limit } => write!(
                f,
                "HLOD declaration byte budget exceeded: required {required}, limit {limit}"
            ),
            Self::GeneratedIdCollision(id) => write!(f, "generated HLOD ID collision {id}"),
            Self::SizeOverflow => f.write_str("HLOD baked output size overflow"),
            Self::ProxyGeometryOverflow { cluster_index } => write!(
                f,
                "HLOD proxy geometry for cluster {cluster_index} cannot be represented finitely"
            ),
        }
    }
}

/// Baked, engine-neutral HLOD declarations and concrete proxy artifacts.
#[derive(Clone, Debug, PartialEq)]
pub struct HlodBakeOutput {
    input_hash: u64,
    declaration_hash: u64,
    artifact_hash: u64,
    estimated_output_bytes: u64,
    clusters: Vec<HlodCluster>,
    representations: Vec<HlodRepresentation>,
    artifacts: Vec<HlodProxyArtifact>,
}

impl HlodBakeOutput {
    /// Returns the hash of the canonical sorted input records.
    pub const fn input_hash(&self) -> u64 {
        self.input_hash
    }

    /// Returns the hash of the complete declaration set.
    pub const fn declaration_hash(&self) -> u64 {
        self.declaration_hash
    }

    /// Returns the estimated bytes occupied by declarations and artifacts.
    pub const fn estimated_declaration_bytes(&self) -> u64 {
        self.estimated_output_bytes
    }

    /// Returns the stable hash of all concrete proxy artifacts.
    pub const fn artifact_hash(&self) -> u64 {
        self.artifact_hash
    }

    /// Returns generated clusters in stable cluster order.
    pub fn clusters(&self) -> &[HlodCluster] {
        &self.clusters
    }

    /// Returns generated representations in stable cluster/level order.
    pub fn representations(&self) -> &[HlodRepresentation] {
        &self.representations
    }

    /// Returns artifacts in the same stable order as [`Self::representations`].
    pub fn artifacts(&self) -> &[HlodProxyArtifact] {
        &self.artifacts
    }
}

/// Bakes deterministic cluster and representation declarations.
///
/// Records are copied and sorted by `(id, source_hash, center bits, radius
/// bits)`, so caller iteration order cannot affect the result. Clusters are
/// contiguous chunks of that canonical order. Each representation receives a
/// deterministic AABB proxy mesh and material payload. Renderer submission,
/// asset residency, and streaming remain explicit downstream contracts.
pub fn bake_hlod(
    config: HlodBakeConfig,
    input: &[HlodProxyRecord],
) -> Result<HlodBakeOutput, HlodBakeError> {
    validate_config(config)?;
    let input_len = u32::try_from(input.len()).map_err(|_| HlodBakeError::SizeOverflow)?;
    if input_len > config.budget.max_input_proxies {
        return Err(HlodBakeError::CountBudgetExceeded {
            kind: HlodBudgetKind::InputProxies,
            required: u64::from(input_len),
            limit: u64::from(config.budget.max_input_proxies),
        });
    }

    let mut records = input.to_vec();
    for record in &records {
        if !record.center.iter().all(|value| value.is_finite()) || !record.radius.is_finite() {
            return Err(HlodBakeError::NonFiniteProxy { id: record.id });
        }
        if record.radius < 0.0 {
            return Err(HlodBakeError::NegativeRadius { id: record.id });
        }
    }
    records.sort_unstable_by_key(|record| {
        (
            record.id,
            record.source_hash,
            record.center[0].to_bits(),
            record.center[1].to_bits(),
            record.center[2].to_bits(),
            record.radius.to_bits(),
        )
    });
    for pair in records.windows(2) {
        if pair[0].id == pair[1].id {
            return Err(HlodBakeError::DuplicateProxyId(pair[0].id));
        }
    }

    let input_hash = hash_records(config.identity_seed, &records);
    if records.is_empty() {
        return Ok(HlodBakeOutput {
            input_hash,
            declaration_hash: hash64(config.identity_seed ^ input_hash ^ 0x484c_4f44),
            artifact_hash: hash64(config.identity_seed ^ input_hash ^ 0x5052_4f58),
            estimated_output_bytes: 0,
            clusters: Vec::new(),
            representations: Vec::new(),
            artifacts: Vec::new(),
        });
    }

    let cluster_count = records.len().div_ceil(config.proxies_per_cluster as usize);
    check_count(
        HlodBudgetKind::Clusters,
        cluster_count,
        config.budget.max_clusters,
    )?;
    let representation_count = cluster_count
        .checked_mul(config.levels as usize)
        .ok_or(HlodBakeError::SizeOverflow)?;
    check_count(
        HlodBudgetKind::Representations,
        representation_count,
        config.budget.max_representations,
    )?;
    let declaration_bytes = declaration_bytes(records.len(), cluster_count, representation_count)?;
    let artifact_bytes = artifact_bytes(representation_count)?;
    let output_bytes = declaration_bytes
        .checked_add(artifact_bytes)
        .ok_or(HlodBakeError::SizeOverflow)?;
    if output_bytes > config.budget.max_declaration_bytes {
        return Err(HlodBakeError::DeclarationBytesBudgetExceeded {
            required: output_bytes,
            limit: config.budget.max_declaration_bytes,
        });
    }

    let mut clusters = Vec::with_capacity(cluster_count);
    let mut representations = Vec::with_capacity(representation_count);
    let mut artifacts = Vec::with_capacity(representation_count);
    let mut generated_ids = BTreeSet::new();
    for (cluster_index, group) in records
        .chunks(config.proxies_per_cluster as usize)
        .enumerate()
    {
        let members = group.iter().map(|record| record.id).collect::<Vec<_>>();
        let source_hash = hash_records(config.identity_seed ^ cluster_index as u64, group);
        let cluster_id = hash64(
            config.identity_seed
                ^ input_hash
                ^ (cluster_index as u64).rotate_left(17)
                ^ source_hash,
        );
        insert_generated_id(&mut generated_ids, cluster_id)?;
        let bounds = cluster_bounds(group).ok_or(HlodBakeError::ProxyGeometryOverflow {
            cluster_index: cluster_index as u32,
        })?;
        let geometry = build_proxy_geometry(bounds, config.identity_seed ^ cluster_id);

        let mut levels = Vec::with_capacity(config.levels as usize);
        for level in 0..config.levels {
            let representation_id = hash64(
                cluster_id
                    ^ source_hash.rotate_left(7)
                    ^ u64::from(level).wrapping_mul(0x9e37_79b9_7f4a_7c15),
            );
            insert_generated_id(&mut generated_ids, representation_id)?;
            let enter_distance = level
                .checked_mul(config.distance_step)
                .ok_or(HlodBakeError::SizeOverflow)?;
            levels.push(HlodLevel {
                asset_key: representation_id,
                enter_distance,
                transition_width: config.transition_width,
            });
            representations.push(HlodRepresentation::new(
                representation_id,
                cluster_id,
                level,
                source_hash,
                group.len() as u32,
                REPRESENTATION_BYTES,
            ));
            let material = build_proxy_material(config.identity_seed, source_hash, level);
            let artifact_hash = hash_artifact(representation_id, &geometry, material);
            artifacts.push(HlodProxyArtifact::new(
                representation_id,
                geometry.clone(),
                material,
                artifact_hash,
            ));
        }
        clusters.push(
            HlodCluster::new(cluster_id, members, levels)
                .map_err(|_| HlodBakeError::InvalidConfig)?,
        );
    }

    let declaration_hash = hash_declarations(config.identity_seed, &clusters, &representations);
    let artifact_hash = hash_artifacts(config.identity_seed, &artifacts);
    Ok(HlodBakeOutput {
        input_hash,
        declaration_hash,
        artifact_hash,
        estimated_output_bytes: output_bytes,
        clusters,
        representations,
        artifacts,
    })
}

fn validate_config(config: HlodBakeConfig) -> Result<(), HlodBakeError> {
    if config.levels == 0
        || config.proxies_per_cluster == 0
        || config.distance_step == 0
        || config.transition_width == 0
        || config.budget.max_input_proxies == 0
        || config.budget.max_clusters == 0
        || config.budget.max_representations == 0
    {
        return Err(HlodBakeError::InvalidConfig);
    }
    Ok(())
}

fn check_count(kind: HlodBudgetKind, required: usize, limit: u32) -> Result<(), HlodBakeError> {
    if required > limit as usize {
        return Err(HlodBakeError::CountBudgetExceeded {
            kind,
            required: required as u64,
            limit: u64::from(limit),
        });
    }
    Ok(())
}

fn declaration_bytes(
    input_count: usize,
    cluster_count: usize,
    representation_count: usize,
) -> Result<u64, HlodBakeError> {
    let input_bytes = (input_count as u64)
        .checked_mul(MEMBER_BYTES)
        .ok_or(HlodBakeError::SizeOverflow)?;
    let cluster_bytes = (cluster_count as u64)
        .checked_mul(CLUSTER_BASE_BYTES)
        .ok_or(HlodBakeError::SizeOverflow)?;
    let representation_bytes = (representation_count as u64)
        .checked_mul(REPRESENTATION_BYTES)
        .ok_or(HlodBakeError::SizeOverflow)?;
    input_bytes
        .checked_add(cluster_bytes)
        .and_then(|bytes| bytes.checked_add(representation_bytes))
        .ok_or(HlodBakeError::SizeOverflow)
}

fn artifact_bytes(representation_count: usize) -> Result<u64, HlodBakeError> {
    let vertex_bytes = (PROXY_VERTEX_COUNT as u64)
        .checked_mul(PROXY_VERTEX_BYTES)
        .ok_or(HlodBakeError::SizeOverflow)?;
    let index_bytes = (PROXY_INDEX_COUNT as u64)
        .checked_mul(PROXY_INDEX_BYTES)
        .ok_or(HlodBakeError::SizeOverflow)?;
    let per_artifact = PROXY_BOUNDS_BYTES
        .checked_add(vertex_bytes)
        .and_then(|bytes| bytes.checked_add(index_bytes))
        .and_then(|bytes| bytes.checked_add(PROXY_GEOMETRY_HASH_BYTES))
        .and_then(|bytes| bytes.checked_add(PROXY_MATERIAL_KEY_BYTES))
        .and_then(|bytes| bytes.checked_add(PROXY_MATERIAL_PARAMS_BYTES))
        .and_then(|bytes| bytes.checked_add(PROXY_MATERIAL_HASH_BYTES))
        .and_then(|bytes| bytes.checked_add(PROXY_ARTIFACT_HEADER_BYTES))
        .ok_or(HlodBakeError::SizeOverflow)?;
    (representation_count as u64)
        .checked_mul(per_artifact)
        .ok_or(HlodBakeError::SizeOverflow)
}

fn cluster_bounds(group: &[HlodProxyRecord]) -> Option<HlodBounds> {
    let mut min = [f64::INFINITY; 3];
    let mut max = [f64::NEG_INFINITY; 3];
    for record in group {
        let radius = f64::from(record.radius);
        for axis in 0..3 {
            let center = f64::from(record.center[axis]);
            let lower = center - radius;
            let upper = center + radius;
            if !lower.is_finite() || !upper.is_finite() {
                return None;
            }
            min[axis] = min[axis].min(lower);
            max[axis] = max[axis].max(upper);
        }
    }

    let center = [
        min[0] * 0.5 + max[0] * 0.5,
        min[1] * 0.5 + max[1] * 0.5,
        min[2] * 0.5 + max[2] * 0.5,
    ];
    let radius = (max[0] - min[0]) * 0.5 + (max[1] - min[1]) * 0.5 + (max[2] - min[2]) * 0.5;
    let min = [min[0] as f32, min[1] as f32, min[2] as f32];
    let max = [max[0] as f32, max[1] as f32, max[2] as f32];
    let center = [center[0] as f32, center[1] as f32, center[2] as f32];
    let radius = radius as f32;
    if !min.iter().all(|value| value.is_finite())
        || !max.iter().all(|value| value.is_finite())
        || !center.iter().all(|value| value.is_finite())
        || !radius.is_finite()
    {
        return None;
    }
    Some(HlodBounds::new(min, max, center, radius))
}

fn build_proxy_geometry(bounds: HlodBounds, seed: u64) -> HlodProxyGeometry {
    const CORNER_NORMAL: f32 = 0.577_350_26;
    let min = bounds.min;
    let max = bounds.max;
    let vertices = vec![
        HlodProxyVertex {
            position: [min[0], min[1], min[2]],
            normal: [-CORNER_NORMAL, -CORNER_NORMAL, -CORNER_NORMAL],
            uv: [0.0, 0.0],
        },
        HlodProxyVertex {
            position: [max[0], min[1], min[2]],
            normal: [CORNER_NORMAL, -CORNER_NORMAL, -CORNER_NORMAL],
            uv: [1.0, 0.0],
        },
        HlodProxyVertex {
            position: [max[0], max[1], min[2]],
            normal: [CORNER_NORMAL, CORNER_NORMAL, -CORNER_NORMAL],
            uv: [1.0, 1.0],
        },
        HlodProxyVertex {
            position: [min[0], max[1], min[2]],
            normal: [-CORNER_NORMAL, CORNER_NORMAL, -CORNER_NORMAL],
            uv: [0.0, 1.0],
        },
        HlodProxyVertex {
            position: [min[0], min[1], max[2]],
            normal: [-CORNER_NORMAL, -CORNER_NORMAL, CORNER_NORMAL],
            uv: [0.0, 0.0],
        },
        HlodProxyVertex {
            position: [max[0], min[1], max[2]],
            normal: [CORNER_NORMAL, -CORNER_NORMAL, CORNER_NORMAL],
            uv: [1.0, 0.0],
        },
        HlodProxyVertex {
            position: [max[0], max[1], max[2]],
            normal: [CORNER_NORMAL, CORNER_NORMAL, CORNER_NORMAL],
            uv: [1.0, 1.0],
        },
        HlodProxyVertex {
            position: [min[0], max[1], max[2]],
            normal: [-CORNER_NORMAL, CORNER_NORMAL, CORNER_NORMAL],
            uv: [0.0, 1.0],
        },
    ];
    let indices = vec![
        0, 2, 1, 0, 3, 2, // -Z
        4, 5, 6, 4, 6, 7, // +Z
        0, 4, 7, 0, 7, 3, // -X
        1, 2, 6, 1, 6, 5, // +X
        0, 1, 5, 0, 5, 4, // -Y
        3, 7, 6, 3, 6, 2, // +Y
    ];
    let hash = hash_geometry(seed, &bounds, &vertices, &indices);
    HlodProxyGeometry::new(bounds, vertices, indices, hash)
}

fn build_proxy_material(seed: u64, source_hash: u64, level: u32) -> HlodProxyMaterial {
    let material_key = hash64(
        seed ^ source_hash.rotate_left(13) ^ u64::from(level).wrapping_mul(0xd6e8_feb8_6659_fd93),
    );
    let sample = |shift: u32| -> f32 { ((material_key.rotate_left(shift) & 0xff) as f32) / 255.0 };
    let params = HlodMaterialParams {
        base_color: [
            0.15 + 0.70 * sample(0),
            0.15 + 0.70 * sample(8),
            0.15 + 0.70 * sample(16),
            1.0,
        ],
        roughness: 0.25 + 0.70 * sample(24),
        metallic: 0.05 + 0.35 * sample(32),
    };
    let hash = hash_material(material_key, params);
    HlodProxyMaterial::new(material_key, params, hash)
}

fn hash_geometry(
    seed: u64,
    bounds: &HlodBounds,
    vertices: &[HlodProxyVertex],
    indices: &[u32],
) -> u64 {
    let mut hash = seed;
    for value in bounds
        .min
        .into_iter()
        .chain(bounds.max)
        .chain(bounds.center)
    {
        hash = hash64(hash ^ u64::from(value.to_bits()));
    }
    hash = hash_f32(hash, bounds.radius);
    for vertex in vertices {
        for value in vertex.position {
            hash = hash_f32(hash, value);
        }
        for value in vertex.normal {
            hash = hash_f32(hash, value);
        }
        for value in vertex.uv {
            hash = hash_f32(hash, value);
        }
    }
    for index in indices {
        hash = hash64(hash ^ u64::from(*index));
    }
    hash
}

fn hash_material(material_key: u64, params: HlodMaterialParams) -> u64 {
    let mut hash = hash64(material_key);
    for value in params.base_color {
        hash = hash_f32(hash, value);
    }
    hash = hash_f32(hash, params.roughness);
    hash_f32(hash, params.metallic)
}

fn hash_f32(hash: u64, value: f32) -> u64 {
    hash64(hash ^ u64::from(value.to_bits()))
}

fn hash_artifact(
    representation_id: u64,
    geometry: &HlodProxyGeometry,
    material: HlodProxyMaterial,
) -> u64 {
    hash64(representation_id ^ geometry.hash().rotate_left(17) ^ material.hash().rotate_left(31))
}

fn hash_artifacts(seed: u64, artifacts: &[HlodProxyArtifact]) -> u64 {
    let mut hash = seed ^ 0x5052_4f58;
    for artifact in artifacts {
        hash = hash64(hash ^ artifact.representation_id());
        hash = hash64(hash ^ artifact.geometry().hash());
        hash = hash64(hash ^ artifact.material().hash());
        hash = hash64(hash ^ artifact.hash());
    }
    hash
}

fn insert_generated_id(ids: &mut BTreeSet<u64>, id: u64) -> Result<(), HlodBakeError> {
    if ids.insert(id) {
        Ok(())
    } else {
        Err(HlodBakeError::GeneratedIdCollision(id))
    }
}

fn hash_records(seed: u64, records: &[HlodProxyRecord]) -> u64 {
    let mut hash = seed ^ 0xcbf2_9ce4_8422_2325;
    for record in records {
        hash = hash64(hash ^ record.id);
        hash = hash64(hash ^ record.center[0].to_bits() as u64);
        hash = hash64(hash ^ record.center[1].to_bits() as u64);
        hash = hash64(hash ^ record.center[2].to_bits() as u64);
        hash = hash64(hash ^ u64::from(record.radius.to_bits()));
        hash = hash64(hash ^ record.source_hash);
    }
    hash
}

fn hash_declarations(
    seed: u64,
    clusters: &[HlodCluster],
    representations: &[HlodRepresentation],
) -> u64 {
    let mut hash = seed;
    for cluster in clusters {
        hash = hash64(hash ^ cluster.id());
        for member in cluster.members() {
            hash = hash64(hash ^ *member);
        }
        for level in cluster.levels() {
            hash = hash64(hash ^ level.asset_key);
            hash = hash64(hash ^ u64::from(level.enter_distance));
            hash = hash64(hash ^ u64::from(level.transition_width));
        }
    }
    for representation in representations {
        hash = hash64(hash ^ representation.id());
        hash = hash64(hash ^ representation.source_hash());
    }
    hash
}

fn hash64(input: u64) -> u64 {
    let mut value = input ^ 0x9e37_79b9_7f4a_7c15;
    value ^= value >> 30;
    value = value.wrapping_mul(0xbf58_476d_1ce4_e5b9);
    value ^= value >> 27;
    value = value.wrapping_mul(0x94d0_49bb_1331_11eb);
    value ^ (value >> 31)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn config() -> HlodBakeConfig {
        HlodBakeConfig::new(3, 2, 100, 10, 0x1234, HlodBakeBudget::new(16, 8, 24, 4_096))
    }

    fn proxy(id: u64) -> HlodProxyRecord {
        HlodProxyRecord {
            id,
            center: [id as f32, 0.0, 1.0],
            radius: 2.0,
            source_hash: id.wrapping_mul(17),
        }
    }

    #[test]
    fn bake_is_deterministic_and_canonicalizes_input_order() {
        let first = bake_hlod(config(), &[proxy(3), proxy(1), proxy(2)]).unwrap();
        let second = bake_hlod(config(), &[proxy(2), proxy(3), proxy(1)]).unwrap();
        assert_eq!(first, second);
        assert_eq!(first.clusters().len(), 2);
        assert_eq!(first.representations().len(), 6);
        assert_eq!(first.artifacts().len(), 6);
        assert_eq!(first.clusters()[0].members(), &[1, 2]);
        assert_ne!(first.input_hash(), 0);
        assert_ne!(first.declaration_hash(), 0);
        assert_ne!(first.artifact_hash(), 0);
        for (first_artifact, second_artifact) in first.artifacts().iter().zip(second.artifacts()) {
            assert_eq!(
                first_artifact.geometry().hash(),
                second_artifact.geometry().hash()
            );
            assert_eq!(
                first_artifact.material().hash(),
                second_artifact.material().hash()
            );
            assert_eq!(first_artifact.hash(), second_artifact.hash());
        }
    }

    #[test]
    fn proxy_geometry_has_valid_triangle_indices_and_finite_bounds() {
        let output = bake_hlod(config(), &[proxy(3), proxy(1), proxy(2)]).unwrap();
        for artifact in output.artifacts() {
            let geometry = artifact.geometry();
            assert_eq!(geometry.vertices().len(), PROXY_VERTEX_COUNT);
            assert_eq!(geometry.indices().len(), PROXY_INDEX_COUNT);
            assert_eq!(geometry.indices().len() % 3, 0);
            assert!(geometry
                .indices()
                .iter()
                .all(|index| (*index as usize) < geometry.vertices().len()));
            assert!(geometry
                .vertices()
                .iter()
                .flat_map(|vertex| vertex.position)
                .all(|value| value.is_finite()));
            assert!(geometry.bounds().min.iter().all(|value| value.is_finite()));
            assert!(geometry.bounds().max.iter().all(|value| value.is_finite()));
            assert!(geometry
                .bounds()
                .center
                .iter()
                .all(|value| value.is_finite()));
            assert!(geometry.bounds().radius.is_finite());
            let material = artifact.material();
            assert!(material
                .params()
                .base_color
                .iter()
                .all(|value| value.is_finite()));
            assert!(material.params().roughness.is_finite());
            assert!(material.params().metallic.is_finite());
            assert!((0.0..=1.0).contains(&material.params().roughness));
            assert!((0.0..=1.0).contains(&material.params().metallic));
        }
    }

    #[test]
    fn bake_rejects_count_and_byte_budgets() {
        let mut limited = config();
        limited.budget = HlodBakeBudget::new(2, 8, 24, 4_096);
        assert!(matches!(
            bake_hlod(limited, &[proxy(1), proxy(2), proxy(3)]),
            Err(HlodBakeError::CountBudgetExceeded {
                kind: HlodBudgetKind::InputProxies,
                ..
            })
        ));

        limited = config();
        limited.budget.max_declaration_bytes = 1;
        assert!(matches!(
            bake_hlod(limited, &[proxy(1)]),
            Err(HlodBakeError::DeclarationBytesBudgetExceeded { .. })
        ));
    }

    #[test]
    fn bake_rejects_duplicate_ids_and_non_finite_values() {
        assert_eq!(
            bake_hlod(config(), &[proxy(4), proxy(4)]),
            Err(HlodBakeError::DuplicateProxyId(4))
        );
        let mut invalid = proxy(5);
        invalid.center[1] = f32::NAN;
        assert_eq!(
            bake_hlod(config(), &[invalid]),
            Err(HlodBakeError::NonFiniteProxy { id: 5 })
        );
        invalid = proxy(7);
        invalid.radius = f32::INFINITY;
        assert_eq!(
            bake_hlod(config(), &[invalid]),
            Err(HlodBakeError::NonFiniteProxy { id: 7 })
        );
        invalid = proxy(6);
        invalid.radius = -1.0;
        assert_eq!(
            bake_hlod(config(), &[invalid]),
            Err(HlodBakeError::NegativeRadius { id: 6 })
        );
    }

    #[test]
    fn empty_input_is_a_stable_empty_declaration() {
        let first = bake_hlod(config(), &[]).unwrap();
        let second = bake_hlod(config(), &[]).unwrap();
        assert_eq!(first, second);
        assert!(first.clusters().is_empty());
        assert!(first.representations().is_empty());
        assert!(first.artifacts().is_empty());
        assert_eq!(first.estimated_declaration_bytes(), 0);
        assert_eq!(first.artifact_hash(), second.artifact_hash());
    }
}
