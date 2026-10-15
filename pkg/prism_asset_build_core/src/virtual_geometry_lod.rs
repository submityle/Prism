//! Attribute-preserving endpoint-QEM static virtual-geometry LOD chains.

use alloc::collections::{BTreeMap, BTreeSet};
use alloc::vec::Vec;

use prism_mesh::{content_hash, Mesh, MeshContentHash, MeshError, POSITION_ATTRIBUTE};
use prism_render_architecture::ray_scene::{
    decimate_preserving_vertices, DecimationChannel, DecimationError, TriangleMesh,
    VertexDecimationConstraints,
};
use prism_render_architecture::virtual_geometry::{
    select_lod, GeometryLodPolicy, LodLevel, LodProjection, LodSelection, VirtualGeometryAsset,
};
use prism_render_driver::PrimitiveTopology;

use crate::virtual_geometry_builder::{
    build_static_virtual_geometry, settings_hash, submesh_ranges, validate_settings,
    StaticClusterBuildError, StaticClusterBuildSettings,
};

mod attributes;
mod surface_error;
pub use attributes::{LodAttributeMetric, LodAttributePolicy, StaticLodPolicy};

/// Algorithm revision bound into every chain and page identity.
pub const STATIC_LOD_ALGORITHM_VERSION: u32 = 2;

/// One independently serialized level in a progressively simplified LOD chain.
#[derive(Clone, Debug, PartialEq)]
pub struct StaticVirtualGeometryLod {
    /// Contiguous level index, beginning at zero.
    pub level: u32,
    /// Requested triangle count; protected seams/topology may prevent reaching it.
    pub target_triangles: u32,
    /// Actual count, including all material partitions after degenerate removal.
    pub achieved_triangles: u32,
    /// Derived page-key asset identity, bound to content, policy and chain.
    pub asset_id: u32,
    /// Authored content identity shared by the chain, before simplification.
    pub source_content_hash: MeshContentHash,
    /// Full chain recipe digest, including source, targets and algorithm revision.
    pub chain_build_hash: [u8; 32],
    /// Measured maximum authored-vertex to this level's unquantized triangle
    /// surface distance. One-sided vertex samples, not a Hausdorff bound.
    pub measured_vertex_error: f32,
    /// Nondecreasing prefix envelope of measured_vertex_error plus this level's
    /// maximum CLST position displacement. Used by the runtime LOD selector.
    /// This does not turn the sampled metric into a continuous surface bound.
    pub vertex_error: f32,
    /// Independent paged asset preserving all surviving authored attributes.
    pub asset: VirtualGeometryAsset,
}

/// A validated LOD chain plus the runtime metrics used to select it.
#[derive(Clone, Debug, PartialEq)]
pub struct StaticVirtualGeometryLodSet {
    levels: Vec<StaticVirtualGeometryLod>,
    metrics: Vec<LodLevel>,
}

/// A runtime LOD selection with concrete display and prefetch assets.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct StaticVirtualGeometryLodPick<'a> {
    /// Selector result, including display and prefetch level indices.
    pub selection: LodSelection,
    /// Asset used for the current view.
    pub display: &'a StaticVirtualGeometryLod,
    /// Asset that should be requested ahead of camera motion.
    pub prefetch: &'a StaticVirtualGeometryLod,
}

impl StaticVirtualGeometryLodSet {
    /// Validates contiguous identities, monotonic targets/counts/errors, and
    /// asset tables. Never clamps invalid metadata into a seemingly valid set.
    pub fn new(
        mut levels: Vec<StaticVirtualGeometryLod>,
    ) -> Result<Self, StaticVirtualGeometryLodError> {
        use StaticVirtualGeometryLodError as Error;
        if levels.is_empty() {
            return Err(Error::InvalidTarget);
        }
        levels.sort_by_key(|level| level.level);
        let source = levels[0].source_content_hash;
        let recipe = levels[0].chain_build_hash;
        let mut identities = BTreeSet::new();
        for (index, level) in levels.iter().enumerate() {
            if usize::try_from(level.level) != Ok(index)
                || level.target_triangles == 0
                || level.achieved_triangles == 0
                || !level.vertex_error.is_finite()
                || level.vertex_error < 0.0
                || !level.measured_vertex_error.is_finite()
                || level.measured_vertex_error < 0.0
                || level.source_content_hash != source
                || level.chain_build_hash != recipe
                || level.asset.header.asset_id != level.asset_id
                || level.asset.header.build_settings_hash != level_hash(recipe, level.level)
                || level.asset_id != page_asset_id(level.asset.header.build_settings_hash)
            {
                return Err(Error::InvalidLevel("identity/metric"));
            }
            if !identities.insert(level.asset_id) {
                return Err(Error::IdentityCollision(level.asset_id));
            }
            level
                .asset
                .validate()
                .map_err(|e| Error::Build(StaticClusterBuildError::InvalidAsset(e)))?;
            let actual: u64 = level
                .asset
                .clusters
                .iter()
                .map(|c| u64::from(c.triangle_count))
                .sum();
            let quantization = level
                .asset
                .nodes
                .iter()
                .map(|n| n.self_error)
                .fold(0.0_f32, f32::max);
            let own_error = surface_error::upper_f32(
                f64::from(level.measured_vertex_error) + f64::from(quantization),
            )
            .ok_or(Error::InvalidMetric)?;
            if actual != u64::from(level.achieved_triangles) || level.vertex_error < own_error {
                return Err(Error::InvalidLevel("triangle count/error envelope"));
            }
        }
        if levels.windows(2).any(|pair| {
            pair[1].target_triangles > pair[0].target_triangles
                || pair[1].achieved_triangles > pair[0].achieved_triangles
                || pair[1].vertex_error < pair[0].vertex_error
        }) {
            return Err(Error::InvalidLevel("nonmonotonic chain"));
        }
        let metrics = levels
            .iter()
            .map(|level| LodLevel {
                level: level.level,
                geometric_error: level.vertex_error,
            })
            .collect();
        Ok(Self { levels, metrics })
    }

    /// Selects display and prefetch assets using shared hysteresis policy.
    #[must_use]
    pub fn select(
        &self,
        projection: LodProjection,
        view_distance: f32,
        closing_speed: f32,
        policy: GeometryLodPolicy,
        previous: Option<u32>,
    ) -> Option<StaticVirtualGeometryLodPick<'_>> {
        let selection = select_lod(
            &self.metrics,
            projection,
            view_distance,
            closing_speed,
            policy,
            previous,
        )?;
        Some(StaticVirtualGeometryLodPick {
            selection,
            display: self.levels.get(selection.level as usize)?,
            prefetch: self.levels.get(selection.prefetch_level as usize)?,
        })
    }

    /// Indexed levels in ascending order.
    #[must_use]
    pub fn levels(&self) -> &[StaticVirtualGeometryLod] {
        &self.levels
    }
}

/// Failure while building or validating a static LOD chain.
#[derive(Clone, Debug, PartialEq)]
pub enum StaticVirtualGeometryLodError {
    /// Source mesh validation failed.
    InvalidMesh(MeshError),
    /// Source must be an indexed triangle list with Float32x3 positions.
    UnsupportedSource,
    /// Empty, increasing, zero, or out-of-source-range targets.
    InvalidTarget,
    /// Unknown/duplicate semantic, position override, or invalid metric weight.
    InvalidPolicy(&'static str),
    /// A measured/enveloped distance cannot be represented as a finite f32.
    InvalidMetric,
    /// Removing degenerates left no triangle surface.
    EmptySurface,
    /// Inconsistent externally supplied LOD metadata.
    InvalidLevel(&'static str),
    /// Endpoint-QEM failure.
    Decimation(DecimationError),
    /// Paged geometry construction or validation failed.
    Build(StaticClusterBuildError),
    /// Two levels derived the same truncated page-key asset identity.
    IdentityCollision(u32),
}

impl core::fmt::Display for StaticVirtualGeometryLodError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::InvalidMesh(e) => write!(f, "invalid source mesh: {e}"),
            Self::UnsupportedSource => {
                f.write_str("LOD source requires indexed TriangleList and Float32x3 positions")
            }
            Self::InvalidTarget => f.write_str(
                "LOD targets must be nonempty, nonincreasing and within the source triangle range",
            ),
            Self::InvalidPolicy(name) => write!(f, "invalid LOD attribute policy for {name}"),
            Self::InvalidMetric => f.write_str("LOD error is invalid or exceeds finite f32 range"),
            Self::EmptySurface => f.write_str("LOD source has no nondegenerate surface"),
            Self::InvalidLevel(reason) => write!(f, "inconsistent LOD level: {reason}"),
            Self::Decimation(e) => write!(f, "QEM decimation failed: {e}"),
            Self::Build(e) => write!(f, "LOD page build failed: {e}"),
            Self::IdentityCollision(id) => {
                write!(f, "LOD page-key asset identity collision for {id}")
            }
        }
    }
}

/// Builds attribute-preserving LOD assets using default semantic metrics.
pub fn build_static_virtual_geometry_lod_chain(
    asset_id: u32,
    mesh: &Mesh,
    settings: StaticClusterBuildSettings,
    targets: &[u32],
) -> Result<Vec<StaticVirtualGeometryLod>, StaticVirtualGeometryLodError> {
    build_static_virtual_geometry_lod_chain_with_policy(
        asset_id,
        mesh,
        settings,
        targets,
        &StaticLodPolicy::default(),
    )
}

/// Simplifies each level from the previous one, preserving material partitions,
/// pinned chart boundaries and exact endpoint attribute bytes. Targets must be
/// nonincreasing. Reports actual counts when protected topology stops reduction.
pub fn build_static_virtual_geometry_lod_chain_with_policy(
    asset_id: u32,
    mesh: &Mesh,
    settings: StaticClusterBuildSettings,
    targets: &[u32],
    policy: &StaticLodPolicy,
) -> Result<Vec<StaticVirtualGeometryLod>, StaticVirtualGeometryLodError> {
    use StaticVirtualGeometryLodError as Error;
    validate_settings(settings).map_err(Error::Build)?;
    if mesh.primitive_topology() != PrimitiveTopology::TriangleList || mesh.indices().is_none() {
        return Err(Error::UnsupportedSource);
    }
    mesh.validate().map_err(Error::InvalidMesh)?;
    let source_positions = positions(mesh)?;
    let source_triangles = mesh.indices().expect("checked").len() / 3;
    abi_u32(source_positions.len(), "vertex")?;
    abi_u32(mesh.indices().expect("checked").len(), "index")?;
    if targets.is_empty()
        || targets.len() > u32::MAX as usize
        || targets
            .iter()
            .any(|t| *t == 0 || *t as usize > source_triangles)
        || targets.windows(2).any(|p| p[1] > p[0])
    {
        return Err(Error::InvalidTarget);
    }
    let policies = attributes::resolve(mesh, policy)?;
    let source_content_hash = content_hash(mesh);
    let chain_build_hash = chain_hash(asset_id, source_content_hash, settings, targets, &policies);
    let mut output = Vec::with_capacity(targets.len());
    let mut identities = BTreeSet::new();
    let mut previous = None;
    let mut envelope = 0.0_f32;
    for (level, &target) in targets.iter().enumerate() {
        let current = previous.as_ref().unwrap_or(mesh);
        let simplified = simplify_partitions(current, target as usize, &policies)?;
        let indices = simplified.indices().expect("indexed output").to_u32();
        let achieved_triangles = abi_u32(indices.len() / 3, "triangle")?;
        if achieved_triangles == 0 {
            return Err(Error::EmptySurface);
        }
        let measured_vertex_error =
            surface_error::measure(source_positions, positions(&simplified)?, &indices)?;
        let build_hash = level_hash(chain_build_hash, level as u32);
        let level_asset_id = page_asset_id(build_hash);
        if !identities.insert(level_asset_id) {
            return Err(Error::IdentityCollision(level_asset_id));
        }
        let mut asset = build_static_virtual_geometry(level_asset_id, &simplified, settings)
            .map_err(Error::Build)?;
        // The recipe includes the cluster builder's own versioned settings hash.
        asset.header.build_settings_hash = build_hash;
        let quantization = asset
            .nodes
            .iter()
            .map(|n| n.self_error)
            .fold(0.0_f32, f32::max);
        envelope = envelope.max(
            surface_error::upper_f32(f64::from(measured_vertex_error) + f64::from(quantization))
                .ok_or(Error::InvalidMetric)?,
        );
        output.push(StaticVirtualGeometryLod {
            level: level as u32,
            target_triangles: target,
            achieved_triangles,
            asset_id: level_asset_id,
            source_content_hash,
            chain_build_hash,
            measured_vertex_error,
            vertex_error: envelope,
            asset,
        });
        previous = Some(simplified);
    }
    Ok(output)
}

fn chain_hash(
    asset_id: u32,
    source: MeshContentHash,
    settings: StaticClusterBuildSettings,
    targets: &[u32],
    policies: &BTreeMap<&'static str, LodAttributeMetric>,
) -> [u8; 32] {
    let mut hasher = blake3::Hasher::new();
    hasher.update(b"Prism.StaticLOD.EndpointQEM");
    hasher.update(&STATIC_LOD_ALGORITHM_VERSION.to_le_bytes());
    hasher.update(&asset_id.to_le_bytes());
    hasher.update(&source.as_bytes());
    hasher.update(&settings_hash(settings));
    hasher.update(&(targets.len() as u64).to_le_bytes());
    for target in targets {
        hasher.update(&target.to_le_bytes());
    }
    attributes::hash_policies(&mut hasher, policies);
    *hasher.finalize().as_bytes()
}

fn level_hash(recipe: [u8; 32], level: u32) -> [u8; 32] {
    let mut hasher = blake3::Hasher::new();
    hasher.update(b"Prism.StaticLOD.Level");
    hasher.update(&recipe);
    hasher.update(&level.to_le_bytes());
    *hasher.finalize().as_bytes()
}

fn page_asset_id(hash: [u8; 32]) -> u32 {
    u32::from_le_bytes(hash[..4].try_into().expect("digest")).max(1)
}

fn positions(mesh: &Mesh) -> Result<&[[f32; 3]], StaticVirtualGeometryLodError> {
    mesh.attributes()
        .find(|(a, _)| a.name == POSITION_ATTRIBUTE)
        .and_then(|(_, v)| v.as_positions())
        .ok_or(StaticVirtualGeometryLodError::UnsupportedSource)
}

fn abi_u32(count: usize, kind: &'static str) -> Result<u32, StaticVirtualGeometryLodError> {
    u32::try_from(count).map_err(|_| {
        StaticVirtualGeometryLodError::Build(StaticClusterBuildError::AbiCountOverflow(kind))
    })
}

fn simplify_partitions(
    source: &Mesh,
    target: usize,
    policies: &BTreeMap<&'static str, LodAttributeMetric>,
) -> Result<Mesh, StaticVirtualGeometryLodError> {
    use StaticVirtualGeometryLodError as Error;
    let indices = source.indices().expect("indexed source").to_u32();
    let ranges = submesh_ranges(source, abi_u32(indices.len(), "index")?).map_err(Error::Build)?;
    let counts: Vec<_> = ranges
        .iter()
        .map(|(start, end, _)| ((end - start) / 3) as usize)
        .collect();
    let targets = partition_targets(&counts, target);
    let positions = positions(source)?;
    let constraints = attributes::constraints(source, policies)?;
    let mut source_vertices = Vec::new();
    let mut output_indices = Vec::new();
    let mut submeshes = Vec::new();
    for ((start, end, material), target) in ranges.into_iter().zip(targets) {
        let mut remap = BTreeMap::new();
        let mut global_vertices = Vec::new();
        let mut local_indices = Vec::new();
        for &global in &indices[start as usize..end as usize] {
            let next = global_vertices.len() as u32;
            let local = *remap.entry(global).or_insert_with(|| {
                global_vertices.push(global);
                next
            });
            local_indices.push(local);
        }
        let local = TriangleMesh::new(
            global_vertices
                .iter()
                .map(|v| positions[*v as usize])
                .collect(),
            Vec::new(),
            Vec::new(),
            local_indices
                .chunks_exact(3)
                .map(|t| [t[0], t[1], t[2]])
                .collect(),
        )
        .map_err(|_| Error::UnsupportedSource)?;
        let local_constraints = VertexDecimationConstraints {
            locked_vertices: global_vertices
                .iter()
                .map(|v| constraints.locked_vertices[*v as usize])
                .collect(),
            discrete_groups: if constraints.discrete_groups.is_empty() {
                Vec::new()
            } else {
                global_vertices
                    .iter()
                    .map(|v| constraints.discrete_groups[*v as usize])
                    .collect()
            },
            channels: constraints
                .channels
                .iter()
                .map(|c| DecimationChannel {
                    values: global_vertices
                        .iter()
                        .map(|v| c.values[*v as usize])
                        .collect(),
                    components: c.components,
                    weight: c.weight,
                })
                .collect(),
            preserve_boundary: true,
        };
        let reduced = decimate_preserving_vertices(&local, target.max(1), &local_constraints)
            .map_err(Error::Decimation)?;
        if reduced.mesh.triangle_count() == 0 {
            continue;
        }
        let base = abi_u32(source_vertices.len(), "vertex")?;
        // Check the whole sum before translating local indices.
        abi_u32(
            source_vertices
                .len()
                .checked_add(reduced.source_vertices.len())
                .ok_or(Error::Build(StaticClusterBuildError::AbiCountOverflow(
                    "vertex",
                )))?,
            "vertex",
        )?;
        source_vertices.extend(
            reduced
                .source_vertices
                .iter()
                .map(|v| global_vertices[*v as usize]),
        );
        let start = abi_u32(output_indices.len(), "index")?;
        output_indices.extend(
            reduced
                .mesh
                .indices()
                .iter()
                .flat_map(|t| t.iter().map(|v| base + v)),
        );
        submeshes.push(prism_mesh::SubMesh::new(
            material,
            start..abi_u32(output_indices.len(), "index")?,
        ));
    }
    if output_indices.is_empty() {
        return Err(Error::EmptySurface);
    }
    let mut output = Mesh::new(PrimitiveTopology::TriangleList);
    for (attribute, values) in source.attributes() {
        let values = values
            .gather(&source_vertices)
            .ok_or(Error::InvalidPolicy(attribute.name))?;
        output
            .insert_attribute(*attribute, values)
            .map_err(Error::InvalidMesh)?;
    }
    output.with_indices(prism_mesh::Indices::Uint32(output_indices));
    output.with_submeshes(submeshes);
    output.validate().map_err(Error::InvalidMesh)?;
    Ok(output)
}

/// Largest-remainder apportionment preserves one face per nonempty partition.
/// The sum is exactly max(target, partition_count), capped at the current count.
fn partition_targets(counts: &[usize], target: usize) -> Vec<usize> {
    let total: usize = counts.iter().sum();
    let nonempty = counts.iter().filter(|n| **n > 0).count();
    let desired = target.max(nonempty).min(total);
    let available = total - nonempty;
    if available == 0 {
        return counts.to_vec();
    }
    let extra = desired - nonempty;
    let mut output = Vec::with_capacity(counts.len());
    let mut remainders = Vec::new();
    for (index, &count) in counts.iter().enumerate() {
        if count == 0 {
            output.push(0);
            continue;
        }
        let numerator = extra as u64 * (count - 1) as u64;
        output.push(1 + (numerator / available as u64) as usize);
        remainders.push((numerator % available as u64, index));
    }
    remainders.sort_unstable_by(|a, b| b.0.cmp(&a.0).then_with(|| a.1.cmp(&b.1)));
    let missing = desired - output.iter().sum::<usize>();
    for (_, index) in remainders.into_iter().take(missing) {
        output[index] += 1;
    }
    output
}

#[cfg(test)]
mod tests;
