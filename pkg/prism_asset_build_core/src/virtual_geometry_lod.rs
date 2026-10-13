//! QEM-backed static virtual-geometry LOD chain construction.

use alloc::collections::BTreeMap;
use alloc::collections::BTreeSet;
use alloc::vec::Vec;

use prism_mesh::{Mesh, MeshError, POSITION_ATTRIBUTE};
use prism_render_architecture::ray_scene::{decimate, DecimationError, TriangleMesh};
use prism_render_driver::{PrimitiveTopology, VertexFormat};

use crate::virtual_geometry_builder::{
    build_static_virtual_geometry, StaticClusterBuildError, StaticClusterBuildSettings,
};
use prism_render_architecture::virtual_geometry::VirtualGeometryAsset;
use prism_render_architecture::virtual_geometry::{
    select_lod, GeometryLodPolicy, LodLevel, LodProjection, LodSelection,
};

/// One independently serialized level in a static LOD chain.
#[derive(Clone, Debug, PartialEq)]
pub struct StaticVirtualGeometryLod {
    /// Level index in the requested target order.
    pub level: u32,
    /// Triangle target passed to the QEM decimator.
    pub target_triangles: u32,
    /// Derived artifact identity used by page keys for this LOD.
    pub asset_id: u32,
    /// Maximum distance from an authored vertex to the simplified triangle
    /// surface. This is a one-sided vertex-sample metric, not a full Hausdorff
    /// bound.
    pub vertex_error: f32,
    /// Paged static virtual-geometry asset for this level.
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
    /// Validates and indexes a built chain by level.
    pub fn new(
        mut levels: Vec<StaticVirtualGeometryLod>,
    ) -> Result<Self, StaticVirtualGeometryLodError> {
        if levels.is_empty() {
            return Err(StaticVirtualGeometryLodError::InvalidTarget);
        }
        levels.sort_by_key(|level| level.level);
        if levels.windows(2).any(|pair| {
            pair[0].level == pair[1].level
                || !pair[0].vertex_error.is_finite()
                || !pair[1].vertex_error.is_finite()
                || pair[1].vertex_error < pair[0].vertex_error
        }) {
            return Err(StaticVirtualGeometryLodError::InvalidTarget);
        }
        let metrics = levels
            .iter()
            .map(|level| LodLevel {
                level: level.level,
                geometric_error: level.vertex_error.max(0.0),
            })
            .collect();
        Ok(Self { levels, metrics })
    }

    /// Selects the display and prefetch assets using the shared runtime policy.
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
        let display = self
            .levels
            .iter()
            .find(|level| level.level == selection.level)?;
        let prefetch = self
            .levels
            .iter()
            .find(|level| level.level == selection.prefetch_level)?;
        Some(StaticVirtualGeometryLodPick {
            selection,
            display,
            prefetch,
        })
    }

    /// Returns the indexed LOD levels in ascending level order.
    #[must_use]
    pub fn levels(&self) -> &[StaticVirtualGeometryLod] {
        &self.levels
    }
}

/// Failure while building a static QEM LOD chain.
#[derive(Clone, Debug, PartialEq)]
pub enum StaticVirtualGeometryLodError {
    /// Source mesh validation failed.
    InvalidMesh(MeshError),
    /// The source must have indexed triangle-list topology.
    UnsupportedSource,
    /// Reserved for callers using an older boundary policy.
    MaterialSubmeshesUnsupported,
    /// The target is zero or greater than the source triangle count.
    InvalidTarget,
    /// Existing QEM decimator failure.
    Decimation(DecimationError),
    /// The resulting mesh could not be turned into paged geometry.
    Build(StaticClusterBuildError),
    /// Two requested levels derived the same page-key asset identity.
    IdentityCollision(u32),
}

impl core::fmt::Display for StaticVirtualGeometryLodError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::InvalidMesh(error) => write!(f, "invalid source mesh: {error}"),
            Self::UnsupportedSource => {
                f.write_str("LOD source must be indexed TriangleList geometry")
            }
            Self::MaterialSubmeshesUnsupported => {
                f.write_str("LOD simplification cannot cross material submesh boundaries")
            }
            Self::InvalidTarget => f.write_str("LOD target is outside the source triangle range"),
            Self::Decimation(error) => write!(f, "QEM decimation failed: {error}"),
            Self::Build(error) => write!(f, "LOD page build failed: {error}"),
            Self::IdentityCollision(asset_id) => {
                write!(f, "LOD page-key asset identity collision for {asset_id}")
            }
        }
    }
}

/// Builds independent QEM-reduced paged assets in the caller's target order.
pub fn build_static_virtual_geometry_lod_chain(
    asset_id: u32,
    mesh: &Mesh,
    settings: StaticClusterBuildSettings,
    targets: &[u32],
) -> Result<Vec<StaticVirtualGeometryLod>, StaticVirtualGeometryLodError> {
    if mesh.primitive_topology() != PrimitiveTopology::TriangleList || mesh.indices().is_none() {
        return Err(StaticVirtualGeometryLodError::UnsupportedSource);
    }
    mesh.validate()
        .map_err(StaticVirtualGeometryLodError::InvalidMesh)?;
    if targets.is_empty() {
        return Err(StaticVirtualGeometryLodError::InvalidTarget);
    }
    let source_positions = positions(mesh)?;
    let source_indices = mesh.indices().expect("checked above").to_u32();
    let source_triangles = source_indices.len() / 3;
    if targets
        .iter()
        .any(|target| *target == 0 || *target as usize > source_triangles)
    {
        return Err(StaticVirtualGeometryLodError::InvalidTarget);
    }
    let mut output = Vec::with_capacity(targets.len());
    let mut identities = BTreeSet::new();
    for (level, target) in targets.iter().copied().enumerate() {
        let simplified_mesh = simplify_partitions(
            mesh,
            source_positions,
            &source_indices,
            target as usize,
            source_triangles,
        )?;
        let level_asset_id = derive_lod_asset_id(asset_id, level as u32, target, settings);
        if !identities.insert(level_asset_id) {
            return Err(StaticVirtualGeometryLodError::IdentityCollision(
                level_asset_id,
            ));
        }
        let asset = build_static_virtual_geometry(level_asset_id, &simplified_mesh, settings)
            .map_err(StaticVirtualGeometryLodError::Build)?;
        let simplified_positions = positions(&simplified_mesh)?;
        output.push(StaticVirtualGeometryLod {
            level: level as u32,
            target_triangles: target,
            asset_id: level_asset_id,
            vertex_error: vertex_surface_error(
                source_positions,
                &simplified_mesh,
                simplified_positions,
            ),
            asset,
        });
    }
    Ok(output)
}

fn derive_lod_asset_id(
    source_asset_id: u32,
    level: u32,
    target_triangles: u32,
    settings: StaticClusterBuildSettings,
) -> u32 {
    let mut bytes = Vec::with_capacity(4 + 4 + 4 + 9);
    bytes.extend_from_slice(b"PVGLOD01");
    bytes.extend_from_slice(&source_asset_id.to_le_bytes());
    bytes.extend_from_slice(&level.to_le_bytes());
    bytes.extend_from_slice(&target_triangles.to_le_bytes());
    bytes.extend_from_slice(&settings.page_size.to_le_bytes());
    bytes.extend_from_slice(&settings.max_vertices_per_cluster.to_le_bytes());
    bytes.extend_from_slice(&settings.max_triangles_per_cluster.to_le_bytes());
    bytes.push(settings.compression as u8);
    let mut asset_id = u32::from_le_bytes(blake3::hash(&bytes).as_bytes()[..4].try_into().unwrap());
    if asset_id == 0 {
        asset_id = 1;
    }
    asset_id
}

fn positions(mesh: &Mesh) -> Result<&[[f32; 3]], StaticVirtualGeometryLodError> {
    mesh.attributes()
        .find(|(attribute, _)| attribute.name == POSITION_ATTRIBUTE)
        .and_then(|(_, values)| values.as_positions())
        .ok_or(StaticVirtualGeometryLodError::InvalidMesh(
            MeshError::NoAttributes,
        ))
}

fn simplify_partitions(
    source: &Mesh,
    positions: &[[f32; 3]],
    indices: &[u32],
    target_triangles: usize,
    source_triangles: usize,
) -> Result<Mesh, StaticVirtualGeometryLodError> {
    let mut ranges: Vec<(u16, u32, u32)> = if source.submeshes().is_empty() {
        vec![(0, 0, indices.len() as u32)]
    } else {
        source
            .submeshes()
            .iter()
            .map(|submesh| {
                (
                    submesh.material_slot,
                    submesh.index_range.start,
                    submesh.index_range.end,
                )
            })
            .collect()
    };
    ranges.sort_by_key(|range| range.1);
    let mut output_positions = Vec::new();
    let mut output_indices: Vec<u32> = Vec::new();
    let mut output_submeshes = Vec::new();
    for (material_slot, start, end) in ranges {
        let mut remap = BTreeMap::new();
        let mut local_positions = Vec::new();
        let mut local_indices = Vec::new();
        for &global in &indices[start as usize..end as usize] {
            let local = if let Some(local) = remap.get(&global) {
                *local
            } else {
                let local = local_positions.len() as u32;
                remap.insert(global, local);
                local_positions.push(positions[global as usize]);
                local
            };
            local_indices.push(local);
        }
        let local_triangles = local_indices.len() / 3;
        let local_target = if target_triangles >= source_triangles {
            local_triangles
        } else {
            ((target_triangles * local_triangles) / source_triangles)
                .max(1)
                .min(local_triangles)
        };
        let local_mesh = TriangleMesh::new(
            local_positions,
            Vec::new(),
            Vec::new(),
            local_indices
                .chunks_exact(3)
                .map(|triangle| [triangle[0], triangle[1], triangle[2]])
                .collect(),
        )
        .map_err(|_| StaticVirtualGeometryLodError::UnsupportedSource)?;
        let simplified = decimate(&local_mesh, local_target)
            .map_err(StaticVirtualGeometryLodError::Decimation)?;
        let vertex_base = output_positions.len() as u32;
        output_positions.extend_from_slice(simplified.positions());
        let index_start = output_indices.len() as u32;
        output_indices.extend(
            simplified
                .indices()
                .iter()
                .flat_map(|triangle| triangle.iter().map(|index| vertex_base + *index)),
        );
        output_submeshes.push(prism_mesh::SubMesh::new(
            material_slot,
            index_start..output_indices.len() as u32,
        ));
    }
    let mut output = Mesh::new(PrimitiveTopology::TriangleList)
        .with_attribute(
            prism_mesh::MeshVertexAttribute::new(POSITION_ATTRIBUTE, VertexFormat::Float32x3, 0),
            prism_mesh::VertexAttributeValues::Float32x3(output_positions),
        )
        .map_err(StaticVirtualGeometryLodError::InvalidMesh)?;
    output.with_indices(prism_mesh::Indices::Uint32(output_indices));
    output.with_submeshes(output_submeshes);
    output
        .validate()
        .map_err(StaticVirtualGeometryLodError::InvalidMesh)?;
    Ok(output)
}

fn vertex_surface_error(
    source: &[[f32; 3]],
    simplified_mesh: &Mesh,
    simplified: &[[f32; 3]],
) -> f32 {
    let triangles = simplified_mesh
        .indices()
        .map(|indices| {
            indices
                .to_u32()
                .chunks_exact(3)
                .map(|triangle| [triangle[0], triangle[1], triangle[2]])
                .collect::<Vec<_>>()
        })
        .unwrap_or_default();
    source
        .iter()
        .map(|point| {
            triangles
                .iter()
                .map(|triangle| {
                    point_triangle_distance_sq(
                        *point,
                        simplified[triangle[0] as usize],
                        simplified[triangle[1] as usize],
                        simplified[triangle[2] as usize],
                    )
                })
                .fold(f32::INFINITY, f32::min)
        })
        .fold(0.0, f32::max)
        .sqrt()
}

fn point_triangle_distance_sq(p: [f32; 3], a: [f32; 3], b: [f32; 3], c: [f32; 3]) -> f32 {
    let ab = sub(b, a);
    let ac = sub(c, a);
    let ap = sub(p, a);
    let d1 = dot(ab, ap);
    let d2 = dot(ac, ap);
    if d1 <= 0.0 && d2 <= 0.0 {
        return dot(ap, ap);
    }
    let bp = sub(p, b);
    let d3 = dot(ab, bp);
    let d4 = dot(ac, bp);
    if d3 >= 0.0 && d4 <= d3 {
        return dot(bp, bp);
    }
    let vc = d1 * d4 - d3 * d2;
    if vc <= 0.0 && d1 >= 0.0 && d3 <= 0.0 {
        let t = d1 / (d1 - d3);
        return dot(sub(p, add(a, scale(ab, t))), sub(p, add(a, scale(ab, t))));
    }
    let cp = sub(p, c);
    let d5 = dot(ab, cp);
    let d6 = dot(ac, cp);
    if d6 >= 0.0 && d5 <= d6 {
        return dot(cp, cp);
    }
    let vb = d5 * d2 - d1 * d6;
    if vb <= 0.0 && d2 >= 0.0 && d6 <= 0.0 {
        let t = d2 / (d2 - d6);
        let q = add(a, scale(ac, t));
        let delta = sub(p, q);
        return dot(delta, delta);
    }
    let va = d3 * d6 - d5 * d4;
    if va <= 0.0 && (d4 - d3) >= 0.0 && (d5 - d6) >= 0.0 {
        let edge = sub(c, b);
        let t = (d4 - d3) / ((d4 - d3) + (d5 - d6));
        let q = add(b, scale(edge, t));
        let delta = sub(p, q);
        return dot(delta, delta);
    }
    let normal = cross(ab, ac);
    let distance = dot(ap, normal);
    distance * distance / dot(normal, normal).max(f32::EPSILON)
}

fn sub(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

fn add(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [a[0] + b[0], a[1] + b[1], a[2] + b[2]]
}

fn scale(value: [f32; 3], scalar: f32) -> [f32; 3] {
    [value[0] * scalar, value[1] * scalar, value[2] * scalar]
}

fn dot(a: [f32; 3], b: [f32; 3]) -> f32 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

fn cross(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

#[cfg(test)]
mod tests {
    use alloc::vec;

    use super::*;
    use prism_mesh::primitives::cube;

    #[test]
    fn qem_chain_builds_real_reduced_assets_with_errors() {
        let source = cube([2.0, 2.0, 2.0]).expect("cube");
        let result = build_static_virtual_geometry_lod_chain(
            21,
            &source,
            StaticClusterBuildSettings::default(),
            &[12, 4],
        )
        .expect("LOD chain");
        assert_eq!(result.len(), 2);
        assert_eq!(result[0].level, 0);
        assert_eq!(result[1].target_triangles, 4);
        assert_ne!(result[0].asset_id, result[1].asset_id);
        assert!(result.iter().all(|lod| {
            lod.asset.header.asset_id == lod.asset_id
                && lod
                    .asset
                    .pages
                    .iter()
                    .all(|page| page.record.key.asset == lod.asset_id)
        }));
        assert!(result.iter().all(|lod| lod.vertex_error.is_finite()));
        assert!(result.iter().all(|lod| lod.asset.encode().is_ok()));
    }

    #[test]
    fn qem_chain_simplifies_material_partitions_independently() {
        let mut source = cube([2.0, 2.0, 2.0]).expect("cube");
        source = source.submeshed(vec![
            prism_mesh::SubMesh::new(0, 0..18),
            prism_mesh::SubMesh::new(1, 18..36),
        ]);
        let lods = build_static_virtual_geometry_lod_chain(
            1,
            &source,
            StaticClusterBuildSettings::default(),
            &[4],
        )
        .expect("material partitions simplify independently");
        assert_eq!(lods.len(), 1);
        assert!(lods[0]
            .asset
            .clusters
            .iter()
            .any(|cluster| cluster.material_slot == 0));
        assert!(lods[0]
            .asset
            .clusters
            .iter()
            .any(|cluster| cluster.material_slot == 1));
    }

    #[test]
    fn lod_set_uses_shared_hysteresis_and_prefetch_selection() {
        let source = cube([2.0, 2.0, 2.0]).expect("cube");
        let levels = build_static_virtual_geometry_lod_chain(
            21,
            &source,
            StaticClusterBuildSettings::default(),
            &[12, 4],
        )
        .expect("LOD chain");
        let set = StaticVirtualGeometryLodSet::new(levels).expect("indexed chain");
        let pick = set
            .select(
                LodProjection::from_focal_length_pixels(500.0),
                10.0,
                1.0,
                GeometryLodPolicy {
                    target_error_pixels: 0.5,
                    hysteresis_pixels: 0.1,
                    prefetch_velocity_scale: 2.0,
                },
                None,
            )
            .expect("selects");
        assert!(pick.selection.prefetch_level <= pick.selection.level);
        assert_eq!(pick.display.level, pick.selection.level);
        assert_eq!(pick.prefetch.level, pick.selection.prefetch_level);
    }
}
