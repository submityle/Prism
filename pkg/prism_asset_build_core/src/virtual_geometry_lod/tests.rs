use super::*;
use prism_mesh::{Indices, MeshVertexAttribute, SubMesh, VertexAttributeValues as Values};
use prism_render_driver::VertexFormat;

fn grid(n: u32) -> Mesh {
    let mut p = Vec::new();
    let mut uv = Vec::new();
    let mut indices = Vec::new();
    for y in 0..=n {
        for x in 0..=n {
            p.push([x as f32, y as f32, 0.0]);
            uv.push([x as f32 / n as f32, y as f32 / n as f32]);
        }
    }
    for y in 0..n {
        for x in 0..n {
            let a = y * (n + 1) + x;
            indices.extend([a, a + 1, a + n + 2, a, a + n + 2, a + n + 1]);
        }
    }
    let count = p.len();
    Mesh::new(PrimitiveTopology::TriangleList)
        .with_attribute(
            MeshVertexAttribute::new(POSITION_ATTRIBUTE, VertexFormat::Float32x3, 15),
            Values::Float32x3(p),
        )
        .unwrap()
        .with_attribute(
            MeshVertexAttribute::new("Vertex_Normal", VertexFormat::Float32x3, 1),
            Values::Float32x3(vec![[0.0, 0.0, 1.0]; count]),
        )
        .unwrap()
        .with_attribute(
            MeshVertexAttribute::new("Vertex_Uv", VertexFormat::Float32x2, 2),
            Values::Float32x2(uv),
        )
        .unwrap()
        .with_attribute(
            MeshVertexAttribute::new("Vertex_Tangent", VertexFormat::Float32x4, 3),
            Values::Float32x4(vec![[1.0, 0.0, 0.0, 1.0]; count]),
        )
        .unwrap()
        .indexed(Indices::Uint32(indices))
}

fn build(source: &Mesh, targets: &[u32]) -> Vec<StaticVirtualGeometryLod> {
    build_static_virtual_geometry_lod_chain(
        21,
        source,
        StaticClusterBuildSettings::default(),
        targets,
    )
    .unwrap()
}

fn replace_values(source: &Mesh, name: &str, values: Values) -> Mesh {
    let mut output = Mesh::new(source.primitive_topology());
    for (attribute, original) in source.attributes() {
        output
            .insert_attribute(
                *attribute,
                if attribute.name == name {
                    values.clone()
                } else {
                    original.clone()
                },
            )
            .unwrap();
    }
    output.with_indices(source.indices().unwrap().clone());
    output.with_submeshes(source.submeshes().to_vec());
    output
}

fn assert_exact_gather(source: &Mesh, reduced: &Mesh) {
    let original = positions(source).unwrap();
    let remap: Vec<_> = positions(reduced)
        .unwrap()
        .iter()
        .map(|p| original.iter().position(|s| s == p).unwrap() as u32)
        .collect();
    assert_eq!(source.attribute_count(), reduced.attribute_count());
    for ((a, values), (b, gathered)) in source.attributes().zip(reduced.attributes()) {
        assert_eq!(a, b);
        assert_eq!(values.gather(&remap), Some(gathered.clone()));
    }
}

#[test]
fn attributed_grid_really_reduces_and_preserves_chart_boundary() {
    let source = grid(12);
    let policies = attributes::resolve(&source, &StaticLodPolicy::default()).unwrap();
    let reduced = simplify_partitions(&source, 100, &policies).unwrap();
    assert!(reduced.indices().unwrap().len() / 3 < 288);
    assert!(reduced.indices().unwrap().len() / 3 >= 100);
    assert_exact_gather(&source, &reduced);
    let positions = positions(&reduced).unwrap();
    for i in 0..=12 {
        for edge in [
            [0.0, i as f32, 0.0],
            [12.0, i as f32, 0.0],
            [i as f32, 0.0, 0.0],
            [i as f32, 12.0, 0.0],
        ] {
            assert!(positions.contains(&edge));
        }
    }
    let lods = build(&source, &[288, 144, 72]);
    assert_eq!(lods[0].achieved_triangles, 288);
    assert!(lods[1].achieved_triangles < lods[0].achieved_triangles);
    assert!(lods[2].achieved_triangles < lods[1].achieved_triangles);
    assert_eq!(lods[0].source_content_hash, content_hash(&source));
    assert!(lods
        .windows(2)
        .all(|p| p[1].vertex_error >= p[0].vertex_error));
    for lod in &lods {
        assert!(lod.asset.encode().is_ok());
        assert_eq!(
            lod.achieved_triangles as usize,
            lod.asset
                .clusters
                .iter()
                .map(|c| c.triangle_count as usize)
                .sum()
        );
        assert!(
            lod.vertex_error > 0.0,
            "flat chart still has CLST quantization error"
        );
        let page = &lod.asset.pages[0];
        let cluster = &lod.asset.clusters[0];
        let bytes = &page.payload[cluster.payload_offset as usize
            ..(cluster.payload_offset + cluster.payload_size) as usize];
        let payload = crate::decode_cluster_payload(cluster.bounds, bytes).unwrap();
        assert_eq!(payload.position_shader_location, 15);
        assert_eq!(payload.attributes.len(), 3);
    }
    let set = StaticVirtualGeometryLodSet::new(lods).unwrap();
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
        .unwrap();
    assert!(pick.selection.prefetch_level <= pick.selection.level);
    assert_eq!(pick.display.level, pick.selection.level);
}

#[test]
fn every_typed_format_survives_simplification_without_reconstruction() {
    let base = grid(8);
    let p = positions(&base).unwrap().to_vec();
    let count = p.len();
    let mut source = Mesh::new(PrimitiveTopology::TriangleList)
        .with_attribute(
            MeshVertexAttribute::new(POSITION_ATTRIBUTE, VertexFormat::Float32x3, 15),
            Values::Float32x3(p),
        )
        .unwrap()
        .indexed(base.indices().unwrap().clone());
    let attributes = [
        (
            "Float",
            Values::Float32((0..count).map(|i| i as f32 * 0.1).collect()),
        ),
        (
            "Float2",
            Values::Float32x2((0..count).map(|i| [i as f32, -0.0]).collect()),
        ),
        (
            "Float3",
            Values::Float32x3((0..count).map(|i| [0.2, i as f32, -0.1]).collect()),
        ),
        (
            "Float4",
            Values::Float32x4((0..count).map(|i| [i as f32, 0.2, 0.3, 0.4]).collect()),
        ),
        (
            "Half2",
            Values::Float16x2((0..count).map(|i| [0x3c00 + i as u16, 0x8000]).collect()),
        ),
        (
            "Half4",
            Values::Float16x4(
                (0..count)
                    .map(|i| [0x3c00, i as u16, 0x8000, 0x0400])
                    .collect(),
            ),
        ),
        (
            "Id",
            Values::Uint32((0..count).map(|i| 0xfeed_0001 + i as u32).collect()),
        ),
        (
            "Id2",
            Values::Uint32x2((0..count).map(|i| [u32::MAX, i as u32]).collect()),
        ),
        (
            "Id4",
            Values::Uint32x4((0..count).map(|i| [i as u32, u32::MAX, 0, 1]).collect()),
        ),
        (
            "Signed",
            Values::Sint32((0..count).map(|i| i32::MIN + i as i32).collect()),
        ),
        (
            "Unorm",
            Values::Unorm8x4((0..count).map(|i| [i as u8, 0, 128, 255]).collect()),
        ),
        (
            "Snorm",
            Values::Snorm8x4((0..count).map(|i| [i as i8, -128, -127, 127]).collect()),
        ),
    ];
    let mut policy = StaticLodPolicy::default();
    for (location, (name, values)) in attributes.into_iter().enumerate() {
        source
            .insert_attribute(
                MeshVertexAttribute::new(name, values.format(), location as u32),
                values,
            )
            .unwrap();
        policy.attributes.push(LodAttributePolicy {
            name,
            metric: LodAttributeMetric::PreserveOnly,
        });
    }
    let policies = attributes::resolve(&source, &policy).unwrap();
    let reduced = simplify_partitions(&source, 64, &policies).unwrap();
    assert!(reduced.indices().unwrap().len() < source.indices().unwrap().len());
    assert_exact_gather(&source, &reduced);
    let lods = build_static_virtual_geometry_lod_chain_with_policy(
        3,
        &source,
        StaticClusterBuildSettings::default(),
        &[64],
        &policy,
    )
    .unwrap();
    for cluster in &lods[0].asset.clusters {
        let page = lods[0]
            .asset
            .pages
            .iter()
            .find(|p| p.record.key == cluster.page)
            .unwrap();
        let decoded = crate::decode_cluster_payload(
            cluster.bounds,
            &page.payload[cluster.payload_offset as usize
                ..(cluster.payload_offset + cluster.payload_size) as usize],
        )
        .unwrap();
        assert_eq!(decoded.attributes.len(), 12);
    }
}

#[test]
fn categorical_ids_and_tangent_handedness_restrict_collapses() {
    let mut source = grid(8);
    let count = positions(&source).unwrap().len();
    source
        .insert_attribute(
            MeshVertexAttribute::new("Id", VertexFormat::Uint32, 5),
            Values::Uint32((0..count as u32).collect()),
        )
        .unwrap();
    let lods = build(&source, &[32]);
    assert_eq!(
        lods[0].achieved_triangles, 128,
        "unique IDs cannot be merged"
    );
    let source = grid(8);
    let policies = attributes::resolve(&source, &StaticLodPolicy::default()).unwrap();
    let constraints = attributes::constraints(&source, &policies).unwrap();
    assert_eq!(
        constraints
            .channels
            .iter()
            .find(|c| c.components == 3)
            .unwrap()
            .components,
        3
    );
    let changed = replace_values(
        &source,
        "Vertex_Tangent",
        Values::Float32x4(
            (0..count)
                .map(|i| [1.0, 0.0, 0.0, if i % 2 == 0 { 1.0 } else { -1.0 }])
                .collect(),
        ),
    );
    let constraints = attributes::constraints(&changed, &policies).unwrap();
    assert_ne!(
        constraints.discrete_groups[0],
        constraints.discrete_groups[1]
    );
    let policy = StaticLodPolicy {
        attributes: vec![LodAttributePolicy {
            name: "Vertex_Tangent",
            metric: LodAttributeMetric::PreserveOnly,
        }],
    };
    let resolved = attributes::resolve(&changed, &policy).unwrap();
    let preserved = attributes::constraints(&changed, &resolved).unwrap();
    assert_ne!(preserved.discrete_groups[0], preserved.discrete_groups[1]);
}

#[test]
fn material_partitions_and_split_cube_seams_are_not_opened() {
    let mut source = grid(8);
    let count = source.indices().unwrap().len() as u32;
    source.with_submeshes(vec![
        SubMesh::new(0, 0..count / 2),
        SubMesh::new(2, count / 2..count),
    ]);
    let policies = attributes::resolve(&source, &StaticLodPolicy::default()).unwrap();
    let reduced = simplify_partitions(&source, 64, &policies).unwrap();
    assert_eq!(reduced.submeshes().len(), 2);
    assert_eq!(reduced.submeshes()[1].material_slot, 2);
    let indices = reduced.indices().unwrap().to_u32();
    for submesh in reduced.submeshes() {
        let p = positions(&reduced).unwrap();
        let used: Vec<_> = indices
            [submesh.index_range.start as usize..submesh.index_range.end as usize]
            .iter()
            .map(|i| p[*i as usize])
            .collect();
        for x in 0..=8 {
            assert!(used.contains(&[x as f32, 4.0, 0.0]));
        }
    }
    let cube = prism_mesh::primitives::cube([2.0; 3]).unwrap();
    let cube_lods = build(&cube, &[12, 4]);
    assert_eq!(
        cube_lods[1].achieved_triangles, 12,
        "split face edges are intentionally protected"
    );
    assert!(StaticVirtualGeometryLodSet::new(cube_lods).is_ok());
}

#[test]
fn coincident_copies_are_locked_across_charts_including_signed_zero() {
    let source = Mesh::new(PrimitiveTopology::TriangleList)
        .with_attribute(
            MeshVertexAttribute::new(POSITION_ATTRIBUTE, VertexFormat::Float32x3, 0),
            Values::Float32x3(vec![[0.0; 3], [-0.0, 0.0, -0.0], [1.0, 0.0, 0.0]]),
        )
        .unwrap()
        .indexed(Indices::Uint16(vec![0, 1, 2]));
    let c = attributes::constraints(&source, &BTreeMap::new()).unwrap();
    assert_eq!(c.locked_vertices, [true, true, false]);
}

#[test]
fn target_apportionment_is_exact_deterministic_and_never_drops_a_partition() {
    assert_eq!(partition_targets(&[3, 7, 2], 7), [2, 4, 1]);
    assert_eq!(partition_targets(&[3, 7, 2], 1), [1, 1, 1]);
    assert_eq!(partition_targets(&[3, 7, 2], 99), [3, 7, 2]);
    assert_eq!(partition_targets(&[0, 1, 1], 1), [0, 1, 1]);
    for target in 3..=12 {
        assert_eq!(
            partition_targets(&[3, 7, 2], target).iter().sum::<usize>(),
            target
        );
    }
}

#[test]
fn source_policy_targets_and_settings_are_bound_into_identity() {
    let source = grid(6);
    let original = build(&source, &[72, 40]);
    assert_eq!(original, build(&source, &[72, 40]));
    let targets = build(&source, &[72, 36]);
    assert_ne!(
        original[0].asset_id, targets[0].asset_id,
        "whole chain recipe binds every level"
    );
    let changed = replace_values(
        &source,
        "Vertex_Normal",
        Values::Float32x3(vec![[0.0, 1.0, 0.0]; positions(&source).unwrap().len()]),
    );
    assert_ne!(original[0].asset_id, build(&changed, &[72, 40])[0].asset_id);
    let policy = StaticLodPolicy {
        attributes: vec![
            LodAttributePolicy {
                name: "Vertex_Normal",
                metric: LodAttributeMetric::PreserveOnly,
            },
            LodAttributePolicy {
                name: "Vertex_Uv",
                metric: LodAttributeMetric::Continuous { weight: 0.0 },
            },
        ],
    };
    let a = build_static_virtual_geometry_lod_chain_with_policy(
        21,
        &source,
        StaticClusterBuildSettings::default(),
        &[72, 40],
        &policy,
    )
    .unwrap();
    assert_ne!(
        original[0].asset.header.build_settings_hash,
        a[0].asset.header.build_settings_hash
    );
    let mut reordered = policy.clone();
    reordered.attributes.reverse();
    assert_eq!(
        a,
        build_static_virtual_geometry_lod_chain_with_policy(
            21,
            &source,
            StaticClusterBuildSettings::default(),
            &[72, 40],
            &reordered
        )
        .unwrap()
    );
    if let LodAttributeMetric::Continuous { weight } = &mut reordered.attributes[0].metric {
        *weight = -0.0;
    }
    assert_eq!(
        a,
        build_static_virtual_geometry_lod_chain_with_policy(
            21,
            &source,
            StaticClusterBuildSettings::default(),
            &[72, 40],
            &reordered
        )
        .unwrap()
    );
    let settings = StaticClusterBuildSettings {
        max_vertices_per_cluster: 32,
        ..Default::default()
    };
    assert_ne!(
        original[0].asset_id,
        build_static_virtual_geometry_lod_chain(21, &source, settings, &[72, 40]).unwrap()[0]
            .asset_id
    );
}

#[test]
fn invalid_policy_targets_partial_coverage_and_empty_surfaces_are_typed_errors() {
    let source = grid(4);
    for name in [POSITION_ATTRIBUTE, "Unknown"] {
        let policy = StaticLodPolicy {
            attributes: vec![LodAttributePolicy {
                name,
                metric: LodAttributeMetric::PreserveOnly,
            }],
        };
        assert_eq!(
            attributes::resolve(&source, &policy),
            Err(StaticVirtualGeometryLodError::InvalidPolicy(name))
        );
    }
    for weight in [f64::NAN, f64::INFINITY, -1.0] {
        let policy = StaticLodPolicy {
            attributes: vec![LodAttributePolicy {
                name: "Vertex_Uv",
                metric: LodAttributeMetric::Continuous { weight },
            }],
        };
        assert!(attributes::resolve(&source, &policy).is_err());
    }
    let policy = StaticLodPolicy {
        attributes: vec![
            LodAttributePolicy {
                name: "Vertex_Uv",
                metric: LodAttributeMetric::Categorical
            };
            2
        ],
    };
    assert!(attributes::resolve(&source, &policy).is_err());
    for targets in [&[][..], &[0][..], &[33][..], &[8, 32][..]] {
        assert_eq!(
            build_static_virtual_geometry_lod_chain(1, &source, Default::default(), targets),
            Err(StaticVirtualGeometryLodError::InvalidTarget)
        );
    }
    let partial = source.submeshed(vec![SubMesh::new(0, 0..3)]);
    assert_eq!(
        build_static_virtual_geometry_lod_chain(1, &partial, Default::default(), &[16]),
        Err(StaticVirtualGeometryLodError::Build(
            StaticClusterBuildError::SubmeshCoverage
        ))
    );
    let degenerate = Mesh::new(PrimitiveTopology::TriangleList)
        .with_attribute(
            MeshVertexAttribute::new(POSITION_ATTRIBUTE, VertexFormat::Float32x3, 0),
            Values::Float32x3(vec![[0.0; 3]; 3]),
        )
        .unwrap()
        .indexed(Indices::Uint16(vec![0, 1, 2]));
    assert_eq!(
        build_static_virtual_geometry_lod_chain(1, &degenerate, Default::default(), &[1]),
        Err(StaticVirtualGeometryLodError::EmptySurface)
    );
}

#[test]
fn lod_set_rejects_mutated_counts_identity_and_metric_metadata() {
    let source = grid(4);
    let original = build(&source, &[32, 24]);
    assert!(StaticVirtualGeometryLodSet::new(original.clone()).is_ok());
    type Mutation = fn(&mut Vec<StaticVirtualGeometryLod>);
    let mutations: &[Mutation] = &[
        |v| v[0].vertex_error = f32::NAN,
        |v| v[0].vertex_error = -1.0,
        |v| v[0].measured_vertex_error = f32::NAN,
        |v| v[0].measured_vertex_error = -1.0,
        |v| v[0].vertex_error = 0.0,
        |v| v[1].level = 4,
        |v| v[1].level = 0,
        |v| v[1].target_triangles = 33,
        |v| v[0].achieved_triangles += 1,
        |v| v[1].source_content_hash.0[0] ^= 1,
        |v| v[1].chain_build_hash[0] ^= 1,
        |v| v[0].asset_id ^= 1,
        |v| v[0].asset.header.build_settings_hash[0] ^= 1,
    ];
    for mutate in mutations {
        let mut changed = original.clone();
        mutate(&mut changed);
        assert!(StaticVirtualGeometryLodSet::new(changed).is_err());
    }
    let mut reversed = original.clone();
    reversed.reverse();
    assert_eq!(
        StaticVirtualGeometryLodSet::new(original),
        StaticVirtualGeometryLodSet::new(reversed)
    );
}
