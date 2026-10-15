use super::*;

fn grid(size: u32) -> TriangleMesh {
    let mut positions = Vec::new();
    let mut normals = Vec::new();
    let mut uvs = Vec::new();
    for y in 0..=size {
        for x in 0..=size {
            positions.push([x as f32, y as f32, 0.0]);
            normals.push([x as f32 * 0.01, 0.0, 2.0]);
            uvs.push([x as f32 / size as f32, y as f32 / size as f32]);
        }
    }
    let mut triangles = Vec::new();
    for y in 0..size {
        for x in 0..size {
            let a = y * (size + 1) + x;
            let b = a + 1;
            let c = a + size + 1;
            triangles.extend([[a, b, c], [b, c + 1, c]]);
        }
    }
    TriangleMesh::new(positions, normals, uvs, triangles).unwrap()
}

#[test]
fn reduces_grid_with_exact_attribute_provenance_and_deterministic_bytes() {
    let source = grid(10);
    let c = VertexDecimationConstraints {
        preserve_boundary: true,
        channels: vec![DecimationChannel {
            values: source
                .uvs()
                .iter()
                .map(|uv| [uv[0] as f64, uv[1] as f64, 0.0, 0.0])
                .collect(),
            components: 2,
            weight: 0.2,
        }],
        ..Default::default()
    };
    let result = decimate_preserving_vertices(&source, 70, &c).unwrap();
    assert!((70..200).contains(&result.mesh.triangle_count()));
    assert_eq!(
        result,
        decimate_preserving_vertices(&source, 70, &c).unwrap()
    );
    for (out, &original) in result.source_vertices.iter().enumerate() {
        let original = original as usize;
        assert_eq!(result.mesh.positions()[out], source.positions()[original]);
        assert_eq!(result.mesh.normals()[out], source.normals()[original]);
        assert_eq!(result.mesh.uvs()[out], source.uvs()[original]);
    }
    for y in 0..=10 {
        for x in 0..=10 {
            if x == 0 || y == 0 || x == 10 || y == 10 {
                assert!(result.source_vertices.contains(&(y * 11 + x)));
            }
        }
    }
    for triangle in result.mesh.indices() {
        let [a, b, c] = triangle.map(|v| result.mesh.positions()[v as usize]);
        assert!((b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]) > 0.0);
    }
}

#[test]
fn locks_and_categories_report_unattainable_target_without_discarding_surface() {
    let source = grid(4);
    for c in [
        VertexDecimationConstraints {
            locked_vertices: vec![true; 25],
            ..Default::default()
        },
        VertexDecimationConstraints {
            discrete_groups: (0..25).collect(),
            ..Default::default()
        },
    ] {
        let result = decimate_preserving_vertices(&source, 1, &c).unwrap();
        assert_eq!(result.mesh, source);
    }
}

#[test]
fn link_and_duplicate_guards_keep_tetrahedron_closed() {
    let source = TriangleMesh::new(
        vec![
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        vec![],
        vec![],
        vec![[0, 2, 1], [0, 1, 3], [0, 3, 2], [1, 2, 3]],
    )
    .unwrap();
    let result = decimate_preserving_vertices(&source, 1, &Default::default()).unwrap();
    assert_eq!(result.mesh, source);
}

#[test]
fn no_undershoot_and_geometric_degenerates_are_removed() {
    let source = grid(6);
    for target in [1, 3, 15, 71, 72, 100] {
        let result = decimate_preserving_vertices(&source, target, &Default::default()).unwrap();
        assert!(result.mesh.triangle_count() >= target.min(72));
        assert!(result.mesh.triangle_count() <= 72);
    }
    let empty = TriangleMesh::new(
        vec![[0.0; 3], [1.0; 3], [2.0; 3]],
        vec![],
        vec![],
        vec![[0, 1, 2], [0, 0, 1]],
    )
    .unwrap();
    let result = decimate_preserving_vertices(&empty, 1, &Default::default()).unwrap();
    assert_eq!(result.mesh.triangle_count(), 0);
    assert!(result.source_vertices.is_empty());
}

#[test]
fn validates_constraints_and_source_attributes_before_building_state() {
    let source = grid(2);
    assert_eq!(
        decimate_preserving_vertices(&source, 0, &Default::default()),
        Err(DecimationError::ZeroTarget)
    );
    let mut c = VertexDecimationConstraints {
        locked_vertices: vec![false],
        ..Default::default()
    };
    assert!(decimate_preserving_vertices(&source, 1, &c).is_err());
    c.locked_vertices.clear();
    c.channels.push(DecimationChannel {
        values: vec![[0.0; 4]; 9],
        components: 0,
        weight: 1.0,
    });
    for (components, weight, value) in [
        (0, 1.0, 0.0),
        (5, 1.0, 0.0),
        (1, -1.0, 0.0),
        (1, f64::INFINITY, 0.0),
        (1, 1.0, f64::NAN),
        (1, 1.0, f64::MAX),
    ] {
        c.channels[0].components = components;
        c.channels[0].weight = weight;
        c.channels[0].values[0][0] = value;
        assert!(decimate_preserving_vertices(&source, 1, &c).is_err());
    }
    for (positions, normals, uvs) in [
        (vec![[f32::NAN; 3]], vec![], vec![]),
        (vec![[0.0; 3]], vec![[f32::INFINITY; 3]], vec![]),
        (vec![[0.0; 3]], vec![], vec![[f32::NAN; 2]]),
    ] {
        let source = TriangleMesh::new(positions, normals, uvs, vec![]).unwrap();
        assert!(decimate_preserving_vertices(&source, 1, &Default::default()).is_err());
    }
    let duplicate = TriangleMesh::new(
        source.positions().to_vec(),
        vec![],
        vec![],
        vec![[0, 1, 3], [3, 0, 1]],
    )
    .unwrap();
    assert_eq!(
        decimate_preserving_vertices(&duplicate, 1, &Default::default()),
        Err(DecimationError::InvalidConstraints("duplicate face"))
    );
}

#[test]
fn candidates_allow_reverse_endpoint_and_normal_checks_handle_extreme_scale() {
    let source = grid(2);
    let state = Decimator::new(&source);
    let locked = vec![false; source.vertex_count()];
    let masses = vec![1.0; source.vertex_count()];
    assert_eq!(
        score(&state, 0, 1, &locked, &masses, &Default::default())
            .unwrap()
            .keep,
        0
    );
    assert_eq!(
        score(&state, 1, 0, &locked, &masses, &Default::default())
            .unwrap()
            .keep,
        1
    );
    let positions = source
        .positions()
        .iter()
        .map(|p| p.map(|v| v * 1.0e37))
        .collect();
    let huge = TriangleMesh::new(positions, vec![], vec![], source.indices().to_vec()).unwrap();
    let result = decimate_preserving_vertices(&huge, 4, &Default::default()).unwrap();
    assert!((4..8).contains(&result.mesh.triangle_count()));
}
