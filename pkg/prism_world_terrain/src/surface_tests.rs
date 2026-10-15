use super::*;
use alloc::{collections::BTreeMap, vec::Vec};

struct Field<F>([u32; 3], F);

impl<F: Fn([u32; 3]) -> i32> SdfField for Field<F> {
    type Error = core::convert::Infallible;
    fn spec(&self) -> SdfSpec {
        SdfSpec::new(self.0, 2, 1000).unwrap()
    }
    fn sample_q16(&self, p: [u32; 3]) -> Result<i32, Self::Error> {
        Ok((self.1)(p))
    }
}

fn grid(dimensions: [u32; 3], f: impl Fn([u32; 3]) -> i32) -> SdfGrid {
    SdfGrid::bake(
        &Field(dimensions, f),
        VolumeBakeLimits {
            max_dimensions: dimensions,
            max_samples: usize::MAX,
            max_bytes: usize::MAX,
        },
    )
    .unwrap()
}

fn limits() -> SurfaceLimits {
    SurfaceLimits {
        max_cells: usize::MAX,
        max_triangles: usize::MAX,
        max_vertices: usize::MAX,
        max_bytes: usize::MAX,
    }
}

fn geometry() -> SurfaceGeometry {
    SurfaceGeometry::new([0; 3], [0; 3], [1000; 3], 0.01).unwrap()
}

fn check(mesh: &SdfSurfaceMesh) {
    assert_eq!(mesh.positions().len(), mesh.normals().len());
    assert_eq!(mesh.indices().len(), mesh.cost().triangles * 3);
    assert_eq!(mesh.positions().len(), mesh.cost().vertices);
    for normal in mesh.normals() {
        let length: f64 = normal.iter().map(|n| f64::from(*n).powi(2)).sum();
        assert!((length - 1.0).abs() < 2e-6);
    }
    for triangle in mesh.indices().chunks_exact(3) {
        let p = [triangle[0], triangle[1], triangle[2]].map(|i| mesh.positions()[i as usize]);
        let a: [f64; 3] = core::array::from_fn(|i| f64::from(p[1][i]) - f64::from(p[0][i]));
        let b: [f64; 3] = core::array::from_fn(|i| f64::from(p[2][i]) - f64::from(p[0][i]));
        let c = [
            a[1] * b[2] - a[2] * b[1],
            a[2] * b[0] - a[0] * b[2],
            a[0] * b[1] - a[1] * b[0],
        ];
        assert!(c.iter().map(|v| v * v).sum::<f64>() > 0.0);
    }
}

#[test]
fn all_cube_sign_masks_and_distance_extremes() {
    for (negative, positive) in [(-1, 1), (i32::MIN, i32::MAX)] {
        for mask in 0_u16..256 {
            let field = grid([2; 3], |p| {
                let bit = p[0] + 2 * p[1] + 4 * p[2];
                if mask & (1 << bit) != 0 {
                    negative
                } else {
                    positive
                }
            });
            let mesh = SdfSurfaceMesh::build(&field, geometry(), limits()).unwrap();
            check(&mesh);
            assert_eq!(mesh.indices().is_empty(), mask == 0 || mask == 255);
            assert!(mesh.cost().triangles <= 12);
        }
    }
}

#[test]
fn planes_have_outward_normals_with_anisotropic_spacing() {
    for axis in 0..3 {
        for sign in [-1, 1] {
            let field = grid([4; 3], |p| sign * (p[axis] as i32 * 65536 - 98304));
            let spacing = [750, 2000, 1250];
            let embedding = SurfaceGeometry::new([0; 3], [0; 3], spacing, 0.01).unwrap();
            let mesh = SdfSurfaceMesh::build(&field, embedding, limits()).unwrap();
            check(&mesh);
            for (p, n) in mesh.positions().iter().zip(mesh.normals()) {
                assert!((f64::from(p[axis]) - 1.5 * f64::from(spacing[axis]) * 0.001).abs() < 1e-6);
                assert!((n[axis] - sign as f32).abs() < 1e-6);
                for i in 0..3 {
                    if i != axis {
                        assert!(n[i].abs() < 1e-6);
                    }
                }
            }
        }
    }
}

#[test]
fn zero_lattice_vertices_are_welded_and_degenerate_faces_removed() {
    let field = grid([3; 3], |p| (p[0] as i32 - 1) * 65536);
    let mesh = SdfSurfaceMesh::build(&field, geometry(), limits()).unwrap();
    check(&mesh);
    assert_eq!(mesh.cost().vertices, 9);
    assert_eq!(mesh.cost().triangles, 8);
    assert!(mesh.positions().iter().all(|p| p[0] == 1.0));
    assert!(mesh.normals().iter().all(|n| *n == [1.0, 0.0, 0.0]));
    for value in [-1, 0, 1] {
        let field = grid([3; 3], |_| value);
        let empty = SdfSurfaceMesh::build(&field, geometry(), limits()).unwrap();
        assert_eq!(empty.cost().peak_bytes, 0);
    }
}

fn sphere(p: [u32; 3]) -> i32 {
    let radius = libm::sqrt(p.into_iter().map(|c| (f64::from(c) - 4.0).powi(2)).sum());
    ((radius - 2.7) * 65536.0) as i32
}

#[test]
fn closed_sphere_is_an_oriented_two_manifold() {
    let field = grid([9; 3], sphere);
    let mesh = SdfSurfaceMesh::build(&field, geometry(), limits()).unwrap();
    check(&mesh);
    let mut edges = BTreeMap::<[u32; 2], (usize, i32)>::new();
    for t in mesh.indices().chunks_exact(3) {
        for [a, b] in [[t[0], t[1]], [t[1], t[2]], [t[2], t[0]]] {
            let (count, orientation) = edges.entry([a.min(b), a.max(b)]).or_default();
            *count += 1;
            *orientation += if a < b { 1 } else { -1 };
        }
    }
    assert!(edges.values().all(|v| *v == (2, 0)));
    assert_eq!(
        mesh.cost().vertices as isize - edges.len() as isize + mesh.cost().triangles as isize,
        2
    );
    for (p, n) in mesh.positions().iter().zip(mesh.normals()) {
        let outward: f32 = (0..3).map(|i| (p[i] - 4.0) * n[i]).sum();
        assert!(outward > 0.0);
    }
}

fn triangles(mesh: &SdfSurfaceMesh) -> Vec<[[u32; 3]; 3]> {
    mesh.indices()
        .chunks_exact(3)
        .map(|t| {
            let mut p = [t[0], t[1], t[2]].map(|i| mesh.positions()[i as usize].map(f32::to_bits));
            let first = (0..3).min_by_key(|&i| p[i]).unwrap();
            p.rotate_left(first);
            p
        })
        .collect()
}

#[test]
fn arbitrary_cube_partitions_reproduce_bit_exact_oriented_triangles() {
    let field = grid([9; 3], sphere);
    let mesh = SdfSurfaceMesh::build(&field, geometry(), limits()).unwrap();
    let mut full = triangles(&mesh);
    let mut parts = Vec::new();
    for (z0, z1) in [(0, 3), (3, 8)] {
        for (y0, y1) in [(0, 5), (5, 8)] {
            for (x0, x1) in [(0, 2), (2, 8)] {
                let region = VolumeRegion::new([x0, y0, z0], [x1, y1, z1]).unwrap();
                let part =
                    SdfSurfaceMesh::build_region(&field, region, geometry(), limits()).unwrap();
                parts.extend(triangles(&part));
            }
        }
    }
    full.sort_unstable();
    parts.sort_unstable();
    assert_eq!(full, parts);
}

#[test]
fn large_common_origins_cancel_before_float_conversion() {
    let field = grid([3; 3], |p| p[0] as i32 * 65536 - 32768);
    let reference = SdfSurfaceMesh::build(&field, geometry(), limits()).unwrap();
    for origin in [i64::MIN, i64::MAX, -9_000_000_000_000_000_000] {
        let shifted = SurfaceGeometry::new([origin; 3], [origin; 3], [1000; 3], 0.01).unwrap();
        assert_eq!(
            reference,
            SdfSurfaceMesh::build(&field, shifted, limits()).unwrap()
        );
    }
    let shifted = SurfaceGeometry::new([-1000; 3], [0; 3], [1000; 3], 0.01).unwrap();
    let mesh = SdfSurfaceMesh::build(&field, shifted, limits()).unwrap();
    assert!(mesh.positions().iter().all(|p| p[0] == -0.5));
}

#[test]
fn exact_limits_are_admitted_and_failure_preserves_grid() {
    let field = grid([4; 3], |p| p[0] as i32 * 65536 - 98304);
    let before = field.bytes().to_vec();
    let mesh = SdfSurfaceMesh::build(&field, geometry(), limits()).unwrap();
    let c = mesh.cost();
    let exact = SurfaceLimits {
        max_cells: c.cells,
        max_triangles: c.triangles,
        max_vertices: c.vertices,
        max_bytes: c.peak_bytes,
    };
    assert_eq!(
        SdfSurfaceMesh::build(&field, geometry(), exact).unwrap(),
        mesh
    );
    for (budget, error) in [
        (
            SurfaceLimits {
                max_cells: c.cells - 1,
                ..exact
            },
            SurfaceError::CellLimit,
        ),
        (
            SurfaceLimits {
                max_triangles: c.triangles - 1,
                ..exact
            },
            SurfaceError::TriangleLimit,
        ),
        (
            SurfaceLimits {
                max_vertices: c.vertices - 1,
                ..exact
            },
            SurfaceError::VertexLimit,
        ),
        (
            SurfaceLimits {
                max_bytes: c.peak_bytes - 1,
                ..exact
            },
            SurfaceError::ByteLimit,
        ),
    ] {
        assert_eq!(
            SdfSurfaceMesh::build(&field, geometry(), budget),
            Err(error)
        );
        assert_eq!(field.bytes(), before);
        assert_eq!(field.revision(), 0);
    }
}

#[test]
fn rejects_invalid_regions_geometry_and_precision_loss() {
    for spacing in [[0, 1, 1], [1, 0, 1], [1, 1, 0]] {
        assert_eq!(
            SurfaceGeometry::new([0; 3], [0; 3], spacing, 1.0),
            Err(SurfaceError::InvalidGeometry)
        );
    }
    for error in [0.0, -1.0, f64::NAN, f64::INFINITY] {
        assert_eq!(
            SurfaceGeometry::new([0; 3], [0; 3], [1; 3], error),
            Err(SurfaceError::InvalidGeometry)
        );
    }
    let field = grid([2; 3], |p| p[0] as i32 * 3 - 1);
    let region = VolumeRegion::new([0; 3], [2, 1, 1]).unwrap();
    assert_eq!(
        SdfSurfaceMesh::build_region(&field, region, geometry(), limits()),
        Err(SurfaceError::RegionOutOfBounds)
    );
    let tight = SurfaceGeometry::new([0; 3], [0; 3], [1000; 3], 1e-9).unwrap();
    assert_eq!(
        SdfSurfaceMesh::build(&field, tight, limits()),
        Err(SurfaceError::PrecisionLoss)
    );
    let far = SurfaceGeometry::new([i64::MAX; 3], [i64::MIN; 3], [1; 3], 1.0).unwrap();
    assert_eq!(
        SdfSurfaceMesh::build(&field, far, limits()),
        Err(SurfaceError::CoordinateRange)
    );
    let collapsed =
        SurfaceGeometry::new([1_000_000_000_000; 3], [0; 3], [1; 3], 100_000.0).unwrap();
    assert_eq!(
        SdfSurfaceMesh::build(&field, collapsed, limits()),
        Err(SurfaceError::PrecisionLoss)
    );
}

#[test]
fn singleton_axes_need_no_mesh_allocation() {
    for dimensions in [[1, 3, 3], [3, 1, 3], [3, 3, 1]] {
        let field = grid(dimensions, |p| p[0] as i32 - 1);
        let budget = SurfaceLimits {
            max_cells: 0,
            max_triangles: 0,
            max_vertices: 0,
            max_bytes: 0,
        };
        let mesh = SdfSurfaceMesh::build(&field, geometry(), budget).unwrap();
        assert_eq!(mesh.cost(), SurfaceCost::default());
    }
}

#[cfg(feature = "mesh")]
#[test]
fn surface_converts_to_valid_native_mesh() {
    let field = grid([9; 3], sphere);
    let mesh = SdfSurfaceMesh::build(&field, geometry(), limits()).unwrap();
    let native = mesh.into_mesh().unwrap();
    native.validate().unwrap();
}
