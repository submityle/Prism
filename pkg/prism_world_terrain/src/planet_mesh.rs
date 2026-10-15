//! Origin-relative cube-sphere geometry derived from baked terrain heights.

use alloc::vec::Vec;
use core::convert::Infallible;

use crate::{BakeError, BakeLimits, CubeFace, HeightfieldTile, TileKey};

/// Cube-sphere placement with a checked bound on numeric position error.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PlanetMeshConfig {
    radius_mm: i64,
    origin_mm: [i64; 3],
    max_error_mm: f64,
}

impl PlanetMeshConfig {
    /// Creates a placement contract. Geometry exceeding the error budget fails.
    ///
    /// Radii are limited to 2^52 millimetres so integer radius conversion is exact.
    /// Error bounds cover floating arithmetic and local f32 encoding, not source
    /// height quantization, surface approximation between vertices, or LOD seams.
    pub fn new(
        radius_mm: i64,
        origin_mm: [i64; 3],
        max_error_mm: f64,
    ) -> Result<Self, BakeError<Infallible>> {
        if !(1..=1_i64 << 52).contains(&radius_mm)
            || !max_error_mm.is_finite()
            || max_error_mm <= 0.0
        {
            return Err(BakeError::InvalidGeometry);
        }
        Ok(Self {
            radius_mm,
            origin_mm,
            max_error_mm,
        })
    }

    /// Origin retained as integer millimetres; add it after local-space queries.
    pub const fn origin_mm(self) -> [i64; 3] {
        self.origin_mm
    }

    /// Spherical reference radius in millimetres.
    pub const fn radius_mm(self) -> i64 {
        self.radius_mm
    }

    /// Allowed per-coordinate arithmetic and encoding error in millimetres.
    pub const fn max_error_mm(self) -> f64 {
        self.max_error_mm
    }
}

/// Indexed planet patch. Positions are origin-relative metres, axes stay global.
#[derive(Clone, Debug, PartialEq)]
pub struct PlanetHeightfieldMesh {
    config: PlanetMeshConfig,
    error_bound_mm: f64,
    positions: Vec<[f32; 3]>,
    normals: Vec<[f32; 3]>,
    uvs: Vec<[f32; 2]>,
    indices: Vec<u32>,
}

impl PlanetHeightfieldMesh {
    /// Projects height samples to a sphere, with outward CCW triangle winding.
    ///
    /// Tile coordinates remain f64 through projection, including level 30.
    /// No temporary full-sized position array is allocated. Payload admission
    /// uses 32 bytes per vertex plus 24 per quad, before any allocation.
    /// Boundary normals are one-sided geometric derivatives; cross-tile smooth
    /// normals and mixed-LOD stitching require neighboring source samples.
    pub fn build(
        field: &HeightfieldTile,
        config: PlanetMeshConfig,
        limits: BakeLimits,
    ) -> Result<Self, BakeError<Infallible>> {
        let count = field.samples().len();
        if count > u32::MAX as usize {
            return Err(BakeError::SizeOverflow);
        }
        let side = field.spec().samples_per_edge();
        let indices = usize::try_from(side - 1)
            .ok()
            .and_then(|n| n.checked_mul(n))
            .and_then(|n| n.checked_mul(6))
            .ok_or(BakeError::SizeOverflow)?;
        let bytes = count
            .checked_mul(32)
            .and_then(|n| indices.checked_mul(4).and_then(|i| n.checked_add(i)))
            .ok_or(BakeError::SizeOverflow)?;
        if count > limits.max_samples || bytes > limits.max_bytes {
            return Err(BakeError::BudgetExceeded);
        }
        let mut mesh = Self {
            config,
            error_bound_mm: 0.0,
            positions: reserve(count)?,
            normals: reserve(count)?,
            uvs: reserve(count)?,
            indices: reserve(indices)?,
        };
        for y in 0..side {
            for x in 0..side {
                let absolute = point(field, config.radius_mm, x, y)?;
                let mut position = [0.0; 3];
                for axis in 0..3 {
                    let local_mm = absolute[axis] - config.origin_mm[axis] as f64;
                    position[axis] = (local_mm * 0.001) as f32;
                    // A deliberately conservative operation-count bound covers
                    // rational UV, normalization, radius/height and subtraction.
                    let radial_mm = (i128::from(config.radius_mm)
                        + field.height_mm([x, y]).ok_or(BakeError::InvalidGeometry)?)
                        as f64;
                    let arithmetic = 128.0
                        * f64::EPSILON
                        * (absolute[axis].abs()
                            + (config.origin_mm[axis] as f64).abs()
                            + radial_mm.abs()
                            + 1.0);
                    let error = (f64::from(position[axis]) * 1000.0 - local_mm).abs() + arithmetic;
                    if !position[axis].is_finite() || error > config.max_error_mm {
                        return Err(BakeError::InvalidGeometry);
                    }
                    mesh.error_bound_mm = mesh.error_bound_mm.max(error);
                }
                let left = point(field, config.radius_mm, x.saturating_sub(1), y)?;
                let right = point(field, config.radius_mm, (x + 1).min(side - 1), y)?;
                let down = point(field, config.radius_mm, x, y.saturating_sub(1))?;
                let up = point(field, config.radius_mm, x, (y + 1).min(side - 1))?;
                let tangent_u = subtract(right, left);
                let tangent_v = subtract(up, down);
                let normal = cross(tangent_u, tangent_v);
                let length = libm::sqrt(normal.iter().map(|v| v * v).sum());
                if !length.is_finite() || length <= 0.0 {
                    return Err(BakeError::InvalidGeometry);
                }
                mesh.positions.push(position);
                mesh.normals.push(normal.map(|v| (v / length) as f32));
                mesh.uvs
                    .push([x as f32 / (side - 1) as f32, y as f32 / (side - 1) as f32]);
            }
        }
        for y in 0..side - 1 {
            for x in 0..side - 1 {
                let a = y * side + x;
                let b = a + 1;
                let c = a + side;
                mesh.indices.extend_from_slice(&[a, b, c, b, c + 1, c]);
            }
        }
        Ok(mesh)
    }

    /// Local position payload in metres.
    pub fn positions(&self) -> &[[f32; 3]] {
        &self.positions
    }
    /// Outward geometric unit normals in global-axis orientation.
    pub fn normals(&self) -> &[[f32; 3]] {
        &self.normals
    }
    /// Tile-local normalized UVs.
    pub fn uvs(&self) -> &[[f32; 2]] {
        &self.uvs
    }
    /// Outward triangle indices.
    pub fn indices(&self) -> &[u32] {
        &self.indices
    }
    /// Integer-origin placement contract.
    pub const fn config(&self) -> PlanetMeshConfig {
        self.config
    }
    /// Maximum per-coordinate arithmetic/encoding error bound in millimetres.
    pub const fn error_bound_mm(&self) -> f64 {
        self.error_bound_mm
    }

    /// Converts into a native Prism render mesh while preserving local positions.
    #[cfg(feature = "mesh")]
    pub fn into_mesh(self) -> Result<prism_mesh::Mesh, prism_mesh::MeshError> {
        use prism_mesh::{Indices, Mesh, MeshVertexAttribute, VertexAttributeValues};
        use prism_render_driver::VertexFormat;
        let mesh = Mesh::new(prism_render_driver::PrimitiveTopology::TriangleList)
            .with_attribute(
                MeshVertexAttribute::new(
                    prism_mesh::POSITION_ATTRIBUTE,
                    VertexFormat::Float32x3,
                    0,
                ),
                VertexAttributeValues::Float32x3(self.positions),
            )?
            .with_attribute(
                MeshVertexAttribute::new("Vertex_Normal", VertexFormat::Float32x3, 1),
                VertexAttributeValues::Float32x3(self.normals),
            )?
            .with_attribute(
                MeshVertexAttribute::new("Vertex_Uv0", VertexFormat::Float32x2, 2),
                VertexAttributeValues::Float32x2(self.uvs),
            )?
            .indexed(Indices::Uint32(self.indices));
        mesh.validate()?;
        Ok(mesh)
    }
}

fn reserve<T>(count: usize) -> Result<Vec<T>, BakeError<Infallible>> {
    let mut values = Vec::new();
    values
        .try_reserve_exact(count)
        .map_err(|_| BakeError::AllocationFailed)?;
    Ok(values)
}

fn point(
    field: &HeightfieldTile,
    radius: i64,
    x: u32,
    y: u32,
) -> Result<[f64; 3], BakeError<Infallible>> {
    let tile = field.tile();
    let [u, v] = face_coordinate(tile, field.spec().samples_per_edge(), x, y);
    let cube = match tile.face() {
        CubeFace::PositiveX => [1.0, v, -u],
        CubeFace::NegativeX => [-1.0, v, u],
        CubeFace::PositiveY => [u, 1.0, -v],
        CubeFace::NegativeY => [u, -1.0, v],
        CubeFace::PositiveZ => [u, v, 1.0],
        CubeFace::NegativeZ => [-u, v, -1.0],
    };
    let height = field.height_mm([x, y]).ok_or(BakeError::InvalidGeometry)?;
    let radial_mm = i128::from(radius) + height;
    if radial_mm <= 0 {
        return Err(BakeError::InvalidGeometry);
    }
    let length = libm::sqrt(1.0 + (u * u + v * v));
    Ok(cube.map(|v| v / length * radial_mm as f64))
}

fn face_coordinate(tile: TileKey, side: u32, x: u32, y: u32) -> [f64; 2] {
    let tiles = f64::from(tile.side_len());
    let segments = f64::from(side - 1);
    [
        2.0 * (f64::from(tile.x()) + f64::from(x) / segments) / tiles - 1.0,
        2.0 * (f64::from(tile.y()) + f64::from(y) / segments) / tiles - 1.0,
    ]
}

fn subtract(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}
fn cross(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{HeightSample, HeightSampleValue, HeightSampler, HeightfieldSpec};
    struct Constant(i64);
    impl HeightSampler for Constant {
        type Error = Infallible;
        fn sample_height(&self, _: HeightSample) -> Result<HeightSampleValue, Self::Error> {
            Ok(HeightSampleValue::from_millimetres(self.0))
        }
    }
    const LIMITS: BakeLimits = BakeLimits {
        max_samples: 1024,
        max_bytes: 100_000,
    };
    fn field(tile: TileKey, height: i64) -> HeightfieldTile {
        HeightfieldTile::bake(
            tile,
            HeightfieldSpec::new(9, 1).unwrap(),
            &Constant(height),
            LIMITS,
        )
        .unwrap()
    }

    #[test]
    fn all_faces_have_outward_winding_and_unit_normals() {
        for face in CubeFace::ALL {
            let mesh = PlanetHeightfieldMesh::build(
                &field(TileKey::root(face), 0),
                PlanetMeshConfig::new(1_000_000, [0; 3], 1.0).unwrap(),
                LIMITS,
            )
            .unwrap();
            for (p, n) in mesh.positions().iter().zip(mesh.normals()) {
                assert!(p.iter().zip(n).map(|(a, b)| a * b).sum::<f32>() > 0.0);
                assert!((n.iter().map(|v| v * v).sum::<f32>() - 1.0).abs() < 1e-6);
            }
            for tri in mesh.indices().chunks_exact(3) {
                let [a, b, c] =
                    [tri[0], tri[1], tri[2]].map(|i| mesh.positions()[i as usize].map(f64::from));
                assert!(
                    cross(subtract(b, a), subtract(c, a))
                        .iter()
                        .zip(a)
                        .map(|(n, p)| n * p)
                        .sum::<f64>()
                        > 0.0
                );
            }
        }
    }

    #[test]
    fn level_thirty_earth_patch_retains_submillimetre_detail() {
        let tile = TileKey::new(CubeFace::PositiveZ, 30, 1 << 29, 1 << 29).unwrap();
        let config = PlanetMeshConfig::new(6_371_000_000, [0, 0, 6_371_000_000], 0.001).unwrap();
        let mesh = PlanetHeightfieldMesh::build(&field(tile, 0), config, LIMITS).unwrap();
        assert!(mesh.positions()[8][0] > 0.01);
        for row in mesh.positions().chunks_exact(9) {
            assert!(row.windows(2).all(|pair| pair[1][0] > pair[0][0]));
        }
        assert!(mesh.error_bound_mm() < 0.001);
    }

    #[test]
    fn adjacent_tiles_and_cube_faces_share_exact_boundary_positions() {
        let config = PlanetMeshConfig::new(1_000_000, [0; 3], 1.0).unwrap();
        let a = PlanetHeightfieldMesh::build(
            &field(TileKey::new(CubeFace::PositiveZ, 1, 0, 0).unwrap(), 0),
            config,
            LIMITS,
        )
        .unwrap();
        let b = PlanetHeightfieldMesh::build(
            &field(TileKey::new(CubeFace::PositiveZ, 1, 1, 0).unwrap(), 0),
            config,
            LIMITS,
        )
        .unwrap();
        for row in 0..9 {
            assert_eq!(a.positions()[row * 9 + 8], b.positions()[row * 9]);
        }
        let z = PlanetHeightfieldMesh::build(
            &field(TileKey::root(CubeFace::PositiveZ), 0),
            config,
            LIMITS,
        )
        .unwrap();
        let x = PlanetHeightfieldMesh::build(
            &field(TileKey::root(CubeFace::PositiveX), 0),
            config,
            LIMITS,
        )
        .unwrap();
        for row in 0..9 {
            assert_eq!(z.positions()[row * 9 + 8], x.positions()[row * 9]);
        }
    }

    #[test]
    fn rejects_encoding_error_budget_nonpositive_radius_and_payload_budget() {
        let f = field(TileKey::root(CubeFace::PositiveZ), 0);
        let tight = PlanetMeshConfig::new(6_371_000_000, [0; 3], 0.001).unwrap();
        assert_eq!(
            PlanetHeightfieldMesh::build(&f, tight, LIMITS),
            Err(BakeError::InvalidGeometry)
        );
        assert!(PlanetMeshConfig::new(0, [0; 3], 1.0).is_err());
        assert!(PlanetMeshConfig::new((1_i64 << 52) + 1, [0; 3], 1.0).is_err());
        let config = PlanetMeshConfig::new(100, [0; 3], 1.0).unwrap();
        assert_eq!(
            PlanetHeightfieldMesh::build(&field(f.tile(), -100), config, LIMITS),
            Err(BakeError::InvalidGeometry)
        );
        assert_eq!(
            PlanetHeightfieldMesh::build(
                &f,
                config,
                BakeLimits {
                    max_samples: 80,
                    ..LIMITS
                }
            ),
            Err(BakeError::BudgetExceeded)
        );
        let exact = 81 * 32 + 64 * 24;
        assert_eq!(
            PlanetHeightfieldMesh::build(
                &f,
                config,
                BakeLimits {
                    max_bytes: exact - 1,
                    ..LIMITS
                }
            ),
            Err(BakeError::BudgetExceeded)
        );
        assert!(PlanetHeightfieldMesh::build(
            &f,
            config,
            BakeLimits {
                max_bytes: exact,
                ..LIMITS
            }
        )
        .is_ok());
    }

    #[cfg(feature = "mesh")]
    #[test]
    fn projected_payload_is_a_valid_native_mesh() {
        let projected = PlanetHeightfieldMesh::build(
            &field(TileKey::new(CubeFace::NegativeY, 8, 128, 128).unwrap(), 12),
            PlanetMeshConfig::new(6_371_000_000, [0, -6_371_000_000, 0], 10.0).unwrap(),
            LIMITS,
        )
        .unwrap();
        let native = projected.into_mesh().unwrap();
        native.validate().unwrap();
        assert_eq!(native.vertex_count(), 81);
        assert_eq!(native.indices().unwrap().len(), 384);
    }
}
