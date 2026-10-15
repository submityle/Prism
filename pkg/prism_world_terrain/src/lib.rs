//! Engine-neutral terrain and planetary core primitives for Prism.
//!
//! The default build owns geometry/data contracts and executable baking kernels,
//! with no engine, operating-system, or GPU dependency (`no_std + alloc`). The
//! optional `mesh` adapter produces native Prism mesh assets at the render edge.
#![no_std]
#![forbid(unsafe_code)]
#![warn(missing_docs)]

extern crate alloc;

mod bake;
mod coordinates;
mod cube_sphere;
mod height;
mod heightfield_surface;
#[cfg(test)]
mod heightfield_surface_tests;
mod lod;
mod planet_mesh;
mod rebake;
mod representation;
#[cfg(feature = "scatter")]
mod scatter_deletion;
#[cfg(feature = "scatter")]
mod scatter_filter;
#[cfg(all(test, feature = "scatter"))]
mod scatter_filter_tests;
mod stream_adapter;
mod surface;
#[cfg(test)]
mod surface_tests;
mod tile;
mod volume_bake;
mod voxel;

pub use bake::{BakeError, BakeLimits, GridChange, HeightfieldMesh, HeightfieldTile, SampleBounds};
pub use coordinates::{
    CoordinateError, FixedPoint3, I64Point, VerifiedLocalPoint, FIXED_FRAC_BITS, FIXED_SCALE,
};
pub use cube_sphere::{
    direction_to_face_uv, face_uv_to_direction, CubeFace, CubeSphereError, FaceUv, UnitDirection,
};
pub use height::{DeterministicHeightSampler, HeightSample, HeightSampleValue, HeightSampler};
pub use heightfield_surface::{
    HeightfieldQueryError, HeightfieldSurfaceSample, PlanarHeightfieldEmbedding,
    PlanarHeightfieldSurface, HEIGHTFIELD_SURFACE_VERSION,
};
pub use lod::{evaluate_lod, LodError, LodEvaluation, LodMetric};
pub use planet_mesh::{PlanetHeightfieldMesh, PlanetMeshConfig};
pub use rebake::{
    DirtyRegion, RebakeBudget, RebakeError, RebakePlan, RebakePlanner, RebakeRequest, RebakeTask,
};
pub use representation::{
    HeightfieldSpec, RepresentationChoice, RepresentationError, RepresentationKind,
    RepresentationQuery, RepresentationSpec, RepresentationWork, SdfSpec,
    TerrainRepresentationSelector, VoxelSpec,
};
#[cfg(feature = "scatter")]
pub use scatter_deletion::{filter_heightfield_scatter, DeletionFilteredHeightfieldScatter};
#[cfg(feature = "scatter")]
pub use scatter_filter::{
    scatter_with_heightfield, scatter_with_indexed_heightfield, HeightfieldScatterBudget,
    HeightfieldScatterCacheKey, HeightfieldScatterError, HeightfieldScatterFilters,
    HeightfieldScatterOutput, HEIGHTFIELD_SCATTER_FILTER_VERSION,
};
pub use stream_adapter::{TerrainStreamAdapter, TerrainStreamConfig, TerrainStreamError};
pub use surface::{SdfSurfaceMesh, SurfaceCost, SurfaceError, SurfaceGeometry, SurfaceLimits};
pub use tile::{TileEdge, TileError, TileKey, TileQuadrant, MAX_TILE_LEVEL};
pub use volume_bake::{
    quantize_sdf_mm, SdfField, SdfGrid, SdfGridChange, VolumeBakeCost, VolumeBakeError,
    VolumeBakeLimits, VolumeError, VolumeRegion, VoxelSdfSource,
};
pub use voxel::{
    VoxelBuffer, VoxelComponentFormat, VoxelError, VoxelField, VoxelLayout, VoxelSample,
};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cube_sphere_round_trips_faces_and_poles() {
        for face in CubeFace::ALL {
            for &(u, v) in &[(0.0, 0.0), (0.37, -0.61)] {
                let face_uv = FaceUv::new(face, u, v).expect("valid face coordinate");
                let direction = face_uv_to_direction(face_uv);
                let recovered = direction_to_face_uv(direction.components()).expect("direction");
                assert_eq!(
                    recovered.face(),
                    face,
                    "uv=({u},{v}) dir={:?}",
                    direction.components()
                );
                assert!((recovered.u() - u).abs() < 1.0e-6);
                assert!((recovered.v() - v).abs() < 1.0e-6);
            }
        }

        let north = direction_to_face_uv([0.0, 1.0, 0.0]).expect("north pole");
        let south = direction_to_face_uv([0.0, -1.0, 0.0]).expect("south pole");
        assert_eq!(north.face(), CubeFace::PositiveY);
        assert_eq!(south.face(), CubeFace::NegativeY);
        assert_eq!((north.u(), north.v()), (0.0, 0.0));
        assert_eq!((south.u(), south.v()), (0.0, 0.0));
    }

    #[test]
    fn tile_neighbors_are_reciprocal_across_all_cube_edges() {
        let level = 4;
        let side = 1_u32 << level;
        for face in CubeFace::ALL {
            for edge in TileEdge::ALL {
                for index in [0, 1, side / 2, side - 2, side - 1] {
                    let tile = match edge {
                        TileEdge::West => TileKey::new(face, level, 0, index),
                        TileEdge::East => TileKey::new(face, level, side - 1, index),
                        TileEdge::South => TileKey::new(face, level, index, 0),
                        TileEdge::North => TileKey::new(face, level, index, side - 1),
                    }
                    .expect("valid edge tile");
                    let (neighbor, back_edge) = tile.neighbor_with_edge(edge);
                    assert_eq!(neighbor.neighbor(back_edge), tile);
                }
            }
        }
    }

    #[test]
    fn tile_uv_boundaries_and_quadtree_are_stable() {
        let root = TileKey::root(CubeFace::PositiveZ);
        let children = root.children().expect("root has children");
        assert_eq!(
            children[0],
            TileKey::new(CubeFace::PositiveZ, 1, 0, 0).unwrap()
        );
        assert_eq!(
            children[3],
            TileKey::new(CubeFace::PositiveZ, 1, 1, 1).unwrap()
        );
        assert_eq!(children[2].parent(), Some(root));

        let leaf = TileKey::new(CubeFace::PositiveZ, MAX_TILE_LEVEL, 0, 0).unwrap();
        assert_eq!(leaf.children(), Err(TileError::NoChildren));

        let west =
            TileKey::from_face_uv(FaceUv::new(CubeFace::PositiveZ, -1.0, 0.0).unwrap(), 3).unwrap();
        let east =
            TileKey::from_face_uv(FaceUv::new(CubeFace::PositiveZ, 1.0, 0.0).unwrap(), 3).unwrap();
        assert_eq!(west.x(), 0);
        assert_eq!(east.x(), (1_u32 << 3) - 1);
    }

    #[test]
    fn continuous_lod_has_bounded_morph_alpha() {
        let evaluation = evaluate_lod(LodMetric::new(10.0, 80.0, 100.0, 1.0, 8).unwrap())
            .expect("valid LOD metric");
        assert_eq!(evaluation.base_level(), 3);
        assert!(evaluation.morph_alpha() > 0.0);
        assert!(evaluation.morph_alpha() < 1.0);

        let clamped = evaluate_lod(LodMetric::new(1.0, 10_000.0, 1.0, 1.0, 2).unwrap())
            .expect("valid LOD metric");
        assert_eq!(clamped.base_level(), 0);
        assert_eq!(clamped.morph_alpha(), 0.0);

        assert_eq!(
            evaluate_lod(LodMetric::new(f32::MAX, f32::MIN_POSITIVE, f32::MAX, 1.0, 2,).unwrap()),
            Err(LodError::NonFinite)
        );
    }

    #[test]
    fn negative_integer_and_fixed_coordinates_convert_with_a_bound() {
        let point = I64Point::new([-9_000_000_001, 4, -7]);
        let origin = I64Point::new([-9_000_000_000, 1, -10]);
        let local = point
            .to_local(origin)
            .expect("difference does not overflow");
        assert!(local.contains([-1.0, 3.0, 3.0]));
        assert!(local.max_abs_error() > 0.0);

        let fixed = FixedPoint3::from_raw([-6_442_450_944, 0, 2_147_483_648]);
        let fixed_origin = FixedPoint3::from_raw([0, 0, 0]);
        let fixed_local = fixed.to_local(fixed_origin).expect("fixed difference");
        assert!(fixed_local.contains([-1.5, 0.0, 0.5]));
    }

    #[test]
    fn deterministic_height_sampling_is_repeatable() {
        let tile = TileKey::new(CubeFace::NegativeX, 5, 17, 9).unwrap();
        let sample = HeightSample::new(tile, 12_345, 54_321);
        let first = DeterministicHeightSampler::new(42, 2_000)
            .sample_height(sample)
            .unwrap();
        let second = DeterministicHeightSampler::new(42, 2_000)
            .sample_height(sample)
            .unwrap();
        let other = DeterministicHeightSampler::new(43, 2_000)
            .sample_height(sample)
            .unwrap();
        assert_eq!(first, second);
        assert_ne!(first, other);
        assert!(first.height_mm().unsigned_abs() <= 2_000);

        let saturated = DeterministicHeightSampler::new(42, i64::MAX)
            .sample_height(sample)
            .unwrap();
        assert!(saturated.height_mm().unsigned_abs() <= i64::MAX as u64);
    }

    #[test]
    fn voxel_contract_is_little_endian_and_bounds_checked() {
        let layout =
            VoxelLayout::new([2, 2, 1], 2, VoxelComponentFormat::SignedDistanceQ8_8).unwrap();
        let mut bytes = alloc::vec![0; layout.expected_byte_len().unwrap()];
        bytes[0..2].copy_from_slice(&(-256_i16).to_le_bytes());
        let buffer = VoxelBuffer::new(layout, bytes).unwrap();
        assert_eq!(
            buffer.sample([0, 0, 0]).unwrap(),
            VoxelSample::SignedDistance(-256)
        );
        assert_eq!(buffer.linear_index([1, 0, 0]).unwrap(), 1);
        assert!(matches!(
            buffer.sample([2, 0, 0]),
            Err(VoxelError::OutOfBounds { .. })
        ));
    }
}
