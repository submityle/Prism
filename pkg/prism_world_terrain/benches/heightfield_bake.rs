//! Release-mode CPU regression distributions, not a GPU or product-scale claim.

use std::hint::black_box;
use std::time::{Duration, Instant};

use prism_world_terrain::{
    BakeLimits, CubeFace, DirtyRegion, HeightSample, HeightSampleValue, HeightSampler,
    HeightfieldLodSeamPlan, HeightfieldMesh, HeightfieldSpec, HeightfieldTile, LodSeamCoverage,
    LodSeamLimits, LodSeamNeighbor, PlanetHeightfieldMesh, PlanetMeshConfig, SdfField, SdfGrid,
    SdfSpec, TileEdge, TileKey, VolumeBakeLimits, VolumeRegion,
};

struct Plane(i64);

struct DistancePlane {
    spec: SdfSpec,
    bias: i32,
}

impl SdfField for DistancePlane {
    type Error = core::convert::Infallible;

    fn spec(&self) -> SdfSpec {
        self.spec
    }

    fn sample_q16(&self, coordinate: [u32; 3]) -> Result<i32, Self::Error> {
        Ok(coordinate[0] as i32 * 65_536 + self.bias)
    }
}

impl HeightSampler for Plane {
    type Error = core::convert::Infallible;

    fn sample_height(&self, point: HeightSample) -> Result<HeightSampleValue, Self::Error> {
        Ok(HeightSampleValue::from_millimetres(
            self.0 + i64::from(point.u_q16()) * 2,
        ))
    }
}

fn report(name: &str, mut samples: Vec<Duration>) {
    samples.sort_unstable();
    let quantile = |p: usize| samples[(samples.len() * p).div_ceil(100) - 1].as_nanos();
    println!(
        "{name} samples={} p50_ns={} p95_ns={} p99_ns={}",
        samples.len(),
        quantile(50),
        quantile(95),
        quantile(99),
    );
}

fn main() {
    const RUNS: usize = 128;
    const WARMUP: usize = 16;
    for side in [129_u32, 257] {
        let tile = TileKey::root(CubeFace::PositiveY);
        let spec = HeightfieldSpec::new(side, 10).unwrap();
        let vertices = side as usize * side as usize;
        let payload = vertices * 32 + (side as usize - 1).pow(2) * 24;
        let limits = BakeLimits {
            max_samples: vertices,
            max_bytes: payload,
        };
        let mut bakes = Vec::with_capacity(RUNS);
        let mut meshes = Vec::with_capacity(RUNS);
        let mut local_bakes = Vec::with_capacity(RUNS);
        let mut updates = Vec::with_capacity(RUNS);
        let mut planets = Vec::with_capacity(RUNS);
        for iteration in 0..WARMUP + RUNS {
            let start = Instant::now();
            let mut field =
                HeightfieldTile::bake(tile, spec, black_box(&Plane(0)), limits).unwrap();
            let bake_time = start.elapsed();
            let start = Instant::now();
            let mut mesh =
                HeightfieldMesh::build(black_box(&field), [128.0, 128.0], limits).unwrap();
            let mesh_time = start.elapsed();
            let start = Instant::now();
            let planet = PlanetHeightfieldMesh::build(
                black_box(&field),
                PlanetMeshConfig::new(6_371_000_000, [0, 6_371_000_000, 0], 2_000.0).unwrap(),
                limits,
            )
            .unwrap();
            let planet_time = start.elapsed();
            black_box(planet);
            let region = DirtyRegion::new(tile, 32_767, 32_767, 32_768, 32_768).unwrap();
            let start = Instant::now();
            let change = field
                .rebake(region, black_box(&Plane(1_000)), limits)
                .unwrap();
            let local_bake_time = start.elapsed();
            let start = Instant::now();
            let bounds = mesh.update(black_box(&field), change).unwrap();
            let update_time = start.elapsed();
            black_box((&mesh, bounds));
            if iteration >= WARMUP {
                bakes.push(bake_time);
                meshes.push(mesh_time);
                local_bakes.push(local_bake_time);
                updates.push(update_time);
                planets.push(planet_time);
            }
        }
        println!("grid={side}x{side} height_bytes={} mesh_payload_bytes={payload} local_samples=9 local_vertices=25 warmup={WARMUP}", vertices * 2);
        report("height_bake", bakes);
        report("mesh_build", meshes);
        report("height_local_rebake", local_bakes);
        report("mesh_local_update", updates);
        report("planet_mesh_build", planets);

        let tile = TileKey::new(CubeFace::PositiveY, 3, 3, 3).unwrap();
        let field = HeightfieldTile::bake(tile, spec, black_box(&Plane(0)), limits).unwrap();
        let segments = side - 1;
        let coarse = tile.neighbor(TileEdge::East).parent().unwrap();
        let seam_plan = HeightfieldLodSeamPlan::build_strict(
            tile,
            segments,
            &[LodSeamNeighbor::with_boundary(
                TileEdge::East,
                coarse,
                segments,
                LodSeamCoverage::full(segments),
            )],
            LodSeamLimits {
                max_segments: segments,
                max_vertices: segments as usize * 2 + 2,
                max_indices: segments as usize * 9,
                max_bytes: segments as usize * 64,
            },
        )
        .unwrap();
        let seam_budget = payload * 2;
        let mut planar_morph = HeightfieldMesh::build(&field, [128.0, 128.0], limits).unwrap();
        planar_morph
            .apply_lod_seam_plan(&seam_plan, seam_budget)
            .unwrap();
        let config = PlanetMeshConfig::new(6_371_000_000, [0, 6_371_000_000, 0], 2_000.0).unwrap();
        let mut planet_morph = PlanetHeightfieldMesh::build(&field, config, limits).unwrap();
        planet_morph
            .apply_lod_seam_plan(&seam_plan, seam_budget)
            .unwrap();
        let mut planar_morph_times = Vec::with_capacity(RUNS);
        let mut planet_morph_times = Vec::with_capacity(RUNS);
        for iteration in 0..WARMUP + RUNS {
            let alpha = [0.25, 0.5, 0.75, 1.0, 0.0][iteration % 5];
            let start = Instant::now();
            planar_morph
                .apply_lod_morph(black_box(&field), black_box(&seam_plan), black_box(alpha))
                .unwrap();
            let planar_time = start.elapsed();
            let start = Instant::now();
            planet_morph
                .apply_lod_morph(black_box(&field), black_box(&seam_plan), black_box(alpha))
                .unwrap();
            let planet_time = start.elapsed();
            black_box((&planar_morph, &planet_morph));
            if iteration >= WARMUP {
                planar_morph_times.push(planar_time);
                planet_morph_times.push(planet_time);
            }
        }
        println!("lod_morph side={side} seam_edges=1 boundary_samples={} extra_mesh_allocations=0 warmup={WARMUP}", segments / 2);
        report("planar_lod_morph", planar_morph_times);
        report("planet_lod_morph", planet_morph_times);
    }
    for side in [33_u32, 65] {
        let count = (side as usize).pow(3);
        let limits = VolumeBakeLimits {
            max_dimensions: [side; 3],
            max_samples: count,
            max_bytes: count * 4 + 27 * 4,
        };
        let mid = side / 2;
        let region = VolumeRegion::new([mid - 1; 3], [mid + 2; 3]).unwrap();
        let mut bakes = Vec::with_capacity(RUNS);
        let mut updates = Vec::with_capacity(RUNS);
        for iteration in 0..WARMUP + RUNS {
            let mut source = DistancePlane {
                spec: SdfSpec::new([side; 3], 8, 1000).unwrap(),
                bias: -(mid as i32) * 65_536,
            };
            let start = Instant::now();
            let mut grid = SdfGrid::bake(black_box(&source), limits).unwrap();
            let bake_time = start.elapsed();
            source.bias += 1024;
            let start = Instant::now();
            let change = grid.resample(region, black_box(&source), limits).unwrap();
            let update_time = start.elapsed();
            black_box((&grid, change));
            if iteration >= WARMUP {
                bakes.push(bake_time);
                updates.push(update_time);
            }
        }
        println!("sdf_grid={side}x{side}x{side} output_bytes={} local_samples=27 local_scratch_bytes=108 warmup={WARMUP}", count * 4);
        report("sdf_bake", bakes);
        report("sdf_local_resample", updates);
    }
}
