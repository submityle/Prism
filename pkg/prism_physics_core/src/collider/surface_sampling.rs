//! Area-weighted uniform surface sampling for triangle meshes.
//!
//! Scattering points uniformly over a mesh surface is a workhorse primitive in
//! AAA content pipelines: seeding particle emitters, distributing foliage or
//! debris, bootstrapping signed-distance narrow bands and estimating surface
//! statistics all consume a point cloud whose density is proportional to area
//! rather than to triangle count. Sampling triangles with probability
//! proportional to their area, then drawing a uniform barycentric point inside
//! the chosen triangle, yields exactly that distribution.
//!
//! Randomness comes from a self-contained `SplitMix64` generator seeded by the
//! caller, so sampling is fully deterministic and reproducible across machines
//! without pulling in an external RNG crate. This is pure triangle-soup
//! geometry with no coupling to the collision pipeline, and nothing here is
//! derived from Unreal Engine source.

use glam::Vec3;

/// Triangles whose doubled area (cross-product length) is below this threshold
/// are treated as degenerate and excluded from the area distribution.
const DEGENERATE_AREA_EPSILON: f32 = 1.0e-12;

/// A single point sampled on the mesh surface.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SurfaceSample {
    /// World-space position of the sample.
    pub position: Vec3,
    /// Unit face normal of the triangle the sample was drawn from.
    pub normal: Vec3,
    /// Index of the source triangle in the input index slice.
    pub triangle: u32,
    /// Barycentric coordinates `(a, b, c)` of the sample inside its triangle,
    /// summing to one, ordered to match the triangle's vertices.
    pub barycentric: [f32; 3],
}

/// Tuning for [`sample_surface`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SurfaceSampleParams {
    /// Number of surface points to draw. Must be non-zero.
    pub count: usize,
    /// Seed for the deterministic `SplitMix64` generator.
    pub seed: u64,
}

impl Default for SurfaceSampleParams {
    /// A single sample from a fixed seed.
    fn default() -> Self {
        Self { count: 1, seed: 0 }
    }
}

/// A minimal, allocation-free `SplitMix64` pseudo-random generator.
///
/// `SplitMix64` (Steele, Lea & Flood 2014) passes `BigCrush` and needs only a
/// single `u64` of state, which makes it an ideal self-contained source for
/// reproducible geometry sampling.
#[derive(Clone, Copy, Debug)]
struct SplitMix64 {
    state: u64,
}

impl SplitMix64 {
    /// Seeds the generator.
    fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    /// Advances the state and returns the next 64-bit output.
    fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^ (z >> 31)
    }

    /// Returns the next float uniformly distributed in `[0, 1)` using the top
    /// 24 bits of the generator output.
    fn next_f32(&mut self) -> f32 {
        let bits = (self.next_u64() >> 40) as u32;
        f64::from(bits).mul_add(1.0 / f64::from(1u32 << 24), 0.0) as f32
    }
}

/// Returns the total surface area of the mesh, or `None` when the input is
/// empty or no triangle has a finite positive area.
///
/// Degenerate (zero-area) and out-of-bounds triangles contribute nothing.
#[must_use]
pub fn total_surface_area(vertices: &[Vec3], indices: &[[u32; 3]]) -> Option<f32> {
    if vertices.is_empty() || indices.is_empty() {
        return None;
    }
    let mut total = 0.0_f64;
    for tri in indices {
        if let Some(area) = triangle_area(vertices, *tri) {
            total += f64::from(area);
        }
    }
    if total > 0.0 {
        Some(total as f32)
    } else {
        None
    }
}

/// Draws `params.count` points over the mesh surface with probability density
/// proportional to area.
///
/// Returns `None` when the vertex or index slice is empty, `params.count` is
/// zero, or the mesh has no finite positive total area (every triangle
/// degenerate or out of bounds). Degenerate and out-of-bounds triangles are
/// skipped and never sampled. The result is deterministic for a given
/// `params.seed`.
#[must_use]
pub fn sample_surface(
    vertices: &[Vec3],
    indices: &[[u32; 3]],
    params: SurfaceSampleParams,
) -> Option<Vec<SurfaceSample>> {
    if vertices.is_empty() || indices.is_empty() || params.count == 0 {
        return None;
    }

    // Build the cumulative-area table over valid triangles only.
    let mut valid: Vec<u32> = Vec::new();
    let mut cumulative: Vec<f64> = Vec::new();
    let mut running = 0.0_f64;
    for (idx, tri) in indices.iter().enumerate() {
        if let Some(area) = triangle_area(vertices, *tri) {
            running += f64::from(area);
            valid.push(idx as u32);
            cumulative.push(running);
        }
    }
    if valid.is_empty() || running <= 0.0 {
        return None;
    }

    let mut rng = SplitMix64::new(params.seed);
    let mut samples = Vec::with_capacity(params.count);
    for _ in 0..params.count {
        // Pick a triangle with probability proportional to area.
        let target = f64::from(rng.next_f32()) * running;
        let slot = cumulative
            .partition_point(|&c| c < target)
            .min(valid.len() - 1);
        let tri_index = valid[slot];
        let tri = indices[tri_index as usize];

        // Uniform barycentric point inside the triangle (Osada et al. 2002).
        let r1 = rng.next_f32();
        let r2 = rng.next_f32();
        let sqrt_r1 = (f64::from(r1)).sqrt() as f32;
        let a = 1.0 - sqrt_r1;
        let b = sqrt_r1 * (1.0 - r2);
        let c = sqrt_r1 * r2;

        let v0 = vertices[tri[0] as usize];
        let v1 = vertices[tri[1] as usize];
        let v2 = vertices[tri[2] as usize];
        let position = v0 * a + v1 * b + v2 * c;
        let normal = (v1 - v0).cross(v2 - v0).normalize_or_zero();

        samples.push(SurfaceSample {
            position,
            normal,
            triangle: tri_index,
            barycentric: [a, b, c],
        });
    }

    Some(samples)
}

/// Returns the area of a triangle, or `None` when any index is out of bounds or
/// the triangle is degenerate.
fn triangle_area(vertices: &[Vec3], tri: [u32; 3]) -> Option<f32> {
    let n = vertices.len();
    if tri[0] as usize >= n || tri[1] as usize >= n || tri[2] as usize >= n {
        return None;
    }
    let v0 = vertices[tri[0] as usize];
    let v1 = vertices[tri[1] as usize];
    let v2 = vertices[tri[2] as usize];
    let doubled = (v1 - v0).cross(v2 - v0).length();
    if doubled <= DEGENERATE_AREA_EPSILON {
        None
    } else {
        Some(0.5 * doubled)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A unit cube centred at the origin, outward wound, as a triangle soup
    /// with shared vertices. Total surface area is `6`.
    fn unit_cube() -> (Vec<Vec3>, Vec<[u32; 3]>) {
        let v = vec![
            Vec3::new(-0.5, -0.5, -0.5),
            Vec3::new(0.5, -0.5, -0.5),
            Vec3::new(0.5, 0.5, -0.5),
            Vec3::new(-0.5, 0.5, -0.5),
            Vec3::new(-0.5, -0.5, 0.5),
            Vec3::new(0.5, -0.5, 0.5),
            Vec3::new(0.5, 0.5, 0.5),
            Vec3::new(-0.5, 0.5, 0.5),
        ];
        let f = vec![
            [0, 2, 1],
            [0, 3, 2],
            [4, 5, 6],
            [4, 6, 7],
            [0, 1, 5],
            [0, 5, 4],
            [3, 7, 6],
            [3, 6, 2],
            [0, 4, 7],
            [0, 7, 3],
            [1, 2, 6],
            [1, 6, 5],
        ];
        (v, f)
    }

    #[test]
    fn empty_or_invalid_input_is_rejected() {
        let (verts, tris) = unit_cube();
        let p = SurfaceSampleParams { count: 4, seed: 1 };
        assert!(sample_surface(&[], &tris, p).is_none());
        assert!(sample_surface(&verts, &[], p).is_none());
        assert!(sample_surface(&verts, &tris, SurfaceSampleParams { count: 0, seed: 1 }).is_none());
        assert!(total_surface_area(&[], &tris).is_none());
        assert!(total_surface_area(&verts, &[]).is_none());
    }

    #[test]
    fn cube_total_area_is_six() {
        let (verts, tris) = unit_cube();
        let area = total_surface_area(&verts, &tris).unwrap();
        assert!((area - 6.0).abs() < 1.0e-5, "area = {area}");
    }

    #[test]
    fn degenerate_only_mesh_is_rejected() {
        let verts = vec![Vec3::ZERO, Vec3::X, Vec3::X * 2.0];
        let tris = vec![[0, 1, 2]]; // collinear -> zero area
        assert!(total_surface_area(&verts, &tris).is_none());
        assert!(sample_surface(&verts, &tris, SurfaceSampleParams { count: 3, seed: 7 }).is_none());
    }

    #[test]
    fn samples_lie_on_the_surface_with_valid_barycentrics() {
        let (verts, tris) = unit_cube();
        let samples = sample_surface(
            &verts,
            &tris,
            SurfaceSampleParams {
                count: 256,
                seed: 42,
            },
        )
        .unwrap();
        assert_eq!(samples.len(), 256);
        for s in &samples {
            // Barycentrics are a convex combination.
            let sum = s.barycentric[0] + s.barycentric[1] + s.barycentric[2];
            assert!((sum - 1.0).abs() < 1.0e-5, "bary sum = {sum}");
            assert!(s.barycentric.iter().all(|&w| w >= -1.0e-6));
            // Every point sits on a face of the cube: one coordinate pinned to
            // +/-0.5, all within the cube.
            let on_face = (s.position.x.abs() - 0.5).abs() < 1.0e-4
                || (s.position.y.abs() - 0.5).abs() < 1.0e-4
                || (s.position.z.abs() - 0.5).abs() < 1.0e-4;
            assert!(on_face, "position off surface: {:?}", s.position);
            assert!(s.position.x.abs() <= 0.5 + 1.0e-4);
            assert!(s.position.y.abs() <= 0.5 + 1.0e-4);
            assert!(s.position.z.abs() <= 0.5 + 1.0e-4);
            // Normals are unit length.
            assert!((s.normal.length() - 1.0).abs() < 1.0e-5);
            assert!((s.triangle as usize) < tris.len());
        }
    }

    #[test]
    fn sampling_is_deterministic_for_a_seed() {
        let (verts, tris) = unit_cube();
        let p = SurfaceSampleParams {
            count: 64,
            seed: 123,
        };
        let a = sample_surface(&verts, &tris, p).unwrap();
        let b = sample_surface(&verts, &tris, p).unwrap();
        assert_eq!(a, b);
    }

    #[test]
    fn different_seeds_differ() {
        let (verts, tris) = unit_cube();
        let a = sample_surface(&verts, &tris, SurfaceSampleParams { count: 64, seed: 1 }).unwrap();
        let b = sample_surface(&verts, &tris, SurfaceSampleParams { count: 64, seed: 2 }).unwrap();
        assert_ne!(a, b);
    }

    #[test]
    fn area_weighting_favours_the_larger_triangle() {
        // Two coplanar triangles: a tiny one and one 100x larger in area.
        let verts = vec![
            // Small triangle near origin (area 0.5).
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(0.0, 1.0, 0.0),
            // Large triangle (area 50).
            Vec3::new(10.0, 0.0, 0.0),
            Vec3::new(20.0, 0.0, 0.0),
            Vec3::new(10.0, 10.0, 0.0),
        ];
        let tris = vec![[0, 1, 2], [3, 4, 5]];
        let samples = sample_surface(
            &verts,
            &tris,
            SurfaceSampleParams {
                count: 4000,
                seed: 9,
            },
        )
        .unwrap();
        let large = samples.iter().filter(|s| s.triangle == 1).count();
        // Large triangle holds 50/50.5 ~= 99% of the area.
        assert!(large > samples.len() * 90 / 100, "large share = {large}");
    }

    #[test]
    fn out_of_bounds_triangles_are_skipped() {
        let verts = vec![Vec3::ZERO, Vec3::X, Vec3::Y];
        // One valid triangle, one with an out-of-range index.
        let tris = vec![[0, 1, 2], [0, 1, 99]];
        let area = total_surface_area(&verts, &tris).unwrap();
        assert!((area - 0.5).abs() < 1.0e-6);
        let samples =
            sample_surface(&verts, &tris, SurfaceSampleParams { count: 32, seed: 3 }).unwrap();
        assert!(samples.iter().all(|s| s.triangle == 0));
    }
}
