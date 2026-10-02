//! The framebuffer and the sampling driver that turn the integrator into an
//! image.
//!
//! [`Film`] is a flat, row-major buffer of per-pixel linear radiance. [`render`]
//! drives the [`PathIntegrator`] over every pixel of a [`PinholeCamera`],
//! averaging `samples_per_pixel` jittered primary rays per pixel to form a
//! converged reference image. Each pixel owns an independent, seed-addressed
//! `RNG` stream (keyed by its flat index), so the whole render is deterministic
//! (same arguments, bit-identical image) and could be parallelized per pixel
//! without changing any result.

use alloc::vec;
use alloc::vec::Vec;

use super::camera::PinholeCamera;
use super::halton::HaltonPixelSampler;
use super::integrator::{PathIntegrator, Scene};
use super::sampler::Rng;
use super::Vec3;

/// A row-major framebuffer of linear per-pixel radiance.
#[derive(Clone, Debug)]
pub struct Film {
    /// Image width in pixels.
    width: u32,
    /// Image height in pixels.
    height: u32,
    /// `width * height` pixels, row-major (`index = y * width + x`).
    pixels: Vec<Vec3>,
}

impl Film {
    /// Allocates a `width`×`height` film cleared to [`Vec3::ZERO`].
    #[must_use]
    pub fn new(width: u32, height: u32) -> Self {
        let count = (width as usize) * (height as usize);
        Self {
            width,
            height,
            pixels: vec![Vec3::ZERO; count],
        }
    }

    /// Image width in pixels.
    #[must_use]
    pub fn width(&self) -> u32 {
        self.width
    }

    /// Image height in pixels.
    #[must_use]
    pub fn height(&self) -> u32 {
        self.height
    }

    /// The pixel at `(x, y)`, or [`Vec3::ZERO`] when `(x, y)` is out of range.
    #[must_use]
    pub fn pixel(&self, x: u32, y: u32) -> Vec3 {
        if x >= self.width || y >= self.height {
            return Vec3::ZERO;
        }
        let index = (y as usize) * (self.width as usize) + (x as usize);
        self.pixels[index]
    }

    /// The backing pixel slice in row-major order.
    #[must_use]
    pub fn pixels(&self) -> &[Vec3] {
        &self.pixels
    }
}

/// Renders `scene` through `camera` with `integrator`, averaging
/// `samples_per_pixel` jittered primary rays per pixel into a `width`×`height`
/// [`Film`].
///
/// The image resolution is passed explicitly (the camera carries only the view
/// frame and field of view, not a pixel count). `seed` selects the
/// deterministic sampling sequence: each pixel derives an independent `RNG`
/// stream from its flat index, so rendering the same arguments twice yields a
/// bit-identical [`Film`]. A `samples_per_pixel` of zero (or a zero-sized film)
/// produces a cleared image without panicking.
#[must_use]
pub fn render(
    scene: &Scene,
    camera: &PinholeCamera,
    integrator: &PathIntegrator,
    width: u32,
    height: u32,
    samples_per_pixel: u32,
    seed: u64,
) -> Film {
    let mut film = Film::new(width, height);
    if width == 0 || height == 0 || samples_per_pixel == 0 {
        return film;
    }
    let inv_spp = 1.0 / (samples_per_pixel as f32);
    for y in 0..height {
        for x in 0..width {
            let index = (y as usize) * (width as usize) + (x as usize);
            // A per-pixel stream keyed by the flat index keeps the render
            // deterministic and order-independent (each pixel is self-contained).
            let mut rng = Rng::with_stream(seed, index as u64 + 1);
            // Low-discrepancy sub-pixel jitter (decorrelated per pixel) resolves
            // primary-visibility edges far faster than independent jitter would.
            let jitter_sampler = HaltonPixelSampler::new(seed, index as u64);
            let mut sum = Vec3::ZERO;
            for s in 0..samples_per_pixel {
                let jitter = jitter_sampler.sample(u64::from(s));
                // A finite aperture engages the thin-lens model (depth of
                // field); a zero aperture keeps the cheaper pinhole path.
                let ray = if camera.aperture_radius() > 0.0 {
                    camera.primary_ray_lens(x, y, width, height, jitter, &mut rng)
                } else {
                    camera.primary_ray(x, y, width, height, jitter)
                };
                sum = sum.add(integrator.radiance(scene, ray, &mut rng));
            }
            film.pixels[index] = sum.scale(inv_spp);
        }
    }
    film
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ray_scene::triangle_mesh::{TriangleMesh, TriangleMeshBvh};
    use crate::reference_pt::bsdf::Bsdf;
    use crate::reference_pt::integrator::Material;

    /// A single large upward-facing triangle in the `y = 0` plane (the shading
    /// normal is faced toward the viewer by the integrator).
    fn ground_plane() -> TriangleMeshBvh {
        let positions = alloc::vec![
            [-10.0_f32, 0.0, -10.0],
            [10.0, 0.0, -10.0],
            [0.0, 0.0, 10.0],
        ];
        let indices = alloc::vec![[0u32, 1, 2]];
        let mesh = TriangleMesh::new(positions, alloc::vec![], alloc::vec![], indices)
            .expect("valid ground triangle");
        TriangleMeshBvh::build(mesh)
    }

    /// A white-furnace scene: an albedo-one floor under unit environment.
    fn white_furnace_scene() -> Scene {
        let materials = alloc::vec![Material::new(Bsdf::Lambert { albedo: Vec3::ONE })];
        Scene::new(ground_plane(), materials, alloc::vec![], Vec3::ONE)
            .expect("white furnace scene")
    }

    /// A camera above the origin looking straight down at the floor, whose
    /// footprint stays well inside the ground triangle.
    fn overhead_camera() -> PinholeCamera {
        PinholeCamera::look_at(
            Vec3::new(0.0, 3.0, 0.0),
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(0.0, 0.0, -1.0),
            0.3,
            1.0,
        )
        .expect("valid overhead camera")
    }

    #[test]
    fn new_film_is_cleared() {
        let film = Film::new(4, 3);
        assert_eq!(film.width(), 4);
        assert_eq!(film.height(), 3);
        assert_eq!(film.pixels().len(), 12);
        assert!(film.pixels().iter().all(|p| *p == Vec3::ZERO));
    }

    #[test]
    fn out_of_range_pixel_is_zero() {
        let film = Film::new(2, 2);
        assert_eq!(film.pixel(2, 0), Vec3::ZERO);
        assert_eq!(film.pixel(0, 2), Vec3::ZERO);
        assert_eq!(film.pixel(99, 99), Vec3::ZERO);
    }

    #[test]
    fn white_furnace_image_is_uniform_unit() {
        // Every primary ray hits an albedo-one floor under unit environment, so
        // each pixel must converge to unit radiance (the white-furnace test
        // lifted to a full image through the camera and film).
        let scene = white_furnace_scene();
        let camera = overhead_camera();
        let integrator = PathIntegrator::new(8, 5);
        let film = render(&scene, &camera, &integrator, 4, 4, 256, 1);
        for p in film.pixels() {
            assert!(p.is_finite());
            assert!(
                (p.x - 1.0).abs() < 2e-2,
                "white furnace pixel {} should converge to 1",
                p.x
            );
        }
    }

    #[test]
    fn thin_lens_render_is_unbiased_and_deterministic() {
        // Depth of field only redistributes where rays land, never how much
        // energy they carry, so a thin-lens white-furnace render must still
        // converge to unit radiance and stay bit-identical across runs.
        let scene = white_furnace_scene();
        let camera = overhead_camera()
            .with_thin_lens(0.05, 3.0)
            .expect("valid thin-lens camera");
        let integrator = PathIntegrator::new(8, 5);
        let a = render(&scene, &camera, &integrator, 4, 4, 256, 9);
        let b = render(&scene, &camera, &integrator, 4, 4, 256, 9);
        for (pa, pb) in a.pixels().iter().zip(b.pixels()) {
            assert_eq!(pa.x.to_bits(), pb.x.to_bits());
        }
        for p in a.pixels() {
            assert!(p.is_finite());
            assert!(
                (p.x - 1.0).abs() < 2e-2,
                "thin-lens white furnace pixel {} should converge to 1",
                p.x
            );
        }
    }

    #[test]
    fn render_is_deterministic() {
        let scene = white_furnace_scene();
        let camera = overhead_camera();
        let integrator = PathIntegrator::new(8, 5);
        let a = render(&scene, &camera, &integrator, 3, 3, 16, 42);
        let b = render(&scene, &camera, &integrator, 3, 3, 16, 42);
        assert_eq!(a.width(), b.width());
        assert_eq!(a.height(), b.height());
        for (pa, pb) in a.pixels().iter().zip(b.pixels()) {
            assert_eq!(pa.x.to_bits(), pb.x.to_bits());
            assert_eq!(pa.y.to_bits(), pb.y.to_bits());
            assert_eq!(pa.z.to_bits(), pb.z.to_bits());
        }
    }

    #[test]
    fn zero_sized_film_does_not_panic() {
        let scene = white_furnace_scene();
        let camera = overhead_camera();
        let integrator = PathIntegrator::new(8, 5);
        let film = render(&scene, &camera, &integrator, 0, 0, 16, 1);
        assert_eq!(film.width(), 0);
        assert_eq!(film.height(), 0);
        assert!(film.pixels().is_empty());
    }

    #[test]
    fn zero_samples_leaves_cleared_image() {
        let scene = white_furnace_scene();
        let camera = overhead_camera();
        let integrator = PathIntegrator::new(8, 5);
        let film = render(&scene, &camera, &integrator, 2, 2, 0, 1);
        assert_eq!(film.pixels().len(), 4);
        assert!(film.pixels().iter().all(|p| *p == Vec3::ZERO));
    }
}
