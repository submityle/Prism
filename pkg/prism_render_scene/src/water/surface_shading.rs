//! Authored, per-body water-surface shading state and its per-view packing.
//!
//! The raster draw ([`super::surface_node`], the following slice) renders one
//! displaced water surface per [`WaterBody`](super::body::WaterBody) that
//! carries a [`WaterSurfaceShading`]. This module owns the *authored* half of
//! that draw's uniform: the frontend selection plus the body-constant lighting
//! and style scalars a game sets once when it spawns the water.
//!
//! The per-frame, per-view camera / key-light state (which changes every frame
//! as the camera moves) lives in [`SurfaceViewInputs`]; the draw node fills it
//! from the active view. [`WaterSurfaceShading::view_params`] is the single
//! place the two halves combine into the
//! [`SurfaceViewParams`](super::surface_mesh::SurfaceViewParams) that
//! [`build_surface_view`](super::surface_mesh::build_surface_view) then packs
//! into the `std140` `WaterSurfaceView` uniform the shader reads.
//!
//! Keeping the authored body constants separate from the per-view inputs lets
//! the body own its stable style (uploaded once, compared cheaply via
//! `PartialEq`) while the node supplies only the volatile camera state each
//! frame.

use prism_render_architecture::water::ShadingFrontend;

use super::surface_mesh::SurfaceViewParams;

/// The authored, body-constant water-surface shading state.
///
/// This is the per-body half of the raster draw's view uniform: the lighting
/// response frontend plus the style and body scalars a game authors once. The
/// per-view camera / key-light state is supplied separately each frame by
/// [`SurfaceViewInputs`] and merged in [`WaterSurfaceShading::view_params`].
///
/// `Copy` and `PartialEq` so a body can carry it inline and the extract stage
/// can diff it cheaply; it deliberately has no `Default` because the frontend
/// is a required authoring decision ([`ShadingFrontend`] has no default), so
/// use [`WaterSurfaceShading::new`] to start from physically-plausible defaults
/// for a chosen frontend.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct WaterSurfaceShading {
    /// Lighting-response frontend; selects the raster pipeline variant.
    pub(crate) frontend: ShadingFrontend,
    /// Base water albedo (`rgb`, linear).
    pub(crate) water_albedo: [f32; 3],
    /// Minimum surface alpha floor, so still water keeps some opacity.
    pub(crate) min_alpha: f32,
    /// Base perceptual roughness.
    pub(crate) roughness: f32,
    /// Base reflectance (`f0` at normal incidence).
    pub(crate) reflectance: f32,
    /// Optical thickness driving the `Beer-Lambert` body attenuation.
    pub(crate) optical_thickness: f32,
    /// Foam whiten strength: how strongly foam coverage washes the body white.
    pub(crate) foam_whiten: f32,
    /// Screen-space refraction offset scale; `0.0` disables the refraction
    /// displacement.
    pub(crate) refraction_screen_offset: f32,
    /// `NPR` ramp step count (number of discrete toon bands).
    pub(crate) npr_ramp_steps: f32,
    /// `NPR` toon foam threshold: foam coverage above this draws a hard edge.
    pub(crate) toon_foam_threshold: f32,
    /// Custom-frontend emissive tint strength.
    pub(crate) tint_strength: f32,
    /// Hybrid-frontend shore blend bias added to foam coverage when crossfading
    /// the `PBR` body into the stylized shallows.
    pub(crate) hybrid_shore_bias: f32,
}

impl WaterSurfaceShading {
    /// Physically-plausible default style for `frontend`.
    ///
    /// The defaults describe a mid-roughness blue-green body with a thin
    /// `Beer-Lambert` tint, a modest refraction offset, and neutral stylization
    /// scalars. A game overrides any field after construction; the frontend is
    /// the one required decision, which is why there is no `Default`.
    #[must_use]
    #[cfg_attr(
        not(test),
        expect(
            dead_code,
            reason = "constructed by the water-surface authoring presets and raster draw node (the following slices); exercised now by the unit tests in this module"
        )
    )]
    pub(crate) fn new(frontend: ShadingFrontend) -> Self {
        Self {
            frontend,
            water_albedo: [0.02, 0.08, 0.12],
            min_alpha: 0.08,
            roughness: 0.08,
            reflectance: 0.02,
            optical_thickness: 1.5,
            foam_whiten: 0.9,
            refraction_screen_offset: 0.02,
            npr_ramp_steps: 4.0,
            toon_foam_threshold: 0.6,
            tint_strength: 0.0,
            hybrid_shore_bias: 0.0,
        }
    }

    /// Merge this body-constant style with the per-view `view` state into the
    /// [`SurfaceViewParams`] the raster draw packs into one uniform.
    ///
    /// Pure: the body contributes the style / lighting scalars and the view
    /// contributes the camera transform, key light and viewport, so the same
    /// `(self, view)` always yields the same params.
    #[must_use]
    #[cfg_attr(
        not(test),
        expect(
            dead_code,
            reason = "called by the water-surface raster draw node (the following slice); exercised now by the unit tests in this module"
        )
    )]
    pub(crate) fn view_params(&self, view: &SurfaceViewInputs) -> SurfaceViewParams {
        SurfaceViewParams {
            clip_from_world: view.clip_from_world,
            camera_world_position: view.camera_world_position,
            refraction_screen_offset: self.refraction_screen_offset,
            sun_direction: view.sun_direction,
            sun_illuminance: view.sun_illuminance,
            roughness: self.roughness,
            reflectance: self.reflectance,
            optical_thickness: self.optical_thickness,
            foam_whiten: self.foam_whiten,
            water_albedo: self.water_albedo,
            min_alpha: self.min_alpha,
            npr_ramp_steps: self.npr_ramp_steps,
            toon_foam_threshold: self.toon_foam_threshold,
            tint_strength: self.tint_strength,
            hybrid_shore_bias: self.hybrid_shore_bias,
            viewport_size: view.viewport_size,
        }
    }
}

/// The per-frame, per-view camera and key-light state the raster draw reads.
///
/// Unlike the body-constant [`WaterSurfaceShading`], every field here changes as
/// the camera and sun move, so the draw node fills it fresh each frame from the
/// active view rather than storing it on the body.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct SurfaceViewInputs {
    /// Clip-from-world transform for the surface vertices (column-major).
    pub(crate) clip_from_world: [[f32; 4]; 4],
    /// World-space camera position.
    pub(crate) camera_world_position: [f32; 3],
    /// Direction *towards* the key light (world space); the shader renormalizes.
    pub(crate) sun_direction: [f32; 3],
    /// Key-light illuminance (`rgb`, linear).
    pub(crate) sun_illuminance: [f32; 3],
    /// Framebuffer size in pixels (`width`, `height`).
    pub(crate) viewport_size: [f32; 2],
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fixture_view() -> SurfaceViewInputs {
        SurfaceViewInputs {
            clip_from_world: [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 2.0, 0.0, 0.0],
                [0.0, 0.0, 3.0, 0.0],
                [4.0, 5.0, 6.0, 1.0],
            ],
            camera_world_position: [7.0, 8.0, 9.0],
            sun_direction: [0.0, 1.0, 0.0],
            sun_illuminance: [10.0, 11.0, 12.0],
            viewport_size: [1920.0, 1080.0],
        }
    }

    #[test]
    fn new_sets_the_requested_frontend() {
        for frontend in [
            ShadingFrontend::Pbr,
            ShadingFrontend::Npr,
            ShadingFrontend::Custom,
            ShadingFrontend::Hybrid,
        ] {
            assert_eq!(WaterSurfaceShading::new(frontend).frontend, frontend);
        }
    }

    #[test]
    fn new_defaults_are_physically_plausible() {
        let s = WaterSurfaceShading::new(ShadingFrontend::Pbr);
        // Roughness and reflectance are valid closure inputs.
        assert!((0.0..=1.0).contains(&s.roughness));
        assert!((0.0..=1.0).contains(&s.reflectance));
        // The alpha floor keeps still water visible.
        assert!(s.min_alpha > 0.0 && s.min_alpha < 1.0);
        // A positive optical thickness gives a real Beer-Lambert tint.
        assert!(s.optical_thickness > 0.0);
        // At least two toon bands, else the NPR ramp degenerates to flat.
        assert!(s.npr_ramp_steps >= 2.0);
    }

    #[test]
    fn view_params_takes_style_from_body_and_camera_from_view() {
        let shading = WaterSurfaceShading {
            frontend: ShadingFrontend::Hybrid,
            water_albedo: [0.1, 0.2, 0.3],
            min_alpha: 0.25,
            roughness: 0.4,
            reflectance: 0.05,
            optical_thickness: 2.0,
            foam_whiten: 0.75,
            refraction_screen_offset: 0.03,
            npr_ramp_steps: 5.0,
            toon_foam_threshold: 0.55,
            tint_strength: 0.6,
            hybrid_shore_bias: 0.15,
        };
        let view = fixture_view();

        let params = shading.view_params(&view);

        // Camera / view lanes come from the per-view inputs.
        assert_eq!(params.clip_from_world, view.clip_from_world);
        assert_eq!(params.camera_world_position, view.camera_world_position);
        assert_eq!(params.sun_direction, view.sun_direction);
        assert_eq!(params.sun_illuminance, view.sun_illuminance);
        assert_eq!(params.viewport_size, view.viewport_size);

        // Style / body lanes come from the authored shading.
        assert_eq!(params.water_albedo, shading.water_albedo);
        assert_eq!(params.min_alpha, shading.min_alpha);
        assert_eq!(params.roughness, shading.roughness);
        assert_eq!(params.reflectance, shading.reflectance);
        assert_eq!(params.optical_thickness, shading.optical_thickness);
        assert_eq!(params.foam_whiten, shading.foam_whiten);
        assert_eq!(
            params.refraction_screen_offset,
            shading.refraction_screen_offset
        );
        assert_eq!(params.npr_ramp_steps, shading.npr_ramp_steps);
        assert_eq!(params.toon_foam_threshold, shading.toon_foam_threshold);
        assert_eq!(params.tint_strength, shading.tint_strength);
        assert_eq!(params.hybrid_shore_bias, shading.hybrid_shore_bias);
    }

    #[test]
    fn view_params_feeds_build_surface_view_consistently() {
        // The packing is the single-sourced concern of build_surface_view; this
        // guards that view_params produces params that round-trip through it.
        let shading = WaterSurfaceShading::new(ShadingFrontend::Npr);
        let view = fixture_view();

        let gpu = super::super::surface_mesh::build_surface_view(&shading.view_params(&view));

        // Spot-check the two cross-cutting packings: the refraction offset rides
        // in world_camera_position.w and the albedo rides in water_color.rgb.
        assert_eq!(
            gpu.world_camera_position[3],
            shading.refraction_screen_offset
        );
        assert_eq!(
            [gpu.water_color[0], gpu.water_color[1], gpu.water_color[2]],
            shading.water_albedo
        );
        assert_eq!(gpu.water_color[3], shading.min_alpha);
    }
}
