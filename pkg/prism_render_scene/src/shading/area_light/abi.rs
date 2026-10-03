//! `ABI` shared between the area-light `LTC` evaluation and its `WESL` twin
//! (`shaders/area_light_ltc.wesl`).
//!
//! The on-wire area-light record [`GpuAreaLight`] is the storage-buffer element
//! the clustered-lighting resolve path will iterate (wired in block 2); it
//! mirrors the Heitz et al. 2016 Linearly Transformed Cosines (`LTC`) polygon
//! parameterization used by the `CPU` golden in
//! [`prism_render_shading::gi::area_light`]. Every field is an all-scalar
//! std430 member (never a `vec3`) so the 16-byte array stride is explicit and
//! carries no implicit padding, matching the `WESL` struct byte-for-byte.
//!
//! The companion [`GpuAreaLightLtcParams`] is the single `var<immediate>`
//! (push-constant) block the standalone `area_light_ltc_main` entry consumes to
//! exercise the inlined `LTC` math on a device without a bound `LUT` texture; it
//! packs one quad plus the two `LTC` inverse-matrix coefficient rows.

use bytemuck::{Pod, Zeroable};

/// Workgroup size (x) of the standalone `area_light_ltc_main` compute entry.
///
/// Must match `@workgroup_size(N, 1, 1)` in `area_light_ltc.wesl`.
pub(crate) const AREA_LIGHT_LTC_WORKGROUP_SIZE: u32 = 64;

/// Shape tag: a planar rectangle (quad) area light.
pub(crate) const AREA_LIGHT_SHAPE_RECT: u32 = 0;
/// Shape tag: a planar disk area light.
pub(crate) const AREA_LIGHT_SHAPE_DISK: u32 = 1;
/// Shape tag: a capsule / tube (line) area light.
pub(crate) const AREA_LIGHT_SHAPE_TUBE: u32 = 2;

/// One area-light storage-buffer record: the std430 twin of the `WESL`
/// `AreaLight` struct and the geometry the golden
/// [`prism_render_shading::gi::area_light::polygon`] /
/// [`prism_render_shading::gi::area_light::shapes`] helpers integrate.
///
/// Five 16-byte rows (80 bytes, 80-byte array stride), all scalar:
///
/// ```text
/// row 0: pos_x   pos_y   pos_z   range
/// row 1: color_x color_y color_z intensity
/// row 2: axis_u_x axis_u_y axis_u_z half_width
/// row 3: axis_v_x axis_v_y axis_v_z half_height
/// row 4: shape   two_sided radius  _pad0
/// ```
///
/// `pos` is the world-space emitter centre, `color * intensity` is the linear
/// pre-exposed radiance, `axis_u` / `axis_v` are the orthonormal in-plane axes
/// with half-extents `half_width` / `half_height`, `shape` is one of
/// [`AREA_LIGHT_SHAPE_RECT`] / [`AREA_LIGHT_SHAPE_DISK`] / [`AREA_LIGHT_SHAPE_TUBE`],
/// `two_sided` is a `0`/`1` flag and `radius` is the disk/tube radius.
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable, PartialEq)]
pub(crate) struct GpuAreaLight {
    /// World-space emitter centre x.
    pub pos_x: f32,
    /// World-space emitter centre y.
    pub pos_y: f32,
    /// World-space emitter centre z.
    pub pos_z: f32,
    /// Influence radius; `0` disables the range window.
    pub range: f32,
    /// Linear pre-exposed radiance r (before the `intensity` multiplier).
    pub color_x: f32,
    /// Linear pre-exposed radiance g.
    pub color_y: f32,
    /// Linear pre-exposed radiance b.
    pub color_z: f32,
    /// Scalar radiance multiplier.
    pub intensity: f32,
    /// Unit in-plane U axis x.
    pub axis_u_x: f32,
    /// Unit in-plane U axis y.
    pub axis_u_y: f32,
    /// Unit in-plane U axis z.
    pub axis_u_z: f32,
    /// Half-extent along `axis_u` in world units.
    pub half_width: f32,
    /// Unit in-plane V axis x.
    pub axis_v_x: f32,
    /// Unit in-plane V axis y.
    pub axis_v_y: f32,
    /// Unit in-plane V axis z.
    pub axis_v_z: f32,
    /// Half-extent along `axis_v` in world units.
    pub half_height: f32,
    /// Shape tag (`0` rect, `1` disk, `2` tube).
    pub shape: u32,
    /// `1` if the emitter radiates from both faces, else `0`.
    pub two_sided: u32,
    /// Disk / tube radius in world units.
    pub radius: f32,
    /// Padding to the 16-byte std430 row boundary.
    pub _pad0: u32,
}

impl GpuAreaLight {
    /// Builds a one-sided rectangle (quad) area light from its centre, the two
    /// orthonormal in-plane axes, the half-extents and the linear radiance.
    pub(crate) fn rect(
        pos: [f32; 3],
        axis_u: [f32; 3],
        axis_v: [f32; 3],
        half_width: f32,
        half_height: f32,
        color: [f32; 3],
        intensity: f32,
    ) -> Self {
        Self {
            pos_x: pos[0],
            pos_y: pos[1],
            pos_z: pos[2],
            range: 0.0,
            color_x: color[0],
            color_y: color[1],
            color_z: color[2],
            intensity,
            axis_u_x: axis_u[0],
            axis_u_y: axis_u[1],
            axis_u_z: axis_u[2],
            half_width,
            axis_v_x: axis_v[0],
            axis_v_y: axis_v[1],
            axis_v_z: axis_v[2],
            half_height,
            shape: AREA_LIGHT_SHAPE_RECT,
            two_sided: 0,
            radius: 0.0,
            _pad0: 0,
        }
    }
}

/// Single `var<immediate>` block consumed by the standalone
/// `area_light_ltc_main` entry point.
///
/// Packs one quad (four corner positions relative to the shaded point, each a
/// `vec4<f32>` with an unused `w`) plus the two rows of the `LTC` inverse-matrix
/// coefficients so the kernel can evaluate the inlined polygon integration
/// without a bound `LUT` texture — 96 bytes, a multiple of 16.
///
/// `coeffs0 = (a00, a02, a11, a20)` and `coeffs1 = (a22, amplitude, 0, 0)`
/// follow the golden [`prism_render_shading::gi::area_light::ltc_lut::LtcCoeffs`]
/// block-sparse layout.
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable, PartialEq)]
pub(crate) struct GpuAreaLightLtcParams {
    /// Quad corner positions relative to the shaded point (`w` unused).
    pub points: [[f32; 4]; 4],
    /// `LTC` inverse-matrix coefficients `(a00, a02, a11, a20)`.
    pub coeffs0: [f32; 4],
    /// `LTC` inverse-matrix tail `(a22, amplitude, 0, 0)`.
    pub coeffs1: [f32; 4],
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn area_light_is_the_80_byte_scalar_record() {
        // 20 scalars x 4 bytes = 80, a multiple of the 16-byte std430 row.
        assert_eq!(size_of::<GpuAreaLight>(), 80);
        assert_eq!(align_of::<GpuAreaLight>(), 4);
        assert_eq!(size_of::<GpuAreaLight>() % 16, 0);
    }

    #[test]
    fn ltc_params_is_the_96_byte_immediate_block() {
        assert_eq!(size_of::<GpuAreaLightLtcParams>(), 96);
        assert_eq!(align_of::<GpuAreaLightLtcParams>(), 4);
        assert!(size_of::<GpuAreaLightLtcParams>() <= 128);
    }

    #[test]
    fn rect_builder_tags_a_one_sided_quad() {
        let light = GpuAreaLight::rect(
            [1.0, 2.0, 3.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            0.5,
            0.25,
            [0.8, 0.9, 1.0],
            4.0,
        );
        assert_eq!(light.shape, AREA_LIGHT_SHAPE_RECT);
        assert_eq!(light.two_sided, 0);
        assert_eq!(light.pos_x, 1.0);
        assert_eq!(light.half_width, 0.5);
        assert_eq!(light.half_height, 0.25);
        assert_eq!(light.intensity, 4.0);
        assert_eq!(light._pad0, 0);
    }

    #[test]
    fn shape_tags_are_distinct() {
        assert_eq!(AREA_LIGHT_SHAPE_RECT, 0);
        assert_eq!(AREA_LIGHT_SHAPE_DISK, 1);
        assert_eq!(AREA_LIGHT_SHAPE_TUBE, 2);
        assert_eq!(AREA_LIGHT_LTC_WORKGROUP_SIZE, 64);
    }
}
