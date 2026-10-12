//! The `#[repr(C)]` GPU parameter struct shared with the WGSL uniform, plus the
//! Rust byte packing that fills a uniform buffer.
//!
//! # Field layout
//!
//! WGSL lays a `vec3<f32>` member out with 16-byte alignment and 12-byte size;
//! a `#[repr(C)] [f32; 3]` has 4-byte alignment. The two agree byte-for-byte as
//! long as **every vector member starts on a 16-byte offset**, so the table
//! below is the contract both sides satisfy. Fields are grouped into 16-byte
//! blocks; a `f32` that follows a vector lands in the vector's tail slot, and
//! the `_pad` members are explicit rather than implicit compiler padding so the
//! offsets are self-documenting.
//!
//! | offset | bytes | field | WGSL type |
//! |--------|-------|-------|-----------|
//! |  0 | 12 | `base_color` | `vec3<f32>` |
//! | 12 |  4 | `base_weight` | `f32` |
//! | 16 | 12 | `specular_color` | `vec3<f32>` |
//! | 28 |  4 | `diffuse_roughness` | `f32` |
//! | 32 | 12 | `emissive` | `vec3<f32>` |
//! | 44 |  4 | `metallic` | `f32` |
//! | 48 |  4 | `perceptual_roughness` | `f32` |
//! | 52 |  4 | `specular_weight` | `f32` |
//! | 56 |  4 | `specular_ior` | `f32` |
//! | 60 |  4 | `normal_variance` | `f32` |
//! | 64 |  4 | `anisotropy` | `f32` |
//! | 68 |  4 | `anisotropy_rotation` | `f32` |
//! | 72 |  4 | `reserved0` | `f32` |
//! | 76 |  4 | `reserved1` | `f32` |
//! | 80 | 12 | `normal` | `vec3<f32>` |
//! | 92 |  4 | `normal_pad` | `f32` |
//! | 96 | 12 | `view` | `vec3<f32>` |
//! | 108 | 4 | `view_pad` | `f32` |
//! | 112 | 12 | `tangent` | `vec3<f32>` |
//! | 124 | 4 | `tangent_pad` | `f32` |
//! | 128 | 12 | `bitangent` | `vec3<f32>` |
//! | 140 | 4 | `bitangent_pad` | `f32` |
//! | 144 | 12 | `direction` | `vec3<f32>` |
//! | 156 | 4 | `visibility` | `f32` |
//! | 160 | 12 | `illuminance` | `vec3<f32>` |
//! | 172 | 4 | `illuminance_pad` | `f32` |
//!
//! Total: 176 bytes, a multiple of 16, so an `array<GpuSurfaceParams, N>` in a
//! uniform buffer has the required 16-byte element stride.
//!
//! # Rust <-> GPU packing
//!
//! The struct is written as little-endian IEEE-754 `f32`s in declaration order:
//! [`GpuSurfaceParams::to_bytes`] emits exactly [`SURFACE_PARAMS_SIZE`] bytes,
//! which the WGSL `SurfaceParams` struct reads back at the offsets above.
//! Because the CPU golden and the shader both consume `f32`, no format
//! conversion, quantisation, or half-precision step is involved.

use core::mem::size_of;

/// The maximum number of surface evaluations in one uniform batch.
///
/// Mirrors `MAX_BATCH` in `surface.wgsl`. The uniform buffer the pass uploads
/// is sized `MAX_BATCH * SURFACE_PARAMS_SIZE` bytes (20 KiB), comfortably inside
/// the default 64 KiB `maxUniformBufferBindingSize` limit.
pub const MAX_BATCH: u32 = 128;

/// The byte size of one [`GpuSurfaceParams`].
pub const SURFACE_PARAMS_SIZE: usize = 176;

/// A complete surface-shading input: the base-lobe material, the shading frame,
/// and the direct light, in the exact shape `surface.wgsl` reads.
///
/// Every field default from [`Default`] matches
/// `prism_render_shading::lighting::SurfaceSample` / `ShadingFrame` /
/// `DirectLightSample` for the mirrored subset, so a `..Default::default()` call
/// site is a valid, neutral surface.
///
/// The struct carries no coat / fuzz / transmission / subsurface / iridescence
/// field: those `OpenPBR` lobes are outside the mirror (see the crate docs), and
/// omitting them means a call site cannot ask the GPU for a lobe it does not
/// evaluate.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GpuSurfaceParams {
    /// Base albedo (`OpenPBR` `base_color`), linear RGB.
    pub base_color: [f32; 3],
    /// `OpenPBR` `base_weight`: scales the whole base diffuse + specular layer.
    pub base_weight: f32,
    /// `OpenPBR` `specular_color`: per-channel tint of the dielectric specular.
    pub specular_color: [f32; 3],
    /// `OpenPBR` `diffuse_roughness` in `[0, 1]`; drives the Oren-Nayar factor.
    pub diffuse_roughness: f32,
    /// Emissive radiance, added after the `n.l` weighting.
    pub emissive: [f32; 3],
    /// Metalness in `[0, 1]`; blends the dielectric and metal Fresnel and
    /// removes the diffuse lobe as it rises.
    pub metallic: f32,
    /// Perceptual roughness in `[0, 1]`; squared into the GGX width.
    pub perceptual_roughness: f32,
    /// `OpenPBR` `specular_weight`: scales the dielectric specular interface.
    pub specular_weight: f32,
    /// `OpenPBR` `specular_ior`: relative index of the dielectric interface.
    pub specular_ior: f32,
    /// Specular anti-aliasing normal variance in the `alpha^2` domain.
    pub normal_variance: f32,
    /// `OpenPBR` `specular_roughness_anisotropy` in `[0, 1]`.
    pub anisotropy: f32,
    /// Rotation of the anisotropy axes around the normal, in radians.
    pub anisotropy_rotation: f32,
    /// Reserved padding slot; keeps the following vector 16-byte aligned.
    pub reserved0: f32,
    /// Reserved padding slot; keeps the following vector 16-byte aligned.
    pub reserved1: f32,
    /// World-space shading normal.
    pub normal: [f32; 3],
    /// Padding after [`Self::normal`].
    pub normal_pad: f32,
    /// World-space view direction (surface to eye).
    pub view: [f32; 3],
    /// Padding after [`Self::view`].
    pub view_pad: f32,
    /// Unit tangent (major anisotropy axis), orthogonal to [`Self::normal`].
    pub tangent: [f32; 3],
    /// Padding after [`Self::tangent`].
    pub tangent_pad: f32,
    /// Unit bitangent carrying the frame handedness.
    pub bitangent: [f32; 3],
    /// Padding after [`Self::bitangent`].
    pub bitangent_pad: f32,
    /// World-space direction toward the light.
    pub direction: [f32; 3],
    /// Scalar shadow visibility in `[0, 1]`.
    pub visibility: f32,
    /// Light illuminance, linear RGB.
    pub illuminance: [f32; 3],
    /// Padding after [`Self::illuminance`].
    pub illuminance_pad: f32,
}

impl Default for GpuSurfaceParams {
    fn default() -> Self {
        // Mirrors the CPU `SurfaceSample::default()` for every mirrored field,
        // plus the canonical shading frame and a unit white light.
        Self {
            base_color: [0.8; 3],
            base_weight: 1.0,
            specular_color: [1.0; 3],
            diffuse_roughness: 0.0,
            emissive: [0.0; 3],
            metallic: 0.0,
            perceptual_roughness: 0.3,
            specular_weight: 1.0,
            specular_ior: 1.5,
            normal_variance: 0.0,
            anisotropy: 0.0,
            anisotropy_rotation: 0.0,
            reserved0: 0.0,
            reserved1: 0.0,
            normal: [0.0, 1.0, 0.0],
            normal_pad: 0.0,
            view: [0.0, 1.0, 0.0],
            view_pad: 0.0,
            tangent: [1.0, 0.0, 0.0],
            tangent_pad: 0.0,
            bitangent: [0.0, 0.0, -1.0],
            bitangent_pad: 0.0,
            direction: [0.0, 1.0, 0.0],
            visibility: 1.0,
            illuminance: [1.0; 3],
            illuminance_pad: 0.0,
        }
    }
}

/// The 44 `f32` slots of one [`GpuSurfaceParams`], in declaration order.
///
/// This is the single place the field order is written down for serialisation;
/// [`GpuSurfaceParams::to_bytes`] drives off it, and the layout tests pin its
/// length to [`SURFACE_PARAMS_SIZE`].
const FIELD_COUNT: usize = SURFACE_PARAMS_SIZE / size_of::<f32>();

impl GpuSurfaceParams {
    /// The number of `f32` slots written per surface (44).
    pub const FLOATS: usize = FIELD_COUNT;

    /// The packed byte size of one surface ([`SURFACE_PARAMS_SIZE`]).
    pub const SIZE: usize = SURFACE_PARAMS_SIZE;

    /// The fields as a flat `f32` array in WGSL declaration order.
    #[must_use]
    pub fn to_floats(self) -> [f32; FIELD_COUNT] {
        [
            self.base_color[0],
            self.base_color[1],
            self.base_color[2],
            self.base_weight,
            self.specular_color[0],
            self.specular_color[1],
            self.specular_color[2],
            self.diffuse_roughness,
            self.emissive[0],
            self.emissive[1],
            self.emissive[2],
            self.metallic,
            self.perceptual_roughness,
            self.specular_weight,
            self.specular_ior,
            self.normal_variance,
            self.anisotropy,
            self.anisotropy_rotation,
            self.reserved0,
            self.reserved1,
            self.normal[0],
            self.normal[1],
            self.normal[2],
            self.normal_pad,
            self.view[0],
            self.view[1],
            self.view[2],
            self.view_pad,
            self.tangent[0],
            self.tangent[1],
            self.tangent[2],
            self.tangent_pad,
            self.bitangent[0],
            self.bitangent[1],
            self.bitangent[2],
            self.bitangent_pad,
            self.direction[0],
            self.direction[1],
            self.direction[2],
            self.visibility,
            self.illuminance[0],
            self.illuminance[1],
            self.illuminance[2],
            self.illuminance_pad,
        ]
    }

    /// Packs the surface into `out` as little-endian `f32`s, using exactly
    /// [`SURFACE_PARAMS_SIZE`] bytes.
    ///
    /// # Panics
    ///
    /// Panics when `out` is shorter than [`SURFACE_PARAMS_SIZE`].
    pub fn write_into(self, out: &mut [u8]) {
        assert!(
            out.len() >= SURFACE_PARAMS_SIZE,
            "the destination slice must hold at least {SURFACE_PARAMS_SIZE} bytes",
        );
        for (index, value) in self.to_floats().iter().enumerate() {
            let start = index * size_of::<f32>();
            out[start..start + size_of::<f32>()].copy_from_slice(&value.to_le_bytes());
        }
    }

    /// Packs the surface into a fresh byte array (see [`Self::write_into`]).
    #[must_use]
    pub fn to_bytes(self) -> [u8; SURFACE_PARAMS_SIZE] {
        let mut out = [0u8; SURFACE_PARAMS_SIZE];
        self.write_into(&mut out);
        out
    }
}

/// Byte size of the whole uniform batch the pass uploads.
pub const BATCH_BYTES: u64 = MAX_BATCH as u64 * SURFACE_PARAMS_SIZE as u64;

const _: () = assert!(
    size_of::<GpuSurfaceParams>() == SURFACE_PARAMS_SIZE,
    "GpuSurfaceParams must be 176 bytes to match the WGSL surface.wgsl layout",
);

const _: () = assert!(
    BATCH_BYTES < 64 * 1024,
    "the uniform batch must fit inside the default 64 KiB binding limit",
);
