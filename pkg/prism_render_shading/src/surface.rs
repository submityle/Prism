//! Stable geometry-to-surface reconstruction contract for compute shading.
//!
//! The visibility buffer identifies a scene instance, a geometry primitive and
//! barycentrics.  It deliberately does not depend on Bevy's vertex-buffer
//! layout.  These compact rows are the format consumed by the future Vulkan
//! resolve pass and by the CPU reference tests.

use core::fmt;

/// One vertex in the compute-friendly shading table.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, bytemuck::Pod, bytemuck::Zeroable)]
pub struct GpuShadingVertex {
    pub position: [f32; 3],
    pub _position_padding: f32,
    pub normal: [f32; 3],
    pub _normal_padding: f32,
    pub uv: [f32; 2],
    pub flags: u32,
    pub _padding: u32,
}

impl Default for GpuShadingVertex {
    fn default() -> Self {
        Self {
            position: [0.0; 3],
            _position_padding: 0.0,
            normal: [0.0, 1.0, 0.0],
            _normal_padding: 0.0,
            uv: [0.0; 2],
            flags: SurfaceReconstructionFlags::MISSING_NORMAL.bits()
                | SurfaceReconstructionFlags::MISSING_UV.bits(),
            _padding: 0,
        }
    }
}

/// Three vertex indices for one triangle primitive.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, bytemuck::Pod, bytemuck::Zeroable)]
pub struct GpuShadingPrimitive {
    pub indices: [u32; 3],
    pub flags: u32,
}

/// Inputs required to reconstruct a surface from one visibility pixel.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SurfaceReconstructionInput {
    pub primitive_id: u32,
    pub barycentrics: [f32; 3],
    pub geometry_generation: u32,
    pub expected_geometry_generation: u32,
}

/// Reconstructed interpolants used by PBR, NPR, motion and material sampling.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SurfaceSampleGeometry {
    pub position: [f32; 3],
    pub normal: [f32; 3],
    pub uv: [f32; 2],
    pub flags: SurfaceReconstructionFlags,
}

/// Non-fatal conditions recorded alongside a reconstructed sample.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct SurfaceReconstructionFlags(u32);

impl SurfaceReconstructionFlags {
    pub const MISSING_NORMAL: Self = Self(1 << 0);
    pub const MISSING_UV: Self = Self(1 << 1);
    pub const DEGENERATE_NORMAL: Self = Self(1 << 2);
    pub const INVALID_PRIMITIVE: Self = Self(1 << 3);

    pub const fn bits(self) -> u32 {
        self.0
    }

    const fn from_bits(bits: u32) -> Self {
        Self(bits)
    }

    pub const fn contains(self, other: Self) -> bool {
        self.0 & other.0 != 0
    }
}

impl core::ops::BitOr for SurfaceReconstructionFlags {
    type Output = Self;

    fn bitor(self, rhs: Self) -> Self::Output {
        Self(self.0 | rhs.0)
    }
}

/// Errors that invalidate a visibility pixel for compute shading.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum SurfaceReconstructionError {
    StaleGeometryGeneration,
    PrimitiveOutOfBounds,
    VertexOutOfBounds,
    NonFiniteBarycentrics,
    BarycentricsOutsideTriangle,
}

impl fmt::Display for SurfaceReconstructionError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(match self {
            Self::StaleGeometryGeneration => "stale geometry generation",
            Self::PrimitiveOutOfBounds => "primitive index is out of bounds",
            Self::VertexOutOfBounds => "vertex index is out of bounds",
            Self::NonFiniteBarycentrics => "barycentrics are not finite",
            Self::BarycentricsOutsideTriangle => "barycentrics are outside the triangle",
        })
    }
}

/// Reconstructs interpolated position, normal and UV without depending on a
/// backend vertex layout.  Missing attributes are explicit fallbacks, never
/// silently interpreted as valid authored data.
pub fn reconstruct_surface(
    input: SurfaceReconstructionInput,
    primitives: &[GpuShadingPrimitive],
    vertices: &[GpuShadingVertex],
) -> Result<SurfaceSampleGeometry, SurfaceReconstructionError> {
    if input.geometry_generation != input.expected_geometry_generation {
        return Err(SurfaceReconstructionError::StaleGeometryGeneration);
    }
    if !input.barycentrics.iter().all(|value| value.is_finite()) {
        return Err(SurfaceReconstructionError::NonFiniteBarycentrics);
    }
    let sum = input.barycentrics.into_iter().sum::<f32>();
    if input.barycentrics.iter().any(|value| *value < -1.0e-4) || (sum - 1.0).abs() > 1.0e-4 {
        return Err(SurfaceReconstructionError::BarycentricsOutsideTriangle);
    }
    let primitive = primitives
        .get(input.primitive_id as usize)
        .ok_or(SurfaceReconstructionError::PrimitiveOutOfBounds)?;
    let [a, b, c] = primitive.indices;
    let vertex = |index| {
        vertices
            .get(index as usize)
            .ok_or(SurfaceReconstructionError::VertexOutOfBounds)
    };
    let [va, vb, vc] = [vertex(a)?, vertex(b)?, vertex(c)?];
    let weights = input.barycentrics;
    let position = interpolate3(va.position, vb.position, vc.position, weights);
    let uv = interpolate2(va.uv, vb.uv, vc.uv, weights);
    let mut flags = SurfaceReconstructionFlags::from_bits(
        va.flags | vb.flags | vc.flags | primitive.flags,
    );
    // Prefer the interpolated authored normal, then the geometric face
    // normal, and only as a last resort a canonical up vector.  A degenerate
    // last-resort fallback is always flagged so downstream passes can react.
    let normal = {
        let interpolated =
            normalize_or(interpolate3(va.normal, vb.normal, vc.normal, weights), [0.0; 3]);
        if interpolated != [0.0; 3] {
            interpolated
        } else {
            let face = face_normal(va.position, vb.position, vc.position);
            if face != [0.0; 3] {
                face
            } else {
                flags = flags | SurfaceReconstructionFlags::DEGENERATE_NORMAL;
                [0.0, 1.0, 0.0]
            }
        }
    };
    Ok(SurfaceSampleGeometry {
        position,
        normal,
        uv,
        flags,
    })
}

fn interpolate3(a: [f32; 3], b: [f32; 3], c: [f32; 3], weights: [f32; 3]) -> [f32; 3] {
    [
        a[0] * weights[0] + b[0] * weights[1] + c[0] * weights[2],
        a[1] * weights[0] + b[1] * weights[1] + c[1] * weights[2],
        a[2] * weights[0] + b[2] * weights[1] + c[2] * weights[2],
    ]
}

fn interpolate2(a: [f32; 2], b: [f32; 2], c: [f32; 2], weights: [f32; 3]) -> [f32; 2] {
    [
        a[0] * weights[0] + b[0] * weights[1] + c[0] * weights[2],
        a[1] * weights[0] + b[1] * weights[1] + c[1] * weights[2],
    ]
}

fn face_normal(a: [f32; 3], b: [f32; 3], c: [f32; 3]) -> [f32; 3] {
    normalize_or(cross(sub(b, a), sub(c, a)), [0.0; 3])
}

fn cross(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

fn sub(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

fn normalize_or(value: [f32; 3], fallback: [f32; 3]) -> [f32; 3] {
    let length_squared = value.into_iter().map(|component| component * component).sum::<f32>();
    if length_squared > 1.0e-12 && length_squared.is_finite() {
        let inverse = length_squared.sqrt().recip();
        [value[0] * inverse, value[1] * inverse, value[2] * inverse]
    } else {
        fallback
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn triangle() -> ([GpuShadingPrimitive; 1], [GpuShadingVertex; 3]) {
        (
            [GpuShadingPrimitive {
                indices: [0, 1, 2],
                flags: 0,
            }],
            [
                GpuShadingVertex {
                    position: [0.0, 0.0, 0.0],
                    _position_padding: 0.0,
                    normal: [0.0, 0.0, 1.0],
                    _normal_padding: 0.0,
                    uv: [0.0, 0.0],
                    flags: 0,
                    _padding: 0,
                },
                GpuShadingVertex {
                    position: [1.0, 0.0, 0.0],
                    _position_padding: 0.0,
                    normal: [0.0, 0.0, 1.0],
                    _normal_padding: 0.0,
                    uv: [1.0, 0.0],
                    flags: 0,
                    _padding: 0,
                },
                GpuShadingVertex {
                    position: [0.0, 1.0, 0.0],
                    _position_padding: 0.0,
                    normal: [0.0, 0.0, 1.0],
                    _normal_padding: 0.0,
                    uv: [0.0, 1.0],
                    flags: 0,
                    _padding: 0,
                },
            ],
        )
    }

    #[test]
    fn interpolates_surface_attributes_and_preserves_layout() {
        let (primitives, vertices) = triangle();
        let sample = reconstruct_surface(
            SurfaceReconstructionInput {
                primitive_id: 0,
                barycentrics: [0.25, 0.5, 0.25],
                geometry_generation: 7,
                expected_geometry_generation: 7,
            },
            &primitives,
            &vertices,
        )
        .unwrap();
        assert_eq!(sample.position, [0.5, 0.25, 0.0]);
        assert_eq!(sample.uv, [0.5, 0.25]);
        assert_eq!(sample.normal, [0.0, 0.0, 1.0]);
        assert_eq!(size_of::<GpuShadingVertex>(), 48);
        assert_eq!(size_of::<GpuShadingPrimitive>(), 16);
    }

    #[test]
    fn rejects_stale_and_out_of_bounds_visibility_data() {
        let (primitives, vertices) = triangle();
        let base = SurfaceReconstructionInput {
            primitive_id: 0,
            barycentrics: [1.0, 0.0, 0.0],
            geometry_generation: 2,
            expected_geometry_generation: 3,
        };
        assert_eq!(
            reconstruct_surface(base, &primitives, &vertices),
            Err(SurfaceReconstructionError::StaleGeometryGeneration)
        );
        assert_eq!(
            reconstruct_surface(
                SurfaceReconstructionInput {
                    primitive_id: 1,
                    geometry_generation: 3,
                    expected_geometry_generation: 3,
                    ..base
                },
                &primitives,
                &vertices,
            ),
            Err(SurfaceReconstructionError::PrimitiveOutOfBounds)
        );
    }

    #[test]
    fn falls_back_to_face_normal_for_missing_or_degenerate_authored_normals() {
        let (primitives, mut vertices) = triangle();
        for vertex in &mut vertices {
            vertex.normal = [0.0; 3];
            vertex.flags = SurfaceReconstructionFlags::MISSING_NORMAL.bits();
        }
        let sample = reconstruct_surface(
            SurfaceReconstructionInput {
                primitive_id: 0,
                barycentrics: [1.0, 0.0, 0.0],
                geometry_generation: 1,
                expected_geometry_generation: 1,
            },
            &primitives,
            &vertices,
        )
        .unwrap();
        assert_eq!(sample.normal, [0.0, 0.0, 1.0]);
        assert!(sample.flags.contains(SurfaceReconstructionFlags::MISSING_NORMAL));
    }

    #[test]
    fn flags_and_recovers_from_fully_degenerate_geometry() {
        let (primitives, mut vertices) = triangle();
        // Collapse the triangle to a single point so neither the authored nor
        // the face normal can be recovered.
        for vertex in &mut vertices {
            vertex.position = [1.0, 2.0, 3.0];
            vertex.normal = [0.0; 3];
        }
        let sample = reconstruct_surface(
            SurfaceReconstructionInput {
                primitive_id: 0,
                barycentrics: [0.2, 0.3, 0.5],
                geometry_generation: 1,
                expected_geometry_generation: 1,
            },
            &primitives,
            &vertices,
        )
        .unwrap();
        assert_eq!(sample.normal, [0.0, 1.0, 0.0]);
        assert!(sample
            .flags
            .contains(SurfaceReconstructionFlags::DEGENERATE_NORMAL));
    }
}
