//! GPU surface-reconstruction library.
//!
//! Byte-for-byte numerical twin of the CPU golden reference in
//! `prism_render_shading::surface` (`reconstruct_surface`).  The table ABI
//! mirrors the `#[repr(C)]` records in `geometry/shading.rs`
//! (`RenderShadingGeometryHeader`, `GpuShadingVertex`, `GpuShadingPrimitive`),
//! validated there against `size_of`, so the same storage buffers bind
//! identically on both sides.
//!
//! Bounds/offset addressing and the stale-generation guards live in the
//! `shading_resolve` entry point (they need the storage arrays); this module
//! owns the pure interpolation math so it can be unit-compiled standalone and
//! kept in lock-step with the CPU reference.

// Geometry-table header flags, mirroring the `SHADING_GEOMETRY_FLAG_*`
// constants in `geometry/shading.rs`.
const SHADING_GEOMETRY_FLAG_ACTIVE: u32 = 1u;
const SHADING_GEOMETRY_FLAG_MISSING_NORMAL: u32 = 2u;
const SHADING_GEOMETRY_FLAG_MISSING_UV: u32 = 4u;
const SHADING_GEOMETRY_FLAG_INVALID: u32 = 8u;

// Non-fatal reconstruction flags, mirroring `SurfaceReconstructionFlags`.
const SURFACE_FLAG_MISSING_NORMAL: u32 = 1u;
const SURFACE_FLAG_MISSING_UV: u32 = 2u;
const SURFACE_FLAG_DEGENERATE_NORMAL: u32 = 4u;
const SURFACE_FLAG_INVALID_PRIMITIVE: u32 = 8u;

// Geometry-table header addressed by a stable geometry index. Mirrors
// `RenderShadingGeometryHeader` (32 bytes).
struct RenderShadingGeometryHeader {
    generation: u32,
    revision: u32,
    vertex_offset: u32,
    vertex_count: u32,
    primitive_offset: u32,
    primitive_count: u32,
    flags: u32,
    padding: u32,
}

// One vertex row (48 bytes). Mirrors `GpuShadingVertex`; the explicit padding
// scalars keep the `vec3<f32>` members on their natural 16-byte alignment so
// the std430 layout matches the Rust `#[repr(C)]` record exactly.
struct GpuShadingVertex {
    position: vec3<f32>,
    position_padding: f32,
    normal: vec3<f32>,
    normal_padding: f32,
    uv: vec2<f32>,
    flags: u32,
    padding: u32,
}

// One triangle row (16 bytes). Mirrors `GpuShadingPrimitive`.
struct GpuShadingPrimitive {
    indices: array<u32, 3>,
    flags: u32,
}

// Reconstructed interpolants used by PBR/NPR shading. Mirrors
// `SurfaceSampleGeometry`.
struct SurfaceSampleGeometry {
    position: vec3<f32>,
    normal: vec3<f32>,
    uv: vec2<f32>,
    flags: u32,
}

// Normalizes `value`, falling back to `fallback` for degenerate inputs.
// Mirrors `normalize_or` in `surface.rs`.
fn surface_normalize_or(value: vec3<f32>, fallback: vec3<f32>) -> vec3<f32> {
    let length_squared = dot(value, value);
    if length_squared > 1.0e-12 {
        return value * inverseSqrt(length_squared);
    }
    return fallback;
}

fn surface_interpolate3(a: vec3<f32>, b: vec3<f32>, c: vec3<f32>, weights: vec3<f32>) -> vec3<f32> {
    return a * weights.x + b * weights.y + c * weights.z;
}

fn surface_interpolate2(a: vec2<f32>, b: vec2<f32>, c: vec2<f32>, weights: vec3<f32>) -> vec2<f32> {
    return a * weights.x + b * weights.y + c * weights.z;
}

// Geometric face normal from three positions. Mirrors `face_normal`.
fn surface_face_normal(a: vec3<f32>, b: vec3<f32>, c: vec3<f32>) -> vec3<f32> {
    return surface_normalize_or(cross(b - a, c - a), vec3<f32>(0.0));
}

// Decodes barycentric weights from the packed visibility metadata channel.
// `metadata_w = pack2x16unorm(clamp(bary.xy, 0, 1))`; the third weight is
// recovered as `1 - x - y` (clamped to non-negative to absorb rounding).
fn surface_decode_barycentrics(metadata_w: u32) -> vec3<f32> {
    let bxy = unpack2x16unorm(metadata_w);
    let bz = max(1.0 - bxy.x - bxy.y, 0.0);
    return vec3<f32>(bxy.x, bxy.y, bz);
}

// Interpolates the three fetched vertices into a shaded surface sample.
//
// Mirrors the interpolation half of `reconstruct_surface` (the bounds and
// generation guards are applied by the caller before the vertices are read):
// prefer the interpolated authored normal, then the geometric face normal, and
// only as a last resort a canonical up vector, always flagging that fallback.
fn interpolate_surface(
    va: GpuShadingVertex,
    vb: GpuShadingVertex,
    vc: GpuShadingVertex,
    primitive_flags: u32,
    weights: vec3<f32>,
) -> SurfaceSampleGeometry {
    var result: SurfaceSampleGeometry;
    result.position = surface_interpolate3(va.position, vb.position, vc.position, weights);
    result.uv = surface_interpolate2(va.uv, vb.uv, vc.uv, weights);
    var flags = va.flags | vb.flags | vc.flags | primitive_flags;

    let interpolated = surface_normalize_or(
        surface_interpolate3(va.normal, vb.normal, vc.normal, weights),
        vec3<f32>(0.0),
    );
    if dot(interpolated, interpolated) > 0.0 {
        result.normal = interpolated;
    } else {
        let face = surface_face_normal(va.position, vb.position, vc.position);
        if dot(face, face) > 0.0 {
            result.normal = face;
        } else {
            flags = flags | SURFACE_FLAG_DEGENERATE_NORMAL;
            result.normal = vec3<f32>(0.0, 1.0, 0.0);
        }
    }
    result.flags = flags;
    return result;
}
