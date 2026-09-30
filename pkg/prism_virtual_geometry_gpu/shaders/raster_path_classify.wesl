// Raster-path classification twin: chooses a GPU rasterization strategy per
// cluster on-device, mirroring the CPU golden `select_raster_path`.
//
// A GPU-driven virtual-geometry pipeline classifies every visible cluster into
// one of four raster paths before it bins draws: sub-pixel clusters take the
// compute software rasterizer (no hardware setup overhead); larger clusters
// prefer mesh shaders, then hardware indirect draws, then a plain mesh
// fallback on capability-poor backends. The CPU golden
// `select_raster_path` (in `prism_render_architecture::virtual_geometry`) owns
// that decision; this kernel is the on-device twin that runs one thread per
// cluster and writes the same path index the reference returns.
//
// Path indices match `GeometryRasterPath as u32` in declaration order:
//   0 = MeshShader, 1 = ComputeSoftware, 2 = IndirectHardware, 3 = FallbackMesh.
//
// The whole decision is integer branching plus a single `<=` compare of the
// per-cluster `max_triangle_pixels` against the clamped `threshold`; there is
// no floating-point arithmetic (only the compare and a `max(threshold, 0.0)`
// clamp evaluated identically on both sides), so every emitted path index is
// bit-exact against the reference. It uses only the portable core-WGSL subset -
// no optional feature - so this twin runs unmodified on Metal, Vulkan and DX12.
//
// Provenance: standard cluster raster-path selection heuristic; no Unreal
// Engine source or derived code.

struct Params {
    // Number of valid clusters in `stats` and outputs in `paths`.
    count: u32,
    // Backend capability bits: bit 0 = mesh_shader, bit 1 = hardware_indirect.
    capability_bits: u32,
    // Software-raster pixel threshold; a cluster whose largest triangle is at
    // or below `max(threshold, 0.0)` takes the compute software path.
    threshold: f32,
    // Padding to a 16-byte uniform tail.
    pad0: u32,
}

// One cluster's raster statistics. `8`-byte stride matching the `bytemuck`
// upload struct.
struct ClusterStats {
    // Largest projected triangle area in the cluster, in pixels squared.
    max_triangle_pixels: f32,
    // Number of triangles in the cluster.
    triangle_count: u32,
}

// Path index constants, matching `GeometryRasterPath as u32`.
const PATH_MESH_SHADER: u32 = 0u;
const PATH_COMPUTE_SOFTWARE: u32 = 1u;
const PATH_INDIRECT_HARDWARE: u32 = 2u;
const PATH_FALLBACK_MESH: u32 = 3u;

// Capability bit masks.
const CAP_MESH_SHADER: u32 = 1u;
const CAP_HARDWARE_INDIRECT: u32 = 2u;

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> stats: array<ClusterStats>;
@group(0) @binding(2) var<storage, read_write> paths: array<u32>;

// One thread per cluster.
@compute @workgroup_size(64, 1, 1)
fn classify(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= params.count) {
        return;
    }

    let s = stats[i];

    // Empty clusters resolve to the inert fallback default.
    if (s.triangle_count == 0u) {
        paths[i] = PATH_FALLBACK_MESH;
        return;
    }

    // Tiny clusters take the compute software rasterizer. The threshold is
    // clamped to `>= 0` exactly as the reference's `software_pixel_threshold.max(0.0)`.
    let clamped_threshold = max(params.threshold, 0.0);
    if (s.max_triangle_pixels <= clamped_threshold) {
        paths[i] = PATH_COMPUTE_SOFTWARE;
        return;
    }

    // Larger clusters take the best available hardware path.
    let mesh_shader = (params.capability_bits & CAP_MESH_SHADER) != 0u;
    let hardware_indirect = (params.capability_bits & CAP_HARDWARE_INDIRECT) != 0u;
    if (mesh_shader) {
        paths[i] = PATH_MESH_SHADER;
    } else if (hardware_indirect) {
        paths[i] = PATH_INDIRECT_HARDWARE;
    } else {
        paths[i] = PATH_FALLBACK_MESH;
    }
}
