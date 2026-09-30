// Sphere-frustum cull twin: decides per bounding sphere whether it is at least
// partially inside the view frustum, mirroring the CPU golden
// `Frustum::contains_sphere` (in `prism_render_architecture::virtual_geometry`):
// a sphere survives iff `signed_distance(center) >= -radius` for all six
// inward-facing planes.
//
// This is the bounding-sphere culling primitive, distinct from the AABB
// projected-radius test the cluster culler uses: cheaper (one radius instead of
// three projected half-extents) and the natural test when a cluster carries a
// bounding sphere rather than a box. A GPU-driven pipeline runs this per
// candidate before it spends a LOD or a resident page on the cluster.
//
// Verdict indices: 1 = inside (at least partially visible), 0 = fully outside.
// The arithmetic mirrors the reference term order
// (`nx*px + ny*py + nz*pz + d` for each plane's signed distance) so that, away
// from the razor-thin plane boundary, the discrete verdict is identical to the
// reference regardless of fused-multiply-add contraction. It uses only the
// portable core-WGSL subset - no optional feature - so this twin runs
// unmodified on Metal, Vulkan and DX12.
//
// Provenance: standard sphere-frustum half-space culling; no Unreal Engine
// source or derived code.

struct Params {
    // Six inward-facing frustum planes, each packed as (normal.xyz, distance).
    // The half-space `dot(normal, p) + distance >= 0` is the interior.
    planes: array<vec4<f32>, 6>,
    // Number of valid spheres in `spheres` and outputs in `verdicts`.
    count: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

// One bounding sphere. 16-byte stride matching the host upload struct; scalar
// fields avoid vec3 storage-alignment padding.
struct Sphere {
    center_x: f32,
    center_y: f32,
    center_z: f32,
    radius: f32,
}

const VERDICT_OUTSIDE: u32 = 0u;
const VERDICT_INSIDE: u32 = 1u;

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> spheres: array<Sphere>;
@group(0) @binding(2) var<storage, read_write> verdicts: array<u32>;

@compute @workgroup_size(64)
fn cull(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= params.count) {
        return;
    }
    let s = spheres[i];
    var verdict = VERDICT_INSIDE;
    for (var p = 0u; p < 6u; p = p + 1u) {
        let plane = params.planes[p];
        // signed_distance(center) = nx*px + ny*py + nz*pz + d
        let signed_distance =
            plane.x * s.center_x + plane.y * s.center_y + plane.z * s.center_z + plane.w;
        if (signed_distance < -s.radius) {
            verdict = VERDICT_OUTSIDE;
        }
    }
    verdicts[i] = verdict;
}
