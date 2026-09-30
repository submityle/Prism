// AABB-frustum cull twin: decides per axis-aligned bounding box whether it is at
// least partially inside the view frustum, mirroring the CPU golden
// `Frustum::intersects_bounds` (in `prism_render_architecture::virtual_geometry`):
// a box survives iff `signed_distance(center) >= -projected_extent` for all six
// inward-facing planes, where `projected_extent = |nx|*hx + |ny|*hy + |nz|*hz`
// is the box's support radius along the plane normal.
//
// This is the box sibling of `sphere_frustum_cull`: a cluster that carries an
// AABB is tested with its three projected half-extents instead of one radius.
// A GPU-driven pipeline runs this per candidate before it spends a LOD or a
// resident page on the cluster. It is the frustum half of the full
// `cluster_cull` decision, isolated so the projected-extent test is validated
// on its own rather than only as a buried sub-step.
//
// Verdict indices: 1 = inside (at least partially visible), 0 = fully outside.
// The arithmetic mirrors the reference term order (`nx*px + ny*py + nz*pz + d`
// for the signed distance, `|nx|*hx + |ny|*hy + |nz|*hz` for the extent) so
// that, away from the razor-thin plane boundary, the discrete verdict is
// identical to the reference regardless of fused-multiply-add contraction. It
// uses only the portable core-WGSL subset - no optional feature - so this twin
// runs unmodified on Metal, Vulkan and DX12.
//
// Provenance: standard AABB projected-radius frustum culling; no Unreal Engine
// source or derived code.

struct Params {
    // Six inward-facing frustum planes, each packed as (normal.xyz, distance).
    // The half-space `dot(normal, p) + distance >= 0` is the interior.
    planes: array<vec4<f32>, 6>,
    // Number of valid boxes in `boxes` and outputs in `verdicts`.
    count: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

// One axis-aligned bounding box. 24-byte stride matching the host upload
// struct; scalar fields avoid vec3 storage-alignment padding.
struct Aabb {
    center_x: f32,
    center_y: f32,
    center_z: f32,
    half_x: f32,
    half_y: f32,
    half_z: f32,
}

const VERDICT_OUTSIDE: u32 = 0u;
const VERDICT_INSIDE: u32 = 1u;

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> boxes: array<Aabb>;
@group(0) @binding(2) var<storage, read_write> verdicts: array<u32>;

@compute @workgroup_size(64)
fn cull(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= params.count) {
        return;
    }
    let b = boxes[i];
    var verdict = VERDICT_INSIDE;
    for (var p = 0u; p < 6u; p = p + 1u) {
        let plane = params.planes[p];
        // signed_distance(center) = nx*cx + ny*cy + nz*cz + d
        let signed_distance =
            plane.x * b.center_x + plane.y * b.center_y + plane.z * b.center_z + plane.w;
        // projected_extent = |nx|*hx + |ny|*hy + |nz|*hz
        let projected_extent =
            abs(plane.x) * b.half_x + abs(plane.y) * b.half_y + abs(plane.z) * b.half_z;
        if (signed_distance < -projected_extent) {
            verdict = VERDICT_OUTSIDE;
        }
    }
    verdicts[i] = verdict;
}
