// Cluster-cull twin: decides per cluster whether it survives frustum and
// occlusion culling on-device, mirroring the CPU golden `cluster_cull`.
//
// A GPU-driven virtual-geometry pipeline culls every candidate cluster before
// it selects a LOD or a resident page: a cluster that leaves the view frustum,
// or that a nearer occluder fully hides, needs neither. The CPU golden
// `cluster_cull` (in `prism_render_architecture::virtual_geometry`) owns that
// decision; this kernel is the on-device twin that runs one thread per cluster
// and writes the same verdict index the reference returns.
//
// Verdict indices match `CullVerdict as u32` in declaration order:
//   0 = Visible, 1 = FrustumCulled, 2 = OcclusionCulled.
//
// Frustum rejection takes precedence: occlusion is only consulted for clusters
// that survive all six planes, exactly as the reference short-circuits. The
// arithmetic mirrors the reference term order (`nx*px + ny*py + nz*pz + d` for
// the signed distance and `|nx|*hx + |ny|*hy + |nz|*hz` for the projected
// extent) so that, away from the razor-thin plane boundary, the discrete
// verdict is identical to the reference regardless of fused-multiply-add
// contraction. It uses only the portable core-WGSL subset - no optional
// feature - so this twin runs unmodified on Metal, Vulkan and DX12.
//
// Provenance: standard frustum projected-radius / Hi-Z occlusion culling; no
// Unreal Engine source or derived code.

struct Params {
    // Six inward-facing frustum planes, each packed as (normal.xyz, distance).
    // The half-space `dot(normal, p) + distance >= 0` is the interior.
    planes: array<vec4<f32>, 6>,
    // Number of valid clusters in `clusters` and outputs in `verdicts`.
    count: u32,
    // Padding to a 16-byte uniform tail.
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

// One cluster's bounds plus optional occlusion probe. `48`-byte stride matching
// the `bytemuck` upload struct; `center`/`half_extents` are `vec3<f32>` (16-byte
// aligned) with a trailing scalar packed into each vec3's padding word.
struct Cluster {
    // AABB centre in world space.
    center: vec3<f32>,
    // `1` when the occlusion probe is present, `0` when it should be skipped.
    has_occlusion: u32,
    // AABB half-extents (support radius per axis).
    half_extents: vec3<f32>,
    // Depth of the cluster's closest point (smaller is nearer).
    closest_depth: f32,
    // Conservative nearest occluder depth over the cluster footprint.
    occluder_depth: f32,
    // Padding to the 48-byte stride.
    pad0: f32,
    pad1: f32,
    pad2: f32,
}

// Verdict index constants, matching `CullVerdict as u32`.
const VERDICT_VISIBLE: u32 = 0u;
const VERDICT_FRUSTUM_CULLED: u32 = 1u;
const VERDICT_OCCLUSION_CULLED: u32 = 2u;

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> clusters: array<Cluster>;
@group(0) @binding(2) var<storage, read_write> verdicts: array<u32>;

// One thread per cluster.
@compute @workgroup_size(64, 1, 1)
fn cull(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= params.count) {
        return;
    }

    let c = clusters[i];
    let center = c.center;
    let he = c.half_extents;

    // Frustum takes precedence: reject on the first plane that fully excludes
    // the AABB. The AABB is inside the frustum only when, for every plane,
    // `signed_distance(center) >= -projected_extent(half_extents)`.
    for (var p = 0u; p < 6u; p = p + 1u) {
        let plane = params.planes[p];
        let signed_distance =
            plane.x * center.x + plane.y * center.y + plane.z * center.z + plane.w;
        let projected_extent =
            abs(plane.x) * he.x + abs(plane.y) * he.y + abs(plane.z) * he.z;
        if (signed_distance < -projected_extent) {
            verdicts[i] = VERDICT_FRUSTUM_CULLED;
            return;
        }
    }

    // Inside the frustum: consult the occlusion probe when present. A cluster
    // is occluded when its nearest point is strictly farther than the nearest
    // occluder covering its footprint.
    if (c.has_occlusion == 1u && c.closest_depth > c.occluder_depth) {
        verdicts[i] = VERDICT_OCCLUSION_CULLED;
        return;
    }

    verdicts[i] = VERDICT_VISIBLE;
}
