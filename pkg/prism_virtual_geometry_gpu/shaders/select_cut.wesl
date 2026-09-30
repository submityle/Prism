// Cut-selection twin: chooses the screen-space-error *cut* through a cluster LOD
// hierarchy on-device, mirroring the CPU golden `ClusterHierarchy::select_cut`
// (in `prism_render_architecture::virtual_geometry`).
//
// Rendering one virtual-geometry frame means choosing one cluster per visible
// surface region: descend the hierarchy while a node is too coarse for the pixel
// budget, stop and draw as soon as a node is fine enough (or is a leaf). The CPU
// golden performs that as a top-down stack walk. Because the walk's decision at
// each node is a *pure local* predicate - is the node frustum-visible, and is it
// a leaf or within the pixel budget - this twin evaluates those predicates in
// parallel (one thread per node) and then propagates reachability from the roots
// down the parent links, so the drawn set it produces is identical to the set
// the reference walk emits.
//
// Two entry points share one bind group:
//   * `classify` runs once, one thread per node, and computes the per-node
//     `descend` and `draw_local` predicates plus the initial `reached` bit (set
//     only on roots). This is where every float comparison lives, so it is the
//     only stage that can diverge from the reference under fused-multiply-add
//     contraction; the term order matches the reference exactly.
//   * `relax` runs repeatedly, one thread per node, and sets a node's `reached`
//     bit once its parent is both reached and descended-into. Reachability only
//     ever flips 0 -> 1, so the host re-dispatches to a fixed point; the final
//     bit pattern is independent of intra-dispatch ordering.
//
// The frustum term order (`nx*px + ny*py + nz*pz + d` for the signed distance,
// `|nx|*hx + |ny|*hy + |nz|*hz` for the projected extent) and the budget test
// (`projected*projected <= budget*budget*distance_sq`, the reference's
// square-root-free squared form) are copied verbatim from the golden so, away
// from the razor-thin budget/plane boundary, the discrete draw set is identical
// regardless of FMA contraction. It uses only the portable core-WGSL subset - no
// optional feature - so this twin runs unmodified on Metal, Vulkan and DX12.
//
// Provenance: standard Nanite-style hierarchical screen-space-error cut
// selection plus wgpu compute dispatch; no Unreal Engine source or derived code.

struct Params {
    // Six inward-facing frustum planes, each packed as (normal.xyz, distance).
    // The half-space `dot(normal, p) + distance >= 0` is the interior.
    planes: array<vec4<f32>, 6>,
    // Camera world position used for error projection.
    view_origin: vec3<f32>,
    // Error projection focal length in pixels.
    focal_length_pixels: f32,
    // Per-frame pixel budget, already clamped to `>= 0` on the host.
    budget: f32,
    // Number of valid nodes in `nodes` and entries in `state`.
    node_count: u32,
    // Padding to a 16-byte uniform tail.
    pad0: u32,
    pad1: u32,
}

// One hierarchy node. `48`-byte stride matching the `bytemuck` upload struct;
// `center`/`half_extents` are `vec3<f32>` (16-byte aligned) with the trailing
// scalar packed into each vec3's padding word.
struct Node {
    // AABB centre in world space.
    center: vec3<f32>,
    // Object-space geometric error bound, in world units.
    self_error: f32,
    // AABB half-extents (support radius per axis).
    half_extents: vec3<f32>,
    // `1` when the node is a leaf (no children), `0` for an interior node.
    is_leaf: u32,
    // Index of this node's parent, or `SENTINEL` when the node has no parent.
    parent: u32,
    // `1` when the node is a hierarchy root (walk starts here), else `0`.
    is_root: u32,
    // Padding to the 48-byte stride.
    pad0: u32,
    pad1: u32,
}

// Sentinel parent index for nodes that have no parent.
const SENTINEL: u32 = 0xffffffffu;

// `state` bit layout, one `u32` per node.
const BIT_DESCEND: u32 = 1u;    // node is visible, interior and too coarse.
const BIT_DRAW_LOCAL: u32 = 2u; // node is visible and (leaf or within budget).
const BIT_REACHED: u32 = 4u;    // node is reachable through descended ancestors.

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> nodes: array<Node>;
@group(0) @binding(2) var<storage, read_write> state: array<u32>;
@group(0) @binding(3) var<storage, read_write> changed: atomic<u32>;

// Whether an AABB is at least partially inside the frustum, using the
// reference's exact projected-radius test with matching term order.
fn frustum_visible(center: vec3<f32>, he: vec3<f32>) -> bool {
    for (var p = 0u; p < 6u; p = p + 1u) {
        let plane = params.planes[p];
        let signed_distance =
            plane.x * center.x + plane.y * center.y + plane.z * center.z + plane.w;
        let projected_extent =
            abs(plane.x) * he.x + abs(plane.y) * he.y + abs(plane.z) * he.z;
        if (signed_distance < -projected_extent) {
            return false;
        }
    }
    return true;
}

// One thread per node: evaluate the local predicates and seed reachability.
@compute @workgroup_size(64, 1, 1)
fn classify(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= params.node_count) {
        return;
    }
    let node = nodes[i];
    var flags = 0u;

    if (frustum_visible(node.center, node.half_extents)) {
        // Squared-form budget test, matching the golden term order:
        //   projected = self_error.max(0) * focal
        //   projected^2 <= budget^2 * distance_sq
        let dx = node.center.x - params.view_origin.x;
        let dy = node.center.y - params.view_origin.y;
        let dz = node.center.z - params.view_origin.z;
        let distance_sq = dx * dx + dy * dy + dz * dz;
        let projected = max(node.self_error, 0.0) * params.focal_length_pixels;
        let fits = projected * projected <= params.budget * params.budget * distance_sq;

        if (node.is_leaf == 1u || fits) {
            flags = flags | BIT_DRAW_LOCAL;
        } else {
            flags = flags | BIT_DESCEND;
        }
    }
    if (node.is_root == 1u) {
        flags = flags | BIT_REACHED;
    }
    state[i] = flags;
}

// One thread per node: set `reached` once the parent is reached and descended.
@compute @workgroup_size(64, 1, 1)
fn relax(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= params.node_count) {
        return;
    }
    let flags = state[i];
    if ((flags & BIT_REACHED) != 0u) {
        return;
    }
    let parent = nodes[i].parent;
    if (parent == SENTINEL) {
        return;
    }
    let pflags = state[parent];
    if ((pflags & BIT_REACHED) != 0u && (pflags & BIT_DESCEND) != 0u) {
        state[i] = flags | BIT_REACHED;
        atomicStore(&changed, 1u);
    }
}
