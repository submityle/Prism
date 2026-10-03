// Screen-space fluid surface-reconstruction compute kernel
// (entry point `water_surface_reconstruct`).
//
// Particle fluids (`FLIP`/`APIC`, `PBF`) have no mesh, so the shipping route to
// a shaded surface is screen space: splat the particles into an eye-depth
// buffer, smooth it, then recover a per-pixel view-space normal from the depth
// field. This kernel is the normal-reconstruction half, the one `van der Laan`
// ("Screen Space Fluid Rendering with Curvature Flow") and every real-time
// fluid renderer rasterizes. It consumes the smoothed depth buffer plus a
// per-pixel accumulated thickness and writes a view-space normal texture whose
// alpha carries the thickness through for the downstream absorption/opacity
// shade.
//
// Per pixel it unprojects the center and its four axis neighbours to view space
// through a pinhole model, forms a one-sided derivative on each axis choosing
// the neighbour with the smaller depth step (the `van der Laan` silhouette
// guard that stops normals bleeding across the fluid's edge), and takes the
// cross product. Background pixels (depth <= 0) and pixels with no valid
// neighbour on an axis degrade gracefully: a flat camera-facing normal, never a
// `NaN`. The normal is flipped to face the camera (negative view `z`).
//
// It mirrors the `CPU` golden `gpu::surface_reconstruct_kernel`'s
// `dispatch_surface_reconstruct` byte-for-byte: the only float intrinsic is
// `sqrt` (the normalize), everything else is `+ - * /`, comparisons, `abs`,
// `cross` and `dot`, all permitted by the workspace determinism policy.
// `EPS_LEN_SQ` matches the crate's `water::EPS_LEN_SQ` (1e-12) exactly so the
// degenerate-normal fallback fires on the same inputs as the twin.
//
// ABI (matches `WaterKernel::SurfaceReconstruct.descriptor()`: 2 storage
// buffers, 1 uniform, 1 storage texture, 0 sampled, 8x8 tile, Screen domain):
//   @binding(0) depth_in     : array<f32>, smoothed per-pixel eye depth (> 0)
//   @binding(1) thickness_in  : array<f32>, per-pixel accumulated thickness
//   @binding(2) params        : ReconstructParams uniform
//   @binding(3) normal_out     : texture_storage_2d<rgba32float, write>,
//                               (nx, ny, nz, thickness) in view space (written)

const EPS_LEN_SQ: f32 = 1e-12;

struct ReconstructParams {
    width: u32,
    height: u32,
    tan_half_fov_x: f32,
    tan_half_fov_y: f32,
}

@group(0) @binding(0) var<storage, read> depth_in: array<f32>;
@group(0) @binding(1) var<storage, read> thickness_in: array<f32>;
@group(0) @binding(2) var<uniform> params: ReconstructParams;
@group(0) @binding(3) var normal_out: texture_storage_2d<rgba32float, write>;

// Unproject a pixel center at eye depth `z` to view space through the pinhole
// model: NDC in [-1, 1] (y up), scaled by the per-axis half-fov tangent.
fn view_pos(px: f32, py: f32, z: f32) -> vec3<f32> {
    let wf = f32(params.width);
    let hf = f32(params.height);
    let ndc_x = 2.0 * ((px + 0.5) / wf) - 1.0;
    let ndc_y = 1.0 - 2.0 * ((py + 0.5) / hf);
    return vec3<f32>(ndc_x * z * params.tan_half_fov_x, ndc_y * z * params.tan_half_fov_y, z);
}

@compute @workgroup_size(8, 8, 1)
fn water_surface_reconstruct(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= params.width || gid.y >= params.height) {
        return;
    }
    let coord = vec2<i32>(i32(gid.x), i32(gid.y));
    let idx = gid.y * params.width + gid.x;
    let zc = depth_in[idx];
    let thick = thickness_in[idx];

    // Background: no fluid here, emit a null record (zero normal, zero
    // thickness) rather than reconstructing from a bogus depth.
    if (zc <= 0.0) {
        textureStore(normal_out, coord, vec4<f32>(0.0, 0.0, 0.0, 0.0));
        return;
    }

    let gx = i32(gid.x);
    let gy = i32(gid.y);
    let w = i32(params.width);
    let h = i32(params.height);
    let pc = view_pos(f32(gx), f32(gy), zc);

    // x-axis derivative with the van der Laan silhouette guard: prefer the
    // neighbour (left or right) whose depth step is smaller.
    var dxv = vec3<f32>(0.0, 0.0, 0.0);
    let have_r = gx + 1 < w;
    let have_l = gx > 0;
    var zr = 0.0;
    var zl = 0.0;
    if (have_r) { zr = depth_in[idx + 1u]; }
    if (have_l) { zl = depth_in[idx - 1u]; }
    let vr = have_r && zr > 0.0;
    let vl = have_l && zl > 0.0;
    if (vr && vl) {
        let dpr = view_pos(f32(gx + 1), f32(gy), zr) - pc;
        let dpl = pc - view_pos(f32(gx - 1), f32(gy), zl);
        if (abs(zr - zc) <= abs(zc - zl)) {
            dxv = dpr;
        } else {
            dxv = dpl;
        }
    } else if (vr) {
        dxv = view_pos(f32(gx + 1), f32(gy), zr) - pc;
    } else if (vl) {
        dxv = pc - view_pos(f32(gx - 1), f32(gy), zl);
    }

    // y-axis derivative, same guard over the up/down neighbours.
    var dyv = vec3<f32>(0.0, 0.0, 0.0);
    let have_u = gy > 0;
    let have_d = gy + 1 < h;
    var zu = 0.0;
    var zd = 0.0;
    if (have_u) { zu = depth_in[idx - params.width]; }
    if (have_d) { zd = depth_in[idx + params.width]; }
    let vu = have_u && zu > 0.0;
    let vd = have_d && zd > 0.0;
    if (vu && vd) {
        let dpu = view_pos(f32(gx), f32(gy - 1), zu) - pc;
        let dpd = pc - view_pos(f32(gx), f32(gy + 1), zd);
        if (abs(zu - zc) <= abs(zc - zd)) {
            dyv = dpu;
        } else {
            dyv = dpd;
        }
    } else if (vu) {
        dyv = view_pos(f32(gx), f32(gy - 1), zu) - pc;
    } else if (vd) {
        dyv = pc - view_pos(f32(gx), f32(gy + 1), zd);
    }

    let n0 = cross(dxv, dyv);
    let len_sq = dot(n0, n0);
    var n = vec3<f32>(0.0, 0.0, -1.0);
    if (len_sq > EPS_LEN_SQ) {
        n = n0 * (1.0 / sqrt(len_sq));
    }
    // Face the camera: the eye looks down +z, so a visible fluid normal has a
    // negative view-space z.
    if (n.z > 0.0) {
        n = -n;
    }

    textureStore(normal_out, coord, vec4<f32>(n.x, n.y, n.z, thick));
}
