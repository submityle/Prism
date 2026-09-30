// Triangle-gradient setup twin: computes per-triangle screen-space edge and
// depth gradients on-device, mirroring the CPU golden `TriangleGradients::new`.
//
// The shipping vis-buffer software rasterizer does not recompute the three
// edge functions at every pixel; it sets up per-triangle *gradients* once
// (`w_x`/`w_y` per-column/row edge steps and the screen-space-linear depth
// plane `z_x`/`z_y`) then walks them incrementally across each row. The CPU
// golden `TriangleGradients::new` (in `prism_render_architecture::
// virtual_geometry`) pins that setup; this kernel is the on-device twin that
// runs one thread per triangle and writes the same gradient fields the
// reference derives, so the twin's incremental walk starts from a device-side
// setup validated bit-for-bit against the reference.
//
// Winding / culling convention matches the reference exactly: the signed
// double area `A = edge(v0, v1, v2)` must be strictly positive (front-facing
// under the y-down, positive-area convention). Degenerate or back-facing
// triangles (`A <= 0`) are the cases the reference culls before setup, so the
// twin marks them invalid (`valid = 0u`) and zeroes the gradient payload.
//
// For triangles on integer / dyadic pixel coordinates the signed edge function
// is evaluated on exactly representable operands, so its value is identical on
// CPU and GPU regardless of fused-multiply-add contraction. When the double
// area is an exact power of two, `inv_area = 1 / A` is exact and each
// `vertices_z[i] = depth_i * inv_area` product and the `z_x`/`z_y` dot sums are
// bit-exact too, so the parity test asserts field-for-field equality rather
// than a tolerance. It uses only the portable core-WGSL subset - no optional
// feature - so this twin runs unmodified on Metal, Vulkan and DX12.
//
// Provenance: standard signed-edge affine gradient setup for software
// rasterization; no Unreal Engine source or derived code.

struct Params {
    // Number of triangles in `triangles` / outputs in `gradients`.
    triangle_count: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

// One triangle's three screen vertices. `36`-byte stride matching the
// `bytemuck` upload struct: (x, y, depth) per vertex, y-down pixel space,
// reversed-Z depth in `[0, 1]`.
struct Triangle {
    x0: f32, y0: f32, d0: f32,
    x1: f32, y1: f32, d1: f32,
    x2: f32, y2: f32, d2: f32,
}

// One triangle's gradient setup result. `56`-byte stride.
struct Gradients {
    // `1u` front-facing (setup valid), `0u` degenerate/back-facing.
    valid: u32,
    // Signed double area `edge(v0, v1, v2)`, strictly positive when valid.
    double_area: f32,
    // Per-column (`+x`) edge-function increments.
    wx0: f32, wx1: f32, wx2: f32,
    // Per-row (`+y`) edge-function increments.
    wy0: f32, wy1: f32, wy2: f32,
    // Depth weights `(d0, d1, d2) / double_area` (shipping `vertices_z`).
    vz0: f32, vz1: f32, vz2: f32,
    // Per-column / per-row increment of screen-space-linear depth.
    z_x: f32,
    z_y: f32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> triangles: array<Triangle>;
@group(0) @binding(2) var<storage, read_write> gradients: array<Gradients>;

// Signed edge function, matching the reference `edge(a, b, p)`:
// `(b.x - a.x) * (p.y - a.y) - (b.y - a.y) * (p.x - a.x)`.
fn edge(a: vec2<f32>, b: vec2<f32>, p: vec2<f32>) -> f32 {
    return (b.x - a.x) * (p.y - a.y) - (b.y - a.y) * (p.x - a.x);
}

fn zeroed() -> Gradients {
    return Gradients(
        0u, 0.0,
        0.0, 0.0, 0.0,
        0.0, 0.0, 0.0,
        0.0, 0.0, 0.0,
        0.0, 0.0,
    );
}

@compute @workgroup_size(64)
fn setup(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= params.triangle_count) {
        return;
    }

    let t = triangles[i];
    let p0 = vec2<f32>(t.x0, t.y0);
    let p1 = vec2<f32>(t.x1, t.y1);
    let p2 = vec2<f32>(t.x2, t.y2);

    let double_area = edge(p0, p1, p2);
    if (double_area <= 0.0) {
        // Degenerate or back-facing: the reference returns `None` here.
        gradients[i] = zeroed();
        return;
    }

    let inv_area = 1.0 / double_area;

    // Per-column (+x) edge increments: d(w)/dx = (v1.y - v2.y, v2.y - v0.y,
    // v0.y - v1.y).
    let wx0 = p1.y - p2.y;
    let wx1 = p2.y - p0.y;
    let wx2 = p0.y - p1.y;
    // Per-row (+y) edge increments: d(w)/dy = (v2.x - v1.x, v0.x - v2.x,
    // v1.x - v0.x).
    let wy0 = p2.x - p1.x;
    let wy1 = p0.x - p2.x;
    let wy2 = p1.x - p0.x;

    let vz0 = t.d0 * inv_area;
    let vz1 = t.d1 * inv_area;
    let vz2 = t.d2 * inv_area;

    // Depth plane gradients: dot(vertices_z, w_x) and dot(vertices_z, w_y),
    // summed left-to-right to match the reference association.
    let z_x = vz0 * wx0 + vz1 * wx1 + vz2 * wx2;
    let z_y = vz0 * wy0 + vz1 * wy1 + vz2 * wy2;

    gradients[i] = Gradients(
        1u, double_area,
        wx0, wx1, wx2,
        wy0, wy1, wy2,
        vz0, vz1, vz2,
        z_x, z_y,
    );
}
