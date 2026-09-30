// Prism virtual-geometry paging priority twin: squared projected screen
// coverage of a cluster's bounding sphere at the camera, mirroring the CPU
// golden `coverage_priority`
// (in `prism_render_architecture::virtual_geometry::hierarchy`):
// `extent^2 / max(dx^2 + dy^2 + dz^2, f32::EPSILON)`
// with `extent = max(radius, 0.0) * focal_length_pixels` and
// `(dx, dy, dz) = center - view_origin`.
//
// The page streamer prioritises clusters by how much screen area they cover,
// so the coarsest visible cut is resident first. The true coverage is
// `radius * focal / distance`; the squared form keeps the priority
// square-root-free and preserves ordering because every term is non-negative.
// A camera resting on the sphere centre collapses `distance_sq` to the
// `f32::EPSILON` floor, yielding the maximum finite priority so an enveloping
// cluster is never starved (matching the golden's `.max(f32::EPSILON)`).
//
// Numerical model: `max` is exact and every multiply is correctly rounded, but
// a GPU divide by a non-power-of-two `distance_sq` is only guaranteed correctly
// rounded to within one ULP (Metal implements it as a refined reciprocal), so
// the twin matches the golden within one ULP in general and bit-for-bit
// whenever the clamped `distance_sq` is an exact power of two (the divide
// degenerates to exponent scaling). The sum of squares `dx^2 + dy^2 + dz^2`
// may be contracted into an FMA on-device, so the parity domain restricts the
// difference operands to values whose squares and partial sums are exactly
// representable f32 - then FMA fusion and the sequential CPU adds coincide to
// the bit and only the final divide carries the one-ULP slack. The
// `f32::EPSILON = 2^-23` floor is written as the exact literal `0x1p-23` so the
// CPU and GPU clamps coincide to the bit.
// The kernel uses only the portable core-WGSL subset - no optional feature - so
// it runs unmodified on Metal, Vulkan and DX12.
//
// Provenance: standard perspective screen-coverage streaming priority for a
// paged cluster hierarchy; no Unreal Engine source or derived code.

struct Params {
    // Number of queries in `inputs` / priorities in `priorities`.
    count: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

// One paging-priority query. 32-byte stride, matching `GpuCoverageInput` on the
// host: bounding-sphere centre, radius, view origin and focal length. Scalar
// fields avoid vec3 storage-alignment padding.
struct CoverageQuery {
    center_x: f32,
    center_y: f32,
    center_z: f32,
    radius: f32,
    origin_x: f32,
    origin_y: f32,
    origin_z: f32,
    focal: f32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> inputs: array<CoverageQuery>;
@group(0) @binding(2) var<storage, read_write> priorities: array<f32>;

@compute @workgroup_size(64)
fn evaluate(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= params.count) {
        return;
    }
    let q = inputs[i];
    let dx = q.center_x - q.origin_x;
    let dy = q.center_y - q.origin_y;
    let dz = q.center_z - q.origin_z;
    // max(dx^2 + dy^2 + dz^2, 2^-23) - mirrors coverage_priority exactly.
    let distance_sq = max(dx * dx + dy * dy + dz * dz, 0x1p-23);
    let extent = max(q.radius, 0.0) * q.focal;
    priorities[i] = extent * extent / distance_sq;
}
