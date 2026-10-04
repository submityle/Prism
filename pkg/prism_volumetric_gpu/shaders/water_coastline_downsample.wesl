// `wgpu` compute twin of the dependency-free `CPU` golden coastline
// heightfield downsample
// (`prism_render_architecture::water::coastline::downsample_heightfield`).
//
// An authoring layer coarsens an over-resolved terrain tile to the water
// solver-grid budget before flooding, exactly the way a large `UE5` Landscape
// is decimated to the water grid it can afford. Each output cell is the
// unbiased mean of the up-to `stride x stride` source samples it covers;
// partial edge blocks average only the samples that exist.
//
// The reduction is embarrassingly parallel: one invocation owns one output
// cell, with no barrier and no workgroup memory. Each thread walks its source
// block in the *same* row-major order the `CPU` golden uses
// (`for row { for col }`), so the floating-point summation order matches and
// the result is bit-identical up to the last-place slack of a `GPU` fused
// multiply-add. The kernel uses only integer arithmetic and add/divide in the
// portable core-`WGSL` subset, so it runs unmodified on Metal, Vulkan and DX12.

// Uniform parameters for one dispatch. Layout matches `Params` in
// `src/water_coastline_downsample.rs` (`32` bytes, 16-byte aligned).
struct Params {
    // Source grid width (cells along x), `> 0`.
    nx: u32,
    // Source grid height (cells along z), `> 0`.
    nz: u32,
    // Integer block edge; each output cell averages a `stride x stride` block.
    stride: u32,
    // Output grid width, `ceil(nx / stride)`.
    out_w: u32,
    // Output grid height, `ceil(nz / stride)`.
    out_h: u32,
    // Padding to a 16-byte-aligned 32-byte struct.
    pad0: u32,
    pad1: u32,
    pad2: u32,
};

@group(0) @binding(0) var<uniform> params: Params;
// Row-major `nx * nz` source terrain heights (meters).
@group(0) @binding(1) var<storage, read> heights: array<f32>;
// Row-major `out_w * out_h` coarsened terrain heights (meters).
@group(0) @binding(2) var<storage, read_write> out_heights: array<f32>;

@compute @workgroup_size(256)
fn downsample_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    let out_cells = params.out_w * params.out_h;
    if idx >= out_cells {
        return;
    }

    let out_row = idx / params.out_w;
    let out_col = idx % params.out_w;

    let row0 = out_row * params.stride;
    let col0 = out_col * params.stride;
    var row1 = row0 + params.stride;
    if row1 > params.nz {
        row1 = params.nz;
    }
    var col1 = col0 + params.stride;
    if col1 > params.nx {
        col1 = params.nx;
    }

    // Walk the source block in the golden's row-major order so the summation
    // order (and therefore the rounding) matches cell for cell.
    var sum = 0.0;
    var count = 0u;
    var row = row0;
    while row < row1 {
        let base = row * params.nx;
        var col = col0;
        while col < col1 {
            sum = sum + heights[base + col];
            count = count + 1u;
            col = col + 1u;
        }
        row = row + 1u;
    }

    // Every output cell covers at least its (row0, col0) sample, so the count is
    // always positive; guard defensively regardless, matching the golden.
    var value = 0.0;
    if count > 0u {
        value = sum / f32(count);
    }
    out_heights[idx] = value;
}
