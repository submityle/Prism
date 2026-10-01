// Trilinear voxel-field sampling twin.
//
// Device mirror of two `CPU` goldens on Prism's hair self-collision density
// field,
// `prism_render_architecture::hair::self_collision_voxel::DensityField::sample_density`
// and `self_collision_voxel::sample_gradient`. For each world-space query
// position the kernel returns the trilinearly interpolated scalar density
// `rho(p)` and the central-difference density gradient `grad(rho)(p)` of the
// splatted voxel field.
//
// Density is the standard `2x2x2` trilinear gather: with the continuous cell
// coordinate `g = (p - origin)/cell_size - 0.5`, base cell `b = floor(g)` and
// fraction `f = g - b`, each of the up-to-eight neighbour cells contributes
// `w * cells[idx]` with `w = wx*wy*wz`, `wk in {1-fk, fk}`, and corners that
// fall outside the grid contribute `0` (density vanishes outside the field).
// The gradient is a boundary-clamped central difference: per axis
// `(rho(p+h) - rho(p-h)) / (actual separation)` with `h = cell_size`, the two
// stencil points clamped into the grid box, and a `0` derivative when the clamp
// collapses the separation below `EPS`.
//
// # Why one thread per query
//
// Each query's `(density, gradient)` depends only on its own position and the
// shared, read-only voxel field, so this is embarrassingly parallel: one thread
// owns one position, gathers from the shared `cells` buffer and writes its own
// `vec4` output row. No thread writes the field, so there is no aliasing.
//
// # Correctness model
//
// The reference sanitizes its grid (forcing `cell_size > 0`, every `dims` axis
// `>= 1`, `origin` finite) and its query positions (non-finite components -> 0).
// The host supplies the already-sanitized grid and finite positions, so the
// kernel keeps only the in-bounds corner test and the boundary clamp verbatim
// and omits the non-finite guards. Both goldens are closed-form gather/stencil
// evaluations (add/sub/mul/divide/floor/clamp) with no chained recurrence, so
// the only `CPU` vs `GPU` divergence is legal fused-multiply-add contraction and
// correctly rounded division; parity is asserted to `abs < 1e-4` or
// `rel < 1e-3`. Query positions stay clear of integer cell faces so a stray
// `fma` cannot move `floor(g)` to a different base cell, and clear of the grid
// box faces so the gradient clamp picks the identical stencil on both sides.
//
// # Portability
//
// The kernel uses only add/sub/mul/divide/floor/clamp/select in the portable
// core-`WGSL` subset — no `exp`, `pow`, `log` or optional device feature — so
// the twin runs unmodified on Metal, Vulkan and DX12.
//
// Provenance: standard trilinear voxel sampling plus a central-difference
// stencil and a `wgpu` compute dispatch; no Unreal Engine source or derived
// code.

// Sanitized voxel grid (origin, cell size, dims) plus the query count, packed
// into two `16`-byte slots. Mirrors (the sanitized subset of) `VoxelGrid`.
struct Params {
    origin: vec3<f32>,
    cell_size: f32,
    dims: vec3<u32>,
    query_count: u32,
}

const EPS: f32 = 1.0e-6;

@group(0) @binding(0) var<uniform> params: Params;
// Row-major per-cell accumulated density (`x + dims.x*(y + dims.y*z)`).
@group(0) @binding(1) var<storage, read> cells: array<f32>;
// Flat query positions, three `f32` (x, y, z) per query.
@group(0) @binding(2) var<storage, read> queries: array<f32>;
// Per query: density in `x`, gradient in `yzw`.
@group(0) @binding(3) var<storage, read_write> out_vals: array<vec4<f32>>;

// Flat row-major cell index, or `-1` when the integer cell is out of bounds.
fn index_of(cell: vec3<i32>) -> i32 {
    if (cell.x < 0 || cell.y < 0 || cell.z < 0) {
        return -1;
    }
    if (cell.x >= i32(params.dims.x) || cell.y >= i32(params.dims.y) || cell.z >= i32(params.dims.z)) {
        return -1;
    }
    let x = u32(cell.x);
    let y = u32(cell.y);
    let z = u32(cell.z);
    return i32(x + params.dims.x * (y + params.dims.y * z));
}

// Trilinearly samples the density field at a world position. Corners outside the
// grid contribute `0`. Corner order is z-outer, y, x-inner — matching the
// reference accumulation order.
fn sample_density_at(pos: vec3<f32>) -> f32 {
    let inv = 1.0 / params.cell_size;
    let gx = (pos.x - params.origin.x) * inv - 0.5;
    let gy = (pos.y - params.origin.y) * inv - 0.5;
    let gz = (pos.z - params.origin.z) * inv - 0.5;
    let bx = floor(gx);
    let by = floor(gy);
    let bz = floor(gz);
    let fx = gx - bx;
    let fy = gy - by;
    let fz = gz - bz;
    let count = i32(arrayLength(&cells));
    var sum: f32 = 0.0;
    for (var cz: u32 = 0u; cz < 2u; cz = cz + 1u) {
        let wz = select(fz, 1.0 - fz, cz == 0u);
        for (var cy: u32 = 0u; cy < 2u; cy = cy + 1u) {
            let wy = select(fy, 1.0 - fy, cy == 0u);
            for (var cx: u32 = 0u; cx < 2u; cx = cx + 1u) {
                let wx = select(fx, 1.0 - fx, cx == 0u);
                let w = wx * wy * wz;
                let cell = vec3<i32>(
                    i32(bx) + i32(cx),
                    i32(by) + i32(cy),
                    i32(bz) + i32(cz),
                );
                let idx = index_of(cell);
                if (idx >= 0 && idx < count) {
                    sum = sum + w * cells[u32(idx)];
                }
            }
        }
    }
    return sum;
}

// Clamps a world position into the grid's axis-aligned box.
fn clamp_to_bounds(pos: vec3<f32>) -> vec3<f32> {
    let hi = params.origin + vec3<f32>(params.dims) * params.cell_size;
    return vec3<f32>(
        clamp(pos.x, params.origin.x, hi.x),
        clamp(pos.y, params.origin.y, hi.y),
        clamp(pos.z, params.origin.z, hi.z),
    );
}

@compute @workgroup_size(64)
fn evaluate(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= params.query_count) {
        return;
    }
    let base_i = i * 3u;
    let pos = vec3<f32>(queries[base_i], queries[base_i + 1u], queries[base_i + 2u]);

    let density = sample_density_at(pos);

    let h = params.cell_size;
    let base = clamp_to_bounds(pos);

    let hix = clamp_to_bounds(vec3<f32>(base.x + h, base.y, base.z));
    let lox = clamp_to_bounds(vec3<f32>(base.x - h, base.y, base.z));
    let denomx = hix.x - lox.x;
    var gx: f32 = 0.0;
    if (denomx > EPS) {
        gx = (sample_density_at(hix) - sample_density_at(lox)) / denomx;
    }

    let hiy = clamp_to_bounds(vec3<f32>(base.x, base.y + h, base.z));
    let loy = clamp_to_bounds(vec3<f32>(base.x, base.y - h, base.z));
    let denomy = hiy.y - loy.y;
    var gy: f32 = 0.0;
    if (denomy > EPS) {
        gy = (sample_density_at(hiy) - sample_density_at(loy)) / denomy;
    }

    let hiz = clamp_to_bounds(vec3<f32>(base.x, base.y, base.z + h));
    let loz = clamp_to_bounds(vec3<f32>(base.x, base.y, base.z - h));
    let denomz = hiz.z - loz.z;
    var gz: f32 = 0.0;
    if (denomz > EPS) {
        gz = (sample_density_at(hiz) - sample_density_at(loz)) / denomz;
    }

    out_vals[i] = vec4<f32>(density, gx, gy, gz);
}
