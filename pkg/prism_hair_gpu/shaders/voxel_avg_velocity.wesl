// Trilinear voxel velocity-field sampling twin.
//
// Device mirror of the `CPU` golden
// `prism_render_architecture::hair::self_collision_voxel::DensityField::sample_avg_velocity`.
// For each world-space query position the kernel returns the local mass-weighted
// average velocity of the splatted self-collision field. The density field holds
// a per-cell accumulated splat mass `cells[i]` and a per-cell mass-weighted
// velocity *sum* `vel_sum[i]` (not yet divided by density). The gather divides
// the trilinearly interpolated velocity sum by the trilinearly interpolated
// density, recovering a true average, and returns zero where there is no nearby
// mass (density `<= EPS`).
//
// With the continuous cell coordinate `g = (p - origin)/cell_size - 0.5`, base
// cell `b = floor(g)` and fraction `f = g - b`, each of the up-to-eight
// neighbour cells contributes `w = wx*wy*wz`, `wk in {1-fk, fk}` to both the
// density sum `rho = sum w*cells[idx]` and the velocity sum
// `V = sum w*vel_sum[idx]`; corners outside the grid contribute `0`. The result
// is `V / rho` when `rho > EPS`, else the zero vector.
//
// # Why one thread per query
//
// Each query's average velocity depends only on its own position and the shared,
// read-only density + velocity field, so this is embarrassingly parallel: one
// thread owns one position, gathers from the shared `cells` and `vel_sum`
// buffers and writes its own `vec4` output row. No thread writes the field, so
// there is no aliasing.
//
// # Correctness model
//
// The reference sanitizes its grid (forcing `cell_size > 0`, every `dims` axis
// `>= 1`, `origin` finite) and its query positions (non-finite components -> 0).
// The host supplies the already-sanitized grid and finite positions, so the
// kernel keeps only the in-bounds corner test verbatim and omits the non-finite
// guards. The gather is a closed-form sum-and-divide (add/sub/mul/divide/floor)
// with no chained recurrence, so the only `CPU` vs `GPU` divergence is legal
// fused-multiply-add contraction and correctly rounded division; parity is
// asserted to `abs < 1e-4` or `rel < 1e-3`. Query positions stay clear of
// integer cell faces so a stray `fma` cannot move `floor(g)` to a different base
// cell. Where density `<= EPS` the result is a bit-exact zero vector on both
// sides.
//
// # Portability
//
// The kernel uses only add/sub/mul/divide/floor/select in the portable
// core-`WGSL` subset — no `exp`, `pow`, `log` or optional device feature — so
// the twin runs unmodified on Metal, Vulkan and DX12.
//
// Provenance: standard trilinear voxel sampling plus a `wgpu` compute dispatch;
// no Unreal Engine source or derived code.

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
// Flat per-cell mass-weighted velocity sum, three `f32` (x, y, z) per cell,
// row-major in the same order as `cells`.
@group(0) @binding(2) var<storage, read> vel_sum: array<f32>;
// Flat query positions, three `f32` (x, y, z) per query.
@group(0) @binding(3) var<storage, read> queries: array<f32>;
// Per query: average velocity in `xyz`, `w` unused (zero).
@group(0) @binding(4) var<storage, read_write> out_vals: array<vec4<f32>>;

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

@compute @workgroup_size(64)
fn evaluate(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= params.query_count) {
        return;
    }
    let base_i = i * 3u;
    let pos = vec3<f32>(queries[base_i], queries[base_i + 1u], queries[base_i + 2u]);

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

    let cell_count = i32(arrayLength(&cells));
    var density: f32 = 0.0;
    var vel: vec3<f32> = vec3<f32>(0.0, 0.0, 0.0);
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
                if (idx >= 0 && idx < cell_count) {
                    let u = u32(idx);
                    density = density + w * cells[u];
                    let vb = u * 3u;
                    vel = vel + w * vec3<f32>(vel_sum[vb], vel_sum[vb + 1u], vel_sum[vb + 2u]);
                }
            }
        }
    }

    var avg: vec3<f32> = vec3<f32>(0.0, 0.0, 0.0);
    if (density > EPS) {
        avg = vel / density;
    }

    out_vals[i] = vec4<f32>(avg.x, avg.y, avg.z, 0.0);
}
