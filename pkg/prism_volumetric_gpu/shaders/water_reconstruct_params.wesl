// `wgpu` compute twin of the dependency-free `CPU` golden per-domain surface
// reconstruction scalar parameter selectors
// (`prism_render_architecture::water::reconstruct::{surface_isovalue,
// narrow_band_width_cells, screen_space_smoothing_iterations}`).
//
// Each fluid domain picks three scalar knobs before its surface is extracted:
// the iso-surface level the mesh routes cross, the narrow-band half-width in
// cells the `SDF` route evaluates within, and the screen-space smoothing
// iteration count that fades with camera distance. All three are small, branchy
// pure functions of per-domain inputs with no cross-domain coupling, so the twin
// batches a whole array of domains into one dispatch -- one invocation per
// domain -- keeping the parameter choice on device next to the reconstruction
// passes that consume it.
//
// The batch is embarrassingly parallel: one invocation owns one domain and reads
// only that domain's own inputs, so there is no write hazard, no barrier and no
// workgroup memory. Each domain performs the identical clamp / `max` /
// linear-fade / round-to-`u32` operations the golden performs, in the same
// order. The kernel uses only the portable core-`WGSL` subset (multiply/add,
// `clamp`, `min`/`max`, a divide, and `f32`-to-`u32` conversion) -- no
// transcendental intrinsic, no optional device feature -- so it runs unmodified
// on Metal, Vulkan and DX12.
//
// Provenance: 孪生自本仓 `prism_render_architecture::water::reconstruct`；无第三方引擎源码或衍生代码。

// Mirror of `prism_render_architecture::water::EPS` (the floating-point epsilon
// the golden uses for the band round-up and the near/far distance guards).
const EPS: f32 = 1.0e-6;

// Uniform parameters for one dispatch. Layout matches `Params` in
// `src/water_reconstruct_params.rs` (`16` bytes, 16-byte aligned).
struct Params {
    // Number of domains (`>= 1`).
    count: u32,
    // Padding to the 16-byte alignment the uniform address space requires.
    pad0: u32,
    pad1: u32,
    pad2: u32,
};

// Device layout of one domain's inputs. Matches `DomainInput` in
// `src/water_reconstruct_params.rs` (`48` bytes: nine `4`-byte scalars plus
// three padding words).
struct DomainInput {
    // Fluid rest density; the isovalue is a fraction of it.
    rest_density: f32,
    // Iso level as a fraction of rest density, clamped to `0..=1`.
    iso_fraction: f32,
    // Particle influence radius (meters); scales the narrow band.
    particle_radius: f32,
    // Reconstruction cell size `dx` (meters).
    dx: f32,
    // Narrow-band factor on the influence radius.
    band_factor: f32,
    // Camera distance (meters); fades the smoothing iteration count.
    camera_distance: f32,
    // Maximum screen-space smoothing iterations near the camera.
    max_iterations: u32,
    // Distance at or below which smoothing runs at `max_iterations`.
    near_distance: f32,
    // Distance at or beyond which smoothing runs a single pass.
    far_distance: f32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
};

// Device layout of one domain's results. Matches `DomainResult` in
// `src/water_reconstruct_params.rs` (`16` bytes).
struct DomainResult {
    // Iso-surface level per domain.
    isovalue: f32,
    // Narrow-band half-width in cells per domain.
    narrow_band: u32,
    // Screen-space smoothing iteration count per domain.
    smoothing_iterations: u32,
    pad0: u32,
};

@group(0) @binding(0) var<uniform> params: Params;
// Row-major per-domain reconstruction inputs.
@group(0) @binding(1) var<storage, read> inputs: array<DomainInput>;
// Row-major per-domain reconstruction scalar parameters.
@group(0) @binding(2) var<storage, read_write> results: array<DomainResult>;

// Mirror of the golden `surface_isovalue`: `rest_density * clamp(iso_fraction)`.
// The negative rest density folds to zero; the fraction is clamped to `0..=1`.
// Keeps the golden's `max`-then-multiply order.
fn surface_isovalue(rest_density: f32, iso_fraction: f32) -> f32 {
    return max(rest_density, 0.0) * clamp(iso_fraction, 0.0, 1.0);
}

// Mirror of the golden `narrow_band_width_cells`: scale the influence radius by
// the band factor and the reciprocal cell size, round up to a whole cell count
// (floor of one) so the band always straddles the interface. The `+1 - EPS`
// then truncate reproduces the golden's `as u32` round-up for a non-negative
// value, including passing an exact integer through unchanged.
fn narrow_band_width_cells(
    particle_radius: f32,
    dx: f32,
    band_factor: f32,
) -> u32 {
    let cell = max(dx, EPS);
    let radius = max(particle_radius, 0.0);
    let factor = max(band_factor, 0.0);
    let cells = factor * radius / cell;
    let rounded = u32(cells + 1.0 - EPS);
    return max(rounded, 1u);
}

// Mirror of the golden `screen_space_smoothing_iterations`: hold `max_iters`
// (floor of one) at or inside `near`, drop to a single pass at or beyond `far`
// (which is itself floored a hair above `near`), and linearly fade the rounded
// remainder in between. Keeps the golden's clamp / branch order so the integer
// result is exactly equal away from the rounding seam.
fn screen_space_smoothing_iterations(
    camera_distance: f32,
    max_iterations: u32,
    near_distance: f32,
    far_distance: f32,
) -> u32 {
    let max_iters = max(max_iterations, 1u);
    let near = max(near_distance, 0.0);
    let far = max(far_distance, near + EPS);
    let distance = max(camera_distance, 0.0);
    if distance <= near {
        return max_iters;
    }
    if distance >= far {
        return 1u;
    }
    let t = (distance - near) / (far - near);
    let span = f32(max_iters - 1u);
    let remaining = span * (1.0 - t);
    return 1u + u32(remaining + 0.5);
}

@compute @workgroup_size(256)
fn params_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if idx >= params.count {
        return;
    }
    let d = inputs[idx];
    var r: DomainResult;
    r.isovalue = surface_isovalue(d.rest_density, d.iso_fraction);
    r.narrow_band = narrow_band_width_cells(d.particle_radius, d.dx, d.band_factor);
    r.smoothing_iterations = screen_space_smoothing_iterations(
        d.camera_distance,
        d.max_iterations,
        d.near_distance,
        d.far_distance,
    );
    r.pad0 = 0u;
    results[idx] = r;
}
