//! `wgpu` compute twin of the `GI`-probe / irradiance golden
//! ([`gi_probe`](prism_render_architecture::particle::gi_probe), particle design
//! §8.3, aligned to the shading contract of §17).
//!
//! The `CPU` golden
//! [`gi_probe`](prism_render_architecture::particle::gi_probe) owns the
//! *receive-light* data interface: it evaluates the real spherical-harmonic
//! (`SH`) basis, the Lambert cosine-lobe convolution weights, the octahedral
//! direction map, the six-face ambient-cube irradiance sampler, and the colored
//! `SH` radiance/irradiance reconstruction a particle bathes in. Every one of
//! those answers is a Cartesian polynomial of a unit direction (plus `abs`,
//! `select`-style sign folds and one guarded `sqrt`), so the whole interface
//! ports to a portable core-`WGSL` kernel with no transcendental call.
//!
//! [`GpuGiProbe`] is the on-device twin: one thread per [`GiProbeQuery`]
//! reproduces the matching golden answer, so a passing real-device parity test
//! is direct evidence the ported kernel evaluates the same polynomials, folds
//! the same octahedral seam, and blends the same ambient faces the reference
//! does, not merely that the shader compiles.
//!
//! # What is twinned
//!
//! The twin reproduces, per query, exactly the golden functions:
//! [`sh_basis_l1`](prism_render_architecture::particle::gi_probe::sh_basis_l1)
//! and
//! [`sh_basis_l2`](prism_render_architecture::particle::gi_probe::sh_basis_l2)
//! (the four / nine real basis values),
//! [`cosine_lobe_weight`](prism_render_architecture::particle::gi_probe::cosine_lobe_weight)
//! (the Lambert transfer weight for a band),
//! [`octa_encode`](prism_render_architecture::particle::gi_probe::octa_encode),
//! [`octa_decode`](prism_render_architecture::particle::gi_probe::octa_decode),
//! [`octa_to_unorm`](prism_render_architecture::particle::gi_probe::octa_to_unorm)
//! and
//! [`octa_from_unorm`](prism_render_architecture::particle::gi_probe::octa_from_unorm)
//! (the octahedral direction map and its texture-space remaps),
//! [`AmbientCube::sample`](prism_render_architecture::particle::gi_probe::AmbientCube::sample)
//! (the squared-component six-face blend), and the colored
//! [`ShColorL1`](prism_render_architecture::particle::gi_probe::ShColorL1) /
//! [`ShColorL2`](prism_render_architecture::particle::gi_probe::ShColorL2)
//! `evaluate_radiance` / `evaluate_irradiance` reconstruction (the coefficient
//! vs basis dot product, with the cosine-lobe weights for irradiance).
//!
//! # What is not twinned
//!
//! The host-only, index-and-state parts of the golden are deliberately left on
//! the `CPU`: the `usize` addressing and bounds of
//! [`ProbeGrid`](prism_render_architecture::particle::gi_probe::ProbeGrid) /
//! [`GridDims`](prism_render_architecture::particle::gi_probe::GridDims), and the
//! accumulating projection state of
//! [`ShProjector`](prism_render_architecture::particle::gi_probe::ShProjector).
//! Those are grid bookkeeping and bake-time accumulation, not the per-direction
//! arithmetic a one-thread-per-element kernel is for.
//!
//! # Portability
//!
//! The kernel uses only the portable core-`WGSL` subset — `abs`, `max`, `dot`,
//! `select`, `+ - * /` and one guarded `sqrt` — with no `sin`, `cos`, `exp`,
//! `log`, `pow`, `tan`, no inverse trigonometry, no transcendental and no
//! optional device feature, so it runs unmodified on `Metal`, `Vulkan` and
//! `DX12`. The sign fold the reference writes as a branch is written here as a
//! `select`, since core-`WGSL` has no `sign` builtin in this subset.
//!
//! # Correctness model
//!
//! Each query is a fixed, non-reorderable sequence of multiplies, adds and one
//! guarded divide / `sqrt`, so `CPU` and `GPU` evaluate the same closed form in
//! the same associativity. They are not bit-exact: a `GPU` may fuse a
//! multiply-add the scalar reference leaves separate, perturbing the low
//! mantissa bits by a few units in the last place. The parity test therefore
//! asserts a tolerance (`abs_diff <= 1e-4` or `rel_diff <= 1e-3`) on the `f32`
//! fields, tight enough to catch a genuinely wrong port (a swapped `SH`
//! coefficient, a dropped octahedral fold, a wrong ambient weight) yet loose
//! enough to admit legal fused multiply-add contraction.
//!
//! # Safety
//!
//! The crate forbids `unsafe`; it relies solely on the safe `wgpu` and
//! `bytemuck` surfaces.
//!
//! Provenance: twinned from this repository's
//! [`gi_probe`](prism_render_architecture::particle::gi_probe); no third-party
//! engine source or derived code.

use bytemuck::{Pod, Zeroable};
use wgpu::util::{BufferInitDescriptor, DeviceExt};
use wgpu::{
    BindGroupDescriptor, BindGroupEntry, BindGroupLayout, BindGroupLayoutDescriptor,
    BindGroupLayoutEntry, BindingType, BufferBindingType, BufferDescriptor, BufferUsages,
    CommandEncoderDescriptor, ComputePassDescriptor, ComputePipeline, ComputePipelineDescriptor,
    MapMode, PipelineCompilationOptions, PipelineLayoutDescriptor, ShaderModule,
    ShaderModuleDescriptor, ShaderSource, ShaderStages,
};

use crate::context::GpuContext;

/// Number of threads per workgroup. `64` is the portable, warp-friendly default
/// used across this crate's one-thread-per-element kernels.
const WORKGROUP_SIZE: u32 = 64;

/// Op code for the band 0-1 `SH` basis evaluation.
const OP_SH_BASIS_L1: u32 = 0;
/// Op code for the band 0-2 `SH` basis evaluation.
const OP_SH_BASIS_L2: u32 = 1;
/// Op code for the cosine-lobe convolution weight lookup.
const OP_COSINE_LOBE: u32 = 2;
/// Op code for the octahedral direction encode.
const OP_OCTA_ENCODE: u32 = 3;
/// Op code for the octahedral direction decode.
const OP_OCTA_DECODE: u32 = 4;
/// Op code for the octahedral `[-1, 1]^2` -> `[0, 1]^2` remap.
const OP_OCTA_TO_UNORM: u32 = 5;
/// Op code for the octahedral `[0, 1]^2` -> `[-1, 1]^2` remap.
const OP_OCTA_FROM_UNORM: u32 = 6;
/// Op code for the ambient-cube irradiance sample.
const OP_AMBIENT_CUBE: u32 = 7;
/// Op code for the colored `L1` `SH` radiance/irradiance reconstruction.
const OP_SH_RECONSTRUCT_L1: u32 = 8;
/// Op code for the colored `L2` `SH` radiance/irradiance reconstruction.
const OP_SH_RECONSTRUCT_L2: u32 = 9;

/// The portable core-`WGSL` `GI`-probe kernel, embedded inline so the twin ships
/// as a single source file. The single entry point `solve` dispatches on an op
/// code to mirror each golden
/// [`gi_probe`](prism_render_architecture::particle::gi_probe) routine for
/// routine; see the module documentation for the algorithm.
const GI_PROBE_WGSL: &str = r#"
// GI-probe twin: one thread per query reproduces the real SH basis (bands 0-2),
// the Lambert cosine-lobe convolution weights, the octahedral direction map and
// its texture-space remaps, the squared-component ambient-cube blend, and the
// colored SH radiance/irradiance reconstruction. It mirrors the CPU golden
// gi_probe routines for routine.
//
// Portability: only the core subset (abs, max, dot, select, + - * / and one
// guarded sqrt) is used; no sin/cos/exp/log/pow/tan, no inverse trigonometry
// and no optional device feature, so it runs unmodified on Metal, Vulkan and
// DX12.
//
// Provenance: twinned from this repository's particle::gi_probe; no third-party
// engine source or derived code.

// Absolute tolerance guarding the ambient-cube degenerate-normal test, matching
// the reference `EPS`; never an exact == / != on an f32.
const EPS: f32 = 1.0e-6;
// Squared-length floor of the robust normalise, matching the reference
// `EPS_LEN_SQ`, so a (near) zero direction collapses to zero instead of NaN.
const EPS_LEN_SQ: f32 = 1.0e-12;
// Archimedes' constant as an f32 literal (no transcendental call), matching the
// reference `PI`.
const PI: f32 = 3.1415927;

// Real SH basis constants, byte-for-byte the reference literals.
const SH_K0: f32 = 0.28209479;
const SH_K1: f32 = 0.4886025;
const SH_K2_XY: f32 = 1.0925484;
const SH_K2_Z2: f32 = 0.31539157;
const SH_K2_X2: f32 = 0.5462742;

// Lambert cosine-lobe convolution weights, matching the reference constants.
const COSINE_LOBE_L0: f32 = PI;
const COSINE_LOBE_L1: f32 = 2.0 * PI / 3.0;
const COSINE_LOBE_L2: f32 = PI / 4.0;

// Op codes, matching the host-side constants.
const OP_SH_BASIS_L1: u32 = 0u;
const OP_SH_BASIS_L2: u32 = 1u;
const OP_COSINE_LOBE: u32 = 2u;
const OP_OCTA_ENCODE: u32 = 3u;
const OP_OCTA_DECODE: u32 = 4u;
const OP_OCTA_TO_UNORM: u32 = 5u;
const OP_OCTA_FROM_UNORM: u32 = 6u;
const OP_AMBIENT_CUBE: u32 = 7u;
const OP_SH_RECONSTRUCT_L1: u32 = 8u;
const OP_SH_RECONSTRUCT_L2: u32 = 9u;

struct Params {
    // Number of queries in the storage arrays.
    count: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

struct Query {
    // Op code selecting the routine.
    op: u32,
    // Auxiliary integer: the band for the cosine-lobe weight, or the
    // irradiance flag (1 = irradiance, 0 = radiance) for the SH reconstruction.
    aux: u32,
    pad0: u32,
    pad1: u32,
    // Direction / normal input for the basis, octahedral encode, ambient-cube
    // sample and SH reconstruction.
    dir_x: f32,
    dir_y: f32,
    dir_z: f32,
    pad2: f32,
    // Octahedral coordinate input for decode and the texture-space remaps.
    uv_u: f32,
    uv_v: f32,
    pad3: f32,
    pad4: f32,
    // Colored coefficient / face payload: up to nine RGB SH coefficients (27
    // lanes) for the reconstruction, or six RGB ambient-cube faces (18 lanes)
    // in pos_x, neg_x, pos_y, neg_y, pos_z, neg_z order.
    c: array<f32, 27>,
}

struct Result {
    // The four / nine SH basis values (unused slots left zero).
    basis: array<f32, 9>,
    // Vector output: the decoded direction, the ambient-cube sample, or the
    // reconstructed radiance/irradiance.
    vx: f32,
    vy: f32,
    vz: f32,
    // Scalar-pair output: the octahedral encode / remap coordinates.
    uu: f32,
    vv: f32,
    // Single-scalar output: the cosine-lobe weight.
    scalar: f32,
    pad0: f32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> queries: array<Query>;
@group(0) @binding(2) var<storage, read_write> results: array<Result>;

// Returns the unit vector along v, or zero when v is (numerically) the zero
// vector, mirroring the reference `Vec3::normalize_or_zero`.
fn normalize_or_zero(v: vec3<f32>) -> vec3<f32> {
    let len_sq = dot(v, v);
    if (len_sq > EPS_LEN_SQ) {
        return v * (1.0 / sqrt(len_sq));
    }
    return vec3<f32>(0.0, 0.0, 0.0);
}

// Returns +1 for non-negative inputs and -1 otherwise, mirroring the reference
// `sign_nonzero`; written with select since core-WGSL has no sign builtin here.
fn sign_nonzero(v: f32) -> f32 {
    return select(1.0, -1.0, v < 0.0);
}

// Cosine-lobe convolution weight for a band in 0..=2; higher bands convolve to
// zero. Mirrors the reference `cosine_lobe_weight`.
fn cosine_lobe_weight(band: u32) -> f32 {
    if (band == 0u) {
        return COSINE_LOBE_L0;
    }
    if (band == 1u) {
        return COSINE_LOBE_L1;
    }
    if (band == 2u) {
        return COSINE_LOBE_L2;
    }
    return 0.0;
}

// Lambert transfer weight for the k-th L2 coefficient slot, mirroring the
// reference `BAND_OF_INDEX` lookup [0, 1, 1, 1, 2, 2, 2, 2, 2].
fn weight_of_index(k: u32) -> f32 {
    if (k == 0u) {
        return COSINE_LOBE_L0;
    }
    if (k <= 3u) {
        return COSINE_LOBE_L1;
    }
    return COSINE_LOBE_L2;
}

// Reads the k-th RGB coefficient / face from the flat payload.
fn coeff_at(qd: Query, k: u32) -> vec3<f32> {
    let b = k * 3u;
    return vec3<f32>(qd.c[b], qd.c[b + 1u], qd.c[b + 2u]);
}

// Octahedral encode: projects onto |x|+|y|+|z|=1 and folds the lower hemisphere
// outward, mirroring the reference `octa_encode`.
fn octa_encode(dir: vec3<f32>) -> vec2<f32> {
    let n = normalize_or_zero(dir);
    let denom = abs(n.x) + abs(n.y) + abs(n.z);
    if (denom <= EPS) {
        return vec2<f32>(0.0, 0.0);
    }
    let inv = 1.0 / denom;
    let px = n.x * inv;
    let py = n.y * inv;
    if (n.z >= 0.0) {
        return vec2<f32>(px, py);
    }
    return vec2<f32>(
        (1.0 - abs(py)) * sign_nonzero(px),
        (1.0 - abs(px)) * sign_nonzero(py),
    );
}

// Octahedral decode, inverse of octa_encode, mirroring the reference
// `octa_decode`.
fn octa_decode(u: f32, v: f32) -> vec3<f32> {
    let z = 1.0 - abs(u) - abs(v);
    let t = max(-z, 0.0);
    let x = u - t * sign_nonzero(u);
    let y = v - t * sign_nonzero(v);
    return normalize_or_zero(vec3<f32>(x, y, z));
}

// Colored L1 reconstruction: coefficient vs basis dot product, with the
// cosine-lobe weights when irradiance is requested, mirroring the reference
// `ShColorL1::evaluate_radiance` / `evaluate_irradiance`.
fn sh_reconstruct_l1(qd: Query, dir: vec3<f32>, irradiance: u32) -> vec3<f32> {
    let n = normalize_or_zero(dir);
    var basis: array<f32, 4>;
    basis[0] = SH_K0;
    basis[1] = SH_K1 * n.y;
    basis[2] = SH_K1 * n.z;
    basis[3] = SH_K1 * n.x;
    var acc = vec3<f32>(0.0, 0.0, 0.0);
    for (var k = 0u; k < 4u; k = k + 1u) {
        var w = 1.0;
        if (irradiance == 1u) {
            w = select(COSINE_LOBE_L1, COSINE_LOBE_L0, k == 0u);
        }
        acc = acc + coeff_at(qd, k) * (w * basis[k]);
    }
    return acc;
}

// Colored L2 reconstruction, mirroring the reference
// `ShColorL2::evaluate_radiance` / `evaluate_irradiance`.
fn sh_reconstruct_l2(qd: Query, dir: vec3<f32>, irradiance: u32) -> vec3<f32> {
    let n = normalize_or_zero(dir);
    var basis: array<f32, 9>;
    basis[0] = SH_K0;
    basis[1] = SH_K1 * n.y;
    basis[2] = SH_K1 * n.z;
    basis[3] = SH_K1 * n.x;
    basis[4] = SH_K2_XY * (n.x * n.y);
    basis[5] = SH_K2_XY * (n.y * n.z);
    basis[6] = SH_K2_Z2 * (3.0 * n.z * n.z - 1.0);
    basis[7] = SH_K2_XY * (n.x * n.z);
    basis[8] = SH_K2_X2 * (n.x * n.x - n.y * n.y);
    var acc = vec3<f32>(0.0, 0.0, 0.0);
    for (var k = 0u; k < 9u; k = k + 1u) {
        var w = 1.0;
        if (irradiance == 1u) {
            w = weight_of_index(k);
        }
        acc = acc + coeff_at(qd, k) * (w * basis[k]);
    }
    return acc;
}

// Ambient-cube sample: squared-component blend of the three faces the normal
// points toward, mirroring the reference `AmbientCube::sample`. A degenerate
// normal returns the mean of the six faces.
fn ambient_cube_sample(qd: Query, n_in: vec3<f32>) -> vec3<f32> {
    let pos_x = coeff_at(qd, 0u);
    let neg_x = coeff_at(qd, 1u);
    let pos_y = coeff_at(qd, 2u);
    let neg_y = coeff_at(qd, 3u);
    let pos_z = coeff_at(qd, 4u);
    let neg_z = coeff_at(qd, 5u);
    let d = normalize_or_zero(n_in);
    if (dot(d, d) <= EPS) {
        let sum = pos_x + neg_x + pos_y + neg_y + pos_z + neg_z;
        return sum * (1.0 / 6.0);
    }
    let wx = d.x * d.x;
    let wy = d.y * d.y;
    let wz = d.z * d.z;
    let fx = select(neg_x, pos_x, d.x >= 0.0);
    let fy = select(neg_y, pos_y, d.y >= 0.0);
    let fz = select(neg_z, pos_z, d.z >= 0.0);
    return fx * wx + fy * wy + fz * wz;
}

@compute @workgroup_size(64)
fn solve(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.count) {
        return;
    }
    let qd = queries[idx];
    var r: Result;
    for (var i = 0u; i < 9u; i = i + 1u) {
        r.basis[i] = 0.0;
    }
    r.vx = 0.0;
    r.vy = 0.0;
    r.vz = 0.0;
    r.uu = 0.0;
    r.vv = 0.0;
    r.scalar = 0.0;
    r.pad0 = 0.0;

    let dir = vec3<f32>(qd.dir_x, qd.dir_y, qd.dir_z);
    let op = qd.op;

    if (op == OP_SH_BASIS_L1) {
        let n = normalize_or_zero(dir);
        r.basis[0] = SH_K0;
        r.basis[1] = SH_K1 * n.y;
        r.basis[2] = SH_K1 * n.z;
        r.basis[3] = SH_K1 * n.x;
    } else if (op == OP_SH_BASIS_L2) {
        let n = normalize_or_zero(dir);
        r.basis[0] = SH_K0;
        r.basis[1] = SH_K1 * n.y;
        r.basis[2] = SH_K1 * n.z;
        r.basis[3] = SH_K1 * n.x;
        r.basis[4] = SH_K2_XY * (n.x * n.y);
        r.basis[5] = SH_K2_XY * (n.y * n.z);
        r.basis[6] = SH_K2_Z2 * (3.0 * n.z * n.z - 1.0);
        r.basis[7] = SH_K2_XY * (n.x * n.z);
        r.basis[8] = SH_K2_X2 * (n.x * n.x - n.y * n.y);
    } else if (op == OP_COSINE_LOBE) {
        r.scalar = cosine_lobe_weight(qd.aux);
    } else if (op == OP_OCTA_ENCODE) {
        let e = octa_encode(dir);
        r.uu = e.x;
        r.vv = e.y;
    } else if (op == OP_OCTA_DECODE) {
        let d = octa_decode(qd.uv_u, qd.uv_v);
        r.vx = d.x;
        r.vy = d.y;
        r.vz = d.z;
    } else if (op == OP_OCTA_TO_UNORM) {
        r.uu = qd.uv_u * 0.5 + 0.5;
        r.vv = qd.uv_v * 0.5 + 0.5;
    } else if (op == OP_OCTA_FROM_UNORM) {
        r.uu = qd.uv_u * 2.0 - 1.0;
        r.vv = qd.uv_v * 2.0 - 1.0;
    } else if (op == OP_AMBIENT_CUBE) {
        let s = ambient_cube_sample(qd, dir);
        r.vx = s.x;
        r.vy = s.y;
        r.vz = s.z;
    } else if (op == OP_SH_RECONSTRUCT_L1) {
        let s = sh_reconstruct_l1(qd, dir, qd.aux);
        r.vx = s.x;
        r.vy = s.y;
        r.vz = s.z;
    } else {
        let s = sh_reconstruct_l2(qd, dir, qd.aux);
        r.vx = s.x;
        r.vy = s.y;
        r.vz = s.z;
    }

    results[idx] = r;
}
"#;

/// One `GI`-probe query: the routine and its inputs.
///
/// Each variant mirrors one golden
/// [`gi_probe`](prism_render_architecture::particle::gi_probe) routine. The
/// direction inputs are normalized robustly on-device exactly as the reference
/// does, so a non-unit or zero direction never yields `NaN`.
///
/// Provenance: twinned from this repository's
/// [`gi_probe`](prism_render_architecture::particle::gi_probe); no third-party
/// engine source or derived code.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum GiProbeQuery {
    /// Evaluate the four band 0-1 `SH` basis values for `dir`
    /// ([`sh_basis_l1`](prism_render_architecture::particle::gi_probe::sh_basis_l1)).
    ShBasisL1 {
        /// Direction to evaluate the basis along.
        dir: [f32; 3],
    },
    /// Evaluate the nine band 0-2 `SH` basis values for `dir`
    /// ([`sh_basis_l2`](prism_render_architecture::particle::gi_probe::sh_basis_l2)).
    ShBasisL2 {
        /// Direction to evaluate the basis along.
        dir: [f32; 3],
    },
    /// Look up the Lambert cosine-lobe convolution weight for a band
    /// ([`cosine_lobe_weight`](prism_render_architecture::particle::gi_probe::cosine_lobe_weight)).
    CosineLobe {
        /// Band index (`0..=2`; higher bands convolve to zero).
        band: u32,
    },
    /// Encode `dir` into octahedral `[-1, 1]^2` coordinates
    /// ([`octa_encode`](prism_render_architecture::particle::gi_probe::octa_encode)).
    OctaEncode {
        /// Direction to encode.
        dir: [f32; 3],
    },
    /// Decode octahedral `(u, v)` into a unit direction
    /// ([`octa_decode`](prism_render_architecture::particle::gi_probe::octa_decode)).
    OctaDecode {
        /// Octahedral `u` coordinate.
        u: f32,
        /// Octahedral `v` coordinate.
        v: f32,
    },
    /// Remap octahedral `[-1, 1]^2` into texture-space `[0, 1]^2`
    /// ([`octa_to_unorm`](prism_render_architecture::particle::gi_probe::octa_to_unorm)).
    OctaToUnorm {
        /// Octahedral `u` coordinate.
        u: f32,
        /// Octahedral `v` coordinate.
        v: f32,
    },
    /// Remap texture-space `[0, 1]^2` back into octahedral `[-1, 1]^2`
    /// ([`octa_from_unorm`](prism_render_architecture::particle::gi_probe::octa_from_unorm)).
    OctaFromUnorm {
        /// Texture-space `u` coordinate.
        u: f32,
        /// Texture-space `v` coordinate.
        v: f32,
    },
    /// Sample a six-face ambient cube along `normal`
    /// ([`AmbientCube::sample`](prism_render_architecture::particle::gi_probe::AmbientCube::sample)).
    AmbientCube {
        /// The six `RGB` faces in `pos_x`, `neg_x`, `pos_y`, `neg_y`, `pos_z`,
        /// `neg_z` order.
        faces: [[f32; 3]; 6],
        /// The surface normal to sample along.
        normal: [f32; 3],
    },
    /// Reconstruct colored `L1` radiance or irradiance along `dir`
    /// ([`ShColorL1`](prism_render_architecture::particle::gi_probe::ShColorL1)).
    ShReconstructL1 {
        /// The four `RGB` coefficients in `sh_basis_l1` order.
        coeffs: [[f32; 3]; 4],
        /// Direction to reconstruct along.
        dir: [f32; 3],
        /// When `true`, apply the cosine-lobe weights (irradiance); when
        /// `false`, reconstruct raw radiance.
        irradiance: bool,
    },
    /// Reconstruct colored `L2` radiance or irradiance along `dir`
    /// ([`ShColorL2`](prism_render_architecture::particle::gi_probe::ShColorL2)).
    ShReconstructL2 {
        /// The nine `RGB` coefficients in `sh_basis_l2` order.
        coeffs: [[f32; 3]; 9],
        /// Direction to reconstruct along.
        dir: [f32; 3],
        /// When `true`, apply the cosine-lobe weights (irradiance); when
        /// `false`, reconstruct raw radiance.
        irradiance: bool,
    },
}

/// The resolved answer for one [`GiProbeQuery`], one variant per query kind.
///
/// Provenance: twinned from this repository's
/// [`gi_probe`](prism_render_architecture::particle::gi_probe); no third-party
/// engine source or derived code.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum GiProbeResult {
    /// The four band 0-1 `SH` basis values, matching `sh_basis_l1`.
    ShBasisL1([f32; 4]),
    /// The nine band 0-2 `SH` basis values, matching `sh_basis_l2`.
    ShBasisL2([f32; 9]),
    /// The cosine-lobe convolution weight, matching `cosine_lobe_weight`.
    CosineLobe(f32),
    /// The octahedral coordinates, matching `octa_encode`.
    OctaEncode {
        /// Octahedral `u` coordinate.
        u: f32,
        /// Octahedral `v` coordinate.
        v: f32,
    },
    /// The decoded unit direction, matching `octa_decode`.
    OctaDecode([f32; 3]),
    /// The texture-space coordinates, matching `octa_to_unorm`.
    OctaToUnorm {
        /// Texture-space `u` coordinate.
        u: f32,
        /// Texture-space `v` coordinate.
        v: f32,
    },
    /// The octahedral coordinates, matching `octa_from_unorm`.
    OctaFromUnorm {
        /// Octahedral `u` coordinate.
        u: f32,
        /// Octahedral `v` coordinate.
        v: f32,
    },
    /// The sampled irradiance, matching `AmbientCube::sample`.
    AmbientCube([f32; 3]),
    /// The reconstructed radiance/irradiance, matching the `ShColorL1`
    /// reconstruction.
    ShReconstructL1([f32; 3]),
    /// The reconstructed radiance/irradiance, matching the `ShColorL2`
    /// reconstruction.
    ShReconstructL2([f32; 3]),
}

/// `repr(C)` `std430` layout of one packed query: four `u32` control words
/// (`op`, `aux`, two pads), a direction slot and an octahedral-coordinate slot
/// (each four `f32`), then a `27`-lane `f32` coefficient / face payload — `156`
/// bytes, matching the `WGSL` `Query` struct lane for lane.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuQuery {
    /// Op code selecting the routine.
    op: u32,
    /// Band (cosine lobe) or irradiance flag (`SH` reconstruction).
    aux: u32,
    /// Padding word.
    pad0: u32,
    /// Padding word.
    pad1: u32,
    /// Direction / normal `x`.
    dir_x: f32,
    /// Direction / normal `y`.
    dir_y: f32,
    /// Direction / normal `z`.
    dir_z: f32,
    /// Padding lane.
    pad2: f32,
    /// Octahedral `u` input.
    uv_u: f32,
    /// Octahedral `v` input.
    uv_v: f32,
    /// Padding lane.
    pad3: f32,
    /// Padding lane.
    pad4: f32,
    /// Up to nine `RGB` `SH` coefficients, or six ambient-cube faces, flattened.
    c: [f32; 27],
}

impl GpuQuery {
    /// Packs one public query into its `std430` image.
    fn new(query: &GiProbeQuery) -> GpuQuery {
        let mut g = GpuQuery::zeroed();
        match query {
            GiProbeQuery::ShBasisL1 { dir } => {
                g.op = OP_SH_BASIS_L1;
                set_dir(&mut g, *dir);
            }
            GiProbeQuery::ShBasisL2 { dir } => {
                g.op = OP_SH_BASIS_L2;
                set_dir(&mut g, *dir);
            }
            GiProbeQuery::CosineLobe { band } => {
                g.op = OP_COSINE_LOBE;
                g.aux = *band;
            }
            GiProbeQuery::OctaEncode { dir } => {
                g.op = OP_OCTA_ENCODE;
                set_dir(&mut g, *dir);
            }
            GiProbeQuery::OctaDecode { u, v } => {
                g.op = OP_OCTA_DECODE;
                g.uv_u = *u;
                g.uv_v = *v;
            }
            GiProbeQuery::OctaToUnorm { u, v } => {
                g.op = OP_OCTA_TO_UNORM;
                g.uv_u = *u;
                g.uv_v = *v;
            }
            GiProbeQuery::OctaFromUnorm { u, v } => {
                g.op = OP_OCTA_FROM_UNORM;
                g.uv_u = *u;
                g.uv_v = *v;
            }
            GiProbeQuery::AmbientCube { faces, normal } => {
                g.op = OP_AMBIENT_CUBE;
                set_dir(&mut g, *normal);
                for (slot, face) in faces.iter().enumerate() {
                    set_coeff(&mut g, slot, *face);
                }
            }
            GiProbeQuery::ShReconstructL1 {
                coeffs,
                dir,
                irradiance,
            } => {
                g.op = OP_SH_RECONSTRUCT_L1;
                set_dir(&mut g, *dir);
                g.aux = u32::from(*irradiance);
                for (slot, coeff) in coeffs.iter().enumerate() {
                    set_coeff(&mut g, slot, *coeff);
                }
            }
            GiProbeQuery::ShReconstructL2 {
                coeffs,
                dir,
                irradiance,
            } => {
                g.op = OP_SH_RECONSTRUCT_L2;
                set_dir(&mut g, *dir);
                g.aux = u32::from(*irradiance);
                for (slot, coeff) in coeffs.iter().enumerate() {
                    set_coeff(&mut g, slot, *coeff);
                }
            }
        }
        g
    }
}

/// Writes a direction triple into the packed query's direction slot.
fn set_dir(g: &mut GpuQuery, dir: [f32; 3]) {
    g.dir_x = dir[0];
    g.dir_y = dir[1];
    g.dir_z = dir[2];
}

/// Writes the `slot`-th `RGB` coefficient / face into the flat payload.
fn set_coeff(g: &mut GpuQuery, slot: usize, rgb: [f32; 3]) {
    let base = slot * 3;
    g.c[base] = rgb[0];
    g.c[base + 1] = rgb[1];
    g.c[base + 2] = rgb[2];
}

/// `repr(C)` `std430` layout of one result: a nine-lane `f32` basis block, a
/// vector slot, an octahedral-coordinate pair, a scalar and one pad — `64`
/// bytes, matching the `WGSL` `Result` struct lane for lane.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuResult {
    /// The four / nine `SH` basis values.
    basis: [f32; 9],
    /// Vector output `x`.
    vx: f32,
    /// Vector output `y`.
    vy: f32,
    /// Vector output `z`.
    vz: f32,
    /// Octahedral / texture-space `u` output.
    uu: f32,
    /// Octahedral / texture-space `v` output.
    vv: f32,
    /// Single-scalar output (the cosine-lobe weight).
    scalar: f32,
    /// Padding lane.
    pad0: f32,
}

/// Uniform parameters for one dispatch: the query count plus three pad words to
/// fill a `16`-byte, `16`-byte-aligned uniform struct.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuParams {
    /// Number of queries in the storage arrays.
    count: u32,
    /// Padding word.
    pad0: u32,
    /// Padding word.
    pad1: u32,
    /// Padding word.
    pad2: u32,
}

/// Decodes one packed [`GpuResult`] into the public [`GiProbeResult`] matching
/// the originating `query`'s variant.
fn decode_result(query: &GiProbeQuery, raw: &GpuResult) -> GiProbeResult {
    match query {
        GiProbeQuery::ShBasisL1 { .. } => {
            GiProbeResult::ShBasisL1([raw.basis[0], raw.basis[1], raw.basis[2], raw.basis[3]])
        }
        GiProbeQuery::ShBasisL2 { .. } => GiProbeResult::ShBasisL2(raw.basis),
        GiProbeQuery::CosineLobe { .. } => GiProbeResult::CosineLobe(raw.scalar),
        GiProbeQuery::OctaEncode { .. } => GiProbeResult::OctaEncode {
            u: raw.uu,
            v: raw.vv,
        },
        GiProbeQuery::OctaDecode { .. } => GiProbeResult::OctaDecode([raw.vx, raw.vy, raw.vz]),
        GiProbeQuery::OctaToUnorm { .. } => GiProbeResult::OctaToUnorm {
            u: raw.uu,
            v: raw.vv,
        },
        GiProbeQuery::OctaFromUnorm { .. } => GiProbeResult::OctaFromUnorm {
            u: raw.uu,
            v: raw.vv,
        },
        GiProbeQuery::AmbientCube { .. } => GiProbeResult::AmbientCube([raw.vx, raw.vy, raw.vz]),
        GiProbeQuery::ShReconstructL1 { .. } => {
            GiProbeResult::ShReconstructL1([raw.vx, raw.vy, raw.vz])
        }
        GiProbeQuery::ShReconstructL2 { .. } => {
            GiProbeResult::ShReconstructL2([raw.vx, raw.vy, raw.vz])
        }
    }
}

/// Builds a compute-visible buffer binding layout entry.
fn buffer_entry(binding: u32, ty: BufferBindingType) -> BindGroupLayoutEntry {
    BindGroupLayoutEntry {
        binding,
        visibility: ShaderStages::COMPUTE,
        ty: BindingType::Buffer {
            ty,
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

/// A compiled, reusable `GI`-probe compute pipeline.
pub struct GpuGiProbe {
    #[expect(
        dead_code,
        reason = "kept alive so the pipeline it produced stays valid"
    )]
    module: ShaderModule,
    layout: BindGroupLayout,
    pipeline: ComputePipeline,
}

impl GpuGiProbe {
    /// Compiles the `GI`-probe kernel on `ctx`.
    ///
    /// The kernel uses only the portable core-`WGSL` subset, so no optional
    /// device feature is required.
    #[must_use]
    pub fn new(ctx: &GpuContext) -> GpuGiProbe {
        let device = ctx.device();
        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("prism_volumetric_gi_probe"),
            source: ShaderSource::Wgsl(GI_PROBE_WGSL.into()),
        });
        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_volumetric_gi_probe_layout"),
            entries: &[
                buffer_entry(0, BufferBindingType::Uniform),
                buffer_entry(1, BufferBindingType::Storage { read_only: true }),
                buffer_entry(2, BufferBindingType::Storage { read_only: false }),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("prism_volumetric_gi_probe_pipeline_layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("prism_volumetric_gi_probe_pipeline"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("solve"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });
        GpuGiProbe {
            module,
            layout,
            pipeline,
        }
    }

    /// Solves every query on-device and returns one [`GiProbeResult`] per input,
    /// in order.
    ///
    /// Each result equals the reference answer for the query's variant to within
    /// the tolerance documented on this module. An empty input returns an empty
    /// vector with no dispatch issued, since a storage buffer cannot be
    /// zero-sized.
    #[must_use]
    pub fn eval(&self, ctx: &GpuContext, queries: &[GiProbeQuery]) -> Vec<GiProbeResult> {
        if queries.is_empty() {
            return Vec::new();
        }
        let device = ctx.device();
        let count = queries.len();

        let packed: Vec<GpuQuery> = queries.iter().map(GpuQuery::new).collect();
        let input_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_gi_probe_input"),
            contents: bytemuck::cast_slice(&packed),
            usage: BufferUsages::STORAGE,
        });

        let out_bytes = (count * size_of::<GpuResult>()) as u64;
        let output_buf = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_gi_probe_output"),
            size: out_bytes,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let params = GpuParams {
            count: count as u32,
            pad0: 0,
            pad1: 0,
            pad2: 0,
        };
        let params_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_gi_probe_params"),
            contents: bytemuck::bytes_of(&params),
            usage: BufferUsages::UNIFORM,
        });

        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("prism_volumetric_gi_probe_bind_group"),
            layout: &self.layout,
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: params_buf.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: input_buf.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 2,
                    resource: output_buf.as_entire_binding(),
                },
            ],
        });

        let stage = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_gi_probe_stage"),
            size: out_bytes,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("prism_volumetric_gi_probe_encoder"),
        });
        {
            let groups = (count as u32).div_ceil(WORKGROUP_SIZE);
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("prism_volumetric_gi_probe_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            // One thread per query, flattened to a 1-D dispatch.
            pass.dispatch_workgroups(groups, 1, 1);
        }
        encoder.copy_buffer_to_buffer(&output_buf, 0, &stage, 0, out_bytes);
        ctx.queue().submit([encoder.finish()]);

        stage.slice(..).map_async(MapMode::Read, |_| {});
        ctx.wait();
        let view = stage
            .slice(..)
            .get_mapped_range()
            .expect("mapped readback range should be available after poll");
        let raw = bytemuck::cast_slice::<u8, GpuResult>(&view).to_vec();
        drop(view);
        stage.unmap();

        queries
            .iter()
            .zip(raw.iter())
            .map(|(query, result)| decode_result(query, result))
            .collect()
    }
}
