//! `wgpu` compute twin of the particle-probe spherical-harmonic (`SH`) rotation
//! golden
//! [`particle::spherical_harmonics_rotate`](prism_render_architecture::particle::spherical_harmonics_rotate),
//! design sections 16 and 17 ("球谐 probe 重光照/混合").
//!
//! When a particle system carries baked `SH` irradiance and the emitter or the
//! scene rotates, the stored band-`L1`/band-`L2` coefficients must be
//! re-expressed in the new frame *without* re-projecting any radiance sample.
//! The `CPU` golden owns that recombination as pure linear algebra: it never
//! evaluates a basis function and never projects a sample, it only applies a
//! rotation to existing coefficient vectors. Band `L1` transforms like a
//! direction vector (a 3x3 matrix multiply after a fixed coordinate shuffle),
//! and band `L2` transforms by a 5x5 matrix built algebraically from the 3x3
//! rotation with the `Ivanic`-`Ruedenberg` recurrence — no trigonometry.
//!
//! [`GpuSphericalHarmonicsRotate`] is the on-device twin: one compute thread per
//! probe rebuilds the 3x3 rotation from the probe's quaternion, rotates every
//! `RGB` channel of the band-`L1` payload, builds the 5x5 band-`L2` matrix with
//! the identical algebra, and rotates every `RGB` channel of the band-`L2`
//! payload. The band-`L0` (`DC`) term is rotation-invariant and passes through
//! untouched. A passing real-device parity test is therefore direct evidence
//! the ported kernel reproduces the reference coefficient-for-coefficient, not
//! merely that its shader compiles.
//!
//! # Step-for-step parity
//!
//! The kernel mirrors the reference exactly:
//!
//! * The quaternion-to-matrix conversion is the same robust normalized
//!   polynomial as
//!   [`rotation_matrix_from_quat`](prism_render_architecture::particle::spherical_harmonics_rotate::rotation_matrix_from_quat)
//!   (scale by `2 / |q|^2`, zero quaternion returns identity).
//! * Band-`L1` rotation uses the same `(y, z, x)` -> Cartesian `(x, y, z)`
//!   shuffle, 3x3 multiply, and inverse shuffle as
//!   [`rotate_l1`](prism_render_architecture::particle::spherical_harmonics_rotate::rotate_l1).
//! * Band-`L2` uses the same `Ivanic`-`Ruedenberg` `band1`/`p2`/`u`/`v` terms
//!   and `u`/`v` scalar coefficients, assembled into the same `m = -2..=2`
//!   row/column order as
//!   [`l2_rotation_matrix`](prism_render_architecture::particle::spherical_harmonics_rotate::l2_rotation_matrix),
//!   then applied with the same per-row dot product as
//!   [`rotate_l2`](prism_render_architecture::particle::spherical_harmonics_rotate::rotate_l2).
//! * The storage layout reuses the reference `std430` convention of one padded
//!   `vec4` slot per `RGB` triple (see
//!   [`ShL1Rgb::to_std430`](prism_render_architecture::particle::spherical_harmonics_rotate::ShL1Rgb::to_std430)).
//!
//! # Portability
//!
//! The kernel uses only the portable core-`WGSL` subset — `sqrt`, `min`, `max`,
//! `+ - * /` and signed/unsigned integer compares — with no `sin`, `cos`,
//! `exp`, `log`, `pow` or optional device feature, so it runs unmodified on
//! Metal, Vulkan and DX12. The only non-integer primitive is `sqrt`, used
//! solely to form the band-`L2` `u`/`v` normalization coefficients, exactly as
//! the reference does; the band-`L1` path uses no `sqrt` at all.
//!
//! # Correctness model
//!
//! The whole pipeline is rational-plus-`sqrt` algebra with no transcendental
//! call and no reorderable reduction, so `CPU` and `GPU` evaluate the same
//! closed form in the same order. They are not bit-exact: a `GPU` may fuse a
//! multiply-add the scalar reference leaves separate, perturbing the low
//! mantissa bits by a few units in the last place. The parity test asserts a
//! tolerance (`abs_diff <= 1e-5` or `rel_diff <= 1e-5`) tight enough to catch a
//! genuinely wrong port yet loose enough to admit legal fused multiply-add
//! contraction.
//!
//! # Safety
//!
//! The crate forbids `unsafe`; it relies solely on the safe `wgpu` and
//! `bytemuck` surfaces.
//!
//! Provenance: standard `Ivanic`-`Ruedenberg` real-`SH` rotation plus `wgpu`
//! compute dispatch; no Unreal Engine source or derived code.

use bytemuck::{Pod, Zeroable};
use prism_render_architecture::particle::spherical_harmonics_rotate::ShL1Rgb;
use wgpu::util::{BufferInitDescriptor, DeviceExt};
use wgpu::{
    BindGroupDescriptor, BindGroupEntry, BindGroupLayout, BindGroupLayoutDescriptor,
    BindGroupLayoutEntry, BindingType, BufferBindingType, BufferDescriptor, BufferUsages,
    CommandEncoderDescriptor, ComputePassDescriptor, ComputePipeline, ComputePipelineDescriptor,
    MapMode, PipelineCompilationOptions, PipelineLayoutDescriptor, ShaderModule,
    ShaderModuleDescriptor, ShaderSource, ShaderStages,
};

use crate::context::GpuContext;

/// The portable core-`WGSL` `SH`-rotation kernel, embedded inline so the twin
/// ships as a single source file. Mirrors the `CPU` golden step for step; see
/// the module documentation for the algorithm.
const SPHERICAL_HARMONICS_ROTATE_WGSL: &str = r#"
// Spherical-harmonic rotation twin: one thread per probe rebuilds the 3x3
// rotation from the probe quaternion, rotates every RGB channel of the band-L1
// payload (a direction-vector transform), builds the 5x5 band-L2 matrix from the
// 3x3 with the Ivanic-Ruedenberg recurrence, and rotates every RGB channel of
// the band-L2 payload. The band-L0 (DC) term is rotation-invariant and passes
// through. It mirrors the CPU golden
// `particle::spherical_harmonics_rotate`, uses only the portable core-WGSL
// subset (sqrt/min/max and + - * / plus integer compares), and takes no optional
// feature, so it runs unmodified on Metal, Vulkan and DX12.
//
// Provenance: standard Ivanic-Ruedenberg real-SH rotation; no Unreal Engine
// source or derived code.

struct Params {
    // Number of probes (one thread each).
    count: u32,
    // Padding to a 16-byte std430 struct.
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

// One input probe. 160-byte std430 stride: the rotation quaternion, then the
// band-L0 RGB term, three band-L1 RGB coefficients, and five band-L2 RGB
// coefficients, each triple in one padded vec4 slot. Matches the host
// `GpuProbeIn`.
struct ProbeIn {
    quat: vec4<f32>,
    l0: vec4<f32>,
    l1_0: vec4<f32>,
    l1_1: vec4<f32>,
    l1_2: vec4<f32>,
    l2_0: vec4<f32>,
    l2_1: vec4<f32>,
    l2_2: vec4<f32>,
    l2_3: vec4<f32>,
    l2_4: vec4<f32>,
}

// One output probe. 144-byte std430 stride: the passthrough band-L0 RGB term,
// three rotated band-L1 RGB coefficients, and five rotated band-L2 RGB
// coefficients. Matches the host `GpuProbeOut`.
struct ProbeOut {
    l0: vec4<f32>,
    l1_0: vec4<f32>,
    l1_1: vec4<f32>,
    l1_2: vec4<f32>,
    l2_0: vec4<f32>,
    l2_1: vec4<f32>,
    l2_2: vec4<f32>,
    l2_3: vec4<f32>,
    l2_4: vec4<f32>,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> inputs: array<ProbeIn>;
@group(0) @binding(2) var<storage, read_write> outputs: array<ProbeOut>;

// Square root of two, matching the reference `core::f32::consts::SQRT_2`; the
// decimal literal rounds to the same nearest f32. It is the only fixed constant
// the band-L2 `v` term needs.
const SQRT_2: f32 = 1.4142135623730951;

// Builds a 3x3 row-major rotation matrix (stored row-major as nine scalars)
// from the quaternion `(x, y, z, w)`, matching the reference
// `rotation_matrix_from_quat`: a robust normalized polynomial that scales by
// `2 / |q|^2`, so a non-unit quaternion still yields an orthonormal matrix and
// the zero quaternion returns the identity. No sqrt and no trigonometry.
fn quat_to_mat3(q: vec4<f32>) -> array<f32, 9> {
    let x = q.x;
    let y = q.y;
    let z = q.z;
    let w = q.w;
    let norm = x * x + y * y + z * z + w * w;
    var s = 0.0;
    if (norm > 0.0) {
        s = 2.0 / norm;
    }
    let xx = x * x * s;
    let yy = y * y * s;
    let zz = z * z * s;
    let xy = x * y * s;
    let xz = x * z * s;
    let yz = y * z * s;
    let wx = w * x * s;
    let wy = w * y * s;
    let wz = w * z * s;
    return array<f32, 9>(
        1.0 - (yy + zz), xy - wz, xz + wy,
        xy + wz, 1.0 - (xx + zz), yz - wx,
        xz - wy, yz + wx, 1.0 - (xx + yy),
    );
}

// Applies the row-major 3x3 matrix to a column vector.
fn apply_mat3(r1: ptr<function, array<f32, 9>>, v: vec3<f32>) -> vec3<f32> {
    return vec3<f32>(
        (*r1)[0] * v.x + (*r1)[1] * v.y + (*r1)[2] * v.z,
        (*r1)[3] * v.x + (*r1)[4] * v.y + (*r1)[5] * v.z,
        (*r1)[6] * v.x + (*r1)[7] * v.y + (*r1)[8] * v.z,
    );
}

// Rotates one color channel's band-L1 triple ordered `(y, z, x)`: shuffle to
// Cartesian `(x, y, z)`, apply the rotation, shuffle back, matching the
// reference `rotate_l1`.
fn rotate_l1_channel(r1: ptr<function, array<f32, 9>>, c0: f32, c1: f32, c2: f32) -> vec3<f32> {
    let cartesian = vec3<f32>(c2, c0, c1);
    let rotated = apply_mat3(r1, cartesian);
    return vec3<f32>(rotated.y, rotated.z, rotated.x);
}

// Reads the band-L1 matrix at signed indices in -1..=1, returning zero
// out of range, matching the reference `band1`.
fn band1(r1: ptr<function, array<f32, 9>>, row: i32, col: i32) -> f32 {
    if (row < -1 || row > 1 || col < -1 || col > 1) {
        return 0.0;
    }
    let r = u32(row + 1);
    let c = u32(col + 1);
    return (*r1)[r * 3u + c];
}

// The Ivanic-Ruedenberg `P` term for band L2 (previous band is L1), matching the
// reference `p2`.
fn p2(r1: ptr<function, array<f32, 9>>, i: i32, mu: i32, n: i32) -> f32 {
    if (n == 2) {
        return band1(r1, i, 1) * band1(r1, mu, 1) - band1(r1, i, -1) * band1(r1, mu, -1);
    }
    if (n == -2) {
        return band1(r1, i, 1) * band1(r1, mu, -1) + band1(r1, i, -1) * band1(r1, mu, 1);
    }
    return band1(r1, i, 0) * band1(r1, mu, n);
}

// The Ivanic-Ruedenberg `U` term for band L2, matching the reference `u_term`.
fn u_term(r1: ptr<function, array<f32, 9>>, m: i32, n: i32) -> f32 {
    return p2(r1, 0, m, n);
}

// The Ivanic-Ruedenberg `V` term for band L2, matching the reference `v_term`.
fn v_term(r1: ptr<function, array<f32, 9>>, m: i32, n: i32) -> f32 {
    if (m > 0) {
        var s1 = 1.0;
        var s2 = 1.0;
        if (m == 1) {
            s1 = SQRT_2;
            s2 = 0.0;
        }
        return p2(r1, 1, m - 1, n) * s1 - p2(r1, -1, -m + 1, n) * s2;
    }
    if (m < 0) {
        var s1 = 1.0;
        var s2 = 1.0;
        if (m == -1) {
            s1 = 0.0;
            s2 = SQRT_2;
        }
        return p2(r1, 1, m + 1, n) * s1 + p2(r1, -1, -m - 1, n) * s2;
    }
    return p2(r1, 1, 1, n) + p2(r1, -1, -1, n);
}

// The Ivanic-Ruedenberg `u` and `v` scalar coefficients for band L2 (the `w`
// coefficient is identically zero for l = 2 and omitted), matching the
// reference `uv_coeff`. The returned `vec2` packs `(u, v)`.
fn uv_coeff(m: i32, n: i32) -> vec2<f32> {
    // Denominator is 12 for |n| = 2, else (2 + n)(2 - n).
    var denom = (2 + n) * (2 - n);
    if (n == 2 || n == -2) {
        denom = 12;
    }
    let denom_f = f32(denom);
    let u = sqrt(f32((2 + m) * (2 - m)) / denom_f);
    var abs_m = m;
    if (m < 0) {
        abs_m = -m;
    }
    var delta0 = 0;
    if (m == 0) {
        delta0 = 1;
    }
    let v_num = f32((1 + delta0) * (1 + abs_m) * (2 + abs_m));
    var v_sign = 1.0;
    if (m == 0) {
        v_sign = -1.0;
    }
    let v = 0.5 * sqrt(v_num / denom_f) * v_sign;
    return vec2<f32>(u, v);
}

// Rotates one color channel's band-L2 coefficient vector by the 5x5 matrix
// (stored row-major as 25 scalars), matching the reference `rotate_l2`.
fn rotate_l2_channel(
    l2: ptr<function, array<f32, 25>>,
    c0: f32,
    c1: f32,
    c2: f32,
    c3: f32,
    c4: f32,
) -> array<f32, 5> {
    var out: array<f32, 5>;
    for (var row = 0u; row < 5u; row = row + 1u) {
        out[row] = (*l2)[row * 5u + 0u] * c0
            + (*l2)[row * 5u + 1u] * c1
            + (*l2)[row * 5u + 2u] * c2
            + (*l2)[row * 5u + 3u] * c3
            + (*l2)[row * 5u + 4u] * c4;
    }
    return out;
}

@compute @workgroup_size(64)
fn rotate_probes(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.count) {
        return;
    }
    let probe = inputs[idx];

    var r1 = quat_to_mat3(probe.quat);

    // Rotate the three RGB channels of the band-L1 payload independently.
    let lr = rotate_l1_channel(&r1, probe.l1_0.x, probe.l1_1.x, probe.l1_2.x);
    let lg = rotate_l1_channel(&r1, probe.l1_0.y, probe.l1_1.y, probe.l1_2.y);
    let lb = rotate_l1_channel(&r1, probe.l1_0.z, probe.l1_1.z, probe.l1_2.z);

    // Build the 5x5 band-L2 rotation matrix in the m = -2..=2 row/column order.
    var l2mat: array<f32, 25>;
    for (var m = -2; m <= 2; m = m + 1) {
        for (var n = -2; n <= 2; n = n + 1) {
            let uv = uv_coeff(m, n);
            let value = uv.x * u_term(&r1, m, n) + uv.y * v_term(&r1, m, n);
            let mi = u32(m + 2);
            let ni = u32(n + 2);
            l2mat[mi * 5u + ni] = value;
        }
    }

    // Rotate the three RGB channels of the band-L2 payload independently.
    let qr = rotate_l2_channel(&l2mat, probe.l2_0.x, probe.l2_1.x, probe.l2_2.x, probe.l2_3.x, probe.l2_4.x);
    let qg = rotate_l2_channel(&l2mat, probe.l2_0.y, probe.l2_1.y, probe.l2_2.y, probe.l2_3.y, probe.l2_4.y);
    let qb = rotate_l2_channel(&l2mat, probe.l2_0.z, probe.l2_1.z, probe.l2_2.z, probe.l2_3.z, probe.l2_4.z);

    var out: ProbeOut;
    // Band-L0 (DC) term is rotation-invariant: pass through verbatim.
    out.l0 = probe.l0;
    out.l1_0 = vec4<f32>(lr.x, lg.x, lb.x, 0.0);
    out.l1_1 = vec4<f32>(lr.y, lg.y, lb.y, 0.0);
    out.l1_2 = vec4<f32>(lr.z, lg.z, lb.z, 0.0);
    out.l2_0 = vec4<f32>(qr[0], qg[0], qb[0], 0.0);
    out.l2_1 = vec4<f32>(qr[1], qg[1], qb[1], 0.0);
    out.l2_2 = vec4<f32>(qr[2], qg[2], qb[2], 0.0);
    out.l2_3 = vec4<f32>(qr[3], qg[3], qb[3], 0.0);
    out.l2_4 = vec4<f32>(qr[4], qg[4], qb[4], 0.0);
    outputs[idx] = out;
}
"#;

/// Number of band-`L2` coefficients, ordered `(Y(2,-2)..Y(2,2))`.
pub const L2_COEFF_COUNT: usize = 5;

/// One probe's rotation request: a rotation quaternion plus the band-`L0`/`L1`
/// `RGB` payload and the band-`L2` `RGB` payload to re-express in the rotated
/// frame.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ShRotationProbe {
    /// Rotation quaternion `(x, y, z, w)`; need not be unit length (the robust
    /// conversion normalizes it, and the zero quaternion acts as identity).
    pub quat: [f32; 4],
    /// The band-`L0` (`DC`) term plus the three band-`L1` `RGB` coefficients in
    /// `(y, z, x)` order; `l1[k]` is the `RGB` triple of the `k`-th coefficient.
    pub l1: ShL1Rgb,
    /// The five band-`L2` `RGB` coefficients in `(Y(2,-2)..Y(2,2))` order;
    /// `l2[k]` is the `RGB` triple of the `k`-th coefficient.
    pub l2: [[f32; 3]; L2_COEFF_COUNT],
}

/// One probe's rotated result: the (unchanged) band-`L0` term with the rotated
/// band-`L1` coefficients, plus the rotated band-`L2` coefficients.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ShRotationResult {
    /// The band-`L0` (`DC`) term (unchanged) and the rotated band-`L1` `RGB`
    /// coefficients in `(y, z, x)` order.
    pub l1: ShL1Rgb,
    /// The rotated band-`L2` `RGB` coefficients in `(Y(2,-2)..Y(2,2))` order.
    pub l2: [[f32; 3]; L2_COEFF_COUNT],
}

/// Uniform parameters for one rotation dispatch. `repr(C)` `std430` layout
/// matching `Params` in [`SPHERICAL_HARMONICS_ROTATE_WGSL`]: one count word plus
/// three pad words — `16` bytes with no interior padding.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Params {
    /// Number of probes.
    count: u32,
    /// Padding word.
    pad0: u32,
    /// Padding word.
    pad1: u32,
    /// Padding word.
    pad2: u32,
}

/// One input probe as uploaded. `160`-byte `std430` stride matching `ProbeIn`
/// in the shader: the quaternion, then one padded `vec4` slot per `RGB` triple
/// (band `L0`, three band-`L1`, five band-`L2`).
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuProbeIn {
    /// Rotation quaternion `(x, y, z, w)`.
    quat: [f32; 4],
    /// Band-`L0` `RGB` term plus a pad word.
    l0: [f32; 4],
    /// First band-`L1` `RGB` coefficient plus a pad word.
    l1_0: [f32; 4],
    /// Second band-`L1` `RGB` coefficient plus a pad word.
    l1_1: [f32; 4],
    /// Third band-`L1` `RGB` coefficient plus a pad word.
    l1_2: [f32; 4],
    /// First band-`L2` `RGB` coefficient plus a pad word.
    l2_0: [f32; 4],
    /// Second band-`L2` `RGB` coefficient plus a pad word.
    l2_1: [f32; 4],
    /// Third band-`L2` `RGB` coefficient plus a pad word.
    l2_2: [f32; 4],
    /// Fourth band-`L2` `RGB` coefficient plus a pad word.
    l2_3: [f32; 4],
    /// Fifth band-`L2` `RGB` coefficient plus a pad word.
    l2_4: [f32; 4],
}

/// One output probe as read back. `144`-byte `std430` stride matching
/// `ProbeOut` in the shader: the passthrough band-`L0` term, three rotated
/// band-`L1` coefficients, and five rotated band-`L2` coefficients, each triple
/// in one padded `vec4` slot.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuProbeOut {
    /// Band-`L0` `RGB` term plus a pad word.
    l0: [f32; 4],
    /// First rotated band-`L1` `RGB` coefficient plus a pad word.
    l1_0: [f32; 4],
    /// Second rotated band-`L1` `RGB` coefficient plus a pad word.
    l1_1: [f32; 4],
    /// Third rotated band-`L1` `RGB` coefficient plus a pad word.
    l1_2: [f32; 4],
    /// First rotated band-`L2` `RGB` coefficient plus a pad word.
    l2_0: [f32; 4],
    /// Second rotated band-`L2` `RGB` coefficient plus a pad word.
    l2_1: [f32; 4],
    /// Third rotated band-`L2` `RGB` coefficient plus a pad word.
    l2_2: [f32; 4],
    /// Fourth rotated band-`L2` `RGB` coefficient plus a pad word.
    l2_3: [f32; 4],
    /// Fifth rotated band-`L2` `RGB` coefficient plus a pad word.
    l2_4: [f32; 4],
}

impl GpuProbeIn {
    /// Packs a host [`ShRotationProbe`] into the padded `vec4` storage layout.
    fn from_probe(p: &ShRotationProbe) -> GpuProbeIn {
        let l0 = [p.l1.l0[0], p.l1.l0[1], p.l1.l0[2], 0.0];
        let pad3 = |c: [f32; 3]| [c[0], c[1], c[2], 0.0];
        GpuProbeIn {
            quat: p.quat,
            l0,
            l1_0: pad3(p.l1.l1[0]),
            l1_1: pad3(p.l1.l1[1]),
            l1_2: pad3(p.l1.l1[2]),
            l2_0: pad3(p.l2[0]),
            l2_1: pad3(p.l2[1]),
            l2_2: pad3(p.l2[2]),
            l2_3: pad3(p.l2[3]),
            l2_4: pad3(p.l2[4]),
        }
    }
}

impl GpuProbeOut {
    /// Unpacks a read-back record into a host [`ShRotationResult`], dropping the
    /// per-slot pad word.
    fn to_result(self) -> ShRotationResult {
        let drop_pad = |v: [f32; 4]| [v[0], v[1], v[2]];
        ShRotationResult {
            l1: ShL1Rgb {
                l0: drop_pad(self.l0),
                l1: [drop_pad(self.l1_0), drop_pad(self.l1_1), drop_pad(self.l1_2)],
            },
            l2: [
                drop_pad(self.l2_0),
                drop_pad(self.l2_1),
                drop_pad(self.l2_2),
                drop_pad(self.l2_3),
                drop_pad(self.l2_4),
            ],
        }
    }
}

/// A compiled, reusable `SH`-rotation pipeline.
pub struct GpuSphericalHarmonicsRotate {
    #[expect(
        dead_code,
        reason = "kept alive so the pipeline it produced stays valid"
    )]
    module: ShaderModule,
    layout: BindGroupLayout,
    pipeline: ComputePipeline,
}

impl GpuSphericalHarmonicsRotate {
    /// Compiles the `SH`-rotation kernel on `ctx`.
    ///
    /// The kernel uses only the portable core-`WGSL` subset, so no optional
    /// device feature is required.
    #[must_use]
    pub fn new(ctx: &GpuContext) -> GpuSphericalHarmonicsRotate {
        let device = ctx.device();
        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("prism_volumetric_spherical_harmonics_rotate"),
            source: ShaderSource::Wgsl(SPHERICAL_HARMONICS_ROTATE_WGSL.into()),
        });
        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("prism_volumetric_spherical_harmonics_rotate_layout"),
            entries: &[
                buffer_entry(0, BufferBindingType::Uniform),
                buffer_entry(1, BufferBindingType::Storage { read_only: true }),
                buffer_entry(2, BufferBindingType::Storage { read_only: false }),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("prism_volumetric_spherical_harmonics_rotate_pipeline_layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("prism_volumetric_spherical_harmonics_rotate_pipeline"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("rotate_probes"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });
        GpuSphericalHarmonicsRotate {
            module,
            layout,
            pipeline,
        }
    }

    /// Rotates every probe in `probes`, returning one [`ShRotationResult`] per
    /// input probe in the same order.
    ///
    /// For each probe the returned `l1` equals
    /// [`ShL1Rgb::rotate`](prism_render_architecture::particle::spherical_harmonics_rotate::ShL1Rgb::rotate)
    /// applied to the matrix from
    /// [`rotation_matrix_from_quat`](prism_render_architecture::particle::spherical_harmonics_rotate::rotation_matrix_from_quat),
    /// and each `RGB` channel of the returned `l2` equals
    /// [`rotate_l2`](prism_render_architecture::particle::spherical_harmonics_rotate::rotate_l2)
    /// on that channel, to within the tolerance documented on this module. An
    /// empty input yields an empty result — a storage buffer cannot be
    /// zero-sized, so it is handled by an early return.
    #[must_use]
    pub fn eval(&self, ctx: &GpuContext, probes: &[ShRotationProbe]) -> Vec<ShRotationResult> {
        let count = probes.len();
        if count == 0 {
            return Vec::new();
        }
        let device = ctx.device();

        let inputs: Vec<GpuProbeIn> = probes.iter().map(GpuProbeIn::from_probe).collect();

        let gpu_params = Params {
            count: count as u32,
            pad0: 0,
            pad1: 0,
            pad2: 0,
        };

        let out_bytes = (count as u64) * (size_of::<GpuProbeOut>() as u64);

        let params_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_spherical_harmonics_rotate_params"),
            contents: bytemuck::bytes_of(&gpu_params),
            usage: BufferUsages::UNIFORM,
        });
        let inputs_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("prism_volumetric_spherical_harmonics_rotate_inputs"),
            contents: bytemuck::cast_slice(&inputs),
            usage: BufferUsages::STORAGE,
        });
        let results_buf = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_spherical_harmonics_rotate_results"),
            size: out_bytes,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let results_stage = device.create_buffer(&BufferDescriptor {
            label: Some("prism_volumetric_spherical_harmonics_rotate_results_stage"),
            size: out_bytes,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("prism_volumetric_spherical_harmonics_rotate_bind_group"),
            layout: &self.layout,
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: params_buf.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: inputs_buf.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 2,
                    resource: results_buf.as_entire_binding(),
                },
            ],
        });

        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("prism_volumetric_spherical_harmonics_rotate_encoder"),
        });
        {
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("prism_volumetric_spherical_harmonics_rotate_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            // One thread per probe, in workgroups of 64 (the kernel's size).
            let groups = (count as u32).div_ceil(64);
            pass.dispatch_workgroups(groups, 1, 1);
        }
        encoder.copy_buffer_to_buffer(&results_buf, 0, &results_stage, 0, out_bytes);
        ctx.queue().submit([encoder.finish()]);

        results_stage.slice(..).map_async(MapMode::Read, |_| {});
        ctx.wait();
        let view = results_stage
            .slice(..)
            .get_mapped_range()
            .expect("mapped readback range should be available after poll");
        let gpu_results = bytemuck::cast_slice::<u8, GpuProbeOut>(&view).to_vec();
        drop(view);
        results_stage.unmap();
        debug_assert_eq!(gpu_results.len(), count);

        gpu_results.into_iter().map(GpuProbeOut::to_result).collect()
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
