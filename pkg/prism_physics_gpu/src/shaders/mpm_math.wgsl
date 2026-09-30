// Shared pure-function math library for the GPU MLS-MPM kernels.
//
// This module holds only side-effect-free helpers (no bindings, no entry
// points): the deterministic exponential, the quadratic B-spline weights, the
// sqrt-based 3x3 symmetric eigen-solve and signed SVD, the fixed-corotated
// stress, and the snow return-mapping. Each helper is a line-for-line device
// mirror of its CPU golden twin in `prism_physics_core::mpm` so a passing
// real-device parity test is direct evidence the ported arithmetic matches the
// reference rather than merely that the WGSL compiles.
//
// Column-major convention: glam `Mat3` and WGSL `mat3x3<f32>` are both
// column-major, so `m[col]` is a column vec3 and element `(row, col)` is
// `m[col][row]`. The helpers below follow that indexing exactly.
//
// The host concatenates this file ahead of a stage module (bindings + entry
// points) with `format!`, because WGSL has no include directive.
//
// Provenance: the MLS-MPM quadratic B-spline weights (Hu et al. 2018; Steffen
// et al. 2008), the fixed-corotated energy and snow return-mapping (Stomakhin
// et al. 2013), the cyclic Jacobi eigen-solve and sqrt-based rotations (Golub &
// Van Loan, *Matrix Computations*), and the range-reduced Taylor exponential
// are all standard, publicly documented techniques. No Unreal Engine source or
// derived code.

// log2(e) as an f32 literal; matches `core::f32::consts::LOG2_E`.
const MPM_LOG2_E: f32 = 1.4426950408889634;
// ln(2) as an f32 literal; matches `core::f32::consts::LN_2`.
const MPM_LN_2: f32 = 0.6931471805599453;
// Cyclic Jacobi sweep count; matches the CPU `JACOBI_SWEEPS`.
const MPM_JACOBI_SWEEPS: i32 = 8;

// The symmetric eigen-decomposition result: eigenvector columns and the
// matching eigenvalues, both sorted so the eigenvalues descend.
struct MpmEigen {
    v: mat3x3<f32>,
    vals: vec3<f32>,
};

// The signed SVD `F = U diag(sigma) Vᵀ` with proper rotations `U`, `V` and the
// smallest singular value carrying `sign(det F)`.
struct MpmSvd3 {
    u: mat3x3<f32>,
    sigma: vec3<f32>,
    v: mat3x3<f32>,
};

// The snow return-mapping result: corrected elastic deformation and new `Jp`.
struct MpmPlasticUpdate {
    deformation: mat3x3<f32>,
    plastic_det: f32,
};

// Per-axis quadratic B-spline weights and the integer base node / fractional
// offset for one particle. `wx`/`wy`/`wz` hold the three offset weights for the
// x/y/z axes respectively.
struct MpmQuadWeights {
    base: vec3<i32>,
    fx: vec3<f32>,
    wx: vec3<f32>,
    wy: vec3<f32>,
    wz: vec3<f32>,
};

// Rounds half away from zero, mirroring Rust `f32::round` (WGSL's builtin
// `round` is ties-to-even, which would break exp parity).
fn mpm_round_half_away(y: f32) -> f32 {
    if (y >= 0.0) {
        return floor(y + 0.5);
    }
    return -floor(-y + 0.5);
}

// Returns `2^n` by repeated multiplication (deterministic, no `pow`).
fn mpm_pow2_int(n: i32) -> f32 {
    var result: f32 = 1.0;
    var k: i32 = abs(n);
    loop {
        if (k <= 0) { break; }
        result = result * 2.0;
        k = k - 1;
    }
    if (n < 0) {
        return 1.0 / result;
    }
    return result;
}

// Deterministic `exp(x)` via base-two range reduction plus a 7-term Taylor
// series; mirrors `expf::exp_stable` (the engine forbids the intrinsic `exp`
// for cross-platform determinism).
fn mpm_exp_stable(x: f32) -> f32 {
    let y = x * MPM_LOG2_E;
    let n = mpm_round_half_away(y);
    let r = (y - n) * MPM_LN_2;
    let series = 1.0
        + r * (1.0
            + r * (0.5
                + r * ((1.0 / 6.0)
                    + r * ((1.0 / 24.0) + r * ((1.0 / 120.0) + r * (1.0 / 720.0))))));
    return mpm_pow2_int(i32(n)) * series;
}

// Returns element `(row, col)` of a column-major matrix.
fn mpm_at3(m: mat3x3<f32>, row: i32, col: i32) -> f32 {
    return m[col][row];
}

// Builds the Givens rotation in the `(p, q)` plane so that `Jᵀ A J` zeroes the
// `(p, q)` entry. Element `(row, col)` is `M[col][row]`, so the CPU assignments
// `(p,p)=c (q,q)=c (row p,col q)=s (row q,col p)=-s` become the column-index
// forms below.
fn mpm_givens3(p: i32, q: i32, c: f32, s: f32) -> mat3x3<f32> {
    var m = mat3x3<f32>(
        vec3<f32>(1.0, 0.0, 0.0),
        vec3<f32>(0.0, 1.0, 0.0),
        vec3<f32>(0.0, 0.0, 1.0),
    );
    m[p][p] = c;
    m[q][q] = c;
    m[q][p] = s;
    m[p][q] = -s;
    return m;
}

// Reorders the three eigenpairs into descending eigenvalue order via the same
// insertion sort the CPU twin uses, carrying the eigenvector columns along.
fn mpm_sort_descending(v: mat3x3<f32>, eig: vec3<f32>) -> MpmEigen {
    var cols = array<vec3<f32>, 3>(v[0], v[1], v[2]);
    var vals = array<f32, 3>(eig.x, eig.y, eig.z);
    for (var i: i32 = 1; i < 3; i = i + 1) {
        var j: i32 = i;
        loop {
            if (!(j > 0 && vals[j - 1] < vals[j])) { break; }
            let tv = vals[j - 1];
            vals[j - 1] = vals[j];
            vals[j] = tv;
            let tc = cols[j - 1];
            cols[j - 1] = cols[j];
            cols[j] = tc;
            j = j - 1;
        }
    }
    var out: MpmEigen;
    out.v = mat3x3<f32>(cols[0], cols[1], cols[2]);
    out.vals = vec3<f32>(vals[0], vals[1], vals[2]);
    return out;
}

// Symmetric eigen-decomposition `A = V Λ Vᵀ` via a cyclic Jacobi sweep, using
// only `sqrt` (never a trig call). Mirrors `svd::symmetric_eigen`.
fn mpm_symmetric_eigen(a_in: mat3x3<f32>) -> MpmEigen {
    var a = a_in;
    var v = mat3x3<f32>(
        vec3<f32>(1.0, 0.0, 0.0),
        vec3<f32>(0.0, 1.0, 0.0),
        vec3<f32>(0.0, 0.0, 1.0),
    );
    var ps = array<i32, 3>(0, 0, 1);
    var qs = array<i32, 3>(1, 2, 2);
    for (var sweep: i32 = 0; sweep < MPM_JACOBI_SWEEPS; sweep = sweep + 1) {
        for (var idx: i32 = 0; idx < 3; idx = idx + 1) {
            let p = ps[idx];
            let q = qs[idx];
            let apq = mpm_at3(a, p, q);
            if (abs(apq) < 1.0e-20) {
                continue;
            }
            let app = mpm_at3(a, p, p);
            let aqq = mpm_at3(a, q, q);
            let tau = (aqq - app) / (2.0 * apq);
            var t: f32;
            if (tau >= 0.0) {
                t = 1.0 / (tau + sqrt(1.0 + tau * tau));
            } else {
                t = -1.0 / (-tau + sqrt(1.0 + tau * tau));
            }
            let c = 1.0 / sqrt(1.0 + t * t);
            let s = t * c;
            let j = mpm_givens3(p, q, c, s);
            a = transpose(j) * a * j;
            v = v * j;
        }
    }
    let eig = vec3<f32>(mpm_at3(a, 0, 0), mpm_at3(a, 1, 1), mpm_at3(a, 2, 2));
    return mpm_sort_descending(v, eig);
}

// Returns any unit vector orthogonal to `n`; mirrors `svd::orthogonal_unit`.
fn mpm_orthogonal_unit(n: vec3<f32>) -> vec3<f32> {
    var a: vec3<f32>;
    if (abs(n.x) <= abs(n.y) && abs(n.x) <= abs(n.z)) {
        a = vec3<f32>(1.0, 0.0, 0.0);
    } else if (abs(n.y) <= abs(n.z)) {
        a = vec3<f32>(0.0, 1.0, 0.0);
    } else {
        a = vec3<f32>(0.0, 0.0, 1.0);
    }
    let c = cross(n, a);
    let len = length(c);
    if (len > 1.0e-12) {
        return c / len;
    }
    return vec3<f32>(1.0, 0.0, 0.0);
}

// Normalises `v`, returning zero for a zero-length input; mirrors glam
// `normalize_or_zero` for the well-conditioned inputs the SVD produces.
fn mpm_normalize_or_zero(v: vec3<f32>) -> vec3<f32> {
    let len = length(v);
    if (len > 0.0) {
        return v / len;
    }
    return vec3<f32>(0.0, 0.0, 0.0);
}

// Signed 3x3 SVD `F = U diag(sigma) Vᵀ`; mirrors `svd::svd3`.
fn mpm_svd3(f: mat3x3<f32>) -> MpmSvd3 {
    let ata = transpose(f) * f;
    let e = mpm_symmetric_eigen(ata);
    var v = e.v;
    if (determinant(v) < 0.0) {
        v = mat3x3<f32>(v[0], v[1], -v[2]);
    }
    var sigma = vec3<f32>(
        sqrt(max(e.vals.x, 0.0)),
        sqrt(max(e.vals.y, 0.0)),
        sqrt(max(e.vals.z, 0.0)),
    );
    let b = f * v;
    var cols_b = array<vec3<f32>, 3>(b[0], b[1], b[2]);
    var sig = array<f32, 3>(sigma.x, sigma.y, sigma.z);
    let tol = 1.0e-9 * max(sigma.x, 1.0);
    var valid = array<bool, 3>(false, false, false);
    var u_cols = array<vec3<f32>, 3>(
        vec3<f32>(0.0, 0.0, 0.0),
        vec3<f32>(0.0, 0.0, 0.0),
        vec3<f32>(0.0, 0.0, 0.0),
    );
    for (var i: i32 = 0; i < 3; i = i + 1) {
        if (sig[i] > tol) {
            u_cols[i] = cols_b[i] / sig[i];
            valid[i] = true;
        }
    }
    if (valid[0] && valid[1] && valid[2]) {
        // All columns well-defined; nothing to complete.
    } else if (valid[0] && valid[1] && !valid[2]) {
        var c2 = mpm_normalize_or_zero(cross(u_cols[0], u_cols[1]));
        if (c2.x == 0.0 && c2.y == 0.0 && c2.z == 0.0) {
            c2 = mpm_orthogonal_unit(u_cols[0]);
        }
        u_cols[2] = c2;
    } else if (valid[0] && !valid[1] && !valid[2]) {
        let e1 = mpm_orthogonal_unit(u_cols[0]);
        let e2 = mpm_normalize_or_zero(cross(u_cols[0], e1));
        u_cols[1] = e1;
        u_cols[2] = e2;
    } else {
        u_cols[0] = vec3<f32>(1.0, 0.0, 0.0);
        u_cols[1] = vec3<f32>(0.0, 1.0, 0.0);
        u_cols[2] = vec3<f32>(0.0, 0.0, 1.0);
    }
    var u = mat3x3<f32>(u_cols[0], u_cols[1], u_cols[2]);
    if (determinant(u) < 0.0) {
        u = mat3x3<f32>(u[0], u[1], -u[2]);
        sigma.z = -sigma.z;
    }
    var out: MpmSvd3;
    out.u = u;
    out.sigma = sigma;
    out.v = v;
    return out;
}

// Polar-decomposition rotation `R = U Vᵀ`; mirrors `svd::polar_rotation`.
fn mpm_polar_rotation(f: mat3x3<f32>) -> mat3x3<f32> {
    let s = mpm_svd3(f);
    return s.u * transpose(s.v);
}

// Fixed-corotated `P Fᵀ = 2μ(F − R)Fᵀ + λ(J − 1) J I`; mirrors
// `constitutive::corotated_pf`.
fn mpm_corotated_pf(f: mat3x3<f32>, mu: f32, lambda: f32) -> mat3x3<f32> {
    let r = mpm_polar_rotation(f);
    let j = determinant(f);
    var m = ((f - r) * transpose(f)) * (2.0 * mu);
    let term_vol = lambda * (j - 1.0) * j;
    m[0][0] = m[0][0] + term_vol;
    m[1][1] = m[1][1] + term_vol;
    m[2][2] = m[2][2] + term_vol;
    return m;
}

// Cofactor matrix `cof(F) = J F⁻ᵀ` from column cross products; mirrors
// `constitutive::cofactor`.
fn mpm_cofactor3(f: mat3x3<f32>) -> mat3x3<f32> {
    let c0 = cross(f[1], f[2]);
    let c1 = cross(f[2], f[0]);
    let c2 = cross(f[0], f[1]);
    return mat3x3<f32>(c0, c1, c2);
}

// Full first Piola–Kirchhoff stress `P = 2μ(F − R) + λ(J − 1) cof(F)`; mirrors
// `constitutive::corotated_piola`.
fn mpm_corotated_piola(f: mat3x3<f32>, mu: f32, lambda: f32) -> mat3x3<f32> {
    let r = mpm_polar_rotation(f);
    let j = determinant(f);
    let cof = mpm_cofactor3(f);
    return (f - r) * (2.0 * mu) + cof * (lambda * (j - 1.0));
}

// Hardening multiplier `exp(ξ(1 − Jp))`, exactly `1.0` when disabled; mirrors
// `constitutive::hardening_factor`.
fn mpm_hardening_factor(hardening: f32, plastic_det: f32) -> f32 {
    if (hardening == 0.0) {
        return 1.0;
    }
    return mpm_exp_stable(hardening * (1.0 - plastic_det));
}

// Snow return-mapping: clamp singular values into `[1−θc, 1+θs]` and fold the
// clamped-off volume into `Jp`; mirrors `constitutive::snow_return_mapping`.
fn mpm_snow_return_mapping(
    f_trial: mat3x3<f32>,
    prev_jp: f32,
    theta_c: f32,
    theta_s: f32,
) -> MpmPlasticUpdate {
    let svd = mpm_svd3(f_trial);
    let lo = 1.0 - theta_c;
    let hi = 1.0 + theta_s;
    let clamped = vec3<f32>(
        clamp(svd.sigma.x, lo, hi),
        clamp(svd.sigma.y, lo, hi),
        clamp(svd.sigma.z, lo, hi),
    );
    let sig_elastic = mat3x3<f32>(
        vec3<f32>(clamped.x, 0.0, 0.0),
        vec3<f32>(0.0, clamped.y, 0.0),
        vec3<f32>(0.0, 0.0, clamped.z),
    );
    let f_elastic = svd.u * sig_elastic * transpose(svd.v);
    let j_total = svd.sigma.x * svd.sigma.y * svd.sigma.z;
    let j_elastic = clamped.x * clamped.y * clamped.z;
    var jp_new: f32;
    if (abs(j_elastic) > 1.0e-12) {
        jp_new = prev_jp * j_total / j_elastic;
    } else {
        jp_new = prev_jp;
    }
    var out: MpmPlasticUpdate;
    out.deformation = f_elastic;
    out.plastic_det = jp_new;
    return out;
}

// Outer product `a ⊗ b = a bᵀ`; mirrors `transfer::outer`.
fn mpm_outer(a: vec3<f32>, b: vec3<f32>) -> mat3x3<f32> {
    return mat3x3<f32>(a * b.x, a * b.y, a * b.z);
}

// Per-axis quadratic B-spline weights for one axis offset `f ∈ [0.5, 1.5]`:
// `(0.5·(1.5−f)², 0.75−(f−1)², 0.5·(f−0.5)²)`.
fn mpm_axis_weights(f: f32) -> vec3<f32> {
    let a = 1.5 - f;
    let b = f - 1.0;
    let c = f - 0.5;
    return vec3<f32>(0.5 * a * a, 0.75 - b * b, 0.5 * c * c);
}

// Computes the quadratic B-spline weights, base node, and fractional offset for
// a particle at world position `x`; mirrors `weights::QuadraticWeights::new`.
fn mpm_quad_weights(x: vec3<f32>, origin: vec3<f32>, dx: f32) -> MpmQuadWeights {
    let inv_dx = 1.0 / dx;
    let cell = (x - origin) * inv_dx;
    let base = vec3<i32>(
        i32(floor(cell.x - 0.5)),
        i32(floor(cell.y - 0.5)),
        i32(floor(cell.z - 0.5)),
    );
    let fx = vec3<f32>(
        cell.x - f32(base.x),
        cell.y - f32(base.y),
        cell.z - f32(base.z),
    );
    var out: MpmQuadWeights;
    out.base = base;
    out.fx = fx;
    out.wx = mpm_axis_weights(fx.x);
    out.wy = mpm_axis_weights(fx.y);
    out.wz = mpm_axis_weights(fx.z);
    return out;
}
