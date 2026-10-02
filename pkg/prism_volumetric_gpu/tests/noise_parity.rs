//! Real-device parity for the procedural-noise twin:
//! [`GpuNoise`](prism_volumetric_gpu::noise::GpuNoise) must reproduce the `CPU`
//! golden [`noise`](prism_render_architecture::particle::noise) across the
//! integer lattice hash
//! ([`hash_lattice`](prism_render_architecture::particle::noise::hash_lattice)),
//! the selected gradient
//! ([`lattice_gradient`](prism_render_architecture::particle::noise::lattice_gradient)),
//! trilinear value noise
//! ([`value_noise_3d`](prism_render_architecture::particle::noise::value_noise_3d)),
//! `Perlin`-style gradient noise
//! ([`gradient_noise_3d`](prism_render_architecture::particle::noise::gradient_noise_3d)),
//! the fractal-Brownian-motion sum
//! ([`fbm`](prism_render_architecture::particle::noise::fbm)), the single-octave
//! and multi-octave analytic curl flow
//! ([`curl_noise_3d`](prism_render_architecture::particle::noise::curl_noise_3d),
//! [`curl_noise_fbm`](prism_render_architecture::particle::noise::curl_noise_fbm)),
//! and the wrapped turbulence force
//! ([`turbulence_force`](prism_render_architecture::particle::noise::turbulence_force)).
//!
//! The fixtures walk the branches the golden unit tests call out: a canonical
//! off-lattice probe under the default `fBm` structure, the `octaves = 0`
//! degenerate field (`fbm` and `curl_noise_fbm` collapse to zero without a
//! `NaN`), single- and multi-octave counts (`1`, `2`, `4`, `8`), negative
//! sample positions and negative lattice cells (two's-complement hashing),
//! turbulence `frequency` / `amplitude` scaling, an empty-batch short-circuit,
//! and a large randomised batch. Every `f32` fixture is produced with a
//! host-side `u64` linear-congruential generator and integer / `floor` math
//! only, with rejection sampling that keeps each sample component away from the
//! integer lattice boundary, so no fixture uses a transcendental method and
//! none sits on a classification edge.
//!
//! The tests skip (returning early) when the host has no `wgpu` adapter, so the
//! suite stays green everywhere while still exercising the full
//! dispatch-and-readback on any real device such as an Apple `M`-series `GPU`.
//! The kernel is portable core-`WGSL`, so it needs no optional device feature.
//!
//! # Parity criterion
//!
//! The `hash_lattice` output is pure integer work, so `CPU` and `GPU` must agree
//! exactly: the comparison is an `assert_eq!` on the `u32`. The continuous
//! fields thread through the quintic fade, trilinear blends, the
//! amplitude-normalised octave sum and the central-difference curl, so they are
//! compared under tolerance (`abs_diff <= 1e-4` or `rel_diff <= 1e-3`,
//! `REL_FLOOR = 1e-6`), tight enough to catch a dropped octave, a swapped
//! gradient or a wrong stencil yet loose enough to admit a legal fused
//! multiply-add.
//!
//! Provenance: 孪生自本仓 `prism_render_architecture::particle::noise`；无第三方引擎源码或衍生代码。

use prism_render_architecture::particle::noise::{
    curl_noise_3d, curl_noise_fbm, fbm, gradient_noise_3d, hash_lattice, lattice_gradient,
    turbulence_force, value_noise_3d, FbmParams, TurbulenceParams,
};
use prism_render_architecture::particle::Vec3;
use prism_volumetric_gpu::noise::{
    GpuFbmParams, GpuNoise, GpuTurbulenceParams, NoiseQuery, NoiseResult,
};
use prism_volumetric_gpu::GpuContext;

/// Absolute tolerance for the continuous quantities.
const ABS_EPS: f32 = 1.0e-4;
/// Relative tolerance for the continuous quantities.
const REL_EPS: f32 = 1.0e-3;
/// Floor for the relative-tolerance denominator.
const REL_FLOOR: f32 = 1.0e-6;

/// Mixed absolute / relative tolerance comparison for one `f32` lane.
fn approx(a: f32, b: f32) -> bool {
    let diff = (a - b).abs();
    if diff <= ABS_EPS {
        return true;
    }
    let scale = a.abs().max(b.abs()).max(REL_FLOOR);
    diff / scale <= REL_EPS
}

/// Tolerant comparison of two `vec3` triples lane by lane.
fn approx3(a: [f32; 3], b: [f32; 3]) -> bool {
    approx(a[0], b[0]) && approx(a[1], b[1]) && approx(a[2], b[2])
}

/// Advances the host-side `u64` linear-congruential generator and returns its
/// new state. Only integer multiply / add appear, so no transcendental is used.
fn next_bits(state: &mut u64) -> u64 {
    *state = state
        .wrapping_mul(6_364_136_223_846_793_005)
        .wrapping_add(1_442_695_040_888_963_407);
    *state
}

/// Draws the next `f32` in `[0, 1)` from the generator.
fn lcg(state: &mut u64) -> f32 {
    let bits = (next_bits(state) >> 40) as u32;
    (bits & 0x00ff_ffff) as f32 / 16_777_216.0
}

/// Draws the next `u32` seed word from the generator.
fn next_u32(state: &mut u64) -> u32 {
    (next_bits(state) >> 32) as u32
}

/// True when `c`'s fractional part sits at least `0.1` away from either integer
/// boundary, so a `floor` split and the `CURL_EPS` half-step both stay inside
/// one lattice cell.
fn frac_safe(c: f32) -> bool {
    let frac = c - c.floor();
    (0.1..=0.9).contains(&frac)
}

/// Rejection-samples a position in `[-4, 4]^3` whose every component clears the
/// lattice-boundary margin via [`frac_safe`].
fn sample_pos(state: &mut u64) -> [f32; 3] {
    loop {
        let p = [
            lcg(state) * 8.0 - 4.0,
            lcg(state) * 8.0 - 4.0,
            lcg(state) * 8.0 - 4.0,
        ];
        if p.iter().copied().all(frac_safe) {
            return p;
        }
    }
}

/// Builds host-side `fBm` parameters.
fn fbm_params(octaves: u32, lacunarity: f32, gain: f32) -> GpuFbmParams {
    GpuFbmParams::new(octaves, lacunarity, gain)
}

/// Builds host-side turbulence parameters.
fn turb(frequency: f32, amplitude: f32, fbm: GpuFbmParams, seed: u32) -> GpuTurbulenceParams {
    GpuTurbulenceParams {
        frequency,
        amplitude,
        fbm,
        seed,
    }
}

/// Bridges a [`GpuFbmParams`] into the reference
/// [`FbmParams`](prism_render_architecture::particle::noise::FbmParams).
fn gold_fbm(p: GpuFbmParams) -> FbmParams {
    FbmParams::new(p.octaves, p.lacunarity, p.gain)
}

/// Bridges a [`GpuTurbulenceParams`] into the reference
/// [`TurbulenceParams`](prism_render_architecture::particle::noise::TurbulenceParams).
fn gold_turb(p: GpuTurbulenceParams) -> TurbulenceParams {
    TurbulenceParams {
        frequency: p.frequency,
        amplitude: p.amplitude,
        fbm: gold_fbm(p.fbm),
        seed: p.seed,
    }
}

/// Asserts one on-device [`NoiseResult`](prism_volumetric_gpu::noise::NoiseResult)
/// matches the `CPU` golden across every twinned field.
fn check(g: &NoiseResult, q: &NoiseQuery) {
    let pos = Vec3::new(q.pos[0], q.pos[1], q.pos[2]);
    let [ci, cj, ck] = q.cell;

    // Integer hash is bit-exact.
    assert_eq!(
        g.hash,
        hash_lattice(ci, cj, ck, q.seed),
        "hash mismatch at cell {:?} seed {}",
        q.cell,
        q.seed
    );

    // Selected lattice gradient.
    let lg = lattice_gradient(ci, cj, ck, q.seed);
    assert!(
        approx3(g.lattice_gradient, [lg.x, lg.y, lg.z]),
        "lattice_gradient mismatch: gpu {:?} vs cpu {:?}",
        g.lattice_gradient,
        [lg.x, lg.y, lg.z]
    );

    // Scalar value and gradient noise.
    let cpu_value = value_noise_3d(pos, q.seed);
    assert!(
        approx(g.value_noise, cpu_value),
        "value_noise mismatch: gpu {} vs cpu {cpu_value}",
        g.value_noise
    );
    let cpu_gradient = gradient_noise_3d(pos, q.seed);
    assert!(
        approx(g.gradient_noise, cpu_gradient),
        "gradient_noise mismatch: gpu {} vs cpu {cpu_gradient}",
        g.gradient_noise
    );

    // fBm sum with the query octave structure.
    let fb = gold_fbm(q.fbm);
    let cpu_fbm = fbm(pos, fb, q.seed);
    assert!(
        approx(g.fbm, cpu_fbm),
        "fbm mismatch: gpu {} vs cpu {cpu_fbm}",
        g.fbm
    );

    // Single-octave curl noise.
    let c3 = curl_noise_3d(pos, q.seed);
    assert!(
        approx3(g.curl_noise_3d, [c3.x, c3.y, c3.z]),
        "curl_noise_3d mismatch: gpu {:?} vs cpu {:?}",
        g.curl_noise_3d,
        [c3.x, c3.y, c3.z]
    );

    // Multi-octave curl noise with the query octave structure.
    let cf = curl_noise_fbm(pos, fb, q.seed);
    assert!(
        approx3(g.curl_noise_fbm, [cf.x, cf.y, cf.z]),
        "curl_noise_fbm mismatch: gpu {:?} vs cpu {:?}",
        g.curl_noise_fbm,
        [cf.x, cf.y, cf.z]
    );

    // Turbulence force under the turbulence parameters.
    let tf = turbulence_force(pos, gold_turb(q.turbulence));
    assert!(
        approx3(g.turbulence_force, [tf.x, tf.y, tf.z]),
        "turbulence_force mismatch: gpu {:?} vs cpu {:?}",
        g.turbulence_force,
        [tf.x, tf.y, tf.z]
    );
}

/// Dispatches a single query and checks every field against the golden.
fn assert_parity(gpu: &GpuNoise, ctx: &GpuContext, q: &NoiseQuery) {
    let got = gpu.eval(ctx, std::slice::from_ref(q));
    assert_eq!(got.len(), 1, "one result per query");
    check(&got[0], q);
}

#[test]
fn canonical_point_matches_every_field() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuNoise::new(&ctx);
    // An off-lattice probe with the cinematic default fBm / turbulence structure
    // exercises all eight fields at once.
    let q = NoiseQuery::new(
        [0.37, 1.23, -0.62],
        [3, -7, 11],
        0x0051_7E3D,
        GpuFbmParams::DEFAULT,
        GpuTurbulenceParams::DEFAULT,
    );
    assert_parity(&gpu, &ctx, &q);
}

#[test]
fn octave_counts_cover_zero_one_many() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuNoise::new(&ctx);
    // octaves = 0 collapses fbm / curl_noise_fbm to zero (no NaN); 1 reduces fbm
    // to plain gradient noise; 2 / 4 / 8 layer detail. All are <= MAX_OCTAVES.
    let pos = [0.44, -1.18, 2.31];
    let cell = [-2, 5, 9];
    let seed = 0x00AB_CDEF;
    let batch: Vec<NoiseQuery> = [0u32, 1, 2, 4, 8]
        .iter()
        .map(|&o| {
            NoiseQuery::new(
                pos,
                cell,
                seed,
                fbm_params(o, 2.0, 0.5),
                GpuTurbulenceParams::DEFAULT,
            )
        })
        .collect();
    let got = gpu.eval(&ctx, &batch);
    assert_eq!(got.len(), batch.len(), "one result per query");
    for (g, q) in got.iter().zip(batch.iter()) {
        check(g, q);
    }
}

#[test]
fn negative_positions_and_cells_match() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuNoise::new(&ctx);
    // Negative sample positions and negative lattice cells cross the
    // two's-complement hashing path and the negative-floor split.
    let batch = [
        NoiseQuery::new(
            [-3.63, -0.74, -2.21],
            [-5, 3, -11],
            0x1357_9BDF,
            fbm_params(3, 2.0, 0.5),
            GpuTurbulenceParams::DEFAULT,
        ),
        NoiseQuery::new(
            [-0.28, 2.66, -1.49],
            [0, 0, 0],
            0x2468_ACE0,
            fbm_params(4, 1.8, 0.55),
            turb(1.5, 0.75, fbm_params(2, 2.0, 0.5), 0x00C0_FFEE),
        ),
    ];
    let got = gpu.eval(&ctx, &batch);
    assert_eq!(got.len(), batch.len(), "one result per query");
    for (g, q) in got.iter().zip(batch.iter()) {
        check(g, q);
    }
}

#[test]
fn turbulence_frequency_and_amplitude_scale() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuNoise::new(&ctx);
    // Vary the turbulence frequency (sample scale) and amplitude (force scale),
    // including a zero amplitude that must yield an exactly zero force.
    let pos = [0.91, -0.43, 1.82];
    let cell = [1, 2, 3];
    let seed = 0x0F0F_0F0F;
    let batch = [
        NoiseQuery::new(
            pos,
            cell,
            seed,
            GpuFbmParams::DEFAULT,
            turb(2.0, 1.0, GpuFbmParams::DEFAULT, 17),
        ),
        NoiseQuery::new(
            pos,
            cell,
            seed,
            GpuFbmParams::DEFAULT,
            turb(0.5, 2.5, fbm_params(2, 2.2, 0.45), 29),
        ),
        NoiseQuery::new(
            pos,
            cell,
            seed,
            GpuFbmParams::DEFAULT,
            turb(1.0, 0.0, GpuFbmParams::DEFAULT, 17),
        ),
    ];
    let got = gpu.eval(&ctx, &batch);
    assert_eq!(got.len(), batch.len(), "one result per query");
    for (g, q) in got.iter().zip(batch.iter()) {
        check(g, q);
    }
}

#[test]
fn empty_batch_short_circuits() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuNoise::new(&ctx);
    // No dispatch is issued and the result vector is empty.
    assert!(gpu.eval(&ctx, &[]).is_empty());
}

#[test]
fn random_batch_matches_cpu() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuNoise::new(&ctx);

    let mut state: u64 = 0x9E37_79B9_7F4A_7C15;
    let mut queries: Vec<NoiseQuery> = Vec::with_capacity(192);
    while queries.len() < 192 {
        let pos = sample_pos(&mut state);
        // Lattice cells span the signed range, including negatives.
        let cell = [
            (lcg(&mut state) * 24.0 - 12.0) as i32,
            (lcg(&mut state) * 24.0 - 12.0) as i32,
            (lcg(&mut state) * 24.0 - 12.0) as i32,
        ];
        let seed = next_u32(&mut state);

        // fBm structure near the canonical lacunarity / gain, octaves 0..=5.
        let octaves = next_u32(&mut state) % 6;
        let lacunarity = 1.6 + lcg(&mut state) * 0.8;
        let gain = 0.35 + lcg(&mut state) * 0.3;
        let fbm = fbm_params(octaves, lacunarity, gain);

        // Independent turbulence structure.
        let turb_octaves = next_u32(&mut state) % 5;
        let turb_lacunarity = 1.6 + lcg(&mut state) * 0.8;
        let turb_gain = 0.35 + lcg(&mut state) * 0.3;
        let frequency = 0.5 + lcg(&mut state) * 1.5;
        let amplitude = lcg(&mut state) * 2.0;
        let turbulence = turb(
            frequency,
            amplitude,
            fbm_params(turb_octaves, turb_lacunarity, turb_gain),
            next_u32(&mut state),
        );

        queries.push(NoiseQuery::new(pos, cell, seed, fbm, turbulence));
    }

    let got = gpu.eval(&ctx, &queries);
    assert_eq!(got.len(), queries.len(), "one result per query");
    for (g, q) in got.iter().zip(queries.iter()) {
        check(g, q);
    }
}
