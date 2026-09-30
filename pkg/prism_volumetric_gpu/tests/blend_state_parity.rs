//! Real-device parity for the sky-state blend twin:
//! [`GpuBlendState`] must reproduce the `CPU` golden
//! [`blend_state`](prism_render_architecture::volumetric::weather::blend_state)
//! across every state pair and a deterministic sweep of the transition
//! parameter, including the exact endpoints and out-of-range `t`.
//!
//! The tests skip (with a printed notice) when the host has no `wgpu` adapter,
//! so the suite stays green everywhere while still exercising the full
//! dispatch-and-readback on any real device. The kernel is portable core-WGSL,
//! so it needs no optional device feature.
//!
//! # Parity criterion
//!
//! The kernel mirrors `math::smoothstep` bit for bit, so agreement is asserted
//! within a tight absolute tolerance. The suite also checks the invariants that
//! make the blend correct: the endpoints are exact (`t = 0` yields the source
//! target, `t = 1` the destination), and the result is monotone in `t`.
//!
//! Provenance: standard smoothstep coverage blend / weather state machine; no
//! Unreal Engine source or derived code.

use prism_render_architecture::volumetric::weather::{blend_state, SkyState};
use prism_volumetric_gpu::{BlendStateQuery, GpuBlendState, GpuContext};

/// Absolute tolerance: the mirrored smoothstep differs from the `CPU` golden's
/// own only in the last few ULPs, well under this.
const TOL: f32 = 1e-6;

const STATES: [SkyState; 4] = [
    SkyState::Clear,
    SkyState::Fair,
    SkyState::Overcast,
    SkyState::Storm,
];

#[test]
#[expect(
    clippy::print_stderr,
    reason = "the skip notice must reach the test log on hosts without a wgpu adapter"
)]
fn gpu_blend_state_matches_cpu_golden() {
    let Some(ctx) = GpuContext::try_headless() else {
        eprintln!("skipping blend-state parity: no wgpu adapter on this host");
        return;
    };
    let gpu_kernel = GpuBlendState::new(&ctx);

    // Every ordered state pair, over a sweep of `t` that includes the exact
    // endpoints and out-of-range values (which saturate).
    let mut queries: Vec<BlendStateQuery> = Vec::new();
    for &from in &STATES {
        for &to in &STATES {
            for k in -2..=12 {
                let t = (k as f32) / 10.0;
                queries.push(BlendStateQuery { from, to, t });
            }
        }
    }

    let gpu = gpu_kernel.eval(&ctx, &queries);
    assert_eq!(gpu.len(), queries.len(), "one result per query");

    for (i, q) in queries.iter().enumerate() {
        let exp = blend_state(q.from, q.to, q.t);
        assert!(
            (gpu[i] - exp).abs() <= TOL,
            "blend-state mismatch for query {i} (from {:?}, to {:?}, t {}): \
             gpu {}, cpu {exp}, |diff| {}",
            q.from,
            q.to,
            q.t,
            gpu[i],
            (gpu[i] - exp).abs()
        );
    }

    // Endpoints are exact for a representative pair: t=0 -> from target,
    // t=1 -> to target.
    let ends = gpu_kernel.eval(
        &ctx,
        &[
            BlendStateQuery {
                from: SkyState::Clear,
                to: SkyState::Storm,
                t: 0.0,
            },
            BlendStateQuery {
                from: SkyState::Clear,
                to: SkyState::Storm,
                t: 1.0,
            },
        ],
    );
    assert!(
        (ends[0] - blend_state(SkyState::Clear, SkyState::Storm, 0.0)).abs() <= TOL,
        "t=0 must yield the source target"
    );
    assert!(
        (ends[1] - blend_state(SkyState::Clear, SkyState::Storm, 1.0)).abs() <= TOL,
        "t=1 must yield the destination target"
    );

    // Monotone in `t` for an increasing transition (Clear -> Storm).
    let mono_queries: Vec<BlendStateQuery> = (0..=20)
        .map(|k| BlendStateQuery {
            from: SkyState::Clear,
            to: SkyState::Storm,
            t: (k as f32) / 20.0,
        })
        .collect();
    let mono = gpu_kernel.eval(&ctx, &mono_queries);
    for w in mono.windows(2) {
        assert!(
            w[1] >= w[0] - TOL,
            "blend from Clear to Storm must be monotone non-decreasing in t: {} then {}",
            w[0],
            w[1]
        );
    }
}

#[test]
fn empty_queries_yields_no_results() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu_kernel = GpuBlendState::new(&ctx);
    let out = gpu_kernel.eval(&ctx, &[]);
    assert!(out.is_empty(), "an empty query slice yields no values");
}
