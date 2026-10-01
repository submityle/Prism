//! Real-device parity for the groom motion-energy reduction twin:
//! [`GpuHairMotionEnergy`] must reproduce the `CPU` golden
//! [`reference_motion_energy`](prism_hair_gpu::motion_energy::reference_motion_energy)
//! (which forwards
//! [`groom_motion_energy`](prism_render_architecture::hair::sleep::groom_motion_energy))
//! for a groom's flat particle slice, folding the sum of every particle's
//! squared implicit velocity (`position - prev_position`, self-dotted) into one
//! groom-global scalar. The suite drives a single moving particle (closed-form
//! energy), a mix of moving and pinned particles (a pinned particle holds
//! `prev == position`, so it contributes zero), an all-at-rest groom (energy
//! `0`), the empty no-op, and a large multi-stride batch that forces the
//! grid-stride load to wrap the 256-wide workgroup several times.
//!
//! The tests skip (with a printed notice) when the host has no `wgpu` adapter,
//! so the suite stays green everywhere while still exercising the full
//! dispatch-and-readback on any real device. The kernel is portable core-WGSL
//! plus workgroup shared memory, so it needs no optional device feature.
//!
//! # Parity criterion
//!
//! The fold is floating-point addition, which is commutative but not
//! associative, so the tree's pairwise order differs from the golden's
//! left-to-right walk by a few low-mantissa `ULP`; each value is asserted
//! within `abs_diff < 1e-4` or `rel_diff < 1e-3` rather than bit-for-bit. All
//! inputs are explicit literals or integer-derived fractions kept finite; no
//! `sin`/`cos` appears anywhere.
//!
//! Provenance: standard single-workgroup shared-memory tree reduction over a
//! Verlet velocity proxy; no Unreal Engine source or derived code.

use prism_hair_gpu::motion_energy::{reference_motion_energy, GpuHairMotionEnergy};
use prism_hair_gpu::GpuContext;
use prism_render_architecture::hair::dynamics::{StrandParticle, Vec3};

/// Acquires a headless context, or `None` (with a skip notice) when the host has
/// no `wgpu` adapter so the suite stays green off-device.
#[expect(
    clippy::print_stderr,
    reason = "the skip notice must reach the test log on hosts without a wgpu adapter"
)]
fn context_or_skip(label: &str) -> Option<GpuContext> {
    match GpuContext::try_headless() {
        Some(ctx) => Some(ctx),
        None => {
            eprintln!("skipping {label}: no wgpu adapter on this host");
            None
        }
    }
}

/// Asserts two scalars agree within the documented fma / reassociation
/// tolerance.
fn assert_close(got: f32, want: f32, what: &str) {
    let abs = (got - want).abs();
    let rel = abs / want.abs().max(1.0);
    assert!(
        abs < 1e-4 || rel < 1e-3,
        "{what}: got {got}, want {want} (abs {abs}, rel {rel})"
    );
}

/// Dispatches one groom through the device twin.
fn run(ctx: &GpuContext, particles: &[StrandParticle]) -> f32 {
    GpuHairMotionEnergy::new(ctx).energy(ctx, particles)
}

/// A particle whose implicit velocity is `pos - prev`.
fn particle(pos: [f32; 3], prev: [f32; 3]) -> StrandParticle {
    StrandParticle {
        position: Vec3::new(pos[0], pos[1], pos[2]),
        prev_position: Vec3::new(prev[0], prev[1], prev[2]),
        inverse_mass: 1.0,
    }
}

#[test]
fn single_particle_energy_matches_closed_form() {
    let Some(ctx) = context_or_skip("single_particle_energy_matches_closed_form") else {
        return;
    };
    // velocity (3, 4, 0) -> squared length 25.
    let particles = [particle([3.0, 4.0, 0.0], [0.0, 0.0, 0.0])];
    let got = run(&ctx, &particles);
    assert_close(got, 25.0, "single particle energy");
    assert_close(got, reference_motion_energy(&particles), "single vs golden");
}

#[test]
fn pinned_particles_contribute_zero() {
    let Some(ctx) = context_or_skip("pinned_particles_contribute_zero") else {
        return;
    };
    // One mover (velocity (1,2,2) -> 9) plus two "pinned" particles whose prev
    // equals position (zero velocity): total energy is the mover's alone.
    let particles = [
        particle([1.0, 2.0, 2.0], [0.0, 0.0, 0.0]),
        particle([5.0, 5.0, 5.0], [5.0, 5.0, 5.0]),
        particle([-3.0, 7.0, 1.0], [-3.0, 7.0, 1.0]),
    ];
    let got = run(&ctx, &particles);
    assert_close(got, 9.0, "mix energy");
    assert_close(got, reference_motion_energy(&particles), "mix vs golden");
}

#[test]
fn all_at_rest_is_zero() {
    let Some(ctx) = context_or_skip("all_at_rest_is_zero") else {
        return;
    };
    let particles = [
        particle([1.0, 1.0, 1.0], [1.0, 1.0, 1.0]),
        particle([2.0, 0.0, -4.0], [2.0, 0.0, -4.0]),
    ];
    let got = run(&ctx, &particles);
    assert_close(got, 0.0, "rest energy");
    assert_close(got, reference_motion_energy(&particles), "rest vs golden");
}

#[test]
fn empty_groom_is_zero() {
    let Some(ctx) = context_or_skip("empty_groom_is_zero") else {
        return;
    };
    let got = run(&ctx, &[]);
    assert_close(got, 0.0, "empty energy");
    assert_close(got, reference_motion_energy(&[]), "empty vs golden");
}

#[test]
fn large_groom_wraps_workgroup_several_times() {
    let Some(ctx) = context_or_skip("large_groom_wraps_workgroup_several_times") else {
        return;
    };
    // 1000 particles force the 256-wide grid-stride load to wrap ~4 times; a
    // deterministic integer-derived velocity sweep keeps every value finite.
    let mut particles = Vec::new();
    for k in 0u32..1000 {
        let vx = (k % 11) as f32 / 8.0 - 0.5;
        let vy = (k % 7) as f32 / 4.0 - 1.0;
        let vz = (k % 5) as f32 / 2.0 - 1.0;
        let base = k as f32 * 0.25;
        particles.push(particle(
            [base + vx, base + vy, base + vz],
            [base, base, base],
        ));
    }
    let got = run(&ctx, &particles);
    assert_close(got, reference_motion_energy(&particles), "large vs golden");
    assert!(got.is_finite(), "large energy finite, got {got}");
}
