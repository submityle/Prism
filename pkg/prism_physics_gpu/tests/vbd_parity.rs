//! Real-device parity for the `GPU` Vertex Block Descent (VBD) solver against
//! its `CPU` golden twin.
//!
//! Every test acquires a headless device with `GpuContext::try_headless()` and
//! skips cleanly when no adapter is available (for example inside a sandbox),
//! so the suite is a no-op rather than a failure off a real `GPU`.
//!
//! The `CPU` twin ([`cpu_vbd`]) delegates to `prism_physics_core`'s
//! `VbdSolver::step_colored`, while the kernel ([`GpuVbd::solve`]) reimplements
//! the same colour-major Gauss-Seidel block descent in `WGSL` over a host-built
//! incident-spring `CSR` and colour ordering. The only divergence is a few
//! `ULP` in the per-vertex `3x3` solve's division, so parity is checked within
//! a tight tolerance. Inputs use integer coordinates so the topology and rest
//! lengths are exact and the two paths share one `color_springs` colouring.
//!
//! Provenance: VBD follows Chen et al., "Vertex Block Descent" (SIGGRAPH 2024);
//! greedy graph colouring is a standard, publicly documented technique. No
//! Unreal Engine source or derived code.

use glam::Vec3;
use prism_physics_core::vbd::{color_springs, SpringElement, SpringSet, VbdColoring, VbdConfig};
use prism_physics_core::{ParticleHandle, ParticleStorage};
use prism_physics_gpu::context::GpuContext;
use prism_physics_gpu::{cpu_vbd, GpuVbd};

/// Absolute/relative tolerance for single-step position and velocity parity.
const TOL: f32 = 1.0e-4;

/// Looser tolerance for multi-step stiff accumulation.
const TOL_MULTI: f32 = 1.0e-3;

/// Frame step used across the suite.
const DT: f32 = 1.0 / 60.0;

#[expect(
    clippy::print_stderr,
    reason = "the suite is a deliberate no-op when no GPU adapter is present"
)]
fn headless() -> Option<GpuContext> {
    match GpuContext::try_headless() {
        Some(ctx) => Some(ctx),
        None => {
            eprintln!("skipping vbd parity: no GPU adapter available");
            None
        }
    }
}

/// Asserts two vectors agree within `tol` (combined absolute/relative).
#[track_caller]
fn assert_vec_close(a: Vec3, b: Vec3, tol: f32, what: &str) {
    let scale = a.length().max(b.length()).max(1.0);
    assert!(
        a.distance(b) <= tol * scale,
        "{what}: cpu={a:?} gpu={b:?} (dist {})",
        a.distance(b)
    );
}

/// Steps both paths once from the current `storage` state, asserts full parity
/// of the advanced positions and velocities within `tol`, then writes the
/// golden (`CPU`) result back into `storage` so a multi-step run advances both
/// paths from an identical, deterministic state each step.
#[track_caller]
fn step_and_sync(
    ctx: &GpuContext,
    gpu: &GpuVbd,
    storage: &mut ParticleStorage,
    handles: &[ParticleHandle],
    springs: &SpringSet,
    config: &VbdConfig,
    coloring: &VbdColoring,
    dt: f32,
    tol: f32,
) {
    let (cpu_pos, cpu_vel) = cpu_vbd(storage, springs, config, coloring, dt);
    let (gpu_pos, gpu_vel) = gpu.solve(ctx, storage, springs, config, coloring, dt);

    assert_eq!(cpu_pos.len(), gpu_pos.len(), "position count");
    assert_eq!(cpu_vel.len(), gpu_vel.len(), "velocity count");

    for i in 0..cpu_pos.len() {
        assert_vec_close(cpu_pos[i], gpu_pos[i], tol, "position");
        assert_vec_close(cpu_vel[i], gpu_vel[i], tol, "velocity");
    }

    for (i, &h) in handles.iter().enumerate() {
        storage.set_position(h, cpu_pos[i]);
        storage.set_velocity(h, cpu_vel[i]);
    }
}

/// Builds a `w x h` integer-coordinate grid in the `XY` plane with structural
/// (axis) and shear (diagonal) springs. The top row (`y == h - 1`) is pinned.
/// Returns the storage, spring set, and the row-major vertex handles.
fn pinned_grid(w: usize, h: usize, stiffness: f32) -> (ParticleStorage, SpringSet, Vec<ParticleHandle>) {
    let mut particles = ParticleStorage::new();
    let mut handles = Vec::with_capacity(w * h);
    for y in 0..h {
        for x in 0..w {
            let pos = Vec3::new(x as f32, y as f32, 0.0);
            let handle = if y == h - 1 {
                particles.spawn_pinned(pos)
            } else {
                particles.spawn(pos, 1.0)
            };
            handles.push(handle);
        }
    }
    let idx = |x: usize, y: usize| y * w + x;
    let mut springs = SpringSet::new();
    // Structural springs (rest length 1).
    for y in 0..h {
        for x in 0..w {
            if x + 1 < w {
                springs.push(SpringElement::new(handles[idx(x, y)], handles[idx(x + 1, y)], 1.0, stiffness));
            }
            if y + 1 < h {
                springs.push(SpringElement::new(handles[idx(x, y)], handles[idx(x, y + 1)], 1.0, stiffness));
            }
        }
    }
    // Shear springs (rest length sqrt(2)).
    let diag = 2.0_f32.sqrt();
    for y in 0..h - 1 {
        for x in 0..w - 1 {
            springs.push(SpringElement::new(handles[idx(x, y)], handles[idx(x + 1, y + 1)], diag, stiffness));
            springs.push(SpringElement::new(handles[idx(x + 1, y)], handles[idx(x, y + 1)], diag, stiffness));
        }
    }
    (particles, springs, handles)
}

#[test]
fn pinned_grid_multi_step_matches_cpu() {
    let Some(ctx) = headless() else {
        return;
    };
    let gpu = GpuVbd::new(&ctx);
    let (mut storage, springs, handles) = pinned_grid(4, 4, 500.0);
    let coloring = color_springs(&springs, storage.len());
    assert!(coloring.color_count() >= 2, "shear+structural grid needs several colours");
    let config = VbdConfig {
        gravity: Vec3::new(0.0, -9.81, 0.0),
        substeps: 2,
        iterations: 6,
        damping: 0.5,
    };
    for _ in 0..24 {
        step_and_sync(&ctx, &gpu, &mut storage, &handles, &springs, &config, &coloring, DT, TOL_MULTI);
    }
}

#[test]
fn free_fall_without_springs_matches_cpu() {
    let Some(ctx) = headless() else {
        return;
    };
    let gpu = GpuVbd::new(&ctx);
    let mut storage = ParticleStorage::new();
    let handles = vec![
        storage.spawn(Vec3::new(0.0, 10.0, 0.0), 1.0),
        storage.spawn(Vec3::new(1.0, 10.0, 0.0), 2.0),
        storage.spawn(Vec3::new(2.0, 10.0, 0.0), 0.5),
    ];
    let springs = SpringSet::new();
    let coloring = color_springs(&springs, storage.len());
    let config = VbdConfig::default();
    for _ in 0..10 {
        step_and_sync(&ctx, &gpu, &mut storage, &handles, &springs, &config, &coloring, DT, TOL);
    }
}

#[test]
fn single_pinned_vertex_stays_put() {
    let Some(ctx) = headless() else {
        return;
    };
    let gpu = GpuVbd::new(&ctx);
    let mut storage = ParticleStorage::new();
    let handles = vec![storage.spawn_pinned(Vec3::new(3.0, 7.0, -2.0))];
    let springs = SpringSet::new();
    let coloring = color_springs(&springs, storage.len());
    let config = VbdConfig::default();
    step_and_sync(&ctx, &gpu, &mut storage, &handles, &springs, &config, &coloring, DT, TOL);
    assert_vec_close(storage.positions()[0], Vec3::new(3.0, 7.0, -2.0), TOL, "pinned stays put");
}

#[test]
fn two_vertex_spring_matches_cpu() {
    let Some(ctx) = headless() else {
        return;
    };
    let gpu = GpuVbd::new(&ctx);
    let mut storage = ParticleStorage::new();
    // Stretched past rest so the spring pulls while gravity acts.
    let top = storage.spawn_pinned(Vec3::new(0.0, 2.0, 0.0));
    let bot = storage.spawn(Vec3::new(0.0, 0.0, 0.0), 1.0);
    let handles = vec![top, bot];
    let mut springs = SpringSet::new();
    springs.push(SpringElement::new(top, bot, 1.0, 200.0));
    let coloring = color_springs(&springs, storage.len());
    let config = VbdConfig {
        gravity: Vec3::new(0.0, -9.81, 0.0),
        substeps: 1,
        iterations: 8,
        damping: 0.5,
    };
    for _ in 0..16 {
        step_and_sync(&ctx, &gpu, &mut storage, &handles, &springs, &config, &coloring, DT, TOL);
    }
}

#[test]
fn zero_dt_is_a_no_op() {
    let Some(ctx) = headless() else {
        return;
    };
    let gpu = GpuVbd::new(&ctx);
    let (storage, springs, _handles) = pinned_grid(3, 3, 100.0);
    let coloring = color_springs(&springs, storage.len());
    let config = VbdConfig::default();
    let (gpu_pos, gpu_vel) = gpu.solve(&ctx, &storage, &springs, &config, &coloring, 0.0);
    for (i, p) in storage.positions().iter().enumerate() {
        assert_vec_close(*p, gpu_pos[i], TOL, "dt=0 position unchanged");
    }
    for (i, v) in storage.velocities().iter().enumerate() {
        assert_vec_close(*v, gpu_vel[i], TOL, "dt=0 velocity unchanged");
    }
}

#[test]
fn empty_storage_returns_empty() {
    let Some(ctx) = headless() else {
        return;
    };
    let gpu = GpuVbd::new(&ctx);
    let storage = ParticleStorage::new();
    let springs = SpringSet::new();
    let coloring = color_springs(&springs, 0);
    let config = VbdConfig::default();
    let (pos, vel) = gpu.solve(&ctx, &storage, &springs, &config, &coloring, DT);
    assert!(pos.is_empty(), "no particles yields no positions");
    assert!(vel.is_empty(), "no particles yields no velocities");
}
