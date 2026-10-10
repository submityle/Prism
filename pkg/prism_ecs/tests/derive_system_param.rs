//! End-to-end integration test for `#[derive(SystemParam)]` exposed through the
//! `prism_ecs` prelude.
//!
//! The macro-crate unit tests check the *generated token stream*; this test
//! checks that the generated `unsafe impl prism_ecs::system::SystemParam`
//! actually compiles and behaves when the derive is invoked from a downstream
//! crate (here, `prism_ecs` itself via its `prism_ecs` dev-dependency). It
//! drives a real `World` by using the derived composite param as the single
//! argument of a function system, then runs that system through the normal
//! `System::run` path (which also flushes `Commands` via `apply_deferred`).
//!
//! Crucially this exercises the design assumption that a composite `State`
//! tuple built from projections like `<Res<'w, T> as SystemParam>::State`
//! (which mention the impl lifetimes `'w`/`'s`) still resolves to a `'static`
//! tuple and compiles.

use prism_ecs::prelude::*;

#[derive(Component)]
struct Position {
    x: i64,
}

#[derive(Resource)]
struct Gravity(i64);

#[derive(Resource, Default)]
struct Tally {
    runs: u32,
    sum: i64,
    spawned: u32,
}

/// A composite system param mixing all four param kinds plus a read-only query,
/// across both the `'w` (world) and `'s` (system-state) lifetimes.
#[derive(SystemParam)]
struct StepCtx<'w, 's> {
    gravity: Res<'w, Gravity>,
    tally: ResMut<'w, Tally>,
    seen_runs: Local<'s, u32>,
    bodies: Query<'w, 's, &'static Position>,
    commands: Commands<'w, 's>,
}

#[test]
fn derived_system_param_drives_a_real_system() {
    let mut world = World::new();
    world.insert_resource(Gravity(10));
    world.init_resource::<Tally>();
    world.spawn(Position { x: 1 });
    world.spawn(Position { x: 2 });
    world.spawn(Position { x: 4 });

    // The whole system takes exactly one argument: the derived composite param.
    // This only compiles because `#[derive(SystemParam)]` produced a real
    // `impl SystemParam for StepCtx`, and the function-system bridge accepts it.
    fn step(mut ctx: StepCtx) {
        *ctx.seen_runs += 1;

        // Read access through `Res`.
        let g = ctx.gravity.0;

        // Query iteration over the three bodies.
        let mut sum = 0;
        for pos in ctx.bodies.iter() {
            sum += pos.x * g;
        }

        // Write access through `ResMut`.
        ctx.tally.runs = *ctx.seen_runs;
        ctx.tally.sum = sum;

        // Deferred structural change through `Commands` (flushed by `run`).
        ctx.commands.spawn(Position { x: 100 });
        ctx.tally.spawned += 1;
    }

    let mut sys = IntoSystem::into_system(step);
    sys.initialize(&mut world);

    // Run 1: 3 bodies, each x * gravity(10) => (1+2+4)*10 = 70.
    sys.run(&mut world);
    {
        let t = world.resource::<Tally>();
        assert_eq!(t.runs, 1, "Local persisted across the single run");
        assert_eq!(t.sum, 70, "Res * Query sum for the first run");
        assert_eq!(t.spawned, 1);
    }

    // Run 2: the previous run's `Commands::spawn` was flushed, so there is now a
    // 4th body (x = 100). Sum => (1+2+4+100)*10 = 1070, and Local increments.
    sys.run(&mut world);
    {
        let t = world.resource::<Tally>();
        assert_eq!(t.runs, 2, "Local is per-system state and survives runs");
        assert_eq!(t.sum, 1070, "the deferred spawn from run 1 is now visible");
        assert_eq!(t.spawned, 2);
    }
}

/// A unit-struct param (empty `State` tuple) is also a valid `SystemParam` and
/// can drive a system that takes nothing from the world.
#[derive(SystemParam)]
struct NoCtx;

#[test]
fn unit_system_param_is_usable() {
    let mut world = World::new();

    fn noop(_ctx: NoCtx) {}

    let mut sys = IntoSystem::into_system(noop);
    sys.initialize(&mut world);
    sys.run(&mut world);
}
