//! Run identical deterministic content through real windowed and headless runners.
//!
//! `cargo run -p prism_app --features winit,determinism --example runner_parity`
//! opens an OS window and exits after 128 frames. `--headless-only` runs the
//! portable reference twice without a display. The digest excludes OS events,
//! wall time and presentation: this checks simulation and lifecycle parity.
#![expect(clippy::print_stdout, reason = "acceptance binary emits its result")]

use std::sync::{Arc, Mutex};

use prism_app::determinism::{DeterministicRng, FrameHash};
use prism_app::{
    App, AppExit, AppExitRequest, EngineClocks, First, FixedUpdate, HeadlessRunner, Last, Shutdown,
    Startup, TimeUpdateStrategy, Update, WindowSettings, WinitRunner,
};
use prism_ecs::{resource::Resource, world::World};
use prism_time::Duration;

const FRAMES: usize = 128;
const STEP_US: u64 = 10_000;
const JITTER_US: [u64; 8] = [0, 10_000, 19_000, 1_000, 33_000, 27_000, 17_000, 4_000];

#[derive(Debug, Default, PartialEq, Eq)]
struct Report {
    startup: u32,
    shutdown: u32,
    first: u32,
    update: u32,
    last: u32,
    samples: Vec<(u64, u64, u64, u64)>,
}

struct Content {
    rng: DeterministicRng,
    values: [u64; 256],
    frame: usize,
    steps: u64,
    phase: u8,
    first_runs: u32,
    update_runs: u32,
    last_runs: u32,
    report: Arc<Mutex<Report>>,
}
impl Resource for Content {}

fn startup(world: &mut World) {
    let content = world.resource_mut::<Content>();
    for value in &mut content.values {
        *value = content.rng.next_u64();
    }
    content.report.lock().unwrap().startup += 1;
}

fn first(world: &mut World) {
    let content = world.resource_mut::<Content>();
    assert_eq!(content.phase, 0);
    content.phase = 1;
    content.first_runs += 1;
}

fn fixed(world: &mut World) {
    assert_eq!(
        world.resource::<EngineClocks>().fixed().delta(),
        Duration::from_micros(STEP_US)
    );
    let content = world.resource_mut::<Content>();
    assert_eq!(content.phase, 1);
    let input = content.rng.next_u64();
    for (index, value) in content.values.iter_mut().enumerate() {
        *value = value
            .wrapping_add(input ^ index as u64)
            .rotate_left(13)
            .wrapping_mul(0x9e37_79b9_7f4a_7c15);
    }
    content.steps += 1;
}

fn update(world: &mut World) {
    let content = world.resource_mut::<Content>();
    assert_eq!(content.phase, 1);
    content.phase = 2;
    content.update_runs += 1;
}

fn last(world: &mut World) {
    let clocks = world.resource::<EngineClocks>();
    let elapsed = clocks.virtual_time().elapsed().as_micros() as u64;
    let overstep = clocks.fixed().overstep().as_micros() as u64;
    let content = world.resource_mut::<Content>();
    assert_eq!(content.phase, 2);
    content.phase = 0;
    content.last_runs += 1;
    let mut hash = FrameHash::with_window(1);
    for &value in &content.values {
        hash.write_u64(value);
    }
    hash.write_u64(content.steps);
    content
        .report
        .lock()
        .unwrap()
        .samples
        .push((hash.current(), content.steps, elapsed, overstep));
    content.frame += 1;
    let frame = content.frame;
    world.insert_resource(TimeUpdateStrategy::ManualDelta(Duration::from_micros(
        JITTER_US[frame % JITTER_US.len()],
    )));
    if frame == FRAMES {
        world.resource_mut::<AppExitRequest>().send_success();
    }
}

fn shutdown(world: &mut World) {
    let content = world.resource::<Content>();
    let mut report = content.report.lock().unwrap();
    report.shutdown += 1;
    report.first = content.first_runs;
    report.update = content.update_runs;
    report.last = content.last_runs;
}

fn app(report: Arc<Mutex<Report>>) -> App {
    let mut app = App::new();
    app.set_fixed_timestep(Duration::from_micros(STEP_US))
        .set_time_update_strategy(TimeUpdateStrategy::ManualDelta(Duration::from_micros(
            JITTER_US[0],
        )))
        .insert_resource(Content {
            rng: DeterministicRng::seeded(0x0050_5249_534d),
            values: [0; 256],
            frame: 0,
            steps: 0,
            phase: 0,
            first_runs: 0,
            update_runs: 0,
            last_runs: 0,
            report,
        })
        .add_systems(Startup, startup)
        .add_systems(First, first)
        .add_systems(FixedUpdate, fixed)
        .add_systems(Update, update)
        .add_systems(Last, last)
        .add_systems(Shutdown, shutdown);
    app
}

fn validate(report: &Report) {
    assert_eq!(
        (
            report.startup,
            report.shutdown,
            report.first,
            report.update,
            report.last,
            report.samples.len()
        ),
        (1, 1, FRAMES as u32, FRAMES as u32, FRAMES as u32, FRAMES)
    );
    let mut elapsed = 0;
    for (frame, &(_, steps, actual_elapsed, overstep)) in report.samples.iter().enumerate() {
        elapsed += JITTER_US[frame % JITTER_US.len()];
        assert_eq!(
            (steps, actual_elapsed, overstep),
            (elapsed / STEP_US, elapsed, elapsed % STEP_US)
        );
    }
}

fn trace_hash(report: &Report) -> u64 {
    let mut hash = FrameHash::with_window(1);
    for &(frame_hash, steps, elapsed, overstep) in &report.samples {
        hash.write_u64(frame_hash);
        hash.write_u64(steps);
        hash.write_u64(elapsed);
        hash.write_u64(overstep);
    }
    hash.current()
}

fn main() {
    let headless = Arc::new(Mutex::new(Report::default()));
    assert_eq!(
        HeadlessRunner::with_max_frames(FRAMES as u64 + 1).run(app(Arc::clone(&headless))),
        AppExit::Success
    );
    let comparison = Arc::new(Mutex::new(Report::default()));
    let headless_only = std::env::args().any(|arg| arg == "--headless-only");
    let other = app(Arc::clone(&comparison));
    let exit = if headless_only {
        HeadlessRunner::with_max_frames(FRAMES as u64 + 1).run(other)
    } else {
        let settings = WindowSettings {
            title: "Prism runner parity — 128 deterministic frames".into(),
            ..WindowSettings::default()
        };
        WinitRunner::new(other).with_settings(settings).run()
    };
    assert_eq!(exit, AppExit::Success);
    let reference = headless.lock().unwrap();
    let actual = comparison.lock().unwrap();
    validate(&reference);
    validate(&actual);
    assert_eq!(*reference, *actual, "runner simulation/lifecycle diverged");
    let reference_trace = trace_hash(&reference);
    let actual_trace = trace_hash(&actual);
    assert_eq!(
        reference_trace, actual_trace,
        "runner trace digest diverged"
    );
    println!(
        "runner_parity: frames={FRAMES} fixed_steps={} final_hash={:016x} trace_hash={reference_trace:016x} first={} update={} last={} windowed={} parity=true",
        reference.samples[FRAMES - 1].1,
        reference.samples[FRAMES - 1].0,
        reference.first,
        reference.update,
        reference.last,
        !headless_only
    );
}
