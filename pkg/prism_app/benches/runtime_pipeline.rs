//! CPU runtime acceptance measurements for App design §18/§22.
//!
//! Reports distribution tails, actual interval overlap and input-to-render-work
//! latency separately. This is a synthetic CPU workload, not a GPU/present or
//! game-content benchmark. Timings have no machine-independent pass threshold;
//! every mode must produce the identical per-frame output sequence.
#![expect(
    clippy::print_stdout,
    reason = "benchmark emits machine-readable measurements"
)]

use std::hint::black_box;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use prism_app::{App, SubApp, TimeUpdateStrategy, Update};
use prism_ecs::{
    resource::Resource,
    schedule::{Schedule, Schedules},
    system::ResMut,
};
use prism_tasks::TaskPool;

const WARMUP: usize = 16;
const SAMPLES: usize = 240;

#[derive(Clone, Copy, Default)]
struct Frame {
    index: usize,
    digest: u64,
    input: Option<Instant>,
}
impl Resource for Frame {}

#[derive(Clone, Copy)]
struct Interval {
    start: Instant,
    end: Instant,
}

struct RenderSample {
    frame: usize,
    digest: u64,
    interval: Interval,
    input: Instant,
}

#[derive(Default)]
struct Trace {
    simulation: Vec<(usize, Interval)>,
    render: Vec<RenderSample>,
}

#[derive(Clone, Copy, Default)]
struct OverlapStats {
    overlap: Duration,
    eligible_render_work: Duration,
    overlap_samples: usize,
    eligible_samples: usize,
}

struct PipelineReport {
    outputs: Vec<(usize, u64)>,
    overlap: OverlapStats,
    frame_p99_ns: u128,
    overlap_ratio: f64,
}

#[derive(Clone, Copy, Debug)]
struct PerformanceThresholds {
    max_frame_p99_ns: u128,
    min_overlap_ratio: f64,
}

impl PerformanceThresholds {
    fn from_environment() -> Option<Self> {
        let max_frame_p99 = std::env::var("PRISM_PIPELINE_MAX_FRAME_P99_NS").ok();
        let min_overlap = std::env::var("PRISM_PIPELINE_MIN_OVERLAP_RATIO").ok();
        let required = std::env::var("PRISM_PIPELINE_REQUIRE_THRESHOLDS")
            .is_ok_and(|value| matches!(value.as_str(), "1" | "true" | "yes"));
        assert_eq!(
            max_frame_p99.is_some(),
            min_overlap.is_some(),
            "configure both PRISM_PIPELINE_MAX_FRAME_P99_NS and PRISM_PIPELINE_MIN_OVERLAP_RATIO, or neither"
        );
        if required {
            assert!(
                max_frame_p99.is_some(),
                "PRISM_PIPELINE_REQUIRE_THRESHOLDS is set but performance thresholds are missing"
            );
        }
        let max_frame_p99_ns = max_frame_p99?
            .parse()
            .expect("PRISM_PIPELINE_MAX_FRAME_P99_NS must be an unsigned integer");
        let min_overlap_ratio: f64 = min_overlap?
            .parse()
            .expect("PRISM_PIPELINE_MIN_OVERLAP_RATIO must be a finite float");
        assert!(
            min_overlap_ratio.is_finite() && (0.0..=1.0).contains(&min_overlap_ratio),
            "PRISM_PIPELINE_MIN_OVERLAP_RATIO must be between 0 and 1"
        );
        Some(Self {
            max_frame_p99_ns,
            min_overlap_ratio,
        })
    }
}

#[derive(Clone, Copy)]
enum Mode {
    Serial,
    Dedicated,
    Tasks,
}

impl Mode {
    fn name(self) -> &'static str {
        match self {
            Self::Serial => "serial",
            Self::Dedicated => "dedicated",
            Self::Tasks => "task_lane",
        }
    }
}

fn work(mut value: u64, iterations: usize) -> u64 {
    for _ in 0..iterations {
        value = value.wrapping_add(0x9e37_79b9_7f4a_7c15);
        value = (value ^ (value >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        value ^= value >> 31;
    }
    black_box(value)
}

fn calibrated_iterations() -> usize {
    let pilot = 100_000;
    let started = Instant::now();
    black_box(work(1, pilot));
    let elapsed = started.elapsed().as_nanos().max(1);
    ((pilot as u128 * 1_000_000 / elapsed) as usize).clamp(1_000, 10_000_000)
}

fn percentiles(values: &[Duration]) -> [u128; 3] {
    assert!(!values.is_empty());
    let mut sorted: Vec<_> = values.iter().map(Duration::as_nanos).collect();
    sorted.sort_unstable();
    [50, 95, 99].map(|p| sorted[(sorted.len() * p).div_ceil(100).saturating_sub(1)])
}

fn drive(mode: Mode, iterations: usize) -> PipelineReport {
    let trace = Arc::new(Mutex::new(Trace::default()));
    let pool = TaskPool::with_threads(if matches!(mode, Mode::Tasks) { 2 } else { 1 });
    let mut app = App::new();
    app.set_task_pool(pool.clone());
    app.set_time_update_strategy(TimeUpdateStrategy::ManualDelta(Duration::from_millis(16)));
    app.insert_resource(Frame::default());
    let simulation_trace = trace.clone();
    app.add_systems(Update, move |mut state: ResMut<Frame>| {
        let start = Instant::now();
        state.index += 1;
        state.input = Some(start);
        state.digest = work(state.digest ^ state.index as u64, iterations);
        let end = Instant::now();
        simulation_trace
            .lock()
            .unwrap()
            .simulation
            .push((state.index, Interval { start, end }));
    });
    let mut render = SubApp::new();
    render.world.insert_resource(Frame::default());
    render.set_extract(|main, render| {
        *render.resource_mut::<Frame>() = *main.resource::<Frame>();
    });
    let render_trace = trace.clone();
    let mut schedule = Schedule::new();
    schedule.add_systems(move |state: ResMut<Frame>| {
        let start = Instant::now();
        let digest = work(state.digest, iterations);
        let end = Instant::now();
        render_trace.lock().unwrap().render.push(RenderSample {
            frame: state.index,
            digest,
            interval: Interval { start, end },
            input: state.input.unwrap(),
        });
    });
    render
        .world
        .resource_mut::<Schedules>()
        .insert(Update, schedule);
    app.insert_sub_app("bench-render", render);
    match mode {
        Mode::Serial => {}
        Mode::Dedicated => {
            app.enable_pipelined_rendering();
        }
        Mode::Tasks => {
            app.enable_pipelined_rendering_with_task_pool(pool);
        }
    }
    for _ in 0..WARMUP {
        app.update();
    }
    app.sync_sub_apps();
    let started = Instant::now();
    let mut frame_times = Vec::with_capacity(SAMPLES);
    for _ in 0..SAMPLES {
        let frame = Instant::now();
        app.update();
        frame_times.push(frame.elapsed());
    }
    app.sync_sub_apps();
    let elapsed = started.elapsed();
    let trace = trace.lock().unwrap();
    assert_eq!(trace.render.len(), WARMUP + SAMPLES);
    let rendered: Vec<_> = trace.render.iter().filter(|r| r.frame > WARMUP).collect();
    assert_eq!(rendered.len(), SAMPLES);
    assert_eq!(
        rendered
            .iter()
            .map(|sample| sample.frame)
            .collect::<Vec<_>>(),
        (WARMUP + 1..=WARMUP + SAMPLES).collect::<Vec<_>>(),
        "render must consume every frame exactly once"
    );
    assert_eq!(
        trace
            .simulation
            .iter()
            .map(|(frame, _)| *frame)
            .collect::<Vec<_>>(),
        (1..=WARMUP + SAMPLES).collect::<Vec<_>>(),
        "simulation must execute every frame exactly once"
    );
    let latencies: Vec<_> = rendered.iter().map(|r| r.interval.end - r.input).collect();
    // A render of frame N is only eligible to overlap simulation of frame N+1.
    // Pairing adjacent frames avoids counting unrelated intervals as pipeline
    // overlap and makes the reported value auditable from the trace.
    let mut overlap = OverlapStats::default();
    for r in &rendered {
        assert!(r.interval.end > r.interval.start);
        if r.frame == WARMUP + SAMPLES {
            continue;
        }
        let (_, simulation) = trace
            .simulation
            .iter()
            .find(|(frame, _)| *frame == r.frame + 1)
            .expect("every measured render frame except the tail has a successor");
        let render_work = r.interval.end - r.interval.start;
        overlap.eligible_samples += 1;
        overlap.eligible_render_work += render_work;
        let start = simulation.start.max(r.interval.start);
        let end = simulation.end.min(r.interval.end);
        if end > start {
            overlap.overlap_samples += 1;
            overlap.overlap += end - start;
        }
    }
    assert_eq!(overlap.eligible_samples, SAMPLES - 1);
    assert!(overlap.overlap <= overlap.eligible_render_work);
    let overlap_ratio = overlap.overlap.as_secs_f64()
        / overlap
            .eligible_render_work
            .as_secs_f64()
            .max(f64::MIN_POSITIVE);
    assert!((0.0..=1.000_001).contains(&overlap_ratio));
    if matches!(mode, Mode::Serial) {
        assert_eq!(overlap.overlap, Duration::ZERO);
        assert_eq!(overlap.overlap_samples, 0);
    } else {
        assert!(
            overlap.overlap_samples > 0,
            "{} pipeline did not observe any adjacent-frame overlap",
            mode.name()
        );
    }
    let frame_percentiles = percentiles(&frame_times);
    println!("{{\"kind\":\"pipeline\",\"mode\":\"{}\",\"samples\":{},\"elapsed_ns\":{},\"frame_ns_p50_p95_p99\":{:?},\"input_to_render_done_ns_p50_p95_p99\":{:?},\"overlap_ns\":{},\"overlap_samples\":{},\"overlap_candidates\":{},\"overlap_ratio\":{:.6},\"overlap_observed\":{}}}",
        mode.name(), SAMPLES, elapsed.as_nanos(), frame_percentiles, percentiles(&latencies), overlap.overlap.as_nanos(), overlap.overlap_samples, overlap.eligible_samples, overlap_ratio, overlap.overlap > Duration::ZERO);
    PipelineReport {
        outputs: trace.render.iter().map(|r| (r.frame, r.digest)).collect(),
        overlap,
        frame_p99_ns: frame_percentiles[2],
        overlap_ratio,
    }
}

fn scaling(workers: usize, iterations: usize) {
    const SYSTEMS: usize = 8;
    let mut app = App::new();
    app.set_task_pool(TaskPool::with_threads(workers));
    app.set_time_update_strategy(TimeUpdateStrategy::ManualDelta(Duration::from_millis(16)));
    let counts: Arc<Vec<std::sync::atomic::AtomicU64>> =
        Arc::new((0..SYSTEMS).map(|_| 0.into()).collect());
    for index in 0..SYSTEMS {
        let counts = counts.clone();
        app.add_systems(Update, move || {
            black_box(work(index as u64, iterations));
            counts[index].fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        });
    }
    for _ in 0..WARMUP {
        app.update();
    }
    let started = Instant::now();
    let mut times = Vec::with_capacity(SAMPLES);
    for _ in 0..SAMPLES {
        let frame = Instant::now();
        app.update();
        times.push(frame.elapsed());
    }
    let elapsed = started.elapsed();
    for count in counts.iter() {
        assert_eq!(
            count.load(std::sync::atomic::Ordering::Relaxed),
            (WARMUP + SAMPLES) as u64
        );
    }
    println!("{{\"kind\":\"schedule_scaling\",\"workers\":{workers},\"systems\":{SYSTEMS},\"samples\":{SAMPLES},\"elapsed_ns\":{},\"frame_ns_p50_p95_p99\":{:?}}}", elapsed.as_nanos(), percentiles(&times));
}

fn main() {
    let iterations = calibrated_iterations();
    let thresholds = PerformanceThresholds::from_environment();
    println!("{{\"kind\":\"environment\",\"os\":\"{}\",\"arch\":\"{}\",\"available_parallelism\":{},\"iterations_per_system\":{iterations},\"warmup\":{WARMUP},\"samples\":{SAMPLES}}}",
        std::env::consts::OS, std::env::consts::ARCH, std::thread::available_parallelism().map_or(1, |n| n.get()));
    let serial = drive(Mode::Serial, iterations);
    let dedicated = drive(Mode::Dedicated, iterations);
    assert_eq!(
        serial.outputs, dedicated.outputs,
        "dedicated pipeline output diverged"
    );
    let tasks = drive(Mode::Tasks, iterations);
    assert_eq!(
        serial.outputs, tasks.outputs,
        "task pipeline output diverged"
    );
    for workers in [1, 2, 4, 8] {
        scaling(workers, iterations);
    }
    let thresholds_passed = thresholds.is_some_and(|thresholds| {
        let frame_ok = dedicated.frame_p99_ns <= thresholds.max_frame_p99_ns
            && tasks.frame_p99_ns <= thresholds.max_frame_p99_ns;
        let overlap_ok = dedicated.overlap_ratio >= thresholds.min_overlap_ratio
            && tasks.overlap_ratio >= thresholds.min_overlap_ratio;
        assert!(
            frame_ok,
            "pipeline frame P99 exceeded PRISM_PIPELINE_MAX_FRAME_P99_NS: dedicated={} task_lane={} limit={}",
            dedicated.frame_p99_ns,
            tasks.frame_p99_ns,
            thresholds.max_frame_p99_ns
        );
        assert!(
            overlap_ok,
            "pipeline overlap ratio below PRISM_PIPELINE_MIN_OVERLAP_RATIO: dedicated={:.6} task_lane={:.6} limit={:.6}",
            dedicated.overlap_ratio,
            tasks.overlap_ratio,
            thresholds.min_overlap_ratio
        );
        true
    });
    println!("{{\"kind\":\"acceptance\",\"output_parity\":true,\"frame_sequence\":true,\"serial_overlap_ns\":{},\"dedicated_overlap_ns\":{},\"task_lane_overlap_ns\":{},\"overlap_is_observed\":{{\"dedicated\":{},\"task_lane\":{}}},\"performance_thresholds_configured\":{},\"performance_thresholds_passed\":{},\"performance_gate_required\":{}}}",
        serial.overlap.overlap.as_nanos(), dedicated.overlap.overlap.as_nanos(), tasks.overlap.overlap.as_nanos(), dedicated.overlap.overlap > Duration::ZERO, tasks.overlap.overlap > Duration::ZERO, thresholds.is_some(), thresholds_passed, std::env::var("PRISM_PIPELINE_REQUIRE_THRESHOLDS").is_ok_and(|value| matches!(value.as_str(), "1" | "true" | "yes")));
}
