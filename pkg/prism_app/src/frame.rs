//! Persistent, system-granular complete-frame stepping for PIE.
//!
//! A frame session owns the same schedule cursors used by the ECS executor. It
//! advances the main clock once, expands the fixed driver into its real ticks,
//! crosses extraction seams in order, and never replays a completed prefix.

use alloc::collections::BTreeSet;
use alloc::vec::Vec;

use prism_ecs::schedule::{ContinueReport, ContinueStop, StepReport, Stepping};
use prism_time::DefaultSource;

use crate::fixed::{
    AfterFixedMainLoop, BeforeFixedMainLoop, FixedFirst, FixedLast, FixedPostUpdate,
    FixedPreUpdate, FixedUpdate,
};
use crate::sub_app::{continue_schedule, step_schedule, SubApp};
use crate::time::advance_time;
use crate::App;

/// Which world produced a frame-step event.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum FrameApp {
    /// The authoritative main world.
    Main,
    /// A secondary world in insertion order.
    Secondary(usize),
}

/// A system's location in complete-frame order.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct FrameSystem {
    /// The world containing the system.
    pub app: FrameApp,
    /// Variable-frame phase index, or the fixed-driver slot for fixed systems.
    pub phase: usize,
    /// ECS schedule node index.
    pub node: usize,
}

/// Result of one complete-frame system step.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FrameStepReport {
    /// A system ran.
    Ran(FrameSystem),
    /// A system was gated off.
    Skipped(FrameSystem),
    /// A system was retired by panic isolation.
    Failed(FrameSystem),
    /// The extract seam was crossed atomically between worlds.
    Extracted(usize),
    /// The session reached the frame boundary.
    Complete,
}

/// Why a complete-frame continuation stopped.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FrameContinueStop {
    /// A configured break-before breakpoint.
    Breakpoint(FrameSystem),
    /// The complete frame finished.
    Complete,
}

/// Outcome of `continue_frame`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FrameContinueReport {
    /// Systems whose bodies ran during this continuation.
    pub ran: Vec<FrameSystem>,
    /// The reason continuation stopped.
    pub stop: FrameContinueStop,
}

#[derive(Default)]
struct Driver {
    phase: usize,
    stepping: Stepping,
    fixed: Option<FixedDriver>,
    done: bool,
}

struct DriverContinue {
    ran: Vec<(usize, usize)>,
    breakpoint: Option<(usize, usize)>,
    done: bool,
}

struct FixedDriver {
    stage: FixedStage,
    stepping: Stepping,
}

#[derive(Clone, Copy)]
enum FixedStage {
    Before,
    Tick,
    Phase(usize),
    After,
    Done,
}

impl FixedDriver {
    fn new() -> Self {
        Self {
            stage: FixedStage::Before,
            stepping: Stepping::new(),
        }
    }

    fn step(&mut self, sub_app: &mut SubApp) -> Option<StepReport> {
        loop {
            match self.stage {
                FixedStage::Before => {
                    let report =
                        step_schedule(&mut sub_app.world, BeforeFixedMainLoop, &mut self.stepping);
                    if !matches!(report, StepReport::FrameBoundary) {
                        return Some(report);
                    }
                    self.stepping.reset();
                    if sub_app
                        .world
                        .get_resource::<crate::time::EngineClocks>()
                        .is_none()
                    {
                        self.stage = FixedStage::Done;
                        continue;
                    }
                    sub_app
                        .world
                        .resource_mut::<crate::time::EngineClocks>()
                        .set_source(DefaultSource::Fixed);
                    self.stage = FixedStage::Tick;
                }
                FixedStage::Tick => {
                    let stepped = {
                        let clocks = sub_app.world.resource_mut::<crate::time::EngineClocks>();
                        let stepped = clocks.fixed_mut().expend();
                        if stepped {
                            clocks.sync_default();
                        }
                        stepped
                    };
                    if stepped {
                        self.stepping.reset();
                        self.stage = FixedStage::Phase(0);
                    } else {
                        let clocks = sub_app.world.resource_mut::<crate::time::EngineClocks>();
                        clocks.set_source(DefaultSource::Virtual);
                        clocks.sync_default();
                        self.stepping.reset();
                        self.stage = FixedStage::After;
                    }
                }
                FixedStage::Phase(index) => {
                    let report = match index {
                        0 => step_schedule(&mut sub_app.world, FixedFirst, &mut self.stepping),
                        1 => step_schedule(&mut sub_app.world, FixedPreUpdate, &mut self.stepping),
                        2 => step_schedule(&mut sub_app.world, FixedUpdate, &mut self.stepping),
                        3 => step_schedule(&mut sub_app.world, FixedPostUpdate, &mut self.stepping),
                        4 => step_schedule(&mut sub_app.world, FixedLast, &mut self.stepping),
                        _ => unreachable!(),
                    };
                    if !matches!(report, StepReport::FrameBoundary) {
                        return Some(report);
                    }
                    self.stepping.reset();
                    self.stage = if index == 4 {
                        FixedStage::Tick
                    } else {
                        FixedStage::Phase(index + 1)
                    };
                }
                FixedStage::After => {
                    let report =
                        step_schedule(&mut sub_app.world, AfterFixedMainLoop, &mut self.stepping);
                    if !matches!(report, StepReport::FrameBoundary) {
                        return Some(report);
                    }
                    self.stage = FixedStage::Done;
                }
                FixedStage::Done => return None,
            }
        }
    }

    fn continue_frame(
        &mut self,
        sub_app: &mut SubApp,
        app: FrameApp,
        frame_phase: usize,
        breakpoints: &BTreeSet<FrameSystem>,
    ) -> (Vec<usize>, Option<usize>, bool) {
        let mut ran = Vec::new();
        loop {
            match self.stage {
                FixedStage::Before => {
                    let (systems, stop) = self.continue_schedule(
                        sub_app,
                        BeforeFixedMainLoop,
                        app,
                        frame_phase,
                        breakpoints,
                    );
                    ran.extend(systems);
                    if let Some(node) = stop {
                        return (ran, Some(node), false);
                    }
                    self.stepping.reset();
                    if sub_app
                        .world
                        .get_resource::<crate::time::EngineClocks>()
                        .is_none()
                    {
                        self.stage = FixedStage::Done;
                        continue;
                    }
                    sub_app
                        .world
                        .resource_mut::<crate::time::EngineClocks>()
                        .set_source(DefaultSource::Fixed);
                    self.stage = FixedStage::Tick;
                }
                FixedStage::Tick => {
                    let stepped = {
                        let clocks = sub_app.world.resource_mut::<crate::time::EngineClocks>();
                        let stepped = clocks.fixed_mut().expend();
                        if stepped {
                            clocks.sync_default();
                        }
                        stepped
                    };
                    if stepped {
                        self.stepping.reset();
                        self.stage = FixedStage::Phase(0);
                    } else {
                        let clocks = sub_app.world.resource_mut::<crate::time::EngineClocks>();
                        clocks.set_source(DefaultSource::Virtual);
                        clocks.sync_default();
                        self.stepping.reset();
                        self.stage = FixedStage::After;
                    }
                }
                FixedStage::Phase(index) => {
                    let (systems, stop) = match index {
                        0 => self.continue_schedule(
                            sub_app,
                            FixedFirst,
                            app,
                            frame_phase,
                            breakpoints,
                        ),
                        1 => self.continue_schedule(
                            sub_app,
                            FixedPreUpdate,
                            app,
                            frame_phase,
                            breakpoints,
                        ),
                        2 => self.continue_schedule(
                            sub_app,
                            FixedUpdate,
                            app,
                            frame_phase,
                            breakpoints,
                        ),
                        3 => self.continue_schedule(
                            sub_app,
                            FixedPostUpdate,
                            app,
                            frame_phase,
                            breakpoints,
                        ),
                        4 => self.continue_schedule(
                            sub_app,
                            FixedLast,
                            app,
                            frame_phase,
                            breakpoints,
                        ),
                        _ => unreachable!(),
                    };
                    ran.extend(systems);
                    if let Some(node) = stop {
                        return (ran, Some(node), false);
                    }
                    self.stepping.reset();
                    self.stage = if index == 4 {
                        FixedStage::Tick
                    } else {
                        FixedStage::Phase(index + 1)
                    };
                }
                FixedStage::After => {
                    let (systems, stop) = self.continue_schedule(
                        sub_app,
                        AfterFixedMainLoop,
                        app,
                        frame_phase,
                        breakpoints,
                    );
                    ran.extend(systems);
                    if let Some(node) = stop {
                        return (ran, Some(node), false);
                    }
                    self.stage = FixedStage::Done;
                }
                FixedStage::Done => return (ran, None, true),
            }
        }
    }

    fn continue_schedule<L: prism_ecs::schedule::ScheduleLabel + Clone>(
        &mut self,
        sub_app: &mut SubApp,
        label: L,
        app: FrameApp,
        phase: usize,
        breakpoints: &BTreeSet<FrameSystem>,
    ) -> (Vec<usize>, Option<usize>) {
        self.stepping.clear_breakpoints();
        for breakpoint in breakpoints {
            if breakpoint.app == app && breakpoint.phase == phase {
                self.stepping.add_breakpoint(breakpoint.node);
            }
        }
        let report: ContinueReport =
            continue_schedule(&mut sub_app.world, label, &mut self.stepping);
        let stop = match report.stop {
            ContinueStop::Breakpoint(node) => Some(node),
            ContinueStop::FrameBoundary => None,
        };
        (report.ran, stop)
    }
}

impl Driver {
    fn step(&mut self, sub_app: &mut SubApp) -> Option<(usize, StepReport)> {
        loop {
            if self.done {
                return None;
            }
            if self.phase >= sub_app.frame_phase_len() {
                self.done = true;
                return None;
            }
            if sub_app.frame_phase_is_fixed_driver(self.phase) {
                let fixed = self.fixed.get_or_insert_with(FixedDriver::new);
                if let Some(report) = fixed.step(sub_app) {
                    return Some((self.phase, report));
                }
                self.fixed = None;
                self.phase += 1;
                self.stepping.reset();
                continue;
            }
            let report = sub_app.step_frame_phase(self.phase, &mut self.stepping);
            if matches!(report, StepReport::FrameBoundary) {
                self.phase += 1;
                self.stepping.reset();
                continue;
            }
            return Some((self.phase, report));
        }
    }

    fn continue_frame(
        &mut self,
        sub_app: &mut SubApp,
        app: FrameApp,
        breakpoints: &BTreeSet<FrameSystem>,
    ) -> DriverContinue {
        let mut ran = Vec::new();
        loop {
            if self.done || self.phase >= sub_app.frame_phase_len() {
                self.done = true;
                return DriverContinue {
                    ran,
                    breakpoint: None,
                    done: true,
                };
            }
            if sub_app.frame_phase_is_fixed_driver(self.phase) {
                let fixed = self.fixed.get_or_insert_with(FixedDriver::new);
                let (systems, stop, done) =
                    fixed.continue_frame(sub_app, app, self.phase, breakpoints);
                ran.extend(systems.into_iter().map(|node| (self.phase, node)));
                if let Some(node) = stop {
                    return DriverContinue {
                        ran,
                        breakpoint: Some((self.phase, node)),
                        done: false,
                    };
                }
                if done {
                    self.fixed = None;
                    self.phase += 1;
                    self.stepping.reset();
                }
                continue;
            }
            self.stepping.clear_breakpoints();
            for breakpoint in breakpoints {
                if breakpoint.app == app && breakpoint.phase == self.phase {
                    self.stepping.add_breakpoint(breakpoint.node);
                }
            }
            let report = sub_app.continue_frame_phase(self.phase, &mut self.stepping);
            ran.extend(report.ran.into_iter().map(|node| (self.phase, node)));
            match report.stop {
                ContinueStop::Breakpoint(node) => {
                    return DriverContinue {
                        ran,
                        breakpoint: Some((self.phase, node)),
                        done: false,
                    };
                }
                ContinueStop::FrameBoundary => {
                    self.phase += 1;
                    self.stepping.reset();
                }
            }
        }
    }
}

/// A live complete-frame session. It is intentionally not `Clone`: it owns
/// mutable ECS condition caches and schedule cursors.
pub(crate) struct FrameSession {
    main: Driver,
    secondary: Option<(usize, Driver)>,
    next_secondary: usize,
    breakpoints: BTreeSet<FrameSystem>,
    complete: bool,
    aborted: bool,
}

impl FrameSession {
    pub(crate) fn start(app: &mut App, delta: prism_time::Duration) -> Self {
        app.sync_sub_apps();
        app.run_frame_boundary_hooks();
        app.set_time_update_strategy(crate::TimeUpdateStrategy::ManualDelta(delta));
        advance_time(app.world_mut());
        Self {
            main: Driver::default(),
            secondary: None,
            next_secondary: 0,
            breakpoints: BTreeSet::new(),
            complete: false,
            aborted: false,
        }
    }

    pub(crate) fn add_breakpoint(&mut self, breakpoint: FrameSystem) -> bool {
        self.breakpoints.insert(breakpoint)
    }

    pub(crate) fn remove_breakpoint(&mut self, breakpoint: FrameSystem) -> bool {
        self.breakpoints.remove(&breakpoint)
    }

    pub(crate) fn abort(&mut self) {
        self.aborted = true;
        self.secondary = None;
    }

    pub(crate) fn is_complete(&self) -> bool {
        self.complete
    }

    fn map_report(app: FrameApp, phase: usize, report: StepReport) -> FrameStepReport {
        match report {
            StepReport::Ran(node) => FrameStepReport::Ran(FrameSystem { app, phase, node }),
            StepReport::Skipped(node) => FrameStepReport::Skipped(FrameSystem { app, phase, node }),
            StepReport::Failed(node) => FrameStepReport::Failed(FrameSystem { app, phase, node }),
            StepReport::FrameBoundary => FrameStepReport::Complete,
        }
    }

    fn drive_one(&mut self, app: &mut App) -> FrameStepReport {
        if self.aborted || self.complete {
            return FrameStepReport::Complete;
        }
        let secondary_step = if let Some((index, driver)) = self.secondary.as_mut() {
            let sub_app = app
                .sub_apps_mut()
                .secondary_mut_at(*index)
                .expect("secondary frame driver index");
            driver
                .step(sub_app)
                .map(|(phase, report)| (*index, phase, report))
        } else {
            None
        };
        if let Some((index, phase, report)) = secondary_step {
            return Self::map_report(FrameApp::Secondary(index), phase, report);
        }
        if self.secondary.is_some() {
            self.secondary = None;
            self.next_secondary += 1;
        } else if let Some((phase, report)) = self.main.step(&mut app.sub_apps_mut().main) {
            return Self::map_report(FrameApp::Main, phase, report);
        }
        let secondary_len = app.sub_apps_mut().secondary_len();
        if self.next_secondary < secondary_len {
            let index = self.next_secondary;
            app.sub_apps_mut().run_extract_at(index);
            if let Some(sub_app) = app.sub_apps_mut().secondary_mut_at(index) {
                advance_time(&mut sub_app.world);
            }
            self.secondary = Some((index, Driver::default()));
            return FrameStepReport::Extracted(index);
        }
        app.sync_sub_apps();
        self.complete = true;
        FrameStepReport::Complete
    }

    pub(crate) fn step(&mut self, app: &mut App) -> FrameStepReport {
        self.drive_one(app)
    }

    pub(crate) fn continue_frame(&mut self, app: &mut App) -> FrameContinueReport {
        let breakpoints = self.breakpoints.clone();
        let mut ran = Vec::new();
        loop {
            if self.aborted || self.complete {
                return FrameContinueReport {
                    ran,
                    stop: FrameContinueStop::Complete,
                };
            }
            if let Some((index, driver)) = self.secondary.as_mut() {
                let sub_app = app
                    .sub_apps_mut()
                    .secondary_mut_at(*index)
                    .expect("secondary frame driver index");
                let result =
                    driver.continue_frame(sub_app, FrameApp::Secondary(*index), &breakpoints);
                ran.extend(result.ran.into_iter().map(|(phase, node)| FrameSystem {
                    app: FrameApp::Secondary(*index),
                    phase,
                    node,
                }));
                if let Some((phase, node)) = result.breakpoint {
                    return FrameContinueReport {
                        ran,
                        stop: FrameContinueStop::Breakpoint(FrameSystem {
                            app: FrameApp::Secondary(*index),
                            phase,
                            node,
                        }),
                    };
                }
                if result.done {
                    self.secondary = None;
                    self.next_secondary += 1;
                }
                continue;
            }
            let result = self.main.continue_frame(
                &mut app.sub_apps_mut().main,
                FrameApp::Main,
                &breakpoints,
            );
            ran.extend(result.ran.into_iter().map(|(phase, node)| FrameSystem {
                app: FrameApp::Main,
                phase,
                node,
            }));
            if let Some((phase, node)) = result.breakpoint {
                return FrameContinueReport {
                    ran,
                    stop: FrameContinueStop::Breakpoint(FrameSystem {
                        app: FrameApp::Main,
                        phase,
                        node,
                    }),
                };
            }
            if !result.done {
                continue;
            }
            let secondary_len = app.sub_apps_mut().secondary_len();
            if self.next_secondary < secondary_len {
                let index = self.next_secondary;
                app.sub_apps_mut().run_extract_at(index);
                if let Some(sub_app) = app.sub_apps_mut().secondary_mut_at(index) {
                    advance_time(&mut sub_app.world);
                }
                self.secondary = Some((index, Driver::default()));
                continue;
            }
            app.sync_sub_apps();
            self.complete = true;
            return FrameContinueReport {
                ran,
                stop: FrameContinueStop::Complete,
            };
        }
    }
}
