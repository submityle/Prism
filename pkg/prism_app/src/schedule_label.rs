//! Stable, hashable identities for the labeled schedules an [`App`] runs.
//!
//! A running engine owns *many* [`Schedule`](prism_ecs::schedule::Schedule)s, each keyed
//! by a **label**: `Startup`, `Update`, `Last`, and (in later milestones)
//! user-defined phases. A label is any `'static` value that can produce a
//! stable [`ScheduleLabelId`]; the engine stores schedules in a map keyed by
//! that id (see [`Schedules`](crate::schedules::Schedules)).
//!
//! [`App`]: crate::app::App
//!
//! # Honestly deferred
//!
//! The full main-frame order from the design doc (§7) also includes
//! `RunFixedMainLoop` (fixed-timestep inner loop, needs `prism_time`) and
//! `StateTransition` (the state machine, §11). Those phases are **not** part of
//! M0 and are intentionally absent from [`CoreSchedule`] rather than stubbed as
//! empty variants — they arrive with M1/M2 together with the machinery that
//! gives them meaning.

use core::any::TypeId;

/// A stable, hashable identity for one schedule label.
///
/// Two labels compare equal iff they come from the same label *type* and carry
/// the same discriminant. This lets any `'static` type (an enum of engine
/// phases, a unit struct for a user phase, …) act as a schedule key without a
/// central registry.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct ScheduleLabelId {
    type_id: TypeId,
    discriminant: u64,
    name: &'static str,
}

impl ScheduleLabelId {
    /// Build an id from the label type `L`, a per-variant `discriminant`, and a
    /// human-readable `name` (used in diagnostics).
    #[inline]
    pub fn new<L: 'static>(discriminant: u64, name: &'static str) -> Self {
        Self {
            type_id: TypeId::of::<L>(),
            discriminant,
            name,
        }
    }

    /// The human-readable label name, for diagnostics and error messages.
    #[inline]
    pub fn name(&self) -> &'static str {
        self.name
    }
}

/// A type that names a [`Schedule`](prism_ecs::schedule::Schedule) slot in an
/// [`App`](crate::app::App).
///
/// Implementors must return a *stable* [`ScheduleLabelId`]: the same logical
/// label must always produce the same id within a process run.
pub trait ScheduleLabel: 'static {
    /// The stable identity of this label.
    fn id(&self) -> ScheduleLabelId;
}

/// The built-in schedules an [`App`](crate::app::App) installs and drives.
///
/// M0 ships the startup trio plus the variable-step main-frame subset. The
/// per-frame order actually driven by [`SubApp::update`](crate::sub_app::SubApp::update)
/// is `First → PreUpdate → Update → PostUpdate → Last`; the startup trio
/// (`PreStartup → Startup → PostStartup`) runs exactly once before the first
/// frame.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum CoreSchedule {
    /// Runs once, first of all, before [`Startup`](CoreSchedule::Startup).
    PreStartup,
    /// Runs once at boot: one-time world setup (spawn initial entities, insert
    /// resources).
    Startup,
    /// Runs once, after [`Startup`](CoreSchedule::Startup).
    PostStartup,
    /// First phase of every frame.
    First,
    /// Before the main [`Update`](CoreSchedule::Update) work each frame.
    PreUpdate,
    /// The main per-frame, variable-step work (gameplay, input sampling,
    /// camera smoothing).
    Update,
    /// After the main [`Update`](CoreSchedule::Update) work each frame.
    PostUpdate,
    /// Final phase of every frame (frame cleanup / bookkeeping).
    Last,
}

impl CoreSchedule {
    /// The startup schedules, in run order. Driven exactly once by
    /// [`App::run`](crate::app::App::run) before the first frame.
    pub const STARTUP_ORDER: [CoreSchedule; 3] = [
        CoreSchedule::PreStartup,
        CoreSchedule::Startup,
        CoreSchedule::PostStartup,
    ];

    /// The per-frame schedules, in run order (M0 variable-step subset of design
    /// §7). `RunFixedMainLoop` and `StateTransition` are deferred to M1/M2.
    pub const FRAME_ORDER: [CoreSchedule; 5] = [
        CoreSchedule::First,
        CoreSchedule::PreUpdate,
        CoreSchedule::Update,
        CoreSchedule::PostUpdate,
        CoreSchedule::Last,
    ];

    #[inline]
    fn discriminant(self) -> u64 {
        self as u64
    }

    #[inline]
    fn as_str(self) -> &'static str {
        match self {
            CoreSchedule::PreStartup => "PreStartup",
            CoreSchedule::Startup => "Startup",
            CoreSchedule::PostStartup => "PostStartup",
            CoreSchedule::First => "First",
            CoreSchedule::PreUpdate => "PreUpdate",
            CoreSchedule::Update => "Update",
            CoreSchedule::PostUpdate => "PostUpdate",
            CoreSchedule::Last => "Last",
        }
    }
}

impl ScheduleLabel for CoreSchedule {
    #[inline]
    fn id(&self) -> ScheduleLabelId {
        ScheduleLabelId::new::<CoreSchedule>(self.discriminant(), self.as_str())
    }
}
