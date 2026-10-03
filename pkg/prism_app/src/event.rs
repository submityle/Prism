//! Buffered-event registration on the [`App`] (design §22 M1: "Events update").
//!
//! An [`Event`] type is any `Send + Sync + 'static` message a system wants to
//! broadcast to other systems within a bounded window of frames. The storage,
//! double-buffering, and cursor logic all live in [`prism_ecs`] as
//! [`Events<E>`]; this module adds the single App-side entry point that both
//! installs the per-type [`Events<E>`] resource and schedules its once-per-frame
//! buffer rotation.
//!
//! # What [`App::add_event`] wires up
//!
//! 1. It inserts an empty [`Events<E>`] resource into the main world, so systems
//!    can take [`Res<Events<E>>`](prism_ecs::system::Res) /
//!    [`ResMut<Events<E>>`](prism_ecs::system::ResMut) to read and send events.
//! 2. It registers an exclusive-free update system into the [`First`] phase that
//!    calls [`Events::update`] once per frame. That call rotates the double
//!    buffer: events sent on frame *N* stay readable through the end of frame
//!    *N + 1*, then are retired on the following [`First`]. This is the standard
//!    "one full frame of grace regardless of system ordering" guarantee.
//!
//! Running the rotation in [`First`] (the very first phase of each frame, design
//! §7) means every event is swept exactly once per frame before any user system
//! in `PreUpdate`/`Update`/… observes the world, keeping the retirement cadence
//! independent of where senders and readers sit in the frame.
//!
//! # Idempotence
//!
//! Calling [`App::add_event`] more than once for the same `E` is a no-op after
//! the first call: the type is tracked by [`TypeId`] so the resource is **not**
//! reinserted (which would silently discard already-buffered events) and the
//! update system is registered exactly once. This mirrors the deduplication
//! that [`App::insert_state`](crate::App::insert_state) performs for state
//! transition systems.
//!
//! [`Event`]: prism_ecs::event::Event
//! [`Events`]: prism_ecs::event::Events
//! [`Events<E>`]: prism_ecs::event::Events
//! [`Events::update`]: prism_ecs::event::Events::update
//! [`First`]: crate::schedule::First

use core::any::TypeId;

use prism_ecs::event::{Event, Events};
use prism_ecs::system::ResMut;

use crate::app::App;
use crate::schedule::First;

impl App {
    /// Register the buffered event type `E`.
    ///
    /// Installs an empty [`Events<E>`](prism_ecs::event::Events) resource and
    /// schedules its once-per-frame [`update`](prism_ecs::event::Events::update)
    /// rotation in the [`First`] phase. After this,
    /// systems can send events with
    /// [`ResMut<Events<E>>`](prism_ecs::system::ResMut) and read them with a
    /// cursor obtained from [`Res<Events<E>>`](prism_ecs::system::Res); every
    /// event stays readable for the frame it is sent and the frame after.
    ///
    /// Calling this repeatedly for the same `E` is safe and idempotent: the
    /// resource is created and the rotation system is scheduled only on the
    /// first call, so no buffered events are ever lost to re-registration.
    pub fn add_event<E: Event>(&mut self) -> &mut Self {
        if self.added_events.insert(TypeId::of::<E>()) {
            self.insert_resource(Events::<E>::new());
            self.add_systems(First, |mut events: ResMut<Events<E>>| events.update());
        }
        self
    }
}
