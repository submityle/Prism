//! Gallery instances for the `motion` family. Dev-only; see `gallery/mod.rs`.

use alloc::vec::Vec;

use prism_ui::Element;
use prism_ui_component::mount_component;

use super::Showcase;
use crate::motion::{
    Collapse, CollapseProps, Fade, FadeProps, Scale, ScaleProps, Slide, SlideFrom, SlideProps,
    Transition, TransitionPhase, TransitionProps,
};

/// Real, named instances of every `motion` control.
///
/// Motion controls are externally driven; each is placed at a *visible* phase
/// (or `open`) with a real child so it paints a frame.
#[must_use]
pub fn instances() -> Vec<Showcase> {
    alloc::vec![
        Showcase::new(
            "Transition / Entered",
            mount_component(
                &Transition,
                TransitionProps::new()
                    .phase(TransitionPhase::Entered)
                    .child(Element::box_().child(Element::text("Transition"))),
            ),
        ),
        Showcase::new(
            "Fade / Entered",
            mount_component(
                &Fade,
                FadeProps::new()
                    .phase(TransitionPhase::Entered)
                    .child(Element::box_().child(Element::text("Fade"))),
            ),
        ),
        Showcase::new(
            "Slide / From Top",
            mount_component(
                &Slide,
                SlideProps::new()
                    .phase(TransitionPhase::Entered)
                    .from(SlideFrom::Top)
                    .child(Element::box_().child(Element::text("Slide"))),
            ),
        ),
        Showcase::new(
            "Scale / Entered",
            mount_component(
                &Scale,
                ScaleProps::new()
                    .phase(TransitionPhase::Entered)
                    .child(Element::box_().child(Element::text("Scale"))),
            ),
        ),
        Showcase::new(
            "Collapse / Open",
            mount_component(
                &Collapse,
                CollapseProps::new()
                    .open(true)
                    .child(Element::box_().child(Element::text("Collapse"))),
            ),
        ),
    ]
}
