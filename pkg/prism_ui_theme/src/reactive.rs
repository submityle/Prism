//! The dynamic, reactive theme.
//!
//! [`ReactiveTheme`] drives theming from a [`prism_ui_reactive::Signal`]:
//! switching appearance is a single [`ReactiveTheme::set_mode`] write. The
//! compiled theme is a [`Memo`], so it recomputes lazily and only when the mode
//! changes. Per-token memos ([`ReactiveTheme::token`],
//! [`ReactiveTheme::color`]) derive from that compiled memo and, thanks to the
//! runtime's value-change pruning, disturb their own dependents *only when the
//! token's resolved value actually changes* across the switch — so the cost of
//! a theme switch is proportional to the number of affected nodes, not the size
//! of the theme.

use alloc::rc::Rc;
use alloc::string::String;

use prism_ui_reactive::{Memo, Runtime, Signal};
use prism_ui_style::{Color, StyleError, StyleValue};

use crate::compile::{compile_theme, CompiledTheme};
use crate::logical::Direction;
use crate::theme::{ThemeDefinition, ThemeMode};

/// A theme bound to a reactive runtime.
///
/// Clones share the same signals and memo graph node, so a `ReactiveTheme` is a
/// cheap handle that can be passed into components.
///
/// # Example
///
/// ```
/// use prism_ui_reactive::Runtime;
/// use prism_ui_theme::{ReactiveTheme, ThemeDefinition, ThemeMode};
/// use prism_ui_style::Color;
///
/// let rt = Runtime::new();
/// let theme = ReactiveTheme::new(&rt, ThemeDefinition::studio(), ThemeMode::Light);
/// let surface = theme.color("color.surface");
///
/// assert_eq!(surface.get().unwrap(), Some(Color::rgba8(249, 250, 251, 255)));
/// theme.set_mode(ThemeMode::Dark);
/// assert_eq!(surface.get().unwrap(), Some(Color::rgba8(17, 24, 39, 255)));
/// ```
#[derive(Clone)]
pub struct ReactiveTheme {
    runtime: Runtime,
    mode: Signal<ThemeMode>,
    direction: Signal<Direction>,
    definition: Rc<ThemeDefinition>,
    compiled: Memo<Result<CompiledTheme, StyleError>>,
}

impl ReactiveTheme {
    /// Creates a reactive theme over `definition`, starting in `mode` with a
    /// left-to-right text direction.
    #[must_use]
    pub fn new(runtime: &Runtime, definition: ThemeDefinition, mode: ThemeMode) -> Self {
        let definition = Rc::new(definition);
        let mode = runtime.signal(mode);
        let direction = runtime.signal(Direction::Ltr);
        let compiled = {
            let definition = definition.clone();
            let mode = mode.clone();
            runtime.memo(move || compile_theme(&definition, &mode.get()))
        };
        Self {
            runtime: runtime.clone(),
            mode,
            direction,
            definition,
            compiled,
        }
    }

    /// Returns the runtime this theme is bound to.
    #[must_use]
    pub fn runtime(&self) -> &Runtime {
        &self.runtime
    }

    /// Returns the underlying definition.
    #[must_use]
    pub fn definition(&self) -> &ThemeDefinition {
        &self.definition
    }

    /// Reads the current mode, recording a dependency on the mode signal.
    #[must_use]
    pub fn mode(&self) -> ThemeMode {
        self.mode.get()
    }

    /// Reads the current mode without recording a dependency.
    #[must_use]
    pub fn mode_untracked(&self) -> ThemeMode {
        self.mode.get_untracked()
    }

    /// Switches the active mode. This is the single write that drives a theme
    /// switch; dependents recompute lazily on their next read.
    pub fn set_mode(&self, mode: ThemeMode) {
        self.mode.set(mode);
    }

    /// Reads the current text direction, recording a dependency.
    #[must_use]
    pub fn direction(&self) -> Direction {
        self.direction.get()
    }

    /// Sets the text direction (used by the RTL logical-property layer).
    pub fn set_direction(&self, direction: Direction) {
        self.direction.set(direction);
    }

    /// Returns the memo holding the whole compiled theme for the current mode.
    #[must_use]
    pub fn compiled(&self) -> Memo<Result<CompiledTheme, StyleError>> {
        self.compiled.clone()
    }

    /// Returns a memo for one token's resolved value in the current mode.
    ///
    /// The memo yields `Ok(None)` for an unknown token and propagates any
    /// resolution error. Because the memo compares its output, it only disturbs
    /// its dependents when this specific token's value changes.
    #[must_use]
    pub fn token(&self, name: impl Into<String>) -> Memo<Result<Option<StyleValue>, StyleError>> {
        let compiled = self.compiled.clone();
        let name = name.into();
        self.runtime.memo(move || {
            compiled.with(|result| match result {
                Ok(theme) => Ok(theme.get(&name).cloned()),
                Err(err) => Err(err.clone()),
            })
        })
    }

    /// Returns a memo for one token resolved as a [`Color`] in the current mode.
    #[must_use]
    pub fn color(&self, name: impl Into<String>) -> Memo<Result<Option<Color>, StyleError>> {
        let compiled = self.compiled.clone();
        let name = name.into();
        self.runtime.memo(move || {
            compiled.with(|result| match result {
                Ok(theme) => Ok(theme.color(&name)),
                Err(err) => Err(err.clone()),
            })
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use core::cell::RefCell;

    #[test]
    fn mode_switch_recomputes_compiled_theme() {
        let rt = Runtime::new();
        let theme = ReactiveTheme::new(&rt, ThemeDefinition::studio(), ThemeMode::Light);
        let surface = theme.color("color.surface");

        assert_eq!(
            surface.get().unwrap(),
            Some(Color::rgba8(249, 250, 251, 255))
        );
        theme.set_mode(ThemeMode::Dark);
        assert_eq!(surface.get().unwrap(), Some(Color::rgba8(17, 24, 39, 255)));
    }

    #[test]
    fn switch_only_wakes_dependents_whose_value_changed() {
        let rt = Runtime::new();
        let theme = ReactiveTheme::new(&rt, ThemeDefinition::studio(), ThemeMode::Light);

        let surface = theme.color("color.surface");
        let primary = theme.color("color.primary");

        let surface_runs = Rc::new(RefCell::new(0usize));
        let primary_runs = Rc::new(RefCell::new(0usize));

        let _surface_effect = rt.effect({
            let surface = surface.clone();
            let runs = surface_runs.clone();
            move || {
                let _ = surface.get();
                *runs.borrow_mut() += 1;
            }
        });
        let _primary_effect = rt.effect({
            let primary = primary.clone();
            let runs = primary_runs.clone();
            move || {
                let _ = primary.get();
                *runs.borrow_mut() += 1;
            }
        });

        // Both effects run once on creation.
        assert_eq!(*surface_runs.borrow(), 1);
        assert_eq!(*primary_runs.borrow(), 1);

        // Light -> Dark: surface changes, primary (constant brand color) does not.
        theme.set_mode(ThemeMode::Dark);

        assert_eq!(*surface_runs.borrow(), 2, "surface should recompute");
        assert_eq!(
            *primary_runs.borrow(),
            1,
            "primary is unchanged across light/dark and must not wake its dependent",
        );
    }

    #[test]
    fn high_contrast_switch_wakes_primary() {
        let rt = Runtime::new();
        let theme = ReactiveTheme::new(&rt, ThemeDefinition::studio(), ThemeMode::Light);
        let primary = theme.color("color.primary");

        let runs = Rc::new(RefCell::new(0usize));
        let _effect = rt.effect({
            let primary = primary.clone();
            let runs = runs.clone();
            move || {
                let _ = primary.get();
                *runs.borrow_mut() += 1;
            }
        });

        assert_eq!(*runs.borrow(), 1);
        // High contrast deepens the brand color, so the dependent must wake.
        theme.set_mode(ThemeMode::HighContrast);
        assert_eq!(*runs.borrow(), 2);
    }

    #[test]
    fn token_memo_reports_unknown_as_none() {
        let rt = Runtime::new();
        let theme = ReactiveTheme::new(&rt, ThemeDefinition::studio(), ThemeMode::Light);
        let ghost = theme.token("color.ghost");
        assert_eq!(ghost.get().unwrap(), None);
    }

    #[test]
    fn clones_share_mode_state() {
        let rt = Runtime::new();
        let theme = ReactiveTheme::new(&rt, ThemeDefinition::studio(), ThemeMode::Light);
        let clone = theme.clone();
        theme.set_mode(ThemeMode::Dark);
        assert_eq!(clone.mode_untracked(), ThemeMode::Dark);
    }

    #[test]
    fn direction_signal_round_trips() {
        let rt = Runtime::new();
        let theme = ReactiveTheme::new(&rt, ThemeDefinition::studio(), ThemeMode::Light);
        assert_eq!(theme.direction(), Direction::Ltr);
        theme.set_direction(Direction::Rtl);
        assert_eq!(theme.direction(), Direction::Rtl);
    }
}
