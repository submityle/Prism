//! Configuration hot-reload (design §24.2 "资产/配置热重载").
//!
//! A dev-time convenience: watch a config file and **re-apply it through the
//! [`User`](crate::settings::SettingsLayer::User) cascade layer** whenever its
//! contents change, so editing `user.cfg` takes effect without restarting the
//! app. The apply path is exactly [`App::load_user_config`] — the same
//! declarative `name value` parser the cold load uses — so a hot reload can
//! never diverge from a cold one.
//!
//! The watch is a **content hash**, not an OS filesystem notification: each
//! [`poll`](ConfigHotReload::poll) reads the file once and compares its hash to
//! the last applied contents, so an unchanged frame costs a single read and the
//! mechanism is identical on every platform (no watcher backend to fake). An
//! OS-watcher fast path can layer on later without changing this contract.

use std::path::{Path, PathBuf};

use prism_ecs::resource::Resource;

use crate::app::App;
use crate::cvar::ConsoleOutcome;

/// The result of a [`ConfigHotReload::poll`].
#[derive(Debug, Clone, PartialEq)]
pub enum ReloadOutcome {
    /// The file changed and its lines were applied to the `User` layer. Carries
    /// the per-line [`ConsoleOutcome`]s, exactly as a cold load would report.
    Applied(Vec<ConsoleOutcome>),
    /// The file is byte-identical to the last applied contents — nothing done.
    Unchanged,
    /// The file could not be read (missing, unreadable, or not UTF-8). Nothing
    /// was applied, and the watch keeps its previous contents so a transiently
    /// missing file does not trigger a spurious reload when it reappears.
    Unavailable,
}

/// Watches a config file and re-applies it when its contents change (design
/// §24.2).
///
/// Installed as a world resource by
/// [`App::enable_config_hot_reload`](crate::app::App::enable_config_hot_reload);
/// when present, [`App::update`](crate::app::App::update) polls it once per
/// frame.
#[derive(Debug, Clone)]
pub struct ConfigHotReload {
    path: PathBuf,
    /// FNV-1a hash of the contents last applied, or `None` before the first
    /// successful apply.
    last: Option<u64>,
}

impl Resource for ConfigHotReload {}

impl ConfigHotReload {
    /// Watch `path` (nothing is read until the first [`poll`](Self::poll)).
    #[must_use]
    pub fn new(path: impl Into<PathBuf>) -> Self {
        Self {
            path: path.into(),
            last: None,
        }
    }

    /// The watched file.
    #[must_use]
    pub fn path(&self) -> &Path {
        &self.path
    }

    /// The content hash last applied, or `None` if nothing has been applied.
    #[must_use]
    pub fn last_applied(&self) -> Option<u64> {
        self.last
    }

    /// Re-read the file and, if it changed since the last apply, apply it with
    /// [`App::load_user_config`] and report the per-line outcomes.
    pub fn poll(&mut self, app: &mut App) -> ReloadOutcome {
        let Ok(bytes) = std::fs::read(&self.path) else {
            return ReloadOutcome::Unavailable;
        };
        let hash = fnv1a(&bytes);
        if self.last == Some(hash) {
            return ReloadOutcome::Unchanged;
        }
        let Ok(text) = core::str::from_utf8(&bytes) else {
            return ReloadOutcome::Unavailable;
        };
        let outcomes = app.load_user_config(text);
        self.last = Some(hash);
        ReloadOutcome::Applied(outcomes)
    }
}

/// FNV-1a 64-bit hash — no external dependency, deterministic, `no_std`-clean.
fn fnv1a(bytes: &[u8]) -> u64 {
    let mut hash = 0xcbf2_9ce4_8422_2325_u64;
    for &byte in bytes {
        hash ^= u64::from(byte);
        hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
    }
    hash
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::app::App;
    use crate::cvar::{CvarBounds, CvarSpec};
    use std::sync::atomic::{AtomicU64, Ordering};

    static SEQ: AtomicU64 = AtomicU64::new(0);

    /// Write a uniquely-named temp config and return its path.
    fn temp_file(contents: &str) -> PathBuf {
        let seq = SEQ.fetch_add(1, Ordering::Relaxed);
        let mut path = std::env::temp_dir();
        path.push(format!(
            "prism_app_hot_reload_{}_{seq}.cfg",
            std::process::id()
        ));
        std::fs::write(&path, contents).expect("write temp config");
        path
    }

    fn app_with_shadows() -> App {
        let mut app = App::new();
        app.register_cvar(CvarSpec::new("r.shadows", 1_i64).bounds(CvarBounds::Int(0, 4)))
            .expect("register");
        app
    }

    #[test]
    fn first_poll_applies_and_a_repeat_is_unchanged() {
        let path = temp_file("r.shadows 3\n");
        let mut app = app_with_shadows();
        app.enable_config_hot_reload(&path);
        assert_eq!(app.cvar_int("r.shadows"), Some(1), "starts at the default");

        assert!(
            matches!(app.poll_config_hot_reload(), ReloadOutcome::Applied(_)),
            "the first poll applies the file"
        );
        assert_eq!(app.cvar_int("r.shadows"), Some(3));
        assert_eq!(app.poll_config_hot_reload(), ReloadOutcome::Unchanged);

        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn rewriting_the_file_reapplies() {
        let path = temp_file("r.shadows 3\n");
        let mut app = app_with_shadows();
        app.enable_config_hot_reload(&path);
        assert!(matches!(
            app.poll_config_hot_reload(),
            ReloadOutcome::Applied(_)
        ));
        assert_eq!(app.cvar_int("r.shadows"), Some(3));

        std::fs::write(&path, "r.shadows 0\n").expect("rewrite");
        assert!(matches!(
            app.poll_config_hot_reload(),
            ReloadOutcome::Applied(_)
        ));
        assert_eq!(app.cvar_int("r.shadows"), Some(0));

        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn missing_file_is_unavailable_and_keeps_state() {
        let mut path = std::env::temp_dir();
        path.push(format!(
            "prism_app_hot_reload_missing_{}.cfg",
            std::process::id()
        ));
        let _ = std::fs::remove_file(&path);

        let mut app = app_with_shadows();
        app.enable_config_hot_reload(&path);
        assert_eq!(app.poll_config_hot_reload(), ReloadOutcome::Unavailable);
        assert_eq!(
            app.config_hot_reload()
                .and_then(ConfigHotReload::last_applied),
            None,
            "nothing was applied"
        );
    }

    #[test]
    fn update_polls_the_watch_automatically() {
        let path = temp_file("r.shadows 2\n");
        let mut app = app_with_shadows();
        app.enable_config_hot_reload(&path);

        // A normal frame applies the change without the caller polling by hand.
        app.update();
        assert_eq!(app.cvar_int("r.shadows"), Some(2));

        let _ = std::fs::remove_file(&path);
    }
}
