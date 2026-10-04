//! Console variables (`cvar`s) with declared schema and validation (design
//! §24.6, §25.3).
//!
//! A shipping engine exposes a flat namespace of **named, runtime-tunable
//! variables** — Quake/Source-style `cvar`s such as `r.shadows 2` — that can be
//! poked from a console, a config file, or the command line. Unlike a free-form
//! [`Settings`] key, every `cvar` is *declared up front* with a schema: a
//! category, a typed default, optional numeric bounds, and a set of
//! [`CvarFlags`] (cheat protection, read-only, archive, change-notify). The
//! design calls for exactly this (§24.6 "cvar：运行时可改的命名变量（`r.shadows
//! 2`），分类（渲染/网络/调试），带默认/范围/权限（作弊保护），控制台/配置文件/
//! 命令行可设" and "cvar 变更可触发回调").
//!
//! This module layers that **declared, validated front door** on top of the
//! existing [`Settings`] cascade rather than duplicating storage:
//!
//! * [`CvarRegistry`] is a [`Resource`] holding each declared cvar's immutable
//!   schema ([`Cvar`]) plus a process-wide [cheats-enabled](CvarRegistry::cheats_enabled)
//!   flag.
//! * Validation ([`CvarRegistry::validate_set`]) is a pure function: it checks
//!   the cvar is registered, coerces/checks the value against the declared
//!   type, enforces [read-only](CvarFlags::READ_ONLY) and
//!   [cheat](CvarFlags::CHEAT) permissions, and **clamps** numeric values into
//!   their declared [bounds](CvarBounds). Illegal input (wrong type, a blocked
//!   write) is *rejected at the boundary with a [`CvarError`]*, never panics
//!   (design §25.3 "非法输入按安全边界拒绝而非崩溃").
//! * The actual value lives in the [`Settings`] cascade: registering a cvar
//!   seeds its default into the [`EngineDefault`](crate::settings::SettingsLayer::EngineDefault)
//!   layer, a console/runtime write lands in the
//!   [`Runtime`](crate::settings::SettingsLayer::Runtime) layer, and the usual
//!   *default → platform → user → command line → runtime* precedence resolves
//!   the effective value. So a cvar is just a Settings key with a validated,
//!   documented schema in front of it, and lower-layer overrides (platform
//!   tier, user prefs) compose with cvar writes exactly as the cascade dictates.
//! * A change to a cvar's resolved value broadcasts the existing
//!   [`SettingChanged`](crate::settings::SettingChanged) event, and — for cvars
//!   that opt in with [`CvarFlags::NOTIFY`] — also a dedicated [`CvarChanged`]
//!   event carrying the cvar name, category, and old/new values, so a
//!   subsystem can react (e.g. rebuild the swap chain when `r.vsync` flips).
//!
//! The [`App`] helpers ([`register_cvar`](App::register_cvar),
//! [`set_cvar`](App::set_cvar), [`reset_cvar`](App::reset_cvar), the
//! [`cvar`](App::cvar) getters, and [`set_cheats_enabled`](App::set_cheats_enabled))
//! wire the registry, the settings cascade, and the two events together.
//!
//! # Deterministic iteration
//!
//! The registry is a [`BTreeMap`], so [`iter`](CvarRegistry::iter) and
//! [`iter_category`](CvarRegistry::iter_category) walk cvars in stable
//! ascending-name order regardless of registration order or hash seeding —
//! the same determinism contract the settings store keeps (design §15).
//!
//! # Honestly deferred: `reflect`-backed schema
//!
//! Design §25.3 routes cvar range-clamping / defaults / rename compatibility
//! through `prism_reflect` attribute metadata once that lands. This module
//! declares the same schema **imperatively** at registration (a real,
//! self-contained mechanism — not a fake of reflect), and the reflect-attribute
//! path is documented as absent, to be layered on without changing these
//! validation semantics. Nothing here is stubbed.
//!
//! [`Resource`]: prism_ecs::resource::Resource
//! [`Settings`]: crate::settings::Settings
//! [`BTreeMap`]: std::collections::BTreeMap

use std::collections::BTreeMap;
use std::error::Error;
use std::fmt;

use prism_ecs::event::Event;
use prism_ecs::resource::Resource;

use crate::app::App;
use crate::settings::{SettingValue, Settings, SettingsLayer};

/// A cvar's subsystem category (design §24.6 "分类（渲染/网络/调试）").
///
/// Categories group cvars for console listing / filtering and documentation;
/// they carry no behaviour of their own. [`General`](CvarCategory::General) is
/// the catch-all default.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
pub enum CvarCategory {
    /// Rendering / graphics (`r.*`).
    Render,
    /// Audio / mixing (`s.*`).
    Audio,
    /// Networking / replication (`net.*`).
    Network,
    /// Input / bindings (`input.*`).
    Input,
    /// Physics / simulation (`phys.*`).
    Physics,
    /// Debug / developer tools (`debug.*`).
    Debug,
    /// Gameplay / rules (`g.*`).
    Gameplay,
    /// Engine / system (`sys.*`).
    System,
    /// Uncategorised — the default.
    #[default]
    General,
}

impl CvarCategory {
    /// A short, stable lowercase label for console listing and logs.
    #[must_use]
    pub fn as_str(self) -> &'static str {
        match self {
            CvarCategory::Render => "render",
            CvarCategory::Audio => "audio",
            CvarCategory::Network => "network",
            CvarCategory::Input => "input",
            CvarCategory::Physics => "physics",
            CvarCategory::Debug => "debug",
            CvarCategory::Gameplay => "gameplay",
            CvarCategory::System => "system",
            CvarCategory::General => "general",
        }
    }
}

impl fmt::Display for CvarCategory {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

/// Permission / behaviour flags on a cvar (design §24.6 "权限（作弊保护）",
/// "cvar 变更可触发回调").
///
/// A small bit set, combinable with `|`. The flags are:
///
/// * [`CHEAT`](CvarFlags::CHEAT) — the cvar may only be changed while cheats are
///   enabled ([`CvarRegistry::cheats_enabled`]); otherwise a runtime write is
///   rejected with [`CvarError::CheatProtected`]. Reads are always allowed.
/// * [`READ_ONLY`](CvarFlags::READ_ONLY) — the cvar is informational: it takes
///   its registered default and can never be written at runtime
///   ([`CvarError::ReadOnly`]).
/// * [`ARCHIVE`](CvarFlags::ARCHIVE) — the cvar's user-set value should be
///   persisted to the user config (a hint for a config writer; this crate does
///   not perform file I/O, see [`archived`](CvarRegistry::archived)).
/// * [`NOTIFY`](CvarFlags::NOTIFY) — a resolved-value change additionally
///   broadcasts a [`CvarChanged`] event (on top of the always-sent
///   [`SettingChanged`](crate::settings::SettingChanged)).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub struct CvarFlags(u8);

impl CvarFlags {
    /// No flags set.
    pub const EMPTY: CvarFlags = CvarFlags(0);
    /// Changing requires cheats enabled.
    pub const CHEAT: CvarFlags = CvarFlags(1 << 0);
    /// Cannot be written at runtime.
    pub const READ_ONLY: CvarFlags = CvarFlags(1 << 1);
    /// Should be persisted to the user config.
    pub const ARCHIVE: CvarFlags = CvarFlags(1 << 2);
    /// Broadcast a [`CvarChanged`] event on change.
    pub const NOTIFY: CvarFlags = CvarFlags(1 << 3);

    /// Whether every bit in `other` is set in `self`.
    #[must_use]
    pub const fn contains(self, other: CvarFlags) -> bool {
        self.0 & other.0 == other.0
    }

    /// Whether no flags are set.
    #[must_use]
    pub const fn is_empty(self) -> bool {
        self.0 == 0
    }

    /// The union of two flag sets.
    #[must_use]
    pub const fn union(self, other: CvarFlags) -> CvarFlags {
        CvarFlags(self.0 | other.0)
    }

    /// The raw bits, for stable serialisation / logging.
    #[must_use]
    pub const fn bits(self) -> u8 {
        self.0
    }
}

impl core::ops::BitOr for CvarFlags {
    type Output = CvarFlags;
    fn bitor(self, rhs: CvarFlags) -> CvarFlags {
        self.union(rhs)
    }
}

impl core::ops::BitOrAssign for CvarFlags {
    fn bitor_assign(&mut self, rhs: CvarFlags) {
        self.0 |= rhs.0;
    }
}

/// Declared numeric bounds for a cvar (design §24.6 "范围").
///
/// Bounds apply only to numeric cvars and are enforced by **clamping** on every
/// write (design §25.3 "范围钳制"): a value below `min` becomes `min`, above
/// `max` becomes `max`. The bound kind must match the cvar's default kind
/// (an [`Int`](CvarBounds::Int) bound on an [`Int`](SettingValue::Int) default,
/// a [`Float`](CvarBounds::Float) bound on a [`Float`](SettingValue::Float)
/// default); a mismatch is rejected at registration with
/// [`CvarError::InvalidBounds`]. [`None`](CvarBounds::None) leaves a cvar
/// unbounded (the only legal choice for `bool` / string cvars).
#[derive(Clone, Copy, Debug, PartialEq, Default)]
pub enum CvarBounds {
    /// No bounds — any value of the declared type is accepted.
    #[default]
    None,
    /// Inclusive integer bounds `[min, max]`.
    Int(i64, i64),
    /// Inclusive float bounds `[min, max]`.
    Float(f64, f64),
}

impl CvarBounds {
    /// Whether `min <= max` (vacuously true for [`None`](CvarBounds::None)).
    #[must_use]
    fn is_well_formed(self) -> bool {
        match self {
            CvarBounds::None => true,
            CvarBounds::Int(min, max) => min <= max,
            // `<=` on NaN is false, so a NaN bound is reported ill-formed.
            CvarBounds::Float(min, max) => min <= max,
        }
    }

    /// Clamp `value` into these bounds, returning the clamped value and whether
    /// clamping altered it. Non-matching value/bound kinds pass through
    /// unchanged (kind agreement is enforced earlier by validation).
    #[must_use]
    fn clamp(self, value: SettingValue) -> (SettingValue, bool) {
        match (self, &value) {
            (CvarBounds::Int(min, max), SettingValue::Int(v)) => {
                let clamped = (*v).clamp(min, max);
                (SettingValue::Int(clamped), clamped != *v)
            }
            (CvarBounds::Float(min, max), SettingValue::Float(v)) => {
                // `f64::clamp` panics if min > max; well-formedness is checked
                // at registration, and NaN inputs are rejected by validation
                // before reaching here.
                let clamped = v.clamp(min, max);
                (SettingValue::Float(clamped), clamped != *v)
            }
            _ => (value, false),
        }
    }
}

/// The declared schema of one cvar (design §24.6).
///
/// Immutable after [registration](CvarRegistry::register); the live value lives
/// in the [`Settings`] cascade, not here.
#[derive(Clone, Debug, PartialEq)]
pub struct Cvar {
    category: CvarCategory,
    default: SettingValue,
    bounds: CvarBounds,
    flags: CvarFlags,
    description: String,
}

impl Cvar {
    /// The cvar's subsystem [category](CvarCategory).
    #[must_use]
    pub fn category(&self) -> CvarCategory {
        self.category
    }

    /// The registered default value (the
    /// [`EngineDefault`](crate::settings::SettingsLayer::EngineDefault)
    /// contribution).
    #[must_use]
    pub fn default_value(&self) -> &SettingValue {
        &self.default
    }

    /// The declared numeric [bounds](CvarBounds).
    #[must_use]
    pub fn bounds(&self) -> CvarBounds {
        self.bounds
    }

    /// The permission / behaviour [flags](CvarFlags).
    #[must_use]
    pub fn flags(&self) -> CvarFlags {
        self.flags
    }

    /// The human-readable description (may be empty).
    #[must_use]
    pub fn description(&self) -> &str {
        &self.description
    }

    /// The [`SettingValue`] variant kind this cvar accepts, as a stable label.
    #[must_use]
    pub fn kind(&self) -> &'static str {
        value_kind(&self.default)
    }
}

/// A builder describing a cvar to [register](CvarRegistry::register) (design
/// §24.6).
///
/// Start from [`CvarSpec::new`] with a name and typed default, then layer on a
/// [category](CvarSpec::category), [bounds](CvarSpec::bounds), [flags](CvarSpec::flag),
/// and a [description](CvarSpec::description). The default's [`SettingValue`]
/// kind *is* the cvar's type: all later writes must be of (or coercible to)
/// that kind.
#[derive(Clone, Debug)]
pub struct CvarSpec {
    name: String,
    category: CvarCategory,
    default: SettingValue,
    bounds: CvarBounds,
    flags: CvarFlags,
    description: String,
}

impl CvarSpec {
    /// A new spec for `name` with the given typed `default`.
    ///
    /// The default's kind fixes the cvar's type. Category defaults to
    /// [`General`](CvarCategory::General), bounds to [`None`](CvarBounds::None),
    /// flags to empty, and the description to empty.
    #[must_use]
    pub fn new(name: impl Into<String>, default: impl Into<SettingValue>) -> Self {
        CvarSpec {
            name: name.into(),
            category: CvarCategory::General,
            default: default.into(),
            bounds: CvarBounds::None,
            flags: CvarFlags::EMPTY,
            description: String::new(),
        }
    }

    /// Set the [category](CvarCategory).
    #[must_use]
    pub fn category(mut self, category: CvarCategory) -> Self {
        self.category = category;
        self
    }

    /// Set numeric [bounds](CvarBounds) (clamped on every write).
    #[must_use]
    pub fn bounds(mut self, bounds: CvarBounds) -> Self {
        self.bounds = bounds;
        self
    }

    /// Add one or more [flags](CvarFlags) (unioned with any already set).
    #[must_use]
    pub fn flag(mut self, flags: CvarFlags) -> Self {
        self.flags |= flags;
        self
    }

    /// Set the human-readable description.
    #[must_use]
    pub fn description(mut self, description: impl Into<String>) -> Self {
        self.description = description.into();
        self
    }

    /// The cvar name being described.
    #[must_use]
    pub fn name(&self) -> &str {
        &self.name
    }
}

/// A resolved-value change to a cvar, broadcast when a
/// [`NOTIFY`](CvarFlags::NOTIFY) cvar changes (design §24.6 "cvar 变更可触发
/// 回调").
///
/// This is additional to the always-sent
/// [`SettingChanged`](crate::settings::SettingChanged): it is scoped to
/// *declared* cvars that opted into notification, and carries the cvar
/// [category](CvarCategory) so a listener can filter (e.g. react only to
/// `render` cvars).
#[derive(Clone, Debug, PartialEq)]
pub struct CvarChanged {
    /// The cvar name.
    pub name: String,
    /// The cvar's category.
    pub category: CvarCategory,
    /// The resolved value before the change, or `None` if previously unset.
    pub previous: Option<SettingValue>,
    /// The resolved value after the change.
    pub current: SettingValue,
}

impl Event for CvarChanged {}

/// Errors from registering or writing a cvar (design §25.3 "非法输入按安全边界
/// 拒绝而非崩溃").
///
/// Every mutating cvar entry point returns `Result<_, CvarError>` instead of
/// panicking, so a console / config parser can surface the problem to the user
/// and continue.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum CvarError {
    /// A cvar with this name is already registered.
    Redeclared(String),
    /// The spec's [bounds](CvarBounds) kind does not match the default's kind,
    /// or `min > max`.
    InvalidBounds(String),
    /// No cvar with this name is registered.
    Unregistered(String),
    /// The written value's type is incompatible with the cvar's declared type.
    TypeMismatch {
        /// The cvar name.
        name: String,
        /// The declared value kind.
        expected: &'static str,
        /// The written value kind.
        found: &'static str,
    },
    /// The cvar is [read-only](CvarFlags::READ_ONLY) and cannot be written.
    ReadOnly(String),
    /// The cvar is [cheat-protected](CvarFlags::CHEAT) and cheats are disabled.
    CheatProtected(String),
}

impl fmt::Display for CvarError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            CvarError::Redeclared(name) => {
                write!(f, "cvar `{name}` is already registered")
            }
            CvarError::InvalidBounds(name) => write!(
                f,
                "cvar `{name}` has bounds that do not match its default type or are inverted"
            ),
            CvarError::Unregistered(name) => {
                write!(f, "cvar `{name}` is not registered")
            }
            CvarError::TypeMismatch {
                name,
                expected,
                found,
            } => write!(
                f,
                "cvar `{name}` expects a {expected} value but got a {found}"
            ),
            CvarError::ReadOnly(name) => {
                write!(f, "cvar `{name}` is read-only")
            }
            CvarError::CheatProtected(name) => write!(
                f,
                "cvar `{name}` is cheat-protected and cheats are disabled"
            ),
        }
    }
}

impl Error for CvarError {}

/// The outcome of a successful cvar write (design §24.6).
///
/// Returned by [`App::set_cvar`] / [`App::reset_cvar`] so a console can report
/// what happened: whether the resolved value actually changed, whether the
/// input was [clamped](CvarBounds) into bounds, and the resulting resolved
/// value.
#[derive(Clone, Debug, PartialEq)]
pub struct CvarSetOutcome {
    /// Whether the resolved value changed as a result of the write.
    pub changed: bool,
    /// Whether the written value was clamped into the declared bounds.
    pub clamped: bool,
    /// The cvar's resolved value after the write.
    pub resolved: SettingValue,
}

/// The outcome of executing one console command line via
/// [`App::exec_console`] (design §24.6, a Quake/Source-style `r.shadows 2`
/// command line).
///
/// A console line is one of: a **query** (a bare cvar name, which reports the
/// current resolved value); a **write** (`name value`, which sets the cvar at
/// the [`Runtime`](crate::settings::SettingsLayer::Runtime) layer); or the
/// reserved **`reset`** command (`reset <name>` to revert one cvar's runtime
/// override, or a bare `reset` to revert them all). This enum captures every
/// outcome so a console front end can echo an accurate response without
/// panicking, mirroring the reject-at-the-boundary contract of [`CvarError`]
/// (design §25.3).
#[derive(Clone, Debug, PartialEq)]
pub enum ConsoleOutcome {
    /// The line was empty or a `//` comment, so nothing happened.
    Empty,
    /// A bare cvar name queried its current resolved value.
    Queried {
        /// The queried cvar name.
        name: String,
        /// The cvar's resolved value from the cascade.
        value: SettingValue,
    },
    /// A `name value` line wrote the cvar; carries the [`CvarSetOutcome`]
    /// describing whether the resolved value changed and whether the written
    /// value was clamped into the declared [bounds](CvarBounds).
    Set(CvarSetOutcome),
    /// The named cvar is not registered. The console is the *declared* front
    /// door (like the rest of this module), so an undeclared name is reported
    /// here rather than silently creating an untyped setting.
    Unknown(String),
    /// The write was rejected by validation: a type mismatch, a
    /// [read-only](CvarFlags::READ_ONLY) cvar, or a
    /// [cheat-protected](CvarFlags::CHEAT) cvar while cheats are disabled.
    Rejected(CvarError),
    /// The `reset <name>` command cleared one cvar's
    /// [`Runtime`](crate::settings::SettingsLayer::Runtime) override; carries
    /// the [`CvarSetOutcome`] describing whether the resolved value fell back to
    /// a lower cascade layer.
    Reset {
        /// The reset cvar's name.
        name: String,
        /// The outcome of clearing the runtime override.
        outcome: CvarSetOutcome,
    },
    /// The bare `reset` command cleared **every** cvar's
    /// [`Runtime`](crate::settings::SettingsLayer::Runtime) override; carries
    /// the ascending-name list of cvars whose resolved value actually changed.
    ResetAll {
        /// The names of the cvars whose resolved value changed, ascending.
        changed: Vec<String>,
    },
}

/// A read-only snapshot of one cvar, joining its immutable [schema](Cvar) with
/// its resolved value from the [`Settings`] cascade (design §24.6 "分类（渲染/
/// 网络/调试）" + console enumeration/help).
///
/// Produced by the console listing entry points [`App::list_cvars`],
/// [`App::list_cvars_in_category`], and [`App::find_cvars`]. It is an owned,
/// point-in-time copy — the live schema stays in [`CvarRegistry`] and the live
/// value in [`Settings`] — so a console / developer overlay can render a cvar
/// list without holding a borrow on the [`App`].
#[derive(Clone, Debug, PartialEq)]
pub struct CvarListing {
    /// The cvar's flat-namespace name (e.g. `r.shadows`).
    pub name: String,
    /// The subsystem [category](CvarCategory) the cvar belongs to.
    pub category: CvarCategory,
    /// The resolved value from the cascade (falls back to [`default`](CvarListing::default)
    /// when the cvar is otherwise unset).
    pub value: SettingValue,
    /// The registered default value (the
    /// [`EngineDefault`](crate::settings::SettingsLayer::EngineDefault) contribution).
    pub default: SettingValue,
    /// The declared numeric [bounds](CvarBounds) (`None` for bool/string cvars).
    pub bounds: CvarBounds,
    /// The permission / behaviour [flags](CvarFlags).
    pub flags: CvarFlags,
    /// The accepted [`SettingValue`] kind, as a stable label (see [`Cvar::kind`]).
    pub kind: &'static str,
    /// The cascade [layer](crate::settings::SettingsLayer) the resolved value
    /// came from, or `None` if the value is the schema default with no cascade
    /// entry.
    pub source: Option<SettingsLayer>,
    /// The human-readable description (may be empty).
    pub description: String,
}

impl CvarListing {
    /// Render a single-line console summary of this cvar, in the shape a
    /// shipping `cvarlist` / `find` command prints: `name = value`, the default
    /// in parentheses when the value has diverged from it, a bracketed list of
    /// active flag labels, and the description after a dash. For example
    /// `r.shadows = 4 (default 2) [archive] - shadow quality`.
    #[must_use]
    pub fn summary_line(&self) -> String {
        let mut line = format!("{} = {}", self.name, format_cvar_token(&self.value));
        if self.value != self.default {
            line.push_str(" (default ");
            line.push_str(&format_cvar_token(&self.default));
            line.push(')');
        }
        let mut labels: Vec<&str> = Vec::new();
        if self.flags.contains(CvarFlags::CHEAT) {
            labels.push("cheat");
        }
        if self.flags.contains(CvarFlags::READ_ONLY) {
            labels.push("readonly");
        }
        if self.flags.contains(CvarFlags::ARCHIVE) {
            labels.push("archive");
        }
        if self.flags.contains(CvarFlags::NOTIFY) {
            labels.push("notify");
        }
        if !labels.is_empty() {
            line.push_str(" [");
            line.push_str(&labels.join(","));
            line.push(']');
        }
        if !self.description.is_empty() {
            line.push_str(" - ");
            line.push_str(&self.description);
        }
        line
    }
}

/// The result of [`CvarRegistry::validate_set`]: the value that should be
/// written to the cascade (type-coerced into the declared kind and clamped into
/// the declared [bounds](CvarBounds)), plus whether clamping altered it.
///
/// Separating validation from the cascade write lets [`App::set_cvar_at`] report
/// an accurate [`clamped`](CvarSetOutcome::clamped) flag without re-clamping an
/// already-clamped value.
#[derive(Clone, Debug, PartialEq)]
pub struct ValidatedWrite {
    /// The value to store (coerced and clamped).
    pub value: SettingValue,
    /// Whether clamping changed the written value.
    pub clamped: bool,
}

/// A per-argument report from a cvar-aware launch-override pass
/// ([`App::apply_cvar_cli_overrides`] / [`App::apply_cvar_env_overrides`],
/// design §24.6, §25.3, §14).
///
/// Launch overrides (command-line flags, environment variables) are the
/// *outermost* input to the settings cascade, so they are also where a typo or
/// an out-of-range value first arrives. Rather than letting such input bypass a
/// declared cvar's schema — or crash the process — each argument is classified:
///
/// * [`cvars`](CvarCliReport::cvars) — keys that name a **declared** cvar and
///   passed [validation](CvarRegistry::validate_set); their type-coerced,
///   clamped value was written to the
///   [`CommandLine`](crate::settings::SettingsLayer::CommandLine) cascade layer.
/// * [`rejected`](CvarCliReport::rejected) — keys that name a declared cvar but
///   whose value was refused at the boundary (wrong type, read-only, or
///   cheat-protected while cheats are off). The cascade is left untouched for
///   that key (design §25.3 "非法输入按安全边界拒绝而非崩溃"), so a launcher can
///   log the [`CvarError`] and keep running.
/// * [`settings`](CvarCliReport::settings) — keys that are **not** declared
///   cvars; these are written verbatim into the `CommandLine` layer as ordinary
///   [`Settings`] keys (cvars are opt-in, so undeclared launch keys keep working
///   exactly like [`Settings::apply_cli_args`]).
///
/// The report preserves argument order within each bucket.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct CvarCliReport {
    /// Declared cvars that were accepted, in argument order, each with the
    /// [`CvarSetOutcome`] describing whether the resolved value changed and
    /// whether the input was clamped.
    pub cvars: Vec<CvarCliApplied>,
    /// Undeclared keys written as ordinary settings, in argument order, each
    /// carrying the [`SettingChange`](crate::settings::SettingChange) that
    /// actually altered a resolved value.
    pub settings: Vec<crate::settings::SettingChange>,
    /// Declared cvars whose value was rejected at the boundary, in argument
    /// order, each with the [`CvarError`] explaining why and leaving the
    /// cascade untouched for that key.
    pub rejected: Vec<CvarCliRejection>,
}

impl CvarCliReport {
    /// Whether no argument produced any effect or error (every argument was
    /// empty or left its resolved value unchanged).
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.cvars.is_empty() && self.settings.is_empty() && self.rejected.is_empty()
    }

    /// Whether at least one declared-cvar argument was rejected at the boundary.
    #[must_use]
    pub fn has_rejections(&self) -> bool {
        !self.rejected.is_empty()
    }

    /// The number of rejected declared-cvar arguments.
    #[must_use]
    pub fn rejected_count(&self) -> usize {
        self.rejected.len()
    }
}

/// One accepted declared-cvar launch override (see [`CvarCliReport::cvars`]).
#[derive(Clone, Debug, PartialEq)]
pub struct CvarCliApplied {
    /// The cvar name as declared (the key with any leading `--` stripped).
    pub name: String,
    /// The validated write's outcome: whether the resolved value changed and
    /// whether the input was clamped into the declared bounds.
    pub outcome: CvarSetOutcome,
}

/// One rejected declared-cvar launch override (see [`CvarCliReport::rejected`]).
#[derive(Clone, Debug, PartialEq)]
pub struct CvarCliRejection {
    /// The offending key (with any leading `--` stripped).
    pub key: String,
    /// Why the write was refused at the boundary; the cascade was left
    /// unchanged for this key.
    pub error: CvarError,
}

/// The stable kind label for a [`SettingValue`], used in error messages and in
/// [`Cvar::kind`].
#[must_use]
fn value_kind(value: &SettingValue) -> &'static str {
    match value {
        SettingValue::Bool(_) => "bool",
        SettingValue::Int(_) => "int",
        SettingValue::Float(_) => "float",
        SettingValue::Str(_) => "string",
    }
}

/// Coerce `value` to the same kind as `target`, if a lossless coercion exists.
///
/// The only coercion is `Int → Float` (console/config text frequently infers
/// `2` as [`Int`](SettingValue::Int) for a float cvar). All other cross-kind
/// pairs return `None` so validation reports a
/// [`TypeMismatch`](CvarError::TypeMismatch). Same-kind values pass through.
#[must_use]
fn coerce_to_kind(value: SettingValue, target: &SettingValue) -> Option<SettingValue> {
    match (target, &value) {
        (SettingValue::Bool(_), SettingValue::Bool(_))
        | (SettingValue::Int(_), SettingValue::Int(_))
        | (SettingValue::Float(_), SettingValue::Float(_))
        | (SettingValue::Str(_), SettingValue::Str(_)) => Some(value),
        (SettingValue::Float(_), SettingValue::Int(i)) => Some(SettingValue::Float(*i as f64)),
        _ => None,
    }
}

/// Whether `bounds`'s kind is legal for a cvar whose default is `default`.
///
/// [`None`](CvarBounds::None) is legal for any type; [`Int`](CvarBounds::Int)
/// requires an [`Int`](SettingValue::Int) default and
/// [`Float`](CvarBounds::Float) a [`Float`](SettingValue::Float) default.
#[must_use]
fn bounds_match_kind(bounds: CvarBounds, default: &SettingValue) -> bool {
    match bounds {
        CvarBounds::None => true,
        CvarBounds::Int(..) => matches!(default, SettingValue::Int(_)),
        CvarBounds::Float(..) => matches!(default, SettingValue::Float(_)),
    }
}

/// The declared-cvar registry (design §24.6).
///
/// Holds each cvar's immutable [`Cvar`] schema plus the process-wide
/// cheats-enabled flag. The live value is **not** stored here — it lives in the
/// [`Settings`] cascade — so the registry is a pure schema + validation
/// authority. It is a [`Resource`] and is auto-installed by the [`App`] cvar
/// helpers; it is never installed by [`App::new`](crate::app::App::new).
#[derive(Clone, Debug, Default)]
pub struct CvarRegistry {
    cvars: BTreeMap<String, Cvar>,
    cheats_enabled: bool,
}

impl Resource for CvarRegistry {}

impl CvarRegistry {
    /// An empty registry with cheats disabled.
    #[must_use]
    pub fn new() -> Self {
        CvarRegistry::default()
    }

    /// Register a cvar from `spec`, returning the cvar's name and clamped
    /// default on success.
    ///
    /// Validates that the name is not already taken
    /// ([`Redeclared`](CvarError::Redeclared)) and that any
    /// [bounds](CvarBounds) match the default's kind and are well-formed
    /// ([`InvalidBounds`](CvarError::InvalidBounds)). The default is clamped
    /// into its own bounds so a registered default is always in range. The
    /// returned `(name, default)` lets a caller seed the settings cascade
    /// without re-deriving them.
    pub fn register(&mut self, spec: CvarSpec) -> Result<(String, SettingValue), CvarError> {
        if self.cvars.contains_key(&spec.name) {
            return Err(CvarError::Redeclared(spec.name));
        }
        if !bounds_match_kind(spec.bounds, &spec.default) || !spec.bounds.is_well_formed() {
            return Err(CvarError::InvalidBounds(spec.name));
        }
        let (default, _) = spec.bounds.clamp(spec.default);
        let cvar = Cvar {
            category: spec.category,
            default: default.clone(),
            bounds: spec.bounds,
            flags: spec.flags,
            description: spec.description,
        };
        self.cvars.insert(spec.name.clone(), cvar);
        Ok((spec.name, default))
    }

    /// Whether a cvar named `name` is registered.
    #[must_use]
    pub fn contains(&self, name: &str) -> bool {
        self.cvars.contains_key(name)
    }

    /// The schema of a registered cvar, or `None`.
    #[must_use]
    pub fn get(&self, name: &str) -> Option<&Cvar> {
        self.cvars.get(name)
    }

    /// The number of registered cvars.
    #[must_use]
    pub fn len(&self) -> usize {
        self.cvars.len()
    }

    /// Whether no cvars are registered.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.cvars.is_empty()
    }

    /// Iterate every registered cvar as `(name, schema)` in ascending-name
    /// order.
    pub fn iter(&self) -> impl Iterator<Item = (&str, &Cvar)> {
        self.cvars.iter().map(|(name, cvar)| (name.as_str(), cvar))
    }

    /// Iterate registered cvars in `category`, ascending-name order.
    pub fn iter_category(&self, category: CvarCategory) -> impl Iterator<Item = (&str, &Cvar)> {
        self.iter().filter(move |(_, cvar)| cvar.category == category)
    }

    /// The names of every [`ARCHIVE`](CvarFlags::ARCHIVE) cvar, ascending.
    ///
    /// A config writer uses this to decide which cvars to persist to the user
    /// profile; this crate performs no file I/O itself.
    pub fn archived(&self) -> impl Iterator<Item = &str> {
        self.iter()
            .filter(|(_, cvar)| cvar.flags.contains(CvarFlags::ARCHIVE))
            .map(|(name, _)| name)
    }

    /// Whether cheats are currently enabled (gates [`CHEAT`](CvarFlags::CHEAT)
    /// cvar writes).
    #[must_use]
    pub fn cheats_enabled(&self) -> bool {
        self.cheats_enabled
    }

    /// Enable or disable cheats, affecting subsequent
    /// [`CHEAT`](CvarFlags::CHEAT) cvar writes.
    pub fn set_cheats_enabled(&mut self, enabled: bool) {
        self.cheats_enabled = enabled;
    }

    /// Validate a prospective write of `value` to cvar `name`, returning the
    /// [`ValidatedWrite`] (the type-coerced, clamped value plus whether it was
    /// clamped) or a [`CvarError`] (design §25.3).
    ///
    /// This is a **pure** check — it does not touch the [`Settings`] cascade —
    /// so it can be unit-tested and reused. It enforces, in order:
    ///
    /// 1. the cvar is registered ([`Unregistered`](CvarError::Unregistered));
    /// 2. it is not [read-only](CvarFlags::READ_ONLY)
    ///    ([`ReadOnly`](CvarError::ReadOnly));
    /// 3. cheats are enabled if it is [cheat-protected](CvarFlags::CHEAT)
    ///    ([`CheatProtected`](CvarError::CheatProtected));
    /// 4. the value is of (or coercible to) the declared type, and not a
    ///    non-finite float ([`TypeMismatch`](CvarError::TypeMismatch));
    /// 5. the value is clamped into the declared [bounds](CvarBounds).
    pub fn validate_set(&self, name: &str, value: SettingValue) -> Result<ValidatedWrite, CvarError> {
        let cvar = self
            .cvars
            .get(name)
            .ok_or_else(|| CvarError::Unregistered(name.to_owned()))?;
        if cvar.flags.contains(CvarFlags::READ_ONLY) {
            return Err(CvarError::ReadOnly(name.to_owned()));
        }
        if cvar.flags.contains(CvarFlags::CHEAT) && !self.cheats_enabled {
            return Err(CvarError::CheatProtected(name.to_owned()));
        }
        let found = value_kind(&value);
        let coerced = coerce_to_kind(value, &cvar.default).ok_or(CvarError::TypeMismatch {
            name: name.to_owned(),
            expected: value_kind(&cvar.default),
            found,
        })?;
        // Reject non-finite floats at the boundary (NaN/inf would poison
        // clamping and comparisons).
        if let SettingValue::Float(f) = &coerced
            && !f.is_finite()
        {
            return Err(CvarError::TypeMismatch {
                name: name.to_owned(),
                expected: "finite float",
                found: "non-finite float",
            });
        }
        let (value, clamped) = cvar.bounds.clamp(coerced);
        Ok(ValidatedWrite { value, clamped })
    }
}

impl App {
    /// Ensure the [`CvarRegistry`] resource, the [`Settings`] cascade, and both
    /// the [`SettingChanged`](crate::settings::SettingChanged) and
    /// [`CvarChanged`] events are installed; returns `&mut self` for chaining.
    ///
    /// Idempotent: the registry is inserted only if absent, so an already
    /// populated registry is never wiped. The cvar helpers call this first, so
    /// explicit use is only needed to install an empty registry up front.
    pub fn init_cvars(&mut self) -> &mut Self {
        self.init_settings();
        if self.world().get_resource::<CvarRegistry>().is_none() {
            self.insert_resource(CvarRegistry::new());
        }
        self.add_event::<CvarChanged>();
        self
    }

    /// Register a cvar from `spec`, seeding its default into the
    /// [`EngineDefault`](crate::settings::SettingsLayer::EngineDefault) settings
    /// layer (design §24.6).
    ///
    /// Auto-installs the registry, cascade, and events. Returns the error from
    /// [`CvarRegistry::register`] (duplicate name or invalid bounds) without
    /// mutating any state on failure. On success a
    /// [`SettingChanged`](crate::settings::SettingChanged) event is broadcast
    /// iff seeding the default changed the key's resolved value.
    pub fn register_cvar(&mut self, spec: CvarSpec) -> Result<(), CvarError> {
        self.init_cvars();
        let (name, default) = self.world_mut().resource_mut::<CvarRegistry>().register(spec)?;
        let change = self
            .world_mut()
            .resource_mut::<Settings>()
            .set(SettingsLayer::EngineDefault, name, default);
        if let Some(change) = change {
            self.send_event(change);
        }
        Ok(())
    }

    /// Enable or disable cheats, gating future [`CHEAT`](CvarFlags::CHEAT) cvar
    /// writes; returns `&mut self` for chaining. Auto-installs the registry.
    pub fn set_cheats_enabled(&mut self, enabled: bool) -> &mut Self {
        self.init_cvars();
        self.world_mut()
            .resource_mut::<CvarRegistry>()
            .set_cheats_enabled(enabled);
        self
    }

    /// Write `value` to the cvar `name` at the given settings `layer`, applying
    /// the registry's validation (design §24.6, §25.3).
    ///
    /// The value is type-checked, coerced, and clamped by
    /// [`CvarRegistry::validate_set`]; on success it is written to `layer` of
    /// the [`Settings`] cascade. If the write changes the key's resolved value,
    /// a [`SettingChanged`](crate::settings::SettingChanged) event is always
    /// broadcast and — if the cvar carries [`CvarFlags::NOTIFY`] — a
    /// [`CvarChanged`] event as well. Returns a [`CvarSetOutcome`] describing
    /// the result, or a [`CvarError`] (which leaves all state untouched).
    pub fn set_cvar_at(
        &mut self,
        layer: SettingsLayer,
        name: &str,
        value: impl Into<SettingValue>,
    ) -> Result<CvarSetOutcome, CvarError> {
        self.init_cvars();
        let registry = self.world().resource::<CvarRegistry>();
        let raw = value.into();
        let ValidatedWrite {
            value: clamped_value,
            clamped,
        } = registry.validate_set(name, raw)?;
        let notify = registry
            .get(name)
            .is_some_and(|cvar| cvar.flags.contains(CvarFlags::NOTIFY));
        let category = registry.get(name).map(Cvar::category).unwrap_or_default();

        let change = self
            .world_mut()
            .resource_mut::<Settings>()
            .set(layer, name.to_owned(), clamped_value.clone());

        let resolved = self
            .world()
            .resource::<Settings>()
            .get(name)
            .cloned()
            .unwrap_or_else(|| clamped_value.clone());

        let changed = change.is_some();
        if let Some(change) = change {
            let previous = change.previous.clone();
            self.send_event(change);
            if notify {
                self.send_event(CvarChanged {
                    name: name.to_owned(),
                    category,
                    previous,
                    current: resolved.clone(),
                });
            }
        }

        Ok(CvarSetOutcome {
            changed,
            clamped,
            resolved,
        })
    }

    /// Write `value` to the cvar `name` at the
    /// [`Runtime`](crate::settings::SettingsLayer::Runtime) layer (the
    /// console/runtime write path, design §24.6).
    ///
    /// A convenience wrapper over [`set_cvar_at`](App::set_cvar_at) with the
    /// highest-precedence layer; see it for validation and event semantics.
    pub fn set_cvar(
        &mut self,
        name: &str,
        value: impl Into<SettingValue>,
    ) -> Result<CvarSetOutcome, CvarError> {
        self.set_cvar_at(SettingsLayer::Runtime, name, value)
    }

    /// Execute one Quake/Source-style console command `line` against the cvar
    /// registry (design §24.6, which quotes the console form `r.shadows 2`).
    ///
    /// The line is tokenised exactly as a shipping console would: it is trimmed,
    /// and an empty line or one beginning with `//` is a no-op
    /// ([`ConsoleOutcome::Empty`]). Otherwise the first whitespace-delimited
    /// token is the cvar name and the remainder (trimmed) is the value:
    ///
    /// * a **bare name** (no remainder) is a *query* — a registered cvar reports
    ///   its resolved value as [`Queried`](ConsoleOutcome::Queried), an
    ///   unregistered one is [`Unknown`](ConsoleOutcome::Unknown);
    /// * a **`name value`** form *writes* the cvar at the
    ///   [`Runtime`](crate::settings::SettingsLayer::Runtime) layer (the
    ///   highest-precedence console layer) via [`set_cvar`](App::set_cvar),
    ///   after inferring the value's type with
    ///   [`SettingValue::parse`](crate::settings::SettingValue::parse). The
    ///   whole remainder is the value, so `name a b c` sets the string `a b c`.
    ///   A successful write yields [`Set`](ConsoleOutcome::Set); an unregistered
    ///   cvar is [`Unknown`](ConsoleOutcome::Unknown); any other validation
    ///   failure (type mismatch, read-only, cheat-gated) is
    ///   [`Rejected`](ConsoleOutcome::Rejected). Out-of-range numeric input is
    ///   *not* an error: it is clamped into bounds and reported through
    ///   [`CvarSetOutcome::clamped`].
    ///
    /// Like every mutating entry point in this module, a rejected line leaves
    /// all state untouched and never panics (design §25.3). A successful write
    /// broadcasts the same [`SettingChanged`](crate::settings::SettingChanged)
    /// (and, for a [`NOTIFY`](CvarFlags::NOTIFY) cvar whose value changed,
    /// [`CvarChanged`]) events as [`set_cvar`](App::set_cvar).
    pub fn exec_console(&mut self, line: &str) -> ConsoleOutcome {
        let line = line.trim();
        if line.is_empty() || line.starts_with("//") {
            return ConsoleOutcome::Empty;
        }
        let first = line
            .split_once(char::is_whitespace)
            .map_or(line, |(name, _)| name);
        self.init_cvars();
        if first == "reset" {
            // `reset` is a reserved *console* command (it shadows cvar get/set
            // for that first token): `reset <name>` reverts one cvar's runtime
            // override, and a bare `reset` reverts every cvar's. A target name
            // never contains whitespace, so only the first remaining token is
            // taken. This keyword is interactive-only — it is deliberately *not*
            // honoured by [`load_user_config`](App::load_user_config), whose
            // lines are declarative assignments rather than commands.
            let rest = line
                .split_once(char::is_whitespace)
                .map_or("", |(_, rest)| rest.trim());
            return match rest.split_whitespace().next() {
                None => ConsoleOutcome::ResetAll {
                    changed: self.reset_all_runtime_cvars(),
                },
                Some(target) => match self.reset_cvar(target) {
                    Ok(outcome) => ConsoleOutcome::Reset {
                        name: target.to_owned(),
                        outcome,
                    },
                    Err(CvarError::Unregistered(name)) => ConsoleOutcome::Unknown(name),
                    Err(error) => ConsoleOutcome::Rejected(error),
                },
            };
        }
        // The query/assignment grammar is shared with config loading; the
        // console writes successful assignments into the highest-precedence
        // [`Runtime`](crate::settings::SettingsLayer::Runtime) layer.
        self.apply_config_line(line, SettingsLayer::Runtime)
    }

    /// Apply one line of the shared cvar config grammar, writing a successful
    /// assignment into `layer` (design §24.6 console/config-file form, §14
    /// config layering).
    ///
    /// This is the common core of [`exec_console`](App::exec_console) (which
    /// targets [`Runtime`](crate::settings::SettingsLayer::Runtime) and layers
    /// the `reset` command on top) and
    /// [`load_user_config`](App::load_user_config) (which targets
    /// [`User`](crate::settings::SettingsLayer::User)). The line is trimmed; an
    /// empty line or one beginning with `//` is a no-op
    /// ([`ConsoleOutcome::Empty`]). Otherwise the first whitespace-delimited
    /// token is the cvar name and the trimmed remainder is the value: a bare
    /// name is a *query* (resolved value as [`Queried`](ConsoleOutcome::Queried),
    /// or [`Unknown`](ConsoleOutcome::Unknown) when undeclared), and a
    /// `name value` form *writes* the cvar at `layer` via
    /// [`set_cvar_at`](App::set_cvar_at) after inferring the value's type with
    /// [`SettingValue::parse`](crate::settings::SettingValue::parse). Validation
    /// is identical regardless of layer; a rejected line leaves all state
    /// untouched (design §25.3).
    fn apply_config_line(&mut self, line: &str, layer: SettingsLayer) -> ConsoleOutcome {
        let line = line.trim();
        if line.is_empty() || line.starts_with("//") {
            return ConsoleOutcome::Empty;
        }
        let (name, rest) = match line.split_once(char::is_whitespace) {
            Some((name, rest)) => (name, rest.trim()),
            None => (line, ""),
        };
        self.init_cvars();
        if rest.is_empty() {
            // Query form: a bare cvar name reports its resolved value. The
            // console is the declared front door, so an undeclared name is
            // Unknown rather than resolving an arbitrary settings key.
            if !self.world().resource::<CvarRegistry>().contains(name) {
                return ConsoleOutcome::Unknown(name.to_owned());
            }
            let value = self.cvar(name).cloned().unwrap_or_else(|| {
                self.world()
                    .resource::<CvarRegistry>()
                    .get(name)
                    .map(|cvar| cvar.default.clone())
                    .expect("cvar is registered")
            });
            return ConsoleOutcome::Queried {
                name: name.to_owned(),
                value,
            };
        }
        match self.set_cvar_at(layer, name, SettingValue::parse(rest)) {
            Ok(outcome) => ConsoleOutcome::Set(outcome),
            Err(CvarError::Unregistered(name)) => ConsoleOutcome::Unknown(name),
            Err(error) => ConsoleOutcome::Rejected(error),
        }
    }

    /// Execute a multi-line console `script`, running each line through
    /// [`exec_console`](App::exec_console) in order and collecting the per-line
    /// [`ConsoleOutcome`]s (design §24.6 console/config-file form, §14 config
    /// layering).
    ///
    /// The script is split on **newlines** only (never `;`), so a value with
    /// interior spaces survives intact — [`exec_console`](App::exec_console)
    /// takes the whole trimmed line remainder as the value. Blank lines and
    /// `//` comments become [`ConsoleOutcome::Empty`], so the header emitted by
    /// [`write_archive_config`](App::write_archive_config) is a no-op and
    /// `write_archive_config()` → `exec_console_script(..)` is a lossless
    /// save/load round-trip for archived cvars.
    ///
    /// Each line is independent: a [`Rejected`](ConsoleOutcome::Rejected) or
    /// [`Unknown`](ConsoleOutcome::Unknown) line leaves all state untouched
    /// (design §25.3) and never aborts the remaining lines, so a single stale
    /// entry in a config file cannot discard the rest of it.
    pub fn exec_console_script(&mut self, script: &str) -> Vec<ConsoleOutcome> {
        script.lines().map(|line| self.exec_console(line)).collect()
    }

    /// Load a persisted user configuration, routing each cvar assignment into
    /// the [`User`](crate::settings::SettingsLayer::User) cascade layer rather
    /// than `Runtime` (design §14 config layering `默认 → 平台 → 用户 → 命令行
    /// → 运行时`, §24.6).
    ///
    /// This is the layer-correct counterpart to
    /// [`exec_console_script`](App::exec_console_script): an interactive console
    /// writes the highest-precedence `Runtime` layer, but a user config file is
    /// a *persistent preference* that should sit below launch flags
    /// ([`CommandLine`](crate::settings::SettingsLayer::CommandLine), set by
    /// [`apply_cvar_cli_overrides`](App::apply_cvar_cli_overrides)) and runtime
    /// console tweaks. Loading into `User` therefore keeps the full cascade
    /// precedence intact: a `--r.shadows=0` launch flag or a live `r.shadows 0`
    /// console write still wins over the config file, and clearing a `Runtime`
    /// override with [`reset`](App::reset_cvar) falls back to the user config
    /// value rather than the engine default.
    ///
    /// Each line uses the same `name value` grammar as the console and as
    /// [`write_archive_config`](App::write_archive_config) output, so a config
    /// previously written to disk reloads cleanly; blank lines and `//`
    /// comments are ignored, and a bare cvar name is a harmless query. Unlike
    /// [`exec_console`](App::exec_console) a config is declarative, so the
    /// `reset` command keyword is **not** special here — a line beginning with
    /// `reset` is treated as an ordinary (and almost certainly
    /// [`Unknown`](ConsoleOutcome::Unknown)) cvar name. Each line is applied
    /// independently through the shared config applier: a
    /// rejected or unknown entry leaves all state untouched (design §25.3) and
    /// never aborts the remaining lines, so one stale key cannot discard the
    /// rest of the file.
    ///
    /// This crate performs no file I/O: callers read the config file themselves
    /// and pass its contents here.
    pub fn load_user_config(&mut self, script: &str) -> Vec<ConsoleOutcome> {
        script
            .lines()
            .map(|line| self.apply_config_line(line, SettingsLayer::User))
            .collect()
    }

    /// Serialise every [`ARCHIVE`](CvarFlags::ARCHIVE) cvar as console lines
    /// suitable for persisting to a user config file and replaying through
    /// [`exec_console_script`](App::exec_console_script) (design §24.6 "控制台/
    /// 配置文件/命令行可设", §14 config layering).
    ///
    /// Each archived cvar is written as a `name value` line — the exact form
    /// [`exec_console`](App::exec_console) consumes — in ascending-name order
    /// via [`CvarRegistry::archived`], so the output is deterministic. The
    /// resolved value from the cascade is used, falling back to the declared
    /// default when a cvar is unset. The text is prefixed with a `//` comment
    /// header, which [`exec_console`](App::exec_console) treats as a no-op on
    /// reload.
    ///
    /// This crate performs no file I/O: callers write the returned string to
    /// disk themselves. Reloading `write_archive_config()` output through
    /// [`exec_console_script`](App::exec_console_script) restores the archived
    /// values losslessly, with one honest limitation — a string value
    /// containing a newline cannot be represented in this line-based format and
    /// is skipped (such a value is pathological for an archived cvar). Interior
    /// spaces and tabs in a string value survive without quoting.
    #[must_use]
    pub fn write_archive_config(&self) -> String {
        let mut out = String::from(
            "// Prism archived cvars — generated config; reload via exec_console_script.\n",
        );
        let Some(registry) = self.world().get_resource::<CvarRegistry>() else {
            return out;
        };
        let settings = self.world().get_resource::<Settings>();
        for name in registry.archived() {
            let resolved = settings
                .and_then(|table| table.get(name).cloned())
                .or_else(|| registry.get(name).map(|cvar| cvar.default.clone()));
            let Some(value) = resolved else {
                continue;
            };
            if let SettingValue::Str(text) = &value
                && text.contains('\n')
            {
                // A newline would split one value across config lines; skip it
                // rather than emit a corrupt, non-round-tripping entry.
                continue;
            }
            out.push_str(name);
            out.push(' ');
            out.push_str(&format_cvar_token(&value));
            out.push('\n');
        }
        out
    }

    /// Serialise the [`User`](crate::settings::SettingsLayer::User)-layer value
    /// of every [`ARCHIVE`](CvarFlags::ARCHIVE) cvar as console lines suitable
    /// for persisting to a user config file and reloading through
    /// [`load_user_config`](App::load_user_config) (design §14 config layering,
    /// §24.6).
    ///
    /// This is the save half of the User-layer round-trip, the layer-correct
    /// counterpart to [`write_archive_config`](App::write_archive_config):
    /// where that method writes each cvar's *resolved* cascade value (useful
    /// for a full snapshot), this writes **only** the value the user explicitly
    /// set at the [`User`](crate::settings::SettingsLayer::User) layer. A cvar
    /// whose current value comes from a lower layer (engine default, platform
    /// tier) or a higher one (command-line flag, runtime console tweak) is
    /// *not* emitted, so transient launch flags and live console edits never
    /// leak into the persisted user preferences. Reloading the output through
    /// [`load_user_config`](App::load_user_config) restores exactly those user
    /// choices into the `User` layer.
    ///
    /// Only [`ARCHIVE`](CvarFlags::ARCHIVE) cvars are considered — the flag is
    /// the declared opt-in for "persist this to the user config" — and they are
    /// emitted in ascending-name order via [`CvarRegistry::archived`], so the
    /// output is deterministic. As with
    /// [`write_archive_config`](App::write_archive_config), a string value
    /// containing a newline cannot be represented in this line-based format and
    /// is skipped. This crate performs no file I/O: callers write the returned
    /// string to disk themselves.
    #[must_use]
    pub fn write_user_config(&self) -> String {
        let mut out = String::from(
            "// Prism user cvars — generated config; reload via load_user_config.\n",
        );
        let Some(registry) = self.world().get_resource::<CvarRegistry>() else {
            return out;
        };
        let Some(settings) = self.world().get_resource::<Settings>() else {
            return out;
        };
        for name in registry.archived() {
            // Only the value the user actually set at the User layer is
            // persisted; lower/higher layers are left to their own sources.
            let Some(value) = settings
                .layers_for(name)
                .and_then(|by_layer| by_layer.get(&SettingsLayer::User))
            else {
                continue;
            };
            if let SettingValue::Str(text) = value
                && text.contains('\n')
            {
                // A newline would split one value across config lines; skip it
                // rather than emit a corrupt, non-round-tripping entry.
                continue;
            }
            out.push_str(name);
            out.push(' ');
            out.push_str(&format_cvar_token(value));
            out.push('\n');
        }
        out
    }

    /// Snapshot every registered cvar as a [`CvarListing`], in ascending-name
    /// order (design §24.6 console enumeration/help).
    ///
    /// Each listing joins the immutable [schema](Cvar) with the value resolved
    /// from the [`Settings`] cascade, so a console `cvarlist` command or a
    /// developer overlay can render the full cvar table in one call. Returns an
    /// empty vector when no cvar has ever been registered.
    #[must_use]
    pub fn list_cvars(&self) -> Vec<CvarListing> {
        let Some(registry) = self.world().get_resource::<CvarRegistry>() else {
            return Vec::new();
        };
        let settings = self.world().get_resource::<Settings>();
        registry
            .iter()
            .map(|(name, cvar)| build_cvar_listing(name, cvar, settings))
            .collect()
    }

    /// Snapshot every registered cvar in `category` as a [`CvarListing`], in
    /// ascending-name order (design §24.6 "分类（渲染/网络/调试）").
    ///
    /// This is the category-filtered form of [`list_cvars`](App::list_cvars)
    /// (e.g. a console `cvarlist render`); see it for the join semantics.
    #[must_use]
    pub fn list_cvars_in_category(&self, category: CvarCategory) -> Vec<CvarListing> {
        let Some(registry) = self.world().get_resource::<CvarRegistry>() else {
            return Vec::new();
        };
        let settings = self.world().get_resource::<Settings>();
        registry
            .iter_category(category)
            .map(|(name, cvar)| build_cvar_listing(name, cvar, settings))
            .collect()
    }

    /// Snapshot every cvar whose name **or** description contains `needle`
    /// (ASCII/Unicode case-insensitive), in ascending-name order — the console
    /// `find` command (design §24.6 console help).
    ///
    /// Matching is a plain case-folded substring test, so `find shadow` surfaces
    /// `r.shadows` and any cvar documented with "shadow". An empty `needle`
    /// matches everything, making `find ""` an alias for
    /// [`list_cvars`](App::list_cvars).
    #[must_use]
    pub fn find_cvars(&self, needle: &str) -> Vec<CvarListing> {
        let needle = needle.to_lowercase();
        let Some(registry) = self.world().get_resource::<CvarRegistry>() else {
            return Vec::new();
        };
        let settings = self.world().get_resource::<Settings>();
        registry
            .iter()
            .filter(|(name, cvar)| {
                name.to_lowercase().contains(&needle)
                    || cvar.description().to_lowercase().contains(&needle)
            })
            .map(|(name, cvar)| build_cvar_listing(name, cvar, settings))
            .collect()
    }

    /// Snapshot every registered cvar whose resolved cascade value differs from
    /// its registered [default](Cvar::default_value), in ascending-name order —
    /// the "show changed settings" / settings-diff view over the config cascade
    /// `默认 → 平台 → 用户 → 命令行 → 运行时` (design §24.6 / §14).
    ///
    /// This is the diff complement to [`list_cvars`](App::list_cvars): where
    /// that renders the full table, this surfaces only the cvars a platform
    /// tier, user config, launch override, or console command has actually
    /// moved off their engine default — exactly what a `cvarlist modified`
    /// command, a settings-diff export, or a "reset to defaults" confirmation
    /// prompt needs. Each returned [`CvarListing`] carries the
    /// [`source`](CvarListing::source) layer that won the cascade, so the caller
    /// can show *where* each override came from. Returns an empty vector when no
    /// cvar has ever been registered or when every cvar still resolves to its
    /// default.
    #[must_use]
    pub fn list_modified_cvars(&self) -> Vec<CvarListing> {
        let Some(registry) = self.world().get_resource::<CvarRegistry>() else {
            return Vec::new();
        };
        let settings = self.world().get_resource::<Settings>();
        registry
            .iter()
            .map(|(name, cvar)| build_cvar_listing(name, cvar, settings))
            .filter(|listing| listing.value != listing.default)
            .collect()
    }

    /// Clear the cvar `name`'s [`Runtime`](crate::settings::SettingsLayer::Runtime)
    /// override, letting it fall back to a lower cascade layer (design §24.6).
    ///
    /// Fails with [`Unregistered`](CvarError::Unregistered) for an unknown cvar
    /// and with [`ReadOnly`](CvarError::ReadOnly) for a read-only one.
    /// Broadcasts a [`SettingChanged`](crate::settings::SettingChanged) (and, if
    /// applicable, a [`CvarChanged`]) event iff the resolved value changed.
    pub fn reset_cvar(&mut self, name: &str) -> Result<CvarSetOutcome, CvarError> {
        self.init_cvars();
        let registry = self.world().resource::<CvarRegistry>();
        let cvar = registry
            .get(name)
            .ok_or_else(|| CvarError::Unregistered(name.to_owned()))?;
        if cvar.flags.contains(CvarFlags::READ_ONLY) {
            return Err(CvarError::ReadOnly(name.to_owned()));
        }
        let notify = cvar.flags.contains(CvarFlags::NOTIFY);
        let category = cvar.category;

        let change = self
            .world_mut()
            .resource_mut::<Settings>()
            .clear(SettingsLayer::Runtime, name);

        let resolved = self
            .world()
            .resource::<Settings>()
            .get(name)
            .cloned()
            .unwrap_or_else(|| {
                self.world()
                    .resource::<CvarRegistry>()
                    .get(name)
                    .map(|cvar| cvar.default.clone())
                    .expect("cvar is registered")
            });

        let changed = change.is_some();
        if let Some(change) = change {
            let previous = change.previous.clone();
            self.send_event(change);
            if notify {
                self.send_event(CvarChanged {
                    name: name.to_owned(),
                    category,
                    previous,
                    current: resolved.clone(),
                });
            }
        }

        Ok(CvarSetOutcome {
            changed,
            clamped: false,
            resolved,
        })
    }

    /// Clear **every** cvar's [`Runtime`](crate::settings::SettingsLayer::Runtime)
    /// override at once, letting each fall back to its lower cascade layer — the
    /// bulk form of [`reset_cvar`](App::reset_cvar) and the engine for the bare
    /// `reset` console command (design §24.6).
    ///
    /// Each registered cvar is reset in ascending-name order via
    /// [`reset_cvar`](App::reset_cvar), so the per-cvar event broadcast
    /// ([`SettingChanged`](crate::settings::SettingChanged) and any
    /// [`CvarChanged`]) and the fall-back semantics are identical to resetting
    /// each by hand. [`Read-only`](CvarFlags::READ_ONLY) cvars never carry a
    /// runtime override (writes to them are rejected at the boundary), so they
    /// are skipped silently rather than reported as errors. Returns the
    /// ascending-name list of cvars whose resolved value actually changed; an
    /// empty vector means nothing had a runtime override (or no cvar is
    /// registered).
    pub fn reset_all_runtime_cvars(&mut self) -> Vec<String> {
        self.init_cvars();
        let names: Vec<String> = self
            .world()
            .resource::<CvarRegistry>()
            .iter()
            .map(|(name, _)| name.to_owned())
            .collect();
        let mut changed = Vec::new();
        for name in names {
            if let Ok(outcome) = self.reset_cvar(&name)
                && outcome.changed
            {
                changed.push(name);
            }
        }
        changed
    }

    /// Resolve the cvar `name`'s effective [`SettingValue`] from the cascade,
    /// or `None` if the registry/cascade was never initialised or the cvar is
    /// unset.
    #[must_use]
    pub fn cvar(&self, name: &str) -> Option<&SettingValue> {
        self.world().get_resource::<Settings>()?.get(name)
    }

    /// Resolve the cvar `name` as a `bool`, if set and boolean.
    #[must_use]
    pub fn cvar_bool(&self, name: &str) -> Option<bool> {
        self.cvar(name).and_then(SettingValue::as_bool)
    }

    /// Resolve the cvar `name` as an `i64`, if set and integral.
    #[must_use]
    pub fn cvar_int(&self, name: &str) -> Option<i64> {
        self.cvar(name).and_then(SettingValue::as_int)
    }

    /// Resolve the cvar `name` as an `f64`, if set and floating-point.
    #[must_use]
    pub fn cvar_float(&self, name: &str) -> Option<f64> {
        self.cvar(name).and_then(SettingValue::as_float)
    }

    /// Resolve the cvar `name` as a `&str`, if set and a string.
    #[must_use]
    pub fn cvar_str(&self, name: &str) -> Option<&str> {
        self.cvar(name).and_then(SettingValue::as_str)
    }

    /// Apply command-line-style launch overrides, routing **declared cvars**
    /// through [validation](CvarRegistry::validate_set) and writing everything
    /// into the [`CommandLine`](crate::settings::SettingsLayer::CommandLine)
    /// cascade layer (design §24.6, §25.3, §14).
    ///
    /// This is the cvar-aware counterpart to
    /// [`apply_cli_overrides`](App::apply_cli_overrides): it accepts the same
    /// argument syntax (`--key=value`, `key=value`, or a bare `--flag` meaning
    /// [`Bool(true)`](crate::settings::SettingValue::Bool), with a leading `--`
    /// stripped), but it does not blindly write every argument. For each key:
    ///
    /// * if the key names a **registered cvar**, the parsed value is run through
    ///   [`set_cvar_at`](App::set_cvar_at) at the `CommandLine` layer, so it is
    ///   type-coerced, clamped into the declared [bounds](CvarBounds), and
    ///   refused if the cvar is [read-only](CvarFlags::READ_ONLY) or
    ///   [cheat-protected](CvarFlags::CHEAT) while cheats are off. Illegal input
    ///   is rejected at the boundary with a [`CvarError`] (recorded in
    ///   [`CvarCliReport::rejected`]) and leaves the cascade untouched for that
    ///   key, never panicking (design §25.3);
    /// * otherwise the key is treated as an ordinary, undeclared
    ///   [`Settings`] key and written verbatim into
    ///   the `CommandLine` layer, exactly like
    ///   [`Settings::apply_cli_args`].
    ///
    /// Each accepted or rejected argument broadcasts the usual
    /// [`SettingChanged`](crate::settings::SettingChanged) (and, for a `NOTIFY`
    /// cvar whose resolved value changed, [`CvarChanged`]) events through the
    /// same paths as the other setters. The returned [`CvarCliReport`] lets a
    /// launcher surface what was applied, clamped, or rejected.
    ///
    /// Auto-initialises the registry, cascade, and events. Because launch flags
    /// are a *lower* cascade layer than `Runtime`, a declared cvar already
    /// overridden at `Runtime` resolves unchanged (its
    /// [`CvarSetOutcome::changed`] is `false`); the `CommandLine` contribution is
    /// still recorded underneath. Note that cheat-protected cvars are refused
    /// here unless [cheats are enabled](App::set_cheats_enabled) first, so a dev
    /// build that wants to honour cheat cvars from the command line must enable
    /// cheats before calling this.
    pub fn apply_cvar_cli_overrides<I, S>(&mut self, args: I) -> CvarCliReport
    where
        I: IntoIterator<Item = S>,
        S: AsRef<str>,
    {
        self.init_cvars();
        let mut report = CvarCliReport::default();
        for arg in args {
            let arg = arg.as_ref();
            let (key, value) = match arg.split_once('=') {
                Some((k, v)) => (
                    k.trim_start_matches("--").to_owned(),
                    SettingValue::parse(v),
                ),
                None => (
                    arg.trim_start_matches("--").to_owned(),
                    SettingValue::Bool(true),
                ),
            };
            if key.is_empty() {
                continue;
            }
            self.fold_launch_override(&key, value, &mut report);
        }
        report
    }

    /// Apply environment-style launch overrides, routing **declared cvars**
    /// through [validation](CvarRegistry::validate_set) and writing everything
    /// into the [`CommandLine`](crate::settings::SettingsLayer::CommandLine)
    /// cascade layer (design §24.6, §25.3, §14).
    ///
    /// The cvar-aware counterpart to
    /// [`apply_env_vars`](crate::settings::Settings::apply_env_vars): only
    /// variables whose name starts with `prefix` are considered; the prefix is
    /// stripped, the remainder is lowercased with each `_` mapped to `.` (so
    /// `PRISM_R_SHADOWS` → `r.shadows`, matching the dotted cvar namespace), and
    /// the value is parsed with
    /// [`SettingValue::parse`](crate::settings::SettingValue::parse). Each
    /// resulting `(key, value)` is then classified and applied exactly like
    /// [`apply_cvar_cli_overrides`](App::apply_cvar_cli_overrides) — declared
    /// cvars validated and clamped, undeclared keys written verbatim, illegal
    /// cvar input rejected at the boundary into [`CvarCliReport::rejected`].
    ///
    /// Launch flags and the environment are a single launch-time layer, so these
    /// share the `CommandLine` layer with
    /// [`apply_cvar_cli_overrides`](App::apply_cvar_cli_overrides).
    pub fn apply_cvar_env_overrides<I, K, V>(&mut self, vars: I, prefix: &str) -> CvarCliReport
    where
        I: IntoIterator<Item = (K, V)>,
        K: AsRef<str>,
        V: AsRef<str>,
    {
        self.init_cvars();
        let mut report = CvarCliReport::default();
        for (name, value) in vars {
            let name = name.as_ref();
            let Some(rest) = name.strip_prefix(prefix) else {
                continue;
            };
            if rest.is_empty() {
                continue;
            }
            let key = rest.to_lowercase().replace('_', ".");
            let value = SettingValue::parse(value.as_ref());
            self.fold_launch_override(&key, value, &mut report);
        }
        report
    }

    /// Classify and apply a single parsed launch override into the
    /// [`CommandLine`](crate::settings::SettingsLayer::CommandLine) layer,
    /// recording the outcome in `report`.
    ///
    /// Shared by [`apply_cvar_cli_overrides`](App::apply_cvar_cli_overrides) and
    /// [`apply_cvar_env_overrides`](App::apply_cvar_env_overrides): a declared
    /// cvar goes through [`set_cvar_at`](App::set_cvar_at) (validated/clamped,
    /// rejected on error); any other key is written verbatim as an ordinary
    /// setting and its [`SettingChanged`](crate::settings::SettingChanged) event
    /// broadcast.
    fn fold_launch_override(
        &mut self,
        key: &str,
        value: SettingValue,
        report: &mut CvarCliReport,
    ) {
        let is_cvar = self.world().resource::<CvarRegistry>().contains(key);
        if is_cvar {
            match self.set_cvar_at(SettingsLayer::CommandLine, key, value) {
                Ok(outcome) => report.cvars.push(CvarCliApplied {
                    name: key.to_owned(),
                    outcome,
                }),
                Err(error) => report.rejected.push(CvarCliRejection {
                    key: key.to_owned(),
                    error,
                }),
            }
        } else {
            let change = self
                .world_mut()
                .resource_mut::<Settings>()
                .set(SettingsLayer::CommandLine, key.to_owned(), value);
            if let Some(change) = change {
                self.send_event(change.clone());
                report.settings.push(change);
            }
        }
    }
}

/// Format a [`SettingValue`] back into a single console token that
/// [`SettingValue::parse`](crate::settings::SettingValue::parse) reads back as
/// an equivalent value, so [`App::write_archive_config`] output round-trips
/// through [`App::exec_console_script`].
///
/// A float holding an integral value is written with a trailing `.0` so it
/// parses back as a [`Float`](SettingValue::Float) rather than an
/// [`Int`](SettingValue::Int); every other variant uses its natural textual
/// form. A string is emitted verbatim: the console treats the whole trimmed
/// line remainder as the value, so interior spaces survive without quoting.
fn format_cvar_token(value: &SettingValue) -> String {
    match value {
        SettingValue::Bool(flag) => flag.to_string(),
        SettingValue::Int(int) => int.to_string(),
        SettingValue::Float(float) => {
            if float.is_finite() && float.fract() == 0.0 {
                format!("{float}.0")
            } else {
                float.to_string()
            }
        }
        SettingValue::Str(text) => text.clone(),
    }
}

/// Build a [`CvarListing`] snapshot for one cvar, joining its [schema](Cvar)
/// with the value resolved from the [`Settings`] cascade (and the layer that
/// value came from). When the cvar has no cascade entry the schema default is
/// used and the source layer is `None`.
fn build_cvar_listing(name: &str, cvar: &Cvar, settings: Option<&Settings>) -> CvarListing {
    let value = settings
        .and_then(|table| table.get(name).cloned())
        .unwrap_or_else(|| cvar.default_value().clone());
    let source = settings.and_then(|table| table.resolved_layer(name));
    CvarListing {
        name: name.to_owned(),
        category: cvar.category(),
        value,
        default: cvar.default_value().clone(),
        bounds: cvar.bounds(),
        flags: cvar.flags(),
        kind: cvar.kind(),
        source,
        description: cvar.description().to_owned(),
    }
}
