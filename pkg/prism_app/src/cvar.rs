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
}
