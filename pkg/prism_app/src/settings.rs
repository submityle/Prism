//! Layered configuration and settings cascade (design §14, §24.6, §25.3).
//!
//! A shipping engine resolves one effective value for each setting from several
//! independent sources that are authored at different times by different
//! actors: the engine's built-in defaults, a per-platform quality tier, the
//! user's saved preferences, process launch flags, and live runtime overrides
//! (a console `cvar` write). The design calls for a strict precedence cascade —
//! *engine default → platform tier → user → command line → runtime* — where a
//! higher layer overrides a lower one **without** destroying it, so clearing the
//! higher layer transparently falls back to whatever was underneath (design §14
//! "分层覆盖（后者覆盖前者）", §24.6 "配置级联：默认 < 平台 < 用户 < 命令行 <
//! 运行时").
//!
//! This module implements that cascade as a real, self-contained layered store:
//!
//! * [`SettingsLayer`] names the five layers, ordered lowest→highest so the
//!   derived [`Ord`] *is* the precedence order.
//! * [`SettingValue`] is a small dynamically-typed value (`bool` / `i64` /
//!   `f64` / `String`) with type-inferring [`parse`](SettingValue::parse) and
//!   typed accessors.
//! * [`Settings`] is a [`Resource`] holding, per key, the value contributed by
//!   each layer; [`get`](Settings::get) resolves the highest present layer.
//! * [`SettingChanged`] is an [`Event`] broadcast whenever a mutation changes a
//!   key's *resolved* value, so interested subsystems can react (design §14
//!   "设置变更以事件广播给关心的子系统"; §24.6 "cvar 变更可触发回调").
//! * [`App`] helpers ([`init_settings`](App::init_settings),
//!   [`insert_setting`](App::insert_setting),
//!   [`clear_setting`](App::clear_setting),
//!   [`apply_cli_overrides`](App::apply_cli_overrides), and the
//!   [`setting`](App::setting) getters) wire the resource and the change event
//!   together and broadcast every effective change.
//!
//! # Deterministic iteration
//!
//! The per-key and per-layer maps are [`BTreeMap`]s, so iteration order is a
//! stable function of the keys (and of the layer enum's order), never of hash
//! seeding or insertion order. This keeps settings-driven assembly reproducible
//! under the determinism goal (design §15): two runs that insert the same
//! settings resolve and iterate them identically.
//!
//! # Honestly deferred: `reflect`-backed typed validation
//!
//! Design §25.3 (and §24.6) specify that per-field **range clamping, defaults,
//! and rename compatibility** are delegated to `prism_reflect`'s attribute
//! system (`clamp` / `default` / `rename`, reflect §24.7), and that illegal
//! input is *rejected at a safe boundary rather than crashing* (reflect §24.8).
//! That reflect attribute machinery is `prism_reflect`'s own in-flight
//! milestone and the `reflect` feature is not yet wired into this crate's
//! `Cargo.toml`. Rather than fake typed validation, this module implements the
//! full layered *resolution* now with best-effort type **inference** on parse,
//! and leaves reflect-backed typed read/write and schema validation **honestly
//! absent** — to be layered on top without changing the cascade semantics once
//! `prism_reflect`'s attribute system lands. Nothing here is stubbed; the
//! deferred piece is simply not present and is documented as such.
//!
//! [`Resource`]: prism_ecs::resource::Resource
//! [`Event`]: prism_ecs::event::Event
//! [`BTreeMap`]: std::collections::BTreeMap

use std::collections::btree_map::Entry;
use std::collections::BTreeMap;

use prism_ecs::event::Event;
use prism_ecs::resource::Resource;

use crate::app::App;

/// The configuration layers, in **ascending precedence** (design §14, §24.6).
///
/// The variants are declared lowest→highest so the derived [`Ord`]/[`PartialOrd`]
/// *is* the precedence order: [`Runtime`](SettingsLayer::Runtime) compares
/// greater than every other layer, [`EngineDefault`](SettingsLayer::EngineDefault)
/// less than every other. [`Settings`] relies on this: it stores each layer's
/// contribution in an ordered map and resolves a key by taking the
/// highest-ordered layer that is present.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum SettingsLayer {
    /// The engine's compiled-in default — the lowest-precedence fallback.
    EngineDefault,
    /// Values chosen for the detected platform / quality tier (design §3).
    PlatformTier,
    /// The user's saved preferences.
    User,
    /// Values supplied on the process command line / environment at launch.
    CommandLine,
    /// Live runtime overrides (a console `cvar` write) — highest precedence.
    Runtime,
}

impl SettingsLayer {
    /// Every layer in ascending precedence order.
    ///
    /// Index `0` is the lowest-precedence [`EngineDefault`](SettingsLayer::EngineDefault)
    /// and the last entry is the highest-precedence [`Runtime`](SettingsLayer::Runtime).
    pub const ALL: [SettingsLayer; 5] = [
        SettingsLayer::EngineDefault,
        SettingsLayer::PlatformTier,
        SettingsLayer::User,
        SettingsLayer::CommandLine,
        SettingsLayer::Runtime,
    ];

    /// This layer's precedence rank, `0` for the lowest
    /// ([`EngineDefault`](SettingsLayer::EngineDefault)) up to `4` for the
    /// highest ([`Runtime`](SettingsLayer::Runtime)).
    ///
    /// Equivalent to the layer's position in [`ALL`](SettingsLayer::ALL); a
    /// higher rank wins the cascade.
    #[must_use]
    pub fn precedence(self) -> u8 {
        match self {
            SettingsLayer::EngineDefault => 0,
            SettingsLayer::PlatformTier => 1,
            SettingsLayer::User => 2,
            SettingsLayer::CommandLine => 3,
            SettingsLayer::Runtime => 4,
        }
    }
}

/// A dynamically-typed setting value (design §14, §24.6).
///
/// Settings are authored from heterogeneous sources (code defaults, text config
/// files, command-line strings, console input), so the stored value is a small
/// tagged union rather than a single Rust type. Typed access is via
/// [`as_bool`](SettingValue::as_bool) / [`as_int`](SettingValue::as_int) /
/// [`as_float`](SettingValue::as_float) / [`as_str`](SettingValue::as_str),
/// and text is turned into the most specific variant by
/// [`parse`](SettingValue::parse).
///
/// `PartialEq` (not `Eq`) is derived because [`Float`](SettingValue::Float)
/// wraps `f64`. One consequence is deliberate and documented: a `Float(NaN)`
/// never compares equal to itself, so re-setting a key to `NaN` always counts
/// as a change and re-broadcasts a [`SettingChanged`] event. Finite values
/// compare normally.
#[derive(Clone, Debug, PartialEq)]
pub enum SettingValue {
    /// A boolean flag.
    Bool(bool),
    /// A signed integer.
    Int(i64),
    /// A floating-point number.
    Float(f64),
    /// A string value.
    Str(String),
}

impl SettingValue {
    /// The boolean payload, or `None` if this is not a [`Bool`](SettingValue::Bool).
    #[must_use]
    pub fn as_bool(&self) -> Option<bool> {
        match self {
            SettingValue::Bool(b) => Some(*b),
            _ => None,
        }
    }

    /// The integer payload, or `None` if this is not an [`Int`](SettingValue::Int).
    #[must_use]
    pub fn as_int(&self) -> Option<i64> {
        match self {
            SettingValue::Int(i) => Some(*i),
            _ => None,
        }
    }

    /// The float payload, or `None` if this is not a [`Float`](SettingValue::Float).
    #[must_use]
    pub fn as_float(&self) -> Option<f64> {
        match self {
            SettingValue::Float(f) => Some(*f),
            _ => None,
        }
    }

    /// The string payload, or `None` if this is not a [`Str`](SettingValue::Str).
    #[must_use]
    pub fn as_str(&self) -> Option<&str> {
        match self {
            SettingValue::Str(s) => Some(s.as_str()),
            _ => None,
        }
    }

    /// Infer the most specific [`SettingValue`] from a textual token.
    ///
    /// The inference order is intentionally most-specific first: the exact
    /// tokens `"true"`/`"false"` become a [`Bool`](SettingValue::Bool); an
    /// otherwise valid [`i64`] becomes an [`Int`](SettingValue::Int); an
    /// otherwise valid [`f64`] becomes a [`Float`](SettingValue::Float); and
    /// anything else is kept verbatim as a [`Str`](SettingValue::Str). This is
    /// best-effort inference for untyped text sources (config files, the
    /// command line, the console); schema-driven typed parsing is the
    /// `prism_reflect`-backed work documented as deferred at the module level.
    #[must_use]
    pub fn parse(text: &str) -> SettingValue {
        match text {
            "true" => return SettingValue::Bool(true),
            "false" => return SettingValue::Bool(false),
            _ => {}
        }
        if let Ok(i) = text.parse::<i64>() {
            return SettingValue::Int(i);
        }
        if let Ok(f) = text.parse::<f64>() {
            return SettingValue::Float(f);
        }
        SettingValue::Str(text.to_owned())
    }
}

impl From<bool> for SettingValue {
    fn from(value: bool) -> Self {
        SettingValue::Bool(value)
    }
}

impl From<i64> for SettingValue {
    fn from(value: i64) -> Self {
        SettingValue::Int(value)
    }
}

impl From<f64> for SettingValue {
    fn from(value: f64) -> Self {
        SettingValue::Float(value)
    }
}

impl From<&str> for SettingValue {
    fn from(value: &str) -> Self {
        SettingValue::Str(value.to_owned())
    }
}

impl From<String> for SettingValue {
    fn from(value: String) -> Self {
        SettingValue::Str(value)
    }
}

/// A change to a key's **resolved** value (design §14).
///
/// Returned by the mutating [`Settings`] methods and broadcast as the
/// [`SettingChanged`] event by the [`App`] helpers. `previous` and `current`
/// are the effective (cascade-resolved) values before and after the mutation;
/// either is `None` when the key had, or now has, no value in any layer. A
/// mutation that does not alter the resolved value (for example writing a lower
/// layer while a higher layer still overrides it) produces no change record and
/// no event.
///
/// This is the broadcast [`SettingChanged`] event type itself (they are the
/// same struct, re-exported under both names): the resource layer returns it as
/// a plain record and the [`App`] layer sends it as an event.
#[derive(Clone, Debug, PartialEq)]
pub struct SettingChanged {
    /// The affected setting key.
    pub key: String,
    /// The resolved value before the mutation, or `None` if previously unset.
    pub previous: Option<SettingValue>,
    /// The resolved value after the mutation, or `None` if now unset.
    pub current: Option<SettingValue>,
}

impl Event for SettingChanged {}

/// Alias for [`SettingChanged`]: the record returned by the mutating
/// [`Settings`] methods is exactly the broadcast event payload.
pub type SettingChange = SettingChanged;

/// The layered settings store (design §14, §24.6).
///
/// For each key it keeps a map from [`SettingsLayer`] to the value contributed
/// by that layer. [`get`](Settings::get) resolves a key by returning the value
/// of the **highest** present layer; lower layers remain intact underneath, so
/// [`clear`](Settings::clear)ing the top layer transparently falls back to the
/// next one down. Both maps are [`BTreeMap`]s, giving deterministic iteration
/// (see the module docs).
///
/// The store is intentionally **not** installed by
/// [`App::new`](crate::app::App::new): it is opt-in and auto-initialised by the
/// [`App`] settings helpers on first use.
#[derive(Clone, Debug, Default)]
pub struct Settings {
    layers: BTreeMap<String, BTreeMap<SettingsLayer, SettingValue>>,
}

impl Resource for Settings {}

impl Settings {
    /// An empty settings store.
    #[must_use]
    pub fn new() -> Self {
        Settings::default()
    }

    /// Resolve `key` to its effective value: the value of the highest present
    /// [`SettingsLayer`], or `None` if no layer sets it.
    #[must_use]
    pub fn get(&self, key: &str) -> Option<&SettingValue> {
        // The per-layer map is ordered by the layer enum (lowest→highest), so
        // the last entry is the highest-precedence layer that is present.
        self.layers.get(key)?.iter().next_back().map(|(_, v)| v)
    }

    /// Resolve `key` and return it as a `bool`, if set and boolean.
    #[must_use]
    pub fn get_bool(&self, key: &str) -> Option<bool> {
        self.get(key).and_then(SettingValue::as_bool)
    }

    /// Resolve `key` and return it as an `i64`, if set and integral.
    #[must_use]
    pub fn get_int(&self, key: &str) -> Option<i64> {
        self.get(key).and_then(SettingValue::as_int)
    }

    /// Resolve `key` and return it as an `f64`, if set and floating-point.
    #[must_use]
    pub fn get_float(&self, key: &str) -> Option<f64> {
        self.get(key).and_then(SettingValue::as_float)
    }

    /// Resolve `key` and return it as a `&str`, if set and a string.
    #[must_use]
    pub fn get_str(&self, key: &str) -> Option<&str> {
        self.get(key).and_then(SettingValue::as_str)
    }

    /// The [`SettingsLayer`] that currently provides `key`'s resolved value, or
    /// `None` if the key is unset.
    #[must_use]
    pub fn resolved_layer(&self, key: &str) -> Option<SettingsLayer> {
        self.layers.get(key)?.keys().next_back().copied()
    }

    /// Whether any layer sets `key`.
    #[must_use]
    pub fn contains(&self, key: &str) -> bool {
        self.layers.get(key).is_some_and(|m| !m.is_empty())
    }

    /// The per-layer contributions for `key`, if any (lowest→highest layer).
    #[must_use]
    pub fn layers_for(&self, key: &str) -> Option<&BTreeMap<SettingsLayer, SettingValue>> {
        self.layers.get(key)
    }

    /// Iterate every set key with its resolved value, in deterministic
    /// (ascending key) order.
    pub fn iter(&self) -> impl Iterator<Item = (&str, &SettingValue)> {
        self.layers.iter().filter_map(|(key, by_layer)| {
            by_layer
                .iter()
                .next_back()
                .map(|(_, value)| (key.as_str(), value))
        })
    }

    /// Set `key` in `layer` to `value`, returning a [`SettingChange`] iff the
    /// key's **resolved** value changed.
    ///
    /// Writing a layer below the current top layer records the contribution but
    /// does not change the resolved value, so it returns `None`.
    pub fn set(
        &mut self,
        layer: SettingsLayer,
        key: impl Into<String>,
        value: impl Into<SettingValue>,
    ) -> Option<SettingChange> {
        let key = key.into();
        let value = value.into();
        let previous = self.get(&key).cloned();
        self.layers
            .entry(key.clone())
            .or_default()
            .insert(layer, value);
        let current = self.get(&key).cloned();
        if previous == current {
            None
        } else {
            Some(SettingChange {
                key,
                previous,
                current,
            })
        }
    }

    /// Clear `key`'s contribution from `layer`, returning a [`SettingChange`]
    /// iff the key's **resolved** value changed.
    ///
    /// If removing the layer empties the key's map, the key entry itself is
    /// dropped so [`contains`](Settings::contains) reports it absent. Clearing a
    /// layer that was being overridden by a higher one changes nothing and
    /// returns `None`.
    pub fn clear(&mut self, layer: SettingsLayer, key: &str) -> Option<SettingChange> {
        let previous = self.get(key).cloned();
        if let Entry::Occupied(mut entry) = self.layers.entry(key.to_owned()) {
            entry.get_mut().remove(&layer);
            if entry.get().is_empty() {
                entry.remove();
            }
        }
        let current = self.get(key).cloned();
        if previous == current {
            None
        } else {
            Some(SettingChange {
                key: key.to_owned(),
                previous,
                current,
            })
        }
    }

    /// Apply command-line-style overrides into the
    /// [`CommandLine`](SettingsLayer::CommandLine) layer, returning one
    /// [`SettingChange`] per argument that changed a resolved value.
    ///
    /// Each argument is parsed as one of:
    ///
    /// * `--key=value` or `key=value` → `key` set to
    ///   [`SettingValue::parse`]`(value)`;
    /// * a bare `--flag` (no `=`) → `flag` set to
    ///   [`Bool(true)`](SettingValue::Bool).
    ///
    /// A leading `--` is stripped from the key in both forms. Arguments are
    /// applied in order, so a later argument for the same key wins within this
    /// call.
    pub fn apply_cli_args<I, S>(&mut self, args: I) -> Vec<SettingChange>
    where
        I: IntoIterator<Item = S>,
        S: AsRef<str>,
    {
        let mut changes = Vec::new();
        for arg in args {
            let arg = arg.as_ref();
            let (key, value) = match arg.split_once('=') {
                Some((k, v)) => (k.trim_start_matches("--"), SettingValue::parse(v)),
                None => (arg.trim_start_matches("--"), SettingValue::Bool(true)),
            };
            if key.is_empty() {
                continue;
            }
            if let Some(change) = self.set(SettingsLayer::CommandLine, key, value) {
                changes.push(change);
            }
        }
        changes
    }

    /// Fold environment-style variables into the
    /// [`CommandLine`](SettingsLayer::CommandLine) layer, returning one
    /// [`SettingChange`] per variable that changed a resolved value.
    ///
    /// Only variables whose name starts with `prefix` are considered; the
    /// prefix is stripped, the remainder is lowercased and each `_` is mapped to
    /// `.` (so `PRISM_R_SHADOWS` → `r.shadows`, matching the dotted `cvar`
    /// namespace of design §24.6), and the value is run through
    /// [`SettingValue::parse`]. Launch flags and the environment are a single
    /// launch-time layer, so these share the
    /// [`CommandLine`](SettingsLayer::CommandLine) layer with
    /// [`apply_cli_args`](Settings::apply_cli_args).
    ///
    /// This is the core, dependency-free entry point taking an explicit
    /// iterator so it is fully testable;
    /// [`apply_process_env`](Settings::apply_process_env) is the thin wrapper
    /// over the real process environment.
    pub fn apply_env_vars<I, K, V>(&mut self, vars: I, prefix: &str) -> Vec<SettingChange>
    where
        I: IntoIterator<Item = (K, V)>,
        K: AsRef<str>,
        V: AsRef<str>,
    {
        let mut changes = Vec::new();
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
            if let Some(change) = self.set(SettingsLayer::CommandLine, key, value) {
                changes.push(change);
            }
        }
        changes
    }

    /// Fold the real process environment (via [`std::env::vars`]) into the
    /// [`CommandLine`](SettingsLayer::CommandLine) layer, keeping only variables
    /// named with `prefix`.
    ///
    /// A thin convenience wrapper over
    /// [`apply_env_vars`](Settings::apply_env_vars); see it for the key mapping
    /// and return semantics.
    pub fn apply_process_env(&mut self, prefix: &str) -> Vec<SettingChange> {
        self.apply_env_vars(std::env::vars(), prefix)
    }
}

impl App {
    /// Ensure the [`Settings`] resource and the [`SettingChanged`] event are
    /// installed, returning `&mut self` for chaining.
    ///
    /// Idempotent: the resource is inserted only if absent (so an already
    /// populated store is never wiped), and registering the event is itself
    /// idempotent. [`Settings`] is deliberately **not** installed by
    /// [`App::new`](crate::app::App::new); the other settings helpers call this
    /// first, so explicit use is only needed to install an empty store up front.
    pub fn init_settings(&mut self) -> &mut Self {
        if self.world().get_resource::<Settings>().is_none() {
            self.insert_resource(Settings::new());
        }
        self.add_event::<SettingChanged>();
        self
    }

    /// Set `key` in `layer` and broadcast a [`SettingChanged`] event if the
    /// resolved value changed (design §14, §24.6).
    ///
    /// Auto-initialises the settings store and event. The event is buffered via
    /// the normal [`add_event`](App::add_event) rotation, so systems reading a
    /// [`SettingChanged`] cursor observe it for the frame it is sent and the
    /// frame after.
    pub fn insert_setting(
        &mut self,
        layer: SettingsLayer,
        key: impl Into<String>,
        value: impl Into<SettingValue>,
    ) -> &mut Self {
        self.init_settings();
        let change = self
            .world_mut()
            .resource_mut::<Settings>()
            .set(layer, key, value);
        if let Some(change) = change {
            self.send_event(change);
        }
        self
    }

    /// Clear `key`'s contribution from `layer` and broadcast a
    /// [`SettingChanged`] event if the resolved value changed.
    ///
    /// Auto-initialises the settings store and event.
    pub fn clear_setting(&mut self, layer: SettingsLayer, key: &str) -> &mut Self {
        self.init_settings();
        let change = self
            .world_mut()
            .resource_mut::<Settings>()
            .clear(layer, key);
        if let Some(change) = change {
            self.send_event(change);
        }
        self
    }

    /// Apply command-line-style overrides into the
    /// [`CommandLine`](SettingsLayer::CommandLine) layer, broadcasting one
    /// [`SettingChanged`] event per argument that changed a resolved value.
    ///
    /// Auto-initialises the settings store and event. See
    /// [`Settings::apply_cli_args`] for the accepted argument syntax; a
    /// headless/server build can be driven purely from these (design §14 "无头/
    /// 服务器可纯命令行驱动").
    pub fn apply_cli_overrides<I, S>(&mut self, args: I) -> &mut Self
    where
        I: IntoIterator<Item = S>,
        S: AsRef<str>,
    {
        self.init_settings();
        let changes = self
            .world_mut()
            .resource_mut::<Settings>()
            .apply_cli_args(args);
        for change in changes {
            self.send_event(change);
        }
        self
    }

    /// Resolve `key`'s effective [`SettingValue`], or `None` if unset (or if the
    /// settings store was never initialised).
    #[must_use]
    pub fn setting(&self, key: &str) -> Option<&SettingValue> {
        self.world().get_resource::<Settings>()?.get(key)
    }

    /// Resolve `key` as a `bool`, if set and boolean.
    #[must_use]
    pub fn setting_bool(&self, key: &str) -> Option<bool> {
        self.setting(key).and_then(SettingValue::as_bool)
    }

    /// Resolve `key` as an `i64`, if set and integral.
    #[must_use]
    pub fn setting_int(&self, key: &str) -> Option<i64> {
        self.setting(key).and_then(SettingValue::as_int)
    }

    /// Resolve `key` as an `f64`, if set and floating-point.
    #[must_use]
    pub fn setting_float(&self, key: &str) -> Option<f64> {
        self.setting(key).and_then(SettingValue::as_float)
    }

    /// Resolve `key` as a `&str`, if set and a string.
    #[must_use]
    pub fn setting_str(&self, key: &str) -> Option<&str> {
        self.setting(key).and_then(SettingValue::as_str)
    }
}
