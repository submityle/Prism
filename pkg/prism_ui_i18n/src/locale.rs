//! Locale identifiers and the reactive [`I18n`] facade.
//!
//! [`I18n`] owns a reactive `Signal<LocaleId>` for the current locale plus a
//! registry of per-locale catalogs, direction flags, and plural rules.
//! Translations resolved through [`I18n::t`], [`I18n::t_plural`], and
//! [`I18n::translation`] read the current-locale signal, so values built with
//! `rt.memo(...)` recompute automatically when [`I18n::set_locale`] is called.

use alloc::collections::BTreeMap;
use alloc::rc::Rc;
use alloc::string::String;
use core::cell::RefCell;

use prism_ui_reactive::{Memo, Runtime, Signal};

use crate::catalog::{Catalog, Message};
use crate::format::{interpolate, Args, Value};
use crate::plural::PluralRules;

/// A small owned, interned locale identifier such as `LocaleId("en")`.
#[derive(Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct LocaleId(String);

impl LocaleId {
    /// Create a locale identifier from any string-like value.
    pub fn new(id: impl Into<String>) -> Self {
        LocaleId(id.into())
    }

    /// Borrow the underlying identifier string.
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl From<&str> for LocaleId {
    fn from(value: &str) -> Self {
        LocaleId(String::from(value))
    }
}

impl From<String> for LocaleId {
    fn from(value: String) -> Self {
        LocaleId(value)
    }
}

/// Per-locale registered data: its catalog, text direction, and plural rules.
struct LocaleData {
    catalog: Catalog,
    is_rtl: bool,
    rules: PluralRules,
}

/// Shared interior state behind an [`I18n`] handle.
struct Inner {
    runtime: Runtime,
    current: Signal<LocaleId>,
    default_locale: LocaleId,
    locales: RefCell<BTreeMap<LocaleId, LocaleData>>,
}

impl Inner {
    /// Resolve `key` against the current locale, falling back to the default
    /// locale and finally to the key string itself. Reads the current-locale
    /// signal, recording a reactive dependency.
    fn render(&self, key: &str, count: Option<u64>, args: &Args) -> String {
        let locale = self.current.get();
        if let Some(text) = self.try_locale(&locale, key, count, args) {
            return text;
        }
        if locale != self.default_locale
            && let Some(text) = self.try_locale(&self.default_locale, key, count, args)
        {
            return text;
        }
        String::from(key)
    }

    /// Attempt to resolve `key` within a specific locale's catalog.
    fn try_locale(
        &self,
        locale: &LocaleId,
        key: &str,
        count: Option<u64>,
        args: &Args,
    ) -> Option<String> {
        let locales = self.locales.borrow();
        let data = locales.get(locale)?;
        let message = data.catalog.message(key)?;
        match message {
            Message::Simple(template) => Some(interpolate(template, args)),
            Message::Plural(variants) => {
                let category = data.rules.select(count.unwrap_or(0));
                let template = variants
                    .get(&category)
                    .or_else(|| variants.get(&crate::plural::PluralCategory::Other))
                    .or_else(|| variants.values().next())?;
                Some(interpolate(template, args))
            }
        }
    }
}

/// The reactive internationalization facade.
#[derive(Clone)]
pub struct I18n {
    inner: Rc<Inner>,
}

impl I18n {
    /// Create a new facade bound to `runtime`, starting at `default_locale`.
    pub fn new(runtime: &Runtime, default_locale: impl Into<LocaleId>) -> Self {
        let default_locale = default_locale.into();
        let current = runtime.signal(default_locale.clone());
        I18n {
            inner: Rc::new(Inner {
                runtime: runtime.clone(),
                current,
                default_locale,
                locales: RefCell::new(BTreeMap::new()),
            }),
        }
    }

    /// Register (or replace) a locale with its catalog, direction, and rules.
    pub fn register(
        &mut self,
        locale: impl Into<LocaleId>,
        catalog: Catalog,
        is_rtl: bool,
        plural_rules: PluralRules,
    ) -> &mut Self {
        self.inner.locales.borrow_mut().insert(
            locale.into(),
            LocaleData {
                catalog,
                is_rtl,
                rules: plural_rules,
            },
        );
        self
    }

    /// Reactively switch the current locale. No notification is emitted when the
    /// locale is unchanged.
    pub fn set_locale(&self, locale: impl Into<LocaleId>) {
        self.inner.current.set_if_changed(locale.into());
    }

    /// The current locale (records a reactive dependency).
    pub fn locale(&self) -> LocaleId {
        self.inner.current.get()
    }

    /// Whether the current locale is right-to-left (records a dependency).
    /// Unknown locales report left-to-right.
    pub fn is_rtl(&self) -> bool {
        let locale = self.inner.current.get();
        self.inner
            .locales
            .borrow()
            .get(&locale)
            .is_some_and(|data| data.is_rtl)
    }

    /// Translate `key` with `args` against the current locale.
    ///
    /// Resolution falls back to the default locale when the key is missing, and
    /// finally yields the key string itself. When `key` names a plural message,
    /// the category for count `0` is used.
    pub fn t(&self, key: &str, args: &Args) -> String {
        self.inner.render(key, None, args)
    }

    /// Translate a plural `key`, selecting a category from `count`.
    ///
    /// The `count` is also exposed to the template as an implicit `{count}`
    /// argument unless `args` already binds `count`.
    pub fn t_plural(&self, key: &str, count: u64, args: &Args) -> String {
        let mut full = args.clone();
        if full.get("count").is_none() {
            full.set(
                "count",
                Value::Num(i64::try_from(count).unwrap_or(i64::MAX)),
            );
        }
        self.inner.render(key, Some(count), &full)
    }

    /// Build a reactive [`Memo`] translating a fixed `key` with fixed `args`.
    ///
    /// The memo recomputes whenever the current locale changes, making it ideal
    /// for binding a translation into reactive UI.
    pub fn translation(&self, key: impl Into<String>, args: Args) -> Memo<String> {
        let inner = self.inner.clone();
        let key = key.into();
        self.inner
            .runtime
            .memo(move || inner.render(&key, None, &args))
    }

    /// The reactive runtime backing this facade.
    pub fn runtime(&self) -> &Runtime {
        &self.inner.runtime
    }
}
