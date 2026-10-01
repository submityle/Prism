//! `prism_ui_i18n` — Loom's internationalization layer.
//!
//! This crate provides reactive message catalogs with argument interpolation
//! and `CLDR`-style plural selection, layered on top of `prism_ui_reactive`.
//! The current locale is held in a reactive `Signal`, so translations built via
//! `rt.memo(...)` recompute automatically when the locale is switched.
//!
//! # Building blocks
//!
//! * [`LocaleId`] — a cheap owned locale identifier.
//! * [`Catalog`] — one locale's keyed [`Message`]s (simple or plural).
//! * [`PluralCategory`] / [`PluralRules`] — integer-only plural selection.
//! * [`Args`] / [`Value`] — the interpolation argument map.
//! * [`I18n`] — the reactive facade tying everything together.
//!
//! # Example
//!
//! ```
//! use prism_ui_i18n::{Args, Catalog, I18n, PluralRules};
//! use prism_ui_reactive::Runtime;
//!
//! let rt = Runtime::new();
//! let mut i18n = I18n::new(&rt, "en");
//! i18n.register(
//!     "en",
//!     Catalog::new().with("greet", "Hello, {name}!"),
//!     false,
//!     PluralRules::English,
//! );
//! i18n.register(
//!     "fr",
//!     Catalog::new().with("greet", "Bonjour, {name}!"),
//!     false,
//!     PluralRules::English,
//! );
//!
//! let hello = i18n.translation("greet", Args::new().with("name", "Ada"));
//! assert_eq!(hello.get(), "Hello, Ada!");
//!
//! i18n.set_locale("fr");
//! assert_eq!(hello.get(), "Bonjour, Ada!");
//! ```
#![cfg_attr(not(feature = "std"), no_std)]
#![forbid(unsafe_code)]

extern crate alloc;

pub mod catalog;
pub mod format;
pub mod locale;
pub mod plural;

pub use catalog::{Catalog, Message};
pub use format::{interpolate, Args, Value};
pub use locale::{I18n, LocaleId};
pub use plural::{PluralCategory, PluralRules};

#[cfg(all(test, feature = "std"))]
mod tests {
    use alloc::rc::Rc;
    use core::cell::RefCell;

    use prism_ui_reactive::Runtime;

    use super::*;

    fn english_i18n() -> I18n {
        let rt = Runtime::new();
        let mut i18n = I18n::new(&rt, "en");
        i18n.register(
            "en",
            Catalog::new()
                .with("greet", "Hello, {name}!")
                .with("brace", "Use {{ and }} like {sym}.")
                .with_plural(
                    "items",
                    [
                        (PluralCategory::One, "{count} item"),
                        (PluralCategory::Other, "{count} items"),
                    ],
                ),
            false,
            PluralRules::English,
        );
        i18n
    }

    #[test]
    fn basic_interpolation() {
        let i18n = english_i18n();
        let out = i18n.t("greet", &Args::new().with("name", "Ada"));
        assert_eq!(out, "Hello, Ada!");
    }

    #[test]
    fn escaped_braces() {
        let i18n = english_i18n();
        let out = i18n.t("brace", &Args::new().with("sym", "x"));
        assert_eq!(out, "Use { and } like x.");
    }

    #[test]
    fn unknown_placeholder_left_verbatim() {
        let out = interpolate("Hi {name}, {missing}!", &Args::new().with("name", "Ada"));
        assert_eq!(out, "Hi Ada, {missing}!");
    }

    #[test]
    fn missing_key_falls_back_then_to_key() {
        let rt = Runtime::new();
        let mut i18n = I18n::new(&rt, "en");
        i18n.register(
            "en",
            Catalog::new().with("only_en", "English only"),
            false,
            PluralRules::English,
        );
        i18n.register("fr", Catalog::new(), false, PluralRules::English);

        i18n.set_locale("fr");
        // Falls back to the default (en) catalog.
        assert_eq!(i18n.t("only_en", &Args::new()), "English only");
        // Not present anywhere: yields the key itself.
        assert_eq!(i18n.t("nope", &Args::new()), "nope");
    }

    #[test]
    fn english_plural_one_other() {
        let i18n = english_i18n();
        assert_eq!(i18n.t_plural("items", 1, &Args::new()), "1 item");
        assert_eq!(i18n.t_plural("items", 0, &Args::new()), "0 items");
        assert_eq!(i18n.t_plural("items", 5, &Args::new()), "5 items");
    }

    #[test]
    fn slavic_plural_one_few_many() {
        let rt = Runtime::new();
        let mut i18n = I18n::new(&rt, "en");
        i18n.register("en", Catalog::new(), false, PluralRules::English);
        i18n.register(
            "ru",
            Catalog::new().with_plural(
                "items",
                [
                    (PluralCategory::One, "{count} товар"),
                    (PluralCategory::Few, "{count} товара"),
                    (PluralCategory::Many, "{count} товаров"),
                ],
            ),
            false,
            PluralRules::Slavic,
        );
        i18n.set_locale("ru");

        assert_eq!(i18n.t_plural("items", 1, &Args::new()), "1 товар");
        assert_eq!(i18n.t_plural("items", 2, &Args::new()), "2 товара");
        assert_eq!(i18n.t_plural("items", 5, &Args::new()), "5 товаров");
        assert_eq!(i18n.t_plural("items", 11, &Args::new()), "11 товаров");
        assert_eq!(i18n.t_plural("items", 21, &Args::new()), "21 товар");
        assert_eq!(i18n.t_plural("items", 22, &Args::new()), "22 товара");
    }

    #[test]
    fn reactive_locale_switch_updates_memo() {
        let rt = Runtime::new();
        let mut i18n = I18n::new(&rt, "en");
        i18n.register(
            "en",
            Catalog::new().with("greet", "Hello, {name}!"),
            false,
            PluralRules::English,
        );
        i18n.register(
            "fr",
            Catalog::new().with("greet", "Bonjour, {name}!"),
            false,
            PluralRules::English,
        );

        let memo = i18n.translation("greet", Args::new().with("name", "Ada"));
        let log: Rc<RefCell<Vec<String>>> = Rc::new(RefCell::new(Vec::new()));
        let _effect = rt.effect({
            let memo = memo.clone();
            let log = log.clone();
            move || log.borrow_mut().push(memo.get())
        });

        assert_eq!(memo.get(), "Hello, Ada!");
        i18n.set_locale("fr");
        assert_eq!(memo.get(), "Bonjour, Ada!");
        assert_eq!(
            *log.borrow(),
            vec!["Hello, Ada!".to_string(), "Bonjour, Ada!".to_string()]
        );
    }

    #[test]
    fn is_rtl_tracks_locale() {
        let rt = Runtime::new();
        let mut i18n = I18n::new(&rt, "en");
        i18n.register("en", Catalog::new(), false, PluralRules::English);
        i18n.register("ar", Catalog::new(), true, PluralRules::OtherOnly);

        let rtl = rt.memo({
            let i18n = i18n.clone();
            move || i18n.is_rtl()
        });

        assert!(!rtl.get());
        i18n.set_locale("ar");
        assert!(rtl.get());
        i18n.set_locale("en");
        assert!(!rtl.get());
    }

    #[test]
    fn negative_integer_formatting() {
        let out = interpolate("{n}", &Args::new().with("n", -42i64));
        assert_eq!(out, "-42");
        let zero = interpolate("{n}", &Args::new().with("n", 0i64));
        assert_eq!(zero, "0");
        let min = interpolate("{n}", &Args::new().with("n", i64::MIN));
        assert_eq!(min, "-9223372036854775808");
    }
}
