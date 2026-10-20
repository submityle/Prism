//! Typed substitutes must not alter identity, load state, failures or readiness.

use crate::asset::Asset;
use crate::error::{AssetError, ErrorRegistry};
use crate::fallback::{AssetFallbacks, FallbackKind, FallbackSource};
use crate::load_state::LoadState;
use crate::storage::Assets;

#[derive(Debug, PartialEq, Eq)]
struct Sample(u32);

impl Asset for Sample {
    const TYPE_NAME: &'static str = "test::fallback::Sample";
}

/// Loading chooses the placeholder; failure chooses its own fallback and keeps diagnostics.
#[test]
fn substitutes_preserve_true_state_and_events() {
    let mut assets = Assets::<Sample>::new();
    assets.fallbacks_mut().set_budget_bytes(8).unwrap();
    assets
        .fallbacks_mut()
        .register(FallbackKind::Placeholder, Sample(1), 4)
        .unwrap();
    assets
        .fallbacks_mut()
        .register(FallbackKind::Failure, Sample(2), 4)
        .unwrap();
    let handle = assets.reserve();
    let loading = assets.get_or_fallback(handle.id()).unwrap();
    assert_eq!(loading.value, &Sample(1));
    assert_eq!(loading.source, FallbackSource::Placeholder);
    assert_eq!(loading.load_state, LoadState::Loading);
    assert!(assets.get(handle.id()).is_none());
    assert!(!assets.is_ready(handle.id()));
    assert!(assets.drain_events().is_empty());
    let mut errors = ErrorRegistry::new();
    let error = errors.record(AssetError::new("sample", "decode rejected"));
    assert!(assets.fail(handle.id(), error));
    let failed = assets.get_or_fallback(handle.id()).unwrap();
    assert_eq!(failed.value, &Sample(2));
    assert_eq!(failed.source, FallbackSource::Fallback);
    assert_eq!(failed.load_state, LoadState::Failed(error));
    let events = assets.drain_events();
    assert_eq!(events.len(), 1);
    assert_eq!(events[0].error(), Some(error));
    assert_eq!(errors.get(error).unwrap().reason, "decode rejected");
    assert!(assets.fulfill(handle.id(), Sample(3)));
    let loaded = assets.get_or_fallback(handle.id()).unwrap();
    assert_eq!(loaded.value, &Sample(3));
    assert_eq!(loaded.source, FallbackSource::Asset);
    assert_eq!(loaded.load_state, LoadState::Loaded);
}

/// Without a dedicated placeholder, loading may present the failure substitute.
#[test]
fn loading_can_use_fallback_but_failure_never_uses_placeholder() {
    let mut assets = Assets::<Sample>::new();
    let handle = assets.reserve();
    assert!(assets.get_or_fallback(handle.id()).is_none());
    assets
        .fallbacks_mut()
        .register(FallbackKind::Failure, Sample(8), 0)
        .unwrap();
    assert_eq!(
        assets.get_or_fallback(handle.id()).unwrap().source,
        FallbackSource::Fallback
    );
    assets.fallbacks_mut().clear(FallbackKind::Failure);
    assets
        .fallbacks_mut()
        .register(FallbackKind::Placeholder, Sample(9), 0)
        .unwrap();
    let error = ErrorRegistry::new().record(AssetError::new("sample", "failure"));
    assets.fail(handle.id(), error);
    assert!(assets.get_or_fallback(handle.id()).is_none());
}

/// Stale ids stay invalid despite a globally available substitute of the same type.
#[test]
fn recycled_generation_never_receives_old_fallback_view() {
    let mut assets = Assets::<Sample>::new();
    assets
        .fallbacks_mut()
        .register(FallbackKind::Failure, Sample(99), 0)
        .unwrap();
    let old = assets.reserve();
    assets.remove(old.id());
    let new = assets.reserve();
    assert_eq!(old.id().index().index(), new.id().index().index());
    assert!(assets.get_or_fallback(old.id()).is_none());
    assert_eq!(assets.get_or_fallback(new.id()).unwrap().value, &Sample(99));
}

/// Failed replacement/shrink is atomic; removal releases exactly its own reservation.
#[test]
fn fallback_budget_replacement_is_atomic() {
    let mut values = AssetFallbacks::<Sample>::new(8);
    values
        .register(FallbackKind::Placeholder, Sample(1), 4)
        .unwrap();
    values
        .register(FallbackKind::Failure, Sample(2), 4)
        .unwrap();
    let error = values
        .register(FallbackKind::Failure, Sample(3), 5)
        .unwrap_err();
    assert_eq!(error.required_bytes, 9);
    assert_eq!(values.get(FallbackKind::Failure), Some(&Sample(2)));
    assert!(values.set_budget_bytes(7).is_err());
    assert_eq!(values.budget_bytes(), 8);
    assert_eq!(values.used_bytes(), 8);
    assert_eq!(values.clear(FallbackKind::Placeholder), Some(Sample(1)));
    assert_eq!(values.used_bytes(), 4);
    assert_eq!(
        values
            .register(FallbackKind::Failure, Sample(4), 8)
            .unwrap(),
        Some(Sample(2))
    );
    assert_eq!(values.used_bytes(), 8);
}

/// Proposed sums use u128 and cannot wrap a u64 budget into an apparent fit.
#[test]
fn fallback_byte_sum_cannot_overflow() {
    let mut values = AssetFallbacks::<Sample>::new(u64::MAX);
    values
        .register(FallbackKind::Placeholder, Sample(1), u64::MAX)
        .unwrap();
    let error = values
        .register(FallbackKind::Failure, Sample(2), 1)
        .unwrap_err();
    assert_eq!(error.required_bytes, u128::from(u64::MAX) + 1);
    assert_eq!(values.used_bytes(), u64::MAX);
}
