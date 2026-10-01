//! Shared-element (Hero) transitions.
//!
//! A shared-element transition animates an element that exists in two layouts
//! (for example two screens) so it appears to fly from its source rectangle to
//! its destination rectangle, like Flutter's `Hero` widget or `SwiftUI`'s
//! `matchedGeometryEffect`.
//!
//! Elements are identified across the two layouts by their [`Key`].
//! [`SharedElementTransition::match_pairs`] pairs keys present in both the
//! `from` and `to` layouts; each pair yields a [`Transform`] that tweens from
//! source to destination. Keys present in only one layout are reported as
//! *entering* or *leaving* and fall back to a scalar appearance value so the
//! caller can fade / scale them in or out.
//!
//! Multiple matched elements can be scheduled together with
//! [`SharedElementTransition::stagger`], which reuses
//! [`prism_ui_anim::Choreography`].

use alloc::collections::BTreeMap;
use alloc::vec::Vec;

use prism_ui::Key;
use prism_ui_anim::{Choreography, Easing, Lerp};

use crate::geometry::{Rect, Transform};

/// Why an unpaired key uses a fallback instead of a matched transform.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FallbackRole {
    /// The key exists only in the destination layout: it is appearing.
    Entering,
    /// The key exists only in the source layout: it is disappearing.
    Leaving,
}

/// A matched source/destination pair for one key.
#[derive(Clone, Debug, PartialEq)]
pub struct SharedPair {
    key: Key,
    from: Rect,
    to: Rect,
    invert: Transform,
    easing: Easing,
}

impl SharedPair {
    fn new(key: Key, from: Rect, to: Rect, easing: Easing) -> Self {
        // Treat `to` as the final layout; invert maps it back onto `from`.
        let invert = Transform::from_rects(&from, &to);
        Self {
            key,
            from,
            to,
            invert,
            easing,
        }
    }

    /// The key this pair animates.
    #[must_use]
    pub fn key(&self) -> &Key {
        &self.key
    }

    /// The source rectangle.
    #[must_use]
    pub fn from_rect(&self) -> Rect {
        self.from
    }

    /// The destination rectangle.
    #[must_use]
    pub fn to_rect(&self) -> Rect {
        self.to
    }

    /// The inverted transform at the start of the transition.
    #[must_use]
    pub fn invert(&self) -> Transform {
        self.invert
    }

    /// Samples the transform at normalized progress `t`.
    ///
    /// `t` is clamped into `[0, 1]` and shaped by the pair's easing. At
    /// `t == 0` the element sits at its source rectangle; at `t == 1` it has
    /// reached its destination (identity transform).
    #[must_use]
    pub fn sample(&self, t: f32) -> Transform {
        let eased = self.easing.sample(clamp01(t));
        self.invert.lerp(&Transform::IDENTITY, eased)
    }
}

/// A set of shared-element transitions between two keyed layouts.
#[derive(Clone, Debug, Default)]
pub struct SharedElementTransition {
    duration: f32,
    easing: Easing,
    pairs: BTreeMap<Key, SharedPair>,
    entering: BTreeMap<Key, Rect>,
    leaving: BTreeMap<Key, Rect>,
}

impl SharedElementTransition {
    /// Creates an empty transition with the given default `duration` and
    /// `easing` applied to every matched pair.
    #[must_use]
    pub fn new(duration: f32, easing: Easing) -> Self {
        Self {
            duration,
            easing,
            pairs: BTreeMap::new(),
            entering: BTreeMap::new(),
            leaving: BTreeMap::new(),
        }
    }

    /// Builds a transition and immediately matches `from` against `to`.
    #[must_use]
    pub fn from_frames(
        from: &[(Key, Rect)],
        to: &[(Key, Rect)],
        duration: f32,
        easing: Easing,
    ) -> Self {
        let mut transition = Self::new(duration, easing);
        transition.match_pairs(from, to);
        transition
    }

    /// Pairs keys across the `from` and `to` layouts.
    ///
    /// Keys present in both layouts become matched [`SharedPair`]s. Keys only
    /// in `to` are recorded as *entering*; keys only in `from` as *leaving*.
    /// Any previous matching is discarded. When a key is duplicated within a
    /// layout, the last occurrence wins.
    pub fn match_pairs(&mut self, from: &[(Key, Rect)], to: &[(Key, Rect)]) {
        self.pairs.clear();
        self.entering.clear();
        self.leaving.clear();

        let from_map: BTreeMap<Key, Rect> = from.iter().cloned().collect();
        let to_map: BTreeMap<Key, Rect> = to.iter().cloned().collect();

        for (key, to_rect) in &to_map {
            match from_map.get(key) {
                Some(from_rect) => {
                    self.pairs.insert(
                        key.clone(),
                        SharedPair::new(key.clone(), *from_rect, *to_rect, self.easing),
                    );
                }
                None => {
                    self.entering.insert(key.clone(), *to_rect);
                }
            }
        }
        for (key, from_rect) in &from_map {
            if !to_map.contains_key(key) {
                self.leaving.insert(key.clone(), *from_rect);
            }
        }
    }

    /// The default transition duration in seconds.
    #[must_use]
    pub fn duration(&self) -> f32 {
        self.duration
    }

    /// Whether there are no matched, entering or leaving elements.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.pairs.is_empty() && self.entering.is_empty() && self.leaving.is_empty()
    }

    /// The number of matched pairs.
    #[must_use]
    pub fn matched_len(&self) -> usize {
        self.pairs.len()
    }

    /// The matched pair for `key`, if any.
    #[must_use]
    pub fn pair(&self, key: &Key) -> Option<&SharedPair> {
        self.pairs.get(key)
    }

    /// Samples the matched transform for `key` at normalized progress `t`.
    ///
    /// Returns `None` when `key` has no matched pair.
    #[must_use]
    pub fn sample(&self, key: &Key, t: f32) -> Option<Transform> {
        self.pairs.get(key).map(|pair| pair.sample(t))
    }

    /// The matched keys, in [`Key`] order.
    #[must_use]
    pub fn matched_keys(&self) -> Vec<Key> {
        self.pairs.keys().cloned().collect()
    }

    /// The entering keys (present only in the destination), in [`Key`] order.
    #[must_use]
    pub fn entering_keys(&self) -> Vec<Key> {
        self.entering.keys().cloned().collect()
    }

    /// The leaving keys (present only in the source), in [`Key`] order.
    #[must_use]
    pub fn leaving_keys(&self) -> Vec<Key> {
        self.leaving.keys().cloned().collect()
    }

    /// The [`FallbackRole`] of an unpaired `key`, if it is entering or leaving.
    ///
    /// Returns `None` for matched keys and keys that are not part of this
    /// transition at all.
    #[must_use]
    pub fn fallback_role(&self, key: &Key) -> Option<FallbackRole> {
        if self.entering.contains_key(key) {
            Some(FallbackRole::Entering)
        } else if self.leaving.contains_key(key) {
            Some(FallbackRole::Leaving)
        } else {
            None
        }
    }

    /// The fallback appearance of an unpaired `key` at normalized progress `t`.
    ///
    /// Entering elements fade in (`0 -> 1`); leaving elements fade out
    /// (`1 -> 0`). The value is shaped by the transition easing and clamped to
    /// `[0, 1]`. Returns `None` for matched or unknown keys.
    #[must_use]
    pub fn fallback_appearance(&self, key: &Key, t: f32) -> Option<f32> {
        let eased = self.easing.sample(clamp01(t));
        match self.fallback_role(key) {
            Some(FallbackRole::Entering) => Some(eased),
            Some(FallbackRole::Leaving) => Some(1.0 - eased),
            None => None,
        }
    }

    /// Builds a [`Choreography`] over the matched keys, staggering their
    /// starts by `stagger` seconds and giving each `item_duration` seconds.
    ///
    /// Track `i` corresponds to the `i`-th key in [`matched_keys`] order, so a
    /// list of shared elements can cascade instead of moving in lockstep.
    ///
    /// [`matched_keys`]: SharedElementTransition::matched_keys
    #[must_use]
    pub fn stagger(&self, item_duration: f32, stagger: f32, easing: Easing) -> Choreography {
        Choreography::stagger(self.pairs.len(), item_duration, stagger, easing)
    }
}

/// Clamps `t` into the normalized `[0, 1]` range.
fn clamp01(t: f32) -> f32 {
    t.clamp(0.0, 1.0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::string::ToString;

    const EPS: f32 = 1.0e-4;

    fn close(a: f32, b: f32) {
        assert!((a - b).abs() < EPS, "expected {a} ~= {b}");
    }

    fn k(id: i64) -> Key {
        Key::Int(id)
    }

    #[test]
    fn matched_pair_flies_from_source_to_destination() {
        let from = [(k(1), Rect::new(0.0, 0.0, 100.0, 100.0))];
        let to = [(k(1), Rect::new(300.0, 0.0, 50.0, 50.0))];
        let transition = SharedElementTransition::from_frames(&from, &to, 1.0, Easing::Linear);
        assert_eq!(transition.matched_len(), 1);

        let start = transition.sample(&k(1), 0.0).unwrap();
        // At the start the element should map its destination back onto source.
        let mapped = start.apply_rect(&to[0].1);
        close(mapped.x, from[0].1.x);
        close(mapped.width, from[0].1.width);

        let end = transition.sample(&k(1), 1.0).unwrap();
        assert!(end.is_identity());
    }

    #[test]
    fn unmatched_keys_enter_and_leave() {
        let from = [
            (k(1), Rect::new(0.0, 0.0, 10.0, 10.0)),
            (k(2), Rect::new(0.0, 0.0, 10.0, 10.0)),
        ];
        let to = [
            (k(1), Rect::new(5.0, 0.0, 10.0, 10.0)),
            (k(3), Rect::new(9.0, 0.0, 10.0, 10.0)),
        ];
        let transition = SharedElementTransition::from_frames(&from, &to, 1.0, Easing::Linear);

        assert_eq!(transition.matched_keys(), alloc::vec![k(1)]);
        assert_eq!(transition.entering_keys(), alloc::vec![k(3)]);
        assert_eq!(transition.leaving_keys(), alloc::vec![k(2)]);

        assert_eq!(
            transition.fallback_role(&k(3)),
            Some(FallbackRole::Entering)
        );
        assert_eq!(transition.fallback_role(&k(2)), Some(FallbackRole::Leaving));
        assert_eq!(transition.fallback_role(&k(1)), None);

        close(transition.fallback_appearance(&k(3), 0.25).unwrap(), 0.25);
        close(transition.fallback_appearance(&k(2), 0.25).unwrap(), 0.75);
        assert!(transition.sample(&k(2), 0.5).is_none());
    }

    #[test]
    fn string_keys_are_supported() {
        let from = [(
            Key::Str("card".to_string()),
            Rect::new(0.0, 0.0, 20.0, 20.0),
        )];
        let to = [(
            Key::Str("card".to_string()),
            Rect::new(40.0, 0.0, 20.0, 20.0),
        )];
        let transition = SharedElementTransition::from_frames(&from, &to, 0.5, Easing::EaseOut);
        let pair = transition.pair(&Key::Str("card".to_string())).unwrap();
        close(pair.invert().tx, -40.0);
        close(pair.from_rect().x, 0.0);
        close(pair.to_rect().x, 40.0);
    }

    #[test]
    fn rematching_replaces_previous_state() {
        let mut transition = SharedElementTransition::new(1.0, Easing::Linear);
        transition.match_pairs(
            &[(k(1), Rect::new(0.0, 0.0, 10.0, 10.0))],
            &[(k(1), Rect::new(5.0, 0.0, 10.0, 10.0))],
        );
        assert_eq!(transition.matched_len(), 1);
        transition.match_pairs(&[], &[(k(9), Rect::new(0.0, 0.0, 1.0, 1.0))]);
        assert_eq!(transition.matched_len(), 0);
        assert_eq!(transition.entering_keys(), alloc::vec![k(9)]);
        assert!(transition.leaving_keys().is_empty());
    }

    #[test]
    fn stagger_builds_one_track_per_pair() {
        let from = [
            (k(1), Rect::new(0.0, 0.0, 10.0, 10.0)),
            (k(2), Rect::new(0.0, 0.0, 10.0, 10.0)),
            (k(3), Rect::new(0.0, 0.0, 10.0, 10.0)),
        ];
        let to = [
            (k(1), Rect::new(1.0, 0.0, 10.0, 10.0)),
            (k(2), Rect::new(2.0, 0.0, 10.0, 10.0)),
            (k(3), Rect::new(3.0, 0.0, 10.0, 10.0)),
        ];
        let transition = SharedElementTransition::from_frames(&from, &to, 1.0, Easing::Linear);
        let choreo = transition.stagger(0.2, 0.1, Easing::Linear);
        assert_eq!(choreo.len(), 3);
        close(choreo.duration(), 0.4);
    }

    #[test]
    fn empty_transition_reports_empty() {
        let transition = SharedElementTransition::new(1.0, Easing::Linear);
        assert!(transition.is_empty());
        assert_eq!(transition.duration(), 1.0);
    }
}
