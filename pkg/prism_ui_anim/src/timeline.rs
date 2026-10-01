//! A sorted keyframe [`Timeline`] with per-segment easing.

use crate::easing::Easing;
use crate::lerp::Lerp;
use crate::math::clampf;
use alloc::vec::Vec;

/// A single keyframe: a `value` reached at a given `time`, plus the [`Easing`]
/// applied to the **outgoing** segment that starts at this keyframe.
///
/// The easing of the final keyframe is unused, since no segment starts there.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Keyframe<T> {
    /// The time (in the timeline's own units) at which `value` is reached.
    pub time: f32,
    /// The value at `time`.
    pub value: T,
    /// Easing applied from this keyframe to the next one.
    pub easing: Easing,
}

impl<T> Keyframe<T> {
    /// Create a keyframe with an explicit easing.
    #[inline]
    pub fn new(time: f32, value: T, easing: Easing) -> Self {
        Self {
            time,
            value,
            easing,
        }
    }

    /// Create a keyframe that eases linearly into the next segment.
    #[inline]
    pub fn linear(time: f32, value: T) -> Self {
        Self::new(time, value, Easing::Linear)
    }
}

/// A timeline of keyframes, kept sorted by ascending time.
///
/// Sampling clamps the query time to the timeline's `[first, last]` range and
/// interpolates within the containing segment using that segment's easing.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Timeline<T: Lerp> {
    keyframes: Vec<Keyframe<T>>,
}

impl<T: Lerp + Clone> Timeline<T> {
    /// Create an empty timeline.
    #[inline]
    pub fn new() -> Self {
        Self {
            keyframes: Vec::new(),
        }
    }

    /// Build a timeline from an unordered set of keyframes.
    ///
    /// The keyframes are sorted by time; equal times keep their relative
    /// order (a stable sort).
    pub fn from_keyframes(mut keyframes: Vec<Keyframe<T>>) -> Self {
        keyframes.sort_by(|a, b| {
            a.time
                .partial_cmp(&b.time)
                .unwrap_or(core::cmp::Ordering::Equal)
        });
        Self { keyframes }
    }

    /// Insert a keyframe, preserving the ascending-time ordering.
    pub fn push(&mut self, keyframe: Keyframe<T>) {
        let index = self
            .keyframes
            .partition_point(|existing| existing.time <= keyframe.time);
        self.keyframes.insert(index, keyframe);
    }

    /// The keyframes, in ascending-time order.
    #[inline]
    pub fn keyframes(&self) -> &[Keyframe<T>] {
        &self.keyframes
    }

    /// Whether the timeline has no keyframes.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.keyframes.is_empty()
    }

    /// Number of keyframes.
    #[inline]
    pub fn len(&self) -> usize {
        self.keyframes.len()
    }

    /// The time span covered by the timeline (last time minus first time).
    ///
    /// Returns `0.0` for an empty timeline.
    #[inline]
    pub fn duration(&self) -> f32 {
        match (self.keyframes.first(), self.keyframes.last()) {
            (Some(first), Some(last)) => last.time - first.time,
            _ => 0.0,
        }
    }

    /// Sample the timeline at `time`, returning `None` if it is empty.
    ///
    /// `time` is clamped to the `[first, last]` range before interpolation.
    pub fn try_sample(&self, time: f32) -> Option<T> {
        let first = self.keyframes.first()?;
        let last = self.keyframes.last()?;

        if time <= first.time {
            return Some(first.value.clone());
        }
        if time >= last.time {
            return Some(last.value.clone());
        }

        // Find the segment [i, i+1] containing `time`. The partition point is
        // the first keyframe strictly after `time`, so the segment starts at
        // the keyframe just before it.
        let next = self.keyframes.partition_point(|kf| kf.time <= time).max(1);
        let start = &self.keyframes[next - 1];
        let end = &self.keyframes[next];

        let span = end.time - start.time;
        let local = if span > 0.0 {
            clampf((time - start.time) / span, 0.0, 1.0)
        } else {
            0.0
        };
        let eased = start.easing.sample(local);
        Some(start.value.lerp(&end.value, eased))
    }

    /// Sample the timeline at `time`.
    ///
    /// # Panics
    ///
    /// Panics if the timeline is empty. Use [`Timeline::try_sample`] to handle
    /// that case explicitly.
    #[inline]
    pub fn sample(&self, time: f32) -> T {
        self.try_sample(time)
            .expect("Timeline::sample called on an empty timeline")
    }
}
