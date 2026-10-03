//! Section 35 dialogue: runtime resolution, decision trees, localization,
//! captions, and viseme timelines.
//!
//! # Provenance
//! Original work; no Unreal Engine, Unity, Godot, Wwise, FMOD, Steam Audio, or
//! Google Resonance Audio source or derived code; no AI/ML.
//!
//! # Relationship
//! Implements the dialogue and localization pipeline of design section 35. A
//! [`DialogueResolver`] turns a semantic key (role / emotion / language /
//! variant) into a media reference; a [`DialogueDecisionTree`] walks
//! State/Switch values to a line or random variant with a generic fallback; a
//! [`LanguageBankSet`] hot-swaps per-language voice banks; a [`CaptionTrack`]
//! and [`VisemeTrack`] carry playhead-driven caption and lip-sync data. All of
//! this runs off the real-time thread; a miss yields silence plus a warning,
//! never a panic.

pub mod caption;
pub mod decision_tree;
pub mod localization;
pub mod resolver;
pub mod viseme;

pub use caption::{CaptionEvent, CaptionTrack, CaptionWord};
pub use decision_tree::{
    DecisionNode, DialogueDecisionTree, DialogueState, NodeIndex, TreeError,
};
pub use localization::{LanguageBankSet, LanguageId, SwitchOutcome};
pub use resolver::{DialogueKey, DialogueResolution, DialogueResolver, MediaRef};
pub use viseme::{Viseme, VisemeKeyframe, VisemePose, VisemeTrack};

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bank::manifest::BankId;
    use crate::bank::entry::EntryId;

    const EPSILON: f32 = 1.0e-4;

    fn media(bank: u32, entry: u32) -> MediaRef {
        MediaRef::new(BankId(bank), EntryId(entry))
    }

    #[test]
    fn resolver_exact_then_generic_then_silence() {
        let en = LanguageId::new("en");
        let mut resolver = DialogueResolver::new();
        resolver.insert(
            DialogueKey::new("hero", "angry", en.clone(), 2),
            media(1, 10),
        );
        resolver.insert(
            DialogueKey::new("hero", "angry", en.clone(), 0),
            media(1, 11),
        );

        // Exact variant hit.
        let exact = resolver.resolve(&DialogueKey::new("hero", "angry", en.clone(), 2));
        assert_eq!(exact.media(), Some(media(1, 10)));

        // Missing variant falls back to the generic variant 0.
        let generic = resolver.resolve(&DialogueKey::new("hero", "angry", en.clone(), 5));
        assert_eq!(generic.media(), Some(media(1, 11)));
        assert_eq!(resolver.miss_count(), 0);

        // Completely unknown role yields silence and warns.
        let miss = resolver.resolve(&DialogueKey::new("villain", "calm", en, 1));
        assert!(miss.is_silence());
        assert_eq!(resolver.miss_count(), 1);
    }

    #[test]
    fn decision_tree_walks_state_to_leaf() {
        // The arena root is node 0; leaves are referenced by index.
        let mut tree = DialogueDecisionTree::new();
        let root = tree.push(DecisionNode::Branch {
            key: String::from("who"),
            arms: vec![(String::from("hero"), NodeIndex(1))],
            fallback: Some(NodeIndex(2)),
        });
        assert_eq!(root.0, 0);
        tree.push(DecisionNode::Leaf {
            variants: vec![media(1, 1)],
        });
        tree.push(DecisionNode::Leaf {
            variants: vec![media(1, 99)],
        });
        tree.validate().unwrap();

        let hero = DialogueState::new().with("who", "hero");
        assert_eq!(tree.evaluate(&hero, 0), Some(media(1, 1)));

        // Unknown speaker takes the generic fallback (no silence).
        let other = DialogueState::new().with("who", "stranger");
        assert_eq!(tree.evaluate(&other, 0), Some(media(1, 99)));

        // Absent key also takes the fallback.
        let empty = DialogueState::new();
        assert_eq!(tree.evaluate(&empty, 0), Some(media(1, 99)));
    }

    #[test]
    fn decision_tree_variant_selection_is_deterministic() {
        let mut tree = DialogueDecisionTree::new();
        tree.push(DecisionNode::Leaf {
            variants: vec![media(1, 1), media(1, 2), media(1, 3), media(1, 4)],
        });
        let state = DialogueState::new();
        let a = tree.evaluate(&state, 12345);
        let b = tree.evaluate(&state, 12345);
        assert_eq!(a, b);
        assert!(a.is_some());
        // A different seed may pick a different take; across many seeds we
        // should see more than one distinct variant chosen.
        let mut seen = alloc::collections::BTreeSet::new();
        for seed in 0..64u64 {
            if let Some(m) = tree.evaluate(&state, seed) {
                seen.insert(m.entry.0);
            }
        }
        assert!(seen.len() > 1);
    }

    #[test]
    fn decision_tree_dangling_index_is_rejected() {
        let mut tree = DialogueDecisionTree::new();
        tree.push(DecisionNode::Branch {
            key: String::from("x"),
            arms: vec![(String::from("y"), NodeIndex(9))],
            fallback: None,
        });
        assert_eq!(tree.validate(), Err(TreeError::DanglingIndex(NodeIndex(9))));
    }

    #[test]
    fn language_hot_swap_falls_back_on_missing() {
        let en = LanguageId::new("en");
        let zh = LanguageId::new("zh");
        let ja = LanguageId::new("ja");
        let mut set = LanguageBankSet::new(en.clone(), BankId(100));
        set.register(zh.clone(), BankId(200));

        assert_eq!(set.active_bank(), BankId(100));

        // Switch to a registered language swaps only the voice bank.
        let outcome = set.set_active(zh.clone());
        assert_eq!(outcome, SwitchOutcome::Switched(zh.clone()));
        assert_eq!(set.active_bank(), BankId(200));

        // Switch to an unregistered language falls back + warns.
        let outcome = set.set_active(ja.clone());
        assert!(outcome.is_fallback());
        assert_eq!(set.active_bank(), BankId(100));
        assert_eq!(set.active_language(), &en);
        assert_eq!(set.fallback_count(), 1);
    }

    #[test]
    fn caption_track_drives_sentence_and_word_highlight() {
        let mut track = CaptionTrack::new(48_000);
        track.push(
            CaptionEvent::new(0, 100, "hello world")
                .with_word(CaptionWord::new(0, 50, "hello"))
                .with_word(CaptionWord::new(50, 100, "world")),
        );
        track.push(CaptionEvent::new(100, 200, "second line"));

        // Insertion keeps order even when pushed out of order.
        track.push(CaptionEvent::new(50, 60, "overlap"));
        assert_eq!(track.len(), 3);

        let event = track.active_at(10).unwrap();
        assert_eq!(event.text, "hello world");
        assert_eq!(event.highlighted_word(10), Some(0));
        assert_eq!(event.highlighted_word(70), Some(1));

        assert!(track.active_at(150).is_some());
        assert!(track.active_at(500).is_none());

        let in_range = track.events_in_range(90, 110);
        // "hello world" (ends at 100) and "second line" (starts at 100).
        assert_eq!(in_range.len(), 2);
    }

    #[test]
    fn viseme_track_interpolates_weight() {
        let mut track = VisemeTrack::new(48_000);
        track.push(VisemeKeyframe::new(0, Viseme::Silence, 0.0));
        track.push(VisemeKeyframe::new(100, Viseme::Aa, 1.0));
        track.push(VisemeKeyframe::new(200, Viseme::Silence, 0.0));

        let start = track.sample_at(0);
        assert_eq!(start.viseme, Viseme::Silence);
        assert!((start.weight - 0.0).abs() < EPSILON);

        // Halfway from the first to the second keyframe: weight ~ 0.5, shape
        // held at the earlier keyframe (Silence).
        let mid = track.sample_at(50);
        assert_eq!(mid.viseme, Viseme::Silence);
        assert!((mid.weight - 0.5).abs() < EPSILON);

        let peak = track.sample_at(100);
        assert_eq!(peak.viseme, Viseme::Aa);
        assert!((peak.weight - 1.0).abs() < EPSILON);

        // After the last keyframe the pose is held.
        let tail = track.sample_at(500);
        assert_eq!(tail.viseme, Viseme::Silence);
        assert!((tail.weight - 0.0).abs() < EPSILON);
    }
}
