//! Algebra over [`CapabilitySet`] allow-lists.
//!
//! Capability negotiation in a server-driven UI is defense-in-depth: the policy
//! a document is actually sanitized against should be the *intersection* of what
//! the host build trusts and what a given tenant (A/B slot, live-ops surface) is
//! granted, so neither side can unilaterally widen the surface. Likewise a host
//! may wish to *union* several role grants, compute the *difference* between a
//! requested and an effective policy to report what was dropped, or assert that
//! one policy *is a subset* of another before promoting it.
//!
//! These are pure set operations applied independently across the three
//! namespaces ([`CapabilitySet`] tracks kinds, style tokens and events
//! separately), implemented on top of the public membership queries and
//! iterators so they never need privileged access to the internal ordered sets.

use crate::capability::CapabilitySet;

impl CapabilitySet {
    /// Returns the capabilities present in **both** `self` and `other`.
    ///
    /// Each namespace is intersected independently. This is the operation used
    /// to combine a host whitelist with a tenant grant: the effective policy can
    /// never exceed either input.
    #[must_use]
    pub fn intersect(&self, other: &Self) -> Self {
        Self::new()
            .allow_kinds(self.kinds().filter(|&k| other.allows_kind(k)))
            .allow_styles(self.styles().filter(|&s| other.allows_style(s)))
            .allow_events(self.events().filter(|&e| other.allows_event(e)))
    }

    /// Returns the capabilities present in **either** `self` or `other`.
    ///
    /// Each namespace is unioned independently; duplicates collapse because the
    /// underlying allow-lists are ordered sets.
    #[must_use]
    pub fn union(&self, other: &Self) -> Self {
        Self::new()
            .allow_kinds(self.kinds().chain(other.kinds()))
            .allow_styles(self.styles().chain(other.styles()))
            .allow_events(self.events().chain(other.events()))
    }

    /// Returns the capabilities present in `self` but **not** in `other`.
    ///
    /// Useful for reporting exactly which capabilities a stricter policy stripped
    /// away relative to a requested one.
    #[must_use]
    pub fn difference(&self, other: &Self) -> Self {
        Self::new()
            .allow_kinds(self.kinds().filter(|&k| !other.allows_kind(k)))
            .allow_styles(self.styles().filter(|&s| !other.allows_style(s)))
            .allow_events(self.events().filter(|&e| !other.allows_event(e)))
    }

    /// Whether every capability in `self` is also granted by `other`.
    ///
    /// The empty set is a subset of everything. All three namespaces must be
    /// contained for the whole set to be a subset.
    #[must_use]
    pub fn is_subset(&self, other: &Self) -> bool {
        self.kinds().all(|k| other.allows_kind(k))
            && self.styles().all(|s| other.allows_style(s))
            && self.events().all(|e| other.allows_event(e))
    }
}

#[cfg(test)]
mod tests {
    use super::CapabilitySet;
    use alloc::vec::Vec;

    struct SplitMix64(u64);
    impl SplitMix64 {
        fn next_u64(&mut self) -> u64 {
            self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut z = self.0;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            z ^ (z >> 31)
        }
    }

    // A deliberately small token alphabet per namespace so random sets overlap
    // frequently, exercising the interesting intersect/difference cases.
    const KIND_TOKENS: [&str; 6] = ["box", "text", "image", "row", "col", "script"];
    const STYLE_TOKENS: [&str; 6] = ["card", "danger", "muted", "bold", "pad", "warn"];
    const EVENT_TOKENS: [&str; 6] = ["tap", "hover", "drag", "drop", "focus", "blur"];

    fn random_set(rng: &mut SplitMix64) -> CapabilitySet {
        let mut caps = CapabilitySet::new();
        // Each token is included with ~50% probability, drawn from one random word.
        let bits = rng.next_u64();
        for (i, &k) in KIND_TOKENS.iter().enumerate() {
            if bits & (1 << i) != 0 {
                caps = caps.allow_kind(k);
            }
        }
        for (i, &s) in STYLE_TOKENS.iter().enumerate() {
            if bits & (1 << (i + 8)) != 0 {
                caps = caps.allow_style(s);
            }
        }
        for (i, &e) in EVENT_TOKENS.iter().enumerate() {
            if bits & (1 << (i + 16)) != 0 {
                caps = caps.allow_event(e);
            }
        }
        caps
    }

    // Membership check across every namespace token, for exhaustive oracles.
    fn check_all(
        caps: &CapabilitySet,
        kind: impl Fn(&str) -> bool,
        style: impl Fn(&str) -> bool,
        event: impl Fn(&str) -> bool,
    ) {
        for &k in &KIND_TOKENS {
            assert_eq!(caps.allows_kind(k), kind(k), "kind {k}");
        }
        for &s in &STYLE_TOKENS {
            assert_eq!(caps.allows_style(s), style(s), "style {s}");
        }
        for &e in &EVENT_TOKENS {
            assert_eq!(caps.allows_event(e), event(e), "event {e}");
        }
    }

    #[test]
    fn intersect_matches_membership_oracle() {
        let mut rng = SplitMix64(0x1234_5678);
        for _ in 0..2000 {
            let a = random_set(&mut rng);
            let b = random_set(&mut rng);
            let r = a.intersect(&b);
            check_all(
                &r,
                |k| a.allows_kind(k) && b.allows_kind(k),
                |s| a.allows_style(s) && b.allows_style(s),
                |e| a.allows_event(e) && b.allows_event(e),
            );
            // Intersection is commutative.
            assert_eq!(r, b.intersect(&a));
            // Idempotent.
            assert_eq!(a.intersect(&a), a);
            // The intersection is a subset of each input.
            assert!(r.is_subset(&a));
            assert!(r.is_subset(&b));
        }
    }

    #[test]
    fn union_matches_membership_oracle() {
        let mut rng = SplitMix64(0x9ABC_DEF0);
        for _ in 0..2000 {
            let a = random_set(&mut rng);
            let b = random_set(&mut rng);
            let r = a.union(&b);
            check_all(
                &r,
                |k| a.allows_kind(k) || b.allows_kind(k),
                |s| a.allows_style(s) || b.allows_style(s),
                |e| a.allows_event(e) || b.allows_event(e),
            );
            // Union is commutative and idempotent.
            assert_eq!(r, b.union(&a));
            assert_eq!(a.union(&a), a);
            // Each input is a subset of the union.
            assert!(a.is_subset(&r));
            assert!(b.is_subset(&r));
        }
    }

    #[test]
    fn difference_matches_membership_oracle() {
        let mut rng = SplitMix64(0xFEED_FACE);
        for _ in 0..2000 {
            let a = random_set(&mut rng);
            let b = random_set(&mut rng);
            let r = a.difference(&b);
            check_all(
                &r,
                |k| a.allows_kind(k) && !b.allows_kind(k),
                |s| a.allows_style(s) && !b.allows_style(s),
                |e| a.allows_event(e) && !b.allows_event(e),
            );
            // A difference is always a subset of the minuend.
            assert!(r.is_subset(&a));
            // (a - b) and b are disjoint.
            assert_eq!(r.intersect(&b), CapabilitySet::new());
            // (a - b) union (a intersect b) reconstructs a.
            assert_eq!(r.union(&a.intersect(&b)), a);
        }
    }

    #[test]
    fn is_subset_matches_token_oracle() {
        let mut rng = SplitMix64(0x0BAD_C0DE);
        for _ in 0..2000 {
            let a = random_set(&mut rng);
            let b = random_set(&mut rng);
            let expected = KIND_TOKENS
                .iter()
                .all(|&k| !a.allows_kind(k) || b.allows_kind(k))
                && STYLE_TOKENS
                    .iter()
                    .all(|&s| !a.allows_style(s) || b.allows_style(s))
                && EVENT_TOKENS
                    .iter()
                    .all(|&e| !a.allows_event(e) || b.allows_event(e));
            assert_eq!(a.is_subset(&b), expected);
            // Reflexive.
            assert!(a.is_subset(&a));
            // Antisymmetry: mutual subset implies equality.
            if a.is_subset(&b) && b.is_subset(&a) {
                assert_eq!(a, b);
            }
        }
    }

    #[test]
    fn empty_set_edge_cases() {
        let empty = CapabilitySet::new();
        let full = CapabilitySet::new()
            .allow_kinds(["box", "text"])
            .allow_style("card")
            .allow_event("tap");
        assert_eq!(empty.intersect(&full), empty);
        assert_eq!(full.intersect(&empty), empty);
        assert_eq!(empty.union(&full), full);
        assert_eq!(full.difference(&full), empty);
        assert_eq!(full.difference(&empty), full);
        assert_eq!(empty.difference(&full), empty);
        assert!(empty.is_subset(&full));
        assert!(empty.is_subset(&empty));
        assert!(!full.is_subset(&empty));
    }

    #[test]
    fn fixed_small_case() {
        let a = CapabilitySet::new()
            .allow_kinds(["box", "text", "image"])
            .allow_styles(["card", "danger"])
            .allow_events(["tap"]);
        let b = CapabilitySet::new()
            .allow_kinds(["text", "image", "script"])
            .allow_styles(["danger", "muted"])
            .allow_events(["hover"]);

        let inter = a.intersect(&b);
        let inter_kinds: Vec<&str> = inter.kinds().collect();
        assert_eq!(inter_kinds, ["image", "text"]);
        let inter_styles: Vec<&str> = inter.styles().collect();
        assert_eq!(inter_styles, ["danger"]);
        assert_eq!(inter.events().count(), 0);

        let diff = a.difference(&b);
        let diff_kinds: Vec<&str> = diff.kinds().collect();
        assert_eq!(diff_kinds, ["box"]);
        let diff_events: Vec<&str> = diff.events().collect();
        assert_eq!(diff_events, ["tap"]);

        let uni = a.union(&b);
        let uni_kinds: Vec<&str> = uni.kinds().collect();
        assert_eq!(uni_kinds, ["box", "image", "script", "text"]);
    }
}
