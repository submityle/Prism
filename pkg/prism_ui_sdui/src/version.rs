//! Schema versioning and client/server version negotiation.
//!
//! A server-driven UI document declares the schema [`Version`] it was authored
//! against. A client advertises the [`VersionSet`] it is able to render, and
//! [`negotiate`] picks the version both can speak, degrading gracefully when
//! the server is ahead of the client.

use alloc::vec::Vec;

/// A server-driven UI schema version, expressed as `major.minor`.
///
/// Compatibility follows the usual rule: equal `major` values describe the same
/// schema family, and a higher `minor` is a strict superset of the features of
/// a lower `minor` within that family. A client that only speaks an older
/// `minor` can therefore still render a newer document by degrading the nodes
/// it does not understand (see [`crate::sandbox`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Version {
    /// Breaking schema family. Different `major` values are mutually
    /// incompatible.
    pub major: u16,
    /// Backwards-compatible feature level within a `major`.
    pub minor: u16,
}

impl Version {
    /// Creates a version from its `major` and `minor` components.
    ///
    /// # Examples
    ///
    /// ```
    /// use prism_ui_sdui::Version;
    ///
    /// let v = Version::new(2, 3);
    /// assert_eq!(v.major, 2);
    /// assert_eq!(v.minor, 3);
    /// ```
    #[must_use]
    pub const fn new(major: u16, minor: u16) -> Self {
        Self { major, minor }
    }

    /// Whether `self` and `other` belong to the same schema family, meaning
    /// they share a `major` version.
    #[must_use]
    pub const fn same_family(self, other: Self) -> bool {
        self.major == other.major
    }
}

/// The set of schema versions a client is able to render.
///
/// The set is kept sorted and de-duplicated so negotiation is deterministic
/// regardless of insertion order.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct VersionSet {
    versions: Vec<Version>,
}

impl VersionSet {
    /// Creates an empty set that supports no versions.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Adds a supported version, keeping the set sorted and de-duplicated.
    pub fn insert(&mut self, version: Version) {
        if let Err(pos) = self.versions.binary_search(&version) {
            self.versions.insert(pos, version);
        }
    }

    /// The supported versions, in ascending order.
    #[must_use]
    pub fn versions(&self) -> &[Version] {
        &self.versions
    }

    /// Whether `version` is supported exactly.
    #[must_use]
    pub fn supports(&self, version: Version) -> bool {
        self.versions.binary_search(&version).is_ok()
    }

    /// The highest supported version in the same family as `target`, if the
    /// client supports that family at all.
    #[must_use]
    pub fn highest_in_family(&self, target: Version) -> Option<Version> {
        self.versions
            .iter()
            .copied()
            .filter(|v| v.same_family(target))
            .max()
    }
}

impl FromIterator<Version> for VersionSet {
    fn from_iter<I: IntoIterator<Item = Version>>(iter: I) -> Self {
        let mut versions: Vec<Version> = iter.into_iter().collect();
        versions.sort_unstable();
        versions.dedup();
        Self { versions }
    }
}

/// The outcome of [`negotiate`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Resolution {
    /// The client can render the server's feature level in full; render at the
    /// contained version.
    Exact(Version),
    /// The client speaks an older compatible version. Render against
    /// `resolved`; nodes introduced after it degrade to placeholders.
    Downgraded {
        /// The version the server advertised.
        requested: Version,
        /// The compatible version the client will render against.
        resolved: Version,
    },
    /// No shared schema family exists; the document cannot be rendered.
    Unsupported {
        /// The version the server advertised.
        requested: Version,
    },
}

impl Resolution {
    /// The version to render against, if a renderable version was agreed.
    #[must_use]
    pub fn render_version(self) -> Option<Version> {
        match self {
            Resolution::Exact(v) => Some(v),
            Resolution::Downgraded { resolved, .. } => Some(resolved),
            Resolution::Unsupported { .. } => None,
        }
    }

    /// Whether a renderable version was agreed.
    #[must_use]
    pub fn is_renderable(self) -> bool {
        self.render_version().is_some()
    }
}

/// Negotiates a render version between a client's supported set and the
/// server-advertised `server` version.
///
/// The client renders against `min(server, highest_client_in_family)`:
///
/// * when the client can cover the server's feature level the result is
///   [`Resolution::Exact`], even if the client is strictly newer (a newer
///   client renders an older document at full fidelity),
/// * when the client is behind the server it renders against its highest
///   compatible version as [`Resolution::Downgraded`], and
/// * when the client does not speak the server's family at all the result is
///   [`Resolution::Unsupported`].
///
/// # Examples
///
/// ```
/// use prism_ui_sdui::{negotiate, Resolution, Version, VersionSet};
///
/// let mut client = VersionSet::new();
/// client.insert(Version::new(2, 0));
/// client.insert(Version::new(2, 1));
///
/// // Server speaks a newer minor the client has not caught up to.
/// let outcome = negotiate(&client, Version::new(2, 4));
/// assert_eq!(
///     outcome,
///     Resolution::Downgraded {
///         requested: Version::new(2, 4),
///         resolved: Version::new(2, 1),
///     }
/// );
/// ```
#[must_use]
pub fn negotiate(client: &VersionSet, server: Version) -> Resolution {
    match client.highest_in_family(server) {
        None => Resolution::Unsupported { requested: server },
        Some(client_max) => {
            let resolved = client_max.min(server);
            if resolved == server {
                Resolution::Exact(server)
            } else {
                Resolution::Downgraded {
                    requested: server,
                    resolved,
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{negotiate, Resolution, Version, VersionSet};

    fn client(versions: &[(u16, u16)]) -> VersionSet {
        versions
            .iter()
            .map(|&(ma, mi)| Version::new(ma, mi))
            .collect()
    }

    #[test]
    fn insert_keeps_set_sorted_and_deduplicated() {
        let mut set = VersionSet::new();
        set.insert(Version::new(2, 2));
        set.insert(Version::new(1, 0));
        set.insert(Version::new(2, 2));
        set.insert(Version::new(2, 0));
        assert_eq!(
            set.versions(),
            &[Version::new(1, 0), Version::new(2, 0), Version::new(2, 2),]
        );
    }

    #[test]
    fn supports_reports_exact_membership() {
        let set = client(&[(2, 0), (2, 1)]);
        assert!(set.supports(Version::new(2, 1)));
        assert!(!set.supports(Version::new(2, 2)));
    }

    #[test]
    fn negotiate_exact_when_version_listed() {
        let set = client(&[(1, 0), (2, 3)]);
        assert_eq!(
            negotiate(&set, Version::new(2, 3)),
            Resolution::Exact(Version::new(2, 3))
        );
    }

    #[test]
    fn negotiate_downgrades_when_client_is_behind() {
        let set = client(&[(2, 0), (2, 1)]);
        assert_eq!(
            negotiate(&set, Version::new(2, 5)),
            Resolution::Downgraded {
                requested: Version::new(2, 5),
                resolved: Version::new(2, 1),
            }
        );
    }

    #[test]
    fn negotiate_is_exact_when_client_is_newer() {
        // A newer client renders an older document at full fidelity.
        let set = client(&[(2, 9)]);
        assert_eq!(
            negotiate(&set, Version::new(2, 3)),
            Resolution::Exact(Version::new(2, 3))
        );
    }

    #[test]
    fn negotiate_unsupported_without_shared_family() {
        let set = client(&[(1, 0), (1, 4)]);
        assert_eq!(
            negotiate(&set, Version::new(3, 0)),
            Resolution::Unsupported {
                requested: Version::new(3, 0)
            }
        );
    }

    #[test]
    fn resolution_render_version_matches_variant() {
        assert_eq!(
            Resolution::Exact(Version::new(2, 0)).render_version(),
            Some(Version::new(2, 0))
        );
        assert_eq!(
            Resolution::Downgraded {
                requested: Version::new(2, 2),
                resolved: Version::new(2, 0),
            }
            .render_version(),
            Some(Version::new(2, 0))
        );
        assert!(!Resolution::Unsupported {
            requested: Version::new(9, 9)
        }
        .is_renderable());
    }
}
