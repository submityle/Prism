//! Bounded collection-member filtering using the catalog's public query policy.

use prism_editor_content::{AssetRecord, ContentCatalog, TreeQuery};

// Collections span all folders. Their bounded member set is filtered without
// scanning the full catalog; folder root is intentionally not applied.
pub(crate) fn matches(catalog: &ContentCatalog, query: &TreeQuery, record: &AssetRecord) -> bool {
    if query.kind.as_ref().is_some_and(|kind| kind != &record.kind)
        || (query.favorites_only && !record.favorite)
        || (query.pinned_only && !record.pinned)
        || query
            .vcs
            .is_some_and(|state| catalog.vcs_status(record.id).unwrap_or_default().state != state)
    {
        return false;
    }
    if let Some(tag) = &query.tag {
        let tag = tag.trim().to_ascii_lowercase();
        if !tag.is_empty() && !record.tags.contains(&tag) {
            return false;
        }
    }
    if let Some(text) = &query.text {
        let text = text.trim().to_ascii_lowercase();
        if !text.is_empty()
            && !record.path.to_ascii_lowercase().contains(&text)
            && !record.kind.as_str().to_ascii_lowercase().contains(&text)
            && !record
                .tags
                .iter()
                .any(|tag| tag.to_ascii_lowercase().contains(&text))
        {
            return false;
        }
    }
    true
}
