//! Nonblocking, page-owned thumbnail requests and genuine decoded pixels.

use crate::{ContentBrowserController, MAX_VISIBLE_ROWS};
use prism_editor_content::{
    thumbnail::{Thumbnail, ThumbnailError, ThumbnailKey, ThumbnailRequest, ThumbnailService},
    AssetId,
};
use std::{
    collections::{BTreeMap, BTreeSet},
    sync::Arc,
    task::Poll,
};

/// Hard cap for decoded pixels retained by one UI page (excludes host clones).
pub const MAX_CONTENT_THUMBNAIL_BYTES: usize = 32 * 1024 * 1024;

/// Genuine service state; a ready result always contains decoded pixel data.
#[derive(Clone, Debug)]
pub enum ContentThumbnailState {
    /// Waiting for shared service capacity; retried on the next explicit poll.
    Queued,
    /// A service request exists and has not published its result.
    Loading,
    /// Decoded pixels retained within the page budget.
    Ready(Arc<Thumbnail>),
    /// Service failure or the page's decoded-byte limit.
    Failed(ThumbnailError),
}

struct Entry {
    key: ThumbnailKey,
    request: Option<ThumbnailRequest>,
    state: ContentThumbnailState,
}

/// Owns at most one request/result per visible asset. Request creation, polling
/// and cancellation perform no caller file I/O. The host supplies authoritative
/// source revisions, schedules poll/redraw, and owns GPU textures.
pub struct ContentThumbnails {
    service: ThumbnailService,
    entries: BTreeMap<AssetId, Entry>,
    generation: Option<u64>,
    bytes: usize,
    byte_budget: usize,
}

impl ContentThumbnails {
    /// Shares the host's bounded service; does not create a second cache.
    #[must_use]
    pub fn new(service: ThumbnailService) -> Self {
        Self {
            service,
            entries: BTreeMap::new(),
            generation: None,
            bytes: 0,
            byte_budget: MAX_CONTENT_THUMBNAIL_BYTES,
        }
    }

    /// Shares a service with a smaller explicit page-owned pixel budget.
    /// Zero or a value above the hard UI ceiling is rejected.
    pub fn with_budget(service: ThumbnailService, bytes: usize) -> Result<Self, ThumbnailError> {
        if bytes == 0 || bytes > MAX_CONTENT_THUMBNAIL_BYTES {
            return Err(ThumbnailError::InvalidConfiguration);
        }
        Ok(Self {
            byte_budget: bytes,
            ..Self::new(service)
        })
    }

    /// Atomically replaces the wanted visible keys, cancelling removed or
    /// superseded requests. Rejects duplicate/nonvisible IDs, excessive keys,
    /// invalid paths or output rectangles before changing retained state.
    /// Call after browser build and on every catalog publication. Omitted assets
    /// have no thumbnail state; no source path or revision is invented by UI.
    pub fn sync_visible(
        &mut self,
        browser: &ContentBrowserController,
        keys: impl IntoIterator<Item = ThumbnailKey>,
    ) -> Result<(), ThumbnailError> {
        let mut wanted = BTreeMap::new();
        let visible: BTreeSet<_> = browser.visible_assets().iter().copied().collect();
        for key in keys {
            if wanted.len() >= MAX_VISIBLE_ROWS {
                return Err(ThumbnailError::Limit("visible thumbnail keys"));
            }
            if !visible.contains(&key.asset)
                || !key.source.is_absolute()
                || key.source.as_os_str().as_encoded_bytes().len() > crate::MAX_ACTION_PATH_BYTES
                || key.width == 0
                || key.height == 0
                || wanted.contains_key(&key.asset)
            {
                return Err(ThumbnailError::InvalidConfiguration);
            }
            wanted.insert(key.asset, key);
        }
        // A new publication invalidates every view/result until the host submits
        // current keys, even when its catalog changes only metadata.
        self.generation = browser.generation();
        self.entries
            .retain(|id, entry| wanted.get(id) == Some(&entry.key));
        for (id, key) in wanted {
            self.entries.entry(id).or_insert(Entry {
                key,
                request: None,
                state: ContentThumbnailState::Queued,
            });
        }
        self.recount();
        self.poll();
        Ok(())
    }

    /// Nonblocking event-loop tick. Returns whether exposed state changed.
    /// Busy requests stay queued; terminal errors need explicit retry or new key.
    pub fn poll(&mut self) -> bool {
        let mut changed = false;
        // Consume first so ready workers release shared slots before queueing.
        for entry in self.entries.values_mut() {
            if let Some(request) = &mut entry.request
                && let Poll::Ready(result) = request.poll()
            {
                entry.request = None;
                entry.state = match result {
                    Ok(image)
                        if image.rgba8().len() <= self.byte_budget.saturating_sub(self.bytes) =>
                    {
                        self.bytes += image.rgba8().len();
                        ContentThumbnailState::Ready(image)
                    }
                    Ok(_) => ContentThumbnailState::Failed(ThumbnailError::Limit(
                        "visible thumbnail pixels",
                    )),
                    Err(error) => ContentThumbnailState::Failed(error),
                };
                changed = true;
            }
        }
        for entry in self.entries.values_mut() {
            if matches!(entry.state, ContentThumbnailState::Queued) {
                match self.service.request(entry.key.clone()) {
                    Ok(request) => {
                        entry.request = Some(request);
                        entry.state = ContentThumbnailState::Loading;
                        changed = true;
                    }
                    Err(ThumbnailError::Busy) => break,
                    Err(error) => {
                        entry.state = ContentThumbnailState::Failed(error);
                        changed = true;
                    }
                }
            }
        }
        changed
    }

    /// Requeues one failed visible request; successful results remain retained.
    pub fn retry(&mut self, asset: AssetId) -> bool {
        if let Some(entry) = self.entries.get_mut(&asset)
            && matches!(entry.state, ContentThumbnailState::Failed(_))
        {
            entry.state = ContentThumbnailState::Queued;
            return true;
        }
        false
    }

    /// Cancels requests and drops page-owned pixels. Does not clear the shared
    /// service cache; host calls service.clear() when closing/changing project.
    pub fn clear(&mut self) {
        self.entries.clear();
        self.bytes = 0;
        self.generation = None;
    }

    /// Service state for an explicitly supplied visible key.
    #[must_use]
    pub fn state(&self, asset: AssetId) -> Option<&ContentThumbnailState> {
        self.entries.get(&asset).map(|entry| &entry.state)
    }

    /// Returns identity and actual pixels only after successful decode. Host
    /// must still synchronize current visible keys before GPU submission.
    #[must_use]
    pub fn ready(&self, asset: AssetId) -> Option<(&ThumbnailKey, &Arc<Thumbnail>)> {
        let entry = self.entries.get(&asset)?;
        if let ContentThumbnailState::Ready(image) = &entry.state {
            Some((&entry.key, image))
        } else {
            None
        }
    }

    /// Page-owned decoded bytes, excluding shared cache and host Arc clones.
    #[must_use]
    pub const fn retained_bytes(&self) -> usize {
        self.bytes
    }

    /// Whether queued/running work requires another host tick.
    #[must_use]
    pub fn needs_poll(&self) -> bool {
        self.entries.values().any(|entry| {
            matches!(
                entry.state,
                ContentThumbnailState::Queued | ContentThumbnailState::Loading
            )
        })
    }

    pub(crate) fn current_for(&self, generation: u64) -> bool {
        self.generation == Some(generation)
    }

    fn recount(&mut self) {
        self.bytes = self
            .entries
            .values()
            .filter_map(|entry| match &entry.state {
                ContentThumbnailState::Ready(image) => Some(image.rgba8().len()),
                _ => None,
            })
            .sum();
    }
}
