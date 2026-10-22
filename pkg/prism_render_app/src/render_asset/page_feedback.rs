//! Opt-in native buffer-page feedback. This does not populate a material atlas.

use std::sync::{Arc, Mutex};

use prism_app::App;
use prism_asset::{FeedbackDemand, PageKey};
use prism_ecs::{prelude::Resource, world::World};
use prism_render_architecture::texture_streaming::TexturePageKey;
use prism_render_driver::BufferUsages;
use prism_render_driver_wgpu::WgpuInstance;
use prism_tasks::Priority;

use super::page_streaming::StreamGeneration;
use super::{FixedPageSlotKey, FixedPageStreamingConfig, FixedPageStreamingSource};

#[path = "page_feedback_gpu.rs"]
mod gpu;
pub use gpu::FixedPageFeedbackGpu;

/// Hard admission bounds for one stream. Storage is allocated once per device.
#[derive(Clone, Copy, Debug)]
pub struct FixedPageFeedbackConfig {
    /// Maximum virtual-to-asset mappings and published table entries.
    pub max_bindings: usize,
    /// Maximum lookups and feedback records in one frame.
    pub max_queries: usize,
    /// Aggregate device bytes, including the physical pool and readback.
    pub max_device_bytes: u64,
}

impl Default for FixedPageFeedbackConfig {
    fn default() -> Self {
        Self {
            max_bindings: 4096,
            max_queries: 1024,
            max_device_bytes: 64 * 1024 * 1024,
        }
    }
}

/// Admission, identity, or device failure. No failed capture produces demands.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum FixedPageFeedbackError {
    /// An invalid pool, workload, or bounded configuration.
    InvalidConfig(&'static str),
    /// A fixed registry, query, or ABI capacity would be exceeded.
    Capacity,
    /// A bounded host reservation failed.
    HostAllocation,
    /// The registry epoch cannot be advanced without aliasing old captures.
    EpochExhausted,
    /// The registry or source changed while the capture was in flight.
    StaleCapture,
    /// The asynchronous staging mapping failed.
    Readback,
    /// A scoped device operation failed.
    Gpu(String),
}
impl core::fmt::Display for FixedPageFeedbackError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "fixed-page feedback: {self:?}")
    }
}
impl std::error::Error for FixedPageFeedbackError {}

/// A virtual lookup and one word to consume from its physical page on a hit.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct FixedPageQuery {
    /// Exact virtual coordinate to look up in the persistent table.
    pub key: TexturePageKey,
    /// Zero-based word within a physical page.
    pub word_in_page: u32,
}

/// Small GPU result; `slot == u32::MAX` is GPU-generated missing-page feedback.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct FixedPageFeedbackSample {
    /// The lookup that produced this record.
    pub query: FixedPageQuery,
    /// Physical page index, or `u32::MAX` for a miss.
    pub slot: u32,
    /// Word read by the downstream GPU consumer, or zero for a miss.
    pub value: u32,
}

#[derive(Clone)]
struct Binding {
    key: TexturePageKey,
    page: PageKey,
    slot: FixedPageSlotKey,
    source_generation: Option<StreamGeneration>,
}

struct Completed {
    source_generation: StreamGeneration,
    target_generation: StreamGeneration,
    epoch: u64,
    revision: u64,
    frame: u64,
    samples: Vec<FixedPageFeedbackSample>,
}

#[derive(Default)]
struct HandoffState {
    generation: StreamGeneration,
    initialized: bool,
    completed: Option<Result<Completed, FixedPageFeedbackError>>,
}
type Handoff = Arc<Mutex<HandoffState>>;

/// Main-world identity registry, bounded lookup input, and feedback diagnostics.
/// Registry mutations invalidate captures already in flight.
#[derive(Resource)]
pub struct FixedPageFeedback {
    config: FixedPageFeedbackConfig,
    page_words: u32,
    bindings: Vec<Binding>,
    queries: Vec<FixedPageQuery>,
    epoch: u64,
    frame: u64,
    source_generation: Option<StreamGeneration>,
    handoff: Handoff,
    /// Last successfully decoded capture, including GPU-read pool words.
    pub last_samples: Vec<FixedPageFeedbackSample>,
    /// Captures admitted with matching registry and source identities.
    pub captures: u64,
    /// Captures rejected without creating asset demands.
    pub rejected_captures: u64,
    /// GPU misses without a virtual-to-asset identity mapping.
    pub unknown_misses: u64,
    /// Last capture rejection, cleared by the next successful capture.
    pub last_error: Option<FixedPageFeedbackError>,
}

fn copy_vec<T: Clone>(items: &[T]) -> Result<Vec<T>, FixedPageFeedbackError> {
    let mut copy = Vec::new();
    copy.try_reserve_exact(items.len())
        .map_err(|_| FixedPageFeedbackError::HostAllocation)?;
    copy.extend_from_slice(items);
    Ok(copy)
}

impl FixedPageFeedback {
    fn new(
        config: FixedPageFeedbackConfig,
        page_words: u32,
    ) -> Result<Self, FixedPageFeedbackError> {
        if config.max_bindings == 0
            || config.max_queries == 0
            || config.max_bindings > (u32::MAX / 4) as usize
            || config.max_queries > (u32::MAX / 4) as usize
            || page_words == 0
        {
            return Err(FixedPageFeedbackError::InvalidConfig(
                "nonzero u32 capacities required",
            ));
        }
        Ok(Self {
            config,
            page_words,
            bindings: Vec::new(),
            queries: Vec::new(),
            epoch: 1,
            frame: 0,
            source_generation: None,
            handoff: Arc::new(Mutex::new(HandoffState::default())),
            last_samples: Vec::new(),
            captures: 0,
            rejected_captures: 0,
            unknown_misses: 0,
            last_error: None,
        })
    }

    /// Binds one virtual coordinate to a caller-supplied asset layout mapping.
    /// An explicit bind resolves against the source at the next pump/extract;
    /// retained bindings do not transfer to independently constructed sources.
    pub fn bind(
        &mut self,
        key: TexturePageKey,
        page: PageKey,
    ) -> Result<(), FixedPageFeedbackError> {
        let index = self
            .bindings
            .binary_search_by_key(&key, |binding| binding.key);
        if let Ok(index) = index {
            if self.bindings[index].page == page && self.bindings[index].source_generation.is_none()
            {
                return Ok(());
            }
        } else {
            if self.bindings.len() == self.config.max_bindings {
                return Err(FixedPageFeedbackError::Capacity);
            }
            self.bindings
                .try_reserve(1)
                .map_err(|_| FixedPageFeedbackError::HostAllocation)?;
        }
        let epoch = self
            .epoch
            .checked_add(1)
            .ok_or(FixedPageFeedbackError::EpochExhausted)?;
        let binding = Binding {
            key,
            // Resolve an explicit bind against the source present at the next
            // pump/extract. Equal local revision numbers do not transfer ownership.
            source_generation: None,
            slot: FixedPageSlotKey {
                path: page.path().clone(),
                page_index: page.page_index(),
            },
            page,
        };
        match index {
            Ok(index) => self.bindings[index] = binding,
            Err(index) => self.bindings.insert(index, binding),
        }
        self.epoch = epoch;
        Ok(())
    }

    /// Removes an identity. The next table publication withholds its mapping.
    pub fn unbind(&mut self, key: TexturePageKey) -> Result<(), FixedPageFeedbackError> {
        if let Ok(index) = self
            .bindings
            .binary_search_by_key(&key, |binding| binding.key)
        {
            let epoch = self
                .epoch
                .checked_add(1)
                .ok_or(FixedPageFeedbackError::EpochExhausted)?;
            self.bindings.remove(index);
            self.epoch = epoch;
        }
        Ok(())
    }

    /// Replaces the bounded consumer workload. Unknown coordinates may miss
    /// on GPU but cannot cause an asset read.
    pub fn set_queries(
        &mut self,
        queries: &[FixedPageQuery],
    ) -> Result<(), FixedPageFeedbackError> {
        if queries.len() > self.config.max_queries {
            return Err(FixedPageFeedbackError::Capacity);
        }
        if queries
            .iter()
            .any(|query| query.word_in_page >= self.page_words)
        {
            return Err(FixedPageFeedbackError::InvalidConfig(
                "query outside fixed page",
            ));
        }
        self.queries = copy_vec(queries)?;
        Ok(())
    }

    fn synchronize_source(
        &mut self,
        generation: StreamGeneration,
    ) -> Result<(), FixedPageFeedbackError> {
        if self.source_generation.as_ref() != Some(&generation) {
            self.epoch = self
                .epoch
                .checked_add(1)
                .ok_or(FixedPageFeedbackError::EpochExhausted)?;
            self.source_generation = Some(generation.clone());
            self.last_samples.clear();
        }
        for binding in &mut self.bindings {
            if binding.source_generation.is_none() {
                binding.source_generation = Some(generation.clone());
            }
        }
        Ok(())
    }

    pub(super) fn admit_completed(
        &mut self,
        source: &mut FixedPageStreamingSource,
        frame: u64,
        limit: usize,
    ) {
        self.frame = frame;
        if let Err(error) = self.synchronize_source(source.generation()) {
            self.reject(error);
            return;
        }
        let (completed, target_generation) = {
            let mut handoff = self.handoff.lock().expect("feedback handoff");
            (handoff.completed.take(), handoff.generation.clone())
        };
        let Some(completed) = completed else {
            return;
        };
        let completed = match completed {
            Ok(capture)
                if capture.epoch == self.epoch
                    && capture.revision == source.sources().revision()
                    && capture.source_generation == source.generation()
                    && capture.target_generation == target_generation =>
            {
                capture
            }
            Ok(_) => {
                self.reject(FixedPageFeedbackError::StaleCapture);
                return;
            }
            Err(error) => {
                self.reject(error);
                return;
            }
        };
        let mut admitted: Vec<usize> = Vec::new();
        if admitted
            .try_reserve_exact(limit.min(completed.samples.len()))
            .is_err()
        {
            self.reject(FixedPageFeedbackError::HostAllocation);
            return;
        }
        for sample in &completed.samples {
            if sample.slot != u32::MAX {
                continue;
            }
            let Ok(index) = self
                .bindings
                .binary_search_by_key(&sample.query.key, |binding| binding.key)
            else {
                self.unknown_misses = self.unknown_misses.saturating_add(1);
                continue;
            };
            let binding = &self.bindings[index];
            if binding.page.source_revision() != completed.revision
                || binding.source_generation.as_ref() != Some(&completed.source_generation)
            {
                continue;
            }
            if !admitted
                .iter()
                .any(|&accepted| self.bindings[accepted].page == binding.page)
            {
                if admitted.len() == limit {
                    continue;
                }
                admitted.push(index);
            }
            source.submit_feedback(FeedbackDemand::new(
                binding.page.clone(),
                binding.key.mip,
                1,
                completed.frame,
                Priority::High,
            ));
        }
        self.last_samples = completed.samples;
        self.captures = self.captures.saturating_add(1);
        self.last_error = None;
    }

    fn reject(&mut self, error: FixedPageFeedbackError) {
        self.rejected_captures = self.rejected_captures.saturating_add(1);
        self.last_error = Some(error);
    }
}

#[derive(Resource)]
struct Input {
    source_generation: StreamGeneration,
    target_generation: StreamGeneration,
    publication_error: Option<FixedPageFeedbackError>,
    config: FixedPageFeedbackConfig,
    bindings: Vec<Binding>,
    queries: Vec<FixedPageQuery>,
    epoch: u64,
    revision: u64,
    frame: u64,
    handoff: Handoff,
}

pub(super) fn extract_feedback(main: &mut World, render: &mut World) {
    let Some(source) = main.get_resource::<FixedPageStreamingSource>() else {
        invalidate_input(render, FixedPageFeedbackError::StaleCapture);
        return;
    };
    let revision = source.sources().revision();
    let generation = source.generation();
    let Some(feedback) = main.get_resource_mut::<FixedPageFeedback>() else {
        invalidate_input(render, FixedPageFeedbackError::StaleCapture);
        return;
    };
    let input: Result<Input, FixedPageFeedbackError> = (|| {
        feedback.synchronize_source(generation.clone())?;
        Ok(Input {
            source_generation: generation,
            target_generation: feedback
                .handoff
                .lock()
                .expect("feedback handoff")
                .generation
                .clone(),
            publication_error: None,
            config: feedback.config,
            bindings: copy_vec(&feedback.bindings)?,
            queries: copy_vec(&feedback.queries)?,
            epoch: feedback.epoch,
            revision,
            frame: feedback.frame,
            handoff: feedback.handoff.clone(),
        })
    })();
    apply_extracted_feedback(feedback, render, input);
}

fn apply_extracted_feedback(
    feedback: &mut FixedPageFeedback,
    render: &mut World,
    input: Result<Input, FixedPageFeedbackError>,
) {
    match input {
        Ok(input) => {
            render.insert_resource(input);
        }
        Err(error) => {
            feedback.reject(error.clone());
            // Never dispatch the previous registry/workload after failed extraction.
            invalidate_input(render, error);
        }
    }
}

fn invalidate_input(render: &mut World, error: FixedPageFeedbackError) {
    if let Some(input) = render.get_resource_mut::<Input>() {
        input.bindings.clear();
        input.queries.clear();
        input.publication_error = Some(error);
    }
}

/// Installs the native buffer-page loop on an existing wgpu RenderApp.
/// Data and table publication share its queue; this does not build a texture atlas.
/// The source's feedback queue and streaming admission budgets govern asset demand.
/// Consumers use the current table/count after `PrepareResourcesFlush` and submit
/// on the runtime queue before its frame completion signal.
pub fn register_fixed_page_feedback(
    app: &mut App,
    source: FixedPageStreamingSource,
    streaming: FixedPageStreamingConfig,
    feedback: FixedPageFeedbackConfig,
) -> Result<(), FixedPageFeedbackError> {
    if streaming.gpu_pool.page_size != source.residency().page_size().bytes()
        || streaming.gpu_pool.page_size == 0
        || streaming.gpu_pool.page_size % 4 != 0
        || streaming.gpu_pool.page_count == 0
        || !streaming
            .gpu_pool
            .usage
            .contains(BufferUsages::STORAGE | BufferUsages::COPY_DST)
    {
        return Err(FixedPageFeedbackError::InvalidConfig(
            "matching aligned storage pool required",
        ));
    }
    let page_words = u32::try_from(streaming.gpu_pool.page_size / 4)
        .map_err(|_| FixedPageFeedbackError::Capacity)?;
    let feedback = FixedPageFeedback::new(feedback, page_words)?;
    let Some(render) = app.get_sub_app(crate::RenderApp) else {
        return Err(FixedPageFeedbackError::InvalidConfig(
            "wgpu RenderApp required",
        ));
    };
    let Some(runtime) = render
        .world
        .get_resource::<crate::RenderRuntimeResource<WgpuInstance>>()
    else {
        return Err(FixedPageFeedbackError::InvalidConfig(
            "wgpu runtime required",
        ));
    };
    if app
        .world()
        .get_resource::<FixedPageStreamingConfig>()
        .is_some()
    {
        return Err(FixedPageFeedbackError::InvalidConfig(
            "fixed-page stream already registered",
        ));
    }
    gpu::validate_device(
        runtime.runtime().device().wgpu_device(),
        feedback.config,
        &streaming.gpu_pool,
    )?;
    app.insert_resource(feedback);
    super::register_fixed_page_streaming::<WgpuInstance>(app, source, streaming);
    gpu::install(app);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use prism_asset::{AssetSources, FeedbackDemandQueue, PageBudget, PageResidency, PageSize};

    fn key(x: u16) -> TexturePageKey {
        TexturePageKey {
            texture: 1,
            mip: 0,
            layer: 0,
            x,
            y: 0,
        }
    }

    fn coordinator() -> FixedPageFeedback {
        FixedPageFeedback::new(
            FixedPageFeedbackConfig {
                max_bindings: 1,
                max_queries: 2,
                max_device_bytes: 1024,
            },
            2,
        )
        .unwrap()
    }

    fn source() -> FixedPageStreamingSource {
        FixedPageStreamingSource::with_feedback_queue(
            AssetSources::new(),
            PageResidency::new(PageSize::new(8).unwrap(), PageBudget::new(16, 1).unwrap()).unwrap(),
            FeedbackDemandQueue::new(4, 60),
        )
    }

    #[test]
    fn late_invalid_workload_and_registry_overflow_preserve_current_state() {
        let mut feedback = coordinator();
        let page = PageKey::new("pages.bin", 0, 0);
        feedback.bind(key(0), page.clone()).unwrap();
        let epoch = feedback.epoch;
        feedback.bind(key(0), page.clone()).unwrap();
        assert_eq!(feedback.epoch, epoch);
        assert_eq!(
            feedback.bind(key(1), page),
            Err(FixedPageFeedbackError::Capacity)
        );
        assert_eq!(feedback.epoch, epoch);
        let valid = FixedPageQuery {
            key: key(0),
            word_in_page: 0,
        };
        feedback.set_queries(&[valid]).unwrap();
        assert!(feedback
            .set_queries(&[
                valid,
                FixedPageQuery {
                    word_in_page: 2,
                    ..valid
                }
            ])
            .is_err());
        assert_eq!(feedback.queries, [valid]);
        assert_eq!(
            feedback.set_queries(&[valid; 3]),
            Err(FixedPageFeedbackError::Capacity)
        );
        assert_eq!(feedback.queries, [valid]);
        feedback.epoch = u64::MAX;
        assert_eq!(
            feedback.unbind(key(0)),
            Err(FixedPageFeedbackError::EpochExhausted)
        );
        assert_eq!(feedback.bindings.len(), 1);
    }

    #[test]
    fn stale_epoch_or_source_rejects_whole_capture_and_next_capture_recovers() {
        let mut feedback = coordinator();
        let mut source = source();
        feedback
            .bind(key(0), PageKey::new("pages.bin", 0, 0))
            .unwrap();
        feedback.synchronize_source(source.generation()).unwrap();
        let source_generation = source.generation();
        let target_generation = feedback.handoff.lock().unwrap().generation.clone();
        let capture = |epoch, revision| Completed {
            source_generation: source_generation.clone(),
            target_generation: target_generation.clone(),
            epoch,
            revision,
            frame: 0,
            samples: vec![FixedPageFeedbackSample {
                query: FixedPageQuery {
                    key: key(0),
                    word_in_page: 0,
                },
                slot: u32::MAX,
                value: 0,
            }],
        };
        for (epoch, revision) in [(feedback.epoch - 1, 0), (feedback.epoch, 1)] {
            feedback.handoff.lock().unwrap().completed = Some(Ok(capture(epoch, revision)));
            feedback.admit_completed(&mut source, 1, 1);
            assert_eq!(source.feedback_len(), 0);
            assert!(feedback.last_samples.is_empty());
            assert_eq!(
                feedback.last_error,
                Some(FixedPageFeedbackError::StaleCapture)
            );
        }
        feedback.handoff.lock().unwrap().completed = Some(Ok(capture(feedback.epoch, 0)));
        feedback.admit_completed(&mut source, 2, 1);
        assert_eq!(source.feedback_len(), 1);
        assert_eq!(feedback.rejected_captures, 2);
        assert_eq!(feedback.captures, 1);
        assert_eq!(feedback.last_error, None);
    }

    fn misses(
        feedback: &FixedPageFeedback,
        source: &FixedPageStreamingSource,
        keys: &[TexturePageKey],
    ) -> Completed {
        Completed {
            source_generation: source.generation(),
            target_generation: feedback.handoff.lock().unwrap().generation.clone(),
            epoch: feedback.epoch,
            revision: source.sources().revision(),
            frame: feedback.frame,
            samples: keys
                .iter()
                .map(|&key| FixedPageFeedbackSample {
                    query: FixedPageQuery {
                        key,
                        word_in_page: 0,
                    },
                    slot: u32::MAX,
                    value: 0,
                })
                .collect(),
        }
    }

    fn deliver(feedback: &mut FixedPageFeedback, completed: Completed) {
        feedback.handoff.lock().unwrap().completed = Some(Ok(completed));
    }

    #[test]
    fn independent_same_revision_source_requires_rebind_and_rejects_late_capture() {
        let mut feedback = coordinator();
        let old = source();
        let page = PageKey::new("pages.bin", 0, 0);
        feedback.bind(key(0), page.clone()).unwrap();
        feedback.synchronize_source(old.generation()).unwrap();
        let late = misses(&feedback, &old, &[key(0)]);
        let mut new = source();
        assert_eq!(old.sources().revision(), new.sources().revision());
        assert_ne!(old.generation(), new.generation());
        deliver(&mut feedback, late);
        feedback.admit_completed(&mut new, 1, 1);
        assert_eq!(
            feedback.last_error,
            Some(FixedPageFeedbackError::StaleCapture)
        );
        assert_eq!(new.feedback_len(), 0);
        // A fresh capture cannot reuse the old source's registry binding.
        let fresh = misses(&feedback, &new, &[key(0)]);
        deliver(&mut feedback, fresh);
        feedback.admit_completed(&mut new, 2, 1);
        assert_eq!(new.feedback_len(), 0);
        feedback.bind(key(0), page).unwrap();
        feedback.synchronize_source(new.generation()).unwrap();
        let fresh = misses(&feedback, &new, &[key(0)]);
        deliver(&mut feedback, fresh);
        feedback.admit_completed(&mut new, 3, 1);
        assert_eq!(new.feedback_len(), 1);
    }

    #[test]
    fn rebuild_generation_rejects_completed_capture_even_with_same_source_and_epoch() {
        let mut feedback = coordinator();
        let mut source = source();
        feedback
            .bind(key(0), PageKey::new("pages.bin", 0, 0))
            .unwrap();
        feedback.synchronize_source(source.generation()).unwrap();
        let old = misses(&feedback, &source, &[key(0)]);
        feedback.handoff.lock().unwrap().generation = StreamGeneration::default();
        deliver(&mut feedback, old);
        feedback.admit_completed(&mut source, 1, 1);
        assert_eq!(source.feedback_len(), 0);
        assert_eq!(
            feedback.last_error,
            Some(FixedPageFeedbackError::StaleCapture)
        );
        let fresh = misses(&feedback, &source, &[key(0)]);
        deliver(&mut feedback, fresh);
        feedback.admit_completed(&mut source, 2, 1);
        assert_eq!(source.feedback_len(), 1);
    }

    #[test]
    fn zero_and_partial_admission_leave_no_native_backlog_across_epoch_change() {
        let mut feedback = FixedPageFeedback::new(
            FixedPageFeedbackConfig {
                max_bindings: 2,
                max_queries: 3,
                max_device_bytes: 1024,
            },
            2,
        )
        .unwrap();
        let mut source = source();
        feedback
            .bind(key(0), PageKey::new("pages.bin", 0, 0))
            .unwrap();
        feedback
            .bind(key(1), PageKey::new("pages.bin", 1, 0))
            .unwrap();
        feedback.synchronize_source(source.generation()).unwrap();
        let capture = misses(&feedback, &source, &[key(0), key(1)]);
        deliver(&mut feedback, capture);
        feedback.admit_completed(&mut source, 0, 0);
        assert_eq!(source.feedback_len(), 0);
        let capture = misses(&feedback, &source, &[key(0), key(0), key(1)]);
        deliver(&mut feedback, capture);
        feedback.admit_completed(&mut source, 1, 1);
        assert_eq!(source.admit_feedback(1, 1).len(), 1);
        assert_eq!(source.feedback_len(), 0);
        feedback.unbind(key(1)).unwrap();
        feedback
            .bind(key(1), PageKey::new("other.bin", 0, 0))
            .unwrap();
        feedback.admit_completed(&mut source, 2, 1);
        assert_eq!(source.feedback_len(), 0);
        assert!(source.residency().eviction_candidates().is_empty());
        assert!(source.feedback_stats().merged >= 1);
    }

    #[test]
    fn failed_extract_invalidates_bindings_and_next_extract_recovers() {
        let mut main = World::new();
        let mut render = World::new();
        let mut feedback = coordinator();
        feedback
            .bind(key(0), PageKey::new("pages.bin", 0, 0))
            .unwrap();
        feedback
            .set_queries(&[FixedPageQuery {
                key: key(0),
                word_in_page: 0,
            }])
            .unwrap();
        main.insert_resource(source());
        main.insert_resource(feedback);
        extract_feedback(&mut main, &mut render);
        assert_eq!(render.resource::<Input>().bindings.len(), 1);
        // Controlled reservation failure after a previous successful publication.
        apply_extracted_feedback(
            main.resource_mut::<FixedPageFeedback>(),
            &mut render,
            Err(FixedPageFeedbackError::HostAllocation),
        );
        let input = render.resource::<Input>();
        assert!(input.bindings.is_empty());
        assert!(input.queries.is_empty());
        assert_eq!(
            input.publication_error,
            Some(FixedPageFeedbackError::HostAllocation)
        );
        extract_feedback(&mut main, &mut render);
        let input = render.resource::<Input>();
        assert!(input.publication_error.is_none());
        assert_eq!(input.bindings.len(), 1);
        assert_eq!(input.queries.len(), 1);
        main.remove_resource::<FixedPageStreamingSource>();
        extract_feedback(&mut main, &mut render);
        assert!(render.resource::<Input>().bindings.is_empty());
        assert!(render.resource::<Input>().publication_error.is_some());
    }

    #[test]
    fn aggregate_budget_includes_pool_uniform_and_readback() {
        let pool = super::super::FixedPagePoolDescriptor::new(
            8,
            1,
            BufferUsages::COPY_DST | BufferUsages::STORAGE,
        );
        let mut config = FixedPageFeedbackConfig {
            max_bindings: 1,
            max_queries: 1,
            max_device_bytes: 72,
        };
        assert_eq!(
            gpu::validate_sizes(config, &pool).unwrap(),
            [16, 16, 8, 8, 16]
        );
        config.max_device_bytes = 71;
        assert!(gpu::validate_sizes(config, &pool).is_err());
    }
}
