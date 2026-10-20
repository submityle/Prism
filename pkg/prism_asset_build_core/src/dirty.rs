//! Deterministic incremental invalidation for local build graphs.
//!
//! A [`DirtySet`] is the exact transitive successor closure of a set of
//! changed jobs. It contains no timestamps, filesystem paths, thread order, or
//! cache policy. Callers can therefore use it for plan-only diagnostics and as
//! the admission set for a later incremental executor without changing build
//! identity semantics.

use crate::graph::{BuildError, BuildGraph};
use crate::job::JobId;
use crate::profile::{ProfileHash, ProfileProjection};
use crate::source::{SourceIdentity, SourceSnapshot};
use crate::types::ContentHash;
use std::collections::{BTreeMap, BTreeSet, VecDeque};
use std::fmt;
use std::fs;
use std::io;
use std::path::Path;

/// Deterministic set of jobs invalidated by direct changes and their graph
/// successors.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct DirtySet {
    direct: BTreeSet<JobId>,
    affected: BTreeSet<JobId>,
}

impl DirtySet {
    /// Computes a dirty set from a validated graph and direct changes.
    ///
    /// Every direct change must name a job in `graph`. Missing dependencies and
    /// cycles are rejected before propagation, so a successful result is safe
    /// to consume as a deterministic incremental plan.
    pub fn from_graph(
        graph: &BuildGraph,
        direct_changes: impl IntoIterator<Item = JobId>,
    ) -> Result<Self, DirtySetError> {
        graph.execution_order().map_err(DirtySetError::Graph)?;

        let direct: BTreeSet<_> = direct_changes.into_iter().collect();
        for &job in &direct {
            if graph.job(job).is_none() {
                return Err(DirtySetError::UnknownJob(job));
            }
        }

        let mut dependents: BTreeMap<JobId, Vec<JobId>> = BTreeMap::new();
        for (&job, node) in graph.jobs() {
            for &dependency in node.dependencies() {
                dependents.entry(dependency).or_default().push(job);
            }
        }
        for children in dependents.values_mut() {
            children.sort_unstable();
            children.dedup();
        }

        let mut affected = direct.clone();
        let mut queue: VecDeque<_> = direct.iter().copied().collect();
        while let Some(job) = queue.pop_front() {
            for &dependent in dependents.get(&job).into_iter().flatten() {
                if affected.insert(dependent) {
                    queue.push_back(dependent);
                }
            }
        }

        Ok(Self { direct, affected })
    }

    /// Returns the directly changed jobs, in ascending stable ID order.
    #[must_use]
    pub fn direct_jobs(&self) -> impl Iterator<Item = JobId> + '_ {
        self.direct.iter().copied()
    }

    /// Returns all invalidated jobs, in ascending stable ID order.
    #[must_use]
    pub fn jobs(&self) -> impl Iterator<Item = JobId> + '_ {
        self.affected.iter().copied()
    }

    /// Returns whether `job` is directly changed or transitively affected.
    #[must_use]
    pub fn contains(&self, job: JobId) -> bool {
        self.affected.contains(&job)
    }

    /// Number of invalidated jobs.
    #[must_use]
    pub fn len(&self) -> usize {
        self.affected.len()
    }

    /// Returns whether no job was invalidated.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.affected.is_empty()
    }
}

/// Failure while constructing a deterministic dirty set.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum DirtySetError {
    /// The supplied change names no graph job.
    UnknownJob(JobId),
    /// The graph is not a valid DAG.
    Graph(BuildError),
}

impl fmt::Display for DirtySetError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::UnknownJob(job) => write!(formatter, "dirty change names unknown job {}", job.0),
            Self::Graph(error) => write!(formatter, "cannot compute dirty set: {error}"),
        }
    }
}

impl std::error::Error for DirtySetError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Graph(error) => Some(error),
            Self::UnknownJob(_) => None,
        }
    }
}

/// Semantic inputs used to compare two build plans.
///
/// Each map is keyed by the graph job that consumes the input. A source
/// change invalidates its source job, settings changes invalidate the job
/// whose processor consumes those settings, and profile changes invalidate
/// the job whose declared projection changed. The graph then expands those
/// direct changes to their deterministic successor closure.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct BuildInputSnapshot {
    sources: BTreeMap<JobId, SourceIdentity>,
    settings: BTreeMap<JobId, Vec<u8>>,
    profiles: BTreeMap<JobId, ProfileInput>,
}

#[derive(Clone, Debug, PartialEq)]
struct ProfileInput {
    hash: ProfileHash,
    fields: Option<BTreeMap<String, ProfileHash>>,
}

impl BuildInputSnapshot {
    /// Creates an empty input snapshot.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Adds or replaces the source identity consumed by `job`.
    pub fn with_source(mut self, job: JobId, source: &SourceSnapshot) -> Self {
        self.set_source(job, source);
        self
    }

    /// Adds or replaces canonical processor settings consumed by `job`.
    pub fn with_settings(mut self, job: JobId, settings: impl Into<Vec<u8>>) -> Self {
        self.set_settings(job, settings);
        self
    }

    /// Adds or replaces a profile projection consumed by `job`.
    pub fn with_profile(mut self, job: JobId, projection: ProfileProjection) -> Self {
        self.set_profile(job, projection);
        self
    }

    /// Adds or replaces a profile hash when the projection is not available.
    /// Such a change remains invalidatable, but its explanation has no field
    /// list because the caller did not provide the dependency projection.
    pub fn with_profile_hash(mut self, job: JobId, hash: ProfileHash) -> Self {
        self.set_profile_hash(job, hash);
        self
    }

    /// Adds or replaces a source identity consumed by `job`.
    pub fn set_source(&mut self, job: JobId, source: &SourceSnapshot) {
        self.sources.insert(job, source.identity());
    }

    /// Adds or replaces canonical processor settings consumed by `job`.
    pub fn set_settings(&mut self, job: JobId, settings: impl Into<Vec<u8>>) {
        self.settings.insert(job, settings.into());
    }

    /// Adds or replaces a profile projection consumed by `job`.
    pub fn set_profile(&mut self, job: JobId, projection: ProfileProjection) {
        self.profiles.insert(
            job,
            ProfileInput {
                hash: projection.hash(),
                fields: Some(
                    projection
                        .fields()
                        .map(|field| {
                            (
                                field.to_owned(),
                                projection
                                    .field_hash(field)
                                    .expect("selected profile field exists"),
                            )
                        })
                        .collect(),
                ),
            },
        );
    }

    /// Adds or replaces a profile hash consumed by `job`.
    pub fn set_profile_hash(&mut self, job: JobId, hash: ProfileHash) {
        self.profiles
            .insert(job, ProfileInput { hash, fields: None });
    }

    /// Compares two input snapshots and computes the deterministic dirty plan.
    pub fn invalidation_plan(
        &self,
        current: &Self,
        graph: &BuildGraph,
    ) -> Result<InvalidationPlan, DirtySetError> {
        // Validate the complete snapshots, not just changed inputs. Otherwise
        // an unknown job present in both snapshots would disappear from the
        // diff and be accepted as a valid empty plan.
        let unknown_job = self
            .sources
            .keys()
            .chain(current.sources.keys())
            .chain(self.settings.keys())
            .chain(current.settings.keys())
            .chain(self.profiles.keys())
            .chain(current.profiles.keys())
            .copied()
            .filter(|job| graph.job(*job).is_none())
            .min();
        if let Some(job) = unknown_job {
            return Err(DirtySetError::UnknownJob(job));
        }

        let mut direct = BTreeSet::new();
        let mut changes = Vec::new();

        let source_jobs: BTreeSet<_> = self
            .sources
            .keys()
            .chain(current.sources.keys())
            .copied()
            .collect();
        for job in source_jobs {
            let before = self.sources.get(&job);
            let after = current.sources.get(&job);
            if before != after {
                direct.insert(job);
                changes.push(InvalidationReason::Source {
                    job,
                    before: before.cloned(),
                    after: after.cloned(),
                });
            }
        }

        let settings_jobs: BTreeSet<_> = self
            .settings
            .keys()
            .chain(current.settings.keys())
            .copied()
            .collect();
        for job in settings_jobs {
            let before = self.settings.get(&job);
            let after = current.settings.get(&job);
            if before != after {
                direct.insert(job);
                changes.push(InvalidationReason::Settings {
                    job,
                    before: before.map(|bytes| ContentHash::digest(bytes)),
                    after: after.map(|bytes| ContentHash::digest(bytes)),
                });
            }
        }

        let profile_jobs: BTreeSet<_> = self
            .profiles
            .keys()
            .chain(current.profiles.keys())
            .copied()
            .collect();
        for job in profile_jobs {
            let before = self.profiles.get(&job);
            let after = current.profiles.get(&job);
            let before_hash = before.map(|profile| profile.hash);
            let after_hash = after.map(|profile| profile.hash);
            if before_hash == after_hash {
                continue;
            }
            direct.insert(job);
            changes.push(InvalidationReason::Profile {
                job,
                before: before_hash,
                after: after_hash,
                fields: profile_field_changes(before, after),
            });
        }

        changes.sort_by_key(|change| (change.job(), change.tag()));
        let dirty = DirtySet::from_graph(graph, direct)?;
        Ok(InvalidationPlan { dirty, changes })
    }

    /// Canonical semantic identities, settings, and profile field fingerprints.
    /// No source payload or diagnostic merge history is persisted.
    #[must_use]
    pub fn canonical_bytes(&self) -> Vec<u8> {
        let mut bytes = b"PRISM-BUILD-INPUTS\0\x01".to_vec();
        put_len(&mut bytes, self.sources.len());
        for (job, source) in &self.sources {
            bytes.extend_from_slice(&job.0.to_le_bytes());
            encode_identity(&mut bytes, Some(source));
        }
        put_len(&mut bytes, self.settings.len());
        for (job, settings) in &self.settings {
            bytes.extend_from_slice(&job.0.to_le_bytes());
            put_bytes(&mut bytes, settings);
        }
        put_len(&mut bytes, self.profiles.len());
        for (job, profile) in &self.profiles {
            bytes.extend_from_slice(&job.0.to_le_bytes());
            bytes.extend_from_slice(&profile.hash);
            bytes.push(u8::from(profile.fields.is_some()));
            if let Some(fields) = &profile.fields {
                put_len(&mut bytes, fields.len());
                for (field, hash) in fields {
                    put_bytes(&mut bytes, field.as_bytes());
                    bytes.extend_from_slice(hash);
                }
            }
        }
        bytes
    }

    /// Strict, bounded decoding; rejects noncanonical paths, order and duplicates.
    pub fn from_canonical_bytes(bytes: &[u8]) -> Result<Self, RebuildExplanationError> {
        let mut cursor = Cursor::bounded(bytes)?;
        let header = b"PRISM-BUILD-INPUTS\0\x01";
        if cursor.take(header.len())? != header {
            return Err(RebuildExplanationError::Invalid(
                "unknown input snapshot header",
            ));
        }
        let mut snapshot = Self::new();
        for _ in 0..cursor.count(58)? {
            let job = JobId(cursor.u128()?);
            let source = decode_identity(&mut cursor)?.ok_or(RebuildExplanationError::Invalid(
                "missing snapshot source identity",
            ))?;
            snapshot.sources.insert(job, source);
        }
        for _ in 0..cursor.count(24)? {
            let job = JobId(cursor.u128()?);
            snapshot.settings.insert(job, cursor.blob()?.to_vec());
        }
        for _ in 0..cursor.count(49)? {
            let job = JobId(cursor.u128()?);
            let hash = cursor.array32()?;
            let fields = match cursor.u8()? {
                0 => None,
                1 => {
                    let mut fields = BTreeMap::new();
                    for _ in 0..cursor.count(41)? {
                        let name = cursor.string()?;
                        validate_field_name(&name)?;
                        fields.insert(name, cursor.array32()?);
                    }
                    Some(fields)
                }
                _ => {
                    return Err(RebuildExplanationError::Invalid(
                        "invalid projection marker",
                    ))
                }
            };
            snapshot.profiles.insert(job, ProfileInput { hash, fields });
        }
        if !cursor.is_empty() || snapshot.canonical_bytes() != bytes {
            return Err(RebuildExplanationError::Invalid(
                "noncanonical input snapshot",
            ));
        }
        Ok(snapshot)
    }
}

fn profile_field_changes(
    before: Option<&ProfileInput>,
    after: Option<&ProfileInput>,
) -> Vec<ProfileFieldChange> {
    // A hash-only input is unknown, not an absent projection. Do not invent
    // field additions/removals when either side lacks field evidence.
    if before.is_some_and(|input| input.fields.is_none())
        || after.is_some_and(|input| input.fields.is_none())
    {
        return Vec::new();
    }
    let before = before.and_then(|input| input.fields.as_ref());
    let after = after.and_then(|input| input.fields.as_ref());
    let names: BTreeSet<_> = before
        .into_iter()
        .flat_map(|fields| fields.keys())
        .chain(after.into_iter().flat_map(|fields| fields.keys()))
        .collect();
    names
        .into_iter()
        .filter_map(|name| {
            let before_hash = before.and_then(|fields| fields.get(name)).copied();
            let after_hash = after.and_then(|fields| fields.get(name)).copied();
            (before_hash != after_hash).then(|| ProfileFieldChange {
                field: name.clone(),
                before_hash,
                after_hash,
            })
        })
        .collect()
}

/// One semantic input change that directly invalidated a graph job.
#[derive(Clone, Debug, PartialEq)]
pub enum InvalidationReason {
    /// A source path or source content digest changed.
    Source {
        job: JobId,
        before: Option<SourceIdentity>,
        after: Option<SourceIdentity>,
    },
    /// Canonical processor settings changed.
    Settings {
        job: JobId,
        before: Option<ContentHash>,
        after: Option<ContentHash>,
    },
    /// A declared profile projection changed.
    Profile {
        job: JobId,
        before: Option<ProfileHash>,
        after: Option<ProfileHash>,
        fields: Vec<ProfileFieldChange>,
    },
}

impl InvalidationReason {
    /// The graph job directly invalidated by this reason.
    #[must_use]
    pub const fn job(&self) -> JobId {
        match self {
            Self::Source { job, .. } | Self::Settings { job, .. } | Self::Profile { job, .. } => {
                *job
            }
        }
    }

    fn tag(&self) -> u8 {
        match self {
            Self::Source { .. } => 0,
            Self::Settings { .. } => 1,
            Self::Profile { .. } => 2,
        }
    }
}

/// A changed profile field included in a rebuild explanation.
#[derive(Clone, Debug, PartialEq)]
pub struct ProfileFieldChange {
    /// Stable profile field name.
    pub field: String,
    /// Previous field fingerprint, or None when not present in the projection.
    pub before_hash: Option<ProfileHash>,
    /// Current field fingerprint, or None when not present in the projection.
    pub after_hash: Option<ProfileHash>,
}

/// A dirty set plus deterministic evidence explaining every direct change.
#[derive(Clone, Debug, PartialEq)]
pub struct InvalidationPlan {
    dirty: DirtySet,
    changes: Vec<InvalidationReason>,
}

impl InvalidationPlan {
    /// The graph closure consumed by the incremental executor.
    #[must_use]
    pub fn dirty_set(&self) -> &DirtySet {
        &self.dirty
    }

    /// Direct source/settings/profile changes in JobId and change-kind order.
    #[must_use]
    pub fn changes(&self) -> &[InvalidationReason] {
        &self.changes
    }

    /// Returns whether no source, settings, or profile input changed.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.changes.is_empty()
    }

    /// Stable bytes used for persistence and cross-process comparison.
    #[must_use]
    pub fn canonical_bytes(&self) -> Vec<u8> {
        let mut bytes = REBUILD_EXPLANATION_HEADER.to_vec();
        put_len(&mut bytes, self.dirty.direct.len());
        for job in &self.dirty.direct {
            bytes.extend_from_slice(&job.0.to_le_bytes());
        }
        put_len(&mut bytes, self.dirty.affected.len());
        for job in &self.dirty.affected {
            bytes.extend_from_slice(&job.0.to_le_bytes());
        }
        put_len(&mut bytes, self.changes.len());
        for change in &self.changes {
            encode_reason(&mut bytes, change);
        }
        bytes
    }

    /// Decodes bounded canonical bytes without normalizing malformed evidence.
    pub fn from_canonical_bytes(bytes: &[u8]) -> Result<Self, RebuildExplanationError> {
        let mut cursor = Cursor::bounded(bytes)?;
        if cursor.take(REBUILD_EXPLANATION_HEADER.len())? != REBUILD_EXPLANATION_HEADER {
            return Err(RebuildExplanationError::Invalid(
                "unknown explanation header",
            ));
        }
        let direct = decode_jobs(&mut cursor)?;
        let affected = decode_jobs(&mut cursor)?;
        if !direct.is_subset(&affected) {
            return Err(RebuildExplanationError::Invalid(
                "direct jobs are not a subset of affected jobs",
            ));
        }
        // A valid reason needs a tag, job id, and at least one present hash.
        // Bound allocation by valid encoded evidence, not all-absent markers.
        let change_count = cursor.count(51)?;
        let mut changes = Vec::with_capacity(change_count);
        for _ in 0..change_count {
            changes.push(decode_reason(&mut cursor)?);
        }
        if !cursor.is_empty() {
            return Err(RebuildExplanationError::Invalid(
                "trailing explanation bytes",
            ));
        }
        let expected_direct: BTreeSet<_> = changes.iter().map(InvalidationReason::job).collect();
        if expected_direct != direct {
            return Err(RebuildExplanationError::Invalid(
                "explanation changes do not match direct jobs",
            ));
        }
        let plan = Self {
            dirty: DirtySet { direct, affected },
            changes,
        };
        if plan
            .changes
            .windows(2)
            .any(|pair| (pair[0].job(), pair[0].tag()) >= (pair[1].job(), pair[1].tag()))
            || (plan.dirty.direct.is_empty() && !plan.dirty.affected.is_empty())
            || plan.canonical_bytes() != bytes
        {
            return Err(RebuildExplanationError::Invalid("noncanonical explanation"));
        }
        for reason in &plan.changes {
            let changed = match reason {
                InvalidationReason::Source { before, after, .. } => before != after,
                InvalidationReason::Settings { before, after, .. } => before != after,
                InvalidationReason::Profile {
                    before,
                    after,
                    fields,
                    ..
                } => {
                    if fields.windows(2).any(|pair| pair[0].field >= pair[1].field) {
                        return Err(RebuildExplanationError::Invalid(
                            "noncanonical profile fields",
                        ));
                    }
                    for field in fields {
                        validate_field_name(&field.field)?;
                        if field.before_hash == field.after_hash
                            || (before.is_none() && field.before_hash.is_some())
                            || (after.is_none() && field.after_hash.is_some())
                        {
                            return Err(RebuildExplanationError::Invalid(
                                "invalid profile field change",
                            ));
                        }
                    }
                    before != after
                }
            };
            if !changed {
                return Err(RebuildExplanationError::Invalid(
                    "explanation contains unchanged input",
                ));
            }
        }
        Ok(plan)
    }
}

const REBUILD_EXPLANATION_HEADER: &[u8] = b"PRISM-REBUILD-EXPLANATION\0\x01";
const MAX_EXPLANATION_ITEMS: usize = 1_000_000;
const MAX_EXPLANATION_BYTES: usize = 16 * 1024 * 1024;

/// Failure while persisting or decoding a rebuild explanation.
#[derive(Debug)]
pub enum RebuildExplanationError {
    /// Local file access failed.
    Io(io::Error),
    /// The bytes do not satisfy the explanation protocol.
    Invalid(&'static str),
}

impl fmt::Display for RebuildExplanationError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Io(error) => write!(formatter, "rebuild explanation I/O error: {error}"),
            Self::Invalid(reason) => write!(formatter, "invalid rebuild explanation: {reason}"),
        }
    }
}

impl std::error::Error for RebuildExplanationError {}

impl From<io::Error> for RebuildExplanationError {
    fn from(value: io::Error) -> Self {
        Self::Io(value)
    }
}

fn put_len(bytes: &mut Vec<u8>, length: usize) {
    bytes.extend_from_slice(&(length as u64).to_le_bytes());
}

fn put_bytes(bytes: &mut Vec<u8>, value: &[u8]) {
    put_len(bytes, value.len());
    bytes.extend_from_slice(value);
}

fn put_optional_hash(bytes: &mut Vec<u8>, hash: Option<ContentHash>) {
    match hash {
        Some(hash) => {
            bytes.push(1);
            bytes.extend_from_slice(&hash.0);
        }
        None => bytes.push(0),
    }
}

fn put_optional_profile_hash(bytes: &mut Vec<u8>, hash: Option<ProfileHash>) {
    match hash {
        Some(hash) => {
            bytes.push(1);
            bytes.extend_from_slice(&hash);
        }
        None => bytes.push(0),
    }
}

fn encode_identity(bytes: &mut Vec<u8>, identity: Option<&SourceIdentity>) {
    match identity {
        Some(identity) => {
            bytes.push(1);
            put_bytes(bytes, identity.logical_path.as_bytes());
            bytes.extend_from_slice(&identity.content_hash.0);
        }
        None => bytes.push(0),
    }
}

fn encode_reason(bytes: &mut Vec<u8>, reason: &InvalidationReason) {
    bytes.push(reason.tag());
    bytes.extend_from_slice(&reason.job().0.to_le_bytes());
    match reason {
        InvalidationReason::Source { before, after, .. } => {
            encode_identity(bytes, before.as_ref());
            encode_identity(bytes, after.as_ref());
        }
        InvalidationReason::Settings { before, after, .. } => {
            put_optional_hash(bytes, *before);
            put_optional_hash(bytes, *after);
        }
        InvalidationReason::Profile {
            before,
            after,
            fields,
            ..
        } => {
            put_optional_profile_hash(bytes, *before);
            put_optional_profile_hash(bytes, *after);
            put_len(bytes, fields.len());
            for field in fields {
                put_bytes(bytes, field.field.as_bytes());
                put_optional_profile_hash(bytes, field.before_hash);
                put_optional_profile_hash(bytes, field.after_hash);
            }
        }
    }
}

fn decode_jobs(cursor: &mut Cursor<'_>) -> Result<BTreeSet<JobId>, RebuildExplanationError> {
    let count = cursor.count(16)?;
    let mut jobs = BTreeSet::new();
    for _ in 0..count {
        if !jobs.insert(JobId(cursor.u128()?)) {
            return Err(RebuildExplanationError::Invalid(
                "duplicate explanation job",
            ));
        }
    }
    Ok(jobs)
}

fn decode_reason(cursor: &mut Cursor<'_>) -> Result<InvalidationReason, RebuildExplanationError> {
    let tag = cursor.u8()?;
    let job = JobId(cursor.u128()?);
    match tag {
        0 => Ok(InvalidationReason::Source {
            job,
            before: decode_identity(cursor)?,
            after: decode_identity(cursor)?,
        }),
        1 => Ok(InvalidationReason::Settings {
            job,
            before: decode_hash(cursor)?,
            after: decode_hash(cursor)?,
        }),
        2 => {
            let before = decode_profile_hash(cursor)?;
            let after = decode_profile_hash(cursor)?;
            let count = cursor.count(43)?;
            let mut fields = Vec::with_capacity(count);
            for _ in 0..count {
                fields.push(ProfileFieldChange {
                    field: cursor.string()?,
                    before_hash: decode_profile_hash(cursor)?,
                    after_hash: decode_profile_hash(cursor)?,
                });
            }
            Ok(InvalidationReason::Profile {
                job,
                before,
                after,
                fields,
            })
        }
        _ => Err(RebuildExplanationError::Invalid(
            "unknown explanation change",
        )),
    }
}

fn decode_identity(
    cursor: &mut Cursor<'_>,
) -> Result<Option<SourceIdentity>, RebuildExplanationError> {
    match cursor.u8()? {
        0 => Ok(None),
        1 => {
            let logical_path = cursor.string()?;
            if !crate::source::is_canonical_path(&logical_path) {
                return Err(RebuildExplanationError::Invalid("noncanonical source path"));
            }
            Ok(Some(SourceIdentity {
                logical_path,
                content_hash: ContentHash(cursor.array32()?),
            }))
        }
        _ => Err(RebuildExplanationError::Invalid(
            "invalid source identity presence marker",
        )),
    }
}

fn decode_hash(cursor: &mut Cursor<'_>) -> Result<Option<ContentHash>, RebuildExplanationError> {
    match cursor.u8()? {
        0 => Ok(None),
        1 => Ok(Some(ContentHash(cursor.array32()?))),
        _ => Err(RebuildExplanationError::Invalid(
            "invalid hash presence marker",
        )),
    }
}

fn decode_profile_hash(
    cursor: &mut Cursor<'_>,
) -> Result<Option<ProfileHash>, RebuildExplanationError> {
    match cursor.u8()? {
        0 => Ok(None),
        1 => Ok(Some(cursor.array32()?)),
        _ => Err(RebuildExplanationError::Invalid(
            "invalid profile hash presence marker",
        )),
    }
}

pub(crate) struct Cursor<'a> {
    bytes: &'a [u8],
    offset: usize,
}

impl<'a> Cursor<'a> {
    pub(crate) fn bounded(bytes: &'a [u8]) -> Result<Self, RebuildExplanationError> {
        if bytes.len() > MAX_EXPLANATION_BYTES {
            return Err(RebuildExplanationError::Invalid(
                "metadata exceeds maximum encoded size",
            ));
        }
        Ok(Self { bytes, offset: 0 })
    }

    pub(crate) fn take(&mut self, length: usize) -> Result<&'a [u8], RebuildExplanationError> {
        if length > self.bytes.len() - self.offset {
            return Err(RebuildExplanationError::Invalid("truncated explanation"));
        }
        let result = &self.bytes[self.offset..self.offset + length];
        self.offset += length;
        Ok(result)
    }

    pub(crate) fn u8(&mut self) -> Result<u8, RebuildExplanationError> {
        Ok(*self.take(1)?.first().expect("one-byte slice is non-empty"))
    }

    pub(crate) fn u64(&mut self) -> Result<u64, RebuildExplanationError> {
        Ok(u64::from_le_bytes(self.take(8)?.try_into().unwrap()))
    }

    pub(crate) fn u128(&mut self) -> Result<u128, RebuildExplanationError> {
        Ok(u128::from_le_bytes(self.take(16)?.try_into().unwrap()))
    }

    fn len(&mut self) -> Result<usize, RebuildExplanationError> {
        usize::try_from(self.u64()?)
            .map_err(|_| RebuildExplanationError::Invalid("explanation length overflows usize"))
    }

    fn count(&mut self, minimum_bytes: usize) -> Result<usize, RebuildExplanationError> {
        let count = self.len()?;
        if count > MAX_EXPLANATION_ITEMS || count > (self.bytes.len() - self.offset) / minimum_bytes
        {
            return Err(RebuildExplanationError::Invalid(
                "metadata count exceeds remaining bytes",
            ));
        }
        Ok(count)
    }

    pub(crate) fn blob(&mut self) -> Result<&'a [u8], RebuildExplanationError> {
        let length = self.len()?;
        self.take(length)
    }

    fn string(&mut self) -> Result<String, RebuildExplanationError> {
        let length = self.len()?;
        let bytes = self.take(length)?;
        String::from_utf8(bytes.to_vec())
            .map_err(|_| RebuildExplanationError::Invalid("explanation contains invalid UTF-8"))
    }

    pub(crate) fn array32(&mut self) -> Result<[u8; 32], RebuildExplanationError> {
        Ok(self.take(32)?.try_into().unwrap())
    }

    pub(crate) fn is_empty(&self) -> bool {
        self.offset == self.bytes.len()
    }
}

fn validate_field_name(name: &str) -> Result<(), RebuildExplanationError> {
    if name.is_empty() || name.contains('\0') {
        return Err(RebuildExplanationError::Invalid(
            "invalid profile field name",
        ));
    }
    Ok(())
}

pub(crate) fn read_bounded(path: &Path, limit: usize) -> Result<Vec<u8>, RebuildExplanationError> {
    use std::io::Read;
    let file = fs::File::open(path)?;
    if !file.metadata()?.is_file() {
        return Err(RebuildExplanationError::Invalid(
            "metadata is not a regular file",
        ));
    }
    let mut bytes = Vec::new();
    file.take(limit as u64 + 1).read_to_end(&mut bytes)?;
    if bytes.len() > limit {
        return Err(RebuildExplanationError::Invalid(
            "metadata exceeds maximum encoded size",
        ));
    }
    Ok(bytes)
}

pub(crate) fn sync_directory(path: &Path) -> io::Result<()> {
    #[cfg(unix)]
    fs::File::open(path)?.sync_all()?;
    #[cfg(not(unix))]
    let _ = path;
    Ok(())
}
