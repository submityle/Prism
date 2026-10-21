//! Contiguous index buckets: two buffers, rather than one allocation per island.

#[derive(Clone, Debug, Default)]
pub(crate) struct IndexBuckets {
    offsets: Vec<usize>,
    indices: Vec<usize>,
}

impl IndexBuckets {
    pub(crate) fn with_counts(counts: impl ExactSizeIterator<Item = usize>) -> Self {
        let mut offsets = Vec::with_capacity(counts.len() + 1);
        offsets.push(0_usize);
        for count in counts {
            let end = offsets.last().copied().unwrap() + count;
            offsets.push(end);
        }
        let indices = vec![0; offsets.last().copied().unwrap()];
        Self { offsets, indices }
    }

    pub(crate) fn set(&mut self, bucket: usize, offset: usize, index: usize) {
        let start = self.offsets[bucket];
        assert!(offset < self.offsets[bucket + 1] - start);
        self.indices[start + offset] = index;
    }

    pub(crate) fn get(&self, bucket: usize) -> &[usize] {
        let Some(&start) = self.offsets.get(bucket) else {
            return &[];
        };
        let Some(&end) = self.offsets.get(bucket + 1) else {
            return &[];
        };
        &self.indices[start..end]
    }

    pub(crate) fn len(&self) -> usize {
        self.offsets.len().saturating_sub(1)
    }

    pub(crate) fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

#[cfg(test)]
mod tests {
    use super::IndexBuckets;

    #[test]
    fn empty_buckets_and_arbitrary_access_are_bounded() {
        let mut buckets = IndexBuckets::with_counts([2, 0, 1].into_iter());
        buckets.set(0, 0, 7);
        buckets.set(0, 1, 9);
        buckets.set(2, 0, 3);
        assert_eq!(buckets.len(), 3);
        assert_eq!(buckets.get(0), [7, 9]);
        assert!(buckets.get(1).is_empty());
        assert_eq!(buckets.get(2), [3]);
        assert!(buckets.get(3).is_empty());
        assert!(buckets.get(usize::MAX).is_empty());
        assert!(IndexBuckets::default().is_empty());
        assert!(IndexBuckets::default().get(0).is_empty());
        assert!(IndexBuckets::with_counts([].into_iter()).is_empty());
    }
}
