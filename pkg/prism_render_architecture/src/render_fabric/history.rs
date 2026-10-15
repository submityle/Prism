//! Backend-neutral publication ledger for physically distinct lag-one slots.
//! Resource ownership, device reset and layout reset are the caller's lifetime.

#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct LagOneHistory {
    published: Option<u64>,
}

impl LagOneHistory {
    /// Missing frames never fall back to an older, semantically wrong version.
    pub fn read_slot(&self, frame: u64) -> Option<usize> {
        let previous = frame.checked_sub(1)?;
        (self.published == Some(previous)).then_some(Self::write_slot(previous))
    }

    pub const fn write_slot(frame: u64) -> usize {
        (frame & 1) as usize
    }

    pub fn can_publish(&self, frame: u64) -> bool {
        self.published.is_none_or(|last| frame > last)
    }

    /// Call only after the complete writer has been recorded for submission.
    pub fn publish(&mut self, frame: u64) -> bool {
        if !self.can_publish(frame) {
            return false;
        }
        self.published = Some(frame);
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lag_one_requires_exact_publication_and_distinct_current_slot() {
        let mut history = LagOneHistory::default();
        assert_eq!(history.read_slot(0), None);
        assert_eq!(history.read_slot(1), None);
        assert!(history.publish(1));
        assert_eq!(history.read_slot(2), Some(1));
        assert_ne!(history.read_slot(2).unwrap(), LagOneHistory::write_slot(2));
        assert!(!history.publish(1));
        assert!(!history.publish(0));
        assert_eq!(history.read_slot(3), None);
        assert!(history.publish(4));
        assert_eq!(history.read_slot(5), Some(0));
    }
}
