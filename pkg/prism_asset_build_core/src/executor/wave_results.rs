//! Canonical application order, independent of worker result arrival order.

use crate::job::JobId;

pub(super) fn ordered<T>(mut results: Vec<(JobId, T)>) -> Vec<(JobId, T)> {
    results.sort_unstable_by_key(|(id, _)| *id);
    results
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_arrival_order_applies_success_then_smallest_failure() {
        // Feed completed results directly: processor markers cannot establish
        // an OS thread's actual return order. Exercise every permutation at
        // the production ordering boundary instead of relying on timing.
        for arrivals in [
            [1, 2, 5],
            [1, 5, 2],
            [2, 1, 5],
            [2, 5, 1],
            [5, 1, 2],
            [5, 2, 1],
        ] {
            let completed = arrivals
                .map(|id| (JobId(id), if id == 1 { Ok(id) } else { Err(id) }))
                .to_vec();
            let canonical = ordered(completed);
            assert_eq!(
                canonical,
                vec![(JobId(1), Ok(1)), (JobId(2), Err(2)), (JobId(5), Err(5))]
            );
            let mut applied = Vec::new();
            let outcome: Result<(), u128> = canonical.into_iter().try_for_each(|(job, result)| {
                result?;
                applied.push(job);
                Ok(())
            });
            assert_eq!(applied, vec![JobId(1)]);
            assert_eq!(outcome, Err(2));
        }
    }

    #[test]
    fn empty_and_successful_waves_keep_all_results() {
        assert!(ordered::<()>(Vec::new()).is_empty());
        assert_eq!(
            ordered(vec![
                (JobId(9), b"nine".as_slice()),
                (JobId(3), b"three".as_slice())
            ]),
            vec![
                (JobId(3), b"three".as_slice()),
                (JobId(9), b"nine".as_slice())
            ]
        );
    }
}
