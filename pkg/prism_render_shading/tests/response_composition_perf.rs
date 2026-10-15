#![allow(missing_docs)]

#[path = "support/response_composition_workload.rs"]
mod workload;

use workload::{build_plan, continuous_item_count, plan_digest, standard_workload, WorkloadLedger};

#[test]
fn fixed_response_composition_workload_is_deterministic_and_accounted() {
    let workload = standard_workload();
    let ledger = WorkloadLedger::for_workload(&workload);
    let first = build_plan(&workload);
    let second = build_plan(&workload);

    assert_eq!(
        workload.name,
        "response-composition-1280x720-mixed-20pct-continuous"
    );
    assert_eq!(ledger.workload_pixels, 921_600);
    assert_eq!(ledger.bucket_count, 28);
    assert_eq!(ledger.bucket_count, workload.buckets.len() as u32);
    assert_eq!(ledger.auxiliary_jobs, 115_200);
    assert_eq!(first, second);
    assert_eq!(plan_digest(&first), plan_digest(&second));
    assert_eq!(first.work.len(), ledger.workload_pixels as usize);
    assert_eq!(continuous_item_count(&first), 184_320);
    assert_eq!(first.offsets.len(), workload.buckets.len());
    assert_eq!(first.counts.iter().sum::<u32>(), ledger.workload_pixels);
    for (bucket, (&offset, &count)) in workload
        .buckets
        .buckets()
        .iter()
        .zip(first.offsets.iter().zip(first.counts.iter()))
    {
        let end = offset + count;
        assert!(end as usize <= first.work.len());
        assert!(first.work[offset as usize..end as usize]
            .iter()
            .all(|item| item.shading_class == bucket.bucket_id));
    }

    let composition_per_pixel = 2 * 64 + 16 + 4;
    assert_eq!(
        ledger.composition_payload_bytes,
        u64::from(ledger.workload_pixels) * composition_per_pixel
            + u64::from(ledger.auxiliary_jobs) * 64
    );
    assert_eq!(
        ledger.known_gpu_payload_bytes,
        ledger.composition_payload_bytes + ledger.classification_payload_bytes
    );
    assert!(ledger.cpu_input_payload_bytes > 0);
    assert_eq!(
        ledger.work_item_payload_bytes,
        u64::from(ledger.workload_pixels) * 32
    );
}
