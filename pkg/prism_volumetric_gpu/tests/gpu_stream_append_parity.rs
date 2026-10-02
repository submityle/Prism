//! Real-device parity for the atomic stream-append twin:
//! [`GpuStreamAppend`](prism_volumetric_gpu::gpu_stream_append::GpuStreamAppend)
//! must reproduce the terminal state of the `CPU` golden
//! [`gpu_stream_append`](prism_render_architecture::particle::gpu_stream_append)
//! across an empty input (host short-circuit), inputs that fit entirely, an
//! exact capacity fill, overflowing batches, a zero-capacity buffer that drops
//! everything, duplicate values, and large pseudo-random batches with and
//! without overflow.
//!
//! The tests skip (returning early) when the host has no `wgpu` adapter, so the
//! suite stays green everywhere while still exercising the full
//! dispatch-and-readback on any real device such as an Apple `M`-series `GPU`.
//! The kernel is portable core-`WGSL` plus `atomicAdd`, so it needs no optional
//! device feature.
//!
//! # Parity criterion
//!
//! The deterministic facts are integers: the terminal counter record
//! `[count, overflow, capacity, reserved]` is asserted with **exact `==`**
//! against the golden `AppendCounter::to_std430`. `GPU` `atomicAdd` does not
//! promise which lane claims which slot, so the written slot *order* is never
//! asserted. When nothing overflows (`capacity >= count`) the written multiset
//! equals the whole input multiset, so the *sorted* readback equals the sorted
//! golden store; when the batch overflows, the written values are asserted to
//! be a sub-multiset of the input (every surviving value came from the input)
//! and to fill exactly the dense `[0, appended)` prefix. There is no float math
//! anywhere, so the comparison is bit-exact by construction with no
//! `ULP`-boundary degenerate region to avoid.
//!
//! Provenance: 孪生自本仓 `prism_render_architecture::particle::gpu_stream_append`；无第三方引擎源码或衍生代码。

use prism_render_architecture::particle::gpu_stream_append::AppendCounter;
use prism_volumetric_gpu::gpu_stream_append::{GpuStreamAppend, GpuStreamAppendQuery};
use prism_volumetric_gpu::GpuContext;

/// A tiny integer linear-congruential generator; only integer arithmetic, so no
/// transcendental appears. Returns the full 32-bit high word of the state.
struct Lcg {
    state: u64,
}

impl Lcg {
    fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    fn next_u32(&mut self) -> u32 {
        self.state = self
            .state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        (self.state >> 32) as u32
    }

    /// Fills a length-`len` vector of pseudo-random payload values.
    fn fill_values(&mut self, len: usize) -> Vec<u32> {
        (0..len).map(|_| self.next_u32()).collect()
    }
}

/// Builds the golden terminal `AppendCounter` by appending every value in order
/// against `capacity`, exactly as the `CPU` reference does.
fn golden_counter(values: &[u32], capacity: usize) -> AppendCounter {
    let mut counter = AppendCounter::new(capacity);
    for &value in values {
        counter.append(value);
    }
    counter
}

/// Whether every value in `sub` appears in `sup` at least as many times (a
/// sub-multiset test), used for the overflow case where which values survived
/// is not deterministic but each must have come from the input.
fn is_sub_multiset(sub: &[u32], sup: &[u32]) -> bool {
    let mut a = sub.to_vec();
    a.sort_unstable();
    let mut b = sup.to_vec();
    b.sort_unstable();
    let mut j = 0usize;
    for &value in &a {
        while j < b.len() && b[j] < value {
            j += 1;
        }
        if j >= b.len() || b[j] != value {
            return false;
        }
        j += 1;
    }
    true
}

/// Runs the `GPU` append and asserts parity against the `CPU` golden: the
/// terminal counter record matches exactly, and the written slots match the
/// golden as a multiset (no overflow) or as a sub-multiset of the input
/// (overflow).
fn check(ctx: &GpuContext, gpu: &GpuStreamAppend, values: &[u32], capacity: u32) {
    let query = GpuStreamAppendQuery {
        values: values.to_vec(),
        capacity,
    };
    let result = gpu.evaluate(ctx, &query);

    let golden = golden_counter(values, capacity as usize);
    let want_words = golden.to_std430();

    // The terminal counter record is deterministic: exact u32 equality.
    assert_eq!(
        result.counter.words(),
        want_words,
        "counter record [count, overflow, capacity, reserved] must match (capacity {capacity})"
    );
    assert_eq!(
        u64::from(result.appended),
        golden.count() as u64,
        "appended count must match the golden slot count (capacity {capacity})"
    );
    assert_eq!(
        u64::from(result.overflow),
        golden.overflow_count() as u64,
        "overflow tally must match (capacity {capacity})"
    );
    assert_eq!(
        result.slots.len(),
        result.appended as usize,
        "the written store is exactly the dense `[0, appended)` prefix (capacity {capacity})"
    );

    let no_overflow = (values.len() as u64) <= u64::from(capacity);
    if no_overflow {
        // Nothing dropped: the written multiset equals the whole input, so the
        // sorted readback equals the sorted golden store element for element.
        let mut got = result.slots.clone();
        got.sort_unstable();
        let mut want = golden.slots().to_vec();
        want.sort_unstable();
        assert_eq!(
            got, want,
            "with no overflow the written multiset equals the input (capacity {capacity})"
        );
    } else {
        // The surviving values are not a deterministic subset, but each must
        // have come from the input and the prefix must be full.
        assert!(
            is_sub_multiset(&result.slots, values),
            "every surviving slot value must come from the input (capacity {capacity})"
        );
        assert_eq!(
            result.appended, capacity,
            "an overflowing batch fills the whole capacity (capacity {capacity})"
        );
    }
}

#[test]
fn empty_input_is_degenerate_but_valid() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuStreamAppend::new(&ctx);
    // No value issues no dispatch; the terminal state is empty with capacity
    // preserved in the counter record.
    let query = GpuStreamAppendQuery {
        values: Vec::new(),
        capacity: 8,
    };
    let result = gpu.evaluate(&ctx, &query);
    assert_eq!(result.counter.words(), [0, 0, 8, 0]);
    assert!(result.slots.is_empty(), "empty input appends nothing");
    assert_eq!(result.appended, 0);
    assert_eq!(result.overflow, 0);
}

#[test]
fn all_values_fit_preserves_full_multiset() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuStreamAppend::new(&ctx);
    let values = [10u32, 20, 30, 40, 50];
    check(&ctx, &gpu, &values, 16);
}

#[test]
fn exact_capacity_fill_has_no_overflow() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuStreamAppend::new(&ctx);
    let mut lcg = Lcg::new(0x1234_5678_9ABC_DEF0);
    let values = lcg.fill_values(64);
    // capacity == count: every value fits, the counter reports zero overflow.
    check(&ctx, &gpu, &values, 64);
}

#[test]
fn single_value_fits() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuStreamAppend::new(&ctx);
    check(&ctx, &gpu, &[42u32], 4);
}

#[test]
fn duplicate_values_keep_their_multiplicity() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuStreamAppend::new(&ctx);
    // Duplicates verify a multiset (not set) comparison when nothing overflows.
    let values = [7u32, 7, 7, 3, 3, 9];
    check(&ctx, &gpu, &values, 32);
}

#[test]
fn overflow_drops_the_excess_and_tallies_it() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuStreamAppend::new(&ctx);
    let mut lcg = Lcg::new(0xDEAD_BEEF_0BAD_F00D);
    let values = lcg.fill_values(100);
    // capacity < count: 10 survive, 90 overflow; survivors come from the input.
    check(&ctx, &gpu, &values, 10);
}

#[test]
fn zero_capacity_overflows_everything() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuStreamAppend::new(&ctx);
    let values = [1u32, 2, 3];
    check(&ctx, &gpu, &values, 0);
    // Spell out the terminal record for the degenerate zero-capacity buffer.
    let result = gpu.evaluate(
        &ctx,
        &GpuStreamAppendQuery {
            values: values.to_vec(),
            capacity: 0,
        },
    );
    assert_eq!(result.counter.words(), [0, 3, 0, 0]);
    assert!(result.slots.is_empty());
}

#[test]
fn capacity_one_keeps_a_single_slot() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuStreamAppend::new(&ctx);
    let mut lcg = Lcg::new(0x0F0F_0F0F_1234_9999);
    let values = lcg.fill_values(50);
    // capacity 1: one survivor, 49 overflow; the survivor is one of the inputs.
    check(&ctx, &gpu, &values, 1);
}

#[test]
fn large_random_batch_without_overflow() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuStreamAppend::new(&ctx);
    let mut lcg = Lcg::new(0xC0FF_EE11_2233_4455);
    let values = lcg.fill_values(4096);
    // capacity comfortably above count: full-multiset parity across many warps.
    check(&ctx, &gpu, &values, 8192);
}

#[test]
fn large_random_batch_with_overflow() {
    let Some(ctx) = GpuContext::try_headless() else {
        return;
    };
    let gpu = GpuStreamAppend::new(&ctx);
    let mut lcg = Lcg::new(0xA5A5_5A5A_7777_1111);
    let values = lcg.fill_values(4096);
    // capacity below count: 1000 survive, the rest overflow; counter is exact.
    check(&ctx, &gpu, &values, 1000);
}
