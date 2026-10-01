#![allow(
    clippy::std_instead_of_alloc,
    reason = "integration tests are std-only binaries"
)]
//! Behavioural tests for the fine-grained reactive core.

use std::cell::Cell;
use std::rc::Rc;

use prism_ui_reactive::Runtime;

#[test]
fn signal_get_set_roundtrip() {
    let rt = Runtime::new();
    let s = rt.signal(1);
    assert_eq!(s.get(), 1);
    s.set(42);
    assert_eq!(s.get(), 42);
    s.update(|v| *v += 1);
    assert_eq!(s.get(), 43);
}

#[test]
fn memo_derives_and_caches() {
    let rt = Runtime::new();
    let a = rt.signal(2);
    let b = rt.signal(3);
    let runs = Rc::new(Cell::new(0));
    let sum = rt.memo({
        let (a, b, runs) = (a.clone(), b.clone(), runs.clone());
        move || {
            runs.set(runs.get() + 1);
            a.get() + b.get()
        }
    });

    // Lazy: not evaluated until first read.
    assert_eq!(runs.get(), 0);
    assert_eq!(sum.get(), 5);
    assert_eq!(runs.get(), 1);
    // Cached: repeated reads do not recompute.
    assert_eq!(sum.get(), 5);
    assert_eq!(runs.get(), 1);
    // Recompute only after a dependency changes.
    a.set(10);
    assert_eq!(sum.get(), 13);
    assert_eq!(runs.get(), 2);
}

#[test]
fn effect_runs_immediately_and_on_change() {
    let rt = Runtime::new();
    let s = rt.signal(0);
    let seen = Rc::new(std::cell::RefCell::new(Vec::new()));
    let _e = rt.effect({
        let (s, seen) = (s.clone(), seen.clone());
        move || seen.borrow_mut().push(s.get())
    });
    assert_eq!(*seen.borrow(), vec![0]);
    s.set(1);
    s.set(2);
    assert_eq!(*seen.borrow(), vec![0, 1, 2]);
}

#[test]
fn diamond_is_glitch_free() {
    // a -> b, a -> c, (b,c) -> d. A single write to `a` must recompute `d`
    // exactly once with consistent inputs.
    let rt = Runtime::new();
    let a = rt.signal(1);
    let b = rt.memo({
        let a = a.clone();
        move || a.get() + 1
    });
    let c = rt.memo({
        let a = a.clone();
        move || a.get() * 10
    });
    let d_runs = Rc::new(Cell::new(0));
    let d = rt.memo({
        let (b, c, d_runs) = (b.clone(), c.clone(), d_runs.clone());
        move || {
            d_runs.set(d_runs.get() + 1);
            b.get() + c.get()
        }
    });

    assert_eq!(d.get(), 12); // (1+1) + (1*10)
    assert_eq!(d_runs.get(), 1);
    a.set(2);
    assert_eq!(d.get(), 23); // (2+1) + (2*10)
    assert_eq!(d_runs.get(), 2); // exactly one recompute, not two
}

#[test]
fn unchanged_memo_output_prunes_downstream() {
    let rt = Runtime::new();
    let n = rt.signal(4);
    let is_even = rt.memo({
        let n = n.clone();
        move || n.get() % 2 == 0
    });
    let downstream_runs = Rc::new(Cell::new(0));
    let _label = rt.effect({
        let (is_even, downstream_runs) = (is_even.clone(), downstream_runs.clone());
        move || {
            let _ = is_even.get();
            downstream_runs.set(downstream_runs.get() + 1);
        }
    });
    assert_eq!(downstream_runs.get(), 1);
    // 4 -> 6: still even, so `is_even` output is unchanged and the effect must
    // not re-run.
    n.set(6);
    assert_eq!(downstream_runs.get(), 1);
    // 6 -> 7: parity flips, effect re-runs once.
    n.set(7);
    assert_eq!(downstream_runs.get(), 2);
}

#[test]
fn dynamic_dependencies_are_tracked() {
    let rt = Runtime::new();
    let toggle = rt.signal(true);
    let x = rt.signal(10);
    let y = rt.signal(20);
    let runs = Rc::new(Cell::new(0));
    let out = rt.memo({
        let (toggle, x, y, runs) = (toggle.clone(), x.clone(), y.clone(), runs.clone());
        move || {
            runs.set(runs.get() + 1);
            if toggle.get() {
                x.get()
            } else {
                y.get()
            }
        }
    });
    assert_eq!(out.get(), 10);
    let runs_after_init = runs.get();
    // While reading `x`, changing `y` must not trigger a recompute.
    y.set(999);
    assert_eq!(out.get(), 10);
    assert_eq!(runs.get(), runs_after_init);
    // Switch the branch: now `y` is a dependency and `x` is not.
    toggle.set(false);
    assert_eq!(out.get(), 999);
    let runs_after_switch = runs.get();
    x.set(0);
    assert_eq!(out.get(), 999);
    assert_eq!(runs.get(), runs_after_switch);
}

#[test]
fn batch_coalesces_updates() {
    let rt = Runtime::new();
    let a = rt.signal(1);
    let b = rt.signal(2);
    let runs = Rc::new(Cell::new(0));
    let _e = rt.effect({
        let (a, b, runs) = (a.clone(), b.clone(), runs.clone());
        move || {
            let _ = a.get() + b.get();
            runs.set(runs.get() + 1);
        }
    });
    assert_eq!(runs.get(), 1);
    rt.batch(|| {
        a.set(10);
        b.set(20);
    });
    // Two writes inside a batch produce a single effect run.
    assert_eq!(runs.get(), 2);
}

#[test]
fn untrack_suppresses_dependency() {
    let rt = Runtime::new();
    let tracked = rt.signal(1);
    let hidden = rt.signal(1);
    let runs = Rc::new(Cell::new(0));
    let _e = rt.effect({
        let (tracked, hidden, runs) = (tracked.clone(), hidden.clone(), runs.clone());
        let rt2 = rt.clone();
        move || {
            let _ = tracked.get() + rt2.untrack(|| hidden.get());
            runs.set(runs.get() + 1);
        }
    });
    assert_eq!(runs.get(), 1);
    hidden.set(100); // untracked read: no re-run
    assert_eq!(runs.get(), 1);
    tracked.set(2); // tracked read: re-run
    assert_eq!(runs.get(), 2);
}

#[test]
fn dispose_frees_nodes_and_stops_effects() {
    let rt = Runtime::new();
    let s = rt.signal(0);
    let runs = Rc::new(Cell::new(0));
    let e = rt.effect({
        let (s, runs) = (s.clone(), runs.clone());
        move || {
            let _ = s.get();
            runs.set(runs.get() + 1);
        }
    });
    assert_eq!(runs.get(), 1);
    e.dispose();
    s.set(1);
    assert_eq!(runs.get(), 1); // disposed effect does not re-run
}
