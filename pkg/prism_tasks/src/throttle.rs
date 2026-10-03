//! Concurrency and rate throttling of job submission (design §17 节流/配额,
//! §24.8 背压).
//!
//! Background lanes (streaming decode, asset bake, async-compute) must not
//! starve foreground frame work by flooding the pool. A [`Throttle`] caps how
//! many throttled jobs run concurrently (a counting semaphore) and, optionally,
//! how fast they may be *admitted* (a token bucket). [`TaskPool::spawn_throttled`]
//! acquires a [`Permit`] before enqueuing a job; the permit rides into the job
//! and is released when the job finishes, so the number of simultaneously
//! running throttled jobs never exceeds the configured concurrency.
//!
//! ## Deadlock contract
//! [`Throttle::acquire`] *blocks the calling thread* until a permit is free.
//! Submit throttled jobs from a control thread (or any thread that is not
//! itself one of the throttled jobs it waits on). Acquiring from inside a pool
//! worker while all permits are held by jobs that need that worker to make
//! progress can deadlock — that is inherent to blocking admission control, so
//! keep submission off the worker critical path.

use alloc::sync::Arc;
use std::sync::{Condvar, Mutex};
use std::time::{Duration, Instant};

use crate::{Counter, TaskPool};

/// Longest a blocked [`Throttle::acquire`] sleeps before re-checking the token
/// bucket. Bounds the latency of a rate-limited wakeup without busy-spinning.
const MAX_TOKEN_WAIT: Duration = Duration::from_millis(50);

/// Token-bucket rate parameters.
#[derive(Clone, Copy)]
struct Rate {
    /// Maximum tokens the bucket holds (the burst size).
    capacity: f64,
    /// Tokens replenished per second.
    per_sec: f64,
}

/// Mutable throttle state behind the lock.
struct State {
    /// Free concurrency permits.
    available: usize,
    /// Current token-bucket level (only meaningful when a [`Rate`] is set).
    tokens: f64,
    /// Timestamp the bucket was last refilled.
    last_refill: Instant,
}

/// Shared throttle internals.
struct Inner {
    /// Maximum concurrent permits.
    max: usize,
    /// Optional admission rate limit.
    rate: Option<Rate>,
    /// Guarded mutable state.
    state: Mutex<State>,
    /// Signalled on permit release and used for timed rate waits.
    cvar: Condvar,
}

/// A concurrency (and optionally rate) limiter for job submission.
///
/// Cheap to clone; clones share the same permit pool. See the module docs for
/// the blocking/deadlock contract.
#[derive(Clone)]
pub struct Throttle {
    inner: Arc<Inner>,
}

impl Throttle {
    /// Build a throttle allowing at most `max` concurrent permits (clamped to
    /// at least `1`), with no rate limit.
    #[must_use]
    pub fn with_concurrency(max: usize) -> Self {
        let max = max.max(1);
        Self {
            inner: Arc::new(Inner {
                max,
                rate: None,
                state: Mutex::new(State {
                    available: max,
                    tokens: 0.0,
                    last_refill: Instant::now(),
                }),
                cvar: Condvar::new(),
            }),
        }
    }

    /// Build a throttle with both a concurrency cap and an admission rate limit
    /// of `permits_per_second` (token bucket, burst = one second's worth).
    ///
    /// `max_concurrency` is clamped to at least `1` and `permits_per_second` to
    /// a small positive value, so the throttle always makes forward progress.
    #[must_use]
    pub fn with_rate(max_concurrency: usize, permits_per_second: f64) -> Self {
        let max = max_concurrency.max(1);
        let per_sec = if permits_per_second.is_finite() && permits_per_second > 0.0 {
            permits_per_second
        } else {
            1.0
        };
        let capacity = per_sec.max(1.0);
        Self {
            inner: Arc::new(Inner {
                max,
                rate: Some(Rate { capacity, per_sec }),
                state: Mutex::new(State {
                    available: max,
                    tokens: capacity,
                    last_refill: Instant::now(),
                }),
                cvar: Condvar::new(),
            }),
        }
    }

    /// Maximum number of concurrent permits this throttle grants.
    #[must_use]
    pub fn max_concurrency(&self) -> usize {
        self.inner.max
    }

    /// Number of concurrency permits currently free.
    #[must_use]
    pub fn available(&self) -> usize {
        self.inner.state.lock().unwrap().available
    }

    /// Acquire a permit, blocking the calling thread until one is free (and,
    /// when rate-limited, until the token bucket has a token). The permit is
    /// released when the returned [`Permit`] is dropped.
    #[must_use]
    pub fn acquire(&self) -> Permit {
        let mut state = self.inner.state.lock().unwrap();
        loop {
            if let Some(rate) = self.inner.rate {
                refill(&rate, &mut state);
            }
            let rate_ready = match self.inner.rate {
                Some(_) => state.tokens >= 1.0,
                None => true,
            };
            if state.available > 0 && rate_ready {
                state.available -= 1;
                if self.inner.rate.is_some() {
                    state.tokens -= 1.0;
                }
                drop(state);
                return Permit {
                    inner: Arc::clone(&self.inner),
                };
            }
            // A permit is free but the bucket is dry: timed wait so the next
            // refill is observed even without a release notification.
            if let Some(rate) = self.inner.rate
                && state.available > 0
                && !rate_ready
            {
                let wait = token_wait(&rate, &state);
                let (next, _timeout) = self.inner.cvar.wait_timeout(state, wait).unwrap();
                state = next;
            } else {
                state = self.inner.cvar.wait(state).unwrap();
            }
        }
    }

    /// Try to acquire a permit without blocking, returning `None` if none is
    /// immediately available (or the rate bucket is empty).
    #[must_use]
    pub fn try_acquire(&self) -> Option<Permit> {
        let mut state = self.inner.state.lock().unwrap();
        if let Some(rate) = self.inner.rate {
            refill(&rate, &mut state);
        }
        let rate_ready = match self.inner.rate {
            Some(_) => state.tokens >= 1.0,
            None => true,
        };
        if state.available > 0 && rate_ready {
            state.available -= 1;
            if self.inner.rate.is_some() {
                state.tokens -= 1.0;
            }
            drop(state);
            Some(Permit {
                inner: Arc::clone(&self.inner),
            })
        } else {
            None
        }
    }
}

/// Refill the token bucket for the time elapsed since the last refill.
fn refill(rate: &Rate, state: &mut State) {
    let now = Instant::now();
    let elapsed = now.duration_since(state.last_refill).as_secs_f64();
    if elapsed > 0.0 {
        state.tokens = (state.tokens + elapsed * rate.per_sec).min(rate.capacity);
        state.last_refill = now;
    }
}

/// Time to wait for the bucket to accumulate one token, capped at
/// [`MAX_TOKEN_WAIT`].
fn token_wait(rate: &Rate, state: &State) -> Duration {
    let needed = (1.0 - state.tokens).max(0.0);
    let secs = needed / rate.per_sec;
    let wait = Duration::from_secs_f64(secs.max(0.0));
    wait.min(MAX_TOKEN_WAIT)
}

/// A held throttle permit. Releasing it (on drop) frees a concurrency slot and
/// wakes one waiter.
pub struct Permit {
    inner: Arc<Inner>,
}

impl Drop for Permit {
    fn drop(&mut self) {
        {
            let mut state = self.inner.state.lock().unwrap();
            state.available += 1;
        }
        self.inner.cvar.notify_one();
    }
}

impl core::fmt::Debug for Permit {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("Permit").finish_non_exhaustive()
    }
}

impl TaskPool {
    /// Spawn `f` under `throttle`, tracking it on `counter`.
    ///
    /// Acquires a [`Permit`] from `throttle` **on the calling thread** (blocking
    /// until one is free), then enqueues the job carrying the permit. The
    /// permit is released when the job finishes, so at most
    /// [`Throttle::max_concurrency`] throttled jobs run at once. See the module
    /// deadlock contract: submit from a control thread, not from a worker whose
    /// progress the throttled jobs depend on. In the single-threaded fallback
    /// the job runs inline and releases its permit immediately.
    pub fn spawn_throttled<F>(&self, counter: &Counter, throttle: &Throttle, f: F)
    where
        F: FnOnce() + Send + 'static,
    {
        let permit = throttle.acquire();
        counter.add(1);
        let counter = counter.clone();
        let job = move || {
            f();
            drop(permit);
            counter.finish_one();
        };
        if self.is_single_threaded() {
            job();
        } else {
            self.push_job(Box::new(job));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::Throttle;

    #[test]
    fn concurrency_permits_bound_available() {
        let t = Throttle::with_concurrency(2);
        assert_eq!(t.max_concurrency(), 2);
        assert_eq!(t.available(), 2);
        let p1 = t.try_acquire().unwrap();
        assert_eq!(t.available(), 1);
        let p2 = t.try_acquire().unwrap();
        assert_eq!(t.available(), 0);
        assert!(t.try_acquire().is_none());
        drop(p1);
        assert_eq!(t.available(), 1);
        drop(p2);
        assert_eq!(t.available(), 2);
    }

    #[test]
    fn zero_concurrency_is_clamped_to_one() {
        let t = Throttle::with_concurrency(0);
        assert_eq!(t.max_concurrency(), 1);
        let p = t.try_acquire().unwrap();
        assert!(t.try_acquire().is_none());
        drop(p);
        assert!(t.try_acquire().is_some());
    }

    #[test]
    fn rate_limit_drains_the_initial_burst() {
        // Capacity is `per_sec.max(1.0)` = 4 tokens; a slow refill means after
        // draining the burst the next acquire is not immediately available.
        let t = Throttle::with_rate(16, 4.0);
        let mut held = Vec::new();
        for _ in 0..4 {
            held.push(t.try_acquire().expect("burst token available"));
        }
        // Burst exhausted: the bucket is dry even though concurrency remains.
        assert!(t.try_acquire().is_none());
        assert!(t.available() > 0);
    }

    #[test]
    fn acquire_blocks_until_release() {
        use std::sync::mpsc;
        use std::thread;

        let t = Throttle::with_concurrency(1);
        let held = t.acquire();
        let t2 = t.clone();
        let (tx, rx) = mpsc::channel();
        let handle = thread::spawn(move || {
            let _p = t2.acquire();
            tx.send(()).unwrap();
        });
        // The second acquire cannot complete while the first permit is held.
        assert!(rx.recv_timeout(std::time::Duration::from_millis(50)).is_err());
        drop(held);
        rx.recv_timeout(std::time::Duration::from_secs(1)).unwrap();
        handle.join().unwrap();
    }
}
