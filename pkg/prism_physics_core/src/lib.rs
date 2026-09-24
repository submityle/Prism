//! Backend-neutral core kernel for Prism's next-generation physics engine.
//!
//! This crate provides the M0 foundation: a Structure-of-Arrays rigid-body
//! state store, math helpers, a real semi-implicit Euler integrator, and the
//! reserved extension points (solver / backend / driver / constraint / island)
//! that later milestones build upon.
//!
//! It is engine-agnostic and contains no Unreal Engine source or derived code.
#![forbid(unsafe_code)]
