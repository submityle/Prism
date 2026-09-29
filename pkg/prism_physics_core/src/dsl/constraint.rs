//! Runtime constraint that projects a compiled residual with XPBD.
//!
//! A [`DslConstraint`] wraps a [`CompiledConstraint`] and binds its point
//! variables to concrete particle indices. Each [`project`](DslConstraint::project)
//! call evaluates the residual `C`, estimates the gradient `∇C` with respect
//! to every coupled particle by central finite differences, and applies the
//! standard XPBD position update
//!
//! ```text
//! alpha_tilde = compliance / dt^2
//! delta_lambda = (-C - alpha_tilde * lambda) / (sum_i w_i |grad_i|^2 + alpha_tilde)
//! x_i += w_i * delta_lambda * grad_i
//! ```
//!
//! # Provenance
//!
//! This module contains **no Unreal Engine source or derived code**. The
//! compliant projection is the canonical XPBD update (Macklin et al. 2016);
//! finite-difference gradients are standard numerical-analysis knowledge.

use glam::Vec3;

use crate::dsl::bytecode::Program;
use crate::dsl::compiler::CompiledConstraint;
use crate::dsl::env::Environment;
use crate::dsl::error::DslError;
use crate::dsl::vm::eval;
use crate::math::scalar::{Real, EPSILON};

/// Finite-difference step used when estimating the residual gradient.
const FD_STEP: Real = 1.0e-3;

/// A compiled DSL residual bound to concrete particles for XPBD projection.
#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub struct DslConstraint {
    name: String,
    residual: Program,
    compliance_program: Program,
    env: Environment,
    /// Base slot of each coupled point, parallel to `point_particles`.
    point_bases: Vec<usize>,
    /// Particle index bound to each point, parallel to `point_bases`.
    point_particles: Vec<usize>,
    /// Current parameter values, occupying slots `0..params_len`.
    params: Vec<Real>,
    /// Cached compliance (inverse stiffness), recomputed when params change.
    compliance: Real,
    /// Accumulated Lagrange multiplier for the current substep.
    lambda: Real,
}

impl DslConstraint {
    /// Binds a [`CompiledConstraint`] to parameter values and particle indices.
    ///
    /// `param_values` must match the compiled parameter count, and
    /// `particle_indices` must match the compiled point count (in the compiled
    /// point order).
    ///
    /// # Errors
    ///
    /// Returns [`DslError::Compile`] on a length mismatch, or [`DslError::Eval`]
    /// if the compliance expression cannot be evaluated.
    pub fn from_compiled(
        compiled: &CompiledConstraint,
        param_values: &[Real],
        particle_indices: &[usize],
    ) -> Result<Self, DslError> {
        if param_values.len() != compiled.params.len() {
            return Err(DslError::compile(format!(
                "expected {} parameter value(s), got {}",
                compiled.params.len(),
                param_values.len()
            )));
        }
        if particle_indices.len() != compiled.point_names.len() {
            return Err(DslError::compile(format!(
                "expected {} particle index(es), got {}",
                compiled.point_names.len(),
                particle_indices.len()
            )));
        }
        let point_bases: Vec<usize> = compiled
            .point_names
            .iter()
            .map(|name| {
                compiled
                    .env
                    .lookup(name)
                    .map(|b| b.base)
                    .ok_or_else(|| DslError::compile(format!("point '{name}' has no slot")))
            })
            .collect::<Result<_, _>>()?;

        let params = param_values.to_vec();
        let compliance = Self::eval_compliance(&compiled.compliance, &params, &compiled.env)?;

        Ok(DslConstraint {
            name: compiled.name.clone(),
            residual: compiled.residual.clone(),
            compliance_program: compiled.compliance.clone(),
            env: compiled.env.clone(),
            point_bases,
            point_particles: particle_indices.to_vec(),
            params,
            compliance,
            lambda: 0.0,
        })
    }

    fn eval_compliance(
        program: &Program,
        params: &[Real],
        env: &Environment,
    ) -> Result<Real, DslError> {
        let mut values = vec![0.0; env.total_slots().max(params.len())];
        values[..params.len()].copy_from_slice(params);
        let value = eval(program, &values)?;
        Ok(value.max(0.0))
    }

    /// Builds the full value slice for the current particle positions.
    ///
    /// Returns `None` if any bound particle index is out of range.
    fn build_values(&self, positions: &[Vec3]) -> Option<Vec<Real>> {
        let mut values = vec![0.0; self.env.total_slots().max(self.params.len())];
        values[..self.params.len()].copy_from_slice(&self.params);
        for (&base, &particle) in self.point_bases.iter().zip(self.point_particles.iter()) {
            let p = positions.get(particle)?;
            values[base] = p.x;
            values[base + 1] = p.y;
            values[base + 2] = p.z;
        }
        Some(values)
    }

    /// Evaluates the residual at the current particle positions, if defined.
    #[must_use]
    pub fn residual_value(&self, positions: &[Vec3]) -> Option<Real> {
        let values = self.build_values(positions)?;
        eval(&self.residual, &values).ok()
    }

    /// Central finite difference of the residual with respect to slot `idx`.
    fn central_diff(&self, values: &mut [Real], idx: usize) -> Option<Real> {
        let saved = values[idx];
        values[idx] = saved + FD_STEP;
        let plus = eval(&self.residual, values).ok()?;
        values[idx] = saved - FD_STEP;
        let minus = eval(&self.residual, values).ok()?;
        values[idx] = saved;
        Some((plus - minus) / (2.0 * FD_STEP))
    }

    /// Projects the constraint once, nudging coupled particle positions toward
    /// `C = 0` under the XPBD compliant-constraint update.
    ///
    /// Out-of-range particle indices or an unevaluable residual make the call a
    /// no-op rather than a panic.
    pub fn project(&mut self, positions: &mut [Vec3], inverse_masses: &[Real], dt: Real) {
        let Some(mut values) = self.build_values(positions) else {
            return;
        };
        let Ok(c0) = eval(&self.residual, &values) else {
            return;
        };

        // Gather per-particle gradient (from three central differences) and
        // inverse mass.
        let mut contributions: Vec<(usize, Vec3, Real)> =
            Vec::with_capacity(self.point_bases.len());
        for (&base, &particle) in self.point_bases.iter().zip(self.point_particles.iter()) {
            let Some(gx) = self.central_diff(&mut values, base) else {
                return;
            };
            let Some(gy) = self.central_diff(&mut values, base + 1) else {
                return;
            };
            let Some(gz) = self.central_diff(&mut values, base + 2) else {
                return;
            };
            let grad = Vec3::new(gx, gy, gz);
            let w = inverse_masses.get(particle).copied().unwrap_or(0.0);
            contributions.push((particle, grad, w));
        }

        let mut denom = 0.0;
        for &(_, grad, w) in &contributions {
            denom += w * grad.length_squared();
        }
        let alpha_tilde = if dt.abs() < EPSILON {
            0.0
        } else {
            self.compliance / (dt * dt)
        };
        denom += alpha_tilde;
        if denom < EPSILON {
            return;
        }

        let delta_lambda = (-c0 - alpha_tilde * self.lambda) / denom;
        self.lambda += delta_lambda;

        for &(particle, grad, w) in &contributions {
            if let Some(slot) = positions.get_mut(particle) {
                *slot += grad * (w * delta_lambda);
            }
        }
    }

    /// Updates the parameter at `index` and recomputes the cached compliance.
    ///
    /// # Errors
    ///
    /// Returns [`DslError::Compile`] if `index` is out of range, or
    /// [`DslError::Eval`] if the compliance expression cannot be re-evaluated.
    pub fn set_param(&mut self, index: usize, value: Real) -> Result<(), DslError> {
        if index >= self.params.len() {
            return Err(DslError::compile("parameter index out of range"));
        }
        self.params[index] = value;
        self.compliance = Self::eval_compliance(&self.compliance_program, &self.params, &self.env)?;
        Ok(())
    }

    /// Returns the constraint name.
    #[must_use]
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Returns the current cached compliance.
    #[must_use]
    pub fn compliance(&self) -> Real {
        self.compliance
    }

    /// Returns the accumulated Lagrange multiplier for the current substep.
    #[must_use]
    pub fn lambda(&self) -> Real {
        self.lambda
    }

    /// Resets the accumulated Lagrange multiplier to zero (call once per
    /// substep before the first projection).
    pub fn reset(&mut self) {
        self.lambda = 0.0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dsl::compiler::compile_constraint;
    use crate::dsl::parser::parse_constraint;

    fn distance_constraint(rest: Real) -> DslConstraint {
        let decl =
            parse_constraint("constraint d(rest) { residual = length(b - a) - rest; }").unwrap();
        let compiled = compile_constraint(&decl).unwrap();
        // Points in first-use order are [b, a]; bind b->1, a->0.
        DslConstraint::from_compiled(&compiled, &[rest], &[1, 0]).unwrap()
    }

    #[test]
    fn projection_reduces_residual_magnitude() {
        let mut c = distance_constraint(1.0);
        let mut positions = [Vec3::ZERO, Vec3::new(2.0, 0.0, 0.0)];
        let inv = [1.0, 1.0];
        let before = c.residual_value(&positions).unwrap().abs();
        c.project(&mut positions, &inv, 1.0 / 60.0);
        let after = c.residual_value(&positions).unwrap().abs();
        assert!(after < before, "before={before}, after={after}");
    }

    #[test]
    fn set_param_updates_compliance() {
        let decl = parse_constraint(
            "constraint d(rest, k) { residual = length(b - a) - rest; compliance = k; }",
        )
        .unwrap();
        let compiled = compile_constraint(&decl).unwrap();
        let mut c = DslConstraint::from_compiled(&compiled, &[1.0, 0.0], &[1, 0]).unwrap();
        assert!((c.compliance() - 0.0).abs() < 1e-9);
        c.set_param(1, 0.5).unwrap();
        assert!((c.compliance() - 0.5).abs() < 1e-6);
    }

    #[test]
    fn out_of_range_particle_is_inert() {
        let decl =
            parse_constraint("constraint d(rest) { residual = length(b - a) - rest; }").unwrap();
        let compiled = compile_constraint(&decl).unwrap();
        let mut c = DslConstraint::from_compiled(&compiled, &[1.0], &[1, 9]).unwrap();
        let mut positions = [Vec3::ZERO, Vec3::new(2.0, 0.0, 0.0)];
        let inv = [1.0, 1.0];
        c.project(&mut positions, &inv, 1.0 / 60.0);
        assert_eq!(positions[1], Vec3::new(2.0, 0.0, 0.0));
    }

    #[test]
    fn set_param_out_of_range_errors() {
        let mut c = distance_constraint(1.0);
        assert!(c.set_param(5, 1.0).is_err());
    }
}
