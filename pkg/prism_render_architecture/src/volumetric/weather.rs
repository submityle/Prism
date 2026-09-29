//! Weather-map dynamics: semi-Lagrangian `advect`ion, the sky-state machine,
//! and precipitation classification (design section 9, testability section 16).
//!
//! The `weather` map is a small 2D `RGBA` field whose channels drive the global
//! cloud distribution (`R` = `coverage`, `G` = cloud type, `B` = `precip`
//! intensity, `A` = wind disturbance); one cell is a [`WeatherSample`]. This
//! module owns the *dynamics* of that field, entirely on the `CPU` as pure,
//! deterministic functions so the reference path is verifiable in the sandbox
//! (there is no `GPU` here) and the numeric properties can be unit-tested:
//!
//! - **[`WeatherField`]** — a caller-owned `alloc::vec::Vec<WeatherSample>` grid
//!   with clamp-to-edge `bilinear` sampling ([`WeatherField::sample_bilinear`])
//!   and `get`/`set` accessors that never panic.
//! - **[`advect_semi_lagrangian`]** — moves the map along the wind by tracing
//!   every cell centre backward and `bilinear`-sampling the old field there.
//!   The backtrace is unconditionally stable and, for an interior feature under
//!   a divergence-free wind, near mass-conserving (the total `coverage` drifts
//!   only by `bilinear` rounding, which the tests bound).
//! - **[`SkyState`]** — the discrete sky-state machine
//!   (`Clear`/`Fair`/`Overcast`/`Storm`) with a target `coverage` per state and
//!   a smooth, monotone, jump-free [`blend_state`] interpolation.
//! - **[`classify_precip`]** — a deterministic threshold classifier turning a
//!   [`WeatherSample`] plus a [`CloudKind`] and a temperature into a
//!   rain/snow/none [`Precip`] signal (signal only; no particles are spawned).
//! - **[`WindField`]** — the wind interface (direction + speed + a `curl`
//!   disturbance strength) that drives advection with a divergence-free
//!   perturbation via [`advect_with_wind`].
//!
//! Only classical numerical methods are used, no `AI`/`ML`. The determinism
//! policy allows only [`f32::sqrt`] among the float intrinsics, so every
//! transcendental (`exp`, `sin`, `cos`) routes through [`super::math`], and all
//! iteration is fixed-order row-major over caller-owned grids.

use alloc::vec;
use alloc::vec::Vec;

use super::math::{clamp, cos_approx, exp_approx, saturate, sin_approx, smoothstep, Vec2};
use super::{CloudKind, WeatherMapHandle, WeatherSample};

/// Target `coverage` of a fully `Clear` sky (a trace of fair-weather cloud).
pub const CLEAR_COVERAGE: f32 = 0.05;

/// Target `coverage` of a `Fair` (scattered fair-weather cloud) sky.
pub const FAIR_COVERAGE: f32 = 0.30;

/// Target `coverage` of an `Overcast` (broken-to-solid deck) sky.
pub const OVERCAST_COVERAGE: f32 = 0.75;

/// Target `coverage` of a `Storm` (near-total, deep-convective) sky.
pub const STORM_COVERAGE: f32 = 0.95;

/// Threshold on the `precip` (`B`) channel at or above which a cell is treated
/// as precipitating; below it [`classify_precip`] returns [`PrecipKind::None`].
pub const PRECIP_TRIGGER: f32 = 0.5;

/// Air temperature (degrees Celsius) at or below which precipitation falls as
/// snow rather than rain; the rain/snow split is deterministic at this edge.
pub const FREEZING_POINT_C: f32 = 0.0;

/// Intensity gain applied to `Cumulonimbus` precipitation: the deep-convective
/// storm cloud is the full-strength precipitation source (design section 9b).
pub const CUMULONIMBUS_PRECIP_GAIN: f32 = 1.0;

/// Intensity gain applied to every non-`Cumulonimbus` [`CloudKind`]: other
/// kinds only signal drizzle-strength precipitation through the `weather` map.
pub const DEFAULT_PRECIP_GAIN: f32 = 0.5;

/// Spatial frequency (cycles per cell) of the divergence-free `curl`
/// stream-function that perturbs the wind in [`WindField::velocity_at`].
pub const CURL_FREQUENCY: f32 = 0.15;

/// A dense 2D `weather` map: a row-major grid of [`WeatherSample`] cells that
/// the dynamics functions read and produce.
///
/// The backing storage is a caller-owned `alloc::vec::Vec<WeatherSample>` of
/// exactly `width * height` cells in row-major order (`index = y * width + x`).
/// All sampling and accessors clamp to the grid, so no operation can panic on
/// an out-of-range coordinate. The [`WeatherMapHandle`] carries the map's
/// stable identity across an [`advect_semi_lagrangian`] step.
#[derive(Clone, Debug, PartialEq)]
pub struct WeatherField {
    /// Stable identity of the `weather`-map tile this field represents.
    handle: WeatherMapHandle,
    /// Cell count along x (columns); may be `0` for an empty field.
    width: u32,
    /// Cell count along y (rows); may be `0` for an empty field.
    height: u32,
    /// Row-major cells, length exactly `width * height`.
    cells: Vec<WeatherSample>,
}

impl WeatherField {
    /// Builds a `width * height` field with every cell set to the default
    /// (all-zero) [`WeatherSample`].
    #[must_use]
    pub fn new(handle: WeatherMapHandle, width: u32, height: u32) -> Self {
        Self::filled(handle, width, height, WeatherSample::default())
    }

    /// Builds a `width * height` field with every cell set to `fill`.
    #[must_use]
    pub fn filled(handle: WeatherMapHandle, width: u32, height: u32, fill: WeatherSample) -> Self {
        let count = (width as usize) * (height as usize);
        Self {
            handle,
            width,
            height,
            cells: vec![fill; count],
        }
    }

    /// Builds a field from an explicit row-major cell list, or returns `None`
    /// when `cells.len()` does not equal `width * height` (guarding the
    /// row-major invariant every accessor relies on).
    #[must_use]
    pub fn from_cells(
        handle: WeatherMapHandle,
        width: u32,
        height: u32,
        cells: Vec<WeatherSample>,
    ) -> Option<Self> {
        if cells.len() == (width as usize) * (height as usize) {
            Some(Self {
                handle,
                width,
                height,
                cells,
            })
        } else {
            None
        }
    }

    /// Stable identity of this `weather`-map tile.
    #[must_use]
    pub fn handle(&self) -> WeatherMapHandle {
        self.handle
    }

    /// Cell count along x (columns).
    #[must_use]
    pub fn width(&self) -> u32 {
        self.width
    }

    /// Cell count along y (rows).
    #[must_use]
    pub fn height(&self) -> u32 {
        self.height
    }

    /// Total number of cells (`width * height`).
    #[must_use]
    pub fn len(&self) -> usize {
        self.cells.len()
    }

    /// `true` when the field holds no cells (`width` or `height` is `0`).
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.cells.is_empty()
    }

    /// Read-only view of the row-major cells.
    #[must_use]
    pub fn cells(&self) -> &[WeatherSample] {
        &self.cells
    }

    /// Row-major flat index of integer cell `(x, y)`, or `None` when the
    /// coordinate lies outside the grid.
    #[must_use]
    pub fn index(&self, x: u32, y: u32) -> Option<usize> {
        if x < self.width && y < self.height {
            Some((y as usize) * (self.width as usize) + (x as usize))
        } else {
            None
        }
    }

    /// Reads the cell at integer `(x, y)`, clamping the coordinate to the grid
    /// edge so an out-of-range read returns the nearest border cell instead of
    /// panicking. An empty field yields the default [`WeatherSample`].
    #[must_use]
    pub fn get(&self, x: u32, y: u32) -> WeatherSample {
        if self.is_empty() {
            return WeatherSample::default();
        }
        let cx = x.min(self.width - 1);
        let cy = y.min(self.height - 1);
        self.cells[(cy as usize) * (self.width as usize) + (cx as usize)]
    }

    /// Writes `sample` into the cell at integer `(x, y)`; returns `true` when a
    /// cell was written and `false` when the coordinate is out of range (a
    /// stale index can never panic).
    pub fn set(&mut self, x: u32, y: u32, sample: WeatherSample) -> bool {
        match self.index(x, y) {
            Some(idx) => {
                self.cells[idx] = sample;
                true
            }
            None => false,
        }
    }

    /// `bilinear`-samples the field at fractional cell coordinates `(u, v)`
    /// (measured in cell units, cell centres at integer coordinates).
    ///
    /// Coordinates are clamped into `[0, width - 1] x [0, height - 1]`
    /// (clamp-to-edge), so sampling never reads out of bounds and never panics.
    /// The four taps are convex-combined via [`WeatherSample::lerp`], so every
    /// channel of the result stays within the range spanned by its taps (and
    /// thus in `0..=1` for an in-range field). An empty field returns the
    /// default [`WeatherSample`].
    #[must_use]
    pub fn sample_bilinear(&self, u: f32, v: f32) -> WeatherSample {
        if self.is_empty() {
            return WeatherSample::default();
        }
        let max_x = (self.width - 1) as f32;
        let max_y = (self.height - 1) as f32;
        let cx = clamp(u, 0.0, max_x);
        let cy = clamp(v, 0.0, max_y);
        // Truncation toward zero is exact here: cx, cy are already non-negative
        // and clamped below the grid maximum.
        let x0 = cx as u32;
        let y0 = cy as u32;
        let x1 = (x0 + 1).min(self.width - 1);
        let y1 = (y0 + 1).min(self.height - 1);
        let fx = cx - (x0 as f32);
        let fy = cy - (y0 as f32);
        let s00 = self.get(x0, y0);
        let s10 = self.get(x1, y0);
        let s01 = self.get(x0, y1);
        let s11 = self.get(x1, y1);
        let top = s00.lerp(s10, fx);
        let bottom = s01.lerp(s11, fx);
        top.lerp(bottom, fy)
    }

    /// Sum of the `coverage` channel over every cell, the conserved quantity the
    /// advection mass tests track.
    #[must_use]
    pub fn total_coverage(&self) -> f32 {
        let mut sum = 0.0_f32;
        let mut i = 0;
        while i < self.cells.len() {
            sum += self.cells[i].coverage;
            i += 1;
        }
        sum
    }
}

/// `advect`s the `weather` map one step along a uniform `wind` by a
/// semi-Lagrangian backtrace.
///
/// Every cell centre `(x, y)` is traced backward to `(x - wind.x * dt,
/// y - wind.y * dt)` (in cell units) and the previous field is
/// `bilinear`-sampled there. The scheme is unconditionally stable for any step,
/// clamps to the grid edge at the domain border, and preserves each channel's
/// `0..=1` range (the taps are convex-combined). For an interior feature under
/// this divergence-free (uniform) wind the total `coverage` is conserved up to
/// `bilinear` rounding; [`WeatherField::total_coverage`] drift is bounded by the
/// module tests.
#[must_use]
pub fn advect_semi_lagrangian(field: &WeatherField, wind: Vec2, dt: f32) -> WeatherField {
    let width = field.width();
    let height = field.height();
    let mut out = WeatherField::filled(field.handle(), width, height, WeatherSample::default());
    if out.is_empty() {
        return out;
    }
    let mut y = 0;
    while y < height {
        let mut x = 0;
        while x < width {
            let px = (x as f32) - wind.x * dt;
            let py = (y as f32) - wind.y * dt;
            out.set(x, y, field.sample_bilinear(px, py));
            x += 1;
        }
        y += 1;
    }
    out
}

/// A global wind that drives advection: a direction, a speed, and a `curl`
/// disturbance strength (design section 9, "风场").
///
/// The base flow is `direction * speed` (cells per unit time). On top of it, a
/// divergence-free `curl` perturbation of magnitude `speed * curl_strength`
/// (further scaled by a cell's local wind disturbance) adds the wispy shearing
/// motion `Cirrus`-style streaks need. Keeping the perturbation divergence-free
/// preserves the near mass-conservation of [`advect_with_wind`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct WindField {
    /// Unit wind direction (the zero vector when the input was degenerate).
    direction: Vec2,
    /// Wind speed in cells per unit time (`>= 0`).
    speed: f32,
    /// `curl` disturbance strength (`>= 0`); `0` gives a pure uniform wind.
    curl_strength: f32,
}

impl WindField {
    /// Builds a wind from a `direction` (normalized internally), a `speed`
    /// (clamped non-negative), and a `curl_strength` (clamped non-negative).
    #[must_use]
    pub fn new(direction: Vec2, speed: f32, curl_strength: f32) -> Self {
        Self {
            direction: direction.normalize_or_zero(),
            speed: speed.max(0.0),
            curl_strength: curl_strength.max(0.0),
        }
    }

    /// Builds a wind from a compass `angle` in radians (0 points along +x),
    /// using the hand-rolled [`cos_approx`] / [`sin_approx`] so no float
    /// intrinsic is used.
    #[must_use]
    pub fn from_polar(angle: f32, speed: f32, curl_strength: f32) -> Self {
        Self::new(
            Vec2::new(cos_approx(angle), sin_approx(angle)),
            speed,
            curl_strength,
        )
    }

    /// Unit wind direction.
    #[must_use]
    pub fn direction(&self) -> Vec2 {
        self.direction
    }

    /// Wind speed in cells per unit time.
    #[must_use]
    pub fn speed(&self) -> f32 {
        self.speed
    }

    /// `curl` disturbance strength.
    #[must_use]
    pub fn curl_strength(&self) -> f32 {
        self.curl_strength
    }

    /// Uniform base velocity `direction * speed`.
    #[must_use]
    pub fn base_velocity(&self) -> Vec2 {
        self.direction.scale(self.speed)
    }

    /// Divergence-free `curl` velocity shape at `pos` from the stream-function
    /// `psi = sin(f x) sin(f y)`, giving `(dpsi/dy, -dpsi/dx)`.
    ///
    /// The analytic partials cancel in the divergence (`d(vx)/dx + d(vy)/dy = 0`),
    /// which is exactly what keeps advection near mass-conserving. Amplitude is
    /// order one (the leading frequency factor is dropped on purpose so the
    /// perturbation scale is governed by `speed * curl_strength`).
    #[must_use]
    fn curl_shape(pos: Vec2) -> Vec2 {
        let ax = CURL_FREQUENCY * pos.x;
        let ay = CURL_FREQUENCY * pos.y;
        Vec2::new(
            sin_approx(ax) * cos_approx(ay),
            -cos_approx(ax) * sin_approx(ay),
        )
    }

    /// Total wind velocity at `pos` (cell units), combining the base flow with
    /// the `curl` perturbation scaled by `speed * curl_strength` and the local
    /// `wind`-disturbance channel `local_disturbance` (saturated to `0..=1`).
    #[must_use]
    pub fn velocity_at(&self, pos: Vec2, local_disturbance: f32) -> Vec2 {
        let gust = self.speed * self.curl_strength * (1.0 + saturate(local_disturbance));
        self.base_velocity().add(Self::curl_shape(pos).scale(gust))
    }
}

/// `advect`s the `weather` map one step along a [`WindField`], sampling the
/// per-cell velocity (base flow plus the divergence-free `curl` gust modulated
/// by each cell's wind-disturbance channel) and tracing that cell backward.
///
/// Like [`advect_semi_lagrangian`] the backtrace is unconditionally stable and
/// clamps to the grid edge; because the perturbation is divergence-free the
/// total `coverage` stays near-conserved for interior features (the tests bound
/// the drift).
#[must_use]
pub fn advect_with_wind(field: &WeatherField, wind: &WindField, dt: f32) -> WeatherField {
    let width = field.width();
    let height = field.height();
    let mut out = WeatherField::filled(field.handle(), width, height, WeatherSample::default());
    if out.is_empty() {
        return out;
    }
    let mut y = 0;
    while y < height {
        let mut x = 0;
        while x < width {
            let cell = field.get(x, y);
            let pos = Vec2::new(x as f32, y as f32);
            let vel = wind.velocity_at(pos, cell.wind_disturbance);
            let px = pos.x - vel.x * dt;
            let py = pos.y - vel.y * dt;
            out.set(x, y, field.sample_bilinear(px, py));
            x += 1;
        }
        y += 1;
    }
    out
}

/// The discrete sky-state machine driving large-scale generation and decay:
/// `Clear` -> `Fair` -> `Overcast` -> `Storm` and back (design section 9,
/// "生消"). Each state maps to a target `coverage`; transitions between states
/// are smoothed by [`blend_state`] so `coverage` never jumps.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum SkyState {
    /// Clear sky: only a trace of `coverage`.
    Clear,
    /// Fair weather: scattered cloud.
    Fair,
    /// Overcast: a broken-to-solid deck.
    Overcast,
    /// Storm: near-total, deep-convective `coverage`.
    Storm,
}

impl SkyState {
    /// Position of this state on the intensification ladder (`Clear` = 0 …
    /// `Storm` = 3), the ordering the state machine advances along.
    #[must_use]
    pub fn ordinal(self) -> u8 {
        match self {
            SkyState::Clear => 0,
            SkyState::Fair => 1,
            SkyState::Overcast => 2,
            SkyState::Storm => 3,
        }
    }

    /// State at ladder position `ordinal`, clamping values above `3` to
    /// `Storm` so the mapping is total and never panics.
    #[must_use]
    pub fn from_ordinal(ordinal: u8) -> Self {
        match ordinal {
            0 => SkyState::Clear,
            1 => SkyState::Fair,
            2 => SkyState::Overcast,
            _ => SkyState::Storm,
        }
    }

    /// The target steady-state `coverage` (`0..=1`) this state relaxes toward.
    #[must_use]
    pub fn target_coverage(self) -> f32 {
        match self {
            SkyState::Clear => CLEAR_COVERAGE,
            SkyState::Fair => FAIR_COVERAGE,
            SkyState::Overcast => OVERCAST_COVERAGE,
            SkyState::Storm => STORM_COVERAGE,
        }
    }
}

/// The next intensifying sky-state (`Clear` -> `Fair` -> `Overcast` ->
/// `Storm`); `Storm` saturates and stays `Storm`.
#[must_use]
pub fn advance_state(state: SkyState) -> SkyState {
    SkyState::from_ordinal(state.ordinal().saturating_add(1))
}

/// The next dissipating sky-state (the reverse of [`advance_state`]); `Clear`
/// saturates and stays `Clear`.
#[must_use]
pub fn dissipate_state(state: SkyState) -> SkyState {
    SkyState::from_ordinal(state.ordinal().saturating_sub(1))
}

/// Target `coverage` (`0..=1`) of `state`, forwarding to
/// [`SkyState::target_coverage`] for call-site symmetry with [`blend_state`].
#[must_use]
pub fn target_coverage(state: SkyState) -> f32 {
    state.target_coverage()
}

/// Smoothly interpolates the target `coverage` from state `from` to state `to`
/// as the transition parameter `t` runs `0..=1`.
///
/// `t` is saturated to `0..=1` and shaped by a Hermite [`smoothstep`], so the
/// blend starts and ends with zero slope (no visible jump at either endpoint),
/// is monotone in `t` (`smoothstep` is monotone), and always lies within the
/// `coverage` interval spanned by the two states — hence within `0..=1`. The
/// endpoints are exact: `t = 0` yields `from`'s target, `t = 1` yields `to`'s.
#[must_use]
pub fn blend_state(from: SkyState, to: SkyState, t: f32) -> f32 {
    let a = from.target_coverage();
    let b = to.target_coverage();
    let w = smoothstep(0.0, 1.0, t);
    saturate(a + (b - a) * w)
}

/// Relaxes a current `coverage` toward `target` over a step `dt` at rate
/// `rate` (per unit time), using the smooth exponential approach
/// `1 - exp(-rate * dt)`.
///
/// The blend weight stays in `0..=1` (via [`saturate`] and the hand-rolled
/// [`exp_approx`], never a float intrinsic), so the result is monotone toward
/// `target` and never overshoots; if both `current` and `target` are in `0..=1`
/// the result is too. This is the per-cell generation/decay coupling that ties
/// the [`SkyState`] machine to a [`WeatherField`].
#[must_use]
pub fn relax_coverage(current: f32, target: f32, rate: f32, dt: f32) -> f32 {
    let k = saturate(1.0 - exp_approx(-rate.max(0.0) * dt.max(0.0)));
    current + (target - current) * k
}

/// The precipitation phase a cell produces.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum PrecipKind {
    /// No precipitation (below the `precip` trigger or dry cloud).
    None,
    /// Liquid precipitation (temperature above [`FREEZING_POINT_C`]).
    Rain,
    /// Frozen precipitation (temperature at or below [`FREEZING_POINT_C`]).
    Snow,
}

/// A deterministic precipitation signal: a phase plus an intensity in `0..=1`.
///
/// This is a *signal only* — it says whether and how hard a cell precipitates so
/// downstream systems (particles, wetness) can react; it never spawns particles
/// itself (design section 9).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Precip {
    /// The precipitation phase.
    pub kind: PrecipKind,
    /// Precipitation intensity in `0..=1` (`0` exactly when `kind` is
    /// [`PrecipKind::None`]).
    pub intensity: f32,
}

impl Precip {
    /// The no-precipitation signal (`PrecipKind::None`, zero intensity).
    pub const NONE: Self = Self {
        kind: PrecipKind::None,
        intensity: 0.0,
    };
}

/// Classifies a cell's precipitation deterministically from its
/// [`WeatherSample`], its [`CloudKind`], and the air `temperature` (Celsius).
///
/// The classification is a pure threshold cascade:
/// 1. If the `precip` (`B`) channel is below [`PRECIP_TRIGGER`], the cell is dry
///    ([`Precip::NONE`]).
/// 2. Otherwise the phase is `Snow` when `temperature <= FREEZING_POINT_C` and
///    `Rain` above it (the freezing edge itself is `Snow`, deterministically).
/// 3. The intensity is the `precip` channel scaled by the kind gain
///    ([`CUMULONIMBUS_PRECIP_GAIN`] for the deep-convective storm cloud,
///    [`DEFAULT_PRECIP_GAIN`] otherwise) and modulated by `coverage`, saturated
///    to `0..=1`.
///
/// Being a pure function of its inputs, it is fully deterministic: identical
/// inputs always yield identical output, with the boundaries above resolved
/// consistently.
#[must_use]
pub fn classify_precip(sample: &WeatherSample, kind: CloudKind, temperature: f32) -> Precip {
    if sample.precipitation < PRECIP_TRIGGER {
        return Precip::NONE;
    }
    let gain = if kind.precipitation_capable() {
        CUMULONIMBUS_PRECIP_GAIN
    } else {
        DEFAULT_PRECIP_GAIN
    };
    let intensity = saturate(sample.precipitation * gain * sample.coverage.max(0.0));
    let phase = if temperature <= FREEZING_POINT_C {
        PrecipKind::Snow
    } else {
        PrecipKind::Rain
    };
    Precip {
        kind: phase,
        intensity,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The handle every test field shares; identity is irrelevant to dynamics.
    const H: WeatherMapHandle = WeatherMapHandle(7);

    /// Absolute tolerance for equality of computed reals in these tests.
    const TOL: f32 = 1e-4;

    /// Builds an `n x n` field whose `coverage` is a smooth interior bump well
    /// away from the border (so clamp-to-edge advection neither gains nor loses
    /// mass), leaving the other channels flat.
    fn interior_bump(n: u32) -> WeatherField {
        let mut f = WeatherField::new(H, n, n);
        let c = (n as f32 - 1.0) * 0.5;
        let mut y = 0;
        while y < n {
            let mut x = 0;
            while x < n {
                let dx = x as f32 - c;
                let dy = y as f32 - c;
                let r2 = dx * dx + dy * dy;
                // A compact quadratic bump vanishing before the border.
                let cov = saturate(1.0 - r2 / 9.0);
                f.set(x, y, WeatherSample::from_rgba(cov, 0.5, 0.0, 0.0));
                x += 1;
            }
            y += 1;
        }
        f
    }

    #[test]
    fn bilinear_clamps_to_edge_without_panicking() {
        let f = interior_bump(8);
        // Far outside in every direction resolves to the nearest corner.
        assert_eq!(f.sample_bilinear(-100.0, -100.0), f.get(0, 0));
        assert_eq!(f.sample_bilinear(1000.0, -5.0), f.get(7, 0));
        assert_eq!(f.sample_bilinear(-5.0, 1000.0), f.get(0, 7));
        assert_eq!(f.sample_bilinear(1000.0, 1000.0), f.get(7, 7));
        // Integer coordinates round-trip the stored cell.
        assert_eq!(f.sample_bilinear(3.0, 4.0), f.get(3, 4));
    }

    #[test]
    fn empty_field_never_panics() {
        let f = WeatherField::new(H, 0, 0);
        assert!(f.is_empty());
        assert_eq!(f.sample_bilinear(2.5, -1.0), WeatherSample::default());
        assert_eq!(f.get(4, 9), WeatherSample::default());
        assert_eq!(f.total_coverage(), 0.0);
        let advected = advect_semi_lagrangian(&f, Vec2::new(1.0, 1.0), 0.5);
        assert!(advected.is_empty());
    }

    #[test]
    fn from_cells_enforces_length_invariant() {
        assert!(WeatherField::from_cells(H, 2, 2, vec![WeatherSample::default(); 4]).is_some());
        assert!(WeatherField::from_cells(H, 2, 2, vec![WeatherSample::default(); 3]).is_none());
    }

    #[test]
    fn set_reports_out_of_range_without_panicking() {
        let mut f = WeatherField::new(H, 3, 3);
        assert!(f.set(2, 2, WeatherSample::from_rgba(1.0, 0.0, 0.0, 0.0)));
        assert!(!f.set(3, 0, WeatherSample::default()));
        assert!(!f.set(0, 9, WeatherSample::default()));
        assert_eq!(f.get(2, 2).coverage, 1.0);
    }

    #[test]
    fn uniform_field_advects_to_itself() {
        let f = WeatherField::filled(H, 6, 6, WeatherSample::from_rgba(0.4, 0.2, 0.1, 0.3));
        let out = advect_semi_lagrangian(&f, Vec2::new(1.7, -0.9), 1.0);
        // A uniform field is invariant under any backtrace (clamp included).
        let mut i = 0;
        while i < out.cells().len() {
            assert!((out.cells()[i].coverage - 0.4).abs() < TOL);
            i += 1;
        }
    }

    #[test]
    fn uniform_wind_advection_conserves_mass() {
        let f = interior_bump(24);
        let before = f.total_coverage();
        // A fractional interior shift: every backtrace tap stays interior.
        let out = advect_semi_lagrangian(&f, Vec2::new(0.73, -0.41), 1.0);
        let after = out.total_coverage();
        let rel = (after - before).abs() / before;
        assert!(rel < 1e-3, "mass drift {rel} exceeds tolerance");
    }

    #[test]
    fn repeated_uniform_advection_stays_bounded() {
        let mut f = interior_bump(32);
        let before = f.total_coverage();
        // Ten small steps: diffusion smears the bump but total is bounded.
        let mut step = 0;
        while step < 10 {
            f = advect_semi_lagrangian(&f, Vec2::new(0.25, 0.35), 1.0);
            step += 1;
        }
        let after = f.total_coverage();
        let rel = (after - before).abs() / before;
        assert!(rel < 1e-2, "accumulated mass drift {rel} exceeds tolerance");
    }

    #[test]
    fn curl_wind_advection_is_near_conserving_and_divergence_free_check() {
        let f = interior_bump(32);
        let before = f.total_coverage();
        let wind = WindField::new(Vec2::new(1.0, 0.2), 0.5, 0.4);
        let out = advect_with_wind(&f, &wind, 1.0);
        let after = out.total_coverage();
        let rel = (after - before).abs() / before;
        assert!(
            rel < 2e-2,
            "curl advection mass drift {rel} exceeds tolerance"
        );
    }

    #[test]
    fn advection_is_deterministic() {
        let f = interior_bump(16);
        let a = advect_semi_lagrangian(&f, Vec2::new(0.6, -0.3), 0.7);
        let b = advect_semi_lagrangian(&f, Vec2::new(0.6, -0.3), 0.7);
        assert_eq!(a, b);
        let wind = WindField::from_polar(0.5, 0.8, 0.3);
        let wa = advect_with_wind(&f, &wind, 0.4);
        let wb = advect_with_wind(&f, &wind, 0.4);
        assert_eq!(wa, wb);
    }

    #[test]
    fn target_coverage_is_in_unit_range_and_ordered() {
        let states = [
            SkyState::Clear,
            SkyState::Fair,
            SkyState::Overcast,
            SkyState::Storm,
        ];
        let mut prev = -1.0_f32;
        for s in states {
            let c = target_coverage(s);
            assert!((0.0..=1.0).contains(&c), "target {c} out of range");
            assert!(
                c > prev,
                "target coverage must strictly increase with intensity"
            );
            prev = c;
        }
    }

    #[test]
    fn blend_state_endpoints_are_exact() {
        let (from, to) = (SkyState::Clear, SkyState::Storm);
        assert!((blend_state(from, to, 0.0) - target_coverage(from)).abs() < TOL);
        assert!((blend_state(from, to, 1.0) - target_coverage(to)).abs() < TOL);
    }

    #[test]
    fn blend_state_is_monotone_in_unit_range_no_jump() {
        // Rising transition: non-decreasing and in range across the sweep.
        let mut prev = blend_state(SkyState::Clear, SkyState::Storm, 0.0);
        let mut i = 1;
        while i <= 40 {
            let t = i as f32 / 40.0;
            let cur = blend_state(SkyState::Clear, SkyState::Storm, t);
            assert!((0.0..=1.0).contains(&cur), "blend {cur} out of range");
            assert!(cur >= prev - TOL, "rising blend must be monotone");
            prev = cur;
            i += 1;
        }
        // Falling transition: non-increasing and in range.
        let mut prev = blend_state(SkyState::Storm, SkyState::Clear, 0.0);
        let mut i = 1;
        while i <= 40 {
            let t = i as f32 / 40.0;
            let cur = blend_state(SkyState::Storm, SkyState::Clear, t);
            assert!((0.0..=1.0).contains(&cur));
            assert!(cur <= prev + TOL, "falling blend must be monotone");
            prev = cur;
            i += 1;
        }
    }

    #[test]
    fn blend_state_saturates_out_of_range_t() {
        let (from, to) = (SkyState::Fair, SkyState::Overcast);
        assert!((blend_state(from, to, -2.0) - target_coverage(from)).abs() < TOL);
        assert!((blend_state(from, to, 5.0) - target_coverage(to)).abs() < TOL);
    }

    #[test]
    fn state_machine_advances_and_dissipates_with_saturation() {
        assert_eq!(advance_state(SkyState::Clear), SkyState::Fair);
        assert_eq!(advance_state(SkyState::Fair), SkyState::Overcast);
        assert_eq!(advance_state(SkyState::Overcast), SkyState::Storm);
        assert_eq!(advance_state(SkyState::Storm), SkyState::Storm);
        assert_eq!(dissipate_state(SkyState::Storm), SkyState::Overcast);
        assert_eq!(dissipate_state(SkyState::Clear), SkyState::Clear);
    }

    #[test]
    fn relax_coverage_approaches_target_monotonically_in_range() {
        let target = OVERCAST_COVERAGE;
        let mut cov = CLEAR_COVERAGE;
        let mut prev = cov;
        let mut step = 0;
        while step < 50 {
            cov = relax_coverage(cov, target, 0.5, 0.2);
            assert!((0.0..=1.0).contains(&cov));
            assert!(
                cov >= prev - TOL,
                "relaxation must be monotone toward target"
            );
            assert!(cov <= target + TOL, "relaxation must not overshoot");
            prev = cov;
            step += 1;
        }
        assert!((cov - target).abs() < 1e-2, "should converge near target");
    }

    #[test]
    fn classify_precip_thresholds_are_deterministic() {
        // Below the trigger -> dry.
        let dry = WeatherSample::from_rgba(0.9, 0.5, PRECIP_TRIGGER - 0.01, 0.0);
        assert_eq!(
            classify_precip(&dry, CloudKind::Cumulonimbus, 10.0).kind,
            PrecipKind::None
        );
        // Exactly at the trigger -> precipitating.
        let edge = WeatherSample::from_rgba(0.9, 0.5, PRECIP_TRIGGER, 0.0);
        assert_ne!(
            classify_precip(&edge, CloudKind::Cumulonimbus, 10.0).kind,
            PrecipKind::None
        );
        // Determinism: identical inputs -> identical outputs.
        let a = classify_precip(&edge, CloudKind::Cumulonimbus, 10.0);
        let b = classify_precip(&edge, CloudKind::Cumulonimbus, 10.0);
        assert_eq!(a, b);
    }

    #[test]
    fn classify_precip_phase_boundary_is_snow_at_freezing() {
        let wet = WeatherSample::from_rgba(0.9, 0.5, 0.8, 0.0);
        assert_eq!(
            classify_precip(&wet, CloudKind::Cumulonimbus, 0.01).kind,
            PrecipKind::Rain
        );
        // Exactly freezing resolves to snow, deterministically.
        assert_eq!(
            classify_precip(&wet, CloudKind::Cumulonimbus, FREEZING_POINT_C).kind,
            PrecipKind::Snow
        );
        assert_eq!(
            classify_precip(&wet, CloudKind::Cumulonimbus, -3.0).kind,
            PrecipKind::Snow
        );
    }

    #[test]
    fn classify_precip_intensity_in_range_and_kind_gain() {
        let wet = WeatherSample::from_rgba(1.0, 0.5, 1.0, 0.0);
        let storm = classify_precip(&wet, CloudKind::Cumulonimbus, 5.0);
        let other = classify_precip(&wet, CloudKind::Cumulus, 5.0);
        assert!((0.0..=1.0).contains(&storm.intensity));
        assert!((0.0..=1.0).contains(&other.intensity));
        // The storm cloud precipitates harder than a fair-weather cumulus.
        assert!(storm.intensity > other.intensity);
        // None always carries zero intensity.
        assert_eq!(Precip::NONE.intensity, 0.0);
    }

    #[test]
    fn wind_field_normalizes_and_curl_is_divergence_free() {
        let w = WindField::new(Vec2::new(3.0, 4.0), 2.0, 0.5);
        assert!((w.direction().length() - 1.0).abs() < 1e-3);
        assert!((w.base_velocity().length() - 2.0).abs() < 1e-3);
        // Numerically verify near-zero divergence of the curl shape.
        let p = Vec2::new(3.2, -1.7);
        let h = 0.05;
        let vx_px = WindField::curl_shape(Vec2::new(p.x + h, p.y)).x;
        let vx_mx = WindField::curl_shape(Vec2::new(p.x - h, p.y)).x;
        let vy_py = WindField::curl_shape(Vec2::new(p.x, p.y + h)).y;
        let vy_my = WindField::curl_shape(Vec2::new(p.x, p.y - h)).y;
        let div = (vx_px - vx_mx) / (2.0 * h) + (vy_py - vy_my) / (2.0 * h);
        assert!(
            div.abs() < 1e-2,
            "curl field divergence {div} should vanish"
        );
    }
}
