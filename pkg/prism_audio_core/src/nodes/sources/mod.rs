//! Signal-generating source nodes (0-input, N-output).
//!
//! Unlike the insert-style processors in [`super::effects`] and
//! [`super::dynamics`], every node here *originates* a signal rather than
//! transforming an upstream one, so it declares no input ports and writes its
//! output buffer from internal state. They are the primitive voices the
//! authoring and voice-management layers instantiate and route into the mix
//! graph.
//!
//! # Source catalogue
//!
//! - [`additive_oscillator::AdditiveOscillatorNode`] -- real-time additive
//!   (Fourier-series) oscillator: a bank of phase-locked harmonic sines whose
//!   per-partial gains are audio-rate controls, Nyquist-muted and sum-normalized
//!   so timbres can be morphed live rather than baked into a table.
//! - [`oscillator::OscillatorNode`] -- band-limited (`PolyBLEP`) sine/saw/square/
//!   triangle geometric oscillator selected by [`oscillator::Waveform`].
//! - [`pwm_oscillator::PwmOscillatorNode`] -- band-limited (`PolyBLEP`) pulse
//!   oscillator with a continuously smoothed, audio-rate duty cycle: the
//!   classic pulse-width-modulation animation. It reuses the oscillator's
//!   `PolyBLEP` edge correction but places the second (falling) edge at the
//!   variable width, generalizing the fixed 50%-duty square.
//! - [`impulse_train::ImpulseTrainNode`] -- band-limited impulse train
//!   (BLIT): an alias-free train of narrow spikes evaluated as the
//!   closed-form normalized Dirichlet kernel (periodic sinc), keeping every
//!   harmonic below Nyquist with equal weight. It is the raw excitation that
//!   integrates into alias-free saw/pulse voices and formant sources.
//! - [`karplus_strong::KarplusStrongNode`] -- extended Karplus-Strong
//!   plucked-string physical model: a noise burst recirculating through a tuned,
//!   damped feedback delay line with an allpass fractional-delay tuning filter.
//! - [`noise::NoiseNode`] -- deterministic white/pink/brown generator
//!   ([`noise::NoiseColor`]) built on a reproducible `xorshift64`/`SplitMix64`
//!   stream with Paul-Kellet pink shaping and a leaky-integrator brown filter.
//! - [`sample_player::SamplePlayerNode`] -- pitch/rate-resampling PCM player with
//!   [`sample_player::LoopMode`] loop points and selectable
//!   [`sample_player::Interpolation`] (linear / Catmull-Rom).
//! - [`wavetable_oscillator::WavetableOscillatorNode`] -- band-limited mipmap
//!   wavetable oscillator: an octave mipmap of additively synthesized tables
//!   (saw/square/triangle presets or an arbitrary harmonic spectrum) read with
//!   periodic Catmull-Rom interpolation.
//! - [`fm_operator::FmOperatorNode`] -- phase-modulation (DX7-style)
//!   operator: a sine core deflected by an optional modulation input and
//!   two-sample-averaged self-feedback, the primitive voice of FM
//!   synthesis algorithms assembled in the mix graph.
//! - [`supersaw::SupersawNode`] -- detuned saw-stack ("super saw"): seven
//!   `PolyBLEP` band-limited sawtooths spread around one fundamental by an
//!   equal-temperament detune control and blended center-vs-sides by a `mix`
//!   control, the lush JP-8000-style unison lead/pad voice.
//! - [`granular_source::GranularSourceNode`] -- Gabor grain-cloud synthesizer:
//!   a scheduler sprays overlapping Hann-windowed sine grains whose carrier
//!   pitch, lifetime, and stereo placement are randomized from a deterministic
//!   PRNG, with power-preserving level normalization. Unlike the capture-based
//!   granulator effect it synthesizes its grains from scratch, so it is a true
//!   zero-input source.
//! - [`fof_source::FofSourceNode`] -- formant-wave-function (FOF) voice
//!   synthesizer: a fundamental phase accumulator fires one damped-sine
//!   formant grain per active formant on every period, so the periodic
//!   triggering fixes the pitch while each grain's exponential decay and
//!   raised-cosine skirt shape an independent formant peak. Unlike the
//!   stochastic grain cloud of `granular_source` the schedule is
//!   deterministic and pitch-synchronous, the classic sung-vowel voice.
//! - [`bowed_string::BowedStringNode`] -- bowed-string digital-waveguide
//!   physical model: a pair of velocity-wave delay lines (bridge-side and
//!   nut-side) terminated by inverting reflections, driven every sample by the
//!   `McIntyre`-Schumacher-Woodhouse bow-friction nonlinearity so the string
//!   self-oscillates. Unlike the once-plucked `karplus_strong` it is
//!   continuously bowed, sustaining as long as the bow moves.
//! - [`reed_woodwind::ReedWoodwindNode`] -- single-reed woodwind
//!   (clarinet-family) digital-waveguide physical model: a pair of pressure-
//!   wave delay lines form a cylindrical bore closed at the mouthpiece by a
//!   nonlinear pressure-controlled reed valve and opened at the bell by a
//!   lossy inverting reflection. The single inversion per round trip resonates
//!   only the odd harmonics, the physical origin of the hollow clarinet timbre.
//!   Unlike the bow-driven `bowed_string` it is sustained by steady breath
//!   pressure through the reed rather than bow friction.
//! - [`air_jet_flute::AirJetFluteNode`] -- air-jet flute (concert-flute /
//!   recorder family) digital-waveguide physical model: two cross-coupled
//!   pressure-wave delay lines form an open-open cylindrical bore, plus a jet
//!   convective-delay line whose cubic edge-tone deflection `jet = JET_DRIVE *
//!   (x - x^3)` pumps the bore. The two inverting end reflections cancel per
//!   round trip, so unlike the odd-only `reed_woodwind` it resonates the full
//!   harmonic series an octave higher; deterministic breath turbulence breaks
//!   the symmetry to start the tone. Driven by an air jet rather than the
//!   `bowed_string` bow or `karplus_strong` pluck.
//! - [`brass_lip_reed::BrassLipReedNode`] -- brass lip-reed (trumpet /
//!   trombone / horn family) digital-waveguide physical model: two
//!   cross-coupled pressure-wave delay lines form an open-open flaring bore
//!   whose two inverting end reflections cancel per round trip, so like
//!   `air_jet_flute` it sounds the full harmonic series. It is excited by an
//!   outward-striking lip valve -- a damped second-order resonator driven by
//!   the high-passed pressure drop (`dp - dp_z1`, zero at DC) so a steady
//!   breath cannot rail the aperture -- whose mechanical resonance, tuned by
//!   `lip_tension`, selects which bore partial sounds (the physics of bugle
//!   calls and lip slurs). A clamped-linear lip flow self-limits the limit
//!   cycle. Unlike the odd-only inward reed of `reed_woodwind`, the air jet of
//!   `air_jet_flute`, or the bow of `bowed_string`, the tunable lip resonance
//!   picking the partial is unique to this brass voice.
//! - [`conical_reed::ConicalReedNode`] -- conical double-reed (oboe /
//!   bassoon / saxophone family) digital-waveguide physical model: two
//!   cross-coupled pressure-wave delay lines form a bore that is driven at
//!   the mouthpiece by the same inward-striking nonlinear reed valve as
//!   `reed_woodwind`, but whose far end is a cone rather than a cylinder. The
//!   conical apex is modelled as a *high-pass* spherical-wave reflection
//!   (zero at DC, corner tracking `f0 / APEX_CORNER_RATIO`) that suppresses
//!   the sub-fundamental relaxation mode, so the net in-phase round trip
//!   resonates the full harmonic series like a cone-equivalent open pipe --
//!   unlike the odd-only cylindrical `reed_woodwind` it sounds every partial
//!   (the reedy oboe timbre). A `tanh` soft-limiter plus a DC blocker bound
//!   the limit cycle. Distinct from the full-harmonic air jet of
//!   `air_jet_flute` and the lip valve of `brass_lip_reed`, this voice is a
//!   reed-driven cone.
//! - [`struck_bar::StruckBarNode`] -- mallet-struck modal percussion source
//!   (marimba / vibraphone / glockenspiel / tubular-bell family): a unit-area
//!   raised-cosine contact pulse excites a parallel bank of [`NUM_MODES`]
//!   decaying two-pole resonators tuned to the transverse bending partials of
//!   a stiff bar. The `inharmonicity` control blends the mode ratios between a
//!   tuned mallet set (`1 : 4 : 10 : ...`) and the ideal Euler-Bernoulli
//!   free-free set (`1 : 2.756 : 5.404 : ...`), spanning the wooden-to-metallic
//!   continuum, while `brightness` sets both pulse width and mode-gain rolloff
//!   (hard vs soft mallet). Unlike the `modal_resonator` effect, which filters
//!   an external input, this voice carries its own excitation and is struck at
//!   construction; unlike the one-dimensional waveguide voices
//!   (`karplus_strong`, `bowed_string`, `reed_woodwind`) its partials are
//!   deliberately inharmonic.
//! - [`membrane_drum::MembraneDrumNode`] -- struck circular-membrane modal
//!   percussion source (timpani / tom / tabla / frame-drum family): the
//!   two-dimensional counterpart of `struck_bar`. A unit-area raised-cosine
//!   contact pulse drives a parallel bank of [`membrane_drum::NUM_MODES`]
//!   decaying two-pole resonators tuned to the Bessel-zero vibration modes of
//!   an ideal drumhead (`1 : 1.593 : 2.136 : ...`). The `inharmonicity` control
//!   blends those ratios toward the near-harmonic air-loaded kettledrum set
//!   (`1 : 1.5 : 2 : ...`) so one node spans the pitchless-tom-to-tuned-timpani
//!   continuum, while `strike_position` weights each mode by the membrane
//!   shape `|J_m(alpha_mn * r)|`: a centre strike excites only the deep
//!   axisymmetric modes (a round "boom"), an edge strike lights the high modes
//!   (a bright "slap"). Unlike the `modal_resonator` effect it carries its own
//!   excitation and is struck at construction.
//! - [`helmholtz_resonator::HelmholtzResonatorNode`] -- blown-bottle Helmholtz
//!   resonator (bottle / ocarina / vessel-flute family): a breath-driven
//!   *lumped* resonance rather than a distributed waveguide. A single
//!   Chamberlin state-variable filter models the vessel's one Helmholtz mode,
//!   and a Van der Pol negative resistance (the edge-tone pump, scaled by
//!   `breath_pressure`) sustains it into a near-sinusoidal limit cycle whose
//!   cubic saturation keeps it bounded. Because the pitch is the lumped
//!   resonance it is pinned by the vessel and does not bend with breath,
//!   unlike the delay-line pitch of `air_jet_flute`; `brightness` adds odd
//!   harmonics through an out-of-loop `tanh` shaper and `breath_noise` mixes
//!   in airy turbulence. The vessel only speaks once the breath exceeds a
//!   `resonance`-dependent threshold.
//! - [`plucked_body::PluckedBodyNode`] -- plucked acoustic-string source
//!   coupled to a parallel instrument-body modal bank (guitar / lute family).
//!   The extended Karplus-Strong string loop of `karplus_strong` (integer
//!   delay line, one-zero brightness filter, allpass sub-sample tuning, and
//!   `60 dB` loop gain) drives a bank of [`plucked_body::NUM_BODY_MODES`]
//!   two-pole resonators tuned to the documented air-cavity and plate
//!   resonances of a soundbox. Each body mode is normalized to a fixed
//!   resonant gain so the box adds sub-fundamental bloom and plate formants
//!   without ringing up; `body_level` crossfades raw string to body-filtered
//!   and `body_size` divides every body-mode frequency (a larger box
//!   resonates lower). Placing the body at the output is, by linearity, the
//!   commuted-synthesis equivalent of pre-convolving the body impulse
//!   response into the pluck. Unlike the bare `karplus_strong` it carries a
//!   built-in body; unlike the `modal_resonator` effect it is self-excited;
//!   unlike the struck modal sources `struck_bar` / `membrane_drum` the modal
//!   bank is a passive body rather than the sounding object itself.
//! - [`struck_plate::StruckPlateNode`] -- mallet-struck two-dimensional
//!   plate modal percussion source (gong / plate-bell / metal-sheet family).
//!   A unit-area raised-cosine contact pulse excites a parallel bank of
//!   [`struck_plate::NUM_MODES`] decaying two-pole resonators tuned to the
//!   lowest simply-supported Kirchhoff thin-plate partials, whose *paired*
//!   `(i, j)` index gives a far denser, beating grid than the single series
//!   of `struck_bar`. The `aspect_ratio` control stretches the plate from a
//!   square (degenerate pairs) to an oblong sheet (split, shimmering
//!   doublets), while `brightness` sets both pulse width and mode-gain
//!   rolloff. Unlike the tension-restored, fast-decaying circular membrane
//!   of `membrane_drum`, the plate is stiffness-restored and rings on; like
//!   the sibling struck sources it carries its own excitation.
//! - [`bell::BellNode`] -- clapper-struck church / cast-bell modal
//!   percussion source (bell / carillon / bell-plate family). A unit-area
//!   raised-cosine contact pulse excites a parallel bank of
//!   [`bell::NUM_MODES`] decaying two-pole resonators tuned to the named
//!   church-bell partials -- hum (`0.5`), prime (`1.0`), **minor-third**
//!   tierce (`1.2`), quint (`1.5`), nominal (`2.0`), and the fast-fading
//!   upper shell modes -- the strongly inharmonic three-dimensional-shell set
//!   that no bar or plate series reproduces. Each partial is voiced as a
//!   near-degenerate `warble` doublet that beats like a real casting, the low
//!   partials ring far longer than the high ones (the hum drones on), and
//!   `brightness` sets both the clapper pulse width and the upper-mode tilt.
//!   Unlike the sparse one-dimensional `struck_bar` beam series or the dense
//!   two-dimensional `struck_plate` grid, it rings at the bell's defining
//!   sub-octave hum and minor-third tierce.
//! - [`blown_pipe::BlownPipeNode`] -- a noise-excited resonant-pipe wind
//!   voice (pan pipe / flue organ pipe / whistle / ocarina). Turbulent breath
//!   noise is shaped by a bank of peak-normalized two-pole resonators tuned to
//!   an air column: open-open pipes ring the full harmonic series, stopped
//!   pipes only the odd harmonics. Unlike the four self-oscillating waveguide
//!   winds it is linear and passive, never closing a nonlinear feedback loop,
//!   and unlike the struck modal sources it is driven continuously so it
//!   sustains instead of decaying.
//! - [`shepard_tone::ShepardToneNode`] -- Shepard/Risset endless-glissando
//!   source: [`shepard_tone::SPAN_OCTAVES`] octave-spaced sine partials whose
//!   shared logarithmic position drifts up or down the frequency axis under a
//!   fixed position-indexed raised-cosine (Hann) envelope that fades partials
//!   in at the bottom and out at the top, manufacturing a tone that seems to
//!   rise (or fall) forever. The summed envelope is exactly constant, so the
//!   illusion is loudness-stable. Unlike `additive_oscillator`'s static
//!   integer-harmonic series the partials are octave-spaced (geometric) and
//!   drift continuously; `speed == 0` degenerates to a static Shepard chord.
//! - [`tonewheel_organ::TonewheelOrganNode`] -- electromechanical drawbar
//!   organ (Hammond-family): [`tonewheel_organ::NUM_DRAWBARS`] fixed musical
//!   footages above the played key, each its own phase accumulator running at
//!   the historically *tempered* gear ratio (notably the sharp seventeenth
//!   ~5.04x), summed under per-drawbar smoothed levels with the octave
//!   *foldback* a finite tonewheel generator imposes on the brightest
//!   footages. Unlike `additive_oscillator`'s phase-locked integer harmonics
//!   it has a sub-octave footage, non-integer tempered ratios, octave
//!   foldback, and partials that gently beat; it pairs naturally with the
//!   `leslie` rotary cabinet.
//! - [`glottal_pulse::GlottalPulseNode`] -- Rosenberg-model glottal pulse
//!   source: the periodic vocal-fold airflow that drives the vocal tract in
//!   the classic source-filter model of speech. Within each period the
//!   glottis opens, peaks, and closes under the open quotient (fraction of
//!   the period that is open) and speed quotient (how much faster it closes
//!   than it opens, which sets the brightness); the single step at the
//!   glottal-closure instant is band-limited with the shared `PolyBLEP`.
//!   [`glottal_pulse::GlottalOutput`] emits either the DC-free flow
//!   derivative (the default excitation) or the raw unipolar flow. Unlike
//!   `fof_source`, which synthesizes an already-*filtered* vowel grain per
//!   period (source and tract fused), this node is only the *unfiltered*
//!   source and pairs with the `formant_filter`; unlike `impulse_train`'s
//!   ideal equal-amplitude harmonic spikes and `oscillator`'s geometric
//!   saw/square it is a physiologically shaped volume-velocity waveform.
//! - [`tine_electric_piano::TineElectricPianoNode`] -- struck tine
//!   electric-piano voice (Rhodes/Wurlitzer family): a felt hammer strikes a
//!   stiff *clamped-free* (cantilever) steel tine whose inharmonic bending
//!   modes sit at `1 : 6.267 : 17.55` (a different boundary condition from the
//!   *free-free* bar of [`struck_bar`], so the bright partials ring far
//!   higher), plus a slightly detuned tonebar partner that beats against the
//!   fundamental for the sustained shimmer. Its defining feature is the
//!   nonlinear electromagnetic *pickup*: the tine displacement is read through
//!   the bounded rational transfer `V(x) = (x + asymmetry*x^2) / (1 + (x/sat)^2)`,
//!   whose even-harmonic term is the "growl" and whose saturating denominator
//!   is the velocity-dependent "bark", so a harder strike swings the tine
//!   further into the nonlinearity and sounds dirtier. Unlike the
//!   [`super::effects::modal_resonator`] *effect* it supplies its own hammer
//!   excitation, and unlike the near-harmonic waveguide voices
//!   [`karplus_strong`]/[`bowed_string`] or the steady [`additive_oscillator`]
//!   its spectrum is struck, inharmonic, and shaped by the pickup.
//! - [`swept_sine::SweptSineNode`] -- swept-sine (chirp) *measurement*
//!   source: a single sine whose frequency glides from a start to an end
//!   frequency over a fixed duration, then falls silent. The frequency law
//!   is one of two [`swept_sine::SweepMode`]s -- the logarithmic
//!   exponential sine sweep (ESS, equal time per octave) used for
//!   impulse-response measurement because its harmonic-distortion orders
//!   deconvolve into separable pre-sweep echoes, or the constant-slope
//!   linear chirp (equal time per hertz, the radar/sonar excitation). Both
//!   are driven by a per-sample recurrence (`f *= k` or `f += step` with
//!   phase accumulation) so the progression stays exact without a logarithm
//!   on the hot path, and a raised-cosine edge taper suppresses the start/
//!   stop click. It pairs with the [`super::reverb::convolver::Convolver`]
//!   to perform a system impulse-response measurement, and unlike the
//!   steady [`oscillator`]/[`additive_oscillator`] tones, the perpetual
//!   octave illusion of [`shepard_tone`], or a [`noise`] excitation it is a
//!   deterministic, phase-coherent, one-shot glissando across the band.
//! - [`hard_sync_oscillator::HardSyncOscillatorNode`] -- hard-sync
//!   sawtooth: a fast slave sawtooth whose phase is forcibly reset to zero
//!   each time a slower master oscillator wraps. The perceived pitch locks to
//!   the master while the slave's higher frequency carves a formant into the
//!   spectrum, and sweeping the slave/master ratio glides that sync formant
//!   through the harmonic series for the aggressive, vocal "tearing" lead.
//!   The slave's own wrap is rounded with the shared two-sided `PolyBLEP`,
//!   while the forced reset -- which lands at a fractional sample position --
//!   recomputes the reset sample from the post-reset ramp and rounds the
//!   step with a one-sided (two-point) `PolyBLEP` residual. Unlike the steady
//!   single [`oscillator`] sawtooth, the detuned de-synchronized stack of
//!   [`supersaw`], or the variable-duty [`pwm_oscillator`], it keeps one
//!   slave edge but re-locks it to the master every cycle.
//!
//! - [`phase_distortion_oscillator::PhaseDistortionOscillatorNode`] -- the
//!   classic (non-resonant) phase-distortion oscillator. A linear phase ramp
//!   is bent through a two-segment time warp around a break point, then read
//!   out of a plain cosine, so advancing time faster through one half-cycle
//!   compresses that lobe and injects progressively brighter harmonics as the
//!   `amount` control slides the break point. At `amount == 0` the warp is the
//!   identity and the output is a mathematically pure sine; as it opens the
//!   tone brightens continuously and click-free. Because the warp knee and the
//!   phase wrap both land where the cosine (and its first derivative) are
//!   zero, the waveform stays C1-continuous -- only the curvature breaks -- so
//!   its aliasing rolls off steeply enough (~18 dB/octave) to need no
//!   band-limiting primitive across the musical range. Unlike the discrete
//!   waveforms of [`oscillator`], the detuned stack of [`supersaw`], the
//!   moving step edge of [`pwm_oscillator`], or the forced reset of
//!   [`hard_sync_oscillator`], it recolours a single tone purely by warping
//!   time.
//! - [`dsf_oscillator::DsfOscillatorNode`] -- discrete summation formula
//!   (DSF) oscillator: a closed-form evaluation of a geometric harmonic
//!   series, so a single sine numerator term plus two correction terms
//!   synthesize an entire `1, a, a^2, ...` stack of partials in O(1) per
//!   sample instead of summing them one by one. A `brightness` control sets
//!   the geometric ratio `a` (spectral tilt): at `brightness == 0` the
//!   series collapses to a mathematically pure sine, and as it opens the
//!   upper partials fill in and the tone brightens. The partial count is
//!   recomputed each block from the fundamental so no partial ever crosses
//!   Nyquist, making it naturally band-limited without an edge-correction
//!   primitive. Unlike the flat-spectrum Dirichlet train of
//!   [`impulse_train`] (its `a == 1` special case), the arbitrary O(N)
//!   partial bank of [`additive_oscillator`], the fixed `PolyBLEP` waveforms
//!   of [`oscillator`], or the time warp of [`phase_distortion_oscillator`],
//!   it sculpts a smoothly decaying harmonic series in constant time.
//!
//!
//! Every generator is real-time safe: `process` performs no allocation, no
//! locking, and no panics, and reproducible generators are fully deterministic
//! across platforms via [`bevy_math::ops`].

pub mod air_jet_flute;
pub mod additive_oscillator;
pub mod bell;
pub mod blown_pipe;
pub mod bowed_string;
pub mod conical_reed;
pub mod brass_lip_reed;
pub mod dsf_oscillator;
pub mod fm_operator;
pub mod fof_source;
pub mod glottal_pulse;
pub mod granular_source;
pub mod hard_sync_oscillator;
pub mod helmholtz_resonator;
pub mod impulse_train;
pub mod karplus_strong;
pub mod membrane_drum;
pub mod noise;
pub mod oscillator;
pub mod phase_distortion_oscillator;
pub mod plucked_body;
pub mod pwm_oscillator;
pub mod reed_woodwind;
pub mod sample_player;
pub mod shepard_tone;
pub mod tonewheel_organ;
pub mod struck_bar;
pub mod struck_plate;
pub mod supersaw;
pub mod swept_sine;
pub mod tine_electric_piano;
pub mod wavetable_oscillator;

pub use air_jet_flute::{AirJetFluteNode, AirJetFluteParams};
pub use additive_oscillator::{AdditiveOscillatorNode, AdditiveOscillatorParams};
pub use bell::{BellNode, BellParams};
pub use blown_pipe::{BlownPipeNode, BlownPipeParams, NUM_HARMONICS};
pub use bowed_string::{BowedStringNode, BowedStringParams};
pub use brass_lip_reed::{BrassLipReedNode, BrassLipReedParams};
pub use conical_reed::{ConicalReedNode, ConicalReedParams};
pub use dsf_oscillator::{DsfOscillatorNode, DsfOscillatorParams};
pub use fm_operator::{FmOperatorNode, FmOperatorParams};
pub use fof_source::{FofSourceNode, FofSourceParams, Formant, MAX_FOF_GRAINS, MAX_FORMANTS};
pub use glottal_pulse::{GlottalOutput, GlottalPulseNode, GlottalPulseParams};
pub use granular_source::{GranularSourceNode, GranularSourceParams, MAX_GRAINS};
pub use hard_sync_oscillator::{HardSyncOscillatorNode, HardSyncOscillatorParams};
pub use helmholtz_resonator::{HelmholtzResonatorNode, HelmholtzResonatorParams};
pub use impulse_train::{ImpulseTrainNode, ImpulseTrainParams};
pub use karplus_strong::{KarplusStrongNode, KarplusStrongParams};
pub use membrane_drum::{MembraneDrumNode, MembraneDrumParams};
pub use noise::{NoiseColor, NoiseNode};
pub use oscillator::{OscillatorNode, Waveform};
pub use phase_distortion_oscillator::{PhaseDistortionOscillatorNode, PhaseDistortionOscillatorParams};
pub use plucked_body::{PluckedBodyNode, PluckedBodyParams, NUM_BODY_MODES};
pub use pwm_oscillator::{PwmOscillatorNode, PwmOscillatorParams};
pub use reed_woodwind::{ReedWoodwindNode, ReedWoodwindParams};
pub use sample_player::{Interpolation, LoopMode, SamplePlayerNode};
pub use shepard_tone::{
    ShepardToneNode, ShepardToneParams, DEFAULT_AMPLITUDE, DEFAULT_BASE_HZ, DEFAULT_SPEED,
    MAX_BASE_HZ, MAX_SPEED, MIN_BASE_HZ, MIN_SPEED, NYQUIST_GUARD, OUTPUT_GAIN, SPAN_OCTAVES,
};
pub use tonewheel_organ::{TonewheelOrganNode, TonewheelOrganParams, NUM_DRAWBARS};
pub use struck_bar::{StruckBarNode, StruckBarParams, NUM_MODES};
pub use struck_plate::{StruckPlateNode, StruckPlateParams};
pub use supersaw::{SupersawNode, SupersawParams};
pub use swept_sine::{SweepMode, SweptSineNode, SweptSineParams};
pub use tine_electric_piano::{TineElectricPianoNode, TineElectricPianoParams, NUM_TINE_MODES};
pub use wavetable_oscillator::{WavetableOscillatorNode, WavetableOscillatorParams};
