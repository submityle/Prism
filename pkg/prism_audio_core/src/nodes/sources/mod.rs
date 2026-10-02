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
//!
//!
//! Every generator is real-time safe: `process` performs no allocation, no
//! locking, and no panics, and reproducible generators are fully deterministic
//! across platforms via [`bevy_math::ops`].

pub mod air_jet_flute;
pub mod additive_oscillator;
pub mod bell;
pub mod bowed_string;
pub mod conical_reed;
pub mod brass_lip_reed;
pub mod fm_operator;
pub mod fof_source;
pub mod granular_source;
pub mod helmholtz_resonator;
pub mod impulse_train;
pub mod karplus_strong;
pub mod membrane_drum;
pub mod noise;
pub mod oscillator;
pub mod plucked_body;
pub mod pwm_oscillator;
pub mod reed_woodwind;
pub mod sample_player;
pub mod struck_bar;
pub mod struck_plate;
pub mod supersaw;
pub mod wavetable_oscillator;

pub use air_jet_flute::{AirJetFluteNode, AirJetFluteParams};
pub use additive_oscillator::{AdditiveOscillatorNode, AdditiveOscillatorParams};
pub use bell::{BellNode, BellParams};
pub use bowed_string::{BowedStringNode, BowedStringParams};
pub use brass_lip_reed::{BrassLipReedNode, BrassLipReedParams};
pub use conical_reed::{ConicalReedNode, ConicalReedParams};
pub use fm_operator::{FmOperatorNode, FmOperatorParams};
pub use fof_source::{FofSourceNode, FofSourceParams, Formant, MAX_FOF_GRAINS, MAX_FORMANTS};
pub use granular_source::{GranularSourceNode, GranularSourceParams, MAX_GRAINS};
pub use helmholtz_resonator::{HelmholtzResonatorNode, HelmholtzResonatorParams};
pub use impulse_train::{ImpulseTrainNode, ImpulseTrainParams};
pub use karplus_strong::{KarplusStrongNode, KarplusStrongParams};
pub use membrane_drum::{MembraneDrumNode, MembraneDrumParams};
pub use noise::{NoiseColor, NoiseNode};
pub use oscillator::{OscillatorNode, Waveform};
pub use plucked_body::{PluckedBodyNode, PluckedBodyParams, NUM_BODY_MODES};
pub use pwm_oscillator::{PwmOscillatorNode, PwmOscillatorParams};
pub use reed_woodwind::{ReedWoodwindNode, ReedWoodwindParams};
pub use sample_player::{Interpolation, LoopMode, SamplePlayerNode};
pub use struck_bar::{StruckBarNode, StruckBarParams, NUM_MODES};
pub use struck_plate::{StruckPlateNode, StruckPlateParams};
pub use supersaw::{SupersawNode, SupersawParams};
pub use wavetable_oscillator::{WavetableOscillatorNode, WavetableOscillatorParams};
