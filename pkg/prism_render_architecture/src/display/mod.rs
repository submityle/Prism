//! Scene-linear to display-output pipeline.

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum DisplayOutput {
    SdrSrgb,
    ScRgb,
    Hdr10Pq,
}

#[derive(Clone, Copy, Debug)]
pub struct DisplaySettings {
    pub output: DisplayOutput,
    pub paper_white_nits: f32,
    pub peak_nits: f32,
    pub local_exposure: bool,
}
