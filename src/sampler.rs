pub struct SamplingParams {
    pub temperature: f32,
    pub top_k: i32,
    pub top_p: f32,
    pub ignore_eos: bool,
}

impl Default for SamplingParams {
    fn default() -> Self {
        Self {
            temperature: 0.0,
            top_k: -1,
            top_p: 1.0,
            ignore_eos: false,
        }
    }
}

impl SamplingParams {
    pub(crate) fn is_greedy(&self) -> bool {
        (self.temperature <= 0.0 || self.top_k == 1) && self.top_p >= 1.0
    }
}
