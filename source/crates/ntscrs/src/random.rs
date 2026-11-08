use core::ops::{Add, Mul, Range, Sub};

#[cfg(not(feature = "std"))]
use core_maths::CoreFloat as _;

/// Lambda parameter for generating a geometric distribution where each trial has probability `p`.
pub fn geometric_lambda(p: f64) -> f64 {
    if p <= 0.0 || p > 1.0 {
        panic!("Invalid probability: {p}");
    }
    (1.0 - p).ln()
}

/// A simple and fast RNG based on the MurmurHash finalizer. It's easy to "mix" additional random seeds into the RNG
/// state, allowing for it to be forked and parallelized.
#[derive(Clone)]
pub struct SplitMix64 {
    state: u64,
}

const PHI: u64 = 0x9e3779b97f4a7c15;

// Adapted from rand_xoshiro
// https://docs.rs/rand_xoshiro/0.8.1/src/rand_xoshiro/splitmix64.rs.html
impl SplitMix64 {
    pub fn random<T: FromState>(&mut self) -> T {
        self.state = self.state.wrapping_add(PHI);
        T::finalize(self.state)
    }

    /// Uniform float in `[low, high)`.
    #[inline]
    pub fn random_range<T: Rangeable>(&mut self, range: Range<T>) -> T {
        range.start + self.random::<T>() * (range.end - range.start)
    }

    pub fn random_geometric(&mut self, lambda: f64) -> usize {
        // We can simulate a geometric distribution by taking the floor of an exponential distribution
        // https://en.wikipedia.org/wiki/Geometric_distribution#Related_distributions
        (self.random::<f64>().ln() / lambda) as usize
    }
}

pub trait Rangeable:
    Add<Output = Self> + Sub<Output = Self> + Mul<Output = Self> + Copy + FromState
{
}
impl Rangeable for f32 {}
impl Rangeable for f64 {}

/// Trait for running the MurmurHash finalizer function over the SplitMix64 state.
pub trait FromState {
    fn finalize(state: u64) -> Self;
}

impl FromState for u32 {
    fn finalize(mut state: u64) -> Self {
        // David Stafford's
        // (http://zimbry.blogspot.com/2011/09/better-bit-mixing-improving-on.html)
        // "Mix4" variant of the 64-bit finalizer in Austin Appleby's
        // MurmurHash3 algorithm.
        state = (state ^ (state >> 33)).wrapping_mul(0x62A9D9ED799705F5);
        state = (state ^ (state >> 28)).wrapping_mul(0xCB24D0A5C88C35B3);
        (state >> 32) as u32
    }
}

impl FromState for i32 {
    fn finalize(state: u64) -> Self {
        u32::finalize(state) as i32
    }
}

impl FromState for u64 {
    fn finalize(mut state: u64) -> Self {
        state = (state ^ (state >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
        state = (state ^ (state >> 27)).wrapping_mul(0x94d049bb133111eb);
        state ^ (state >> 31)
    }
}

impl FromState for i64 {
    fn finalize(state: u64) -> Self {
        u64::finalize(state) as i64
    }
}

impl FromState for f32 {
    fn finalize(state: u64) -> Self {
        (u32::finalize(state) >> 8) as f32 * (1.0 / (1u32 << 24) as f32)
    }
}

impl FromState for f64 {
    fn finalize(state: u64) -> Self {
        (u64::finalize(state) >> 11) as f64 * (1.0 / (1u64 << 53) as f64)
    }
}

impl SplitMix64 {
    pub fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    /// Mix an additional seed into the RNG.
    pub fn mix(mut self, input: u64) -> Self {
        self.state = self.state.wrapping_add(input);
        Self {
            state: self.random(),
        }
    }
}
