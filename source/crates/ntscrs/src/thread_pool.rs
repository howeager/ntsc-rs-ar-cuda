pub struct ThreadPool {
    #[cfg(all(feature = "rayon", not(target_arch = "wasm32")))]
    pool: rayon_core::ThreadPool,
}

impl ThreadPool {
    pub fn new() -> Self {
        #[cfg(all(feature = "rayon", not(target_arch = "wasm32")))]
        {
            // On Windows debug builds, the stack overflows with the default stack size.
            //
            // Not sure how big the stack should actually be here. It *was* 2MB, but that was apparently overflowing in
            // *release* builds of the AE plugin (https://x.com/KazunaEdits/status/2050129514089648321).
            let mut pool = rayon_core::ThreadPoolBuilder::new().stack_size(4 * 1024 * 1024);

            // Use physical core count instead of logical core count. Hyperthreading seems to be ~20-25% slower, at least on
            // a Ryzen 7 7700X.
            if std::env::var("RAYON_NUM_THREADS").is_err() {
                pool = pool.num_threads(num_cpus::get_physical());
            }

            Self {
                pool: pool.build().unwrap(),
            }
        }

        #[cfg(any(not(feature = "rayon"), target_arch = "wasm32"))]
        {
            Self {}
        }
    }

    pub fn install<OP, R>(&self, op: OP) -> R
    where
        OP: FnOnce() -> R + Send,
        R: Send,
    {
        #[cfg(all(feature = "rayon", not(target_arch = "wasm32")))]
        {
            self.pool.install(op)
        }

        #[cfg(any(not(feature = "rayon"), target_arch = "wasm32"))]
        {
            // wasm-bindgen-rayon doesn't support custom thread pools
            // https://github.com/RReverser/wasm-bindgen-rayon/issues/18
            op()
        }
    }
}

// Number of elements per task, independent of the number of zipped planes.
#[cfg(feature = "rayon")]
const TARGET_GRANULARITY: usize = 2048;

pub struct ZipChunks<'a, const N: usize, T> {
    arrays: [&'a mut [T]; N],
    chunk_size: usize,
    start: usize,
}

impl<'a, const N: usize, T> ZipChunks<'a, N, T> {
    pub fn new(arrays: [&'a mut [T]; N], chunk_size: usize) -> Self {
        if let Some((first, rest)) = arrays.split_first() {
            for r in rest.iter() {
                assert_eq!(first.len(), r.len());
            }
        }

        Self {
            arrays,
            chunk_size,
            start: 0,
        }
    }

    pub fn len(&self) -> usize {
        match self.arrays.first() {
            Some(first) => first.len(),
            None => 0,
        }
    }

    pub fn seq_for_each(mut self, mut cb: impl FnMut(usize, [&mut [T]; N])) {
        let len = self.len();
        let num_chunks = len / self.chunk_size;
        let remainder = len % self.chunk_size;
        let mut base = 0;
        for i in 0..num_chunks {
            let chunks: [&mut [T]; N] = self
                .arrays
                .each_mut()
                .map(|s| &mut s[base..base + self.chunk_size]);
            cb(i + self.start, chunks);

            base += self.chunk_size;
        }
        if remainder != 0 {
            let chunks: [&mut [T]; N] = self
                .arrays
                .each_mut()
                .map(|s| &mut s[base..base + remainder]);
            cb(self.start + num_chunks, chunks);
        }
    }

    #[cfg(feature = "rayon")]
    fn split_at(mut self, chunk_idx: usize) -> (Self, Self) {
        let len = self.len();
        let split_point = (chunk_idx * self.chunk_size).min(len);

        // Store raw pointers before consuming self.arrays
        let ptrs: [*mut T; N] = self.arrays.each_mut().map(|s| s.as_mut_ptr());

        let left_halves = core::array::from_fn(|i: usize| unsafe {
            core::slice::from_raw_parts_mut(ptrs[i], split_point)
        });

        let right_halves = core::array::from_fn(|i: usize| unsafe {
            core::slice::from_raw_parts_mut(ptrs[i].add(split_point), len - split_point)
        });

        (
            Self {
                arrays: left_halves,
                chunk_size: self.chunk_size,
                start: self.start,
            },
            Self {
                arrays: right_halves,
                chunk_size: self.chunk_size,
                start: self.start + chunk_idx,
            },
        )
    }

    #[cfg(feature = "rayon")]
    fn par_for_each_inner<'b>(
        self,
        num_threads: usize,
        scope: &rayon_core::Scope<'b>,
        cb: &'b (impl Fn(usize, [&mut [T]; N]) + Send + Sync + Copy),
    ) where
        T: Send,
        'a: 'b,
    {
        // Keep enough work in each task to amortize scheduling, especially for
        // short planes and the small regions used by head switching/tracking.
        let num_work_units = (num_threads * 4)
            .min(self.len().div_ceil(TARGET_GRANULARITY))
            .max(1);

        // Number of chunks, including any partial chunk at the end
        let num_total_chunks = self.len().div_ceil(self.chunk_size);
        let chunks_per_thread = num_total_chunks / num_work_units;
        let remainder = num_total_chunks % num_work_units;

        let n = if chunks_per_thread == 0 {
            remainder
        } else {
            num_work_units
        };

        let mut rest = self;
        for i in 0..n.saturating_sub(1) {
            let mut num_chunks = chunks_per_thread;
            if i < remainder {
                num_chunks += 1;
            }

            if num_chunks == 0 {
                break;
            }

            let (head, tail) = rest.split_at(num_chunks);
            rest = tail;

            scope.spawn(move |_| {
                head.seq_for_each(cb);
            });
        }
        // Use the calling worker too, rather than queueing every task and then
        // immediately waiting for one of them to be stolen.
        rest.seq_for_each(cb);
    }

    pub fn par_for_each(self, cb: impl Fn(usize, [&mut [T]; N]) + Send + Sync + Copy)
    where
        T: Send,
    {
        #[cfg(feature = "rayon")]
        {
            let num_threads = rayon_core::current_num_threads();
            if num_threads == 1 || self.len() <= TARGET_GRANULARITY {
                self.seq_for_each(cb);
            } else {
                rayon_core::scope(|scope| {
                    self.par_for_each_inner(num_threads, scope, &cb);
                });
            }
        }

        #[cfg(not(feature = "rayon"))]
        {
            self.seq_for_each(cb);
        }
    }
}

#[cfg(feature = "rayon")]
pub use rayon_core::join;
#[cfg(not(feature = "rayon"))]
pub fn join<A, B, RA, RB>(oper_a: A, oper_b: B) -> (RA, RB)
where
    A: FnOnce() -> RA + Send,
    B: FnOnce() -> RB + Send,
    RA: Send,
    RB: Send,
{
    (oper_a(), oper_b())
}

#[cfg(all(test, feature = "rayon"))]
mod tests {
    use super::*;
    use alloc::{vec, vec::Vec};

    #[test]
    fn parallel_chunks_match_sequential_including_partial_chunks() {
        for threads in [1, 2, 4] {
            let pool = rayon_core::ThreadPoolBuilder::new()
                .num_threads(threads)
                .build()
                .unwrap();
            pool.install(|| {
                for len in [0, 1, 31, 2048, 2049, 16_384, 16_385, 65_537, 131_073] {
                    for chunk_size in [1, 7, 640, 4096, len + 1] {
                        let mut first = vec![usize::MAX; len];
                        let mut second = vec![0; len];
                        ZipChunks::new([first.as_mut_slice(), second.as_mut_slice()], chunk_size)
                            .par_for_each(|index, [a, b]| {
                                // Encode both the global chunk index and the position
                                // within it to catch dropped, duplicated or offset work.
                                for (offset, (a, b)) in a.iter_mut().zip(b).enumerate() {
                                    *a = index * chunk_size + offset;
                                    *b += 1;
                                }
                            });
                        assert_eq!(first, (0..len).collect::<Vec<_>>());
                        assert!(second.iter().all(|&count| count == 1));
                    }
                }
            });
        }
    }
}
