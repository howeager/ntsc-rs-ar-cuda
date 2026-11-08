//! Adapted from Clatter (<https://github.com/Ralith/clatter/>)
//! which is available under MIT/Apache-2.0/zlib

use core::f32;

use fearless_simd::prelude::*;

/// A distribution of points across a space
pub trait Grid<const DIMENSION: usize> {
    /// Array containing a `T` per dimension per vertex
    type VertexArray<T>: AsRef<[[T; DIMENSION]]>;

    /// Compute integer coordinates of and vectors to `point` from each vertex in the cell enclosing
    /// `point`
    fn get<S: Simd>(
        &self,
        point: [S::f32s; DIMENSION],
    ) -> (Self::VertexArray<S::i32s>, Self::VertexArray<S::f32s>);
}

/// A regular grid of simplices, the simplest possible polytope in a given dimension
pub struct Simplex;

impl Grid<1> for Simplex {
    type VertexArray<T> = [[T; 1]; 2];

    #[inline(always)]
    fn get<S: Simd>(&self, [x]: [S::f32s; 1]) -> ([[S::i32s; 1]; 2], [[S::f32s; 1]; 2]) {
        let i = x.floor();
        let i0 = i.to_int();
        let i1 = i0 + 1;
        let x0 = x - i;
        let x1 = x0 - 1.0;
        ([[i0], [i1]], [[x0], [x1]])
    }
}

impl Grid<2> for Simplex {
    type VertexArray<T> = [[T; 2]; 3];

    #[inline(always)]
    fn get<S: Simd>(&self, [x, y]: [S::f32s; 2]) -> ([[S::i32s; 2]; 3], [[S::f32s; 2]; 3]) {
        const SKEW: f32 = skew_factor(2);
        const UNSKEW: f32 = -unskew_factor(2);

        // Skew to distort simplexes with side length sqrt(2)/sqrt(3) until they make up
        // squares
        let s = (x + y) * SKEW;
        let ips = (x + s).floor();
        let jps = (y + s).floor();

        // Integer coordinates for the base vertex of the triangle
        let i = ips.to_int();
        let j = jps.to_int();

        let t = S::f32s::float_from(i + j) * UNSKEW;

        // Unskewed distances to the first point of the enclosing simplex
        let x0: S::f32s = x - (ips - t);
        let y0: S::f32s = y - (jps - t);

        // Offsets of the second (middle) vertex of the enclosing simplex: (1, 0)
        // for the lower triangle, (0, 1) for the upper one
        let simd = x.witness();
        let one = S::i32s::splat(simd, 1);
        let zero = S::i32s::splat(simd, 0);
        let i1 = x0.simd_ge(y0).select(one, zero);
        let j1 = y0.simd_gt(x0).select(one, zero);

        // Distances to the second and third points of the enclosing simplex
        let x1 = x0 - S::f32s::float_from(i1) + UNSKEW;
        let y1 = y0 - S::f32s::float_from(j1) + UNSKEW;
        let x2 = x0 - 1.0 + 2.0 * UNSKEW;
        let y2 = y0 - 1.0 + 2.0 * UNSKEW;

        (
            [[i, j], [i + i1, j + j1], [i + 1, j + 1]],
            [[x0, y0], [x1, y1], [x2, y2]],
        )
    }
}

#[inline(always)]
pub const fn skew_factor(dimension: usize) -> f32 {
    (SQRT_TABLE[dimension + 1] - 1.0) / dimension as f32
}

#[inline(always)]
pub const fn unskew_factor(dimension: usize) -> f32 {
    ((1.0 / SQRT_TABLE[dimension + 1]) - 1.0) / dimension as f32
}

const SQRT_TABLE: [f32; 4] = [0.0, 1.0, core::f32::consts::SQRT_2, 1.732_050_8_f32];
