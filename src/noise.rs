use core::simd::prelude::*;
use core::simd::{f32x8, i32x8, u32x8};
use std::simd::StdFloat;

#[derive(Clone)]
pub struct PerlinSimd {
    p: [u32; 512],
}

impl PerlinSimd {
    pub fn new(seed: u32) -> Self {
        let mut p = [0u32; 512];
        let mut permutation: Vec<u32> = (0..256).collect();

        // Basic LCG for shuffling (reproducible)
        let mut state = seed as u64;
        for i in (1..256).rev() {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
            let j = (state % (i as u64 + 1)) as usize;
            permutation.swap(i, j);
        }

        for i in 0..256 {
            p[i] = permutation[i];
            p[i + 256] = permutation[i];
        }

        Self { p }
    }

    #[inline(always)]
    fn fade(t: f32x8) -> f32x8 {
        t * t * t * (t * (t * f32x8::splat(6.0) - f32x8::splat(15.0)) + f32x8::splat(10.0))
    }

    #[inline(always)]
    fn lerp(t: f32x8, a: f32x8, b: f32x8) -> f32x8 {
        a + t * (b - a)
    }

    #[inline(always)]
    fn grad(hash: u32x8, x: f32x8, y: f32x8, z: f32x8) -> f32x8 {
        let h = hash & u32x8::splat(15);

        // Selection logic: mask.select(if_true, if_false)
        let u = h.simd_lt(u32x8::splat(8)).select(x, y);

        let v_mask_4 = h.simd_lt(u32x8::splat(4));
        let v_mask_12 = h.simd_eq(u32x8::splat(12));
        let v_mask_14 = h.simd_eq(u32x8::splat(14));

        // Use | for OR logic on masks
        let v = v_mask_4.select(y, (v_mask_12 | v_mask_14).select(x, z));

        let part1 = (h & u32x8::splat(1)).simd_eq(u32x8::splat(0)).select(u, -u);
        let part2 = (h & u32x8::splat(2)).simd_eq(u32x8::splat(0)).select(v, -v);
        part1 + part2
    }

    pub fn get_3d(&self, x: f32x8, y: f32x8, z: f32x8) -> f32x8 {
        let x_floor = x.floor();
        let y_floor = y.floor();
        let z_floor = z.floor();

        // Cast to i32 for bitwise AND, then to u32 for indexing
        let xi = (x_floor.cast::<i32>() & i32x8::splat(255)).cast::<u32>();
        let yi = (y_floor.cast::<i32>() & i32x8::splat(255)).cast::<u32>();
        let zi = (z_floor.cast::<i32>() & i32x8::splat(255)).cast::<u32>();

        let xf = x - x_floor;
        let yf = y - y_floor;
        let zf = z - z_floor;

        let u = Self::fade(xf);
        let v = Self::fade(yf);
        let w = Self::fade(zf);

        let p = &self.p;
        // Gather values from the permutation table
        let fetch = |idx: u32x8| -> u32x8 { u32x8::gather_or_default(p, idx.cast::<usize>()) };

        let a = fetch(xi) + yi;
        let aa = fetch(a) + zi;
        let ab = fetch(a + u32x8::splat(1)) + zi;
        let b = fetch(xi + u32x8::splat(1)) + yi;
        let ba = fetch(b) + zi;
        let bb = fetch(b + u32x8::splat(1)) + zi;

        Self::lerp(
            w,
            Self::lerp(
                v,
                Self::lerp(
                    u,
                    Self::grad(fetch(aa), xf, yf, zf),
                    Self::grad(fetch(ba), xf - f32x8::splat(1.0), yf, zf),
                ),
                Self::lerp(
                    u,
                    Self::grad(fetch(ab), xf, yf - f32x8::splat(1.0), zf),
                    Self::grad(
                        fetch(bb),
                        xf - f32x8::splat(1.0),
                        yf - f32x8::splat(1.0),
                        zf,
                    ),
                ),
            ),
            Self::lerp(
                v,
                Self::lerp(
                    u,
                    Self::grad(fetch(aa + u32x8::splat(1)), xf, yf, zf - f32x8::splat(1.0)),
                    Self::grad(
                        fetch(ba + u32x8::splat(1)),
                        xf - f32x8::splat(1.0),
                        yf,
                        zf - f32x8::splat(1.0),
                    ),
                ),
                Self::lerp(
                    u,
                    Self::grad(
                        fetch(ab + u32x8::splat(1)),
                        xf,
                        yf - f32x8::splat(1.0),
                        zf - f32x8::splat(1.0),
                    ),
                    Self::grad(
                        fetch(bb + u32x8::splat(1)),
                        xf - f32x8::splat(1.0),
                        yf - f32x8::splat(1.0),
                        zf - f32x8::splat(1.0),
                    ),
                ),
            ),
        )
    }
}
