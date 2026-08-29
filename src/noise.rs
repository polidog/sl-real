//! ハッシュベースのノイズ群。地面のテクスチャ、山の稜線、煙のゆらぎに使う。

use crate::math::{v3, V3};

#[inline]
pub fn hash1(mut n: u32) -> f32 {
    n = (n << 13) ^ n;
    n = n.wrapping_mul(n.wrapping_mul(n).wrapping_mul(15731).wrapping_add(789221))
        .wrapping_add(1376312589);
    (n & 0x7fff_ffff) as f32 / 2_147_483_647.0
}

#[inline]
pub fn hash2(x: i32, y: i32) -> f32 {
    hash1((x as u32).wrapping_mul(374_761_393) ^ (y as u32).wrapping_mul(668_265_263))
}

#[inline]
pub fn hash3(x: i32, y: i32, z: i32) -> f32 {
    hash1(
        (x as u32)
            .wrapping_mul(374_761_393)
            .wrapping_add((y as u32).wrapping_mul(668_265_263))
            ^ (z as u32).wrapping_mul(2_246_822_519),
    )
}

#[inline]
fn fade(t: f32) -> f32 {
    t * t * (3.0 - 2.0 * t)
}

/// 1D 値ノイズ。
pub fn noise1(x: f32) -> f32 {
    let i = x.floor();
    let f = fade(x - i);
    let a = hash1(i as i32 as u32);
    let b = hash1((i as i32 + 1) as u32);
    a + (b - a) * f
}

/// 2D 値ノイズ。
pub fn noise2(x: f32, y: f32) -> f32 {
    let ix = x.floor();
    let iy = y.floor();
    let fx = fade(x - ix);
    let fy = fade(y - iy);
    let (ix, iy) = (ix as i32, iy as i32);
    let a = hash2(ix, iy);
    let b = hash2(ix + 1, iy);
    let c = hash2(ix, iy + 1);
    let d = hash2(ix + 1, iy + 1);
    let ab = a + (b - a) * fx;
    let cd = c + (d - c) * fx;
    ab + (cd - ab) * fy
}

/// 3D 値ノイズ。
pub fn noise3(p: V3) -> f32 {
    let ix = p.x.floor();
    let iy = p.y.floor();
    let iz = p.z.floor();
    let fx = fade(p.x - ix);
    let fy = fade(p.y - iy);
    let fz = fade(p.z - iz);
    let (ix, iy, iz) = (ix as i32, iy as i32, iz as i32);
    let mut acc = [0.0f32; 2];
    for (k, dz) in [0, 1].iter().enumerate() {
        let a = hash3(ix, iy, iz + dz);
        let b = hash3(ix + 1, iy, iz + dz);
        let c = hash3(ix, iy + 1, iz + dz);
        let d = hash3(ix + 1, iy + 1, iz + dz);
        let ab = a + (b - a) * fx;
        let cd = c + (d - c) * fx;
        acc[k] = ab + (cd - ab) * fy;
    }
    acc[0] + (acc[1] - acc[0]) * fz
}

pub fn fbm2(mut x: f32, mut y: f32, octaves: u32) -> f32 {
    let mut sum = 0.0;
    let mut amp = 0.5;
    let mut norm = 0.0;
    for _ in 0..octaves {
        sum += noise2(x, y) * amp;
        norm += amp;
        x = x * 2.03 + 11.7;
        y = y * 2.03 - 5.3;
        amp *= 0.5;
    }
    sum / norm
}

pub fn fbm3(mut p: V3, octaves: u32) -> f32 {
    let mut sum = 0.0;
    let mut amp = 0.5;
    let mut norm = 0.0;
    for _ in 0..octaves {
        sum += noise3(p) * amp;
        norm += amp;
        p = p * 2.03 + v3(7.1, -3.7, 13.3);
        amp *= 0.5;
    }
    sum / norm
}

/// 決定論的な小型 PRNG（煙の粒子生成などに使う）。
pub struct Rng(pub u64);

impl Rng {
    pub fn new(seed: u64) -> Rng {
        Rng(seed | 1)
    }
    #[inline]
    pub fn next_u32(&mut self) -> u32 {
        // xorshift64*
        let mut x = self.0;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.0 = x;
        (x.wrapping_mul(0x2545_F491_4F6C_DD1D) >> 32) as u32
    }
    /// [0, 1)
    #[inline]
    pub fn f32(&mut self) -> f32 {
        self.next_u32() as f32 / 4_294_967_296.0
    }
    /// [-1, 1)
    #[inline]
    pub fn sf32(&mut self) -> f32 {
        self.f32() * 2.0 - 1.0
    }
    #[inline]
    pub fn range(&mut self, a: f32, b: f32) -> f32 {
        a + (b - a) * self.f32()
    }
}
