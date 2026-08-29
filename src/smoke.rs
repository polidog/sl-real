//! 煙と蒸気のパーティクル。
//! 煙突からのドラフトはピストンの排気に同期して 1 回転 4 拍で吹き上がる。

use crate::math::*;
use crate::noise::{fbm3, Rng};
use crate::render::Renderer;
use crate::train::Train;

#[derive(Clone, Copy, PartialEq)]
pub enum Kind {
    /// 煙突からの煤けた煙。
    Exhaust,
    /// シリンダのドレンや安全弁の白い蒸気。
    Steam,
}

#[derive(Clone, Copy)]
pub struct Particle {
    pub p: V3,
    pub v: V3,
    pub age: f32,
    pub life: f32,
    pub r0: f32,
    pub r1: f32,
    pub seed: f32,
    pub kind: Kind,
    /// 生まれたときの濃さ。
    pub density: f32,
}

pub struct Smoke {
    pub parts: Vec<Particle>,
    rng: Rng,
    /// 排気の拍を数える位相。
    chuff_phase: f32,
    /// 直近のドラフトの強さ（描画側でも使う）。
    pub last_chuff: f32,
    pub wind: V3,
    max: usize,
}

impl Smoke {
    pub fn new(seed: u64, max: usize) -> Smoke {
        Smoke {
            parts: Vec::with_capacity(max),
            rng: Rng::new(seed ^ 0x9E37_79B9_7F4A_7C15),
            chuff_phase: 0.0,
            last_chuff: 0.0,
            wind: v3(-0.6, 0.0, 1.1),
            max,
        }
    }

    fn spawn(&mut self, p: Particle) {
        if self.parts.len() < self.max {
            self.parts.push(p);
        }
    }

    /// 1 フレーム進める。`d_dist` はこのフレームで進んだ距離。
    pub fn update(&mut self, t: &Train, dt: f32, d_dist: f32) {
        self.emit(t, dt, d_dist);
        self.simulate(t, dt);
    }

    fn emit(&mut self, t: &Train, dt: f32, d_dist: f32) {
        // ---- 煙突：1 回転につき 4 拍。
        let rev = d_dist / (std::f32::consts::TAU * 0.70);
        self.chuff_phase += rev * 4.0;
        // 停車中でもボイラーは炊いているので、ゆっくり湧かせる。
        let idle = t.speed.abs() < 0.4;
        if idle {
            self.chuff_phase += dt * 1.6;
        }
        self.last_chuff = (self.last_chuff - dt * 6.0).max(0.0);

        let mouth = t.stack_mouth();
        let fwd = v3(1.0, 0.0, 0.0);
        while self.chuff_phase >= 1.0 {
            self.chuff_phase -= 1.0;
            self.last_chuff = 1.0;
            let power = 0.35 + t.throttle * 0.85;
            // 1 拍あたり数個の玉を、少しずつ時間をずらして出す。
            let n = if idle { 3 } else { 8 };
            for i in 0..n {
                let k = i as f32 / n as f32;
                let up = (7.5 + 9.0 * power) * (1.0 - k * 0.35) * self.rng.range(0.85, 1.15);
                let spread = 0.55 + k * 0.5;
                let p = mouth
                    + v3(
                        self.rng.sf32() * 0.10,
                        k * 0.16,
                        self.rng.sf32() * 0.10,
                    )
                    // 吹き上がる間に列車が進んだ分をずらす。
                    + fwd * (d_dist * k);
                let q = Particle {
                    p,
                    v: v3(
                        t.speed * 0.55 + self.rng.sf32() * spread,
                        up,
                        self.rng.sf32() * spread,
                    ),
                    age: 0.0,
                    life: self.rng.range(3.4, 6.2) + power * 1.6,
                    r0: self.rng.range(0.22, 0.38),
                    r1: self.rng.range(3.6, 6.5),
                    seed: self.rng.f32(),
                    kind: Kind::Exhaust,
                    density: self.rng.range(0.55, 0.85) * (0.45 + power * 0.7),
                };
                self.spawn(q);
            }
        }

        // ---- シリンダのドレン（発車直後に強く吹く）。
        let cock = saturate(1.0 - t.speed / 7.0) * t.throttle;
        if cock > 0.02 {
            let want = cock * 46.0 * dt;
            let n = want.floor() as u32 + if self.rng.f32() < want.fract() { 1 } else { 0 };
            for _ in 0..n {
                let left = self.rng.f32() < 0.5;
                let base = t.cock_pos(left);
                let sz = if left { 1.0 } else { -1.0 };
                let q = Particle {
                    p: base + v3(self.rng.sf32() * 0.1, self.rng.sf32() * 0.06, 0.0),
                    v: v3(
                        t.speed * 0.5 + self.rng.range(1.0, 3.5),
                        self.rng.range(-1.2, 0.4),
                        sz * self.rng.range(2.5, 5.5),
                    ),
                    age: 0.0,
                    life: self.rng.range(1.1, 2.2),
                    r0: self.rng.range(0.18, 0.32),
                    r1: self.rng.range(1.6, 2.8),
                    seed: self.rng.f32(),
                    kind: Kind::Steam,
                    density: self.rng.range(0.65, 0.95),
                };
                self.spawn(q);
            }
        }

        // ---- 安全弁。ときどき勢いよく抜ける。
        let blow = ((t.time * 0.21).sin() * 0.5 + 0.5).powf(14.0) * (0.3 + t.throttle * 0.7);
        if blow > 0.45 {
            let want = (blow - 0.45) * 110.0 * dt;
            let n = want.floor() as u32 + if self.rng.f32() < want.fract() { 1 } else { 0 };
            let sp = t.safety_pos();
            for _ in 0..n {
                let q = Particle {
                    p: sp + v3(self.rng.sf32() * 0.07, 0.0, self.rng.sf32() * 0.07),
                    v: v3(
                        t.speed * 0.6 + self.rng.sf32() * 0.6,
                        self.rng.range(6.0, 10.0),
                        self.rng.sf32() * 0.6,
                    ),
                    age: 0.0,
                    life: self.rng.range(0.7, 1.4),
                    r0: self.rng.range(0.07, 0.13),
                    r1: self.rng.range(0.55, 1.05),
                    seed: self.rng.f32(),
                    kind: Kind::Steam,
                    density: self.rng.range(0.20, 0.40),
                };
                self.spawn(q);
            }
        }
    }

    fn simulate(&mut self, t: &Train, dt: f32) {
        let wind = self.wind;
        let time = t.time;
        self.parts.retain_mut(|q| {
            q.age += dt;
            if q.age >= q.life {
                return false;
            }
            let u = q.age / q.life;

            // 浮力。熱い間は強く、冷えるにつれて弱まる。
            let heat = (-u * 3.2).exp();
            let buoy = match q.kind {
                Kind::Exhaust => 5.5 * heat + 0.5,
                Kind::Steam => 3.0 * heat + 0.25,
            };
            q.v.y += buoy * dt;

            // 空気抵抗。周囲の風速へ収束させる。
            let drag = match q.kind {
                Kind::Exhaust => 1.5,
                Kind::Steam => 2.4,
            };
            let rel = q.v - wind;
            q.v += -rel * (drag * dt).min(1.0);

            // 乱流。位置と時間で変化するベクトル場でかき混ぜる。
            let s = 0.18;
            let base = v3(q.p.x * s, q.p.y * s, q.p.z * s) + v3(0.0, -time * 0.25, 0.0);
            let n1 = fbm3(base, 2) - 0.5;
            let n2 = fbm3(base + v3(31.4, 7.7, -12.9), 2) - 0.5;
            let n3 = fbm3(base + v3(-8.3, 22.1, 5.5), 2) - 0.5;
            // 育つほど大きな渦に乗る。
            let turb = (1.4 + u * 5.5) * (0.5 + q.seed * 0.8);
            q.v += v3(n1, n2 * 0.7, n3) * (turb * dt);

            q.p += q.v * dt;

            // 地面で止める。
            if q.p.y < 0.15 {
                q.p.y = 0.15;
                q.v.y *= -0.15;
                q.v.x *= 0.85;
                q.v.z *= 0.85;
            }
            true
        });
    }

    /// 半透明なので奥から手前へ並べて描く。
    pub fn draw(&mut self, r: &mut Renderer, night: f32) {
        let cam = r.cam_pos;
        let mut order: Vec<(f32, usize)> = self
            .parts
            .iter()
            .enumerate()
            .map(|(i, q)| ((q.p - cam).dot(q.p - cam), i))
            .collect();
        order.sort_unstable_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));

        for (_, i) in order {
            let q = self.parts[i];
            let u = q.age / q.life;
            let radius = lerp(q.r0, q.r1, u.powf(0.62));

            // 濃さ：生まれてすぐ立ち上がり、広がるにつれて薄まる。
            let fade_in = smoothstep(0.0, 0.05, u);
            let fade_out = 1.0 - smoothstep(0.35, 1.0, u);
            let dilute = (q.r0 / radius).powf(0.85);
            let alpha = q.density * fade_in * fade_out * dilute * 1.9;
            if alpha < 0.006 {
                continue;
            }

            let tint = match q.kind {
                // 出たては煤で黒く、薄まりながら灰色へ。
                Kind::Exhaust => v3(0.045, 0.042, 0.040)
                    .lerp(v3(0.30, 0.295, 0.285), smoothstep(0.0, 0.65, u))
                    * lerp(1.0, 0.55, night),
                Kind::Steam => v3(0.94, 0.95, 0.97) * lerp(1.0, 0.7, night),
            };
            r.puff(q.p, radius, tint, saturate(alpha), q.seed);
        }
    }

    /// 煙突の直上に立ちのぼる熱の揺らぎ（陽炎）を弱い発光として足す。
    pub fn draw_glow(&self, r: &mut Renderer, t: &Train, night: f32) {
        if self.last_chuff > 0.01 && night > 0.05 {
            // 夜は火の粉が舞う。
            let m = t.stack_mouth();
            r.glow(m + v3(0.0, 0.35, 0.0), 0.5, v3(1.0, 0.45, 0.12), self.last_chuff * 0.9 * night);
        }
    }
}
