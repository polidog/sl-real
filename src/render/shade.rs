//! 陰影・霧・影の計算。スレッドから直接呼べるようフリー関数にしてある。

use super::{Env, Material};
use crate::math::*;

/// 影の箱による遮蔽率。地面のシェーディングから直接呼べるようフリー関数にしてある。
///
/// 点 `p` から太陽へ向かう線分が箱を貫いているかを、
/// 2D のスラブ判定で求める。貫通長が長いほど濃い影になる。
pub fn shadow_of(boxes: &[(V3, V3)], sun: V3, p: V3) -> f32 {
    if boxes.is_empty() || sun.y < 0.06 {
        return 0.0;
    }
    let l = sun;
    let mut occ: f32 = 0.0;
    for (lo, hi) in boxes {
        // 箱の高さの範囲を通る t の区間。
        let mut tmin = ((lo.y - p.y) / l.y).max(0.0);
        let mut tmax = (hi.y - p.y) / l.y;
        if tmax <= tmin {
            continue;
        }
        // x と z のスラブで区間を削る。
        let mut ok = true;
        for (o, d, a, b) in [(p.x, l.x, lo.x, hi.x), (p.z, l.z, lo.z, hi.z)] {
            if d.abs() < 1e-6 {
                if o < a || o > b {
                    ok = false;
                    break;
                }
                continue;
            }
            let (t0, t1) = ((a - o) / d, (b - o) / d);
            tmin = tmin.max(t0.min(t1));
            tmax = tmax.min(t0.max(t1));
            if tmax <= tmin {
                ok = false;
                break;
            }
        }
        if !ok {
            continue;
        }
        // 距離が離れるほど輪郭をぼかす（半影）。
        let soft = 0.20 + tmin * 0.10;
        occ = occ.max(smoothstep(0.0, soft, tmax - tmin));
    }
    saturate(occ) * 0.94
}

/// 大気による減衰。
pub fn fog_of(env: &Env, c: V3, dist: f32, dir: V3) -> V3 {
    let f = 1.0 - (-dist * env.fog_density).exp();
    // 太陽方向の霧はほんのり明るい。
    let g = saturate(dir.dot(env.sun_dir));
    let glow = g * g * g * g;
    let fog_col = env.horizon_color.lerp(env.sun_color, glow * 0.5);
    c.lerp(fog_col, f)
}

/// 陰影計算の本体。スレッドから直接呼べるようフリー関数にしてある。
pub fn shade_of(env: &Env, cam_pos: V3, p: V3, n: V3, albedo: V3, m: &Material, shadow: f32) -> V3 {
    let e = env;
    let view = (cam_pos - p).norm();
    // 裏面が見えているときは法線を反転して両面ライティングにする。
    let n = if n.dot(view) < 0.0 { -n } else { n };

    let ndl = saturate(n.dot(e.sun_dir));
    let vis = 1.0 - shadow;
    let mut lit = albedo.mul3(e.sun_color) * (ndl * vis);

    // 半球状の環境光。上は空、下は地面の照り返し。
    let up = saturate(n.y * 0.5 + 0.5);
    let amb = e.sky_color * up + e.ground_color * (1.0 - up);
    lit += albedo.mul3(amb) * 0.55;

    // Blinn-Phong のハイライト。金属は素材色に着色する。
    let spec_tint = V3::splat(1.0).lerp(albedo, m.metal);
    let hv = (e.sun_dir + view).norm();
    let ndh = saturate(n.dot(hv));
    let shin = 2.0 / (m.rough * m.rough + 1e-3) + 2.0;
    // 指数が大きいので、少し外れただけでほぼ 0 になる。そこは計算しない。
    if ndh * ndh > 0.25 || shin < 12.0 {
        let spec = ndh.powf(shin) * (1.0 - m.rough) * vis * ndl.max(0.0);
        lit += spec_tint.mul3(e.sun_color) * (spec * (0.32 + m.metal * 0.85));
    }

    // 空と地面の映り込み。黒い車体に立体感を与えるのはほぼこれ。
    let refl = n * (2.0 * n.dot(view)) - view;
    let ry = refl.y;
    let env_col = if ry >= 0.0 {
        e.horizon_color.lerp(e.zenith_color, saturate(ry).sqrt())
    } else {
        e.horizon_color.lerp(e.ground_color, saturate(-ry).sqrt())
    };
    // ざらついた面ほど反射はぼやけ、彩度も落ちる。
    let env_lum = env_col.x * 0.2126 + env_col.y * 0.7152 + env_col.z * 0.0722;
    let env_col = env_col.lerp(V3::splat(env_lum), m.rough * 0.55);
    let fres = (1.0 - saturate(n.dot(view))).powi(5);
    let k = (0.018 + m.metal * 0.20) * (1.0 - m.rough * 0.65) + fres * (0.08 + m.metal * 0.40);
    lit += env_col.mul3(spec_tint) * k;

    // 前照灯（点光源 + スポット）。
    if e.head_power > 0.0 {
        let d = e.head_pos - p;
        let dist = d.len().max(0.5);
        let ld = d / dist;
        let spot = saturate((-ld).dot(e.head_dir));
        let spot = spot.powf(42.0);
        let att = e.head_power / (1.0 + dist * dist * 0.02);
        lit += albedo.mul3(v3(1.0, 0.93, 0.75)) * (saturate(n.dot(ld)) * att * spot);
    }
    // かまどの火の漏れ光。
    if e.fire_power > 0.0 {
        let d = e.fire_pos - p;
        let dist = d.len().max(0.4);
        let att = e.fire_power / (1.0 + dist * dist * 0.55);
        lit += albedo.mul3(v3(1.0, 0.38, 0.10)) * (saturate(n.dot(d / dist)) * att);
    }

    lit + m.emissive
}
