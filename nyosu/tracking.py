from __future__ import annotations

import math



def target_point_ratio(box, ratio):
    x1, y1, x2, y2 = box
    return x1 + (x2 - x1) / 2, y1 + (y2 - y1) * ratio


def target_point(box, tp):
    x1, y1, x2, y2 = box
    w, h = x2 - x1, y2 - y1
    if tp == "head":
        if h < 40:
            ratio = 0.22
        elif h < 90:
            ratio = 0.16
        elif h < 160:
            ratio = 0.12
        else:
            ratio = 0.10
        return x1 + w / 2, y1 + h * ratio
    return x1 + w / 2, y1 + h * 0.45


class KalmanCV:
    R = 9.0
    Q_POS = 2.0
    Q_VEL = 1400.0

    def __init__(self):
        self._ax = None
        self._ay = None
        self.last_time = 0.0

    @property
    def ready(self):
        return self._ax is not None

    @property
    def vx(self):
        return self._ax[1] if self._ax else 0.0

    @property
    def vy(self):
        return self._ay[1] if self._ay else 0.0

    def reset(self, pos=None, now=0.0):
        if pos is None:
            self._ax = self._ay = None
            return
        self._ax = [float(pos[0]), 0.0, 25.0, 0.0, 400.0]
        self._ay = [float(pos[1]), 0.0, 25.0, 0.0, 400.0]
        self.last_time = now

    @staticmethod
    def _predict(st, dt):
        pos, vel, p00, p01, p11 = st
        st[0] = pos + vel * dt
        st[2] = p00 + 2 * dt * p01 + dt * dt * p11 + KalmanCV.Q_POS
        st[3] = p01 + dt * p11
        st[4] = p11 + KalmanCV.Q_VEL * dt

    @staticmethod
    def _correct(st, z):
        pos, vel, p00, p01, p11 = st
        y = z - pos
        s_ = p00 + KalmanCV.R
        k0 = p00 / s_
        k1 = p01 / s_
        st[0] = pos + k0 * y
        st[1] = vel + k1 * y
        st[2] = (1 - k0) * p00
        st[3] = (1 - k0) * p01
        st[4] = p11 - k1 * p01

    def update(self, pos, now):
        if not self.ready:
            self.reset(pos, now)
            return (float(pos[0]), float(pos[1]))
        dt = now - self.last_time
        if dt <= 0.001 or dt > 0.25 or \
                math.hypot(pos[0] - self._ax[0], pos[1] - self._ay[0]) > 250:
            self.reset(pos, now)
            return (float(pos[0]), float(pos[1]))
        self._predict(self._ax, dt)
        self._predict(self._ay, dt)
        self._correct(self._ax, float(pos[0]))
        self._correct(self._ay, float(pos[1]))
        self.last_time = now
        return (self._ax[0], self._ay[0])

    def extrapolate(self, now, extra_lead=0.0):
        if not self.ready:
            return None
        dt = now - self.last_time
        if dt < 0 or dt > 0.5:
            return None
        t = dt + max(0.0, extra_lead)
        return (self._ax[0] + self._ax[1] * t, self._ay[0] + self._ay[1] * t)


class DetTracker:

    def __init__(self, alpha=0.6, assoc_px=90.0, keep_s=0.15):
        self.alpha = alpha
        self.assoc_px = assoc_px
        self.keep_s = keep_s
        self.tracks = []

    def reset(self):
        self.tracks = []

    def update(self, dets, now):
        used = set()
        out = []
        alpha = self.alpha
        keep_alpha = 1.0 - alpha
        centers = [((t["box"][0] + t["box"][2]) * 0.5, (t["box"][1] + t["box"][3]) * 0.5)
                   for t in self.tracks]
        parsed = []
        for d in dets:
            try:
                bx = d["box"]
                vals = (float(bx[0]), float(bx[1]), float(bx[2]), float(bx[3]), float(d["conf"]))
            except Exception:
                continue
            if not all(math.isfinite(v) for v in vals):
                continue
            parsed.append(vals)
        parsed.sort(key=lambda v: -v[4])
        for x1, y1, x2, y2, dconf in parsed:
            cx, cy = (x1 + x2) * 0.5, (y1 + y2) * 0.5
            best_i, best_d = -1, self.assoc_px
            for i in range(len(self.tracks)):
                if i in used:
                    continue
                tcx, tcy = centers[i]
                dist = math.hypot(tcx - cx, tcy - cy)
                if dist < best_d:
                    best_i, best_d = i, dist
            if best_i >= 0:
                t = self.tracks[best_i]
                ob = t["box"]
                sb = (alpha * x1 + keep_alpha * ob[0], alpha * y1 + keep_alpha * ob[1],
                      alpha * x2 + keep_alpha * ob[2], alpha * y2 + keep_alpha * ob[3])
                t["box"] = sb
                t["conf"] = dconf
                t["seen"] = now
                used.add(best_i)
                centers[best_i] = ((sb[0] + sb[2]) * 0.5, (sb[1] + sb[3]) * 0.5)
            else:
                sb = (x1, y1, x2, y2)
                self.tracks.append({"box": sb, "conf": dconf, "seen": now})
                centers.append((cx, cy))
                used.add(len(self.tracks) - 1)
            out.append({"box": [int(round(v)) for v in sb], "conf": dconf})
        self.tracks = [t for t in self.tracks if now - t["seen"] <= self.keep_s]
        return out
