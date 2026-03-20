def perclos_from_events(events, now, window_s):
    if len(events) < 2:
        return 0.0
    t0 = now - window_s
    ev = [(t, c) for (t, c) in events if t >= t0]
    if len(ev) < 2:
        return 0.0

    closed_time = 0.0
    total_time = ev[-1][0] - ev[0][0]
    for i in range(1, len(ev)):
        t_prev, c_prev = ev[i - 1]
        t_curr, _ = ev[i]
        if c_prev:
            closed_time += (t_curr - t_prev)

    return float(closed_time / (total_time + 1e-9))

def fatigue_score_0_1(perclos, long_closure):
    score = 0.85 * perclos + (0.15 if long_closure else 0.0)
    return float(min(1.0, max(0.0, score)))

def fatigue_percent(perclos, long_closure):
    return int(round(fatigue_score_0_1(perclos, long_closure) * 100.0))
