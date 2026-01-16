import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon

# ============================================================
# Helper math functions
# ============================================================

def angle_wrap(th):
    # wrap angle to [-pi, pi]
    th_wrapped = (th + np.pi) % (2*np.pi) - np.pi
    return th_wrapped

def poly3_eval(coef, x, x0):
    # coef: [c1, c2, c3, c4] for y = c1 + c2*(x-x0) + c3*(x-x0)^2 + c4*(x-x0)^3
    dx = (x - x0)
    return coef[0] + coef[1]*dx + coef[2]*dx**2 + coef[3]*dx**3

def splineinter(xa, ya, da, xb, yb, db):
    """
    Hermite-like cubic between (xa,ya) and (xb,yb),
    with slope da at xa and slope db at xb.
    We solve for coefficients c1..c4 of:
    y(x) = c1 + c2*(x-xa) + c3*(x-xa)^2 + c4*(x-xa)^3
    """
    dx = xb - xa

    # boundary conditions
    # yb = ya + da*dx + c3*dx^2 + c4*dx^3
    # db = da + 2*c3*dx + 3*c4*dx^2
    A = np.array([[dx**2,     dx**3],
                  [2*dx,  3*dx**2]], dtype=float)
    b = np.array([
        yb - ya - da*dx,
        db - da
    ], dtype=float)

    if abs(np.linalg.det(A)) < 1e-12:
        # fallback: if dx == 0 or degenerate, just do linear
        c3 = 0.0
        c4 = 0.0
    else:
        c3, c4 = np.linalg.solve(A, b)

    c1 = ya
    c2 = da
    return np.array([c1, c2, c3, c4], dtype=float)

def arclength_param(pts):
    """
    pts: Nx2 array
    returns:
      s_norm: normalized arclength in [0,1]
      xs, ys: arrays of same length
    Ensures closure.
    """
    pts = np.array(pts, dtype=float)

    # remove consecutive duplicates
    diffs = np.vstack(([0,0], np.diff(pts, axis=0)))
    keep = np.linalg.norm(diffs, axis=1) > 1e-9
    keep[0] = True
    pts = pts[keep]

    xs = pts[:,0].copy()
    ys = pts[:,1].copy()

    # close loop if not already
    if (xs[-1] != xs[0]) or (ys[-1] != ys[0]):
        xs = np.append(xs, xs[0])
        ys = np.append(ys, ys[0])

    seg = np.sqrt(np.diff(xs)**2 + np.diff(ys)**2)
    s = np.concatenate(([0.0], np.cumsum(seg)))
    s_norm = s / s[-1]
    return s_norm, xs, ys

def interp_track(sq, s_dist, waypts):
    """
    Interpolate along centerline by arc-length distance sq with wrap.
    s_dist: array of cumulative arc lengths
    waypts: Nx2 array of centerline points
    """
    track_length_local = s_dist[-1]

    # wrap sq
    while sq < 0:
        sq += track_length_local
    while sq > track_length_local:
        sq -= track_length_local

    # find segment where s_dist[i] <= sq < s_dist[i+1]
    idx = np.searchsorted(s_dist, sq, side='right') - 1
    if idx < 0:
        idx = 0
    if idx >= len(s_dist)-1:
        idx = len(s_dist)-2

    s0 = s_dist[idx]
    s1 = s_dist[idx+1]
    p0 = waypts[idx,:]
    p1 = waypts[idx+1,:]

    if s1 > s0:
        alpha = (sq - s0) / (s1 - s0)
    else:
        alpha = 0.0

    p = (1-alpha)*p0 + alpha*p1
    return p  # (2,)

def project_to_track(p, s_dist, waypts):
    """
    Project arbitrary point p=[x,y] onto the centerline polyline (waypts).
    Returns:
      s_hat: arc-length location of closest point
      p_hat: coordinates of closest point
    """
    p = np.array(p, dtype=float)
    best_d2 = np.inf
    best_s = 0.0
    best_pt = waypts[0,:]

    for k in range(len(waypts)-1):
        p0 = waypts[k,:]
        p1 = waypts[k+1,:]
        v = p1 - p0
        vv = np.dot(v,v)
        if vv < 1e-12:
            continue
        w = p - p0
        alpha = np.dot(w,v)/vv
        alpha = np.clip(alpha, 0.0, 1.0)
        proj = p0 + alpha*v
        d2 = np.dot(p - proj, p - proj)
        if d2 < best_d2:
            best_d2 = d2
            best_pt = proj
            # arc-length of projection
            seg_len = np.linalg.norm(v)
            best_s = s_dist[k] + alpha*seg_len

    # wrap s in [0, track_length]
    track_length_local = s_dist[-1]
    while best_s > track_length_local:
        best_s -= track_length_local
    while best_s < 0:
        best_s += track_length_local

    return best_s, best_pt

# ============================================================
# Track construction from your control points
# ============================================================

def make_centerline_from_custom_track():
    """
    Reconstructs the track you defined with control points and cubic splines:
     - Build outer boundary loop
     - Build inner boundary loop
     - Match them by normalized arclength
     - Centerline = midpoint(inner, outer)
     - Smooth and resample uniformly
    Returns:
     tx, ty : arrays (centerline)
    """

    # ----- control points (from your MATLAB script) -----

    p = {
        1: np.array([542, 744]),
        2: np.array([978, 950]),
        3: np.array([883, 417]),
        4: np.array([ 40, 286]),
        5: np.array([229, 823]),
    }
    q = {
        1: np.array([635, 922]),
        2: np.array([978, 550]),
        3: np.array([341, 144]),
        4: np.array([ 40, 756]),
        5: np.array([433, 719]),
    }
    s_ = {
        1: np.array([613, 696]),
        2: np.array([895, 850]),
        3: np.array([843, 485]),
        4: np.array([118, 290]),
        5: np.array([203, 742]),
    }
    v = {
        1: np.array([702, 877]),
        2: np.array([895, 550]),
        3: np.array([315, 219]),
        4: np.array([118, 692]),
        5: np.array([401, 643]),
    }
    r = {
        1: np.array([824,1031]),
        2: np.array([947, 453]),
        3: np.array([134, 116]),
        4: np.array([126, 860]),
        5: np.array([497, 705]),
    }
    d_slp = {
        1: 0,
        2: 1,
        3: -0.5,
        4: 0,
        5: 0,
    }
    w = {
        1: np.array([822, 942]),
        2: np.array([873, 504]),
        3: np.array([200, 190]),
        4: np.array([155, 759]),
        5: np.array([497, 612]),
    }
    e_slp = {
        1: 0,
        2: 1,
        3: 0,
        4: 0,
        5: 0,
    }

    # Optional: uniformly scale all control points (direct multiply about origin)
    # Set `scale` to a value != 1.0 (e.g. 2.0) to enlarge, <1.0 to shrink.
    scale = 0.1  # change this value to apply scaling (set to 2 to double control-point coordinates)
    if scale != 1.0:
        for D in (p, q, s_, v, r, w):
            for k in list(D.keys()):
                D[k] = D[k].astype(float) * scale

    # slopes on "outer" boundaries
    d_1 = {}
    d_2 = {}
    for i in range(1,6):
        # default slope from line through p->q
        denom = (q[i][0] - p[i][0])
        if abs(denom) < 1e-12:
            d_1[i] = 0.0
        else:
            d_1[i] = (q[i][1] - p[i][1]) / denom
        d_2[i] = d_1[i]

    d_1[2] = 4
    d_2[2] = -2
    d_1[4] = 4
    d_2[4] = -4

    # slopes on "inner" boundaries
    e_1 = {}
    e_2 = {}
    for i in range(1,6):
        denom = (v[i][0] - s_[i][0])
        if abs(denom) < 1e-12:
            e_1[i] = 0.0
        else:
            e_1[i] = (v[i][1] - s_[i][1]) / denom
        e_2[i] = e_1[i]

    e_1[2] = 4
    e_2[2] = -4
    e_1[4] = 4
    e_2[4] = -4

    # number of samples per spline "piece"
    Nseg = 30

    outer_pts = []
    inner_pts = []

    # build the loops
    for i in range(1,6):
        ip1 = (i % 5) + 1  # i+1 with wrap 5->1

        # OUTER piece q{i} -> r{i}
        coef1 = splineinter(q[i][0], q[i][1], d_1[i],
                            r[i][0], r[i][1], d_slp[i])
        x0 = q[i][0]
        xspan = np.linspace(q[i][0], r[i][0], Nseg)
        yspan = poly3_eval(coef1, xspan, x0)
        outer_pts.append(np.column_stack((xspan, yspan)))

        # OUTER piece r{i} -> p{ip1}
        coef2 = splineinter(r[i][0], r[i][1], d_slp[i],
                            p[ip1][0], p[ip1][1], d_2[ip1])
        x0 = r[i][0]
        xspan = np.linspace(r[i][0], p[ip1][0], Nseg)
        yspan = poly3_eval(coef2, xspan, x0)
        outer_pts.append(np.column_stack((xspan, yspan)))

        # INNER piece v{i} -> w{i}
        coef3 = splineinter(v[i][0], v[i][1], e_1[i],
                            w[i][0], w[i][1], e_slp[i])
        x0 = v[i][0]
        xspan = np.linspace(v[i][0], w[i][0], Nseg)
        yspan = poly3_eval(coef3, xspan, x0)
        inner_pts.append(np.column_stack((xspan, yspan)))

        # INNER piece w{i} -> s{ip1}
        coef4 = splineinter(w[i][0], w[i][1], e_slp[i],
                            s_[ip1][0], s_[ip1][1], e_2[ip1])
        x0 = w[i][0]
        xspan = np.linspace(w[i][0], s_[ip1][0], Nseg)
        yspan = poly3_eval(coef4, xspan, x0)
        inner_pts.append(np.column_stack((xspan, yspan)))

    outer_pts = np.vstack(outer_pts)
    inner_pts = np.vstack(inner_pts)

    # parameterize outer and inner boundaries by normalized arclength
    sO, xO_s, yO_s = arclength_param(outer_pts)
    sI, xI_s, yI_s = arclength_param(inner_pts)

    # common param u in [0,1]
    Nu = 600
    u = np.linspace(0,1,Nu)

    x_outer_u = np.interp(u, sO, xO_s)
    y_outer_u = np.interp(u, sO, yO_s)

    x_inner_u = np.interp(u, sI, xI_s)
    y_inner_u = np.interp(u, sI, yI_s)

    x_center = 0.5*(x_outer_u + x_inner_u)
    y_center = 0.5*(y_outer_u + y_inner_u)

    # smooth (moving average window=5)
    def movmean(arr, win=5):
        # simple centered moving average with wrap padding
        pad = win//2
        arr_pad = np.pad(arr, (pad,pad), mode='wrap')
        cumsum = np.cumsum(arr_pad, dtype=float)
        res = (cumsum[win:] - cumsum[:-win]) / win
        return res

    x_center_s = movmean(x_center, win=5)
    y_center_s = movmean(y_center, win=5)

    # close loop
    x_center_s[-1] = x_center_s[0]
    y_center_s[-1] = y_center_s[0]

    # resample uniformly in arc length
    pts_center = np.column_stack((x_center_s, y_center_s))
    sC, xC_s, yC_s = arclength_param(pts_center)
    totalL = sC[-1]

    Nsamp = 500
    s_query = np.linspace(0,totalL,Nsamp)
    tx = np.interp(s_query, sC, xC_s)
    ty = np.interp(s_query, sC, yC_s)

    # final closure
    tx[-1] = tx[0]
    ty[-1] = ty[0]

    return tx, ty

# ============================================================
# Vehicle dynamics (bicycle model)
# ============================================================

def bicycle_dynamics(state, delta, ax_cmd, params):
    """
    state = [x, y, psi, vx, vy, r]
    delta = steering angle (rad)
    ax_cmd = longitudinal accel command (m/s^2)
    params has m, Iz, lf, lr, Cf, Cr
    """
    x, y, psi, vx, vy, r = state

    m   = params["m"]
    Iz  = params["Iz"]
    lf  = params["lf"]
    lr  = params["lr"]
    Cf  = params["Cf"]
    Cr  = params["Cr"]

    vx_safe = vx if vx >= 0.5 else 0.5

    # slip angles
    alpha_f = delta - np.arctan2(vy + lf*r, vx_safe)
    alpha_r =      - np.arctan2(vy - lr*r, vx_safe)

    # lateral forces
    Fyf = Cf * alpha_f
    Fyr = Cr * alpha_r

    # global kinematics
    x_dot   = vx*np.cos(psi) - vy*np.sin(psi)
    y_dot   = vx*np.sin(psi) + vy*np.cos(psi)
    psi_dot = r

    # body-frame dynamics
    vx_dot = ax_cmd + (-Fyf * np.sin(delta))/m + r*vy
    vy_dot = (Fyf * np.cos(delta) + Fyr)/m - r*vx
    r_dot  = (lf*Fyf * np.cos(delta) - lr*Fyr)/Iz

    return np.array([x_dot, y_dot, psi_dot, vx_dot, vy_dot, r_dot])

# ============================================================
# Main simulation
# ============================================================

def main():
    # ---- build track ----
    track_x, track_y = make_centerline_from_custom_track()

    # close loop
    track_x[-1] = track_x[0]
    track_y[-1] = track_y[0]

    # normals to generate "asphalt" ribbon for visualization
    dx = np.gradient(track_x)
    dy = np.gradient(track_y)
    tang = np.stack([dx, dy], axis=1)
    tang_norm = np.linalg.norm(tang, axis=1).reshape(-1,1)
    tang_norm[tang_norm < 1e-12] = 1e-12
    tang = tang / tang_norm
    normal = np.stack([-tang[:,1], tang[:,0]], axis=1)

    track_width = 4.0  # same logic as MATLAB
    left_x  = track_x + track_width*normal[:,0]
    left_y  = track_y + track_width*normal[:,1]
    right_x = track_x - track_width*normal[:,0]
    right_y = track_y - track_width*normal[:,1]

    waypts = np.column_stack((track_x, track_y))
    # arclength along centerline
    diffs = np.diff(waypts, axis=0)
    seglen = np.sqrt(np.sum(diffs**2, axis=1))
    s_dist = np.concatenate(([0.0], np.cumsum(seglen)))
    track_length = s_dist[-1]

    # ---- vehicle + controller params ----
    params = {
        "m": 150.0,
        "Iz": 20.0,
        "lf": 0.7,
        "lr": 0.7,
        "Cf": 800.0,
        "Cr": 800.0,
    }

    k_delta   = 2.0
    max_steer = np.deg2rad(25.0)

    vx_ref    = 12.0
    k_vx      = 2.0
    max_accel =  4.0
    max_brake = -6.0

    look_ahead = 6.0
    dt = 0.02

    # ---- initial state ----
    x0 = track_x[0]
    y0 = track_y[0]
    psi0 = np.arctan2(dy[0], dx[0])
    vx0  = 0.1
    vy0  = 0.0
    r0   = 0.0
    state = np.array([x0, y0, psi0, vx0, vy0, r0])

    s_progress = 0.0
    penalty_offtrack = 0.0

    Tend = track_length / max(vx_ref, 0.1) * 2

    traj_x = []
    traj_y = []

    # ---- plotting setup ----
    plt.ion()
    fig, ax = plt.subplots()
    fig.set_facecolor("white")
    ax.set_aspect("equal", adjustable="box")

    # asphalt ribbon
    road_poly_x = np.concatenate([left_x, right_x[::-1]])
    road_poly_y = np.concatenate([left_y, right_y[::-1]])
    road_patch = Polygon(
        np.column_stack([road_poly_x, road_poly_y]),
        closed=True,
        facecolor=(0.2,0.2,0.2),
        edgecolor="none",
        alpha=0.95
    )
    ax.add_patch(road_patch)

    # centerline
    ax.plot(track_x, track_y, 'w--', linewidth=1.0)

    # start/finish line (normal at index 0)
    sf_n = normal[0,:]
    sf_pt = waypts[0,:]
    ax.plot([sf_pt[0]-sf_n[0]*track_width, sf_pt[0]+sf_n[0]*track_width],
            [sf_pt[1]-sf_n[1]*track_width, sf_pt[1]+sf_n[1]*track_width],
            color='y', linewidth=3)

    # walls
    ax.plot(left_x, left_y, 'k', linewidth=2)
    ax.plot(right_x,right_y,'k', linewidth=2)

    # trajectory line (live update)
    traj_line, = ax.plot([], [], 'c', linewidth=2)

    # kart patch (triangle) – we rotate/translate each frame
    kart_size = 1.0
    kart_scale = 1.1
    # base body polygon (slightly rounded triangular kart)
    kart_shape_local = np.array([
        [ 1.6,  0.0],
        [-1.0,  0.7],
        [-1.0, -0.7]
    ]) * kart_size * kart_scale

    # body patch
    body_local = kart_shape_local.copy()
    body_patch = Polygon(body_local, closed=True, facecolor=(0.9,0.12,0.12),
                         edgecolor='k', linewidth=1.4, zorder=10)
    ax.add_patch(body_patch)

    # canopy (windshield) patch
    canopy_local = np.array([[0.45,0.18], [ -0.05,0.45], [ -0.35,0.05]]) * kart_size * kart_scale
    canopy_patch = Polygon(canopy_local, closed=True, facecolor=(0.2,0.6,1.0),
                           edgecolor='k', linewidth=0.9, zorder=12)
    ax.add_patch(canopy_patch)

    # four wheels (as circles) with offsets in body frame
    wheel_offsets_local = np.array([[0.9,-0.6], [-0.9,-0.6], [0.9,0.6], [-0.9,0.6]]) * kart_size * kart_scale
    wheel_radius = 0.28 * kart_size * kart_scale
    wheel_patches = []
    for _ in range(4):
        c = plt.Circle((0,0), wheel_radius, facecolor='black', edgecolor='k', linewidth=0.6, zorder=9)
        ax.add_patch(c)
        wheel_patches.append(c)

    # HUD text
    margin = 50
    ax.set_xlim(np.min(track_x)-margin, np.max(track_x)+margin)
    ax.set_ylim(np.min(track_y)-margin, np.max(track_y)+margin)
    info_txt = ax.text(
        np.min(track_x)-margin+10,
        np.max(track_y)+margin-10,
        f"Time: {0.0:.1f}s\nPenalty: {penalty_offtrack:.1f}",
        fontsize=12,
        fontweight='bold',
        color='black',
        bbox=dict(facecolor=(1,1,1,0.6), edgecolor='k')
    )

    ax.set_title("Mario Kart Python Demo (Your Track)")
    ax.set_xlabel("x [px-ish]")
    ax.set_ylabel("y [px-ish]")

    # ---- main sim loop ----
    t = 0.0
    offtrack_timer = 0.0

    # lap tracking: detect crossing of start/finish by s wrap
    laps_completed = 0
    last_s_progress = s_progress

    # runtime flag: set to False for baseline controller, True for time-optimal
    use_time_optimal_controller = True

    while t < Tend:
        x, y, psi, vx, vy, r_ = state  # r_ = yaw rate, unused directly below

        if use_time_optimal_controller:
            # Time-optimal controller
            delta, ax_cmd, vx_target = time_optimal_controller(
                state, s_progress, s_dist, waypts,
                params, vx_ref, max_accel, max_brake,
                look_ahead, k_delta=k_delta, max_steer=max_steer,
                a_lat_max=6.0, nsamples=20
            )
        else:
            # Baseline controller (pure pursuit + speed PID)
            # pure-pursuit target s_target = s_progress + look_ahead
            s_target = s_progress + look_ahead
            if s_target > track_length:
                s_target -= track_length
            while s_target < 0:
                s_target += track_length
            p_target = interp_track(s_target, s_dist, waypts)

            dir_vec = p_target - np.array([x, y])
            desired_psi = np.arctan2(dir_vec[1], dir_vec[0])

            heading_err = angle_wrap(desired_psi - psi)
            delta = k_delta * heading_err
            delta = np.clip(delta, -max_steer, max_steer)

            ax_cmd = k_vx*(vx_ref - vx)
            ax_cmd = np.clip(ax_cmd, max_brake, max_accel)

        # Euler step
        xdot = bicycle_dynamics(state, delta, ax_cmd, params)
        state = state + xdot * dt

        # progress along lap
        s_progress, closest_pt = project_to_track(state[0:2], s_dist, waypts)

        # off-track penalty
        dist_center = np.linalg.norm(state[0:2] - closest_pt)
        if dist_center > track_width:
            penalty_offtrack += (dist_center - track_width)*dt
            offtrack_timer = 0.2
        else:
            offtrack_timer = max(0.0, offtrack_timer - dt)

        # store trajectory
        traj_x.append(state[0])
        traj_y.append(state[1])

        # detect crossing of finish line (wrap from high s to low s)
        if (last_s_progress > track_length * 0.9) and (s_progress < track_length * 0.1) and (t > 1.0):
            laps_completed += 1
            print(f"Finish line crossed. Laps completed = {laps_completed} at t={t:.2f}s")
            break

        last_s_progress = s_progress

        # ---- update drawing ----
        # rotate + translate kart triangle
        c = np.cos(state[2])
        s = np.sin(state[2])
        Rmat = np.array([[c, -s],
                         [s,  c]])
        # body
        kart_world = (Rmat @ body_local.T).T + state[0:2]
        body_patch.set_xy(kart_world)
        # canopy
        canopy_world = (Rmat @ canopy_local.T).T + state[0:2]
        canopy_patch.set_xy(canopy_world)
        # wheels
        for i, off in enumerate(wheel_offsets_local):
            wp = (Rmat @ off) + state[0:2]
            wheel_patches[i].center = (wp[0], wp[1])

        if offtrack_timer > 0:
            body_patch.set_facecolor((1,1,0))  # flash yellow
        else:
            body_patch.set_facecolor((0.9,0.12,0.12))  # normal red

        traj_line.set_xdata(traj_x)
        traj_line.set_ydata(traj_y)

        controller_type = "Time-optimal" if use_time_optimal_controller else "Baseline"
        info_txt.set_text(f"Time: {t:.1f}s\nPenalty: {penalty_offtrack:.1f}\nController: {controller_type}\nTarget speed: {vx_ref:.1f} m/s")

        plt.pause(0.001)  # let matplotlib update GUI

        t += dt

    print("Lap complete!")
    print(f"Total off-track penalty = {penalty_offtrack:.3f}")

    plt.ioff()
    plt.show()

def curvature_at_s(s, s_dist, waypts, ds=1.0):
    """Estimate curvature kappa at arc position s using three-point circle method.
    Uses interp_track which wraps s automatically. ds is an arc-length offset.
    Returns unsigned curvature (>=0).
    """
    p1 = interp_track(s - ds, s_dist, waypts)
    p2 = interp_track(s, s_dist, waypts)
    p3 = interp_track(s + ds, s_dist, waypts)

    a = np.linalg.norm(p2 - p1)
    b = np.linalg.norm(p3 - p2)
    c = np.linalg.norm(p3 - p1)
    area = 0.5 * abs(np.cross(p2 - p1, p3 - p1))
    denom = a * b * c
    if denom < 1e-9:
        return 0.0
    kappa = 4.0 * area / denom
    return kappa


def vmax_from_lateral_acc(kappa, a_lat_max):
    """Max speed satisfying v^2 * |kappa| <= a_lat_max."""
    kabs = abs(kappa)
    if kabs < 1e-12:
        return 1e6
    return np.sqrt(max(a_lat_max / kabs, 0.0))


def time_optimal_controller(state, s_progress, s_dist, waypts,
                            params, vx_ref, max_accel, max_brake,
                            look_ahead, k_delta=2.0, max_steer=np.deg2rad(25.0),
                            a_lat_max=6.0, nsamples=20):
    """Return (delta, ax_cmd) using curvature-based speed planning.

    Samples curvature ahead, computes vmax per sample from lateral acceleration
    limit, picks conservative vx_target = min(vx_ref, min(vmax_samples)). Steering
    uses pure-pursuit style look-ahead similar to baseline.
    """
    x, y, psi, vx, vy, r = state

    # steering (pure-pursuit style)
    s_target = s_progress + look_ahead
    track_length_local = s_dist[-1]
    while s_target > track_length_local:
        s_target -= track_length_local
    while s_target < 0:
        s_target += track_length_local
    p_target = interp_track(s_target, s_dist, waypts)
    dir_vec = p_target - np.array([x, y])
    desired_psi = np.arctan2(dir_vec[1], dir_vec[0])
    heading_err = angle_wrap(desired_psi - psi)
    delta = k_delta * heading_err
    delta = np.clip(delta, -max_steer, max_steer)

    # speed planning: sample ahead
    s_vals = np.linspace(s_progress, s_progress + max(look_ahead*2.0, 1.0), nsamples)
    vmax_vals = []
    for sv in s_vals:
        kappa = curvature_at_s(sv, s_dist, waypts, ds=max(1.0, look_ahead/10.0))
        vmax_vals.append(vmax_from_lateral_acc(kappa, a_lat_max))
    planned_vmax = min(vmax_vals) if len(vmax_vals) > 0 else vx_ref
    vx_target = min(vx_ref, planned_vmax)

    # simple proportional speed controller toward vx_target
    k_vx_local = 2.0
    ax_cmd = k_vx_local * (vx_target - vx)
    ax_cmd = np.clip(ax_cmd, max_brake, max_accel)

    return delta, ax_cmd, vx_target

# ============================================================
# run
# ============================================================

if __name__ == "__main__":
    main()