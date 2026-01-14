function marioKartMatlabDemo_fixed()
% Mario Kart Python Demo (Your Track) - MATLAB conversion (FIXED)
% Fix: removed containers.Map usage (MATLAB indexing issue) and replaced with matrices.
%
% To run:
%   marioKartMatlabDemo_fixed

% ---- build track ----
[track_x, track_y] = make_centerline_from_custom_track();

% ensure closure
track_x(end) = track_x(1);
track_y(end) = track_y(1);

% normals to generate "asphalt" ribbon for visualization
dx = gradient(track_x);
dy = gradient(track_y);

tang = [dx(:), dy(:)];
tang_norm = sqrt(sum(tang.^2,2));
tang_norm(tang_norm < 1e-12) = 1e-12;
tang = tang ./ tang_norm;

normal = [-tang(:,2), tang(:,1)];

track_width = 4.0;
left_x  = track_x + track_width*normal(:,1);
left_y  = track_y + track_width*normal(:,2);
right_x = track_x - track_width*normal(:,1);
right_y = track_y - track_width*normal(:,2);

waypts = [track_x(:), track_y(:)];
diffs = diff(waypts,1,1);
seglen = sqrt(sum(diffs.^2,2));
s_dist = [0.0; cumsum(seglen)];
track_length = s_dist(end);

% ---- vehicle + controller params ----
params.m  = 150.0;
params.Iz = 20.0;
params.lf = 0.7;
params.lr = 0.7;
params.Cf = 800.0;
params.Cr = 800.0;

k_delta   = 2.0;
max_steer = deg2rad(25.0);

vx_ref    = 12.0;
k_vx      = 2.0;
max_accel =  4.0;
max_brake = -6.0;

look_ahead = 6.0;
dt = 0.02;

% ---- initial state ----
x0 = track_x(1);
y0 = track_y(1);
psi0 = atan2(dy(1), dx(1));
vx0  = 0.1;
vy0  = 0.0;
r0   = 0.0;
state = [x0; y0; psi0; vx0; vy0; r0]; % [x;y;psi;vx;vy;r]

s_progress = 0.0;
penalty_offtrack = 0.0;

Tend = track_length / max(vx_ref,0.1) * 2;

traj_x = [];
traj_y = [];

% ---- plotting setup ----
figure('Color','w');
ax = axes();
axis(ax,'equal');
hold(ax,'on');

% asphalt ribbon patch
road_poly_x = [left_x(:); flipud(right_x(:))];
road_poly_y = [left_y(:); flipud(right_y(:))];
patch(ax, road_poly_x, road_poly_y, [0.2 0.2 0.2], ...
    'EdgeColor','none','FaceAlpha',0.95);

% centerline
plot(ax, track_x, track_y, 'w--', 'LineWidth', 1.0);

% start/finish line (normal at index 1)
sf_n  = normal(1,:);
sf_pt = waypts(1,:);
plot(ax, [sf_pt(1)-sf_n(1)*track_width, sf_pt(1)+sf_n(1)*track_width], ...
    [sf_pt(2)-sf_n(2)*track_width, sf_pt(2)+sf_n(2)*track_width], ...
    'y-', 'LineWidth', 3);

% walls
plot(ax, left_x, left_y, 'k', 'LineWidth', 2);
plot(ax, right_x, right_y,'k', 'LineWidth', 2);

% trajectory line
traj_line = plot(ax, NaN, NaN, 'c-', 'LineWidth', 2);

% kart patches
kart_size = 1.0;
kart_scale = 1.1;

body_local = [ 1.6  0.0;
    -1.0  0.7;
    -1.0 -0.7] * kart_size * kart_scale;

canopy_local = [ 0.45  0.18;
    -0.05  0.45;
    -0.35  0.05] * kart_size * kart_scale;

body_patch = patch(ax, body_local(:,1), body_local(:,2), [0.9 0.12 0.12], ...
    'EdgeColor','k','LineWidth',1.4);

canopy_patch = patch(ax, canopy_local(:,1), canopy_local(:,2), [0.2 0.6 1.0], ...
    'EdgeColor','k','LineWidth',0.9);

wheel_offsets_local = [ 0.9 -0.6;
    -0.9 -0.6;
    0.9  0.6;
    -0.9  0.6] * kart_size * kart_scale;
wheel_radius = 0.28 * kart_size * kart_scale;

wheel_patches = gobjects(4,1);
th = linspace(0,2*pi,25);
for i=1:4
    cx = 0; cy = 0;
    wheel_patches(i) = patch(ax, cx + wheel_radius*cos(th), cy + wheel_radius*sin(th), [0 0 0], ...
        'EdgeColor','k','LineWidth',0.6);
end

margin = 50;
xlim(ax, [min(track_x)-margin, max(track_x)+margin]);
ylim(ax, [min(track_y)-margin, max(track_y)+margin]);

info_txt = text(ax, min(track_x)-margin+10, max(track_y)+margin-10, ...
    sprintf('Time: %.1fs\nPenalty: %.1f',0.0, penalty_offtrack), ...
    'FontSize',12,'FontWeight','bold','Color','k', ...
    'BackgroundColor',[1 1 1],'Margin',6);

title(ax, 'Mario Kart MATLAB Demo (Your Track) - FIXED');
xlabel(ax, 'x [px-ish]');
ylabel(ax, 'y [px-ish]');

% ---- main sim loop ----
t = 0.0;
offtrack_timer = 0.0;

laps_completed = 0;
last_s_progress = s_progress;

use_time_optimal_controller = true;

while t < Tend
    x = state(1); y = state(2); psi = state(3);
    vx = state(4);

    if use_time_optimal_controller
        [delta, ax_cmd] = time_optimal_controller( ...
            state, s_progress, s_dist, waypts, ...
            vx_ref, max_accel, max_brake, ...
            look_ahead, k_delta, max_steer, ...
            6.0, 20);
    else
        % Baseline controller (pure pursuit + speed P)
        s_target = s_progress + look_ahead;
        if s_target > track_length, s_target = s_target - track_length; end
        while s_target < 0, s_target = s_target + track_length; end

        p_target = interp_track(s_target, s_dist, waypts);

        dir_vec = p_target - [x; y];
        desired_psi = atan2(dir_vec(2), dir_vec(1));

        heading_err = angle_wrap(desired_psi - psi);
        delta = k_delta * heading_err;
        delta = min(max(delta, -max_steer), max_steer);

        ax_cmd = k_vx*(vx_ref - vx);
        ax_cmd = min(max(ax_cmd, max_brake), max_accel);
    end

    % Euler step
    xdot = bicycle_dynamics(state, delta, ax_cmd, params);
    state = state + xdot * dt;

    % progress along lap
    [s_progress, closest_pt] = project_to_track(state(1:2), s_dist, waypts);

    % off-track penalty
    dist_center = norm(state(1:2) - closest_pt);
    if dist_center > track_width
        penalty_offtrack = penalty_offtrack + (dist_center - track_width)*dt;
        offtrack_timer = 0.2;
    else
        offtrack_timer = max(0.0, offtrack_timer - dt);
    end

    traj_x(end+1,1) = state(1); %#ok<AGROW>
    traj_y(end+1,1) = state(2); %#ok<AGROW>

    % detect finish-line crossing by s wrap
    if (last_s_progress > track_length*0.9) && (s_progress < track_length*0.1) && (t > 1.0)
        laps_completed = laps_completed + 1;
        fprintf('Finish line crossed. Laps completed = %d at t=%.2fs\n', laps_completed, t);
        break
    end
    last_s_progress = s_progress;

    % ---- update drawing ----
    c = cos(state(3)); s = sin(state(3));
    R = [c -s; s c];

    kart_world = (R*body_local')' + state(1:2)';
    set(body_patch, 'XData', kart_world(:,1), 'YData', kart_world(:,2));

    canopy_world = (R*canopy_local')' + state(1:2)';
    set(canopy_patch, 'XData', canopy_world(:,1), 'YData', canopy_world(:,2));

    for i=1:4
        off = wheel_offsets_local(i,:)';
        wp = R*off + state(1:2);
        set(wheel_patches(i), 'XData', wp(1) + wheel_radius*cos(th), ...
            'YData', wp(2) + wheel_radius*sin(th));
    end

    if offtrack_timer > 0
        set(body_patch, 'FaceColor', [1 1 0]); % flash yellow
    else
        set(body_patch, 'FaceColor', [0.9 0.12 0.12]);
    end

    set(traj_line, 'XData', traj_x, 'YData', traj_y);

    controller_type = ternary(use_time_optimal_controller, 'Time-optimal', 'Baseline');
    set(info_txt, 'String', sprintf('Time: %.1fs\nPenalty: %.2f\nController: %s\nTarget speed: %.1f m/s', ...
        t, penalty_offtrack, controller_type, vx_ref));

    drawnow limitrate;
    pause(0.001);

    t = t + dt;
end

fprintf('Lap complete!\n');
fprintf('Total off-track penalty = %.3f\n', penalty_offtrack);

end

% ============================================================
% Helper math functions
% ============================================================

% Wraps an angle to the range [-pi, pi]
% Input:
%   th - angle in radians
% Output:
%   th_wrapped - angle wrapped to [-pi, pi]
function th_wrapped = angle_wrap(th)
th_wrapped = mod(th + pi, 2*pi) - pi;
end

% Evaluates a cubic polynomial at given points
% Inputs:
%   coef - polynomial coefficients [c0, c1, c2, c3]
%   x - evaluation points
%   x0 - reference point
% Output:
%   y - polynomial values at x: y = c0 + c1*(x-x0) + c2*(x-x0)^2 + c3*(x-x0)^3
function y = poly3_eval(coef, x, x0)
dx = (x - x0);
y = coef(1) + coef(2).*dx + coef(3).*dx.^2 + coef(4).*dx.^3;
end

% Computes cubic spline coefficients between two points with specified derivatives
% Inputs:
%   (xa,ya) - start point
%   da - slope at start
%   (xb,yb) - end point
%   db - slope at end
% Output:
%   coef - polynomial coefficients [c0, c1, c2, c3] for spline segment
function coef = splineinter(xa, ya, da, xb, yb, db)
dx = xb - xa;
A = [dx^2,   dx^3;
    2*dx, 3*dx^2];
b = [yb - ya - da*dx;
    db - da];

if abs(det(A)) < 1e-12
    c3 = 0.0; c4 = 0.0;
else
    sol = A\b;
    c3 = sol(1); c4 = sol(2);
end

coef = [ya, da, c3, c4];
end

% Computes arc-length parameterization of a curve
% Input:
%   pts - n×2 matrix of [x,y] points defining the curve
% Outputs:
%   s_norm - normalized arc-length parameter [0,1]
%   xs - x coordinates
%   ys - y coordinates
function [s_norm, xs, ys] = arclength_param(pts)
pts = double(pts);

diffs = [0 0; diff(pts,1,1)];
keep = sqrt(sum(diffs.^2,2)) > 1e-9;
keep(1) = true;
pts = pts(keep,:);

xs = pts(:,1);
ys = pts(:,2);

if xs(end) ~= xs(1) || ys(end) ~= ys(1)
    xs(end+1,1) = xs(1);
    ys(end+1,1) = ys(1);
end

seg = sqrt(diff(xs).^2 + diff(ys).^2);
s = [0; cumsum(seg)];
s_norm = s / s(end);
end

% Interpolates position on track at a given arc-length distance
% Inputs:
%   sq - query arc-length distance
%   s_dist - cumulative distances
%   waypts - n×2 waypoint matrix
% Output:
%   p - interpolated 2D position [x; y] on track
function p = interp_track(sq, s_dist, waypts)
track_length = s_dist(end);

while sq < 0
    sq = sq + track_length;
end
while sq > track_length
    sq = sq - track_length;
end

idx = find(s_dist <= sq, 1, 'last');
if isempty(idx), idx = 1; end
if idx >= numel(s_dist), idx = numel(s_dist)-1; end

s0 = s_dist(idx);
s1 = s_dist(idx+1);
p0 = waypts(idx,:)';
p1 = waypts(idx+1,:)';

if s1 > s0
    alpha = (sq - s0) / (s1 - s0);
else
    alpha = 0.0;
end

p = (1-alpha)*p0 + alpha*p1;
end

% Projects a point onto the track centerline and computes arc-length distance
% Inputs:
%   p - 2D point [x; y] to project
%   s_dist - cumulative distances
%   waypts - n×2 waypoint matrix
% Outputs:
%   s_hat - arc-length distance along track
%   p_hat - projected point [x; y] on track
function [s_hat, p_hat] = project_to_track(p, s_dist, waypts)
p = double(p(:));
best_d2 = inf;
best_s = 0.0;
best_pt = waypts(1,:)';

for k = 1:(size(waypts,1)-1)
    p0 = waypts(k,:)';
    p1 = waypts(k+1,:)';
    v = p1 - p0;
    vv = dot(v,v);
    if vv < 1e-12
        continue
    end
    w = p - p0;
    alpha = dot(w,v)/vv;
    alpha = min(max(alpha,0.0),1.0);

    proj = p0 + alpha*v;
    d2 = dot(p-proj, p-proj);

    if d2 < best_d2
        best_d2 = d2;
        best_pt = proj;
        seg_len = norm(v);
        best_s = s_dist(k) + alpha*seg_len;
    end
end

track_length = s_dist(end);
while best_s > track_length
    best_s = best_s - track_length;
end
while best_s < 0
    best_s = best_s + track_length;
end

s_hat = best_s;
p_hat = best_pt;
end

% ============================================================
% Track construction (FIXED: matrices instead of Map)
% ============================================================

% Constructs a custom race track centerline from control points using cubic splines
% Outputs:
%   tx - x coordinates of centerline
%   ty - y coordinates of centerline
function [tx, ty] = make_centerline_from_custom_track()
% control points as 5x2 matrices
p = [542 744;
    978 950;
    883 417;
    40 286;
    229 823];

q = [635 922;
    978 550;
    341 144;
    40 756;
    433 719];

s_ = [613 696;
    895 850;
    843 485;
    118 290;
    203 742];

v = [702 877;
    895 550;
    315 219;
    118 692;
    401 643];

r = [824 1031;
    947  453;
    134  116;
    126  860;
    497  705];

w = [822 942;
    873 504;
    200 190;
    155 759;
    497 612];

d_slp = [0; 1; -0.5; 0; 0];
e_slp = [0; 1; 0; 0; 0];

% optional scaling
scale = 0.1;
if abs(scale - 1.0) > eps
    p = p*scale; q = q*scale; s_ = s_*scale; v = v*scale; r = r*scale; w = w*scale;
end

% outer slopes
d_1 = zeros(5,1);
d_2 = zeros(5,1);
for i=1:5
    denom = (q(i,1) - p(i,1));
    if abs(denom) < 1e-12
        d_1(i) = 0.0;
    else
        d_1(i) = (q(i,2) - p(i,2)) / denom;
    end
    d_2(i) = d_1(i);
end
d_1(2) = 4; d_2(2) = -2;
d_1(4) = 4; d_2(4) = -4;

% inner slopes
e_1 = zeros(5,1);
e_2 = zeros(5,1);
for i=1:5
    denom = (v(i,1) - s_(i,1));
    if abs(denom) < 1e-12
        e_1(i) = 0.0;
    else
        e_1(i) = (v(i,2) - s_(i,2)) / denom;
    end
    e_2(i) = e_1(i);
end
e_1(2) = 4; e_2(2) = -4;
e_1(4) = 4; e_2(4) = -4;

Nseg = 30;
outer_pts = [];
inner_pts = [];

for i=1:5
    ip1 = mod(i,5) + 1;

    % OUTER q{i} -> r{i}
    coef1 = splineinter(q(i,1), q(i,2), d_1(i), r(i,1), r(i,2), d_slp(i));
    x0 = q(i,1);
    xspan = linspace(q(i,1), r(i,1), Nseg);
    yspan = poly3_eval(coef1, xspan, x0);
    outer_pts = [outer_pts; [xspan(:), yspan(:)]]; %#ok<AGROW>

    % OUTER r{i} -> p{ip1}
    coef2 = splineinter(r(i,1), r(i,2), d_slp(i), p(ip1,1), p(ip1,2), d_2(ip1));
    x0 = r(i,1);
    xspan = linspace(r(i,1), p(ip1,1), Nseg);
    yspan = poly3_eval(coef2, xspan, x0);
    outer_pts = [outer_pts; [xspan(:), yspan(:)]]; %#ok<AGROW>

    % INNER v{i} -> w{i}
    coef3 = splineinter(v(i,1), v(i,2), e_1(i), w(i,1), w(i,2), e_slp(i));
    x0 = v(i,1);
    xspan = linspace(v(i,1), w(i,1), Nseg);
    yspan = poly3_eval(coef3, xspan, x0);
    inner_pts = [inner_pts; [xspan(:), yspan(:)]]; %#ok<AGROW>

    % INNER w{i} -> s{ip1}
    coef4 = splineinter(w(i,1), w(i,2), e_slp(i), s_(ip1,1), s_(ip1,2), e_2(ip1));
    x0 = w(i,1);
    xspan = linspace(w(i,1), s_(ip1,1), Nseg);
    yspan = poly3_eval(coef4, xspan, x0);
    inner_pts = [inner_pts; [xspan(:), yspan(:)]]; %#ok<AGROW>
end

[sO, xO_s, yO_s] = arclength_param(outer_pts);
[sI, xI_s, yI_s] = arclength_param(inner_pts);

Nu = 600;
u = linspace(0,1,Nu);

x_outer_u = interp1(sO, xO_s, u, 'linear');
y_outer_u = interp1(sO, yO_s, u, 'linear');

x_inner_u = interp1(sI, xI_s, u, 'linear');
y_inner_u = interp1(sI, yI_s, u, 'linear');

x_center = 0.5*(x_outer_u + x_inner_u);
y_center = 0.5*(y_outer_u + y_inner_u);

x_center_s = movmean_wrap(x_center, 5);
y_center_s = movmean_wrap(y_center, 5);

x_center_s(end) = x_center_s(1);
y_center_s(end) = y_center_s(1);

pts_center = [x_center_s(:), y_center_s(:)];
[sC, xC_s, yC_s] = arclength_param(pts_center);
totalL = sC(end);

Nsamp = 500;
s_query = linspace(0,totalL,Nsamp);
tx = interp1(sC, xC_s, s_query, 'linear')';
ty = interp1(sC, yC_s, s_query, 'linear')';

tx(end) = tx(1);
ty(end) = ty(1);
end

% Computes moving average with wraparound for periodic data
% Inputs:
%   arr - 1D array of values
%   win - window size for averaging
% Output:
%   out - smoothed array with same length as input
function out = movmean_wrap(arr, win)
arr = arr(:)';
pad = floor(win/2);
arr_pad = [arr(end-pad+1:end), arr, arr(1:pad)];
c = cumsum([0, arr_pad]);
out = (c(win+1:end) - c(1:end-win)) / win;
end

% ============================================================
% Vehicle dynamics (bicycle model)
% ============================================================

% Computes vehicle state derivatives using bicycle model dynamics
% Inputs:
%   state - [x; y; psi; vx; vy; r] vehicle state
%   delta - steering angle
%   ax_cmd - longitudinal acceleration
%   params - vehicle parameters struct
% Output:
%   xdot - state derivatives [x_dot; y_dot; psi_dot; vx_dot; vy_dot; r_dot]
function xdot = bicycle_dynamics(state, delta, ax_cmd, params)
x   = state(1); %#ok<NASGU>
y   = state(2); %#ok<NASGU>
psi = state(3);
vx  = state(4);
vy  = state(5);
r   = state(6);

m  = params.m;
Iz = params.Iz;
lf = params.lf;
lr = params.lr;
Cf = params.Cf;
Cr = params.Cr;

vx_safe = max(vx, 0.5);

alpha_f = delta - atan2(vy + lf*r, vx_safe);
alpha_r =       - atan2(vy - lr*r, vx_safe);

Fyf = Cf * alpha_f;
Fyr = Cr * alpha_r;

x_dot   = vx*cos(psi) - vy*sin(psi);
y_dot   = vx*sin(psi) + vy*cos(psi);
psi_dot = r;

vx_dot = ax_cmd + (-Fyf*sin(delta))/m + r*vy;
vy_dot = (Fyf*cos(delta) + Fyr)/m - r*vx;
r_dot  = (lf*Fyf*cos(delta) - lr*Fyr)/Iz;

xdot = [x_dot; y_dot; psi_dot; vx_dot; vy_dot; r_dot];
end

% ============================================================
% Time-optimal controller helpers
% ============================================================

% Estimates track curvature at a given arc-length position using three-point circle fit
% Inputs:
%   s - arc-length position
%   s_dist - cumulative distances
%   waypts - n×2 waypoint matrix
%   ds - distance offset for finite difference
% Output:
%   kappa - estimated curvature (1/radius)
function kappa = curvature_at_s(s, s_dist, waypts, ds)
p1 = interp_track(s - ds, s_dist, waypts);
p2 = interp_track(s,       s_dist, waypts);
p3 = interp_track(s + ds,  s_dist, waypts);

a = norm(p2 - p1);
b = norm(p3 - p2);
c = norm(p3 - p1);

v1 = p2 - p1;
v2 = p3 - p1;
area = 0.5 * abs(v1(1)*v2(2) - v1(2)*v2(1));

denom = a*b*c;
if denom < 1e-9
    kappa = 0.0;
else
    kappa = 4.0 * area / denom;
end
end

% Computes maximum safe velocity from track curvature and lateral acceleration limit
% Inputs:
%   kappa - track curvature (1/radius)
%   a_lat_max - maximum lateral acceleration limit
% Output:
%   vmax - maximum velocity satisfying lateral acceleration constraint
function vmax = vmax_from_lateral_acc(kappa, a_lat_max)
kabs = abs(kappa);
if kabs < 1e-12
    vmax = 1e6;
else
    vmax = sqrt(max(a_lat_max / kabs, 0.0));
end
end

% Time-optimal racing controller combining pure pursuit steering with curvature-based speed planning
% Inputs:
%   state - [x;y;psi;vx;vy;r] vehicle state
%   s_progress - current arc-length position
%   s_dist - cumulative distances
%   waypts - n×2 waypoint matrix
%   vx_ref - reference velocity
%   max_accel - max acceleration
%   max_brake - max braking
%   look_ahead - look-ahead distance for path tracking
%   k_delta - steering gain
%   max_steer - max steering angle
%   a_lat_max - max lateral acceleration
%   nsamples - number of samples for speed planning
% Outputs:
%   delta - steering angle command
%   ax_cmd - longitudinal acceleration command
function [delta, ax_cmd ] = time_optimal_controller( ...
    state, s_progress, s_dist, waypts, ...
    vx_ref, max_accel, max_brake, ...
    look_ahead, k_delta, max_steer, a_lat_max, nsamples)

x = state(1); y = state(2); psi = state(3);
vx = state(4);

% steering
s_target = s_progress + look_ahead;
track_length = s_dist(end);
while s_target > track_length, s_target = s_target - track_length; end
while s_target < 0,          s_target = s_target + track_length; end

p_target = interp_track(s_target, s_dist, waypts);
dir_vec = p_target - [x; y];
desired_psi = atan2(dir_vec(2), dir_vec(1));
heading_err = angle_wrap(desired_psi - psi);

delta = k_delta * heading_err;
delta = min(max(delta, -max_steer), max_steer);

% speed planning
s_vals = linspace(s_progress, s_progress + max(look_ahead*2.0, 1.0), nsamples);
vmax_vals = zeros(size(s_vals));
ds = max(1.0, look_ahead/10.0);

for i=1:numel(s_vals)
    kappa = curvature_at_s(s_vals(i), s_dist, waypts, ds);
    vmax_vals(i) = vmax_from_lateral_acc(kappa, a_lat_max);
end

planned_vmax = min(vmax_vals);
vx_target = min(vx_ref, planned_vmax);

k_vx_local = 2.0;
ax_cmd = k_vx_local * (vx_target - vx);
ax_cmd = min(max(ax_cmd, max_brake), max_accel);
end

% Ternary conditional operator (returns a if cond is true, else b)
% Inputs:
%   cond - boolean condition
%   a - value if true
%   b - value if false
% Output:
%   out - selected value
function out = ternary(cond, a, b)
if cond, out = a; else, out = b; end
end
