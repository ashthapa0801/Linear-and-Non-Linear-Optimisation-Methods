% Run steepest descent (with linesearch)
% Lindon Roberts, 2019

clear, close all

% Problem and x0
if 1
    a = 10;
    objfun = @(x)scaled_quadratic(x,a);
    x = [1; a];
    xmin = [0;0]; % true minimiser
else
    objfun = @(x)rosenbrock(x);
    x = [-1; 0.8];
    %x = [-1.2; 1];
    %x = [0.4; 0.2]; % start close to the solution
    xmin = [1;1]; % true minimiser
end
nhistory = 10; % use last N iterates to check asymptotic rate

% True info
[fmin, gmin, Hmin] = objfun(xmin); % true minimum
kappa = cond(Hmin);

% Solver settings
max_iterations = 800;
tol_g = 1e-5; % termination condition ||gradient|| <= tol
alpha0 = 1; % initial step length
tau = 0.5; % backtracking parameter
beta = 0.001; % for Armijo condition

% Useful data to see progress of solver
n = numel(x);
xs = zeros(max_iterations+1, n); % iterate
fs = zeros(max_iterations+1,1); % objective value at each iteration
norm_gs = zeros(max_iterations+1,1); % ||gradient|| at each iteration

% Set initial data
xs(1,:) = x;
[f, g] = objfun(x);
fs(1,:) = f;
norm_gs(1,:) = norm(g);

k = 1;
fprintf('  k  |  f(xk)       |  ||grad|| \n');
fprintf('--------------------------------\n');
fprintf('  %i  |  %.4e  |  %.4e  \n', k-1, f, norm(g));
while k <= max_iterations && norm(g) >= tol_g
    s = -g; % steepest descent direction
    % Backtracking Armijo linesearch
    alpha = alpha0;
    xtest = x + alpha*s;
    while objfun(xtest) > f + beta*alpha*(g'*s)
        alpha = tau*alpha;
        xtest = x + alpha*s;
    end
    x = xtest;
    [f, g] = objfun(x);
    if mod(k, 10) == 0
        fprintf('  %i  |  %.4e  |  %.4e  \n', k, f, norm(g));
    end
    % Save info
    xs(k+1,:) = x;
    fs(k+1,:) = f;
    norm_gs(k+1,:) = norm(g);
    k = k + 1;
end
fprintf('  %i  |  %.4e  |  %.4e   <- finished\n', k-1, f, norm(g));
fprintf('Finished after %g iterations\n', k-1)
xs = xs(1:k, :);
fs = fs(1:k, :);
norm_gs = norm_gs(1:k, :);
xdists = zeros(k,1);
for i=1:k
    xdists(i) = norm(xs(i,:) - xmin);
end

% Check asymptotic order of convergence
if numel(fs) < nhistory
    asym_fs = fs;
else
    asym_fs = fs(end-nhistory:end);
end
fit_fs = polyfit(log(asym_fs(1:end-1)), log(asym_fs(2:end)), 1);
fprintf('Objective values converge with order %1.2f\n', fit_fs(1));

%=====================================================
% Plot iterates, objective decrease, gradient decrease
%=====================================================

subplot(2,2,1);
npts = 30;
xplt = linspace(min(min(xs)), max(max(xs)), npts);
yplt = xplt;
[X,Y] = meshgrid(xplt,yplt);
Z = zeros(npts,npts);
for i=1:npts
    for j=1:npts
        Z(i,j) = log(objfun([X(i,j); Y(i,j)]));
    end
end
contour(X,Y,Z,20)
hold on
axis equal
plot(xs(:,1), xs(:,2), 'r.-', 'MarkerSize', 15);
xlabel('x1');
ylabel('x2');
hold off

subplot(2,2,2);
semilogy(fs-fmin, 'b-', 'Linewidth', 2);
hold on
xlabel('Iteration');
ylabel('Objective value - fmin');
rho = ((kappa-1)/(kappa+1))^2;
fprintf('rho_{SD} convergence rate <= %g (from kappa = %g)\n', rho, kappa);
semilogy(1:numel(fs), (fs(1)-fmin) * rho.^(0:1:numel(fs)-1), 'r--', 'Linewidth', 2);
grid on
hold off

subplot(2,2,3);
semilogy(norm_gs, 'b-', 'Linewidth', 2);
xlabel('Iteration');
ylabel('Norm of gradient');
grid on

subplot(2,2,4);
semilogy(xdists, 'b-', 'Linewidth', 2);
hold on
xlabel('Iteration');
ylabel('||x-x*||');
if exist('a')
    xs_rate = zeros(numel(xdists),1);
    xs_rate(1) = xdists(1);
    for i=2:numel(fs)
        xs_rate(i) = (a-1)/(a+1) * xs_rate(i-1);
    end
    semilogy(xs_rate, 'r--', 'Linewidth', 2);
end
grid on
hold off
