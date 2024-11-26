% Run Newton (with linesearch)
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
    %x = [0.4; 0.2]; % start in convex region near the solution
    %x = [-0.9; 1]; % start in nonconvex region
    %x = [-50; 40]; % start very far away
    xmin = [1;1]; % true minimiser
end
nhistory = 5; % use last N iterates to check asymptotic rate

% True info
[fmin, gmin, Hmin] = objfun(xmin); % true minimum
kappa = cond(Hmin);

% Solver settings
damp = 1; % 1 = damped Newton, 0 = regular Newton (fails when nonconvex)
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
[f, g, H] = objfun(x);
fs(1,:) = f;
norm_gs(1,:) = norm(g);

k = 1;
fprintf('  k  |  f(xk)       |  ||grad|| \n');
fprintf('--------------------------------\n');
fprintf('  %i  |  %.4e  |  %.4e  \n', k-1, f, norm(g));
while k <= max_iterations && norm(g) >= tol_g
    lambda_min = min(eig(H)); % note: can speed up by using eigs
    if lambda_min < 1e-5 && damp
        fprintf('Nonconvex region (lambda_min = %g), using damped Newton\n', lambda_min);
        Htmp = H + 1.01*abs(lambda_min)*eye(n); % nonconvex - damped Newton
    else
        Htmp = H; % convex - regular Newton
    end
    s = -Htmp\g; % Newton direction
    % Backtracking Armijo linesearch
     alpha = alpha0;
     xtest = x + alpha*s;
       while objfun(xtest) > f + beta*alpha*(g'*s)
        alpha = tau*alpha;
         xtest = x + alpha*s;
     end
     x = xtest;
     
    [f, g, H] = objfun(x);
    fprintf('  %i  |  %.4e  |  %.4e  \n', k, f, norm(g));
    % Save info
    xs(k+1,:) = x;
    fs(k+1,:) = f;
    norm_gs(k+1,:) = norm(g);
    k = k + 1;
end
fprintf('Finished after %g iterations\n', k-1);
xs = xs(1:k, :);
fs = fs(1:k, :);
norm_gs = norm_gs(1:k, :);
xdists = zeros(k,1);
for i=1:k
    xdists(i) = norm(xs(i,:) - xmin);
end

% Check asymptotic order of convergence
if numel(xdists) < nhistory
    asym_xdists = xdists;
else
    asym_xdists = xdists(end-nhistory:end);
end
fit_xdists = polyfit(log(asym_xdists(1:end-1)), log(asym_xdists(2:end)), 1);
fprintf('Iterates converge with order %1.2f\n', fit_xdists(1));

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
contour(X,Y,Z,30)
hold on
plot(xs(:,1), xs(:,2), 'r.-', 'MarkerSize', 15);
xlabel('x1');
ylabel('x2');
axis equal
hold off

subplot(2,2,2);
semilogy(fs-fmin, 'b-', 'Linewidth', 2);
hold on
xlabel('Iteration');
ylabel('Objective value - fmin');
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
grid on
hold off
