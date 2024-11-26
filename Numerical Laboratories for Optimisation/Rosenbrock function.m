function varargout = rosenbrock(x)
    % Rosenbrock test function
    % An interesting start point is x = [-1; 0.8]
    % Unique local/global minimum is f=0 at [1;1]
    % Lindon Roberts, 2019
    
    % Call test functions as:
    %   f         = rosenbrock(x);  % function value only
    %   [f, g]    = rosenbrock(x);  % function value and gradient
    %   [f, g, H] = rosenbrock(x);  % function value, gradient and Hessian
    
    if nargout > 3
        error('Invalid number of output arguments');
    end
    
    % Function value
    a = 10;
    f = a*(x(2)-x(1).^2).^2 + (x(1)-1).^2;
    varargout{1} = f;
    
    if nargout > 1
        % Gradient
        g = [2*(x(1)-1)-4*a*x(1).*(x(2)-x(1).^2); 2*a*(-x(1).^2 + x(2))];
        varargout{2} = g;
    end
    
    if nargout > 2
        % Hessian
        H = [2 + 12*a*x(1).^2 - 4*a*x(2), -4*a*x(1); -4*a*x(1), 2*a];
        varargout{3} = H;
    end
end