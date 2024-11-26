function varargout = scaled_quadratic(x,a)
    % Scaled quadratic, parametrised by a > 0
    % Large a -> poorly scaled
    % An interesting start point is x = [1; a]
    % Unique local/global minimum is f=0 at [0;0]
    % Lindon Roberts, 2019
    
    % Call test functions as:
    %   a = 10;
    %   objective = @(x)scaled_quadratic(x,a);
    %   f         = objective(x);  % function value only
    %   [f, g]    = objective(x);  % function value and gradient
    %   [f, g, H] = objective(x);  % function value, gradient and Hessian
    
    if a <= 0
        error('Parameter a must be strictly positive');
    end
    
    if nargout > 3
        error('Invalid number of output arguments');
    end
    
    % Function value
    f = (a*x(1).^2 + x(2).^2) / 2;
    varargout{1} = f;
    
    if nargout > 1
        % Gradient
        g = [a*x(1); x(2)];
        varargout{2} = g;
    end
    
    if nargout > 2
        % Hessian
        H = [a, 0; 0, 1];
        varargout{3} = H;
    end
end