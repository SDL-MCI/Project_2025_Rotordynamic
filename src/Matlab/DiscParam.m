% Parameters (you can change these values)
density = 2700;    % in kg/m^3 (e.g., steel)
diameter = 0.177;    % in meters
thickness = 0.016;  % in meters

% Derived quantities
radius = diameter / 2;           % Radius in meters
volume = pi * radius^2 * thickness;  % Volume of the disc in m^3
mass = density * volume;             % Mass in kg

% Moments of inertia
Ip = 0.5 * mass * radius^2;      % Polar mass moment of inertia (about central axis)
Id = 0.25 * mass * radius^2;  % Diametral mass moment of inertia (about a diameter)

% Display results
fprintf('Mass: %.4f kg\n', mass);
fprintf('Polar Mass Moment of Inertia: %.6f kg·m²\n', Ip);
fprintf('Diametral Mass Moment of Inertia: %.6f kg·m²\n', Id);
