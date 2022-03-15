function [InitialObservation, LoggedSignal] = ...
    QuadcopterResetFunction(radius,deviation)
% Reset function to place quadcopter environment into a random initial
%state defined as a hover at a given altitude, with no roll, pitch, or yaw,
% and no linear velocity; however, there is a random angular velocity
% caused by a disturbance which the RL program must overcome.

% Initial position
spherical_1 = rand(1)*2*pi;
spherical_2 = rand(1)*2*pi;
x1 = radius*cos(spherical_1)*sin(spherical_2);
y1 = radius*sin(spherical_1)*sin(spherical_2);
z1 = radius*cos(spherical_2);
r_0 = [x1; y1; z1]; %Initial pos relative to target - m

rdot_0 = [0; 0; 0]; %initial velocity [x; y; z] in inertial frame - m/s
E_0 = [0; 0; 0]; %initial [pitch;roll;yaw] relative to inertial frame -deg

sim_time = 0;

%Add initial random roll, pitch, and yaw rates
Edot_0 = ((2* deviation * rand(3,1) - deviation)*pi/180);
%Edot_0 = [0; 0; 0];

% Return initial environment state variables as logged signals.
LoggedSignal.State = [r_0;rdot_0;E_0;Edot_0;sim_time];
InitialObservation = LoggedSignal.State;
end
