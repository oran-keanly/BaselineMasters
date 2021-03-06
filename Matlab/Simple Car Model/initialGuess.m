%clearvars -except out
writematrix(out.Steer_angle,'steer.csv');
writematrix(out.Yaw_rate, 'yawRate.csv');
writematrix(out.X_Velocity_VehicleAxis,'longVel.csv');
writematrix(out.Y_Velocity_VehicleAxis,'latVel.csv');

distance_steps = zeros(length(out.X_Position), 1);
X_pos = out.X_Position(1);
Y_pos = out.Y_Position(1);

for x = 2:length(out.X_Position)
    distance_steps(x) = sqrt(((X_pos - out.X_Position(x))^2) + ((Y_pos - out.Y_Position(x))^2));
    X_pos = out.X_Position(x);
    Y_pos = out.Y_Position(x);
end
cumalative_distance = cumsum(distance_steps, 1);

writematrix(cumalative_distance,'distance.csv');
writematrix(curvature,'Curvature.csv');
plot(curvature);
    
   
    
    
    
