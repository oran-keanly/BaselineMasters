delta_s = 0.1;
B_f = 10;
B_r = 10;
C_f = 1.9;
C_r = 1.9;
D_r = 1;
D_f = 1;
C_m = 5;
C_r0 = 0.1;
C_r2 = 0.2;
FN_r = 43.6;
FN_f = 54.5;
l_f = 2.5;
l_r = 2;
mass = 10;
I_z = 0.152;
p_tv = 0;

Curvature = zeros([200, 2]);
delta_steer = zeros([200, 2]);
delta_T = zeros([200, 2]);
delta_T(1, 2) =  1;

for i = 1:length(Curvature)
    Curvature(i, 1) = delta_s*i;
    delta_steer(i, 1) = delta_s*i;
    delta_T(i, 1) = delta_s*i;
end
