close all
clear
clc

%%

M1 = dlmread('./../log_files/GT_log.txt'); 
M2 = dlmread('./../log_files/state_log.txt'); 
M3 = dlmread('./../log_files/gps_log.txt'); 
M4 = dlmread('./../log_files/encoder_log.txt'); 
M5 = dlmread('./../log_files/imu_log.txt');


%%
% GT
x_gt = M1(:,1);
y_gt = M1(:,2);
z_gt = M1(:,3);
qw_gt = M1(:,4);
qx_gt = M1(:,5);
qy_gt = M1(:,6);
qz_gt = M1(:,7);
vx_gt = M1(:,8);
vy_gt = M1(:,9);
vz_gt = M1(:,10);
eul_gt = quat2eul([qw_gt qx_gt qy_gt qz_gt]);
time = M1(:,11) - M1(1,11);

% EKF
x = M2(:,1);
y = M2(:,2);
z = M2(:,3);
qw = M2(:,4);
qx = M2(:,5);
qy = M2(:,6);
qz = M2(:,7);
vx = M2(:,8);
vy = M2(:,9);
vz = M2(:,10);
eul = quat2eul([qw qx qy qz]);

% GPS
x_gps = M3(:,1);
y_gps = M3(:,2);
z_gps = M3(:,3);

% Encoder
vx_enc = M4(:,1);
vy_enc = M4(:,2);
vz_enc = M4(:,3);

% IMU
ax = M5(:,1);
ay = M5(:,2);
az = M5(:,3);
gx = M5(:,4);
gy = M5(:,5);
gz = M5(:,6);
ow = M5(:,7);
ox = M5(:,8);
oy = M5(:,9);
oz = M5(:,10);
eul_imu = quat2eul([ow ox oy oz]);

%% Plots
figure(1)
h1 = plot(x_gt,y_gt,'b-','LineWidth',1.2);
hold on
h2 = plot(x,y,'r-','LineWidth',1.2);
h3 = plot(x_gps,y_gps,'g-','LineWidth',1.2);
hold off
legend([h1, h2, h3], 'Performed (GT)', 'EKF', 'GPS')
xlabel('x (m)')
ylabel('y (m)')
grid on


%
figure(2)
subplot(2,1,1)
plot(time,x_gt,'b-','LineWidth',1.2);
hold on
plot(time,x,'r-','LineWidth',1.2);
plot(time,x_gps,'g-','LineWidth',1.2);
ylabel('distance (m)')
title('Displacement x')
hold off
grid on
legend('Performed (GT)', 'EKF', 'GPS')

subplot(2,1,2)
plot(time,y_gt,'b-','LineWidth',1.2);
hold on
plot(time,y,'r-','LineWidth',1.2);
plot(time,y_gps,'g-','LineWidth',1.2);
ylabel('distance (m)')
title('Displacement y')
hold off
xlabel('t (s)')
grid on

% subplot(3,1,3)
% plot(time,z_gt,'b-','LineWidth',1.2);
% hold on
% plot(time,z,'r-','LineWidth',1.2);
% plot(time,z_gps,'g-','LineWidth',1.2);
% ylabel('distance (m)')
% title('Displacement z')
% hold off
% xlabel('t (s)')

%
figure(3)
% subplot(3,1,1)
% plot(time,eul_gt(:,2),'b-','LineWidth',1.2);
% hold on
% plot(time,eul(:,2),'r-','LineWidth',1.2);
% plot(time,eul_imu(:,2),'g-','LineWidth',1.2);
% ylabel('$\phi$ (rad)','fontsize',13,'interpreter','latex')
% title('Roll angle')
% hold off
% 
% subplot(3,1,2)
% plot(time,eul_gt(:,3),'b-','LineWidth',1.2);
% hold on
% plot(time,eul(:,3),'r-','LineWidth',1.2);
% plot(time,eul_imu(:,3),'g-','LineWidth',1.2);
% ylabel('$\theta$ (rad)','fontsize',13,'interpreter','latex')
% title('Pitch angle')
% hold off

% subplot(3,1,3)
plot(time,eul_gt(:,1),'b-','LineWidth',1.2);
hold on
plot(time,eul(:,1),'r-','LineWidth',1.2);
plot(time,eul_imu(:,1),'g-','LineWidth',1.2);
ylabel('$\psi$ (rad)','fontsize',13,'interpreter','latex')
title('Yaw angle')
hold off
xlabel('t (s)')
grid on
legend('Performed (GT)', 'EKF', 'IMU')

%
figure(4)
% subplot(3,1,1)
plot(time,vx_gt,'b-','LineWidth',1.2);
hold on
plot(time,vx,'r-','LineWidth',1.2);
plot(time,vx_enc,'g-','LineWidth',1.2);
ylabel('velocity (m/s)')
title('Velocity x')
hold off
xlabel('t (s)')
grid on
legend('Performed (GT)', 'EKF', 'Encoder')

% subplot(3,1,2)
% plot(time,vy_gt,'b-','LineWidth',1.2);
% hold on
% plot(time,vy,'r-','LineWidth',1.2);
% plot(time,vy_enc,'g-','LineWidth',1.2);
% ylabel('velocity (m/s)')
% title('Velocity y')
% hold off
% 
% subplot(3,1,3)
% plot(time,vz_gt,'b-','LineWidth',1.2);
% hold on
% plot(time,vz,'r-','LineWidth',1.2);
% plot(time,vz_enc,'g-','LineWidth',1.2);
% ylabel('velocity (m/s)')
% title('Velocity z')
% hold off
% xlabel('t (s)')
