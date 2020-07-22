# ekf_filter

This package has a extended Kalman filter (EKF) that can be used in several mobile robots as a robust states estimator.

In this case, was developed to estimate states of a ground vehicle. However, can be adapted to other models and vehicles (including aerial).

There are two codes inside of the package:

1 - EKF with 15 states: positions (x,y,z); velocities (vx,vy,vz); orientation (roll,pitch,yaw); Bias (accelerometer and gyroscope).

  Sensors used: GPS (x,y,z), encoder(Vx), IMU (accelerometer, gyroscope and full orientation estimation)
  
2 - EKF with 12 states: positions (x,y); velocities (vx); orientation (roll,pitch,yaw); Bias (accelerometer and gyroscope).

  Sensors used: GPS (x,y,z), encoder(Vx), IMU (accelerometer, gyroscope and full orientation estimation)


# Complete Explanation - TO DO
