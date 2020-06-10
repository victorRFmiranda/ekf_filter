#include "EKF_class.h"

#define PI 3.1415926535

using namespace std;
using namespace Eigen;



// States
// x, y, z
// roll, pitch, yaw
// vx, vy, vz (body)
// bgx, bgy, bgz (bias imu gyro)
// bax, bay, baz (bias imu accelerometer)

#define N_STATES 15

// constructor
EKF_class::EKF_class(VectorXd initial_pos, double Frequency, double init_angle){
  VectorXd state_init(N_STATES), imu_init(6), read_pose_init(6);
  state_init.setZero();
  read_pose_init.setZero();
  imu_init << 0,0,0,  0,0,9.81;

  state_init.block(0,0,3,1) = initial_pos;
  state_init(5) = init_angle;
  read_pose_init.block(0,0,3,1) = initial_pos;

  states = state_init; // (x, y, z, phi, theta, psi, vx, vy, vz, bgx, bgy, bgz, bax, bay, baz)
  imu_data = imu_init; // (ang_vel, acc_linear)
  read_pose = read_pose_init; // (x, y, z, phi, theta, psi)

  dt = 1.0/Frequency;

  enable_slam = false;

  rangeF_data = 0.0;

  rangeF_data_prev = 0.0;
  rf_time_prev = 0.0;

  //filter_mutex.unlock();

  // Jacobian of the measurement model
  H = MatrixXd::Zero(9,N_STATES);
  H << 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
       0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
       0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,

       0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
       0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
       0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,

       0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
       0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
       0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0;


  // Jacobian of the measurement model - YOLO (gate detection)
  H_yolo = MatrixXd::Zero(6,N_STATES);
  H_yolo << 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
		  	    0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
			      0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,

			      0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
			      0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
			      0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0;


  // Jacobian of the measurement model - YOLO (gate detection)
  H_slam = MatrixXd::Zero(6,N_STATES);
  H_slam << 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
		  	    0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
			      0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,

			      0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
			      0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
			      0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0;



  // Model Propagation Covariance
  Q = MatrixXd::Zero(N_STATES,N_STATES);
  Q << 0.00011, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
       0.0, 0.00011, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
       0.0, 0.0, 0.00016, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,

       0.0, 0.0, 0.0, 0.00001, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
       0.0, 0.0, 0.0, 0.0, 0.00001, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
       0.0, 0.0, 0.0, 0.0, 0.0, 0.00001, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,

       0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0001, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
       0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0001, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
       0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0001, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,

       0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.000000001, 0.0, 0.0, 0.0, 0.0, 0.0,
       0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.000000001, 0.0, 0.0, 0.0, 0.0,
       0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.000000001, 0.0, 0.0, 0.0,

       0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.000000001, 0.0, 0.0,
       0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.000000001, 0.0,
       0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.000000001;


  // IMU covariance
  Q_bar = MatrixXd::Zero(6,6);
  Q_bar << 0.2, 0.0, 0.0, 0.0, 0.0, 0.0,
  		   0.0, 0.2, 0.0, 0.0, 0.0, 0.0,
  		   0.0, 0.0, 0.2, 0.0, 0.0, 0.0,
  		   0.0, 0.0, 0.0, 2.0, 0.0, 0.0,
  		   0.0, 0.0, 0.0, 0.0, 2.0, 0.0,
  		   0.0, 0.0, 0.0, 0.0, 0.0, 2.0;
  Q_bar = Q_bar*10.0;

  // Measurement Covariance
  R = MatrixXd::Zero(9,9);
  R << 0.4, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
       0.0, 0.4, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
       0.0, 0.0, 0.4, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,

       0.0, 0.0, 0.0, 1.1, 0.0, 0.0, 0.0, 0.0, 0.0,
       0.0, 0.0, 0.0, 0.0, 1.1, 0.0, 0.0, 0.0, 0.0,
       0.0, 0.0, 0.0, 0.0, 0.0, 1.1, 0.0, 0.0, 0.0,

       0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5, 0.0, 0.0,
       0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5, 0.0,
       0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5;


  // Measurement Covariance - YOLO
  R_yolo = MatrixXd::Zero(6,6);
  R_yolo << 6.0, 0.0, 0.0, 0.0, 0.0, 0.0,
		    0.0, 6.0, 0.0, 0.0, 0.0, 0.0,
			0.0, 0.0, 6.0, 0.0, 0.0, 0.0,

			0.0, 0.0, 0.0, 150.0, 0.0, 0.0,
			0.0, 0.0, 0.0, 0.0, 150.0, 0.0,
			0.0, 0.0, 0.0, 0.0, 0.0, 50.0;


  // Measurement Covariance - YOLO
  R_slam = MatrixXd::Zero(6,6);
  R_slam << 0.4, 0.0, 0.0, 0.0, 0.0, 0.0,
		        0.0, 0.4, 0.0, 0.0, 0.0, 0.0,
			      0.0, 0.0, 0.4, 0.0, 0.0, 0.0,

			      0.0, 0.0, 0.0, 1.1, 0.0, 0.0,
			      0.0, 0.0, 0.0, 0.0, 1.1, 0.0,
			      0.0, 0.0, 0.0, 0.0, 0.0, 1.1;
            R_slam = R_slam*10;



  // Initial covariance
  P = MatrixXd::Zero(N_STATES,N_STATES);
  P << 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
       0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
       0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,

       0.0, 0.0, 0.0, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
       0.0, 0.0, 0.0, 0.0, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
       0.0, 0.0, 0.0, 0.0, 0.0, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,

       0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
       0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
       0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,

       0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0,
       0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.1, 0.0, 0.0, 0.0, 0.0,
       0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.1, 0.0, 0.0, 0.0,

       0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.1, 0.0, 0.0,
       0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.1, 0.0,
       0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.1;
  //P = P + 0.01;
  for (int k1 = 0; k1<N_STATES; k1++){
	  for (int k2 = 0; k2<N_STATES; k2++){
		  if(k1!=k2){
			  P(k1,k2) = P(k1,k2) + 0.01*0;
		  }
	  }
  }

  // Kalman Gain
  K = MatrixXd::Zero(N_STATES,N_STATES);
}


// EKF_class::EKF_class(VectorXd states_0, double Frequency, MatrixXd H_, MatrixXd Q_, MatrixXd Q_bar_, MatrixXd R_, MatrixXd P_){

EKF_class::EKF_class(VectorXd states_0, double Frequency, bool enable_slam_, MatrixXd H_, MatrixXd H_yolo_, MatrixXd H_slam_, MatrixXd Q_, MatrixXd Q_bar_, MatrixXd R_, MatrixXd R_yolo_, MatrixXd R_slam_, MatrixXd P_){



  VectorXd imu_init(6);
  imu_init << 0,0,0,  0,0,9.81;

  states = states_0; // (x, y, z, phi, theta, psi, vx, vy, vz, bgx, bgy, bgz, bax, bay, baz)
  imu_data = imu_init; // (ang_vel, acc_linear)

  dt = 1.0/Frequency;

  //Flag to enable or not the use of SLAM
  enable_slam = enable_slam_;

  // Jacobian of the measurement model
  H = H_;

  // Jacobian of the measurement model - YOLO (gate detection)
  H_yolo = H_yolo_;

  // Jacobian of the measurement model - YOLO (gate detection)
  H_slam = H_slam_;

  // Model Propagation Covariance
  Q = Q_;

  // IMU covariance
  Q_bar = Q_bar_;

  // Measurement Covariance
  R = R_;

  // Measurement Covariance - YOLO
  R_yolo = R_yolo_;

  // Measurement Covariance - YOLO
  R_slam = R_slam_;

  // Initial covariance
  P = P_;
  //P = P + 0.01;
  for (int k1 = 0; k1<N_STATES; k1++){
    for (int k2 = 0; k2<N_STATES; k2++){
      if(k1!=k2){
        P(k1,k2) = P(k1,k2) + 0.01*0;
      }
    }
  }

  // Kalman Gain
  K = MatrixXd::Zero(N_STATES,N_STATES);

}




// callback_imu
void EKF_class::callback_imu(VectorXd gyro_read, VectorXd acc_read){
  // Low-pass filter
  double alpha = 0.04;  //We can make alpha be a function of Ts
  double beta = 0.04;  //We can make alpha be a function of Ts
  imu_data.block(0,0,3,1) = (1.0-beta)*imu_data.block(0,0,3,1) + beta*gyro_read;
  imu_data.block(3,0,3,1) = (1.0-alpha)*imu_data.block(3,0,3,1) + alpha*acc_read;
}




// Range Finder
void EKF_class::callback_rangeF(double range_finder){

	double alpha = 0.2;
	rangeF_data = (1.0-alpha)*rangeF_data + alpha*range_finder;
}





// Jacobian of the model with respect to the states
MatrixXd EKF_class::Jacobian_F(VectorXd x, VectorXd u, double dt){

    MatrixXd F(N_STATES,N_STATES);

    VectorXd f0(N_STATES);
    VectorXd f1(N_STATES);

    VectorXd state_now(N_STATES), imu_now(6);


//    state_now = states;
    state_now = x;
    imu_now = u;
    VectorXd state_plus(N_STATES), state_diff(N_STATES);
    double delta = 0.0001;

    f0 = discrete_model(state_now, imu_data, dt);

    for (int k = 0; k<N_STATES; k++){
        state_plus = state_now;
        state_plus(k) = state_plus(k) + delta;
        f1 = discrete_model(state_plus, imu_now, dt);
        state_diff = f1-f0;
        state_diff(3) = sin(state_diff(3)); //Orientation   !!!!!!!!!!!!!!!!!!!!! CHECK THAT
        state_diff(4) = sin(state_diff(4)); //Orientation   !!!!!!!!!!!!!!!!!!!!! CHECK THAT
        state_diff(5) = sin(state_diff(5)); //Orientation   !!!!!!!!!!!!!!!!!!!!! CHECK THAT
        state_diff(18) = sin(state_diff(18)); //Bias SLAM orientation
        state_diff(19) = sin(state_diff(19)); //Bias SLAM orientation
        state_diff(20) = sin(state_diff(20)); //Bias SLAM orientation
        F.block(0,k,N_STATES,1) = state_diff/delta;
    }

    return F;
}




// Jacobian of the model with respect to the IMU
MatrixXd EKF_class::Jacobian_G(VectorXd x, VectorXd u, double dt){

   MatrixXd G(N_STATES,6);

   VectorXd f0(N_STATES);
   VectorXd f1(N_STATES);

   VectorXd state_now(N_STATES), imu_now(6);


//    state_now = states;
   state_now = x;
   imu_now = u;
   VectorXd imu_plus(6), state_diff(N_STATES);
   double delta = 0.0001;

   f0 = discrete_model(state_now, imu_now, dt);

   for (int k = 0; k<6; k++){
       imu_plus = imu_now;
       imu_plus(k) = imu_plus(k) + delta;
       f1 = discrete_model(state_now, imu_plus, dt);
       state_diff = f1-f0;
       state_diff(3) = sin(state_diff(3)); //Orientation   !!!!!!!!!!!!!!!!!!!!! CHECK THAT
       state_diff(4) = sin(state_diff(4)); //Orientation   !!!!!!!!!!!!!!!!!!!!! CHECK THAT
       state_diff(5) = sin(state_diff(5)); //Orientation   !!!!!!!!!!!!!!!!!!!!! CHECK THAT
       state_diff(18) = sin(state_diff(18)); //Bias SLAM orientation
       state_diff(19) = sin(state_diff(19)); //Bias SLAM orientation
       state_diff(20) = sin(state_diff(20)); //Bias SLAM orientation
       G.block(0,k,N_STATES,1) = state_diff/delta;
   }

   return G;
}






// Model
VectorXd EKF_class::discrete_model(VectorXd x, VectorXd u, double dt){

  //Create output vector
  VectorXd f(N_STATES);

  // Remove bias of imu, create new variable
  VectorXd u_imu(6);
  u_imu = u - x.block(9,0,6,1);

  // Create a skew_symmetric_matrix of angular velocities
  MatrixXd S_omega(3,3);
  S_omega << 0.0, -u_imu(2), u_imu(1),
             u_imu(2), 0.0, -u_imu(0),
             -u_imu(1), u_imu(0), 0.0;

  // Create a rotation matrix from body frame to world
  MatrixXd R_bw(3,3);
  double phi, theta, psi;
  phi = x(3);
  theta = x(4);
  psi = x(5);

  R_bw << (cos(theta)*cos(psi)), (sin(phi)*sin(theta)*cos(psi)-cos(phi)*sin(psi)), (cos(phi)*sin(theta)*cos(psi)+sin(phi)*sin(psi)),
					(cos(theta)*sin(psi)), (sin(phi)*sin(theta)*sin(psi)+cos(phi)*cos(psi)), (cos(phi)*sin(theta)*sin(psi)-sin(phi)*cos(psi)),
					(-sin(theta)), (sin(phi)*cos(theta)), (cos(phi)*cos(theta));

  // Create the Jacobian of transformation
  MatrixXd JT(3,3);
  //if(cos(theta)*cos(theta) < 0.01^2){
  //  JT << 1.0, sin(phi)*tan(theta), cos(phi)*tan(theta),
  //        0.0, cos(phi), -sin(phi),
  //        0.0, sin(phi)/0.01, cos(phi)/0.01;
  //}
  //else{
    JT << 1.0, sin(phi)*tan(theta), cos(phi)*tan(theta),
          0.0, cos(phi), -sin(phi),
          0.0, sin(phi)/cos(theta), cos(phi)/cos(theta);
  //}


  // Gravity vector
  Vector3d g_vec;
  g_vec << 0.0, 0.0, -9.81;

  // // position
  // x.block(0,0,3,1) = x.block(0,0,3,1) + dt*R_bw*x.block(6,0,3,1);
  // // orientation
  // x.block(3,0,3,1) = x.block(3,0,3,1) + dt*J*u_imu.block(0,0,3,1);
  // // velocities
  // x.block(6,0,3,1) = x.block(6,0,3,1) + dt*(u_imu.block(3,0,3,1) + (-S_omega*x.block(6,0,3,1)) + R_bw.transpose()*g_vec);
  // // bias
  // x.block(9,0,6,1) = x.block(9,0,6,1);


  //filter_mutex.lock();
  // Position
  f.block(0,0,3,1) = x.block(0,0,3,1) + ( R_bw*x.block(6,0,3,1) )*dt;
  // Orientation
  f.block(3,0,3,1) = x.block(3,0,3,1) + ( JT*u_imu.block(0,0,3,1) )*dt;
  // Velocities
  f.block(6,0,3,1) = x.block(6,0,3,1) + ( u_imu.block(3,0,3,1) - S_omega*x.block(6,0,3,1) + R_bw.transpose()*g_vec )*dt;
  // Bias of IMU
  f.block(9,0,6,1) = x.block(9,0,6,1);// +  ( 0 )*dt;
  // Bias of SLAM
  f.block(15,0,6,1) = x.block(15,0,6,1);// +  ( 0 )*dt;
//  f.block(15,0,6,1) << 0.0,0.0,0.0,   0.0,0.0,0.0;
  //filter_mutex.unlock();


  return f;

}




// Prediction Step
void EKF_class::prediction(double Ts){

  // Set dt
  dt = Ts;


  // Next state function
  VectorXd f(N_STATES);
  // Jacobian of the next state with respect to the states
  MatrixXd F(N_STATES,N_STATES);
  // Jacobian of the next state with respect to the IMU
  MatrixXd G(N_STATES,6);




  //Compute the next state
  f = discrete_model(states, imu_data, dt);
  //Compute the Jacobian matrix
  F = Jacobian_F(states, imu_data, dt);
  //Compute the Jacobian matrix
  G = Jacobian_G(states, imu_data, dt);


  //Propagate the states

  //filter_mutex.lock();
  states = f;
  //filter_mutex.unlock();

  //Compute the covariance of the propagation given the current state and the covariance of the IMU
  Q = G*Q_bar*G.transpose();

  //cout << "Q = \n" << Q << endl;

  //Propagate the covariance matrix
  //filter_mutex.lock();
  P = F*P*F.transpose() + Q;
  //filter_mutex.unlock();
  // // Resymetrize the matrix
  // P = (P+P.transpose())/2.0;


  //cout << "states = \n" << states.transpose() << endl << endl;
  static int c_print_f = 0;
  c_print_f++;
  if (c_print_f == 50){
//    cout << "bias_SLAM: " << states.block(15,0,6,1).transpose() << endl << endl; // rezeck
    c_print_f = 0;
  }


}







// Prediction Step
void EKF_class::prediction_z(){

  // Next state function
  VectorXd f(N_STATES);
  // Jacobian of the next state with respect to the states
  MatrixXd F(N_STATES,N_STATES);
  // Jacobian of the next state with respect to the IMU
  MatrixXd G(N_STATES,6);


  //Compute the next state
  f = discrete_model(states, imu_data, dt);
  //Compute the Jacobian matrix
  F = Jacobian_F(states, imu_data, dt);
  //Compute the Jacobian matrix
  G = Jacobian_G(states, imu_data, dt);


  //Propagate the states

  //filter_mutex.lock();
  states(2) = f(2);
  states(8) = f(8);
  //filter_mutex.unlock();

  //Compute the covariance of the propagation given the current state and the covariance of the IMU
  Q = G*Q_bar*G.transpose();

  //cout << "Q = \n" << Q << endl;

  //Propagate the covariance matrix
  //filter_mutex.lock();
  P = F*P*F.transpose() + Q;
  //filter_mutex.unlock();
  // // Resymetrize the matrix
  // P = (P+P.transpose())/2.0;


  //cout << "states = \n" << states.transpose() << endl << endl;
  static int c_print_f = 0;
  c_print_f++;
  if (c_print_f == 50){
//    cout << "bias_SLAM: " << states.block(15,0,6,1).transpose() << endl << endl; // rezeck
    c_print_f = 0;
  }


}










// Propagate the measurement in order to compensate for time delays
VectorXd EKF_class::compensate_delay(int iterations, VectorXd input){

  // Compensated measurement
  VectorXd output(6);

  VectorXd states_aux(N_STATES);
  VectorXd f(N_STATES);
  states_aux = states;
  states_aux.block(0,0,6,1) = input;

  for (int k=0; k<iterations; k++){
	  f = discrete_model(states_aux, imu_data, dt);
	  states_aux = f;
  }

  output = states_aux.block(0,0,6,1);


  return output;


}














// EKF UPDATE - Z pose
void EKF_class::callback_High(){


    //cout << "callback_High" << endl;

    //Measurement model only for pose
    MatrixXd H_aux(1,N_STATES);
    H_aux << 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0;
    H_aux(2) = 1.0/(cos(states(3))*cos(states(4)));
    H_aux(3) = -(states(2)/cos(states(4)))*log(cos(states(3)))*sin(states(3));
    H_aux(4) = -(states(2)/cos(states(3)))*log(cos(states(4)))*sin(states(4));




    //Covariance of the measurement only for pose
    double R_aux = 0.42291;

    // Compute Inovation
    double inovation;
    inovation = rangeF_data - (H_aux*states)(0);


//    // Compute Kalman Gain
    double S;
    S = (H_aux*P*H_aux.transpose())(0);
    S = S + R_aux;
    K = P*H_aux.transpose()/S;

   // Actualization of the states
    //filter_mutex.lock();
    states = states + K*inovation;

   // Actualization of covariance matrix
    P = (MatrixXd::Identity(N_STATES,N_STATES) - K*H_aux)*P;
    //filter_mutex.unlock();



}







// EKF UPDATE - Z pose
void EKF_class::callback_vertical_velocity(double vz_measured){


    //cout << "callback_High" << endl;

    //Measurement model only for pose
    MatrixXd H_aux(1,N_STATES);
    H_aux << 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0;


    //Covariance of the measurement only for pose
    double R_aux = 1.02291;

    // Compute Inovation
    double inovation;
    inovation = rangeF_data - (H_aux*states)(0);


//    // Compute Kalman Gain
    double S;
    S = (H_aux*P*H_aux.transpose())(0);
    S = S + R_aux;
    K = P*H_aux.transpose()/S;

   // Actualization of the states
    //filter_mutex.lock();
    states = states + K*inovation;

   // Actualization of covariance matrix
    P = (MatrixXd::Identity(N_STATES,N_STATES) - K*H_aux)*P;
    //filter_mutex.unlock();

}







// Measurement model of the SLAM
VectorXd EKF_class::h_slam(VectorXd x){

	VectorXd h(6);
	VectorXd last_line(4);
	last_line << 0.0, 0.0, 0.0, 1.0;


	//cout << "A1" << endl;
	//Construct bias homogeneous matrix
	MatrixXd H_bias(4,4); //Bias of the slam
	H_bias.block(0,3,3,1) = x.block(15,0,3,1);
	H_bias.block(0,0,3,3) = angle2rotm(x.block(18,0,3,1));
	H_bias.block(3,0,1,4) = last_line.transpose();

	//cout << "B1" << endl;
	//Construct state homogeneous matrix
	MatrixXd H_state(4,4);
	H_state.block(0,3,3,1) = x.block(0,0,3,1);
	H_state.block(0,0,3,3) = angle2rotm(x.block(3,0,3,1));
	H_state.block(3,0,1,4) = last_line.transpose();

	//cout << "C1" << endl;
	MatrixXd H_measurement(4,4);
	H_measurement = H_bias*H_state; //this is
//	H_measurement = H_state*H_bias; //this is
	//cout << "D1" << endl;
	h.block(0,0,3,1) = H_measurement.block(0,3,3,1);
	//cout << "E1" << endl;
	h.block(3,0,3,1) = rotm2angle(H_measurement.block(0,0,3,3));
	//cout << "F1" << endl;


	//cout << "H_bias = \n" << H_bias << endl;
	//cout << "H_state = \n" << H_state << endl;
	//cout << "H_measurement = \n" << H_measurement << endl << endl;

	return h;
}


//Jacobian of the measurement model of the SLAM
MatrixXd EKF_class::H_slam_jac(VectorXd x){

	MatrixXd H_jac(6,N_STATES);

	VectorXd h0(6);
	VectorXd h1(6);

	VectorXd state_now(N_STATES);

	state_now = x;
	VectorXd state_plus(N_STATES), h_diff(6);
	double delta = 0.0001;
//	double delta = 0.01;

	h0 = h_slam(state_now);

	//cout << "h0 = " << h0.transpose() << endl;

	for (int k = 0; k<N_STATES; k++){
		state_plus = state_now;
		state_plus(k) = state_plus(k) + delta;
		h1 = h_slam(state_plus);
		//cout << "h"<< k+1 <<" = " << h1.transpose() << endl;
		h_diff = h1-h0;
		h_diff(3) = sin(h_diff(3)); //SLAM orientation
		h_diff(4) = sin(h_diff(4)); //SLAM orientation
		h_diff(5) = sin(h_diff(5)); //SLAM orientation
		H_jac.block(0,k,6,1) = h_diff/delta;
	}
	//cout << endl;

	return H_jac;



}














// EKF UPDATE - POSE - YOLO
void EKF_class::callback_pose_yolo(VectorXd pose){
    // Measurements
    read_pose = pose; // position and euler angles

    //Measurement model only for pose
    MatrixXd H_aux(6,N_STATES);
    H_aux = H_yolo;

    //Covariance of the measurement only for pose
    MatrixXd R_aux(6,6);
//    R_aux = R_yolo/5.0;
    R_aux = R_yolo;


    //cout << "bias = " << states.block(9,0,6,1).transpose() << endl;

    // Compute Inovation
    VectorXd inovation(6);
    inovation = read_pose - H_aux*states;
    inovation(3) = sin(inovation(3));
    inovation(4) = sin(inovation(4));
    inovation(5) = sin(inovation(5));

    // Compute Kalman Gain
    MatrixXd S(6,6);
    S = H_aux*P*H_aux.transpose() + R_aux;
    K = P*H_aux.transpose()*S.inverse();


    //filter_mutex.lock();
    // Actualization of the states
    //inovation(2) = 0.0;
    states = states + K*inovation;

    // Actualization of covariance matrix
    P = (MatrixXd::Identity(N_STATES,N_STATES) - K*H_aux)*P;
    //filter_mutex.unlock();

}



// EKF UPDATE - position x and y and orientation yaw - YOLO
void EKF_class::callback_yolo_xy_y(VectorXd pose){
    // Measurements
    //read_pose = pose; // position and euler angles


	  // cout <<  "A" << endl;
    //Measurement model only for pose
    MatrixXd H_aux(3,N_STATES);
    //H_aux = H_yolo;
    // cout <<  "B" << endl;
    H_aux.block(0,0,2,N_STATES) = H_yolo.block(0,0,2,N_STATES);
    // cout <<  "C" << endl;
    H_aux.block(2,0,1,N_STATES) = H_yolo.block(5,0,1,N_STATES);
    // cout <<  "D" << endl;

    //Covariance of the measurement only for pose
    MatrixXd R_aux(3,3);
//    R_aux = R_yolo;
    R_aux << 0.0,0.0,0.0, 0.0,0.0,0.0, 0.0,0.0,0.0;
    // cout <<  "E" << endl;
    R_aux.block(0,0,2,2) = R_yolo.block(0,0,2,2);
    // cout <<  "F" << endl;
    R_aux.block(2,2,1,1) = R_yolo.block(5,5,1,1);
    // cout <<  "G" << endl;
    // Compute Inovation
    VectorXd inovation(3);
    inovation = pose - H_aux*states;
    // cout <<  "H" << endl;
    inovation(2) = sin(inovation(2)); // yaw
    // cout <<  "I" << endl;

    // Compute Kalman Gain
    MatrixXd S(3,3);
    S = H_aux*P*H_aux.transpose() + R_aux;
    //cout <<  "J" << endl;
    K = P*H_aux.transpose()*S.inverse();
    //cout <<  "K" << endl;

    //filter_mutex.lock();
    // Actualization of the states
    states = states + K*inovation;


    // Actualization of covariance matrix
    P = (MatrixXd::Identity(N_STATES,N_STATES) - K*H_aux)*P;
    //filter_mutex.unlock();

}





// EKF UPDATE - POSE - SLAM
void EKF_class::callback_pose_slam(VectorXd pose){
    // Measurements
    read_pose = pose; // position and Euler angles

    //Measurement model only for pose
    MatrixXd H_aux(6,N_STATES);
    H_aux = H_slam_jac(states);

    //Covariance of the measurement only for pose
    MatrixXd R_aux(6,6);
    R_aux = R_slam;


    //cout << "bias_slam = " << states.block(15,0,6,1).transpose() << endl; //rezeck

    // Compute Inovation
    VectorXd inovation(6);
//    inovation = read_pose - H_aux*states;
    inovation = read_pose - h_slam(states);
    inovation(3) = sin(inovation(3));
    inovation(4) = sin(inovation(4));
    inovation(5) = sin(inovation(5));

 
    // Compute Kalman Gain
    MatrixXd S(6,6);
    S = H_aux*P*H_aux.transpose() + R_aux;
    K = P*H_aux.transpose()*S.inverse();

    //filter_mutex.lock();
    // Actualization of the states
    //inovation(2) = 0.0;
    states = states + K*inovation;

    // Actualization of covariance matrix
    P = (MatrixXd::Identity(N_STATES,N_STATES) - K*H_aux)*P;
    //filter_mutex.unlock();

}













// EKF UPDATE - POSE
void EKF_class::callback_pose(VectorXd pose){
    // Measurements
    read_pose = pose;

    //Measurement model only for pose
    MatrixXd H_aux(6,N_STATES);
    H_aux = H.block(0,0,6,N_STATES);

    //Covariance of the measurement only for pose
    MatrixXd R_aux(6,6);
    R_aux = R.block(0,0,6,6);


    //cout << "bias = " << states.block(9,0,6,1).transpose() << endl;

    // Compute Inovation
    VectorXd inovation(6);
    inovation = read_pose - H_aux*states;
    inovation(3) = sin(inovation(3));
    inovation(4) = sin(inovation(4));
    inovation(5) = sin(inovation(5));

    // Compute Kalman Gain
    MatrixXd S(6,6);
    S = H_aux*P*H_aux.transpose() + R_aux;
    K = P*H_aux.transpose()*S.inverse();





    MatrixXd lixo(6,N_STATES);
    lixo = H_slam_jac(states);

    //cout << "P: " << P(0,0) << "\t" << P(1,1) << "\t" << P(2,2) << "\n";

    //filter_mutex.lock();
    // Actualization of the states
    //inovation(2) = 0.0;
    states = states + K*inovation;

    // Actualization of covariance matrix
    P = (MatrixXd::Identity(N_STATES,N_STATES) - K*H_aux)*P;
    //filter_mutex.unlock();

}


// EKF UPDATE - POSITION ONLY
void EKF_class::callback_position(Vector3d pos){
    // Measurements
    read_pose = pos;

    //Measurement model only for position
    MatrixXd H_aux(3,N_STATES);
    H_aux = H.block(0,0,3,N_STATES);

    //Covariance of the measurement only for position
    MatrixXd R_aux(3,6);
    R_aux = R.block(0,0,3,3);


    // Compute Inovation
    VectorXd inovation(3);
    inovation = read_pose - H_aux*states;


    // Compute Kalman Gain
    MatrixXd S(3,3);
    S = H_aux*P*H_aux.transpose() + R_aux;
    K = P*H_aux.transpose()*S.inverse();

    //filter_mutex.lock();
    // Actualization of the states
    //inovation(2) = 0.0;
    states = states + K*inovation;

    // Actualization of covariance matrix
    P = (MatrixXd::Identity(N_STATES,N_STATES) - K*H_aux)*P;
    //filter_mutex.unlock();

}





// EKF UPDATE - POSE X Y Z
void EKF_class::callback_pose_xyz(VectorXd pose){


    //Measurement model only for pose
	MatrixXd H_aux(3,N_STATES);
	H_aux << 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
			0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
			0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0;


	//Covariance of the measurement only for pose
	MatrixXd R_aux(3,3);
	R_aux = R.block(0,0,3,3);
//	R_aux = R.block(0,0,3,3)*30.0;

	// Compute Inovation
	VectorXd inovation(3);
	inovation = pose - H_aux*states;

	 // Compute Kalman Gain
	MatrixXd S(3,3);
	S = H_aux*P*H_aux.transpose() + R_aux*10;
	K = P*H_aux.transpose()*S.inverse();

  //filter_mutex.lock();
	// Actualization of the states
	states = states + K*inovation;

	// Actualization of covariance matrix
	P = (MatrixXd::Identity(N_STATES,N_STATES) - K*H_aux)*P;
  //filter_mutex.unlock();

}



// EKF UPDATE - VELOCITY
void EKF_class::callback_velocity(VectorXd body_vel){

    // Rotation to World Frame
	//VectorXd vel_world(3);
	//vel_world = body_vel;


    //Measurement model only for velocity
    MatrixXd H_aux(3,N_STATES);
    H_aux = H.block(6,0,3,N_STATES);

    //Covariance of the measurement only for velocity
    MatrixXd R_aux(3,3);
    R_aux = R.block(6,6,3,3);




    // Compute Inovation
    VectorXd inovation(3);
    inovation = body_vel - H_aux*states;

    // Compute Kalman Gain
    MatrixXd S(3,3);
    S = H_aux*P*H_aux.transpose() + R_aux;
    K = P*H_aux.transpose()*S.inverse();

    //filter_mutex.lock();
    // Actualization of the states
    states = states + K*inovation;

    // Covariance Propagation
    P = (MatrixXd::Identity(N_STATES,N_STATES) - K*H_aux)*P;
    //filter_mutex.unlock();

}




// destructor
EKF_class::~EKF_class(){
}





//VectorXd EKF_class::EulertoQuaternion( double roll, double pitch, double yaw) // yaw (Z), pitch (Y), roll (X)
VectorXd EKF_class::EulertoQuaternion( VectorXd rpy) // yaw (Z), pitch (Y), roll (X)
{

	double roll = rpy(0);
	double pitch = rpy(1);
	double yaw = rpy(2);

    // Abbreviations for the various angular functions
    double cy = cos(yaw * 0.5);
    double sy = sin(yaw * 0.5);
    double cp = cos(pitch * 0.5);
    double sp = sin(pitch * 0.5);
    double cr = cos(roll * 0.5);
    double sr = sin(roll * 0.5);

    double w = cy * cp * cr + sy * sp * sr;
    double x = cy * cp * sr - sy * sp * cr;
    double y = sy * cp * sr + cy * sp * cr;
    double z = sy * cp * cr - cy * sp * sr;

    VectorXd q(4);
    q << w,x,y,z;
    return q;
}





MatrixXd EKF_class::angle2rotm( VectorXd rpy) // yaw (Z), pitch (Y), roll (X)
{

	MatrixXd Rot(3,3);
	double phi = rpy(0);
	double theta = rpy(1);
	double psi = rpy(2);
	// Get rotation matrix
	Rot << (cos(theta)*cos(psi)), (sin(phi)*sin(theta)*cos(psi)-cos(phi)*sin(psi)), (cos(phi)*sin(theta)*cos(psi)+sin(phi)*sin(psi)),
		   (cos(theta)*sin(psi)), (sin(phi)*sin(theta)*sin(psi)+cos(phi)*cos(psi)), (cos(phi)*sin(theta)*sin(psi)-sin(phi)*cos(psi)),
		   (-sin(theta)), (sin(phi)*cos(theta)), (cos(phi)*cos(theta));

    return Rot;
}



VectorXd EKF_class::rotm2angle( MatrixXd Rot)
{
	VectorXd rpy(3);
	Matrix3d Rot2;
	//cout << "A2" << endl;
	Rot2 = Rot;
	//cout << "B2" << endl;
	Quaterniond quat1(Rot2);
	//cout << "C2" << endl;
	VectorXd quat2(4);
	//cout << "D2" << endl;
	quat2 << quat1.w(), quat1.x(), quat1.y(), quat1.z();
	//cout << "E2" << endl;
	rpy = quat2eulerangle(quat2);
	//cout << "F2" << endl;
    return rpy;
}






// Unit Quaternion to Euler angle
VectorXd EKF_class::quat2eulerangle(VectorXd q){
	// w x y z

   VectorXd angle(3,1);
   angle.setZero();

   // roll (x-axis rotation)
	double sinr_cosp = +2.0 * (q[0] * q[1] + q[2] * q[3]);
	double cosr_cosp = +1.0 - 2.0 * (q[1] * q[1] + q[2] * q[2]);
	angle[0] = atan2(sinr_cosp, cosr_cosp);

	// pitch (y-axis rotation)
	double sinp = +2.0 * (q[0] * q[2] - q[3] * q[1]);
	if (fabs(sinp) >= 1)
		angle[1] = copysign(M_PI / 2, sinp); // use 90 degrees if out of range
	else
		angle[1] = asin(sinp);

	// yaw (z-axis rotation)
	double siny_cosp = +2.0 * (q[0] * q[3] + q[1] * q[2]);
	double cosy_cosp = +1.0 - 2.0 * (q[2] * q[2] + q[3] * q[3]);
	angle[2] = atan2(siny_cosp, cosy_cosp);

   return angle;
}

//Method to get position orientation and velocity
VectorXd EKF_class::get_states(){

  Eigen::VectorXd states_return(10);
  states_return.block(0,0,3,1) = states.block(0,0,3,1);
  states_return.block(3,0,4,1) = EKF_class::EulertoQuaternion(states.block(3,0,3,1));
  states_return.block(7,0,3,1) = states.block(6,0,3,1);

  return states_return;
}


//Method to get raw EKF filter states
VectorXd EKF_class::get_raw_states(){
  return states;
}

VectorXd EKF_class::get_IMU(){
  return imu_data - states.block(9,0,6,1)*0; //ATENTION, THIS MAY CAUSE PROBLEM
}
