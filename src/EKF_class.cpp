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
EKF_class::EKF_class(VectorXd initial_pos, double Frequency){
  VectorXd state_init(N_STATES), imu_init(6), read_pose_init(6);
  state_init.setZero();
  read_pose_init.setZero();
  imu_init << 0,0,0,  0,0,9.81;

  state_init.block(0,0,3,1) = initial_pos;
  read_pose_init.block(0,0,3,1) = initial_pos;

  states = state_init; // (x, y, z, phi, theta, psi, vx, vy, vz, bgx, bgy, bgz, bax, bay, baz)
  imu_data = imu_init; // (ang_vel, acc_linear)
  read_pose = read_pose_init; // (x, y, z, phi, theta, psi)

  dt = 1.0/Frequency;

  rangeF_data = 0.0;

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

  // Model Propagation Covariance
  Q = MatrixXd::Zero(N_STATES,N_STATES);
  Q << 0.00011, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
       0.0, 0.00011, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
       0.0, 0.0, 0.00016, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,

       0.0, 0.0, 0.0, 0.00001, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
       0.0, 0.0, 0.0, 0.0, 0.00001, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
       0.0, 0.0, 0.0, 0.0, 0.0, 0.00001, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,

       0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0001, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
       0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0001, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
       0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0001, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,

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

  // Measurement Covariance
  R = MatrixXd::Zero(9,9);
  R << 0.8, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
       0.0, 0.8, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
       0.0, 0.0, 0.8, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,

       0.0, 0.0, 0.0, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0,
       0.0, 0.0, 0.0, 0.0, 0.1, 0.0, 0.0, 0.0, 0.0,
       0.0, 0.0, 0.0, 0.0, 0.0, 0.1, 0.0, 0.0, 0.0,

       0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5, 0.0, 0.0,
       0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5, 0.0,
       0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5;

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

  // Kalman Gain
  K = MatrixXd::Zero(N_STATES,N_STATES);
}

// constructor 2
EKF_class::EKF_class(VectorXd states_0, double Frequency, MatrixXd H_, MatrixXd Q_, MatrixXd Q_bar_, MatrixXd R_, MatrixXd P_){
  VectorXd imu_init(6);
  imu_init << 0,0,0,  0,0,9.81;

  states = states_0; // (x, y, z, phi, theta, psi, vx, vy, vz, bgx, bgy, bgz, bax, bay, baz)
  imu_data = imu_init; // (ang_vel, acc_linear)

  dt = 1.0/Frequency;

  // Jacobian of the measurement model
  H = H_;

  // Model Propagation Covariance
  Q = Q_;

  // IMU covariance
  Q_bar = Q_bar_;

  // Measurement Covariance
  R = R_;

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
void EKF_class::callback_imu(VectorXd gyro_read, VectorXd acc_read, double Ts){
  // Set dt
  dt = Ts;

  // Low-pass filter
  double alpha = 0.121;  //We can make alpha be a function of Ts
  double beta = 0.121;  //We can make alpha be a function of Ts
  imu_data.block(0,0,3,1) = (1.0-beta)*imu_data.block(0,0,3,1) + beta*gyro_read;
  imu_data.block(3,0,3,1) = (1.0-alpha)*imu_data.block(3,0,3,1) + alpha*acc_read;
}


// Range Finder
void EKF_class::callback_rangeF(double range_finder){

  double alpha = 0.2;
  rangeF_data = (1.0-alpha)*rangeF_data + alpha*range_finder;
}




// Jacobian of the mode with respect to the states
MatrixXd EKF_class::Jacobian_F(VectorXd x, VectorXd u, double dt){

    MatrixXd F(N_STATES,N_STATES);

    VectorXd f0(N_STATES);
    VectorXd f1(N_STATES);

    VectorXd state_now(N_STATES), imu_now(6);


    // state_now = states;
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
        state_diff(6) = sin(state_diff(6));
        state_diff(7) = sin(state_diff(7));
        state_diff(8) = sin(state_diff(8));
        F.block(0,k,N_STATES,1) = state_diff/delta;
    }

    return F;
}



// Jacobian of the mode with respect to the IMU
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
       state_diff(6) = sin(state_diff(6));
       state_diff(7) = sin(state_diff(7));
       state_diff(8) = sin(state_diff(8));
       G.block(0,k,N_STATES,1) = state_diff/delta;
   }

   return G;
}



// !!!! MODIFICAR AQUI PARA O MODELO ACKERMANN

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
  JT << 1.0, sin(phi)*tan(theta), cos(phi)*tan(theta),
        0.0, cos(phi), -sin(phi),
        0.0, sin(phi)/cos(theta), cos(phi)/cos(theta);
  

  // Gravity vector
  Vector3d g_vec;
  g_vec << 0.0, 0.0, -9.81;

  // Position
  f.block(0,0,3,1) = x.block(0,0,3,1) + ( R_bw*x.block(6,0,3,1) )*dt;
  // Orientation
  f.block(3,0,3,1) = x.block(3,0,3,1) + ( JT*u_imu.block(0,0,3,1) )*dt;
  // Velocities
  f.block(6,0,3,1) = x.block(6,0,3,1) + ( u_imu.block(3,0,3,1) - S_omega*x.block(6,0,3,1) + R_bw.transpose()*g_vec )*dt;
  // Bias
  f.block(9,0,6,1) = x.block(9,0,6,1);

  return f;

}




// Prediction Step
void EKF_class::prediction(){

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
  states = f;

  //Compute the covariance of the propagation given the current state and the covariance of the IMU
  Q = G*Q_bar*G.transpose();

  //cout << "Q = \n" << Q << endl;

  //Propagate the covariance matrix
  P = F*P*F.transpose() + Q;


}



// EKF UPDATE - Z pose
void EKF_class::callback_High(){

    //Measurement model only for pose
    MatrixXd H_aux(1,15);
    H_aux << 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0;
    H_aux(2) = 1.0/(cos(states(3))*cos(states(4)));
    H_aux(3) = -(states(2)/cos(states(4)))*log(cos(states(3)))*sin(states(3));
    H_aux(4) = -(states(2)/cos(states(3)))*log(cos(states(4)))*sin(states(4));

    //Covariance of the measurement only for pose
    double R_aux = 0.1;




    //cout << "bias = " << states.block(9,0,6,1).transpose() << endl;

    // Compute Inovation
    double inovation;
    inovation = rangeF_data - (H_aux*states)(0);


//    // Compute Kalman Gain
    double S;
    S = (H_aux*P*H_aux.transpose())(0);
    S = S + R_aux;
    K = P*H_aux.transpose()/S;
//
//    // Actualization of the states
    states = states + K*inovation;

//    // Actualization of covariance matrix
    P = (MatrixXd::Identity(15,15) - K*H_aux)*P;

}





// EKF UPDATE - Position
void EKF_class::callback_position(VectorXd pose){
    // Measurements
    read_pose = pose;

    //Measurement model only for pose
    MatrixXd H_aux(3,N_STATES);
    H_aux = H.block(0,0,3,N_STATES);

    //Covariance of the measurement only for pose
    // MatrixXd R_aux(6,N_STATES);
    MatrixXd R_aux(3,3);
    R_aux = R.block(0,0,3,3);

    //cout << "bias = " << states.block(9,0,6,1).transpose() << endl;

    // Compute Inovation
    VectorXd inovation(3);
    inovation = read_pose - H_aux*states;

    // Compute Kalman Gain
    MatrixXd S(3,3);
    S = H_aux*P*H_aux.transpose() + R_aux;
    K = P*H_aux.transpose()*S.inverse();

    // Actualization of the states
    states = states + K*inovation;

    // Actualization of covariance matrix
    P = (MatrixXd::Identity(N_STATES,N_STATES) - K*H_aux)*P;
}


// EKF UPDATE - ORIENTATION
void EKF_class::callback_orientation(VectorXd orient){
    // Measurements
    VectorXd read_orientation(3);
    read_orientation = quat2eulerangle(orient);

    //Measurement model only for pose
    MatrixXd H_aux(3,N_STATES);
    H_aux = H.block(3,0,3,N_STATES);

    //Covariance of the measurement only for pose
    // MatrixXd R_aux(6,N_STATES);
    MatrixXd R_aux(3,3);
    R_aux = R.block(3,3,3,3);

    //cout << "bias = " << states.block(9,0,6,1).transpose() << endl;

    // Compute Inovation
    VectorXd inovation(3);
    inovation = read_orientation - H_aux*states;
    inovation(0) = sin(inovation(0));
    inovation(1) = sin(inovation(1));
    inovation(2) = sin(inovation(2));

    // Compute Kalman Gain
    MatrixXd S(3,3);
    S = H_aux*P*H_aux.transpose() + R_aux;
    K = P*H_aux.transpose()*S.inverse();

    // Actualization of the states
    states = states + K*inovation;

    // Actualization of covariance matrix
    P = (MatrixXd::Identity(N_STATES,N_STATES) - K*H_aux)*P;
}



// EKF UPDATE - VELOCITY
void EKF_class::callback_velocity(VectorXd body_vel){

    // Rotation to World Frame
    VectorXd vel_world(3);
    vel_world = body_vel;


    //Measurement model only for velocity
    MatrixXd H_aux(3,N_STATES);
    H_aux = H.block(6,0,3,N_STATES);

    //Covariance of the measurement only for velocity
    // MatrixXd R_aux(6,15);
    MatrixXd R_aux(3,3);
    R_aux = R.block(6,6,3,3);


    // Compute Inovation
    VectorXd inovation(3);
    inovation = vel_world - H_aux*states;

    // Compute Kalman Gain
    MatrixXd S(3,3);
    S = H_aux*P*H_aux.transpose() + R_aux;
    K = P*H_aux.transpose()*S.inverse();

    // Actualization of the states
    states = states + K*inovation;

    // Covariance Propagation
    P = (MatrixXd::Identity(N_STATES,N_STATES) - K*H_aux)*P;

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

// Metodos modificadores
VectorXd EKF_class::get_states(){

  Eigen::VectorXd states_return(10);
  states_return.block(0,0,3,1) = states.block(0,0,3,1);
  states_return.block(3,0,4,1) = EKF_class::EulertoQuaternion(states.block(3,0,3,1));
  states_return.block(7,0,3,1) = states.block(6,0,3,1);


  return states_return;
}

VectorXd EKF_class::get_IMU(){
  return imu_data;
}