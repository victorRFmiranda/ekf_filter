#include <eigen3/Eigen/Dense>
#include <eigen3/Eigen/Geometry>
#include <iostream>
#include <math.h>

using namespace Eigen;

class EKF_class{
  private:
    // States (12) and IMU (6) Data
    VectorXd states, imu_data;
    // Pose Measurement (3)
    VectorXd read_pose;
    // Time stamp
    double dt;
    // Range Finder
    double rangeF_data;

    // Covariation Matrix
    MatrixXd P, Q, R, H, Q_bar;
    // Kalman Gain
    MatrixXd K;

  public:
  	EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    // EKF_class(Eigen::VectorXd, double); //Construtor
    EKF_class(VectorXd, double, MatrixXd, MatrixXd, MatrixXd, MatrixXd, MatrixXd); //Construtor
    ~EKF_class(); //Destructor

    //callback imu Data (imu, dt)
    void callback_imu(VectorXd, VectorXd, double);
    // callback range Finder data (range finder)
    void callback_rangeF(double);

    // Model fuction (states, imu)
    VectorXd discrete_model(VectorXd, VectorXd, double);

    // Facobian of the model functionwith respect to the states (VectorXd x, VectorXd u)
	MatrixXd Jacobian_F(VectorXd, VectorXd, double);
	//Facobian of the model functionwith respect to the IMU (VectorXd x, VectorXd u)
	MatrixXd Jacobian_G(VectorXd, VectorXd, double);

    // Prediction Step
    void prediction();

    // Actualization position measu (position)
    void callback_position(VectorXd);
    // Actualization orientation measu (position)
    void callback_orientation(VectorXd);
    // Actualization Vel measu (velocity)
    void callback_velocity(VectorXd);
    // Actualization RangeFinder
    // void callback_High();

    // Convert Euler to Quaternion
    //VectorXd EulertoQuaternion(VectorXd);
    VectorXd EulertoQuaternion(VectorXd);
    VectorXd quat2eulerangle(VectorXd);
    MatrixXd angle2rotm(VectorXd);
    VectorXd rotm2angle(MatrixXd);

    // Get EKF States Data
    VectorXd get_states();
    // Get Filtered IMU Data
    VectorXd get_IMU();
};