#include <eigen3/Eigen/Dense>
#include <eigen3/Eigen/Geometry>
#include <iostream>
#include <math.h>
// #include <mutex>

using namespace Eigen;

class EKF_class{
  private:
    // States (15) and IMU (6) Data
    VectorXd states, imu_data;
    //----- ----- states ----- -----
    //position
    //velocity on the body frame
    //orientation
    //bias IMU
    //bias IMU
    //bias SLAM position
    //bias SLAM orientation
    //----- ----- ------ ----- -----


    bool enable_slam;

    // Pose Measurement (6)
    VectorXd read_pose;
    // Time stamp
    double dt;
    // Range Finder
    double rangeF_data;


    double rangeF_data_prev;
    double rf_time_prev;

    // Covariation Matrix
    MatrixXd P, Q, R, H, Q_bar, H_yolo, H_slam, R_yolo, R_slam;
    // Kalman Gain
    MatrixXd K;

    // std::mutex filter_mutex;

  public:
  	EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    EKF_class(Eigen::VectorXd, double, double); //Construtor
    EKF_class(VectorXd, double, bool, MatrixXd, MatrixXd, MatrixXd, MatrixXd, MatrixXd, MatrixXd, MatrixXd, MatrixXd, MatrixXd); //Construtor
    ~EKF_class(); //Destructor



    //callback imu Data (imu, dt)
    void callback_imu(VectorXd, VectorXd);
    // callback range Finder data (range finder)
    void callback_rangeF(double);

    // Model fuction (states, imu)
    VectorXd discrete_model(VectorXd, VectorXd, double);

    // Measurement model of the range finder
    // double range_finder_model(VectorXd);

    // Facobian of the model functionwith respect to the states (VectorXd x, VectorXd u)
	MatrixXd Jacobian_F(VectorXd, VectorXd, double);
	//Facobian of the model functionwith respect to the IMU (VectorXd x, VectorXd u)
	MatrixXd Jacobian_G(VectorXd, VectorXd, double);

    // Jacobian of the range finger with respect to the states
    // MatrixXd Jacobian_range_finder(VectorXd x);

    // Prediction Step
    void prediction(double);
    void prediction_z();

    // Propagate the measurement in order to compensate for time delays
    VectorXd compensate_delay(int, VectorXd);


    // Actualization Pose measurement (pose)
    void callback_pose(VectorXd);
    // Actualization Position measurement (position)
    void callback_position(Vector3d);
    // Actualization Vel measurement (velocity)
    void callback_velocity(VectorXd);
    // Actualization RangeFinder
    void callback_High();
    // Actualization Position x,y
    void callback_pose_xyz(VectorXd pose);


    //
    void callback_vertical_velocity(double);


    // Correction - pose - YOLO
    void callback_pose_yolo(VectorXd);
    // Correction - pose - SLAM
    void callback_pose_slam(VectorXd);
    // EKF UPDATE - position x and y and orientation yaw - YOLO
    void callback_yolo_xy_y(VectorXd);


    //Measurement model of the SLAM algorithm
    VectorXd h_slam(VectorXd);
    //Jacobian of the measurement model of the SLAM
    MatrixXd H_slam_jac(VectorXd);



    // Convert Euler to Quaternion
    //VectorXd EulertoQuaternion(VectorXd);
    VectorXd EulertoQuaternion(VectorXd);
    VectorXd quat2eulerangle(VectorXd);
    MatrixXd angle2rotm(VectorXd);
    VectorXd rotm2angle(MatrixXd);

    // Get EKF States Data
    VectorXd get_states();
    // Get EKF States Data
    VectorXd get_raw_states();
    // Get Filtered IMU Data
    VectorXd get_IMU();
};
