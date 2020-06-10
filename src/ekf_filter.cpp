// kalman filter
// ROS stuffs
#include "ros/ros.h"
#include <sensor_msgs/Imu.h>
#include <geometry_msgs/PoseStamped.h>
#include <nav_msgs/Odometry.h>
#include <tf/tf.h>
#include <tf/transform_listener.h>
#include <tf/transform_broadcaster.h>


//Eigen stuff
#include <eigen3/Eigen/Cholesky>
#include <eigen3/Eigen/Dense>
#include <eigen3/Eigen/Core>

//File management
#include <fstream>
#include <iostream>
#include <sstream>
#include <sys/stat.h> // commands
#include <stdio.h>


// EKF lib
#include "EKF_class.h"



using namespace std;

// Global variables
#define N_STATES 15
bool enable_txt_log;
Eigen::VectorXd EKF_states_0(N_STATES);
Eigen::MatrixXd EKF_H(3,N_STATES);
Eigen::MatrixXd EKF_Q(N_STATES,N_STATES);
Eigen::MatrixXd EKF_Q_bar(6,6);
Eigen::MatrixXd EKF_R(3,3);
Eigen::MatrixXd EKF_R_yolo(6,6);
Eigen::MatrixXd EKF_P(N_STATES,N_STATES);


//Flags for new measured data
bool new_u_imu = false; //imu
bool new_z_gps = false; //GPS

//Flag for initialization
bool filter_init = false;

//Variables to store new received data
Eigen::VectorXd u_imu(10,1);
Eigen::VectorXd z_gps(7,1);


// callbacks for IMU data
void imu_callback(const sensor_msgs::Imu::ConstPtr& msg){

  u_imu << msg->linear_acceleration.x, msg->linear_acceleration.y, msg->linear_acceleration.z,
         1*msg->angular_velocity.x, 1*msg->angular_velocity.y, 1*msg->angular_velocity.z,
         msg->orientation.w, msg->orientation.x, msg->orientation.y, msg->orientation.z;

  new_u_imu = true;

}


void gps_callback(const geometry_msgs::PoseStamped::ConstPtr& msg){
  Eigen::VectorXd new_gps(7,1);
  double alpha = 0.3;

  filter_init = true;//TEMPORARY!!!!!!!!!!!!!!!

  //Get position
  z_gps.block(0,0,3,1) << msg->pose.position.x, msg->pose.position.y, msg->pose.position.z;

  //Get orientation
  z_gps.block(3,0,4,1) << msg->pose.orientation.w, msg->pose.orientation.x, msg->pose.orientation.y, msg->pose.orientation.z;

  new_z_gps = true;
}





void load_EKF_parameters(ros::NodeHandle nh){

  std::vector<double> temp_vector;

  try{

    nh.getParam("/EKF/enable_txt_log", enable_txt_log);

    nh.getParam("/EKF/states_0", temp_vector);
    EKF_states_0 = VectorXd::Map(temp_vector.data(), temp_vector.size());

    nh.getParam("/EKF/H", temp_vector);
    EKF_H = Eigen::Map<Eigen::Matrix<double, N_STATES, 3> >(temp_vector.data()).transpose();

    nh.getParam("/EKF/Q", temp_vector);
    EKF_Q = Eigen::Map<Eigen::Matrix<double, N_STATES, N_STATES> >(temp_vector.data()).transpose();

    nh.getParam("/EKF/Q_bar", temp_vector);
    EKF_Q_bar = Eigen::Map<Eigen::Matrix<double, 6, 6> >(temp_vector.data()).transpose();

    nh.getParam("/EKF/R", temp_vector);
    EKF_R = Eigen::Map<Eigen::Matrix<double, 3, 3> >(temp_vector.data()).transpose();

    nh.getParam("/EKF/P", temp_vector);
    EKF_P = Eigen::Map<Eigen::Matrix<double, N_STATES, N_STATES> >(temp_vector.data()).transpose();


    printf("\33[92m\nThe following EKF parameters were loaded:\33[0m\n\n");
    cout << "\33[92menable_txt_log:\n" << enable_txt_log << "\33[0m" << endl << endl;
    cout << "\33[92mEKF_states_0:\n" << EKF_states_0 << "\33[0m" << endl << endl;
    cout << "\33[92mEKF_H:\n" << EKF_H << "\33[0m" << endl << endl;
    cout << "\33[92mEKF_Q:\n" << EKF_Q << "\33[0m" << endl << endl;
    cout << "\33[92mEKF_Q_bar:\n" << EKF_Q_bar << "\33[0m" << endl << endl;
    cout << "\33[92mEKF_R:\n" << EKF_R << "\33[0m" << endl << endl;
    cout << "\33[92mEKF_P:\n" << EKF_P << "\33[0m" << endl << endl;


  }catch(...){

    printf("\33[41mError when trying to read EKF parameters\33[0m");

  }



}



int main(int argc, char *argv[]){
	//Initialize node
	ros::init(argc, argv, "ekf_filter");

	// Get handle
	ros::NodeHandle n;
	// Handle for getting parameters
	ros::NodeHandle n2("ekf_filter");

	// Callbacks
    ros::Subscriber sub_imu = n.subscribe("/uav/sensors/imu", 1, &imu_callback);
    ros::Subscriber sub_gps = n.subscribe("/gps", 1, &gps_callback);

    // Publishers
    ros::Publisher ekf_pub = n.advertise<nav_msgs::Odometry>("/ekf_odom", 1);

    // Pub Messages
    nav_msgs::Odometry ekf_odom;


    // Initialize more variables
	double dt = 0.0;
	int imu_step = 1; // decimate imu data
	int gps_step = 1; // decimate beacons ddata
	Eigen::Matrix3d R;
	Eigen::Vector3d velo_w;


	//Read parameters of the EKF filter
    load_EKF_parameters(n2);


	// EKF filter object
	double frequency = 150.0;
	EKF_class *Filter;
	Filter = new EKF_class(EKF_states_0, frequency, EKF_H, EKF_Q, EKF_Q_bar, EKF_R, EKF_P);
	Eigen::VectorXd ekf_states(N_STATES);
	//Local states variable
    ekf_states.setZero();

    //Define frequency
    ros::Rate loop_rate(frequency*1.05);



    // ================================ Create log =================================
    FILE *state_log;
    FILE *imu_log;
    FILE *gps_log;

    if(enable_txt_log){
    	std::string log_path;
    	std::string log_path_1;
    	std::string log_path_2;
    	std::string log_path_3;
    	if (n2.getParam ("log_path", log_path)){
    		log_path_1 = log_path+"state_log.txt";
      		log_path_2 = log_path+"imu_log.txt";
      		log_path_3 = log_path+"gps_log.txt";
    	}
    	try{
    		state_log = fopen(log_path_1.c_str(),"w");
      		imu_log = fopen(log_path_2.c_str(),"w");
      		gps_log = fopen(log_path_3.c_str(),"w");
    	}catch(...){
      		cout << "\33[41mError when oppening the log files\33[0m" << endl;
    	}
    }



    // ================================ Main loop =================================
	while (ros::ok()){
		if(filter_init){
	      // ----------  ----------  ---------- ----------  ----------
	      //Publish the pose estimation of the robot

	      ekf_odom.header.frame_id = "world";
	      ekf_odom.header.stamp = ros::Time::now();
	      ekf_odom.pose.pose.position.x = ekf_states(0);
	      ekf_odom.pose.pose.position.y = ekf_states(1);
	      ekf_odom.pose.pose.position.z = ekf_states(2);
	      ekf_odom.pose.pose.orientation.w = ekf_states(3);
	      ekf_odom.pose.pose.orientation.x = ekf_states(4);
	      ekf_odom.pose.pose.orientation.y = ekf_states(5);
	      ekf_odom.pose.pose.orientation.z = ekf_states(6);

	      Eigen::Quaterniond quat_states(ekf_states(3),ekf_states(4),ekf_states(5),ekf_states(6));
	      R = quat_states.toRotationMatrix();
	      velo_w = R*ekf_states.block(7,0,3,1);

	      ekf_odom.twist.twist.linear.x = velo_w(0);
	      ekf_odom.twist.twist.linear.y = velo_w(1);
	      ekf_odom.twist.twist.linear.z = velo_w(2);

	      // ----------  ----------  ---------- ----------  ----------

	      //Publish a transform between the world frame and the filter estimation
	      static tf::TransformBroadcaster br;
	      tf::Transform transform;
	      transform.setOrigin( tf::Vector3(ekf_states(0), ekf_states(1), ekf_states(2)) );
	      tf::Quaternion q(ekf_states(4),ekf_states(5),ekf_states(6),ekf_states(3));
	      transform.setRotation(q);
	      br.sendTransform(tf::StampedTransform(transform, ros::Time::now(), "world", "/ekf/estimation"));
	    }
	}

	if(enable_txt_log){
    //Close log files
    	fclose(state_log);
    	fclose(imu_log);
    	fclose(gps_log);
  	}

}