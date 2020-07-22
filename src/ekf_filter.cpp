// kalman filter

/*
Universidade Federal de Minas Gerais (UFMG) - 2020
Laboraorio CORO
Instituto Tecnologico Vale (ITV)
Contact:
Victor R. F. Miranda, <victormrfm@gmail.com>
*/

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
#define PRINT_STATES 1
#define N_STATES 15
bool enable_txt_log;
Eigen::VectorXd EKF_states_0(N_STATES);
Eigen::MatrixXd EKF_H(9,N_STATES);
Eigen::MatrixXd EKF_Q(N_STATES,N_STATES);
Eigen::MatrixXd EKF_Q_bar(6,6);
Eigen::MatrixXd EKF_R(9,9);
Eigen::MatrixXd EKF_P(N_STATES,N_STATES);

// vehicle number
std::string vehicle_number;


//Counters for decimation
int count_imu = 0;
int count_gps = 0;
int count_encoder = 0;


//Flags for new measured data
bool new_u_imu = false;     //imu
bool new_z_gps = false;     //GPS
bool new_z_orient = false;  //Orientation (IMU)
bool new_z_vel = false;     //Encoder

//Variables to transform quaternion <-> Euler
Eigen::VectorXd pos_euler(3,1);
Eigen::VectorXd pos_quat(7,1);

//Flag for initialization
bool filter_init = false;

//Variables to store new received data
Eigen::VectorXd u_imu(10,1);
Eigen::VectorXd orientation_imu(4,1);
Eigen::VectorXd vel_encoder(3,1);
Eigen::VectorXd z_gps(3,1);
Eigen::VectorXd gt_data(10,1);


void GT_callback(const tf2_msgs::TFMessage::ConstPtr &msg){

  if(msg->transforms[0].header.frame_id == "world"){

    if(msg->transforms[0].child_frame_id != "base_footprintmaster") return;

    // Estimate time elapsed since last measurement
    geometry_msgs::TransformStamped ugv_tf = msg->transforms[0];
    static ros::Time t_anterior = ugv_tf.header.stamp;
    ros::Duration dt = ugv_tf.header.stamp - t_anterior;
    t_anterior = ugv_tf.header.stamp ;
    double dt_sec = dt.toSec();
    if(dt_sec == 0.0 )
      dt_sec = 1/50.0;

    // Compute velocity
    gt_data.block(7,0,3,1) << (msg->transforms[0].transform.translation.x - gt_data(0))/dt_sec, (msg->transforms[0].transform.translation.y - gt_data(1))/dt_sec, (msg->transforms[0].transform.translation.z - gt_data(2))/dt_sec;

    //Get position
    gt_data.block(0,0,3,1) << msg->transforms[0].transform.translation.x, msg->transforms[0].transform.translation.y, msg->transforms[0].transform.translation.z;

    //Get orientation
    gt_data.block(3,0,4,1) << msg->transforms[0].transform.rotation.w, msg->transforms[0].transform.rotation.x, msg->transforms[0].transform.rotation.y, msg->transforms[0].transform.rotation.z;

  }
}


// callbacks for IMU data
void imu_callback(const sensor_msgs::Imu::ConstPtr& msg){

  if (msg->header.frame_id == (("imu_link_")+vehicle_number)){
    u_imu << -msg->linear_acceleration.x, -msg->linear_acceleration.y, -msg->linear_acceleration.z,
         1*msg->angular_velocity.x, 1*msg->angular_velocity.y, 1*msg->angular_velocity.z,
         msg->orientation.w, msg->orientation.x, msg->orientation.y, msg->orientation.z;

    orientation_imu << msg->orientation.w, msg->orientation.x, msg->orientation.y, msg->orientation.z;

    new_u_imu = true;
    new_z_orient = true;
  }  

}

// callback Encoder
void encoder_callback(const nav_msgs::Odometry::ConstPtr& msg){

  if (msg->header.frame_id == (("vehicle_")+vehicle_number)){
    Eigen::VectorXd new_vel(3,1);
    double alpha = 0.3;

    //Get vel
    vel_encoder << msg->twist.twist.linear.x, msg->twist.twist.linear.y, msg->twist.twist.linear.z;

    new_z_vel = true;
  }
}


// void gps_callback(const geometry_msgs::PoseStamped::ConstPtr& msg){
void gps_callback(const nav_msgs::Odometry::ConstPtr& msg){
  if (msg->header.frame_id == (("gps_link_")+vehicle_number)){
    Eigen::VectorXd new_gps(3,1);
    double alpha = 0.3;

    // filter_init = true;//TEMPORARY!!!!!!!!!!!!!!!

    //Get position
    z_gps << msg->pose.pose.position.x, msg->pose.pose.position.y, msg->pose.pose.position.z;

    new_z_gps = true;
  }
}





void load_EKF_parameters(ros::NodeHandle nh){

  std::vector<double> temp_vector;

  try{

    nh.getParam("/EKF/enable_txt_log", enable_txt_log);

    nh.getParam("/EKF/states_0", temp_vector);
    EKF_states_0 = VectorXd::Map(temp_vector.data(), temp_vector.size());

    nh.getParam("/EKF/H", temp_vector);
    EKF_H = Eigen::Map<Eigen::Matrix<double, N_STATES, 9> >(temp_vector.data()).transpose();

    nh.getParam("/EKF/Q", temp_vector);
    EKF_Q = Eigen::Map<Eigen::Matrix<double, N_STATES, N_STATES> >(temp_vector.data()).transpose();

    nh.getParam("/EKF/Q_bar", temp_vector);
    EKF_Q_bar = Eigen::Map<Eigen::Matrix<double, 6, 6> >(temp_vector.data()).transpose();

    nh.getParam("/EKF/R", temp_vector);
    EKF_R = Eigen::Map<Eigen::Matrix<double, 9, 9> >(temp_vector.data()).transpose();

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
    cout << "\33[92mVehicle Number:\t"<<vehicle_number<<"\33[0m"<<endl<<endl;


  }catch(...){

    printf("\33[41mError when trying to read EKF parameters\33[0m");

  }



}



int main(int argc, char *argv[]){
  // set vehicle number
  vehicle_number = argv[1];


  //Initialize node
  ros::init(argc, argv, "ekf_filter");

  // Get handle
  ros::NodeHandle n;
  // Handle for getting parameters
  ros::NodeHandle n2("ekf_filter");

  // Callbacks
  ros::Subscriber sub_imu = n.subscribe("/imu_data", 1, &imu_callback);
  ros::Subscriber sub_gps = n.subscribe("/gps", 1, &gps_callback);
  ros::Subscriber sub_encoder = n.subscribe("/odom", 1, &encoder_callback);
  ros::Subscriber sub_GT = n.subscribe("/tf", 1, &GT_callback);


  // Publishers
  ros::Publisher ekf_pub = n.advertise<nav_msgs::Odometry>("/ekf_odom", 1);

  // Pub Messages
  nav_msgs::Odometry ekf_odom;


    // Initialize more variables
  double dt = 0.0;
  double t_init = ros::Time::now().toSec();
  double time_log = 0.0;
  int imu_step = 1; // decimate imu data
  int gps_step = 1; // decimate beacons ddata
  int vel_step = 1;
  Eigen::Matrix3d R;
  Eigen::Vector3d velo_w;
  double t_current = ros::Time::now().toSec();
  double t_previous = ros::Time::now().toSec();


  //Read parameters of the EKF filter
  load_EKF_parameters(n2);


  // EKF filter object
  double frequency = 50.0;
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
  FILE *gt_log;
  FILE *encoder_log;

  if(enable_txt_log){
    std::string log_path;
    std::string log_path_1;
    std::string log_path_2;
    std::string log_path_3;
    std::string log_path_4;
    std::string log_path_5;
    if (n2.getParam ("/EKF/log_path", log_path)){
      log_path_1 = log_path+"state_log.txt";
      log_path_2 = log_path+"imu_log.txt";
      log_path_3 = log_path+"gps_log.txt";
      log_path_4 = log_path+"GT_log.txt";
      log_path_5 = log_path+"encoder_log.txt";
    }
    try{
      state_log = fopen(log_path_1.c_str(),"w");
      imu_log = fopen(log_path_2.c_str(),"w");
      gps_log = fopen(log_path_3.c_str(),"w");
      gt_log = fopen(log_path_4.c_str(),"w");
      encoder_log = fopen(log_path_5.c_str(),"w");
    }catch(...){
        cout << "\33[41mError when oppening the log files\33[0m" << endl;
    }
  }



    // ================================ Main loop =================================
  while (ros::ok()){

    // ==================================================
    // Perform a prediction step with the IMU information
    // ==================================================
    if(new_u_imu == true){
      new_u_imu = false;

      //Compute dt
      t_current = ros::Time::now().toSec();
      dt = t_current - t_previous;

      t_previous = t_current;

      // if (filter_init==true){


        //Update the IMU data in the filter object
        double Ts = dt;
        Filter->callback_imu(u_imu.block(3,0,3,1), u_imu.block(0,0,3,1), Ts); //(gyro,accel)
        //Perform a prediction step in the filter
        Filter->prediction();
        // Get the current states of the filter
        ekf_states = Filter->get_states();


#ifdef PRINT_STATES
        cout << "\33[40mpos:" << ekf_states.block(0,0,3,1).transpose() << "\33[0m" << endl;
        cout << "\33[99mrpy:" << Filter->quat2eulerangle(ekf_states.block(3,0,4,1)).transpose() << "\33[0m" << endl;
        cout << "\33[40mvel:" << ekf_states.block(7,0,3,1).transpose() << "\33[0m" << endl;
        cout << "\33[99mvel_World:" << velo_w.transpose() << "\33[0m" << endl;
        cout << endl;
#endif

      //Save data to log files
      if(enable_txt_log){
        
        time_log = ros::Time::now().toSec() - t_init;

        //Save imu data
        fprintf(imu_log,"%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\n",u_imu(0),u_imu(1),u_imu(2),u_imu(3),u_imu(4),u_imu(5),u_imu(6),u_imu(7),u_imu(8),u_imu(9),time_log);
        fflush(imu_log);

        //Save states
        fprintf(state_log,"%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\n",ekf_states(0),ekf_states(1),ekf_states(2),ekf_states(3),ekf_states(4),ekf_states(5),ekf_states(6),ekf_states(7),ekf_states(8),ekf_states(9),time_log);
        fflush(state_log);

        //Save GT
        fprintf(gt_log,"%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\n",gt_data(0),gt_data(1),gt_data(2),gt_data(3),gt_data(4),gt_data(5),gt_data(6), gt_data(7), gt_data(8), gt_data(9),time_log);
        fflush(gt_log);
      }


    // ===========================================
    // Update filter with information from Orinetaion (IMU)
    // ===========================================

        //Call the update for Orientation
        Filter->callback_orientation(orientation_imu);

      // }
    }



    // ===========================================
    // Update filter with information from GPS
    // ===========================================
    if(new_z_gps == true){
      new_z_gps = false;
      count_gps++;

      if (count_gps == gps_step){
        count_gps = 0;

        // if (filter_init==true){

          //Call the update
          Filter->callback_position(z_gps);


        if (enable_txt_log){
          time_log = ros::Time::now().toSec() - t_init;
          //Save GPS
          fprintf(gps_log,"%f\t%f\t%f\t%f\n",z_gps(0),z_gps(1),z_gps(2),time_log);
          fflush(gps_log);
        }

        // }
      }
    }


    // ===========================================
    // Update filter with information from Encoder
    // ===========================================
    if(new_z_vel == true){
      new_z_vel = false;
      count_encoder++;

      if (count_encoder == vel_step){
        count_encoder = 0;

        // if (filter_init==true){

          //Call the update
          Filter->callback_velocity(vel_encoder);


        if (enable_txt_log){
          time_log = ros::Time::now().toSec() - t_init;
          //Save Encoder
          fprintf(encoder_log,"%f\t%f\t%f\t%f\n",vel_encoder(0),vel_encoder(1),vel_encoder(2),time_log);
          fflush(encoder_log);
        }

        // }
      }
    }


    // if(filter_init){
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
      // }


    //Check callbacks and wait
    ros::spinOnce();
    loop_rate.sleep();


  }//end while



  if(enable_txt_log){
    //Close log files
      fclose(state_log);
      fclose(imu_log);
      fclose(gps_log);
      fclose(encoder_log);
      fclose(gt_log);
    }

}
