#include <math.h>
#include <time.h>
#include <stdio.h>
#include <stdlib.h>
#include <ros/ros.h>
#include <Eigen/Dense> 
#include <message_filters/subscriber.h>
#include <message_filters/synchronizer.h>
#include <message_filters/sync_policies/approximate_time.h>

#include <std_msgs/Bool.h>
#include <nav_msgs/Path.h>
#include <nav_msgs/Odometry.h>
#include <geometry_msgs/TwistStamped.h>
#include <gazebo_msgs/ModelState.h>
#include <sensor_msgs/Imu.h>
#include <sensor_msgs/PointCloud2.h>
#include <sensor_msgs/Joy.h>
#include <visualization_msgs/Marker.h>

#include <tf/transform_datatypes.h>
#include <tf/transform_broadcaster.h>
#include <tf/transform_listener.h> // Added for odometry processing

#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>

#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <geometry_msgs/PointStamped.h> 
#include <geometry_msgs/Twist.h> // For /cmd_vel

using namespace std;

const double PI = 3.1415926;

// ======================================================================
// GLOBAL STATE AND CONFIGURATION (Types fixed from float to double for tf)
// ======================================================================

bool use_gazebo_time = false;
double cameraOffsetZ = 0;
double sensorOffsetX = 0;
double sensorOffsetY = 0;
double vehicleHeight = 0.75;
double terrainVoxelSize = 0.05;
double groundHeightThre = 0.1;
bool adjustZ = false;
double terrainRadiusZ = 0.5;
int minTerrainPointNumZ = 10;
double smoothRateZ = 0.2;
bool adjustIncl = false;
double terrainRadiusIncl = 1.5;
int minTerrainPointNumIncl = 500;
double smoothRateIncl = 0.2;
double InclFittingThre = 0.2;
double maxIncl = 30.0;

const int systemDelay = 5;
int systemInitCount = 0;
bool systemInited = false;

// Robot Pose Variables (Changed to double to fix getRPY error)
double vehicleX = 0;
double vehicleY = 0;
double vehicleZ = 0;
double vehicleRoll = 0;
double vehiclePitch = 0;
double vehicleYaw = 0;

// Velocity/Control Variables
double vehicleYawRate = 0;
double vehicleSpeed = 0;

double terrainZ = 0;
double terrainRoll = 0;
double terrainPitch = 0;

// Odometry History Stack (FIX: All arrays are now declared globally as double)
const int stackNum = 400;
double vehicleXStack[stackNum];
double vehicleYStack[stackNum];
double vehicleZStack[stackNum];
double vehicleRollStack[stackNum];
double vehiclePitchStack[stackNum];
double vehicleYawStack[stackNum];
double terrainRollStack[stackNum];
double terrainPitchStack[stackNum];
double odomTimeStack[stackNum];
int odomSendIDPointer = -1;
int odomRecIDPointer = 0;
double goalX = 0;
double goalY = 0; 

pcl::PointCloud<pcl::PointXYZI>::Ptr scanData(new pcl::PointCloud<pcl::PointXYZI>());
pcl::PointCloud<pcl::PointXYZI>::Ptr terrainCloud(new pcl::PointCloud<pcl::PointXYZI>());
pcl::PointCloud<pcl::PointXYZI>::Ptr terrainCloudIncl(new pcl::PointCloud<pcl::PointXYZI>());
pcl::PointCloud<pcl::PointXYZI>::Ptr terrainCloudDwz(new pcl::PointCloud<pcl::PointXYZI>());

pcl::VoxelGrid<pcl::PointXYZI> terrainDwzFilter;
ros::Publisher* pubScanPointer = NULL;
ros::Publisher* pubMotionPointer = NULL;
tf::TransformBroadcaster *tfBroadcasterPointer = NULL;


// ======================================================================
// HANDLER FUNCTIONS
// ======================================================================

// NEW HANDLER: Integrates External Odometry (e.g., from EKF via remapping)
void odometryHandler(const nav_msgs::Odometry::ConstPtr& odomIn)
{
  // 1. Update Global State Variables (replaces simulation integration)
  vehicleX = odomIn->pose.pose.position.x;
  vehicleY = odomIn->pose.pose.position.y;
  vehicleZ = odomIn->pose.pose.position.z;

  // Convert Quaternion to RPY (Fixed type issue: vehicleRoll/Pitch/Yaw are now double)
  tf::Quaternion q;
  tf::quaternionMsgToTF(odomIn->pose.pose.orientation, q);
  tf::Matrix3x3(q).getRPY(vehicleRoll, vehiclePitch, vehicleYaw);

  // 2. Update Velocity/Rate Variables (used for command relay)
  vehicleSpeed = odomIn->twist.twist.linear.x;
  vehicleYawRate = odomIn->twist.twist.angular.z;

  // 3. Update Odometry History Stack (Used for Point Cloud Time Lookup)
  ros::Time odomTime = odomIn->header.stamp;
  odomSendIDPointer = (odomSendIDPointer + 1) % stackNum;
  odomTimeStack[odomSendIDPointer] = odomTime.toSec();
  
  // Stacking the current pose
  vehicleXStack[odomSendIDPointer] = vehicleX;
  vehicleYStack[odomSendIDPointer] = vehicleY;
  vehicleZStack[odomSendIDPointer] = vehicleZ;
  vehicleRollStack[odomSendIDPointer] = vehicleRoll;
  vehiclePitchStack[odomSendIDPointer] = vehiclePitch;
  vehicleYawStack[odomSendIDPointer] = vehicleYaw;
  
  // Stacking the terrain tilt (which comes from terrainCloudHandler, not odometry)
  terrainRollStack[odomSendIDPointer] = terrainRoll;
  terrainPitchStack[odomSendIDPointer] = terrainPitch;
  
  // 4. Publish TF for the new pose
  /*
  tf::StampedTransform odomTrans;
  odomTrans.stamp_ = odomTime;
  odomTrans.frame_id_ = "map";
  odomTrans.child_frame_id_ = "sensor";

  odomTrans.setRotation(tf::Quaternion(odomIn->pose.pose.orientation.x, odomIn->pose.pose.orientation.y,
                                       odomIn->pose.pose.orientation.z, odomIn->pose.pose.orientation.w));
  odomTrans.setOrigin(tf::Vector3(vehicleX, vehicleY, vehicleZ));
  tfBroadcasterPointer->sendTransform(odomTrans);
  */
}


void scanHandler(const sensor_msgs::PointCloud2::ConstPtr& scanIn)
{
  if (!systemInited) {
    systemInitCount++;
    if (systemInitCount > systemDelay) {
      systemInited = true;
    }
    return;
  }

  double scanTime = scanIn->header.stamp.toSec();

  if (odomSendIDPointer < 0)
  {
    return;
  }
  
  // Time-synchronized odometry lookup
  // This loop finds the pose closest in time to the scan time
  while (odomTimeStack[(odomRecIDPointer + 1) % stackNum] < scanTime &&
         odomRecIDPointer != (odomSendIDPointer + 1) % stackNum)
  {
    odomRecIDPointer = (odomRecIDPointer + 1) % stackNum;
  }

  // FIX: odomRecTime is now retrieved from the stack, making it locally defined
  double odomRecTime = odomTimeStack[odomRecIDPointer]; 
  
  // Retrieve the vehicle state at the time of the scan
  double vehicleRecX = vehicleXStack[odomRecIDPointer];
  double vehicleRecY = vehicleYStack[odomRecIDPointer];
  double vehicleRecZ = vehicleZStack[odomRecIDPointer];
  double vehicleRecRoll = vehicleRollStack[odomRecIDPointer];
  double vehicleRecPitch = vehiclePitchStack[odomRecIDPointer];
  double vehicleRecYaw = vehicleYawStack[odomRecIDPointer];

  // Point Cloud Registration (Transformation)
  
  // Compute rotation matrix using the time-synced pose
  Eigen::Matrix3f rotationMatrix;
  rotationMatrix = Eigen::AngleAxisf(vehicleRecYaw, Eigen::Vector3f::UnitZ()) *
                  Eigen::AngleAxisf(vehicleRecPitch, Eigen::Vector3f::UnitY()) *
                  Eigen::AngleAxisf(vehicleRecRoll, Eigen::Vector3f::UnitX());

  scanData->clear();
  std::vector<int> scanInd;
  pcl::fromROSMsg(*scanIn, *scanData);
  pcl::removeNaNFromPointCloud(*scanData, *scanData, scanInd);

  int scanDataSize = scanData->points.size();
  for (int i = 0; i < scanDataSize; i++)
  {
    Eigen::Vector3f point(scanData->points[i].x, scanData->points[i].y, scanData->points[i].z);

    // Apply rotation and translation
    point = rotationMatrix * point;

    point.x() += vehicleRecX;
    point.y() += vehicleRecY;
    point.z() += vehicleRecZ;

    scanData->points[i].x = point.x();
    scanData->points[i].y = point.y();
    scanData->points[i].z = point.z();
  }

  // Publish registered scan messages
  sensor_msgs::PointCloud2 scanData2;
  pcl::toROSMsg(*scanData, scanData2);
  scanData2.header.stamp = ros::Time().fromSec(odomRecTime);
  scanData2.header.frame_id = "map";
  pubScanPointer->publish(scanData2);
}

void terrainCloudHandler(const sensor_msgs::PointCloud2ConstPtr& terrainCloud2)
{
  // This function remains the same as its logic only reads global state variables (vehicleX/Y)
  // for local terrain lookup and updates terrainRoll/Pitch/Z.

  if (!adjustZ && !adjustIncl)
  {
    return;
  }

  terrainCloud->clear();
  pcl::fromROSMsg(*terrainCloud2, *terrainCloud);

  pcl::PointXYZI point;
  terrainCloudIncl->clear();
  int terrainCloudSize = terrainCloud->points.size();
  double elevMean = 0;
  int elevCount = 0;
  bool terrainValid = true;
  for (int i = 0; i < terrainCloudSize; i++)
  {
    point = terrainCloud->points[i];

    // Note: uses global vehicleX and vehicleY (latest filtered state)
    float dis = sqrt((point.x - vehicleX) * (point.x - vehicleX) + (point.y - vehicleY) * (point.y - vehicleY));

    if (dis < terrainRadiusZ)
    {
      if (point.intensity < groundHeightThre)
      {
        elevMean += point.z;
        elevCount++;
      }
      else
      {
        terrainValid = false;
      }
    }

    if (dis < terrainRadiusIncl && point.intensity < groundHeightThre)
    {
      terrainCloudIncl->push_back(point);
    }
  }

  if (elevCount >= minTerrainPointNumZ)
    elevMean /= elevCount;
  else
    terrainValid = false;

  if (terrainValid && adjustZ)
  {
    // Updates global terrainZ
    terrainZ = (1.0 - smoothRateZ) * terrainZ + smoothRateZ * elevMean;
  }

  // Voxel filter for terrain
  terrainCloudDwz->clear();
  terrainDwzFilter.setInputCloud(terrainCloudIncl);
  terrainDwzFilter.filter(*terrainCloudDwz);
  int terrainCloudDwzSize = terrainCloudDwz->points.size();

  if (terrainCloudDwzSize < minTerrainPointNumIncl || !terrainValid)
  {
    return;
  }

  // Least Squares Fitting for Inclination (Roll/Pitch)
  cv::Mat matA(terrainCloudDwzSize, 2, CV_32F, cv::Scalar::all(0));
  cv::Mat matAt(2, terrainCloudDwzSize, CV_32F, cv::Scalar::all(0));
  cv::Mat matAtA(2, 2, CV_32F, cv::Scalar::all(0));
  cv::Mat matB(terrainCloudDwzSize, 1, CV_32F, cv::Scalar::all(0));
  cv::Mat matAtB(2, 1, CV_32F, cv::Scalar::all(0));
  cv::Mat matX(2, 1, CV_32F, cv::Scalar::all(0));

  int inlierNum = 0;
  matX.at<float>(0, 0) = terrainPitch;
  matX.at<float>(1, 0) = terrainRoll;
  for (int iterCount = 0; iterCount < 5; iterCount++)
  {
    int outlierCount = 0;
    for (int i = 0; i < terrainCloudDwzSize; i++)
    {
      point = terrainCloudDwz->points[i];

      matA.at<float>(i, 0) = -point.x + vehicleX;
      matA.at<float>(i, 1) = point.y - vehicleY;
      matB.at<float>(i, 0) = point.z - elevMean;

      if (fabs(matA.at<float>(i, 0) * matX.at<float>(0, 0) + matA.at<float>(i, 1) * matX.at<float>(1, 0) -
               matB.at<float>(i, 0)) > InclFittingThre &&
          iterCount > 0)
      {
        matA.at<float>(i, 0) = 0;
        matA.at<float>(i, 1) = 0;
        matB.at<float>(i, 0) = 0;
        outlierCount++;
      }
    }

    cv::transpose(matA, matAt);
    matAtA = matAt * matA;
    matAtB = matAt * matB;
    cv::solve(matAtA, matAtB, matX, cv::DECOMP_QR);

    if (inlierNum == terrainCloudDwzSize - outlierCount)
      break;
    inlierNum = terrainCloudDwzSize - outlierCount;
  }

  if (inlierNum < minTerrainPointNumIncl || fabs(matX.at<float>(0, 0)) > maxIncl * PI / 180.0 ||
      fabs(matX.at<float>(1, 0)) > maxIncl * PI / 180.0)
  {
    terrainValid = false;
  }

  if (terrainValid && adjustIncl)
  {
    // Updates global terrainPitch/Roll
    terrainPitch = (1.0 - smoothRateIncl) * terrainPitch + smoothRateIncl * matX.at<float>(0, 0);
    terrainRoll = (1.0 - smoothRateIncl) * terrainRoll + smoothRateIncl * matX.at<float>(1, 0);
  }
}

void speedHandler(const geometry_msgs::TwistStamped::ConstPtr& speedIn)
{
  // This is kept to allow control commands to modulate the velocity variables
  vehicleSpeed = 0.1 * speedIn->twist.linear.x;
  vehicleYawRate = 0.1 * speedIn->twist.angular.z;
}

void goalHandlerR(const geometry_msgs::PointStamped::ConstPtr& goal) 
{
  goalX = goal->point.x;
  goalY = goal->point.y;
}

// ======================================================================
// MAIN FUNCTION
// ======================================================================
int main(int argc, char** argv)
{
  ros::init(argc, argv, "vehicleStateProcessor");
  ros::NodeHandle nh;
  ros::NodeHandle nhPrivate = ros::NodeHandle("~");

  // Get Parameters (UNCHANGED)
  nhPrivate.getParam("use_gazebo_time", use_gazebo_time);
  nhPrivate.getParam("cameraOffsetZ", cameraOffsetZ);
  nhPrivate.getParam("sensorOffsetX", sensorOffsetX);
  nhPrivate.getParam("sensorOffsetY", sensorOffsetY);
  nhPrivate.getParam("vehicleHeight", vehicleHeight);
  nhPrivate.getParam("vehicleX", vehicleX);
  nhPrivate.getParam("vehicleY", vehicleY);
  nhPrivate.getParam("vehicleZ", vehicleZ);
  nhPrivate.getParam("terrainZ", terrainZ);
  nhPrivate.getParam("vehicleYaw", vehicleYaw);
  nhPrivate.getParam("terrainVoxelSize", terrainVoxelSize);
  nhPrivate.getParam("groundHeightThre", groundHeightThre);
  nhPrivate.getParam("adjustZ", adjustZ);
  nhPrivate.getParam("terrainRadiusZ", terrainRadiusZ);
  nhPrivate.getParam("minTerrainPointNumZ", minTerrainPointNumZ);
  nhPrivate.getParam("smoothRateZ", smoothRateZ);
  nhPrivate.getParam("adjustIncl", adjustIncl);
  nhPrivate.getParam("terrainRadiusIncl", terrainRadiusIncl);
  nhPrivate.getParam("minTerrainPointNumIncl", minTerrainPointNumIncl);
  nhPrivate.getParam("smoothRateIncl", smoothRateIncl);
  nhPrivate.getParam("InclFittingThre", InclFittingThre);
  nhPrivate.getParam("maxIncl", maxIncl);
  
  // --- SUBSCRIBERS ---
  // NEW: Subscribes to the external filtered odometry (remapped from /odometry/filtered)
  ros::Subscriber subOdometry = nh.subscribe<nav_msgs::Odometry>("/state_estimation", 5, odometryHandler); 
  
  ros::Subscriber subScan = nh.subscribe<sensor_msgs::PointCloud2>("/velodyne_points", 2, scanHandler);
  ros::Subscriber subTerrainCloud = nh.subscribe<sensor_msgs::PointCloud2>("/terrain_map", 2, terrainCloudHandler);
  ros::Subscriber subSpeed = nh.subscribe<geometry_msgs::TwistStamped>("/cmd_vel_in", 5, speedHandler);
  ros::Subscriber subGoal = nh.subscribe<geometry_msgs::PointStamped> ("/way_point", 5, goalHandlerR);

  // --- PUBLISHERS ---
  ros::Publisher pubScan = nh.advertise<sensor_msgs::PointCloud2>("/registered_scan", 2);
  pubScanPointer = &pubScan;
  
  ros::Publisher huskymotion = nh.advertise<geometry_msgs::Twist>("/cmd_vel",5);
  pubMotionPointer = &huskymotion;

  // --- TF BROADCASTER (Used in odometryHandler now) ---
  tf::TransformBroadcaster tfBroadcaster;
  tfBroadcasterPointer = &tfBroadcaster;

  terrainDwzFilter.setLeafSize(terrainVoxelSize, terrainVoxelSize, terrainVoxelSize);

  printf("\nVehicle State Processor started. Using external Odometry for pose.\n\n");

  geometry_msgs::Twist twist_msg;
  ros::Rate rate(200); 
  bool status = ros::ok();
  
  while (status)
  {
    ros::spinOnce();

    // The motion integration loop is REMOVED.
    // vehicleX/Y/Z/Roll/Pitch/Yaw are updated by the odometryHandler callback.
    
    // Command Relay: Use the speed/rate updated by the odometryHandler to command the vehicle
    twist_msg.linear.x = vehicleSpeed;
    twist_msg.linear.y = 0.0;
    twist_msg.linear.z = 0.0;
    twist_msg.angular.x = 0.0;
    twist_msg.angular.y = 0.0;
    twist_msg.angular.z = vehicleYawRate;
    pubMotionPointer->publish(twist_msg);

    status = ros::ok();
    rate.sleep();
  }

  return 0;
}