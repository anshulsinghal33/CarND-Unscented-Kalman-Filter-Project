#include "ukf.h"
#include "Eigen/Dense"

using namespace std;
using std::vector;

/**
 * Initializes Unscented Kalman filter
 */
UKF::UKF() {
  // if this is false, laser measurements will be ignored (except during init)
  use_laser_ = true;

  // if this is false, radar measurements will be ignored (except during init)
  use_radar_ = true;

  // initial state vector
  x_ = VectorXd(5);

  // initial covariance matrix
  P_ = MatrixXd(5, 5);

  // Process noise standard deviation longitudinal acceleration in m/s^2
  std_a_ = 2;

  // Process noise standard deviation yaw acceleration in rad/s^2
  std_yawdd_ = 0.3;
  
  /**
   * DO NOT MODIFY measurement noise values below.
   * These are provided by the sensor manufacturer.
   */

  // Laser measurement noise standard deviation position1 in m
  std_laspx_ = 0.15;

  // Laser measurement noise standard deviation position2 in m
  std_laspy_ = 0.15;

  // Radar measurement noise standard deviation radius in m
  std_radr_ = 0.3;

  // Radar measurement noise standard deviation angle in rad
  std_radphi_ = 0.03;

  // Radar measurement noise standard deviation radius change in m/s
  std_radrd_ = 0.3;
  
  /**
   * End DO NOT MODIFY section for measurement noise values 
   */
  
  /**
   * TODO: Complete the initialization. See ukf.h for other member properties.
   * Hint: one or more values initialized above might be wildly off...
   */
  
  // initially set to false, set to true in first call of ProcessMeasurement
  is_initialized_ = false;

  // time when the state is true, in us
  time_us_ = 0.0;

  // state dimension
  n_x_ = 5;

  // Augmented state dimension
  n_aug_ = 7;

  // Sigma point spreading parameter
  lambda_ = 3 - n_x_;

  // predicted sigma points matrix
  Xsig_pred_ = MatrixXd(n_x_, 2 * n_aug_ + 1);

  //create vector for weights
  weights_ = VectorXd(2 * n_aug_ + 1);

  // Measurement noise covariance matrices initialization
  R_radar_ = MatrixXd(3, 3);
  R_radar_ << std_radr_*std_radr_, 0, 0,
              0, std_radphi_*std_radphi_, 0,
              0, 0,std_radrd_*std_radrd_;
  R_lidar_ = MatrixXd(2, 2);
  R_lidar_ << std_laspx_*std_laspx_,0,
              0,std_laspy_*std_laspy_;

  // the current NIS for radar
  NIS_radar_ = 0.0;

  // the current NIS for laser
  NIS_laser_ = 0.0;
}

void UKF::NormAng(double *ang) {
    while (*ang > M_PI) *ang -= 2. * M_PI;
    while (*ang < -M_PI) *ang += 2. * M_PI;
}

UKF::~UKF() {}

void UKF::ProcessMeasurement(MeasurementPackage meas_package) {
  /**
   * TODO: Complete this function! Make sure you switch between lidar and radar
   * measurements.
   */
  if ((meas_package.sensor_type_ == MeasurementPackage::RADAR && use_radar_) ||
      (meas_package.sensor_type_ == MeasurementPackage::LASER && use_laser_)){

      // Initiallization

      if(!is_initialized_){
        
        // Initialize State and Covariance matrix
        x_ << 1,1,0,0,0;

        P_ << 1,0,0,0,0,
              0,1,0,0,0,
              0,0,1,0,0,
              0,0,0,1,0,
              0,0,0,0,1;

        // Get timestamp
        time_us_ = meas_package.timestamp_;

        if(meas_package.sensor_type_ == MeasurementPackage::LASER && use_laser_){

          x_(0) = meas_package.raw_measurements_(0);
          x_(1) = meas_package.raw_measurements_(1);

        }
        else if(meas_package.sensor_type_ == MeasurementPackage::RADAR && use_radar_){
        //Convert radar from polar to cartesian coordinates and initialize state.
          float rho = meas_package.raw_measurements_[0]; // range
          float phi = meas_package.raw_measurements_[1]; // bearing
          //float rho_dot = meas_package.raw_measurements_[2]; // velocity of rho
          // Coordinates convertion from polar to cartesian
          x_(0) = rho * cos(phi);
          x_(1) = rho * sin(phi);
        }

        //Done Initializing
        is_initialized_ = true;

        return;
      }

      // Prediction Step

      // Compute time elapsed
      float dt = (meas_package.timestamp_ - time_us_) / 1000000.0;
      time_us_ = meas_package.timestamp_;

      Prediction(dt);

      //Update Step
      if (meas_package.sensor_type_ == MeasurementPackage::LASER){
        UpdateLidar(meas_package);
      }
      else if (meas_package.sensor_type_ == MeasurementPackage::RADAR){
        UpdateRadar(meas_package);
      }
  }
}

void UKF::Prediction(double delta_t) {
  /**
   * TODO: Complete this function! Estimate the object's location. 
   * Modify the state vector, x_. Predict sigma points, the state, 
   * and the state covariance matrix.
   */
  
  // Step 1: Generate Sigma Points

  // Create sigma points matrix
  MatrixXd Xsig = MatrixXd(n_x_, 2 * n_x_ + 1);

  // Calculate sqaure root of P
  MatrixXd A = P_.llt().matrixL();

  // Set lambda for non-augmented sigma points
  lambda_ = 3 - n_x_;

  // Set first column of sigma point matrix
  Xsig.col(0) = x_;

  // Set remaining sigma points
  for (int i = 0; i < n_x_; i++)
  {
    Xsig.col(i + 1) = x_ + sqrt(lambda_ + n_x_) * A.col(i);
    Xsig.col(i + 1 + n_x_) = x_ - sqrt(lambda_ + n_x_) * A.col(i);
  }


  // Step 2: Augment Sigma Points

  // Create Augmented Mean vector
  VectorXd x_aug = VectorXd(n_aug_);

  // Create Augmented State Covariance Matrix
  MatrixXd P_aug = MatrixXd(n_aug_,n_aug_);

  // Create Augmented Sigma Points Matrix
  MatrixXd Xsig_aug = MatrixXd(n_aug_, 2 * n_aug_ +1);

  // Set lambda
  lambda_ = 3 - n_aug_;

  // Augmented Mean State
  x_aug.head(5) = x_;
  x_aug(5) = 0;
  x_aug(6) = 0;

  // Augmented Covariance Matrix
  P_aug.fill(0.0);
  P_aug.topLeftCorner(n_x_,n_x_) = P_;
  P_aug(5,5) = std_a_*std_a_;
  P_aug(6,6) = std_yawdd_*std_yawdd_;

  // Find square root of P_aug
  MatrixXd L = P_aug.llt().matrixL();

  // Create Augmented Sigma Points
  Xsig_aug.col(0) = x_aug;
  for(int i = 0; i < n_aug_; i++)
  {
    Xsig_aug.col(i+1) = x_aug + (sqrt(lambda_ + n_aug_)*L.col(i));
    Xsig_aug.col(i+1+n_aug_) = x_aug - (sqrt(lambda_ + n_aug_)*L.col(i));
  }

  // Step 3: Predict Sigma Points

  for(int i = 0; i < 2 * n_aug_+ 1; i++){

    double p_x      = Xsig_aug(0, i);
    double p_y      = Xsig_aug(1, i);
    double v        = Xsig_aug(2, i);
    double yaw      = Xsig_aug(3, i);
    double yawd     = Xsig_aug(4, i);
    double nu_a     = Xsig_aug(5, i);
    double nu_yawdd = Xsig_aug(6, i);

    // Predicted State Values
    double px_p, py_p;

    // Avoid Division by Zero
    if(fabs(yawd)>0.001){
      px_p = p_x + v/yawd*(sin(yaw + yawd * delta_t) - sin(yaw));
      py_p = p_y + v/yawd*(cos(yaw) - cos(yawd * delta_t + yaw));
    }
    else{
      px_p = p_x + v*delta_t * cos(yaw);
      py_p = p_y + v*delta_t * sin(yaw);
    }

    double v_p = v;
    double yaw_p = yaw + yawd * delta_t;
    double yawd_p = yawd;

    // Add noise
    px_p = px_p + 0.5 * nu_a * delta_t * delta_t * cos(yaw);
    py_p = py_p + 0.5 * nu_a * delta_t * delta_t * sin(yaw);
    v_p = v_p + nu_a*delta_t;

    yaw_p = yaw_p + 0.5*nu_yawdd*delta_t*delta_t;
    yawd_p = yawd_p + nu_yawdd*delta_t;

    // Write predicted sigma point into right column
    Xsig_pred_(0, i) = px_p;
    Xsig_pred_(1, i) = py_p;
    Xsig_pred_(2, i) = v_p;
    Xsig_pred_(3, i) = yaw_p;
    Xsig_pred_(4, i) = yawd_p;
  }

  // Step 4: Predict Mean and Covariance

  // Set Weights
  double weight_0 = lambda_/(lambda_ + n_aug_);
  weights_(0) = weight_0;

  for(int i = 1; i < 2 * n_aug_ + 1; i++){
    double weight = 0.5 / (n_aug_ + lambda_);
    weights_(i) = weight;
  }

  // Predict Mean
  x_.fill(0.0);
  for (int i = 0; i < 2 * n_aug_ + 1; i++) {  //iterate over sigma points
    x_ = x_ + weights_(i) * Xsig_pred_.col(i);
  }

  // Predict Covariance Matrix
  P_.fill(0.0);

  for(int i = 0; i < 2 * n_aug_ + 1; i++){
    VectorXd x_diff = Xsig_pred_.col(i) - x_;
    // Angle normalization
    while (x_diff(3)> M_PI) x_diff(3) -= 2.*M_PI;
    while (x_diff(3)<-M_PI) x_diff(3) += 2.*M_PI;

    P_ = P_ + weights_(i) * x_diff * x_diff.transpose();
  }
}

void UKF::UpdateLidar(MeasurementPackage meas_package) {
  /**
   * TODO: Complete this function! Use lidar data to update the belief 
   * about the object's position. Modify the state vector, x_, and 
   * covariance, P_.
   * You can also calculate the lidar NIS, if desired.
   */
  // Extract Measurements
  VectorXd z = meas_package.raw_measurements_;

  // Set dimension for Lidar measurement
  int n_z = 2;

  // Create Sigma Points in Measurement Space
  MatrixXd Zsig = MatrixXd(n_z, 2 * n_aug_ + 1);

  // Transform Predicted Sigma Points into measurement space
  for(int i = 0; i < 2 * n_aug_ +1; i++){

    // extract values for better readibility
    double p_x = Xsig_pred_(0, i);
    double p_y = Xsig_pred_(1, i);

    Zsig(0,i) = p_x; // px
    Zsig(1,i) = p_y; // py
  }
  UpdateUKF(meas_package,Zsig,n_z);
}

void UKF::UpdateRadar(MeasurementPackage meas_package) {
  /**
   * TODO: Complete this function! Use radar data to update the belief 
   * about the object's position. Modify the state vector, x_, and 
   * covariance, P_.
   * You can also calculate the radar NIS, if desired.
   */
  // Extract Measurements
  VectorXd z = meas_package.raw_measurements_;

  // Set dimension for Lidar measurement
  int n_z = 3;

  // Create Sigma Points in Measurement Space
  MatrixXd Zsig = MatrixXd(n_z, 2 * n_aug_ + 1);

  // Transform Predicted Sigma Points into measurement space
  for(int i = 0; i < 2 * n_aug_ +1; i++){

    double p_x = Xsig_pred_(0, i);
    double p_y = Xsig_pred_(1, i);
    double v   = Xsig_pred_(2, i);
    double yaw = Xsig_pred_(3, i);

    double v1 = cos(yaw)*v;
    double v2 = sin(yaw)*v;

    // measurement model
    Zsig(0, i) = sqrt(p_x*p_x + p_y*p_y);                      //r
    Zsig(1, i) = atan2(p_y, p_x);                              //phi
    Zsig(2, i) = (p_x*v1 + p_y*v2) / sqrt(p_x*p_x + p_y*p_y);  //r_dot
  }
  UpdateUKF(meas_package,Zsig,n_z);
}

void UKF::UpdateUKF(MeasurementPackage meas_package, MatrixXd Zsig, int n_z){

  // Extract Measurements
  VectorXd z = meas_package.raw_measurements_;

  // Predicted measurement mean
  VectorXd z_pred = VectorXd(n_z);
  z_pred.fill(0.0);
  for (int i = 0; i < 2 * n_aug_ + 1; i++) {
    z_pred = z_pred + weights_(i) * Zsig.col(i);
  }

  // Measurement covariance matrix S
  MatrixXd S = MatrixXd(n_z, n_z);
  S.fill(0.0);
  for (int i = 0; i < 2 * n_aug_ + 1; i++) {  //2n+1 simga points

    //residual
    VectorXd z_diff = Zsig.col(i) - z_pred;
    if(meas_package.sensor_type_ == MeasurementPackage::RADAR){
      NormAng(&(z_diff(1)));
    }

    S = S + weights_(i) * z_diff * z_diff.transpose();
  }

  // Add measurement noise covariance matrix
  MatrixXd R = MatrixXd(n_z, n_z);
  if(meas_package.sensor_type_ == MeasurementPackage::RADAR){
    R << R_radar_;
  }
  else if(meas_package.sensor_type_ == MeasurementPackage::LASER){
    R << R_lidar_;
  }
  S = S + R;

  // Create matrix for cross correlation Tc
  MatrixXd Tc = MatrixXd(n_x_, n_z);


  Tc.fill(0.0);
  for (int i = 0; i < 2 * n_aug_ + 1; i++) {  //2n+1 simga points

    //residual
    VectorXd z_diff = Zsig.col(i) - z_pred;
    if(meas_package.sensor_type_ == MeasurementPackage::RADAR){
      NormAng(&(z_diff(1)));
    }

    // State difference
    VectorXd x_diff = Xsig_pred_.col(i) - x_;
    if(meas_package.sensor_type_ == MeasurementPackage::RADAR){
      NormAng(&(x_diff(3)));
    }

    Tc = Tc + weights_(i) * x_diff * z_diff.transpose();
  }

  // Kalman gain K;
  MatrixXd K = Tc * S.inverse();

  //residual
  VectorXd z_diff = z - z_pred;
  if(meas_package.sensor_type_ == MeasurementPackage::RADAR){
      NormAng(&(z_diff(1)));
    }

  // Calculate NIS
  if(meas_package.sensor_type_ == MeasurementPackage::RADAR){
    NIS_radar_ = z_diff.transpose() * S.inverse() * z_diff;
  }
  else if(meas_package.sensor_type_ == MeasurementPackage::LASER){
    NIS_laser_ = z_diff.transpose() * S.inverse() * z_diff;
  }

  // Update state mean and covariance matrix
  x_ = x_ + K * z_diff;
  P_ = P_ - K*S*K.transpose();
}