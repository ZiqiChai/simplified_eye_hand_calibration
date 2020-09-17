/*hand-eye calibration using TSAI method*/
#include <stdlib.h>
#include <iostream>
#include <fstream>

#include <opencv2/core/core.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>


#include <eigen3/Eigen/Core>
#include <eigen3/Eigen/Geometry>
#include <eigen3/Eigen/LU>
#include <eigen3/Eigen/Dense>
#include <eigen3/Eigen/StdVector>

#include <opencv2/core/eigen.hpp>

#define PI 3.1415926

using namespace std;
using namespace cv;
using namespace Eigen;


int num_of_all_images =10;
// 行列方向内角点数量
cv::Size board_size = cv::Size(9, 6);
// 标定板棋盘格实际尺寸(单位要与pose.txt中机器人位置的单位一致) mm
cv::Size2f square_size = cv::Size2f(25, 25);


Eigen::Matrix3d skew(Eigen::Vector3d V);
Eigen::Matrix4d quat2rot(Eigen::Vector3d q);
Eigen::Vector3d rot2quat(Eigen::MatrixXd R);
Eigen::Matrix4d transl(Eigen::Vector3d x);
Eigen::Matrix4d handEye(std::vector<Eigen::Matrix4d, Eigen::aligned_allocator<Eigen::Matrix4d> > bHg,
	std::vector<Eigen::Matrix4d, Eigen::aligned_allocator<Eigen::Matrix4d> > cHw);
Eigen::Matrix4d handEye1(std::vector<Eigen::Matrix4d, Eigen::aligned_allocator<Eigen::Matrix4d> > bHg,
	std::vector<Eigen::Matrix4d, Eigen::aligned_allocator<Eigen::Matrix4d> > cHw);
int handEye_calib(Eigen::Matrix4d &gHc, std::string path );




// skew - returns skew matrix of a 3x1 vector.
//        cross(V,U) = skew(V)*U
//    S = skew(V)
//          0  -Vz  Vy
//    S =   Vz  0  -Vx
//         -Vy  Vx  0
// See also: cross
Eigen::Matrix3d skew(Eigen::Vector3d V)
{
	Eigen::Matrix3d S;
	S <<
	0, -V(2), V(1),
	V(2), 0, -V(0),
	-V(1), V(0), 0;
	return S;
}



// quat2rot - a unit quaternion(3x1) to converts a rotation matrix (3x3)
//
//    R = quat2rot(q)
//
//    q - 3x1 unit quaternion
//    R - 4x4 homogeneous rotation matrix (translation component is zero)
//        q = sin(theta/2) * v
//        theta - rotation angle
//        v    - unit rotation axis, |v| = 1
//
// See also: rot2quat, rotx, roty, rotz, transl, rotvec
Eigen::Matrix4d quat2rot(Eigen::Vector3d q)
{
	double p = q.transpose()*q;
	if (p > 1)
		std::cout << "Warning: quat2rot: quaternion greater than 1";
	double w = sqrt(1 - p);                // w = cos(theta/2)
	Eigen::Matrix4d R;
	R << Eigen::MatrixXd::Identity(4, 4);
	Eigen::Matrix3d res;
	res = 2 * (q*q.transpose()) + 2 * w*skew(q);
	res = res + Eigen::MatrixXd::Identity(3, 3) - 2 * p*Eigen::MatrixXd::Identity(3, 3);
	R.topLeftCorner(3, 3) << res;
	return R;
}



// rot2quat - converts a rotation matrix (3x3) to a unit quaternion(3x1)
//
//    q = rot2quat(R)
//
//    R - 3x3 rotation matrix, or 4x4 homogeneous matrix
//    q - 3x1 unit quaternion
//        q = sin(theta/2) * v
//        teta - rotation angle
//        v    - unit rotation axis, |v| = 1

// See also: quat2rot, rotx, roty, rotz, transl, rotvec
Eigen::Vector3d rot2quat(Eigen::MatrixXd R)
{
	// can this be imaginary?
	double w4 = 2 * sqrt(1 + R.topLeftCorner(3, 3).trace());
	Eigen::Vector3d q;
	q << (R(2, 1) - R(1, 2)) / w4,
	(R(0, 2) - R(2, 0)) / w4,
	(R(1, 0) - R(0, 1)) / w4;
	return q;
}

//  TRANSL	Translational transform
//
//	T= TRANSL(X, Y, Z)
//	T= TRANSL( [X Y Z] )
//
//	[X Y Z]' = TRANSL(T)
//
//	[X Y Z] = TRANSL(TG)
//
//	Returns a homogeneous transformation representing a
//	translation of X, Y and Z.
//
//	The third form returns the translational part of a
//	homogenous transform as a 3-element column vector.
//
//	The fourth form returns a  matrix of the X, Y and Z elements
//	extracted from a Cartesian trajectory matrix TG.
//
//	See also ROTX, ROTY, ROTZ, ROTVEC.
// 	Copyright (C) Peter Corke 1990
Eigen::Matrix4d transl(Eigen::Vector3d x)
{
	Eigen::Matrix4d r;
	r << Eigen::MatrixXd::Identity(4, 4);
	r.topRightCorner(3, 1) << x;
	return r;
}



//用每两个 K = (M*M - M) / 2
Eigen::Matrix4d handEye(std::vector<Eigen::Matrix4d, Eigen::aligned_allocator<Eigen::Matrix4d> > bHg,
	std::vector<Eigen::Matrix4d, Eigen::aligned_allocator<Eigen::Matrix4d> > cHw)
{
	int M = bHg.size();
	// Number of unique camera position pairs
	int K = (M*M - M) / 2;
	// will store: skew(Pgij+Pcij)
	Eigen::MatrixXd A;
	A = Eigen::MatrixXd::Zero(3 * K, 3);
	// will store: Pcij - Pgij
	Eigen::MatrixXd B;
	B = Eigen::MatrixXd::Zero(3 * K, 1);
	int k = 0;

	// Now convert from wHc notation to Hc notation used in Tsai paper.
	std::vector<Eigen::Matrix4d, Eigen::aligned_allocator<Eigen::Matrix4d> > Hg = bHg;
	std::vector<Eigen::Matrix4d, Eigen::aligned_allocator<Eigen::Matrix4d> > Hc = cHw;

	for (int i = 0; i<M; i++)
	{
		for (int j = i + 1; j<M; j++)
		{
			//Transformation from i-th to j-th gripper pose and the corresponding quaternion
			Eigen::Matrix4d Hgij = Hg.at(j).lu().solve(Hg.at(i));
			Eigen::Vector3d Pgij = 2 * rot2quat(Hgij);

			// Transformation from i-th to j-th camera pose and the corresponding quaternion
			Eigen::Matrix4d Hcij = Hc.at(j)*Hc.at(i).inverse();
			Eigen::Vector3d Pcij = 2 * rot2quat(Hcij);
			// Form linear system of equations
			k = k + 1;
			// left-hand side
			A.block(3 * k - 3, 0, 3, 3) << skew(Pgij + Pcij);
			// right-hand side
			B.block(3 * k - 3, 0, 3, 1) << Pcij - Pgij;
		}
	}
	// Rotation from camera to gripper is obtained from the set of equations:
	//    skew(Pgij+Pcij) * Pcg_ = Pcij - Pgij
	// Gripper with camera is first moved to M different poses, then the gripper
	// .. and camera poses are obtained for all poses. The above equation uses
	// .. invariances present between each pair of i-th and j-th pose.

	// Solve the equation A*Pcg_ = B
	Eigen::Vector3d Pcg_ = A.colPivHouseholderQr().solve(B);
	// Obtained non-unit quaternin is scaled back to unit value that
	// .. designates camera-gripper rotation
	Eigen::Vector3d Pcg = 2 * Pcg_ / sqrt(1 + (double)(Pcg_.transpose()*Pcg_));
	// Rotation matrix
	Eigen::Matrix4d Rcg = quat2rot(Pcg / 2);

	// Calculate translational component
	k = 0;
	for (int i = 0; i<M; i++)
	{
		for (int j = i + 1; j<M; j++)
		{
			// Transformation from i-th to j-th gripper pose
			Eigen::Matrix4d Hgij = Hg.at(j).lu().solve(Hg.at(i));
			// Transformation from i-th to j-th camera pose
			Eigen::Matrix4d Hcij = Hc.at(j)*Hc.at(i).inverse();

			k = k + 1;
			// Form linear system of equations
			// left-hand side
			A.block(3 * k - 3, 0, 3, 3) << Hgij.topLeftCorner(3, 3) - Eigen::MatrixXd::Identity(3, 3);
			// right-hand side
			B.block(3 * k - 3, 0, 3, 1) << Rcg.topLeftCorner(3, 3)*Hcij.block(0, 3, 3, 1) - Hgij.block(0, 3, 3, 1);
		}
	}

	Eigen::Vector3d Tcg = A.colPivHouseholderQr().solve(B);

	// incorporate translation with rotation
	Eigen::Matrix4d gHc = transl(Tcg) * Rcg;
	return gHc;
}

//只用相邻两个 K = M-1
Eigen::Matrix4d handEye1(std::vector<Eigen::Matrix4d, Eigen::aligned_allocator<Eigen::Matrix4d> > bHg,
	std::vector<Eigen::Matrix4d, Eigen::aligned_allocator<Eigen::Matrix4d> > cHw)
{
	int M = bHg.size();
	// Number of unique camera position pairs
	int K = M-1;
	// will store: skew(Pgij+Pcij)
	Eigen::MatrixXd A;
	A = Eigen::MatrixXd::Zero(3 * K, 3);
	// will store: Pcij - Pgij
	Eigen::MatrixXd B;
	B = Eigen::MatrixXd::Zero(3 * K, 1);
	int k = 0;

	// Now convert from wHc notation to Hc notation used in Tsai paper.
	std::vector<Eigen::Matrix4d, Eigen::aligned_allocator<Eigen::Matrix4d> > Hg = bHg;
	std::vector<Eigen::Matrix4d, Eigen::aligned_allocator<Eigen::Matrix4d> > Hc = cHw;
	for (int i = 0; i<M-1; i++)
	{
		//Transformation from i-th to j-th gripper pose and the corresponding quaternion
		Eigen::Matrix4d Hgij = Hg.at(i+1).lu().solve(Hg.at(i));
		Eigen::Vector3d Pgij = 2 * rot2quat(Hgij);
		// Transformation from i-th to j-th camera pose and the corresponding quaternion
		Eigen::Matrix4d Hcij = Hc.at(i+1)*Hc.at(i).inverse();
		Eigen::Vector3d Pcij = 2 * rot2quat(Hcij);
		//Form linear system of equations
		k = k + 1;
		//left-hand side
		A.block(3 * k - 3, 0, 3, 3) << skew(Pgij + Pcij);
		//right-hand side
		B.block(3 * k - 3, 0, 3, 1) << Pcij - Pgij;
	}
	// Rotation from camera to gripper is obtained from the set of equations:
	//    skew(Pgij+Pcij) * Pcg_ = Pcij - Pgij
	// Gripper with camera is first moved to M different poses, then the gripper
	// .. and camera poses are obtained for all poses. The above equation uses
	// .. invariances present between each pair of i-th and j-th pose.

	// Solve the equation A*Pcg_ = B
	Eigen::Vector3d Pcg_ = A.colPivHouseholderQr().solve(B);

	// Obtained non-unit quaternin is scaled back to unit value that
	// .. designates camera-gripper rotation
	Eigen::Vector3d Pcg = 2 * Pcg_ / sqrt(1 + (double)(Pcg_.transpose()*Pcg_));

	// Rotation matrix
	Eigen::Matrix4d Rcg = quat2rot(Pcg / 2);
	// Calculate translational component
	k = 0;
	for (int i = 0; i<M-1; i++)
	{
		// Transformation from i-th to j-th gripper pose
		Eigen::Matrix4d Hgij = Hg.at(i+1).lu().solve(Hg.at(i));
		// Transformation from i-th to j-th camera pose
		Eigen::Matrix4d Hcij = Hc.at(i+1)*Hc.at(i).inverse();
		// Form linear system of equations
		k = k + 1;
		// left-hand side
		A.block(3 * k - 3, 0, 3, 3) << Hgij.topLeftCorner(3, 3) - Eigen::MatrixXd::Identity(3, 3);
		B.block(3 * k - 3, 0, 3, 1) << Rcg.topLeftCorner(3, 3)*Hcij.block(0, 3, 3, 1) - Hgij.block(0, 3, 3, 1);
		B.block(3 * k - 3, 0, 3, 1) << Rcg.topLeftCorner(3, 3)*Hcij.block(0, 3, 3, 1) - Hgij.block(0, 3, 3, 1);
	}
	Eigen::Vector3d Tcg = A.colPivHouseholderQr().solve(B);
	// incorporate translation with rotation
	Eigen::Matrix4d gHc = transl(Tcg) * Rcg;
	return gHc;
}


int handEye_calib(Eigen::Matrix4d &gHc, std::string path)
{
	ofstream ofs(path + "/output.txt");
	std::vector<cv::Mat> images;
	// 读入图片
	std::cout 	<< "******************读入图片......******************" << std::endl;
	ofs 		<< "******************读入图片......******************" << std::endl;

	for (int i = 0; i < num_of_all_images; i++)
	{
		std::string image_path;
		image_path = path + "/" + std::to_string(i) + ".png";
		cv::Mat image = cv::imread(image_path, 0);
		std::cout << "image_path: " << image_path << std::endl;
		if (!image.empty())
			images.push_back(image);
		else
		{
			std::cout << "can not find " << image_path << std::endl;
			exit(-1);
		}
	}
	// 更新实际图片数量
	int num_of_images = images.size();
	std::cout 	<< "******************读入图片完成******************" << std::endl;
	ofs 		<< "******************读入图片完成！******************" << std::endl;
	std::cout 	<< "实际图片数量： " << num_of_images << std::endl;
	ofs  		<< "实际图片数量： " << num_of_images << std::endl;

	// 提取角点
	std::cout 	<< "******************开始提取角点......******************" << std::endl;
	ofs 		<< "******************开始提取角点......******************" << std::endl;
	// 图像尺寸
	cv::Size image_size;
	std::vector<cv::Point2f> image_points_buff;
	std::vector<std::vector<cv::Point2f>> image_points_seq;
	int num_img_successed_processing  =0;
	for (int i = 0; i<images.size(); i++)
	{
		cv::Mat image = images[i];
		if (i == 0)
		{
			// 第一幅图像
			image_size.width = image.cols;
			image_size.height = image.rows;
		}

		if (0 == cv::findChessboardCorners(image, board_size, image_points_buff,
						cv::CALIB_CB_ADAPTIVE_THRESH |cv::CALIB_CB_NORMALIZE_IMAGE /*+ cv::CALIB_CB_FAST_CHECK*/))
		{
			std::cout << "can not find chessboard corners! " << std::endl;
			ofs << "can not find chessboard corners! " << std::endl;
			continue;
		}
		else
		{
			cv::cornerSubPix(image, image_points_buff, cv::Size(3,3), cv::Size(-1, -1),
				cv::TermCriteria(CV_TERMCRIT_EPS + CV_TERMCRIT_ITER, 30, 0.1));
			// 对粗提取的角点精细化
			// cv::find4QuadCornerSubpix(image, image_points_buff, cv::Size(5, 5));
			// 保存亚像素角点
			image_points_seq.push_back(image_points_buff);
			std::cout<<"successed processing :" <<num_img_successed_processing++<<std::endl;
			cv::Mat image_color;
			cv::cvtColor(image, image_color, CV_GRAY2BGR);
			cv::drawChessboardCorners(image_color, board_size, image_points_buff, true);
			std::stringstream namestream;
			std::string name;
			namestream << "/" << i << "_corner.png";
			namestream >> name;
			cv::imwrite(path + name, image_color);
		}

	}

	// 根据检测到角点的图像的数量再次更新图像数目
	num_of_images = image_points_seq.size();
	std::cout 	<< "******************提取角点完成!******************" << std::endl;
	ofs 		<< "******************提取角点完成!******************" << std::endl;

	// 摄像机标定
	std::cout 	<< "******************开始进行相机标定......******************" << std::endl;
	ofs 		<< "******************开始进行相机标定......******************" << std::endl;
	// 保存标定板上的角点的三维坐标
	std::vector<std::vector<cv::Point3f>> object_points;
	// 内外参
	// 内参
	cv::Mat camera_matrix = cv::Mat(3, 3, CV_32FC1, cv::Scalar::all(0));
	std::vector<int> point_counts;
	// 畸变系数: k1, k2, p1, p2, k3
	cv::Mat dist_coeff = cv::Mat(1, 5, CV_32FC1, cv::Scalar::all(0));
	//平移向量
	std::vector<cv::Mat> t_vec;
	//旋转向量
	std::vector<cv::Mat> r_vec;
	// 初始化标定板上角点的三维坐标
	int i, j, k;
	for (k = 0; k<num_of_images; k++)
	{
		//std::cout << "image.NO:" << k << std::endl;
		std::vector<cv::Point3f> temp_point_set;
		for (i = 0; i<board_size.height; i++)
		{
			for (j = 0; j<board_size.width; j++)
			{
				cv::Point3f real_point;
				real_point.x = j*square_size.width;
				real_point.y = i*square_size.height;
				real_point.z = 0;
				// std::cout << "real_point cordinates" << real_point << std::endl;
				temp_point_set.push_back(real_point);
			}
		}
		object_points.push_back(temp_point_set);
	}
	//初始化每幅图像上的角点数量
	for (int i = 0; i<num_of_images; i++)
	{
		point_counts.push_back(board_size.width*board_size.height);
	}
	// 开始标定
	std::cout << "******************开始运行calibrateCamera!******************" << std::endl;
	cv::calibrateCamera(object_points, image_points_seq, image_size, camera_matrix, dist_coeff, r_vec, t_vec, 0);
	std::cout << "******************标定完成!******************" << std::endl;
	std::cout << camera_matrix << std::endl;
	std::cout << dist_coeff << std::endl;

	// 评价标定结果
	std::cout << "******************开始评定标定结果......******************" << std::endl;
	double total_err = 0.0;
	double err = 0.0;
	//保存重新计算得到的投影点
	std::vector<cv::Point2f> image_points2;
	std::cout << std::endl << "每幅图像的标定误差: " << std::endl;
	for (int i = 0; i<num_of_images; i++)
	{
		std::vector<cv::Point3f> temp_point_set = object_points[i];
		// 重映射
		cv::projectPoints(temp_point_set, r_vec[i], t_vec[i], camera_matrix, dist_coeff, image_points2);

		std::vector<cv::Point2f> temp_image_points = image_points_seq[i];
		cv::Mat temp_image_points_Mat = cv::Mat(1, temp_image_points.size(), CV_32FC2);
		cv::Mat temp_image_points2_Mat = cv::Mat(1, image_points2.size(), CV_32FC2);
		for (int j = 0; j<temp_image_points.size(); j++)
		{
			temp_image_points_Mat.at<cv::Vec2f>(0, j) = cv::Vec2f(temp_image_points[j].x, temp_image_points[j].y);
			temp_image_points2_Mat.at<cv::Vec2f>(0, j) = cv::Vec2f(image_points2[j].x, image_points2[j].y);
		}
		err = cv::norm(temp_image_points_Mat, temp_image_points2_Mat, cv::NORM_L2);
		total_err += err /= point_counts[i];
		std::cout << "第" << i + 1 << "幅图像的平均误差: " << err << "像素" << std::endl;
	}

	std::cout << std::endl << "总体平均误差: " << total_err / num_of_images << "像素" << std::endl;
	std::cout << "******************评价完成!******************" << std::endl;

	// 输出标定结果
	std::cout << "******************标定结果如下:******************" << std::endl;
	std::cout << "相机内参:" << std::endl;
	std::cout << camera_matrix << std::endl;
	std::cout << "畸变系数:" << std::endl;
	std::cout << dist_coeff.t() << std::endl;
	// 输出到文件
	ofs << "******************标定结果如下:******************" << std::endl;
	ofs << "相机内参:" << std::endl;
	ofs << camera_matrix << std::endl;
	ofs << "畸变系数:" << std::endl;
	ofs << dist_coeff.t() << std::endl;


	std::cout << "******************开始运行calibrate handEye!******************" << std::endl;
	std::vector<Eigen::Matrix4d, Eigen::aligned_allocator<Eigen::Matrix4d> > bHg;
	std::vector<Eigen::Matrix4d, Eigen::aligned_allocator<Eigen::Matrix4d> > cHw;

	std::ifstream ifs;
	ifs.open(path + "/pose.txt", std::ios::in);

	if (!ifs.is_open())
	{
		std::cout << "pose file not found" << std::endl;
		return -1;
	}

	else
	{
		std::cout << "pose file found" << std::endl;
	}

	//读取机器人位姿及转化为eigen matrix
	for (int i = 0; i<num_of_images; i++)
	{
		double temp;
		Eigen::Vector3d trans_temp;

		std::cout << "pose[" << i << "]:\n";
		for (int j = 0; j<3; j++)
		{
			ifs >> temp;
			std::cout << temp<<" ";
			trans_temp(j) = temp;
		}

		std::vector<double> v(3, 0.0);
		ifs >>v[0] >> v[1] >> v[2];
		std::cout << "RPY: " << v[0] <<  " "<< v[1] <<  " "<< v[2] <<  " " << std::endl;
		Eigen::Quaterniond m = Eigen::AngleAxisd(v[2] / 180 * M_PI, Eigen::Vector3d::UnitZ())\
		* Eigen::AngleAxisd(v[1] / 180 * M_PI, Eigen::Vector3d::UnitY())\
		* Eigen::AngleAxisd(v[0] / 180 * M_PI, Eigen::Vector3d::UnitX());


		double w, x, y, z;
		x = m.x(); y= m.y(); z= m.z();  w= m.w();
		Eigen::Quaterniond rot_temp(w, x, y, z);

		Eigen::Matrix4d pose_temp;
		pose_temp << Eigen::MatrixXd::Identity(4, 4);
		//四元数转旋转矩阵
		pose_temp.topLeftCorner(3, 3) << rot_temp.toRotationMatrix();
		pose_temp.topRightCorner(3, 1) << trans_temp;

		std::cout << "bHg:" << std::endl;
		std::cout << pose_temp << "\n" << std::endl;

		pose_temp = pose_temp.inverse().eval();

		std::cout << "gHb:" << std::endl;
		std::cout << pose_temp << "\n" << std::endl;
		bHg.push_back(pose_temp);
	}
	ifs.close();


	//r_vec和t_vec转化为Eigen matrix
	for (int i = 0; i<num_of_images; i++)
	{
		Eigen::Matrix4d pose_temp;
		cv::Mat rot_temp;
		//罗德里格斯变换，旋转向量转旋转矩阵
		cv::Rodrigues(r_vec.at(i), rot_temp);
		Eigen::MatrixXd rot_eigen;
		Eigen::MatrixXd trans_eigen;
		cv::cv2eigen(rot_temp, rot_eigen);
		cv::cv2eigen(t_vec.at(i), trans_eigen);
		pose_temp.topLeftCorner(3, 3) << rot_eigen;
		pose_temp.topRightCorner(3, 1) << trans_eigen;
		pose_temp(3, 0) = 0;
		pose_temp(3, 1) = 0;
		pose_temp(3, 2) = 0;
		pose_temp(3, 3) = 1;
		cHw.push_back(pose_temp);
	}


	Eigen::AngleAxisd v;
	for (int m = 0; m < bHg.size(); m++)
	{
		ofs << "num:" << m << std::endl;
		ofs << "bHg" << std::endl;
		ofs << bHg.at(m) << std::endl;
		v.fromRotationMatrix((Matrix3d)bHg.at(m).topLeftCorner(3, 3));
		ofs << "axis: " << v.axis().transpose() << std::endl << "angle: " << v.angle()/PI*180 << std::endl;
		ofs << "cHw" << std::endl;
		ofs << cHw.at(m) << std::endl;
		v.fromRotationMatrix((Matrix3d)cHw.at(m).topLeftCorner(3, 3));
		ofs << "axis: " << v.axis().transpose() << std::endl << "angle: " << v.angle() / PI * 180 << std::endl;
	}
	gHc = handEye(bHg, cHw);
	std::cout << "\n\n\nCalibration Finished: \n" <<std::endl;
	std::cout << "gHc" << std::endl;
	ofs <<std::endl<< "gHc" << std::endl;
	std::cout << gHc << std::endl;
	ofs << gHc << std::endl;
	v.fromRotationMatrix((Matrix3d)gHc.topLeftCorner(3, 3));
	std::cout << "axis: " << v.axis().transpose() << std::endl << "angle: " << v.angle() / PI * 180 << std::endl;
	ofs << "axis: " << v.axis().transpose() << std::endl << "angle: " << v.angle() / PI * 180 << std::endl;
	return 0;
}



int main()
{
	Eigen::Matrix4d gHc;
	handEye_calib(gHc, "./data");
	return 1;
}