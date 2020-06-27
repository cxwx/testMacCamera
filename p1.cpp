#include <iostream>

#include <opencv4/opencv2/opencv.hpp>
#include <opencv4/opencv2/videoio.hpp>
#include <opencv4/opencv2/opencv.hpp>
#include <dlib/opencv.h>
#include <dlib/image_processing/frontal_face_detector.h>
#include <dlib/image_processing/render_face_detections.h>
#include <dlib/image_processing.h>
#include <dlib/image_io.h>

void p1()
{
	cv::VideoCapture cap(0); // default camera
	if(!cap.isOpened()){
		std::cerr<<"cannot open webcam"<<std::endl; 
		exit(1);
	}
//	cv::namedWindow("edges",CV_WINDOW_NORMAL);
	dlib::image_window aWindow;
	dlib::frontal_face_detector detector = dlib::get_frontal_face_detector();
	dlib::shape_predictor pose_model;
	dlib::deserialize("usedfile/shape_predictor_68_face_landmarks.dat") >> pose_model;
	while(1){
		cv::Mat temp;
		cap >> temp;
//		if (!cap.read(temp))break;
		cv::Mat temp2;
		cv::resize(temp, temp2, cv::Size(), 0.25,0.25);
		dlib::cv_image<dlib::bgr_pixel> cimg(temp2);
		std::vector<dlib::rectangle> faces = detector(cimg);
		std::vector<dlib::full_object_detection> shapes;
		for (auto i: faces)
			shapes.push_back(pose_model(cimg, i));

		// Display it all on the screen
		aWindow.set_image(cimg);
		aWindow.clear_overlay();
		aWindow.add_overlay(render_face_detections(shapes));
	}


}
int main(int, char**)
{
	p1();
}
