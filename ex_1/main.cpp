#include<stdio.h>
#include <iostream>
#include <cmath>
#include <algorithm>
#include <numeric>
#include <fstream>
#include <string>
#include  <opencv2\opencv.hpp>
#include "net.h"
#include "mat.h"
#include "benchmark.h"
#include "mobilenetv2.id.h"
/*
	@brief 读取标签文件
	@param [input] strFileName 文件名
	@param [input] vecLabels 标签
*/
// 自定义指数函数
double myfunction(double num) {
	return exp(num);
}
// 按行获取label
void read_labels(std::string strFileName,std::vector<std::string> &vecLabels)
{
	std::ifstream in(strFileName);

	if (in)
	{	
		std::string line;
		while (std::getline(in, line)) 
		{
			// std::cout << line << std::endl;
			vecLabels.push_back(line);
		}
	}
	else 
	{
		std::cout << "label file is not exit!!!" << std::endl;
	}
}
/*
	@brief squeezenet_v_1			预测单张图的类别
	@param [input] strImagePath		图片路径
*/
void forward_mobilenetv2(std::string strImagePath)
{
	// data
	std::string strLabelPath = "../model/synset_words.txt";
	std::vector<std::string> vecLabel;
	read_labels(strLabelPath, vecLabel);

	const float mean_vals[3] = { 0.f, 0.f, 0.f };
	const float norm_vals[3] = { 0.0039, 0.0039, 0.0039 };
	// getImage
	cv::Mat matImage = cv::imread(strImagePath);
	cv::resize(matImage, matImage, cv::Size(32, 32));

	if (matImage.empty()) 
	{
		printf("image is empty!!!\n");
	}
	
	const int nImageWidth = matImage.cols;
	const int nImageHeight = matImage.rows;

	// input and output
	ncnn::Mat matIn;
	ncnn::Mat matOut;
	// net
	ncnn::Net net;
	net.load_param_bin("../model/mobilenetv2.param.bin");
	net.load_model("../model/mobilenetv2.bin");
	
	const int nNetInputWidth = 32;
	const int nNetInputHeight = 32;

	// time
	double dStart = ncnn::get_current_time();

	// 判断图片大小是否和网络输入相同
	if (nNetInputWidth != nImageWidth || nNetInputHeight != nImageHeight)
	{
		matIn = ncnn::Mat::from_pixels_resize(matImage.data, ncnn::Mat::PIXEL_BGR, nImageWidth, nImageHeight, nNetInputWidth, nNetInputHeight);
	}
	else
	{
		matIn = ncnn::Mat::from_pixels(matImage.data, ncnn::Mat::PIXEL_BGR, nNetInputWidth, nNetInputHeight);
	}
	// 数据预处理
	matIn.substract_mean_normalize(mean_vals, norm_vals);

	// forward
	ncnn::Extractor ex = net.create_extractor();
	ex.set_light_mode(true);
	ex.input(mobilenetv2_param_id::BLOB_input_1, matIn);
	ex.extract(mobilenetv2_param_id::BLOB_457, matOut);

	printf("output_size: %d, %d, %d \n", matOut.w, matOut.h, matOut.c);
	
	// cls 1000 class
	std::vector<float> cls_scores;
	cls_scores.resize(matOut.w);
	for (int i = 0; i <matOut.w; i ++)
	{
		cls_scores[i] = matOut[i];
	}
	// return top class
	int top_class = 0;
	float max_score = 0.f;
	double sum = 0.0;
	transform(cls_scores.begin(), cls_scores.end(), cls_scores.begin(), myfunction);
	sum = accumulate(cls_scores.begin(), cls_scores.end(), sum);
	for (size_t i = 0; i<cls_scores.size(); i++)
	{
		float s = cls_scores[i];
		if (s > max_score)
		{
			top_class = i;
			max_score = s/sum;
		}
	}
	double dEnd = ncnn::get_current_time();

	printf("%d  score: %f   spend time: %.2f ms\n", top_class, max_score, (dEnd - dStart));
	std::cout << vecLabel[top_class] << std::endl;
	cv::putText(matImage, vecLabel[top_class], cv::Point(5, 10), 1, 0.8, cv::Scalar(0, 0, 255), 1);
	cv::putText(matImage, " score:" + std::to_string(max_score), cv::Point(5, 20), 1, 0.8, cv::Scalar(0, 0, 255), 1);
	cv::putText(matImage, " time: " + std::to_string(dEnd - dStart) + "ms", cv::Point(5, 30), 1, 0.8, cv::Scalar(0, 0, 255), 1);
	cv::imwrite("..\\images\\result.jpg", matImage);
	//cv::imshow("result", matImage);
	//cv::waitKey(-1);
	net.clear();
	
	
}


int main()
{
	forward_mobilenetv2("..\\images\\38.jpg");
	printf("hello ncnn");
	system("pause");
}