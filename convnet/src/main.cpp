#include <torch/torch.h>

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/opencv.hpp>

#include <map>
#include <algorithm>
#include <random>
#include <sstream>
#include <fstream>
#include <vector>
#include <tuple>
#include <iostream>
#include <iomanip>

#include "convnet.h"

int main()
{
	// Device
	bool cuda_available = torch::cuda::is_available();
	torch::Device device(cuda_available ? torch::kCUDA : torch::kCPU);
	std::cout << (cuda_available ? "CUDA available. Training on GPU." : "Training on CPU.") << '\n';

	std::string data_path = "D:\\Projects\\pytorch-test\\data\\dogcat\\train";  // data root
	std::vector<std::string> name_classes = { "dog", "cat" };

	// Hyperparameters
	const int NUM_IMAGES = 25000;					// 12500 -cat, 12500 -dog
	const int NUM_CLASSES = name_classes.size();	// 2
	const int INPUT_SIZE = 224;						// 224 -w, 224 -h
	const int BATCH_SIZE = 64;
	const double LEARING_RATE = 0.001;
	const size_t NUM_EPOCHS = 5;
	const float TESTSET_SIZE = 0.2;					// 20 -val, 80 -train

	return 0;
}