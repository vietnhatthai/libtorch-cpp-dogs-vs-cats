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

	return 0;
}