#pragma once
#include <torch/torch.h>

class ConvNetImpl : public torch::nn::Module {
public:
	explicit ConvNetImpl(int64_t num_classes);
	torch::Tensor forward(torch::Tensor x);

private:

	torch::nn::Sequential layer1{
		torch::nn::Conv2d(torch::nn::Conv2dOptions(3, 32, 5)),
		torch::nn::BatchNorm2d(32),
		torch::nn::ReLU(),
		torch::nn::MaxPool2d(torch::nn::MaxPool2dOptions(2))
	};

	torch::nn::Sequential layer2{
		torch::nn::Conv2d(torch::nn::Conv2dOptions(32, 64, 5)),
		torch::nn::BatchNorm2d(64),
		torch::nn::ReLU(),
		torch::nn::MaxPool2d(torch::nn::MaxPool2dOptions(2))
	};

	torch::nn::Sequential layer3{
		torch::nn::Conv2d(torch::nn::Conv2dOptions(64, 128, 5)),
		torch::nn::BatchNorm2d(128),
		torch::nn::ReLU(),
		torch::nn::MaxPool2d(torch::nn::MaxPool2dOptions(2))
	};

	torch::nn::Sequential layer4{
		torch::nn::Conv2d(torch::nn::Conv2dOptions(128, 128, 5)),
		torch::nn::BatchNorm2d(128),
		torch::nn::ReLU(),
		torch::nn::MaxPool2d(torch::nn::MaxPool2dOptions(2))
	};

	torch::nn::Linear fc1 = torch::nn::Linear(128 * 10 * 10, 50);
	torch::nn::Linear fc2;
};

TORCH_MODULE(ConvNet);