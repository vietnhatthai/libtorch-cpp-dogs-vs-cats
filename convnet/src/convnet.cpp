#include "convnet.h"

ConvNetImpl::ConvNetImpl(int64_t num_classes) :
	fc2(50, num_classes)
{
	register_module("layer1", layer1);
	register_module("layer2", layer2);
	register_module("layer3", layer3);
	register_module("layer4", layer4);
	register_module("fc1", fc1);
	register_module("fc2", fc2);
}

torch::Tensor ConvNetImpl::forward(torch::Tensor x) {
	//std::cout << "Layer1" << std::endl;
	x = layer1->forward(x);
	//std::cout << "Layer2" << std::endl;
	x = layer2->forward(x);
	//std::cout << "Layer3" << std::endl;
	x = layer3->forward(x);
	//std::cout << "Layer4" << std::endl;
	x = layer4->forward(x);
	//std::cout << "Layer5" << std::endl;
	x = x.view({ -1, 128 * 10 * 10 });
	x = fc1->forward(x);
	x = torch::relu(x);
	//std::cout << "Layer6" << std::endl;
	x = fc2->forward(x);
	//x = torch::softmax(x, 1);

	return x;
}