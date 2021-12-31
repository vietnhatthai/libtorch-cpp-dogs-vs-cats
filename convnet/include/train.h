#pragma once
#include <torch/torch.h>
#include "convnet.h"

ConvNet train(ConvNet model, torch::Device device,
	size_t num_epochs, int batch_size, double learning_rate);