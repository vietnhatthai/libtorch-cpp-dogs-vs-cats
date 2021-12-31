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
#include "dataset.h"

int main()
{
	// Device
	bool cuda_available = torch::cuda::is_available();
	torch::Device device(cuda_available ? torch::kCUDA : torch::kCPU);
	std::cout << (cuda_available ? "CUDA available. Training on GPU." : "Training on CPU.") << '\n';

	std::string data_path = "D:\\Projects\\pytorch-test\\data\\dogcat\\train";  // data root
	std::string test_path = "D:\\Projects\\pytorch-test\\data\\dogcat\\test1";  // test root
	std::vector<std::string> name_classes = { "dog", "cat" };

	// Hyperparameters
	const int NUM_IMAGES = 25000;					// 12500 -cat, 12500 -dog
	const int NUM_IMAGES_TEST = 12500;				// 12500 images test
	const int NUM_CLASSES = name_classes.size();	// 2
	const int INPUT_SIZE = 224;						// 224 -w, 224 -h
	const int BATCH_SIZE = 64;
	const double LEARING_RATE = 0.001;
	const size_t NUM_EPOCHS = 5;
	const float TESTSET_SIZE = 0.2;					// 20 -val, 80 -train
	const bool IS_TRAIN = true;						// 
	
	// Neural Network model
	ConvNet model(NUM_CLASSES);
	torch::load(model, "model.pt");
	model->to(device);
	std::cout << std::endl << model << std::endl << std::endl;

	if (IS_TRAIN)
	{
		model->train();

		auto datasets = Dataset(data_path, name_classes, INPUT_SIZE, NUM_IMAGES)
			.map(torch::data::transforms::Normalize<>({ 0.5, 0.5, 0.5 }, { 0.5, 0.5, 0.5 }))
			.map(torch::data::transforms::Stack<>());
		// Number of samples in the training set
		int num_train_samples = datasets.size().value();
		int num_inters = num_train_samples / BATCH_SIZE;
		std::cout << "[INFO] num train samples : " << num_train_samples << std::endl;

		// Data loader
		auto data_loader = torch::data::make_data_loader<torch::data::samplers::RandomSampler>(std::move(datasets), BATCH_SIZE);

		// Optimizer
		torch::optim::Adam optimizer(model->parameters(), torch::optim::AdamOptions(LEARING_RATE));
		// Set floating point output precision
		std::cout << std::fixed << std::setprecision(3);
		std::cout << "[INFO] Training...\n";

		// Train the model
		for (size_t epoch = 0; epoch != NUM_EPOCHS; ++epoch) {
			// Initialize running metrics
			double running_loss = 0.0;
			size_t num_correct = 0;
			int iter = 0;
			for (auto& batch : *data_loader) {
				auto data = batch.data.to(device);
				auto target = batch.target.squeeze().to(device);
				// Forward pass
				auto output = model->forward(data);
				auto loss = torch::nn::functional::cross_entropy(output, target);
				//auto loss = torch::nll_loss(output.squeeze(), target);
				//auto loss = torch::nn::functional::binary_cross_entropy(output, target);
				// Update running loss
				running_loss += loss.item<double>() * data.size(0);

				// Calculate prediction
				auto prediction = output.argmax(1);
				// Update number of correctly classified samples
				size_t n_correct = prediction.eq(target).sum().item<int64_t>();
				num_correct += n_correct;

				auto n_accuracy = static_cast<double>(n_correct) / data.size(0);

				if (!iter % 100) {
					std::cout << "Epoch [" << (epoch + 1) << "/" << NUM_EPOCHS << "][" << inter << "/" << num_inters
						<< "], Loss = " << loss.item<double>() << ", Accuracy: " << n_accuracy << std::endl;
				}
				iter += 1;

				// Backward and optimize
				optimizer.zero_grad();
				loss.backward();
				optimizer.step();
			}

			auto sample_mean_loss = running_loss / num_train_samples;
			auto accuracy = static_cast<double>(num_correct) / num_train_samples;

			std::cout << "\nEpoch [" << (epoch + 1) << "/" << NUM_EPOCHS << "], Trainset - Loss: "
				<< sample_mean_loss << ", Accuracy: " << accuracy << "\n\n";
		}

		// Save model
		std::cout << "[INFO] Saving model..." << std::endl;
		torch::save(model, "model.pt");
	}

	// Test model
	std::cout << "[INFO] Testing model..." << std::endl;
	model->eval();
	torch::NoGradGuard no_grad;

	// Prediction data test
	cv::Mat image;
	for (int i = 1; i <= NUM_IMAGES_TEST; i++)
	{
		image = cv::imread(test_path + "\\" + std::to_string(i) + ".jpg");
		cv::resize(image, image, cv::Size(INPUT_SIZE, INPUT_SIZE));
		torch::Tensor img_tensor = torch::from_blob(image.data, { image.rows, image.cols, 3 }, torch::kByte).clone();
		img_tensor = img_tensor.permute({ 2, 0, 1 });			// convert to CxHxW
		img_tensor = torch::data::transforms::Normalize<>({ 0.5, 0.5, 0.5 }, { 0.5, 0.5, 0.5 })(img_tensor);
		img_tensor = img_tensor.unsqueeze(0).to(device);
		auto output = model->forward(img_tensor);
		std::string prediction = name_classes[output.argmax(1).item<int>()];
		std::cout << "[TEST] Prediction : " << prediction << std::endl;

		cv::putText(image, prediction, { 0, 30 }, cv::FONT_HERSHEY_SIMPLEX, 1, { 0, 0, 255 }, 2);
		cv::imshow("image", image);
		if (cv::waitKey(1) == 'q')
			break;
	}

	return 0;
}