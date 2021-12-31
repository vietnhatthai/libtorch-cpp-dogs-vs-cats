#pragma once
#include <torch/torch.h>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/opencv.hpp>

#include <iostream>
#include <iomanip>

class Dataset : public torch::data::Dataset<Dataset>
{
public:
	explicit Dataset(std::string data_path, std::vector<std::string> name_classes, int input_size, int num_images = 25000);

	// Override the get method to load custom data.
	torch::data::Example<> get(size_t index) override;

	// Override the size method to infer the size of the data set.
	torch::optional<size_t> size() const override { return datasets.size(); };

private:
	std::vector<std::vector<int>> datasets;
	std::string data_path;
	std::vector<std::string> name_classes;
	int input_size;
};