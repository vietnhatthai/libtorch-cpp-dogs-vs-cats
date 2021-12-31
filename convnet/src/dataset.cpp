#include "dataset.h"

std::string get_img_path(std::string data_path, std::vector<std::string> name_classes, int i_class, int i_img)
{
	// https://www.kaggle.com/c/dogs-vs-cats/data
	return data_path + "\\" + name_classes[i_class] + "." + std::to_string(i_img) + ".jpg";
}

Dataset::Dataset(std::string data_path, std::vector<std::string> name_classes, int input_size, int num_images)
	: data_path(data_path), name_classes(name_classes), input_size(input_size)
{
	int num_classes = name_classes.size();
	for (int i = 0; i < num_classes; i++)
		for (int j = 0; j < num_images / 2; j++)
			datasets.push_back({ i, j });
	std::cout << "[INFO] Dataset size : " << datasets.size() << std::endl;
	std::cout << "[INFO] Classes      : " << name_classes << std::endl;
	//std::cout << "[TEST] " << name_classes[datasets[1][0]] << " ID :" << datasets[1][1] << std::endl;
	//std::cout << "[TEST] " << name_classes[datasets[2][0]] << " ID :" << datasets[2][1] << std::endl;
	//std::cout << "[TEST] " << name_classes[datasets[3][0]] << " ID :" << datasets[3][1] << std::endl;
}

torch::data::Example<> Dataset::get(size_t index)
{
	std::string file_location = get_img_path(data_path, name_classes, datasets[index][0], datasets[index][1]);
	int label = datasets[index][0];

	// Load image with OpenCV.
	cv::Mat img = cv::imread(file_location);
	cv::resize(img, img, cv::Size(input_size, input_size));

	torch::Tensor img_tensor = torch::from_blob(img.data, { img.rows, img.cols, 3 }, torch::kByte).clone();
	img_tensor = img_tensor.permute({ 2, 0, 1 });			// convert to CxHxW
	torch::Tensor label_tensor = torch::tensor({ label });

	return { img_tensor, label_tensor };
};