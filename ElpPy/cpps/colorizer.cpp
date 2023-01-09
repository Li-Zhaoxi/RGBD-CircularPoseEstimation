#include "colorizer.h"

#include "ElpTools.h"

static color_map hue{ {
		{ 255, 0, 0 },
		{ 255, 255, 0 },
		{ 0, 255, 0 },
		{ 0, 255, 255 },
		{ 0, 0, 255 },
		{ 255, 0, 255 },
		{ 255, 0, 0 },
		} };

static color_map jet{ {
	{ 0, 0, 255 },
	{ 0, 255, 255 },
	{ 255, 255, 0 },
	{ 255, 0, 0 },
	{ 50, 0, 0 },
	} };

static color_map classic{ {
	{ 30, 77, 203 },
	{ 25, 60, 192 },
	{ 45, 117, 220 },
	{ 204, 108, 191 },
	{ 196, 57, 178 },
	{ 198, 33, 24 },
	} };

static color_map grayscale{ {
	{ 255, 255, 255 },
	{ 0, 0, 0 },
	} };

static color_map inv_grayscale{ {
	{ 0, 0, 0 },
	{ 255, 255, 255 },
	} };

static color_map biomes{ {
	{ 0, 0, 204 },
	{ 204, 230, 255 },
	{ 255, 255, 153 },
	{ 170, 255, 128 },
	{ 0, 153, 0 },
	{ 230, 242, 255 },
	} };

static color_map cold{ {
	{ 230, 247, 255 },
	{ 0, 92, 230 },
	{ 0, 179, 179 },
	{ 0, 51, 153 },
	{ 0, 5, 15 }
	} };

static color_map warm{ {
	{ 255, 255, 230 },
	{ 255, 204, 0 },
	{ 255, 136, 77 },
	{ 255, 51, 0 },
	{ 128, 0, 0 },
	{ 10, 0, 0 }
	} };

static color_map quantized{ {
	{ 255, 255, 255 },
	{ 0, 0, 0 },
	}, 6 };

static color_map pattern{ {
	{ 255, 255, 255 },
	{ 0, 0, 0 },
	{ 255, 255, 255 },
	{ 0, 0, 0 },
	{ 255, 255, 255 },
	{ 0, 0, 0 },
	{ 255, 255, 255 },
	{ 0, 0, 0 },
	{ 255, 255, 255 },
	{ 0, 0, 0 },
	{ 255, 255, 255 },
	{ 0, 0, 0 },
	{ 255, 255, 255 },
	{ 0, 0, 0 },
	{ 255, 255, 255 },
	{ 0, 0, 0 },
	{ 255, 255, 255 },
	{ 0, 0, 0 },
	{ 255, 255, 255 },
	{ 0, 0, 0 },
	{ 255, 255, 255 },
	{ 0, 0, 0 },
	{ 255, 255, 255 },
	{ 0, 0, 0 },
	{ 255, 255, 255 },
	{ 0, 0, 0 },
	{ 255, 255, 255 },
	{ 0, 0, 0 },
	{ 255, 255, 255 },
	{ 0, 0, 0 },
	{ 255, 255, 255 },
	{ 0, 0, 0 },
	{ 255, 255, 255 },
	{ 0, 0, 0 },
	{ 255, 255, 255 },
	{ 0, 0, 0 },
	{ 255, 255, 255 },
	{ 0, 0, 0 },
	{ 255, 255, 255 },
	{ 0, 0, 0 },
	{ 255, 255, 255 },
	{ 0, 0, 0 },
	{ 255, 255, 255 },
	{ 0, 0, 0 },
	{ 255, 255, 255 },
	{ 0, 0, 0 },
	{ 255, 255, 255 },
	{ 0, 0, 0 },
	{ 255, 255, 255 },
	{ 0, 0, 0 },
	} };


colorizer::colorizer()
{
	_histogram = std::vector<int>(MAX_DEPTH, 0);
	_hist_data = _histogram.data();

	_maps = { &jet, &classic, &grayscale, &inv_grayscale, &biomes, &cold, &warm, &quantized, &pattern, &hue };
	_map_index = 0;
}

void colorizer::process(cv::Mat& imgD, cv::Mat& outColor)
{
	auto make_equalized_histogram = [this](const cv::Mat& imgD, cv::Mat& outColor)
	{
		const auto w = imgD.cols, h = imgD.rows;
		outColor.create(h, w, CV_8UC3);
		auto rgb_data = (cv::Vec3b*)outColor.data;
		auto coloring_function = [&, this](float data) {
			auto hist_data = _hist_data[(int)data];
			auto pixels = (float)_hist_data[MAX_DEPTH - 1];
			return (hist_data / pixels);
		};

		auto depth_data = reinterpret_cast<const unsigned short*>(imgD.data);
		update_histogram(_hist_data, depth_data, w, h);
		make_rgb_data<uint16_t>(depth_data, rgb_data, w, h, coloring_function);
	};

	auto make_value_cropped_frame = [this](const cv::Mat& imgD, cv::Mat& outColor)
	{
		const auto w = imgD.cols, h = imgD.rows;
		outColor.create(h, w, CV_8UC3);
		auto rgb_data = (cv::Vec3b*)outColor.data;
		auto depth_data = reinterpret_cast<const unsigned short*>(imgD.data);
		auto min_val = 0.01f;
		auto max_val = 6.0f;
		auto _depth_units = 0.001;
		auto coloring_function = [&, this](float data) {
			return (data * _depth_units - min_val) / (max_val - min_val);
		};
		make_rgb_data<uint16_t>(depth_data, rgb_data, w, h, coloring_function);
	};

	make_equalized_histogram(imgD, outColor);
}

void depthColorizer(int rows, int cols, unsigned short* _data, unsigned char* _outdata)
{
	cv::Mat dep(rows, cols, CV_16UC1, _data);
	cv::Mat outrgb(rows, cols, CV_8UC3);
	colorizer clr;
	clr.process(dep, outrgb);
	memcpy(_outdata, outrgb.data, sizeof(unsigned char) * rows * cols * 3);
}