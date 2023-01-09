#pragma once

#include <map>
#include <vector>
#include <opencv2/opencv.hpp>


// 在libRealsense的基础上进行修改

class color_map
{
public:
	color_map(std::map<float, cv::Point3f> map, int steps = 4000) : _map(map)
	{
		initialize(steps);
	}

	color_map(const std::vector<cv::Point3f>& values, int steps = 4000)
	{
		for (size_t i = 0; i < values.size(); i++)
		{
			_map[(float)i / (values.size() - 1)] = values[i];
		}
		initialize(steps);
	}

	color_map() {}

	inline cv::Point3f get(float value) const
	{
		if (_max == _min) return *_data;
		auto t = (value - _min) / (_max - _min);
		//std::cout << value << " , " << t <<" , " << _min << " , " << _max << " , " << _size << " , " << (t * (_size - 1)) << std::endl;
		//system("pause");
		t = std::min(std::max(t, 0.f), 1.f);
		return _data[(int)(t * (_size - 1))];
	}

	float min_key() const { return _min; }
	float max_key() const { return _max; }

	const std::vector<cv::Point3f>& get_cache() const { return _cache; }

private:
	inline cv::Point3f lerp(const cv::Point3f& a, const cv::Point3f& b, float t) const
	{
		return b * t + a * (1 - t);
	}

	cv::Point3f calc(float value) const
	{
		if (_map.size() == 0) return{ value, value, value };
		// if we have exactly this value in the map, just return it
		if (_map.find(value) != _map.end()) return _map.at(value);
		// if we are beyond the limits, return the first/last element
		if (value < _map.begin()->first)   return _map.begin()->second;
		if (value > _map.rbegin()->first)  return _map.rbegin()->second;

		auto lower = _map.lower_bound(value) == _map.begin() ? _map.begin() : --(_map.lower_bound(value));
		auto upper = _map.upper_bound(value);

		auto t = (value - lower->first) / (upper->first - lower->first);
		auto c1 = lower->second;
		auto c2 = upper->second;
		return lerp(c1, c2, t);
	}

	void initialize(int steps)
	{
		if (_map.size() == 0) return;

		_min = _map.begin()->first;
		_max = _map.rbegin()->first;

		_cache.resize(steps + 1);
		for (int i = 0; i <= steps; i++)
		{
			auto t = (float)i / steps;
			auto x = _min + t * (_max - _min);
			_cache[i] = calc(x);
		}

		// Save size and data to avoid STL checks penalties in DEBUG
		_size = _cache.size();
		_data = _cache.data();
	}

	std::map<float, cv::Point3f> _map;
	std::vector<cv::Point3f> _cache;
	float _min, _max;
	size_t _size; cv::Point3f* _data;
};




class colorizer
{
public:
	colorizer();

	template<typename T>
	static void update_histogram(int* hist, const T* depth_data, int w, int h)
	{
		memset(hist, 0, MAX_DEPTH * sizeof(int));
		for (auto i = 0; i < w*h; ++i)
		{
			T depth_val = depth_data[i];
			int index = static_cast<int>(depth_val);
			hist[index] += 1;
		}

		for (auto i = 2; i < MAX_DEPTH; ++i) hist[i] += hist[i - 1]; // Build a cumulative histogram for the indices in [1,0xFFFF]
	}

	static const int MAX_DEPTH = 0x10000;
	static const int MAX_DISPARITY = 0x2710;

	void process(cv::Mat& imgD, cv::Mat& outColor);

protected:

	template<typename T, typename F>
	void make_rgb_data(const T* depth_data, cv::Vec3b* rgb_data, int width, int height, F coloring_func)
	{
		auto cm = _maps[_map_index];
		for (auto i = 0; i < width*height; ++i)
		{
			auto d = depth_data[i];
			//std::cout << "d:" << d << std::endl;
			colorize_pixel(rgb_data, i, cm, d, coloring_func);
		}
	}

	template<typename T, typename F>
	static void colorize_pixel(cv::Vec3b* rgb_data, int idx, color_map* cm, T data, F coloring_func)
	{
		if (data)
		{
			auto f = coloring_func(data); // 0-255 based on histogram locationcolorize_pixel
			auto c = cm->get(f);
			rgb_data[idx][0] = (uint8_t)c.x;
			rgb_data[idx][1] = (uint8_t)c.y;
			rgb_data[idx][2] = (uint8_t)c.z;
			//std::cout << rgb_data[idx] << std::endl;
		}
		else
		{
			rgb_data[idx][0] = 0;
			rgb_data[idx][1] = 0;
			rgb_data[idx][2] = 0;
		}
	}


	std::vector<color_map*> _maps;
	int _map_index = 0;

	std::vector<int> _histogram;
	int* _hist_data;


};