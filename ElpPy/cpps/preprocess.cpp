#include "ElpTools.h"
#include <opencv2/opencv.hpp>

void calCannyThreshold(cv::Mat &ImgG, int &low, int &high)
{
	cv::Mat ImgT, dx, dy, grad;
	cv::resize(ImgG, ImgT, cv::Size(ImgG.cols / 10, ImgG.rows / 10));
	cv::Sobel(ImgT, dx, CV_16SC1, 1, 0);
	cv::Sobel(ImgT, dy, CV_16SC1, 0, 1);
	short *_dx = (short*)dx.data, *_dy = (short*)dy.data;

	int subpixel_num = dx.rows*dx.cols;
	grad.create(1, subpixel_num, CV_32SC1);
	int* _grad = (int*)grad.data;
	int maxGrad(0);
	for (int i = 0; i < subpixel_num; i++)
	{
		_grad[i] = std::abs(_dx[i]) + std::abs(_dy[i]);
		if (maxGrad < _grad[i])
			maxGrad = _grad[i];
	}

	//set magic numbers
	const int NUM_BINS = 64;
	const double percent_of_pixels_not_edges = 0.7;
	const double threshold_ratio = 0.4;
	int bins[NUM_BINS] = { 0 };


	//compute histogram
#if defined(__GNUC__)
        int bin_size = std::floor(maxGrad / float(NUM_BINS) + 0.5f) + 1;
#else
        int bin_size = std::floorf(maxGrad / float(NUM_BINS) + 0.5f) + 1;
#endif
	if (bin_size < 1) bin_size  = 1;
	for (int i = 0; i < subpixel_num; i++)
	{
		bins[_grad[i] / bin_size]++;
	}

	//% Select the thresholds
	float total(0.f);
	float target = float(subpixel_num * percent_of_pixels_not_edges);

	high = 0;
	while (total < target)
	{
		total += bins[high];
		high++;
	}
	//	high *= bin_size;
	high *= (255.0f / NUM_BINS);
	//	low = std::min((int)std::floor(threshold_ratio * float(high)), 30);
	low = threshold_ratio*float(high);

}

void calCannyThreshold(unsigned char *_ImgG, int rows, int cols, int *low, int *high)
{
    cv::Mat imgG(rows, cols, CV_8UC1, _ImgG);
    int tlow = 0, thigh = 10;

    calCannyThreshold(imgG, tlow, thigh);

    *low = tlow;
    *high = thigh;
}

void findContour(const int Wise[8][2], const int antiWise[8][2], unsigned char *Edge, int cols, int x, int y, 
					std::vector<std::vector<cv::Point>> &edgeContours, int min_edge_num)
{
	auto iIDX = [cols](int x, int y) {return x * cols + y;};
	std::vector<cv::Point> oneContour, oneContourOpp;
	bool isEnd;
	int find_x, find_y;
	int move_x = x, move_y = y;
	oneContour.clear(); oneContourOpp.clear();
	oneContour.push_back(cv::Point(x, y));
	int idxiMove = iIDX(x, y), idxiFind;

	while (1)
	{
		isEnd = true;
		idxiMove = iIDX(move_x, move_y);
		for (int i = 0; i < 8; i++)
		{
			find_x = move_x + Wise[i][0];
			find_y = move_y + Wise[i][1];
			idxiFind = iIDX(find_x, find_y);
			if (Edge[idxiFind])
			{
				Edge[idxiFind] = 0;
				isEnd = false;
				move_x = find_x; move_y = find_y;
				oneContour.push_back(cv::Point(move_x, move_y));
				break;
			}
		}
		if (isEnd)
		{
			break;
		}
	}
	move_x = oneContour[0].x; move_y = oneContour[0].y;
	while (1)
	{
		isEnd = true;
		idxiMove = iIDX(move_x, move_y);
		for (int i = 0; i < 8; i++)
		{
			find_x = move_x + antiWise[i][0];
			find_y = move_y + antiWise[i][1];
			idxiFind = iIDX(find_x, find_y);
			if (Edge[idxiFind])
			{
				Edge[idxiFind] = 0;
				isEnd = false;
				move_x = find_x; move_y = find_y;
				oneContourOpp.push_back(cv::Point(move_x, move_y));
				break;
			}
		}
		if (isEnd)
		{
			break;
		}
	}
	if (oneContour.size() + oneContourOpp.size() > min_edge_num)
	{
		if (oneContourOpp.size() > 0)
		{
			cv::Point temp;
			for (int i = 0; i < (oneContourOpp.size() + 1) / 2; i++)
			{
				temp = oneContourOpp[i];
				oneContourOpp[i] = oneContourOpp[oneContourOpp.size() - 1 - i];
				oneContourOpp[oneContourOpp.size() - 1 - i] = temp;
			}
			oneContourOpp.insert(oneContourOpp.end(), oneContour.begin(), oneContour.end());
			edgeContours.push_back(oneContourOpp);
		}
		else
			edgeContours.push_back(oneContour);
	}
}


void findContours(unsigned char *_edge, int rows, int cols, int min_edge_num, std::vector<std::vector<cv::Point>> &edgeContours)
{
	const int clockWise[8][2] = { { 0,1 },{ 1,0 },{ 0,-1 },{ -1,0 },{ -1,1 },{ 1,1 },{ 1,-1 },{ -1,-1 } };
	const int anticlockWise[8][2] = { { 0,-1 },{ 1,0 },{ 0,1 },{ -1,0 },{ -1,-1 },{ 1,-1 },{ 1,1 },{ -1,1 } };
	int idx_first = (rows - 1)*cols;

	for (int i = 0; i < cols; i++)
	{
		_edge[i] = 0;
		_edge[idx_first + i] = 0;
	}
	for (int i = 1; i < rows - 1; i++)
	{
		_edge[i*cols] = 0;
		_edge[i*cols + cols - 1] = 0;
	}
	for (int i = 1; i < rows; i++)
	{
		idx_first = i*cols;
		for (int j = 1; j < cols; j++)
		{
			if (_edge[idx_first + j])
			{
				_edge[idx_first + j] = 0;
				if (_edge[idx_first + cols + j - 1] && _edge[idx_first + cols + j] && _edge[idx_first + cols + j + 1])
					continue;
				else
				{
					findContour(clockWise, anticlockWise, _edge, cols, i, j, edgeContours, min_edge_num);
				}
			}
		}
	}
}

void findContours(unsigned char *_edge, int rows, int cols, int min_edge_num, int *contour_num, int **contours, int *each_contour_num)
{
	std::vector<std::vector<cv::Point>> edgeContours;
	findContours(_edge, rows, cols, min_edge_num, edgeContours);
	int total_edge_num = edgeContours.size();
	*contour_num = total_edge_num;

	if (total_edge_num == 0)
		return;

	each_contour_num = new int[total_edge_num];
	contours = new int*[total_edge_num];
	for (int i = 0; i < total_edge_num; i++)
	{
		int each_num = edgeContours[i].size();
		each_contour_num[i] = each_num;
		contours[i] = new int[each_num * 2];
		for (int j = 0; j < each_num; j++)
		{
			contours[i][j * 2] = edgeContours[i][j].x;
			contours[i][j * 2 + 1] = edgeContours[i][j].y;
		}
	}

}