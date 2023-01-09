#include "lsd.h"
#include <vector>

#include "json.hpp"

using json = nlohmann::json;

int main(int argc, char**argv)
{
    std::string img_path = std::string(argv[1]);
    cv::Mat imgC = cv::imread(img_path), imgG;

    if(imgC.channels() == 3)
    {
        cv::cvtColor(imgC, imgG, cv::COLOR_BGR2GRAY);
    }
    else
    {
        imgC.copyTo(imgG);
        cv::cvtColor(imgG, imgC, cv::COLOR_GRAY2BGR);
    }
    cv::Ptr<LSD::LineSegmentDetector> ls = LSD::createLineSegmentDetector(LSD_REFINE_STD); 

    std::vector<cv::Vec4i>lines; // 或vector<Vec4f>lines; 
    ls->detect(imgG, lines);

    std::vector<std::vector<int>> stdLines;
    int num_lines = lines.size();
    stdLines.resize(num_lines);
    for(int i = 0; i < num_lines; i++)
    {
        stdLines[i].resize(4);
        for(int k = 0; k < 4; k++)
        {
            stdLines[i][k] = lines[i][k];
        }
    }

    std::string out_json = img_path + ".json";
    json node;
    node["lines"] = stdLines;

    std::ofstream(out_json) << node;

    return 0;
}