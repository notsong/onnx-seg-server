#pragma once
#include <array>
#include <chrono>
#include <filesystem>
#include <memory>
#include <string>
#include <vector>
#include <opencv2/core.hpp>
#include <onnxruntime_cxx_api.h>

namespace seg {
struct Config {
    std::string catagory;
    int item = 0;
    std::filesystem::path model;
    int height = 0, width = 0, classes = 0;
    bool tiled = false, letterbox = false;
    std::array<float, 3> mean{0.0f, 0.0f, 0.0f};
    std::array<float, 3> stddev{1.0f, 1.0f, 1.0f};
    int stride = 0;
    std::vector<cv::Vec3b> bgr_colors;
};
struct Timings {
    double decode_ms = 0, preprocess_ms = 0, inference_ms = 0;
    double postprocess_ms = 0, encode_ms = 0;
    size_t windows = 0;
};
using Clock = std::chrono::steady_clock;
double milliseconds(Clock::time_point start);
std::string model_key(const std::string& catagory, int item);
int parse_item(const std::string& value);
Config load_config(const std::filesystem::path& file,
                   const std::filesystem::path& models_dir);

class Segmenter {
public:
    Segmenter(Ort::Env& env, Config config, int threads, size_t max_work_bytes,
              bool use_openvino = false);
    cv::Mat predict(const cv::Mat& image, Timings& timings);
    const Config& config() const { return config_; }
private:
    std::vector<cv::Mat> infer_scores(const cv::Mat& image,
                                     cv::Size target_size, Timings& timings);
    Config config_;
    Ort::Session session_{nullptr};
    std::string input_name_, output_name_;
    size_t max_work_bytes_;
};
}
