#include "segmenter.hpp"
#include "json.hpp"
#include <algorithm>
#include <charconv>
#include <climits>
#include <cstdint>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <unordered_map>
#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>

namespace seg {
using json = nlohmann::json;
namespace fs = std::filesystem;
namespace {
int integer(const json& value, int minimum, int maximum, const char* name) {
    if (!value.is_number_integer())
        throw std::invalid_argument(std::string(name) + " must be an integer");
    if (value.is_number_unsigned()) {
        auto n = value.get<uint64_t>();
        if (n > static_cast<uint64_t>(maximum))
            throw std::invalid_argument(std::string(name) + " is out of range");
    }
    auto n = value.get<int64_t>();
    if (n < minimum || n > maximum)
        throw std::invalid_argument(std::string(name) + " is out of range");
    return static_cast<int>(n);
}
void category_valid(const std::string& category) {
    if (category.empty() || category.size() > 12 ||
        !std::all_of(category.begin(), category.end(),
                     [](unsigned char c) { return c >= '0' && c <= '9'; }))
        throw std::invalid_argument("catagory must contain 1-12 ASCII digits");
}
bool inside(const fs::path& child, const fs::path& parent) {
    auto p = parent.begin(), c = child.begin();
    for (; p != parent.end(); ++p, ++c)
        if (c == child.end() || *c != *p) return false;
    return c != child.end();
}
std::vector<int> starts(int length, int window, int stride) {
    std::vector<int> positions;
    int last = std::max(0, length - window);
    for (int position = 0;;) {
        positions.push_back(position);
        if (position == last) break;
        position = static_cast<int>(std::min<int64_t>(
            static_cast<int64_t>(position) + stride, last));
    }
    return positions;
}
cv::Mat paint(const std::vector<cv::Mat>& scores,
              const std::vector<cv::Vec3b>& colors) {
    cv::Mat mask(scores.front().size(), CV_8UC3);
    std::vector<const float*> rows(scores.size());
    for (int y = 0; y < mask.rows; ++y) {
        for (size_t c = 0; c < scores.size(); ++c)
            rows[c] = scores[c].ptr<float>(y);
        auto* pixels = mask.ptr<cv::Vec3b>(y);
        for (int x = 0; x < mask.cols; ++x) {
            size_t winner = 0;
            for (size_t c = 1; c < rows.size(); ++c)
                if (rows[c][x] > rows[winner][x]) winner = c;
            pixels[x] = colors[winner];
        }
    }
    return mask;
}
}
double milliseconds(Clock::time_point start) {
    return std::chrono::duration<double, std::milli>(Clock::now() - start).count();
}
int parse_item(const std::string& value) {
    if (value.empty() || value.size() > 10 ||
        !std::all_of(value.begin(), value.end(),
                     [](unsigned char c) { return c >= '0' && c <= '9'; }))
        throw std::invalid_argument("item must be a nonnegative integer");
    int result = 0;
    auto parsed = std::from_chars(value.data(), value.data() + value.size(), result);
    if (parsed.ec != std::errc{} || parsed.ptr != value.data() + value.size())
        throw std::invalid_argument("item is out of range");
    return result;
}
std::string model_key(const std::string& category, int item) {
    category_valid(category);
    if (item < 0) throw std::invalid_argument("item must be nonnegative");
    std::ostringstream name;
    name << category << '_' << std::setfill('0') << std::setw(3) << item;
    return name.str();
}
Config load_config(const fs::path& file, const fs::path& models_dir) {
    std::ifstream stream(file, std::ios::binary);
    if (!stream) throw std::runtime_error("Cannot open config: " + file.u8string());
    json document;
    stream >> document;
    Config result;
    result.catagory = document.at("catagory").get<std::string>();
    result.item = integer(document.at("item"), 0, INT_MAX, "item");
    if (file.filename().u8string() != model_key(result.catagory, result.item) + ".json")
        throw std::invalid_argument("Config filename and catagory/item disagree: " +
                                    file.filename().u8string());
    const auto& detail = document.at("model_detail");
    const auto& shape = detail.at("input_shape");
    if (!shape.is_array() || shape.size() != 2)
        throw std::invalid_argument("input_shape must be [height,width]");
    result.height = integer(shape[0], 1, 16384, "input height");
    result.width = integer(shape[1], 1, 16384, "input width");
    result.classes = integer(detail.at("num_classes"), 2, 4096, "num_classes");
    const auto& cut = detail.at("cut_flag");
    result.tiled = cut.is_boolean() ? cut.get<bool>()
                                  : integer(cut, 0, 1, "cut_flag") != 0;
    if (result.tiled)
        result.stride = integer(detail.at("stride"), 1,
                                std::min(result.height, result.width), "stride");
    const auto resize_mode = detail.value("resize_mode", std::string("stretch"));
    if (resize_mode != "stretch" && resize_mode != "letterbox")
        throw std::invalid_argument("resize_mode must be stretch or letterbox");
    result.letterbox = resize_mode == "letterbox";
    if (detail.contains("mean") != detail.contains("std"))
        throw std::invalid_argument("mean and std must be specified together");
    if (detail.contains("mean")) {
        const auto& mean = detail.at("mean");
        const auto& stddev = detail.at("std");
        if (!mean.is_array() || mean.size() != 3 ||
            !stddev.is_array() || stddev.size() != 3)
            throw std::invalid_argument("mean/std must each contain three RGB numbers");
        for (size_t channel = 0; channel < 3; ++channel) {
            if (!mean[channel].is_number() || !stddev[channel].is_number())
                throw std::invalid_argument("mean/std must be numeric");
            result.mean[channel] = mean[channel].get<float>();
            result.stddev[channel] = stddev[channel].get<float>();
            if (!std::isfinite(result.mean[channel]) ||
                !std::isfinite(result.stddev[channel]) || result.stddev[channel] <= 0)
                throw std::invalid_argument("mean/std must be finite and std must be positive");
        }
    }
    const auto& colors = detail.at("colors");
    if ((!colors.is_array() && !colors.is_object()) ||
        colors.size() != static_cast<size_t>(result.classes))
        throw std::invalid_argument("colors must contain exactly num_classes entries");
    for (int c = 0; c < result.classes; ++c) {
        const auto& color = colors.is_array() ? colors.at(c)
                                              : colors.at(std::to_string(c));
        if (!color.is_array() || color.size() != 3)
            throw std::invalid_argument("Each color must be [R,G,B]");
        int red = integer(color[0], 0, 255, "R");
        int green = integer(color[1], 0, 255, "G");
        int blue = integer(color[2], 0, 255, "B");
        result.bgr_colors.emplace_back(static_cast<uchar>(blue),
                                      static_cast<uchar>(green),
                                      static_cast<uchar>(red));
    }
    fs::path relative = fs::u8path(detail.at("model_name").get<std::string>());
    if (relative.empty() || relative.is_absolute() || relative.has_root_name())
        throw std::invalid_argument("model_name must be relative to models/");
    fs::path base = fs::weakly_canonical(models_dir);
    result.model = fs::weakly_canonical(base / relative);
    if (!inside(result.model, base) || result.model.extension() != ".onnx" ||
        !fs::is_regular_file(result.model))
        throw std::invalid_argument("model_name must resolve to an existing .onnx inside models/");
    return result;
}
Segmenter::Segmenter(Ort::Env& env, Config config, int threads, size_t max_work_bytes,
                       bool use_openvino)
    : config_(std::move(config)), max_work_bytes_(max_work_bytes) {
    const uint64_t warm_bytes = static_cast<uint64_t>(config_.height) * config_.width *
                              (12ull + (config_.letterbox ? 6ull : 0ull) +
                               8ull * config_.classes);
    if (warm_bytes > max_work_bytes_)
        throw std::invalid_argument("Model window exceeds --max-work-mb");
    Ort::SessionOptions options;
    options.SetIntraOpNumThreads(threads);
    options.SetGraphOptimizationLevel(use_openvino ? GraphOptimizationLevel::ORT_DISABLE_ALL
                                                   : GraphOptimizationLevel::ORT_ENABLE_ALL);
    if (use_openvino) {
        // OpenVINO owns graph optimization; one FP32 CPU stream targets latency.
        Ort::ThrowOnError(Ort::GetApi().AddFreeDimensionOverrideByName(options, "batch_size", 1));
        std::unordered_map<std::string, std::string> provider{
            {"device_type", "CPU"}, {"precision", "FP32"}, {"num_streams", "1"},
            {"load_config", R"({"CPU":{"PERFORMANCE_HINT":"LATENCY","INFERENCE_PRECISION_HINT":"f32","SCHEDULING_CORE_TYPE":"ANY_CORE","ENABLE_CPU_PINNING":true,"ENABLE_HYPER_THREADING":false}})"}};
        if (threads > 0) provider["num_of_threads"] = std::to_string(threads);
        options.AppendExecutionProvider_OpenVINO_V2(provider);
    }
    options.AddConfigEntry("session.intra_op.allow_spinning", "0");
    options.AddConfigEntry("session.inter_op.allow_spinning", "0");
    // std::filesystem::path supplies UTF-16 on Windows.
    session_ = Ort::Session(env, config_.model.c_str(), options);
    if (session_.GetInputCount() != 1 || session_.GetOutputCount() == 0)
        throw std::invalid_argument("Model must have exactly one input and at least one output");
    Ort::AllocatorWithDefaultOptions allocator;
    input_name_ = session_.GetInputNameAllocated(0, allocator).get();
    // Keep TypeInfo alive while accessing its borrowed tensor-info wrapper.
    auto input_type = session_.GetInputTypeInfo(0);
    auto input_info = input_type.GetTensorTypeAndShapeInfo();
    auto shape = input_info.GetShape();
    const std::vector<int64_t> expected{1, 3, config_.height, config_.width};
    if (input_info.GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT ||
        shape.size() != 4)
        throw std::invalid_argument("Input must be float32 NCHW rank 4");
    for (size_t d = 0; d < 4; ++d)
        if (shape[d] != -1 && shape[d] != expected[d])
            throw std::invalid_argument("Input shape disagrees with [1,3,H,W] config");
    // Select the unique class-score output; auxiliary boundary heads may coexist.
    for (size_t index = 0; index < session_.GetOutputCount(); ++index) {
        auto output_type = session_.GetOutputTypeInfo(index);
        if (output_type.GetONNXType() != ONNX_TYPE_TENSOR) continue;
        auto output_info = output_type.GetTensorTypeAndShapeInfo();
        auto output_shape = output_info.GetShape();
        if (output_info.GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT ||
            output_shape.size() != 4 ||
            (output_shape[0] != -1 && output_shape[0] != 1) ||
            (output_shape[1] != -1 && output_shape[1] != config_.classes))
            continue;
        if (!output_name_.empty())
            throw std::invalid_argument("Ambiguous model: multiple float32 [1,num_classes,h,w] outputs");
        output_name_ = session_.GetOutputNameAllocated(index, allocator).get();
    }
    if (output_name_.empty())
        throw std::invalid_argument("No segmentation output: need float32 [1,num_classes,h,w]");
    Timings warm;
    infer_scores(cv::Mat::zeros(config_.height, config_.width, CV_8UC3),
                 cv::Size(config_.width, config_.height), warm);
}
std::vector<cv::Mat> Segmenter::infer_scores(const cv::Mat& image,
                                            cv::Size target_size, Timings& timings) {
    auto start = Clock::now();
    const cv::Size input_size(config_.width, config_.height);
    cv::Mat prepared = image;
    cv::Rect content(0, 0, config_.width, config_.height);
    if (config_.letterbox) {
        const double scale = std::min(static_cast<double>(config_.height) / image.rows,
                                      static_cast<double>(config_.width) / image.cols);
        // Match Python int(h*scale), including odd padding split toward bottom/right.
        const int height = std::clamp(static_cast<int>(image.rows * scale), 1, config_.height);
        const int width = std::clamp(static_cast<int>(image.cols * scale), 1, config_.width);
        content = cv::Rect((config_.width - width) / 2,
                           (config_.height - height) / 2, width, height);
        cv::Mat resized;
        cv::resize(image, resized, content.size(), 0, 0, cv::INTER_LINEAR);
        prepared = cv::Mat(input_size, CV_8UC3, cv::Scalar(128, 128, 128));
        resized.copyTo(prepared(content));
    }
    cv::Mat blob = cv::dnn::blobFromImage(prepared, 1.0 / 255.0,
        input_size, cv::Scalar(), true, false, CV_32F);
    const size_t input_plane = static_cast<size_t>(config_.height) * config_.width;
    auto* input_values = blob.ptr<float>();
    for (size_t channel = 0; channel < 3; ++channel) {
        if (config_.mean[channel] == 0.0f && config_.stddev[channel] == 1.0f) continue;
        auto* values = input_values + channel * input_plane;
        for (size_t pixel = 0; pixel < input_plane; ++pixel)
            values[pixel] = (values[pixel] - config_.mean[channel]) / config_.stddev[channel];
    }
    const int64_t dims[]{1, 3, config_.height, config_.width};
    auto memory = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    auto input = Ort::Value::CreateTensor<float>(memory, blob.ptr<float>(),
                                               blob.total(), dims, 4);
    timings.preprocess_ms += milliseconds(start);
    const char* input_names[]{input_name_.c_str()};
    const char* output_names[]{output_name_.c_str()};
    start = Clock::now();
    auto output = session_.Run(Ort::RunOptions{nullptr}, input_names, &input, 1,
                               output_names, 1);
    timings.inference_ms += milliseconds(start);
    ++timings.windows;
    start = Clock::now();
    auto& tensor = output.front();
    if (!tensor.IsTensor()) throw std::runtime_error("Output is not a tensor");
    auto info = tensor.GetTensorTypeAndShapeInfo();
    auto shape = info.GetShape();
    if (info.GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT ||
        shape.size() != 4 || shape[0] != 1 || shape[1] != config_.classes ||
        shape[2] <= 0 || shape[3] <= 0 || shape[2] > INT_MAX || shape[3] > INT_MAX)
        throw std::runtime_error("Runtime output must be float32 [1,C,h,w]");
    const size_t plane_size = static_cast<size_t>(shape[2]) * shape[3];
    if (plane_size > max_work_bytes_ / sizeof(float) / config_.classes ||
        info.GetElementCount() != plane_size * config_.classes)
        throw std::runtime_error("Output exceeds --max-work-mb or has invalid shape");
    const float* values = tensor.GetTensorData<float>();
    std::vector<cv::Mat> scores;
    scores.reserve(config_.classes);
    for (int c = 0; c < config_.classes; ++c) {
        cv::Mat plane(static_cast<int>(shape[2]), static_cast<int>(shape[3]),
                      CV_32F, const_cast<float*>(values + c * plane_size));
        if (!cv::checkRange(plane, true, nullptr))
            throw std::runtime_error("Model returned NaN or infinite scores");
        cv::Mat spatial = plane;
        if (config_.letterbox) {
            if (plane.size() != input_size)
                cv::resize(plane, spatial, input_size, 0, 0, cv::INTER_LINEAR);
            spatial = spatial(content); // Remove padding before restoring original geometry.
        }
        cv::Mat restored;
        if (spatial.size() == target_size) restored = spatial.clone();
        else cv::resize(spatial, restored, target_size, 0, 0, cv::INTER_LINEAR);
        scores.push_back(std::move(restored));
    }
    timings.postprocess_ms += milliseconds(start);
    return scores;
}
cv::Mat Segmenter::predict(const cv::Mat& image, Timings& timings) {
    if (image.empty() || image.type() != CV_8UC3)
        throw std::invalid_argument("Image must decode to three-channel color");
    uint64_t pixels = static_cast<uint64_t>(image.rows) * image.cols;
    uint64_t window_pixels = static_cast<uint64_t>(config_.height) * config_.width;
    uint64_t estimated = pixels * (9ull + 4ull * config_.classes) +
                         window_pixels * (15ull + (config_.letterbox ? 6ull : 0ull) +
                                          8ull * config_.classes);
    if (estimated > max_work_bytes_)
        throw std::invalid_argument("Image working buffers exceed --max-work-mb");
    if (!config_.tiled) {
        auto scores = infer_scores(image, image.size(), timings);
        auto start = Clock::now();
        cv::Mat mask = paint(scores, config_.bgr_colors);
        timings.postprocess_ms += milliseconds(start);
        return mask;
    }
    auto start = Clock::now();
    std::vector<cv::Mat> sums;
    for (int c = 0; c < config_.classes; ++c)
        sums.push_back(cv::Mat::zeros(image.size(), CV_32F));
    cv::Mat counts = cv::Mat::zeros(image.size(), CV_32F);
    timings.postprocess_ms += milliseconds(start);
    // Window positioning and score/count averaging adapted from MMSegmentation's
    // EncoderDecoder.slide_inference (Apache-2.0); see THIRD_PARTY_NOTICES.md.
    for (int y : starts(image.rows, config_.height, config_.stride)) {
        for (int x : starts(image.cols, config_.width, config_.stride)) {
            start = Clock::now();
            cv::Rect region(x, y, std::min(config_.width, image.cols - x),
                                 std::min(config_.height, image.rows - y));
            cv::Mat patch(config_.height, config_.width, CV_8UC3,
                          config_.letterbox ? cv::Scalar(128, 128, 128) : cv::Scalar());
            image(region).copyTo(patch(cv::Rect(0, 0, region.width, region.height)));
            timings.preprocess_ms += milliseconds(start);
            auto scores = infer_scores(patch, patch.size(), timings);
            start = Clock::now();
            for (int c = 0; c < config_.classes; ++c)
                sums[c](region) += scores[c](cv::Rect(0, 0, region.width, region.height));
            counts(region) += 1.0f;
            timings.postprocess_ms += milliseconds(start);
        }
    }
    start = Clock::now();
    double minimum = 0;
    cv::minMaxLoc(counts, &minimum, nullptr);
    if (minimum <= 0) throw std::runtime_error("Sliding windows left uncovered pixels");
    for (auto& scores : sums) cv::divide(scores, counts, scores);
    cv::Mat mask = paint(sums, config_.bgr_colors);
    timings.postprocess_ms += milliseconds(start);
    return mask;
}
}
