#define CPPHTTPLIB_FORM_URL_ENCODED_PAYLOAD_MAX_LENGTH 8192
#include "httplib.h"
#include "json.hpp"
#include "segmenter.hpp"
#include <algorithm>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <map>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <opencv2/imgcodecs.hpp>
#ifdef _WIN32
#include <windows.h>
#endif

namespace fs = std::filesystem;
using json = nlohmann::json;
namespace {
fs::path default_root() {
#ifdef _WIN32
    std::wstring module(32768, L'\0');
    DWORD size = GetModuleFileNameW(nullptr, module.data(),
                                   static_cast<DWORD>(module.size()));
    if (size == 0 || size >= module.size())
        throw std::runtime_error("Cannot locate executable directory");
    module.resize(size);
    return fs::path(module).parent_path();
#else
    return fs::current_path();
#endif
}
void keep_error_visible() {
#ifdef _WIN32
    DWORD processes[2]{};
    // A window created by Explorer has only this process attached.
    if (GetConsoleProcessList(processes, 2) == 1) {
        std::cerr << "\nPress Enter to close this window..." << std::endl;
        std::cin.get();
    }
#endif
}
void error(httplib::Response& response, int status, const std::string& message) {
    response.status = status;
    response.set_content(json{{"error", message}}.dump(), "application/json; charset=utf-8");
}
std::string form_text(const httplib::Request& request, const char* name) {
    auto values = request.form.get_fields(name);
    const size_t query_count = request.get_param_value_count(name);
    if (request.form.has_file(name) || values.size() + query_count != 1)
        throw std::invalid_argument(std::string("Supply exactly one text field: ") + name);
    if (!values.empty()) return values.front();
    return request.get_param_value(name);
}
std::string decimal(double value) {
    std::ostringstream text;
    text << std::fixed << std::setprecision(3) << value;
    return text.str();
}
}
int main(int argc, char** argv) {
#ifdef _WIN32
    SetConsoleOutputCP(CP_UTF8);
#endif
    try {
        fs::path root = default_root();
        std::string host = "127.0.0.1", backend = "onnxruntime";
        int port = 9600, threads = 0, work_mb = 1024;
        for (int i = 1; i < argc; ++i) {
            std::string flag = argv[i];
            if (flag == "--help") {
                std::cout << "onnx-seg-server [--root DIR] [--host 127.0.0.1] "
                             "[--port 9600] [--threads 0] [--max-work-mb 1024] "
                             "[--backend onnxruntime|openvino]\n";
                return 0;
            }
            if (++i >= argc) throw std::invalid_argument("Missing value for " + flag);
            std::string value = argv[i];
            if (flag == "--root") root = fs::u8path(value);
            else if (flag == "--host") host = value;
            else if (flag == "--backend") backend = value;
            else if (flag == "--port") port = seg::parse_item(value);
            else if (flag == "--threads") threads = seg::parse_item(value);
            else if (flag == "--max-work-mb") work_mb = seg::parse_item(value);
            else throw std::invalid_argument("Unknown option: " + flag);
        }
        if (backend != "onnxruntime" && backend != "openvino")
            throw std::invalid_argument("backend must be onnxruntime or openvino");
        if (port < 1 || port > 65535 || threads > 256 || work_mb < 16 || work_mb > 65536)
            throw std::invalid_argument("Invalid port/threads/max-work-mb");
        root = fs::canonical(root);
        if (!fs::is_directory(root / "config"))
            throw std::runtime_error("Missing config/ under --root");
        std::cout << "Inference backend: " << backend << " (CPU)\n"
                  << "Service directory: " << root.u8string() << '\n'
                  << "Configs: " << (root / "config").u8string() << '\n'
                  << "Models: " << (root / "models").u8string() << std::endl;
        Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "onnx-seg-server");
        std::vector<fs::path> config_files;
        for (const auto& entry : fs::directory_iterator(root / "config"))
            if (entry.is_regular_file() && entry.path().extension() == ".json")
                config_files.push_back(entry.path());
        std::sort(config_files.begin(), config_files.end());
        if (config_files.empty()) throw std::runtime_error("No .json configs found");
        std::map<std::string, std::unique_ptr<seg::Segmenter>> models;
        for (const auto& file : config_files) {
            try {
                auto config = seg::load_config(file, root / "models");
                auto key = seg::model_key(config.catagory, config.item);
                std::cout << "Loading " << key << "..." << std::endl;
                auto model = std::make_unique<seg::Segmenter>(
                    env, std::move(config), threads, static_cast<size_t>(work_mb) * 1024 * 1024,
                    backend == "openvino");
                models.emplace(key, std::move(model));
                std::cout << "Loaded and warmed " << key << '\n';
            } catch (const std::exception& exception) {
                throw std::runtime_error("Invalid config/model " + file.filename().u8string() +
                                         ": " + exception.what());
            }
        }
        httplib::Server server;
        // One worker serializes requests and bounds concurrent image buffers.
        server.new_task_queue = [] { return new httplib::ThreadPool(1, 16); };
#ifdef _WIN32
        // Prevent Windows SO_REUSEADDR from silently sharing an occupied port.
        server.set_socket_options([](socket_t socket) {
            const int exclusive = 1;
            if (::setsockopt(socket, SOL_SOCKET, SO_EXCLUSIVEADDRUSE,
                             reinterpret_cast<const char*>(&exclusive),
                             sizeof(exclusive)) != 0)
                throw std::runtime_error("Cannot set exclusive socket binding");
        });
#endif
        server.set_payload_max_length(64ull * 1024 * 1024);
        server.set_read_timeout(10, 0);
        server.set_write_timeout(30, 0);
        server.set_keep_alive_timeout(2);
        server.Get("/health", [&](const httplib::Request&, httplib::Response& response) {
            response.set_content(json{{"status", "ok"}, {"models", models.size()}}.dump(),
                                 "application/json");
        });
        server.Post("/infer", [&](const httplib::Request& request, httplib::Response& response) {
            const auto total_start = seg::Clock::now();
            try {
                if (!request.is_multipart_form_data())
                    throw std::invalid_argument("Use multipart/form-data with file, catagory, item");
                const auto category = form_text(request, "catagory");
                const int item = seg::parse_item(form_text(request, "item"));
                const auto key = seg::model_key(category, item);
                auto found = models.find(key);
                if (found == models.end()) {
                    error(response, 404, "Config not found: " + key + ".json");
                    return;
                }
                auto uploads = request.form.get_files("file");
                if (uploads.size() != 1 || uploads.front().content.empty())
                    throw std::invalid_argument("Supply exactly one nonempty file");
                const auto& content = uploads.front().content;
                seg::Timings timings;
                auto start = seg::Clock::now();
                cv::Mat encoded(1, static_cast<int>(content.size()), CV_8U,
                                const_cast<char*>(content.data()));
                cv::Mat image;
                try { image = cv::imdecode(encoded, cv::IMREAD_COLOR); }
                catch (const cv::Exception&) {
                    throw std::invalid_argument("File is not a valid image");
                }
                timings.decode_ms = seg::milliseconds(start);
                if (image.empty()) throw std::invalid_argument("File is not a valid image");
                cv::Mat mask = found->second->predict(image, timings);
                start = seg::Clock::now();
                std::vector<uchar> png;
                if (!cv::imencode(".png", mask, png, {cv::IMWRITE_PNG_COMPRESSION, 3}))
                    throw std::runtime_error("PNG encoding failed");
                timings.encode_ms = seg::milliseconds(start);
                response.set_content(reinterpret_cast<const char*>(png.data()), png.size(),
                                     "image/png");
                response.set_header("X-Model-Key", key);
                response.set_header("X-Tile-Count", std::to_string(timings.windows));
                response.set_header("X-Inference-Ms", decimal(timings.inference_ms));
                response.set_header("X-Total-Ms", decimal(seg::milliseconds(total_start)));
                std::cout << key << " image=" << image.cols << 'x' << image.rows
                          << " windows=" << timings.windows
                          << " decode_ms=" << decimal(timings.decode_ms)
                          << " preprocess_ms=" << decimal(timings.preprocess_ms)
                          << " inference_ms=" << decimal(timings.inference_ms)
                          << " postprocess_ms=" << decimal(timings.postprocess_ms)
                          << " encode_ms=" << decimal(timings.encode_ms)
                          << " total_ms=" << decimal(seg::milliseconds(total_start))
                          << std::endl;
            } catch (const std::invalid_argument& exception) {
                error(response, 400, exception.what());
            } catch (const std::exception& exception) {
                error(response, 500, exception.what());
            }
        });
        server.set_error_handler([](const httplib::Request&, httplib::Response& response) {
            if (response.body.empty())
                error(response, response.status, "HTTP request rejected");
        });
        std::cout << "Listening on http://" << host << ':' << port
                  << " (CPU, serial requests)" << std::endl;
        if (!server.listen(host, port))
            throw std::runtime_error("Cannot bind host/port");
        return 0;
    } catch (const std::exception& exception) {
        std::cerr << "Startup failed: " << exception.what() << std::endl;
        keep_error_visible();
        return 1;
    }
}
