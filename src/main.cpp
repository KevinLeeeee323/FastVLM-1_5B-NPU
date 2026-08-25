#include "image_encoder.h"
#include "model_contract.h"
#include "rkllm.h"

#include <opencv2/opencv.hpp>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <csignal>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

namespace {

LLMHandle g_llm = nullptr;

struct Options {
    std::string image;
    std::string vision;
    std::string llm;
    std::string manifest = "models/manifest.json";
    std::string prompt;
    int max_new_tokens = 0;
    int max_context = 0;
    int vision_cores = 0;
    bool once = false;
    bool text_only = false;
    bool help = false;
};

void print_usage(const char* program) {
    std::cerr << "Usage: " << program << " --image IMAGE --vision VISION.rknn --llm MODEL.rkllm [options]\n"
              << "Options:\n"
              << "  --max-new-tokens N  Maximum generated tokens (default: 256)\n"
              << "  --max-context N     Runtime context, <= manifest build maximum\n"
              << "  --vision-cores N    RKNN cores: 1, 2, or 3\n"
              << "  --manifest FILE     Versioned model contract (default: models/manifest.json)\n"
              << "  --prompt TEXT       Run one prompt instead of interactive mode\n"
              << "  --text-only         Load only the RKLLM for a pure-text smoke test\n"
              << "  --once              Exit after the initial image prompt\n"
              << "  --help              Show this message\n";
}

bool take_value(int argc, char** argv, int* index, std::string* value) {
    if (*index + 1 >= argc) return false;
    *value = argv[++(*index)];
    return true;
}

bool parse_int(const std::string& value, int* result) {
    char* end = nullptr;
    long parsed = std::strtol(value.c_str(), &end, 10);
    if (end == value.c_str() || *end != '\0' || parsed < 0 || parsed > std::numeric_limits<int>::max()) return false;
    *result = static_cast<int>(parsed);
    return true;
}

bool parse_args(int argc, char** argv, Options* options) {
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        std::string value;
        if (arg == "--help" || arg == "-h") { options->help = true; return true; }
        if (arg == "--image" && take_value(argc, argv, &i, &value)) options->image = value;
        else if (arg == "--vision" && take_value(argc, argv, &i, &value)) options->vision = value;
        else if (arg == "--llm" && take_value(argc, argv, &i, &value)) options->llm = value;
        else if (arg == "--manifest" && take_value(argc, argv, &i, &value)) options->manifest = value;
        else if (arg == "--prompt" && take_value(argc, argv, &i, &value)) { options->prompt = value; options->once = true; }
        else if (arg == "--max-new-tokens" && take_value(argc, argv, &i, &value) && parse_int(value, &options->max_new_tokens)) {}
        else if (arg == "--max-context" && take_value(argc, argv, &i, &value) && parse_int(value, &options->max_context)) {}
        else if (arg == "--vision-cores" && take_value(argc, argv, &i, &value) && parse_int(value, &options->vision_cores)) {}
        else if (arg == "--once") options->once = true;
        else if (arg == "--text-only") options->text_only = true;
        else { std::cerr << "Unknown or incomplete argument: " << arg << "\n"; print_usage(argv[0]); return false; }
    }
    if (options->help) return true;
    if (options->llm.empty() || (!options->text_only && (options->image.empty() || options->vision.empty()))) {
        print_usage(argv[0]);
        return false;
    }
    if (options->max_new_tokens < 0 || options->max_context < 0 || options->vision_cores < 0 || options->vision_cores > 3) {
        std::cerr << "Invalid numeric option\n";
        return false;
    }
    return true;
}

void destroy_llm(int) {
    if (g_llm != nullptr) {
        LLMHandle handle = g_llm;
        g_llm = nullptr;
        rkllm_destroy(handle);
    }
    std::_Exit(130);
}

int callback(RKLLMResult* result, void*, LLMCallState state) {
    if (state == RKLLM_RUN_NORMAL && result != nullptr && result->text != nullptr) {
        std::fputs(result->text, stdout);
        std::fflush(stdout);
    } else if (state == RKLLM_RUN_ERROR) {
        std::fprintf(stderr, "\nRKLLM inference error\n");
    } else if (state == RKLLM_RUN_FINISH) {
        std::printf("\n");
    }
    return 0;
}

cv::Mat pad_and_resize(const cv::Mat& source, int width, int height) {
    cv::Mat rgb;
    if (source.channels() == 1) cv::cvtColor(source, rgb, cv::COLOR_GRAY2RGB);
    else if (source.channels() == 4) cv::cvtColor(source, rgb, cv::COLOR_BGRA2RGB);
    else cv::cvtColor(source, rgb, cv::COLOR_BGR2RGB);
    const int side = std::max(rgb.cols, rgb.rows);
    cv::Mat square(side, side, CV_8UC3, cv::Scalar(0, 0, 0));
    const int x = (side - rgb.cols) / 2;
    const int y = (side - rgb.rows) / 2;
    rgb.copyTo(square(cv::Rect(x, y, rgb.cols, rgb.rows)));
    cv::Mat resized;
    cv::resize(square, resized, cv::Size(width, height), 0, 0, cv::INTER_CUBIC);
    return resized;
}

int run_prompt(const std::string& prompt, const ImageEncoder& encoder,
               const std::vector<float>& embedding, int image_width, int image_height) {
    RKLLMInput input{};
    input.role = "user";
    input.enable_thinking = false;
    input.input_type = prompt.find("<image>") == std::string::npos ? RKLLM_INPUT_PROMPT : RKLLM_INPUT_MULTIMODAL;
    RKLLMInferParam infer{};
    infer.mode = RKLLM_INFER_GENERATE;
    infer.keep_history = 1;
    infer.max_new_tokens = -1;
    if (input.input_type == RKLLM_INPUT_PROMPT) {
        input.prompt_input = prompt.c_str();
    } else {
        input.multimodal_input.prompt = const_cast<char*>(prompt.c_str());
        input.multimodal_input.image.image_embed = const_cast<float*>(embedding.data());
        input.multimodal_input.image.n_image_tokens = encoder.image_tokens;
        input.multimodal_input.image.n_image = 1;
        input.multimodal_input.image.image_start = "";
        input.multimodal_input.image.image_end = "";
        input.multimodal_input.image.image_content = "<image>";
        input.multimodal_input.image.image_width = static_cast<size_t>(image_width);
        input.multimodal_input.image.image_height = static_cast<size_t>(image_height);
    }
    const int ret = rkllm_run(g_llm, &input, &infer, nullptr);
    if (ret != 0) std::fprintf(stderr, "rkllm_run failed: %d\n", ret);
    return ret;
}

}  // namespace

int main(int argc, char** argv) {
    Options options;
    if (!parse_args(argc, argv, &options)) return 2;
    if (options.help) { print_usage(argv[0]); return 0; }
    std::signal(SIGINT, destroy_llm);
    std::signal(SIGTERM, destroy_llm);

    ModelContract contract;
    std::string contract_error;
    if (!load_model_contract(options.manifest, &contract, &contract_error)) {
        std::cerr << contract_error << "\n";
        return 1;
    }
    if (contract.target_platform != "rk3588" || contract.rkllm_version != "1.3.0") {
        std::cerr << "Manifest targets " << contract.target_platform << " with RKLLM "
                  << contract.rkllm_version << "; this executable requires RK3588/RKLLM 1.3.0\n";
        return 1;
    }
    if (!contract_contains_artifact(contract, options.llm) ||
        (!options.text_only && !contract_contains_artifact(contract, options.vision))) {
        std::cerr << "Selected model file is not part of manifest " << options.manifest << "\n";
        return 1;
    }
    if (options.max_context == 0) options.max_context = contract.default_max_context;
    if (options.max_new_tokens == 0) options.max_new_tokens = contract.default_max_new_tokens;
    if (options.vision_cores == 0) options.vision_cores = contract.default_vision_cores;
    if (options.max_context > contract.build_max_context) {
        std::cerr << "--max-context " << options.max_context << " exceeds RKLLM build maximum "
                  << contract.build_max_context << "\n";
        return 1;
    }
    if (options.max_context <= 0 || options.max_new_tokens <= 0 ||
        options.vision_cores < 1 || options.vision_cores > 3) {
        std::cerr << "Invalid runtime defaults in model manifest\n";
        return 1;
    }
    std::cout << "model contract: " << contract.model_id << ", RKLLM " << contract.rkllm_version
              << ", max_context=" << options.max_context << "/" << contract.build_max_context << "\n";

    if (options.text_only) {
        RKLLMParam param = rkllm_createDefaultParam();
        param.model_path = options.llm.c_str();
        param.max_new_tokens = options.max_new_tokens;
        param.max_context_len = options.max_context;
        param.top_k = 1;
        param.skip_special_token = true;
        param.extend_param.base_domain_id = 1;
        RKLLMCallback callback_config{};
        callback_config.result_callback = callback;
        int text_ret = rkllm_init(&g_llm, &param, &callback_config);
        if (text_ret != 0) {
            std::fprintf(stderr, "rkllm_init failed: %d\n", text_ret);
            return 1;
        }
        std::cout << "RKLLM text-only smoke. Type 'clear' to reset history, 'exit' to quit.\n";
        std::string prompt = options.prompt.empty() ? "请用一句话介绍你自己。" : options.prompt;
        if (prompt.find("<image>") != std::string::npos) {
            std::cerr << "--text-only prompt must not contain <image>\n";
            rkllm_destroy(g_llm);
            g_llm = nullptr;
            return 1;
        }
        ImageEncoder unused_encoder;
        std::vector<float> unused_embedding;
        text_ret = run_prompt(prompt, unused_encoder, unused_embedding, 0, 0);
        int exit_code = text_ret == 0 ? 0 : 1;
        if (!options.once && text_ret == 0) {
            for (;;) {
                std::cout << "user> " << std::flush;
                if (!std::getline(std::cin, prompt) || prompt == "exit") break;
                if (prompt == "clear") {
                    text_ret = rkllm_clear_kv_cache(g_llm, 1, nullptr, nullptr);
                    std::cout << (text_ret == 0 ? "history cleared\n" : "clear failed\n");
                    continue;
                }
                if (!prompt.empty() && run_prompt(prompt, unused_encoder, unused_embedding, 0, 0) != 0) {
                    exit_code = 1;
                    break;
                }
            }
        }
        rkllm_destroy(g_llm);
        g_llm = nullptr;
        return exit_code;
    }

    cv::Mat source = cv::imread(options.image, cv::IMREAD_COLOR);
    if (source.empty()) {
        std::cerr << "Cannot read image: " << options.image << "\n";
        return 1;
    }

    ImageEncoder encoder;
    int ret = image_encoder_init(options.vision.c_str(), options.vision_cores, &encoder);
    if (ret != 0) return 1;
    cv::Mat image = pad_and_resize(source, encoder.width, encoder.height);
    std::vector<float> embedding(encoder.image_tokens * encoder.embedding_size);
    const auto vision_start = std::chrono::steady_clock::now();
    ret = image_encoder_run(&encoder, image.ptr<std::uint8_t>(), embedding.data(), embedding.size());
    const auto vision_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - vision_start).count();
    if (ret != 0) {
        std::cerr << "Vision inference failed: " << ret << "\n";
        image_encoder_release(&encoder);
        return 1;
    }
    std::printf("vision inference: %.2f ms, image tokens=%zu, embedding=%zu\n",
                vision_ms, encoder.image_tokens, encoder.embedding_size);

    RKLLMParam param = rkllm_createDefaultParam();
    param.model_path = options.llm.c_str();
    param.max_new_tokens = options.max_new_tokens;
    param.max_context_len = options.max_context;
    param.top_k = 1;
    param.skip_special_token = true;
    param.extend_param.base_domain_id = 1;
    RKLLMCallback callback_config{};
    callback_config.result_callback = callback;
    ret = rkllm_init(&g_llm, &param, &callback_config);
    if (ret != 0) {
        std::fprintf(stderr, "rkllm_init failed: %d\n", ret);
        image_encoder_release(&encoder);
        return 1;
    }
    std::printf("RKLLM initialized. Type 'clear' to reset history, 'exit' to quit.\n");

    std::string first_prompt = options.prompt.empty() ? "请描述图片中的主要物体和场景。" : options.prompt;
    if (first_prompt.find("<image>") == std::string::npos) first_prompt = "<image>\n" + first_prompt;
    ret = run_prompt(first_prompt, encoder, embedding, encoder.width, encoder.height);
    if (ret != 0) {
        rkllm_destroy(g_llm);
        g_llm = nullptr;
        image_encoder_release(&encoder);
        return 1;
    }
    if (options.once) {
        rkllm_destroy(g_llm);
        g_llm = nullptr;
        image_encoder_release(&encoder);
        return 0;
    }

    int exit_code = 0;
    for (;;) {
        std::cout << "user> " << std::flush;
        std::string prompt;
        if (!std::getline(std::cin, prompt) || prompt == "exit") break;
        if (prompt == "clear") {
            ret = rkllm_clear_kv_cache(g_llm, 1, nullptr, nullptr);
            std::cout << (ret == 0 ? "history cleared\n" : "clear failed\n");
            continue;
        }
        if (prompt.empty()) continue;
        if (run_prompt(prompt, encoder, embedding, encoder.width, encoder.height) != 0) {
            exit_code = 1;
            break;
        }
    }

    rkllm_destroy(g_llm);
    g_llm = nullptr;
    image_encoder_release(&encoder);
    return exit_code;
}
