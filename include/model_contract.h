#pragma once

#include <string>

struct ModelContract {
    int schema_version = 0;
    int embedding_size = 0;
    int build_max_context = 0;
    int default_max_context = 0;
    int default_max_new_tokens = 0;
    int default_vision_cores = 0;
    std::string model_id;
    std::string target_platform;
    std::string rkllm_version;
    std::string raw_json;
};

bool load_model_contract(const std::string& path, ModelContract* contract, std::string* error);
bool contract_contains_artifact(const ModelContract& contract, const std::string& path);

