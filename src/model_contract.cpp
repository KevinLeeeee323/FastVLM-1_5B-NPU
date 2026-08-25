#include "model_contract.h"

#include <cctype>
#include <fstream>
#include <sstream>

namespace {

bool locate_value(const std::string& json, const std::string& key, size_t* position) {
    const std::string needle = "\"" + key + "\"";
    size_t found = json.find(needle);
    if (found == std::string::npos) return false;
    found = json.find(':', found + needle.size());
    if (found == std::string::npos) return false;
    *position = found + 1;
    while (*position < json.size() && std::isspace(static_cast<unsigned char>(json[*position]))) ++(*position);
    return *position < json.size();
}

bool read_int(const std::string& json, const std::string& key, int* value) {
    size_t position = 0;
    if (!locate_value(json, key, &position)) return false;
    bool negative = false;
    if (json[position] == '-') { negative = true; ++position; }
    if (position >= json.size() || !std::isdigit(static_cast<unsigned char>(json[position]))) return false;
    int result = 0;
    while (position < json.size() && std::isdigit(static_cast<unsigned char>(json[position]))) {
        result = result * 10 + (json[position++] - '0');
    }
    *value = negative ? -result : result;
    return true;
}

bool read_string(const std::string& json, const std::string& key, std::string* value) {
    size_t position = 0;
    if (!locate_value(json, key, &position) || json[position] != '"') return false;
    const size_t end = json.find('"', position + 1);
    if (end == std::string::npos) return false;
    *value = json.substr(position + 1, end - position - 1);
    return true;
}

std::string basename(const std::string& path) {
    const size_t slash = path.find_last_of("/\\");
    return slash == std::string::npos ? path : path.substr(slash + 1);
}

}  // namespace

bool load_model_contract(const std::string& path, ModelContract* contract, std::string* error) {
    if (contract == nullptr) return false;
    std::ifstream stream(path.c_str(), std::ios::binary);
    if (!stream) {
        if (error) *error = "cannot open model manifest: " + path;
        return false;
    }
    std::ostringstream buffer;
    buffer << stream.rdbuf();
    ModelContract parsed;
    parsed.raw_json = buffer.str();
    const bool valid = read_int(parsed.raw_json, "schema_version", &parsed.schema_version) &&
        read_int(parsed.raw_json, "embedding_size", &parsed.embedding_size) &&
        read_int(parsed.raw_json, "build_max_context", &parsed.build_max_context) &&
        read_int(parsed.raw_json, "default_max_context", &parsed.default_max_context) &&
        read_int(parsed.raw_json, "default_max_new_tokens", &parsed.default_max_new_tokens) &&
        read_int(parsed.raw_json, "default_vision_cores", &parsed.default_vision_cores) &&
        read_string(parsed.raw_json, "model_id", &parsed.model_id) &&
        read_string(parsed.raw_json, "target_platform", &parsed.target_platform) &&
        read_string(parsed.raw_json, "rkllm_version", &parsed.rkllm_version);
    if (!valid || parsed.schema_version != 1 || parsed.embedding_size != 1536 ||
        parsed.build_max_context <= 0 || parsed.default_max_context > parsed.build_max_context) {
        if (error) *error = "invalid or incompatible model manifest: " + path;
        return false;
    }
    *contract = parsed;
    return true;
}

bool contract_contains_artifact(const ModelContract& contract, const std::string& path) {
    const std::string name = basename(path);
    return !name.empty() && contract.raw_json.find("\"" + name + "\"") != std::string::npos;
}

