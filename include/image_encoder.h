#pragma once

#include <cstddef>
#include <cstdint>

#include "rknn_api.h"

struct ImageEncoder {
    rknn_context context = 0;
    rknn_input_output_num io_num{};
    rknn_tensor_attr input{};
    rknn_tensor_attr output{};
    int width = 0;
    int height = 0;
    int channels = 0;
    size_t image_tokens = 0;
    size_t embedding_size = 0;
};

int image_encoder_init(const char* model_path, int core_count, ImageEncoder* encoder);
int image_encoder_run(ImageEncoder* encoder, const std::uint8_t* rgb_hwc,
                      float* embedding, size_t embedding_floats);
void image_encoder_release(ImageEncoder* encoder);

