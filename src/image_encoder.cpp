#include "image_encoder.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>

namespace {

void print_attr(const char* label, const rknn_tensor_attr& attr) {
    std::printf("%s: index=%u name=%s dims=", label, attr.index, attr.name);
    for (uint32_t i = 0; i < attr.n_dims; ++i) {
        std::printf("%s%u", i == 0 ? "[" : ", ", attr.dims[i]);
    }
    std::printf("] fmt=%s type=%s elems=%u size=%u\n",
                get_format_string(attr.fmt), get_type_string(attr.type),
                attr.n_elems, attr.size);
}

int set_core_mask(rknn_context context, int core_count) {
    if (core_count == 1) {
        return rknn_set_core_mask(context, RKNN_NPU_CORE_0);
    }
    if (core_count == 2) {
        return rknn_set_core_mask(context, RKNN_NPU_CORE_0_1);
    }
    if (core_count == 3) {
        return rknn_set_core_mask(context, RKNN_NPU_CORE_0_1_2);
    }
    return rknn_set_core_mask(context, RKNN_NPU_CORE_AUTO);
}

size_t product_except_batch(const rknn_tensor_attr& attr) {
    if (attr.n_dims < 2) return 0;
    size_t result = 1;
    for (uint32_t i = 1; i < attr.n_dims; ++i) {
        result *= attr.dims[i];
    }
    return result;
}

}  // namespace

int image_encoder_init(const char* model_path, int core_count, ImageEncoder* encoder) {
    if (model_path == nullptr || encoder == nullptr) return -1;
    *encoder = ImageEncoder{};

    int ret = rknn_init(&encoder->context, const_cast<char*>(model_path), 0, 0, nullptr);
    if (ret != RKNN_SUCC) {
        std::fprintf(stderr, "rknn_init failed: %d\n", ret);
        return ret;
    }
    ret = set_core_mask(encoder->context, core_count);
    if (ret != RKNN_SUCC) {
        std::fprintf(stderr, "rknn_set_core_mask failed: %d\n", ret);
        image_encoder_release(encoder);
        return ret;
    }
    ret = rknn_query(encoder->context, RKNN_QUERY_IN_OUT_NUM,
                     &encoder->io_num, sizeof(encoder->io_num));
    if (ret != RKNN_SUCC || encoder->io_num.n_input != 1 || encoder->io_num.n_output != 1) {
        std::fprintf(stderr, "expected one input and one output, got %u/%u (ret=%d)\n",
                     encoder->io_num.n_input, encoder->io_num.n_output, ret);
        image_encoder_release(encoder);
        return ret == RKNN_SUCC ? -1 : ret;
    }

    encoder->input.index = 0;
    encoder->output.index = 0;
    ret = rknn_query(encoder->context, RKNN_QUERY_INPUT_ATTR,
                     &encoder->input, sizeof(encoder->input));
    if (ret == RKNN_SUCC) {
        ret = rknn_query(encoder->context, RKNN_QUERY_OUTPUT_ATTR,
                         &encoder->output, sizeof(encoder->output));
    }
    if (ret != RKNN_SUCC) {
        std::fprintf(stderr, "rknn_query tensor attributes failed: %d\n", ret);
        image_encoder_release(encoder);
        return ret;
    }
    print_attr("vision input", encoder->input);
    print_attr("vision output", encoder->output);

    if (encoder->input.n_dims != 4 || encoder->input.dims[0] != 1 ||
        encoder->input.dims[1] == 0 || encoder->input.dims[2] == 0 || encoder->input.dims[3] == 0) {
        std::fprintf(stderr, "unsupported vision input shape\n");
        image_encoder_release(encoder);
        return -1;
    }
    if (encoder->input.fmt == RKNN_TENSOR_NCHW) {
        encoder->channels = static_cast<int>(encoder->input.dims[1]);
        encoder->height = static_cast<int>(encoder->input.dims[2]);
        encoder->width = static_cast<int>(encoder->input.dims[3]);
    } else if (encoder->input.fmt == RKNN_TENSOR_NHWC) {
        encoder->height = static_cast<int>(encoder->input.dims[1]);
        encoder->width = static_cast<int>(encoder->input.dims[2]);
        encoder->channels = static_cast<int>(encoder->input.dims[3]);
    } else {
        std::fprintf(stderr, "unsupported vision input format: %d\n", encoder->input.fmt);
        image_encoder_release(encoder);
        return -1;
    }
    if (encoder->channels != 3 || encoder->output.n_dims < 2) {
        std::fprintf(stderr, "expected RGB input and [tokens, embedding] output\n");
        image_encoder_release(encoder);
        return -1;
    }
    encoder->embedding_size = encoder->output.dims[encoder->output.n_dims - 1];
    const size_t output_values = product_except_batch(encoder->output);
    if (output_values == 0 || encoder->embedding_size != 1536 ||
        output_values % encoder->embedding_size != 0) {
        std::fprintf(stderr, "unexpected output token/embedding shape\n");
        image_encoder_release(encoder);
        return -1;
    }
    // The queried shape is [1, tokens, embedding]. Derive tokens after validating
    // the final embedding dimension instead of treating the flattened element
    // count as the token count.
    encoder->image_tokens = output_values / encoder->embedding_size;
    std::printf("vision contract: %dx%d RGB -> %zu tokens x %zu embedding\n",
                encoder->width, encoder->height, encoder->image_tokens, encoder->embedding_size);
    return RKNN_SUCC;
}

int image_encoder_run(ImageEncoder* encoder, const std::uint8_t* rgb_hwc,
                      float* embedding, size_t embedding_floats) {
    if (encoder == nullptr || encoder->context == 0 || rgb_hwc == nullptr || embedding == nullptr ||
        embedding_floats < encoder->image_tokens * encoder->embedding_size) {
        return -1;
    }
    const size_t pixels = static_cast<size_t>(encoder->width) * encoder->height;
    const size_t input_values = pixels * encoder->channels;
    // The RKNN graph was built with mean=0/std=255. Supplying UINT8 NHWC lets
    // rknn_inputs_set perform the layout conversion and compiled preprocessing,
    // even though RKNN_QUERY_INPUT_ATTR reports the native graph tensor as FP16.
    rknn_input input{};
    input.index = 0;
    input.type = RKNN_TENSOR_UINT8;
    input.fmt = RKNN_TENSOR_NHWC;
    input.size = static_cast<uint32_t>(input_values);
    input.buf = const_cast<std::uint8_t*>(rgb_hwc);
    int ret = rknn_inputs_set(encoder->context, 1, &input);
    if (ret != RKNN_SUCC) return ret;
    ret = rknn_run(encoder->context, nullptr);
    if (ret != RKNN_SUCC) return ret;

    rknn_output output{};
    output.want_float = 1;
    ret = rknn_outputs_get(encoder->context, 1, &output, nullptr);
    if (ret == RKNN_SUCC) {
        const size_t expected = encoder->image_tokens * encoder->embedding_size * sizeof(float);
        if (output.buf == nullptr || output.size < expected) {
            ret = -1;
        } else {
            std::memcpy(embedding, output.buf, expected);
            for (size_t i = 0; i < expected / sizeof(float); ++i) {
                if (!std::isfinite(embedding[i])) { ret = -1; break; }
            }
        }
    }
    const int release_ret = rknn_outputs_release(encoder->context, 1, &output);
    return ret == RKNN_SUCC ? release_ret : ret;
}

void image_encoder_release(ImageEncoder* encoder) {
    if (encoder == nullptr) return;
    if (encoder->context != 0) rknn_destroy(encoder->context);
    *encoder = ImageEncoder{};
}
