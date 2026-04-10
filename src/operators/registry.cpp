#include "edge_ml/operators/registry.h"
#include "edge_ml/operators/activations.h"
#include "edge_ml/operators/dense.h"
#include "edge_ml/operators/conv2d.h"
#include "edge_ml/operators/pooling.h"
#include "edge_ml/operators/batchnorm.h"
#include <stdexcept>

namespace edge_ml {
namespace ops {

OpRegistry& OpRegistry::instance() {
    static OpRegistry reg;
    return reg;
}

void OpRegistry::register_op(const std::string& name, OpFunction fn) {
    registry_[name] = std::move(fn);
}

Tensor OpRegistry::dispatch(const std::string& name, const OpContext& ctx) const {
    auto it = registry_.find(name);
    if (it == registry_.end()) {
        throw std::runtime_error("OpRegistry: unknown operator '" + name + "'");
    }
    return it->second(ctx);
}

bool OpRegistry::has_op(const std::string& name) const {
    return registry_.find(name) != registry_.end();
}

std::vector<std::string> OpRegistry::list_ops() const {
    std::vector<std::string> names;
    names.reserve(registry_.size());
    for (const auto& pair : registry_) {
        names.push_back(pair.first);
    }
    return names;
}

static int get_int_attr(const OpContext& ctx, const std::string& key, int default_val) {
    auto it = ctx.int_attrs.find(key);
    return (it != ctx.int_attrs.end()) ? it->second : default_val;
}

void register_all_ops() {
    auto& reg = OpRegistry::instance();

    reg.register_op("Relu", [](const OpContext& ctx) {
        return relu(ctx.inputs[0]);
    });

    reg.register_op("Sigmoid", [](const OpContext& ctx) {
        return sigmoid(ctx.inputs[0]);
    });

    reg.register_op("Silu", [](const OpContext& ctx) {
        return silu(ctx.inputs[0]);
    });

    reg.register_op("Softmax", [](const OpContext& ctx) {
        int axis = get_int_attr(ctx, "axis", -1);
        return softmax(ctx.inputs[0], axis);
    });

    reg.register_op("MatMul", [](const OpContext& ctx) {
        return dense(ctx.inputs[0], ctx.inputs[1], ctx.inputs[2]);
    });

    reg.register_op("Conv", [](const OpContext& ctx) {
        int sh = get_int_attr(ctx, "stride_h", 1);
        int sw = get_int_attr(ctx, "stride_w", 1);
        int ph = get_int_attr(ctx, "pad_h", 0);
        int pw = get_int_attr(ctx, "pad_w", 0);
        return conv2d_im2col(ctx.inputs[0], ctx.inputs[1], ctx.inputs[2], sh, sw, ph, pw);
    });

    reg.register_op("MaxPool", [](const OpContext& ctx) {
        int kh = get_int_attr(ctx, "kernel_h", 2);
        int kw = get_int_attr(ctx, "kernel_w", 2);
        int sh = get_int_attr(ctx, "stride_h", 2);
        int sw = get_int_attr(ctx, "stride_w", 2);
        int ph = get_int_attr(ctx, "pad_h", 0);
        int pw = get_int_attr(ctx, "pad_w", 0);
        return maxpool2d(ctx.inputs[0], kh, kw, sh, sw, ph, pw);
    });

    reg.register_op("BatchNormalization", [](const OpContext& ctx) {
        float eps = 1e-5f;
        auto it = ctx.float_attrs.find("epsilon");
        if (it != ctx.float_attrs.end()) eps = it->second;
        return batchnorm(ctx.inputs[0], ctx.inputs[1], ctx.inputs[2],
                         ctx.inputs[3], ctx.inputs[4], eps);
    });
}

} // namespace ops
} // namespace edge_ml
