#pragma once

#include "edge_ml/tensor.h"
#include <functional>
#include <string>
#include <unordered_map>
#include <vector>

namespace edge_ml {
namespace ops {

struct OpContext {
    std::vector<Tensor> inputs;
    std::unordered_map<std::string, float> float_attrs;
    std::unordered_map<std::string, int> int_attrs;
    std::unordered_map<std::string, std::vector<int>> int_list_attrs;
};

using OpFunction = std::function<Tensor(const OpContext&)>;

class OpRegistry {
public:
    static OpRegistry& instance();

    void register_op(const std::string& name, OpFunction fn);
    Tensor dispatch(const std::string& name, const OpContext& ctx) const;
    bool has_op(const std::string& name) const;
    std::vector<std::string> list_ops() const;

private:
    OpRegistry() = default;
    std::unordered_map<std::string, OpFunction> registry_;
};

void register_all_ops();

} // namespace ops
} // namespace edge_ml
