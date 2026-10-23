// Python 扩展入口：导入即触发 TORCH_LIBRARY 静态初始化（注册 schema 与实现）
#include <torch/extension.h>

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "cann_bench direct-launch operator extension";
}
