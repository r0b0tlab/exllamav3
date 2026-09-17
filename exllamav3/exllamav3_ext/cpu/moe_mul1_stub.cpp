#include "moe_mul1.h"
#include <stdexcept>

static void unsupported()
{
    throw std::runtime_error("CPU MoE offload is not built on this architecture");
}

int64_t exl3_moe_cpu_make_layer(
    const std::vector<at::Tensor>&,
    const std::vector<at::Tensor>&,
    const std::vector<at::Tensor>&,
    const std::vector<at::Tensor>&,
    const std::vector<at::Tensor>&,
    const std::vector<at::Tensor>&,
    const std::vector<at::Tensor>&,
    const std::vector<at::Tensor>&,
    const std::vector<at::Tensor>&,
    const std::vector<at::Tensor>&,
    const std::vector<at::Tensor>&,
    const std::vector<at::Tensor>&,
    int64_t, double, int64_t)
{
    unsupported();
    return -1;
}

void exl3_moe_cpu_free_layer(int64_t) {}

void exl3_moe_cpu_forward(
    int64_t,
    const at::Tensor&,
    const at::Tensor&,
    const at::Tensor&,
    at::Tensor&,
    int64_t)
{
    unsupported();
}

void exl3_moe_cpu_forward_raw(
    int64_t,
    const at::Half*,
    const int32_t*,
    const at::Half*,
    float*,
    int, int, int)
{
    unsupported();
}

void exl3_moe_cpu_stage_experts(
    int64_t,
    const uint32_t*,
    int,
    uint8_t*,
    int)
{
    unsupported();
}

void exl3_moe_cpu_set_prof(bool) {}
int64_t exl3_moe_cpu_pool_stress(int, int, int, int)
{
    unsupported();
    return 0;
}

bool exl3_moe_cpu_has_avx2() { return false; }
bool exl3_moe_cpu_has_avx512_bw() { return false; }
bool exl3_moe_cpu_has_avx512_vnni() { return false; }
bool exl3_moe_cpu_has_avx512_vbmi() { return false; }
