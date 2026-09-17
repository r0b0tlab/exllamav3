#include "all_reduce_cpu_avx2.h"
#include <stdexcept>

void enable_fast_fp() {}
void enable_fast_fp_avx2() {}

void perform_cpu_reduce(
    PGContext*,
    size_t,
    uint32_t,
    uint32_t,
    uint8_t*,
    size_t)
{
    throw std::runtime_error("CPU all-reduce is not built on this architecture");
}

void perform_cpu_reduce_avx2(
    PGContext*,
    size_t,
    uint32_t,
    uint32_t,
    uint8_t*,
    size_t)
{
    throw std::runtime_error("CPU all-reduce AVX2 is not built on this architecture");
}

void cpu_reduce_parallel(
    void (*)(uint16_t*, const uint16_t*, const uint16_t*, size_t),
    void (*)(uint16_t*, const uint16_t*, size_t),
    uint16_t*,
    const uint16_t*,
    const uint16_t*,
    size_t,
    int)
{
    throw std::runtime_error("cpu_reduce_parallel is not built on this architecture");
}
