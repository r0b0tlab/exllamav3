#include "avx2_target.h"

bool is_avx2_supported()
{
    static bool avx2_check = false;
    static bool avx2_supported = false;
    if (avx2_check) return avx2_supported;
#if defined(__x86_64__) || defined(__i386__)
    #ifdef __linux__
        avx2_supported = __builtin_cpu_supports("avx2");
    #else
        int cpuInfo[4];
        __cpuidex(cpuInfo, 7, 0);
        avx2_supported = (cpuInfo[1] & (1 << 5)) != 0;
    #endif
#else
    avx2_supported = false;
#endif
    avx2_check = true;
    return avx2_supported;
}
bool is_f16c_supported()
{
    static bool f16c_check = false;
    static bool f16c_supported = false;
    if (f16c_check) return f16c_supported;
#if defined(__x86_64__) || defined(__i386__)
    #ifdef __linux__
        f16c_supported = __builtin_cpu_supports("f16c");
    #else
        int cpuInfo[4];
        __cpuidex(cpuInfo, 1, 0);
        f16c_supported = (cpuInfo[2] & (1 << 29)) != 0;
    #endif
#else
    f16c_supported = false;
#endif
    f16c_check = true;
    return f16c_supported;
}
