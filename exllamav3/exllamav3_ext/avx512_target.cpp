#include "avx512_target.h"

bool is_avx512_supported()
{
    static bool avx512_check = false;
    static bool avx512_supported = false;
    if (avx512_check) return avx512_supported;

#if defined(__x86_64__) || defined(__i386__)
#ifdef __linux__
    avx512_supported = __builtin_cpu_supports("avx512f") && __builtin_cpu_supports("avx512bw");
#else
    int cpuInfo[4];
    __cpuid(cpuInfo, 0);
    if (cpuInfo[0] < 7)
    {
        avx512_check = true;
        return false;
    }
    __cpuidex(cpuInfo, 7, 0);
    bool avx512f = (cpuInfo[1] & (1 << 16)) != 0;
    bool avx512bw = (cpuInfo[1] & (1 << 30)) != 0;
    avx512_supported = avx512f && avx512bw;
#endif
#else
    avx512_supported = false;
#endif

    avx512_check = true;
    return avx512_supported;
}
