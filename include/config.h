#pragma once

#ifndef ML_USE_SIMD
    #define ML_USE_SIMD 1 
#endif

// Check if compiler/CPU supports AVX2
#ifdef ML_USE_SIMD
    #ifdef __AVX2__
        #define ML_HAS_AVX2 1
    #else
        #define ML_HAS_AVX2 0
        #pragma message("SIMD requested but AVX2 not available. Falling back to scalar.")
    #endif
#else
    #define ML_HAS_AVX2 0
#endif