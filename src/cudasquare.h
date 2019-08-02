#pragma once
#include <gmp.h>

template<class params>
__host__ void launch_repeat_square(mpz_t* result, const uint64_t* a, const uint64_t* t, const mpz_t* N, const uint64_t len);

template<class params>
__global__ void kernel_repeat_square();
