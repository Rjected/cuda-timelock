#pragma once
#include <gmp.h>

__host__ void launch_repeat_square(mpz_t*, const uint64_t*, const uint64_t*, const mpz_t*, const uint64_t);

template<class params>
__global__ void kernel_repeat_square();
