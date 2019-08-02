#include "powm_odd.cu"
#include "cudasquare.h"

/**
 *
 * launch_repeat_square sets z = x**(2**y) mod |m| (i.e. the sign of m is ignored), and returns z.
 * If y < 0, the result is 1; if y == 0 the result is x, if m == nil or m == 0, z = x**(2**y).
 * See Knuth, volume 2, section 4.6.3.
 *
 * @param variable arrays a, t, and N for the above calculation, and len, denoting the length of the arras for each a, t, N. This assumes they are all the same size, otherwise segfault.
 *
 * @return uhh TODO !!
 *
 * @throws idk TODO
 *
 * @exceptsafe TODO
 */
template<class params>
__host__ void launch_repeat_square(mpz_t* result, const uint32_t* a, const uint32_t* t, const mpz_t* N, const uint64_t len) {

    // Since we want this API to be globally accessible, we're going to initialize as mpz
    // and convert to cgbn_env_t<cgbn_context_t<params::TPI, params>, params::BITS>::cgbn_t
    for (int i = 0; i < len; ++i) {
        // init the result
        mpz_init(result[i]);
    }

    // TODO write a kernel like powm that calls powm but in groups according to another parameter.
    // The reason for this is because we might have some exponents for a (in a^2^t) that are
    // 2^t, which I'm guessing the in memory representation will be a single 1 bit, and t zero bits.
    // This was an issue with the library in Go, since the t parameter made the exponent gigabytes
    // in size.
    // So we need to write something that we can call, after generating instances, that will be very
    // similar to powm but will be grouped. Say the parameter is 1024, every exponent will be 2^1024
    // but we will run it a bunch of times.

    return;
}

int main() {
  typedef powm_params_t<8, 1024, 5> params;

  run_test<params>(10000);
  return 0;
}
