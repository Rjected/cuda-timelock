#include "powm_odd.cu"

/**
 *
 * expSquare sets z = x**(2**y) mod |m| (i.e. the sign of m is ignored), and returns z.
 * If y < 0, the result is 1; if y == 0 the result is x, if m == nil or m == 0, z = x**(2**y).
 * See Knuth, volume 2, section 4.6.3.
 *
 * @param vectors of mpz_t variables a, t, and N for the above calculation.
 *
 * @return uhh TODO !!
 *
 * @throws idk TODO
 *
 * @exceptsafe TODO
 */

// If you put this __global__ back then it breaks because GMP is host code, not device code.
//__global__


/* template<class params> */
/* __global__ void kernel_repeat_square(mpz_t *result, mpz_t *a, const mpz_t t, const mpz_t *N) { */
/*     return; */
/* } */

int main() {
  typedef powm_params_t<8, 1024, 5> params;

  run_test<params>(10000);
  return 0;
}
