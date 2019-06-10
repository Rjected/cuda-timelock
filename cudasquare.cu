#include<gmp.h>
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
void expSquare(mpz_t *result, mpz_t *a, const mpz_t t, const mpz_t *N) {
  int tid = blockIdx.x;
  mpz_t two;
  mpz_init(two);
  mpz_set_ui(two, 2);
  mpz_t i;
  for (mpz_init(i); mpz_sgn(i) != 0; mpz_sub_ui(i, i, 1)) {
    mpz_powm(a[tid], a[tid], two, N[tid]);
  }
  mpz_set(result[tid], a[tid]);
}
