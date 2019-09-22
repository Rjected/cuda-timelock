#include <gtest/gtest.h>
#include "../src/powm_odd.cu"
int main() {
  typedef powm_params_t<8, 1024, 5> params;
  run_test<params>(128);
  typedef powm_params_t<32, 32768, 16> params_puzzle;
  /* run_simple_test<params_puzzle>(2, 15000); */
  /* run_simple_test<params_puzzle>(2, 15000); */
  /* run_simple_test<params_puzzle>(2, 15000); */
  /* run_simple_test<params_puzzle>(2, 15000); */

  printf("\n ==================== SIMPLE TESTS ==================== \n");

  /* run_simple_test<params_puzzle>(2, 2); */
  /* run_simple_test<params_puzzle>(2, 4); */
  /* run_simple_test<params_puzzle>(2, 8); */
  /* run_simple_test<params_puzzle>(2, 16); */
  /* run_simple_test<params_puzzle>(2, 32); */
  /* run_simple_test<params_puzzle>(2, 64); */
  /* run_simple_test<params_puzzle>(2, 128); */
  /* run_simple_test<params_puzzle>(2, 256); */
  /* run_simple_test<params_puzzle>(2, 512); */
  /* run_simple_test<params_puzzle>(2, 1024); */
  run_simple_test<params_puzzle>(2, 2048);
  run_simple_test<params_puzzle>(2, 4096);
  run_simple_test<params_puzzle>(2, 8192);
  /* run_simple_test<params_puzzle>(2, 16384); */
  /* run_simple_test<params_puzzle>(2, 32768); */
  /* run_simple_test<params_puzzle>(2, 65536); */

  printf("\n ==================== THROUGHPUT TESTS 1 ==================== \n");

  run_puzzle_test<params_puzzle>(1, 16384);
  run_puzzle_test<params_puzzle>(10, 16384);
  run_puzzle_test<params_puzzle>(100, 16384);
  run_puzzle_test<params_puzzle>(1000, 16384);
  run_puzzle_test<params_puzzle>(10000, 16384);

  printf("\n ==================== THROUGHPUT TESTS 2 ==================== \n");

  run_puzzle_test<params_puzzle>(1, 4096);
  run_puzzle_test<params_puzzle>(10, 4096);
  run_puzzle_test<params_puzzle>(100, 4096);
  run_puzzle_test<params_puzzle>(1000, 4096);
  run_puzzle_test<params_puzzle>(10000, 4096);
  run_puzzle_test<params_puzzle>(100000, 4096);

  printf("\n ==================== DONE ==================== \n");
  return 0;
}
