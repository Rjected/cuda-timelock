#include <gtest/gtest.h>
#include <sstream>
#include <string>
#include <chrono>
#include <time.h>
#include "../src/powm_odd.cu"
int main() {
  typedef powm_params_t<8, 1024, 5> params;
  std::stringstream ss;
  time_t now;
  now = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
  ss << std::put_time(localtime(&now), "%y-%m-%d %h:%m:%s");
  printf("[%s]: Running normal test\n", ss.str());
  run_test<params>(10000);
  typedef powm_params_t<16, 4096, 5> params_puzzle;
  now = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
  ss << std::put_time(localtime(&now), "%y-%m-%d %h:%m:%s");
  printf("[%s]: Running puzzle test\n", ss.str());
  run_puzzle_test<params_puzzle>(10000, 1024);
  return 0;
}
