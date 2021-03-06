# cuda-timelock
[![License](http://img.shields.io/badge/License-MIT-brightgreen.svg)](./LICENSE)
[![Build Status](http://spice.dancline.net:8080/buildStatus/icon?job=cuda-timelock%2Fmaster)](http://spice.dancline.net:8080/job/cuda-timelock/job/master/)
<!-- [![Build Status](http://spice.dancline.net:8080/job/cuda-timelock/job/master/badge/icon)](http://spice.dancline.net:8080/job/cuda-timelock/job/master/) -->

cuda-timelock is code to solve timelock puzzles in parallel, using a GPU.
While a GPU is not the best in terms of latency for each individual multiple precision operation, the goal is to increase throughput on traditional hardware.

## Requirements
 * [GMP](https://gmplib.org)
 * [CGBN](https://github.com/NVlabs/CGBN)
 * [Google Test Framework](https://github.com/google/googletest)
 * A graphics card
