# cuda-timelock

cuda-timelock is code to solve timelock puzzles in parallel, using a GPU.
While a GPU is not the best in terms of latency for each individual multiple precision operation, the goal is to increase throughput on traditional hardware.

## Requirements
 * [GMP](https://gmplib.org)
 * [CGBN](https://github.com/NVlabs/CGBN)
 * A graphics card
