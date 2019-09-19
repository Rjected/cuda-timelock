/***
Copyright (c) 2018-2019, NVIDIA CORPORATION.  All rights reserved.
Permission is hereby granted, free of charge, to any person obtaining a
copy of this software and associated documentation files (the "Software"),
to deal in the Software without restriction, including without limitation
the rights to use, copy, modify, merge, publish, distribute, sublicense,
and/or sell copies of the Software, and to permit persons to whom the
Software is furnished to do so, subject to the following conditions:
The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
IN THE SOFTWARE.
***/

#include <time.h>
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <cuda.h>
#include <gmp.h>
#include <inttypes.h>
#include "../include/cgbn/cgbn.h"
#include "../include/insecure_rsa/rsa.h"
#include "support.h"

// For this example, there are quite a few template parameters that are used to generate the actual code.
// In order to simplify passing many parameters, we use the same approach as the CGBN library, which is to
// create a container class with static constants and then pass the class.

// The CGBN context uses the following three parameters:
//   TBP             - threads per block (zero means to use the blockDim.x)
//   MAX_ROTATION    - must be small power of 2, imperically, 4 works well
//   SHM_LIMIT       - number of bytes of dynamic shared memory available to the kernel
//   CONSTANT_TIME   - require constant time algorithms (currently, constant time algorithms are not available)

// Locally it will also be helpful to have several parameters:
//   TPI             - threads per instance
//   BITS            - number of bits per instance
//   WINDOW_BITS     - number of bits to use for the windowed exponentiation

template<uint32_t tpi, uint32_t bits, uint32_t window_bits>
class powm_params_t {
  public:
  // parameters used by the CGBN context
  static const uint32_t TPB=0;                     // get TPB from blockDim.x
  static const uint32_t MAX_ROTATION=4;            // good default value
  static const uint32_t SHM_LIMIT=0;               // no shared mem available
  static const bool     CONSTANT_TIME=false;       // constant time implementations aren't available yet

  // parameters used locally in the application
  static const uint32_t TPI=tpi;                   // threads per instance
  static const uint32_t BITS=bits;                 // instance size
  static const uint32_t WINDOW_BITS=window_bits;   // window size
};

template<class params>
class powm_odd_t {
  public:
  static const uint32_t window_bits=params::WINDOW_BITS;  // used a lot, give it an instance variable

  // define the instance structure
  typedef struct {
    cgbn_mem_t<params::BITS> x;
    cgbn_mem_t<params::BITS> power;
    cgbn_mem_t<params::BITS> modulus;
    cgbn_mem_t<params::BITS> result;
  } instance_t;

  typedef cgbn_context_t<params::TPI, params>   context_t;
  typedef cgbn_env_t<context_t, params::BITS>   env_t;
  typedef typename env_t::cgbn_t                bn_t;
  typedef typename env_t::cgbn_local_t          bn_local_t;

  context_t _context;
  env_t     _env;
  int32_t   _instance;

  __device__ __forceinline__ powm_odd_t(cgbn_monitor_t monitor, cgbn_error_report_t *report, int32_t instance) : _context(monitor, report, (uint32_t)instance), _env(_context), _instance(instance) {
  }

  __device__ __forceinline__ void fixed_window_powm_odd(bn_t &result, const bn_t &x, const bn_t &power, const bn_t &modulus) {
    bn_t       t;
    bn_local_t window[1<<window_bits];
    int32_t    index, position, offset;
    uint32_t   np0;

    // conmpute x^power mod modulus, using the fixed window algorithm
    // requires:  x<modulus,  modulus is odd

    // compute x^0 (in Montgomery space, this is just 2^BITS - modulus)
    cgbn_negate(_env, t, modulus);
    cgbn_store(_env, window+0, t);

    // convert x into Montgomery space, store into window table
    np0=cgbn_bn2mont(_env, result, x, modulus);
    cgbn_store(_env, window+1, result);
    cgbn_set(_env, t, result);

    // compute x^2, x^3, ... x^(2^window_bits-1), store into window table
    #pragma nounroll
    for(index=2;index<(1<<window_bits);index++) {
      cgbn_mont_mul(_env, result, result, t, modulus, np0);
      cgbn_store(_env, window+index, result);
    }

    // find leading high bit
    position=params::BITS - cgbn_clz(_env, power);

    // break the exponent into chunks, each window_bits in length
    // load the most significant non-zero exponent chunk
    offset=position % window_bits;
    if(offset==0)
      position=position-window_bits;
    else
      position=position-offset;
    index=cgbn_extract_bits_ui32(_env, power, position, window_bits);
    cgbn_load(_env, result, window+index);

    // process the remaining exponent chunks
    while(position>0) {
      // square the result window_bits times
      #pragma nounroll
      for(int sqr_count=0;sqr_count<window_bits;sqr_count++)
        cgbn_mont_sqr(_env, result, result, modulus, np0);

      // multiply by next exponent chunk
      position=position-window_bits;
      index=cgbn_extract_bits_ui32(_env, power, position, window_bits);
      cgbn_load(_env, t, window+index);
      cgbn_mont_mul(_env, result, result, t, modulus, np0);
    }

    // we've processed the exponent now, convert back to normal space
    cgbn_mont2bn(_env, result, result, modulus, np0);
  }

  __device__ __forceinline__ void sliding_window_powm_odd(bn_t &result, const bn_t &x, const bn_t &power, const bn_t &modulus) {
    bn_t         t, starts;
    int32_t      index, position, leading;
    uint32_t     mont_inv;
    bn_local_t   odd_powers[1<<window_bits-1];

    // conmpute x^power mod modulus, using Constant Length Non-Zero windows (CLNZ).
    // requires:  x<modulus,  modulus is odd

    // find the leading one in the power
    leading=params::BITS-1-cgbn_clz(_env, power);
    if(leading>=0) {
      // convert x into Montgomery space, store in the odd powers table
      mont_inv=cgbn_bn2mont(_env, result, x, modulus);

      // compute t=x^2 mod modulus
      cgbn_mont_sqr(_env, t, result, modulus, mont_inv);

      // compute odd powers window table: x^1, x^3, x^5, ...
      cgbn_store(_env, odd_powers, result);
      #pragma nounroll
      for(index=1;index<(1<<window_bits-1);index++) {
        cgbn_mont_mul(_env, result, result, t, modulus, mont_inv);
        cgbn_store(_env, odd_powers+index, result);
      }

      // starts contains an array of bits indicating the start of a window
      cgbn_set_ui32(_env, starts, 0);

      // organize p as a sequence of odd window indexes
      position=0;
      while(true) {
        if(cgbn_extract_bits_ui32(_env, power, position, 1)==0)
          position++;
        else {
          cgbn_insert_bits_ui32(_env, starts, starts, position, 1, 1);
          if(position+window_bits>leading)
            break;
          position=position+window_bits;
        }
      }

      // load first window.  Note, since the window index must be odd, we have to
      // divide it by two before indexing the window table.  Instead, we just don't
      // load the index LSB from power
      index=cgbn_extract_bits_ui32(_env, power, position+1, window_bits-1);
      cgbn_load(_env, result, odd_powers+index);
      position--;

      // Process remaining windows
      while(position>=0) {
        cgbn_mont_sqr(_env, result, result, modulus, mont_inv);
        if(cgbn_extract_bits_ui32(_env, starts, position, 1)==1) {
          // found a window, load the index
          index=cgbn_extract_bits_ui32(_env, power, position+1, window_bits-1);
          cgbn_load(_env, t, odd_powers+index);
          cgbn_mont_mul(_env, result, result, t, modulus, mont_inv);
        }
        position--;
      }

      // convert result from Montgomery space
      cgbn_mont2bn(_env, result, result, modulus, mont_inv);
    }
    else {
      // p=0, thus x^p mod modulus=1
      cgbn_set_ui32(_env, result, 1);
    }
  }

  // this is assuming power will equal 2^t. then we would have a lot of memory issues.
  __device__ __forceinline__ void grouped_fixed_window_powm_odd(bn_t &result, const bn_t &x, const uint32_t t, const bn_t &modulus, const uint32_t grouping) {

    // First we calculate the exponent, in this case 2^grouping.
    // Then we divide to get an index, and take the modulus to get the last exponent.
    bn_t primary_exponent;
    // this is 1 because 1 = 2^0, we shift grouping times so the result will be
    // 2^0 * 2^grouping = 2^(0 + grouping) = 2^grouping
    cgbn_set_ui32(_env, primary_exponent, 1);
    cgbn_shift_left(_env, primary_exponent, primary_exponent, grouping);

    bn_t two;
    cgbn_set_ui32(_env, two, 2);
    bn_t tee;
    cgbn_set_ui32(_env, tee, t);
    bn_t expon;
    fixed_window_powm_odd(expon, two, tee, modulus);
    fixed_window_powm_odd(result, x, expon, modulus);


    /* // limit = t / grouping */
    /* // we don't care about the result being stored in a bn_t */
    /* uint32_t limit = t / grouping; */

    /* // final_grouping = t % grouping */
    /* // we don't care about the result being stored in a bn_t */
    /* const uint32_t final_grouping = t % grouping; */

    /* bn_t one; */
    /* cgbn_set_ui32(_env, one, 1); */
    /* bn_t zero; */
    /* cgbn_set_ui32(_env, zero, 0); */

    /* // Now we take 2^final_grouping for the final exponent */
    /* bn_t final_exponent; */
    /* cgbn_set_ui32(_env, final_exponent, 1); */
    /* cgbn_shift_left(_env, final_exponent, final_exponent, final_grouping); */

    /* // x is not constant so we create a mutable one */
    /* bn_t mut_x; */
    /* cgbn_set(_env, mut_x, x); */
    /* // now we do this a bunch of times */
    /* while (limit > 0) { */
    /*   // x = x ^ primary_exponent (mod N) */
    /*   fixed_window_powm_odd(mut_x, mut_x, primary_exponent, modulus); */
    /*   limit--; */
    /* } */

    /* // and finally, the last will store the result */
    /* fixed_window_powm_odd(result, mut_x, final_exponent, modulus); */

    return;
  }

  // this is assuming power will equal 2^t. if we actually calculated 2^t then we would have a lot of memory issues.
  __device__ __forceinline__ void grouped_sliding_window_powm_odd(bn_t &result, const bn_t &x, const bn_t &t, const bn_t &modulus, const uint32_t grouping) {

    // First we calculate the exponent, in this case 2^grouping.
    // Then we divide to get an index, and take the modulus to get the last exponent.
    bn_t primary_exponent;
    // this is 1 because 1 = 2^0, we shift grouping times so the result will be
    // 2^0 * 2^grouping = 2^(0 + grouping) = 2^grouping
    cgbn_set_ui32(_env, primary_exponent, 1);
    cgbn_shift_left(_env, primary_exponent, primary_exponent, grouping);

    // limit = t / grouping
    // we don't care about the result being stored in a bn_t
    bn_t limit;
    cgbn_div_ui32(_env, limit, t, grouping);

    // final_grouping = t % grouping
    // we don't care about the result being stored in a bn_t
    const uint32_t final_grouping = cgbn_rem_ui32(_env, t, grouping);

    bn_t one;
    cgbn_set_ui32(_env, one, 1);
    bn_t zero;
    cgbn_set_ui32(_env, zero, 0);

    // Now we take 2^final_grouping for the final exponent
    bn_t final_exponent;
    cgbn_set_ui32(_env, final_exponent, 1);
    cgbn_shift_left(_env, final_exponent, final_exponent, final_grouping);

    // x is not constant so we create a mutable one
    bn_t mut_x;
    cgbn_set(_env, mut_x, x);
    // now we do this a bunch of times
    while (!cgbn_equals(_env, limit, zero)) {
      // x = x ^ primary_exponent (mod N)
      sliding_window_powm_odd(mut_x, mut_x, primary_exponent, modulus);
      cgbn_sub(_env, limit, limit, one);
    }

    // and finally, the last will store the result
    sliding_window_powm_odd(result, mut_x, final_exponent, modulus);

    return;
  }

  __host__ static instance_t *generate_instances(uint32_t count) {
    instance_t *instances=(instance_t *)malloc(sizeof(instance_t)*count);
    int         index;

    for(index=0;index<count;index++) {
      random_words(instances[index].x._limbs, params::BITS/32);
      random_words(instances[index].power._limbs, params::BITS/32);
      random_words(instances[index].modulus._limbs, params::BITS/32);

      // ensure modulus is odd
      instances[index].modulus._limbs[0] |= 1;

      // ensure modulus is greater than
      if(compare_words(instances[index].x._limbs, instances[index].modulus._limbs, params::BITS/32)>0) {
        swap_words(instances[index].x._limbs, instances[index].modulus._limbs, params::BITS/32);

        // modulus might now be even, ensure it's odd
        instances[index].modulus._limbs[0] |= 1;
      }
      else if(compare_words(instances[index].x._limbs, instances[index].modulus._limbs, params::BITS/32)==0) {
        // since modulus is odd and modulus = x, we can just subtract 1 from x
        instances[index].x._limbs[0] -= 1;
      }
    }
    return instances;
  }

  // this generates timelock puzzle instances, for example 2^2^4444 or 5^2^12345 or 7^2^3333
  // all mod the same N. This is the case because when testing, we're generating the instances
  // probably knowing the factorization of N.
  // This way we can test that a^2^t (mod N), calculated from cgbn, = a^(2^t (mod phi(N)) (mod N),
  // calculated from gmp.
  // Either that, or we just let GMP do all of the work
  __host__ static instance_t *generate_puzzle_instances(uint32_t count, const uint32_t t, const mpz_t N) {
    instance_t *instances=(instance_t *)malloc(sizeof(instance_t)*count);
    int         index;

    mpz_t two; // 2
    mpz_init(two);
    mpz_set_ui(two, 2);

    mpz_t five; // 5
    mpz_init(five);
    mpz_set_ui(five, 5);

    mpz_t seven; // 7
    mpz_init(seven);
    mpz_set_ui(seven, 7);

    mpz_t thirteen; // 13
    mpz_init(thirteen);
    mpz_set_ui(thirteen, 13);

    mpz_t maxval;
    mpz_init(maxval);

    for(index=0;index<count;index++) {
        // create 2^whatever
        mpz_t e;
        mpz_init(e);

        mpz_pow_ui(e, two, t);
        // just alternate between our bases
        switch (index % 4) {
          case 0:
            // base = 2
            instances[index] = create_instance(two, e, N);
          case 1:
            instances[index] = create_instance(five, e, N);
          case 2:
            instances[index] = create_instance(seven, e, N);
          case 3:
            instances[index] = create_instance(thirteen, e, N);
        }

        mpz_clear(e);
    }

    mpz_clear(maxval);
    mpz_clear(two);
    mpz_clear(five);
    mpz_clear(seven);
    mpz_clear(thirteen);
    return instances;
  }

  __host__ static instance_t create_instance(const mpz_t x, const mpz_t e, const mpz_t N) {
    instance_t instance;
    // first, we get the number of limbs

    const size_t num_limbs_x = mpz_size(x);
    const size_t num_limbs_e = mpz_size(e);
    const size_t num_limbs_N = mpz_size(N);

    // just have these assertions in case anyone tries to pull any funny business
    // any one of these failing means the input is too big and we would have caused a segfault
    // the size of instance x, e, N limbs should be params::BITS/32, so if these were not true
    // then we would loop over them and cause a segfault.
    assert(num_limbs_x <= params::BITS/32);
    assert(num_limbs_e <= params::BITS/32);
    assert(num_limbs_N <= params::BITS/32);

    // start with x
    for (int i = 0; i < num_limbs_x; ++i) {
        // get limb i, put in x.
        instance.x._limbs[i] = mpz_getlimbn(x, i);
    }

    // now do e
    for (int i = 0; i < num_limbs_e; ++i) {
        // get limb i, put in e.
        instance.power._limbs[i] = mpz_getlimbn(e, i);
    }

    // now do N
    for (int i = 0; i < num_limbs_N; ++i) {
        // get limb i, put in N.
        instance.modulus._limbs[i] = mpz_getlimbn(N, i);
    }

    // now we're going to assert that the modulus is odd, just because we don't want to create
    // instances that would be incorrect for powm_ODD_t.
    assert(instance.modulus._limbs[0] & 1);

    return instance;
  }

  __host__ static instance_t *create_instances(const mpz_t* xs, const mpz_t* es, const mpz_t* Ns, const uint32_t count) {
    instance_t *instances=(instance_t *)malloc(sizeof(instance_t)*count);
    int         index;

    for(index=0;index<count;index++) {
      instances[index] = create_instance(xs[index], es[index], Ns[index]);
    }
    return instances;
  }

  __host__ static void verify_results(instance_t *instances, uint32_t count) {
    mpz_t x, p, m, computed, correct;

    mpz_init(x);
    mpz_init(p);
    mpz_init(m);
    mpz_init(computed);
    mpz_init(correct);
    int wrong = 0;

    for(int index=0;index<count;index++) {
      to_mpz(x, instances[index].x._limbs, params::BITS/32);
      to_mpz(p, instances[index].power._limbs, params::BITS/32);
      to_mpz(m, instances[index].modulus._limbs, params::BITS/32);
      to_mpz(computed, instances[index].result._limbs, params::BITS/32);

      size_t instance_x_size = mpz_sizeinbase(x, 2);
      size_t instance_p_size = mpz_sizeinbase(p, 2);
      size_t instance_m_size = mpz_sizeinbase(m, 2);
      size_t instance_r_size = mpz_sizeinbase(computed, 2);
      /* printf("Instance %d: Number of bits in x: %lu\n", index, instance_x_size); */
      /* printf("Instance %d: Number of bits in p: %lu\n", index, instance_p_size); */
      /* printf("Instance %d: Number of bits in m: %lu\n", index, instance_m_size); */
      /* printf("Instance %d: Number of bits in r: %lu\n", index, instance_r_size); */
      mpz_powm(correct, x, p, m);
      if(mpz_cmp(correct, computed)!=0) {
        /* printf("gpu inverse kernel failed on instance %d\n", index); */
          wrong++;
        // return;
      }
    }
    printf("Number of powm's computed: %d\n", count);

    mpz_clear(x);
    mpz_clear(p);
    mpz_clear(m);
    mpz_clear(computed);
    mpz_clear(correct);

    if (wrong == 0) {
        printf("All results match\n");
    } else {
        printf("Not all results match, %d wrong\n", wrong);
    }
  }
};

// kernel implementation using cgbn
//
// Unfortunately, the kernel must be separate from the powm_odd_t class

template<class params>
__global__ void kernel_powm_odd(cgbn_error_report_t *report, typename powm_odd_t<params>::instance_t *instances, uint32_t count) {
  int32_t instance;

  // decode an instance number from the blockIdx and threadIdx
  instance=(blockIdx.x*blockDim.x + threadIdx.x)/params::TPI;
  if(instance>=count)
    return;

  powm_odd_t<params>                 po(cgbn_report_monitor, report, instance);
  typename powm_odd_t<params>::bn_t  r, x, p, m;

  // the loads and stores can go in the class, but it seems more natural to have them
  // here and to pass in and out bignums
  cgbn_load(po._env, x, &(instances[instance].x));
  cgbn_load(po._env, p, &(instances[instance].power));
  cgbn_load(po._env, m, &(instances[instance].modulus));

  // this can be either fixed_window_powm_odd or sliding_window_powm_odd.
  // when TPI<32, fixed window runs much faster because it is less divergent, so we use it here
  po.fixed_window_powm_odd(r, x, p, m);
  //   OR
  // po.sliding_window_powm_odd(r, x, p, m);

  cgbn_store(po._env, &(instances[instance].result), r);
}

// grouped fixed kernel implementation using cgbn -- IMPORTANT! default grouping = 1024
template<class params>
__global__ void grouped_fixed_kernel_powm_odd(cgbn_error_report_t *report, typename powm_odd_t<params>::instance_t *instances, const uint32_t count, const uint32_t grouping, const uint32_t time_value) {
  int32_t instance;

  // decode an instance number from the blockIdx and threadIdx
  instance=(blockIdx.x*blockDim.x + threadIdx.x)/params::TPI;
  if(instance>=count)
    return;

  powm_odd_t<params>                 po(cgbn_report_monitor, report, instance);
  typename powm_odd_t<params>::bn_t  r, x, p, m;

  // the loads and stores can go in the class, but it seems more natural to have them
  // here and to pass in and out bignums
  cgbn_load(po._env, x, &(instances[instance].x));
  cgbn_load(po._env, p, &(instances[instance].power));
  cgbn_load(po._env, m, &(instances[instance].modulus));

  // this can be either fixed_window_powm_odd or sliding_window_powm_odd.
  // when TPI<32, fixed window runs much faster because it is less divergent, so we use it here
  po.fixed_window_powm_odd(r, x, p, m);
  //   OR
  // po.grouped_sliding_window_powm_odd(r, x, p, m, grouping);

  cgbn_store(po._env, &(instances[instance].result), r);
}

// grouped sliding window kernel implementation using cgbn -- IMPORTANT! default grouping = 1024
template<class params>
__global__ void grouped_sliding_kernel_powm_odd(cgbn_error_report_t *report, typename powm_odd_t<params>::instance_t *instances, const uint32_t count, const uint32_t grouping) {
  int32_t instance;

  // decode an instance number from the blockIdx and threadIdx
  instance=(blockIdx.x*blockDim.x + threadIdx.x)/params::TPI;
  if(instance>=count)
    return;

  powm_odd_t<params>                 po(cgbn_report_monitor, report, instance);
  typename powm_odd_t<params>::bn_t  r, x, p, m;

  // the loads and stores can go in the class, but it seems more natural to have them
  // here and to pass in and out bignums
  cgbn_load(po._env, x, &(instances[instance].x));
  cgbn_load(po._env, p, &(instances[instance].power));
  cgbn_load(po._env, m, &(instances[instance].modulus));

  // this can be either fixed_window_powm_odd or sliding_window_powm_odd.
  // when TPI<32, fixed window runs much faster because it is less divergent, so we use it here
  // po.grouped_fixed_window_powm_odd(r, x, p, m, grouping);
  //   OR
  po.grouped_sliding_window_powm_odd(r, x, p, m, grouping);

  cgbn_store(po._env, &(instances[instance].result), r);
}


template<class params>
void run_test(uint32_t instance_count) {
  typedef typename powm_odd_t<params>::instance_t instance_t;

  instance_t          *instances, *gpuInstances;
  cgbn_error_report_t *report;
  int32_t              TPB=(params::TPB==0) ? 128 : params::TPB;    // default threads per block to 128
  int32_t              TPI=params::TPI, IPB=TPB/TPI;                // IPB is instances per block

  printf("Generating instances ...\n");
  instances=powm_odd_t<params>::generate_instances(instance_count);

  printf("Copying instances to the GPU ...\n");
  CUDA_CHECK(cudaSetDevice(0));
  CUDA_CHECK(cudaMalloc((void **)&gpuInstances, sizeof(instance_t)*instance_count));
  CUDA_CHECK(cudaMemcpy(gpuInstances, instances, sizeof(instance_t)*instance_count, cudaMemcpyHostToDevice));

  // create a cgbn_error_report for CGBN to report back errors
  CUDA_CHECK(cgbn_error_report_alloc(&report));

  printf("Running GPU kernel ...\n");

  // launch kernel with blocks=ceil(instance_count/IPB) and threads=TPB
  kernel_powm_odd<params><<<(instance_count+IPB-1)/IPB, TPB>>>(report, gpuInstances, instance_count);

  // error report uses managed memory, so we sync the device (or stream) and check for cgbn errors
  CUDA_CHECK(cudaDeviceSynchronize());
  CGBN_CHECK(report);

  // copy the instances back from gpuMemory
  printf("Copying results back to CPU ...\n");
  CUDA_CHECK(cudaMemcpy(instances, gpuInstances, sizeof(instance_t)*instance_count, cudaMemcpyDeviceToHost));

  printf("Verifying the results ...\n");
  powm_odd_t<params>::verify_results(instances, instance_count);

  // clean up
  free(instances);
  CUDA_CHECK(cudaFree(gpuInstances));
  CUDA_CHECK(cgbn_error_report_free(report));
}

template<class params>
void run_puzzle_test(const uint32_t instance_count, const uint32_t time_value) {
  typedef typename powm_odd_t<params>::instance_t instance_t;

  instance_t          *instances, *gpuInstances;
  cgbn_error_report_t *report;
  int32_t              TPB=(params::TPB==0) ? 128 : params::TPB;    // default threads per block to 128
  int32_t              TPI=params::TPI, IPB=TPB/TPI;                // IPB is instances per block

  printf("Generating composite to be used in puzzles...\n");

  // initialize private key
  private_key priv;
  mpz_init(priv.n);
  mpz_init(priv.e);
  mpz_init(priv.d);
  mpz_init(priv.p);
  mpz_init(priv.q);

  // initialize public key
  public_key  pub;
  mpz_init(pub.n);
  mpz_init(pub.e);

  // now generate the keys with 4096 bits
  generate_keys(&priv, &pub, 4096);

  printf("Generating puzzle instances ...\n");
  instances=powm_odd_t<params>::generate_puzzle_instances(instance_count, time_value, priv.n);

  printf("Copying instances to the GPU ...\n");
  CUDA_CHECK(cudaSetDevice(0));
  CUDA_CHECK(cudaMalloc((void **)&gpuInstances, sizeof(instance_t)*instance_count));
  CUDA_CHECK(cudaMemcpy(gpuInstances, instances, sizeof(instance_t)*instance_count, cudaMemcpyHostToDevice));

  // create a cgbn_error_report for CGBN to report back errors
  CUDA_CHECK(cgbn_error_report_alloc(&report));

  // declaring argument of time()
  time_t my_time = time(NULL);
  // ctime() used to give the present time
  printf("Start Time: %s", ctime(&my_time));

  printf("Running GPU kernel ...\n");

  // launch kernel with blocks=ceil(instance_count/IPB) and threads=TPB
  kernel_powm_odd<params><<<(instance_count+IPB-1)/IPB, TPB>>>(report, gpuInstances, instance_count);

  // error report uses managed memory, so we sync the device (or stream) and check for cgbn errors
  CUDA_CHECK(cudaDeviceSynchronize());
  CGBN_CHECK(report);

  // declaring argument of time()
  my_time = time(NULL);
  // ctime() used to give the present time
  printf("Stop Time: %s", ctime(&my_time));

  // copy the instances back from gpuMemory
  printf("Copying results back to CPU ...\n");
  CUDA_CHECK(cudaMemcpy(instances, gpuInstances, sizeof(instance_t)*instance_count, cudaMemcpyDeviceToHost));

  // declaring argument of time()
  my_time = time(NULL);
  // ctime() used to give the present time
  printf("Verify Start Time: %s", ctime(&my_time));

  printf("Verifying the results ...\n");
  powm_odd_t<params>::verify_results(instances, instance_count);

  // declaring argument of time()
  my_time = time(NULL);
  // ctime() used to give the present time
  printf("Verify Stop Time: %s", ctime(&my_time));

  // clean up
  free(instances);

  // clear private key
  mpz_clear(priv.n);
  mpz_clear(priv.e);
  mpz_clear(priv.d);
  mpz_clear(priv.p);
  mpz_clear(priv.q);

  // clear public key
  mpz_clear(pub.n);
  mpz_clear(pub.e);

  CUDA_CHECK(cudaFree(gpuInstances));
  CUDA_CHECK(cgbn_error_report_free(report));
}

/* int main() { */
/*   typedef powm_params_t<8, 1024, 5> params; */

/*   run_test<params>(10000); */
/* } */
