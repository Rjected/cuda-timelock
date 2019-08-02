/**********************************************************************
 *                                                                    *
 * Created by Adam Brockett                                           *
 *                                                                    *
 * Copyright (c) 2010                                                 *
 *                                                                    *
 * Redistribution and use in source and binary forms, with or without *
 * modification is allowed.                                           *
 *                                                                    *
 * But if you let me know you're using my code, that would be freaking*
 * sweet.                                                             *
 *                                                                    *
 **********************************************************************/

#include "rsa.h"

// IMPORTANT!!! This RSA key generation function is for TESTING PURPOSES ONLY!
// Never use it, ever, except for testing. Use OpenSSL instead.
// There is a reason why I put this in the insecure_rsa folder.
// I'm just using it to generate primes that are in mpz_t format.

/* NOTE: Assumes mpz_t's are initted in ku and kp */
void generate_keys(private_key* ku, public_key* kp, const uint32_t modulus_size)
{
    // This is where the original code changes
    const uint32_t buffer_size = (modulus_size / 8) / 2;

    char buf[buffer_size];
    int i;
    mpz_t phi; mpz_init(phi);
    mpz_t tmp1; mpz_init(tmp1);
    mpz_t tmp2; mpz_init(tmp2);

    srand(time(NULL));

    /* Insetead of selecting e st. gcd(phi, e) = 1; 1 < e < phi, lets choose e
     * first then pick p,q st. gcd(e, p-1) = gcd(e, q-1) = 1 */
    // We'll set e globally.  I've seen suggestions to use primes like 3, 17 or
    // 65537, as they make coming calculations faster.  Lets use 3.
    mpz_set_ui(ku->e, 3);

    /* Select p and q */
    /* Start with p */
    // Set the bits of tmp randomly
    for(i = 0; i < buffer_size; i++)
        buf[i] = rand() % 0xFF;
    // Set the top two bits to 1 to ensure int(tmp) is relatively large
    buf[0] |= 0xC0;
    // Set the bottom bit to 1 to ensure int(tmp) is odd (better for finding primes)
    buf[buffer_size - 1] |= 0x01;
    // Interpret this char buffer as an int
    mpz_import(tmp1, buffer_size, 1, sizeof(buf[0]), 0, 0, buf);
    // Pick the next prime starting from that random number
    mpz_nextprime(ku->p, tmp1);
    /* Make sure this is a good choice*/
    mpz_mod(tmp2, ku->p, ku->e);        /* If p mod e == 1, gcd(phi, e) != 1 */
    while(!mpz_cmp_ui(tmp2, 1))
    {
        mpz_nextprime(ku->p, ku->p);    /* so choose the next prime */
        mpz_mod(tmp2, ku->p, ku->e);
    }

    /* Now select q */
    do {
        for(i = 0; i < buffer_size; i++)
            buf[i] = rand() % 0xFF;
        // Set the top two bits to 1 to ensure int(tmp) is relatively large
        buf[0] |= 0xC0;
        // Set the bottom bit to 1 to ensure int(tmp) is odd
        buf[buffer_size - 1] |= 0x01;
        // Interpret this char buffer as an int
        mpz_import(tmp1, (buffer_size), 1, sizeof(buf[0]), 0, 0, buf);
        // Pick the next prime starting from that random number
        mpz_nextprime(ku->q, tmp1);
        mpz_mod(tmp2, ku->q, ku->e);
        while(!mpz_cmp_ui(tmp2, 1))
        {
            mpz_nextprime(ku->q, ku->q);
            mpz_mod(tmp2, ku->q, ku->e);
        }
    } while(mpz_cmp(ku->p, ku->q) == 0); /* If we have identical primes (unlikely), try again */

    /* Calculate n = p x q */
    mpz_mul(ku->n, ku->p, ku->q);

    /* Compute phi(n) = (p-1)(q-1) */
    mpz_sub_ui(tmp1, ku->p, 1);
    mpz_sub_ui(tmp2, ku->q, 1);
    mpz_mul(phi, tmp1, tmp2);

    /* Calculate d (multiplicative inverse of e mod phi) */
    if(mpz_invert(ku->d, ku->e, phi) == 0)
    {
        mpz_gcd(tmp1, ku->e, phi);
        printf("gcd(e, phi) = [%s]\n", mpz_get_str(NULL, 16, tmp1));
        printf("Invert failed\n");
    }

    /* Set public key */
    mpz_set(kp->e, ku->e);
    mpz_set(kp->n, ku->n);

    return;
}
