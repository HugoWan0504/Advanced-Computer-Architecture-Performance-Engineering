/*
This file trains the GPT-2 model.
This version is the clean, minimal, reference. As such:
- it runs on CPU.
- it does not make the code too complex; it is readable.
- it does not use any processor-specific instructions, intrinsics and such.
There will be other versions of this code that specialize it and make it fast.
*/

#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <ctype.h>
#include <stdint.h>
#include <assert.h>
#include <math.h>
#include <time.h>
#include <string.h>
#include <unistd.h>
#include "perfstats.h"
// our own utilities
// defines: fopenCheck, freadCheck, fcloseCheck, fseekCheck, mallocCheck
#include "llmc/utils.h"
// defines: tokenizer_init, tokenizer_decode, tokenizer_free
#include "llmc/tokenizer.h"
// defines: dataloader_init, dataloader_reset, dataloader_next_batch, dataloader_free
#include "llmc/dataloader.h"

// ----------------------------------------------------------------------------
// all the individual layers' forward and backward passes
// B = batch_size, T = sequence_length, C = channels, V = vocab_size

void encoder_forward(float* restrict out,
                     int* restrict inp, float* restrict wte, float* restrict wpe,
                     int B, int T, int C) {
    int BT = B * T;

    #pragma omp parallel for schedule(static)
    for (int bt = 0; bt < BT; ++bt) {
        int b = bt / T;
        int t = bt % T;

        float* out_bt = out + (size_t)bt * C;
        int ix        = inp[bt];
        float* wte_ix = wte + (size_t)ix * C;
        float* wpe_t  = wpe + (size_t)t  * C;

        for (int i = 0; i < C; ++i) {
            out_bt[i] = wte_ix[i] + wpe_t[i];
        }
    }
}


void encoder_backward(float* dwte, float* dwpe,
                      float* dout, int* inp,
                      int B, int T, int C) {
    for (int b = 0; b < B; b++) {
        for (int t = 0; t < T; t++) {
            float* dout_bt = dout + b * T * C + t * C;
            int ix = inp[b * T + t];
            float* dwte_ix = dwte + ix * C;
            float* dwpe_t = dwpe + t * C;
            for (int i = 0; i < C; i++) {
                float d = dout_bt[i];
                dwte_ix[i] += d;
                dwpe_t[i] += d;
            }
        }
    }
}


void layernorm_forward(float* restrict out, float* restrict mean, float* restrict rstd,
                       float* restrict inp, float* restrict weight, float* restrict bias,
                       int B, int T, int C) {
    float eps = 1e-5f;
    int BT = B * T;

    #pragma omp parallel for schedule(static)
    for (int bt = 0; bt < BT; ++bt) {
        int b = bt / T;
        int t = bt % T;

        float* x      = inp  + (size_t)b * T * C + (size_t)t * C;
        float* out_bt = out  + (size_t)b * T * C + (size_t)t * C;

        // mean
        float m = 0.0f;
        for (int i = 0; i < C; ++i) {
            m += x[i];
        }
        m /= C;

        // variance
        float v = 0.0f;
        for (int i = 0; i < C; ++i) {
            float d = x[i] - m;
            v += d * d;
        }
        v /= C;

        float s = 1.0f / sqrtf(v + eps);
        mean[bt] = m;
        rstd[bt] = s;

        // normalize + affine
        for (int i = 0; i < C; ++i) {
            float n = (x[i] - m) * s;
            out_bt[i] = n * weight[i] + bias[i];
        }
    }
}


void layernorm_backward(float* restrict dinp, float* restrict dweight, float* restrict dbias,
                        float* restrict dout, float* restrict inp, float* restrict weight,
                        float* restrict mean, float* restrict rstd,
                        int B, int T, int C) {
    int BT = B * T;

    for (int bt = 0; bt < BT; ++bt) {
        int b = bt / T;
        int t = bt % T;

        float* dout_bt = dout + (size_t)b * T * C + (size_t)t * C;
        float* inp_bt  = inp  + (size_t)b * T * C + (size_t)t * C;
        float* dinp_bt = dinp + (size_t)b * T * C + (size_t)t * C;

        float mean_bt = mean[bt];
        float rstd_bt = rstd[bt];

        // two reductions
        float dnorm_sum      = 0.0f;
        float dnorm_norm_sum = 0.0f;

        for (int i = 0; i < C; ++i) {
            float norm_i  = (inp_bt[i] - mean_bt) * rstd_bt;
            float dnorm_i = weight[i] * dout_bt[i];
            dnorm_sum      += dnorm_i;
            dnorm_norm_sum += dnorm_i * norm_i;
        }

        float dnorm_mean      = dnorm_sum      / C;
        float dnorm_norm_mean = dnorm_norm_sum / C;

        // grads
        for (int i = 0; i < C; ++i) {
            float norm_i  = (inp_bt[i] - mean_bt) * rstd_bt;
            float dnorm_i = weight[i] * dout_bt[i];

            dbias[i]   += dout_bt[i];
            dweight[i] += norm_i * dout_bt[i];

            float dval = dnorm_i;
            dval -= dnorm_mean;
            dval -= norm_i * dnorm_norm_mean;
            dval *= rstd_bt;

            dinp_bt[i] += dval;
        }
    }
}


void matmul_forward_naive(float* out,
                         const float* inp, const float* weight, const float* bias,
                         int B, int T, int C, int OC) {
    // the most naive implementation of matrix multiplication
    // this serves as an algorithmic reference, and as a fallback for
    // unfriendly input shapes inside matmul_forward(), below.
    for (int b = 0; b < B; b++) {
        for (int t = 0; t < T; t++) {
            int bt = b * T + t;
            for (int o = 0; o < OC; o++) {
                float val = (bias != NULL) ? bias[o] : 0.0f;
                for (int i = 0; i < C; i++) {
                    val += inp[bt * C + i] * weight[o*C + i];
                }
                out[bt * OC + o] = val;
            }
        }
    }
}


void matmul_forward(float* restrict out,
                    const float* restrict inp,
                    const float* restrict weight,
                    const float* restrict bias,
                    int B, int T, int C, int OC) {

    int BT = B * T;

    #pragma omp parallel for schedule(static)
    for (int o = 0; o < OC; ++o) {

        const float* wrow = weight + (size_t)o * C;
        float b = (bias != NULL) ? bias[o] : 0.0f;

        for (int bt = 0; bt < BT; ++bt) {

            const float* inp_bt = inp + (size_t)bt * C;

            float acc0 = 0.f, acc1 = 0.f, acc2 = 0.f, acc3 = 0.f;

            int i = 0;
            for (; i + 3 < C; i += 4) {
                acc0 += inp_bt[i]     * wrow[i];
                acc1 += inp_bt[i + 1] * wrow[i + 1];
                acc2 += inp_bt[i + 2] * wrow[i + 2];
                acc3 += inp_bt[i + 3] * wrow[i + 3];
            }

            for (; i < C; ++i) {
                acc0 += inp_bt[i] * wrow[i];
            }

            float acc = b + acc0 + acc1 + acc2 + acc3;
            out[(size_t)bt * OC + o] = acc;
        }
    }
}

void matmul_backward(float* restrict dinp, float* restrict dweight, float* restrict dbias,
                     const float* restrict dout, const float* restrict inp, const float* restrict weight,
                     int B, int T, int C, int OC) {

    int BT = B * T;

    // dweight + dbias
    #pragma omp parallel for schedule(static)
    for (int o = 0; o < OC; ++o) {

        float* dwrow = dweight + (size_t)o * C;
        float db = 0.0f;

        for (int bt = 0; bt < BT; ++bt) {

            const float* dout_bt = dout + (size_t)bt * OC;
            const float* inp_bt  = inp  + (size_t)bt * C;

            float d = dout_bt[o];
            db += d;

            int i = 0;
            for (; i + 3 < C; i += 4) {
                dwrow[i]     += inp_bt[i]     * d;
                dwrow[i + 1] += inp_bt[i + 1] * d;
                dwrow[i + 2] += inp_bt[i + 2] * d;
                dwrow[i + 3] += inp_bt[i + 3] * d;
            }
            for (; i < C; ++i) {
                dwrow[i] += inp_bt[i] * d;
            }
        }

        if (dbias) dbias[o] += db;
    }

    // dinp
    #pragma omp parallel for schedule(static)
    for (int bt = 0; bt < BT; ++bt) {

        float*       dinp_bt = dinp + (size_t)bt * C;
        const float* dout_bt = dout + (size_t)bt * OC;

        for (int o = 0; o < OC; ++o) {

            const float* wrow = weight + (size_t)o * C;
            float d = dout_bt[o];

            int i = 0;
            for (; i + 3 < C; i += 4) {
                dinp_bt[i]     += wrow[i]     * d;
                dinp_bt[i + 1] += wrow[i + 1] * d;
                dinp_bt[i + 2] += wrow[i + 2] * d;
                dinp_bt[i + 3] += wrow[i + 3] * d;
            }
            for (; i < C; ++i) {
                dinp_bt[i] += wrow[i] * d;
            }
        }
    }
}



void attention_forward(float* restrict out, float* restrict preatt, float* restrict att,
                       float* restrict inp,
                       int B, int T, int C, int NH) {
    // input is (B, T, 3C) holding Q, K, V
    // preatt, att are (B, NH, T, T)
    // out is (B, T, C)
    int C3 = C * 3;
    int hs = C / NH; // head size
    float scale = 1.0f / sqrtf((float)hs);

    #pragma omp parallel for collapse(2) schedule(static)
    for (int b = 0; b < B; b++) {
        for (int h = 0; h < NH; h++) {

            size_t base_inp_b   = (size_t)b * T * C3;
            size_t base_patt_bh = (size_t)b * NH * T * T + (size_t)h * T * T;
            size_t base_out_b   = (size_t)b * T * C;

            for (int t = 0; t < T; t++) {

                float* query_t    = inp    + base_inp_b   + (size_t)t * C3 + h * hs;
                float* preatt_bth = preatt + base_patt_bh + (size_t)t * T;
                float* att_bth    = att    + base_patt_bh + (size_t)t * T;

                // ---- pass 1: Q·K^T (causal) + maxval ----
                float maxval = -10000.0f;
                for (int t2 = 0; t2 <= t; t2++) {
                    float* key_t2 = inp + base_inp_b + (size_t)t2 * C3 + h * hs + C; // +C: key

                    float s0 = 0.f, s1 = 0.f, s2 = 0.f, s3 = 0.f;
                    int i = 0;
                    for (; i + 3 < hs; i += 4) {
                        s0 += query_t[i]     * key_t2[i];
                        s1 += query_t[i + 1] * key_t2[i + 1];
                        s2 += query_t[i + 2] * key_t2[i + 2];
                        s3 += query_t[i + 3] * key_t2[i + 3];
                    }
                    for (; i < hs; ++i) {
                        s0 += query_t[i] * key_t2[i];
                    }
                    float val = (s0 + s1 + s2 + s3) * scale;
                    preatt_bth[t2] = val;
                    if (val > maxval) maxval = val;
                }

                // ---- pass 2: exp and sum ----
                float expsum = 0.0f;
                for (int t2 = 0; t2 <= t; t2++) {
                    float expv = expf(preatt_bth[t2] - maxval);
                    expsum += expv;
                    att_bth[t2] = expv;
                }
                float expsum_inv = (expsum == 0.0f) ? 0.0f : 1.0f / expsum;

                // ---- pass 3: normalize + causal mask ----
                for (int t2 = 0; t2 < T; t2++) {
                    if (t2 <= t) {
                        att_bth[t2] *= expsum_inv;
                    } else {
                        att_bth[t2] = 0.0f;
                    }
                }

                // ---- pass 4: accumulate V ----
                float* out_bth = out + base_out_b + (size_t)t * C + h * hs;
                for (int i = 0; i < hs; i++) out_bth[i] = 0.0f;

                for (int t2 = 0; t2 <= t; t2++) {
                    float* value_t2 = inp + base_inp_b + (size_t)t2 * C3 + h * hs + 2 * C; // +2C: value
                    float a = att_bth[t2];

                    int i = 0;
                    for (; i + 3 < hs; i += 4) {
                        out_bth[i]     += a * value_t2[i];
                        out_bth[i + 1] += a * value_t2[i + 1];
                        out_bth[i + 2] += a * value_t2[i + 2];
                        out_bth[i + 3] += a * value_t2[i + 3];
                    }
                    for (; i < hs; ++i) {
                        out_bth[i] += a * value_t2[i];
                    }
                }
            }
        }
    }
}


void attention_backward(float* restrict dinp, float* restrict dpreatt, float* restrict datt,
                        float* restrict dout, float* restrict inp, float* restrict att,
                        int B, int T, int C, int NH) {
    int C3 = C * 3;
    int hs = C / NH;
    float scale = 1.0f / sqrtf((float)hs);

    #pragma omp parallel for collapse(2) schedule(static)
    for (int b = 0; b < B; b++) {
        for (int h = 0; h < NH; h++) {

            size_t base_inp_b   = (size_t)b * T * C3;
            size_t base_dinp_b  = (size_t)b * T * C3;
            size_t base_att_bh  = (size_t)b * NH * T * T + (size_t)h * T * T;
            size_t base_dout_b  = (size_t)b * T * C;

            for (int t = 0; t < T; t++) {

                float* att_bth     = att     + base_att_bh + (size_t)t * T;
                float* datt_bth    = datt    + base_att_bh + (size_t)t * T;
                float* dpreatt_bth = dpreatt + base_att_bh + (size_t)t * T;

                float* dquery_t = dinp + base_dinp_b + (size_t)t * C3 + h * hs;
                float* query_t  = inp  + base_inp_b  + (size_t)t * C3 + h * hs;
                float* dout_bth = dout + base_dout_b + (size_t)t * C + h * hs;

                // ---- backward pass 4: through V accumulation ----
                for (int t2 = 0; t2 <= t; t2++) {
                    float* value_t2  = inp  + base_inp_b  + (size_t)t2 * C3 + h * hs + 2 * C;
                    float* dvalue_t2 = dinp + base_dinp_b + (size_t)t2 * C3 + h * hs + 2 * C;
                    float a = att_bth[t2];

                    int i = 0;
                    for (; i + 3 < hs; i += 4) {
                        float dout0 = dout_bth[i];
                        float dout1 = dout_bth[i + 1];
                        float dout2 = dout_bth[i + 2];
                        float dout3 = dout_bth[i + 3];

                        datt_bth[t2] += value_t2[i]     * dout0
                                      + value_t2[i + 1] * dout1
                                      + value_t2[i + 2] * dout2
                                      + value_t2[i + 3] * dout3;

                        dvalue_t2[i]     += a * dout0;
                        dvalue_t2[i + 1] += a * dout1;
                        dvalue_t2[i + 2] += a * dout2;
                        dvalue_t2[i + 3] += a * dout3;
                    }
                    for (; i < hs; ++i) {
                        float douti = dout_bth[i];
                        datt_bth[t2]   += value_t2[i] * douti;
                        dvalue_t2[i]   += a * douti;
                    }
                }

                // ---- backward passes 2 & 3: softmax ----
                for (int t2 = 0; t2 <= t; t2++) {
                    float datt_t2 = datt_bth[t2];
                    for (int t3 = 0; t3 <= t; t3++) {
                        float indicator = (t2 == t3) ? 1.0f : 0.0f;
                        float local_derivative = att_bth[t2] * (indicator - att_bth[t3]);
                        dpreatt_bth[t3] += local_derivative * datt_t2;
                    }
                }

                // ---- backward pass 1: Q @ K ----
                for (int t2 = 0; t2 <= t; t2++) {
                    float* key_t2  = inp  + base_inp_b  + (size_t)t2 * C3 + h * hs + C;
                    float* dkey_t2 = dinp + base_dinp_b + (size_t)t2 * C3 + h * hs + C;
                    float dp = dpreatt_bth[t2] * scale;

                    int i = 0;
                    for (; i + 3 < hs; i += 4) {
                        float q0 = query_t[i];
                        float q1 = query_t[i + 1];
                        float q2 = query_t[i + 2];
                        float q3 = query_t[i + 3];

                        float k0 = key_t2[i];
                        float k1 = key_t2[i + 1];
                        float k2 = key_t2[i + 2];
                        float k3 = key_t2[i + 3];

                        dquery_t[i]     += k0 * dp;
                        dquery_t[i + 1] += k1 * dp;
                        dquery_t[i + 2] += k2 * dp;
                        dquery_t[i + 3] += k3 * dp;

                        dkey_t2[i]      += q0 * dp;
                        dkey_t2[i + 1]  += q1 * dp;
                        dkey_t2[i + 2]  += q2 * dp;
                        dkey_t2[i + 3]  += q3 * dp;
                    }
                    for (; i < hs; ++i) {
                        dquery_t[i] += key_t2[i] * dp;
                        dkey_t2[i]  += query_t[i] * dp;
                    }
                }
            }
        }
    }
}

#define GELU_SCALING_FACTOR sqrtf(2.0f / M_PI)
void gelu_forward(float* restrict out, float* restrict inp, int N) {
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < N; i++) {
        float x = inp[i];
        float cube = 0.044715f * x * x * x;
        out[i] = 0.5f * x * (1.0f + tanhf(GELU_SCALING_FACTOR * (x + cube)));
    }
}


// we want to use -Ofast optimization, but sadly GeLU breaks, so disable this flag just for it (#168)
#pragma float_control(precise, on, push)
#if defined(__GNUC__) && !defined(__clang__)
__attribute__((optimize("no-finite-math-only")))
#endif
void gelu_backward(float* restrict dinp, float* restrict inp, float* restrict dout, int N) {
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < N; i++) {
        float x = inp[i];
        float cube = 0.044715f * x * x * x;
        float tanh_arg = GELU_SCALING_FACTOR * (x + cube);
        float tanh_out = tanhf(tanh_arg);
        float coshf_out = coshf(tanh_arg);
        float sech_out = 1.0f / (coshf_out * coshf_out);
        float local_grad =
            0.5f * (1.0f + tanh_out) +
            x * 0.5f * sech_out * GELU_SCALING_FACTOR *
            (1.0f + 3.0f * 0.044715f * x * x);
        dinp[i] += local_grad * dout[i];
    }
}
#pragma float_control(pop)


void residual_forward(float* restrict out, float* restrict inp1, float* restrict inp2, int N) {
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < N; ++i) {
        out[i] = inp1[i] + inp2[i];
    }
}


void residual_backward(float* restrict dinp1, float* restrict dinp2, float* restrict dout, int N) {
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < N; ++i) {
        float g = dout[i];
        dinp1[i] += g;
        dinp2[i] += g;
    }
}


void softmax_forward(float* restrict probs, float* restrict logits, int B, int T, int V, int Vp) {
    int BT = B * T;

    #pragma omp parallel for schedule(static)
    for (int bt = 0; bt < BT; ++bt) {
        int b = bt / T;
        int t = bt % T;

        float* logits_bt = logits + (size_t)b * T * Vp + (size_t)t * Vp;
        float* probs_bt  = probs  + (size_t)b * T * Vp + (size_t)t * Vp;

        float maxval = -10000.0f;
        for (int i = 0; i < V; ++i) {
            float x = logits_bt[i];
            if (x > maxval) maxval = x;
        }

        float sum = 0.0f;
        for (int i = 0; i < V; ++i) {
            float e = expf(logits_bt[i] - maxval);
            probs_bt[i] = e;
            sum += e;
        }

        float inv_sum = 1.0f / sum;
        for (int i = 0; i < V; ++i) {
            probs_bt[i] *= inv_sum;
        }
        for (int i = V; i < Vp; ++i) {
            probs_bt[i] = 0.0f;
        }
    }
}


void crossentropy_forward(float* restrict losses,
                          float* restrict probs, int* restrict targets,
                          int B, int T, int Vp) {
    int BT = B * T;

    #pragma omp parallel for schedule(static)
    for (int bt = 0; bt < BT; ++bt) {
        int b = bt / T;
        int t = bt % T;

        float* probs_bt = probs + (size_t)b * T * Vp + (size_t)t * Vp;
        int ix = targets[bt];
        losses[bt] = -logf(probs_bt[ix]);
    }
}


void crossentropy_softmax_backward(float* restrict dlogits,
                                   float* restrict dlosses, float* restrict probs, int* restrict targets,
                                   int B, int T, int V, int Vp) {
    int BT = B * T;

    #pragma omp parallel for schedule(static)
    for (int bt = 0; bt < BT; ++bt) {
        int b = bt / T;
        int t = bt % T;

        float* dlogits_bt = dlogits + (size_t)b * T * Vp + (size_t)t * Vp;
        float* probs_bt   = probs   + (size_t)b * T * Vp + (size_t)t * Vp;
        float dloss       = dlosses[bt];
        int ix            = targets[bt];

        for (int i = 0; i < V; ++i) {
            float p = probs_bt[i];
            float indicator = (i == ix) ? 1.0f : 0.0f;
            dlogits_bt[i] += (p - indicator) * dloss;
        }
        // padded tail remains untouched (stays zero)
    }
}


// ----------------------------------------------------------------------------
// GPT-2 model definition

typedef struct {
    int max_seq_len; // max sequence length, e.g. 1024
    int vocab_size; // vocab size, e.g. 50257
    int padded_vocab_size; // padded to e.g. %128==0, 50304
    int num_layers; // number of layers, e.g. 12
    int num_heads; // number of heads in attention, e.g. 12
    int channels; // number of channels, e.g. 768
} GPT2Config;

// the parameters of the model
#define NUM_PARAMETER_TENSORS 16
typedef struct {
    float* wte; // (V, C)
    float* wpe; // (maxT, C)
    float* ln1w; // (L, C)
    float* ln1b; // (L, C)
    float* qkvw; // (L, 3*C, C)
    float* qkvb; // (L, 3*C)
    float* attprojw; // (L, C, C)
    float* attprojb; // (L, C)
    float* ln2w; // (L, C)
    float* ln2b; // (L, C)
    float* fcw; // (L, 4*C, C)
    float* fcb; // (L, 4*C)
    float* fcprojw; // (L, C, 4*C)
    float* fcprojb; // (L, C)
    float* lnfw; // (C)
    float* lnfb; // (C)
} ParameterTensors;

void fill_in_parameter_sizes(size_t* param_sizes, GPT2Config config) {
    size_t Vp = config.padded_vocab_size;
    size_t C = config.channels;
    size_t maxT = config.max_seq_len;
    size_t L = config.num_layers;
    param_sizes[0] = Vp * C; // wte
    param_sizes[1] = maxT * C; // wpe
    param_sizes[2] = L * C; // ln1w
    param_sizes[3] = L * C; // ln1b
    param_sizes[4] = L * (3 * C) * C; // qkvw
    param_sizes[5] = L * (3 * C); // qkvb
    param_sizes[6] = L * C * C; // attprojw
    param_sizes[7] = L * C; // attprojb
    param_sizes[8] = L * C; // ln2w
    param_sizes[9] = L * C; // ln2b
    param_sizes[10] = L * (4 * C) * C; // fcw
    param_sizes[11] = L * (4 * C); // fcb
    param_sizes[12] = L * C * (4 * C); // fcprojw
    param_sizes[13] = L * C; // fcprojb
    param_sizes[14] = C; // lnfw
    param_sizes[15] = C; // lnfb
}

// allocate memory for the parameters and point the individual tensors to the right places
float* malloc_and_point_parameters(ParameterTensors* params, size_t* param_sizes) {
    size_t num_parameters = 0;
    for (size_t i = 0; i < NUM_PARAMETER_TENSORS; i++) {
        num_parameters += param_sizes[i];
    }
    // malloc all parameters all at once
    float* params_memory = (float*)mallocCheck(num_parameters * sizeof(float));
    // assign all the tensors
    float** ptrs[] = {
        &params->wte, &params->wpe, &params->ln1w, &params->ln1b, &params->qkvw, &params->qkvb,
        &params->attprojw, &params->attprojb, &params->ln2w, &params->ln2b, &params->fcw, &params->fcb,
        &params->fcprojw, &params->fcprojb, &params->lnfw, &params->lnfb
    };
    float* params_memory_iterator = params_memory;
    for (size_t i = 0; i < NUM_PARAMETER_TENSORS; i++) {
        *(ptrs[i]) = params_memory_iterator;
        params_memory_iterator += param_sizes[i];
    }
    return params_memory;
}

#define NUM_ACTIVATION_TENSORS 23
typedef struct {
    float* encoded; // (B, T, C)
    float* ln1; // (L, B, T, C)
    float* ln1_mean; // (L, B, T)
    float* ln1_rstd; // (L, B, T)
    float* qkv; // (L, B, T, 3*C)
    float* atty; // (L, B, T, C)
    float* preatt; // (L, B, NH, T, T)
    float* att; // (L, B, NH, T, T)
    float* attproj; // (L, B, T, C)
    float* residual2; // (L, B, T, C)
    float* ln2; // (L, B, T, C)
    float* ln2_mean; // (L, B, T)
    float* ln2_rstd; // (L, B, T)
    float* fch; // (L, B, T, 4*C)
    float* fch_gelu; // (L, B, T, 4*C)
    float* fcproj; // (L, B, T, C)
    float* residual3; // (L, B, T, C)
    float* lnf; // (B, T, C)
    float* lnf_mean; // (B, T)
    float* lnf_rstd; // (B, T)
    float* logits; // (B, T, V)
    float* probs; // (B, T, V)
    float* losses; // (B, T)
} ActivationTensors;

void fill_in_activation_sizes(size_t* act_sizes, GPT2Config config, int B, int T) {
    size_t C = config.channels;
    size_t NH = config.num_heads;
    size_t L = config.num_layers;
    size_t Vp = config.padded_vocab_size;
    act_sizes[0] = B * T * C; // encoded
    act_sizes[1] = L * B * T * C; // ln1
    act_sizes[2] = L * B * T; // ln1_mean
    act_sizes[3] = L * B * T; // ln1_rstd
    act_sizes[4] = L * B * T * 3 * C; // qkv
    act_sizes[5] = L * B * T * C; // atty
    act_sizes[6] = L * B * NH * T * T; // preatt
    act_sizes[7] = L * B * NH * T * T; // att
    act_sizes[8] = L * B * T * C; // attproj
    act_sizes[9] = L * B * T * C; // residual2
    act_sizes[10] = L * B * T * C; // ln2
    act_sizes[11] = L * B * T; // ln2_mean
    act_sizes[12] = L * B * T; // ln2_rstd
    act_sizes[13] = L * B * T * 4 * C; // fch
    act_sizes[14] = L * B * T * 4 * C; // fch_gelu
    act_sizes[15] = L * B * T * C; // fcproj
    act_sizes[16] = L * B * T * C; // residual3
    act_sizes[17] = B * T * C; // lnf
    act_sizes[18] = B * T; // lnf_mean
    act_sizes[19] = B * T; // lnf_rstd
    act_sizes[20] = B * T * Vp; // logits
    act_sizes[21] = B * T * Vp; // probs
    act_sizes[22] = B * T; // losses
}

float* malloc_and_point_activations(ActivationTensors* acts, size_t* act_sizes) {
    size_t num_activations = 0;
    for (size_t i = 0; i < NUM_ACTIVATION_TENSORS; i++) {
        num_activations += act_sizes[i];
    }
    float* acts_memory = (float*)mallocCheck(num_activations * sizeof(float));
    float** ptrs[] = {
        &acts->encoded, &acts->ln1, &acts->ln1_mean, &acts->ln1_rstd, &acts->qkv, &acts->atty,
        &acts->preatt, &acts->att, &acts->attproj, &acts->residual2, &acts->ln2, &acts->ln2_mean,
        &acts->ln2_rstd, &acts->fch, &acts->fch_gelu, &acts->fcproj, &acts->residual3, &acts->lnf,
        &acts->lnf_mean, &acts->lnf_rstd, &acts->logits, &acts->probs, &acts->losses
    };
    float* acts_memory_iterator = acts_memory;
    for (size_t i = 0; i < NUM_ACTIVATION_TENSORS; i++) {
        *(ptrs[i]) = acts_memory_iterator;
        acts_memory_iterator += act_sizes[i];
    }
    return acts_memory;
}

typedef struct {
    GPT2Config config;
    // the weights (parameters) of the model, and their sizes
    ParameterTensors params;
    size_t param_sizes[NUM_PARAMETER_TENSORS];
    float* params_memory;
    size_t num_parameters;
    // gradients of the weights
    ParameterTensors grads;
    float* grads_memory;
    // buffers for the AdamW optimizer
    float* m_memory;
    float* v_memory;
    // the activations of the model, and their sizes
    ActivationTensors acts;
    size_t act_sizes[NUM_ACTIVATION_TENSORS];
    float* acts_memory;
    size_t num_activations;
    // gradients of the activations
    ActivationTensors grads_acts;
    float* grads_acts_memory;
    // other run state configuration
    int batch_size; // the batch size (B) of current forward pass
    int seq_len; // the sequence length (T) of current forward pass
    int* inputs; // the input tokens for the current forward pass
    int* targets; // the target tokens for the current forward pass
    float mean_loss; // after a forward pass with targets, will be populated with the mean loss
} GPT2;

void gpt2_build_from_checkpoint(GPT2 *model, const char* checkpoint_path) {

    // read in model from a checkpoint file
    FILE *model_file = fopenCheck(checkpoint_path, "rb");
    int model_header[256];
    freadCheck(model_header, sizeof(int), 256, model_file);
    if (model_header[0] != 20240326) { printf("Bad magic model file\n"); exit(1); }
    if (model_header[1] != 3) {
        printf("Bad version in model file\n");
        printf("---> HINT: try to re-run `python train_gpt2.py`\n");
        exit(1);
    }

    // read in hyperparameters
    size_t maxT, V, Vp, L, NH, C; // size_t to prevent int overflow
    model->config.max_seq_len = maxT = model_header[2];
    model->config.vocab_size = V = model_header[3];
    model->config.num_layers = L = model_header[4];
    model->config.num_heads = NH = model_header[5];
    model->config.channels = C = model_header[6];
    model->config.padded_vocab_size = Vp = model_header[7];
    printf("[GPT-2]\n");
    printf("max_seq_len: %zu\n", maxT);
    printf("vocab_size: %zu\n", V);
    printf("padded_vocab_size: %zu\n", Vp);
    printf("num_layers: %zu\n", L);
    printf("num_heads: %zu\n", NH);
    printf("channels: %zu\n", C);

    // allocate space for all the parameters and read them in
    fill_in_parameter_sizes(model->param_sizes,  model->config);

    // count the number of parameters
    size_t num_parameters = 0;
    for (size_t i = 0; i < NUM_PARAMETER_TENSORS; i++) {
        num_parameters += model->param_sizes[i];
    }
    printf("num_parameters: %zu\n", num_parameters);
    model->num_parameters = num_parameters;

    // read in all the parameters from file
    model->params_memory = malloc_and_point_parameters(&model->params, model->param_sizes);
    freadCheck(model->params_memory, sizeof(float), num_parameters, model_file);
    fcloseCheck(model_file);

    // other inits
    model->acts_memory = NULL;
    model->grads_memory = NULL;
    model->m_memory = NULL;
    model->v_memory = NULL;
    model->grads_acts_memory = NULL;
    model->inputs = NULL;
    model->targets = NULL;
    model->batch_size = 0;
    model->seq_len = 0;
    model->mean_loss = -1.0f; // -1.0f will designate no loss
}

void gpt2_forward(GPT2 *model, int* inputs, int* targets, size_t B, size_t T) {
    // targets are optional and could be NULL

    // ensure the model was initialized or error out
    if (model->params_memory == NULL) {
        printf("Error: model was not initialized properly.\n");
        exit(1);
    }

    // convenience parameters (size_t to help prevent int overflow)
    size_t V = model->config.vocab_size;
    size_t Vp = model->config.padded_vocab_size;
    size_t L = model->config.num_layers;
    size_t NH = model->config.num_heads;
    size_t C = model->config.channels;

    // validate inputs, all indices must be in the range [0, V)
    for(int i = 0; i < B * T; i++) {
        assert(0 <= inputs[i] && inputs[i] < V);
        if (targets != NULL) {
            assert(0 <= targets[i] && targets[i] < V);
        }
    }

    // allocate space for all the activations if needed (done here, lazily)
    if(model->acts_memory == NULL) {
        // record the current B,T as well
        model->batch_size = B;
        model->seq_len = T;
        // and now allocate the space
        fill_in_activation_sizes(model->act_sizes, model->config, B, T);
        size_t num_activations = 0;
        for (size_t i = 0; i < NUM_ACTIVATION_TENSORS; i++) {
            num_activations += model->act_sizes[i];
        }
        printf("num_activations: %zu\n", num_activations);
        model->num_activations = num_activations;
        model->acts_memory = malloc_and_point_activations(&model->acts, model->act_sizes);
        // also create memory for caching inputs and targets
        model->inputs = (int*)mallocCheck(B * T * sizeof(int));
        model->targets = (int*)mallocCheck(B * T * sizeof(int)); // might be unused if we never have targets but it's small
    } else {
        // validate B,T is consistent with how we've allocated the memory before
        // in principle we could get more clever here in the future, for now this is safest
        if (B != model->batch_size || T != model->seq_len) {
            printf("Model: B=%d T=%d, Desired: B=%d T=%d\n", model->batch_size, model->seq_len, (int)B, (int)T);
            exit(EXIT_FAILURE);
        }
    }

    // cache the inputs/targets
    memcpy(model->inputs, inputs, B * T * sizeof(int));
    if (targets != NULL) {
        memcpy(model->targets, targets, B * T * sizeof(int));
    }

    // forward pass
    ParameterTensors params = model->params; // for brevity
    ActivationTensors acts = model->acts;
    float* residual;
    encoder_forward(acts.encoded, inputs, params.wte, params.wpe, B, T, C); // encoding goes into residual[0]
    for (int l = 0; l < L; l++) {

        residual = l == 0 ? acts.encoded : acts.residual3 + (l-1) * B * T * C;

        // get the pointers of the weights for this layer
        float* l_ln1w = params.ln1w + l * C;
        float* l_ln1b = params.ln1b + l * C;
        float* l_qkvw = params.qkvw + l * 3*C * C;
        float* l_qkvb = params.qkvb + l * 3*C;
        float* l_attprojw = params.attprojw + l * C * C;
        float* l_attprojb = params.attprojb + l * C;
        float* l_ln2w = params.ln2w + l * C;
        float* l_ln2b = params.ln2b + l * C;
        float* l_fcw = params.fcw + l * 4*C * C;
        float* l_fcb = params.fcb + l * 4*C;
        float* l_fcprojw = params.fcprojw + l * C * 4*C;
        float* l_fcprojb = params.fcprojb + l * C;

        // get the pointers of the activations for this layer
        float* l_ln1 = acts.ln1 + l * B * T * C;
        float* l_ln1_mean = acts.ln1_mean + l * B * T;
        float* l_ln1_rstd = acts.ln1_rstd + l * B * T;
        float* l_qkv = acts.qkv + l * B * T * 3*C;
        float* l_atty = acts.atty + l * B * T * C;
        float* l_preatt = acts.preatt + l * B * NH * T * T;
        float* l_att = acts.att + l * B * NH * T * T;
        float* l_attproj = acts.attproj + l * B * T * C;
        float* l_residual2 = acts.residual2 + l * B * T * C;
        float* l_ln2 = acts.ln2 + l * B * T * C;
        float* l_ln2_mean = acts.ln2_mean + l * B * T;
        float* l_ln2_rstd = acts.ln2_rstd + l * B * T;
        float* l_fch = acts.fch + l * B * T * 4*C;
        float* l_fch_gelu = acts.fch_gelu + l * B * T * 4*C;
        float* l_fcproj = acts.fcproj + l * B * T * C;
        float* l_residual3 = acts.residual3 + l * B * T * C;

        // now do the forward pass
        layernorm_forward(l_ln1, l_ln1_mean, l_ln1_rstd, residual, l_ln1w, l_ln1b, B, T, C);
        matmul_forward(l_qkv, l_ln1, l_qkvw, l_qkvb, B, T, C, 3*C);
        attention_forward(l_atty, l_preatt, l_att, l_qkv, B, T, C, NH);
        matmul_forward(l_attproj, l_atty, l_attprojw, l_attprojb, B, T, C, C);
        residual_forward(l_residual2, residual, l_attproj, B*T*C);
        layernorm_forward(l_ln2, l_ln2_mean, l_ln2_rstd, l_residual2, l_ln2w, l_ln2b, B, T, C);
        matmul_forward(l_fch, l_ln2, l_fcw, l_fcb, B, T, C, 4*C);
        gelu_forward(l_fch_gelu, l_fch, B*T*4*C);
        matmul_forward(l_fcproj, l_fch_gelu, l_fcprojw, l_fcprojb, B, T, 4*C, C);
        residual_forward(l_residual3, l_residual2, l_fcproj, B*T*C);
    }
    residual = acts.residual3 + (L-1) * B * T * C; // last residual is in residual3
    layernorm_forward(acts.lnf, acts.lnf_mean, acts.lnf_rstd, residual, params.lnfw, params.lnfb, B, T, C);
    matmul_forward(acts.logits, acts.lnf, params.wte, NULL, B, T, C, Vp);
    softmax_forward(acts.probs, acts.logits, B, T, V, Vp);

    // also forward the cross-entropy loss function if we have the targets
    if (targets != NULL) {
        crossentropy_forward(model->acts.losses, model->acts.probs, targets, B, T, Vp);
        // for convenience also evaluate the mean loss
        float mean_loss = 0.0f;
        for (int i=0; i<B*T; i++) { mean_loss += model->acts.losses[i]; }
        mean_loss /= B*T;
        model->mean_loss = mean_loss;
    } else {
        // if we don't have targets, we don't have a loss
        model->mean_loss = -1.0f;
    }
}

void gpt2_zero_grad(GPT2 *model) {
    if(model->grads_memory != NULL) { memset(model->grads_memory, 0, model->num_parameters * sizeof(float)); }
    if(model->grads_acts_memory != NULL) { memset(model->grads_acts_memory, 0, model->num_activations * sizeof(float)); }
}

void gpt2_backward(GPT2 *model) {

    // double check we forwarded previously, with targets
    if (model->mean_loss == -1.0f) {
        printf("Error: must forward with targets before backward\n");
        exit(1);
    }

    // lazily allocate the memory for gradients of the weights and activations, if needed
    if (model->grads_memory == NULL) {
        model->grads_memory = malloc_and_point_parameters(&model->grads, model->param_sizes);
        model->grads_acts_memory = malloc_and_point_activations(&model->grads_acts, model->act_sizes);
        gpt2_zero_grad(model);
    }

    // convenience shortcuts (and size_t to help prevent int overflow)
    size_t B = model->batch_size;
    size_t T = model->seq_len;
    size_t V = model->config.vocab_size;
    size_t Vp = model->config.padded_vocab_size;
    size_t L = model->config.num_layers;
    size_t NH = model->config.num_heads;
    size_t C = model->config.channels;

    // backward pass: go in the reverse order of the forward pass, and call backward() functions
    ParameterTensors params = model->params; // for brevity
    ParameterTensors grads = model->grads;
    ActivationTensors acts = model->acts;
    ActivationTensors grads_acts = model->grads_acts;

    // we kick off the chain rule by filling in dlosses with 1.0f/(B*T)
    // technically this is a small, inline backward() pass of calculating
    // total, final loss as the mean over all losses over all (B,T) positions in the batch
    float dloss_mean = 1.0f / (B*T);
    for (int i = 0; i < B*T; i++) { grads_acts.losses[i] = dloss_mean; }

    crossentropy_softmax_backward(grads_acts.logits, grads_acts.losses, acts.probs, model->targets, B, T, V, Vp);
    matmul_backward(grads_acts.lnf, grads.wte, NULL, grads_acts.logits, acts.lnf, params.wte, B, T, C, Vp);
    float* residual = acts.residual3 + (L-1) * B * T * C; // last layer's residual
    float* dresidual = grads_acts.residual3 + (L-1) * B * T * C; // write to last layer's residual
    layernorm_backward(dresidual, grads.lnfw, grads.lnfb, grads_acts.lnf, residual, params.lnfw, acts.lnf_mean, acts.lnf_rstd, B, T, C);

    for (int l = L-1; l >= 0; l--) {

        residual = l == 0 ? acts.encoded : acts.residual3 + (l-1) * B * T * C;
        dresidual = l == 0 ? grads_acts.encoded : grads_acts.residual3 + (l-1) * B * T * C;

        // get the pointers of the weights for this layer
        float* l_ln1w = params.ln1w + l * C;
        float* l_qkvw = params.qkvw + l * 3*C * C;
        float* l_attprojw = params.attprojw + l * C * C;
        float* l_ln2w = params.ln2w + l * C;
        float* l_fcw = params.fcw + l * 4*C * C;
        float* l_fcprojw = params.fcprojw + l * C * 4*C;
        // get the pointers of the gradients of the weights for this layer
        float* dl_ln1w = grads.ln1w + l * C;
        float* dl_ln1b = grads.ln1b + l * C;
        float* dl_qkvw = grads.qkvw + l * 3*C * C;
        float* dl_qkvb = grads.qkvb + l * 3*C;
        float* dl_attprojw = grads.attprojw + l * C * C;
        float* dl_attprojb = grads.attprojb + l * C;
        float* dl_ln2w = grads.ln2w + l * C;
        float* dl_ln2b = grads.ln2b + l * C;
        float* dl_fcw = grads.fcw + l * 4*C * C;
        float* dl_fcb = grads.fcb + l * 4*C;
        float* dl_fcprojw = grads.fcprojw + l * C * 4*C;
        float* dl_fcprojb = grads.fcprojb + l * C;
        // get the pointers of the activations for this layer
        float* l_ln1 = acts.ln1 + l * B * T * C;
        float* l_ln1_mean = acts.ln1_mean + l * B * T;
        float* l_ln1_rstd = acts.ln1_rstd + l * B * T;
        float* l_qkv = acts.qkv + l * B * T * 3*C;
        float* l_atty = acts.atty + l * B * T * C;
        float* l_att = acts.att + l * B * NH * T * T;
        float* l_residual2 = acts.residual2 + l * B * T * C;
        float* l_ln2 = acts.ln2 + l * B * T * C;
        float* l_ln2_mean = acts.ln2_mean + l * B * T;
        float* l_ln2_rstd = acts.ln2_rstd + l * B * T;
        float* l_fch = acts.fch + l * B * T * 4*C;
        float* l_fch_gelu = acts.fch_gelu + l * B * T * 4*C;
        // get the pointers of the gradients of the activations for this layer
        float* dl_ln1 = grads_acts.ln1 + l * B * T * C;
        float* dl_qkv = grads_acts.qkv + l * B * T * 3*C;
        float* dl_atty = grads_acts.atty + l * B * T * C;
        float* dl_preatt = grads_acts.preatt + l * B * NH * T * T;
        float* dl_att = grads_acts.att + l * B * NH * T * T;
        float* dl_attproj = grads_acts.attproj + l * B * T * C;
        float* dl_residual2 = grads_acts.residual2 + l * B * T * C;
        float* dl_ln2 = grads_acts.ln2 + l * B * T * C;
        float* dl_fch = grads_acts.fch + l * B * T * 4*C;
        float* dl_fch_gelu = grads_acts.fch_gelu + l * B * T * 4*C;
        float* dl_fcproj = grads_acts.fcproj + l * B * T * C;
        float* dl_residual3 = grads_acts.residual3 + l * B * T * C;

        // backprop this layer
        residual_backward(dl_residual2, dl_fcproj, dl_residual3, B*T*C);
        matmul_backward(dl_fch_gelu, dl_fcprojw, dl_fcprojb, dl_fcproj, l_fch_gelu, l_fcprojw, B, T, 4*C, C);
        gelu_backward(dl_fch, l_fch, dl_fch_gelu, B*T*4*C);
        matmul_backward(dl_ln2, dl_fcw, dl_fcb, dl_fch, l_ln2, l_fcw, B, T, C, 4*C);
        layernorm_backward(dl_residual2, dl_ln2w, dl_ln2b, dl_ln2, l_residual2, l_ln2w, l_ln2_mean, l_ln2_rstd, B, T, C);
        residual_backward(dresidual, dl_attproj, dl_residual2, B*T*C);
        matmul_backward(dl_atty, dl_attprojw, dl_attprojb, dl_attproj, l_atty, l_attprojw, B, T, C, C);
        attention_backward(dl_qkv, dl_preatt, dl_att, dl_atty, l_qkv, l_att, B, T, C, NH);
        matmul_backward(dl_ln1, dl_qkvw, dl_qkvb, dl_qkv, l_ln1, l_qkvw, B, T, C, 3*C);
        layernorm_backward(dresidual, dl_ln1w, dl_ln1b, dl_ln1, residual, l_ln1w, l_ln1_mean, l_ln1_rstd, B, T, C);
    }
    encoder_backward(grads.wte, grads.wpe, grads_acts.encoded, model->inputs, B, T, C);
}

void gpt2_update(GPT2 *model, float learning_rate, float beta1, float beta2,
                 float eps, float weight_decay, int t) {
    // lazily allocate the memory for m_memory and v_memory
    if (model->m_memory == NULL) {
        model->m_memory = (float*)calloc(model->num_parameters, sizeof(float));
        model->v_memory = (float*)calloc(model->num_parameters, sizeof(float));
    }

    size_t N = model->num_parameters;

    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < N; i++) {
        float param = model->params_memory[i];
        float grad  = model->grads_memory[i];

        float m = beta1 * model->m_memory[i] + (1.0f - beta1) * grad;
        float v = beta2 * model->v_memory[i] + (1.0f - beta2) * grad * grad;

        float m_hat = m / (1.0f - powf(beta1, t));
        float v_hat = v / (1.0f - powf(beta2, t));

        model->m_memory[i] = m;
        model->v_memory[i] = v;
        model->params_memory[i] -=
            learning_rate * (m_hat / (sqrtf(v_hat) + eps) + weight_decay * param);
    }
}


void gpt2_free(GPT2 *model) {
    free(model->params_memory);
    free(model->grads_memory);
    free(model->m_memory);
    free(model->v_memory);
    free(model->acts_memory);
    free(model->grads_acts_memory);
    free(model->inputs);
    free(model->targets);
}

void print_my_name()
{
    printf("Trained by @\n");
}
#ifndef TESTING
// if we are TESTING (see test_gpt2.c), we'll skip the int main below
// ----------------------------------------------------------------------------
// sampler

unsigned int random_u32(uint64_t *state) {
    // xorshift rng: https://en.wikipedia.org/wiki/Xorshift#xorshift.2A
    *state ^= *state >> 12;
    *state ^= *state << 25;
    *state ^= *state >> 27;
    return (*state * 0x2545F4914F6CDD1Dull) >> 32;
}
float random_f32(uint64_t *state) { // random float32 in [0,1)
    return (random_u32(state) >> 8) / 16777216.0f;
}

int sample_mult(float* probabilities, int n, float coin) {
    // sample index from probabilities (they must sum to 1!)
    // coin is a random number in [0, 1), usually from random_f32()
    float cdf = 0.0f;
    for (int i = 0; i < n; i++) {
        cdf += probabilities[i];
        if (coin < cdf) {
            return i;
        }
    }
    return n - 1; // in case of rounding errors
}
#endif
