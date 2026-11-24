// FMCompute.hlsl - Xbox Series S/X D3D12 compute shaders for GPU-resident OT-FM training and dreaming.
//
// Every kernel below is compiled to its own .cso by fxc (profile cs_5_1, entry point = file name).
// The C++ side (FMBrain.cpp) binds ALL kernels through ONE root signature:
//
//   root[0]      26 x 32-bit constants -> cbuffer TrainConstants (b0)   (must match GPUTrainConstants in FMBrain.h)
//   root[1..4]   root SRVs             -> t0, t1, t2, t3
//   root[5..7]   root UAVs             -> u0, u1, u2
//
// Tensor layout is NCHW, float32.  Weights of a conv layer are [outC, inC, K, K], bias is [outC].
//
// Rules every kernel follows (so a bad launch can never touch memory outside its buffers):
//   * thread-group sizes are 64 or 256 threads (Xbox-friendly, no TDR-prone giant groups)
//   * every thread bounds-checks itself against the constants before reading or writing
//   * no early `return` ahead of a group barrier (barriers are reached by every thread)

cbuffer TrainConstants : register(b0) {
    uint  gBatch;
    uint  gWidth;
    uint  gHeight;
    uint  gInC;
    uint  gOutC;
    uint  gKernelSize;     // 1 or 3
    uint  gApplySiLU;      // ConvForward: 1 = out = SiLU(acc) and pre-activation is stored, 0 = linear
    uint  gAddResidual;    // ConvForward / ConvBackwardInput: 1 = add the residual tensor (t3)
    uint  gSiLUGradEpi;    // ConvBackwardInput: 1 = multiply the result by SiLU'(pre-activation) (t2)
    uint  gUseConstT;      // EncodeSine: 1 = use gTimeT for every sample, 0 = read t from tValues (t1)
    uint  gParamCount;     // element count for the 1-D elementwise / Adam / reduction kernels
    uint  gSampleStride;   // image channels * W * H   (PairwiseCost only)
    uint  gPartialCount;   // number of partial sums per reduction range
    uint  gPartialOffset;  // where SumSquaresPartial writes its partial sums
    float gTimeT;
    float gDt;
    float gLearningRate;
    float gBeta1;
    float gBeta2;
    float gEpsilon;
    float gBiasCorr1;      // 1 - beta1^step
    float gBiasCorr2;      // 1 - beta2^step
    float gInvOutCount;    // 1 / (N * image channels * W * H)
    float gGradClipNorm;   // <= 0 disables clipping
    uint  gMode;           // CSEncodeSineOperators: 0 = sine encoding (the default), 1..4 = per-pixel operations of the one-pass brain (see 2b)
    float gKlScale;        // mode 4: weight of the KL term per code number (klWeight / (N * code channels * H * W))
};

StructuredBuffer<float>   gInA     : register(t0);
StructuredBuffer<float>   gInB     : register(t1);
StructuredBuffer<float>   gInC_Buf : register(t2);
StructuredBuffer<float>   gInD     : register(t3);
RWStructuredBuffer<float> gOutA    : register(u0);
RWStructuredBuffer<float> gOutB    : register(u1);
RWStructuredBuffer<float> gOutBufC : register(u2);

static const float PI = 3.14159265359f;

// Linear (1-D) dispatches are folded into a 2-D grid so that very large tensors never need more than
// 65535 thread groups in X.  The C++ side always launches at most 32768 groups of 64 threads in X.
static const uint kLinearRowThreads = 2097152u;   // 32768 groups * 64 threads

inline uint LinearIndex(uint3 tid) {
    return tid.x + tid.y * kLinearRowThreads;
}

inline float EvalSiLU(float x) {
    float c = clamp(x, -20.0f, 20.0f);
    return c / (1.0f + exp(-c));
}

inline float EvalSiLUGrad(float x) {
    float c = clamp(x, -20.0f, 20.0f);
    float s = 1.0f / (1.0f + exp(-c));
    return s * (1.0f + c * (1.0f - s));
}

// ---------------------------------------------------------------------------------------------
// 1. Pairwise squared-L2 cost matrix  C[i,j] = ||x0[i] - x1[j]||^2 / stride
//    t0 = x0 [N, stride], t1 = x1 [N, stride], u0 = C [N, N]
//    NOTE: available for experiments, not dispatched by the runtime today (the Hungarian solve that
//    consumes this matrix is sequential and the matrix is tiny, so the CPU keeps doing OT pairing).
// ---------------------------------------------------------------------------------------------
[numthreads(8, 8, 1)]
void CSPairwiseCost(uint3 tid : SV_DispatchThreadID) {
    uint i = tid.x;
    uint j = tid.y;
    if (i >= gBatch || j >= gBatch) return;

    uint baseA = i * gSampleStride;
    uint baseB = j * gSampleStride;
    float sumSq = 0.0f;
    for (uint k = 0; k < gSampleStride; ++k) {
        float diff = gInA[baseA + k] - gInB[baseB + k];
        sumSq += diff * diff;
    }
    gOutA[i * gBatch + j] = sumSq / max(1.0f, (float)gSampleStride);
}

// ---------------------------------------------------------------------------------------------
// 2b. Per-pixel operations of the one-pass (multi-zoom) brain. They live inside the program CSEncodeSineOperators (see 2) so the
//     list of programs the build compiles stays the same: gMode picks the operation (0 = the sine encoding below).
//     The threads walk over the grid gWidth x gHeight; for modes 1 and 2 that is the SMALL grid (half the picture width and height).
//     mode 1  space-to-depth   t0 = [N, C, 2H, 2W]  ->  u0 = [N, 4C, H, W]       (C = gInC; channel c*4 + dy*2 + dx)
//     mode 2  depth-to-space   t0 = [N, 4C, H, W]   ->  u0 = [N, C, 2H, 2W]      (the exact inverse of mode 1)
//     mode 3  reparameterise   t0 = code [N, 2Z, H, W] (Z averages, then Z log-variances)  t1 = noise [N, Z, H, W]
//                              ->  u0 = [N, Z, H, W] with  average + exp(logvariance / 2) * noise      (Z = gInC)
//     mode 4  reparameterise, backward   t0 = code  t1 = noise  t2 = gradient of the result
//                              ->  u0 = gradient of the code [N, 2Z, H, W], including the KL term (weight gKlScale)
//     The log-variance is limited to [-30, 20] (no gradient outside), exactly like the processor version in FMBrain.cpp.
// ---------------------------------------------------------------------------------------------
inline void PixelSpaceToDepth(uint x, uint y, uint n) {
    uint lowW = gWidth;
    uint highW = gWidth * 2;
    uint lowHW = gWidth * gHeight;
    uint highHW = highW * gHeight * 2;
    uint p = y * lowW + x;
    uint srcBase = n * gInC * highHW;
    uint dstBase = n * gInC * 4 * lowHW;
    for (uint c = 0; c < gInC; ++c) {
        for (uint q = 0; q < 4; ++q) {
            uint srcPos = (2 * y + (q >> 1)) * highW + 2 * x + (q & 1);
            gOutA[dstBase + (c * 4 + q) * lowHW + p] = gInA[srcBase + c * highHW + srcPos];
        }
    }
}

inline void PixelDepthToSpace(uint x, uint y, uint n) {
    uint lowW = gWidth;
    uint highW = gWidth * 2;
    uint lowHW = gWidth * gHeight;
    uint highHW = highW * gHeight * 2;
    uint p = y * lowW + x;
    uint srcBase = n * gInC * 4 * lowHW;
    uint dstBase = n * gInC * highHW;
    for (uint c = 0; c < gInC; ++c) {
        for (uint q = 0; q < 4; ++q) {
            uint dstPos = (2 * y + (q >> 1)) * highW + 2 * x + (q & 1);
            gOutA[dstBase + c * highHW + dstPos] = gInA[srcBase + (c * 4 + q) * lowHW + p];
        }
    }
}

inline void PixelReparameterise(uint x, uint y, uint n) {
    uint hw = gWidth * gHeight;
    uint p = y * gWidth + x;
    uint codeBase = n * 2 * gInC * hw;
    uint zBase = n * gInC * hw;
    for (uint c = 0; c < gInC; ++c) {
        float avg = gInA[codeBase + c * hw + p];
        float lv = clamp(gInA[codeBase + (gInC + c) * hw + p], -30.0f, 20.0f);
        float rnd = gInB[zBase + c * hw + p];
        gOutA[zBase + c * hw + p] = avg + exp(0.5f * lv) * rnd;
    }
}

inline void PixelReparameteriseBackward(uint x, uint y, uint n) {
    uint hw = gWidth * gHeight;
    uint p = y * gWidth + x;
    uint codeBase = n * 2 * gInC * hw;
    uint zBase = n * gInC * hw;
    for (uint c = 0; c < gInC; ++c) {
        float avg = gInA[codeBase + c * hw + p];
        float lvRaw = gInA[codeBase + (gInC + c) * hw + p];
        float lv = clamp(lvRaw, -30.0f, 20.0f);
        float rnd = gInB[zBase + c * hw + p];
        float gz = gInC_Buf[zBase + c * hw + p];
        float sd = exp(0.5f * lv);
        float live = (lvRaw > -30.0f && lvRaw < 20.0f) ? 1.0f : 0.0f;
        gOutA[codeBase + c * hw + p] = gz + gKlScale * avg;
        gOutA[codeBase + (gInC + c) * hw + p] = live * (0.5f * sd * gz * rnd + 0.5f * gKlScale * (sd * sd - 1.0f));
    }
}

// ---------------------------------------------------------------------------------------------
// 2. Sine-operator encoder: x_t image channels + [-1,1] grid + harmonics k=0..3 + time embedding
//    gInC = number of image channels (3 for pixels, 12 for the Haar wave form), gOutC = gInC + 22 channels in total
//    t0 = x_t [N,gInC,H,W]   t1 = tValues [N] (unused when gUseConstT)   u0 = encoded [N,gOutC,H,W]
//    (gMode != 0 selects one of the per-pixel operations of 2b instead)
// ---------------------------------------------------------------------------------------------
[numthreads(8, 8, 1)]
void CSEncodeSineOperators(uint3 tid : SV_DispatchThreadID) {
    uint x = tid.x;
    uint y = tid.y;
    uint n = tid.z;
    if (x >= gWidth || y >= gHeight || n >= gBatch) return;

    if (gMode == 1) { PixelSpaceToDepth(x, y, n); return; }
    if (gMode == 2) { PixelDepthToSpace(x, y, n); return; }
    if (gMode == 3) { PixelReparameterise(x, y, n); return; }
    if (gMode == 4) { PixelReparameteriseBackward(x, y, n); return; }

    uint hw = gWidth * gHeight;
    uint p = y * gWidth + x;
    uint imgC = gInC;
    uint cIn = gOutC;
    uint dstBase = n * cIn * hw;
    uint srcBase = n * imgC * hw;

    for (uint c = 0; c < imgC; ++c) {
        gOutA[dstBase + c * hw + p] = gInA[srcBase + c * hw + p];
    }

    float nx = (gWidth > 1)  ? (2.0f * (float)x / (float)(gWidth - 1)  - 1.0f) : 0.0f;
    float ny = (gHeight > 1) ? (2.0f * (float)y / (float)(gHeight - 1) - 1.0f) : 0.0f;
    gOutA[dstBase + (imgC + 0) * hw + p] = nx;
    gOutA[dstBase + (imgC + 1) * hw + p] = ny;

    [unroll]
    for (uint k = 0; k < 4; ++k) {
        float freq = (float)(1u << k) * PI;
        uint ch = imgC + 2 + k * 4;
        gOutA[dstBase + (ch + 0) * hw + p] = sin(freq * nx);
        gOutA[dstBase + (ch + 1) * hw + p] = cos(freq * nx);
        gOutA[dstBase + (ch + 2) * hw + p] = sin(freq * ny);
        gOutA[dstBase + (ch + 3) * hw + p] = cos(freq * ny);
    }

    float t = (gUseConstT != 0) ? gTimeT : gInB[n];
    uint tb = imgC + 18;
    gOutA[dstBase + (tb + 0) * hw + p] = sin(PI * t);
    gOutA[dstBase + (tb + 1) * hw + p] = cos(PI * t);
    gOutA[dstBase + (tb + 2) * hw + p] = sin(2.0f * PI * t);
    gOutA[dstBase + (tb + 3) * hw + p] = cos(2.0f * PI * t);
}

// ---------------------------------------------------------------------------------------------
// 3. Conv2D forward (1x1 or 3x3, zero padding) with fused epilogue.
//    t0 = input [N,inC,H,W]   t1 = weights [outC,inC,K,K]   t2 = bias [outC]   t3 = residual [N,outC,H,W]
//    u0 = output [N,outC,H,W]  u1 = pre-activation [N,outC,H,W] (written only when gApplySiLU != 0)
//    out = (gApplySiLU ? SiLU(acc) : acc) + (gAddResidual ? residual : 0)
//    Dispatch: ceil(W/8) x ceil(H/8) x (N * outC)
// ---------------------------------------------------------------------------------------------
[numthreads(8, 8, 1)]
void CSConvForward(uint3 tid : SV_DispatchThreadID) {
    uint x = tid.x;
    uint y = tid.y;
    uint noc = tid.z;
    uint n = noc / gOutC;
    uint oc = noc % gOutC;
    if (x >= gWidth || y >= gHeight || n >= gBatch) return;

    uint hw = gWidth * gHeight;
    uint inBatchBase = n * gInC * hw;
    uint outIdx = (n * gOutC + oc) * hw + (y * gWidth + x);
    float acc = gInC_Buf[oc];

    if (gKernelSize == 1) {
        uint p = y * gWidth + x;
        for (uint ic = 0; ic < gInC; ++ic) {
            acc += gInA[inBatchBase + ic * hw + p] * gInB[oc * gInC + ic];
        }
    } else {
        for (uint ic = 0; ic < gInC; ++ic) {
            uint inPlane = inBatchBase + ic * hw;
            uint wBase = (oc * gInC + ic) * 9;
            [unroll]
            for (int ky = 0; ky < 3; ++ky) {
                int iy = (int)y + ky - 1;
                if (iy < 0 || iy >= (int)gHeight) continue;
                [unroll]
                for (int kx = 0; kx < 3; ++kx) {
                    int ix = (int)x + kx - 1;
                    if (ix < 0 || ix >= (int)gWidth) continue;
                    acc += gInA[inPlane + (uint)iy * gWidth + (uint)ix] * gInB[wBase + (uint)ky * 3 + (uint)kx];
                }
            }
        }
    }

    float result = acc;
    if (gApplySiLU != 0) {
        gOutB[outIdx] = acc;
        result = EvalSiLU(acc);
    }
    if (gAddResidual != 0) {
        result += gInD[outIdx];
    }
    gOutA[outIdx] = result;
}

// ---------------------------------------------------------------------------------------------
// 4. MSE loss gradient:  dL/dv_pred = 2 * (v_pred - v_target) / (N*3*H*W)
//    t0 = v_pred, t1 = v_target, u0 = gradPred          (1-D, gParamCount elements)
// ---------------------------------------------------------------------------------------------
[numthreads(64, 1, 1)]
void CSMSELossGrad(uint3 tid : SV_DispatchThreadID) {
    uint idx = LinearIndex(tid);
    if (idx >= gParamCount) return;
    float diff = gInA[idx] - gInB[idx];
    gOutA[idx] = 2.0f * gInvOutCount * diff;
}

// ---------------------------------------------------------------------------------------------
// 5. Conv2D backward w.r.t. the INPUT (transposed convolution) with fused epilogue.
//    t0 = gradOutput [N,outC,H,W]  t1 = weights [outC,inC,K,K]  t2 = preActivation [N,inC,H,W]
//    t3 = residual [N,inC,H,W]     u0 = gradInput [N,inC,H,W]
//    result = acc + (gAddResidual ? residual : 0);   if gSiLUGradEpi: result *= SiLU'(preActivation)
//    Dispatch: ceil(W/8) x ceil(H/8) x (N * inC)
// ---------------------------------------------------------------------------------------------
[numthreads(8, 8, 1)]
void CSConvBackwardInput(uint3 tid : SV_DispatchThreadID) {
    uint x = tid.x;
    uint y = tid.y;
    uint nic = tid.z;
    uint n = nic / gInC;
    uint ic = nic % gInC;
    if (x >= gWidth || y >= gHeight || n >= gBatch) return;

    uint hw = gWidth * gHeight;
    uint outBatchBase = n * gOutC * hw;
    uint inIdx = (n * gInC + ic) * hw + (y * gWidth + x);
    float acc = 0.0f;

    if (gKernelSize == 1) {
        uint p = y * gWidth + x;
        for (uint oc = 0; oc < gOutC; ++oc) {
            acc += gInA[outBatchBase + oc * hw + p] * gInB[oc * gInC + ic];
        }
    } else {
        for (uint oc = 0; oc < gOutC; ++oc) {
            uint outPlane = outBatchBase + oc * hw;
            uint wBase = (oc * gInC + ic) * 9;
            [unroll]
            for (int ky = 0; ky < 3; ++ky) {
                int oy = (int)y - ky + 1;
                if (oy < 0 || oy >= (int)gHeight) continue;
                [unroll]
                for (int kx = 0; kx < 3; ++kx) {
                    int ox = (int)x - kx + 1;
                    if (ox < 0 || ox >= (int)gWidth) continue;
                    acc += gInA[outPlane + (uint)oy * gWidth + (uint)ox] * gInB[wBase + (uint)ky * 3 + (uint)kx];
                }
            }
        }
    }

    if (gAddResidual != 0) {
        acc += gInD[inIdx];
    }
    if (gSiLUGradEpi != 0) {
        acc *= EvalSiLUGrad(gInC_Buf[inIdx]);
    }
    gOutA[inIdx] = acc;
}

// ---------------------------------------------------------------------------------------------
// 5b. Elementwise SiLU backward:  u0[i] = t0[i] * SiLU'(t1[i])      (gradPostAct, preAct -> gradPreAct)
// ---------------------------------------------------------------------------------------------
[numthreads(64, 1, 1)]
void CSSiLUBackward(uint3 tid : SV_DispatchThreadID) {
    uint idx = LinearIndex(tid);
    if (idx >= gParamCount) return;
    gOutA[idx] = gInA[idx] * EvalSiLUGrad(gInB[idx]);
}

// ---------------------------------------------------------------------------------------------
// 6. Conv2D backward w.r.t. WEIGHTS and BIAS (parallel reduction, one thread group per (oc, ic) pair).
//    t0 = layer input [N,inC,H,W]   t1 = gradOutput [N,outC,H,W]
//    u1 = gradWeights [outC,inC,K,K] (written, not accumulated)   u2 = gradBias [outC] (written by ic == 0 groups)
//    The group makes one pass over all N*H*W positions per kernel tap (1 pass for 1x1, 9 passes for 3x3).
//    In a pass each of the 256 threads walks a strided share of the positions and keeps ONE running sum,
//    then the group tree-reduces the sums.  The pass over the centre tap (offset 0,0) visits every position,
//    so it also collects the bias gradient (sum of gradOutput) at no extra cost.
//    Written in the plainest possible HLSL (no local arrays, no 2-D groupshared arrays, no unrolled barriers),
//    because fxc's optimiser hit an internal error on an earlier, more elaborate version of this kernel.
//    Dispatch: the (oc, ic) pairs are numbered gid.x + gid.y * 32768 (so more than 65535 pairs are possible):
//    gx = min(pairs, 32768), gy = ceil(pairs / 32768); groups past the last pair do nothing.
// ---------------------------------------------------------------------------------------------
groupshared float sRedW[256];
groupshared float sRedB[256];

[numthreads(256, 1, 1)]
void CSConvBackwardWeights(uint3 gid : SV_GroupID, uint3 gtid : SV_GroupThreadID) {
    uint pair = gid.x + gid.y * 32768u;
    uint t = gtid.x;
    bool valid = (pair < gOutC * gInC);

    uint oc = valid ? (pair / gInC) : 0u;
    uint ic = valid ? (pair % gInC) : 0u;
    uint kk = gKernelSize * gKernelSize;
    int  pad = (int)(gKernelSize / 2);
    uint hw = gWidth * gHeight;

    for (uint tap = 0; tap < kk; ++tap) {
        int dy = (int)(tap / gKernelSize) - pad;
        int dx = (int)(tap % gKernelSize) - pad;
        bool centre = (dy == 0 && dx == 0);

        float acc = 0.0f;
        float bAcc = 0.0f;
        if (valid) {
            for (uint n = 0; n < gBatch; ++n) {
                uint goBase = (n * gOutC + oc) * hw;
                uint inBase = (n * gInC + ic) * hw;
                for (uint p = t; p < hw; p += 256) {
                    uint y = p / gWidth;
                    uint x = p - y * gWidth;
                    int iy = (int)y + dy;
                    int ix = (int)x + dx;
                    if (iy >= 0 && iy < (int)gHeight && ix >= 0 && ix < (int)gWidth) {
                        float go = gInB[goBase + p];
                        acc += go * gInA[inBase + (uint)iy * gWidth + (uint)ix];
                        if (centre) bAcc += go;
                    }
                }
            }
        }

        sRedW[t] = acc;
        sRedB[t] = bAcc;
        GroupMemoryBarrierWithGroupSync();

        for (uint stride = 128; stride > 0; stride >>= 1) {
            if (t < stride) {
                sRedW[t] += sRedW[t + stride];
                sRedB[t] += sRedB[t + stride];
            }
            GroupMemoryBarrierWithGroupSync();
        }

        // Only thread 0 reads index 0 here and only thread 0 rewrites it in the next pass, so no extra barrier is needed.
        if (valid && t == 0) {
            gOutB[(oc * gInC + ic) * kk + tap] = sRedW[0];
            if (centre && ic == 0) {
                gOutBufC[oc] = sRedB[0];
            }
        }
    }
}

// ---------------------------------------------------------------------------------------------
// 6b. EMERGENCY TWIN of CSConvBackwardWeights (same bindings, same dispatch, same results, much slower).
//     Not used normally.  build-xbox.yml compiles it under the name CSConvBackwardWeights.cso ONLY if fxc
//     cannot build the fast kernel above.  No groupshared memory and no barriers: thread `tap` of each group
//     owns kernel tap `tap` and walks every position on its own; the centre tap also collects the bias.
// ---------------------------------------------------------------------------------------------
[numthreads(64, 1, 1)]
void CSConvBackwardWeightsLite(uint3 gid : SV_GroupID, uint3 gtid : SV_GroupThreadID) {
    uint pair = gid.x + gid.y * 32768u;
    uint tap = gtid.x;
    uint kk = gKernelSize * gKernelSize;
    if (pair >= gOutC * gInC || tap >= kk) return;

    uint oc = pair / gInC;
    uint ic = pair % gInC;
    int  pad = (int)(gKernelSize / 2);
    int  dy = (int)(tap / gKernelSize) - pad;
    int  dx = (int)(tap % gKernelSize) - pad;
    bool centre = (dy == 0 && dx == 0);
    uint hw = gWidth * gHeight;

    float acc = 0.0f;
    float bAcc = 0.0f;
    for (uint n = 0; n < gBatch; ++n) {
        uint goBase = (n * gOutC + oc) * hw;
        uint inBase = (n * gInC + ic) * hw;
        for (uint y = 0; y < gHeight; ++y) {
            int iy = (int)y + dy;
            if (iy < 0 || iy >= (int)gHeight) continue;
            for (uint x = 0; x < gWidth; ++x) {
                int ix = (int)x + dx;
                if (ix < 0 || ix >= (int)gWidth) continue;
                float go = gInB[goBase + y * gWidth + x];
                acc += go * gInA[inBase + (uint)iy * gWidth + (uint)ix];
                if (centre) bAcc += go;
            }
        }
    }
    gOutB[(oc * gInC + ic) * kk + tap] = acc;
    if (centre && ic == 0) {
        gOutBufC[oc] = bAcc;
    }
}

// ---------------------------------------------------------------------------------------------
// 7. Sum of squares, stage 1:  u0[gPartialOffset + g] = sum over i = g, g + P, g + 2P ... of t0[i]^2
//    where P = gPartialCount (the total thread count of the dispatch, 64 groups x 64 threads).
// ---------------------------------------------------------------------------------------------
[numthreads(64, 1, 1)]
void CSSumSquaresPartial(uint3 tid : SV_DispatchThreadID) {
    uint g = tid.x;
    if (g >= gPartialCount) return;
    float sum = 0.0f;
    for (uint i = g; i < gParamCount; i += gPartialCount) {
        float v = gInA[i];
        sum += v * v;
    }
    gOutA[gPartialOffset + g] = sum;
}

// ---------------------------------------------------------------------------------------------
// 7b. Sum of squares, stage 2 (one group):  t0 = partials [2 * P]   u0 = stats[4]
//     range A = [0, P)  : squared gradient norm
//     range B = [P, 2P) : sum of squared gradPred entries (the loss)
//     stats[0] = ||grad||^2    stats[1] = Adam gradient scale (0 = non-finite gradient, skip the update)
//     stats[2] = MSE loss      stats[3] = ||grad||
// ---------------------------------------------------------------------------------------------
groupshared float sFinA[64];
groupshared float sFinB[64];

[numthreads(64, 1, 1)]
void CSReduceFinalize(uint3 gtid : SV_GroupThreadID) {
    uint t = gtid.x;
    float a = 0.0f;
    float b = 0.0f;
    for (uint i = t; i < gPartialCount; i += 64) {
        a += gInA[i];
        b += gInA[gPartialCount + i];
    }
    sFinA[t] = a;
    sFinB[t] = b;
    GroupMemoryBarrierWithGroupSync();

    for (uint stride = 32; stride > 0; stride >>= 1) {
        if (t < stride) {
            sFinA[t] += sFinA[t + stride];
            sFinB[t] += sFinB[t + stride];
        }
        GroupMemoryBarrierWithGroupSync();
    }

    if (t == 0) {
        float normSq = sFinA[0];
        float norm = sqrt(max(normSq, 0.0f));
        // non-finite (NaN / Inf) gradient -> scale 0 so the Adam kernel skips the whole update
        // (bit test instead of x == x, so the compiler can never fold the NaN check away)
        bool gradFinite = ((asuint(normSq) & 0x7F800000u) != 0x7F800000u);
        float scale = 1.0f;
        if (gGradClipNorm > 0.0f && norm > gGradClipNorm && norm > 1e-6f) {
            scale = gGradClipNorm / norm;
        }
        gOutA[0] = normSq;
        gOutA[1] = gradFinite ? scale : 0.0f;
        gOutA[2] = sFinB[0] * 0.25f / max(gInvOutCount, 1e-30f);   // sum(g^2) = 4 * invCount^2 * sum(diff^2)
        gOutA[3] = norm;
    }
}

// ---------------------------------------------------------------------------------------------
// 7c. EMERGENCY TWIN of CSReduceFinalize (same bindings, same results up to summation order, a little slower).
//     Used only if fxc cannot build the kernel above (see build-xbox.yml).  No groupshared memory, no barriers:
//     thread 0 adds everything up serially.
// ---------------------------------------------------------------------------------------------
[numthreads(64, 1, 1)]
void CSReduceFinalizeLite(uint3 gtid : SV_GroupThreadID) {
    if (gtid.x != 0) return;
    float a = 0.0f;
    float b = 0.0f;
    for (uint i = 0; i < gPartialCount; ++i) {
        a += gInA[i];
        b += gInA[gPartialCount + i];
    }
    float normSq = a;
    float norm = sqrt(max(normSq, 0.0f));
    bool gradFinite = ((asuint(normSq) & 0x7F800000u) != 0x7F800000u);
    float scale = 1.0f;
    if (gGradClipNorm > 0.0f && norm > gGradClipNorm && norm > 1e-6f) {
        scale = gGradClipNorm / norm;
    }
    gOutA[0] = normSq;
    gOutA[1] = gradFinite ? scale : 0.0f;
    gOutA[2] = b * 0.25f / max(gInvOutCount, 1e-30f);
    gOutA[3] = norm;
}

// ---------------------------------------------------------------------------------------------
// 8. Adam update over a whole parameter arena (weights + biases of every layer, padding stays zero).
//    t0 = grad arena   t1 = stats (stats[1] = gradient scale; 0 -> skip)
//    u0 = param arena (in place)   u1 = m (in place)   u2 = v (in place)
// ---------------------------------------------------------------------------------------------
[numthreads(64, 1, 1)]
void CSAdamUpdate(uint3 tid : SV_DispatchThreadID) {
    uint idx = LinearIndex(tid);
    if (idx >= gParamCount) return;

    float scale = gInB[1];
    if (!(scale > 0.0f)) return;

    float g = gInA[idx] * scale;
    float m = gBeta1 * gOutB[idx] + (1.0f - gBeta1) * g;
    float v = gBeta2 * gOutBufC[idx] + (1.0f - gBeta2) * g * g;
    gOutB[idx] = m;
    gOutBufC[idx] = v;

    float mHat = m / max(1e-8f, gBiasCorr1);
    float vHat = v / max(1e-8f, gBiasCorr2);
    gOutA[idx] -= gLearningRate * mHat / (sqrt(vHat) + gEpsilon);
}

// ---------------------------------------------------------------------------------------------
// 9. Euler ODE step:  x_{t+dt} = x_t + dt * v_pred        t0 = v_pred   u0 = x (in place)
// ---------------------------------------------------------------------------------------------
[numthreads(64, 1, 1)]
void CSEulerStep(uint3 tid : SV_DispatchThreadID) {
    uint idx = LinearIndex(tid);
    if (idx >= gParamCount) return;
    gOutA[idx] = gOutA[idx] + gDt * gInA[idx];
}
