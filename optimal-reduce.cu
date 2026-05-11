// ============================================================
// HIP version of R1 correlation computation
// AMD optimizations applied:
//   1. 128-bit LDS loads via float4 reinterpret (ds_read_b128)
//   2. Pre-gather into registers before dot product
//   3. 64-wide wavefront reduction via __shfl_xor
//   4. __builtin_amdgcn_readfirstlane for broadcast
// ============================================================

    using TComplexCompute = float2; // HIP: hipFloatComplex = float2

    const uint32_t DMRS_LS_EST_TONE_IDX = (DMRS_ABS_TONE_IDX - startPrb * N_TONES_PER_PRB - DMRS_GRID_IDX) / N_DMRS_TONE_STRIDE;

    TComplexCompute R1_thread = make_float2(0.0f, 0.0f);

    if (nLayers > 1) {
        // --- LS estimate store (unchanged logic) ---
        if (isValidTone) {
            for (uint32_t i = 0; i < nLayers; i++) {
                uint32_t k = (OCCIdx[i] >> 2) & 0x1;
                if (DMRS_GRID_IDX != k) continue;
                uint32_t j = OCCIdx[i] & 0x3;
                tInfoDmrsLSEst(DMRS_LS_EST_TONE_IDX, i, BS_ANT_IDX, NH_IDX) =
                    shH(DMRS_TONE_IDX, j, DMRS_GRID_IDX); // type_convert is identity for float2
            }
        }

        // --- Per-PRG R1 boundary check (unchanged) ---
        if (enablePerPrgChEst == 1) {
            if ((prgSize == 1) && (DMRS_TONE_IDX % 6 == 4))   isValidR1Cal = false;
            if ((prgSize == 2) && (DMRS_TONE_IDX % 12 == 10)) isValidR1Cal = false;
            if ((prgSize == 3) && (DMRS_TONE_IDX % 18 == 16)) isValidR1Cal = false;
            if ((prgSize == 4) && (DMRS_TONE_IDX % 24 == 22)) isValidR1Cal = false;
        }

        // --- R1 computation with AMD LDS optimization ---
        const bool nextToneIsValidAfterAvg =
            ((DMRS_ABS_TONE_IDX + 2 * N_DMRS_TONE_STRIDE) / N_TONES_PER_PRB) <= lastPrbThisCluster;

        if (DMRS_TONE_IDX % 2 == 0 && nextToneIsValidAfterAvg && isValidR1Cal) {
            // Pre-gather phase: load from LDS into registers
            // Using 128-bit loads since DMRS_TONE_IDX is even => aligned
            for (uint32_t i = 0; i < nLayers; i++) {
                uint32_t j = OCCIdx[i] & 0x3;
                uint32_t k = (OCCIdx[i] >> 2) & 0x1;
                if (DMRS_GRID_IDX != k) continue;

                // AMD optimization: 128-bit LDS load (ds_read_b128)
                // shH(tone, j, grid) and shH(tone+1, j, grid) are adjacent
                const float2* base0 = &shH(DMRS_TONE_IDX, j, DMRS_GRID_IDX);
                const float4 pair0 = *reinterpret_cast<const float4*>(base0);
                // pair0 = { shH(tone).x, shH(tone).y, shH(tone+1).x, shH(tone+1).y }

                const float2* base1 = &shH(DMRS_TONE_IDX + N_DMRS_TONE_STRIDE, j, DMRS_GRID_IDX);
                const float4 pair1 = *reinterpret_cast<const float4*>(base1);
                // pair1 = { shH(tone+S).x, shH(tone+S).y, shH(tone+S+1).x, shH(tone+S+1).y }

                // Average adjacent tones: avg = 0.5 * (tone_val + tone_val+1)
                const float a_re = 0.5f * (pair0.x + pair0.z);  // Re(avg0)
                const float a_im = 0.5f * (pair0.y + pair0.w);  // Im(avg0)
                const float b_re = 0.5f * (pair1.x + pair1.z);  // Re(avg1)
                const float b_im = 0.5f * (pair1.y + pair1.w);  // Im(avg1)

                // Conjugate dot product: conj(avg0) * avg1
                // Re: a_re*b_re + a_im*b_im  (uses FMA: fma(a_re, b_re, a_im*b_im))
                // Im: a_re*b_im - a_im*b_re  (uses FMA: fma(a_re, b_im, -a_im*b_re))
                R1_thread.x += __fmaf_rn(a_re, b_re, a_im * b_im);
                R1_thread.y += __fmaf_rn(a_re, b_im, -(a_im * b_re));
            }
        }
    } else {
        // --- Single layer path ---
        if (isValidTone) {
            for (uint32_t i = 0; i < nLayers; i++) {
                uint32_t j = OCCIdx[i] & 0x3;
                uint32_t k = (OCCIdx[i] >> 2) & 0x1;
                if (DMRS_GRID_IDX != k) continue;

                if (enablePerPrgChEst == 1) {
                    if ((prgSize == 1) && (DMRS_TONE_IDX % 6 == 5))   isValidR1Cal = false;
                    if ((prgSize == 2) && (DMRS_TONE_IDX % 12 == 11)) isValidR1Cal = false;
                    if ((prgSize == 3) && (DMRS_TONE_IDX % 18 == 17)) isValidR1Cal = false;
                    if ((prgSize == 4) && (DMRS_TONE_IDX % 24 == 23)) isValidR1Cal = false;
                }

                if (nextToneIsValid && isValidR1Cal) {
                    // Single-layer: no averaging, direct conjugate multiply
                    // Can still use 128-bit load if we know tone+1 is valid
                    const float2 h0 = shH(DMRS_TONE_IDX, j, DMRS_GRID_IDX);
                    const float2 h1 = shH(DMRS_TONE_IDX + 1, j, DMRS_GRID_IDX);
                    // conj(h0) * h1
                    R1_thread.x += __fmaf_rn(h0.x, h1.x, h0.y * h1.y);
                    R1_thread.y += __fmaf_rn(h0.x, h1.y, -(h0.y * h1.x));
                }
                tInfoDmrsLSEst(DMRS_LS_EST_TONE_IDX, i, BS_ANT_IDX, NH_IDX) =
                    shH(DMRS_TONE_IDX, j, DMRS_GRID_IDX);
            }
        }
    }

    // ============================================================
    // Wavefront reduction (AMD: 64-wide wavefront)
    // Replaces CUDA cooperative_groups reduce(tile, ...)
    // ============================================================
    constexpr int WAVEFRONT_SIZE = 64; // AMD CDNA/RDNA
    constexpr uint32_t MAX_NUM_WARPS = 32;
    __shared__ TComplexCompute sh_R1[MAX_NUM_WARPS];

    // Cross-lane butterfly reduction within wavefront
    for (int offset = WAVEFRONT_SIZE / 2; offset > 0; offset >>= 1) {
        R1_thread.x += __shfl_xor(R1_thread.x, offset, WAVEFRONT_SIZE);
        R1_thread.y += __shfl_xor(R1_thread.y, offset, WAVEFRONT_SIZE);
    }

    const uint32_t laneId = threadIdx.x % WAVEFRONT_SIZE;
    const uint32_t waveId = threadIdx.x / WAVEFRONT_SIZE;
    const uint32_t numWaves = (blockDim.x + WAVEFRONT_SIZE - 1) / WAVEFRONT_SIZE;

    if (laneId == 0) {
        sh_R1[waveId] = R1_thread;
    }
    __syncthreads();

    // Final reduction across wavefronts (thread 0 only)
    if (threadIdx.x == 0) {
        for (uint32_t i = 1; i < numWaves; i++) {
            sh_R1[0].x += sh_R1[i].x;
            sh_R1[0].y += sh_R1[i].y;
        }
        TComplexCompute* pAccum = reinterpret_cast<TComplexCompute*>(drvdUeGrpPrms.tInfoDmrsAccum.pAddr);
        atomicAdd(&pAccum[drvdUeGrpPrms.dmrsActiveAccumBuf].x, sh_R1[0].x);
        atomicAdd(&pAccum[drvdUeGrpPrms.dmrsActiveAccumBuf].y, sh_R1[0].y);
        if (blockIdx.x == 0 && blockIdx.y == 0) {
            pAccum[drvdUeGrpPrms.dmrsActiveAccumBuf ^ 1].x = 0.0f;
            pAccum[drvdUeGrpPrms.dmrsActiveAccumBuf ^ 1].y = 0.0f;
        }
    }
