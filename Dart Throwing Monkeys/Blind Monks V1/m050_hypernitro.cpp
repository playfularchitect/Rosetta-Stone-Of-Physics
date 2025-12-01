
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <chrono>
#ifdef _OPENMP
#include <omp.h>
#endif
#include <immintrin.h>

using vec = __m256i;

static inline vec vset1_16(int x){ return _mm256_set1_epi16((short)x); }
static inline vec vadd16(vec a, vec b){ return _mm256_add_epi16(a,b); }
static inline vec vand(vec a, vec b){ return _mm256_and_si256(a,b); }
static inline vec vshr16(vec a, int s){ return _mm256_srli_epi16(a, s); }
static inline vec vsub16(vec a, vec b){ return _mm256_sub_epi16(a,b); }
static inline vec vcmpeq16(vec a, vec b){ return _mm256_cmpeq_epi16(a,b); }

#define LANES 16
// -------- Mersenne P=127 folding ----------
static inline vec addmod_127(vec a, vec b){
  const vec MASK = vset1_16(0x7F);
  vec s = vadd16(a,b);
  vec t = vadd16(vand(s, MASK), vshr16(s, 7));
  t = vadd16(vand(t, MASK), vshr16(t, 7));
  vec eq127 = vcmpeq16(t, MASK);
  vec corr  = vand(MASK, eq127);
  return vsub16(t, corr);
}
static inline uint16_t red127_u16(uint32_t x){
  x = (x & 0x7F) + (x >> 7);
  x = (x & 0x7F) + (x >> 7);
  if(x == 127) x = 0;
  return (uint16_t)x;
}

int main(int argc, char** argv){
  // Parameters
  int SE    = 8192;  // store every
  int Kf127 = 32;    // fused-K on P=127 rail
  int Kf2p  = 64;    // fused-K on 2^16 rail
  int U     = 80;    // unroll
  int VECN  = 12;    // live vectors per thread
  double win = 0.90; // seconds

  for(int i=1;i<argc;i++){
    if(!strcmp(argv[i],"--SE")    && i+1<argc) SE    = atoi(argv[++i]);
    else if(!strcmp(argv[i],"--Kf127")&& i+1<argc) Kf127 = atoi(argv[++i]);
    else if(!strcmp(argv[i],"--Kf2p") && i+1<argc) Kf2p  = atoi(argv[++i]);
    else if(!strcmp(argv[i],"--U")    && i+1<argc) U     = atoi(argv[++i]);
    else if(!strcmp(argv[i],"--VECN") && i+1<argc) VECN  = atoi(argv[++i]);
    else if(!strcmp(argv[i],"--win")  && i+1<argc) win   = atof(argv[++i]);
  }

  volatile uint64_t sink = 0;
  double secs = 0.0;
  unsigned long long logical_ops = 0ULL;

  auto t0 = std::chrono::high_resolution_clock::now();

  #pragma omp parallel num_threads(2) reduction(+:logical_ops) reduction(+:sink)
  {
    const int N = VECN;
    vec a127[64], k127[64];
    vec a2p[64],  k2p[64];

    // deterministic seeds in [0,126] for P=127; any 16-bit for 2^16
    for(int i=0;i<N;i++){
      uint16_t xi = (uint16_t)((123 + 17*i) % 127);
      uint16_t ki = (uint16_t)((77  + 31*i) % 127);
      uint16_t kf = red127_u16((uint32_t)Kf127 * (uint32_t)ki);
      a127[i] = vset1_16((int)xi);
      k127[i] = vset1_16((int)kf);

      uint16_t xj = (uint16_t)(0xACE1u + 97*i);  // any pattern
      uint16_t kj = (uint16_t)(0xBEEF + 29*i);
      // for 2^16, fused-K is just Kf2p * kj (wrap naturally)
      uint16_t kf2 = (uint16_t)(kj * (uint16_t)Kf2p);
      a2p[i]  = vset1_16((int)xj);
      k2p[i]  = vset1_16((int)kf2);
    }

    int it = 0;
    do{
      #pragma unroll(64)
      for(int u=0; u<U; ++u){
        // Rail A: P=127
        for(int i=0;i<N;i++){ a127[i] = addmod_127(a127[i], k127[i]); }
        // Rail B: 2^16 (wrap-around via epi16 add)
        for(int i=0;i<N;i++){ a2p[i]  = vadd16(a2p[i],  k2p[i]); }

        logical_ops += (unsigned long long)(N * LANES) * (unsigned long long)(Kf127 + Kf2p);
      }

      if((++it % SE)==0){
        alignas(32) uint16_t tmp[LANES];
        for(int i=0;i<N;i++){
          _mm256_store_si256((__m256i*)tmp, a127[i]); sink += tmp[0];
          _mm256_store_si256((__m256i*)tmp, a2p[i]);  sink += tmp[1];
        }
      }
      auto now = std::chrono::high_resolution_clock::now();
      secs = std::chrono::duration<double>(now - t0).count();
    } while(secs < win);
  }

  double gops = (double)logical_ops / secs / 1e9;
  printf("===== FX20_HYPERNITRO [AVX2 fused-K, P=127 + 2^16] =====\n");
  printf("Threads=2  LANES=%d  U=%d  Kf127=%d  Kf2p=%d  VECN=%d  SE=%d  window=%.2f s\n",
         LANES, U, Kf127, Kf2p, VECN, SE, secs);
  printf("Logical G-ops/s: %.2f\n", gops);
  printf("Kernel hash sink: 0x%llx\n", (unsigned long long)sink);
  return 0;
}
