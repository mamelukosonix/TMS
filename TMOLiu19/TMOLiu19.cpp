/************************************************************************************
*                                                                                   *
*                       Brno University of Technology                               *
*                       CPhoto@FIT                                                  *
*                                                                                   *
*                       Tone Mapping Studio                                         *
*                                                                                   *
*                                                                                   *
*                       Author: Robert Zelníček [xzelni06@stud.fit.vutbr.cz]        *
*                       Brno 2025                                                   *
*                                                                                   *
*                       Implementation of the TMOLiu19 class                        *
*                                                                                   *
************************************************************************************/

#include "TMOLiu19.h"
#include "TMOImage.h"

#include <vector>
#include <cmath>
#include <algorithm>

// --------------------------------------------------------------------------- //
// small helpers
// --------------------------------------------------------------------------- //
static inline int wrap(int x, int n) { x %= n; return (x < 0) ? (x + n) : x; }
static inline double clamp01(double v) { return v < 0.0 ? 0.0 : (v > 1.0 ? 1.0 : v); }

// --------------------------------------------------------------------------- //
// You16 complex contrast operator C via DoG pyramid
// c(G) = sum_{i=1..6} beta_i * (G_{2^i} - G_{2^{i+1}})
// beta = [-4, 1, 4, 4, 1, -2] for i=1..6
// periodic boundary condition
// --------------------------------------------------------------------------- //
static void gaussianBlurSeparable(const std::vector<double>& in, int w, int h, double sigma,
                                 std::vector<double>& out)
{
   if (sigma <= 0.0)
   {
      out = in;
      return;
   }

   const int radius = (int)std::ceil(3.0 * sigma);
   const int kSize  = 2 * radius + 1;

   std::vector<double> kernel(kSize);
   double sum = 0.0;
   for (int i = -radius; i <= radius; ++i)
   {
      double v = std::exp(-(i * i) / (2.0 * sigma * sigma));
      kernel[i + radius] = v;
      sum += v;
   }
   for (double& v : kernel) v /= sum;

   std::vector<double> tmp(w * h, 0.0);

   // horizontal
   for (int y = 0; y < h; ++y)
   {
      for (int x = 0; x < w; ++x)
      {
         double acc = 0.0;
         for (int k = -radius; k <= radius; ++k)
         {
            int xx = wrap(x + k, w);
            acc += kernel[k + radius] * in[y * w + xx];
         }
         tmp[y * w + x] = acc;
      }
   }

   // vertical
   out.assign(w * h, 0.0);
   for (int y = 0; y < h; ++y)
   {
      for (int x = 0; x < w; ++x)
      {
         double acc = 0.0;
         for (int k = -radius; k <= radius; ++k)
         {
            int yy = wrap(y + k, h);
            acc += kernel[k + radius] * tmp[yy * w + x];
         }
         out[y * w + x] = acc;
      }
   }
}

static void applyC(const std::vector<double>& x, int w, int h, std::vector<double>& out)
{
   static const double betas[] = {0.0, -4.0, 1.0, 4.0, 4.0, 1.0, -2.0}; // 1..6
   out.assign(w * h, 0.0);

   std::vector<double> blurA, blurB;
   std::vector<double> dog(w * h);

   for (int i = 1; i <= 6; ++i)
   {
      const double sigmaA = std::pow(2.0, (double)i);
      const double sigmaB = std::pow(2.0, (double)(i + 1));

      gaussianBlurSeparable(x, w, h, sigmaA, blurA);
      gaussianBlurSeparable(x, w, h, sigmaB, blurB);

      for (int p = 0; p < w * h; ++p)
         dog[p] = blurA[p] - blurB[p];

      const double b = betas[i];
      for (int p = 0; p < w * h; ++p)
         out[p] += b * dog[p];
   }
}

// symmetric kernels => C^T = C => C^T C x = C(C(x))
static void applyCTC(const std::vector<double>& x, int w, int h, std::vector<double>& out)
{
   std::vector<double> tmp;
   applyC(x, w, h, tmp);
   applyC(tmp, w, h, out);
}

// --------------------------------------------------------------------------- //
// Dark channel operator: D(g)(x) = min_{y in N(x)} g(y)
// Store Mg and argmin indices for M^T.
// --------------------------------------------------------------------------- //
static void applyM(const std::vector<double>& g, int w, int h, int rad,
                   std::vector<double>& Mg, std::vector<int>& argminIdx)
{
   const int n = w * h;
   Mg.assign(n, 0.0);
   argminIdx.assign(n, 0);

   for (int y = 0; y < h; ++y)
   {
      for (int x = 0; x < w; ++x)
      {
         const int idx = y * w + x;

         double bestVal = g[idx];
         int bestIdx = idx;

         const int y0 = std::max(0, y - rad);
         const int y1 = std::min(h - 1, y + rad);
         const int x0 = std::max(0, x - rad);
         const int x1 = std::min(w - 1, x + rad);

         for (int yy = y0; yy <= y1; ++yy)
         {
            for (int xx = x0; xx <= x1; ++xx)
            {
               const int j = yy * w + xx;
               if (g[j] < bestVal)
               {
                  bestVal = g[j];
                  bestIdx = j;
               }
            }
         }

         Mg[idx] = bestVal;
         argminIdx[idx] = bestIdx;
      }
   }
}

static void applyMT(const std::vector<double>& wvec, const std::vector<int>& argminIdx,
                    std::vector<double>& MTw)
{
   const int n = (int)wvec.size();
   MTw.assign(n, 0.0);
   for (int i = 0; i < n; ++i)
      MTw[argminIdx[i]] += wvec[i];
}

// --------------------------------------------------------------------------- //
// A(x) = (beta+gamma)x + alphaSum * C^T C x
// --------------------------------------------------------------------------- //
static void applyA(const std::vector<double>& x, int w, int h,
                   double beta, double gamma, double alphaSum,
                   std::vector<double>& out)
{
   std::vector<double> ctc;
   applyCTC(x, w, h, ctc);

   const int n = w * h;
   out.assign(n, 0.0);

   const double diag = beta + gamma;
   for (int i = 0; i < n; ++i)
      out[i] = diag * x[i] + alphaSum * ctc[i];
}

static bool solveCG(int w, int h,
                    double beta, double gamma, double alphaSum,
                    const std::vector<double>& b,
                    std::vector<double>& x,
                    int maxIters,
                    double tol)
{
   const int n = w * h;

   auto dot = [&](const std::vector<double>& a, const std::vector<double>& bb) -> double {
      double s = 0.0;
      for (int i = 0; i < n; ++i) s += a[i] * bb[i];
      return s;
   };
 
   std::vector<double> Ax, r, p, Ap;
   applyA(x, w, h, beta, gamma, alphaSum, Ax);

   r.resize(n);
   p.resize(n);
   for (int i = 0; i < n; ++i)
   {
      r[i] = b[i] - Ax[i];
      p[i] = r[i];
   }

   double rsold = dot(r, r);
   const double tol2 = tol * tol;
   if (rsold < tol2) return true;

   for (int it = 0; it < maxIters; ++it)
   {
      applyA(p, w, h, beta, gamma, alphaSum, Ap);
      const double denom = dot(p, Ap);
      if (std::abs(denom) < 1e-30) break;

      const double a = rsold / denom;

      for (int i = 0; i < n; ++i)
      {
         x[i] += a * p[i];
         r[i] -= a * Ap[i];
      }

      const double rsnew = dot(r, r);
      if (rsnew < tol2) return true;

      const double bta = rsnew / rsold;
      for (int i = 0; i < n; ++i)
            p[i] = r[i] + bta * p[i];

      rsold = rsnew;
   }

   return false;
}

// --------------------------------------------------------------------------- //
// Lambda diagonal (Liu19) but with TMS Lab normalized to [0,1]:
// L100 = 100*L
// a100 = 200*(a-0.5)  -> [-100,100]
// b100 = 200*(b-0.5)  -> [-100,100]
// Then apply the paper formula.
// --------------------------------------------------------------------------- //
static void buildLambdaDiag_TMSLab01(const std::vector<double>& L01,
                                    const std::vector<double>& a01,
                                    const std::vector<double>& b01,
                                    std::vector<double>& Lambda,
                                    double eps)
{
   const int n = (int)L01.size();
   Lambda.assign(n, 0.0);

   const double C100 = 100.0;

   for (int i = 0; i < n; ++i)
   {
      const double L = 100.0 * L01[i];
      const double A = 200.0 * (a01[i] - 0.5);
      const double B = 200.0 * (b01[i] - 0.5);

      const double t1 = std::sqrt(C100*C100 - A*A - B*B + eps);
      const double t2 = std::sqrt(C100*C100 - (2.0*L - C100)*(2.0*L - C100) + eps);
      const double denom = t1 * t2;

      Lambda[i] = (denom > 0.0) ? (1.0 / denom) : 0.0;
   }
}

// --------------------------------------------------------------------------- //
// TMOLiu19: ctor/params (Khudair23-like style)
// --------------------------------------------------------------------------- //
TMOLiu19::TMOLiu19()
{
   SetName(L"Liu19");
   SetDescription(L"Color-to-gray conversion with perceptual preservation "
                   L"and dark channel prior (Liu et al., 2019).");

   alpha1.SetName(L"alpha1");
   alpha1.SetDescription(L"Contrast weight for p1 (L channel)");
   alpha1.SetRange(0.0, 1000.0);
   alpha1 = 1.0;
   this->Register(alpha1);

   alpha2.SetName(L"alpha2");
   alpha2.SetDescription(L"Contrast weight for p2,p3 (a,b channels), alpha2=alpha3");
   alpha2.SetRange(0.0, 1000.0);
   alpha2 = 1.0;
   this->Register(alpha2);

   eta.SetName(L"eta");
   eta.SetDescription(L"Dark channel prior weight");
   eta.SetRange(0.0, 1000.0);
   eta = 0.01;
   this->Register(eta);

   beta0.SetName(L"beta0");
   beta0.SetDescription(L"Initial beta (if 0, uses 2*eta as in paper)");
   beta0.SetRange(0.0, 1e6);
   beta0 = 0.0;
   this->Register(beta0);

   gamma0.SetName(L"gamma0");
   gamma0.SetDescription(L"Initial gamma (if 0, uses 2*eta as in paper)");
   gamma0.SetRange(0.0, 1e6);
   gamma0 = 0.0;
   this->Register(gamma0);

   Iterations.SetName(L"Iterations");
   Iterations.SetDescription(L"Maximum number of iterations (safety cap)");
   Iterations.SetRange(1, 5000);
   Iterations = 50;
   this->Register(Iterations);

   PatchRadius.SetName(L"PatchRadius");
   PatchRadius.SetDescription(L"Radius of dark-channel patch N(x)");
   PatchRadius.SetRange(1, 50);
   PatchRadius = 3;
   this->Register(PatchRadius);

   CGIters.SetName(L"CGIters");
   CGIters.SetDescription(L"CG iterations for solving g-subproblem");
   CGIters.SetRange(10, 10000);
   CGIters = 200;
   this->Register(CGIters);

   CGTol.SetName(L"CGTol");
   CGTol.SetDescription(L"CG tolerance");
   CGTol.SetRange(1e-12, 1e-2);
   CGTol = 1e-6;
   this->Register(CGTol);
}

TMOLiu19::~TMOLiu19()
{
    // required for vtable
}

// --------------------------------------------------------------------------- //
// Transform
// --------------------------------------------------------------------------- //
int TMOLiu19::Transform()
{
   // Liu19 fixed constants
   const double sigma1 = 1e-3;
   const double sigma2 = 1e-3;
   const double Pmax   = 26.0;
   const double eps    = 1e-8;

   pSrc->Convert(TMO_RGB);
   pDst->Convert(TMO_RGB);

   const int width  = pSrc->GetWidth();
   const int height = pSrc->GetHeight();
   const int N = width * height;

   // STEP 1: Lab channels in TMS are in [0,1]
   pSrc->Convert(TMO_Lab);
   std::vector<double> p1(N), p2(N), p3(N);
   {
      double* src = pSrc->GetData();
      for (int i = 0; i < N; ++i)
      {
         p1[i] = *src++; // L in [0,1]
         p2[i] = *src++; // a in [0,1]
         p3[i] = *src++; // b in [0,1]
      }
   }
   // Lambda diagonal (using scaled Lab values internally)
   std::vector<double> Lambda;
   buildLambdaDiag_TMSLab01(p1, p2, p3, Lambda, eps);
   
   // STEP 2: init g,f,w and beta,gamma
   std::vector<double> g = p1;
   std::vector<double> f = p1;
   std::vector<double> wvec(N, 0.0);
   double beta  = (beta0 > 0.0) ? (double)beta0 : (2.0 * (double)eta);
   double gamma = (gamma0 > 0.0) ? (double)gamma0 : (2.0 * (double)eta);
   const double a1 = (double)alpha1;
   const double a2 = (double)alpha2;
   const double alphaSum = a1 + a2 + a2;
   
   // C^T C * sum(alpha_i p_i)
   std::vector<double> pWeighted(N);
   for (int i = 0; i < N; ++i)
       pWeighted[i] = a1*p1[i] + a2*p2[i] + a2*p3[i];
   std::vector<double> CTC_pWeighted;
   applyCTC(pWeighted, width, height, CTC_pWeighted);
   
   // STEP 3: Algorithm 2 iterations
   int iter = 0;
   while (beta < Pmax && iter < (int)Iterations)
   {
      pSrc->ProgressBar(iter, Iterations);
      // (a) Eq.(8) w-update
      std::vector<double> Mg;
      std::vector<int> argminIdx;
      applyM(g, width, height, (int)PatchRadius, Mg, argminIdx);

      std::vector<double> wnew(N, 0.0);
      for (int i = 0; i < N; ++i)
      {
         const double lhs = gamma * (Mg[i]*Mg[i]) + sigma1 * (wvec[i]*wvec[i]);
         wnew[i] = (lhs >= (double)eta) ? Mg[i] : 0.0;
      }

      // (b) Eq.(10) g-update: solve A g = b
      std::vector<double> MTw;
      applyMT(wnew, argminIdx, MTw);

      std::vector<double> b(N, 0.0);
      for (int i = 0; i < N; ++i)
            b[i] = beta * f[i] + MTw[i] + CTC_pWeighted[i];

      std::vector<double> gnew = g;
      solveCG(width, height, beta, gamma, alphaSum, b, gnew,
                (int)CGIters, (double)CGTol);

      // (c) Eq.(12) f-update shrinkage
      std::vector<double> fnew(N, 0.0);
      const double denom = 2.0 * (beta + sigma2);

      for (int i = 0; i < N; ++i)
      {
         const double xi = beta * (gnew[i] - p1[i]) + sigma2 * (f[i] - p1[i]);
         const double sign = (xi > 0.0) ? 1.0 : ((xi < 0.0) ? -1.0 : 0.0);
         const double mag = std::abs(xi) / denom - Lambda[i] / denom;
         const double shrink = (mag > 0.0) ? mag : 0.0;
         fnew[i] = shrink * sign + p1[i];
      }
      // commit
      wvec.swap(wnew);
      g.swap(gnew);
      f.swap(fnew);
      // penalty update
      beta *= 2.0;
      gamma *= 2.0;
      ++iter;
   }
   // Output: grayscale in RGB, already in [0,1]
   double* dst = pDst->GetData();
   for (int i = 0; i < N; ++i)
   {
      const double gray = clamp01(g[i]);
      *dst++ = gray;
      *dst++ = gray;
      *dst++ = gray;
   }

   pSrc->ProgressBar(Iterations, Iterations);
   return 0;
}
