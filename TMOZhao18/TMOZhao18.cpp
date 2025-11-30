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
*                       Implementation of the TMOZhao18 class                       *
*                                                                                   *
************************************************************************************/
//ZKONTROLOVAT VYSLEDKY


#include "TMOZhao18.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdarg>
#include <cstdlib>
#include <fstream>
#include <limits>
#include <mutex>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace {

// --------------------------- Utils ------------------------------- //

inline double clamp01(double v){ return v<0.0?0.0:(v>1.0?1.0:v); }

static inline double srgb_to_linear(double c)
{
    if (c <= 0.04045)
        return c / 12.92;
    return std::pow((c + 0.055) / 1.055, 2.4);
}


// ------------------------------ Logging --------------------------------- //

static std::mutex g_log_mtx;

static std::string zhao_temp_path() {
   const char *tmp = std::getenv("TMPDIR");
   if (!tmp) tmp = std::getenv("TMP");
   if (!tmp) tmp = std::getenv("TEMP");
   if (!tmp) tmp = "/tmp";
   std::string base(tmp);
   if (!base.empty() && base.back() != '/') base.push_back('/');
   return base + "zhao18.log";
}

static void zlog(const char *fmt, ...) {
   std::lock_guard<std::mutex> lk(g_log_mtx);
   va_list ap;
   va_start(ap, fmt);
   vfprintf(stderr, fmt, ap);
   fputc('\n', stderr);
   fflush(stderr);
   va_end(ap);
   static std::ofstream fout(zhao_temp_path(), std::ios::app);
   if (fout) {
      va_start(ap, fmt);
      char buf[4096];
      vsnprintf(buf, sizeof(buf), fmt, ap);
      va_end(ap);
      fout << buf << std::endl;
   }
}

// ------------------------------ Structures ------------------------------ //

struct PairIdx { int i, j; };
struct PDPair  { PairIdx idx; double kappa; };

struct Stats { double mn=0, mx=0, mean=0; };
static Stats stats_of(const std::vector<double>& v){
   Stats s{};
   if (v.empty()) return s;
   double mn=v[0], mx=v[0];
   long double sum=0;
   for(double x: v){ if(x<mn) mn=x; if(x>mx) mx=x; sum+=x; }
   s.mn=mn; s.mx=mx; s.mean=(double)(sum/v.size());
   return s;
}

inline uint32_t clamp_u8(int v){ return (uint32_t)(v<0?0:(v>255?255:v)); }

inline uint32_t lab_code24(double L, double a, double b){
   const int Li = (int)std::lround(L);             
   const int ai = (int)std::lround(a + 128.0);     
   const int bi = (int)std::lround(b + 128.0);
   const uint32_t L8 = clamp_u8(Li);
   const uint32_t A8 = clamp_u8(ai);
   const uint32_t B8 = clamp_u8(bi);
   return (L8<<16) | (A8<<8) | B8;
}
inline uint32_t rgb8_code(double r, double g, double b){
   const int Ri = (int)std::lround(clamp01(r)*255.0);
   const int Gi = (int)std::lround(clamp01(g)*255.0);
   const int Bi = (int)std::lround(clamp01(b)*255.0);
   const uint32_t R8 = clamp_u8(Ri);
   const uint32_t G8 = clamp_u8(Gi);
   const uint32_t B8 = clamp_u8(Bi);
   return (R8<<16) | (G8<<8) | B8;
}

// ------------------------------ Downsample ------------------------------ //

struct DownImg {
   int w=0,h=0;
   std::vector<double> R,G,B,L,A,Bb;
};
static DownImg downsample_to(const std::vector<double> &R,
                             const std::vector<double> &G,
                             const std::vector<double> &B,
                             const std::vector<double> &L,
                             const std::vector<double> &A,
                             const std::vector<double> &Bb,
                             int W,int H,int targetS){
   DownImg out;
   if (targetS<=0) return out;
   const int maxSide = std::max(W,H);
   if (targetS>=maxSide) return out;
   const double scale = (double)targetS / (double)maxSide;
   out.w = std::max(1, (int)std::round(W*scale));
   out.h = std::max(1, (int)std::round(H*scale));
   out.R.resize(out.w*out.h);
   out.G.resize(out.w*out.h);
   out.B.resize(out.w*out.h);
   out.L.resize(out.w*out.h);
   out.A.resize(out.w*out.h);
   out.Bb.resize(out.w*out.h);

   const double sx=(double)W/out.w, sy=(double)H/out.h;
   for(int y=0;y<out.h;++y){
      const int syi = std::min(H-1,(int)std::floor(y*sy));
      for(int x=0;x<out.w;++x){
         const int sxi = std::min(W-1,(int)std::floor(x*sx));
         const int si = syi*W + sxi;
         const int di = y*out.w + x;
         out.R[di]=R[si]; out.G[di]=G[si]; out.B[di]=B[si];
         out.L[di]=L[si]; out.A[di]=A[si]; out.Bb[di]=Bb[si];
      }
   }
   return out;
}

// ------------------------------ P_N ------------------------------------- //

static std::vector<PairIdx> build_PN_indices(const std::vector<double>& L,
                                             const std::vector<double>& A,
                                             const std::vector<double>& Bb,
                                             int W,int H,int nhood){
   std::vector<PairIdx> out;
   struct Off{int dx,dy;};
   std::vector<Off> offs;
   offs.push_back({+1,0});
   offs.push_back({0,+1});
   if (nhood>=8){ offs.push_back({+1,+1}); offs.push_back({-1,+1}); }

   const int N=W*H;
   std::vector<uint32_t> code(N);
   for(int i=0;i<N;++i) code[i]=lab_code24(L[i],A[i],Bb[i]);

   std::unordered_set<uint64_t> seen;
   seen.reserve((size_t)N);

   for(int y=0;y<H;++y){
      for(int x=0;x<W;++x){
         const int i=y*W+x;
         for(const auto& o: offs){
            const int nx=x+o.dx, ny=y+o.dy;
            if(nx<0||nx>=W||ny<0||ny>=H) continue;
            const int j=ny*W+nx;
            uint32_t c1=code[i], c2=code[j];
            if(c1==c2) continue;
            uint32_t lo = std::min(c1,c2), hi=std::max(c1,c2);
            uint64_t key = ( (uint64_t)lo<<24 ) | (uint64_t)hi;
            if(seen.insert(key).second) out.push_back(PairIdx{i,j});
         }
      }
   }
   return out;
}

// ------------------------------ P_D ------------------------------------- //

struct DomColor { uint32_t code; int count; int repIdx; double L,A,B; };

static void build_PD(const std::vector<double>& R,
                     const std::vector<double>& G,
                     const std::vector<double>& Bc,
                     const std::vector<double>& L,
                     const std::vector<double>& A,
                     const std::vector<double>& Bb,
                     int W,int H,
                     std::vector<PDPair>& PD_out,
                     double& sum_kappa_pos_out,
                     int& kept_pairs_out,
                     int& dropped_pairs_out,
                     std::array<int,8>& kappa_bins_dbg)
{
   const int N=W*H;
   struct Hist{ int count=0; int repIdx=-1; };
   std::unordered_map<uint32_t,Hist> hist;
   hist.reserve((size_t)N);

   for(int i=0;i<N;++i){
      const uint32_t key = rgb8_code(R[i],G[i],Bc[i]);
      auto &e=hist[key];
      if(e.count==0) e.repIdx=i;
      e.count++;
   }

   const int minCount = std::max(1,(int)std::floor(0.001 * (double)N));

   std::vector<DomColor> dom;
   dom.reserve(hist.size());
   int hmin=INT_MAX, hmed=0, hmax=0;
   std::vector<int> counts; counts.reserve(hist.size());

   for(const auto& kv: hist){
      const auto& e=kv.second;
      counts.push_back(e.count);
      if(e.count>=minCount && e.repIdx>=0){
         const int i=e.repIdx;
         dom.push_back(DomColor{kv.first,e.count,i,L[i],A[i],Bb[i]});
      }
   }
   if(!counts.empty()){
      std::sort(counts.begin(),counts.end());
      hmin=counts.front(); hmax=counts.back(); hmed=counts[counts.size()/2];
   }

   std::sort(dom.begin(),dom.end(),
              [](const DomColor& a,const DomColor& b){return a.count>b.count;});

   //DEBUG
   zlog("[Zhao18] Dominant histogram: min=%d  median=%d  max=%d  minCount=%d",
        hmin,hmed,hmax,minCount);
   for(int i=0;i<std::min<int>(10,(int)dom.size());++i){
      const auto& d=dom[i];
      double frac = 100.0 * (double)d.count / (double)N;
      zlog("[Zhao18]   top%02d: count=%d  %.3f%%  repIdx=%d  Lab(%.2f,%.2f,%.2f)",
             i+1, d.count, frac, d.repIdx, d.L,d.A,d.B);
   }

   // Conscruct P_D pairs and kappa
   PD_out.clear();
   kept_pairs_out=0; dropped_pairs_out=0;
   sum_kappa_pos_out=0.0;

   //DEBUG
   if(dom.size()<2){
      zlog("[Zhao18][WARN] Too few dominant colors (%zu) => P_D empty.", dom.size());
      return;
   }

   const double denom = (0.001 * 0.001) * (double)N * (double)N;

   // histogram bins kappa - debug
   kappa_bins_dbg = {0,0,0,0,0,0,0,0}; // (-inf,-4],(-4,-2],(-2,-1],(-1,0],(0,0.2],(0.2,0.5],(0.5,1],(1,inf))

   int kpos=0, kneg=0;
   double sum_raw=0;

   for(size_t a=0;a<dom.size();++a){
      for(size_t b=a+1;b<dom.size();++b){
         const auto& X=dom[a];
         const auto& Y=dom[b];
         const double NxNy = (double)X.count * (double)Y.count;
         double kappa = std::log( NxNy / denom );
         sum_raw += kappa;
         // debug bins
         auto bin_incr=[&](double v){
         if(v<=-4.0) kappa_bins_dbg[0]++; else
         if(v<=-2.0) kappa_bins_dbg[1]++; else
         if(v<=-1.0) kappa_bins_dbg[2]++; else
         if(v<  0.0) kappa_bins_dbg[3]++; else
         if(v<=0.2) kappa_bins_dbg[4]++; else
         if(v<=0.5) kappa_bins_dbg[5]++; else
         if(v<=1.0) kappa_bins_dbg[6]++; else kappa_bins_dbg[7]++;
      };
         bin_incr(kappa);

         // Negative kappa -> ignore
         if(kappa <= 0.0){
            kneg++;
            dropped_pairs_out++;
            continue;
         }
         kpos++;
         sum_kappa_pos_out += kappa;

         PD_out.push_back(PDPair{ PairIdx{X.repIdx, Y.repIdx}, kappa });
         kept_pairs_out++;
      }
   }

   //DEBUG
   zlog("[Zhao18] P_D: pairs=%d  denom=%.6f  κ_pos=%d  κ_neg=%d",
        kept_pairs_out + dropped_pairs_out, denom, kpos, kneg);
   zlog("[Zhao18] κ stats: sum_raw=%.6f  sum_pos=%.6f", sum_raw, sum_kappa_pos_out);
   zlog("[Zhao18] κ bins: (-inf,-4]=%d  (-4,-2]=%d  (-2,-1]=%d  (-1,0]=%d  (0,0.2]=%d  (0.2,0.5]=%d  (0.5,1]=%d  (1,inf]=%d",
         kappa_bins_dbg[0],kappa_bins_dbg[1],kappa_bins_dbg[2],kappa_bins_dbg[3],
         kappa_bins_dbg[4],kappa_bins_dbg[5],kappa_bins_dbg[6],kappa_bins_dbg[7]);
}

// ------------------------- delta precompute & energy ------------------------ //

struct Deltas {
   // PN
   std::vector<float> dL_PN, da_PN, db_PN;
   // PD 
   std::vector<float> dL_PD, da_PD, db_PD;
   // min/max
   double minL=0,maxL=1,minA=0,maxA=1,minB=0,maxB=1;
};

// Calculates absolute deltas L,A,Bb for PN & PD and normalize to [0,1] by global min-max
static Deltas precompute_lab_diffs_norm(const std::vector<double>& L,
                                        const std::vector<double>& A,
                                        const std::vector<double>& Bb,
                                        const std::vector<PairIdx>& PN,
                                        const std::vector<PDPair>& PD)
{
   Deltas D;
   D.dL_PN.resize(PN.size());
   D.da_PN.resize(PN.size());
   D.db_PN.resize(PN.size());
   D.dL_PD.resize(PD.size());
   D.da_PD.resize(PD.size());
   D.db_PD.resize(PD.size());

   auto upd_minmax = [](double v, double& mn, double& mx){
      if(v<mn) mn=v; if(v>mx) mx=v;
   };

   double mnL=1e300,mxL=-1e300, mnA=1e300,mxA=-1e300, mnB=1e300,mxB=-1e300;

   for(size_t k=0;k<PN.size();++k){
      const int i=PN[k].i, j=PN[k].j;
      const double dL = std::fabs(L[i]-L[j]);
      const double da = std::fabs(A[i]-A[j]);
      const double db = std::fabs(Bb[i]-Bb[j]);
      D.dL_PN[k]=(float)dL; D.da_PN[k]=(float)da; D.db_PN[k]=(float)db;
      upd_minmax(dL,mnL,mxL); upd_minmax(da,mnA,mxA); upd_minmax(db,mnB,mxB);
   }
   for(size_t k=0;k<PD.size();++k){
      const int i=PD[k].idx.i, j=PD[k].idx.j;
      const double dL = std::fabs(L[i]-L[j]);
      const double da = std::fabs(A[i]-A[j]);
      const double db = std::fabs(Bb[i]-Bb[j]);
      D.dL_PD[k]=(float)dL; D.da_PD[k]=(float)da; D.db_PD[k]=(float)db;
      upd_minmax(dL,mnL,mxL); upd_minmax(da,mnA,mxA); upd_minmax(db,mnB,mxB);
   }
   const double eps=1e-12;
   const double rL = (mxL-mnL)>eps ? (mxL-mnL) : 1.0;
   const double rA = (mxA-mnA)>eps ? (mxA-mnA) : 1.0;
   const double rB = (mxB-mnB)>eps ? (mxB-mnB) : 1.0;

   auto norm = [](float v, double mn, double r){ return (float)((v - (float)mn)/r); };

   for(size_t k=0;k<PN.size();++k){
      D.dL_PN[k]=norm(D.dL_PN[k],mnL,rL);
      D.da_PN[k]=norm(D.da_PN[k],mnA,rA);
      D.db_PN[k]=norm(D.db_PN[k],mnB,rB);
   }
   for(size_t k=0;k<PD.size();++k){
      D.dL_PD[k]=norm(D.dL_PD[k],mnL,rL);
      D.da_PD[k]=norm(D.da_PD[k],mnA,rA);
      D.db_PD[k]=norm(D.db_PD[k],mnB,rB);
   }

   D.minL=mnL; D.maxL=mxL;
   D.minA=mnA; D.maxA=mxA;
   D.minB=mnB; D.maxB=mxB;
   return D;
}

static inline double gauss(double x, double mu, double sigma){
   const double s2 = 2.0*sigma*sigma;
   return std::exp( -((x-mu)*(x-mu))/s2 );
}

static inline double D_mixture(double dI, double dL, double da, double db, double mu, double sigma){
   double eL = gauss( (std::fabs(dI) - std::fabs(dL)), mu, sigma );
   double ea = gauss( (std::fabs(dI) - std::fabs(da)), mu, sigma );
   double eb = gauss( (std::fabs(dI) - std::fabs(db)), mu, sigma );
   double s = (eL + ea + eb) / 3.0;
   if(s < 1e-300) s=1e-300;
   return -std::log(s);
}

struct EnergyDbg { double Epn=0, Epd=0; };

static double energy_for_weights(
   double wr,double wg,double wb,
   const std::vector<double>& I1, const std::vector<double>& I2, const std::vector<double>& I3,
   const std::vector<PairIdx>& PN,
   const std::vector<PDPair>& PD,
   const Deltas& Dn,
   double lambda, double mu, double sigma,
   EnergyDbg* dbg = nullptr)
{
   double Epn=0.0, Epd=0.0;
   // PN
   for(size_t k=0;k<PN.size();++k){
      const int i=PN[k].i, j=PN[k].j;
      const double dI = std::fabs( (wr*I1[i] + wg*I2[i] + wb*I3[i]) - (wr*I1[j] + wg*I2[j] + wb*I3[j]) );
      Epn += D_mixture( dI, Dn.dL_PN[k], Dn.da_PN[k], Dn.db_PN[k], mu, sigma );
   }
   // PD
   for(size_t k=0;k<PD.size();++k){
      const int i=PD[k].idx.i, j=PD[k].idx.j;
      const double dI = std::fabs( (wr*I1[i] + wg*I2[i] + wb*I3[i]) - (wr*I1[j] + wg*I2[j] + wb*I3[j]) );
      Epd += PD[k].kappa * D_mixture( dI, Dn.dL_PD[k], Dn.da_PD[k], Dn.db_PD[k], mu, sigma );
   }
   if(dbg){ dbg->Epn=Epn; dbg->Epd=lambda*Epd; }
   return Epn + lambda*Epd;
}

// ------------------------------ Search ---------------------------------- //

struct Candidate { double wr=0,wg=0,wb=1, E=std::numeric_limits<double>::infinity(); EnergyDbg dbg; };

static void coarse_search(int S,
                           const std::vector<double>& I1,const std::vector<double>& I2,const std::vector<double>& I3,
                           const std::vector<PairIdx>& PN,const std::vector<PDPair>& PD,const Deltas& Dn,
                           double lambda,double mu,double sigma,
                           Candidate& best,
                           std::vector<Candidate>* ladder_dbg=nullptr)
{
   const int combos = (S+1)*(S+2)/2;
   zlog("[Zhao18] Search grid: step=%.3f  combos=%d", 1.0/S, combos);
   for(int ir=0; ir<=S; ++ir){
      for(int ig=0; ig<=S-ir; ++ig){
         const int ib = S - ir - ig;
         const double wr = ir/(double)S;
         const double wg = ig/(double)S;
         const double wb = ib/(double)S;
         EnergyDbg dbg;
         const double E = energy_for_weights(wr,wg,wb,I1,I2,I3,PN,PD,Dn,lambda,mu,sigma,&dbg);
         if(ladder_dbg){
            ladder_dbg->push_back(Candidate{wr,wg,wb,E,dbg});
         }
         if(E < best.E){ best = Candidate{wr,wg,wb,E,dbg}; }
      }
   }
}
}//namespace
// --------------------------------------------------------------------------- //
// TMOZhao18: ctor/params
// --------------------------------------------------------------------------- //

TMOZhao18::TMOZhao18(){
   SetName(L"Zhao18");
   SetDescription(L"Zhao 2018: Efficient decolorization with multimodal contrast-preserving measure");

   sigma.SetName(L"sigma");
   sigma.SetDescription(L"Width of the Gaussian kernel in D(x,y) ");
   sigma.SetDefault(0.35); sigma=0.35; sigma.SetRange(1e-4,2.0);
   this->Register(sigma);

   mu.SetName(L"mu");
   mu.SetDescription(L"Tolerance parameter for contrast differences.");
   mu.SetDefault(0.00); mu=0.00; mu.SetRange(0.0,2.0);
   this->Register(mu);

   lambda0.SetName(L"lambda0");
   lambda0.SetDescription(L"The base multiplier for Lambda before normalization");
   lambda0.SetDefault(1.0); lambda0=1.0; lambda0.SetRange(0.0,10.0);
   this->Register(lambda0);

   downsample.SetName(L"downsample");
   downsample.SetDescription(L"Target size of the shorter image");
   downsample.SetDefault(64); downsample=64; downsample.SetRange(16,512);
   this->Register(downsample);
}

TMOZhao18::~TMOZhao18(){}

// --------------------------------------------------------------------------- //
// Transform
// --------------------------------------------------------------------------- //

// --------------------------------------------------------------------------- //
// Transform
// --------------------------------------------------------------------------- //

int TMOZhao18::Transform(){
   // DEBUG
   setvbuf(stdout,nullptr,_IONBF,0);
   setvbuf(stderr,nullptr,_IONBF,0);
   zlog("[Zhao18] Logging to file: %s", zhao_temp_path().c_str());

   pSrc->Convert(TMO_RGB);
   pDst->Convert(TMO_RGB);

   const int W = pSrc->GetWidth();
   const int H = pSrc->GetHeight();
   const int N = W * H;

   double *src = pSrc->GetData();
   double *dst = pDst->GetData();

   std::vector<double> R(N), G(N), B(N);
   for (int y = 0; y < H; ++y) {
      pSrc->ProgressBar(y,H);
      for (int x = 0; x < W; ++x) {
         const int i = y*W + x;
         const double r = clamp01(*src++);
         const double g = clamp01(*src++);
         const double b = clamp01(*src++);
         R[i] = r;
         G[i] = g;
         B[i] = b;
      }
   }

   TMOImage labImg;
   labImg.New(*pSrc, TMO_LAB, true);
   double *labData = labImg.GetData();

   std::vector<double> Lc(N), Ac(N), Bc(N);
   for (int i = 0; i < N; ++i) {
      Lc[i] = labData[3*i + 0];
      Ac[i] = labData[3*i + 1];
      Bc[i] = labData[3*i + 2];
   }

   // DEBUG – statisticsc in Lab
   zlog("[Zhao18] Image: %d x %d (N=%d)", W,H,N);
   zlog("[Zhao18] Params: sigma=%.4f  mu=%.4f  lambda0=%.4f  downsample=%d",
        (double)sigma,(double)mu,(double)lambda0,(int)downsample);

   const Stats SL = stats_of(Lc), SA = stats_of(Ac), SBb = stats_of(Bc);
   zlog("[Zhao18] Lab:    L*[min=%.3f max=%.3f mean=%.3f]  a*[min=%.3f max=%.3f mean=%.3f]  b*[min=%.3f max=%.3f mean=%.3f]",
        SL.mn,SL.mx,SL.mean, SA.mn,SA.mx,SA.mean, SBb.mn,SBb.mx,SBb.mean);

   // --- 3) Downsample for eval domain (RGB + Lab) ---
   const int targetShort = (int)downsample;
   DownImg dsmall = downsample_to(R,G,B, Lc,Ac,Bc, W,H, targetShort);
   if (dsmall.w > 0)
      zlog("[Zhao18] Downsampled: %d x %d (target short side=%d)", dsmall.w,dsmall.h,targetShort);
   else
      zlog("[Zhao18] Downsample disabled or not needed (target short side=%d)", targetShort);

   const bool use_small = (dsmall.w > 0);
   const std::vector<double> &Re   = use_small ? dsmall.R  : R,
                              &Ge   = use_small ? dsmall.G  : G,
                              &Be   = use_small ? dsmall.B  : B,
                              &Le   = use_small ? dsmall.L  : Lc,
                              &Ae   = use_small ? dsmall.A  : Ac,
                              &Be_b = use_small ? dsmall.Bb : Bc;
   const int We = use_small ? dsmall.w : W;
   const int He = use_small ? dsmall.h : H;

   // --- 4) LINEAR RGB for energy
   std::vector<double> ReL( (size_t)We * (size_t)He );
   std::vector<double> GeL(ReL.size());
   std::vector<double> BeL(ReL.size());
   for (size_t i = 0; i < ReL.size(); ++i) {
      ReL[i] = Re[i];
      GeL[i] = Ge[i];
      BeL[i] = Be[i];
   }

   // --- 5) P_N – pairs from Lab ---
   int nh = 8;
   std::vector<PairIdx> PN = build_PN_indices(Le,Ae,Be_b, We,He, nh);
   zlog("[Zhao18] P_N (eval) unique pairs: %zu", PN.size());

   // --- 6) P_D ---
   std::vector<PDPair> PD;
   double sum_kappa_pos = 0.0;
   int kept = 0, dropped = 0;
   std::array<int,8> bins{};
   build_PD(Re,Ge,Be, Le,Ae,Be_b, We,He, PD, sum_kappa_pos, kept, dropped, bins);
   if (!PD.empty()) {
      std::vector<std::pair<double,PDPair>> tmp;
      tmp.reserve(PD.size());
      for (auto &p : PD) tmp.push_back({p.kappa, p});
      std::sort(tmp.begin(), tmp.end(),
                [](auto &a, auto &b){ return a.first > b.first; });
      zlog("[Zhao18] TOP κ_pos pairs:");
      for (int i = 0; i < std::min<int>(10,(int)tmp.size()); ++i) {
         auto &pp = tmp[i].second;
         zlog("  #%d: kappa=%.6f  i=%d j=%d", i+1, tmp[i].first, pp.idx.i, pp.idx.j);
      }
   }

   // --- 7( lambda as in pdf, equation 5
   double lambda_val = 0.0;
   if (sum_kappa_pos > 1e-12) {
      lambda_val = (double)lambda0 * (double)PN.size() / sum_kappa_pos;
   }
   zlog("[Zhao18] P_D (eval) pairs kept=%d dropped=%d  sum_kappa_pos=%.6f", kept, dropped, sum_kappa_pos);
   zlog("[Zhao18] lambda (Eq.5): %.6f (|P_N|=%zu, sum_kappa_pos=%.6f)", lambda_val, PN.size(), sum_kappa_pos);

   // --- 8) deltas LAB ---
   Deltas Dn = precompute_lab_diffs_norm(Le,Ae,Be_b, PN, PD);
   zlog("[Zhao18] Δ ranges (raw): L[%.3f..%.3f] a[%.3f..%.3f] b[%.3f..%.3f] -> normalized to [0,1]",
        Dn.minL,Dn.maxL, Dn.minA,Dn.maxA, Dn.minB,Dn.maxB);

   // --- Debug - reference values for some fixed weight, to see if its working allright ---
   auto report_ref = [&](double wr,double wg,double wb, const char* tag){
      EnergyDbg dbg;
      double E = energy_for_weights(wr,wg,wb,ReL,GeL,BeL,PN,PD,Dn,lambda_val,mu,sigma,&dbg);
      zlog("[Zhao18] Ref %-6s: wr=%.3f wg=%.3f wb=%.3f  E=%.6f  [E_PN=%.6f  E_PD=%.6f]",
           tag, wr,wg,wb, E, dbg.Epn, dbg.Epd);
   };
   report_ref(1,0,0,"wR");
   report_ref(0,1,0,"wG");
   report_ref(0,0,1,"wB");
   report_ref(1.0/3,1.0/3,1.0/3,"avg");
   report_ref(0.299,0.587,0.114,"BT.601");

   // --- 10) 66 candidates, 0.1 step ---
   Candidate best;
   std::vector<Candidate> ladder;
   ladder.reserve(300);
   coarse_search(10, ReL,GeL,BeL, PN,PD, Dn, lambda_val,mu,sigma, best, &ladder);

   std::sort(ladder.begin(), ladder.end(),
             [](const Candidate &a,const Candidate &b){ return a.E < b.E; });
   zlog("[Zhao18] TOP candidates:");
   for (int i = 0; i < std::min<int>(12,(int)ladder.size()); ++i) {
      const auto &c = ladder[i];
      zlog("  #%d: wr=%.3f wg=%.3f wb=%.3f  E=%.6f  [E_PN=%.6f  E_PD=%.6f]",
           i+1, c.wr,c.wg,c.wb, c.E, c.dbg.Epn, c.dbg.Epd);
   }

   zlog("[Zhao18] Best weights: wr=%.3f wg=%.3f wb=%.3f  E=%.6f  [E_PN=%.6f  E_PD=%.6f]",
        best.wr,best.wg,best.wb, best.E, best.dbg.Epn, best.dbg.Epd);

   auto diag_mixture = [&](const std::vector<PairIdx>& PNPairs,
                           const std::vector<PDPair>& PDPairs,
                           const Deltas& Dn){
      double sLpn=0, sapn=0, sbpn=0;
      for(size_t k=0;k<PNPairs.size();++k){
         const int i=PNPairs[k].i, j=PNPairs[k].j;
         const double dI = std::fabs(
            (best.wr*ReL[i]+best.wg*GeL[i]+best.wb*BeL[i]) -
            (best.wr*ReL[j]+best.wg*GeL[j]+best.wb*BeL[j])
         );
         const double eL = gauss(std::fabs(dI-Dn.dL_PN[k]),mu,sigma);
         const double ea = gauss(std::fabs(dI-Dn.da_PN[k]),mu,sigma);
         const double eb = gauss(std::fabs(dI-Dn.db_PN[k]),mu,sigma);
         sLpn+=eL; sapn+=ea; sbpn+=eb;
      }
      sLpn/=std::max<size_t>(1,PNPairs.size());
      sapn/=std::max<size_t>(1,PNPairs.size());
      sbpn/=std::max<size_t>(1,PNPairs.size());
      zlog("[Zhao18]   P_N: eL=%.6f ea=%.6f eb=%.6f  (cnt=%zu)", sLpn,sapn,sbpn, PNPairs.size());

      double sLpd=0,sapd=0,sbpd=0;
      for(size_t k=0;k<PDPairs.size();++k){
         const int i=PDPairs[k].idx.i, j=PDPairs[k].idx.j;
         const double dI = std::fabs(
            (best.wr*ReL[i]+best.wg*GeL[i]+best.wb*BeL[i]) -
            (best.wr*ReL[j]+best.wg*GeL[j]+best.wb*BeL[j])
         );
         const double eL = gauss(std::fabs(dI-Dn.dL_PD[k]),mu,sigma);
         const double ea = gauss(std::fabs(dI-Dn.da_PD[k]),mu,sigma);
         const double eb = gauss(std::fabs(dI-Dn.db_PD[k]),mu,sigma);
         sLpd+=eL; sapd+=ea; sbpd+=eb;
      }
      sLpd/=std::max<size_t>(1,PDPairs.size());
      sapd/=std::max<size_t>(1,PDPairs.size());
      sbpd/=std::max<size_t>(1,PDPairs.size());
      zlog("[Zhao18]   P_D: eL=%.6f ea=%.6f eb=%.6f  (cnt=%zu)", sLpd,sapd,sbpd, PDPairs.size());
   };
   diag_mixture(PN,PD,Dn);

   // --- 12)FInal grayscale ---
   for (int i = 0; i < N; ++i) {
      double g = best.wr*R[i] + best.wg*G[i] + best.wb*B[i];
      g = clamp01(g);
      *dst++ = g;
      *dst++ = g;
      *dst++ = g;
   }
   pSrc->ProgressBar(H,H);

   {
      double absPN = std::fabs(best.dbg.Epn);
      double absPD = std::fabs(best.dbg.Epd);
      double share = (absPN+absPD)>1e-12 ? (100.0*absPD/(absPN+absPD)) : 0.0;
      zlog("[Zhao18] PD_share=%.2f%%  (by |contribution|)", share);
   }

   return 0;
}
