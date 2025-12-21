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
*                       Implementation of the TMOKhudair23 class                    *
*                                                                                   *
************************************************************************************/

#include "TMOKhudair23.h"
#include "matrix.h"     // tmolib: Matrix, Diagonal, mysvd
#include <cmath>

// --------------------------------------------------------------------------- //
// TMOKhudair23: ctor/params
// --------------------------------------------------------------------------- //
TMOKhudair23::TMOKhudair23()
{
   SetName(L"Khudair23");
   SetDescription(L"Color-to-grayscale conversion based on SVD "
                  L"(Khudhair et al., 2023).");

   Kappa.SetName(L"k");
   Kappa.SetDescription(
      L"Divisor k in G = ||S|| / k (Eq. (6)); "
      L"paper examples use k = 2."
   );
   Kappa.SetRange(0.01, 100.0);
   Kappa = 2.0; 
   this->Register(Kappa);

   Mode.SetName(L"Mode");
   Mode.SetDescription(
      L"Select which channel is weighted by factor 3:1==R, 2==G, 3==B\n"
   );
   Mode.SetDefault(1);        // paper often shows red-weighted as default
   Mode = 0;
   Mode.SetRange(1, 3);
   this->Register(Mode);
}

TMOKhudair23::~TMOKhudair23()
{
   // required for vtable
}

// --------------------------------------------------------------------------- //
// Transform
// --------------------------------------------------------------------------- //

int TMOKhudair23::Transform()
{
   int mode = static_cast<int>(Mode);

   pSrc->Convert(TMO_RGB);
   pDst->Convert(TMO_RGB);

   const int width  = pSrc->GetWidth();
   const int height = pSrc->GetHeight();

   double *src = pSrc->GetData();
   double *dst = pDst->GetData();

   // k from Eq. (6)
   double k = static_cast<double>(Kappa);
   if (k == 0.0)
      k = 1.0; // no division by zero

   const double weight = 3.0;


   for (int y = 0; y < height; ++y)
   {
      pSrc->ProgressBar(y, height);

      for (int x = 0; x < width; ++x)
      {
         // Step 6–7: read pixel and separate channels
         double xr = *src++;  // Red
         double xg = *src++;  // Green
         double xb = *src++;  // Blue

         // Step 8: create vector for each pixel: C = [xr, xg, xb]
         // Step 9: add weight to one parameter:
         mtx::Matrix C(1, 3);
         switch (mode)
         {
         case 1: // C1
            C[0][0] = weight * xr;
            C[0][1] = xg;
            C[0][2] = xb;
            break;
         case 2: // C2
            C[0][0] = xr;
            C[0][1] = weight * xg;
            C[0][2] = xb;
            break;
         case 3: // C3
            C[0][0] = xr;
            C[0][1] = xg;
            C[0][2] = weight * xb;
            break;
         }

         // Step 10: [U S V] = SVD(C(i,j))
         mtx::Matrix U, V;
         mtx::Diagonal S;
         mtx::mysvd(C, U, S, V);

         // Eq. (5)
         int n = S.dmin();   // number of singular values
         double sumSq = 0.0;
         for (int i = 0; i < n; ++i)
         {
            double s = S[i];
            sumSq += s * s;
         }
         double normS = std::sqrt(sumSq);

         // Step 11:(Eq. (6))
         double gray = normS / k;

         //Step 12: GrayImage(i,j) = G
         *dst++ = gray;  // R'
         *dst++ = gray;  // G'
         *dst++ = gray;  // B'
      }
   }

   pSrc->ProgressBar(height, height);
   return 0;
}
