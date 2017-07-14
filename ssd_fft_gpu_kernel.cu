#include <cufft.h>

#include "include/ssd_fft_gpu.h"

#define IMUL(a, b) __mul24(a, b)

#ifdef US_SIGNS
	//dead rows starts at the first row, no need to add +1 in iDataIndx calculation in fixDeadPixels kernel
	__device__ __constant__ int iAddOneRow = 0;
#else
	//in EU videos, dead rows start at second row, need to add +1 in iDataIndx calculation in fixDeadPixels kernel
	__device__ __constant__ int iAddOneRow = 1;
#endif

//constant var should be in file scope that is why I got rid of ssd_fft_kernel.h 
//and instead included ssd_fft_kernel.cu in the main.cu (had to deleted customBuild line from
//proj file)
__device__ __constant__ unsigned char d_acLUT[256];

//convert char to float and adjust contrast by doing gamma correction.
__global__ void convertChar4ToFloatDoConGam(uchar4* gd_ac4Scn, float4* d_afScn, int dataN, bool bConGam)
{
	int iIndx = IMUL(blockIdx.x, blockDim.x) + threadIdx.x;
	uchar4 c4DataIn;
	float4 f4DataOut;
	if (iIndx < dataN)
	{
	c4DataIn = gd_ac4Scn[iIndx];
	if (bConGam)
		{
		//doing ConGam takes 0.1 ms more
		f4DataOut.x = (float) d_acLUT[int(c4DataIn.x)];
		f4DataOut.y = (float) d_acLUT[int(c4DataIn.y)];
		f4DataOut.z = (float) d_acLUT[int(c4DataIn.z)];
		f4DataOut.w = (float) d_acLUT[int(c4DataIn.w)];
		}
	else
		{
		f4DataOut.x = (float) c4DataIn.x;
		f4DataOut.y = (float) c4DataIn.y;
		f4DataOut.z = (float) c4DataIn.z;
		f4DataOut.w = (float) c4DataIn.w;
		}
	d_afScn[iIndx] = f4DataOut;
	}
}

//fix dead pixel with averaging 8 immediate neighbors. 
__global__ void fixDeadPixels(cufftReal* d_afScn, int iScnSz, int iScnW, int iScnH)
{
    __shared__ cufftReal afTopRow[(BLOCKDIMX+(HALFWARP+1))];
    __shared__ cufftReal afMidRow[(BLOCKDIMX+(HALFWARP+1))];
    __shared__ cufftReal afBotRow[(BLOCKDIMX+(HALFWARP+1))];
	int iDeadRowDataIndx = IMUL(blockIdx.x, BLOCKDIMX) + (threadIdx.x-HALFWARP);
	int iDataIndx = iDeadRowDataIndx + IMUL((iDeadRowDataIndx/iScnW)+iAddOneRow,iScnW);
	
	afTopRow[threadIdx.x] = 0;
	afMidRow[threadIdx.x] = 0;
	afBotRow[threadIdx.x] = 0;

	if (iDataIndx >= 0 && iDataIndx < iScnSz && threadIdx.x >= HALFWARP-1 && threadIdx.x <= (blockDim.x-1))
	{
		int iRow = iDataIndx/iScnW;
		//read top row
		if (iRow > 0)
			afTopRow[threadIdx.x] = d_afScn[iDataIndx - iScnW];
		//read middle row
		afMidRow[threadIdx.x] = d_afScn[iDataIndx];
		//read bottom row
		if (iRow < iScnH-1)
			afBotRow[threadIdx.x] = d_afScn[iDataIndx + iScnW];
	}

	__syncthreads();
	
	if (iDataIndx >= 0 && iDataIndx < iScnSz && threadIdx.x > HALFWARP-1 && threadIdx.x < (blockDim.x-1))
	{
		cufftReal fSum = 0;
		int iLeftIndx, iRightIndx;
		int iCol = iDataIndx%iScnW;
		if (iCol%2 == 0)
		{
			fSum = fSum + afTopRow[threadIdx.x] + afBotRow[threadIdx.x];
			int iNumNeigh = 2;
			if (iCol > 0)
			{
				iLeftIndx = threadIdx.x - 1 ;
				fSum = fSum + afTopRow[iLeftIndx] + afMidRow[iLeftIndx] + afBotRow[iLeftIndx];
				iNumNeigh = iNumNeigh + 3;
				//fSum = fSum + afMidRow[iLeftIndx];
				//iNumNeigh = iNumNeigh + 1;
			}
			if (iCol < iScnW-1)
			{
				iRightIndx = threadIdx.x + 1 ;
				fSum = fSum + afTopRow[iRightIndx] + afMidRow[iRightIndx] + afBotRow[iRightIndx];
				iNumNeigh = iNumNeigh + 3;
				//fSum = fSum + afMidRow[iRightIndx];
				//iNumNeigh = iNumNeigh + 1;
			}
		d_afScn[iDataIndx] = fSum / (float)iNumNeigh;
		}
	}
}

//take kth law of the data
__global__ void kthLaw(cufftComplex* d_afPadScn, int dataN)
{
	int iIndx = IMUL(blockIdx.x, blockDim.x) + threadIdx.x;
	if (iIndx < dataN)
	{
	//afVals(:) = (abs(afVals(:)).^k) .* (cos(angle(afVals(:))) + sin(angle(afVals(:)))*i);
	cufftComplex cDat = d_afPadScn[iIndx];
	float fNewAbsDat = powf(sqrtf(powf(cDat.x,2)+ powf(cDat.y,2)),FK);
	float fAngDat = atan2f(cDat.y, cDat.x);
	cDat.x = fNewAbsDat*cosf(fAngDat);
	cDat.y = fNewAbsDat*sinf(fAngDat);
	d_afPadScn[iIndx] = cDat;
	}
}

__global__ void pointWiseMul(cufftComplex* d_afCorr, cufftComplex* d_afPadScn, cufftComplex* d_afPadTpl,int dataN, float fScale)
{
	int iIndx = IMUL(blockIdx.x, blockDim.x) + threadIdx.x;
	if (iIndx < dataN)
	{
	cufftComplex cDat = d_afPadScn[iIndx];
	cufftComplex cKer = d_afPadTpl[iIndx];
	//take the conjugate of the kernel
	cKer.y = -cKer.y; 
	cufftComplex cMul = {cDat.x* cKer.x - cDat.y * cKer.y, cDat.y * cKer.x + cDat.x * cKer.y};
	//const float     q = 1.0f / (float)dataN;
    //cMul.x = q * cMul.x;
    //cMul.y = q * cMul.y;

    cMul.x = fScale * cMul.x;
    cMul.y = fScale * cMul.y;
	d_afCorr[iIndx] = cMul;
	}
}


// Complex scale
__global__ void ComplexScale(cufftComplex* a, int size, float scale)
{
    const int numThreads = blockDim.x * gridDim.x;
    const int threadID = blockIdx.x * blockDim.x + threadIdx.x;
    for (int i = threadID; i < size; i += numThreads)
	{
    a[i].x = scale * a[i].x;
    a[i].y = scale * a[i].y;
	}
} 



/* Compute max with reduction (return key-value).
 */

__global__ void max_k(cufftReal* afData, int* aiDataIdxs, int iSizeOfData, cufftReal* afBlockMaxs, int* aiBlockMaxIdxs)
{

	// Block index
    int iBx = blockIdx.x;

    // Thread index
    int iTx = threadIdx.x;

	//Block dim
	int iBDimX = blockDim.x; //BLOCKDIMX_MAX
	__shared__ cufftReal afSubMax[BLOCKDIMX_MAX];
	__shared__ int aiSubMaxIdx[BLOCKDIMX_MAX];

	int iIndx = iBx*(EACHTHREADREADS*iBDimX) + iTx;
	int iIdx;

	//each thread will read EACHTHREADREADS pixels and add them up
	afSubMax[iTx] = 0;
	aiSubMaxIdx[iTx] = -1;
	for (int i = 0; i < EACHTHREADREADS; i++)
		{
			iIdx = iIndx+(i*iBDimX); 
			if ( iIdx < iSizeOfData) 
			{
			afSubMax[iTx] = fmaxf(afSubMax[iTx], afData[iIdx]);
			if (afSubMax[iTx] == afData[iIdx])
				{
				if (aiDataIdxs == NULL)
					aiSubMaxIdx[iTx] = iIdx;
				else
					aiSubMaxIdx[iTx] = aiDataIdxs[iIdx];
				}
			}
		}
	__syncthreads();

	//this for loop does the reduce max!
	for (unsigned int d = iBDimX >> 1; d > 0; d >>= 1) 
    {
        if (iTx < d)
        {
			afSubMax[iTx] = fmaxf(afSubMax[iTx], afSubMax[iTx + d]);
			if (afSubMax[iTx] == afSubMax[iTx + d])
				aiSubMaxIdx[iTx] = aiSubMaxIdx[iTx + d];
        }
		__syncthreads();
    }

	if (iTx == 0) 
	{
		afBlockMaxs[iBx] = afSubMax[0];
		aiBlockMaxIdxs[iBx] = aiSubMaxIdx[0];
	}
}

