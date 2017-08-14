#ifdef US_SIGNS
//dead rows starts at the first row, no need to add +1 in iDataIndx calculation in fixDeadPixels kernel
__constant int iAddOneRow = 0;
#else
//in EU videos, dead rows start at second row, need to add +1 in iDataIndx calculation in fixDeadPixels kernel
__constant int iAddOneRow = 1;
#endif
#define FK 0.1
#define cufftComplex float2
#define cufftReal float
#define BLOCKDIMX 256
#define BLOCKDIMX_MAX 512
#define EACHTHREADREADS 16
#define HALFWARP 16
#define IMUL(a, b) a*b
#define NULL 0
__constant unsigned char d_acLUT[256] = { 0 }; // currently can't initial from cuda 



__kernel void convertChar4ToFloatDoConGam(__global uchar4* restrict gd_ac4Scn, __global float4* restrict d_afScn, int dataN, int bConGam)
{
	//printf("------------------------%d", bConGam);
	uchar4 c4DataIn;
	float4 f4DataOut;
    for (int iIndx = 0; iIndx < dataN; iIndx++)
	{
		c4DataIn = gd_ac4Scn[iIndx];
		if (bConGam)
		{
			//doing ConGam takes 0.1 ms more //invalid for now
			f4DataOut.x = (float)d_acLUT[(int)(c4DataIn.x)];
			f4DataOut.y = (float)d_acLUT[(int)(c4DataIn.y)];
			f4DataOut.z = (float)d_acLUT[(int)(c4DataIn.z)];
			f4DataOut.w = (float)d_acLUT[(int)(c4DataIn.w)];
			//printf("%d",bConGam);
		}
		else
		{
			f4DataOut.x = (float)c4DataIn.x;
			f4DataOut.y = (float)c4DataIn.y;
			f4DataOut.z = (float)c4DataIn.z;
			f4DataOut.w = (float)c4DataIn.w;
			//printf("here2");
		}
		d_afScn[iIndx] = f4DataOut;
	}
}

//fix dead pixel with averaging 8 immediate neighbors. 
__kernel void fixDeadPixels(__global float* d_afScn, int iScnSz, int iScnW, int iScnH)
{
	__local cufftReal afTopRow[(BLOCKDIMX + (HALFWARP + 1))];
	__local cufftReal afMidRow[(BLOCKDIMX + (HALFWARP + 1))];
	__local cufftReal afBotRow[(BLOCKDIMX + (HALFWARP + 1))];
	for (int b = 0; b < 600; b++) { // emulate the GridDim as it is in CUDA
		for (int t = 0; t < BLOCKDIMX; t++) { // emulate the BlockDim as it is in CUDA
			int blockIdx = t / BLOCKDIMX;
			int threadIdx = t % BLOCKDIMX;
			int blockDim = BLOCKDIMX;

			int iDeadRowDataIndx = IMUL(blockIdx, BLOCKDIMX) + (threadIdx - HALFWARP);
			int iDataIndx = iDeadRowDataIndx + IMUL((iDeadRowDataIndx / iScnW) + iAddOneRow, iScnW);

			afTopRow[threadIdx] = 0;
			afMidRow[threadIdx] = 0;
			afBotRow[threadIdx] = 0;

			if (iDataIndx >= 0 && iDataIndx < iScnSz && threadIdx >= HALFWARP - 1 && threadIdx <= (blockDim - 1))
			{
				int iRow = iDataIndx / iScnW;
				//read top row
				if (iRow > 0)
					afTopRow[threadIdx] = d_afScn[iDataIndx - iScnW];
				//read middle row
				afMidRow[threadIdx] = d_afScn[iDataIndx];
				//read bottom row
				if (iRow < iScnH - 1)
					afBotRow[threadIdx] = d_afScn[iDataIndx + iScnW];
			}

			barrier(CLK_LOCAL_MEM_FENCE);

			if (iDataIndx >= 0 && iDataIndx < iScnSz && threadIdx > HALFWARP - 1 && threadIdx < (blockDim - 1))
			{
				cufftReal fSum = 0;
				int iLeftIndx, iRightIndx;
				int iCol = iDataIndx%iScnW;
				if (iCol % 2 == 0)
				{
					fSum = fSum + afTopRow[threadIdx] + afBotRow[threadIdx];
					int iNumNeigh = 2;
					if (iCol > 0)
					{
						iLeftIndx = threadIdx - 1;
						fSum = fSum + afTopRow[iLeftIndx] + afMidRow[iLeftIndx] + afBotRow[iLeftIndx];
						iNumNeigh = iNumNeigh + 3;
						//fSum = fSum + afMidRow[iLeftIndx];
						//iNumNeigh = iNumNeigh + 1;
					}
					if (iCol < iScnW - 1)
					{
						iRightIndx = threadIdx + 1;
						fSum = fSum + afTopRow[iRightIndx] + afMidRow[iRightIndx] + afBotRow[iRightIndx];
						iNumNeigh = iNumNeigh + 3;
						//fSum = fSum + afMidRow[iRightIndx];
						//iNumNeigh = iNumNeigh + 1;
					}
					d_afScn[iDataIndx] = fSum / (float)iNumNeigh;
				}
			}
		}
	}
}

__kernel void kthLaw(__global float2* d_afPadScn, int dataN)
{
	//int iIndx = get_global_id(0);
	for(int iIndx=0; iIndx < dataN; iIndx++)
	{
		//afVals(:) = (abs(afVals(:)).^k) .* (cos(angle(afVals(:))) + sin(angle(afVals(:)))*i);
		float2 cDat = d_afPadScn[iIndx];
		float fNewAbsDat = pow(sqrtf(pow(cDat.x, 2) + pow(cDat.y, 2)), FK);
		float fAngDat = atan2(cDat.y, cDat.x);
		cDat.x = fNewAbsDat*cosf(fAngDat);
		cDat.y = fNewAbsDat*sinf(fAngDat);
		d_afPadScn[iIndx] = cDat;
	}
}

__kernel void pointWiseMul(__global float2* restrict d_afCorr, __global float2* restrict d_afPadScn, __global float2* restrict d_afPadTpl, int dataN, float fScale)
{
	//int iIndx = IMUL(blockIdx.x, blockDim.x) + threadIdx.x;
	for (int iIndx = 0; iIndx < dataN; iIndx++)
	{
		float2 cDat = d_afPadScn[iIndx];
		float2 cKer = d_afPadTpl[iIndx];
		//take the conjugate of the kernel
		cKer.y = -cKer.y;
		float2 cMul = { cDat.x* cKer.x - cDat.y * cKer.y, cDat.y * cKer.x + cDat.x * cKer.y };
		//const float     q = 1.0f / (float)dataN;
		//cMul.x = q * cMul.x;
		//cMul.y = q * cMul.y;

		cMul.x = fScale * cMul.x;
		cMul.y = fScale * cMul.y;
		d_afCorr[iIndx] = cMul;
	}
}

__kernel void ComplexScale(__global float2* a, int size, float scale)
{
	for (int i = 0; i < size; i ++)
	{
		a[i].x = scale * a[i].x;
		a[i].y = scale * a[i].y;
	}
}


__kernel void max_k(__global  cufftReal* restrict  afData, __global int* restrict aiDataIdxs, int iSizeOfData,
	__global cufftReal* restrict afBlockMaxs, __global int* restrict aiBlockMaxIdxs, int iGDx) //add iGDx as input
{
	__local cufftReal afSubMax[BLOCKDIMX_MAX];
	__local int aiSubMaxIdx[BLOCKDIMX_MAX];
	for (int b = 0; b < iGDx; b++) { // emulate the GridDim as it is in CUDA
		for (int t = 0; t < BLOCKDIMX_MAX; t++) { // emulate the BlockDim as it is in CUDA

			// Block index
			int iBx = b;
			// Thread index
			int iTx = t;
			//Block dim
			int iBDimX = BLOCKDIMX_MAX; //blockDim.x; BLOCKDIMX_MAX
			int iIndx = iBx*(EACHTHREADREADS*iBDimX) + iTx;
			int iIdx;

			//each thread will read EACHTHREADREADS pixels and add them up
			afSubMax[iTx] = 0;
			aiSubMaxIdx[iTx] = -1;
			for (int i = 0; i < EACHTHREADREADS; i++)
			{
				iIdx = iIndx + (i*iBDimX);
				if (iIdx < iSizeOfData)
				{
					afSubMax[iTx] = fmax(afSubMax[iTx], afData[iIdx]);
					if (afSubMax[iTx] == afData[iIdx])
					{
						if (aiDataIdxs == NULL)
							aiSubMaxIdx[iTx] = iIdx;
						else
							aiSubMaxIdx[iTx] = aiDataIdxs[iIdx];
					}
				}
			}
			barrier(CLK_LOCAL_MEM_FENCE);

			//this for loop does the reduce max!
			for (unsigned int d = iBDimX >> 1; d > 0; d >>= 1)
			{
				if (iTx < d)
				{
					afSubMax[iTx] = fmax(afSubMax[iTx], afSubMax[iTx + d]);
					if (afSubMax[iTx] == afSubMax[iTx + d])
						aiSubMaxIdx[iTx] = aiSubMaxIdx[iTx + d];
				}
				barrier(CLK_LOCAL_MEM_FENCE);
			}

			if (iTx == 0)
			{
				afBlockMaxs[iBx] = afSubMax[0];
				aiBlockMaxIdxs[iBx] = aiSubMaxIdx[0];
			}
		}
	}
}
