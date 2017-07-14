
#define FK 0.1

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