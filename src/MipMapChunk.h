#pragma once

#include <nppdefs.h>
#include <ImagesNPP.h>
#include <helper_cuda.h>
#include <helper_string.h>


struct MipMapChunk
{
public:
	/// <summary>
	/// Construct a new mip map chunk.
	/// </summary>
	/// <param name="srcImage">The source image memory.</param>
	/// <param name="width">The desired width.</param>
	/// <param name="height">The desired height.</param>
	/// <param name="streamCtx">The stream context.</param>
	/// <param name="eInterpolation">Interpolation method to use, by defualt it is NPPI_INTER_LINEAR</param>
	MipMapChunk(npp::ImageNPP_8u_C4& srcImage, int width, int height, NppStreamContext& streamCtx, int eInterpolation = NPPI_INTER_LINEAR)
	{
		// Source
		NppiSize srcSize = { srcImage.width(), srcImage.height() };
		NppiRect srcROI = { 0,0, srcSize.width, srcSize.height };

		// This destination.
		Size = { width, height };
		NppiRect roi = { 0, 0, Size.width, Size.height };

		GPUMemory = npp::ImageNPP_8u_C4(Size.width, Size.height);

		NPP_CHECK_NPP(nppiResize_8u_C4R_Ctx(
			srcImage.data(), srcImage.pitch(), srcSize, srcROI,
			GPUMemory.data(), GPUMemory.pitch(), Size, roi,
			eInterpolation, streamCtx));
	}

	npp::ImageNPP_8u_C4 GPUMemory;

	NppiSize Size;

};

