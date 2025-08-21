/* Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *  * Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *  * Neither the name of NVIDIA CORPORATION nor the names of its
 *    contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 * OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#if defined(WIN32) || defined(_WIN32) || defined(WIN64) || defined(_WIN64)
#define WINDOWS_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#pragma warning(disable : 4819)
#endif

#include <Exceptions.h>
#include <ImageIO.h>
#include <ImagesCPU.h>
#include <ImagesNPP.h>

#include <string.h>
#include <fstream>
#include <iostream>
#include <vector>

#include <cuda.h>
#include <cuda_runtime.h>
#include <npp.h>        // umbrella include (pulls most of NPP)
#include <nppcore.h>    // for nppSetStream / nppGetStreamContext
#include <nppi.h>       // for image processing functions (resize, filter, etc.)

#include <helper_cuda.h>
#include <helper_string.h>

#include "MipMapChunk.h"

/// <summary>
/// Creates a mipmap image from the source image using NPP's resize function.
/// </summary>
/// <param name="srcImage">The src image.</param>
/// <param name="streamCtx">The stream ctx.</param>
/// <param name="outDstImage">The pointer to the destination image.</param>
/// <param name="interpolationMode">The mode for interpolation.</param>
void createMipMap(
	npp::ImageNPP_8u_C4& srcImage, 
	NppStreamContext& streamCtx,
	npp::ImageNPP_8u_C4* outDstImage,
	int interpolationMode)
{
	if (srcImage.data() == NULL)
	{
		std::cout << "Failed to allocate device image memory." << std::endl;
		throw npp::Exception("Failed to allocate device image memory.");
	}


	NppiSize srcSize = { srcImage.width(), srcImage.height() };
	NppiRect srcROI = { 0, 0, srcSize.width, srcSize.height };

	// We start writing to right and top and then move Y for size of new mip map on each iteration.
	NppiSize finalSize = { srcSize.width * 1.5, srcSize.height };
	npp::ImageNPP_8u_C4 dstDeviceMemory(finalSize.width, finalSize.height);

	// Destination
	int step = 2;
	std::vector<MipMapChunk*> chunks;

	while (true)
	{
		int width = srcSize.width / step;
		int height = srcSize.height / step;

		// Create chunk and increase steps
		MipMapChunk* chunk = new MipMapChunk(srcImage, width, height, streamCtx, interpolationMode);
		step *= 2;
		chunks.push_back(chunk);

		// If 1x1 was last, break
		if (width <= 1 && height <= 1)
		{
			break;
		}
	}

	cudaStreamSynchronize(streamCtx.hStream); // wait only for this stream

	// Copy source image first then chunks
	NPP_CHECK_NPP(nppiCopy_8u_C4R_Ctx(
		srcImage.data(), srcImage.pitch(),
		dstDeviceMemory.data(), dstDeviceMemory.pitch(),
		srcSize,
		streamCtx));

	// Copy chunks now to create mip map.
	int y = 0;
	for (auto& chunk : chunks)
	{
		// Copy chunk to final.
		NPP_CHECK_NPP(nppiCopy_8u_C4R_Ctx(
			chunk->GPUMemory.data(), chunk->GPUMemory.pitch(),
			dstDeviceMemory.data() + y * dstDeviceMemory.pitch() + srcSize.width * 4, dstDeviceMemory.pitch(),
			chunk->Size,
			streamCtx));

		y += chunk->Size.height;
	}

	*outDstImage = dstDeviceMemory;
}

bool initCudaAndSetupStream(NppStreamContext& nppStreamCtx)
{
	// Create a CUDA device and set it as the current device.
	int deviceCount = 0;
	NPP_CHECK_CUDA(cudaGetDeviceCount(&deviceCount));

	if (deviceCount == 0)
	{
		std::cerr << "No CUDA devices found." << std::endl;
		return false;
	}

	NPP_CHECK_CUDA(cudaGetDevice(&nppStreamCtx.nCudaDeviceId));

	nppStreamCtx.hStream = 0; // The NULL stream by default, set this to whatever your stream ID is if not the NULL stream.

	int driverVersion, runtimeVersion;
	cudaDriverGetVersion(&driverVersion);
	cudaRuntimeGetVersion(&runtimeVersion);

	printf("CUDA Driver  Version: %d.%d\n", driverVersion / 1000, (driverVersion % 100) / 10);
	printf("CUDA Runtime Version: %d.%d\n\n", runtimeVersion / 1000, (runtimeVersion % 100) / 10);

	NPP_CHECK_CUDA(cudaDeviceGetAttribute(&nppStreamCtx.nCudaDevAttrComputeCapabilityMajor,
		cudaDevAttrComputeCapabilityMajor,
		nppStreamCtx.nCudaDeviceId));
	

	NPP_CHECK_CUDA(cudaDeviceGetAttribute(&nppStreamCtx.nCudaDevAttrComputeCapabilityMinor,
		cudaDevAttrComputeCapabilityMinor,
		nppStreamCtx.nCudaDeviceId));

	NPP_CHECK_CUDA(cudaStreamGetFlags(nppStreamCtx.hStream, &nppStreamCtx.nStreamFlags));

	cudaDeviceProp oDeviceProperties;

	NPP_CHECK_CUDA(cudaGetDeviceProperties(&oDeviceProperties, nppStreamCtx.nCudaDeviceId));

	nppStreamCtx.nMultiProcessorCount = oDeviceProperties.multiProcessorCount;
	nppStreamCtx.nMaxThreadsPerMultiProcessor = oDeviceProperties.maxThreadsPerMultiProcessor;
	nppStreamCtx.nMaxThreadsPerBlock = oDeviceProperties.maxThreadsPerBlock;
	nppStreamCtx.nSharedMemPerBlock = oDeviceProperties.sharedMemPerBlock;


	return true;
}


/// <summary>
/// Comma split string and fill in result array.
/// </summary>
/// <param name="str">The string to split by ','.</param>
/// <param name="result">The array where split parts fill be filled at.</param>
void commaSplitString(std::string str, std::vector<std::string>& result)
{
	std::string currentStr;

	for (int i = 0; i < str.length(); i++)
	{
		char c = str.c_str()[i];
		if (c == ',')
		{
			result.push_back(currentStr);
			currentStr = "";
		}
		else
		{
			currentStr += c;
		}
	}

	if (currentStr != "")
	{
		result.push_back(currentStr);
	}
}


bool getInputFileParameter(int argc, char* argv[], std::vector<std::string>& result)
{
	char* inputFilePath;

	if (checkCmdLineFlag(argc, (const char**)argv, "--input"))
	{
		getCmdLineArgumentString(argc, (const char**)argv, "--input", &inputFilePath);
	}
	else if (checkCmdLineFlag(argc, (const char**)argv, "--i"))
	{
		getCmdLineArgumentString(argc, (const char**)argv, "--i", &inputFilePath);
	}
	else
	{
		std::cout << "No input file specified. Use --input <filename> or --i <filename>." << std::endl;
		return false;
	}

	commaSplitString(std::string(inputFilePath), result);

	return true;
}

void getOutputFileParameter(int argc, char* argv[], std::vector<std::string>& result)
{
	char* outputFilePath = nullptr;

	if (checkCmdLineFlag(argc, (const char**)argv, "--output"))
	{
		getCmdLineArgumentString(argc, (const char**)argv, "--output", &outputFilePath);
	}
	else if (checkCmdLineFlag(argc, (const char**)argv, "--o"))
	{
		getCmdLineArgumentString(argc, (const char**)argv, "--o", &outputFilePath);
	}

	if (outputFilePath != nullptr)
	{
		commaSplitString(std::string(outputFilePath), result);
	}
}

void getInterpolationMode(int argc, char* argv[], int* outMode)
{
	int mode = NPPI_INTER_LINEAR;
	char* modeStr = nullptr;

	if (checkCmdLineFlag(argc, (const char**)argv, "--mode"))
	{
		getCmdLineArgumentString(argc, (const char**)argv, "--mode", &modeStr);
	}
	else if (checkCmdLineFlag(argc, (const char**)argv, "--m"))
	{
		getCmdLineArgumentString(argc, (const char**)argv, "--m", &modeStr);
	}

	if (modeStr == "1")
	{
		mode = NPPI_INTER_NN;
		std::cout << "MipMaps will be genereted with NearestNeighbour filter." << std::endl;
	}
	if (modeStr == "2")
	{
		mode = NPPI_INTER_CUBIC;
		std::cout << "MipMaps will be genereted with Cubic interpolation." << std::endl;

	}
	else
	{
		std::cout << "MipMaps will be genereted with Linear interpolation." << std::endl;
	}

	*outMode = mode;
}

std::string trimExtension(std::string str)
{
	int index = str.find_last_of('.');
	std::string subStr = str.substr(0, index);
	return subStr;
}


int main(int argc, char* argv[])
{
	printf("%s Starting...\n\n", argv[0]);

	NppStreamContext nppStreamCtx;
	if (!initCudaAndSetupStream(nppStreamCtx))
	{
		return EXIT_FAILURE;
	}

	// INPUT
	std::vector<std::string> inputs;
	if (!getInputFileParameter(argc, argv, inputs))
	{
		// Provide default argument if none are provided.
		std::cout << "Using ./data/Lena.png" << std::endl;
		inputs.push_back("./data/Lena.png");
	}

	// OUTPUT
	std::vector<std::string> outputs;
	getOutputFileParameter(argc, argv, outputs);

	// INTERPOLATION
	int mode;
	getInterpolationMode(argc, argv, &mode);

	try
	{
		for (int i = 0; i < inputs.size(); i++)
		{
			std::string input = inputs[i];

			// If there is output file name provided, use it instead.
			std::string output = trimExtension(input) + "_mipmap.png";
			if (outputs.size() > i)
			{
				output = outputs[i];
			}

			// Load image.
			npp::ImageCPU_8u_C4 inputHostMemory;
			if (npp::loadImage(input, &inputHostMemory))
			{
				std::cout << "Loaded image " << input << std::endl;
			}	
			else
			{
				std::cout << "Failed to load image " << input << std::endl;
				return EXIT_FAILURE;
			}

			// Generate mip map
			npp::ImageNPP_8u_C4 inputDeviceMemory(inputHostMemory);
			npp::ImageNPP_8u_C4 outputDeviceMemory;
			std::cout << "Creating mipmap for " << input << std::endl;
			createMipMap(inputDeviceMemory, nppStreamCtx, &outputDeviceMemory, mode);
			std::cout << "Mipmap created for " << input << std::endl;
			npp::ImageCPU_8u_C4 outputHostMemory(outputDeviceMemory.size());
			outputDeviceMemory.copyTo(outputHostMemory.data(), outputHostMemory.pitch());

			std::cout << "Saving MipMap" << output << std::endl;
			npp::saveImage(output, outputHostMemory);

		}
	}
	catch (npp::Exception& rException)
	{
		std::cerr << "Program error! The following exception occurred: \n";
		std::cerr << rException << std::endl;
		std::cerr << "Aborting." << std::endl;

		exit(EXIT_FAILURE);
	}
	catch (...)
	{
		std::cerr << "Program error! An unknown type of exception occurred. \n";
		std::cerr << "Aborting." << std::endl;

		exit(EXIT_FAILURE);
		return -1;
	}

	return 0;
}
