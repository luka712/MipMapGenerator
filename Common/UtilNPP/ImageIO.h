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

#ifndef NV_UTIL_NPP_IMAGE_IO_H
#define NV_UTIL_NPP_IMAGE_IO_H

#include "ImagesCPU.h"
#include "ImagesNPP.h"

#include "FreeImage.h"
#include "Exceptions.h"

#include <string>
#include "string.h"


 // Error handler for FreeImage library.
 //  In case this handler is invoked, it throws an NPP exception.
void
FreeImageErrorHandler(FREE_IMAGE_FORMAT oFif, const char* zMessage)
{
    throw npp::Exception(zMessage);
}

namespace npp
{
    // Load a gray-scale image from disk.
    void
        loadImage(const std::string& rFileName, ImageCPU_8u_C1& rImage)
    {
        // set your own FreeImage error handler
        FreeImage_SetOutputMessage(FreeImageErrorHandler);

        FREE_IMAGE_FORMAT eFormat = FreeImage_GetFileType(rFileName.c_str());

        // no signature? try to guess the file format from the file extension
        if (eFormat == FIF_UNKNOWN)
        {
            eFormat = FreeImage_GetFIFFromFilename(rFileName.c_str());
        }

        NPP_ASSERT(eFormat != FIF_UNKNOWN);
        // check that the plugin has reading capabilities ...
        FIBITMAP* pBitmap;

        if (FreeImage_FIFSupportsReading(eFormat))
        {
            pBitmap = FreeImage_Load(eFormat, rFileName.c_str());
        }

        NPP_ASSERT(pBitmap != 0);
        // make sure this is an 8-bit single channel image
        NPP_ASSERT(FreeImage_GetColorType(pBitmap) == FIC_MINISBLACK);
        NPP_ASSERT(FreeImage_GetBPP(pBitmap) == 8);

        // create an ImageCPU to receive the loaded image data
        ImageCPU_8u_C1 oImage(FreeImage_GetWidth(pBitmap), FreeImage_GetHeight(pBitmap));

        // Copy the FreeImage data into the new ImageCPU
        unsigned int nSrcPitch = FreeImage_GetPitch(pBitmap);
        const Npp8u* pSrcLine = FreeImage_GetBits(pBitmap) + nSrcPitch * (FreeImage_GetHeight(pBitmap) - 1);
        Npp8u* pDstLine = oImage.data();
        unsigned int nDstPitch = oImage.pitch();

        for (size_t iLine = 0; iLine < oImage.height(); ++iLine)
        {
            memcpy(pDstLine, pSrcLine, oImage.width() * sizeof(Npp8u));
            pSrcLine -= nSrcPitch;
            pDstLine += nDstPitch;
        }

        // swap the user given image with our result image, effecively
        // moving our newly loaded image data into the user provided shell
        oImage.swap(rImage);
    }

    // Load a image from disk.
    bool loadImage(const std::string& fileName, ImageCPU_8u_C4* outCPUMemory)
    {
        // Load image
        FIBITMAP* dib = FreeImage_Load(FIF_PNG, fileName.c_str(), PNG_DEFAULT);
        if (!dib) 
        {
            std::cerr << "Failed to load image\n";
            return false;
        }

        // Convert to 32-bit RGBA
        FIBITMAP* dib32 = FreeImage_ConvertTo32Bits(dib);
        FreeImage_Unload(dib); // free original

        int width = FreeImage_GetWidth(dib32);
        int height = FreeImage_GetHeight(dib32);
        BYTE* h_img = FreeImage_GetBits(dib32); // BGRA (note FreeImage default!)

        // Create CPU-side NPP image wrapper (width x height, 4 channels, 8-bit each)
        npp::ImageCPU_8u_C4 oHostSrc(width, height);

        // Copy row by row because FreeImage might have pitch != width*4
        int destIndex = height - 1; // Flips image.
        for (int srcIndex = 0; srcIndex < height; ++srcIndex) 
        {
            memcpy(oHostSrc.data() + srcIndex * oHostSrc.pitch(),
                h_img + destIndex * FreeImage_GetPitch(dib32),
                width * 4);
            destIndex--;
        }
        FreeImage_Unload(dib32);

        *outCPUMemory = oHostSrc;
    }

    // Save an gray-scale image to disk.
    void
        saveImage(const std::string& rFileName, const ImageCPU_8u_C1& rImage)
    {
        // create the result image storage using FreeImage so we can easily
        // save
        FIBITMAP* pResultBitmap = FreeImage_Allocate(rImage.width(), rImage.height(), 8 /* bits per pixel */);
        NPP_ASSERT_NOT_NULL(pResultBitmap);
        unsigned int nDstPitch = FreeImage_GetPitch(pResultBitmap);
        Npp8u* pDstLine = FreeImage_GetBits(pResultBitmap) + nDstPitch * (rImage.height() - 1);
        const Npp8u* pSrcLine = rImage.data();
        unsigned int nSrcPitch = rImage.pitch();

        for (size_t iLine = 0; iLine < rImage.height(); ++iLine)
        {
            memcpy(pDstLine, pSrcLine, rImage.width() * sizeof(Npp8u));
            pSrcLine += nSrcPitch;
            pDstLine -= nDstPitch;
        }

        // now save the result image
        bool bSuccess;
        bSuccess = FreeImage_Save(FIF_PGM, pResultBitmap, rFileName.c_str(), 0) == TRUE;
        NPP_ASSERT_MSG(bSuccess, "Failed to save result image.");
    }

    // Save an gray-scale image to disk.
    void
        saveImage(const std::string& rFileName, const ImageCPU_8u_C4& rImage)
    {
        // create the result image storage using FreeImage so we can easily
        // save
        FIBITMAP* pResultBitmap = FreeImage_Allocate(rImage.width(), rImage.height(), 8 * 4 /* bits per pixel */);
        NPP_ASSERT_NOT_NULL(pResultBitmap);
        unsigned int nDstPitch = FreeImage_GetPitch(pResultBitmap);
        Npp8u* pDstLine = FreeImage_GetBits(pResultBitmap) + nDstPitch * (rImage.height() - 1);
        const Npp8u* pSrcLine = rImage.data();
        unsigned int nSrcPitch = rImage.pitch();

        for (size_t iLine = 0; iLine < rImage.height(); ++iLine)
        {
            memcpy(pDstLine, pSrcLine, rImage.width() * 4);
            pSrcLine += nSrcPitch;
            pDstLine -= nDstPitch;
        }

        // now save the result image
        bool bSuccess;
        bSuccess = FreeImage_Save(FIF_PNG, pResultBitmap, rFileName.c_str(), 0) == TRUE;
        NPP_ASSERT_MSG(bSuccess, "Failed to save result image.");
    }

    // Load a gray-scale image from disk.
    void
        loadImage(const std::string& rFileName, ImageNPP_8u_C1& rImage)
    {
        ImageCPU_8u_C1 oImage;
        loadImage(rFileName, oImage);
        ImageNPP_8u_C1 oResult(oImage);
        rImage.swap(oResult);
    }

    // Save an gray-scale image to disk.
    void
        saveImage(const std::string& rFileName, const ImageNPP_8u_C1& rImage)
    {
        ImageCPU_8u_C1 oHostImage(rImage.size());
        // copy the device result data
        rImage.copyTo(oHostImage.data(), oHostImage.pitch());
        saveImage(rFileName, oHostImage);
    }
}


#endif // NV_UTIL_NPP_IMAGE_IO_H
