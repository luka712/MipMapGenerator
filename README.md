# Mip Map generation
---

Used to create [mip maps](https://en.wikipedia.org/wiki/Mipmap) which can be used for various rendering libraries (Vulkan, D3D12, Metal ...).

MipMaps are used as textures that are loaded when rendered object is far from the "camera" or when viewing rendered object from an angle (such as floor). In those case it is desirable to have smaller samples of primary texture, which mip map provides.

![MipMap](data/Lena_mipmap.png)

## To use the code
---

Since I am mainly Windows user, I used Visual Studio (for better or worse) to create example.
Simply openning `MipMapGenerator.sln` and building the project should be enought to build the code. By default, it is output to `.bin` folder.

## Details
---
