CUDA Path Tracer
================

**University of Pennsylvania, CIS 565: GPU Programming and Architecture, Project 3**

* (TODO) YOUR NAME HERE
* Tested on: (TODO) Windows 22, i7-2222 @ 2.22GHz 22GB, GTX 222 222MB (Moore 2222 Lab)

### (TODO: Your README)

*DO NOT* leave the README to the last minute! It is a crucial part of the
project, and we will not be able to grade you without a good README.

## Part 2: Additional Features

1. Refraction (2pts). For this feature, sampling must randomly choose between reflection and transmission. Sample proportional to Fresnel reflectance R and complementary transmittance 1-R. 
	- Additional kernel: `FrDielectric` which implements Fresnel equation and Dialectric BSDF from PBR 9.5


2. Depth of Field (2 pts - PBRTv4 5.2.3):
	- Added focal distance and lens size to the Camera object