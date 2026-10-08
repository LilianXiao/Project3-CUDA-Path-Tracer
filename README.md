CUDA Path Tracer
================

**University of Pennsylvania, CIS 565: GPU Programming and Architecture, Project 3**

<img width="1920" height="1200" alt="greenpath" src="https://github.com/user-attachments/assets/7ba13637-5a06-495f-a444-2a4e0188da04" />

* Lilian Xiao
  * [LinkedIn](https://www.linkedin.com/in/lilian-xiao/), [personal website](https://lilianxiao.carrd.co/), [art website](https://lilianxvis.carrd.co/)
* Tested on: Windows 11, Intel(R) Core(TM) Ultra 9 185H (2.50 GHz), 16.0 GB RAM, NVIDIA GeForce RTX 4070 Laptop GPU (8 GB)
Intel(R) Arc(TM) Graphics (128 MB) (Personal Laptop)

## Overview

Path tracing provides a more physically accurate simulation of light by considering its reflection, refraction, transmission, absorption, and scattering properties.  In basic ray tracing, a ray is emitted from the camera/eye for each pixel of the image.  The intersections of these rays with objects in the scene, depending on the surface behaviors of the objects, can result in secondary rays.  For instance, when the view ray hits a specular object, a secondary ray is emitted.  Ray tracing supports direct illumination and approximates the behavior of light in terms of reflection, refraction, and shadowing.

On the other hand, path tracing can simulate global illumination; in nature, light can come from anywhere, and this results in both direct and indirect lighting.  While ray tracing is a deterministic process, Monte Carlo path tracing is stochastic; we trace multiple rays per pixel because light in nature is random.  Path tracing for more iterations allows the render to converge to a less noisy and more usable image, since each iteration is responsible for tracing a random light path, and the average over these iterations grows closer to the Monte Carlo estimate.

The following two renders of the same Cornell box scene are taken at 7 samples and 204 samples respectively.

<img width="800" height="800" alt="cornell 2026-10-08_17-13-56z 7samp" src="https://github.com/user-attachments/assets/b88680e4-6442-4d08-a844-df262f448bfa" />
<img width="800" height="800" alt="cornell 2026-10-08_17-13-56z 204samp" src="https://github.com/user-attachments/assets/82e55449-a7d9-4f10-b70f-37a104569bc1" />

## Core Features

A standard Monte Carlo pathtracer must trace a single path to completion before tracing the next path.  This project implements a CUDA pathtracer, which leverages the GPU to advance paths simultaneously at each bounce.  It uses kernels that run in parallel and buffers important data that these kernels index into.  For instance, there are kernels for generating rays from the camera, computing intersections, and applying material shading.  Scene data (geometry, triangles, textures, materials, BVH nodes, etc.), image accumulation, and path/intersection information are buffered into flat arrays, as kernels do not recurse via pointers.

### Stream Compaction

Thrust partitioning puts live paths in the front of the buffer.  Material sorting organizes intersections such that they are contiguous in memory by material type.  Within a kernel, we can expect different materials and BSDFs evaluations to finish at different times, but paired with idle threads, this results in a lot of branching.  Material sorting allows for memory adjacency, while stream compaction removes paths that have already been finished.  Overall, this results in fewer threads launched and intersection hits for the same material type being in the same group.

Notably, sorting by material is somewhat expensive.  It's not ideal to use for limited materials (for example, only 1-2 diffuse materials), but the tradeoff becomes worth it for scenes with a lot of materials (for example, having diffuse, reflective, transmissive, subsurface scattering, and procedural materials).

### Stochastic Sampled Anti-Aliasing

This was adapted from Paul Bourke's notes [here](https://paulbourke.net/miscellaneous/raytracing/).  Without anti-aliasing, the "edges" of an object appear to be sharp and jagged like a staircase.  This is because the pixels there must be the object's color or the background/not the object.

In each iteration, a random offset is picked using thrust rng and applied to the pixel coordinates.  Based on the rng seed, the offset will be different per iteration.  Averaging over iterations, this ultimately results in smoother edges.

<img width="224" height="224" alt="image" src="https://github.com/user-attachments/assets/d2bd508c-b7fd-417e-9a01-071bcdfb1655" />
<img width="224" height="224" alt="image" src="https://github.com/user-attachments/assets/5907ba53-ae47-47a4-9c55-fd042474c052" />
