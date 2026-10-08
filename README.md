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

This was adapted from Paul Bourke's notes [here](https://paulbourke.net/miscellaneous/raytracing/).  Without anti-aliasing, the "edges" of an object appear to be sharp and jagged like a staircase.  This is because the pixels there must be the object's color or the background/not the object.  One might notice how the jaggies become more apparent for images with smaller resolution.

In each iteration, a random offset is picked using thrust rng and applied to the pixel coordinates.  Based on the rng seed, the offset will be different per iteration.  Averaging over iterations, this ultimately results in smoother edges.

<img width="224" height="224" alt="image" src="https://github.com/user-attachments/assets/d2bd508c-b7fd-417e-9a01-071bcdfb1655" />
<img width="224" height="224" alt="image" src="https://github.com/user-attachments/assets/5907ba53-ae47-47a4-9c55-fd042474c052" />

## Additional Features

Multiple additional features were implemented in this pathtracer, from materials to optimization techniques.

### Fresnel-Schlick Effects for Refraction

Reflective and transmissive materials using [Schlick's Approximation](https://en.wikipedia.org/wiki/Schlick's_approximation).

<img width="800" height="800" alt="cornell 2026-10-07_05-08-23z 412samp" src="https://github.com/user-attachments/assets/d441bc0f-b9b4-4481-bf61-897a3ea97009" />

<img width="800" height="800" alt="cornell 2026-10-07_05-09-38z 330samp" src="https://github.com/user-attachments/assets/2e33411a-8f01-4b87-b33f-d5b57c755a80" />

<img width="800" height="800" alt="cornell 2026-10-07_05-12-38z 362samp" src="https://github.com/user-attachments/assets/bbb4f789-b362-4abf-b9ca-b0ec2dc287b0" />

In the fresnel effect, light either passes through the material or is reflected.  This reflection is viewing-angle dependent, as reflection is greater at glancing angles.  Schlick's approximation evaluates the fraction of light reflected (F) and the fraction transmitted (1 - F). 

<img width="1200" height="1200" alt="cornell 2026-10-08_13-29-28z 201samp" src="https://github.com/user-attachments/assets/4b52573f-610f-420f-a865-7cad78653def" />

<img width="1200" height="1200" alt="cornell 2026-10-08_13-26-12z 88samp" src="https://github.com/user-attachments/assets/acfb655c-8a52-45ae-9cb8-5066712ed687" />

### Physically-Based Depth Of Field

<img width="1200" height="1200" alt="cornell 2026-10-07_10-55-15z 262samp" src="https://github.com/user-attachments/assets/53c81a42-34c3-4291-9bec-d0bba580a78c" />

Originally, the camera is a lensless pinhole that is perfectly focused on everything captured in the scene.  However, adding a lens radius and focal distance allows for one to control both the pinhole size and the distance in front of the viewing position where the "sharp" plane is.

Rays per pixel are guaranteed to pass through the same focal point, which is why originally, everything in the scene is focused.  To blur, jitter is applied, so rays will actually hit some randomized point on the lens.  Overall, the average will cover a range of the object, resulting in this blurring effect for anything that isn't at the focal distance. 

Top image: lens radius of 0.1 and focal distance of 5.  Bottom image: lens radius of 0.5 and focal distance of 5.

<img width="1200" height="1200" alt="silksong 2026-10-08_18-17-12z 38samp" src="https://github.com/user-attachments/assets/9b3ff6c3-840f-4098-920b-a6879919efe3" />
<img width="1200" height="1200" alt="silksong 2026-10-08_18-18-50z 50samp" src="https://github.com/user-attachments/assets/b98e38e1-1f4f-4198-8f45-f721a7ff145f" />

### Procedural Textures

Perlin and Voronoi noise are available as procedural texturing materials.  The images below are: fbm perlin, fbm perlin input voronoi, and voronoi materials, respectively.

<img width="800" height="800" alt="cornell 2026-10-07_01-33-44z 190samp" src="https://github.com/user-attachments/assets/caa320cc-d633-4274-aac6-95cbb54645b6" />
<img width="800" height="800" alt="cornell 2026-10-07_02-03-28z 201samp" src="https://github.com/user-attachments/assets/f64239f7-6793-4c8f-b0de-ddbd8d8e1d43" />
<img width="800" height="800" alt="cornell 2026-10-08_18-28-45z 47samp" src="https://github.com/user-attachments/assets/f4d57da5-edc5-41ce-a696-3d0c48c550de" />

In Perlin noise, the entire space is subdivided into 3D unit cubes.  Then, a hashing function is used to randomly assign some direction to each corner of the cubes.  An input point has some distance along the gradient directions of the cube's 8 corners, and these values are blended using Ken Perlin's smooth fade function.

<img width="442" height="442" alt="image" src="https://github.com/user-attachments/assets/351dcfea-90e3-42f8-81c3-b076e5e876ca" />

<img width="441" height="400" alt="image" src="https://github.com/user-attachments/assets/6080cd61-fcc8-4c9f-8650-f5b9775895f5" />

(Images sourced from [here](https://rtouti.github.io/graphics/perlin-noise-algorithm))

Voronoi also divides space into unit cubes, but has a slighly different behavior.  Each cube has some randomly positioned point; each point is checked against the other 27 cubes to find the distance to the nearest other point, which creates a cell-effect of boundaries created between these "centroids".

<img width="500" height="500" alt="image" src="https://github.com/user-attachments/assets/fd553a1f-eac2-4cc0-be5b-bf9a3ec9a3d9" />

(Image sourced from [here](https://en.wikipedia.org/wiki/Weighted_Voronoi_diagram))

FBM stands for Fractal Brownian Motion and is a method of overlaying several octaves/layers of Perlin noise, depending on frequency and strength parameters for each succeeding layer.  As octaves increase, more fine detail can be achieved.

### BVH Tree Acceleration Structure

<img width="609" height="263" alt="image" src="https://github.com/user-attachments/assets/69e1875b-4ec2-4c70-81d8-5edaa9a37f68" />

(Image sourced from [here](https://developer.nvidia.com/blog/thinking-parallel-part-ii-tree-traversal-gpu/))

BVH stands for "Bounding Volume Hierarchy" and is an tree optimization structure.  Suppose we have a triangulated mesh; then we find a bounding box that encloses the triangles.  Suppose a ray misses this box; then nothing inside the box can be hit either, and we can safely disregard the general group.

A BVH consists of the root node (which is a box enclosing the entire mesh), the interior nodes (which each have two children enclosed by boxes), and leaves (in this implementation, 4 enclosed triangles).  One can see how the mesh is subdivided into halves.

First, the bounding boxes are computed for the current group.  Then, we pick the longest axis to equally halve the triangles by their position, and repeat.  This results in a balanced tree of approximately log_2(n) depth, where n is the total number of triangles.  During building, this tree is reordered such that triangles belonging to the same leaf are oriented next to each other in the buffer.

The GPU is responsible for traversal and tests a ray against the root bounding box, continually tracking the closest hit.  This prevents every single ray from being tested against every single triangle, which can lead to humongous cost for very large and complex meshes.  With BVH, cost is proportional to the tree depth, not the total number of triangles.
