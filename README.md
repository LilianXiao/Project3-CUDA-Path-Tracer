CUDA Path Tracer
================

**University of Pennsylvania, CIS 565: GPU Programming and Architecture, Project 3**

<img width="1920" height="1200" alt="greenpath" src="https://github.com/user-attachments/assets/7ba13637-5a06-495f-a444-2a4e0188da04" />

* Lilian Xiao
  * [LinkedIn](https://www.linkedin.com/in/lilian-xiao/), [personal website](https://lilianxiao.carrd.co/), [art website](https://lilianxvis.carrd.co/)
* Tested on: Windows 11, Intel(R) Core(TM) Ultra 9 185H (2.50 GHz), 16.0 GB RAM, NVIDIA GeForce RTX 4070 Laptop GPU (8 GB)
Intel(R) Arc(TM) Graphics (128 MB) (Personal Laptop)

### Overview

Path tracing provides a more physically accurate simulation of light by considering its reflection, refraction, transmission, absorption, and scattering properties.  In basic ray tracing, a ray is emitted from the camera/eye for each pixel of the image.  The intersections of these rays with objects in the scene, depending on the surface behaviors of the objects, can result in secondary rays.  For instance, when the view ray hits a specular object, a secondary ray is emitted.  Ray tracing supports direct illumination and approximates the behavior of light in terms of reflection, refraction, and shadowing.

On the other hand, path tracing can simulate global illumination; in nature, light can come from anywhere, and this results in both direct and indirect lighting.  While ray tracing is a deterministic process, Monte Carlo path tracing is stochastic; we trace multiple rays per pixel because light in nature is random.  Path tracing for more iterations allows the render to converge to a less noisy and more usable image, since each iteration is responsible for tracing a random light path, and the average over these iterations grows closer to the Monte Carlo estimate.

The following two renders of the same Cornell box scene are taken at 7 samples and 204 samples respectively.

<img width="800" height="800" alt="cornell 2026-10-08_17-13-56z 7samp" src="https://github.com/user-attachments/assets/b88680e4-6442-4d08-a844-df262f448bfa" />
<img width="800" height="800" alt="cornell 2026-10-08_17-13-56z 204samp" src="https://github.com/user-attachments/assets/82e55449-a7d9-4f10-b70f-37a104569bc1" />


