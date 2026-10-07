#include "pathtrace.h"
#include "procedural.h"

#include <cstdio>
#include <cuda.h>
#include <cmath>
#include <thrust/execution_policy.h>
#include <thrust/random.h>
#include <thrust/remove.h>
#include <thrust/partition.h>
#include <thrust/sort.h>

#include "sceneStructs.h"
#include "scene.h"
#include "glm/glm.hpp"
#include "glm/gtx/norm.hpp"
#include "utilities.h"
#include "intersections.h"
#include "interactions.h"

#define ERRORCHECK 1
#define MATERIALSORT 1
#define STREAMCOMPACT 1
#define ANTIALIASING 1

#define FILENAME (strrchr(__FILE__, '/') ? strrchr(__FILE__, '/') + 1 : __FILE__)
#define checkCUDAError(msg) checkCUDAErrorFn(msg, FILENAME, __LINE__)
void checkCUDAErrorFn(const char* msg, const char* file, int line)
{
#if ERRORCHECK
    cudaDeviceSynchronize();
    cudaError_t err = cudaGetLastError();
    if (cudaSuccess == err)
    {
        return;
    }

    fprintf(stderr, "CUDA error");
    if (file)
    {
        fprintf(stderr, " (%s:%d)", file, line);
    }
    fprintf(stderr, ": %s: %s\n", msg, cudaGetErrorString(err));
#ifdef _WIN32
    getchar();
#endif // _WIN32
    exit(EXIT_FAILURE);
#endif // ERRORCHECK
}

__host__ __device__
thrust::default_random_engine makeSeededRandomEngine(int iter, int index, int depth)
{
    int h = utilhash((1 << 31) | (depth << 22) | iter) ^ utilhash(index);
    return thrust::default_random_engine(h);
}

//Kernel that writes the image to the OpenGL PBO directly.
__global__ void sendImageToPBO(uchar4* pbo, glm::ivec2 resolution, int iter, glm::vec3* image)
{
    int x = (blockIdx.x * blockDim.x) + threadIdx.x;
    int y = (blockIdx.y * blockDim.y) + threadIdx.y;

    if (x < resolution.x && y < resolution.y)
    {
        int index = x + (y * resolution.x);

        // ** Here I do a little tone mapping to help the hdri maps
        glm::vec3 pix = image[index] / (float)iter;
        pix = pix / (pix + glm::vec3(1.f));
        pix = glm::pow(pix, glm::vec3(1.f / 2.2f));

        glm::ivec3 color;
        color.x = glm::clamp((int)(pix.x * 255.0), 0, 255);
        color.y = glm::clamp((int)(pix.y * 255.0), 0, 255);
        color.z = glm::clamp((int)(pix.z * 255.0), 0, 255);

        // Each thread writes one pixel location in the texture (textel)
        pbo[index].w = 0;
        pbo[index].x = color.x;
        pbo[index].y = color.y;
        pbo[index].z = color.z;
    }
}

// idea: we will sort intersections by the material id and pass path segments.
struct CompareMaterial {
	__host__ __device__
        bool operator()(const ShadeableIntersection& a, const ShadeableIntersection& b) {
            return a.materialId < b.materialId;
	}
};

// this will help for stream compaction!
struct isPathActive {
    __host__ __device__
        bool operator()(const PathSegment& path) {
            return path.remainingBounces > 0;
        }
};

static Scene* hst_scene = NULL;
static GuiDataContainer* guiData = NULL;
static glm::vec3* dev_image = NULL;
static Geom* dev_geoms = NULL;
static Material* dev_materials = NULL;
static PathSegment* dev_paths = NULL;
static ShadeableIntersection* dev_intersections = NULL;
// static variables for device memory, any extra info you need, etc
// buffer for triangles
static Triangle* dev_triangles = NULL;
static Texture* dev_textures = NULL;
static glm::vec3* dev_texels = NULL;
// buffer for BVH nodes
static BVHNode* dev_bvhNodes = NULL;

void InitDataContainer(GuiDataContainer* imGuiData)
{
    guiData = imGuiData;
}

void pathtraceInit(Scene* scene)
{
    hst_scene = scene;

    const Camera& cam = hst_scene->state.camera;
    const int pixelcount = cam.resolution.x * cam.resolution.y;

    cudaMalloc(&dev_image, pixelcount * sizeof(glm::vec3));
    cudaMemset(dev_image, 0, pixelcount * sizeof(glm::vec3));

    cudaMalloc(&dev_paths, pixelcount * sizeof(PathSegment));

    cudaMalloc(&dev_geoms, scene->geoms.size() * sizeof(Geom));
    cudaMemcpy(dev_geoms, scene->geoms.data(), scene->geoms.size() * sizeof(Geom), cudaMemcpyHostToDevice);

    cudaMalloc(&dev_materials, scene->materials.size() * sizeof(Material));
    cudaMemcpy(dev_materials, scene->materials.data(), scene->materials.size() * sizeof(Material), cudaMemcpyHostToDevice);

    cudaMalloc(&dev_intersections, pixelcount * sizeof(ShadeableIntersection));
    cudaMemset(dev_intersections, 0, pixelcount * sizeof(ShadeableIntersection));

    // initialize any extra device memory you need
    if (!scene->triangles.empty()) {
		cudaMalloc(&dev_triangles, scene->triangles.size() * sizeof(Triangle));
        cudaMemcpy(dev_triangles,
            scene->triangles.data(),
            scene->triangles.size() * sizeof(Triangle),
            cudaMemcpyHostToDevice);
    }

    // texture buffers
    if (!scene->textures.empty()) {
        cudaMalloc(&dev_textures, scene->textures.size() * sizeof(Texture));
        cudaMemcpy(dev_textures,
            scene->textures.data(),
            scene->textures.size() * sizeof(Texture),
            cudaMemcpyHostToDevice);
    }

    if (!scene->texels.empty()) {
        cudaMalloc(&dev_texels, scene->texels.size() * sizeof(glm::vec3));
        cudaMemcpy(dev_texels,
            scene->texels.data(),
            scene->texels.size() * sizeof(glm::vec3),
            cudaMemcpyHostToDevice);
    }

    if (!scene->bvhNodes.empty()) {
        cudaMalloc(&dev_bvhNodes, scene->bvhNodes.size() * sizeof(BVHNode));
        cudaMemcpy(dev_bvhNodes,
            scene->bvhNodes.data(), 
            scene->bvhNodes.size() * sizeof(BVHNode), 
            cudaMemcpyHostToDevice);
    }

    checkCUDAError("pathtraceInit");
}

void pathtraceFree()
{
    cudaFree(dev_image);  // no-op if dev_image is null
    cudaFree(dev_paths);
    cudaFree(dev_geoms);
    cudaFree(dev_materials);
    cudaFree(dev_intersections);
    // clean up any extra device memory you created
    cudaFree(dev_triangles);
    dev_triangles = NULL;
    cudaFree(dev_textures);
    dev_textures = NULL;
    cudaFree(dev_texels);
    dev_texels = NULL;
    cudaFree(dev_bvhNodes);
    dev_bvhNodes = NULL;

    checkCUDAError("pathtraceFree");
}

void pathtraceReset() {
    const Camera& cam = hst_scene->state.camera;
	const int pixelcount = cam.resolution.x * cam.resolution.y;
	cudaMemset(dev_image, 0, pixelcount * sizeof(glm::vec3));
	checkCUDAError("pathtraceReset");
}

/**
* Generate PathSegments with rays from the camera through the screen into the
* scene, which is the first bounce of rays.
*
* Antialiasing - add rays for sub-pixel sampling
* motion blur - jitter rays "in time"
* lens effect - jitter ray origin positions based on a lens
*/
__global__ void generateRayFromCamera(Camera cam, int iter, int traceDepth, PathSegment* pathSegments)
{
    int x = (blockIdx.x * blockDim.x) + threadIdx.x;
    int y = (blockIdx.y * blockDim.y) + threadIdx.y;

    if (x < cam.resolution.x && y < cam.resolution.y) {
        int index = x + (y * cam.resolution.x);
        PathSegment& segment = pathSegments[index];

        segment.ray.origin = cam.position;
        segment.color = glm::vec3(1.0f, 1.0f, 1.0f);

        // implement antialiasing by jittering the ray
        float jx = 0.f;
        float jy = 0.f;
#if ANTIALIASING
		thrust::default_random_engine rng = makeSeededRandomEngine(iter, index, 0);
		thrust::uniform_real_distribution<float> u01(0, 1);
		jx = u01(rng) - 0.5f;
		jy = u01(rng) - 0.5f;
#endif
        segment.ray.direction = glm::normalize(cam.view
            - cam.right * cam.pixelLength.x * ((float)x + jx - (float)cam.resolution.x * 0.5f)
            - cam.up * cam.pixelLength.y * ((float)y + jy - (float)cam.resolution.y * 0.5f)
        );

        segment.pixelIndex = index;
        segment.remainingBounces = traceDepth;
    }
}

// computeIntersections handles generating ray intersections ONLY.
// Generating new rays is handled in your shader(s).
// Feel free to modify the code below.
__global__ void computeIntersections(
    int depth,
    int num_paths,
    PathSegment* pathSegments,
    Geom* geoms,
    int geoms_size,
    Triangle* triangles,
    BVHNode* bvhNodes,
    ShadeableIntersection* intersections)
{
    int path_index = blockIdx.x * blockDim.x + threadIdx.x;

    if (path_index < num_paths)
    {
        PathSegment pathSegment = pathSegments[path_index];

        float t;
        glm::vec3 intersect_point;
        glm::vec3 normal;
        glm::vec3 tangent;
        glm::vec2 uv;
        float t_min = FLT_MAX;
        int hit_geom_index = -1;
        bool outside = true;

        glm::vec3 tmp_intersect;
        glm::vec3 tmp_normal;
        glm::vec3 tmp_tangent;
        glm::vec2 tmp_uv;
        bool tmp_outside = true;

        // naive parse through global geoms

        for (int i = 0; i < geoms_size; i++)
        {
            Geom& geom = geoms[i];
            // reset params so unhandled geometry types won't reuse prev result
			t = -1.0f;
            tmp_tangent = glm::vec3(0.f);
            tmp_uv = glm::vec2(0.f);

            if (geom.type == CUBE)
            {
                t = boxIntersectionTest(geom, pathSegment.ray, tmp_intersect, tmp_normal, tmp_outside);
            }
            else if (geom.type == SPHERE)
            {
                t = sphereIntersectionTest(geom, pathSegment.ray, tmp_intersect, tmp_normal, tmp_outside);
            } else if (geom.type == MESH) {
				int triIdx;
				t = triangleIntersectionTest(geom, triangles, bvhNodes, pathSegment.ray, tmp_intersect, tmp_normal, tmp_tangent, tmp_uv, tmp_outside, triIdx);
			}

            // Compute the minimum t from the intersection tests to determine what
            // scene geometry object was hit first.
            if (t > 0.0f && t_min > t)
            {
                t_min = t;
                hit_geom_index = i;
                intersect_point = tmp_intersect;
                normal = tmp_normal;
                tangent = tmp_tangent;
                uv = tmp_uv;
                outside = tmp_outside;
            }
        }

        if (hit_geom_index == -1)
        {
            intersections[path_index].t = -1.0f;
            // on non hits, obviously material will be 0
			intersections[path_index].materialId = -1;
        }
        else
        {
            // The ray hits something
            intersections[path_index].t = t_min;
            intersections[path_index].materialId = geoms[hit_geom_index].materialid;
            intersections[path_index].surfaceNormal = normal;
			intersections[path_index].tangent = tangent;
			intersections[path_index].uv = uv;
            intersections[path_index].outside = outside;
        }
    }
}

// LOOK: "fake" shader demonstrating what you might do with the info in
// a ShadeableIntersection, as well as how to use thrust's random number
// generator. Observe that since the thrust random number generator basically
// adds "noise" to the iteration, the image should start off noisy and get
// cleaner as more iterations are computed.
//
// Note that this shader does NOT do a BSDF evaluation!
// Your shaders should handle that - this can allow techniques such as
// bump mapping.
__global__ void shadeFakeMaterial(
    int iter,
    int num_paths,
    ShadeableIntersection* shadeableIntersections,
    PathSegment* pathSegments,
    Material* materials)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_paths)
    {
        ShadeableIntersection intersection = shadeableIntersections[idx];
        if (intersection.t > 0.0f) // if the intersection exists...
        {
          // Set up the RNG
          // LOOK: this is how you use thrust's RNG! Please look at
          // makeSeededRandomEngine as well.
            thrust::default_random_engine rng = makeSeededRandomEngine(iter, idx, 0);
            thrust::uniform_real_distribution<float> u01(0, 1);

            Material material = materials[intersection.materialId];
            glm::vec3 materialColor = material.color;

            // If the material indicates that the object was a light, "light" the ray
            if (material.emittance > 0.0f) {
                pathSegments[idx].color *= (materialColor * material.emittance);
            }
            // Otherwise, do some pseudo-lighting computation. This is actually more
            // like what you would expect from shading in a rasterizer like OpenGL.
            // replace this! you should be able to start with basically a one-liner
            else {
                float lightTerm = glm::dot(intersection.surfaceNormal, glm::vec3(0.0f, 1.0f, 0.0f));
                pathSegments[idx].color *= (materialColor * lightTerm) * 0.3f + ((1.0f - intersection.t * 0.02f) * materialColor) * 0.7f;
                pathSegments[idx].color *= u01(rng); // apply some noise because why not
            }
            // If there was no intersection, color the ray black.
            // Lots of renderers use 4 channel color, RGBA, where A = alpha, often
            // used for opacity, in which case they can indicate "no opacity".
            // This can be useful for post-processing and image compositing.
        }
        else {
            pathSegments[idx].color = glm::vec3(0.0f);
        }
    }
}

// true material shading kernel
__global__ void shadeMaterial(
    int iter,
    int depth,
    int num_paths,
    ShadeableIntersection* shadeableIntersections,
    PathSegment* pathSegments,
    Material* materials, 
    Texture* textures,
    glm::vec3* texels,
    int envTexId,
    float envIntensity)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_paths)
    {
        // prevent light rays from bouncing again
        if (pathSegments[idx].remainingBounces <= 0) {
            return;
        }

        ShadeableIntersection intersection = shadeableIntersections[idx];
        if (intersection.t > 0.0f)
        {
            Material material = materials[intersection.materialId];
            glm::vec3 materialColor = material.color;

            // If the material indicates that the object was a light, "light" the ray
            if (material.emittance > 0.0f) {
                pathSegments[idx].color *= (materialColor * material.emittance);
				pathSegments[idx].remainingBounces = 0;
            }
            else {
                // need to use depth for seeding the rng, or else it will be the same random numbers every time
                // in the case of sorting by material, idx will differ for same path segment
                thrust::default_random_engine rng = makeSeededRandomEngine(iter, pathSegments[idx].pixelIndex, depth);
                // ray + t
				glm::vec3 isectP = pathSegments[idx].ray.origin + pathSegments[idx].ray.direction * intersection.t;
                
                // flat mat -> albedo texture IF texture exists
                // also handle bump mapping here!!
                if (material.noiseId == 1) { // fbm perlin
                    float fn = fbm(isectP * material.noiseScale, 5) * 0.5f + 0.5f;
                    material.color *= glm::clamp(fn, 0.f, 1.f);
                }
                else if (material.noiseId == 2) { // voronoi
                    float vn = voronoi(isectP * material.noiseScale);
                    material.color *= glm::clamp(vn, 0.f, 1.f);
                }
                else if (material.noiseId == 3) { // voronoi with fbm input
                    glm::vec3 q = isectP * material.noiseScale;
                    glm::vec3 warp = fbmVec(q * material.warpFreq, 5);
                    unsigned int cellId;
                    float d = voronoiBetter(q + material.warpStrength * warp, 1.f, cellId);
                    material.color *= glm::clamp(d, 0.f, 1.f);
                }
                else if (material.albedoTexId >= 0) { // otherwise look for file texture
                    material.color = sampleTexture(textures[material.albedoTexId], texels, intersection.uv);
                }

                glm::vec3 N = intersection.surfaceNormal;
                if (material.bumpTexId >= 0) {
                    // T is the tangent (directed toward u's increase)
                    glm::vec3 T = intersection.tangent - N * glm::dot(N, intersection.tangent);

                    if (glm::dot(T, T) > 1e-10f) {
                        T = glm::normalize(T);
                        // B is the bitangent (diretion of v's increase)
                        glm::vec3 B = glm::cross(N, T);

                        // calculate height map slope
                        const Texture& bump = textures[material.bumpTexId];
                        glm::vec2 du(1.f / bump.width, 0.f);
                        glm::vec2 dv(0.f, 1.f / bump.height);

                        // for finding slopes
                        float h = sampleTexture(bump, texels, intersection.uv).x;
                        float hu = sampleTexture(bump, texels, intersection.uv + du).x;
                        float hv = sampleTexture(bump, texels, intersection.uv + dv).x;

                        // the height map displaces along N, where T and B move along the slope of u and v respectively
                        // thus the normal is perpendicular to these steps
                        N = glm::normalize(N - material.bumpStrength * ((hu - h) * T + (hv - h) * B));
                    }
                }
                
                scatterRay(pathSegments[idx], isectP, N, intersection.outside, material, rng);
                
                // make sure paths that never hit emitters don't contribute light
                if (pathSegments[idx].remainingBounces == 0) {
                    pathSegments[idx].color = glm::vec3(0.f);
                }
            }
            // If there was no intersection, color the ray black.
            // Lots of renderers use 4 channel color, RGBA, where A = alpha, often
            // used for opacity, in which case they can indicate "no opacity".
            // This can be useful for post-processing and image compositing.
        }
        else {
            // Add environment illumination support
            if (envTexId >= 0) {
                glm::vec3 env = sampleEnvironment(
                    textures[envTexId],
                    texels,
                    pathSegments[idx].ray.direction
                );
                // clamp to remove most fireflies
                env = glm::min(env, glm::vec3(20.f));
                pathSegments[idx].color *= envIntensity * env;
            }
            else {
                // no environment, just do black
                pathSegments[idx].color = glm::vec3(0.0f);
            }
			pathSegments[idx].remainingBounces = 0;
        }
    }
}

// Add the current iteration's output to the overall image
__global__ void finalGather(int nPaths, glm::vec3* image, PathSegment* iterationPaths)
{
    int index = (blockIdx.x * blockDim.x) + threadIdx.x;

    if (index < nPaths)
    {
        PathSegment iterationPath = iterationPaths[index];
        image[iterationPath.pixelIndex] += iterationPath.color;
    }
}

/**
 * Wrapper for the __global__ call that sets up the kernel calls and does a ton
 * of memory management
 */
void pathtrace(uchar4* pbo, int frame, int iter)
{
    const int traceDepth = hst_scene->state.traceDepth;
    const Camera& cam = hst_scene->state.camera;
    const int pixelcount = cam.resolution.x * cam.resolution.y;

    // 2D block for generating ray from camera
    const dim3 blockSize2d(8, 8);
    const dim3 blocksPerGrid2d(
        (cam.resolution.x + blockSize2d.x - 1) / blockSize2d.x,
        (cam.resolution.y + blockSize2d.y - 1) / blockSize2d.y);

    // 1D block for path tracing
    const int blockSize1d = 128;

    ///////////////////////////////////////////////////////////////////////////

    // Recap:
    // * Initialize array of path rays (using rays that come out of the camera)
    //   * You can pass the Camera object to that kernel.
    //   * Each path ray must carry at minimum a (ray, color) pair,
    //   * where color starts as the multiplicative identity, white = (1, 1, 1).
    //   * This has already been done for you.
    // * For each depth:
    //   * Compute an intersection in the scene for each path ray.
    //     A very naive version of this has been implemented for you, but feel
    //     free to add more primitives and/or a better algorithm.
    //     Currently, intersection distance is recorded as a parametric distance,
    //     t, or a "distance along the ray." t = -1.0 indicates no intersection.
    //     * Color is attenuated (multiplied) by reflections off of any object
    //   * Stream compact away all of the terminated paths.
    //     You may use either your implementation or `thrust::remove_if` or its
    //     cousins.
    //     * Note that you can't really use a 2D kernel launch any more - switch
    //       to 1D.
    //   * Shade the rays that intersected something or didn't bottom out.
    //     That is, color the ray by performing a color computation according
    //     to the shader, then generate a new ray to continue the ray path.
    //     We recommend just updating the ray's PathSegment in place.
    //     Note that this step may come before or after stream compaction,
    //     since some shaders you write may also cause a path to terminate.
    //          This step is done by shadeMaterial calling scatterRay.  This multiplies
    //          by the color of the material and generates a new ray direction.
    // * Finally, add this iteration's results to the image. This has been done
    //   for you.

    // perform one iteration of path tracing

    generateRayFromCamera<<<blocksPerGrid2d, blockSize2d>>>(cam, iter, traceDepth, dev_paths);
    checkCUDAError("generate camera ray");

    int depth = 0;
    PathSegment* dev_path_end = dev_paths + pixelcount;
    int num_paths = dev_path_end - dev_paths;

    // --- PathSegment Tracing Stage ---
    // Shoot ray into scene, bounce between objects, push shading chunks

    bool iterationComplete = false;
    while (!iterationComplete)
    {
        // clean shading chunks
        cudaMemset(dev_intersections, 0, pixelcount * sizeof(ShadeableIntersection));

        // tracing
        dim3 numblocksPathSegmentTracing = (num_paths + blockSize1d - 1) / blockSize1d;
        computeIntersections<<<numblocksPathSegmentTracing, blockSize1d>>> (
            depth,
            num_paths,
            dev_paths,
            dev_geoms,
            hst_scene->geoms.size(),
            dev_triangles,
            dev_bvhNodes,
            dev_intersections
        );
        checkCUDAError("trace one bounce");
        cudaDeviceSynchronize();
        depth++;

        // --- Shading Stage ---
        // Shade path segments based on intersections and generate new rays by
        // evaluating the BSDF.
        // Start off with just a big kernel that handles all the different
        // materials you have in the scenefile.
        // Compare between directly shading the path segments and shading
        // path segments that have been reshuffled to be contiguous in memory.

        // Notably, rendering with the material sort is a lot slower than directly shading the path segments.
        // The sort itself can easily create significant overhead by conducting comparison sorts on every struct key.
        // For cases where there are very few materials (like only diffuse and emissive), there isn't a lot of divergence.

        // sort by material: keyed by intersection, values are the actual path segments.
        #if MATERIALSORT
                thrust::sort_by_key(thrust::device,
                    dev_intersections,
                    dev_intersections + num_paths,
                    dev_paths,
                    CompareMaterial());
        #endif

        shadeMaterial<<<numblocksPathSegmentTracing, blockSize1d>>>(
            iter,
            depth,
            num_paths,
            dev_intersections,
            dev_paths,
            dev_materials, 
            dev_textures,
            dev_texels,
            hst_scene->envTexId,
            hst_scene->envIntensity
        );
		checkCUDAError("shadeMaterial");

#if STREAMCOMPACT
        dev_path_end = thrust::partition(thrust::device,
            dev_paths,
            dev_path_end,
            isPathActive());
        num_paths = dev_path_end - dev_paths;
#endif
        iterationComplete = (num_paths == 0) || (depth >= traceDepth); // should be based off stream compaction results.

        if (guiData != NULL)
        {
            guiData->TracedDepth = depth;
        }
    }

    // Assemble this iteration and apply it to the image
    dim3 numBlocksPixels = (pixelcount + blockSize1d - 1) / blockSize1d;
    // Instead of num_paths, should gather the terminated paths (that have final colors)
    finalGather<<<numBlocksPixels, blockSize1d>>>(pixelcount, dev_image, dev_paths);

    ///////////////////////////////////////////////////////////////////////////

    // Send results to OpenGL buffer for rendering
    sendImageToPBO<<<blocksPerGrid2d, blockSize2d>>>(pbo, cam.resolution, iter, dev_image);

    // Retrieve image from GPU
    cudaMemcpy(hst_scene->state.image.data(), dev_image,
        pixelcount * sizeof(glm::vec3), cudaMemcpyDeviceToHost);

    checkCUDAError("pathtrace");
}
