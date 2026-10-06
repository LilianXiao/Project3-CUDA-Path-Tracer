#include "scene.h"

#include "utilities.h"

#define TINYOBJLOADER_IMPLEMENTATION
#include "tiny_obj_loader.h"

#include <glm/gtc/matrix_inverse.hpp>
#include <glm/gtx/string_cast.hpp>
#include "json.hpp"

#include <fstream>
#include <iostream>
#include <string>
#include <unordered_map>
#include <cfloat>

using namespace std;
using json = nlohmann::json;

Scene::Scene(string filename)
{
    cout << "Reading scene from " << filename << " ..." << endl;
    cout << " " << endl;
    auto ext = filename.substr(filename.find_last_of('.'));
    if (ext == ".json")
    {
        loadFromJSON(filename);
        return;
    }
    else
    {
        cout << "Couldn't read from " << filename << endl;
        exit(-1);
    }
}

void Scene::loadFromJSON(const std::string& jsonName)
{
    std::ifstream f(jsonName);
    json data = json::parse(f);
    const auto& materialsData = data["Materials"];
    std::unordered_map<std::string, uint32_t> MatNameToID;
    for (const auto& item : materialsData.items())
    {
        const auto& name = item.key();
        const auto& p = item.value();
        Material newMaterial{};
        // TODO: handle materials loading differently
        if (p["TYPE"] == "Diffuse")
        {
            const auto& col = p["RGB"];
            newMaterial.color = glm::vec3(col[0], col[1], col[2]);
        }
        else if (p["TYPE"] == "Emitting")
        {
            const auto& col = p["RGB"];
            newMaterial.color = glm::vec3(col[0], col[1], col[2]);
            newMaterial.emittance = p["EMITTANCE"];
        }
        else if (p["TYPE"] == "Specular")
        {
            const auto& col = p["RGB"];
            newMaterial.color = glm::vec3(col[0], col[1], col[2]);
        }
        MatNameToID[name] = materials.size();
        materials.emplace_back(newMaterial);
    }
    
    static const std::unordered_map<std::string, GeomType> geomType = {
        {"sphere", SPHERE},
        {"cube", CUBE},
        {"mesh", MESH}
    };
    // we will retrieve from file path
	const std::string basePath = jsonName.substr(0, jsonName.find_last_of("/\\") + 1);

    const auto& objectsData = data["Objects"];
    for (const auto& p : objectsData)
    {
        const auto& type = p["TYPE"];
		auto it = geomType.find(type);
        if (it == geomType.end()) {
			cout << "Unknown object type!" << type << endl;
            exit(-1);
        }

        Geom newGeom;
		newGeom.type = it->second;

        if (type == "cube")
        {
            newGeom.type = CUBE;
        }
        else if (type == "sphere")
        {
            newGeom.type = SPHERE;
        }

        newGeom.materialid = MatNameToID[p["MATERIAL"]];
        const auto& trans = p["TRANS"];
        const auto& rotat = p["ROTAT"];
        const auto& scale = p["SCALE"];
        newGeom.translation = glm::vec3(trans[0], trans[1], trans[2]);
        newGeom.rotation = glm::vec3(rotat[0], rotat[1], rotat[2]);
        newGeom.scale = glm::vec3(scale[0], scale[1], scale[2]);
        newGeom.transform = utilityCore::buildTransformationMatrix(
            newGeom.translation, newGeom.rotation, newGeom.scale);
        newGeom.inverseTransform = glm::inverse(newGeom.transform);
        newGeom.invTranspose = glm::inverseTranspose(newGeom.transform);

        // load mesh only after matrices exist so we can transform everything
        if (type == "mesh")
        {
            newGeom.type = MESH;
            const std::string file = p["FILE"];
            loadMeshFromOBJ(basePath + file, newGeom);
        }

        geoms.push_back(newGeom);
    }
    const auto& cameraData = data["Camera"];
    Camera& camera = state.camera;
    RenderState& state = this->state;
    camera.resolution.x = cameraData["RES"][0];
    camera.resolution.y = cameraData["RES"][1];
    float fovy = cameraData["FOVY"];
    state.iterations = cameraData["ITERATIONS"];
    state.traceDepth = cameraData["DEPTH"];
    state.imageName = cameraData["FILE"];
    const auto& pos = cameraData["EYE"];
    const auto& lookat = cameraData["LOOKAT"];
    const auto& up = cameraData["UP"];
    camera.position = glm::vec3(pos[0], pos[1], pos[2]);
    camera.lookAt = glm::vec3(lookat[0], lookat[1], lookat[2]);
    camera.up = glm::vec3(up[0], up[1], up[2]);

    //calculate fov based on resolution
    float yscaled = tan(fovy * (PI / 180));
    float xscaled = (yscaled * camera.resolution.x) / camera.resolution.y;
    float fovx = (atan(xscaled) * 180) / PI;
    camera.fov = glm::vec2(fovx, fovy);

    camera.right = glm::normalize(glm::cross(camera.view, camera.up));
    camera.pixelLength = glm::vec2(2 * xscaled / (float)camera.resolution.x,
        2 * yscaled / (float)camera.resolution.y);

    camera.view = glm::normalize(camera.lookAt - camera.position);

    //set up render camera stuff
    int arraylen = camera.resolution.x * camera.resolution.y;
    state.image.resize(arraylen);
    std::fill(state.image.begin(), state.image.end(), glm::vec3());
}

// OBJ triangulated mesh loading.  I am adapting from tinyOBJLoader:
// https://github.com/tinyobjloader/tinyobjloader
// This is cpu side loading, and later the GPU will read them via ray-triangle intersection testing.
void Scene::loadMeshFromOBJ(const std::string& objName, Geom& geom) {
    tinyobj::ObjReaderConfig config;
	config.triangulate = true;
	tinyobj::ObjReader reader;
    if (!reader.ParseFromFile(objName, config)) {
		std::cerr << "TinyObjReader loading failed: " << reader.Error() << std::endl;
		exit(-1);
    }
    if (!reader.Warning().empty()) {
		std::cout << "TinyObjReader warning: " << reader.Warning() << std::endl;
    }

	const tinyobj::attrib_t& attrib = reader.GetAttrib();
	const std::vector<tinyobj::shape_t>& shapes = reader.GetShapes();

	const std::vector<tinyobj::real_t>& positions = attrib.vertices;
	const std::vector<tinyobj::real_t>& normals = attrib.normals;
	const std::vector<tinyobj::real_t>& uvs = attrib.texcoords;

    auto position = [&](const tinyobj::index_t& i) {
        glm::vec3 p(
            positions[3 * i.vertex_index + 0],
            positions[3 * i.vertex_index + 1],
            positions[3 * i.vertex_index + 2]
        );
        return glm::vec3(geom.transform * glm::vec4(p, 1.f));
    };

    auto normal = [&](const tinyobj::index_t& i) {
        glm::vec3 n(
            normals[3 * i.normal_index + 0],
            normals[3 * i.normal_index + 1],
            normals[3 * i.normal_index + 2]
        );
		return glm::normalize(glm::vec3(geom.invTranspose * glm::vec4(n, 0.f)));
	};

    // note that uv i is at 2i, 2i + 1 in the array
    auto uv = [&](const tinyobj::index_t& i) {
        glm::vec2 uv(
            uvs[2 * i.texcoord_index + 0],
            uvs[2 * i.texcoord_index + 1]
        );
        return uv;
	};

	geom.triStart = (int)triangles.size();
    geom.bboxMin = glm::vec3(FLT_MAX);
	geom.bboxMax = glm::vec3(-FLT_MAX);

    for (size_t i = 0; i < shapes.size(); ++i) {
		const std::vector<tinyobj::index_t>& indices = shapes[i].mesh.indices;

        for (size_t j = 0; j + 2 < indices.size(); j += 3) {
			const tinyobj::index_t& i1 = indices[j];
			const tinyobj::index_t& i2 = indices[j + 1];
			const tinyobj::index_t& i3 = indices[j + 2];

            Triangle tri{};
			tri.v0 = position(i1);
			tri.v1 = position(i2);
			tri.v2 = position(i3);

            if (i1.normal_index >= 0 && i2.normal_index >= 0 && i3.normal_index >= 0) {
                tri.n0 = normal(i1);
                tri.n1 = normal(i2);
                tri.n2 = normal(i3);
            }
            else {
				tri.n0 = tri.n1 = tri.n2 = 
                    glm::normalize(glm::cross(tri.v1 - tri.v0, tri.v2 - tri.v0));
            }

			geom.bboxMin = glm::min(geom.bboxMin, glm::min(tri.v0, glm::min(tri.v1, tri.v2)));
			geom.bboxMax = glm::max(geom.bboxMax, glm::max(tri.v0, glm::max(tri.v1, tri.v2)));
			triangles.push_back(tri);
        }
    }

	geom.numTris = (int)triangles.size() - geom.triStart;
}
