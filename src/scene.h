#pragma once

#include "sceneStructs.h"
#include <vector>

class Scene
{
private:
    void loadFromJSON(const std::string& jsonName);
	void loadMeshFromOBJ(const std::string& objName, Geom& geom);
	int loadTexture(const std::string& texName);
public:
    Scene(std::string filename);

    std::vector<Geom> geoms;
    std::vector<Material> materials;
    std::vector<Triangle> triangles;
	std::vector<Texture> textures;
	std::vector<glm::vec3> texels;
    RenderState state;
};
