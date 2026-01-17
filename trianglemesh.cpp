// ========================================================================= //
// Authors: Daniel Rutz, Daniel Ströter, Roman Getto, Matthias Bein          //
//                                                                           //
// GRIS - Graphisch Interaktive Systeme                                      //
// Technische Universität Darmstadt                                          //
// Fraunhoferstrasse 5                                                       //
// D-64283 Darmstadt, Germany                                                //
//                                                                           //
// Content: Simple class for reading and rendering triangle meshes, SOLUTION //
//   * readOBJ                                                               //
//   * draw                                                                  //
//   * transformations                                                       //
// ========================================================================= //

#include <cmath>
#include <array>
#include <cfloat>
#include <algorithm>
#include <random>
#include <array>

#include <fstream>
#include <iostream>
#include <sstream>
#include <iomanip>

#include <QtMath>
#include <QOpenGLFunctions_3_3_Core>

#include "trianglemesh.h"
#include "renderstate.h"
#include "utilities.h"
#include "clipplane.h"
#include "shader.h"

using glVertexAttrib3fvPtr = void (*)(GLuint index, const GLfloat* v);
using glVertexAttrib3fPtr = void (*)(GLuint index, GLfloat v1, GLfloat v2, GLfloat v3);

TriangleMesh::TriangleMesh(QOpenGLFunctions_3_3_Core* f)
    : staticColor(1.f, 1.f, 1.f), f(f)
{
    clear();
}

TriangleMesh::~TriangleMesh() {
    // clear data
    clear();
}

void TriangleMesh::clear() {
    // clear mesh data
    vertices.clear();
    triangles.clear();
    normals.clear();
    colors.clear();
    texCoords.clear();
    // clear bounding box data
    boundingBoxMin = Vec3f(FLT_MAX, FLT_MAX, FLT_MAX);
    boundingBoxMax = Vec3f(-FLT_MAX, -FLT_MAX, -FLT_MAX);
    boundingBoxMid.zero();
    boundingBoxSize.zero();
    // draw mode data
    coloringType = ColoringType::STATIC_COLOR;
    withBB = false;
    withNormals = false;
    textureID.val = 0;
    cleanupVBO();
}

void TriangleMesh::coutData() {
    std::cout << std::endl;
    std::cout << "=== MESH DATA ===" << std::endl;
    std::cout << "nr. triangles: " << triangles.size() << std::endl;
    std::cout << "nr. vertices:  " << vertices.size() << std::endl;
    std::cout << "nr. normals:   " << normals.size() << std::endl;
    std::cout << "nr. colors:    " << colors.size() << std::endl;
    std::cout << "nr. texCoords: " << texCoords.size() << std::endl;
    std::cout << "BB: (" << boundingBoxMin << ") - (" << boundingBoxMax << ")" << std::endl;
    std::cout << "  BBMid: (" << boundingBoxMid << ")" << std::endl;
    std::cout << "  BBSize: (" << boundingBoxSize << ")" << std::endl;
    std::cout << "  VAO ID: " << VAO() << ", VBO IDs: f=" << VBOf() << ", v=" << VBOv() << ", n=" << VBOn() << ", c=" << VBOc() << ", t=" << VBOt() << std::endl;
    std::cout << "coloring using: ";
    switch (coloringType) {
        case ColoringType::STATIC_COLOR:
            std::cout << "a static color" << std::endl;
            break;
        case ColoringType::COLOR_ARRAY:
            std::cout << "a color array" << std::endl;
            break;
        case ColoringType::TEXTURE:
            std::cout << "a texture" << std::endl;
            break;

        case ColoringType::BUMP_MAPPING:
            std::cout << "a bump map" << std::endl;
            break;
    }
}

// ================
// === RAW DATA ===
// ================

void TriangleMesh::flipNormals(bool createVBOs) {
    for (auto& n : normals) n *= -1.0f;
    //correct VBO
    if (createVBOs && VBOn() != 0) {
        if (!f) return;
        f->glBindBuffer(GL_ARRAY_BUFFER, VBOn());
        f->glBufferSubData(GL_ARRAY_BUFFER, 0, normals.size() * sizeof(Normal), normals.data());
        f->glBindBuffer(GL_ARRAY_BUFFER, 0);
    }
}

void TriangleMesh::translateToCenter(const Vec3f& newBBmid, bool createVBOs) {
    Vec3f trans = newBBmid - boundingBoxMid;
    for (auto& vertex : vertices) vertex += trans;
    boundingBoxMin += trans;
    boundingBoxMax += trans;
    boundingBoxMid += trans;
    // data changed => delete VBOs and create new ones (not efficient but easy)
    if (createVBOs) {
        cleanupVBO();
        createAllVBOs();
    }
}

void TriangleMesh::scaleToLength(const float newLength, bool createVBOs) {
    float length = std::max(std::max(boundingBoxSize.x(), boundingBoxSize.y()), boundingBoxSize.z());
    float scale = newLength / length;
    for (auto& vertex : vertices) vertex *= scale;
    boundingBoxMin *= scale;
    boundingBoxMax *= scale;
    boundingBoxMid *= scale;
    boundingBoxSize *= scale;
    // data changed => delete VBOs and create new ones (not efficient but easy)
    if (createVBOs) {
        cleanupVBO();
        createAllVBOs();
    }
}

// =================
// === LOAD MESH ===
// =================

void TriangleMesh::loadOBJ(const char* filename, bool createVBOs) {
    // clear any existing mesh
    clear();
    // load from obj
    std::ifstream in(filename);
    if (!in.is_open()) {
        std::cout << "loadOBJ: can not find " << filename << std::endl;
        return;
    }

    vertices.resize(0);
    triangles.resize(0);
    normals.resize(0);

    // Load all vertices and triangles and ignore other entries.
    while (!in.eof()) 
    {
        std::string s;        
        in >> s;

        if (s == "v") {
            // read and store a vertex
            GLfloat x, y, z;
            in >> x >> y >> z;
            vertices.push_back({ x, y, z });

			// update bounding box
			boundingBoxMin[0] = std::min(x, boundingBoxMin[0]);
			boundingBoxMin[1] = std::min(y, boundingBoxMin[1]);
			boundingBoxMin[2] = std::min(z, boundingBoxMin[2]);
			boundingBoxMax[0] = std::max(x, boundingBoxMax[0]);
			boundingBoxMax[1] = std::max(y, boundingBoxMax[1]);
			boundingBoxMax[2] = std::max(z, boundingBoxMax[2]);
			
			// Ignore rest of line
            in.ignore(std::numeric_limits<std::streamsize>::max(), '\n'); 
        } else if (s == "vn") {
            // read and store a vertex normal
            GLfloat x, y, z;
            in >> x >> y >> z;
            normals.push_back({ x, y, z });

			// Ignore rest of line
            in.ignore(std::numeric_limits<std::streamsize>::max(), '\n'); 
        } else if (s == "f") 
        {
            // read and store a face
            std::string line;
            getline(in, line);
            std::stringstream ss(line);
            std::vector<int> indices;
            int num;

            while (ss >> num) {
                // convert negative indices to positive indices
                if (num < 0) num = vertices.size() + 1 + num;

                // Store and convert OBJ's 1-based indices to the required 0-based indices!
                indices.push_back(num - 1);
            }
            if (indices.size() == 3)
                triangles.push_back({ (unsigned int)indices[0], (unsigned int)indices[1], (unsigned int)indices[2] });
            else
                qWarning("The OBJ file contains polygons that are not triangles! Ignoring "
                        "entry, this will lead to holes in your mesh!");
        } else 
        {
            // Skip comments and other entries
            in.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
        }

    }

	// update bounding box
	boundingBoxMid = 0.5f*boundingBoxMin + 0.5f*boundingBoxMax;
	boundingBoxSize = boundingBoxMax - boundingBoxMin;

    // close ifstream
    in.close();

    // calculate normals if they are not present in the file
    if(normals.size() != vertices.size())
        calculateNormalsByArea();

    // calculate texture coordinates
    calculateTexCoordsSphereMapping();

    // createVBO
    if (createVBOs) {
        createAllVBOs();
    }
}

void TriangleMesh::loadOBJ(const char* filename, const Vec3f& BBmid, const float BBlength) {
    loadOBJ(filename, false);
    translateToCenter(BBmid, false);
    scaleToLength(BBlength, true);
}

void TriangleMesh::calculateNormalsByArea() {
    // sum up triangle normals in each vertex
    normals.resize(vertices.size());
    for (auto& triangle : triangles) {
        unsigned int
            id0 = triangle[0],
            id1 = triangle[1],
            id2 = triangle[2];
        Vec3f
            vec1 = vertices[id1] - vertices[id0],
            vec2 = vertices[id2] - vertices[id0],
            normal = cross(vec1, vec2);
        normals[id0] += normal;
        normals[id1] += normal;
        normals[id2] += normal;
    }
    // normalize normals
    for (auto& normal : normals) normal.normalize();
}

void TriangleMesh::calculateTexCoordsSphereMapping() {
    texCoords.clear();
    // texCoords by central projection on unit sphere
    // optional ...
    for (const auto& vertex : vertices) {
        const auto dist = vertex - boundingBoxMid;
        float u = (M_1_PI / 2) * std::atan2(dist.x(), dist.z()) + 0.5;
        float v = M_1_PI * std::asin(dist.y() / std::sqrt(dist.x() * dist.x() + dist.y() * dist.y() + dist.z() * dist.z()));
        texCoords.push_back(TexCoord{ u, v });
    }

}

void TriangleMesh::calculateBB() {
    // clear bounding box data
    boundingBoxMin = Vec3f(FLT_MAX, FLT_MAX, FLT_MAX);
    boundingBoxMax = Vec3f(-FLT_MAX, -FLT_MAX, -FLT_MAX);
    boundingBoxMid.zero();
    boundingBoxSize.zero();
    // iterate over vertices
    for (auto& vertex : vertices) {
        boundingBoxMin[0] = std::min(vertex[0], boundingBoxMin[0]);
        boundingBoxMin[1] = std::min(vertex[1], boundingBoxMin[1]);
        boundingBoxMin[2] = std::min(vertex[2], boundingBoxMin[2]);
        boundingBoxMax[0] = std::max(vertex[0], boundingBoxMax[0]);
        boundingBoxMax[1] = std::max(vertex[1], boundingBoxMax[1]);
        boundingBoxMax[2] = std::max(vertex[2], boundingBoxMax[2]);
    }
    boundingBoxMid = 0.5f*boundingBoxMin + 0.5f*boundingBoxMax;
    boundingBoxSize = boundingBoxMax - boundingBoxMin;
}

GLuint TriangleMesh::createVBO(QOpenGLFunctions_3_3_Core* f, const void* data, int dataSize, GLenum target, GLenum usage) {

    // 0 is reserved, glGenBuffers() will return non-zero id if success
    GLuint id = 0;
    // create a vbo
    f->glGenBuffers(1, &id);
    // activate vbo id to use
    f->glBindBuffer(target, id);
    // upload data to video card
    f->glBufferData(target, dataSize, data, usage);
    // check data size in VBO is same as input array, if not return 0 and delete VBO
    int bufferSize = 0;
    f->glGetBufferParameteriv(target, GL_BUFFER_SIZE, &bufferSize);
    if(dataSize != bufferSize) {
        f->glDeleteBuffers(1, &id);
        id = 0;
        std::cout << "createVBO() ERROR: Data size (" << dataSize << ") is mismatch with input array (" << bufferSize << ")." << std::endl;
    }
    // unbind after copying data
    f->glBindBuffer(target, 0);
    return id;
}

void TriangleMesh::createBBVAO(QOpenGLFunctions_3_3_Core* f) {
    f->glGenVertexArrays(1, &VAObb.val);

    // create VBOs of bounding box
    VBOvbb.val = createVBO(f, BoxVertices, BoxVerticesSize, GL_ARRAY_BUFFER, GL_STATIC_DRAW);
    VBOfbb.val = createVBO(f, BoxLineIndices, BoxLineIndicesSize, GL_ELEMENT_ARRAY_BUFFER, GL_STATIC_DRAW);

    // bind VAO of bounding box
    f->glBindVertexArray(VAObb.val);
    f->glBindBuffer(GL_ARRAY_BUFFER, VBOvbb.val);
    f->glVertexAttribPointer(POSITION_LOCATION, 3, GL_FLOAT, GL_FALSE, 0, nullptr);
    f->glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, VBOfbb.val);

    f->glEnableVertexAttribArray(POSITION_LOCATION);
    f->glBindVertexArray(0);
    f->glBindBuffer(GL_ARRAY_BUFFER, 0);
    f->glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
}

void TriangleMesh::createNormalVAO(QOpenGLFunctions_3_3_Core* f) {
    if (vertices.size() != normals.size()) return;
    std::vector<Vec3f> normalArrowVertices;
    normalArrowVertices.reserve(2 * vertices.size());
    for (size_t i = 0; i < vertices.size(); ++i) {
        normalArrowVertices.push_back(vertices[i]);
        normalArrowVertices.push_back(vertices[i] + 0.1 * normals[i]);
    }

    f->glGenVertexArrays(1, &VAOn.val);
    VBOvn.val = createVBO(f, normalArrowVertices.data(), normalArrowVertices.size() * sizeof(Vertex), GL_ARRAY_BUFFER, GL_STATIC_DRAW);
    f->glBindVertexArray(VAOn.val);
    f->glBindBuffer(GL_ARRAY_BUFFER, VBOvn.val);
    f->glEnableVertexAttribArray(POSITION_LOCATION);
    f->glVertexAttribPointer(POSITION_LOCATION, 3, GL_FLOAT, GL_FALSE, 0, nullptr);
    f->glBindVertexArray(0);
    f->glBindBuffer(GL_ARRAY_BUFFER, 0);
}

void TriangleMesh::createAllVBOs() {
    if (!f) return;
    // create VAOs
    f->glGenVertexArrays(1, &VAO.val);

    // create VBOs
    VBOf.val = createVBO(f, triangles.data(), triangles.size() * sizeof(Triangle), GL_ELEMENT_ARRAY_BUFFER, GL_STATIC_DRAW);
    VBOv.val = createVBO(f, vertices.data(), vertices.size() * sizeof(Vertex), GL_ARRAY_BUFFER, GL_STATIC_DRAW);
    VBOn.val = createVBO(f, normals.data(), normals.size() * sizeof(Normal), GL_ARRAY_BUFFER, GL_STATIC_DRAW);
    if (colors.size() == vertices.size()) {
        VBOc.val = createVBO(f, colors.data(), colors.size() * sizeof(Color), GL_ARRAY_BUFFER, GL_STATIC_DRAW);
        f->glEnableVertexAttribArray(COLOR_LOCATION);
    }
    if (texCoords.size() == vertices.size()) {
        VBOt.val = createVBO(f, texCoords.data(), texCoords.size() * sizeof(TexCoord), GL_ARRAY_BUFFER, GL_STATIC_DRAW);
        f->glEnableVertexAttribArray(TEXCOORD_LOCATION);
    }
    if (tangents.size() == vertices.size()) {
        VBOtan.val = createVBO(f, tangents.data(), tangents.size() * sizeof(Tangent), GL_ARRAY_BUFFER, GL_STATIC_DRAW);
        f->glEnableVertexAttribArray(TANGENT_LOCATION);   
    }

    // bind VBOs to VAO object
    f->glBindVertexArray(VAO.val);
    f->glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, VBOf.val);
    f->glBindBuffer(GL_ARRAY_BUFFER, VBOv.val);
    f->glVertexAttribPointer(POSITION_LOCATION, 3, GL_FLOAT, GL_FALSE, 0, nullptr);
    f->glEnableVertexAttribArray(POSITION_LOCATION);
    f->glBindBuffer(GL_ARRAY_BUFFER, VBOn.val);
    f->glVertexAttribPointer(NORMAL_LOCATION, 3, GL_FLOAT, GL_FALSE, 0, nullptr);
    f->glEnableVertexAttribArray(NORMAL_LOCATION);
    if (VBOc.val) {
        f->glBindBuffer(GL_ARRAY_BUFFER, VBOc.val);
        f->glVertexAttribPointer(COLOR_LOCATION, 3, GL_FLOAT, GL_FALSE, 0, nullptr);
        f->glEnableVertexAttribArray(COLOR_LOCATION);
    }

    if (VBOt.val) {
        f->glBindBuffer(GL_ARRAY_BUFFER, VBOt.val);
        f->glVertexAttribPointer(TEXCOORD_LOCATION, 2, GL_FLOAT, GL_FALSE, 0, nullptr);
        f->glEnableVertexAttribArray(TEXCOORD_LOCATION);
    }
    if (VBOtan.val) {
        f->glBindBuffer(GL_ARRAY_BUFFER, VBOtan.val);
        f->glVertexAttribPointer(TANGENT_LOCATION, 3, GL_FLOAT, GL_FALSE, 0, nullptr);
        f->glEnableVertexAttribArray(TANGENT_LOCATION);   
    }

    f->glBindVertexArray(0);

    createBBVAO(f);

    createNormalVAO(f);
}

void TriangleMesh::cleanupVBO() {
    if (!f) return;
    cleanupVBO(f);
}

void TriangleMesh::cleanupVBO(QOpenGLFunctions_3_3_Core* f) {
    // delete VAO
    if (VAO.val != 0) f->glDeleteVertexArrays(1, &VAO.val);
    // delete VBO
    if (VBOv.val != 0) f->glDeleteBuffers(1, &VBOv.val);
    if (VBOn.val != 0) f->glDeleteBuffers(1, &VBOn.val);
    if (VBOf.val != 0) f->glDeleteBuffers(1, &VBOf.val);
    if (VBOc.val != 0) f->glDeleteBuffers(1, &VBOc.val);
    if (VBOt.val != 0) f->glDeleteBuffers(1, &VBOt.val);
    if (VBOtan.val != 0) f->glDeleteBuffers(1, &VBOtan.val);
    if (VAObb.val != 0) f->glDeleteVertexArrays(1, &VAObb.val);
    if (VBOvbb.val != 0) f->glDeleteBuffers(1, &VBOvbb.val);
    if (VBOfbb.val != 0) f->glDeleteBuffers(1, &VBOfbb.val);
    if (VAOn.val != 0) f->glDeleteVertexArrays(1, &VAOn.val);
    if (VBOvn.val != 0) f->glDeleteBuffers(1, &VBOvn.val);
    VBOv.val = 0;
    VBOn.val = 0;
    VBOf.val = 0;
    VBOc.val = 0;
    VBOt.val = 0;
    VBOtan.val = 0;
    VAO.val = 0;
    VAObb.val = 0;
    VBOfbb.val = 0;
    VBOvbb.val = 0;
    VAOn.val = 0;
    VBOvn.val = 0;
}

unsigned int TriangleMesh::draw(RenderState& state) {
    if (!boundingBoxIsVisible(state)) return 0;
    if (VAO.val == 0) return 0;
    if (withBB || withNormals) {
        GLuint formerProgram = state.getCurrentProgram();
        state.switchToStandardProgram();
        if (withBB) drawBB(state);
        if (withNormals) drawNormals(state);
        state.setCurrentProgram(formerProgram);
    }
    drawVBO(state);

    return triangles.size();
}

void TriangleMesh::drawVBO(RenderState& state) {
    auto* f = state.getOpenGLFunctions();

    //Bug in Qt: They flagged glVertexAttrib3f as deprecated in modern OpenGL, which is not true.
    //We have to load it manually. Make it static so we do it only once.
    static auto glVertexAttrib3fv = reinterpret_cast<glVertexAttrib3fvPtr>(QOpenGLContext::currentContext()->getProcAddress("glVertexAttrib3fv"));
    
    // The VAO keeps track of all the buffers and the element buffer, so we do not need to bind else except for the VAO
    f->glBindVertexArray(VAO.val);
    f->glUniformMatrix4fv(state.getModelViewUniform(), 1, GL_FALSE, state.getCurrentModelViewMatrix().data());
    f->glUniformMatrix3fv(state.getNormalMatrixUniform(), 1, GL_FALSE, state.calculateNormalMatrix().data());
    switch (coloringType) {
        case ColoringType::TEXTURE:
            if (textureID.val != 0) {
                f->glUniform1ui(state.getUseTextureUniform(), GL_TRUE);
                f->glActiveTexture(GL_TEXTURE0);
                f->glBindTexture(GL_TEXTURE_2D, textureID.val);
                f->glUniform1i(state.getTextureUniform(), 0);
                break;
            }
            //[[fallthrough]];

        case ColoringType::COLOR_ARRAY:
            if (VBOc.val != 0) {
                f->glUniform1ui(state.getUseTextureUniform(), GL_FALSE);
                f->glEnableVertexAttribArray(COLOR_LOCATION);
                break;
            }
            //[[fallthrough]];

        case ColoringType::STATIC_COLOR:
            f->glUniform1ui(state.getUseTextureUniform(), GL_FALSE);
            f->glDisableVertexAttribArray(COLOR_LOCATION); //By disabling the attribute array, it uses the value set in the following line.
            glVertexAttrib3fv(2, reinterpret_cast<const GLfloat*>(&staticColor));
            break;

        case ColoringType::BUMP_MAPPING:
            // Use static color as base color.
            f->glDisableVertexAttribArray(COLOR_LOCATION);
            glVertexAttrib3fv(2, reinterpret_cast<const GLfloat*>(&staticColor));

            GLint location;
            auto program = state.getCurrentProgram();

            location = f->glGetUniformLocation(program, "useDiffuse");
            f->glUniform1ui(location, enableDiffuseTexture);

            location = f->glGetUniformLocation(program, "useNormal");
            f->glUniform1ui(location, enableNormalMapping);

            location = f->glGetUniformLocation(program, "useDisplacement");
            f->glUniform1ui(location, enableDisplacementMapping);

            location = f->glGetUniformLocation(program, "diffuseTexture");
            f->glUniform1i(location, 0);
            f->glActiveTexture(GL_TEXTURE0);
            f->glBindTexture(GL_TEXTURE_2D, textureID.val);

            location = f->glGetUniformLocation(program, "normalTexture");
            f->glUniform1i(location, 1);
            f->glActiveTexture(GL_TEXTURE1);
            f->glBindTexture(GL_TEXTURE_2D, normalMapID.val);

            location = f->glGetUniformLocation(program, "displacementTexture");
            f->glUniform1i(location, 3);
            f->glActiveTexture(GL_TEXTURE3);
            f->glBindTexture(GL_TEXTURE_2D, displacementMapID.val);
            break;
    }
    f->glDrawElements(GL_TRIANGLES, 3*triangles.size(), GL_UNSIGNED_INT, nullptr);
}

// ===========
// === VFC ===
// ===========

// method determining whether the bounding box is in the frustum or outside
bool TriangleMesh::isInsideFrustum(std::vector<Plane> planes)
{
    QVector3D cmin(boundingBoxMin.x(), boundingBoxMin.y(), boundingBoxMin.z());
    QVector3D cmax(boundingBoxMax.x(), boundingBoxMax.y(), boundingBoxMax.z());

    for (auto& plane : planes)
    {
        QVector3D p(boundingBoxMid.x(), boundingBoxMid.y(), boundingBoxMid.z());
        // QVector3D p(
        //     (n.x() >= 0.0f) ? cmax.x() : cmin.x(),
        //     (n.y() >= 0.0f) ? cmax.y() : cmin.y(),
        //     (n.z() >= 0.0f) ? cmax.z() : cmin.z()
        // );

        // equation xn - d = ax1+bx2+cx3-d > 0
        if (QVector3D::dotProduct(p, plane.n) - plane.d > 0.0f) {
            return false; // return early if the box is outside any planes to cull this obj
        }
    }

	return true;
}

bool TriangleMesh::boundingBoxIsVisible(const RenderState& state) {
    // 3.3 Implement view frustum culling.
    // projectionMatrix and model view matrix in 4x4
    QMatrix4x4 projection = state.getCurrentProjectionMatrix();
	QMatrix4x4 modelView = state.getCurrentModelViewMatrix();

    // product of projection matrix and vertex
	QMatrix4x4 VP = projection * modelView;

	// raw data of VP matrix
    float* vp = VP.data();
    /*
    m[0] m[4] m[ 8] m[12]
    m[1] m[5] m[ 9] m[13]
    m[2] m[6] m[10] m[14]
    m[3] m[7] m[11] m[15]
    */

    std::vector<Plane> planes(6);
    // left plane
    planes[0].n = QVector3D(vp[3] + vp[0], vp[7] + vp[4], vp[11] + vp[8]);
    planes[0].d = vp[15] + vp[12];
	// right plane
	planes[1].n = QVector3D(vp[3] - vp[0], vp[7] - vp[4], vp[11] - vp[8]);
	planes[1].d = vp[15] - vp[12];
    // bottom plane
    planes[2].n = QVector3D(vp[3] + vp[1], vp[7] + vp[5], vp[11] + vp[9]);
    planes[2].d = vp[15] + vp[13];
	// top plane
	planes[3].n = QVector3D(vp[3] - vp[1], vp[7] - vp[5], vp[11] - vp[9]);
	planes[3].d = vp[15] - vp[13];
    // near plane
	planes[4].n = QVector3D(vp[3] + vp[2], vp[7] + vp[6], vp[11] + vp[10]);
	planes[4].d = vp[15] + vp[14];
	// far plane
	planes[5].n = QVector3D(vp[3] - vp[2], vp[7] - vp[6], vp[11] - vp[10]);
	planes[5].d = vp[15] - vp[14];

	// normalize plane normals
    for (auto& plane : planes)
    {
        float magnitude = sqrt(plane.n.x() * plane.n.x() + plane.n.y() * plane.n.y() + plane.n.z() * plane.n.z());
        plane.n /= magnitude;
        plane.d /= magnitude;
    }

    int culled = 0;
    bool shouldDraw = isInsideFrustum(planes);
    if (!shouldDraw)
        culled++;

    return shouldDraw;
}

void TriangleMesh::setStaticColor(Vec3f color) {
    staticColor = color;
}

void TriangleMesh::drawBB(RenderState &state) {
    auto* f = state.getOpenGLFunctions();
    f->glBindVertexArray(VAObb.val);
    //Transform BB to correct position.
    state.pushModelViewMatrix();
    state.getCurrentModelViewMatrix().translate(boundingBoxMid.x(), boundingBoxMid.y(), boundingBoxMid.z());
    state.getCurrentModelViewMatrix().scale(boundingBoxSize.x(), boundingBoxSize.y(), boundingBoxSize.z());
    f->glUniformMatrix4fv(state.getModelViewUniform(), 1, GL_FALSE, state.getCurrentModelViewMatrix().data());
    //Set color to constant white.
    //Bug in Qt: They flagged glVertexAttrib3f as deprecated in modern OpenGL, which is not true.
    //We have to load it manually. Make it static so we do it only once.
    static auto glVertexAttrib3f = reinterpret_cast<glVertexAttrib3fPtr>(QOpenGLContext::currentContext()->getProcAddress("glVertexAttrib3f"));
    glVertexAttrib3f(2, 1.0f, 1.0f, 1.0f);

    f->glDrawElements(GL_LINES, 24, GL_UNSIGNED_INT, nullptr);
    state.popModelViewMatrix();
}

void TriangleMesh::drawNormals(RenderState &state) {
    auto* f = state.getOpenGLFunctions();
    f->glBindVertexArray(VAOn.val);
    f->glUniformMatrix4fv(state.getModelViewUniform(), 1, GL_FALSE, state.getCurrentModelViewMatrix().data());

    //Set color to constant white.
    //Bug in Qt: They flagged glVertexAttrib3f as deprecated in modern OpenGL, which is not true.
    //We have to load it manually. Make it static so we do it only once.
    static auto glVertexAttrib3f = reinterpret_cast<glVertexAttrib3fPtr>(QOpenGLContext::currentContext()->getProcAddress("glVertexAttrib3f"));
    glVertexAttrib3f(2, 1.0f, 1.0f, 1.0f);

    f->glDrawArrays(GL_LINES, 0, vertices.size() * 2);
}

void TriangleMesh::generateSphere(QOpenGLFunctions_3_3_Core* f) {
    // The sphere consists of latdiv rings of longdiv faces.
    int longdiv = 200; // minimum 4
    int latdiv  = 100; // minimum 2

    setGLFunctionPtr(f);

    // Generate vertices.
    for (int latitude = 0; latitude <= latdiv; latitude++) {
        float v = static_cast<float>(latitude) / static_cast<float>(latdiv);
        float latangle = v * M_PI;

        float extent = std::sin(latangle);
        float y = -std::cos(latangle);

        for (int longitude = 0; longitude <= longdiv; longitude++) {
            float u = static_cast<float>(longitude) / static_cast<float>(longdiv);
            float longangle = u * 2.0f * M_PI;

            float z = std::sin(longangle) * extent;
            float x = std::cos(longangle) * extent;

            Vec3f pos(x, y, z);

            vertices.push_back(pos);
            normals.push_back(pos);
            texCoords.push_back({ 2.0f - 2.0f * u, v });
            tangents.push_back(cross(Vec3f(0, 1, 0), pos));
        }
    }

    for (int latitude = 0; latitude < latdiv; latitude++) {
        unsigned int bottomBase = latitude * (longdiv + 1);
        unsigned int topBase = (latitude + 1) * (longdiv + 1);
        for (int longitude = 0; longitude < longdiv; longitude++) {
            unsigned int bottomCurrent = bottomBase + longitude;
            unsigned int bottomNext = bottomBase + (longitude + 1);
            unsigned int topCurrent = topBase + longitude;
            unsigned int topNext = topBase + (longitude + 1);
            triangles.emplace_back(bottomCurrent, bottomNext, topNext);
            triangles.emplace_back(topNext, topCurrent, bottomCurrent);
        }
    }

    boundingBoxMid = Vec3f(0, 0, 0);
    boundingBoxSize = Vec3f(2, 2, 2);
    boundingBoxMin = Vec3f(-1, -1, -1);
    boundingBoxMax = Vec3f(1, 1, 1);

    createAllVBOs();
}

void TriangleMesh::generateTerrain(unsigned int l, unsigned int w, unsigned int iterations) {
    // 3.1: Implement terrain generation.
    // The terrain should be a grid of size l x w nodes.

    // generate heightmap using The Fault Algorithm
    int displacementType = rand() % 4;
    std::vector<std::vector<double>> heightmap = generateHeightmap(l, w, iterations, displacementType);
   
    vertices.clear();
    colors.clear();
    triangles.clear();

    for (int x = 0; x < l; x++) 
    for (int z = 0; z < w; z++)
    {
	    double height = heightmap[x][z];

    	// for each cell (x,z) add vertices (x, height, z)
        vertices.emplace_back(x, height, z);

	    calculateTerrainColor(height, displacementType);
    }

    // for each cell create two triangles: 
    // triangle 1 has the cell, the cell to the right and the cell below
    // triangle 2 has the cell to the right, the cell below and the cell below of the cell to the right
    for (int x = 0; x < l - 1; x++) {
        for (int z = 0; z < w - 1; z++) {
            int cell = x * w + z;
            int right = cell + 1;
            int below = cell + w;
            int belowRight = below + 1;

            triangles.emplace_back(cell, right, below);
            triangles.emplace_back(right, below, belowRight);
        }
    }

    calculateNormalsByArea();
    calculateBB();
    createAllVBOs();
}

std::vector<std::vector<double>> TriangleMesh::generateHeightmap(int l, int w, int iterations, int displacementType)
{
    std::vector<std::vector<double>> heightmap(l, std::vector<double>(w));

    float d = std::sqrt(w * w + l * l);
    double displacement = 0.1;
    float waveSize = d / 10.0f;

    for (int i = 0; i < iterations; i++)
    {
        float v = std::rand();
        // convert v to radian
        v = v * M_PI / 180.0f;
        float a = std::sin(v);
        float b = std::cos(v);
        // rand() / RAND_MAX gives a random number between 0 and 1.
        // therefore c will be a random number between -d/2 and d/2
        float c = (static_cast<float>(rand()) / RAND_MAX) * d - d / 2.0f;

        for (int x = 0; x < heightmap.size(); x++)
        {
            for (int z = 0; z < heightmap[0].size(); z++)
            {
                float dist = a * x + b * z - c;
              
                // cosine function
            	if (displacementType == 0) {
                    float cosValue = std::cos(dist / waveSize * M_PI);
                    heightmap[x][z] += displacement / 2.0f * cosValue;
                }
                // sine function
                else if (displacementType == 1) {
                    float sinValue = std::sin(dist / waveSize * M_PI);
                    heightmap[x][z] += displacement / 2.0f * sinValue;
                }
                // step function
                else {
                    heightmap[x][z] += dist > 0 ? displacement : -displacement;
                }
            }
        }
    }

    return heightmap;
}

void TriangleMesh::calculateTerrainColor(double height, int displacementType)
{
    Vec3f deepWater(0.0f, 0.0f, 0.5f);      // deep blue
    Vec3f shallowWater(0.0f, 0.5f, 1.0f);   // light blue
    Vec3f sand(0.93f, 0.87f, 0.5f);         // sand/yellow
    Vec3f lowLand(0.2f, 0.8f, 0.2f);        // light green
    Vec3f grass(0.0f, 0.6f, 0.0f);          // dark green
    Vec3f forest(0.0f, 0.4f, 0.0f);         // forest green
    Vec3f mountain(0.6f, 0.4f, 0.2f);       // brown
    Vec3f rock(0.5f, 0.5f, 0.5f);           // grey
    Vec3f snow(1.0f, 1.0f, 1.0f);           // white

    Vec3f color = 0;

    // terrain color for step function
    if (displacementType > 1)
    {
        if (height < -7.0f) 
            color = deepWater;
        else if (height < -4.0f)
            color = shallowWater;
        else if (height < -3.0f)
            color = sand;
        else if (height < 0.0f)
            color = lowLand;
        else if (height < 5.0f)
            color = grass;
        else if (height < 8.0f)
            color = forest;
        else if (height < 9.0f)
            color = mountain;
        else if (height < 10.0f)
            color = rock;
        else
            color = snow;
    }
    // ignore deepWater for cosine and sine functions since they do not look realistic
    // shrink down the color height so have more colors for these functions
    else
    {
        if (height < -5.5f)
            color = shallowWater;
        else if (height < -4.5f)
            color = sand;
        else if (height < -3.5f)
            color = lowLand;
        else if (height < -1.5f)
            color = grass;
        else if (height < 0.5f)
            color = forest;
        else if (height < 2.0f)
            color = mountain;
        else if (height < 3.5f)
            color = rock;
        else
            color = snow;
    }
    
    colors.push_back(color);
}