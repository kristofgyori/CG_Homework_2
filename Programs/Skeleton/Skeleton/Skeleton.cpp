// A beadott program csak ebben a fajlban lehet, a fajl 1 byte-os ASCII karaktereket tartalmazhat, BOM kihuzando.
// Tilos:
// - mast "beincludolni", illetve mas konyvtarat hasznalni
// - faljmuveleteket vegezni a printf-et kiveve
// - Mashonnan atvett programresszleteket forrasmegjeloles nelkul felhasznalni es
// - felesleges programsorokat a beadott programban hagyni!!!!!!! 
// - felesleges kommenteket a beadott programba irni a forrasmegjelolest kommentjeit kiveve
// ---------------------------------------------------------------------------------------------
// A feladatot ANSI C++ nyelvu forditoprogrammal ellenorizzuk, a Visual Studio-hoz kepesti elteresekrol
// es a leggyakoribb hibakrol (pl. ideiglenes objektumot nem lehet referencia tipusnak ertekul adni)
// a hazibeado portal ad egy osszefoglalot.
// ---------------------------------------------------------------------------------------------
// A feladatmegoldasokban csak olyan OpenGL fuggvenyek hasznalhatok, amelyek az oran a feladatkiadasig elhangzottak 
// A keretben nem szereplo GLUT fuggvenyek tiltottak.
//
// NYILATKOZAT
// ---------------------------------------------------------------------------------------------
// Nev    : Gyõri Kristóf
// Neptun : HV0R9S
// ---------------------------------------------------------------------------------------------
// ezennel kijelentem, hogy a feladatot magam keszitettem, es ha barmilyen segitseget igenybe vettem vagy
// mas szellemi termeket felhasznaltam, akkor a forrast es az atvett reszt kommentekben egyertelmuen jeloltem.
// A forrasmegjeloles kotelme vonatkozik az eloadas foliakat es a targy oktatoi, illetve a
// grafhazi doktor tanacsait kiveve barmilyen csatornan (szoban, irasban, Interneten, stb.) erkezo minden egyeb
// informaciora (keplet, program, algoritmus, stb.). Kijelentem, hogy a forrasmegjelolessel atvett reszeket is ertem,
// azok helyessegere matematikai bizonyitast tudok adni. Tisztaban vagyok azzal, hogy az atvett reszek nem szamitanak
// a sajat kontribucioba, igy a feladat elfogadasarol a tobbi resz mennyisege es minosege alapjan szuletik dontes.
// Tudomasul veszem, hogy a forrasmegjeloles kotelmenek megsertese eseten a hazifeladatra adhato pontokat
// negativ elojellel szamoljak el es ezzel parhuzamosan eljaras is indul velem szemben.
//=============================================================================================
#include "framework.h"

// Konstansok //
const float epsilon = 0.0001f;

////////////////

// Anyag
struct Material {
	vec3 ka, kd, ks;
	float  shininess;
	bool rough;
	bool refracitve;
	Material(vec3 _kd, vec3 _ks, float _shininess) : ka(_kd * M_PI), kd(_kd), ks(_ks) { shininess = _shininess; }
};

// Metszéspont
struct Hit {
	float t;
	vec3 position, normal;
	Material * material;
	Hit() { t = -1; }
};

// Sugáregyenes
struct Ray {
	vec3 start, dir;
	Ray(vec3 _start, vec3 _dir) { start = _start; dir = normalize(_dir); }
};

class Intersectable {
protected:
	Material * material;
public:
	virtual Hit intersect(const Ray& ray) = 0;
};
struct Point3D {
	vec3 r;
	vec3 n;
};

struct Plane : public Intersectable {
	vec3 point, normal;

	Plane(const vec3& _point, const vec3& _normal, Material* mat) {
		point = _point;
		normal = normalize(_normal);
		material = mat;
	}
	Hit intersect(const Ray& ray) {
		Hit hit;
		double NdotV = dot(normal, ray.dir);
		if (fabs(NdotV) < epsilon) return hit;
		double t = dot(normal, point - ray.start) / NdotV;
		if (t < epsilon) return hit;
		hit.t = t;
		hit.position = ray.start + ray.dir * hit.t;
		hit.normal = normal;
		if (dot(hit.normal, ray.dir) > 0) hit.normal = hit.normal * (-1); // flip the normal
		hit.material = material;
		return hit;
	}
};
class Triangle : Intersectable {

	Point3D r1;
	Point3D r2;
	Point3D r3;

	vec3 normal;
public:
	Triangle(vec3 _r1, vec3 _r2, vec3 _r3, vec3 _n1, vec3 _n2, vec3 _n3) {
		r1.r = _r1;	r1.n = _n1;
		r2.r = _r2;	r2.n = _n2;
		r3.r = _r3;	r3.n = _n3;

		normal = cross(r2.r - r1.r, r3.r - r1.r);

		// Linear Interpolation

	}

	vec3 GetNormal() {
		return normal;
	}

	Hit intersect(const Ray& ray) {
		Hit hit;
		hit.t = dot((r1.r - ray.start), normal) / (dot(ray.dir, normal));

		hit.position = ray.start + ray.dir * hit.t;

		/*	if (dot(ray.dir, normal) > 0)
		normal = normal * (-1);*/

		hit.normal = normal;
		if (isItInside(hit.position)) {
			return hit;
		}
		else
			return Hit();		// default t = -1
	}

	bool isItInside(const vec3& p) {
		bool e1 = dot(cross(r2.r - r1.r, p - r1.r), normal) > 0;
		bool e2 = dot(cross(r3.r - r2.r, p - r2.r), normal) > 0;
		bool e3 = dot(cross(r1.r - r3.r, p - r3.r), normal) > 0;

		return e1 && e2 && e3;
	}

	vec3 GetClosestVertex(const vec3& p) {
		vec3 closest = r1.r;

		if (length(closest - p) > length(r2.r - p))
			closest = r2.r;
		if (length(closest - p) > length(r3.r - p))
			closest = r3.r;

		return closest;
	}

};


class DiniSurface : public Intersectable
{
private:
	const float u_min = 0.0;
	const float u_max = 4 * M_PI;
	const float v_min = 0.01;
	const float v_max = 1;
	const float a = 1;
	const float b = 0.15;
	//const float epsilon = 0.0001;

	const int N = 20;		// u
	const int M = 20;		// v

	std::vector<Triangle> triangles;

	void fillTriangles() {
		for (int i = 0; i < N; i++) {
			for (int j = 0; j < M; j++) {
				float ui = u_min + (u_max - u_min)*((float)i) / N;
				float vj = v_min + (v_max - v_min)*((float)j) / M;

				vec2 u1(u_min + (u_max - u_min)*((float)i) / N, vj = v_min + (v_max - v_min)*((float)j) / M);
				vec2 u2(u_min + (u_max - u_min)*((float)(i + 1)) / N, vj = v_min + (v_max - v_min)*((float)j) / M);
				vec2 u3(u_min + (u_max - u_min)*((float)i) / N, vj = v_min + (v_max - v_min)*((float)(j + 1)) / M);

				Triangle triangle1(
					r(u1.x, u1.y),
					r(u2.x, u2.y),
					r(u3.x, u3.y),
					GetNormalVector(u1.x, u1.y),
					GetNormalVector(u2.x, u2.y),
					GetNormalVector(u3.x, u3.y)
				);

				vec2 u4 = u2;
				vec2 u5(u_min + (u_max - u_min)*((float)(i + 1)) / N, vj = v_min + (v_max - v_min)*((float)(j + 1)) / M);
				vec2 u6 = u3;

				Triangle triangle2(
					r(u4.x, u4.y),
					r(u5.x, u5.y),
					r(u6.x, u6.y),
					GetNormalVector(u4.x, u4.y),
					GetNormalVector(u5.x, u5.y),
					GetNormalVector(u6.x, u6.y)
				);

				triangles.push_back(triangle1);
				triangles.push_back(triangle2);
			}
		}
	}

public:
	DiniSurface(Material* _material) {
		material = _material;
		fillTriangles();
	}

	vec3 r(float u, float v) {
		float x = a*cosf(u)*sinf(v);
		float y = a*sinf(u)*sinf(v);
		float z = a*(cosf(v) + logf(tanf(v / 2.0))) + b*u;
		return vec3(x, y, z);
	}

	vec3 GetNormalVector(float u, float v) {
		float rv_x = a*cosf(u)*cosf(v);
		float rv_y = a*sinf(u)*cosf(v);
		float rv_z = a*sinf(v) + (1.0 / (2 * sinf(v / 2.0)*cosf(v / 2.0)));

		float ru_x = a*sinf(v)*(-sinf(u));
		float ru_y = a*sinf(v)*cosf(u);
		float ru_z = b;

		vec3 rv(rv_x, rv_y, rv_z);
		vec3 ru(ru_x, ru_y, ru_z);

		return cross(ru, rv);
	}

	// Hol metszi az adott sugár a felületet?
	Hit intersect(const Ray& ray) {
		Hit hit;
		float tempt = -1;
		//int i = 0;
		for (int k = 0; k < triangles.size(); k++)
		{
			hit = triangles[k].intersect(ray);
			if (hit.t > 0) {
				tempt = hit.t;
				break;
			}
		}
		hit.t = tempt;
		hit.material = material;
		return hit;
	}



};

class Camera {
	vec3 eye, lookat, right, up;
public:
	void set(vec3 _eye, vec3 _lookat, vec3 vup, double fov) {
		eye = _eye;
		lookat = _lookat;
		vec3 w = eye - lookat;
		float f = length(w);
		right = normalize(cross(vup, w)) * f * tan(fov / 2);
		up = normalize(cross(w, right)) * f * tan(fov / 2);
	}
	Ray getRay(int X, int Y) {
		vec3 dir = lookat + right * (2.0 * (X + 0.5) / windowWidth - 1) + up * (2.0 * (Y + 0.5) / windowHeight - 1) - eye;
		return Ray(eye, dir);
	}
};

struct Light {
	vec3 direction;
	vec3 Le;
	Light(vec3 _direction, vec3 _Le) {
		direction = normalize(_direction);
		Le = _Le;
	}
};


class Scene {
	std::vector<Intersectable *> objects;
	std::vector<Light *> lights;
	Camera camera;
	vec3 La;
public:
	void build() {
		vec3 eye = vec3(0, -7.5, 5), vup = vec3(0, 1, 0), lookat = vec3(0, 0, 0);
		float fov = 45 * M_PI / 180;
		camera.set(eye, lookat, vup, fov);

		La = vec3(0.4f, 0.4f, 0.4f);
		vec3 lightDirection(0, 1, 2), Le(50, 50, 50);
		lights.push_back(new Light(lightDirection, Le));

		La = vec3(0.4f, 0.4f, 0.4f);
		vec3 lightDirection2(0, -1, 2), Le2(50, 50, 50);
		lights.push_back(new Light(lightDirection2, Le2));

		vec3 kd(0.3f, 0.2f, 0.1f), ks(2, 2, 2);
		Material * material = new Material(kd, ks, 50);
		objects.push_back(new DiniSurface(material));
		
	//	objects.push_back(new Plane(vec3(0, 0, 0), vec3(0, 0, 1), material));


	}

	void render(std::vector<vec4>& image) {
		for (int Y = 0; Y < windowHeight; Y++) {
#pragma omp parallel for
			for (int X = 0; X < windowWidth; X++) {
				vec3 color = trace(camera.getRay(X, Y));
				image[Y * windowWidth + X] = vec4(color.x, color.y, color.z, 1);
			}
		}
	}

	Hit firstIntersect(Ray ray) {
		Hit bestHit;
		for (Intersectable * object : objects) {
			Hit hit = object->intersect(ray); //  hit.t < 0 if no intersection
			if (hit.t > 0 && (bestHit.t < 0 || hit.t < bestHit.t))  bestHit = hit;
		}
		if (dot(ray.dir, bestHit.normal) > 0) bestHit.normal = bestHit.normal * (-1);
		return bestHit;
	}

	bool shadowIntersect(Ray ray) {	// for directional lights
		for (Intersectable * object : objects)
			if (object->intersect(ray).t > 0) return true;
		return false;
	}

	vec3 trace(Ray ray, int depth = 0) {
		Hit hit = firstIntersect(ray);
		if (hit.t < 0) return La;
		vec3 outRadiance = hit.material->ka * La;
		for (Light * light : lights) {
			Ray shadowRay(hit.position + hit.normal * epsilon, light->direction);
			float cosTheta = dot(hit.normal, light->direction);
			if (cosTheta > 0 && !shadowIntersect(shadowRay)) {	// shadow computation
				outRadiance = outRadiance + light->Le * hit.material->kd * cosTheta;
				vec3 halfway = normalize(-ray.dir + light->direction);
				float cosDelta = dot(hit.normal, halfway);
				if (cosDelta > 0)
					outRadiance = outRadiance + light->Le * hit.material->ks * powf(cosDelta, hit.material->shininess);
			}
		}
		return outRadiance;
	}
};

GPUProgram gpuProgram; // vertex and fragment shaders
Scene scene;

// vertex shader in GLSL
const char *vertexSource = R"(
	#version 330
    precision highp float;

	layout(location = 0) in vec2 cVertexPosition;	// Attrib Array 0
	out vec2 texcoord;

	void main() {
		texcoord = (cVertexPosition + vec2(1, 1))/2;							// -1,1 to 0,1
		gl_Position = vec4(cVertexPosition.x, cVertexPosition.y, 0, 1); 		// transform to clipping space
	}
)";

// fragment shader in GLSL
const char *fragmentSource = R"(
	#version 330
    precision highp float;

	uniform sampler2D textureUnit;
	in  vec2 texcoord;			// interpolated texture coordinates
	out vec4 fragmentColor;		// output that goes to the raster memory as told by glBindFragDataLocation

	void main() {
		fragmentColor = texture(textureUnit, texcoord); 
	}
)";

class FullScreenTexturedQuad {
	unsigned int vao;	// vertex array object id and texture id
	Texture * pTexture;
public:
	void Create(std::vector<vec4>& image) {
		glGenVertexArrays(1, &vao);	// create 1 vertex array object
		glBindVertexArray(vao);		// make it active

		unsigned int vbo;		// vertex buffer objects
		glGenBuffers(1, &vbo);	// Generate 1 vertex buffer objects

								// vertex coordinates: vbo0 -> Attrib Array 0 -> vertexPosition of the vertex shader
		glBindBuffer(GL_ARRAY_BUFFER, vbo); // make it active, it is an array
		float vertexCoords[] = { -1, -1,  1, -1,  1, 1,  -1, 1 };	// two triangles forming a quad
		glBufferData(GL_ARRAY_BUFFER, sizeof(vertexCoords), vertexCoords, GL_STATIC_DRAW);	   // copy to that part of the memory which is not modified 
		glEnableVertexAttribArray(0);
		glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, NULL);     // stride and offset: it is tightly packed

		pTexture = new Texture(windowWidth, windowHeight, image);
	}

	void Draw() {
		glBindVertexArray(vao);	// make the vao and its vbos active playing the role of the data source
		pTexture->SetUniform(gpuProgram.getId(), "textureUnit");
		glDrawArrays(GL_TRIANGLE_FAN, 0, 4);	// draw two triangles forming a quad
	}
};

FullScreenTexturedQuad fullScreenTexturedQuad;

// Initialization, create an OpenGL context
void onInitialization() {
	glViewport(0, 0, windowWidth, windowHeight);
	scene.build();

	std::vector<vec4> image(windowWidth * windowHeight);
	long timeStart = glutGet(GLUT_ELAPSED_TIME);
	scene.render(image);
	long timeEnd = glutGet(GLUT_ELAPSED_TIME);
	printf("Rendering time: %d milliseconds\n", (timeEnd - timeStart));
	fullScreenTexturedQuad.Create(image);

	// create program for the GPU
	gpuProgram.Create(vertexSource, fragmentSource, "fragmentColor");
}

// Window has become invalid: Redraw
void onDisplay() {
	fullScreenTexturedQuad.Draw();
	glutSwapBuffers();									// exchange the two buffers
}

// Key of ASCII code pressed
void onKeyboard(unsigned char key, int pX, int pY) {
}

// Key of ASCII code released
void onKeyboardUp(unsigned char key, int pX, int pY) {

}

// Mouse click event
void onMouse(int button, int state, int pX, int pY) {
}

// Move mouse with key pressed
void onMouseMotion(int pX, int pY) {
}

// Idle event indicating that some time elapsed: do animation here
void onIdle() {
}

