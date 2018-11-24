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
inline vec3 operator/(const vec3& v, const vec3& u) { return vec3(v.x / u.x, v.y / u.y, v.z / u.z); }
// Anyag
struct Material {
	vec3 ka, kd, ks;
	float  shininess;
	bool rough = true;
	bool reflective = false;
	bool refractive = false;
	Material(vec3 _kd, vec3 _ks, float _shininess) : ka(_kd * M_PI), kd(_kd), ks(_ks) { shininess = _shininess; }
};


struct Hit {
	float t;
	vec3 position, normal;
	Material * material;
	Hit() { t = -1; }
};


struct Ray {
	vec3 start, dir, reflectionDir;
	bool out = true;
	Ray(vec3 _start, vec3 _dir) { start = _start; dir = normalize(_dir); }
	Ray(vec3 _start, vec3 _dir, bool _out) { start = _start; dir = normalize(_dir); out = _out; }
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


class Triangle : public Intersectable {

	Point3D r1;
	Point3D r2;
	Point3D r3;

	float a;
	float b;
	float c;

	vec3 normal;
public:

	Triangle(vec3 _r1, vec3 _r2, vec3 _r3, Material* _material) {
		r1.r = _r1;
		r2.r = _r2;
		r3.r = _r3;
		material = _material;
		normal = cross(r2.r - r1.r, r3.r - r1.r);

		// Linear Interpolation

	}

	Triangle(vec3 _r1, vec3 _r2, vec3 _r3, vec3 _n1, vec3 _n2, vec3 _n3) {
		r1.r = _r1;    r1.n = _n1;
		r2.r = _r2;    r2.n = _n2;
		r3.r = _r3;    r3.n = _n3;

		normal = cross(r2.r - r1.r, r3.r - r1.r);

		// Linear Interpolation
		vec3 normal1 = normalize(r1.n);
		vec3 normal2 = normalize(r2.n);
		vec3 normal3 = normalize(r3.n);
		/*float X1 = normal1.x; float Y1 = normal1.y; float U1 = normal1.z;
		float X2 = normal2.x; float Y2 = normal2.y; float U2 = normal2.z;
		float X3 = normal3.x; float Y3 = normal3.y; float U3 = normal3.z;*/

		float X1 = r1.n.x; float Y1 = r1.n.y; float U1 = r1.n.z;
		float X2 = r2.n.x; float Y2 = r2.n.y; float U2 = r2.n.z;
		float X3 = r3.n.x; float Y3 = r3.n.y; float U3 = r3.n.z;

		a = ((U3 - U1)*(Y2 - Y1) - (Y3 - Y1)*(U2 - U1)) / ((X3 - X1)*(Y2 - Y1) - (Y3 - Y1)*(X2 - X1));
		b = ((U3 - U1) - a*(X3 - X1)) / (Y3 - Y1);
		c = U1 - a*X2 - b*Y2;


	}
	vec3 setUpTriangle(vec3 point) {
		return vec3(point.x, point.y, a*point.x + b*point.y + c);
	}
	vec3 GetNormal() {
		return normal;
	}

	Hit intersect(const Ray& ray) {
		Hit hit;
		hit.t = dot((r1.r - ray.start), normal) / (dot(ray.dir, normal));
		hit.position = ray.start + ray.dir * hit.t;
		hit.material = material;
		
		hit.normal = normalize(normal);
		if (dot(hit.normal, ray.dir) > 0) hit.normal = hit.normal * (-1); // flip the normal, we are inside the sphere
		if (isItInside(hit.position)) {
			return hit;
		}
		else
			return Hit();        // default t = -1
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

class AABB {
	float x_min;
	float x_max;
	float z_min;
	float z_max;
	float y;
	vec3 normal;
	vec3 point;

public:
	AABB() {};
	AABB(float xmin, float xmax, float zmin, float zmax, float _y) {
		x_min = xmin;x_max = xmax;
		z_min = zmin; z_max = zmax;
		y = _y;
		point = vec3(x_min, y, z_min);
		normal = cross(vec3(x_max, y, z_min) - vec3(x_max, y, z_max), vec3(x_max, y, z_min) - vec3(x_min, y, z_max));
	}
	Hit intersect(const Ray& ray) {
		Hit hit;
		hit.t = dot((point - ray.start), normal) / (dot(ray.dir, normal));
		hit.position = ray.start + ray.dir * hit.t;

		vec3 hp = hit.position;

		if (hp.x > x_min && hp.x < x_max &&  hp.z > z_min && hp.z < z_max)
			return hit;
		hit.normal = normalize(normal);
		hit.t = -1;
		return hit;
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
	const float b = 0.2;

	const int N = 60;        // u
	const int M = 60;        // v

	std::vector<Triangle> triangles;
	AABB bV;
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
		bV = AABB(-5.0f, 1.0f, -3.0f, 5.0f, 0);
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


	Hit intersect(const Ray& ray) {
		//Hit hit;
		Hit hit = bV.intersect(ray);
		if (hit.t < 0)
			return hit;

		float tempt = -1;
		for (int k = 0; k < triangles.size(); k++)
		{
			hit = triangles[k].intersect(ray);
			if (hit.t > 0) {
				tempt = hit.t;
				hit.normal  = normalize(triangles[k].setUpTriangle(hit.position));
				 
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
		vec3 eye = vec3(0, -15, 3), vup = vec3(0, 1, 0), lookat = vec3(0, 0, -2);
		float fov = 45 * M_PI / 180;
		camera.set(eye, lookat, vup, fov);

		La = vec3(0.1f, 0.1f, 0.1f);
		vec3 lightDirection(0,-5,2), Le(0.8, 0.8, 0.8);
		lights.push_back(new Light(lightDirection, Le));
		//lights.push_back(new Light(vec3(-1, 0, 10), Le));
		vec3 kd(1.0f, 1.0f, 0.0f), ks(2, 2, 2);
		Material * material = new Material(kd, ks, 50);
		material->rough = true;
		//material->reflective = true;
		//material->refractive = true;
		objects.push_back(new DiniSurface(material));


		Material * material2 = new Material(vec3(0.682, 0.714, 0.749), ks, 0);
		Material * material3 = new Material(vec3(0.157, 0.706, 0.388), ks, 100);

		float e = 6.0f;
		float z = -3.0f;
		// Floor
		objects.push_back(new Triangle(vec3(e, e, z), vec3(e, -e, z), vec3(-e, -e, z), material3));
		objects.push_back(new Triangle(vec3(e, e, z), vec3(-e, e, z), vec3(-e, -e, z), material3));

		// Right
		objects.push_back(new Triangle(vec3(e, e, z), vec3(e, -e, z), vec3(e, e, e + z), material2));
		objects.push_back(new Triangle(vec3(e, -e, e + z), vec3(e, -e, z), vec3(e, e, e + z), material2));

		// Front
		objects.push_back(new Triangle(vec3(e, e, z), vec3(e, e, z + e), vec3(-e, e, z), material2));
		objects.push_back(new Triangle(vec3(e, e, z + e), vec3(-e, e, z), vec3(-e, e, z + e), material2));

		// Left
		objects.push_back(new Triangle(vec3(-e, -e, z), vec3(-e, e, z), vec3(-e, e, e + z), material2));
		objects.push_back(new Triangle(vec3(-e, -e, z), vec3(-e, e, z + e), vec3(-e, -e, e + z), material2));

	}

	void render(std::vector<vec4>& image) {
		for (int Y = 0; Y < windowHeight; Y++) {
#pragma omp parallel for
			for (int X = 0; X < windowWidth; X++) {
				vec3 color = trace(camera.getRay(X, Y), 3);
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

	bool shadowIntersect(Ray ray) {    // for directional lights
		for (Intersectable * object : objects)
			if (object->intersect(ray).t > 0) return true;
		return false;
	}


	vec3 trace(Ray ray, int depth = 0) {
		Hit hit = firstIntersect(ray);
		if (hit.t < 0) return La;

		vec3 outRadiance(0, 0, 0);
		outRadiance = hit.material->ka * La;
		if (hit.material->rough) {
			for (Light * light : lights) {
				Ray shadowRay(hit.position + hit.normal * epsilon, light->direction);
				float cosTheta = dot(hit.normal, light->direction);
				if (cosTheta > 0 && !shadowIntersect(shadowRay)) {    // shadow computation
					outRadiance = outRadiance + light->Le * hit.material->kd * cosTheta;
					vec3 halfway = normalize(-ray.dir + light->direction);
					float cosDelta = dot(hit.normal, halfway);
					if (cosDelta > 0)
						outRadiance = outRadiance + light->Le * hit.material->ks * powf(cosDelta, hit.material->shininess);
				}
			}
		}

		if (hit.material->reflective) {
			vec3 reflectionDir = reflect(ray.dir, hit.normal);
			Ray reflectRay(hit.position + hit.normal * epsilon, reflectionDir, ray.out);
			outRadiance = outRadiance + trace(reflectRay, depth + 1)*Fresnel(ray.dir, hit.normal);
		}

		if (hit.material->refractive) {
			float ior = (ray.out) ? 0.17 : 1 / 0.17;
			vec3 refractionDir = refract(ray.dir, hit.normal, ior);
			if (length(refractionDir) > 0) {
				Ray refractRay(hit.position + hit.normal * epsilon, refractionDir, !ray.out);
				outRadiance = outRadiance + trace(refractRay, depth + 1)*(vec3(1, 1, 1) - Fresnel(ray.dir, hit.normal));
			}
		}

		return outRadiance;
	}

	vec3 reflect(vec3 inDir, vec3 normal) {
		return inDir - normal * dot(normal, inDir) * 2.0f;
	}

	vec3 refract(vec3 inDir, vec3 normal, float ns) {
		float cosa = -dot(inDir, normal);
		float disc = 1 - (1 - cosa*cosa) / ns / ns; // scalar n
		if (disc < 0) return vec3(0, 0, 0);
		return inDir / ns + normal * (cosa / ns - sqrt(disc));
	}



	//float one = 1;
	vec3 Fresnel(vec3 inDir, vec3 normal) {
		vec3 n(0.17f, 0.35f, 1.5f);
		vec3 kappa(3.1f, 2.7f, 1.9f);
		float cosa = -dot(inDir, normal);
		vec3 one(1, 1, 1);
		vec3 F0 = ((n - one)*(n - one) + kappa*kappa) / ((n + one)*(n + one) + kappa*kappa);

		return F0 + (one - F0) * pow(1 - cosa, 5);
	}


};

//vec3::operator/ 
GPUProgram gpuProgram; // vertex and fragment shaders
Scene scene;

// vertex shader in GLSL
const char *vertexSource = R"(
    #version 330
    precision highp float;
 
    layout(location = 0) in vec2 cVertexPosition;    // Attrib Array 0
    out vec2 texcoord;
 
    void main() {
        texcoord = (cVertexPosition + vec2(1, 1))/2;                            // -1,1 to 0,1
        gl_Position = vec4(cVertexPosition.x, cVertexPosition.y, 0, 1);         // transform to clipping space
    }
)";

// fragment shader in GLSL
const char *fragmentSource = R"(
    #version 330
    precision highp float;
 
    uniform sampler2D textureUnit;
    in  vec2 texcoord;            // interpolated texture coordinates
    out vec4 fragmentColor;        // output that goes to the raster memory as told by glBindFragDataLocation
 
    void main() {
        fragmentColor = texture(textureUnit, texcoord); 
    }
)";

class FullScreenTexturedQuad {
	unsigned int vao;    // vertex array object id and texture id
	Texture * pTexture;
public:
	void Create(std::vector<vec4>& image) {
		glGenVertexArrays(1, &vao);    // create 1 vertex array object
		glBindVertexArray(vao);        // make it active

		unsigned int vbo;        // vertex buffer objects
		glGenBuffers(1, &vbo);    // Generate 1 vertex buffer objects

								  // vertex coordinates: vbo0 -> Attrib Array 0 -> vertexPosition of the vertex shader
		glBindBuffer(GL_ARRAY_BUFFER, vbo); // make it active, it is an array
		float vertexCoords[] = { -1, -1,  1, -1,  1, 1,  -1, 1 };    // two triangles forming a quad
		glBufferData(GL_ARRAY_BUFFER, sizeof(vertexCoords), vertexCoords, GL_STATIC_DRAW);       // copy to that part of the memory which is not modified 
		glEnableVertexAttribArray(0);
		glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, NULL);     // stride and offset: it is tightly packed

		pTexture = new Texture(windowWidth, windowHeight, image);
	}

	void Draw() {
		glBindVertexArray(vao);    // make the vao and its vbos active playing the role of the data source
		pTexture->SetUniform(gpuProgram.getId(), "textureUnit");
		glDrawArrays(GL_TRIANGLE_FAN, 0, 4);    // draw two triangles forming a quad
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
	glutSwapBuffers();                                    // exchange the two buffers
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