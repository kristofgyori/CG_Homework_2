## Dini's surface

```cpp
vec3 r(float u, float v) {
    float x = a*cosf(u)*sinf(v);
    float y = a*sinf(u)*sinf(v);
    float z = a*(cosf(v) + logf(tanf(v / 2.0))) + b*u;
    return vec3(x, y, z);
}
```

```cpp
vec3 getNormalVector(float u, float v) {
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
```

