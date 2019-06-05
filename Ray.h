#pragma once
#include "vec3.h"
#include "RayObject.h"

class Ray : public RayObject
{
public:
	__device__ Ray(){};
	//Constructor: two vec3 specify a line in 3D.
	__device__ Ray(const vec3& a, const vec3& b) { A = a; B = b; }
	//Get the intercept
	__device__ vec3 origin() const { return A; }
	//Get the direction of the ray (not unitized)
	__device__ vec3 direction() const { return B; }
	//Evaluate the function that defines the line at t
	__device__ vec3 operator[](double t) const { return A + t*B; }
	//The intercept of the line (the value when t = 0)
	vec3 A;
	//The direction of the line.
	vec3 B;
};