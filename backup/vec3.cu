#include "vec3.h"

__host__ __device__ inline vec3 & vec3::operator+=(const vec3 & v2)
{
	e[0] += v2.e[0];
	e[1] += v2.e[1];
	e[2] += v2.e[2];
	return *this;
}

__host__ __device__ inline vec3 & vec3::operator-=(const vec3 & v2)
{
	e[0] -= v2.e[0];
	e[1] -= v2.e[1];
	e[2] -= v2.e[2];
	return *this;
}

__host__ __device__ inline vec3 & vec3::operator*=(const vec3 & v2)
{
	e[0] *= v2.e[0];
	e[1] *= v2.e[1];
	e[2] *= v2.e[2];
	return *this;
}

__host__ __device__ inline vec3 & vec3::operator/=(const vec3 & v2)
{
	e[0] /= v2.e[0];
	e[1] /= v2.e[1];
	e[2] /= v2.e[2];
	return *this;
}

__host__ __device__ inline vec3 & vec3::operator*=(const float t)
{
	e[0] *= t;
	e[1] *= t;
	e[2] *= t;
	return *this;
}

__host__ __device__ inline vec3 & vec3::operator/=(const float t)
{
	//multiplication is faster than division.
	float div = 1 / t;
	e[0] *= div;
	e[1] *= div;
	e[2] *= div;
}

