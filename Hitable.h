#pragma once
#include "vec3.h"
#include "Ray.h"


class Material;


struct HitRecord
{
	//time variable for things like motion blur
	float t;
	//point that was hit
	vec3 p;
	//normal vector at the point that was his
	vec3 normal;

	//hitables and materials need to know about eachother
	//materials will define how rays interact with surface
	//When a ray hits a surface (a particular sphere for example), the material pointer in the
	//	hit_record will be set to point at the material pointer the sphere was given when it was set up in
	//	main() when we start.When the color() routine gets the hit_record it can call member
	//	functions of the material pointer to find out what ray, if any, is scattered.

	Material* mat_p;
};

class hitable : public RayObject {
public:
	//hit only counts if it is between tmin and tmax
	__device__ virtual bool hit(const Ray& r, float tmin, float tmax, HitRecord& rec) const = 0;
};

class hitable_list : public hitable {
public:
	__device__ hitable_list() { p_list = NULL; list_size = 0; }
	__device__ hitable_list(hitable** l, int n){ 
		p_list = l; 
		list_size = n;
	}
	__device__ virtual bool hit(const Ray& r, float tmin, float tmax, HitRecord& rec) const{
		HitRecord temp_rec;
		bool hit_anything = false;
		float closest_so_far = tmax;
		for (size_t i = 0; i < list_size; i++)
		{
			if (p_list[i]->hit(r,tmin,closest_so_far,temp_rec))
			{
				hit_anything = true;
				closest_so_far = temp_rec.t;
				rec = temp_rec;
			}
		}
		return hit_anything;
	}
	hitable **p_list;
	//needed because std::vector is not on device
	int list_size;
	
};

class Sphere : public hitable {
public:
	__device__ Sphere() {}
	__device__ Sphere(vec3 cen, float r, Material* m){
		center = cen;
		radius = r;
		mat = m;
	}
	__device__ virtual bool hit(const Ray& r, float tmin, float tmax, HitRecord& rec) const{
		vec3 o_c = r.origin() - center;
		float a = dot(r.direction(), r.direction());
		float b = 2.0*dot(r.direction(), o_c);
		float c = dot(o_c, o_c) - radius * radius;
		float discr = b * b - 4*a*c;
		if (discr > 0) {
			float temp = (-b - sqrt(b*b - 4*a * c)) / (2*a);
			if (temp < tmax && temp > tmin) {
				//set the hit record variables
				rec.t = temp;
				rec.p = r[rec.t];
				rec.normal = (rec.p - center) / radius;
				rec.mat_p = mat;
				return true;
			}
			temp = -b + sqrt(b*b - 4*a * c) / (2 * a);
			if (temp < tmax && temp > tmin) {
				//set the hit record variables
				rec.t = temp;
				rec.p = r[rec.t];
				rec.normal = (rec.p - center) / radius;
				rec.mat_p = mat;
				return true;
			}
		}
		return false;
	}
	vec3 center;
	float radius;
	Material* mat;
};

class Cube : public hitable {
	public:
	__device__ Cube(const vec3 &min, const vec3 &max, Material *m){
		//these are the corners of the cube
		min_bound = min;
		max_bound = max;
		mat = m;
	}
	__device__ virtual bool hit(const Ray& r, float tmin, float tmax, HitRecord& rec) const{
		float tx0, tx1, ty0, ty1, tz0, tz1;
		//determine if for some t, the ray A + Bt intersects the 
		//min_bound = A + Bt
		//
		float divx = 1/(r.direction().x());
		float divy = 1/(r.direction().y());
		float divz = 1/(r.direction().z());
		if(divx >= 0.0f){
			
			tx0 = (min_bound.x() - r.origin().x())*divx;
			tx1 = (max_bound.x() - r.origin().x())*divx;
		}
		else{
			tx0 = (max_bound.x() - r.origin().x())*divx;
			tx1 = (min_bound.x() - r.origin().x())*divx;
		}
		if(divy >= 0.0f){
			ty0 = (min_bound.y() - r.origin().y())*divy;
			ty1 = (max_bound.y() - r.origin().y())*divy;
		}
		else{
			ty1 = (min_bound.y() - r.origin().y())*divy;
			ty0 = (max_bound.y() - r.origin().y())*divy;
		}
		if(tx0 > ty1 || ty0 > tx1){
			return false;
		}
		if(ty0 > tx0){
			tx0 = ty0;
		}
		if(ty1 < tx1){
			tx1 = ty1;
		}
		if(divz >= 0.0f){
			tz0 = (min_bound.z() - r.origin().z())*divz;
			tz1 = (max_bound.z() - r.origin().z())*divz;
		}
		else{
			tz1 = (min_bound.z() - r.origin().z())*divz;
			tz0 = (max_bound.z() - r.origin().z())*divz;
		}
		if(tx0 > tz1 || tz0 > tx1){
			return false;
		}
		if(tz0 > tx0){
			tx0 = tz0;
		}
		if(tz1 < tx1){
			tx1 = tz1;
		}
		if ( (tx0 < tmax ) && (tx1 > tmin)){
			if (min(tx0,tx1) < tmax && min(tx0,tx1) > tmin) {
					//set the hit record variables
					rec.t = min(tx0,tx1);
					rec.p = r[rec.t];
					vec3 temp = (rec.p - vec3((min_bound.x() + max_bound.x())/2,(min_bound.y() + max_bound.y())/2,(min_bound.z() + max_bound.z())/2));
					vec3 d = vec3(abs(min_bound.x() - max_bound.x())/2,abs(min_bound.y() - max_bound.y())/2,abs(min_bound.z() - max_bound.z())/2);
					rec.normal = vec3(int(temp.x()/d.x()),int(temp.y()/d.y()),int(temp.z()/d.z()));
					rec.mat_p = mat;
					return true;
			}
			if (max(tx0,tx1) < tmax && max(tx0,tx1) > tmin) {
					//set the hit record variables
					rec.t = max(tx0,tx1);
					rec.p = r[rec.t];
					vec3 temp = (rec.p - vec3((min_bound.x() + max_bound.x())/2,(min_bound.y() + max_bound.y())/2,(min_bound.z() + max_bound.z())/2));
					vec3 d = vec3(abs(min_bound.x() - max_bound.x())/2,abs(min_bound.y() - max_bound.y())/2,abs(min_bound.z() - max_bound.z())/2);
					rec.normal = vec3(float(int(temp.x()/d.x())),float(int(temp.y()/d.y())),float(int(temp.z()/d.z())));
					rec.mat_p = mat;
					return true;
			}
		}
		return false;
	}
	//min corner of the cube
	vec3 min_bound;
	//max corner of the cube
	vec3 max_bound;
	//material of the cube
	Material *mat;

};

// class Hyperboloid : public hitable{
	// __device__ Hyperboloid() {}
	// __device__ Hyperboloid(vec3 cen, float rr, Material* m){
		// c = cen;
		// r = rr;
		// mat = m;
	// }
	// __device__ virtual bool hit(const Ray& ray, float tmin, float tmax, HitRecord& rec) const{
		// //A = -1
		// //B = 1
		// //C = 1
		
	
	// }
	// vec3 c; //center
	// float r; //radius 
	// Material *mat; //material

// }

class Sphube : public hitable {
public:
	__device__ Sphube() {}
	__device__ Sphube(vec3 cen, float rr, float ss, Material* m){
			_c = cen;
			mat = m;
			float r = rr;
			tau = ss;
			s = ss;

			_r = rr;
	}

	__device__ virtual bool hit(const Ray& ray, float tmin, float tmax, HitRecord& rec) const{
		float r = _r;
		vec3 max_bound(_c.x() + r,_c.y() + r,_c.z() + r);
		vec3 min_bound(_c.x() - r,_c.y() - r,_c.z() - r);
		float tx0, tx1, ty0, ty1, tz0, tz1;
		//determine if for some t, the ray A + Bt intersects the 
		//min_bound = A + Bt
		//
		float divx = 1/(ray.direction().x());
		float divy = 1/(ray.direction().y());
		float divz = 1/(ray.direction().z());
		if(divx >= 0.0f){
			
			tx0 = (min_bound.x() - ray.origin().x())*divx;
			tx1 = (max_bound.x() - ray.origin().x())*divx;
		}
		else{
			tx0 = (max_bound.x() - ray.origin().x())*divx;
			tx1 = (min_bound.x() - ray.origin().x())*divx;
		}
		if(divy >= 0.0f){
			ty0 = (min_bound.y() - ray.origin().y())*divy;
			ty1 = (max_bound.y() - ray.origin().y())*divy;
		}
		else{
			ty1 = (min_bound.y() - ray.origin().y())*divy;
			ty0 = (max_bound.y() - ray.origin().y())*divy;
		}
		if(tx0 > ty1 || ty0 > tx1){
			return false;
		}
		if(ty0 > tx0){
			tx0 = ty0;
		}
		if(ty1 < tx1){
			tx1 = ty1;
		}
		if(divz >= 0.0f){
			tz0 = (min_bound.z() - ray.origin().z())*divz;
			tz1 = (max_bound.z() - ray.origin().z())*divz;
		}
		else{
			tz1 = (min_bound.z() - ray.origin().z())*divz;
			tz0 = (max_bound.z() - ray.origin().z())*divz;
		}
		if(tx0 > tz1 || tz0 > tx1){
			return false;
		}
		if(tz0 > tx0){
			tx0 = tz0;
		}
		if(tz1 < tx1){
			tx1 = tz1;
		}
		if ( (tx0 < tmax ) && (tx1 > tmin)){
			float a1 =  -_c.x() + ray.origin().x();//- _c.x());//- _c.x());
			float a2 =  -_c.y() + ray.origin().y();//- _c.y());//- _c.y());
			float a3 =  -_c.z() + ray.origin().z();//- _c.z());//- _c.z());
			vec3 dir = ray.direction(); 
			float dir_length = ray.direction().length();
			//dir = unitize(ray.direction());
			float b1 = (float)(dir.x());//ray.direction().x());
			float b2 = (float)(dir.y());//ray.direction().y());
			float b3 = (float)(dir.z());//ray.direction().z());
			float div = 1/(r*r);
			float div2 = 1/(r*r*r*r);
			float x0 = min(tx0,tx1);//min(tx0,tx1);//v2.length()- (v2 - ray.origin()).length();
			float x1 = -FLT_MAX;
			bool first = true;
			int MAX_ITERATIONS = 100;
			int NUM_INTERATIONS = 0;
			//first guess is r
			//x1 stores first root
			while(NUM_INTERATIONS < MAX_ITERATIONS){
				first ? first = false : x0 = x1;
				x1 =  newtonsmethod(x0,a1,a2,a3,b1,b2,b3,s,r);
				NUM_INTERATIONS++;
			}
			//x5 is estimate of first root. y5 is estimate of second root.
			float y0 = max(tx0,tx1);//min(tx0,tx1);//v2.length()- (v2 - ray.origin()).length();
			float y1 = -FLT_MAX;
			first = true;
			NUM_INTERATIONS = 0;
			//first guess is r
			//x1 stores first root
			while(NUM_INTERATIONS < MAX_ITERATIONS){
				first ? first = false : y0 = y1;
				y1 = newtonsmethod(y0,a1,a2,a3,b1,b2,b3,s,r);
				NUM_INTERATIONS++;
			}
			vec3 sol1 = (ray[x1] - _c);
			vec3 sol2 = (ray[y1] - _c);
			float zero1 = (a1 + b1*x1)*(a1 + b1*x1) + (a2 + b2*x1)*(a2 + b2*x1) + (a3 + b3*x1)*(a3 + b3*x1) - (r*r) 
				- (s*s*(a1 + b1*x1)*(a1 + b1*x1)*(a2 + b2*x1)*(a2 + b2*x1))*div 
				- (s*s*(a1 + b1*x1)*(a1 + b1*x1)*(a3 + b3*x1)*(a3 + b3*x1))*div 
				- (s*s*(a2 + b2*x1)*(a2 + b2*x1)*(a3 + b3*x1)*(a3 + b3*x1))*div 
				+ ((s*s*s*s)*(a1 + b1*x1)*(a1 + b1*x1)*(a2 + b2*x1)*(a2 + b2*x1)*(a3 + b3*x1)*(a3 + b3*x1))*div2;
			float zero2 = (a1 + b1*y1)*(a1 + b1*y1) + (a2 + b2*y1)*(a2 + b2*y1) + (a3 + b3*y1)*(a3 + b3*y1) - (r*r) 
				- (s*s*(a1 + b1*y1)*(a1 + b1*y1)*(a2 + b2*y1)*(a2 + b2*y1))*div 
				- (s*s*(a1 + b1*y1)*(a1 + b1*y1)*(a3 + b3*y1)*(a3 + b3*y1))*div 
				- (s*s*(a2 + b2*y1)*(a2 + b2*y1)*(a3 + b3*y1)*(a3 + b3*y1))*div 
				+ ((s*s*s*s)*(a1 + b1*y1)*(a1 + b1*y1)*(a2 + b2*y1)*(a2 + b2*y1)*(a3 + b3*y1)*(a3 + b3*y1))*div2;
			float zero;
			vec3 sol;
			if((sol1 - ray.origin()).length() < (sol2 - ray.origin()).length()){
				sol = sol1;
				zero = zero1;
				
			} else { sol = sol2; zero = zero2; }
			if ( min(x1,y1) < tmax && min(x1,y1) > tmin && sol.length() < 0.5f*(min_bound-max_bound).length()*(s) + r*(1-s) && abs(zero) < .001f) {
				//set the hit record variables
				rec.t = min(x1,y1);
				rec.p = ray[min(x1,y1)];
				float x = (rec.p.x() - _c.x());
				float y = (rec.p.y() - _c.y());
				float z = (rec.p.z() - _c.z());
				float dfdx = (2.0f*x - (2.0f*s*s*x*y*y)*div - (2.0f*s*s*x*z*z)*div + (2.0f*s*s*s*s*x*y*y*z*z)*div2);
				float dfdy = (2.0f*y - (2.0f*s*s*y*x*x)*div - (2.0f*s*s*y*z*z)*div + (2.0f*s*s*s*s*y*x*x*z*z)*div2);
				float dfdz = (2.0f*z - (2.0f*s*s*z*x*x)*div - (2.0f*s*s*z*y*y)*div + (2.0f*s*s*s*s*z*y*y*x*x)*div2);		
				vec3 n(dfdx,dfdy,dfdz);
				rec.normal = unitize(n);
				rec.mat_p = mat;
				return true;
			}
			if (max(x1,y1) < tmax && max(x1,y1) > tmin && sol.length() < 0.5f*(min_bound-max_bound).length()*(s) + r*(1-s) && abs(zero) < .001f){
				//set the hit record variables
				rec.t = max(x1,y1);
				rec.p = ray[max(x1,y1)];
				float x = (rec.p.x() - _c.x());
				float y = (rec.p.y() - _c.y());
				float z = (rec.p.z() - _c.z());
				float dfdx = (2.0f*x - (2.0f*s*s*x*y*y)*div - (2.0f*s*s*x*z*z)*div + (2.0f*s*s*s*s*x*y*y*z*z)*div2);
				float dfdy = (2.0f*y - (2.0f*s*s*y*x*x)*div - (2.0f*s*s*y*z*z)*div + (2.0f*s*s*s*s*y*x*x*z*z)*div2);
				float dfdz = (2.0f*z - (2.0f*s*s*z*x*x)*div - (2.0f*s*s*z*y*y)*div + (2.0f*s*s*s*s*z*y*y*x*x)*div2);
				vec3 n(dfdx,dfdy,dfdz);
				n = vec3(samesign(n.x(),rec.p.x() - _c.x()) ? n.x() : -1.0f*n.x(),samesign(n.y(),rec.p.y() - _c.y()) ? n.y() : -1.0f*n.y(),samesign(n.z(),rec.p.z() - _c.z()) ? n.z() : -1.0f*n.z());
				rec.normal = unitize(n);
				rec.mat_p = mat;
				return true;
			}
		}
		return false;
	}
	//center of the sphube
	vec3 _c;
	//squircularity of the sphube
	float s;
	//radius of the sphube
	float _r;
	float tau;
	//material the sphube (determines how rays interract)
	Material* mat;
};
