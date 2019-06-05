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
			float a1 = (float)(ray.origin().x() - _c.x());
			float a2 = (float)(ray.origin().y() - _c.y());
			float a3 = (float)(ray.origin().z() - _c.z());
			float b1 = (float)(ray.direction().x());
			float b2 = (float)(ray.direction().y());
			float b3 = (float)(ray.direction().z());
			float div = (float)(1/(r*r));
			float t6 = div*(((b1*b1)*(b2*b2)*(b3*b3)*(s*s*s*s))/(r*r*r*r));
			float t5 = div*((b3*b3)*((2*a1*b1*(b2*b2)*(s*s*s*s))/(r*r*r*r) + (2*a2*(b1*b1)*b2*(s*s*s*s))/(r*r*r*r)) + (2*a3*(b1*b1)*(b2*b2)*b3*(s*s*s*s))/(r*r*r*r));
			float t4 = div*((b3*b3)*(((a1*a1)*(b2*b2)*(s*s*s*s))/(r*r*r*r) + ((a2*a2)*(b1*b1)*(s*s*s*s))/(r*r*r*r) + (4*a1*a2*b1*b2*(s*s*s*s))/(r*r*r*r)) + 2*a3*b3*((2*a1*b1*(b2*b2)*(s*s*s*s))/(r*r*r*r) + (2*a2*(b1*b1)*b2*(s*s*s*s))/(r*r*r*r)) - ((b1*b1)*(b2*b2)*(s*s))/(r*r) - ((b1*b1)*(b3*b3)*(s*s))/(r*r) - ((b2*b2)*(b3*b3)*(s*s))/(r*r) + ((a3*a3)*(b1*b1)*(b2*b2)*(s*s*s*s))/(r*r*r*r));
			float t3 = div*((b3*b3)*((2*a1*(a2*a2)*b1*(s*s*s*s))/(r*r*r*r) + (2*(a1*a1)*a2*b2*(s*s*s*s))/(r*r*r*r)) + (a3*a3)*((2*a1*b1*(b2*b2)*(s*s*s*s))/(r*r*r*r) + (2*a2*(b1*b1)*b2*(s*s*s*s))/(r*r*r*r)) + 2*a3*b3*(((a1*a1)*(b2*b2)*(s*s*s*s))/(r*r*r*r) + ((a2*a2)*(b1*b1)*(s*s*s*s))/(r*r*r*r) + (4*a1*a2*b1*b2*(s*s*s*s))/(r*r*r*r)) - (2*a1*b1*(b2*b2)*(s*s))/(r*r) - (2*a1*b1*(b3*b3)*(s*s))/(r*r) - (2*a2*(b1*b1)*b2*(s*s))/(r*r) - (2*a2*b2*(b3*b3)*(s*s))/(r*r) - (2*a3*(b1*b1)*b3*(s*s))/(r*r) - (2*a3*(b2*b2)*b3*(s*s))/(r*r));
			float t2 = div*((a3*a3)*(((a1*a1)*(b2*b2)*(s*s*s*s))/(r*r*r*r) + ((a2*a2)*(b1*b1)*(s*s*s*s))/(r*r*r*r) + (4*a1*a2*b1*b2*(s*s*s*s))/(r*r*r*r)) + (b1*b1) + (b2*b2) + (b3*b3) + 2*a3*b3*((2*a1*(a2*a2)*b1*(s*s*s*s))/(r*r*r*r) + (2*(a1*a1)*a2*b2*(s*s*s*s))/(r*r*r*r)) - ((a1*a1)*(b2*b2)*(s*s))/(r*r) - ((a2*a2)*(b1*b1)*(s*s))/(r*r) - ((a1*a1)*(b3*b3)*(s*s))/(r*r) - ((a3*a3)*(b1*b1)*(s*s))/(r*r) - ((a2*a2)*(b3*b3)*(s*s))/(r*r) - ((a3*a3)*(b2*b2)*(s*s))/(r*r) + ((a1*a1)*(a2*a2)*(b3*b3)*(s*s*s*s))/(r*r*r*r) - (4*a1*a2*b1*b2*(s*s))/(r*r) - (4*a1*a3*b1*b3*(s*s))/(r*r) - (4*a2*a3*b2*b3*(s*s))/(r*r));
			float t1 = div*(2*a1*b1 + 2*a2*b2 + 2*a3*b3 + (a3*a3)*((2*a1*(a2*a2)*b1*(s*s*s*s))/(r*r*r*r) + (2*(a1*a1)*a2*b2*(s*s*s*s))/(r*r*r*r)) - (2*a1*(a2*a2)*b1*(s*s))/(r*r) - (2*a1*(a3*a3)*b1*(s*s))/(r*r) - (2*(a1*a1)*a2*b2*(s*s))/(r*r) - (2*a2*(a3*a3)*b2*(s*s))/(r*r) - (2*(a1*a1)*a3*b3*(s*s))/(r*r) - (2*(a2*a2)*a3*b3*(s*s))/(r*r) + (2*(a1*a1)*(a2*a2)*a3*b3*(s*s*s*s))/(r*r*r*r));
			float t0 = div*((a1*a1) + (a2*a2) + (a3*a3) - r*r - ((a1*a1)*(a2*a2)*(s*s))/(r*r) - ((a1*a1)*(a3*a3)*(s*s))/(r*r) - ((a2*a2)*(a3*a3)*(s*s))/(r*r) + ((a1*a1)*(a2*a2)*(a3*a3)*(s*s*s*s))/(r*r*r*r));
			
			float tp5 = div*((6*(b1*b1)*(b2*b2)*(b3*b3)*(s*s*s*s))/(r*r*r*r));
			float tp4 = div*(5*(b3*b3)*((2*a1*b1*(b2*b2)*(s*s*s*s))/(r*r*r*r) + (2*a2*(b1*b1)*b2*(s*s*s*s))/(r*r*r*r)) + (10*a3*(b1*b1)*(b2*b2)*b3*(s*s*s*s))/(r*r*r*r));
			float tp3 = div*(4*(b3*b3)*(((a1*a1)*(b2*b2)*(s*s*s*s))/(r*r*r*r) + ((a2*a2)*(b1*b1)*(s*s*s*s))/(r*r*r*r) + (4*a1*a2*b1*b2*(s*s*s*s))/(r*r*r*r)) + 8*a3*b3*((2*a1*b1*(b2*b2)*(s*s*s*s))/(r*r*r*r) + (2*a2*(b1*b1)*b2*(s*s*s*s))/(r*r*r*r)) - (4*(b1*b1)*(b2*b2)*(s*s))/(r*r) - (4*(b1*b1)*(b3*b3)*(s*s))/(r*r) - (4*(b2*b2)*(b3*b3)*(s*s))/(r*r) + (4*(a3*a3)*(b1*b1)*(b2*b2)*(s*s*s*s))/(r*r*r*r));
			float tp2 = div*(3*(b3*b3)*((2*a1*(a2*a2)*b1*(s*s*s*s))/(r*r*r*r) + (2*(a1*a1)*a2*b2*(s*s*s*s))/(r*r*r*r)) + 3*(a3*a3)*((2*a1*b1*(b2*b2)*(s*s*s*s))/(r*r*r*r) + (2*a2*(b1*b1)*b2*(s*s*s*s))/(r*r*r*r)) + 6*a3*b3*(((a1*a1)*(b2*b2)*(s*s*s*s))/(r*r*r*r) + ((a2*a2)*(b1*b1)*(s*s*s*s))/(r*r*r*r) + (4*a1*a2*b1*b2*(s*s*s*s))/(r*r*r*r)) - (6*a1*b1*(b2*b2)*(s*s))/(r*r) - (6*a1*b1*(b3*b3)*(s*s))/(r*r) - (6*a2*(b1*b1)*b2*(s*s))/(r*r) - (6*a2*b2*(b3*b3)*(s*s))/(r*r) - (6*a3*(b1*b1)*b3*(s*s))/(r*r) - (6*a3*(b2*b2)*b3*(s*s))/(r*r));
			float tp1 = div*(2*(a3*a3)*(((a1*a1)*(b2*b2)*(s*s*s*s))/(r*r*r*r) + ((a2*a2)*(b1*b1)*(s*s*s*s))/(r*r*r*r) + (4*a1*a2*b1*b2*(s*s*s*s))/(r*r*r*r)) + 2*(b1*b1) + 2*(b2*b2) + 2*(b3*b3) + 4*a3*b3*((2*a1*(a2*a2)*b1*(s*s*s*s))/(r*r*r*r) + (2*(a1*a1)*a2*b2*(s*s*s*s))/(r*r*r*r)) - (2*(a1*a1)*(b2*b2)*(s*s))/(r*r) - (2*(a2*a2)*(b1*b1)*(s*s))/(r*r) - (2*(a1*a1)*(b3*b3)*(s*s))/(r*r) - (2*(a3*a3)*(b1*b1)*(s*s))/(r*r) - (2*(a2*a2)*(b3*b3)*(s*s))/(r*r) - (2*(a3*a3)*(b2*b2)*(s*s))/(r*r) + (2*(a1*a1)*(a2*a2)*(b3*b3)*(s*s*s*s))/(r*r*r*r) - (8*a1*a2*b1*b2*(s*s))/(r*r) - (8*a1*a3*b1*b3*(s*s))/(r*r) - (8*a2*a3*b2*b3*(s*s))/(r*r));
			float tp0 = div*(2*a1*b1 + 2*a2*b2 + 2*a3*b3 + (a3*a3)*((2*a1*(a2*a2)*b1*(s*s*s*s))/(r*r*r*r) + (2*(a1*a1)*a2*b2*(s*s*s*s))/(r*r*r*r)) - (2*a1*(a2*a2)*b1*(s*s))/(r*r) - (2*a1*(a3*a3)*b1*(s*s))/(r*r) - (2*(a1*a1)*a2*b2*(s*s))/(r*r) - (2*a2*(a3*a3)*b2*(s*s))/(r*r) - (2*(a1*a1)*a3*b3*(s*s))/(r*r) - (2*(a2*a2)*a3*b3*(s*s))/(r*r) + (2*(a1*a1)*(a2*a2)*a3*b3*(s*s*s*s))/(r*r*r*r));
			float coefs[7] = { t0, t1, t2, t3, t4, t5, t6 };
			float coefsp[6] = {tp0,tp1,tp2,tp3,tp4,tp5};
			
			vec3 v1 = _c - ray.origin();
			vec3 v2 = ((dot(v1,ray.direction())/(ray.direction().length()*ray.direction().length())))*ray.direction();
			
			//x5 is estimate of first root. y5 is estimate of second root.
			float x0 = tx0;//v2.length()- (v2 - ray.origin()).length();
			float x1 = -FLT_MAX;
			bool first = true;
			int MAX_ITERATIONS = 100;
			int NUM_INTERATIONS = 0;
			//first guess is r
			//x1 stores first root
			while((abs(x1 - x0) > .00001f) && NUM_INTERATIONS < MAX_ITERATIONS){
				first ? first = false : x0 = x1;
				x1 = newtonsmethod(coefs,coefsp,.99f*x0);
				NUM_INTERATIONS++;
			}
			//x1*=1.001f;
			first = true;
			float y0 = tx1;//v2.length() + (v2 - ray.origin()).length();
			float y1 = FLT_MAX;
			NUM_INTERATIONS = 0;
			//first guess is -r
			//y1 stores second root
			while((abs(y1 - y0) > .00001f) && NUM_INTERATIONS < MAX_ITERATIONS){
				first ? first = false : y0 = y1;
				y1 = newtonsmethod(coefs,coefsp,.99f*y0);
				NUM_INTERATIONS++;
			}
			//y1 *= 1.001f;
			vec3 sol1 = ray[x1]- _c;
			vec3 sol2 = ray[y1]- _c;
			if(r < 1.0f){
				if(sol1.length() < 1.0001f*(1/r)*(float)sqrt(2.0f) && sol2.length() < 1.0001f*(1/r)*(float)sqrt(2.0f)){
					if (x1 < tmax && x1 > tmin) {
							//set the hit record variables
							rec.t = x1;
							rec.p = ray[rec.t];
							float dfdx = 2.0f*rec.p.x() + (2.0f*s*s*rec.p.x()*rec.p.y()*rec.p.y())/(r*r) + (2.0f*s*s*rec.p.x()*rec.p.z()*rec.p.z())/(r*r) + (2.0f*s*s*s*s*rec.p.x()*rec.p.y()*rec.p.y()*rec.p.z()*rec.p.z())/(r*r*r*r);
							float dfdy = 2.0f*rec.p.y() + (2.0f*s*s*rec.p.y()*rec.p.x()*rec.p.x())/(r*r) + (2.0f*s*s*rec.p.y()*rec.p.z()*rec.p.z())/(r*r) + (2.0f*s*s*s*s*rec.p.y()*rec.p.x()*rec.p.x()*rec.p.z()*rec.p.z())/(r*r*r*r);
							float dfdz = 2.0f*rec.p.z() + (2.0f*s*s*rec.p.z()*rec.p.x()*rec.p.x())/(r*r) + (2.0f*s*s*rec.p.z()*rec.p.y()*rec.p.y())/(r*r) + (2.0f*s*s*s*s*rec.p.y()*rec.p.z()*rec.p.x()*rec.p.y()*rec.p.y())/(r*r*r*r);
							vec3 n(1.01f*dfdx,1.01f*dfdy,1.01f*dfdz);
							n = vec3(samesign(n.x(),rec.p.x() - _c.x()) ? n.x() : -1.0f*n.x(),samesign(n.y(),rec.p.y() - _c.y()) ? n.y() : -1.0f*n.y(),samesign(n.z(),rec.p.z() - _c.z()) ? n.z() : -1.0f*n.z());
							rec.normal = unitize(n);
							rec.mat_p = mat;
							return true;
					}
					if (y1 < tmax && y1 > tmin) {
							//set the hit record variables
							rec.t = y1;
							rec.p = ray[rec.t];
							float dfdx = 2.0f*rec.p.x() + (2.0f*s*s*rec.p.x()*rec.p.y()*rec.p.y())/(r*r) + (2.0f*s*s*rec.p.x()*rec.p.z()*rec.p.z())/(r*r) + (2.0f*s*s*s*s*rec.p.x()*rec.p.y()*rec.p.y()*rec.p.z()*rec.p.z())/(r*r*r*r);
							float dfdy = 2.0f*rec.p.y() + (2.0f*s*s*rec.p.y()*rec.p.x()*rec.p.x())/(r*r) + (2.0f*s*s*rec.p.y()*rec.p.z()*rec.p.z())/(r*r) + (2.0f*s*s*s*s*rec.p.y()*rec.p.x()*rec.p.x()*rec.p.z()*rec.p.z())/(r*r*r*r);
							float dfdz = 2.0f*rec.p.z() + (2.0f*s*s*rec.p.z()*rec.p.x()*rec.p.x())/(r*r) + (2.0f*s*s*rec.p.z()*rec.p.y()*rec.p.y())/(r*r) + (2.0f*s*s*s*s*rec.p.y()*rec.p.z()*rec.p.x()*rec.p.y()*rec.p.y())/(r*r*r*r);
							vec3 n(1.0001f*dfdx,1.0001f*dfdy,1.0001f*dfdz);
							n = vec3(samesign(n.x(),rec.p.x() - _c.x()) ? n.x() : -1.0f*n.x(),samesign(n.y(),rec.p.y() - _c.y()) ? n.y() : -1.0f*n.y(),samesign(n.z(),rec.p.z() - _c.z()) ? n.z() : -1.0f*n.z());
							rec.normal = unitize(n);
							rec.mat_p = mat;
							return true;
					}
				}
				return false;	
			}
			if(sol1.length() < 1.0001f*r*(float)sqrt(2.0f) && sol2.length() < 1.0001f*r*(float)sqrt(2.0f)){
				if (x1 < tmax && x1 > tmin) {
						//set the hit record variables
						rec.t = x1;
						rec.p = ray[rec.t];
						float dfdx = 2.0f*rec.p.x() + (2.0f*s*s*rec.p.x()*rec.p.y()*rec.p.y())/(r*r) + (2.0f*s*s*rec.p.x()*rec.p.z()*rec.p.z())/(r*r) + (2.0f*s*s*s*s*rec.p.x()*rec.p.y()*rec.p.y()*rec.p.z()*rec.p.z())/(r*r*r*r);
						float dfdy = 2.0f*rec.p.y() + (2.0f*s*s*rec.p.y()*rec.p.x()*rec.p.x())/(r*r) + (2.0f*s*s*rec.p.y()*rec.p.z()*rec.p.z())/(r*r) + (2.0f*s*s*s*s*rec.p.y()*rec.p.x()*rec.p.x()*rec.p.z()*rec.p.z())/(r*r*r*r);
						float dfdz = 2.0f*rec.p.z() + (2.0f*s*s*rec.p.z()*rec.p.x()*rec.p.x())/(r*r) + (2.0f*s*s*rec.p.z()*rec.p.y()*rec.p.y())/(r*r) + (2.0f*s*s*s*s*rec.p.y()*rec.p.z()*rec.p.x()*rec.p.y()*rec.p.y())/(r*r*r*r);
						vec3 n(1.01f*dfdx,1.01f*dfdy,1.01f*dfdz);
						n = vec3(samesign(n.x(),rec.p.x() - _c.x()) ? n.x() : -1.0f*n.x(),samesign(n.y(),rec.p.y() - _c.y()) ? n.y() : -1.0f*n.y(),samesign(n.z(),rec.p.z() - _c.z()) ? n.z() : -1.0f*n.z());
						rec.normal = unitize(n);
						rec.mat_p = mat;
						return true;
				}
				if (y1 < tmax && y1 > tmin) {
						//set the hit record variables
						rec.t = y1;
						rec.p = ray[rec.t];
						float dfdx = 2.0f*rec.p.x() + (2.0f*s*s*rec.p.x()*rec.p.y()*rec.p.y())/(r*r) + (2.0f*s*s*rec.p.x()*rec.p.z()*rec.p.z())/(r*r) + (2.0f*s*s*s*s*rec.p.x()*rec.p.y()*rec.p.y()*rec.p.z()*rec.p.z())/(r*r*r*r);
						float dfdy = 2.0f*rec.p.y() + (2.0f*s*s*rec.p.y()*rec.p.x()*rec.p.x())/(r*r) + (2.0f*s*s*rec.p.y()*rec.p.z()*rec.p.z())/(r*r) + (2.0f*s*s*s*s*rec.p.y()*rec.p.x()*rec.p.x()*rec.p.z()*rec.p.z())/(r*r*r*r);
						float dfdz = 2.0f*rec.p.z() + (2.0f*s*s*rec.p.z()*rec.p.x()*rec.p.x())/(r*r) + (2.0f*s*s*rec.p.z()*rec.p.y()*rec.p.y())/(r*r) + (2.0f*s*s*s*s*rec.p.y()*rec.p.z()*rec.p.x()*rec.p.y()*rec.p.y())/(r*r*r*r);
						vec3 n(1.0001f*dfdx,1.0001f*dfdy,1.0001f*dfdz);
						n = vec3(samesign(n.x(),rec.p.x() - _c.x()) ? n.x() : -1.0f*n.x(),samesign(n.y(),rec.p.y() - _c.y()) ? n.y() : -1.0f*n.y(),samesign(n.z(),rec.p.z() - _c.z()) ? n.z() : -1.0f*n.z());
						rec.normal = unitize(n);
						rec.mat_p = mat;
						return true;
				}
			}
			return false;	
		}
		return false;
		//vector from origin to center of the sphube
		vec3 o_c = ray.origin() - _c;
		//first transform the ray to the basis of the sphube in which the sphube is a sphere

	}
	//center of the sphube
	vec3 _c;
	//squircularity of the sphube
	float s;
	//radius of the sphube
	float _r;
	//material the sphube (determines how rays interract)
	Material* mat;
};
