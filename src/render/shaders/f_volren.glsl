#version 330

uniform sampler3D volumeTex;
uniform mat4 modelInvTranspMat;
uniform vec3 ratio;
uniform vec2 imageScale;

out vec4 out_color;

struct ray_t {
  vec3 org;
  vec3 dir;
};
//==============================================================================
// intersect ray with a box
// http://www.siggraph.org/education/materials/HyperGraph/raytrace/rtinter3.htm
void intersectBox(in ray_t ray, in vec3 boxmin, in vec3 boxmax, 
		  out float tnear, out float tfar, out int intersect)
{
  // compute intersection of ray with all six bbox planes
  vec3 invR = vec3(1.0) / ray.dir;
  vec3 tbot = invR * (boxmin - ray.org);
  vec3 ttop = invR * (boxmax - ray.org);

  // re-order intersections to find smallest and largest on each axis
  vec3 tmin = min(ttop, tbot);
  vec3 tmax = max(ttop, tbot);

  // find the largest tmin and the smallest tmax
  float largest_tmin = max(max(tmin.x, tmin.y), max(tmin.x, tmin.z));
  float smallest_tmax = min(min(tmax.x, tmax.y), min(tmax.x, tmax.z));

  tnear = largest_tmin;
  tfar = smallest_tmax;
  intersect = smallest_tmax > largest_tmin ? 1 : 0;
}
//==============================================================================
void main(void)
{
  vec2 tex2DCoord = gl_FragCoord.xy*imageScale.xy; // [0:1]
  float imageRatio = imageScale.y / imageScale.x;
  vec2 imageCoord = tex2DCoord*2.0 - 1.0; // [-1:1]

  if (imageRatio > 1.0) // width > height
    imageCoord.x = imageCoord.x*imageRatio;
  else // width < height
    imageCoord.y = imageCoord.y/imageRatio;

  ray_t ray;
  ray.org = vec4(vec4(0.0, 0.0, 0.0, 1.0)*modelInvTranspMat).xyz;
  ray.dir = normalize(vec3(imageCoord.x, imageCoord.y, -2.0));
  ray.dir = normalize(vec4(vec4(ray.dir, 0.0)*modelInvTranspMat).xyz);

  int intersect;
  float tnear, tfar;
  intersectBox(ray, -ratio, ratio, tnear, tfar, intersect);

  vec4 sum_color = vec4(0.0);

  if (intersect == 1) {

    if (tnear < 0.0) {
      tnear = 0.0;
    }
    float t = tnear;
    vec3 volCoord = ray.org + ray.dir*tnear;//[-1,-1,-1]:[1,1,1]
    float tstep = 0.01;
    vec3 step = ray.dir*tstep;
    int maxNumSteps = 500;

    for(int i = 0; i < maxNumSteps; i++) {

      vec3 tex3DCoord = volCoord/ratio*0.5 + 0.5;//[0,0,0]:[1,1,1]
      vec4 color = vec4(texture(volumeTex, tex3DCoord).x);

      color.x *= color.w;
      color.y *= color.w;
      color.z *= color.w;

      sum_color = sum_color + color*(1.0f - sum_color.w);

      if (sum_color.w > 0.95)
    	break;

      t += tstep;
      if (t > tfar)
    	break;

      volCoord += step;
    }
    // make the background white
    sum_color = sum_color + vec4(1.0)*(1.0 - sum_color.w);    
  }
  else {
    sum_color = vec4(1.0);
  }
  out_color = sum_color;
}
