#ifndef COMPUTE_H
#define COMPUTE_H

#include <GL/gl.h>
#include <GL/freeglut.h>
#include <vector_types.h> // dim3
#include "data_structure.h"

class Compute {

 public:

  void Init(dataInfo_t* dataInfo);
  void RegisterVolumeTexture(GLuint volume);
  void UnregisterVolumeTexture();
  void Update();
  void InitDye();
  void InitVelocity();

 private:

  int NumCellFaces[3];
  int Res[3][3];
  dim3 VolumeSize;
  dataInfo_t *DataInfo;
  float *VelocityX;
  float *VelocityY;
  float *VelocityZ;
  float *Dye;
  float *Divergence;
  float *Pressure;

  cudaArray *ca_VelocityX;
  cudaArray *ca_VelocityY;
  cudaArray *ca_VelocityZ;
  cudaArray *ca_Dye;

  struct cudaGraphicsResource *VolumeResource;

  void InitData();
  void InitTextures();
  void InitSymbols();
  float ComputeTimestep();
  void SetTimestep(float dt);
  void AdvectDye();
  void AdvectVelocity();
  void SetBoundaryConditions();
  void ComputeDivergence();
  void PressureUpdate();

  //kernel wrappers-------------------------------------------------------------
  void ComputeTimestep_kernel(float* data);
  void InitDye_kernel();
  void InitVelocity_kernel();
  void AdvectDye_kernel();
  void AdvectVelocity_kernel();
  void SetBoundaryConditions_kernel();
  void ComputeDivergence_kernel();
  void PressureUpdate_kernel();

  //utilities-------------------------------------------------------------------
  void UpdateCudaArray(cudaArray* ca, int res[3], float* src);
  void CopyCudaArray(cudaArray* ca, int res[3], float* dst);
};

#endif//COMPUTE_H
