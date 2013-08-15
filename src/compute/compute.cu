#include "compute.h"
#include "helper_cuda.h"
#include <iostream>
#include <cuda_gl_interop.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/extrema.h>
#include <thrust/transform_reduce.h>
#include <thrust/functional.h>

//==============================================================================
void Compute::Init(dataInfo_t* dataInfo)
{
  cudaGLSetGLDevice(0);
  DataInfo = dataInfo;

  NumCellFaces[0] = 
    (DataInfo->resolution[0]+1)*
    (DataInfo->resolution[1])*
    (DataInfo->resolution[2]);
  NumCellFaces[1] = 
    (DataInfo->resolution[0])*
    (DataInfo->resolution[1]+1)*
    (DataInfo->resolution[2]);
  NumCellFaces[2] = 
    (DataInfo->resolution[0])*
    (DataInfo->resolution[1])*
    (DataInfo->resolution[2]+1);

  InitData();
  InitTextures();
  InitSymbols();

  Res[0][0] = DataInfo->resolution[0]+1;
  Res[0][1] = DataInfo->resolution[1];
  Res[0][2] = DataInfo->resolution[2];
  Res[1][0] = DataInfo->resolution[0];
  Res[1][1] = DataInfo->resolution[1]+1;
  Res[1][2] = DataInfo->resolution[2];
  Res[2][0] = DataInfo->resolution[0];
  Res[2][1] = DataInfo->resolution[1];
  Res[2][2] = DataInfo->resolution[2]+1;
  Res[3][0] = DataInfo->resolution[0];
  Res[3][1] = DataInfo->resolution[1];
  Res[3][2] = DataInfo->resolution[2];

  VolumeSize.x = DataInfo->resolution[0];
  VolumeSize.y = DataInfo->resolution[1];
  VolumeSize.z = DataInfo->resolution[2];
}
//==============================================================================
void Compute::InitData()
{
  int numCells = 
    (DataInfo->resolution[0])*
    (DataInfo->resolution[1])*
    (DataInfo->resolution[2]);
  cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float>();
  int res[3] = {DataInfo->resolution[0],
		DataInfo->resolution[1],
		DataInfo->resolution[2]};
  cudaExtent volumeSize;

  // Dye
  myCudaCall(cudaMalloc((void**)&Dye, numCells*sizeof(float)), 
	     __LINE__, __FILE__);
  volumeSize = make_cudaExtent(res[0], res[1], res[2]);
  myCudaCall(cudaMalloc3DArray(&ca_Dye, &channelDesc, volumeSize),
	     __LINE__, __FILE__);
  // Velocity X component
  myCudaCall(cudaMalloc((void**)&VelocityX, NumCellFaces[0]*sizeof(float)), 
	     __LINE__, __FILE__);
  volumeSize = make_cudaExtent(res[0]+1, res[1], res[2]);
  myCudaCall(cudaMalloc3DArray(&ca_VelocityX, &channelDesc, volumeSize),
	     __LINE__, __FILE__);
  // Velocity Y component
  myCudaCall(cudaMalloc((void**)&VelocityY, NumCellFaces[1]*sizeof(float)), 
	     __LINE__, __FILE__);
  volumeSize = make_cudaExtent(res[0], res[1]+1, res[2]);
  myCudaCall(cudaMalloc3DArray(&ca_VelocityY, &channelDesc, volumeSize),
	     __LINE__, __FILE__);
  // Velocity Z component
  myCudaCall(cudaMalloc((void**)&VelocityZ, NumCellFaces[2]*sizeof(float)), 
	     __LINE__, __FILE__);
  volumeSize = make_cudaExtent(res[0], res[1], res[2]+1);
  myCudaCall(cudaMalloc3DArray(&ca_VelocityZ, &channelDesc, volumeSize),
	     __LINE__, __FILE__);
  // Divergence
  myCudaCall(cudaMalloc((void**)&NegDivergence, numCells*sizeof(float)), 
	     __LINE__, __FILE__);
  // Pressure
  myCudaCall(cudaMalloc((void**)&Pressure, numCells*sizeof(float)), 
	     __LINE__, __FILE__);
  // Boundary conditions
  myCudaCall(cudaMallocArray(&ca_BCLeft, &channelDesc, res[1], res[2]),
	     __LINE__, __FILE__);
  myCudaCall(cudaMallocArray(&ca_BCRight, &channelDesc, res[1], res[2]),
	     __LINE__, __FILE__);
  myCudaCall(cudaMallocArray(&ca_BCBottom, &channelDesc, res[0], res[2]),
	     __LINE__, __FILE__);
  myCudaCall(cudaMallocArray(&ca_BCTop, &channelDesc, res[0], res[2]),
	     __LINE__, __FILE__);
  myCudaCall(cudaMallocArray(&ca_BCBack, &channelDesc, res[0], res[1]),
	     __LINE__, __FILE__);
  myCudaCall(cudaMallocArray(&ca_BCFront, &channelDesc, res[0], res[1]),
	     __LINE__, __FILE__);
}
//==============================================================================
void Compute::InitDye()
{
  InitDye_kernel();

  cudaGraphicsMapResources(1, &VolumeResource);
  cudaGraphicsSubResourceGetMappedArray(&ca_Dye, VolumeResource, 0, 0);

  UpdateCudaArray(ca_Dye, DataInfo->resolution, Dye);

  cudaGraphicsUnmapResources(1, &VolumeResource);
}
//==============================================================================
void Compute::InitVelocity()
{
  InitVelocity_kernel();

  UpdateCudaArray(ca_VelocityX, Res[0], VelocityX);
  UpdateCudaArray(ca_VelocityY, Res[1], VelocityY);
  UpdateCudaArray(ca_VelocityZ, Res[2], VelocityZ);
}
//==============================================================================
void Compute::RegisterVolumeTexture(GLuint volume)
{
  cudaGraphicsGLRegisterImage(&VolumeResource, volume, GL_TEXTURE_3D, 
			      cudaGraphicsRegisterFlagsNone);
}
//==============================================================================
void Compute::UnregisterVolumeTexture()
{
  cudaGraphicsUnregisterResource(VolumeResource);
}
//==============================================================================
void Compute::AdvectDye()
{
  AdvectDye_kernel();

  cudaGraphicsMapResources(1, &VolumeResource);
  cudaGraphicsResourceSetMapFlags(VolumeResource, 
				  cudaGraphicsMapFlagsWriteDiscard);
  cudaGraphicsSubResourceGetMappedArray(&ca_Dye, VolumeResource, 0, 0);
  UpdateCudaArray(ca_Dye, DataInfo->resolution, Dye);
  cudaGraphicsUnmapResources(1, &VolumeResource);
}
//==============================================================================
void Compute::AdvectVelocity()
{
  AdvectVelocity_kernel();

  UpdateCudaArray(ca_VelocityX, Res[0], VelocityX);
  UpdateCudaArray(ca_VelocityY, Res[1], VelocityY);
  UpdateCudaArray(ca_VelocityZ, Res[2], VelocityZ);
}
//==============================================================================
void Compute::SetBoundaryConditions()
{
  SetBoundaryConditions_kernel();

  float *velobc;
  int count;

  count = Res[3][1]*Res[3][2];
  velobc = new float[count];

  for (int k = 0; k < Res[3][2]; k++) {
    for (int j = 0; j < Res[3][1]; j++) {

      int idx = j + k*Res[3][1];
      if (j > Res[3][1]/4 && j < Res[3][1]*3/4 &&
  	  k > Res[3][2]/4 && k < Res[3][2]*3/4 ) {
  	velobc[idx] = 1.0f;
      }
      else {
  	velobc[idx] = 0.0f;
      }
    }
  }  
  myCudaCall(cudaMemcpyToArray(ca_BCLeft, 0, 0, velobc, count, 
			       cudaMemcpyHostToDevice), __LINE__, __FILE__);

  for (int k = 0; k < Res[3][2]; k++) {
    for (int j = 0; j < Res[3][1]; j++) {

      int idx = j + k*Res[3][1];
      velobc[idx] = 0.0f;     
    }
  }  
  myCudaCall(cudaMemcpyToArray(ca_BCRight, 0, 0, velobc, count,
			       cudaMemcpyHostToDevice), __LINE__, __FILE__);
  delete [] velobc;

  count = Res[3][0]*Res[3][2];
  velobc = new float[count];

  for (int k = 0; k < Res[3][2]; k++) {
    for (int i = 0; i < Res[3][0]; i++) {

      int idx = i + k*Res[3][0];
      velobc[idx] = 0.0f;
    }
  }  
  myCudaCall(cudaMemcpyToArray(ca_BCBottom, 0, 0, velobc, count,
			       cudaMemcpyHostToDevice), __LINE__, __FILE__);
  myCudaCall(cudaMemcpyToArray(ca_BCTop, 0, 0, velobc, count,
			       cudaMemcpyHostToDevice), __LINE__, __FILE__);
  delete [] velobc;

  count = Res[3][0]*Res[3][1];
  velobc = new float[count];

  for (int j = 0; j < Res[3][1]; j++) {
    for (int i = 0; i < Res[3][0]; i++) {

      int idx = i + j*Res[3][0];
      velobc[idx] = 0.0f;
    }
  }  
  myCudaCall(cudaMemcpyToArray(ca_BCBottom, 0, 0, velobc, count,
			       cudaMemcpyHostToDevice), __LINE__, __FILE__);
  myCudaCall(cudaMemcpyToArray(ca_BCTop, 0, 0, velobc, count,
			       cudaMemcpyHostToDevice), __LINE__, __FILE__);
  delete [] velobc;
}
//==============================================================================
void Compute::Update()
{
  SetTimestep(ComputeTimestep());
  //  SetBoundaryConditions();
  ComputeNegDivergence();
  // //Projection();
  // PressureUpdate();
  AdvectDye();
  AdvectVelocity();
}
//==============================================================================
void Compute::UpdateCudaArray(cudaArray* ca, int res[3], float* src)
{
  cudaMemcpy3DParms copyParams[1] = {0};  
  cudaExtent volumeSize = make_cudaExtent(res[0], res[1], res[2]);
  copyParams[0].srcPtr = make_cudaPitchedPtr(src, res[0]*sizeof(float), 
					     res[0], res[1]);
  copyParams[0].dstArray = ca;
  copyParams[0].extent = volumeSize;
  copyParams[0].kind = cudaMemcpyDeviceToDevice;

  cudaMemcpy3D(copyParams);  
}
//==============================================================================
void Compute::CopyCudaArray(cudaArray* ca, int res[3], float* dst)
{
  cudaMemcpy3DParms copyParams[1] = {0};  
  cudaExtent volumeSize = make_cudaExtent(res[0], res[1], res[2]);
  copyParams[0].dstPtr = make_cudaPitchedPtr(dst, res[0]*sizeof(float), 
					     res[0], res[1]);
  copyParams[0].srcArray = ca;
  copyParams[0].extent = volumeSize;
  copyParams[0].kind = cudaMemcpyDeviceToDevice;

  cudaMemcpy3D(copyParams);  
}
//==============================================================================
void Compute::ComputeNegDivergence()
{
  ComputeNegDivergence_kernel();
}
//==============================================================================
float Compute::ComputeTimestep()
{
  thrust::device_vector<float> tdv_velo;
  // enough space to hold each velocity component
  tdv_velo.resize((DataInfo->resolution[0]+1)*
		  (DataInfo->resolution[1]+1)*
		  (DataInfo->resolution[2]+1));
  float *velo_raw_ptr = thrust::raw_pointer_cast(tdv_velo.data());
  thrust::device_vector<float>::iterator iter;

  // TODO: do I have to copy the array?
  // TODO: write my own reduce algorithm

  CopyCudaArray(ca_VelocityX, Res[0], velo_raw_ptr);
  iter = thrust::max_element(tdv_velo.begin(), tdv_velo.end());
  float max_u = fabs(*iter);
  iter = thrust::min_element(tdv_velo.begin(), tdv_velo.end());
  float min_u = fabs(*iter);
  max_u = min_u > max_u ? min_u : max_u;

  CopyCudaArray(ca_VelocityY, Res[1], velo_raw_ptr);
  iter =thrust::max_element(tdv_velo.begin(), tdv_velo.end());
  float max_v = fabs(*iter);
  iter = thrust::min_element(tdv_velo.begin(), tdv_velo.end());
  float min_v = fabs(*iter);
  max_v = min_v > max_v ? min_v : max_v;

  CopyCudaArray(ca_VelocityZ, Res[2], velo_raw_ptr);  
  iter = thrust::max_element(tdv_velo.begin(), tdv_velo.end());
  float max_w = fabs(*iter);
  iter = thrust::min_element(tdv_velo.begin(), tdv_velo.end());
  float min_w = fabs(*iter);
  max_w = min_w > max_w ? min_w : max_w;

  float u_max = max(max(max_u,max_v),max_w);
  float dx = 
    min(min(DataInfo->spacing[0],DataInfo->spacing[1]),DataInfo->spacing[2]);
  float CFL = 0.25f;
  float dt;

  if (u_max > 0.0f) {
    dt = CFL*dx/u_max;
  }
  else {
    dt = 0.0f;
  }
  return dt;
}
//==============================================================================
void Compute::PressureUpdate()
{
  PressureUpdate_kernel();

  UpdateCudaArray(ca_VelocityX, Res[0], VelocityX);
  UpdateCudaArray(ca_VelocityY, Res[1], VelocityY);
  UpdateCudaArray(ca_VelocityZ, Res[2], VelocityZ);
}
// TODO: implement boundary conditions in textures
// TODO: modify rhs to account solid velocities
// TODO: build matrix A
