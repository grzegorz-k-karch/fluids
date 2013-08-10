#include "compute.h"
#include "helper_cuda.h" // my code
#include "helper_math.h" // cuda sdk
#include <iostream>

texture<float, cudaTextureType3D, cudaReadModeElementType> veloXTex;
texture<float, cudaTextureType3D, cudaReadModeElementType> veloYTex;
texture<float, cudaTextureType3D, cudaReadModeElementType> veloZTex;
texture<float, cudaTextureType3D, cudaReadModeElementType> dyeTex;

__device__ __constant__ float c_dt;
__device__ __constant__ float c_rdx;
__device__ __constant__ float c_rrho;

//==============================================================================
__device__
float3 tex3DVelocity(float3 x)
{
  float3 v;
  v.x = tex3D(veloXTex, x.x+0.5f, x.y,      x.z     );
  v.y = tex3D(veloYTex, x.x,      x.y+0.5f, x.z     );
  v.z = tex3D(veloZTex, x.x,      x.y,      x.z+0.5f);
  return v;
}
//==============================================================================
__device__
float3 d_Advect(float3 x, float dt)
{
  float3 xNew;

  // Runge-Kutta 4th order
  const float f6 = 0.1666666666f;

  float3 k1 = dt*tex3DVelocity(x);
  float3 k2 = dt*tex3DVelocity(x+0.5f*k1);
  float3 k3 = dt*tex3DVelocity(x+0.5f*k2);
  float3 k4 = dt*tex3DVelocity(x+k3);

  xNew = x + f6*(k1 + 2.0f*k2 + 2.0f*k3 + k4);
  return xNew;
}
//==============================================================================
__global__
void g_AdvectDye(float* dye, dim3 volumeSize)
{
  int gid = blockIdx.x*blockDim.x + threadIdx.x;
  int x = gid % volumeSize.x;
  int y = (gid / volumeSize.x) % volumeSize.y;
  int z = gid / (volumeSize.x*volumeSize.y);
  float3 pos = make_float3(x+0.5f, y+0.5f, z+0.5f);

  // advect particle
  pos = d_Advect(pos, -c_dt);

  int numCells = volumeSize.x*volumeSize.y*volumeSize.z;
  if (gid < numCells) {
    dye[gid] = tex3D(dyeTex, pos.x, pos.y, pos.z);
  }
}
//==============================================================================
__global__
void g_ComputeDivergence(float* divergence, dim3 volumeSize)
{
  int gid = blockIdx.x*blockDim.x + threadIdx.x;
  int x = gid % volumeSize.x;
  int y = (gid / volumeSize.x) % volumeSize.y;
  int z = gid / (volumeSize.x*volumeSize.y);
  int numCells = volumeSize.x*volumeSize.y*volumeSize.z;
  float3 pos = make_float3(x+0.5f, y+0.5f, z+0.5f);;

  if (gid < numCells) {

    float u0, u1;
    float div = 0.0f;

    u0 = tex3D(veloXTex, pos.x, pos.y, pos.z);
    u1 = tex3D(veloXTex, pos.x+1.0f, pos.y, pos.z);
    div += -c_rdx*(u1 - u0);

    u0 = tex3D(veloYTex, pos.x, pos.y, pos.z);
    u1 = tex3D(veloYTex, pos.x, pos.y+1.0f, pos.z);
    div += -c_rdx*(u1 - u0);

    u0 = tex3D(veloZTex, pos.x, pos.y, pos.z);
    u1 = tex3D(veloZTex, pos.x, pos.y, pos.z+1.0f);
    div += -c_rdx*(u1 - u0);

    divergence[gid] = div;
  }
}
//==============================================================================
__global__
void g_SetBoundaryConditions(float* velo, dim3 volumeSize, int dir)
{
  int gid = blockIdx.x*blockDim.x + threadIdx.x;
  int nfx = volumeSize.x;//number of faces in x direction
  int nfy = volumeSize.y;//number of faces in y direction
  int nfz = volumeSize.z;//number of faces in z direction

  switch (dir) {
  case 0:
    nfx += 1;
    break;
  case 1:
    nfy += 1;
    break;
  case 2:
    nfz += 1;
    break;
  }
  int x = gid % nfx;
  int y = (gid / nfx) % nfy;
  int z = gid / (nfx*nfy);
  int numCells = nfx*nfy*nfz;

  if (gid < numCells) {
    switch (dir) {
    case 0: // X faces
      if (x == 0) {
	if (y > nfy/4 && y < nfy*3/4 && z > nfz/4 && nfz < nfz*3/4) {
	  velo[gid] = 1.0f;
	}
	else {
	  velo[gid] = 0.0f;
	}
      }
      else if (x == nfx-1) {
	velo[gid] = 0.0f;
      }
      break;
    case 1: // Y faces
      if (y == 0 || y == nfy-1) {
	velo[gid] = 0.0f;
      }
      break;
    case 2: // Z faces
      if (z == 0 || z == nfz-1) {
	velo[gid] = 0.0f;
      }
      break;
    default:
      break;
    }
  }
}
//==============================================================================
__global__
void g_AdvectVelocity(float* velo, dim3 volumeSize, int dir)
{
  int gid = blockIdx.x*blockDim.x + threadIdx.x;
  int nfx = volumeSize.x;//number of faces in x direction
  int nfy = volumeSize.y;//number of faces in y direction
  int nfz = volumeSize.z;//number of faces in z direction
  float offx = 0.5f;
  float offy = 0.5f;
  float offz = 0.5f;

  switch (dir) {
  case 0:
    nfx += 1;
    offx = 0.0f;
    break;
  case 1:
    nfy += 1;
    offy = 0.0f;
    break;
  case 2:
    nfz += 1;
    offz = 0.0f;
    break;
  }
  int x = gid % nfx;
  int y = (gid / nfx) % nfy;
  int z = gid / (nfx*nfy);
  float3 pos = make_float3(x+offx, y+offy, z+offz);

  pos = d_Advect(pos, -c_dt);

  int numCells = nfx*nfy*nfz;
  if (gid < numCells) {
    switch (dir) {
    case 0:
      velo[gid] = tex3D(veloXTex, pos.x, pos.y, pos.z);
      break;
    case 1:
      velo[gid] = tex3D(veloYTex, pos.x, pos.y, pos.z);
      break;
    case 2:
      velo[gid] = tex3D(veloZTex, pos.x, pos.y, pos.z);
      break;
    }
  }
}
//==============================================================================
__global__
void g_InitDye(float* dye, dim3 volumeSize)
{
  int gid = blockIdx.x*blockDim.x + threadIdx.x;
  int x = gid % volumeSize.x;
  int y = (gid / volumeSize.x) % volumeSize.y;
  int z = gid / (volumeSize.x*volumeSize.y);

  int dyeSize = volumeSize.x*volumeSize.y*volumeSize.z;

  if (gid < dyeSize) {
    if (x > volumeSize.x/4 && x < volumeSize.x*3/4 &&
	y > volumeSize.y/4 && y < volumeSize.y*3/4 &&
	z > volumeSize.z/4 && z < volumeSize.z*3/4) {
      dye[gid] = 1.0f;
    }
    else {
      dye[gid] = 0.0f;
    }
  }
}
//==============================================================================
__global__
void g_InitVelocity(float* veloX, float* veloY, float* veloZ, dim3 volumeSize)
{
  int gid = blockIdx.x*blockDim.x + threadIdx.x;
  int sizeX = (volumeSize.x+1)*volumeSize.y*volumeSize.z;
  int sizeY = volumeSize.x*(volumeSize.y+1)*volumeSize.z;
  int sizeZ = volumeSize.x*volumeSize.y*(volumeSize.z+1);

  // X
  if (gid < sizeX) {
    int x = gid%(volumeSize.x+1);
    int y = (gid/(volumeSize.x+1))%volumeSize.y;
    int z = gid/((volumeSize.x+1)*volumeSize.y);
    if (x == 0) {
      veloX[gid] = 1.0f;
    }
    else {
      veloX[gid] = 0.0f;
    }
  }
  // Y
  if (gid < sizeY) {
    veloY[gid] = 0.0f;
  }
  // Z
  if (gid < sizeZ) {
    veloZ[gid] = 0.0f;
  }
}
//==============================================================================
#define BLOCK_SIZE 64
//==============================================================================
// 0 1 2 3 4 5 6 7 8
// | | | | | | | | |
//  0 1 2 3 4 5 6 7
__global__
void g_PressureUpdateX(float* pressure, dim3 volumeSize, float* velo)
{
  int x = threadIdx.x;
  int y = blockIdx.x % volumeSize.y;
  int z = blockIdx.x / volumeSize.y;
  __shared__ float p[BLOCK_SIZE];
  int gidp = x + y*volumeSize.x + z*volumeSize.x*volumeSize.y;

  if (x < volumeSize.x) {
    p[threadIdx.x] = pressure[gidp];
  }
  __syncthreads();

  // first thread in each block doesn't take part in the game
  if (x > 0 && x < volumeSize.x) {
    veloUpdate = tex3D(veloXTex, x+0.5f, y+0.5f, z+0.5f);

    float scale = -c_dt*c_rdx*c_rrho;
    veloUpdate += scale*p[threadIdx.x-1];
    veloUpdate -= scale*p[threadIdx.x];

    int gidv = x + y*(volumeSize.x+1) + z*(volumeSize.x+1)*(volumeSize.y+1);
    velo[gidv] = veloUpdate;
  }

}
#if 0
__global__
void g_PressureUpdateX(float* pressure, dim3 volumeSize, float* velo)
{
  int gid = blockIdx.x*blockDim.x + threadIdx.x;
  int x = gid % (volumeSize.x+1);
  int y = (gid / (volumeSize.x+1)) % volumeSize.y;
  int z = gid / ((volumeSize.x+1)*volumeSize.y);
  int gidp = x + y*volumeSize.x + z*volumeSize.x*volumeSize.y;
  __shared__ float p[BLOCK_SIZE+1];
  int numFaces = (volumeSize.x+1)*volumeSize.y*volumeSize.z;
  float veloUpdate;

  if (gid < numFaces) {

    veloUpdate = tex3D(veloXTex, x+0.5f, y+0.5f, z+0.5f);

    // get pressure to the right of the face
    if (x < volumeSize.x) {

      p[threadIdx.x+1] = pressure[gidp];

      // if the first face in thread block is not the leftmost face then load
      // pressure to the right of it
      if (threadIdx.x == 0 && x > 0) {

  	p[0] = pressure[gidp-1];
      }
    }
  }
  __syncthreads();

  float scale = -c_dt*c_rdx*c_rrho;
  if (gid < numFaces) {

    if (x < volumeSize.x) {

      veloUpdate += p[threadIdx.x+1];
    }
    if (x > 0) {

      veloUpdate -= p[threadIdx.x];
    }
    velo[gid] = veloUpdate;
  }
}
//==============================================================================
// Since the data is not aligned in y and z direction, kernel Y and Z will
// process pressure updates as x-y or x-z blocks to minimize memory penalty
// due to y and z memory strides
#define BLOCK_SIZE_X 16
#define BLOCK_SIZE_Y 16
// TODO: use profiling to find optimum, check for different compute capabilities
//==============================================================================
__global__
void g_PressureUpdateY(float* pressure, dim3 volumeSize, float* velo)
{
  int numThreadBlocksX = myDivUp(volumeSize.x, BLOCK_SIZE_X);
  int numThreadBlocksY = myDivUp(volumeSize.y, BLOCK_SIZE_Y);
  int bIdx = blockIdx.x % numThreadBlocksX;
  int gx = bIdx * BLOCK_SIZE_X;
  int bIdy = ((blockIdx.x / numThreadBlocksX) % numThreadBlocksY);
  int gy = bIdy * BLOCK_SIZE_Y;
  int lx = threadIdx.x & (BLOCK_SIZE_X-1);
  int ly = (threadIdx.x / BLOCK_SIZE_X);

  int x = gx + lx;
  int y = gy + ly;
  int z = blockIdx.x / (numThreadBlocksX*numThreadBlocksY);

  __shared__ float p[BLOCK_SIZE_Y+1][BLOCK_SIZE_X];
  float veloUpdate;

  if (x < volumeSize.x && y < volumeSize.y) {

    veloUpdate = tex3D(veloYTex, x+0.5f, y+0.5f, z+0.5f);
    int gid = x + y*volumeSize.x + z*volumeSize.x*volumeSize.y;
    p[ly+1][lx] = pressure[gid];

    if (ly == 0 && y > 0) {
      p[0][lx] = pressure[gid-volumeSize.x];
    }
  }
  __syncthreads();

  int gidv = x + y*volumeSize.x + z*volumeSize.x*(volumeSize.y+1);
  float scale = -c_dt*c_rdx*c_rrho;
  if (x < volumeSize.x && y < volumeSize.y) {

    if (y < volumeSize.y) {

      veloUpdate += p[ly+1][lx];
    }
    if (y > 0) {

      veloUpdate -= p[ly][lx];
    }
    velo[gidv] = veloUpdate;
  }
  // update bottommost face velocities
  if (y == volumeSize.y-1) {

    y += 1;
    veloUpdate = tex3D(veloYTex, x+0.5f, y+0.5f, z+0.5f);

    if (x < volumeSize.x) {
      veloUpdate -= p[ly+1][lx];
    }
    gidv = x + y*volumeSize.x + z*volumeSize.x*(volumeSize.y+1);
    velo[gidv] = veloUpdate;
  }
}
//==============================================================================
__global__
void g_PressureUpdateZ(float* pressure, dim3 volumeSize, float* velo)
{
  int numThreadBlocksX = myDivUp(volumeSize.x, BLOCK_SIZE_X);
  int numThreadBlocksZ = myDivUp(volumeSize.z, BLOCK_SIZE_Y);
  int bIdx = blockIdx.x % numThreadBlocksX;
  int gx = bIdx * BLOCK_SIZE_X;
  int bIdz = ((blockIdx.x / numThreadBlocksX) % numThreadBlocksZ);
  int gz = bIdz * BLOCK_SIZE_Y;
  int lx = threadIdx.x & (BLOCK_SIZE_X-1);
  int lz = (threadIdx.x / BLOCK_SIZE_X);

  int x = gx + lx;
  int y = blockIdx.x / (numThreadBlocksX*numThreadBlocksZ);
  int z = gz + lz;

  __shared__ float p[BLOCK_SIZE_Y][BLOCK_SIZE_X];
  float veloUpdate;

  if (x < volumeSize.x && z < volumeSize.z) {

    veloUpdate = tex3D(veloZTex, x+0.5f, y+0.5f, z+0.5f);
    int gid = x + y*volumeSize.x + z*volumeSize.x*volumeSize.y;
    p[lz+1][lx] = pressure[gid];

    if (lz == 0 && z > 0) {
      p[0][lx] = pressure[gid-volumeSize.x*volumeSize.y];
    }
  }
  __syncthreads();
  // TODO: gid for velocity
  // TODO: update furthermost and bottommost face velocities!!
  float scale = -c_dt*c_rdx*c_rrho;
  if (x < volumeSize.x && z < volumeSize.z) {

    if (z < volumeSize.z) {

      veloUpdate += p[lz+1][lx];
    }
    if (z > 0) {

      veloUpdate -= p[lz][lx];
    }
    velo[gid] = veloUpdate;
  }
}
#endif
//==============================================================================
void Compute::InitDye_kernel()
{
  int dyeSize =
    (DataInfo->resolution[0])*
    (DataInfo->resolution[1])*
    (DataInfo->resolution[2]);
  int numThreads = 256;
  int numBlocks = myDivUp(dyeSize, numThreads);

  g_InitDye<<<numBlocks, numThreads>>>(Dye, VolumeSize);
}
//==============================================================================
void Compute::InitVelocity_kernel()
{
  // instead of running initialization for each velocity component separately,
  // create enough threads to cover the staggered grid, i.e.,
  // stagRes has one additional cell in each direction
  int stagSize =
    (DataInfo->resolution[0]+1)*
    (DataInfo->resolution[1]+1)*
    (DataInfo->resolution[2]+1);
  int numThreads = 256;
  int numBlocks = myDivUp(stagSize, numThreads);

  g_InitVelocity<<<numBlocks, numThreads>>>(VelocityX, VelocityY,
					    VelocityZ, VolumeSize);
}
//==============================================================================
void Compute::AdvectDye_kernel()
{
  int numCells =
    (DataInfo->resolution[0])*
    (DataInfo->resolution[1])*
    (DataInfo->resolution[2]);
  int numThreads = 256;
  int numBlocks = myDivUp(numCells, numThreads);

  cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float>();

  cudaBindTextureToArray(dyeTex, ca_Dye, channelDesc);
  g_AdvectDye<<<numBlocks, numThreads>>>(Dye, VolumeSize);
  cudaUnbindTexture(dyeTex);
}
//==============================================================================
void Compute::AdvectVelocity_kernel()
{
  int numThreads = 256;
  int numBlocks;
  cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float>();

  // X
  numBlocks = myDivUp(NumCellFaces[0], numThreads);
  g_AdvectVelocity<<<numBlocks, numThreads>>>(VelocityX, VolumeSize, 0);
  // Y
  numBlocks = myDivUp(NumCellFaces[1], numThreads);
  g_AdvectVelocity<<<numBlocks, numThreads>>>(VelocityY, VolumeSize, 1);
  // Z
  numBlocks = myDivUp(NumCellFaces[2], numThreads);
  g_AdvectVelocity<<<numBlocks, numThreads>>>(VelocityZ, VolumeSize, 2);
}
//==============================================================================
void Compute::SetBoundaryConditions_kernel()
{
  int numThreads = 256;
  int numBlocks;

  // X
  numBlocks = myDivUp(NumCellFaces[0], numThreads);
  g_SetBoundaryConditions<<<numBlocks, numThreads>>>(VelocityX, VolumeSize, 0);
  // Y
  numBlocks = myDivUp(NumCellFaces[1], numThreads);
  g_SetBoundaryConditions<<<numBlocks, numThreads>>>(VelocityY, VolumeSize, 1);
  // Z
  numBlocks = myDivUp(NumCellFaces[2], numThreads);
  g_SetBoundaryConditions<<<numBlocks, numThreads>>>(VelocityZ, VolumeSize, 2);
}
//==============================================================================
void Compute::InitTextures()
{
  // X
  veloXTex.normalized = false;
  veloXTex.filterMode = cudaFilterModeLinear;
  veloXTex.addressMode[0] = cudaAddressModeClamp;
  veloXTex.addressMode[1] = cudaAddressModeClamp;
  veloXTex.addressMode[2] = cudaAddressModeClamp;
  // Y
  veloYTex.normalized = false;
  veloYTex.filterMode = cudaFilterModeLinear;
  veloYTex.addressMode[0] = cudaAddressModeClamp;
  veloYTex.addressMode[1] = cudaAddressModeClamp;
  veloYTex.addressMode[2] = cudaAddressModeClamp;
  // Z
  veloZTex.normalized = false;
  veloZTex.filterMode = cudaFilterModeLinear;
  veloZTex.addressMode[0] = cudaAddressModeClamp;
  veloZTex.addressMode[1] = cudaAddressModeClamp;
  veloZTex.addressMode[2] = cudaAddressModeClamp;
  // dye
  dyeTex.normalized = false;
  dyeTex.filterMode = cudaFilterModeLinear;
  dyeTex.addressMode[0] = cudaAddressModeClamp;
  dyeTex.addressMode[1] = cudaAddressModeClamp;
  dyeTex.addressMode[2] = cudaAddressModeClamp;

  cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float>();
  cudaBindTextureToArray(veloXTex, ca_VelocityX, channelDesc);
  cudaBindTextureToArray(veloYTex, ca_VelocityY, channelDesc);
  cudaBindTextureToArray(veloZTex, ca_VelocityZ, channelDesc);
}
//==============================================================================
void Compute::InitSymbols()
{
  float rdx = 1.0f/(DataInfo->spacing[0]);
  myCudaCall(cudaMemcpyToSymbol(c_rdx, &rdx, sizeof(float), 0,
  				cudaMemcpyHostToDevice), __LINE__, __FILE__);
  float rho = 997.7735; // [kg/m^3] water density at 22 degrees Celsius
  float rrho = 1.0f/(rrho);
  myCudaCall(cudaMemcpyToSymbol(c_rrho, &rrho, sizeof(float), 0,
  				cudaMemcpyHostToDevice), __LINE__, __FILE__);
}
//==============================================================================
void Compute::ComputeDivergence_kernel()
{
  int numCells =
    (DataInfo->resolution[0])*
    (DataInfo->resolution[1])*
    (DataInfo->resolution[2]);
  int numThreads = 256;
  int numBlocks = myDivUp(numCells, numThreads);
  cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float>();

  g_ComputeDivergence<<<numBlocks, numThreads>>>(Divergence, VolumeSize);
}
//==============================================================================
void Compute::SetTimestep(float dt)
{
  myCudaCall(cudaMemcpyToSymbol(c_dt, &dt, sizeof(float), 0,
  				cudaMemcpyHostToDevice), __LINE__, __FILE__);
}
//==============================================================================
void Compute::PressureUpdate_kernel()
{
  int numThreads;
  int numBlocks;
  int numBlocksX;

  // TODO: compute block size according to the volume size
  // TODO: multiple blocks in x direction, if necessary
  numThreads = BLOCK_SIZE;
  numBlocksX = 1;
  numBlocks = DataInfo->resolution[1]*DataInfo->resolution[2]*numBlocksX;

  g_PressureUpdateX<<<numBlocks, numThreads>>>(Pressure, VolumeSize, VelocityX);

  // int numThreads;
  // int numBlocks;
  // cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float>();

  // numThreads = BLOCK_SIZE;
  // numBlocks = myDivUp(NumCellFaces[0], numThreads);
  // g_PressureUpdateX<<<numBlocks, numThreads>>>(Pressure, VolumeSize, VelocityX);

  // // For y and z use the pressure resolution; the threads at the bottom (y) and
  // // at the back (z) will take care of updating the bottommost (y) and furthest
  // // (z) faces.
  // numThreads = BLOCK_SIZE_X*BLOCK_SIZE_Y;
  // int numThreadBlocksX = myDivUp(DataInfo->resolution[0], BLOCK_SIZE_X);
  // int numThreadBlocksY = myDivUp(DataInfo->resolution[1], BLOCK_SIZE_Y);
  // numBlocks = numThreadBlocksX*numThreadBlocksY*(DataInfo->resolution[2]);
  // g_PressureUpdateY<<<numBlocks, numThreads>>>(Pressure, VolumeSize, VelocityY);

  // // numBlocks = myDivUp(NumCellFaces[2], numThreads);
  // // g_PressureUpdateZ<<<numBlocks, numThreads>>>(Pressure, VolumeSize, VelocityZ);
}
