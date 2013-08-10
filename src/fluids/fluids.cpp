#include "render.h"
#include "compute.h"
#include "data_structure.h"
#include <stdio.h>

//==============================================================================
void initData(dataInfo_t& dataInfo)
{
  dataInfo.resolution[0] = 64;
  dataInfo.resolution[1] = 64;
  dataInfo.resolution[2] = 64;
  dataInfo.origin[0] = 0.0f;
  dataInfo.origin[1] = 0.0f;
  dataInfo.origin[2] = 0.0f;
  dataInfo.spacing[0] = 1.0f;
  dataInfo.spacing[1] = 1.0f;
  dataInfo.spacing[2] = 1.0f;
}
//==============================================================================
int main(int argc, char** argv)
{
  Render *render = new Render;
  Compute *compute = new Compute;
  dataInfo_t dataInfo;

  initData(dataInfo);

  render->Init(512, 512, &dataInfo);
  render->SetDataSource(compute);

  compute->Init(&dataInfo);
  compute->RegisterVolumeTexture(render->GetVolumeTexture());
  compute->InitDye();
  compute->InitVelocity();

  render->Run();
  render->Terminate();

  compute->UnregisterVolumeTexture();

  delete [] compute;
  delete [] render;

  return 0;
}
