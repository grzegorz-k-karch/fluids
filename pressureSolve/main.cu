#include <cstdio>
#include <cuda_runtime.h>
#include <cusparse_v2.h>

void genSparseMatrix(float* csrValA, int* csrRowPtrA, int* csrColIndA, int n, 
		     int nnz, int res[3])
{
  float* h_csrValA = new float[nnz];
  int* h_csrColIndA = new int[nnz];
  int* h_csrRowPtrA = new int[n+1];

  float rho = 1.0f;
  float dt = 1.0f;
  float dx = 1.0f;
  float coeff = dt/(rho*dx*dx);
  int di = 1;
  int dj = res[0];
  int dk = res[0]*res[1];

  int nzInd = 0;

  for (int row = 0; row < n; row++) {

    // get position in volume
    int i = row%res[0];
    int j = (row/res[0])%res[1];
    int k = row/(res[0]*res[1]);
    int fc = 0; // number of fluid cells around (i,j,k)
    int nzIndDiag;

    h_csrRowPtrA[row] = nzInd;

    if (k > 0) {
      fc++;
      h_csrValA[nzInd] = 1.0f;//-coeff;
      h_csrColIndA[nzInd] = row - dk;
      nzInd++;
    }
    if (j > 0) {
      fc++;
      h_csrValA[nzInd] = 1.0f;//-coeff;
      h_csrColIndA[nzInd] = row - dj;
      nzInd++;
    }
    if (i > 0) {
      fc++;
      h_csrValA[nzInd] = 1.0f;//-coeff;
      h_csrColIndA[nzInd] = row - di;
      nzInd++;
    }

    nzIndDiag = nzInd++;

    if (i < res[0]-1) {
      fc++;
      h_csrValA[nzInd] = 1.0f;//-coeff;
      h_csrColIndA[nzInd] = row + di;
      nzInd++;
    }
    if (j < res[1]-1) {
      fc++;
      h_csrValA[nzInd] = 1.0f;//-coeff;
      h_csrColIndA[nzInd] = row + dj;
      nzInd++;
    }
    if (k < res[2]-1) {
      fc++;
      h_csrValA[nzInd] = 1.0f;//-coeff;
      h_csrColIndA[nzInd] = row + dk;
      nzInd++;
    }

    h_csrValA[nzIndDiag] = 1.0f;//fc*coeff;
    h_csrColIndA[nzIndDiag] = row;
  }
  h_csrRowPtrA[n] = nnz;

  cudaMemcpy(csrValA, h_csrValA, nnz*sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(csrColIndA, h_csrColIndA, nnz*sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(csrRowPtrA, h_csrRowPtrA, (n+1)*sizeof(int), 
	     cudaMemcpyHostToDevice);

  //test
  int ind = 0;
  for (int i = 0; i < n; i++) {

    int nzrow = h_csrRowPtrA[i+1] - h_csrRowPtrA[i];
    int nzr = 0;

    for (int j = 0; j < n; j++) {
      
      if (h_csrColIndA[ind] == j && nzr < nzrow) {
	printf("%d ", int(h_csrValA[ind]));
	ind++;
	nzr++;
      }
      else {
	printf("0 ");
      }
    }
    printf("\n");
  }
  //~test


  delete [] h_csrValA;
  delete [] h_csrColIndA;
  delete [] h_csrRowPtrA;
}

int main(int argc, char** argv)
{
  int res[3] = {3, 3, 3};
  int numCells = res[0]*res[1]*res[2];

  //----------------------------------------------------------------------------
  cusparseStatus_t cusparseStatus;
  cusparseHandle_t handle = 0;

  cusparseStatus = cusparseCreate(&handle);
  if (cusparseStatus != CUSPARSE_STATUS_SUCCESS) {
    fprintf(stderr, "could not create cusparse handle\n");
    exit(-1);
  }

  cusparseOperation_t transA = CUSPARSE_OPERATION_NON_TRANSPOSE;
  int n = numCells;
  int nnz = (n - 6)*7 + 30;
  float alpha = 1.0f;
  cusparseMatDescr_t descrA = 0;

  cusparseStatus = cusparseCreateMatDescr(&descrA);
  if (cusparseStatus != CUSPARSE_STATUS_SUCCESS) {
    fprintf(stderr, "could not create matrix description\n");
    exit(-1);
  }

  float *csrValA;
  int *csrRowPtrA;
  int *csrColIndA;
  float *x;
  float beta = 0.0f;
  float *y;

  cudaMalloc((void**)&csrValA, nnz*sizeof(float));
  cudaMalloc((void**)&csrRowPtrA, (n+1)*sizeof(float));
  cudaMalloc((void**)&csrColIndA, nnz*sizeof(float));
  cudaMalloc((void**)&x, n*sizeof(float));
  cudaMalloc((void**)&y, n*sizeof(float));

  float *h_x = new float[n];
  for (int i = 0; i < n; i++) {
    h_x[i] = 1.0f;
  }
  cudaMemcpy(x, h_x, n*sizeof(float), cudaMemcpyHostToDevice);  
  delete [] h_x;

  //-build compressed sparse row format-----------------------------------------
  genSparseMatrix(csrValA, csrRowPtrA, csrColIndA, n, nnz, res);
  //----------------------------------------------------------------------------

  cusparseStatus = cusparseScsrmv(handle, transA, n, n, nnz, &alpha, descrA, 
				  csrValA, csrRowPtrA, csrColIndA, x, &beta, y);

  if (cusparseStatus != CUSPARSE_STATUS_SUCCESS) {
    fprintf(stderr, "could not perform csrmv\n");
    exit(-1);
  }
  //test
  float *h_y = new float[n];
  cudaMemcpy(h_y, y, n*sizeof(float), cudaMemcpyDeviceToHost);  

  for (int i = 0; i < n; i++) {
    printf("%d %f\n", i, h_y[i]);
  }
  delete [] h_y;
  //~test

  cusparseDestroy(handle);
  cudaFree(csrValA);
  cudaFree(csrRowPtrA);
  cudaFree(csrColIndA);
  cudaFree(x);
  cudaFree(y);

  //----------------------------------------------------------------------------

  return 0;
}
