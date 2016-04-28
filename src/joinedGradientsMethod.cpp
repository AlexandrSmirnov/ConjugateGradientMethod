#include <fstream>
#include <iostream>
#include <math.h>
#include <time.h>
#include <vector>
#include <tbb\tbb.h>
#include "tbb\task_scheduler_init.h"
#include "tbb\blocked_range.h"
#include "tbb\parallel_for.h"

using namespace std;

#pragma region Working with sparse matrix

struct crsMatrix { 
  int N;
  int NZ;
  double* Value;
  int* Col;
  int* RowIndex;
};

void InitializeMatrix(int N, int NZ, crsMatrix &mtx) {
  mtx.N = N;
  mtx.NZ = NZ;
  mtx.Value = new double[NZ];
  mtx.Col = new int[NZ];
  mtx.RowIndex = new int[N + 1];
}

void FreeMatrix(crsMatrix &A) { 
	delete[] A.Value; 
	delete[] A.Col; 
	delete[] A.RowIndex; 
}

#pragma endregion

#pragma region Working with Method data

struct MethodData
{
	int size;
	double alpha;
	double beta;
	
	double *r;
	double *p;
	double *x;

	double *AMultOnPCurrent;
	double rCurrentDotProd;	
};

void InitMethodData(MethodData &md, int size)
{
	md.size = size;
	md.r = new double[size];
	md.p = new double[size];
	md.x = new double[size];

	md.AMultOnPCurrent = new double[size];
}

void FreeMethodData(MethodData &md)
{
	delete [] md.r;
	delete [] md.p;
	delete [] md.x;
	delete [] md.AMultOnPCurrent;
}

#pragma endregion

#pragma region Print data to console

void changeToMatrixViewAndPrint(crsMatrix &A)
{
  int i, j, k, ind;
  double** matrix1 = new double*[A.N];
  for(i = 0; i < A.N; i++)
    matrix1[i] = new double[A.N];

  for (k = 0; k < A.N; k++)
  {
    for (j = 0; j < A.N; j++)
	{
		matrix1[k][j] = 0;
    }
  }

  for (ind = 0; ind < A.N; ind++)
  {
    if (A.RowIndex[ind] == A.RowIndex[ind+1]) 
      continue;
    else
	{
      for (k = A.RowIndex[ind]; k < A.RowIndex[ind+1]; k++) {
		  matrix1[ind][A.Col[k]] = A.Value[k];
		  matrix1[ind][A.Col[k]] = A.Value[k];
      }
    }
  }

  cout << "custom matrix view" << endl;
  for (k = 0; k < A.N; k++)
  {
    for (j = 0; j < A.N; j++)
	{
		cout << matrix1[k][j] << " ";
    }
    cout << endl;
  } 
  cout << endl;
  
 for (i = 0; i < A.N; i++)
    delete []matrix1[i];
  delete []matrix1;
}

void printCRS(crsMatrix &A) {
  int i;
  cout << "crs view" << endl;
  for (i = 0; i < A.NZ; i++)
	  cout << A.Value[i] << "   ";
  cout << endl;
  for (i = 0; i < A.NZ; i++)
    cout << A.Col[i] << "   ";
  cout << endl;
  for (i = 0; i <= A.N; i++)
    cout << A.RowIndex[i] << "   ";
  cout << endl;
}

void printPortrait(int **postions, int size)
{
	cout << "portrait" << endl;
	for (int i = 0; i < size; i++)
	{
		for (int j = 0; j < size; j++)
		{
			cout << postions[i][j] << " ";
		}
		cout << endl;
	}
}

#pragma endregion

#pragma region Reading Input Data From File

void ReadMatrixFromFileInMTXToCrs(string filename, crsMatrix &mtx, int &size, int &notZero)
{
	ifstream in(filename, ios::app);
	in >> size;
	in >> size;
	in >> notZero;
	
	InitializeMatrix(size, notZero, mtx);

	mtx.RowIndex[0] = 0;

	int prevI;
	int cntInRow = 0;
	int currentRowIndexPos = 1;
	bool first = true;
	for (int i = 0; i < notZero; i++)
	{
		int IInd;
		in >>  IInd;
		if (first)
		{
			for (int iRow = 1; iRow < IInd; iRow++)
				mtx.RowIndex[iRow] = 0;
			first = false;
		}
		else
		{
			if ( prevI != IInd)
			{
				mtx.RowIndex[currentRowIndexPos] = cntInRow;
				currentRowIndexPos++;
			}
			
		}
		prevI = IInd;
		cntInRow++;

		int JInd;
		in >> JInd;
		mtx.Col[i] = JInd - 1;

		in >> mtx.Value[i];
	}
	mtx.RowIndex[size] = notZero;
}

void ReadVectorBFromFile(string filename, double *b)
{
	ifstream in(filename);
	int size;
	in >> size;
	b = new double[size];
	for (int i = 0; i < size; i++)
		in >> b[i];
}

#pragma endregion

#pragma region Create self data

void CreateSparseMatrixByMySelf(crsMatrix &mtx, int &size, int &notZero)
{
	size = 3;
	notZero = 5;
	InitializeMatrix(size, notZero, mtx);
	mtx.Col[0] = 0;
	mtx.Col[1] = 2;
	mtx.Col[2] = 1;
	mtx.Col[3] = 0;
	mtx.Col[4] = 2;

	mtx.Value[0] = 2.0;
	mtx.Value[1] = 1.0;
	mtx.Value[2] = 3.0;
	mtx.Value[3] = 1.0;
	mtx.Value[4] = 4.0;

	mtx.RowIndex[0] = 0;
	mtx.RowIndex[1] = 2;
	mtx.RowIndex[2] = 3;
	mtx.RowIndex[3] = 5;
}

void CreateVectorByMySelf(double *b, int size)
{
	b = new double[size];
	b[0] = 3;
	b[1] = 3;
	b[2] = 5;
}

#pragma endregion

#pragma region Generate Data

double* GenerateVectorB(int size)
{
	double *b = new double[size];
	for (int i = 0; i < size; i++)
		b[i] = (double)(rand() % 10);
	return b;
}

void GenerateSymmetricPositiveMatrix(crsMatrix &mtx, int &size, int &notZero)
{
	if (notZero < size) notZero = size;
	if ((notZero - size) % 2 != 0) notZero--;
	

	int numNotZeroElementsOnUpTriangle = (notZero - size) / 2;
	int numElementsOnUpTriangle = (size * size - size) / 2;
	int *listIndexes = new int[numElementsOnUpTriangle];
	memset(listIndexes, 0, sizeof(int) * numElementsOnUpTriangle);

	int currentCountNotZeroElementsNeedToAdd = numNotZeroElementsOnUpTriangle; 
	
	//generate random positions
	std::vector<int> added;
	int addedCount = 0;
	while (currentCountNotZeroElementsNeedToAdd > 0)
	{
		for (int i = 0; i < numElementsOnUpTriangle; i++)
		{
			if (rand() % 2 == 1)
			{
				bool contains = false;
				for (int k = 0; k < addedCount; k++)
				{
					if (added[k] == i)
					{
						contains = true;
						break;
					}
				}
				if (!contains)
				{
					added.push_back(i);
					addedCount++;
					listIndexes[i] = 1;
					currentCountNotZeroElementsNeedToAdd--;
				}
			}
			if (currentCountNotZeroElementsNeedToAdd == 0)
				break;
		}
	}

	// create standard matrix positions
	int **matPositions = new int*[size];
	for (int i = 0; i < size; i++)
		matPositions[i] = new int[size];

	for (int i = 0; i < size; i++)
		matPositions[i][i] = 1;

	int curIndex = 0;
	for (int i = 0; i < size - 1; i++)
	{
		for (int j = i + 1; j < size; j++)
		{
			matPositions[i][j] = listIndexes[curIndex];
			matPositions[j][i] = listIndexes[curIndex];
			curIndex++;
		}
	}

	printPortrait(matPositions, size);

	// start to form sparse matrix array
	mtx.NZ = notZero;
	mtx.N = size;
	InitializeMatrix(size, notZero, mtx);
	mtx.RowIndex[0] = 0;
	int colInd = 0;
	int rowInd = 1;
	int rowElementsCount = 0;

	double **values = new double*[size];
	for (int i = 0; i < size; i++)
		values[i] = new double[size];

	// form Value
	for (int i = 0; i < size; i++)
	{
		for (int j = 0; j < size; j++)
		{
			if (matPositions[i][j] != 0)
			{
				if (i == j)
				{
					values[i][j] = (double)(rand()% 10 + notZero * 10);
				}
				else
				{
					double num = (double)(rand()% 9 + 1);
					values[i][j] = num;
					values[j][i] = num;
				}
			}
		}
	}

	for (int i = 0; i < size; i++)
	{
		for (int j = 0; j < size; j++)
		{
			if (matPositions[i][j] != 0)
			{
				// form Col
				mtx.Col[colInd] = j;
				rowElementsCount++;
				// form Value
				mtx.Value[colInd] = values[i][j];
				colInd++;
			}
		}
		//form RowIndex
		mtx.RowIndex[rowInd] = rowElementsCount;
		rowInd++;
	}

	 for (int i = 0; i < size; i++)
		delete [] matPositions[i];
	 delete [] matPositions;

	 for (int i = 0; i < size; i++)
		delete [] values[i];
	 delete [] values;

	 delete [] listIndexes;
}

#pragma endregion

#pragma region Sequantial Basic Math

void MultiplicateMV(crsMatrix A, double *b, double *x)
{
	for (int i = 0; i < A.N; i++)
	{
		x[i] = 0.0;
		for (int j = A.RowIndex[i]; j < A.RowIndex[i + 1]; j++)
			x[i] += A.Value[j] * b[A.Col[j]]; 
	}
}

double DotProduct(double *vector1, double *vector2, int size)
{
	double res = 0.0;
	for (int i = 0; i < size; i++)
		res += vector1[i] * vector2[i];
	return res;
}

double Norm(double *vector, int size)
{
	return sqrt(DotProduct(vector, vector, size));
}

#pragma endregion

#pragma region OpenMP Basic Math

void MultiplicateMV_OpenMP(crsMatrix A, double *b, double *x)
{
	#pragma omp parallel for
	for (int i = 0; i < A.N; i++)
	{
		x[i] = 0.0;
		for (int j = A.RowIndex[i]; j < A.RowIndex[i + 1]; j++)
			x[i] += A.Value[j] * b[A.Col[j]]; 
	}
}

double DotProduct_OpenMP(double *vector1, double *vector2, int size)
{
	double res = 0.0;
	#pragma omp parallel for reduction (+:res)
	for (int i = 0; i < size; i++)
		res += vector1[i] * vector2[i];
	return res;
}

double Norm_OpenMP(double *vector, int size)
{
	return sqrt(DotProduct_OpenMP(vector, vector, size));
}

#pragma endregion

#pragma region TBB Basic Math

void MultiplicateMV_TBB(crsMatrix A, double *b, double *x)
{
	tbb::parallel_for(0, A.N, 1, [=](int i) {
		x[i] = 0.0;
		for (int j = A.RowIndex[i]; j < A.RowIndex[i + 1]; j++)
			x[i] += A.Value[j] * b[A.Col[j]]; 
	});
}

double DotProduct_TBB(double *vector1, double *vector2, int size)
{
	double res = tbb::parallel_reduce(tbb::blocked_range<int>(0, size), 0.0f,
        [=](const tbb::blocked_range<int>& r, double sum)->double {
            for (int i=r.begin(); i!=r.end(); i++)
                sum += vector1[i] * vector2[i];
            return sum;
	} , [](double x, double y) -> double { return 0; });
	return res;
}

double Norm_TBB(double *vector, int size)
{
	return sqrt(DotProduct_TBB(vector, vector, size));
}

#pragma endregion

#pragma region Sequantial MSG

void Step0(crsMatrix &A, double *b, MethodData &md)
{
	for (int i = 0; i < md.size; i++)
	{
		md.x[i] = 1.0;
	}

	MultiplicateMV(A, md.x, md.p);
	
	for (int i = 0; i < md.size; i++)
	{
		md.r[i] = b[i] - md.p[i];
	}

	memcpy(md.p, md.r, sizeof(double) * md.size);
}

void Step1(const crsMatrix &A, MethodData &md)
{
	MultiplicateMV(A, md.p, md.AMultOnPCurrent);

	md.rCurrentDotProd = DotProduct(md.r, md.r, md.size);
	
	md.alpha = md.rCurrentDotProd / DotProduct(md.AMultOnPCurrent, md.p, md.size);
}

void Step2(MethodData &md)
{
	for (int i = 0; i < md.size; i++)
	{
		md.x[i] += md.alpha * md.p[i];
	}
}

void Step3(MethodData &md)
{
	for (int i = 0; i < md.size; i++)
	{
		md.r[i] -= md.alpha * md.AMultOnPCurrent[i];
	}
}

void Step4(MethodData &md)
{
	md.beta = DotProduct(md.r, md.r, md.size) / md.rCurrentDotProd;
}

void Step5(MethodData &md)
{
	for (int i = 0; i < md.size; i++)
	{
		md.p[i] = md.r[i] + md.beta * md.p[i];
	}
}

#pragma endregion

#pragma region OpenMP MSG

void Step0_OpenMP(crsMatrix &A, double *b, MethodData &md)
{
	for (int i = 0; i < md.size; i++)
	{
		md.x[i] = 1.0;
	}

	MultiplicateMV_OpenMP(A, md.x, md.p);
	
	#pragma omp parallel for
	for (int i = 0; i < md.size; i++)
	{
		md.r[i] = b[i] - md.p[i];
	}

	memcpy(md.p, md.r, sizeof(double) * md.size);
}

void Step1_OpenMP(const crsMatrix &A, MethodData &md)
{
	MultiplicateMV_OpenMP(A, md.p, md.AMultOnPCurrent);

	md.rCurrentDotProd = DotProduct_OpenMP(md.r, md.r, md.size);
	
	md.alpha = md.rCurrentDotProd / DotProduct_OpenMP(md.AMultOnPCurrent, md.p, md.size);
}

void Step2_OpenMP(MethodData &md)
{
	#pragma omp parallel for
	for (int i = 0; i < md.size; i++)
	{
		md.x[i] += md.alpha * md.p[i];
	}
}

void Step3_OpenMP(MethodData &md)
{
	#pragma omp parallel for
	for (int i = 0; i < md.size; i++)
	{
		md.r[i] -= md.alpha * md.AMultOnPCurrent[i];
	}
}

void Step4_OpenMP(MethodData &md)
{
	md.beta = DotProduct_OpenMP(md.r, md.r, md.size) / md.rCurrentDotProd;
}

void Step5_OpenMP(MethodData &md)
{
	#pragma omp parallel for
	for (int i = 0; i < md.size; i++)
	{
		md.p[i] = md.r[i] + md.beta * md.p[i];
	}
}

#pragma endregion

#pragma region TBB MSG

void Step0_TBB(crsMatrix &A, double *b, MethodData &md)
{
	for (int i = 0; i < md.size; i++)
	{
		md.x[i] = 1.0;
	}

	MultiplicateMV_TBB(A, md.x, md.p);
	
	tbb::parallel_for(0, md.size, 1, [=](int i) {
		md.r[i] = b[i] - md.p[i];
	});

	memcpy(md.p, md.r, sizeof(double) * md.size);
}

void Step1_TBB(const crsMatrix &A, MethodData &md)
{
	MultiplicateMV_TBB(A, md.p, md.AMultOnPCurrent);

	md.rCurrentDotProd = DotProduct_TBB(md.r, md.r, md.size);
	
	md.alpha = md.rCurrentDotProd / DotProduct_TBB(md.AMultOnPCurrent, md.p, md.size);
}

void Step2_TBB(MethodData &md)
{
	tbb::parallel_for(0, md.size, 1, [=](int i) {
		md.x[i] += md.alpha * md.p[i];
	});
}

void Step3_TBB(MethodData &md)
{
	tbb::parallel_for(0, md.size, 1, [=](int i) {
		md.r[i] -= md.alpha * md.AMultOnPCurrent[i];
	});
}

void Step4_TBB(MethodData &md)
{
	md.beta = DotProduct_TBB(md.r, md.r, md.size) / md.rCurrentDotProd;
}

void Step5_TBB(MethodData &md)
{
	tbb::parallel_for(0, md.size, 1, [=](int i) {
		md.p[i] = md.r[i] + md.beta * md.p[i];
	});
}

#pragma endregion

#pragma region Method versions start

void StartSequantialMethod(crsMatrix& A, double* vectorb, double *result, int size, double eps, int maxIter)
{
	double normDifference;

	// Init Method Data
	MethodData md;
	InitMethodData(md, size);

	// Start method
	auto currentIter = 0;

	Step0(A, vectorb, md);

	normDifference = Norm(md.r, md.size) / Norm(vectorb, size);
	
	if (normDifference >= eps)
	{
		do
		{
			// execute steps
			Step1(A, md);
			Step2(md);
			Step3(md);
			Step4(md);
			Step5(md);

			// check
			normDifference = Norm(md.r, md.size) / Norm(vectorb, size);
		}
		while(++currentIter < maxIter && normDifference >= eps);
	}
	memcpy(result, md.x, size * sizeof(double));
	FreeMethodData(md);	
}

void StartOpenMPMethod(crsMatrix& A, double* vectorb, double *result, int size, double eps, int maxIter)
{
	double normDifference;

	// Init Method Data
	MethodData md;
	InitMethodData(md, size);

	// Start method
	auto currentIter = 0;

	Step0_OpenMP(A, vectorb, md);

	normDifference = Norm_OpenMP(md.r, md.size) / Norm_OpenMP(vectorb, size);
	
	if (normDifference >= eps)
	{
		do
		{
			// execute steps
			Step1_OpenMP(A, md);
			Step2_OpenMP(md);
			Step3_OpenMP(md);
			Step4_OpenMP(md);
			Step5_OpenMP(md);

			// check
			normDifference = Norm_OpenMP(md.r, md.size) / Norm_OpenMP(vectorb, size);
		}
		while(++currentIter < maxIter && normDifference >= eps);
	}
	memcpy(result, md.x, size * sizeof(double));
	FreeMethodData(md);
}

void StartTBBMethod(crsMatrix& A, double* vectorb, double *result, int size, double eps, int maxIter)
{
	tbb::task_scheduler_init init;
	double normDifference;

	// Init Method Data
	MethodData md;
	InitMethodData(md, size);

	// Start method
	auto currentIter = 0;

	Step0_TBB(A, vectorb, md);

	normDifference = Norm_TBB(md.r, md.size) / Norm_TBB(vectorb, size);
	
	if (normDifference >= eps)
	{
		do
		{
			// execute steps
			Step1_TBB(A, md);
			Step2_TBB(md);
			Step3_TBB(md);
			Step4_TBB(md);
			Step5_TBB(md);

			// check
			normDifference = Norm_TBB(md.r, md.size) / Norm_TBB(vectorb, size);
		}
		while(++currentIter < maxIter && normDifference >= eps);
	}
	memcpy(result, md.x, size * sizeof(double));
	FreeMethodData(md);
}

#pragma endregion

#pragma region Checking Correctness Versions

bool AreArraysEquals(double *arr1, double *arr2, int size, double eps)
{
	for (int i = 0; i < size; i++)
		if (abs(arr1[i] - arr2[i]) > eps)
			return false;
	return true;
}

#pragma endregion

int main(int argc, char* argv[])
{
	srand(time(nullptr));
	crsMatrix A;
	double *vectorb;
	int size;
	int notZero;
	int maxIter = 10;
	double eps = 0.01;

	// Init Data
	size = 5;
	notZero = 13;
	GenerateSymmetricPositiveMatrix(A, size, notZero);
	vectorb = GenerateVectorB(size);

	// Print Data
	changeToMatrixViewAndPrint(A);
	printCRS(A);

	// Start
	auto methodSeqResult = new double[size];
	StartSequantialMethod(A, vectorb, methodSeqResult, size, eps, maxIter);

	auto methodParallelResult = new double[size];
	StartOpenMPMethod(A, vectorb, methodParallelResult, size, eps, maxIter);

	// print solution
	for (auto i = 0; i < size; i++)
		cout << methodSeqResult[i] << endl;
	cout  << endl;

	// print solution
	for (auto i = 0; i < size; i++)
		cout << methodParallelResult[i] << endl;
	cout  << endl;

	if (AreArraysEquals(methodSeqResult, methodParallelResult, size, eps))
		cout << "correct";
	else 
		cout << "wrong";
	cout << endl;

	// Free memory
	delete [] methodSeqResult;
	delete [] methodParallelResult;
	FreeMatrix(A);

	return 0;
}