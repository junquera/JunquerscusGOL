
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <stdlib.h>
#include <conio.h>

#include "header.h"

//  declaración de la kernel. Se le introducen por parámetro dos arrays unidimensionales. 
// Se sacan los índices mediante la división entera y el módulo. 
__global__ void compKernel(int* celdas, int* nuevo, int sizeX, int sizeY)
{
	int thx = threadIdx.x;
	int i = thx % sizeX;
	int j = thx / sizeX;

	int xm = (sizeX + ((i - 1) % sizeX)) % sizeX;
	int xM = (i + 1) % sizeX;

	int ym = (sizeY + ((j - 1) % sizeY)) % sizeY;
	int yM = (j + 1) % sizeY;


	int vecinos =	(celdas[xm + yM *sizeX] + celdas[i + yM *sizeX] + celdas[xM + yM *sizeX]) +
					(celdas[xm + j*sizeX] + celdas[xM + j*sizeX]) +
					(celdas[xm + ym * sizeX] + celdas[i + ym * sizeX] + celdas[xM + ym*sizeX]);

	if ((vecinos == 2 && celdas[i + j*sizeX]) || vecinos == 3){
		nuevo[i + j*sizeX] = 1;
	}
	else{
		nuevo[i + j * sizeX] = 0;
	}
}

int main(int argc, char *argv[])
{
	int i, manual = 0;
	struct_grid grid;

	//	Inicializamos los número aleatorios
	srand((int)time(NULL));

	gridInit(10, 10, &grid);
	printGrid(grid);
	juega(&grid, 1);

	printGrid(grid);

	system("pause");

	cudaDeviceReset();

	return 0;
}

void juega(struct_grid *t, int manual){
	while (compruebaCasillas(t)){
		if (manual){
			printf("\nPresiona intro para continuar o \"c\" para parar la ejecución...\n");

			int option;
			do{
				option = getch();
				if (option == 67 || option == 99)
					return;
			} while (option != 13);

		}
		else{
			SLEEP(1000);
		}
		printGrid(*t);
	}
}
void gridInit(int x, int y, struct_grid *t){
	int i, j;

	(*t).sizeX = x;
	(*t).sizeY = y;
	(*t).celdas = (int**)malloc(sizeof(int) * (*t).sizeX);

	for (i = 0; i < (*t).sizeX; i++)
		(*t).celdas[i] = (int*)malloc((*t).sizeY * sizeof(int));

	(*t).nuevo = (int**)malloc(sizeof(int) * (*t).sizeX);
	for (i = 0; i < (*t).sizeX; i++)
		(*t).nuevo[i] = (int*)malloc((*t).sizeY * sizeof(int));

	for (i = 0; i < (*t).sizeX; i++){
		for (j = 0; j < (*t).sizeY; j++){
			(*t).celdas[i][j] = 0;
			(*t).nuevo[i][j] = 0;
		}
	}

	for (i = 0; i < (*t).sizeX; i++)
		for (j = 0; j < (*t).sizeY; j++)
			(*t).celdas[i][j] = rand() % 2;

}

void printGrid(struct_grid t){

	CLRSCR();
	printf("Game of life [CPU]\n\n");

	int i, j;

	for (i = 0; i < t.sizeX; i++){
		printf("[");
		for (j = 0; j < t.sizeY; j++){

			if (t.celdas[i][j])
				printf("%c", VIVA);
			else
				printf("%c", MUERTA);
		}
		printf("]\n");
	}
}

int* convierte(int** matriz, int dimX, int dimY){
	int* vector = (int*)malloc(dimX * dimY * sizeof(int));
	for (int i = 0; i < dimX; i++)
		for (int j = 0; j < dimY; j++)
			vector[i + j*dimX] = matriz[i][j];
	return vector;
}

int** convierte(int* vector, int dimX, int dimY){
	int i;
	int** matriz = (int**)malloc(sizeof(int) * dimX);

	for (i = 0; i < dimX; i++)
		matriz[i] = (int*)malloc(dimY * sizeof(int));

	for (int i = 0; i < dimX; i++)
		for (int j = 0; j < dimY; j++)
			matriz[i][j] = vector[i + j*dimX];

	return matriz;
}

int compruebaCasillas(struct_grid *t){

	int i, j, vivo = 0;

	int *dev_tablero = 0;
	int *dev_nuevo = 0;
	int size = (*t).sizeX * (*t).sizeY;

	// Allocate GPU buffers
	cudaMalloc((void**)&dev_tablero, size*sizeof(int));
	cudaMalloc((void**)&dev_nuevo, size*sizeof(int));

	cudaMemcpy(dev_tablero, convierte((*t).celdas, (*t).sizeX, (*t).sizeY), size*sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_nuevo, convierte((*t).celdas, (*t).sizeX, (*t).sizeY), size*sizeof(int), cudaMemcpyHostToDevice);

	// Launch a kernel on the GPU with one thread for each element.
	compKernel <<<1 ,(*t).sizeX * (*t).sizeY >>>(dev_tablero, dev_nuevo, (*t).sizeX, (*t).sizeY);

	// cudaDeviceSynchronize waits for the kernel to finish, and returns
	// any errors encountered during the launch.
	cudaDeviceSynchronize();


	int *a = (int*)malloc((*t).sizeX * (*t).sizeY * sizeof(int));
	int *b = (int*)malloc((*t).sizeX * (*t).sizeY * sizeof(int)); 
	cudaMemcpy(a, dev_tablero, (*t).sizeX*(*t).sizeY*sizeof(int *),cudaMemcpyDeviceToHost);
	cudaMemcpy(b, dev_nuevo, (*t).sizeX*(*t).sizeY*sizeof(int *), cudaMemcpyDeviceToHost);
	(*t).celdas = convierte(a, (*t).sizeX, (*t).sizeY);
	(*t).nuevo = convierte(b, (*t).sizeX, (*t).sizeY);

	updateGrid(t);

	return 1;
}

void updateGrid(struct_grid *t){
	int i, j;

	for (i = 0; i < (*t).sizeX; i++){
		for (j = 0; j < (*t).sizeY; j++){
			(*t).celdas[i][j] = (*t).nuevo[i][j];
		}
	}
}