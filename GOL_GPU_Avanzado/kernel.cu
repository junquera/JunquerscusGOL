	
#include "header.h"

//  declaración de la kernel. Se le introducen por parámetro dos arrays unidimensionales. 
// Se sacan los índices mediante la división entera y el módulo. 
__global__ void compKernel(int* celdas, int* nuevo, int sizeX, int sizeY)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int j = blockIdx.y * blockDim.y + threadIdx.y;

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
//---------------------------------------------------------------------------------------------------//
int main(int argc, char *argv[])
{
	int i, manual = 0;
	struct_grid grid;

	//Inicializamos los número aleatorios
	srand((int)time(NULL));

	gridInit(16, 16, &grid);
	printGrid(grid);
	juega(&grid, 1);

	printGrid(grid);

	system("pause");

	cudaDeviceReset();

	return 0;
}
//---------------------------------------------------------------------------------------------------//
//métdo jugar. Juega mientras la el método casillas devuelva 1. Mientras sea manual, estará pidiendo 
//		precionar, si no, dormirá un segundo y seguirá ejecutando.
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
//---------------------------------------------------------------------------------------------------//
//Hace un malloc de las estructuras del grid, dados unos tamaños. 
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
//---------------------------------------------------------------------------------------------------//
//Pinta una matriz
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
//---------------------------------------------------------------------------------------------------//
//Convierte una matriz bidimensional a una unidimensional
int* convierte(int** matriz, int dimX, int dimY){
	int* vector = (int*)malloc(dimX * dimY * sizeof(int));
	for (int i = 0; i < dimX; i++)
		for (int j = 0; j < dimY; j++)
			vector[i + j*dimX] = matriz[i][j];
	return vector;
}
//---------------------------------------------------------------------------------------------------//
//Convierte un array unidimensional a bidimensional
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
//---------------------------------------------------------------------------------------------------//
//Recibe una estructura struct_grid. Es la encargada de llamar al kernel. Reserva memoria para los arrays que enviará al kernel
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

	//Bloques de 4x4 hilos
	dim3 block_size;
	block_size.x = 4;
	block_size.y = 4;

	// configura un grid bidimensional de 4 x 4
	dim3 grid_size;
	grid_size.x = (*t).sizeX / block_size.x;
	grid_size.y = (*t).sizeY / block_size.y;

	//llamada al kernel
	compKernel <<<grid_size ,block_size >>>(dev_tablero, dev_nuevo, (*t).sizeX, (*t).sizeY);


	cudaDeviceSynchronize();

	//alojar memoria para los nuevos arrays, copiarlos del host y asignarlos al array de nuevos y celdas.
	int *a = (int*)malloc((*t).sizeX * (*t).sizeY * sizeof(int));
	int *b = (int*)malloc((*t).sizeX * (*t).sizeY * sizeof(int)); 
	cudaMemcpy(a, dev_tablero, (*t).sizeX*(*t).sizeY*sizeof(int *),cudaMemcpyDeviceToHost);
	cudaMemcpy(b, dev_nuevo, (*t).sizeX*(*t).sizeY*sizeof(int *), cudaMemcpyDeviceToHost);
	(*t).celdas = convierte(a, (*t).sizeX, (*t).sizeY);
	(*t).nuevo = convierte(b, (*t).sizeX, (*t).sizeY);

	//actualizar el grid
	updateGrid(t);

	return 1;
}
//---------------------------------------------------------------------------------------------------//
//actualiza el nuevo grid al actual
void updateGrid(struct_grid *t){
	int i, j;

	for (i = 0; i < (*t).sizeX; i++){
		for (j = 0; j < (*t).sizeY; j++){
			(*t).celdas[i][j] = (*t).nuevo[i][j];
		}
	}
}
//---------------------------------------------------------------------------------------------------//