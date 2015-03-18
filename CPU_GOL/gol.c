#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <stdlib.h>

//	Para sleep
#ifdef _WIN32
#include <Windows.h>
#else
#include <unistd.h>
#endif

#include "header.h"

int main(int argc, char *argv[]){

	int i;
	struct_tablero tablero;

	//	Inicializamos los número aleatorios
	srand((int) time(NULL));

	if( argc > 1){

		int posSize = -1;
		for(i = 0; i<argc; i++){
			if(!strcmp(argv[i], "-a")){
				posSize = i;
				break;
			}
		}

		if (posSize == -1)
			return -1;

		gridInit(atoi(argv[posSize + 1]), atoi(argv[posSize + 2]), &tablero);

	}

	while (compruebaCasillas(&tablero)){
		#ifdef _WIN32
		Sleep(1000);
		system("cls");
		#else
		sleep(1000);
		system("clear");
		#endif
	}

	system("pause");
	return 0;
}

void gridInit(int x, int y, struct_tablero *t){
	int i, j;

	(*t).sizeX = x;
	(*t).sizeY = y;
	(*t).celdas = (int**)malloc(sizeof(int) * (*t).sizeX);

	for (i = 0; i<(*t).sizeX; i++)
		(*t).celdas[i] = (int*)malloc((*t).sizeY * sizeof(int));

	(*t).nuevo = (int**)malloc(sizeof(int) * (*t).sizeX);
	for (i = 0; i<(*t).sizeX; i++)
		(*t).nuevo[i] = (int*)malloc((*t).sizeY * sizeof(int));

	for (i = 0; i < (*t).sizeX; i++)
		for (j = 0; j < (*t).sizeY; j++)
			(*t).celdas[i][j] = rand() % 2;
}

void printGrid(struct_tablero t){

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

int compruebaCasillas(struct_tablero *t){

	int i, j, vivo = 0;

	for (i = 0; i < (*t).sizeX; i++){
		for (j = 0; j < (*t).sizeY; j++){
			int vecinos = contarVecinos(i, j, *t);
			if (vecinos == 2 || vecinos == 3){
				(*t).nuevo[i][j] = 1;
				vivo = 1;
			} else{
				(*t).nuevo[i][j] = 0;
			}
		}
	}
	updateGrid(t);
	printGrid(*t);

	return vivo;
}

int contarVecinos(int x, int y, struct_tablero t){

	int xm = (t.sizeX - ((x - 1) % t.sizeX)) % t.sizeX;
	int xM = (x + 1) % t.sizeX;

	int ym = (t.sizeY + ((y - 1) % t.sizeY)) % t.sizeY;
	int yM = (y + 1) % t.sizeY;

	return	(t.celdas[xm][yM] + t.celdas[x][yM] + t.celdas[xM][yM]) + 
			(t.celdas[xm][y] + t.celdas[xM][y]) + 
			(t.celdas[xm][ym] + t.celdas[x][ym] + t.celdas[xM][ym]);

}

void updateGrid(struct_tablero *t){
	int i, j;

	for (i = 0; i < (*t).sizeX; i++){
		for (j = 0; j < (*t).sizeY; j++){
			(*t).celdas[i][j] = (*t).nuevo[i][j];
		}
	}
}

// No hace falta, pensaba que era más complicado...
void verificaCasilla(int *x, int *y, struct_tablero t){
		*x = (*x) % t.sizeX;
		*y = (*y) % t.sizeY;
}
