#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
	int sizeX, sizeY;
	int** celdas;
} struct_tablero;

void printGrid(struct_tablero t);

int main(int argc, char *argv[]){

	int i, j;
	struct_tablero tablero;

	printf("GOL\n");

	if( argc > 1){

		int posSize = -1;
		for(i = 0; i<argc; i++){
			if(!strcmp(argv[i], "-a")){
				posSize = i;
				break;
			}
		}

		tablero.sizeX = atoi(argv[posSize + 1]);
		tablero.sizeY = atoi(argv[posSize + 2]);
		tablero.celdas = (int**) malloc(sizeof(int) * tablero.sizeX);

		for (i = 0; i<tablero.sizeX; i++)
			tablero.celdas[i] = (int*) malloc(tablero.sizeY * sizeof(int));

	}

	for (i = 0; i < tablero.sizeX; i++)
		for (j = 0; j < tablero.sizeY; j++)
			tablero.celdas[i][j] = 0;	
	
	printGrid(tablero);

	system("pause");
	return 0;
}

void gridInit(struct_tablero t){

}

void printGrid(struct_tablero t){
	int i, j;

	for (i = 0; i < t.sizeX; i++){
		printf("[");
		for (j = 0; j < t.sizeY; j++){
			printf("%d", t.celdas[i][j]);
		}
		printf("]\n");
	}
}