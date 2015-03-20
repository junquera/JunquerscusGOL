#include "header.h"
//---------------------------------------------------------------------------------------------------//
int main(int argc, char *argv[]){

	int i, manual = 0;
	struct_grid grid;

	//	Inicializamos los número aleatorios
	srand((int)time(NULL));

	//Comprobación de los parámetros
	if (argc > 1){

		int posSize = -1;
		for (i = 0; i < argc; i++){
			if (!strcmp(argv[i], "-a")){
				posSize = i;
				break;
			}
			if (!strcmp(argv[i], "-m")){
				manual = 1;
			}
		}

		if (posSize == -1)
			return -1;

		//inicializa el grid en función de los parámetros que le hemos enviado
		gridInit(atoi(argv[posSize + 1]), atoi(argv[posSize + 2]), &grid);

		//pinta la primera vez en grid
		printGrid(grid);

	}


	//método jugar principal
	juega(&grid, manual);

	system("pause");
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
//alojar memoria de las estructuras del grid, dados unos tamaños. 
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
//Muestra por pantalla la matriz
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
//Éste método cuenta los vecinos de cada casilla y asigna el nuevo
int compruebaCasillas(struct_grid *t){

	int i, j, vivo = 0;
	//recorrer todas las casillas
	for (i = 0; i < (*t).sizeX; i++){
		for (j = 0; j < (*t).sizeY; j++){
			int vecinos = contarVecinos(i, j, *t);
			if ((vecinos == 2 && (*t).celdas[i][j]) || vecinos == 3){
				(*t).nuevo[i][j] = 1;
				vivo = 1;
			}
			else{
				(*t).nuevo[i][j] = 0;
			}
		}
	}
	//actualizar grid
	updateGrid(t);

	return vivo;
}
//---------------------------------------------------------------------------------------------------//
//Dado un grid, devuelve todos los vecinos
int contarVecinos(int x, int y, struct_grid t){

	int xm = (t.sizeX + ((x - 1) % t.sizeX)) % t.sizeX;
	int xM = (x + 1) % t.sizeX;

	int ym = (t.sizeY + ((y - 1) % t.sizeY)) % t.sizeY;
	int yM = (y + 1) % t.sizeY;

	return	(t.celdas[xm][yM] + t.celdas[x][yM] + t.celdas[xM][yM]) +
		(t.celdas[xm][y] + t.celdas[xM][y]) +
		(t.celdas[xm][ym] + t.celdas[x][ym] + t.celdas[xM][ym]);

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