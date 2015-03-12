#include <stdio.h>
#include <stdlib.h>

typedef struct {
	int sizeX, sizeY;
	char** celdas;
} tablero;

int main(int argc, char *argv[]){

	int i;

	if( argc > 1){
		int posSize = -1;
		for(i = 0; i<argc; i++){
			if(argv[i] == "-a"){
				posSize = i;
				break;
			}
		}
		tablero.sizeX = atoi(argv[posSize + 1]);
		tablero.sizeY = atoi(argv[posSize + 2]);
	}
	return 0;
}