#define VIVA 'O'
#define MUERTA ' '

typedef struct {
	int sizeX, sizeY;
	int** celdas;
	int** nuevo;
} struct_tablero;

void gridInit(int x, int y, struct_tablero *t); 
void printGrid(struct_tablero t);
int compruebaCasillas(struct_tablero *t); 
int contarVecinos(int x, int y, struct_tablero t);
void updateGrid(struct_tablero *t);