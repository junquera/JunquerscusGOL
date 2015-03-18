#define VIVA 'O'
#define MUERTA 'X'

#ifdef _WIN32
#include <Windows.h>
#define SLEEP(x) Sleep(x)
#define CLRSCR() system("cls")
#else
#include <unistd.h>
#define SLEEP(x) sleep(x)
#define CLRSCR() system("clear")
#endif

typedef struct {
	int sizeX, sizeY;
	int** celdas;
	int** nuevo;
} struct_grid;

void gridInit(int x, int y, struct_grid *t); 
void printGrid(struct_grid t);
int compruebaCasillas(struct_grid *t); 
int contarVecinos(int x, int y, struct_grid t);
void updateGrid(struct_grid *t);
void juega(struct_grid *t, int manual);