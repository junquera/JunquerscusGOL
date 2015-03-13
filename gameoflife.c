#include <stdio.h>
#include <stdlib.h>

int contarVecinos(int **grid, int fila, int columna);
int** actualizar(int **new, int **old);


int main(int argc, char const *argv[])
{
  int **new, **old;
  int nF, nC;
  nF = 3;
  nC = 3;
  int i;

  old = malloc(nC*sizeof(int*));
  new = malloc(nC*sizeof(int*));

  for(i=0; i<nC; i++){
    old[i] = malloc(nF*sizeof(int));
    new[i] = malloc(nF*sizeof(int));
  }

  if(argc != 4){
    printf("asdas\n");
  }

  return 0;
}

int contarVecinos(int **grid, int fila, int columna)
{
  return 0;
}

int** actualizar(int **new, int **old)
{
  return new;
}