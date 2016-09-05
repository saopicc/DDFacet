#pragma once
#include <time.h>

extern clock_t start;

void initTime(){start=clock();}

void timeit(char* Name){
  clock_t diff;
  diff = clock() - start;
  start=clock();
  float msec = diff * 1000 / CLOCKS_PER_SEC;
  printf("%s: %f\n",Name,msec);
}

void AddTimeit(struct timespec PreviousTime, long int *aTime){
  long int t0 = PreviousTime.tv_nsec+PreviousTime.tv_sec*1000000000;
  clock_gettime(CLOCK_MONOTONIC_RAW, &PreviousTime);
  (*aTime)+=(PreviousTime.tv_nsec+PreviousTime.tv_sec*1000000000-t0);
}