/* This is a simple algorithm using Pthread. The objective is to show witch thread is entering the critical section.  
 * 
 * */
 
#include <stdio.h>
#include <pthread.h>
#include <stdlib.h>
#include <sys/time.h>
#include <semaphore.h>

#ifndef N
#define N 100000
#endif

#define MAX_THREADS 1

pthread_mutex_t  s;
pthread_mutex_t  gate;
int count = 20;

int sem;

void *processo(void *thid){
    int t;
    t = (int) thid;
    printf("Processo %d entrando seção não critica\n", t);
    pthread_mutex_lock(&gate);
    pthread_mutex_lock(&s);
    count -= 1;
    if (count > 0) pthread_mutex_unlock(&gate);
    pthread_mutex_unlock(&s);
    printf("Processo %d saindo seção não critica\n", t);
    printf("Processo %d entrando seção critica\n", t);
    pthread_mutex_lock(&s);
    count += 1;
    if (count == 1) pthread_mutex_unlock(&gate);
    pthread_mutex_unlock(&s);
    printf("Processo %d saindo seção critica\n", t);
}


int main(int argc, char * argv[]) {
  pthread_t threads[2];
  int t1, t2;
  
  pthread_mutex_init(&s, NULL);
  pthread_mutex_init(&gate, NULL);

  t1 = 0;
  t2 = 1;
  
  pthread_create(&threads[0], NULL, &processo, (void *) t1);
  pthread_create(&threads[1], NULL, &processo, (void *) t2);
  
  pthread_join(threads[0],NULL);
  pthread_join(threads[1],NULL);
  
  return 0;
}
