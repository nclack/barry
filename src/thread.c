#include "thread.h"

#if _MSC_VER

thread_t thread_create (void (*f)(void*),void* p){
    return CreateThread(0,0,(LPTHREAD_START_ROUTINE)f,p,0,0);   
}

void thread_release(thread_t *t) {
    CloseHandle(*t);
}

int thread_join(thread_t *t, unsigned timeout_ms) {
    return WAIT_OBJECT_0==WaitForSingleObject(*t,timeout_ms);
}

mutex_t mutex_create () {
  mutex_t m=SRWLOCK_INIT;
  return m;
}

void mutex_release(mutex_t *m) {/*noop*/}

void mutex_lock(mutex_t *m) {
    AcquireSRWLockExclusive(m);
}

void mutex_unlock(mutex_t *m) { ReleaseSRWLockExclusive(m); }

condition_t condition_create() {
    condition_t c=CONDITION_VARIABLE_INIT;
    return c;
}

void condition_release(condition_t *c) {/*noop*/}

int condition_wait   (condition_t *c,mutex_t *m,unsigned timeout_ms) {
    return SleepConditionVariableSRW(c,m,timeout_ms,0);
}

void condition_notify(condition_t* c) {
    WakeAllConditionVariable(c);
}

#else

#include <pthread.h>

thread_t thread_create (void (*f)(void*),void* p){
  thread_t t;
  pthread_create(&t,0,(void * (*)(void *))f,p);
  return t;
}

void thread_release(thread_t *t) {
  /* noop */
}

int thread_join(thread_t *t, unsigned timeout_ms) {
  pthread_join(*t,0);
}

mutex_t mutex_create () {
  mutex_t m= PTHREAD_MUTEX_INITIALIZER;
  return m;
}

void mutex_release(mutex_t *m) {
  pthread_mutex_destroy(m);
}

void mutex_lock(mutex_t *m) {
  pthread_mutex_lock(m);
}

void mutex_unlock(mutex_t *m) {
  pthread_mutex_unlock(m);
}

condition_t condition_create() {
  condition_t c=PTHREAD_COND_INITIALIZER;
  return c;
}

void condition_release(condition_t *c) {
  pthread_cond_destroy(c);
}

int condition_wait(condition_t *c,mutex_t *m,unsigned timeout_ms) {
  pthread_cond_wait(c,m);
}

void condition_notify(condition_t* c) {
  pthread_cond_signal(c);
}

#endif
