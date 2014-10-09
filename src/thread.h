#pragma once

#if _MSC_VER
#define WIN32_LEAN_AND_MEAN
#include <windows.h>

typedef HANDLE             thread_t;
typedef SRWLOCK            mutex_t;
typedef CONDITION_VARIABLE condition_t;

#else
#error TODO
#endif

thread_t    thread_create (void (*f)(void*),void*);
void        thread_release(thread_t*);
int         thread_join   (thread_t*, unsigned timeout_ms);

mutex_t     mutex_create ();
void        mutex_release(mutex_t*);
void        mutex_lock   (mutex_t*);
void        mutex_unlock (mutex_t*);

condition_t condition_create ();
void        condition_release(condition_t*);
int         condition_wait   (condition_t*,mutex_t*,unsigned timeout_ms);
void        condition_notify (condition_t*);