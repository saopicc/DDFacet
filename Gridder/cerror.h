/**
 * Adds cexcept error handler which prints out to stderr and exits halts execution
 */
#pragma once
#include <stdlib.h>
#include <stdio.h>
#define cexcept(msg)({fprintf(stderr, msg); exit(EXIT_FAILURE);})

