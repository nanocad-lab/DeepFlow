
/*
 * delve.h - Header file containing declarations shared within the
 * DELVE source code.
 *
 * Copyright (c) 1996 by The University of Toronto.
 * 
 * See the file "copyright" for information on usage and redistribution
 * of this file, and for a DISCLAIMER OF ALL WARRANTIES.  
 *
 * Author: Delve (delve@cs.toronto.edu)
 * 
 * $Id: delve.h,v 1.7.2.4.2.2 1997/11/27 16:42:56 revow Exp $
 */

#ifndef DELVE_H
#define DELVE_H

#include <stdio.h>
#include <tcl.h>

#ifdef STDC_HEADERS
#include <stdlib.h>
#endif

#define DELVE_VERSION "1.1"
#define DELVE_MAJOR_VERSION 1
#define DELVE_MINOR_VERSION 1

typedef CONST struct matrix_t {
    double **	x ;
    int		rows ;
    int		cols ;
} matrix_t ;

EXTERN int	Delve_GetMatrix _ANSI_ARGS_((Tcl_Interp *interp,
			CONST char *handle, matrix_t **matrixPtrPtr));
EXTERN int	Delve_InitAnova _ANSI_ARGS_((Tcl_Interp *interp)) ;
EXTERN int	Delve_InitAttr _ANSI_ARGS_((Tcl_Interp *interp)) ;
EXTERN int	Delve_InitMatrix _ANSI_ARGS_((Tcl_Interp *interp)) ;
EXTERN int	Delve_InitRandom _ANSI_ARGS_((Tcl_Interp *interp)) ;
EXTERN int	Delve_RandomNumber _ANSI_ARGS_((Tcl_Interp *interp,
			double * doublePtr)) ;
EXTERN int	Delve_RandomSeed _ANSI_ARGS_((Tcl_Interp *interp,
			long int seed)) ;
EXTERN int	MatrixStatsCmd _ANSI_ARGS_((ClientData dummy, 
			Tcl_Interp * interp, int argc, char **argv));

#endif /* DELVE_H */
