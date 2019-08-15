/*
 * delveStats.c - Implementation of DELVE "d_mstats" command.
 *
 * Copyright (c) 1996 by The University of Toronto.
 * 
 * See the file "copyright" for information on usage and redistribution
 * of this file, and for a DISCLAIMER OF ALL WARRANTIES.
 * 
 * Author: Delve (delve@cs.toronto.edu) 
 */

#ifndef lint
static char rcsid[] = "$Id: delveStats.c,v 1.6.2.1 1996/11/12 16:54:08 revow Exp $" ;
#endif

#include "delve.h"
#include <math.h>

static int      compare _ANSI_ARGS_((CONST void * a, CONST void * b)) ;


/*
 *-----------------------------------------------------------------------------
 * MatrixStatsCmd --
 *
 *    The procedure for the "d_mstats" command. Calculates the mean,
 *    variance, median, and average absolute deviation from the median
 *    for a row or column of a matrix.
 *-----------------------------------------------------------------------------
 */

int
MatrixStatsCmd(dummy, interp, argc, argv)
ClientData 	dummy;		/* unused */
Tcl_Interp *	interp;		/* Current interpreter. */
int 		argc;		/* Number of arguments. */
char **		argv;		/* Argument strings. */
{
    char	dblBuf[TCL_DOUBLE_SPACE] ;
    double	aveAbsDev ;
    double	mean ;
    double	median ;
    double	sum ;
    double	variance ;
    double *	y ;
    int		constIdx ;
    int		dimension ;
    int		doRow ;
    int		idx ;

    matrix_t *	matrixPtr ;

    /*
     * A whole lotta parsing... 
     */

    if (argc != 4) {
	Tcl_AppendResult(interp, "wrong # args: should be \"",
			 argv[0], " -column n matrixHandle\" or \"", 
			 argv[0], " -row n matrixHandle\"",
			 (char *) NULL) ;
	return TCL_ERROR ;
    }

    if (strcmp(argv[1], "-row") == 0) {
	doRow = 1 ;
    } else if (strcmp(argv[1], "-col") == 0 
	       || strcmp(argv[1], "-column") == 0) {
	doRow = 0 ;
    } else {
	Tcl_AppendResult(interp, "bad option \"", argv[1],
			 "\": should be \"-row\" or \"-column\"",
			 (char *) NULL) ;
	return TCL_ERROR ;
    }

    if (Tcl_GetInt(interp, argv[2], &constIdx) != TCL_OK) {
	return TCL_ERROR ;
    }

    if (Delve_GetMatrix(interp, argv[3], &matrixPtr) != TCL_OK) {
	return TCL_ERROR ;
    }

    /*
     * Make sure that the row/column the user wants actually exists.
     */

    dimension = doRow ? matrixPtr->rows : matrixPtr->cols ;
    if (constIdx < 0 || constIdx >= dimension) {
	char	buffer[200] ;
	sprintf(buffer, "expected integer between 0 and %d but got %d", 
		dimension - 1, constIdx) ;
	Tcl_SetResult(interp, buffer, TCL_VOLATILE) ;
	return TCL_ERROR ;
    }

    /*
     * Initialize the local variables that are used over and over.
     * Copy the appropriate row/column into a private vector. This
     * costs, but we need a private arry to find the median (we have
     * to sort it), so why not use it for everything else?
     */

    dimension = doRow ? matrixPtr->cols : matrixPtr->rows ;

    y = (double *)ckalloc(dimension*sizeof(double)) ;
    if (doRow) {
	for (idx = 0 ; idx < dimension ; ++idx) {
	    y[idx] = matrixPtr->x[constIdx][idx] ;
	}
    } else {
	for (idx = 0 ; idx < dimension ; ++idx) {
	    y[idx] = matrixPtr->x[idx][constIdx] ;
	}
    }
    

    /*
     * Calculate the mean of the vector and append it to the result.
     */

    sum = 0.0 ;
    for (idx = 0 ; idx < dimension ; ++idx) {
	sum += y[idx] ;
    }
    mean = sum/dimension ;

    Tcl_PrintDouble(interp, mean, dblBuf) ;
    Tcl_AppendElement(interp, dblBuf) ;

    /*
     * Calculate the variance of the row/col and append it to the
     * result. If the dimension is less than two we can't do the
     * calculation, so just append an empty string.
     */

    if (dimension > 1) {
	sum = 0.0 ;
	for (idx = 0 ; idx < dimension ; ++idx) {
	    sum += (y[idx] - mean)*(y[idx] - mean) ;
	}
	variance = sum/(dimension - 1) ;

	Tcl_PrintDouble(interp, variance, dblBuf) ;
	Tcl_AppendElement(interp, dblBuf) ;

    } else {
	Tcl_AppendElement(interp, "") ;
    }

    /*
     * Calculate the median by sorting the vector (not the fastest
     * method, but at least it works), and append it to the result.
     */

    qsort (y, dimension, sizeof(*y), compare);

    if (dimension % 2 == 1) {
	median = y[dimension/2] ;
    } else {
	median = (y[(dimension/2) - 1] + y[dimension/2])/2.0 ;
    }

    Tcl_PrintDouble(interp, median, dblBuf) ;
    Tcl_AppendElement(interp, dblBuf) ;

    /*
     * Calculate the average absolute deviation from the median, and
     * append it to the result. As with the variance, it can't be
     * calculated if there are fewer than 2 values.
     */

    if (dimension > 1) {
	sum = 0.0 ;
	for (idx = 0 ; idx < dimension ; ++idx) {
	    sum += fabs(y[idx] - median) ;
	}
	aveAbsDev = sum/(dimension - 1) ;

	Tcl_PrintDouble(interp, aveAbsDev, dblBuf) ;
	Tcl_AppendElement(interp, dblBuf) ;

    } else {
	Tcl_AppendElement(interp, "") ;
    }

    /*
     * Finally, release the vector and return TCL_OK.
     */

    ckfree((char *)y) ;

    return TCL_OK ;
}


/*
 *-----------------------------------------------------------------------------
 * compare --
 *
 * Compares two double precision values, and returns -1, 0, or 1
 * depending on whether the first is less than, equal to, or greater
 * than the second.
 *
 *-----------------------------------------------------------------------------
 */

static int
compare (a, b)
CONST void *	a ;
CONST void *	b ;
{
    return *(double*)a > *(double*)b ? 1 : *(double*)a < *(double*)b ? -1 : 0 ;
}
