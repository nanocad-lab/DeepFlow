/*
 * delveMatrix.c - Implementation of DELVE matrix facility.
 *
 * Copyright (c) 1996 by The University of Toronto.
 * 
 * See the file "copyright" for information on usage and redistribution
 * of this file, and for a DISCLAIMER OF ALL WARRANTIES.
 * 
 * Author: Delve (delve@cs.toronto.edu)
 */

#ifndef lint
static char rcsid[] = "$Id: delveMatrix.c,v 1.3.2.3 1996/11/12 16:54:07 revow Exp $" ;
#endif


#include <stdio.h>
#include <tcl.h>
#include "delve.h"

static void	CleanUpMatrix _ANSI_ARGS_((struct matrix_t *matrixPtr)) ;
static struct matrix_t * GetMatrix _ANSI_ARGS_((Tcl_Interp * interp,
			Tcl_HashTable *	matrixTablePtr, CONST char *handle)) ;
static int	MatrixCreate _ANSI_ARGS_((Tcl_Interp *interp,
			Tcl_HashTable * matrixTablePtr, int rows, int cols)) ;
static int	MatrixDelete _ANSI_ARGS_((Tcl_Interp *interp,
			Tcl_HashTable * matrixTablePtr, char *handle)) ;
static int	MatrixSize _ANSI_ARGS_((Tcl_Interp *interp,
			Tcl_HashTable * matrixTablePtr, char *handle)) ;
static int	MatrixCopy _ANSI_ARGS_((Tcl_Interp *interp,
			Tcl_HashTable * matrixTablePtr, char *dstHandle, 
			char *srcHandle)) ;
static int	MatrixSet _ANSI_ARGS_((Tcl_Interp *interp,
			Tcl_HashTable * matrixTablePtr, char *handle, 
			char *valueString));
static int	MatrixSetEntry _ANSI_ARGS_((Tcl_Interp *interp,
			Tcl_HashTable * matrixTablePtr,char *handle, int row,
			int column, char *valueString)) ;
static int	MatrixGetEntry _ANSI_ARGS_((Tcl_Interp *interp,
			Tcl_HashTable * matrixTablePtr, char *handle, int row,
			int column)) ;
static int	MatrixCmd _ANSI_ARGS_((ClientData dummy, Tcl_Interp *interp,
			int argc, char **argv));
static void	MatrixCleanUp _ANSI_ARGS_((ClientData clientData,
			Tcl_Interp *interp)) ;


/*
 *-----------------------------------------------------------------------------
 * CleanUpMatrix --
 *
 *   Release all resources allocated to the specified matrix.  Doesn't
 * free the table entry.
 *-----------------------------------------------------------------------------
 */
static void
CleanUpMatrix (matrixPtr)
struct matrix_t *matrixPtr;
{
    int		idx ;

    for (idx = 0 ; idx < matrixPtr->rows ; ++idx) {
	ckfree((char *)matrixPtr->x[idx]) ;
    }
    ckfree((char *)matrixPtr->x) ;
    ckfree((char *)matrixPtr) ;
}


/*
 *-----------------------------------------------------------------------------
 * MatrixCreate --
 *
 *   Create a new matrix, implements the subcommand:
 *         matrix create rows cols
 *-----------------------------------------------------------------------------
 */
static int
MatrixCreate (interp, matrixTablePtr, rows, cols)
Tcl_Interp *	interp;
Tcl_HashTable *	matrixTablePtr;
int		rows ;
int		cols ;
{
    char	handle[200] ;
    int		id ;
    int		new ;
    Tcl_HashEntry *	entryPtr ;
    struct matrix_t *	matrixPtr;

    double **	x ;
    int		i ;
    int		j ;

    /*
     * Make sure that the row and column sizes are in the right range.
     */

    if (rows <= 0) {
	char	buffer[200] ;
	sprintf(buffer, "expected row size greater than 0, but got %d", 
		rows) ;
	Tcl_SetResult(interp, buffer, TCL_VOLATILE) ;
	return TCL_ERROR ;
    }

    if (cols <= 0) {
	char	buffer[200] ;
	sprintf(buffer, "expected column size greater than 0, but got %d", 
		cols) ;
	Tcl_SetResult(interp, buffer, TCL_VOLATILE) ;
	return TCL_ERROR ;
    }

    /*  
     * Allocate all the memory, and set everything to zero.
     */

    matrixPtr = (struct matrix_t *) ckalloc (sizeof (struct matrix_t));

    x = (double **)ckalloc(rows*sizeof(double *)) ;
    for (i = 0 ; i < rows ; ++i) {
	x[i] = (double *)ckalloc(cols*sizeof(double)) ;
	for (j = 0 ; j < cols ; ++j) {
	    x[i][j] = 0.0 ;
	}
    }

    matrixPtr->x    = x ;
    matrixPtr->rows = rows ;
    matrixPtr->cols = cols ;

    /*
     * Add the matrix into the handle table, and return the handle in
     * the interpreters result.  Only allow 65536 matrices at one time
     * (an arbitrary decision).
     */

    for (new = 0, id = 0 ; new == 0 && id < 65536 ; ++id) {
	sprintf(handle, "matrix%d", id) ;
	entryPtr = Tcl_CreateHashEntry(matrixTablePtr, handle, &new) ;
    }

    if (new == 0) {
	CleanUpMatrix(matrixPtr) ;
	Tcl_AppendResult(interp, "too many open matrices",  (char *)NULL) ;
        return TCL_ERROR ;
    }
    
    Tcl_SetHashValue(entryPtr, (char *)matrixPtr) ;

    Tcl_SetResult (interp, Tcl_GetHashKey(matrixTablePtr, entryPtr),
		   TCL_STATIC);
    return TCL_OK;
}


/*
 *-----------------------------------------------------------------------------
 * MatrixDelete --
 *
 *   Deletes the specified matrix, implements the subcommand:
 *         matrix delete handle
 *-----------------------------------------------------------------------------
 */
static int
MatrixDelete (interp, matrixTablePtr, handle)
Tcl_Interp *	interp;
Tcl_HashTable *	matrixTablePtr;
char *		handle;
{
    Tcl_HashEntry *	entryPtr ;
    struct matrix_t *	matrixPtr = GetMatrix (interp, matrixTablePtr, handle);

    if (matrixPtr == NULL) {
        return TCL_ERROR;
    }

    entryPtr = Tcl_FindHashEntry(matrixTablePtr, handle) ;

    CleanUpMatrix(matrixPtr) ;
    Tcl_DeleteHashEntry (entryPtr);

    return TCL_OK;
}


/*
 *-----------------------------------------------------------------------------
 * MatrixSize --
 *
 *   Prints the number of rows and columns in the matrix into the
 *   interpreters of the result. Implements the subcommand:
 *	matrix size
 *
 *-----------------------------------------------------------------------------
 */
static int
MatrixSize (interp, matrixTablePtr, handle)
Tcl_Interp *	interp;
Tcl_HashTable *	matrixTablePtr;
char *		handle;
{
    matrix_t *	matrixPtr = GetMatrix (interp, matrixTablePtr, handle);

    if (matrixPtr == NULL) {
        return TCL_ERROR;
    }

    sprintf(interp->result, "%d %d", matrixPtr->rows, matrixPtr->cols) ;

    return TCL_OK;
}


/*
 *-----------------------------------------------------------------------------
 * MatrixCopy --
 *
 *   Copies the values in one matrix to another. Implements the
 *   subcommand:
 *	matrix copy dstHandle srcHandle
 *
 *-----------------------------------------------------------------------------
 */
static int
MatrixCopy (interp, matrixTablePtr, dstHandle, srcHandle)
Tcl_Interp *	interp;
Tcl_HashTable *	matrixTablePtr;
char *		srcHandle;
char *		dstHandle;
{
    matrix_t *	matrixPtr[2];
    int		i ;
    int		j ;

    matrixPtr[0]  = GetMatrix (interp, matrixTablePtr, dstHandle);
    if (matrixPtr[0] == NULL) {
        return TCL_ERROR;
    }

    matrixPtr[1]  = GetMatrix (interp, matrixTablePtr, srcHandle);
    if (matrixPtr[1] == NULL) {
        return TCL_ERROR;
    }

    if (matrixPtr[1]->rows != matrixPtr[0]->rows) {
	Tcl_AppendResult(interp, "matrices have different number of rows",
			 (char *) NULL) ;
	return TCL_ERROR;
    }

    if (matrixPtr[1]->cols != matrixPtr[0]->cols) {
	Tcl_AppendResult(interp, "matrices have different number of columns",
			 (char *) NULL) ;
	return TCL_ERROR;
    }

    for (i = 0 ; i < matrixPtr[0]->rows ; ++i) {
	for (j = 0 ; j < matrixPtr[0]->cols ; ++j) {
	    matrixPtr[0]->x[i][j] = matrixPtr[1]->x[i][j] ;
	}
    }

    return TCL_OK;
}


/*
 *-----------------------------------------------------------------------------
 * MatrixSet --
 *
 *   Sets all elements in the matrix to a single value. Implements the
 *   subcommand:
 *	matrix set value
 *
 *-----------------------------------------------------------------------------
 */
static int
MatrixSet (interp, matrixTablePtr, handle, valueString)
Tcl_Interp *	interp;
Tcl_HashTable *	matrixTablePtr;
char *		handle;
char *		valueString ;
{
    matrix_t *	matrixPtr = GetMatrix (interp, matrixTablePtr, handle);
    double	value ;
    int		i ;
    int		j ;

    if (matrixPtr == NULL) {
        return TCL_ERROR;
    }

    if (Tcl_GetDouble(interp, valueString, &value) != TCL_OK) {
	Tcl_AddErrorInfo(interp,
		 "\n    (while parsing value to set all matrix elements to)") ;
	return TCL_ERROR ;
    }

    for (i = 0 ; i < matrixPtr->rows ; ++i) {
	for (j = 0 ; j < matrixPtr->cols ; ++j) {
	    matrixPtr->x[i][j] = value ;
	}
    }

    return TCL_OK;
}


/*
 *-----------------------------------------------------------------------------
 * MatrixSetEntry --
 *
 *   Set an entry in the matrix from a string. Implements the subcommand:
 * 	matrix entryset i j val
 *-----------------------------------------------------------------------------
 */
static int
MatrixSetEntry (interp, matrixTablePtr, handle, row, col, valueString)
Tcl_Interp *	interp;
Tcl_HashTable *	matrixTablePtr;
char *		handle;
int		row ;
int		col ;
char *		valueString ;
{
    matrix_t *	matrixPtr  = GetMatrix (interp, matrixTablePtr, handle);
    char	buffer[200] ;
    double	value ;

    if (matrixPtr == NULL) {
        return TCL_ERROR;
    }

    /*
     * First make sure that the row and column indices are in the
     * right range.
     */

    if (row < 0 || row >= matrixPtr->rows) {
	sprintf(buffer, "expected row index between 0 and %d, but got %d", 
		matrixPtr->rows - 1, row) ;
	Tcl_SetResult(interp, buffer, TCL_VOLATILE) ;
	Tcl_AddErrorInfo(interp, "\n    (while setting element for matrix)") ;
	return TCL_ERROR ;
    }

    if (col < 0 || col >= matrixPtr->cols) {
	sprintf(buffer, "expected column index between 0 and %d, but got %d", 
		matrixPtr->cols - 1, col) ;
	Tcl_SetResult(interp, buffer, TCL_VOLATILE) ;
	Tcl_AddErrorInfo(interp, "\n    (while setting element for matrix)") ;
	return TCL_ERROR ;
    }

    /*
     * Parse the value to set the element to, and set it.
     */

    if (Tcl_GetDouble(interp, valueString, &value) != TCL_OK) {
	sprintf(buffer,
		"\n    (while parsing element for matrix entry %d,%d)",
		row, col) ;
	Tcl_AddErrorInfo(interp, buffer) ;
	return TCL_ERROR ;
    }

    matrixPtr->x[row][col] = value ;

    return TCL_OK;
}


/*
 *-----------------------------------------------------------------------------
 * MatrixGetEntry --
 *
 *   Returns a string representation of the value of matrix[row][col]
 *   in the interpreters result. 
 *   Implements the subcommand: 
 * 	matrix entryset i j 
 *-----------------------------------------------------------------------------
 */

static int
MatrixGetEntry (interp, matrixTablePtr, handle, row, col)
Tcl_Interp *	interp;
Tcl_HashTable *	matrixTablePtr;
char *		handle;
int		row ;
int		col ;
{
    matrix_t *	matrixPtr  = GetMatrix (interp, matrixTablePtr, handle);
    char	buffer[200] ;
    char	dblBuf[TCL_DOUBLE_SPACE] ;

    if (matrixPtr == NULL) {
        return TCL_ERROR;
    }

    /*
     * First make sure that the row and column indices are in the
     * right range.
     */

    if (row < 0 || row >= matrixPtr->rows) {
	sprintf(buffer, "expected row index between 0 and %d, but got %d", 
		matrixPtr->rows - 1, row) ;
	Tcl_SetResult(interp, buffer, TCL_VOLATILE) ;
	Tcl_AddErrorInfo(interp, "\n    (while setting element for matrix)") ;
	return TCL_ERROR ;
    }

    if (col < 0 || col >= matrixPtr->cols) {
	sprintf(buffer, "expected column index between 0 and %d, but got %d", 
		matrixPtr->cols - 1, col) ;
	Tcl_SetResult(interp, buffer, TCL_VOLATILE) ;
	Tcl_AddErrorInfo(interp, "\n    (while setting element for matrix)") ;
	return TCL_ERROR ;
    }

    /*
     * Print the value into the interpreter's result, and return ;
     */

    Tcl_PrintDouble(interp, matrixPtr->x[row][col], dblBuf) ;
    Tcl_SetResult(interp, dblBuf, TCL_VOLATILE) ;

    return TCL_OK;
}


/*
 *-----------------------------------------------------------------------------
 * MatrixCmd --
 *
 *    The procedure for the "matrix" command.
 *-----------------------------------------------------------------------------
 */
static int
MatrixCmd(clientData, interp, argc, argv)
ClientData 	clientData;	/* Handle table pointer. */
Tcl_Interp *	interp;		/* Current interpreter. */
int 		argc;		/* Number of arguments. */
char **		argv;		/* Argument strings. */
{
    char *	option ;
    char	c ;

    if (argc < 2) {
	Tcl_AppendResult(interp, "wrong # args: should be \"",
			 argv[0], " option ?arg ...?\"", (char *) NULL) ;
	return TCL_ERROR;
    }

    option = argv[1] ;
    c = option[0] ;

    if (c == 'c' && strcmp(option, "copy") == 0) {
	if (argc != 4) {
	    Tcl_AppendResult(interp, "wrong # args: should be \"",
			     argv[0],
			     " copy destMatrixHandle srcMatrixHandle\"", 
			     (char *) NULL);
	    return TCL_ERROR;
	}
	if (MatrixCopy(interp, (Tcl_HashTable *) clientData, 
		       argv[2], argv[3]) != TCL_OK) {
	    return TCL_ERROR ;
	}

    } else if (c == 'c' && strcmp(option, "create") == 0) {
	int	rows ;
	int	cols ;

	if (argc != 4) {
	    Tcl_AppendResult(interp, "wrong # args: should be \"",
			     argv[0], " create rows columns\"", (char *) NULL);
	    return TCL_ERROR;
	}
	if (Tcl_GetInt(interp, argv[2], &rows) != TCL_OK) {
	    return TCL_ERROR ;
	}
	if (Tcl_GetInt(interp, argv[3], &cols) != TCL_OK) {
	    return TCL_ERROR ;
	}
	if (MatrixCreate(interp,
			 (Tcl_HashTable *) clientData, rows, cols) != TCL_OK) {
	    return TCL_ERROR ;
	}

    } else if (c == 'd' && strcmp(option, "delete") == 0) {
	if (argc != 3) {
	    Tcl_AppendResult(interp, "wrong # args: should be \"",
			     argv[0], " delete matrixHandle\"", (char *) NULL);
	    return TCL_ERROR;
	}
	if (MatrixDelete(interp,
			 (Tcl_HashTable *) clientData, argv[2]) != TCL_OK) {
	    return TCL_ERROR ;
	}
	
    } else if (c == 'e' && strcmp(option, "entryset") == 0) {
	int	 row ;
	int	 column ;

	if (argc != 5 && argc != 6) {
	    Tcl_AppendResult(interp, "wrong # args: should be \"",
			     argv[0], 
			     " entryset matrixHandle row column ?value?\"", 
			     (char *) NULL) ;
	    return TCL_ERROR;
	}

	if (Tcl_GetInt(interp, argv[3], &row) != TCL_OK) {
	    return TCL_ERROR ;
	}
	if (Tcl_GetInt(interp, argv[4], &column) != TCL_OK) {
	    return TCL_ERROR ;
	}
	if (argc == 6 
	    && MatrixSetEntry(interp, (Tcl_HashTable *) clientData, 
			      argv[2], row, column, argv[5]) != TCL_OK) {
	    return TCL_ERROR ;
	}
	if (MatrixGetEntry(interp, (Tcl_HashTable *) clientData, 
			    argv[2], row, column) != TCL_OK) {
	    return TCL_ERROR ;
	}
	
    } else if (c == 's' && strcmp(option, "set") == 0) {
	if (argc != 4) {
	    Tcl_AppendResult(interp, "wrong # args: should be \"",
			     argv[0], " set matrixHandle value\"",
			     (char *) NULL) ;
	    return TCL_ERROR;
	}
	if (MatrixSet(interp, (Tcl_HashTable *) clientData, 
		      argv[2], argv[3]) != TCL_OK) {
	    return TCL_ERROR ;
	}
	
    } else if (c == 's' && strcmp(option, "size") == 0) {
	if (argc != 3) {
	    Tcl_AppendResult(interp, "wrong # args: should be \"",
			     argv[0], " size matrixHandle\"", (char *) NULL);
	    return TCL_ERROR;
	}
	if (MatrixSize(interp,
		       (Tcl_HashTable *) clientData, argv[2]) != TCL_OK) {
	    return TCL_ERROR ;
	}
	
    } else {
	Tcl_AppendResult(interp, "bad option \"",
			 option, "\": should be copy, create, ",
			 "delete, entryset, set, or size", (char *) NULL) ;
	return TCL_ERROR;
    }

    return TCL_OK ;
}


/*
 *-----------------------------------------------------------------------------
 * MatrixCleanUp --
 *
 *    Called when the interpreter is deleted to cleanup all matrix
 * resources
 *-----------------------------------------------------------------------------
 */
static void
MatrixCleanUp (clientData, interp)
ClientData	clientData ;
Tcl_Interp * 	interp;
{
    while (1) {
	Tcl_HashSearch 	search ;
	Tcl_HashEntry *	entryPtr ;
    
	entryPtr = Tcl_FirstHashEntry((Tcl_HashTable *)clientData, &search) ;
        if (entryPtr == NULL) {
            break;
	}
        CleanUpMatrix ((struct matrix_t *)Tcl_GetHashValue(entryPtr));
    }
    Tcl_DeleteHashTable ((Tcl_HashTable *) clientData);

    ckfree((char *) clientData) ;
}


/*
 *-----------------------------------------------------------------------------
 *  GetMatrix --
 *
 *    Looks up a matrix in the hash table given it's handle
 *-----------------------------------------------------------------------------
 */
static struct matrix_t *
GetMatrix (interp, matrixTablePtr, handle)
Tcl_Interp *	interp ;
Tcl_HashTable *	matrixTablePtr;
CONST char *	handle ;
{
    Tcl_HashEntry *	entryPtr ;
    char *		nonConstHandle ;

    nonConstHandle = (char *)ckalloc((1+strlen(handle))*sizeof(char)) ;
    strcpy(nonConstHandle, handle) ;
    entryPtr = Tcl_FindHashEntry(matrixTablePtr, nonConstHandle) ;
    ckfree((char *)nonConstHandle) ;

    if (entryPtr == NULL) {
	Tcl_AppendResult(interp, "invalid matrix handle \"", handle, "\"",
			 (char *)NULL) ;
        return (struct matrix_t *)NULL;
    }

    return (struct matrix_t *)Tcl_GetHashValue(entryPtr) ;
}


/*
 *-----------------------------------------------------------------------------
 *  Delve_InitMatrix --
 *
 *    Initialize the DELVE matrix facility.
 *-----------------------------------------------------------------------------
 */
int
Delve_InitMatrix (interp)
    Tcl_Interp *interp;
{
    Tcl_HashTable  *matrixTablePtr;

    matrixTablePtr = (Tcl_HashTable *)ckalloc(sizeof(Tcl_HashTable)) ;

    Tcl_InitHashTable(matrixTablePtr, TCL_STRING_KEYS) ;

    Tcl_CallWhenDeleted (interp, MatrixCleanUp, (ClientData) matrixTablePtr);

    /*
     * Initialize the commands.
     */

    Tcl_CreateCommand (interp, "d_matrix", MatrixCmd, 
                       (ClientData) matrixTablePtr, (void (*)()) NULL);

    return TCL_OK ;
}

/*
 *-----------------------------------------------------------------------------
 *  Delve_GetMatrix --
 *
 *    Finds a matrix given it's handle
 *-----------------------------------------------------------------------------
 */
int
Delve_GetMatrix (interp, handle, matrixPtrPtr)
Tcl_Interp *	interp ;
CONST char *	handle ;
matrix_t **	matrixPtrPtr ;
{
    Tcl_HashTable *	matrixTablePtr;
    Tcl_HashEntry *	entryPtr ;
    Tcl_CmdInfo		info ;
    struct matrix_t *	matrixPtr ;
    
    if (Tcl_GetCommandInfo(interp, "d_matrix", &info) == 0
	|| info.proc != MatrixCmd) {
	Tcl_SetResult(interp, "can't find \"d_matrix\" command in interpreter",
		      TCL_VOLATILE) ;
	return TCL_ERROR ;
    }

    matrixPtr = GetMatrix(interp, (Tcl_HashTable *) info.clientData, handle) ;

    if (matrixPtr == NULL) {
	return TCL_ERROR ;
    }

    *matrixPtrPtr = (matrix_t *)matrixPtr ;
    return TCL_OK ;
}


