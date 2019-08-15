/*
 * delveAttr.c - Implementation of DELVE attr facility.
 *
 * Copyright (c) 1996 by The University of Toronto.
 * 
 * See the file "copyright" for information on usage and redistribution
 * of this file, and for a DISCLAIMER OF ALL WARRANTIES.
 * 
 * Author: Delve (delve@cs.toronto.edu)
 */

#ifndef lint
static char rcsid[] = "$Id: delveAttr.c,v 1.2.2.4 1996/11/12 16:54:07 revow Exp $" ;
#endif

#include <stdio.h>
#include <math.h>
#include <tcl.h>
#include "delve.h"

#ifndef M_PI
#define M_PI	3.14159265358979323846
#endif

/*
 * All the different types of attributes that we know about.
 */

typedef enum Attr_Type_t { 
    ATTR_ANGULAR=0,		/* Angular variables have a circular nature,
				 * in that only the value modulus some length
				 * is thought to be important. */
    ATTR_BINARY=1,		/* Binary variables are ones that take on
				 * exactly two values. */
    ATTR_NOMINAL=2,		/* Nominal values have a finite range of
				 * values and are unordered. */
    ATTR_ORDINAL=3,		/* Ordinal values have a finite range of
				 * values, and are ordered. */
    ATTR_INTEGER=4,		/* Integer values are just that: integer */
    ATTR_REAL=5			/* Real variables are just that: real */
} Attr_Type_t ;

/*
 * The names of the above attribute types.  This array *must* contain as many
 * names as there are types in Attr_Type_t, and they must be in the same
 * order.  As well, the final element *must* be NULL.
 */

static char *	attrTypeName[] = {
    "angular",
    "binary",
    "nominal",
    "ordinal",
    "integer",
    "real",
    (char *)NULL
} ;

/*
 * All the different types of encoding methods that we know about.
 */

typedef enum Attr_Method_t { 
    ATTR_COPY=0,		/* Copies the value straight through. */
    ATTR_IGNORE=1,		/* Converts the value to an empty string. */
    ATTR_BINARY_SYMMETRIC=2,	/* Only valid for binary types, converts to
				 * -1,+1. */
    ATTR_BINARY_PASSIVE=3,	/* Only valid for binary types, converts to
				 * 0,1. Requires a passive value. */
    ATTR_ZERO_UP=4,		/* Converts nominal or ordinal variables to an
				 * integer index starting at zero. */
    ATTR_ONE_UP=5,		/* Converts nominal or ordinal variables to an
				 * integer index starting at one. */
    ATTR_ONE_OF_N=6,		/* Converts nominal or ordinal variables to a
				 * list of 0/1 values with only one on. May
				 * have a passive value. */
    ATTR_THERM=7,		/* Converts nominal or ordinal variables to a
				 * list of -1/+1 values, the number +1 being
				 * the index of the value. May have a passive
				 * value. */
    ATTR_NORM=8,		/* Normalizes integer or real variables. */
    ATTR_RECTAN=9		/* Encodes angular variables as cos and sin of
				 * the value. Requires a unit value. */
} Attr_Method_t ;

/*
 * The names of the above encoding methods.  This array *must* contain as many
 * names as there are types in Attr_Method_t, and they must be in the same
 * order.  As well, the final element *must* be NULL.
 */

static char *	attrMethodName[] = {
    "copy",
    "ignore",
    "binary-symmetric",
    "binary-passive",
    "zero-up",
    "one-up",
    "one-of-n",
    "thermometer",
    "normalized",
    "rectan",
    (char *)NULL
} ;

/*
 * Structure used to hold information about variable traces:
 */

typedef struct Attr_t {
    Attr_Type_t		type ;		/* The type of attribute. */
    Attr_Method_t	method ; 	/* How the attribute is to be
					 * encoded. */
    int			rangeArgc ; 	/* If the allowed attribute values are
					 * finite, the number of allowed
					 * values; zero otherwise. */
    char **		rangeArgv ; 	/* If the allowed attribute values are
					 * finite, this contains those values;
					 * NULL otherwise. */
    char **		crangeArgv ; 	/* If the allowed attribute values are
					 * finite, this contains the encoded
					 * representations of those values;
					 * NULL otherwise */
    char *		passive ; 	/* If there is a passive value, this
					 * is it; NULL otherwise. */
    double		mu ;		/* For normalization encoding, this is
					 * the mean. */
    double		sigma ;		/* For normalization encoding, this is
					 * the std deviation. */
    double		unit ;		/* For angular encoding, this is the
					 * unit value. */
    double		scale ;		/* For thermometer encoding,
					   this is scale to divide the
					   values by. */
} Attr_t ;

/*
 * Forward declarations for procedures defined in this file:
 */

static void	AttrCleanUp _ANSI_ARGS_((ClientData clientData,
			Tcl_Interp *interp)) ;
static int	AttrCmd _ANSI_ARGS_((ClientData dummy, Tcl_Interp *interp,
			int argc, char **argv));
static int	AttrCodeRange _ANSI_ARGS_((Tcl_Interp *interp,
			Attr_t *attrPtr)) ;
static int	AttrCreate _ANSI_ARGS_((Tcl_Interp *interp,
			Tcl_HashTable * attrTablePtr, Attr_Type_t type,
			Attr_Method_t method, char *argv0, int argc,
			char **argv)) ;
static int	AttrDecode _ANSI_ARGS_((Tcl_Interp *interp,
			Tcl_HashTable *attrTablePtr,
			char *handle, char *value)) ;
static int	AttrDelete _ANSI_ARGS_((Tcl_Interp *interp,
			Tcl_HashTable * attrTablePtr, char *handle)) ;
static int	AttrEncode _ANSI_ARGS_((Tcl_Interp *interp,
			Tcl_HashTable *attrTablePtr,
			char *handle, char *value)) ;
static int	AttrParseArgs _ANSI_ARGS_((Tcl_Interp *interp,
			Attr_t *attrPtr, Attr_Type_t type,
			Attr_Method_t method, char *argv0, int argc,
			char **argv)) ;
static void	CleanUpAttr _ANSI_ARGS_((Attr_t *attrPtr)) ;
static Attr_t * GetAttr _ANSI_ARGS_((Tcl_Interp * interp,
			Tcl_HashTable *attrTablePtr, CONST char *handle)) ;


/*
 *-----------------------------------------------------------------------------
 * CleanUpAttr --
 *
 *	Release all resources allocated to the specified attribute.
 *	Doesn't free the table entry.
 *-----------------------------------------------------------------------------
 */

static void
CleanUpAttr (attrPtr)
Attr_t *	attrPtr;
{
    if (attrPtr->passive) {
	ckfree((char *)attrPtr->passive) ;
    }
    if (attrPtr->rangeArgv) {
	ckfree((char *)attrPtr->rangeArgv) ;
    }
    if (attrPtr->crangeArgv) {
	ckfree((char *)attrPtr->crangeArgv) ;
    }
    ckfree((char *)attrPtr) ;
}


/*
 *-----------------------------------------------------------------------------
 * AttrCreate --
 *
 *	Create a new attribute. Implements the subcommand:
 *		attr create type method ?args ...?
 *-----------------------------------------------------------------------------
 */

static int
AttrCreate (interp, attrTablePtr, type, method, argv0, argc, argv)
Tcl_Interp *	interp ;
Tcl_HashTable *	attrTablePtr ;
Attr_Type_t	type ;
Attr_Method_t	method ;
char *		argv0 ;
int		argc ;
char **		argv ;
{
    char		handle[200] ;
    int			id ;
    int			new ;
    Tcl_HashEntry *	entryPtr ;
    Attr_t *		attrPtr;
    int			code ;

    /*
     * Make sure that the encoding method and type are compatible.
     */

    code = TCL_OK ;
    switch (type) {
    case ATTR_ANGULAR:
	if (method != ATTR_COPY && method != ATTR_IGNORE
	    && method != ATTR_RECTAN) {
	    code = TCL_ERROR ;
	}
	break ;
    case ATTR_BINARY:
	if (method != ATTR_COPY && method != ATTR_IGNORE
	    && method != ATTR_BINARY_PASSIVE 
	    && method != ATTR_BINARY_SYMMETRIC) {
	    code = TCL_ERROR ;
	}
	break ;
    case ATTR_NOMINAL:
    case ATTR_ORDINAL:
	if (method != ATTR_COPY && method != ATTR_IGNORE
	    && method != ATTR_ONE_OF_N 
	    && method != ATTR_ZERO_UP 
	    && method != ATTR_ONE_UP 
	    && method != ATTR_THERM) {
	    code = TCL_ERROR ;
	}
	break ;
    case ATTR_INTEGER:
    case ATTR_REAL:
	if (method != ATTR_COPY && method != ATTR_IGNORE
	    && method != ATTR_NORM) { 
	    code = TCL_ERROR ;
	}
	break ;
    }

    if (code != TCL_OK) {
	Tcl_AppendResult(interp, "can't use ", attrMethodName[method],
			 " encoding for ", attrTypeName[type],
			 " attributes", (char *)NULL) ;
	return TCL_ERROR ;
    }

    /*  
     * Allocate all the memory, and set everything to good initial values.
     */

    attrPtr = (Attr_t *) ckalloc (sizeof (Attr_t));

    attrPtr->crangeArgv = (char **)0 ;
    attrPtr->method = method ;
    attrPtr->mu = 0.0 ;
    attrPtr->passive = (char *)0 ;
    attrPtr->rangeArgc = 0 ;
    attrPtr->rangeArgv = (char **)0 ;
    attrPtr->sigma = 0.0 ;
    attrPtr->scale = 1.0 ;
    attrPtr->type = type ;
    attrPtr->unit = 0.0 ;

    /*
     * Get all of the options from "argv" and set the appropriate fields in
     * the attribute structure.  Each coding method expects different
     * arguments.
     */

    if (AttrParseArgs(interp, attrPtr,
		      type, method, argv0, argc, argv) != TCL_OK) {
	CleanUpAttr(attrPtr) ;
	return TCL_ERROR ;
    }

    /*  
     * If we're doing binary encoding, make sure we have exactly two values in
     * the range.
     */

    if ((method == ATTR_BINARY_SYMMETRIC
	 || method == ATTR_BINARY_PASSIVE) && attrPtr->rangeArgc != 2) {
	char	buf[200] ;
	sprintf(buf, "expected two values in range, but got %d",
		attrPtr->rangeArgc) ;
	Tcl_SetResult(interp, buf, TCL_VOLATILE) ;
	CleanUpAttr(attrPtr) ;
	return TCL_ERROR ;
    }

    /*  
     * If there is a passive value, make sure it's in the range.
     */

    if (attrPtr->passive != NULL) {
	int 	idx ;
	for (idx = 0 ; idx < attrPtr->rangeArgc ; ++idx) {
	    if (strcmp(attrPtr->passive, attrPtr->rangeArgv[idx]) == 0) {
		break ;
	    }
	}

	if (idx == attrPtr->rangeArgc) {
	    Tcl_AppendResult(interp, "bad passive value \"", attrPtr->passive,
			     "\": ", (char *)NULL) ;
	    if (attrPtr->rangeArgc == 0) {
		Tcl_AppendResult(interp, "range is empty", (char *)NULL) ;
	    } else if (attrPtr->rangeArgc == 1) {
		Tcl_AppendResult(interp, "should be ",
				 attrPtr->rangeArgv[0], (char *)NULL) ;
	    } else if (attrPtr->rangeArgc == 2) {
		Tcl_AppendResult(interp, "should be ",
				 attrPtr->rangeArgv[0], " or ",
				 attrPtr->rangeArgv[1], (char *)NULL) ;
	    } else {
		Tcl_AppendResult(interp, "should be", (char *)NULL) ;
		for (idx = 0 ; idx < attrPtr->rangeArgc - 1 ; ++idx) {
		    Tcl_AppendResult(interp, " ", attrPtr->rangeArgv[idx], ",",
				     (char *)NULL) ;
		}
		Tcl_AppendResult(interp, " or ", attrPtr->rangeArgv[idx],
				 (char *)NULL) ;
	    }
	    CleanUpAttr(attrPtr) ;
	    return TCL_ERROR ;
	}
    }

    /*
     * Now encode all of the values in the attribute's range.  We do this now
     * so we can just look up the encodings when we actually get around to
     * encoding/decoding a value.
     */


    if (AttrCodeRange(interp, attrPtr) != TCL_OK) {
	CleanUpAttr(attrPtr) ;
	return TCL_ERROR ;
    }
    
    /*
     * Add the attr into the handle table, and return the handle in the
     * interpreters result. Only allow 65536 matrices at one time
     * (an arbitrary decision).
     */

    for (new = 0, id = 0 ; new == 0 && id < 65536 ; ++id) {
	sprintf(handle, "attr%d", id) ;
	entryPtr = Tcl_CreateHashEntry(attrTablePtr, handle, &new) ;
    }

    if (new == 0) {
	CleanUpAttr(attrPtr) ;
	Tcl_AppendResult(interp, "too many open attributes",  (char *)NULL) ;
        return TCL_ERROR ;
    }
    
    Tcl_SetHashValue(entryPtr, (char *)attrPtr) ;

    Tcl_SetResult (interp, Tcl_GetHashKey(attrTablePtr, entryPtr),
		   TCL_STATIC);
    return TCL_OK;
}


/*
 *-----------------------------------------------------------------------------
 * AttrParseArgs --
 *
 *	Parses the command line arguments and sets the corresponding
 *	values in the attribute.
 *-----------------------------------------------------------------------------
 */

static int
AttrParseArgs (interp, attrPtr, type, method, argv0, argc, argv)
Tcl_Interp *	interp ;
Attr_t *	attrPtr ;
Attr_Type_t	type ;
Attr_Method_t	method ;
char *		argv0 ;
int		argc ;
char **		argv ;
{
    /*
     * Get all of the options from "argv".  Each coding method expects
     * different arguments.
     */

    switch (method) {
    case ATTR_COPY:
    case ATTR_IGNORE:
	if (argc != 0) {
	    Tcl_AppendResult(interp, "wrong # args: should be \"", argv0,
			     " create ", attrTypeName[type], " ",
			     attrMethodName[method], "\"", (char *)NULL) ;
	    return TCL_ERROR ;
	}
	break ;

    case ATTR_BINARY_SYMMETRIC:
    case ATTR_ZERO_UP:
    case ATTR_ONE_UP:
	if (argc != 1) {
	    Tcl_AppendResult(interp, "wrong # args: should be \"", argv0,
			     " create ", attrTypeName[type], " ",
			     attrMethodName[method], " range\"", 
			     (char *)NULL) ;
	    return TCL_ERROR ;
	}
	if (Tcl_SplitList(interp, argv[0], &attrPtr->rangeArgc,
			  &attrPtr->rangeArgv) != TCL_OK) {
	    return TCL_ERROR ;
	}
	break ;

    case ATTR_ONE_OF_N:
	if (argc != 1 && argc != 2) {
	    Tcl_AppendResult(interp, "wrong # args: should be \"", argv0,
			     " create ", attrTypeName[type], " ",
			     attrMethodName[method], " range ?passive?\"", 
			     (char *)NULL) ;
	    return TCL_ERROR ;
	}
	if (Tcl_SplitList(interp, argv[0], &attrPtr->rangeArgc,
			  &attrPtr->rangeArgv) != TCL_OK) {
	    return TCL_ERROR ;
	}
	if (argc == 2) {
	    attrPtr->passive = (char *)ckalloc(strlen(argv[1]) + 1) ;
	    strcpy(attrPtr->passive, argv[1]) ;
	}
	break ;

    case ATTR_THERM:
	if (argc != 1 && argc != 2) {
	    Tcl_AppendResult(interp, "wrong # args: should be \"", argv0,
			     " create ", attrTypeName[type], " ",
			     attrMethodName[method], " range ?scale?\"", 
			     (char *)NULL) ;
	    return TCL_ERROR ;
	}
	if (Tcl_SplitList(interp, argv[0], &attrPtr->rangeArgc,
			  &attrPtr->rangeArgv) != TCL_OK) {
	    return TCL_ERROR ;
	}
	if (attrPtr->rangeArgc < 2) {
	    Tcl_AppendResult(interp, "can't use thermometer coding with ",
			     "fewer than two values in the range",
			     (char *)NULL) ;
	    return TCL_ERROR ;
	}
	if (argc == 2) {
	    if (Tcl_GetDouble(interp, argv[1], &attrPtr->scale) != TCL_OK) {
		return TCL_ERROR ;
	    }
	}
	break ;

    case ATTR_BINARY_PASSIVE:
	if (argc != 2) {
	    Tcl_AppendResult(interp, "wrong # args: should be \"", argv0,
			     " create ", attrTypeName[type], " ",
			     attrMethodName[method], " range passive\"", 
			     (char *)NULL) ;
	    return TCL_ERROR ;
	}
	if (Tcl_SplitList(interp, argv[0], &attrPtr->rangeArgc,
			  &attrPtr->rangeArgv) != TCL_OK) {
	    return TCL_ERROR ;
	}
	attrPtr->passive = (char *)ckalloc(strlen(argv[1]) + 1) ;
	strcpy(attrPtr->passive, argv[1]) ;
	break ;

    case ATTR_NORM:
	if (argc != 2) {
	    Tcl_AppendResult(interp, "wrong # args: should be \"", argv0,
			     " create ", attrTypeName[type], " ",
			     attrMethodName[method], " mu sigma\"", 
			     (char *)NULL) ;
	    return TCL_ERROR ;
	}
	if (Tcl_GetDouble(interp, argv[0], &attrPtr->mu) != TCL_OK) {
	    return TCL_ERROR ;
	}
	if (Tcl_GetDouble(interp, argv[1], &attrPtr->sigma) != TCL_OK) {
	    return TCL_ERROR ;
	}
	break ;

    case ATTR_RECTAN:
	if (argc != 1) {
	    Tcl_AppendResult(interp, "wrong # args: should be \"", argv0,
			     " create ", attrTypeName[type], " ",
			     attrMethodName[method], " unit\"", 
			     (char *)NULL) ;
	    return TCL_ERROR ;
	}
	if (Tcl_GetDouble(interp, argv[0], &attrPtr->unit) != TCL_OK) {
	    return TCL_ERROR ;
	}
	if (attrPtr->unit <= 0) {
	    Tcl_AppendResult(interp, "bad unit value for ",
			     attrMethodName[method], " encoding \"",
			     argv[1], "\": value must be greater ",
			     "than zero", (char *)NULL) ;
	    return TCL_ERROR ;
	}
	break ;

    }
    return TCL_OK ;
}


/*
 *-----------------------------------------------------------------------------
 * AttrCodeRange --
 *
 *	Builds the array of code strings corresponding to all values in an
 *	attribute's range (attr->rangeArgv).  It stores the result in
 *	"attrPtr->crangeArgv".
 *-----------------------------------------------------------------------------
 */

static int
AttrCodeRange (interp, attrPtr)
Tcl_Interp *	interp;
Attr_t *	attrPtr ;
{
    Tcl_DString	codeList ;
    char **	argv ;
    int		argc ;
    int		idx ;
    int		on ;
    
    /*
     * If there is no range, there is no coded range either.
     */

    if (attrPtr->rangeArgc == 0) {
	attrPtr->crangeArgv = (char **)0 ;
	return TCL_OK ;
    }

    /*
     * If there is a range, then we encode it based on whatever method
     * we're using. We create a list in "codeList" that holds all of
     * the codes. Generate an error if we can't handle ranges for a
     * particular coding method.
     */

    Tcl_DStringInit(&codeList) ;

    switch (attrPtr->method) {
    case ATTR_BINARY_SYMMETRIC:
	Tcl_DStringAppend(&codeList, "-1 1", 4) ;
	break ;
    case ATTR_BINARY_PASSIVE:
	if (strcmp(attrPtr->rangeArgv[0], attrPtr->passive) == 0) {
	    Tcl_DStringAppend(&codeList, "0 1", 3) ;
	} else {
	    Tcl_DStringAppend(&codeList, "1 0", 3) ;
	}
	break ;
    case ATTR_ZERO_UP:
	for (idx = 0 ; idx < attrPtr->rangeArgc ; ++idx) {
	    char	buf[100] ;
	    sprintf(buf, "%d", idx) ;
	    Tcl_DStringAppendElement(&codeList, buf) ;
	}
	break ;
    case ATTR_ONE_UP:
	for (idx = 0 ; idx < attrPtr->rangeArgc ; ++idx) {
	    char	buf[100] ;
	    sprintf(buf, "%d", idx + 1) ;
	    Tcl_DStringAppendElement(&codeList, buf) ;
	}
	break ;
    case ATTR_ONE_OF_N:
	for (idx = 0, on = 0 ; idx < attrPtr->rangeArgc ; ++idx) {
	    Tcl_DString	code ;
	    int		codeIdx ;
	    int		limit = attrPtr->rangeArgc ;

	    if (attrPtr->passive != NULL) {
		--limit ;
	    }

	    Tcl_DStringInit(&code) ;

	    for (codeIdx = 0 ; codeIdx < on ; ++codeIdx) {
		Tcl_DStringAppendElement(&code, "0") ;
	    }
	    if (attrPtr->passive == NULL
		|| strcmp(attrPtr->passive, attrPtr->rangeArgv[idx])) {
		Tcl_DStringAppendElement(&code, "1") ;
		++on ;
	    }
	    for (codeIdx = on ; codeIdx < limit  ; ++codeIdx) {
		Tcl_DStringAppendElement(&code, "0") ;
	    }

	    Tcl_DStringAppendElement(&codeList, Tcl_DStringValue(&code)) ;
	    Tcl_DStringFree(&code) ;
	}
	break ;
    case ATTR_THERM:
	for (idx = 0 ; idx < attrPtr->rangeArgc ; ++idx) {
	    Tcl_DString	code ;
	    int		codeIdx ;
	    char	dblBuf[TCL_DOUBLE_SPACE] ;

	    Tcl_DStringInit(&code) ;

	    Tcl_PrintDouble(interp, 1.0*attrPtr->scale, dblBuf) ;
	    for (codeIdx = 1 ; codeIdx <= idx ; ++codeIdx) {
		Tcl_DStringAppendElement(&code, dblBuf) ;
	    }

	    Tcl_PrintDouble(interp, -1.0*attrPtr->scale, dblBuf) ;
	    for ( ; codeIdx < attrPtr->rangeArgc ; ++codeIdx) {
		Tcl_DStringAppendElement(&code, dblBuf) ;
	    }

	    Tcl_DStringAppendElement(&codeList, Tcl_DStringValue(&code)) ;
	    Tcl_DStringFree(&code) ;
	}
	break ;
    case ATTR_NORM:
    case ATTR_RECTAN:
    case ATTR_IGNORE:
    case ATTR_COPY:
    default:
	Tcl_AppendResult(interp, "can't handle ranges when encoding",
			 " with method ", attrMethodName[attrPtr->method],
			 (char *)NULL) ;
	Tcl_DStringFree(&codeList) ;
	return TCL_ERROR ;
	break ;
    }

    /*
     * Now that we have the list of code strings, split it up and put
     * it into "attrPtr->crangeArgv".  We had better have as many
     * elements in the code string list as in the range list: don't
     * know how we couldn't, but better check anyways.
     */
    
    if (Tcl_SplitList(interp, Tcl_DStringValue(&codeList),
		      &argc, &argv) != TCL_OK) {
	Tcl_DStringFree(&codeList) ;
	return TCL_ERROR ;
    }

    if (argc != attrPtr->rangeArgc) {
	Tcl_AppendResult(interp, "internal error encoding values in range",
			 " with method ", attrMethodName[attrPtr->method],
			 (char *)NULL) ;
	Tcl_DStringFree(&codeList) ;
	ckfree((char *)argv) ;
	return TCL_ERROR ;
    }

    attrPtr->crangeArgv = argv ;
    
    return TCL_OK;
}


/*
 *-----------------------------------------------------------------------------
 * AttrDelete --
 *
 *	Deletes the specified attribute, implements the subcommand:
 *		attr delete handle
 *-----------------------------------------------------------------------------
 */

static int
AttrDelete (interp, attrTablePtr, handle)
Tcl_Interp *	interp;
Tcl_HashTable *	attrTablePtr;
char *		handle;
{
    Tcl_HashEntry *	entryPtr ;
    Attr_t *		attrPtr = GetAttr (interp, attrTablePtr, handle);

    if (attrPtr == NULL) {
        return TCL_ERROR;
    }

    entryPtr = Tcl_FindHashEntry(attrTablePtr, handle) ;

    CleanUpAttr(attrPtr) ;
    Tcl_DeleteHashEntry (entryPtr);

    return TCL_OK;
}


/*
 *-----------------------------------------------------------------------------
 * AttrEncode --
 *
 *	Encodes a string value and prints the code string into the
 *	interpreter's result.  Implements the subcommand:
 * 		attr encode handle value
 *-----------------------------------------------------------------------------
 */

static int
AttrEncode (interp, attrTablePtr, handle, value)
Tcl_Interp *	interp;
Tcl_HashTable *	attrTablePtr;
char *		handle;
char *		value;
{
    Attr_t *	attrPtr = GetAttr (interp, attrTablePtr, handle);
    char	dblBuf[TCL_DOUBLE_SPACE] ;
    double	x ;
    int		idx ;

    if (attrPtr == NULL) {
        return TCL_ERROR;
    }

    switch (attrPtr->method) {
    case ATTR_BINARY_SYMMETRIC:
    case ATTR_BINARY_PASSIVE:
    case ATTR_ZERO_UP:
    case ATTR_ONE_UP:
    case ATTR_ONE_OF_N:
    case ATTR_THERM:
	/*
	 * Find the value in the raw array, then return the corresponding code
	 * string.  If the value isn't in the range we have a complex switch
	 * so that the error message is pretty.
	 */

	for (idx = 0 ; idx < attrPtr->rangeArgc ; ++idx) {
	    if (strcmp(value, attrPtr->rangeArgv[idx]) == 0) {
		break ;
	    }
	}
	if (idx == attrPtr->rangeArgc) {
	    Tcl_AppendResult(interp, "bad value to encode \"", value,
			     "\": ", (char *)NULL) ;
	    if (attrPtr->rangeArgc == 0) {
		Tcl_AppendResult(interp, "range is empty", (char *)NULL) ;
	    } else if (attrPtr->rangeArgc == 1) {
		Tcl_AppendResult(interp, "should be ",
				 attrPtr->rangeArgv[0], (char *)NULL) ;
	    } else if (attrPtr->rangeArgc == 2) {
		Tcl_AppendResult(interp, "should be ",
				 attrPtr->rangeArgv[0], " or ",
				 attrPtr->rangeArgv[1], (char *)NULL) ;
	    } else {
		Tcl_AppendResult(interp, "should be", (char *)NULL) ;
		for (idx = 0 ; idx < attrPtr->rangeArgc - 1 ; ++idx) {
		    Tcl_AppendResult(interp, " ", attrPtr->rangeArgv[idx], ",",
				     (char *)NULL) ;
		}
		Tcl_AppendResult(interp, " or ", attrPtr->rangeArgv[idx],
				 (char *)NULL) ;
	    }
	    return TCL_ERROR ;
	}

	Tcl_SetResult(interp, attrPtr->crangeArgv[idx], TCL_STATIC) ;
	break ;

    case ATTR_COPY:
	/*
	 * Just return whatever value was passed in.
	 */
	
	Tcl_SetResult(interp, value, TCL_VOLATILE) ;
	break ;

    case ATTR_NORM:
	/*
	 * Normalize the value, making sure that the value we read in
	 * corresponds to the type of the attribute (integer or
	 * real). We also have to handle the case where the sigma
	 * value is zero (in which case the value we're encoding had
	 * better be the same as mu!).
	 */
	
	if (attrPtr->type == ATTR_INTEGER) {
	    int		xInt ;
	    if (Tcl_GetInt(interp, value, &xInt) != TCL_OK) {
		return TCL_ERROR ;
	    }
	    x = xInt ;
	} else if (Tcl_GetDouble(interp, value, &x) != TCL_OK) {
	    return TCL_ERROR ;
	}

	if (attrPtr->sigma != 0) {
	    Tcl_PrintDouble(interp, (x - attrPtr->mu)/attrPtr->sigma, dblBuf) ;
	} else if (attrPtr->mu == x) {
	    Tcl_PrintDouble(interp, attrPtr->mu, dblBuf) ;
	} else {
	    char	dblBuf[TCL_DOUBLE_SPACE] ;
	    Tcl_PrintDouble(interp, attrPtr->mu, dblBuf) ;
	    Tcl_AppendResult(interp, "bad value to encode \"", value,
			     "\": should be ", dblBuf, (char *)NULL) ;
	    return TCL_ERROR ;
	}
	Tcl_SetResult(interp, dblBuf, TCL_VOLATILE) ;
	break ;

    case ATTR_RECTAN:
	/*
	 * Return a list of the cos and sin of the value normalized by
	 * the "unit".
	 */
	
	if (Tcl_GetDouble(interp, value, &x) != TCL_OK) {
	    return TCL_ERROR ;
	}
	Tcl_PrintDouble(interp, sin(x*2.0*M_PI/attrPtr->unit), dblBuf) ;
	Tcl_AppendElement(interp, dblBuf) ;

	Tcl_PrintDouble(interp, cos(x*2.0*M_PI/attrPtr->unit), dblBuf) ;
	Tcl_AppendElement(interp, dblBuf) ;
	break ;

    case ATTR_IGNORE:
	/*
	 * Return NOTHING!
	 */
	
	break ;
    }
    return TCL_OK;
}


/*
 *-----------------------------------------------------------------------------
 * AttrDecode --
 *
 *	Decodes a string value and prints the decoded value into the
 *	interpreter's result.  Implements the subcommand:
 * 		attr decode handle value
 *-----------------------------------------------------------------------------
 */

static int
AttrDecode (interp, attrTablePtr, handle, value)
Tcl_Interp *	interp;
Tcl_HashTable *	attrTablePtr;
char *		handle;
char *		value;
{
    Attr_t *	attrPtr = GetAttr (interp, attrTablePtr, handle);
    char	dblBuf[TCL_DOUBLE_SPACE] ;
    double	x ;
    int		idx ;
    
    if (attrPtr == NULL) {
        return TCL_ERROR;
    }

    switch (attrPtr->method) {
    case ATTR_BINARY_SYMMETRIC:
    case ATTR_BINARY_PASSIVE:
    case ATTR_ZERO_UP:
    case ATTR_ONE_UP:
    case ATTR_ONE_OF_N:
    case ATTR_THERM:
	/*
	 * Find the value in the code string array, then return the
	 * corresponding raw string.
	 */

	for (idx = 0 ; idx < attrPtr->rangeArgc ; ++idx) {
	    if (strcmp(value, attrPtr->crangeArgv[idx]) == 0) {
		break ;
	    }
	}

	if (idx == attrPtr->rangeArgc) {
	    Tcl_AppendResult(interp, "bad value to decode \"", value,
			     "\": ", (char *)NULL) ;
	    if (attrPtr->rangeArgc == 0) {
		Tcl_AppendResult(interp, "range is empty", (char *)NULL) ;
	    } else if (attrPtr->rangeArgc == 1) {
		Tcl_AppendResult(interp, "should be ",
				 attrPtr->crangeArgv[0], (char *)NULL) ;
	    } else if (attrPtr->rangeArgc == 2) {
		Tcl_AppendResult(interp, "should be ",
				 attrPtr->crangeArgv[0], " or ",
				 attrPtr->crangeArgv[1], (char *)NULL) ;
	    } else {
		Tcl_AppendResult(interp, "should be", (char *)NULL) ;
		for (idx = 0 ; idx < attrPtr->rangeArgc - 1 ; ++idx) {
		    Tcl_AppendResult(interp, " ", attrPtr->crangeArgv[idx], ",",
				     (char *)NULL) ;
		}
		Tcl_AppendResult(interp, " or ", attrPtr->crangeArgv[idx],
				 (char *)NULL) ;
	    }
	    return TCL_ERROR ;
	}

	Tcl_SetResult(interp, attrPtr->rangeArgv[idx], TCL_STATIC) ;
	break ;
    case ATTR_COPY:
	/*
	 * Just return whatever value was passed in.
	 */
	
	Tcl_SetResult(interp, value, TCL_VOLATILE) ;
	break ;

    case ATTR_NORM:
	/*
	 * De-normalize the value, making sure that the value we use
	 * the right format for printing the result (integer or real).
	 */
	
	if (Tcl_GetDouble(interp, value, &x) != TCL_OK) {
	    return TCL_ERROR ;
	}
	if (attrPtr->type == ATTR_INTEGER) {
	    sprintf(dblBuf, "%d",
		    (int)(attrPtr->mu + attrPtr->sigma*x + 0.5)) ;
	} else {
	    Tcl_PrintDouble(interp, attrPtr->mu + attrPtr->sigma*x, dblBuf) ;
	}
	Tcl_SetResult(interp, dblBuf, TCL_VOLATILE) ;
	break ;

    case ATTR_RECTAN:
    case ATTR_IGNORE:
	/*
	 * Neither of these types can be decoded.
	 */
	
	Tcl_AppendResult(interp, "can't decode ",
			 attrMethodName[attrPtr->method],
			 " encoded variables", (char *)NULL) ;
	return TCL_ERROR ;
	break ;
    }

    return TCL_OK;
}


/*
 *-----------------------------------------------------------------------------
 * AttrCmd --
 *
 *	The procedure for the "attr" command.
 *-----------------------------------------------------------------------------
 */

static int
AttrCmd(clientData, interp, argc, argv)
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

    if (c == 'c' && strcmp(option, "create") == 0) {
	Attr_Type_t	type ;
	Attr_Method_t	method ;
	
	if (argc < 4) {
	    Tcl_AppendResult(interp, "wrong # args: should be \"",
			     argv[0], " create type method ?arg ...?\"",
			     (char *) NULL);
	    return TCL_ERROR;
	}

	for (type = (Attr_Type_t)0 ; attrTypeName[type] != NULL ; ++type) {
	    if (strcmp(argv[2], attrTypeName[type]) == 0) {
		break ;
	    }
	}
	if (attrTypeName[type] == NULL) {
	    Tcl_AppendResult(interp, "unknown attribute type \"", argv[2],
			     "\"", (char *)NULL) ;
	    return TCL_ERROR ;
	}

	for (method = (Attr_Method_t)0 ;
	     attrMethodName[method] != NULL ; ++method) {
	    if (strcmp(argv[3], attrMethodName[method]) == 0) {
		break ;
	    }
	}
	if (attrMethodName[method] == NULL) {
	    Tcl_AppendResult(interp, "unknown encoding method \"", argv[3],
			     "\"", (char *)NULL) ;
	    return TCL_ERROR ;
	}

	if (AttrCreate(interp, (Tcl_HashTable *) clientData, type, method,
		       argv[0], argc - 4, argv + 4) != TCL_OK) {
	    return TCL_ERROR ;
	}

    } else if (c == 'd' && strcmp(option, "decode") == 0) {
	if (argc != 4) {
	    Tcl_AppendResult(interp, "wrong # args: should be \"",
			     argv[0], " decode attrHandle value\"",
			     (char *) NULL);
	    return TCL_ERROR;
	}
	if (AttrDecode(interp, (Tcl_HashTable *) clientData,
		       argv[2], argv[3]) != TCL_OK) {
	    return TCL_ERROR ;
	}
	
    } else if (c == 'd' && strcmp(option, "delete") == 0) {
	if (argc != 3) {
	    Tcl_AppendResult(interp, "wrong # args: should be \"",
			     argv[0], " delete attrHandle\"", (char *) NULL);
	    return TCL_ERROR;
	}
	if (AttrDelete(interp, (Tcl_HashTable *) clientData,
		       argv[2]) != TCL_OK) {
	    return TCL_ERROR ;
	}
	
    } else if (c == 'e' && strcmp(option, "encode") == 0) {
	if (argc != 4) {
	    Tcl_AppendResult(interp, "wrong # args: should be \"",
			     argv[0], " decode attrHandle value\"",
			     (char *) NULL);
	    return TCL_ERROR;
	}
	if (AttrEncode(interp, (Tcl_HashTable *) clientData,
		       argv[2], argv[3]) != TCL_OK) {
	    return TCL_ERROR ;
	}
	
    } else {
	Tcl_AppendResult(interp, "bad option \"",
			 option, "\": should be create, ",
			 "decode, delete, or encode", (char *) NULL) ;
	return TCL_ERROR;
    }

    return TCL_OK ;
}


/*
 *-----------------------------------------------------------------------------
 *  GetAttr --
 *
 *    Looks up an attribute in the hash table given it's handle
 *-----------------------------------------------------------------------------
 */
static Attr_t *
GetAttr (interp, attrTablePtr, handle)
Tcl_Interp *	interp ;
Tcl_HashTable *	attrTablePtr;
CONST char *	handle ;
{
    Tcl_HashEntry *	entryPtr ;
    char *		nonConstHandle ;

    nonConstHandle = (char *)ckalloc((1+strlen(handle))*sizeof(char)) ;
    strcpy(nonConstHandle, handle) ;
    entryPtr = Tcl_FindHashEntry(attrTablePtr, nonConstHandle) ;
    ckfree((char *)nonConstHandle) ;

    if (entryPtr == NULL) {
	Tcl_AppendResult(interp, "invalid attr handle \"", handle, "\"",
			 (char *)NULL) ;
        return (Attr_t *)NULL;
    }

    return (Attr_t *)Tcl_GetHashValue(entryPtr) ;
}


/*
 *-----------------------------------------------------------------------------
 * AttrCleanUp --
 *
 *	Called when the interpreter is deleted to cleanup all
 *	attribute resources.
 *-----------------------------------------------------------------------------
 */

static void
AttrCleanUp (clientData, interp)
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
        CleanUpAttr ((Attr_t *)Tcl_GetHashValue(entryPtr));
    }
    Tcl_DeleteHashTable ((Tcl_HashTable *) clientData);

    ckfree((char *) clientData) ;
}


/*
 *-----------------------------------------------------------------------------
 *  Delve_InitAttr --
 *
 *    Initialize the DELVE attribute encoding/decoding facility.
 *-----------------------------------------------------------------------------
 */

int
Delve_InitAttr (interp)
Tcl_Interp *	interp;
{
    Tcl_HashTable  *attrTablePtr;

    attrTablePtr = (Tcl_HashTable *)ckalloc(sizeof(Tcl_HashTable)) ;

    Tcl_InitHashTable(attrTablePtr, TCL_STRING_KEYS) ;

    Tcl_CallWhenDeleted (interp, AttrCleanUp, (ClientData) attrTablePtr);

    /*
     * Initialize the command.
     */

    Tcl_CreateCommand (interp, "d_attr", AttrCmd, 
                       (ClientData) attrTablePtr, (void (*)()) NULL);

    return TCL_OK ;
}
