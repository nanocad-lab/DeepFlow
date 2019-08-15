/*
 * delveRandom.c - Lagged Fibonnacci random number generator 
 *
 * Copyright (c) 1995 Paul Coddington,
 * Northeast Parallel Architectures Center at Syracuse University,
 *
 * Copyright (c) 1996 by The University of Toronto.
 * 
 * Author: Paul Coddington
 * Modified by: Delve (delve@cs.toronto.edu)
 *
 * See the file "copyright" for information on usage and redistribution
 * of this file, and for a DISCLAIMER OF ALL WARRANTIES.
 * 
 * General lagged Fibonnacci generator using subtraction, with lags 
 * p and q, i.e. F(p,q,-) in Marsaglia's notation.
 * 
 * The random numbers X_{i} are obtained from the sequence:
 * 
 *    X_{i} = X_{i-q} - X_{i-p}   mod M
 *  
 * where M is 1 if the X's are taken to be floating point reals in [0,1),
 * as they are here.
 * 
 * For good results, the biggest lag should be at least 1000, and probably
 * on the order of 10000.
 * 
 * The following lags give the maximal period of the generator, which is 
 * (2^{p} - 1)2^{n-1} on integers mod 2^n or reals with n bits in the 
 * mantissa (see Knuth, or W. Zerler, Information and Control, 15 (1969) 67, 
 * for a complete more list).
 * 
 *     P     Q
 *   9689   471 
 *   4423  1393
 *   2281   715
 *   1279   418
 *    607   273
 *    521   168
 *    127    63
 * 
 * This program is based on the implementation of RANMAR in 
 * F. James, "A Review of Pseudo-random Number Generators", 
 * Comput. Phys. Comm. 60, 329 (1990).
 * 
 * For more details, see:
 * 
 * D.E. Knuth, The Art of Computer Programming Vol. 2: 
 * Seminumerical Methods, (Addison-Wesley, Reading, Mass., 1981).
 * 
 * P. L'Ecuyer, Random numbers for simulation, Comm. ACM 33:10, 85 (1990).
 * 
 * G.A. Marsaglia, A current view of random number generators,
 * in Computational Science and Statistics: The Interface,
 * ed. L. Balliard (Elsevier, Amsterdam, 1985).
 * 
 */

#ifndef lint
static char rcsid[] = "$Id: delveRandom.c,v 1.1.2.3 1996/11/12 16:54:08 revow Exp $" ;
#endif

#include <stdio.h>
#include <tcl.h>
#include "delve.h"

#include <stdio.h>

#define MAXSEED 900000000	/* The maximum value for the random
				 * number seed */
#define SIGBITS 24		/* Number of significant bits */

#define P 1279			/* The two lags */
#define Q  418

/*
 * Structure used to maintain seed table used by lagged Fibonacci.
 */

typedef struct Random_t {
    double	u[P+1] ;	/* seed table */
    int		pt0 ;		/* pointer into the seed table */
    int		pt1 ;		/* second pointer into the seed table */
} Random_t ;

static Random_t *GetRandom _ANSI_ARGS_((Tcl_Interp *interp)) ;
static void	RandomCleanUp _ANSI_ARGS_((ClientData clientData,
			Tcl_Interp *interp)) ;
static int	RandomCmd _ANSI_ARGS_((ClientData clientData,
			Tcl_Interp *interp, int argc, char **argv)) ;
static double	RandomNumber _ANSI_ARGS_((Random_t *randomPtr)) ;
static void	RandomSeed _ANSI_ARGS_((Random_t *randomPtr, long int seed)) ;



/*
 *-----------------------------------------------------------------------------
 * RandomSeed --
 *
 *   Initialize the random number generator. Taken from RMARIN in
 *   James's review -- initializes every significant bit using a
 *   combination linear congruential and small lag Fibonacci
 *   generator.
 *
 *-----------------------------------------------------------------------------
 */

static void
RandomSeed(randomPtr, seed)
Random_t *	randomPtr ;
long int	seed ;
{
    int		ij, kl, i, j, k, l;
    int		ii, jj, m;
    double	t, s;

    if (seed < 0) {
	seed = - seed;
    }
    seed = seed % MAXSEED;

    ij = seed / 30082;
    kl = seed - 30082 * ij;
    i = ((ij/177)% 177) + 2;
    j =  ij%177 + 2;
    k = ((kl/169)% 178) + 1;
    l =  kl%169;
      
    for ( ii = 1 ; ii <= P; ii++ ) {
        s = 0.0;
        t = 0.5;
        for ( jj = 1 ; jj <= SIGBITS; jj++ ){
	    m = (((i*j)% 179)*k)% 179;
	    i = j;
	    j = k;
	    k = m;
	    l = (53*l+1) % 169;
	    if ( ((l*m)%64) >= 32) {
		s = s + t;
	    }
	    t = 0.5 * t;
        }
        randomPtr->u[ii] = s;
    }
    randomPtr->pt0 = P;
    randomPtr->pt1 = Q;

    return ;
}
  

/*
 *-----------------------------------------------------------------------------
 * Delve_RandomNumber --
 *
 *	Return a random double in [0.0, 1.0)
 *
 *-----------------------------------------------------------------------------
 */

static double
RandomNumber(randomPtr)
Random_t *	randomPtr ;
{
    double	uni;
      
    uni = randomPtr->u[randomPtr->pt0] - randomPtr->u[randomPtr->pt1];
    if (uni < 0.0) {
	uni = uni + 1.0;
    }
    randomPtr->u[randomPtr->pt0] = uni;
    randomPtr->pt0 = randomPtr->pt0 - 1;
    if (randomPtr->pt0 == 0) {
	randomPtr->pt0 = P;
    }
    randomPtr->pt1 = randomPtr->pt1 - 1;
    if (randomPtr->pt1 == 0) {
	randomPtr->pt1 = P;
    }

    return uni ;
}


/*
 *-----------------------------------------------------------------------------
 *  Delve_InitRandom --
 *
 *    Initialize the DELVE pseudo-random number generation facility.
 *-----------------------------------------------------------------------------
 */

int
Delve_InitRandom (interp)
Tcl_Interp *	interp;
{
    Random_t	*randomPtr ;

    randomPtr = (Random_t *)ckalloc(sizeof(Random_t)) ;

    RandomSeed(randomPtr, 0) ;

    Tcl_CallWhenDeleted (interp, RandomCleanUp, (ClientData) randomPtr);

    /*
     * Initialize the commands.
     */

    Tcl_CreateCommand (interp, "d_random", RandomCmd, 
                       (ClientData) randomPtr, (void (*)()) NULL);

    return TCL_OK ;
}


/*
 *-----------------------------------------------------------------------------
 *  GetRandom --
 *
 *    Looks up a random number generator data structure in the hash
 *    table given it's handle
 *
 *-----------------------------------------------------------------------------
 */
static Random_t *
GetRandom (interp)
Tcl_Interp *	interp ;
{
    Tcl_CmdInfo		info ;
    
    if (Tcl_GetCommandInfo(interp, "d_random", &info) == 0
	|| info.proc != RandomCmd) {
	Tcl_SetResult(interp, "can't find \"d_random\" command in interpreter",
		      TCL_VOLATILE) ;
	return (Random_t *)NULL ;
    }

    return (Random_t *)info.clientData ;
}


/*
 *-----------------------------------------------------------------------------
 * RandomCleanUp --
 *
 * Called when the interpreter is deleted to cleanup all random
 * number generator resources
 *
 *-----------------------------------------------------------------------------
 */

static void
RandomCleanUp (clientData, interp)
ClientData	clientData ;
Tcl_Interp * 	interp;
{
    ckfree((char *) clientData) ;
}


/*
 *-----------------------------------------------------------------------------
 * RandomCmd --
 *
 *    The procedure for the "d_random" command.
 *-----------------------------------------------------------------------------
 */
static int
RandomCmd(clientData, interp, argc, argv)
ClientData 	clientData;	/* Handle table pointer. */
Tcl_Interp *	interp;		/* Current interpreter. */
int 		argc;		/* Number of arguments. */
char **		argv;		/* Argument strings. */
{
    char *	option ;
    int		limit ;
    char	c ;

    if (argc < 2) {
	Tcl_AppendResult(interp, "wrong # args: should be \"",
			 argv[0], " limit | seed seedval\"", (char *) NULL) ;
	return TCL_ERROR;
    }

    option = argv[1] ;
    c = option[0] ;

    if (c == 's' && strcmp(option, "seed") == 0) {
	int	seed ;
	if (argc != 3) {
	    Tcl_AppendResult(interp, "wrong # args: should be \"",
			     argv[0], " seed seedval\"", (char *) NULL) ;
	    return TCL_ERROR;
	}
	if (Tcl_GetInt(interp, argv[2], &seed) != TCL_OK) {
	    return TCL_ERROR ;
	}
	RandomSeed((Random_t *)clientData, seed) ;
    } else if (argc != 2) {
	Tcl_AppendResult(interp, "wrong # args: should be \"",
			 argv[0], " limit | seed seedval\"", (char *) NULL) ;
	return TCL_ERROR;
    } else {
	if (Tcl_GetInt(interp, argv[1], &limit) != TCL_OK) {
	    return TCL_ERROR ;
	}
	sprintf(interp->result, "%d",
		(int)(limit*RandomNumber((Random_t *)clientData))) ;
    }

    return TCL_OK ;
}


/*
 *-----------------------------------------------------------------------------
 *  Delve_RandomSeed --
 *
 *    External interface to random number generator seed.
 *-----------------------------------------------------------------------------
 */

int
Delve_RandomSeed(interp, seed)
Tcl_Interp	*interp ;
long int	seed ;
{
    Random_t *	randomPtr = GetRandom(interp) ;

    if (randomPtr == NULL) {
	return TCL_ERROR ;
    }

    RandomSeed(randomPtr, seed) ;

    return TCL_OK ;
}
  

/*
 *-----------------------------------------------------------------------------
 *  Delve_RandomNumber --
 *
 *    External interface to random number generator.
 *-----------------------------------------------------------------------------
 */

int
Delve_RandomNumber(interp, doublePtr)
Tcl_Interp *	interp ;
double *	doublePtr ;
{
    Random_t *	randomPtr = GetRandom(interp) ;
      
    if (randomPtr == NULL) {
	return TCL_ERROR ;
    }

    *doublePtr = RandomNumber(randomPtr) ;

    return TCL_OK ;
}
