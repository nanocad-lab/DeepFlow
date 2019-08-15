/*
 * delveAnova.c - Implementation of DELVE analysis of variance utilities.
 *
 * Copyright (c) 1996 by The University of Toronto.
 * 
 * See the file "copyright" for information on usage and redistribution
 * of this file, and for a DISCLAIMER OF ALL WARRANTIES.
 * 
 * Authors: Delve (delve@cs.toronto.edu)
 * 	    
 */

#ifndef lint
static char rcsid[] = "$Id: delveAnova.c,v 1.19.2.2 1996/11/21 15:51:38 revow Exp $" ;
#endif

#include "delve.h"
#include <math.h>

/*
 * We could have either the "gamma" or the "lgamma" function (both of
 * which should return the natural logarithm of the gamma function).
 * We use "lgamma" throughout the code.  As well, the "math.h" header
 * file may or may not include the declaration for it.
 */

#ifdef HAVE_LGAMMA
#   ifndef HAVE_LGAMMA_DECL
        double lgamma _ANSI_ARGS_((double x));
#   endif
#else
#   ifdef HAVE_GAMMA
#       define lgamma(x)	gamma((x))
#	ifndef HAVE_GAMMA_DECL
            double gamma _ANSI_ARGS_((double x));
#       endif
#   endif
#endif

static int	AnovaCmd _ANSI_ARGS_((ClientData dummy, Tcl_Interp * interp,
			int argc, char **argv));
static int      betainc _ANSI_ARGS_((Tcl_Interp *interp, double x, double a,
                                double b, double *pPtr)) ;
static double	sq _ANSI_ARGS_((double x)) ;
 

/*
 *-----------------------------------------------------------------------------
 *  Delve_InitAnova --
 *
 *    Initialize the DELVE analysis of variance facility.
 *-----------------------------------------------------------------------------
 */

int
Delve_InitAnova (interp)
Tcl_Interp *interp;
{
    /*
     * Initialize the command.
     */

    Tcl_CreateCommand (interp, "d_anova", AnovaCmd, 
                       (ClientData) NULL, (void (*)()) NULL);
    return TCL_OK ;
}


#define ESTIMATE 0
#define COMPARE 1
#define HIERARCHICAL 0
#define COMMON 1

/*
 *-----------------------------------------------------------------------------
 * AnovaCmd --
 *
 * The procedure for the "d_anova" command.  It takes two strings and one or
 * two matrices. The first string is the analysis-type (must be either
 * "compare" or "estimate"); second string is the test_set_selection (must
 * be either "heirarchical" or "common"); then follows a martix of losses,
 * and iff analysis-type is "compare" then a second matrix of losses.
 *
 * The procedure outputs [mean1] mean [mean2] [p] sig_a sig_e stdErr [sig_b], 
 * where a square bracket means that quantity is only sometimes computed.
 *-----------------------------------------------------------------------------
 */

static int 
AnovaCmd(dummy, interp, argc, argv)
ClientData 	dummy;		/* unused */
Tcl_Interp *	interp;		/* Current interpreter. */
int 		argc;		/* Number of arguments. */
char **		argv;		/* Argument strings. */
{
    matrix_t *	matrixPtr ;
    int  	I, J, i, j, test_set_selection, analysis;
    char 	buffer[TCL_DOUBLE_SPACE];
    double	**y1, **y2, *yi, *yj, msa = 0.0, msb = 0.0, mse = 0.0,
                mean = 0.0, mean1 = 0.0, p = -1.0, sda2, sdb2, stdErr;
    
    /*
     * Parse input.
     */

    if (argc != 4 && argc != 5) {
	Tcl_AppendResult(interp, "wrong # args: should be \"",
			 argv[0], " selection_mode analysis_mode matrixHandle",
			 " ?matrixHandle?\"", (char *) NULL) ;
	return TCL_ERROR;
    }

    if (!strcmp(argv[1], "common")) {
        test_set_selection = COMMON;
    } else if (!strcmp(argv[1], "hierarchical")) {
        test_set_selection = HIERARCHICAL;
    } else {
	Tcl_AppendResult(interp, "bad test set selection mode \"",
			 argv[1], "\": must be common or hierarchical",
			 (char *) NULL) ;
  	return TCL_ERROR;
    }

    if (!strcmp(argv[2], "estimate")) {
        analysis = ESTIMATE;
    } else if (!strcmp(argv[2], "compare")) {
        analysis = COMPARE;
    } else {
	Tcl_AppendResult(interp, "bad analysis mode \"",
			 argv[1], "\": must be estimate or compare",
			 (char *) NULL) ;
  	return TCL_ERROR;
    }

    if (Delve_GetMatrix(interp, argv[3], &matrixPtr) != TCL_OK) {
	return TCL_ERROR;
    }
    I = matrixPtr->rows; 
    J = matrixPtr->cols; 
    y1 = matrixPtr->x;

    if (analysis == COMPARE) {
	if (argc != 5) {
	    Tcl_AppendResult(interp, "missing second matrix handle for",
			     " analysis mode \"compare\"",
			     (char *) NULL) ;
	    return TCL_ERROR;
	} else if (Delve_GetMatrix(interp, argv[4], &matrixPtr) != TCL_OK) {
	    return TCL_ERROR;
	}
	y2 = matrixPtr->x;
    }

    /* 
     * If we are comparing two methods, then compute the mean of the first
     * matrix and subtract matrix 2 from matrix 1; subsequent analysis will
     * be done on matrix 1. Pass back the mean of matrix 1.
     */

    if (analysis == COMPARE) {
        for (i=0; i<I; i++) {
            for (j=0; j<J; j++) {
                mean1 += y1[i][j];
                y1[i][j] -= y2[i][j];
            }
        }
        mean1 /= I*J;
        Tcl_PrintDouble(interp, mean1, buffer) ;
        Tcl_AppendElement(interp, buffer) ;
    }

    /*
     * Compute marginal means for training sets and overall mean. Pass back
     * overall mean. Compute "mean squared error" (msa) for the training sets. 
     */

    yi = (double *)ckalloc((size_t) I*sizeof(double)); 
    for (i=0; i<I; i++) {
        yi[i] = 0.0;
	for (j=0; j<J; j++) {
	    yi[i] += y1[i][j]; 
        }
	yi[i] /= J;
	mean += yi[i]; 
    }
    mean /= I;

    Tcl_PrintDouble(interp, mean, buffer) ;
    Tcl_AppendElement(interp, buffer) ;

    for (i=0; i<I; i++) {
	msa += sq(mean-yi[i]); 
    }
    msa *= (double)J/(I-1);   

    /*
     * Compute mse and if test_set_selection is "common" then also compute
     * also marginal means for test-cases and msb. Free yi and yj.
     */
    
    if (test_set_selection == HIERARCHICAL) {
        for (i=0; i<I; i++) {
            for (j=0; j<J; j++) {
                mse += sq(y1[i][j]-yi[i]);
            }
        }
        mse /= (J-1.0)*I;
    } else {
        yj = (double *)ckalloc((size_t) J*sizeof(double)); 
        for (j=0; j<J; j++) {
	    yj[j] = 0.0;
            for (i=0; i<I; i++) {
                yj[j] += y1[i][j]; 
            }
            yj[j] /= I;
            msb += sq(mean-yj[j]); 
            for (i=0; i<I; i++) {
                mse += sq(y1[i][j]-yi[i]-yj[j]+mean);
            }
        }
        msb *= I/(J-1.0);
        mse /= (I-1.0)*(J-1.0);
        ckfree((char *) yj);
    }
    ckfree((char *) yi);

    /*
     * If we are comparing methods then pass back mean2; then do a test of
     * significance test: if the test_set_selection is "common" then use the
     * F test otherwise T test. Pass back p values.
     */

    if (analysis == COMPARE) {

        Tcl_PrintDouble(interp, mean1-mean, buffer) ;
        Tcl_AppendElement(interp, buffer) ;

        if (test_set_selection == COMMON && msa + msb > 0.0) {
            double SS, F, df1, df2;
           
            SS = mean*mean*I*J;
	    F = (SS+mse)/(msa+msb);
	    df1 = sq(SS+mse)/(SS*SS + mse*mse/((I-1.0)*(J-1.0))) ;
	    df2 = sq(msa+msb)/(msa*msa/(I-1.0) + msb*msb/(J-1.0)) ;
	    if (betainc(interp, df2/(df2+df1*F), df2/2, df1/2, &p) != TCL_OK) {
	        return TCL_ERROR ;
	    }
        } else {
	    double t2, df;
            
	    if (msa > 0.0)
	    {
	      t2 = I*J*mean*mean/msa;
	      df = I-1.0;
	      if (betainc(interp, df/(df+t2), 0.5*df, 0.5, &p) != TCL_OK) {
	        return TCL_ERROR ;
	      }
	    }
        }
	if (p > 0.0)
	{
	  Tcl_PrintDouble(interp, p, buffer) ;
	}
	else
	{
	  strcpy(buffer, "undefined");
	}
        Tcl_AppendElement(interp, buffer) ;
    }

    /* 
     * Now compute standard deviations for the relevant types of uncertainty
     * and pass them back.
     */

    sda2 = (msa > mse) ? (msa-mse)/J : 0.0;

    Tcl_PrintDouble(interp, sqrt(sda2), buffer) ;
    Tcl_AppendElement(interp, buffer) ;

    Tcl_PrintDouble(interp, sqrt(mse), buffer) ;
    Tcl_AppendElement(interp, buffer) ;

    if (test_set_selection == COMMON) {
       sdb2 = (msb > mse) ? (msb-mse)/I : 0.0;
       stdErr = sqrt(sda2/I+sdb2/J+mse/(I*J));

       Tcl_PrintDouble(interp, stdErr, buffer) ;
       Tcl_AppendElement(interp, buffer) ;
  
       Tcl_PrintDouble(interp, sqrt(sdb2), buffer) ;
       Tcl_AppendElement(interp, buffer) ;
    } else {
       stdErr = sqrt(sda2/I+mse/(I*J));
     
       Tcl_PrintDouble(interp, stdErr, buffer) ;
       Tcl_AppendElement(interp, buffer) ;
    }
    return TCL_OK ;
}


/*
 *-----------------------------------------------------------------------------
 * betainc --
 *
 * Calculates the incomplete Beta function I_x(a,b) and stores the
 * result in the address in "pPtr".  Normally it returns TCL_OK, but
 * will return TCL_ERROR and leave an error message in interp's result
 * if it can't do the calculation for some reason.
 *-----------------------------------------------------------------------------
 */

#define tiny 1.0e-30

static int
betainc(interp, x, a, b, hPtr)
Tcl_Interp *    interp ;
double 		x ;
double 		a ;
double 		b ;
double *        hPtr ;
{
    double c = 1.0, d, e, f = 0.0, eps = 1e-15;
    int    m, n, symtrans = 0;

    if (x < 0.0 || x > 1.0 || a <= 0.0 || b <= 0.0) {
	Tcl_SetResult(interp,
		      "Illegal arguments in call to \"betainc\" function",
		      TCL_VOLATILE) ;
	return TCL_ERROR ;
    }

    /* 
     * make a symmetry transformation? 
     */
    if (x > (a+1.0)/(a+b+2.0)) { 
	e = a; a = b; b = e; x = 1.0-x; symtrans = 1; 
    }
    if (x != 0 && x != 1.0) {
	f = exp(a*log(x)+b*log(1.0-x)+lgamma(a+b)-lgamma(a)-lgamma(b))/a;
    }

    d = 1.0-x*(a+b)/(a+1.0); d = (fabs(d) < tiny) ? 1.0/tiny : 1.0/d;
    for (n=2; m=n/2, n<100; n++) {
	f *= c*d;
	e = (m == n/2.0) ? x*m*(b-m) : -x*(a+m)*(a+b+m); e /= (a+n-1)*(a+n);
	d = 1.0+e*d; d = (fabs(d) < tiny) ? 1.0/tiny : 1.0/d;
	c = 1.0+e/c; if (fabs(d) < tiny) c = tiny;
	if (fabs(1.0-c*d) < eps) {
	    if (symtrans) {
		*hPtr = 1.0-f; 
	    } else {
		*hPtr = f;
	    }
	    return TCL_OK;
	}
    }
}


/*
 *-----------------------------------------------------------------------------
 * sq --
 *
 * Multiplies a floating point number by itself, and returns the result.
 *-----------------------------------------------------------------------------
 */

static double
sq(x)
double x ;
{ 
    return x*x; 
}

