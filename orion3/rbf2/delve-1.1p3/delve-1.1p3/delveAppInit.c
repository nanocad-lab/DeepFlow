/*
 * delveAppInit.c - application initialization for "delve".
 *
 * Copyright (c) 1996 by The University of Toronto.
 * 
 * See the file "copyright" for information on usage and redistribution
 * of this file, and for a DISCLAIMER OF ALL WARRANTIES.
 * 
 * Author: Delve (delve@cs.toronto.edu)
 */

#ifndef lint
static char rcsid[] = "$Id: delveAppInit.c,v 1.13.2.3 1996/11/12 16:54:06 revow Exp $" ;
#endif

#include <stdio.h>
#include <tcl.h>
#include "delve.h"
#include "patchlevel.h"

/*
 * The following variable is a special hack that insures the tcl
 * version of matherr() is used when linking against shared libraries.
 * Even if matherr is not used on this system, there is a dummy version
 * in libtcl.
 */
EXTERN int matherr ();
int (*tclDummyMathPtr)() = matherr;


#if (TCL_MAJOR_VERSION > 7) || ((TCL_MAJOR_VERSION == 7) && (TCL_MINOR_VERSION > 3))

/*
 *----------------------------------------------------------------------
 *
 * main --
 *
 *	This is the main program for the application. It is only
 *	required for major versions of tcl >= 7.4
 *
 * Results:
 *	None: Tcl_Main never returns here, so this procedure never
 *	returns either.
 *
 * Side effects:
 *	Whatever the application does.
 *
 *----------------------------------------------------------------------
 */

int
main(argc, argv)
    int argc;			/* Number of command-line arguments. */
    char **argv;		/* Values of command-line arguments. */
{
    Tcl_Main(argc, argv, Tcl_AppInit);
    return 0;			/* Needed only to prevent compiler warning. */
}

#endif


/*
 *-----------------------------------------------------------------------------
 * Tcl_AppInit -
 *
 *   Application initialization.
 *-----------------------------------------------------------------------------
 */

int 
Tcl_AppInit(interp)
  Tcl_Interp	*interp ;
{
    static char initCmd[] =
    "if [file exists $delve_library/init.tcl] {\n\
        source $delve_library/init.tcl\n\
    } else {\n\
        set msg \"can't find $delve_library/init.tcl\\n\"\n\
        append msg \"Perhaps you need to install Delve \\n\"\n\
        append msg \"or set your DELVE_LIBRARY environment variable?\"\n\
        error $msg\n\
    }";

    char *libDir;
    char *path;

    /*
     * Initialize Extensions.
     */

    if (Tcl_Init(interp) != TCL_OK) {
	return TCL_ERROR ;
    }

    if (Delve_InitMatrix(interp) != TCL_OK) {
	return TCL_ERROR ;
    }

    if (Delve_InitAnova(interp) != TCL_OK) {
	return TCL_ERROR ;
    }

    if (Delve_InitAttr(interp) != TCL_OK) {
	return TCL_ERROR ;
    }

    if (Delve_InitRandom(interp) != TCL_OK) {
	return TCL_ERROR ;
    }

    /*
     * Initialize the commands.
     */

    Tcl_CreateCommand (interp, "d_mstats", MatrixStatsCmd, 
                       (ClientData) NULL, (void (*)()) NULL);

    /*
     *  Set up the delve path and delve library, and load the
     *  "init.itcl" file.  If we get "delve_path" from the
     *  environment, it's a colon separated list, so we have to
     *  convert it to a Tcl list.  
     */

    path = Tcl_GetVar2(interp, "env", "DELVE_PATH", TCL_GLOBAL_ONLY);
    if (path != NULL) {
	Tcl_DString	dstring ;
	char		*elementStart ;
	char		*p ;

	Tcl_DStringInit(&dstring) ;
	for (p = elementStart = path; *p != 0; p++) {
	    if (*p == ':') {
		*p = 0;
		Tcl_DStringAppendElement(&dstring, elementStart);
		*p = ':';
		elementStart = p+1;
	    }
	}
	if (p != path) {
	    Tcl_DStringAppendElement(&dstring, elementStart);
	}

	Tcl_SetVar(interp, 
		   "delve_path", Tcl_DStringValue(&dstring), TCL_GLOBAL_ONLY);
	Tcl_DStringFree(&dstring) ;
    } else {
	Tcl_SetVar(interp, "delve_path", DELVE_PATH, TCL_GLOBAL_ONLY);
    }

    libDir = Tcl_GetVar2(interp, "env", "DELVE_LIBRARY", TCL_GLOBAL_ONLY);
    if (libDir == NULL) {
        libDir = DELVE_LIBRARY;
    }
    Tcl_SetVar(interp, "delve_library",    libDir,	      TCL_GLOBAL_ONLY);
    Tcl_SetVar(interp, "delve_patchLevel", DELVE_PATCH_LEVEL, TCL_GLOBAL_ONLY);
    Tcl_SetVar(interp, "delve_version",    DELVE_VERSION,     TCL_GLOBAL_ONLY);

#if (TCL_MAJOR_VERSION > 7) || ((TCL_MAJOR_VERSION == 7) && (TCL_MINOR_VERSION > 4))
    Tcl_SetVar(interp, "tcl_rcFileName", "~/.delverc", TCL_GLOBAL_ONLY) ;
#else
    tcl_RcFileName = "~/.delverc";
#endif
    

    return Tcl_Eval(interp, initCmd);
}
