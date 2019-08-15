# ----------------------------------------------------------------------
# $Id: init.tcl,v 1.6.2.4.2.1 1997/11/27 15:42:09 revow Exp $
#
# Default system startup file for Delve-based applications.  Appends
# the delve library to the auto_path, and sets the global "delve_path"
# variable, used to find data sets and methods.
#
# Also sets all the specification file names (since they seem to still
# be changing).
#
# AUTHOR HISTORY:
#    Author Delve (delve@cs.toronto.edu)
#    Drew van Camp (drew@cs.toronto.edu)
#
# Copyright (c) 1995-1996 The University of Toronto.
#
# See the file "copyright" for information on usage and redistribution
# of this file, and for a DISCLAIMER OF ALL WARRANTIES.
#
# ----------------------------------------------------------------------

lappend auto_path $delve_library ;

set d_priv(datasetFile)		Dataset.data ;
set d_priv(datasetSpec)		Dataset.spec ;

set d_priv(prototaskFile)	Prototask.data ;
set d_priv(prototaskSpec)	Prototask.spec ;

set d_priv(Randomorder)		Random-order ;

regsub -all " +" [d_compressExt] "|" extList
set d_priv(priorGlobPattern)   .prior($extList)?$ ;

# ----------------------------------------------------------------------
# This procedure strips all but the first line from the variable it
# traces.  It is used as a trace on 'errorInfo' so that when a script
# dies, the user doesn't see the whole stack trace, just the first
# error.  Scripts can keep stack traces or discard them by calling
# 'delve_debug' passing in '1' or '0' respectively.  By default, stack
# traces are kept if tcl_interactive is 1, and discarded if it is 0.
# ----------------------------------------------------------------------

proc no_debug_trace { name1 name2 op } {
    if { [string compare $name2 {}] == 0 } {
	upvar $name1 errorInfo ;
    } else {
	upvar $name1($name2) errorInfo ;
    }
    if ![info exists errorInfo] {
	return ;
    }
    set idx	[string first "\n" $errorInfo] ;

    if { $idx >= 0 } {
	set errorInfo [string range $errorInfo 0 [expr { $idx - 1 }]] ;
    }
}

proc delve_debug { state } {
    global errorInfo ;
    if { "$state" } {
	if { [lsearch [trace vinfo errorInfo] "w no_debug_trace"] >= 0 } {
	    trace vdelete  errorInfo w no_debug_trace ;
	}
    } else {
	if { [lsearch [trace vinfo errorInfo] "w no_debug_trace"] < 0 } {
	    trace variable errorInfo w no_debug_trace ;
	}
    }
}

if { "$tcl_interactive" } {
    delve_debug 1 ;
} else {
    delve_debug 0 ;
}
    
    
set tcl_precision 9
