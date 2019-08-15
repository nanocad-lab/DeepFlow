# ----------------------------------------------------------------------
# $Id: root.tcl,v 1.2.2.3 1996/11/12 16:55:43 revow Exp $
#
# Tcl procedures for getting information about delve root directories.
#
# AUTHOR HISTORY:
#    Author Delve (delve@cs.toronto.edu)
#    Drew van Camp (drew@cs.toronto.edu)
#
# Copyright (c) 1996 The University of Toronto.
#
# See the file "copyright" for information on usage and redistribution
# of this file, and for a DISCLAIMER OF ALL WARRANTIES.
#
# ----------------------------------------------------------------------

# ----------------------------------------------------------------------
# d_loadRootInfo - Collects information about the root data directory
# in "cpath" and puts it into the array "arrayName".  Information keys
# are stored as "$key1,$key" to ease book-keeping.  "Cpath" must be a
# canonical delve data path.
# ----------------------------------------------------------------------

proc d_loadRootInfo { cpath arrayName key1 } {
    upvar $arrayName dinfo ;
    global	delve_path ;
    global	d_priv ;
    
    if { ![d_file exists $cpath] } {
 	error "root data directory \"$cpath\" not found" ;
    } elseif { [string compare [d_type $cpath] "root"] } {
	error "\"$cpath\" is not a data root directory" ;
    }

    if [info exists dinfo] {
	foreach name [array names dinfo] {
	    if [regexp "^$key1," $name] {
		unset dinfo($name) ;
	    }
	}
    }
    set dinfo($key1,delve_path)	$delve_path ;

    set dinfo($key1,datasets)	"" ;
    foreach file [d_readdir $cpath] {
	if [d_file exists $key1/$file/$d_priv(datasetSpec)] {
	    lappend dinfo($key1,datasets) $file ;
	}
    }

    set dinfo($key1,loaded)	"1" ;
}

# ----------------------------------------------------------------------
# m_loadRootInfo - Collects information about the method directory in
# "cpath" and puts it into the array "arrayName".  Information keys
# are stored as "$key1,$key" to ease book-keeping.  "Cpath" must be a
# canonical delve data path.
# ----------------------------------------------------------------------

proc m_loadRootInfo { cpath arrayName key1 } {
    upvar $arrayName minfo ;
    global	delve_path ;
    global	d_priv ;
    
    if { ![m_file exists $cpath] } {
 	error "root methods directory \"$cpath\" not found" ;
    } elseif { [string compare [m_type $cpath] "root"] } {
	error "\"$cpath\" is not a root methods directory" ;
    }

    if [info exists minfo] {
	foreach name [array names minfo] {
	    if [regexp "^$key1," $name] {
		unset minfo($name) ;
	    }
	}
    }
    set minfo($key1,delve_path)	$delve_path ;

    set minfo($key1,methods)	"" ;
    foreach file [m_readdir $key1] {
	if [m_file isdirectory $cpath/$file] {
	    lappend minfo($key1,methods) $file ;
	}
    }

    set minfo($key1,loaded)	"1" ;
}

# ----------------------------------------------------------------------
# m_loadMethodInfo - Same as d_loadRootInfo, but for method directories.
# ----------------------------------------------------------------------

proc m_loadMethodInfo { cpath arrayName key1 } {
    upvar $arrayName minfo ;
    global d_priv ;

    d_loadRootInfo [m_cvtcpath $cpath] minfo $key1 ;

    set minfo($key1,datasets)	"" ;
    foreach file [m_readdir $cpath] {
	if { [m_file isdirectory $cpath/$file]
	     && [d_file isdirectory /$file] } {
	    lappend minfo($key1,datasets) $file ;
	}
    }
}
