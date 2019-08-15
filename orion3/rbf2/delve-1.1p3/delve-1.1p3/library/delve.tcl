# ----------------------------------------------------------------------
# $Id: delve.tcl,v 1.6.2.6 1996/12/04 19:14:05 revow Exp $
#
# Tcl utility procedures used by all scripts in this directory.
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

#----------------------------------------------------------------------#
# Returns 1 if the cpath points to a delve data module that actually
# exists
#----------------------------------------------------------------------#

proc d_exists { cpath } {
    if [catch {d_type $cpath} type] {
	return 0 ;
    }
    switch $type {
	root	-
	dataset	-
	prototask {
	    return [d_file isdirectory $cpath] ;
	}
	task {
	    set dirname [file dirname $cpath] ;
	    set tail	[file tail $cpath] ;
	    return [expr { [d_file isdirectory $dirname]
			   && [lsearch [d_info $dirname tasks] $tail] >= 0 } ]
	}
	default {
	    error "bad type \"$type\"" ;
	}
    }
}

#----------------------------------------------------------------------#
# Returns 1 if the cpath points to a delve method module that actually
# exists
#----------------------------------------------------------------------#

proc m_exists { cpath } {
    if [catch {m_type $cpath} type] {
	return 0 ;
    } else {
	return [m_file isdirectory $cpath] ;
    }
}

#----------------------------------------------------------------------#
# Return the type of the delve "cpath", assuming it points to a delve
# data directory: dataset, prototask, or task. Return an error if we
# can't figure out what "path" is.
#----------------------------------------------------------------------#

proc d_type { cpath } {
    switch [llength [split [string trim $cpath "/"] "/"]] {
	0	{ return root }
	1	{ return dataset }
	2	{ return prototask }
	3	{ return task }
	default	{ error "path contains too many directory names: \"$cpath\"" }
    }
}

#----------------------------------------------------------------------#
# Return the type of the delve "cpath", assuming it points to a delve
# methods directory: dataset, prototask, or task. Return an error if
# we can't figure out what "path" is.
#----------------------------------------------------------------------#

proc m_type { cpath } {
    switch [llength [split [string trim $cpath "/"] "/"]] {
	0	{ return root }
	1	{ return method }
	2	{ return dataset }
	3	{ return prototask }
	4	{ return task }
	default	{ error "path contains too many directory names: \"$cpath\"" }
    }
}

#----------------------------------------------------------------------#
# Evaluate a command; but before doing it do the following
# replacements:
#	%t -> "type"
#	%T -> "Type"
#	%% -> %
# (where "type" is the string value stored in the type argument).
#----------------------------------------------------------------------#

proc delve_eval { type command } {
    set type [string tolower $type]
    set Type [string toupper [string index $type 0]][string range $type 1 end]
    regsub -all {([^%]|^)%t} $command "\\1$type" command ;
    regsub -all {([^%]|^)%T} $command "\\1$Type" command ;
    regsub -all {([^%]|^)%%} $command "\\1" command ;

    set code [catch {uplevel $command} result] ;

    if { $code == 1 } {
	global errorInfo ;
	global errorCode ;
	return -code $code -errorinfo $errorInfo -errorcode $errorCode $result
    } else {
	return -code $code $result ;
    }
}

#----------------------------------------------------------------------#
# Retrieves the value for "key" for the given canonical data path and
# returns the value.
#----------------------------------------------------------------------#

proc d_info { cpath key } {
    global dinfo ;

    while { 1 } {			
       if ![info exists dinfo($cpath,loaded)] {
	    delve_eval [d_type $cpath] {
		d_load%TInfo $cpath dinfo $cpath ;
	    }
	}
	if { [info exists dinfo($cpath,$key)] } {
	    return $dinfo($cpath,$key) ;
	} elseif { [string compare $cpath "/"] != 0 } {
	    set cpath [file dirname $cpath] ;
	} else {
	    error "unknown key \"$key\"" ;
	}
    }
}

#----------------------------------------------------------------------#
# Retrieves the value for "key" for the given canonical method path
# and returns the value.
#----------------------------------------------------------------------#

proc m_info { cpath key } {
    global minfo ;

    while { 1 } {
	if ![info exists minfo($cpath,loaded)] {
	    delve_eval [m_type $cpath] {
		m_load%TInfo $cpath minfo $cpath ;
	    }
	}
	if { [info exists minfo($cpath,$key)] } {
	    return $minfo($cpath,$key) ;
	} elseif { [string compare $cpath "/"] != 0 } {
	    set cpath [file dirname $cpath] ;
	} else {
	    error "unknown key \"$key\"" ;
	}
    }
}

#----------------------------------------------------------------------#
# Loads a key file that must contain exactly the keys in "keys" and
# returns a list of keys and values: {key value key value key value ...}
#----------------------------------------------------------------------#

proc loadKeyFile { file keys }  {
    set list {} ;
    for_file line $file {
	set line [stripComment $line] ;
	if ![lempty $line] {
	    lappend list $line ;
	}
    }

    if { [llength $list] < [llength $keys] } {
	error "expected at least [llength $keys] lines in \"$file\" but got [llength $list]" ;
    }

    foreach spec $list {
	set pair	[split $spec ":"] ;
	set key		[string tolower [lindex $pair 0]] ;
	set array($key)	[string trim 	[lindex $pair 1]] ;
    }


    foreach key $keys {
	if ![info exists array($key)] {
	    error "\"$key\" not specified in \"$file\"" ;
	}
    }

    set result {} ;
    foreach key [array names array] {
	lappend result $key $array($key) ;
    }
    return $result ;
}

#----------------------------------------------------------------------#
# Strips a comment from a string. Comments are started with "#", but
# the "#" can be escaped by preceding it with a "\".
#----------------------------------------------------------------------#

proc stripComment { string } {
    set cidx	[string first "\#" $string] ;
    set pcidx	[expr { $cidx - 1 }] ;

    if { $cidx < 0 } {
	set result	$string ;
    } elseif { $cidx == 0 } {
	set result	{} ;
    } elseif { [string compare [string index $string $pcidx] "\\"] != 0 } {
	set result	[string range $string 0 $pcidx] ;
    } else {
	set result 	[string range $string 0 $cidx] ;
	append result	[stripComment [string range $string [incr cidx] end]] ;
    }
    return $result ;
}
