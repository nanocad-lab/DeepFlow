# ----------------------------------------------------------------------
# $Id: taskInfo.tcl,v 1.4.2.5 1996/12/04 19:15:34 revow Exp $
#
# Tcl procedures for manipulating tasks.
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

# ----------------------------------------------------------------------
# Collects information about the task in directory "path" and puts it
# into the array "arrayName". "Path" can be either a master or a
# working directory.
# ----------------------------------------------------------------------

proc d_loadTaskInfo { cpath arrayName key1 } {
    upvar $arrayName dinfo ;

    set priorcpath 	[delveFileCmd rootname $cpath].prior ;
    set tasksize 	[string range [file extension $cpath] 1 end] ;

    if { ![d_file exists $priorcpath] } {
 	error "prior file \"$priorcpath\" not found" ;
    } elseif { [string compare $tasksize {}] == 0 } {
	error "$cpath does not specify a task training set size" ;
    } elseif { [string compare [d_type $cpath] "task"] } {
	error "$cpath is not a DELVE task" ;
    }

    set dinfo($key1,mtime) 		[d_info [file dirname $cpath] mtime] ;
    set dinfo($key1,task)		$cpath ;
    set dinfo($key1,prior,file)		$priorcpath ;
    set dinfo($key1,training-set-size)	$tasksize ;
    set dinfo($key1,targetAveDev)	"unknown" ;
    set dinfo($key1,targetVariance)	"unknown" ;

    d_loadPriorInfo $cpath dinfo $key1 ;

    set dinfo($key1,loaded)	"1" ;
}

proc m_loadTaskInfo { cpath arrayName key1 } {
    upvar $arrayName minfo ;

    d_loadTaskInfo [m_cvtcpath $cpath] minfo $key1 ;

    loadEncoding [m_getFile $cpath/Coding-used "coding file"] \
	[concat [m_info $cpath inputs] [m_info $cpath targets]] ;
}

# ----------------------------------------------------------------------
# Collects prior information about the task in directory "path" and
# puts it into the array "arrayName". "Path" must be a master
# directory.
# ----------------------------------------------------------------------

proc d_loadPriorInfo { cpath arrayName key1 } {
    upvar $arrayName dinfo ;

    set priorfile [d_getFile $dinfo($key1,prior,file) "prior file"]

    for_file line $priorfile {
	set idx 	[lindex $line 0] ;

	set dinfo($key1,prior,level,$idx) [string tolower [lindex $line 1]] ;
	set dinfo($key1,prior,type,$idx)  [string tolower [lindex $line 2]] ;
	foreach option [lrange $line 3 end] {
	    set list	[split $option "="] ;
	    set key	[string tolower [lindex $list 0]] ;
	    set value	[lindex $list 1] ;

	    set dinfo($key1,prior,$key,$idx) $value ;
	}
    }

    set pcpath	[file dirname $cpath] ;
    set dcpath	[file dirname $pcpath] ;
    set attrs	[concat [d_info $pcpath inputs] [d_info $pcpath targets]] ;
    foreach attr $attrs {
	acdc_new $dinfo($key1,prior,type,$attr) [d_info $dcpath code,$attr] \
	    [list $attr] ;

	acdc_options $attr -range [d_info $dcpath range,$attr] ;
	foreach key [list passive center unit] {
	    if [info exists dinfo($key1,prior,$key,$attr)] {
		acdc_options $attr -$key $dinfo($key1,prior,$key,$attr) ;
	    }
	}

	if { [string compare $dinfo($key1,prior,type,$attr) "binary"] == 0 } {
	    if [info exists dinfo($key1,prior,passive,$attr)] {
		acdc_method $attr -default "0/1"
		acdc_method $attr "0/1"
	    } else {
		acdc_method $attr -default -- "-1/+1"
		acdc_method $attr -- "-1/+1"
	    }
	}
    }
}

# ----------------------------------------------------------------------
# Takes the name of a prior file, and an instance size, and generates
# the name of the corresponding task.
# ----------------------------------------------------------------------

proc genTaskName { prior instance } {
    return [file root $prior].$instance ;
}
