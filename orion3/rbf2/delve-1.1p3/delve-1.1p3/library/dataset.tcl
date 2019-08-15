# ----------------------------------------------------------------------
# $Id: dataset.tcl,v 1.3.2.12 1996/12/04 19:15:32 revow Exp $
#
# Tcl procedures for manipulating datasets.
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
# d_loadDatasetInfo - Collects information about the dataset directory
# in "cpath" and puts it into the array "arrayName".  Information keys
# are stored as "$key1,$key" to ease book-keeping.  "Cpath" must be a
# canonical delve data path.
# ----------------------------------------------------------------------

proc d_loadDatasetInfo { cpath arrayName key1 } {
    upvar $arrayName dinfo ;
    global d_priv ;

    if { ![d_file exists $cpath] } {
 	error "dataset \"$cpath\" not found" ;
    } elseif { [string compare [d_type $cpath] "dataset"] } {
	error "\"$cpath\" is not a dataset" ;
    }

    if [info exists dinfo] {
	foreach name [array names dinfo] {
	    if [regexp "^$key1," $name] {
		unset dinfo($name) ;
	    }
	}
    }
    set dinfo($key1,dataset)	$cpath ;

    set specfile [d_getFile $cpath/$d_priv(datasetSpec) "specification file"]

   set fp	[delve_openForRead $specfile  ]
    set inAttrs	0 ;
    set reqKeys	[list origin usage order attributes] ;
    while { [gets $fp line] >= 0} {
	if { $inAttrs && ![regexp {^[a-zA-Z _]*:.*} $line] } {
	    set line [parseAttribute $line] ;
	    set idx	 [lpop line] ;
	    lappend dinfo($key1,attributes) $idx ;
	    set dinfo($key1,code,$idx)		[lindex $line 0] ;
	    set dinfo($key1,nature,$idx)	[lindex $line 1] ;
	    set dinfo($key1,range,$idx)		[lindex $line 2] ;
	    set dinfo($key1,comment,$idx)	[lindex $line 3] ;
	} else {
	    set index [string first "#" $line] ;
	    if { $index > 0 } {
		set line [string range $line 0 [incr index -1]] ;
	    } elseif { $index == 0 } {
		set line "" ;
	    } 
	    set line [string trim $line] ;
		
	    if { [string compare $line ""] == 0 } {
		continue ;
	    }
	    set list	[split $line ":"] ;
	    set key	[string tolower [lindex $list 0]] ;
	    set value	[lindex $list 1] ;
	    if [info exists dinfo($key1,$key)] {
		error "$key specified twice in \"$cpath/$d_priv(datasetSpec)\"" ;
	    }
	    if { [string compare $key "attributes"] == 0 } {
		set inAttrs	1 ;
	    } else {
		set inAttrs	0 ;
		set dinfo($key1,$key) [string trim $value] ;
	    }
	}
    }
    close $fp ;

    foreach key $reqKeys {
	if ![info exists dinfo($key1,$key)] {
	    error "\"$key\" not specified in \"$cpath/$d_priv(datasetSpec)\"" ;
	}
    }

    if ![info exists dinfo($key1,title)] {
	set dinfo($key1,title)	"no title" ;
    }
    set dinfo($key1,number-of-attributes) [llength $dinfo($key1,attributes)] ;
    set dinfo($key1,mtime) [delveFileCmd mtime $specfile] ;
	
    set dinfo($key1,prototasks)	"" ;
    foreach file [d_readdir $key1] {
	if [d_file exists $key1/$file/$d_priv(prototaskSpec)] {
	    lappend dinfo($key1,prototasks) $file ;
	}
    }

    set dinfo($key1,loaded)	"1" ;
}

# ----------------------------------------------------------------------
# m_loadDatasetInfo - Same as d_loadDatasetInfo, but for method
# directories.
# ----------------------------------------------------------------------

proc m_loadDatasetInfo { cpath arrayName key1 } {
    upvar $arrayName minfo ;
    global d_priv ;

    d_loadDatasetInfo [m_cvtcpath $cpath] minfo $key1 ;

    set minfo($key1,prototasks)	"" ;
    foreach file [m_readdir $cpath] {
	if [m_file isdirectory $cpath/$file] {
	    lappend minfo($key1,prototasks) $file ;
	}
    }
}

# ----------------------------------------------------------------------
# A simple procedure that makes parsing the attributes lines in a
# Dataset.spec file easier.
# ----------------------------------------------------------------------

proc parseAttribute { line } {
    set idx	[lindex $line 0] ;
    set name	[lindex $line 1] ;
    set nature	[lindex $line 2] ;
    set rest	[split [lrange $line 3 end] "\#"] ;
    set range	[string trim [lindex $rest 0]] ;
    if { [llength $rest] > 1 } {
	set comment [string trim [eval concat [lrange $rest 1 end]]] ;
    } else {
	set comment {} ;
    }
    list $idx $name $nature $range $comment ;
}
