# ----------------------------------------------------------------------
# $Id: loss.tcl,v 1.16.2.8 1996/11/12 16:55:41 revow Exp $
#
# Tcl procedures for calculating loss functions.
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
# Returns the name of a loss function given its code.
# ----------------------------------------------------------------------

proc lossName { type } {
    switch $type {
	"A" { return "Absolute error" ; }
	"S" { return "Squared error" ; }
	"L" { return "Negative Log probability" ; }
	"Z" { return "Zero-one" ; }
	"Q" { return "Squared Probability" ; }
	default {
	    error "unknown code \"$type\"" ;
	}
    }
}

# ----------------------------------------------------------------------
# Loads a series of single value loss files into the rows of a matrix
# (which it also creates). Returns the handle of the new matrix.
# ----------------------------------------------------------------------

proc loadLossFiles { fileList } {
    set nFiles	[llength $fileList] ;

    set row	0 ;
    foreach file $fileList {
	set values {} ;
	for_file line $file {
	    lappend values $line ;
	}

	if { ![info exists m] } {
	    set nCases	[llength $values] ;
	    set m	[d_matrix create $nFiles $nCases] ;
	}
	set col	0 ;
	foreach value $values {
	    d_matrix entryset $m $row $col $value ;
	    incr col ;
	}

	if { $col != $nCases } {
	    d_matrix delete $m ;
	    error "expected $nCases cases in $cfile, but got $col"
	}
	incr row ;
    }
    
    return $m ;
}

# ----------------------------------------------------------------------
# Returns a list of appropriate statistics describing the losses in
# the "fileList".  Each file should contain one value per line, the
# same number of lines per file.
#
# The list returned list has 6 or 7 entries (depending on the value of
# "test-set-selection". The returned values are:
# mean sig_a sig_e stdErr [sig_b] I J
# ----------------------------------------------------------------------

proc getLossStats { fileList test-set-selection } {
    if { [llength $fileList] < 2 } {
	error "can't do analysis of variance with fewer than two loss files" ;
    }

    set m	[loadLossFiles $fileList]
    set anova   [d_anova ${test-set-selection} "estimate" $m]
    set size    [d_matrix size $m]     
    d_matrix delete $m;

    lappend anova [lindex $size 0] [lindex $size 1]
    return $anova
}

# ----------------------------------------------------------------------
# Returns a list of appropriate statistics describing the differences
# between the losses in the "fileList1" and the "fileList2".  Each
# file should contain one value per line, the same number of lines per
# file. The "test-set-selection" is supplied to d_anova and used to
# decide which statistical test should be used.
#
# The list returned has 9 or 10 entries (depending on the value of
# "test-set-selection". The returned values are:
# mean1 mean mean2 p sig_a sig_e stdErr [sig_b] I J
# ----------------------------------------------------------------------

proc getLossDiffStats { fileList1 fileList2 test-set-selection } {
    if { [llength $fileList1] < 2 } {
	error "can't do analysis of variance with fewer than two loss files" ;
    }

    set m1	[loadLossFiles $fileList1] ;
    set m2 	[loadLossFiles $fileList2] ;
    set anova   [d_anova ${test-set-selection} "compare" $m1 $m2]
    set size    [d_matrix size $m1]     
    d_matrix delete $m1
    d_matrix delete $m2

    lappend anova [lindex $size 0] [lindex $size 1]
    return $anova
}

# ----------------------------------------------------------------------
# Normalizes all of the statistics in "statList" using the *targets*
# stored in "fileList". "lossType" is one of "S" "A" or "L". It is
# used to figure out the normalization constants. If it is not one of
# the above, all normalized values are "undefined" (as is anything it
# can't calculate. Keys modified in "arrayName" are: mu, stdError,
# s_noise, s_instance, s_cases.
# ----------------------------------------------------------------------

proc normalizeLossStats { loss statList arrayName } {
    upvar $arrayName stat ;
    set result {} ;

    if { ![info exists stat($loss)]
	 || [lsearch $stat($loss) "undefined"] >= 0 } {
	foreach value $statList {
	    lappend result "undefined" ;
	}
	return $result ;
    }
    switch $loss {
	"A" -
	"Q" -
	"S" -
	"Z" {
	    set average	[average $stat($loss)] ;
	    foreach value $statList {
		if { $average != 0.0 && [string compare $value "undefined"] } {
		    lappend result [expr { $value/$average } ] ;
		} else {
		    lappend result "undefined" ;
		}
	    }
	}
	"L" {
	    set expr "" ;
	    foreach element $stat($loss) {
		append expr " + $element" ;
	    }
	    set idxList	0 ;
	    if { [llength $statList] > 7 } {
		lappend idxList 2 ;
	    }
	    set result $statList ;
	    foreach idx $idxList {
		set value	[lindex $statList $idx] ;
		if { [string compare $value "undefined"] } {
		    set value	[expr "$value $expr"] ;
		} else {
		    set value	"undefined" ;
		}
		set result [lreplace $result $idx $idx $value] ;
	    }
	}
	default {
	    foreach value $statList {
		lappend result "undefined" ;
	    }
	}
    }
    return $result
}
