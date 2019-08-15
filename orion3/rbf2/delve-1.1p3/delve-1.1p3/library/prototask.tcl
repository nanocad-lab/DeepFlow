# ----------------------------------------------------------------------
# $Id: prototask.tcl,v 1.9.2.9 1996/12/04 19:15:34 revow Exp $
#
# Tcl procedures for manipulating prototasks.
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
# d_loadPrototaskInfo - Collects information about the prototask data
# directory in "cpath" and puts it into the array "arrayName".
# Information keys are stored as "$key1,$key" to ease book-keeping.
# "Cpath" must be a canonical delve data path.
# ----------------------------------------------------------------------

proc d_loadPrototaskInfo { cpath arrayName key1 } {
    upvar $arrayName dinfo ;
    global d_priv ;

    if { ![d_file exists $cpath] } {
 	error "prototask \"$cpath\" not found" ;
    } elseif { [string compare [d_type $cpath] "prototask"] } {
	error "$cpath is not a DELVE prototask" ;
    }

    set specfile [d_getFile $cpath/$d_priv(prototaskSpec) "specification file"]

    set dinfo($key1,prototask) $cpath ;

    set reqKeys	[list cases origin inputs order targets \
		     maximum-number-of-instances\
		     test-set-size training-set-sizes test-set-selection] ;
    set list	[loadKeyFile $specfile $reqKeys] ;
    while { ! [lempty $list] } {
	set key [lpop list] ;
	set dinfo($key1,$key) [lpop list] ;
    }

    set dinfo($key1,priors)	"" ;
    regsub -all " +" [d_compressExt] "|" extList
    foreach file [d_readdir $cpath] {
       if [regexp $d_priv(priorGlobPattern) $file] {
	  regsub "($extList)?$" $file "" plainName
	  lappend dinfo($key1,priors) $plainName ;
       }
    }

    set dinfo($key1,tasks)	"" ;
    foreach instance $dinfo($key1,training-set-sizes) {
	foreach file $dinfo($key1,priors) {
	    lappend dinfo($key1,tasks) \
		[genTaskName [file tail $file] $instance] ;
	}
    }

    set files $dinfo($key1,priors) ;
    lappend files \
	$d_priv(prototaskSpec) $dinfo($key1,cases) $dinfo($key1,order) ;

    set dinfo($key1,mtime) [d_info [file dirname $cpath] mtime] ;
    foreach file $files {
	if { [d_file exists $cpath/$file] 
	     && [d_file mtime $cpath/$file] > $dinfo($key1,mtime) } {
	    set dinfo($key1,mtime) [d_file mtime $cpath/$file] ;
	}
    }
    
    set dinfo($key1,maximum-training-sets)	"2" ;
    set dinfo($key1,loaded)	"1" ;
}

# ----------------------------------------------------------------------
# m_loadPrototaskInfo - Same as d_loadPrototaskInfo, but for method
# directories.
# ----------------------------------------------------------------------

proc m_loadPrototaskInfo { cpath arrayName key1 } {
    upvar $arrayName minfo ;

    d_loadPrototaskInfo [m_cvtcpath $cpath] minfo $key1 ;

    set minfo($key1,tasks)	"" ;
    foreach file [m_readdir $cpath] {
	if [m_file isdirectory $cpath/$file] {
	    lappend minfo($key1,tasks) $file ;
	}
    }
}

#----------------------------------------------------------------------#
# Proc to generate a prototask data file from the data in the
# canonical path "cpath" and store it in "outfile". If "force" is
# zero, pre-existing files will not be over-written.
#----------------------------------------------------------------------#

proc genPrototask { cpath outfile { force 0 } } {
    puts stderr "generating: $outfile"  ;

    pwrap "  extracting cases..." {
	extractCases $cpath case ;
    }
    pwrap "  creating file..." {
	set out  [wopen $outfile $force] ;
	printCases $cpath case $out ;
	close $out ;
    }
}

#----------------------------------------------------------------------#
# Proc to generate a Random-order file for a prototask.  If "force" is
# zero, a pre-existing file will not be over-written.
#----------------------------------------------------------------------#

proc genRandomorder { cpath outfile { force 0 } } {

    puts stderr "generating: $outfile" ;

    pwrap "  counting cases..." {
	extractCases $cpath case ;
    }
    pwrap "  creating file..." {
	set out  [wopen $outfile $force] ;
	d_randList [array size case] $out;
	close $out ;
    }
}

#----------------------------------------------------------------------#
# The real guts of the task: extract all the relevant cases and stick
# them in the array "case" keyed by an integer index (being the order
# it was extracted from the dataset).
#----------------------------------------------------------------------#

proc extractCases { cpath caseName } {
    upvar $caseName  case ;
    global d_priv ;


    set attrs	  [concat [d_info $cpath inputs] [d_info $cpath targets]] ;
    set rawCase	  {} ;
    set recNum	  1 ;
    set rawRecNum 1 ;

    set datafile [d_getFile [file dirname $cpath]/$d_priv(datasetFile) \
		      "data file"] ;
    for_file line $datafile {
	
	#
	# Build up the raw case by collecting lines ending with "\".
	#

	append rawCase $line ;

	if [regexp {^(.*)\\$} $rawCase all allButSlash] {
	    set rawCase $allButSlash ;
	    continue ;
	}

	#
	# Strip off trailing stuff (comments and commonality indices),
	# then grab the relevant attributes.
	#

	set comment	[removeTail rawCase "#"] ;
	set commonality	[removeTail rawCase "@"] ;

	set string	"" ;
	foreach attr $attrs {
	    append string " [lindex $rawCase [incr attr -1]]" ;
	}

	#
	# We're done with the raw data now, so get it ready for the
	# next pass.
	#

	set rawCase	{} ;

	#
	# Check if the case is to be included, then add back the
	# comments etc.
	#

	if [useCase $cpath $rawRecNum $string] {
	    if { [string compare $commonality {}] != 0 } {
		append string " $commonality"
	    }
	    if { [string compare $comment {}] != 0 } {
		append string " $comment"
	    }

	    set case($recNum) [string trim $string] ;
	    incr recNum ;
	}

	incr rawRecNum ;
    }
}

#----------------------------------------------------------------------#
# Remove all text including and after the final occurence of "$token"
# from the variable called varName. Return the string removed.
#----------------------------------------------------------------------#

proc removeTail { varName token } {
    upvar $varName string ;

    set idx	[string last "$token" $string] ;
    if { $idx < 0 } {
	return ;
    }
    set tail	[string range $string $idx end] ;
    set string	[string range $string 0 [incr idx -1]] ;

    return $tail ;
}

#----------------------------------------------------------------------#
# Check if a case should be included in the prototask. "caseNum" is
# the index of the case *from the database file*. "list" is the list
# of inputs and targets (only the ones that will be included in the
# proto-task). The decision is based on the prototasks "cases"
# property ("all", "no missing", or a file name).
#----------------------------------------------------------------------#

proc useCase { cpath caseNum list } {

    set cases	[d_info $cpath cases] ;
    if { [string compare $cases "all"] == 0 } {
	return 1 ;
    } elseif { [string compare $cases "no missing"] == 0 } {
	expr { [lsearch -exact $list ?] >= 0 ? 0 : 1 } ;
    } else {
	global	caseList ;
	if ![info exists caseList] {
	    for_file idx [d_getFile $cpath/$cases "order file"] {
		set caseList($idx)	1 ;
	    }
	}
	info exists caseList($caseNum) ;
    }
}

#----------------------------------------------------------------------#
# Prints all the cases in the proto-task to the stream "out" using the
# order specified by the prototasks "order" property (either "retain"
# to keep the order from the dataset, or the name of a file with the
# order (one number perline)).
#----------------------------------------------------------------------#

proc printCases { cpath caseName out } {
    upvar $caseName	case ;

    set order	[d_info $cpath order] ;

    if { [string compare $order "retain"] == 0 } {
	set limit	[expr { [array size case] + 1 }] ;
	for { set idx 1 } { $idx < $limit } { incr idx } {
	    puts $out	$case($idx) ;
	}
    } else {
	for_file idx [d_getFile $cpath/$order "order file"] {
	    if ![info exists case([string trim $idx])] {
		error "case [string trim $idx] doesn't exist (must be less than [array size case]" ;
	    }
	    puts $out	$case([string trim $idx]) ;
	}
    }
}
