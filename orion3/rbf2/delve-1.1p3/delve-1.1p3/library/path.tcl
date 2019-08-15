# ----------------------------------------------------------------------
# $Id: path.tcl,v 1.9.2.10.2.1 1997/11/27 15:42:10 revow Exp $
#
# Tcl procedures for manipulating canonical paths.
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
# Get rid of multiple slashes between directories, plus any trailing
# slashes. E.g. "/a//b/c///d/" becomes "/a/b/c/d", and return the result.
# ----------------------------------------------------------------------

proc cleanpath path {
    regsub -all {/+} $path {/}  path
    regsub {(.)/+$}  $path {\1} path
    return $path ;
}

# ----------------------------------------------------------------------
# Returns one if path could possibly be interpreted as a canonical
# path (i.e. it starts with "/").  Some day this may become more
# complex.  If "exist" is present and non-zero, the file must also exist.
# ----------------------------------------------------------------------

proc delve_iscpath { path module { exists 0 } } {
    expr { [string compare [string index $path 0] "/"] == 0
	   && ( $exists == 0 || [delve_file exists $path $module] ) }
}

# ----------------------------------------------------------------------
# Returns one if path could possibly be interpreted as a delve data
# path. If "exist" is present and non-zero, the file must also exist.
# ----------------------------------------------------------------------

proc d_iscpath { path { exist 0 } } {
    uplevel [list delve_iscpath $path "data" $exist] ;
}

# ----------------------------------------------------------------------
# Returns one if path could possibly be interpreted as a delve methods
# path.  If "exist" is present and non-zero, the file must also exist.
# ----------------------------------------------------------------------

proc m_iscpath { path { exist 0 } } {
    uplevel [list delve_iscpath $path "methods" $exist] ;
}

# ----------------------------------------------------------------------
# Given a real path, try to find the delve root part of the path.  A
# delve root directory is defined as any directory that contains a
# directory called "data" and/or "methods".  For example if path were
# /usr/local/delve/data/foo/bar, the procedure would return
# /usr/local/delve.  If path is not part of a delve directory, an
# empty string is returned.  The actual delve root directory is 
# considered to be inside itself.
# ----------------------------------------------------------------------

proc delve_root { path } {
    while { ![file isdirectory $path] } {
	set path [file dirname $path] ;
    }

    set wd	[pwd] ;
    cd $path ;
    set path	[pwd] ;
    cd $wd ;

    while 1 {
	if { [string match "delve*" [file tail $path]]
	     && [file isdirectory $path/data]
	     && [file isdirectory $path/methods] } {
	    return $path ;
	} elseif { [string compare $path "/"] == 0 } {
	    return {} ;
	}
	set path	[file dirname	$path] ;
    }
}

# ----------------------------------------------------------------------
# Returns the canonical path for the real "path". Module should be
# "data" or "methods", depending on which type of path you want.
# ----------------------------------------------------------------------

proc delve_cpath { path module } {
    global delve_path ;

    set dir  $path ;
    set tail {} ;

    while { ![file isdirectory $dir] } {
	lpush tail [file tail $dir] ;
	set dir [file dirname $dir] ;
    }

    set wd [pwd] ;
    cd $dir ;
    set list [concat [split [pwd] "/"] $tail] ;
    cd $wd ;

    set idx 	[lsearch $list $module] ;
    if { $idx < 0 } {
	error "\"$path\" is not in a delve \"$module\" hierarchy" ;
    }

    while { $idx >= 0 } {
	set list	[lrange $list [incr idx] end] ;
	set idx	[lsearch $list $module] ;
    }

    return "/[join $list /]" ;
}

# ----------------------------------------------------------------------
# Returns the canonical data path for the real "path". Note that this
# could be from a method directory as well.
# ----------------------------------------------------------------------

proc d_cpath { path } {
    if [catch [list uplevel [list delve_cpath $path "data"]] result] {
	if [catch [list uplevel [list delve_cpath $path "methods"]] nextResult] {
	    error $result ;
	}
	set result [m_cvtcpath $nextResult] ;
    }
    return $result ;
}

# ----------------------------------------------------------------------
# Returns the canonical method path for the real "path". Note that this
# *cannot* be from a data directory.
# ----------------------------------------------------------------------

proc m_cpath { path } {
    uplevel [list delve_cpath $path "methods"] ;
}

# ----------------------------------------------------------------------
# Given a canonical method path, return the corresponding canonical
# data path.
# ----------------------------------------------------------------------

proc m_cvtcpath { mcpath } {
    list "/[join [lrange [split $mcpath /] 2 end] /]" ;
}

# ----------------------------------------------------------------------
# Return all real paths that map to the canonical path "cpath".
# Modules should be either "data" or "methods", depending on which you
# want.
# ----------------------------------------------------------------------

proc delve_paths { cpath module } {
    global delve_path ;

    set root [delve_root [pwd]] ;
    if { [string compare $root ""] != 0
	 && [lsearch $delve_path $root] < 0 
	 && [delveFileCmd exists ${root}/${module}${cpath}] } {
	set result ${root}/${module}${cpath} ;
    } else {
	set result {} ;
    }

    foreach path $delve_path {
	if [delveFileCmd exists ${path}/$module${cpath}] {
	    lappend result ${path}/$module${cpath} ;
	}
    }
    return $result ;
}

# ----------------------------------------------------------------------
# Return all real paths that map to the canonical data path "cpath".
# ----------------------------------------------------------------------

proc d_paths { cpath } {
    uplevel [list delve_paths $cpath data] ;
}

# ----------------------------------------------------------------------
# Return all real paths that map to the canonical method path "cpath".
# ----------------------------------------------------------------------

proc m_paths { cpath } {
    uplevel [list delve_paths $cpath methods] ;
}

# ----------------------------------------------------------------------
# Return the *single* file that maps to the canonical path "cpath". If
# zero or more than one file map, generate an error. Module should be
# either "data" or "methods".
# ----------------------------------------------------------------------

proc delve_getFile { cpath name module } {
    set files [delve_paths $cpath $module] ;
    if { [llength $files] == 0 } {
	error "$name \"$cpath\" not found in delve path"
    } elseif { [llength $files] > 1 } {
	error "multiple copies of $name found in delve path: $files"
    }

    return [lindex $files 0] ;
}

# ----------------------------------------------------------------------
# Return the *single* file that maps to the canonical data path
# "cpath".  If zero or more than one file map, generate an error.
# ----------------------------------------------------------------------

proc d_getFile { cpath { name "file" } } {
    uplevel [list delve_getFile $cpath $name data] ;
}

# ----------------------------------------------------------------------
# Return the *single* file that maps to the canonical methods path
# "cpath".  If zero or more than one file map, generate an error.
# ----------------------------------------------------------------------

proc m_getFile { cpath { name "file" } } {
    uplevel [list delve_getFile $cpath $name methods] ;
}

# ----------------------------------------------------------------------
# Equivalent to the Tcl "file" command, but acts on canonical paths.
# Module should be either "data" or "methods".
# ----------------------------------------------------------------------

proc delve_file { option cpath module args } {
    set files [delve_paths $cpath $module] ;

    switch -- $option {
	executable -
	exists	-
	isdirectory -
	isfile	-
	owned	-
	readable -
	writable {
	    if [lempty $files] {
		return 0 ;
	    }
	    foreach file $files {
		lappend list [eval [list delveFileCmd $option $file] $args] ;
	    }
	    set result	[lrmdups $list] ;
	    if { [llength $result] != 1 } {
		error "conflicting results for \"file $option $cpath\": $result"
	    }
	    return [lindex $result 0] ;
	}

	readlink -
	size 	-
	type	{
	    if [lempty $files] {
		error "couldn't stat option $option \"$cpath\": No such file or directory" ;
	    }
	    foreach file $files {
		lappend list [eval [list delveFileCmd $option $file] $args] ;
	    }
	    set result	[lrmdups $list] ;
	    if { [llength $result] != 1 } {
		error "conflicting results for \"file $option $cpath\": $result"
	    }
	    return [lindex $result 0] ;
	}

	atime	-
	mtime	{
	    if [lempty $files] {
		error "couldn't stat option $option \"$cpath\": No such file or directory" ;
	    }
	    foreach file $files {
		lappend list [eval [list delveFileCmd $option $file] $args] ;
	    }
	    if { [llength $list] == 1 } {
		return [lindex $list 0] ;
	    } else {
		return [eval max $list] ;
	    }
	}
	
	extension -
	dirname	-
	rootname -
	tail	{
	    return [eval [list file $option $cpath] $args] ;
	}


	lstat	-
	stat	{
	    if { [llength $files] != 1 } {
		error "can't $option $cpath, it doesn't map to exactly one file"
	    }
	    return [uplevel [list delveFileCmd $option [lindex $files 0]] $args] ;
	}
    }
}

# ----------------------------------------------------------------------
# Equivalent to the Tcl "file" command, but acts on canonical data
# paths.
# ----------------------------------------------------------------------

proc d_file { option cpath args } {
    uplevel [list delve_file $option $cpath data] $args ;
}

# ----------------------------------------------------------------------
# Equivalent to the Tcl "file" command, but acts on canonical methods
# paths.
# ----------------------------------------------------------------------

proc m_file { option cpath args } {
    uplevel [list delve_file $option $cpath methods] $args ;
}

# ----------------------------------------------------------------------
# Equivalent to the Tcl "readdir" command, but acts on canonical
# paths.  Module should be either "data" or "methods".
# ----------------------------------------------------------------------

proc delve_readdir { cpath module } {
    set list {}
    foreach path [delve_paths $cpath $module] {
	foreach file [glob -nocomplain $path/*] {
	    lappend list [file tail $file] ;
	}
    }
    lrmdups $list ;
}

# ----------------------------------------------------------------------
# Equivalent to the Tcl "readdir" command, but acts on canonical data
# paths.  
# ----------------------------------------------------------------------

proc d_readdir { cpath } {
    uplevel [list delve_readdir $cpath "data"] ;
}

# ----------------------------------------------------------------------
# Equivalent to the Tcl "readdir" command, but acts on canonical methods
# paths.  
# ----------------------------------------------------------------------

proc m_readdir { cpath } {
    uplevel [list delve_readdir $cpath "methods"] ;
}

# ----------------------------------------------------------------------
# Equivalent to the unix "ls" command, but acts on canonical paths.
# Module should be either "data" or "methods". 
# ----------------------------------------------------------------------

proc delve_ls { cpath module long } {
    if ![delve_file exists $cpath $module] {
 	error "file \"$cpath\" not found in delve path" ;
    }

    if $long {
	if { [delve_file isfile $cpath $module] } {
	    return [delve_paths $cpath $module] ;

	} elseif [delve_file isdirectory $cpath $module] {
	    set list {}
	    foreach path [delve_paths $cpath $module] {
		lappend list $path ;
		set sublist	{} ;
		foreach file [glob -nocomplain $path/*] {
		    lappend sublist [file tail $file] ;
		}
		lappend list $sublist ;
	    }
	    return $list ;
	}
    } else {
	if { [delve_file isfile $cpath $module] } {
	    return $cpath ;
	} elseif [delve_file isdirectory $cpath $module] {
	    return [delve_readdir $cpath $module] ;
	}
    }
}

# ----------------------------------------------------------------------
# Equivalent to the unix "ls" command, but acts on canonical data
# paths.
# ----------------------------------------------------------------------

proc d_ls { cpath { long 0 } } {
    uplevel [list delve_ls $cpath "data" $long] ;
}

# ----------------------------------------------------------------------
# Equivalent to the unix "ls" command, but acts on canonical method
# paths.
# ----------------------------------------------------------------------

proc m_ls { cpath { long 0 } } {
    uplevel [list delve_ls $cpath "methods" $long] ;
}

# ----------------------------------------------------------------------
# Returns a list of all duplicate files in the canonical directory
# "cpath".  Module should be either "data" or "methods".
# ----------------------------------------------------------------------

proc delve_conflicts { cpath module } {
    if ![delve_file exists $cpath $module] {
	error "file \"$cpath\" not found in delve path" ;
    }
    set cpath "[string trimright $cpath /]/"

    if { [delve_file isfile $cpath $module] } {
	set cpaths [delve_paths $cpath $module] ;
	if { [llength $cpaths] > 1 } {
	    return [list $cpaths] ;
	} else {
	    return {} ;
	}
    } elseif [delve_file isdirectory $cpath $module] {
	set dirs	[delve_paths $cpath $module] ;
	set len		[llength $dirs] ;
	set conflicts	{} ;
	for { set i 0 } { $i < $len } { incr i } {
	    set dir(1)	 	[lindex $dirs $i] ;
	    set dirlist(1)	{}
	    foreach file [glob -nocomplain $dir(1)/*] {
		lappend dirlist(1) [file tail $file] ;
	    }
	    for { set j [expr { $i + 1 }] } { $j < $len } { incr j } {
		set dir(2)	[lindex $dirs $j] ;
		set dirlist(2)	{}
		foreach file [glob -nocomplain $dir(2)/*] {
		    lappend dirlist(2) [file tail $file] ;
		}
		set files {} ;
		foreach file [lintersect $dirlist(1) $dirlist(2)] {
		    if { [file isfile $dir(1)/$file] 
			 || [file isfile $dir(2)/$file] } {
			lappend files $file ;
		    }
		}
		set conflicts [lunion $conflicts $files] ;
	    }
	}
	set list {} ;
	foreach file $conflicts {
	    lappend list [delve_paths ${cpath}$file $module] ;
	}
	return $list ;
    }
}

# ----------------------------------------------------------------------
# Returns a list of all duplicate files in the canonical data directory
# "cpath".
# ----------------------------------------------------------------------

proc d_conflicts { cpath } {
    uplevel [list delve_conflicts $cpath "data"] ;
}

# ----------------------------------------------------------------------
# Returns a list of all duplicate files in the canonical methods
# directory "cpath".
# ----------------------------------------------------------------------

proc m_conflicts { cpath } {
    uplevel [list delve_conflicts $cpath "methods"] ;
}

# ----------------------------------------------------------------------
# Returns the name of a method given a canonical path
# ----------------------------------------------------------------------

proc m_name { cpath } {
    lindex [split $cpath /] 1 ;
}

# ----------------------------------------------------------------------
# Formats a list nicely for output to a tty.
# ----------------------------------------------------------------------

proc delve_format { list } {
    if { [string compare [info commands infox] {}] } {
	if { [fstat stdout tty] } {
	    #
	    # Have to call tput twice: the first time to see if it works
	    # (stderr onto /dev/null); the second time redirecting stderr
	    # onto stderr so that we can get the right column width.
	    #
	    
	    if [catch { exec tput cols <@ stdin 2> /dev/null } width] {
		set width 80 ;
	    } else {
		if [catch { exec tput cols <@ stdin 2>@ stderr } width] {
		    set width 80 ;
		}
	    }
	    
	    set max 0 ;
	    foreach word $list {
		set max [max $max [string length $word]] ;
	    }
	    set cols [expr { int($width /($max + 1)) }] ;
	} else {
	    set width 80
	    set cols 1 ;
	}
    } else {
	set width 80
	set max 0 ;
	foreach word $list {
	    set max [max $max [string length $word]] ;
	}
	set cols [expr { int($width /($max + 1)) }] ;
    }
    
    set rows	 [expr { int([llength $list] / $cols + 1) }] ;
    set colWidth [expr { int($width / $cols) }] ;
    
    set result {} ;
    for { set row 0 } { $row < $rows } { incr row } {
	for { set col 0 } { $col < $cols } { incr col } {
	    set index	[expr { int($row + $rows * $col) }] ;
	    append result [format "%-${colWidth}s" [lindex $list $index]] ;
	}
	append result "\n" ;
    }
    
    string trim $result ;
}

