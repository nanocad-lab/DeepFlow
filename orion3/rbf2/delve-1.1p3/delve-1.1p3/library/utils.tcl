##############################################################
#
# $Id: utils.tcl,v 1.3.2.3.2.2 1997/04/22 17:38:00 revow Exp $
#
# DESCRIPTION
#   General common functions used by many routines mainly
#   for accessing the file system
#
# PROJECT:
#
# AUTHOR HISTORY:
#    Author Delve (delve@cs.toronto.edu)
#    Drew van Camp
#
# Copyright (c) 1996 The University of Toronto.
#
# See the file "copyright" for information on usage and redistribution
# of this file, and for a DISCLAIMER OF ALL WARRANTIES.
#
##############################################################

proc frename { oldpath newpath } {
    exec mv $oldpath $newpath ;
}

proc funlink { fileList } {
    if { [llength $fileList] > 0 } {
        eval exec rm -f $fileList ;
    }
}

proc mkdir { dirList args } {
    if { [string compare $dirList "-path"] == 0 } {
        set flag        -path
        set dirList     [lindex $args 0] ;
        set args        [lrange $args 1 end] ;
    } else {
        set flag        {} ;
    }

    if { [llength $args] != 0 } {
        error "wrong \# args: should be \"mkdir ?-path? dirList\"" ;
    }

    foreach dir $dirList {
        if { [string compare $flag "-path"] == 0 } {
            set dirname [file dirname $dir] ;
            if { ![file exists $dirname] } {
                mkdir -path $dirname ;
            }
        } 
        exec mkdir $dir ;
    }
}

proc for_file { varName fileName script } {
    upvar $varName line ;
   global errorInfo

   set fileId [delve_openForRead $fileName]

   if {$fileId != {} } {
    set code    0 ;
    set result  "" ;
    
    while { [gets $fileId line] >= 0 } {
        set code [catch {uplevel $script} result] ;
        if { $code != 0 && $code != 4 } {
	   puts "Warning: $errorInfo"
            break ;
        }
    }

    set code [catch {close $fileId} msg] ;
    if { $code != 0 } {
	puts "for_file: error closing file $msg $code"
    }
    
    if { $code == 1 } {
        global errorInfo errorCode ;
        return -code $code -errorinfo $errorInfo -errorcode $errorCode $result
    } elseif { $code == 0 || $code == 3 || $code == 4 } {
        return -code 0 $result ;
    } else {
        return -code $code $result ;
    }
   } else {
      return "Cannot find $fileName"
   }
}

proc cutFile { inFile outPrefix list { force 0 } } {
    set fid($inFile)    [open $inFile r] ;
    
    set outFiles    {} ;
    set limit       [llength $list] ;
    for { set idx 0 } { $idx < $limit } { incr idx } {
        set outFile $outPrefix.$idx ;
        
        if [catch {wopen $outFile $force} msg] {
            
            foreach name [array names fid] {
                close $fid($name) ;
            }
            if { [info exists outFiles] && ![lempty $outFiles] } {
                funlink $outFiles ;
            }
            
            global errorInfo errorCode ;
            error $msg $errorInfo $errorCode ;
        }
        set fid($outFile) $msg ;
        lappend outFiles $outFile ;
        
    }
    
    
    set numColumns   0 ;
    set column      -1 ;
    foreach file $outFiles {
        set split [lpop list] ;
    
        incr numColumns $split ;
    
        set first($file)    [expr { $column + 1 }] ;
        set last($file)     [incr column $split] ;
    }
    
    
    set lineNum 0 ;
    while { [gets $fid($inFile) line] != -1 } {
        if { [llength $line] != $numColumns } {
            
            foreach name [array names fid] {
                close $fid($name) ;
            }
            if { [info exists outFiles] && ![lempty $outFiles] } {
                funlink $outFiles ;
            }
            
            error "\"$inFile\", line $lineNum has [llength $line] columns, expected $numColumns"
        }
        foreach file $outFiles {
            puts $fid($file) [lrange $line $first($file) $last($file)] ;
        }
        incr lineNum ;
    }
    
    
    foreach file $outFiles {
        close $fid($file) ;
    }
    
    close $fid($inFile) ;
    return $outFiles ;
}

proc splitFile { inFile outPrefix list { force 0 } } {
    set fid($inFile)    [delve_openForRead $inFile] ;
    
    set outFiles    {} ;
    set ext         0 ;
    foreach split $list {
        set outFile $outPrefix.$ext ;
        
        if [catch {wopen $outFile $force} msg] {
            
            foreach name [array names fid] {
                close $fid($name) ;
            }
            if { [info exists outFiles] && ![lempty $outFiles] } {
                funlink $outFiles ;
            }
            
            global errorInfo errorCode ;
            error $msg $errorInfo $errorCode ;
        }
        set fid($outFile) $msg ;
        lappend outFiles $outFile ;
        
        
        if { [copyLines $fid($inFile) $fid($outFile) $split] != $split } {
            
            foreach name [array names fid] {
                close $fid($name) ;
            }
            if { [info exists outFiles] && ![lempty $outFiles] } {
                funlink $outFiles ;
            }
            
            error "$inFile has too few lines to create split $ext ($split lines)" ;
        }
        
        
        close $fid($outFile) ;
        unset fid($outFile) ;
        
        incr ext ;
    }
    
    close $fid($inFile) ;
    return $outFiles ;
}

proc copyLines { in out numLines } {
    for { set idx 0 } { $idx < $numLines } { incr idx } {
        if { [gets $in line] < 0 } {
            break ;
        }
        puts $out $line ;
    }
    return $idx ;
}

proc wopen { fileName { force 0 } } {
    set mode [expr { $force ? {WRONLY CREAT TRUNC} : {WRONLY CREAT EXCL} }] ;
    open $fileName $mode ;
}

proc max { num1 args } {
    lindex [lsort -real -decreasing [concat [list $num1] $args]] 0 ;
}

proc atPrecision { precision script } {
    global tcl_precision ;
    
    if [info exists tcl_precision] {
        set old_tcl_precision $tcl_precision ;
    }
    set tcl_precision $precision ;
    
    
    set code [catch { uplevel $script } result] ;
    
    
    if [info exists old_tcl_precision] {
        set tcl_precision $old_tcl_precision ;
    } else {
        unset tcl_precision ;
    }
    
    
    if { $code == 1 } {
        global errorInfo errorCode ;
        return -code $code -errorinfo $errorInfo -errorcode $errorCode $result
    } else {
        return -code $code $result
    }
    
}
 
proc average { list { k 0 } } {
    expr "double([join $list +])/([llength $list] - $k)" ;
}
 
proc loadMatrix { matrixHandle file args } {
    set list    [d_matrix size $matrixHandle] ;
    set rows    [lindex $list 0]
    set cols    [lindex $list 1]

    set row     0 ;
    if [lempty $args] {
        for_file list $file {
            set col 0 ;
            foreach value $list {
                d_matrix entryset $matrixHandle $row $col $value
                incr col ;
            }
            incr row ;
        }
    } else {
        set indices [eval concat $args] ;
        for_file list $file {
            set col 0 ;
            foreach index $indices {
                d_matrix entryset $matrixHandle $row $col \
                    [lindex $list $index] ;
                incr col ;
            }
            incr row ;
        }
    }
    if { $row != $rows } {
        error "expected $rows lines in $file but got $row" ;
    }
}
 
proc median { list } {
    
    set list   [lsort -real $list] ;
    set len    [llength $list] ;
    
    if { $len % 2 } {
        
        set n1     [expr { $len / 2 }] ;
        set median [lindex $list $n1] ;
        
    } else {
        
        set n1     [expr { $len / 2 }] ;
        set n2     [expr { $n1 - 1 }] ;
        set median [expr { 0.5*([lindex $list $n1]+[lindex $list $n2]) }] ;
        
    }
    
    return $median ;
    
}

proc square { x } {
    uplevel [list expr ($x)*($x)] ;
}

proc sum { list } {
    expr [join $list +] ;
}

proc delve_quiet {{ val 1} } {
    global pwrap_silent ;

   set pwrap_silent $val
}
proc pwrap { msg script } {
    global pwrap_silent ;
    if { [info exists pwrap_silent] && $pwrap_silent != 0 } {
        return [uplevel $script] ;
    }

    puts -nonewline stderr $msg ;
    set code [catch { uplevel $script } result] ;
    puts stderr "" ;
    if { $code == 1 } {
        global errorCode errorInfo ;
        return -code $code -errorinfo $errorInfo -errorcode $errorCode $result
    } else {
        return -code $code $result ;
    }
}

proc lorder { list orderList } {
    set result {} ;
    foreach index $orderList {
        lappend result [lindex $list $index] ;
    }
    return $result ;
}

proc lempty { list } {
    expr { [llength $list] == 0 } ;
}

proc lrmdups { list } {
    if { [llength $list] < 2 } {
        return $list ;
    }
    set list    [lsort $list] ;
    set last    [lindex $list 0] ;
    lappend result $last ;
    foreach element [lrange $list 1 end] {
        if [string compare $element $last] {
            set last $element ;
            lappend result $last ;
        }
    }
    return $result ;
}

proc lunion { lista listb } {
    lrmdups [concat $lista $listb] ;
}

proc lintersect { lista listb } {
    set lista   [lsort $lista] ;
    set listb   [lsort $listb] ;

    set result  {} ;
    while { [llength $lista] > 0 && [llength $listb] > 0 } {
        set compare     [string compare [lindex $lista 0] [lindex $listb 0]] ;

        if { $compare < 0 } {
            set lista [lrange $lista 1 end] ;
        } elseif { $compare > 0 } {
            set listb [lrange $listb 1 end] ;
        } else {
            lappend result [lindex $lista 0] ;
            set lista [lrange $lista 1 end] ;
            set listb [lrange $listb 1 end] ;
        }
    }
    return $result ;
}

proc lpop { varName } {
    upvar $varName list
    set first   [lindex $list 0] ;
    set list    [lrange $list 1 end] ;
    return $first ;
}

proc lpush { varName string } {
    upvar $varName list
    set list    [linsert $list 0 $string] ;
}
########################################################
#
# delveFileCmd
#
#   Wrapper for the tcl command: file option Fname
#   If the operation fails on the file Fname then we try
#   names with possible extensions that might indicate
#   a compressed file
#
########################################################
proc delveFileCmd {cmd fname} {

   if { [catch {set ret [file $cmd $fname]} msg ] == 0 && $ret != 0 } {
      return $ret
   }
   

   foreach ext [concat "" [d_compressExt]] {
      if { [catch {set ret [file $cmd $fname$ext]} msg ]  == 0 && $ret != 0 } {
	 return $ret;
      }
   }
   return $ret
}

##########################################################
#
# delveCat
#
#  Wrapper for the UNIX cat command. If fName does not exits
#  we try to cat a comprresed file with one of the possible
#  extension types
#
##########################################################
proc delveCat {fName} {
global env

   if {[file exists $fName]} {
      return [list cat $fName]
   } else {
      
      if [ info exists env(DELVE_UNCOMPRESS)] {
	 set uncompress $env(DELVE_UNCOMPRESS)
      }	else {
	 set uncompress "zcat"
      }

      foreach ext [concat "" [d_compressExt]] {
	 if [file exists $fName$ext] {
	    return [eval list $uncompress $fName$ext]
	 }
      }

      return [list echo Cannot cat $fName]
   }
}

proc delve_openForRead {fileName} {
global env

   set fileId {}
   if [file exists $fileName] {
      set fileId  [open $fileName r] ;
   } else {
      if [ info exists env(DELVE_UNCOMPRESS)] {
	 set uncompress $env(DELVE_UNCOMPRESS)
      } else {
	 set uncompress "zcat"
      }

      foreach ext [d_compressExt] {
	 if [ file exists $fileName$ext] {
	    set fileId [open "| $uncompress $fileName$ext" r]
	    break;
	 }
      }
   }

   return $fileId
}

# Return valid list of extensions for compressed files
proc  d_compressExt {} {
   return [list .gz .Z .z .zip]
}

#
# Open a file for reading and optionally apply
# filter (d_priv($filtName)) to the input.
# Note: The filter option has been discontinued so this function
# should always do a simple open - not using pipes
proc filtOpenForRead {file filtName} {
global d_priv

   if { [info exists d_priv($filtName)] } {
      regsub -all -- "/" [d_cpath .]  " " path
      return [open "| cat $file | $d_priv($filtName) [file tail $file] $path " ]
   } else {
      return [open $file r]
   }
}

proc filtOpenForWrite { file force filtName } {

   global d_priv

   if { [info exists d_priv($filtName)] } {
      regsub -all -- "/" [d_cpath .]  " " path
      return [open "|  $d_priv($filtName) [file tail $file] $path  > $file " w]
   } else {
      return [open $file w]
   }
}
