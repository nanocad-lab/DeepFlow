# ----------------------------------------------------------------------
# $Id: randList.tcl,v 1.2.2.4 1996/11/12 16:55:42 revow Exp $
#
# This procedure prints the numbers one through <n> to an open file id
# (default stdout), in random order. 
#
# It first builds an ordered array of the numbers (keyed by the
# numbers themselves). It then iterates over the array swapping each
# element with a random element at that position or later.  This
# has the effect of randomly shuffling the numbers, with each permutation
# having equal probability.
#
# The suffled array is then printed out.
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

proc d_randList { n { fileId stdout } } {
    for { set idx 0 } { $idx < $n } { incr idx } {
	set num($idx)	[expr {$idx + 1}] ;
    }

    for { set oldIdx 0 } { $oldIdx < $n } { incr oldIdx } {
	set rand_offset		[d_random [expr {$n - $oldIdx}]]
	set newIdx		[expr {$oldIdx + $rand_offset}]
	set tmp			$num($newIdx) ;
	set num($newIdx)	$num($oldIdx) ;
	set num($oldIdx)	$tmp ;
    }

    for { set idx 0 } { $idx < $n } { incr idx } {
	puts $fileId $num($idx) ;
    }
}
