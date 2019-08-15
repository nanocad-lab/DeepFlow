# ----------------------------------------------------------------------
# $Id: matrix.tcl,v 1.5.2.5 1996/11/12 16:55:41 revow Exp $
#
# File containing procedures dealing with matrices. 
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
# pmatrix - prints a matrix out to an open file descriptor
#
# The procedure prints out the matrix one row to a line to an open
# file descriptor (by default "stdout").
# ----------------------------------------------------------------------

proc pmatrix { matrix { fileId stdout } } {
    set list	[d_matrix size $matrix] ;
    set rows	[lindex $list 0]
    set cols	[lindex $list 1]
    for { set row 0 } { $row < $rows } { incr row } {
	set line	{} ;
	for { set col 0 } { $col < $cols } { incr col } {
	    lappend line [d_matrix entryset $matrix $row $col] ;
	}
	puts $fileId $line
    }
}
