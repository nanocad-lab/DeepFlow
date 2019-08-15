##############################################################
#
# $Id: range.tcl,v 1.1.2.1 1996/11/12 16:55:43 revow Exp $
#
# DESCRIPTION
#   Utilties for manipulating ranges
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

proc range { option range args } {
    switch $option {
        
        contains {
            if { [llength $args] != 1 } {
                error "wrong \# args: should be \"range contains range value\""
            }
            set value [lindex $args 0 ] ;
            set type  [range type $range] ;
        
            if { [string compare $type enumerated] == 0 } {
                return [expr { [lsearch $range $value] >= 0 }] ;
            }
        
            if { [string compare $type integer] == 0 && ![isInt $value] } {
                return 0 ;
            } elseif { [string compare $type real] == 0 && ![isFloat $value] } {
                return 0 ;
            }
        
            set limits  [range limits $range] ;
            set lower   [lindex $limits 0] ;
            set upper   [lindex $limits 1] ;
        
            if { ( [isInf $lower] && [string compare [string index $lower 0] -] != 0 )
                 || ( ![range openlower $range] && $value == $lower)
                 || $value < $lower } {
                return 0 ;
            }
        
            if { ( [isInf $upper] && [string compare [string index $upper 0] -] == 0 )
                 || ( ![range openupper $range] && $value == $upper)
                 || $value > $upper } {
                return 0 ;
            }
        
            return 1 ;
        }
        
        
        finite  {
            if ![lempty $args] {
                error "wrong \# args: should be \"range finite range\""
            }
            set type [range type $range] ;
        
            if { [string compare $type enumerated] == 0 } {
                return 1 ;
            }
        
            set limits  [range limits $range] ;
            set lower   [lindex $limits 0] ;
            set upper   [lindex $limits 1] ;
        
            if { ( [isInf $lower] && [string compare [string index $lower 0] -] == 0 )
                 || ( [isInf $upper] && [string compare [string index $upper 0] -] != 0 ) } {
                return 0 ;
            }
        
            if { [isInf $lower] || [isInf $upper] || $lower >= $upper } {
                return 1 ;
            }
        
            expr { [string compare $type "integer"] == 0 } ;
        }
        
        
        limits {
            if ![lempty $args] {
                error "wrong \# args: should be \"range limits range\""
            }
            switch [range type $range] {
                integer {
                    set regexp "\[ \t\]*\[-+\]?(\[0-9\])+\[ \t\]*\.\.\[ \t\]*\[-+\]?(\[0-9\]\[ \t\]*)+"
                    set regexp "\[ \t\]*(\[^ \t\]*)\[ \t\]*\\.\\.\[ \t\]*(\[^ \t\]*)\[ \t\]*" ;
                }
                real {
                    set regexp "\[\\\[\\(\]\[ \t\]*(\[^ \t\]*)\[ \t\]*,\[ \t\]*(\[^ \t\]*)\[ \t\]*\[]\\)\]" ;
                }
                enumerated {
                    error "range \"$range\" doesn't have limits"
                }
             }
             regexp $regexp $range all lower upper ;
             return [list $lower $upper] ;
        }
        
        
        list    {
            if ![lempty $args] {
                error "wrong \# args: should be \"range list range\""
            }
            if ![range finite $range] {
                error "can't list an infinite range: \"$range\"" ;
            }
            set type    [range type $range] ;
        
            if { [string compare $type enumerated] == 0 } {
                set result $range ;
        
            } else {
                set limits      [range limits $range] ;
                set lower       [lindex $limits 0] ;
                set upper       [lindex $limits 1] ;
        
                if { [isInf $lower] || [isInf $upper]
                     || ( $lower > $upper )
                     || ( $lower == $upper
                         && ( ![range openlower $range] || ![range openupper $range] ) ) } {
                    set result  {} ;
                } elseif { $lower == $upper } {
                    set result  $lower ;
                } else {
                    incr lower [expr { ![range openlower $range] }] ;
                    incr upper [range openupper $range] ;
                    set result {} ;
                    for { set idx $lower } { $idx < $upper } { incr idx } {
                        lappend result $idx ;
                    }
                }
            }
            return $result ;
        }
        
        
        openlower {
            if ![lempty $args] {
                error "wrong \# args: should be \"range openlower range\""
            }
        
            switch [range type $range] {
                "enumerated" {
                    error "enumerated ranges are neither opened or closed" ;
                }
                "integer" {
                    return 1 ;
                }
                "real" {
                    set regexp {([\[\(]).*} ;
                    regexp "^\[ \t]*$regexp" $range all brace ;
                    return [expr { [string compare $brace {[}] == 0 } ] ;
                }
            }
        }
        
        
        openupper {
            if ![lempty $args] {
                error "wrong \# args: should be \"range openupper range\""
            }
        
            switch [range type $range] {
                "enumerated" {
                    error "enumerated ranges are neither opened or closed" ;
                }
                "integer" {
                    return 1 ;
                }
                "real" {
                    set regexp {.*([]\)])} ;
                    regexp "$regexp\[ \t]*$" $range all brace ;
                    return [expr { [string compare $brace {]}] == 0 }] ;
                }
            }
        }
        
        
        type {
            if ![lempty $args] {
                error "wrong \# args: should be \"range type range\""
            }
        
            set regexp "^\[ \t\]*\[\\\[\\(\](.*),(.*)\[]\\)\]\[ \t\]*\$" ;
            if [regexp $regexp $range all lower upper] {
                if { ![isFloat $lower] } {
                    error "expected floating-point limit in range \"$range\" but got \"$lower\""
                }
                if { ![isFloat $upper] } {
                    error "expected floating-point limit in range \"$range\" but got \"$upper\""
                }
                return real ;
            }
        
            set regexp "(.*)\\.\\.(.*)"
            if [regexp $regexp $range all lower upper] {
                if { ![isInt $lower] } {
                    error "expected integer limit in range \"$range\" but got \"$lower\""
                }
                if { ![isInt $upper] } {
                    error "expected integer limit in range \"$range\" but got \"$upper\""
                }
                return integer ;
            }
        
            return "enumerated" ;
        }
        
        
        default {
            error "bad option \"$option\": should be contains, finite, limits, list, openlower, openupper, or type"
        }
        
    }
}

proc isInt { value } {
    set regexp {[-+]?([0-9])+}
    expr { [regexp "^\[ \t]*$regexp\[ \t]*$" $value] || [isInf $value] }
}

proc isFloat { value } {
    set regexp {[-+]?([0-9])*(\.([0-9])*)?([eE][-+]?[0-9]+)?}
    expr { ([regexp "^\[ \t]*$regexp\[ \t]*$" $value \
                 all integer allFraction fraction exponent]
            && !([string compare $integer {}] == 0 
                 && [string compare $fraction {}] == 0))
           || [isInf $value] }
}

proc isInf { value } {
    set regexp {[-+]?[Ii][Nn][Ff]} ;
    regexp "^\[ \t]*$regexp\[ \t]*$" $value ;
}
