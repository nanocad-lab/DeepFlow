##############################################################
#
# $Id: acdc.tcl,v 1.1.2.1.2.1 1997/04/22 17:39:31 revow Exp $
#
# DESCRIPTION
#   Utilities for encoding and decoding attributes.
#
# PROJECT:
#
# AUTHOR HISTORY:
#    Author Delve (delve@cs.toronto.edu)
#
#    Drew van Camp
#
# Copyright (c) 1996 The University of Toronto.
#
# See the file "copyright" for information on usage and redistribution
# of this file, and for a DISCLAIMER OF ALL WARRANTIES.
#
##############################################################

proc acdc_new { type id args } {
    
    global acdc_priv ;
    
    
    if [info exists acdc_priv(acdc_id,$id)] {
        acdc_delete $id ;
    }
    
    set acdc_priv($id,type)         $type ;
    set acdc_priv(acdc_id,$id)      $id ;
    foreach alias $args {
        set acdc_priv(acdc_id,$alias)       $id ;
    }
    
    
    set acdc_priv($id,defaultMethod) [acdcDefault $type] ;
    set acdc_priv($id,method)        $acdc_priv($id,defaultMethod) ;
    
    
    foreach option [list absDev center mean median passive range variance unit] {
        set acdc_priv($id,$option)  {} ;
    }
    set acdc_priv($id,scale)        sqrt ;
    
}

proc acdcDefault { type } {
    switch -- $type {
        angular         { return "rectan" }
        binary          { return "-1/+1" }
        nominal         { return "1-of-n" }
        ordinal         { return "therm" }
        integer         { return "nm-abs" }
        real            { return "nm-abs" }
        default         { error "can't encode attribute type \"$type\"" }
    }
}
proc acdc_clear {} {
   global acdc_priv;

   foreach key [array names acdc_priv] {
      if {[regexp "^acdc_id," $key] && [info exists acdc_priv($key)]} {
	 acdc_delete $acdc_priv($key)
      }
   }
}
proc acdc_delete { attr } {
    
    global acdc_priv ;
    
    
    if ![info exists acdc_priv(acdc_id,$attr)] {
        return ;
    }
    
    
    set id $acdc_priv(acdc_id,$attr) ;
    if [info exists acdc_prive($id,handle)] {
        d_attr delete $acdc_prive($id,handle)
    }
    foreach key [array names acdc_priv] {
        if [regexp "^$id," $key] {
            unset acdc_priv($key) ;
        }
    }
    foreach key [array names acdc_priv] {
        if { [regexp "^acdc_id," $key] && [string match $acdc_priv($key) $id] } {
            unset acdc_priv($key) ;
        }
    }
    
}

proc acdc_method { attr args } {
    
    global acdc_priv ;
    
    
    if ![info exists acdc_priv(acdc_id,$attr)] {
        error "attribute \"$attr\" hasn't been registered for encoding" ;
    } 
    set id $acdc_priv(acdc_id,$attr) ;
    
    
    while { ![lempty $args] } {
        set switch [lpop args] ;
        switch -glob -- $switch {
            --              { break ; }
            -default        { set default 1 ; }
            -*              { error "unknown switch \"$switch\"" }
            default         { lpush args $switch ; break ; }
        }
    }
    
    
    if { [llength $args] > 1 } {
        error "wrong \# args: should be \"acdc_method attr ?switches? ?method?\"" ;
    }
    
    
    if [info exists default] {
        set key defaultMethod ;
    } else {
        set key method ;
    }
    
    
    if { [llength $args] == 1 } {
        set method  [lpop args] ;
        set type    $acdc_priv($id,type) ;
        if { [lsearch [acdcAllowed $type] $method] < 0 } {
            error "can't use \"$method\" encoding for \"$type\" attribute $attr" ;
        }
        set acdc_priv($id,$key)     $method ;
        if [info exists acdc_priv($id,handle)] {
            acdc_delete $acdc_priv($id,handle) ;
            unset acdc_priv($id,handle) ;
        }
    }
    
    
    return $acdc_priv($id,$key) ;
    
}

proc acdcGetId { attr } {
    if ![info exists acdc_priv(acdc_id,$attr)] {
        error "attribute \"$attr\" hasn't been registered for encoding" ;
    } 
    return $acdc_priv(acdc_id,$attr) ;
}

proc acdcAllowed { type } {
    switch -- $type {
        angular     { return "ignore copy rectan" }
        binary      { return "ignore copy -1/+1 0/1" }
        nominal     { return "ignore copy 1-of-n 0-up 1-up therm" }
        ordinal     { return "ignore copy 1-of-n 0-up 1-up therm" }
        integer     { return "ignore copy nm-abs nm-sqr" }
        real        { return "ignore copy nm-abs nm-sqr" }
        default     { error "can't encode attribute type \"$type\"" }
    }
}

proc acdc_options { attr args } {
    
    global acdc_priv ;
    
    
    if ![info exists acdc_priv(acdc_id,$attr)] {
        error "attribute \"$attr\" hasn't been registered for encoding" ;
    } 
    set id $acdc_priv(acdc_id,$attr) ;
    
    
    if { [llength $args] == 0 } {
        error "wrong \# args: should be \"acdc_options attr option ?value option value ...?\"" ;
    }
    

    if { [llength $args] == 1 } {
        
        set option [lpop args] ;
        switch -glob -- $option {
            -absDev     { return $acdc_priv($id,absDev) }
            -center     -
            -centre     { return $acdc_priv($id,center) }
            -mean       { return $acdc_priv($id,mean) }
            -median     { return $acdc_priv($id,median) }
            -passive    { return $acdc_priv($id,passive) }
            -range      { return $acdc_priv($id,range) }
            -scale      { return $acdc_priv($id,scale) }
            -unit       { return $acdc_priv($id,unit) }
            -variance   { return $acdc_priv($id,variance) }
            -stdDev     {
                if { [string compare $acdc_priv($id,variance) {}] == 0 } {
                    return {} ;
                } else {
                    return [expr { sqrt($acdc_priv($id,variance)) }] ;
                }
            }
            default     { error "unknown option \"$option\"" }
        }
        
    } else {
        
        while { ![lempty $args] } {
            set option [lpop args] ;
            if [lempty $args] {
                error "missing value for option \"$option\"" ;
            }
            set value  [lpop args] ;
            switch -glob -- $option {
                -absDev   { set acdc_priv($id,absDev)   $value }
                -center   -
                -centre   { set acdc_priv($id,center)   $value }
                -mean     { set acdc_priv($id,mean)     $value }
                -median   { set acdc_priv($id,median)   $value }
                -passive  { set acdc_priv($id,passive)  $value }
                -range    { set acdc_priv($id,range)    $value }
                -unit     { set acdc_priv($id,unit)     $value }
                -scale    {
                    if { [lsearch "none linear sqrt" $value] >= 0 } {
                        set acdc_priv($id,scale)        $value ;
                    } else {
                        error "bad scale option for \"$id\", \"$value\": should be none, linear, or sqrt" ;
                    }
                }
                -variance { set acdc_priv($id,variance) $value }
                -stdDev   {
                    if { [string compare $value {}] == 0 } {
                        set acdc_priv($id,variance)     {} ;
                    } else {
                        set acdc_priv($id,variance)     [expr { $value * $value }] ;
                    }
                }
                default   { error "unknown option \"$option\"" }
            }
        }
        
        if [info exists acdc_priv($id,handle)] {
            acdc_delete $acdc_priv($id,handle) ;
            unset acdc_priv($id,handle) ;
        }
        
            return ;
        
    }
}

proc acdc_encode { attr value } {
    
    global acdc_priv ;
    
    
    if ![info exists acdc_priv(acdc_id,$attr)] {
        error "attribute \"$attr\" hasn't been registered for encoding" ;
    } 
    set id $acdc_priv(acdc_id,$attr) ;
    
    
    if ![info exists acdc_priv($id,handle)] {
        acdcCompile $attr ;
    }
    
    
    d_attr encode $acdc_priv($id,handle) $value ;
    
}

proc acdcCompile { attr } {
    
    global acdc_priv ;
    
    
    if ![info exists acdc_priv(acdc_id,$attr)] {
        error "attribute \"$attr\" hasn't been registered for encoding" ;
    } 
    set id $acdc_priv(acdc_id,$attr) ;
    
    set type    $acdc_priv($id,type)
    set method  $acdc_priv($id,method)
    switch -exact -- $method {
        
        ignore  {
            set handle [d_attr create $type ignore] ;
        }
        
        
        copy  {
            set handle [d_attr create $type copy] ;
        }
        
        
        nm-abs  {
            set mu      $acdc_priv($id,median) ;
            set sigma   $acdc_priv($id,absDev)
            set center  $acdc_priv($id,center) ;
        
            if { [string compare $mu {}] == 0 } {
                error "no median given for \"$method\" encoded attribute \"$attr\"" ;
            }
            if { [string compare $sigma {}] == 0 } {
                error "no absDev given for \"$method\" encoded attribute \"$attr\"" ;
            }
            if { [string compare  $center {}] != 0 } {
                set mu [expr { $mu - $center }] ;
            }
            set handle [d_attr create $type normalized $mu $sigma]
        }
        
        
        nm-sqr  {
            set mu      $acdc_priv($id,mean) ;
            set sigma   [expr sqrt($acdc_priv($id,variance))]
            set center  $acdc_priv($id,center) ;
        
            if { [string compare $mu {}] == 0 } {
                error "no mean given for \"$method\" encoded attribute \"$attr\"" ;
            }
            if { [string compare  $center {}] != 0 } {
                set mu [expr { $mu - $center }] ;
            }
            if { [string compare $sigma {}] == 0 } {
                error "no variance given for \"$method\" encoded attribute \"$attr\"" ;
            }
            set handle [d_attr create $type normalized $mu $sigma] ;
        }
        
        
        rectan {
            if { [string compare $acdc_priv($id,unit) {}] == 0 } {
                error "no unit given for \"$method\" encoded attribute \"$attr\"" ;
            }
            set handle [d_attr create $type rectan $acdc_priv($id,unit)]
        }
        
        
        0-up {
            set list    {} ;
            foreach range $acdc_priv($id,range) {
                if { [string compare [range type $range] "enumerated"] } {
                    eval lappend list [range list $range] ;
                } else {
                    lappend list $range ;
                }
            }
            set handle [d_attr create $type zero-up $list] ;
        }
        
        
        1-up {
            set list    {} ;
            foreach range $acdc_priv($id,range) {
                if { [string compare [range type $range] "enumerated"] } {
                    eval lappend list [range list $range] ;
                } else {
                    lappend list $range ;
                }
            }
            set handle [d_attr create $type one-up $list] ;
        }
        
        
        0/1 {
            set passive $acdc_priv($id,passive) ;
            if { [string compare $passive {}] == 0 } {
                error "no passive value given for \"$method\" encoded attribute \"$attr\""
            }
            set list    {} ;
            foreach range $acdc_priv($id,range) {
                if { [string compare [range type $range] "enumerated"] } {
                    eval lappend list [range list $range] ;
                } else {
                    lappend list $range ;
                }
            }
            set handle [d_attr create $type binary-passive $list $passive] ;
        }
        
        
        -1/+1 {
            set list    {} ;
            foreach range $acdc_priv($id,range) {
                if { [string compare [range type $range] "enumerated"] } {
                    eval lappend list [range list $range] ;
                } else {
                    lappend list $range ;
                }
            }
            set handle [d_attr create $type binary-symmetric $list] ;
        }
        
        
        1-of-n {
            set passive $acdc_priv($id,passive) ;
            set list    {} ;
            foreach range $acdc_priv($id,range) {
                if { [string compare [range type $range] "enumerated"] } {
                    eval lappend list [range list $range] ;
                } else {
                    lappend list $range ;
                }
            }
            if { [string compare $passive {}] == 0 } {
                set handle [d_attr create $type one-of-n $list] ;
            } else {
                set handle [d_attr create $type one-of-n $list $passive] ;
            }
        }
        
        
        therm {
            set list    {} ;
            foreach range $acdc_priv($id,range) {
                if { [string compare [range type $range] "enumerated"] } {
                    eval lappend list [range list $range] ;
                } else {
                    lappend list $range ;
                }
            }
            set n       [llength $list] ;
            if { $n < 2 } {
                error "can't use thermometer coding for attribute $id: it has too few values in its range" ;
            }
            switch $acdc_priv($id,scale) {
                "linear"        { set scale [expr { 1.0/($n - 1) }] }
                "sqrt"          { set scale [expr { 1.0/sqrt($n - 1) }] }
                "none"          { set scale 1.0 }
                default         { error "unknown scaling method for thermometer coding of attribute \"$id\"" }
            }
            set handle [d_attr create $type thermometer $list $scale] ;
        }
        
    }
    set acdc_priv($id,handle) $handle ;
}

proc acdc_decode { attr value } {
    
    global acdc_priv ;
    
    
    if ![info exists acdc_priv(acdc_id,$attr)] {
        error "attribute \"$attr\" hasn't been registered for encoding" ;
    } 
    set id $acdc_priv(acdc_id,$attr) ;
    
    
    if ![info exists acdc_priv($id,handle)] {
        acdcCompile $attr ;
    }
    
    
    d_attr decode $acdc_priv($id,handle) $value ;
    
}

proc acdc_encodeFile { attrList fromFileId toFileId } {
    
    global acdc_priv ;
    
    
    set numAttrs 0 ;
    foreach attr $attrList {
        
        if ![info exists acdc_priv(acdc_id,$attr)] {
            error "attribute \"$attr\" hasn't been registered for encoding" ;
        } 
        set id $acdc_priv(acdc_id,$attr) ;
        
        if ![info exists acdc_priv($id,handle)] {
            acdcCompile $attr ;
        }
        set handle($numAttrs) $acdc_priv($id,handle) ;
        incr numAttrs ;
    }
    
    
    set lineNum 0 ;
    while { [gets $fromFileId line] >= 0 } {
        if { [llength $line] != $numAttrs } {
            error "wrong \# of values at line $lineNum" ;
        }
        set result  {} ;
        set idx     0 ;
        foreach value $line {
            append result " [d_attr encode $handle($idx) $value]" ;
            incr idx ;
        }
        puts $toFileId $result ;
        incr lineNum ;
    }
    
}

proc acdc_decodeFile { attrList fromFileId toFileId } {
    
    global acdc_priv ;
    
    
    set numAttrs 0 ;
    foreach attr $attrList {
        
        if ![info exists acdc_priv(acdc_id,$attr)] {
            error "attribute \"$attr\" hasn't been registered for encoding" ;
        } 
        set id $acdc_priv(acdc_id,$attr) ;
        
        
        if ![info exists acdc_priv($id,handle)] {
            acdcCompile $attr ;
        }
        
        set handle($numAttrs) $acdc_priv($id,handle) ;
        incr numAttrs ;
    }
    
    
    set start 0 ;
    for { set idx 0 } { $idx < $numAttrs } { incr idx } {
        set num             [llength [acdc_names [lindex $attrList $idx]]] ;
        set first($idx)     $start ;
        set last($idx)      [expr { $first($idx) + $num - 1 }] ;
    
        incr start $num ;
    }
    set numEncodings $start ;
    
    
    set lineNum 0 ;
    while { [gets $fromFileId line] >= 0 } {
        set line [eval list $line] ;
        if { [llength $line] != $numEncodings } {
            error "wrong \# of values at line $lineNum" ;
        }
        set result  {} ;
        for { set idx 0 } { $idx < $numAttrs } { incr idx } {
            set value       [lrange $line $first($idx) $last($idx)] ;
            if { [string compare $value "?"] == 0 } {
                lappend result $value ;
            } else {
                lappend result [d_attr decode $handle($idx) $value] ;
            }
        }
        puts $toFileId $result ;
        incr lineNum ;
    }
    
}

proc acdc_list { args } {
    
    global acdc_priv ;
    
    
    if { [lempty $args] && [info exists acdc_priv] } {
        foreach key [array names acdc_priv] {
            if [regexp "^acdc_id," $key] {
                lappend args $acdc_priv($key) ;
            }
        }
        set args [lrmdups $args] ;
    }
    
    
    set result {} ;
    
    foreach attr $args {
        
        if ![info exists acdc_priv(acdc_id,$attr)] {
            error "attribute \"$attr\" hasn't been registered for encoding" ;
        } 
        set id $acdc_priv(acdc_id,$attr) ;
        
        
        set aliases {}
        foreach key [array names acdc_priv] {
            if { [regexp "^acdc_id," $key] 
                 && [string compare $acdc_priv($key) $id] == 0
                 && [regexp {acdc_id,(.*)} $key all alias]
                 && [string compare $alias $id] != 0 } {
                lappend aliases $alias ;
            }
        }
        
        
        lappend result [list $acdc_priv($id,type) $id $aliases] ;
        
    }
    
    return $result ;
    
}

proc acdc_names { attr } {
    
    global acdc_priv ;
    
    
    if ![info exists acdc_priv(acdc_id,$attr)] {
        error "attribute \"$attr\" hasn't been registered for encoding" ;
    } 
    set id $acdc_priv(acdc_id,$attr) ;
    
    
    switch -exact -- $acdc_priv($id,method) {
        "copy"      -
        "nm-abs"    -
        "nm-sqr"    -
        "0-up"      -
        "1-up"      -
        "0/1"       -
        "-1/+1"     {
            set names       [list $id] ;
        }
        
        "ignore" {
            set names   "" ;
        }
        
        
        "rectan" {
            set names   [list "sin(k*$id)" "cos(k*$id)"] ;
        }
        
        
        1-of-n {
            set names   "" ;
            foreach range $acdc_priv($id,range) {
                if { [string compare [range type $range] "enumerated"] } {
                    set list [range list $range] ;
                } else {
                    set list [list $range] ;
                }
                foreach value $list {
                    if { [string compare $value $acdc_priv($id,passive)] != 0 } {
                        lappend names "$id:$value" ;
                    }
                }
            }
        }
        
        
        therm    {
            set names   "" ;
            foreach range $acdc_priv($id,range) {
                if { [string compare [range type $range] "enumerated"] } {
                    set list [range list $range] ;
                } else {
                    set list [list $range] ;
                }
                foreach value $list {
                    lappend names "$id:$value" ;
                }
            }
            set names   [lrange $names 1 end] ;
        }
        
    }
    
    
    return $names ;
    
}

proc acdc_save args {
    
    set info [eval acdc_list $args] ;
    
    
    set result {} ;
    
    foreach list $info {
        
        set type        [lindex $list 0] ;
        set id          [lindex $list 1] ;
        set aliases     [lindex $list 2] ;
        append result [list acdc_new $type $id $aliases] ;
        append result "\n"
        
        
        set method [acdc_method $id -default] ;
        append result [list acdc_method $id -default -- $method] ;
        append result "\n"
        
        set method [acdc_method $id] ;
        append result [list acdc_method $id -- $method] ;
        append result "\n"
        
        
        foreach option [list absDev center mean median passive range variance unit] {
            set value [acdc_options $id -$option] ;
            append result [list acdc_options $id -$option $value] ;
            append result "\n"
        }
        
    }
    
    return [string trim $result] ;
    
}
