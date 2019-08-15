##############################################################
#
# $Id: taskGen.tcl,v 1.1.2.2.2.2 1997/06/11 18:29:20 revow Exp $
#
# DESCRIPTION
#   Utilities for handling tasks
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


proc saveEncoding { file { force 0 } } {
    
    set out  [wopen $file $force] ;
    
    foreach encoding [acdc_list] {
        
        set name        [lindex $encoding 1] ;
        
        
        set method      [acdc_method $name] ;
        
        
        set options {} ;
        foreach option [list passive center unit scale] {
            set value [acdc_options $name -$option] ;
            if { [string compare $value {}] != 0 } {
                append options " $option=[list $value]"
            }
        }
        
        
        puts $out "[list $name $method] $options" ;
        
    }
    
    close $out ;
    
}

proc loadEncoding { file attrs } {
    for_file line $file {
        
        set attr        [lpop line] ;
        if [catch {acdc_list $attr} list] {
            error "attribute \"$attr\" doesn't exist (file $file)" ;
        }
        set aliases     [lindex $list 0] ;
        if [lempty [lunion [concat [list $attr] $aliases] $attrs]] {
            error "attribute \"$attr\" exists, but isn't valid (file $file)" ;
        }
        
        
        
        set method      [lpop line] ;
        acdc_method $attr -- $method
        
        
        foreach key [list passive center unit] {
            acdc_options $attr -$key {} ;
        }
        
        foreach option $line {
            if ![regexp {^([^=]*)=([^=]+.*)$} $option all key value] {
                error "expected \"key=value\" but got \"$option\" (file $file)"
            }
            set key     [string tolower $key] ;
            acdc_options $attr -$key $value ;
        }
        
    }
}

proc genTaskInstances { cpath path fileList instList force } {
    
    global d_priv ;
    set tail          $d_priv(prototaskFile) ;
    set prototaskFile [d_getFile [file dirname $cpath]/$tail "data file"] ;
    
    
    set files [splitPrototask $prototaskFile \
                   [llength [d_info $cpath inputs]] \
                   [llength [d_info $cpath targets]] \
                   [d_info $cpath test-set-size] \
                   [d_info $cpath training-set-size] \
                   [d_info $cpath maximum-number-of-instances] \
                   [d_info $cpath test-set-selection]] ;
    set rawFiles(test)      [lpop files] ;
    set rawFiles(targets)   [lpop files] ;
    set rawFiles(training)  [lpop files] ;
    
    
    if { [string compare $instList "all"] == 0 } {
        set instList {} ;
        set limit    [llength $rawFiles(training)]
        for { set idx 0 } { $idx < $limit } { incr idx } {
            lappend instList $idx ;
        }
    }
    
    foreach instance $instList {
        set rawTrainingFile [lindex $rawFiles(training) $instance] ;
    
        
        set attrList    [concat [d_info $cpath inputs] [d_info $cpath targets]] ;
        set indexList   {} ;
        set index       0 ;
        foreach list [eval acdc_list $attrList] {
            set type [lindex $list 0] ;
            if { [string compare $type real] == 0
                 || [string compare $type integer] == 0 } {
                lappend indexList $index ;
            }
            incr index ;
        }
        if { ![lempty $indexList] } {
            set matrix \
                [d_matrix create [d_info $cpath training-set-size] [llength $indexList]] ;
            loadMatrix $matrix $rawTrainingFile $indexList ;
        } elseif { [info exists matrix] } {
            unset matrix ;
        }
        set out       [wopen $path/normalize.$instance $force] ;
        
        set col 0 ;
        foreach index $indexList {
            set attr [lindex $attrList $index] ;
            set list [d_mstats -col $col $matrix] ;
        
            puts $out [list normalization options mean $attr [lpop list]] ;
            puts $out [list normalization options stdDev $attr \
                [expr { sqrt( [lpop list] ) }]] ;
            puts $out [list normalization options median $attr [lpop list]] ;
            puts $out [list normalization options absDev $attr [lpop list]] ;
        
            incr col ;
        }
        
        if [info exists matrix] {
            d_matrix delete $matrix ;
        }
        close $out ;
        
    
        set regexp \
            {^normalization options ((mean)|(stdDev)|(median)|(absDev)) (.*) (.*)}
        for_file line $path/normalize.$instance {
            if [regsub $regexp $line {acdc_options \6 -\1 \7} new] {
                eval $new ; 
            } else {
                error "unknown line in $path/normalize.$instance: \"$line\"" ;
            }
        }
    
        if { [lsearch $fileList "training"] >= 0 } {
            pwrap "  encoding instance $instance training data..." {
                set fromFileId [filtOpenForRead $rawTrainingFile filterBeforeCoding] ;
                set toFileId   [filtOpenForWrite $path/train.$instance $force filterAfterCoding] ;
                acdc_encodeFile $attrList $fromFileId $toFileId ;
                close $fromFileId ;
                close $toFileId ;
            }
        }
        if { [lsearch $fileList "testing"] >= 0 
             && [llength [d_info $cpath inputs]] > 0 }  {
            pwrap "  encoding instance $instance test inputs..." {
                set fromFileId [filtOpenForRead [lindex $rawFiles(test) $instance] filterBeforeCoding ] ;
                set toFileId  [filtOpenForWrite $path/test.$instance $force filterAfterCoding] ;
                acdc_encodeFile [d_info $cpath inputs] $fromFileId $toFileId ;
                close $fromFileId ;
                close $toFileId ;
            }
        }
        if { [lsearch $fileList "targets"] >= 0
             && [llength [d_info $cpath targets]] > 0 }  {
            pwrap "  encoding instance $instance test targets..." {
	       set fromFileId [filtOpenForRead [lindex $rawFiles(targets) $instance] filterBeforeCoding ]
                set toFileId [filtOpenForWrite $path/targets.$instance $force filterAfterCoding] ;
                acdc_encodeFile [d_info $cpath targets] $fromFileId $toFileId ;
                close $fromFileId ;
                close $toFileId ;
            }
        }
    }
    
    
    
    set scheme      [d_info $cpath test-set-selection] ;
    if { [string compare $scheme "common"] == 0 } {
        set targetFile [lindex $rawFiles(targets) 0] ;
    } elseif { [string compare $scheme "hierarchical"] == 0 } {
        set targetFile "/tmp/alltargets.[pid]" ;
        set out        [open $targetFile w] ;
        set bufsize    1024 ;
        foreach infile $rawFiles(targets) {
            set in [open $infile r] ;
            while { ![eof $in] } {
                puts -nonewline $out [read $in $bufsize] ;
            }
            close $in ;
        }
        close $out ;
    } else {
        error "unknown test set selection scheme \"$scheme\"" ;
    }
    
    
    set attrList    [d_info $cpath targets] ;
    set indexList   {} ;
    set index       0 ;
    foreach list [eval acdc_list $attrList] {
        set type [lindex $list 0] ;
        if { [string compare $type real] == 0
             || [string compare $type integer] == 0 } {
            lappend indexList $index ;
        }
        incr index ;
    }
    
    if { ![lempty $indexList] } {
        set matrix \
            [d_matrix create [eval exec [delveCat $targetFile] | wc -l] [llength $indexList]] ;
        loadMatrix $matrix $targetFile $indexList ;
    } elseif { [info exists matrix] } {
        unset matrix ;
    }
    
    
    set col 0 ;
    set varList     {} ;
    set absDevList  {} ;
    foreach index $indexList {
        set attr [lindex $attrList $index] ;
        set list [d_mstats -col $col $matrix] ;
        lappend varList     [lindex $list 1] ;
        lappend absDevList  [lindex $list 3] ;
        incr col ;
    }
    
    set stat(S)     {} ;
    set stat(A)     {} ;
    set twopi               6.2831853
    for {set index 0} { $index < [llength $attrList] } { incr index } {
        if { [lsearch $indexList $index] >= 0 } {
            set variance    [lpop varList] ;
            set absDev      [lpop absDevList] ;
            lappend stat(S) $variance ;
            lappend stat(A) $absDev ;
            lappend stat(L) [expr { - 0.5*(log($twopi * $variance) + 1) }]
        } else {
            lappend stat(S) "undefined" ;
            lappend stat(A) "undefined" ;
            lappend stat(L) "undefined" ;
        }
    }
    
    
    if [info exists matrix] {
        d_matrix delete $matrix ;
    }
    
    
    set indexList   {} ;
    set index       0 ;
    set typeList    [list binary ordinal nominal integer] ;
    foreach list [eval acdc_list $attrList] {
        set type [lindex $list 0] ;
        if { [lsearch $typeList $type] >= 0 } {
            lappend indexList $index ;
        }
        incr index ;
    }
    set count(total)        0 ;
    for_file line $targetFile {
        incr count(total) ;
        foreach index $indexList {
            set value       [lindex $line $index] ;
            if [info exists count($index,$value)] {
                incr count($index,$value) ;
            } else {
                lappend count($index)       $value ;
                set count($index,$value) 1
            }
        }
    }
    
    
    set stat(Z)     {} ;
    for {set index 0} { $index < [llength $attrList] } { incr index } {
        if { [lsearch $indexList $index] >= 0 && $count(total) != 0 } {
            set max         0 ;
            foreach value $count($index) {
                if { $count($index,$value) > $max } {
                    set max $count($index,$value) ;
                }
            }
            lappend stat(Z) [expr { 1.0 - double($max)/$count(total) }] ;
        } else {
            lappend stat(Z) "undefined" ;
        }
    
    }
    
    
    set stat(Q)     {} ;
    set denom       [expr { double($count(total)*$count(total)) }] ;
    for {set index 0} { $index < [llength $attrList] } { incr index } {
        if { [lsearch $indexList $index] >= 0 && $count(total) != 0 } {
            set expr "1.0" ;
            foreach value $count($index) {
                set num     [expr { $count($index,$value)*$count($index,$value) }]
                append expr " - $num/$denom" ;
            }
            lappend stat(Q) [expr $expr] ;
        } else {
            lappend stat(Q) "undefined" ;
        }
    }
    
    
    for {set index 0} { $index < [llength $attrList] } { incr index } {
        if { [lsearch $indexList $index] >= 0 && $count(total) != 0 } {
            set expr        "0.0" ;
            foreach value $count($index) {
                append expr " + $count($index,$value)*log($count($index,$value)/double($count(total)))/double($count(total))"
            }
            set stat(L) [lreplace $stat(L) $index $index [expr $expr]]
        } else {
            lappend stat(L) "undefined" ;
        }
    }
    
    
    set out [wopen $path/Test-set-stats $force] ;
    foreach name [array names stat] {
        puts $out "$name $stat($name)"
    }
    close $out ;
    if { [string compare $scheme "hierarchical"] == 0 } {
        funlink $targetFile ;
    }
    
    
    
    saveEncoding $path/Coding-used $force ;
    
    
    foreach key [array names rawFiles] {
        funlink $rawFiles($key) ;
    }
    
}

proc genTask { cpath path { force 0 } } {
    uplevel [list genTaskInstances $cpath \
                 $path "training testing targets" all $force] ;
}

proc splitPrototask { inFile numInputs numTargets 
                      numTestCases numTrainingCases maxInstances scheme} {

    
    set numCases   [eval exec [delveCat $inFile] | wc -l] ;
    set caseList   $numTestCases ;
    incr numCases -$numTestCases ;
    while { $numCases >= $numTrainingCases } {
        lappend caseList  $numTrainingCases ;
        incr numCases    -$numTrainingCases ;
    }
    
    set numInstances        [expr { [llength $caseList] - 1 }] ;
    if { $numInstances > $maxInstances } {
        set numInstances $maxInstances ;
    }
    
    if { [string compare $scheme hierarchical] == 0 } {
        set testSetSize     [expr { $numTestCases/$numInstances }] ;
        for { set idx 0 } { $idx < $numInstances } { incr idx } {
            lappend newCaseList $testSetSize ;
        }
        if { $numTestCases % $numInstances != 0 } {
            lappend newCaseList \
                [expr { $numTestCases - $testSetSize*$numInstances }]
        }
        set caseList        [eval [list lreplace $caseList 0 0] $newCaseList] ;
    } elseif { [string compare $scheme common] != 0 } {
        error "unknown test set selection scheme \"$scheme\"" ;
    }
    
    
    pwrap "  segmenting cases..." {
        set files [splitFile $inFile /tmp/training[pid] $caseList 1] ;
    }
    
    if { [string compare $scheme common] == 0 } {
        set file    [lpop files] ;
        for { set idx 0 } { $idx < $numInstances } { incr idx } {
            lappend uncutTestFiles  $file ;
        }
    } else {
        for { set idx 0 } { $idx < $numInstances } { incr idx } {
            lappend uncutTestFiles  [lpop files] ;
        }
        #
        # unlink the dummy file holding unused test cases.
        #
        if { $numTestCases % $numInstances != 0 } {
            funlink [lpop files] ;
        }
    }
    for { set idx 0 } { $idx < $numInstances } { incr idx } {
        lappend trainingFiles       [lpop files] ;
    }
    funlink $files ;
    
    
    pwrap "  splitting test inputs and targets..." {
        foreach file $uncutTestFiles {
            set files [cutFile $file $file[pid] "$numInputs $numTargets" 1] ;
            lappend testFiles       [lpop files] ;
            lappend targetFiles     [lpop files] ;
        }
        funlink $uncutTestFiles ;
    }
    
    
    return [list $testFiles $targetFiles $trainingFiles] ;
    
}
