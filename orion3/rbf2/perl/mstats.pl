#!/usr/local/bin/perl -w

#
# Simulates mstats -t if mstats had a -t flag.
#
# This script is useful for getting delve results into Matlab.
# The trouble with DELVE's own script, mstats, is that it
# doesn't have a -t flag (terse) like some of the other DELVE
# commands. This makes it awkward to communicate the numbers
# printed by mstats to Matlab. This perl script calls the
# real mstats but strips away all the verbosity to leave
# just the key numbers relating to method performance.
#
# To use mstats.pl from within Matlab first you have to figure
# out the correct arguments (the same ones you would have
# passed to the real mstats). Suppose this string is in args,
# then from Matlab use:
#
#   [status, stdout] = unix(['mstats.pl ' args]);
#
# This assumes Matlab can find mstats.pl in your PATH.
# Otherwise use the full pathname. If the script has
# returned sucessfully status will be 0 and the string
# stdout can be parsed with sscanf for the required
# numbers.
#
# Here are two examples comparing mstats and mstats.pl
# run from the unix command line.
#
# Example 1: Get the scaled mean and standard deviation
# of the squared error loss for lin-1 on demo/age/std.128.
#
#   $ mstats -l S /lin-1/demo/age/std.128
#   /lin-1/demo/age/std.128
#   Loss: S (Squared error)
#                                                         Raw value   Standardized
#   
#                            Estimated expected loss:   400.729718    0.81974466
#                        Standard error for estimate:    28.611146  0.0585278134
#   
#        SD from training sets & stochastic training:   40.8979934  0.0836621548
#   SD from test cases & stoch. pred. & interactions:   790.028701    1.61610627
#   
#       Based on 8 disjoint training sets, each containing 128 cases and
#                8 disjoint test sets, each containing 128 cases.
#
#   $ mstats.pl -l S /lin-1/demo/age/std.128
#   0.81974466 0.0585278134
#
# Example 2: Get the t-test significance probability between
# lin-1 and knn-cv-1 on demo/age/std.128.
#
#   $ mstats -c knn-cv-1 -l S /lin-1/demo/age/std.128
#   /lin-1/demo/age/std.128
#   Loss: S (Squared error)
#                                                         Raw value   Standardized
#   
#                  Estimated expected loss for lin-1:   400.729718    0.81974466
#              Estimated expected loss for /knn-cv-1:   368.002767   0.752797434
#                      Estimated expected difference:   32.7269508  0.0669472264
#             Standard error for difference estimate:   14.0749681  0.0287921744
#   
#        SD from training sets & stochastic training:   27.6978006  0.0566594468
#   SD from test cases & stoch. pred. & interactions:   323.514631   0.661791176
#   
#       Significance of difference (t-test), p = 0.0529880206
#   
#       Based on 8 disjoint training sets, each containing 128 cases and
#                8 disjoint test sets, each containing 128 cases.
#
#   $ mstats.pl -c knn-cv-1 -l S /lin-1/demo/age/std.128
#   0.0529880206
#

# Get argument list (same as would be sent to mstats).
$args = join ' ', @ARGV;

# Check syntax.
$type = 'error';
if ($args =~ /^-l S \/[-_a-z0-9\.]+\/[-_a-z0-9\.]+\/[a-z]+\/std\.\d+$/) {
  $type = 'method';
} elsif ($args =~ /^-c [-_a-z0-9\.]+ -l S \/[-_a-z0-9\.]+\/[-_a-z0-9\.]+\/[a-z]+\/std\.\d+$/) {
  $type = 'compare';
}
if ($type eq 'error') {
  print "error: bad input syntax\n";
  exit 1;
}

# Run delve program to get results.
$output = `mstats $args`;

# Look in each line.
@output = split /\n/, $output;
$return = '';
foreach $line (@output) {

  if ($type eq 'method') {

    # Expected loss.
    if ($line =~ /\s+Estimated expected loss:\s*(\d+\.\d+(e[-+]\d\d)?)\s*(\d+\.\d+(e[-+]\d+)?)$/) {
      $std = $3;
    }

    # Standard errors.
    if ($line =~ /\s+Standard error for estimate:\s*(\d+\.\d+(e[-+]\d\d)?)\s*(\d+\.\d+(e[-+]\d+)?)$/) {
      $serr = $3;
    }
    
    # What to return.
    $return = "$std $serr" if (defined $std) && (defined $serr);
    
    # When to quit.
    last if $return;

  } else {

    # Significance of difference.
    if ($line =~ /^\s+Significance of difference \([Ft]-test\), p [=<] (\d+\.\d+(e[-+]\d+)?)$/) {
      $sig = $1;
    }

    # What to return.
    $return = "$sig" if (defined $sig);
    
    # When to quit.
    last if $return;
  
  }

}

# Check we got something.
if (!$return) {
  print "error: can't find results\n";
  exit 1;
}

# Return result by printing.
print "$return\n";

# Normal exit.
exit 0;
