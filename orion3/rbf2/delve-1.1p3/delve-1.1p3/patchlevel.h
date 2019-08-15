/*
 *  patchlevel.h - This file does nothing except define a "patch
 *  level" for Delve.  The patch level has the form "X.YpZ" where X.Y
 *  is the base release, and Z is a serial number that is used to
 *  sequence patches for a given release.  Thus 7.4p1 is the first
 *  patch to release 7.4, 7.4p2 is the patch that follows 7.4p1, and
 *  so on.  The "pZ" is omitted in an original new release, and it is
 *  replaced with "bZ" for beta releases and "aZ" for alpha releases.
 *  The patch level ensures that patches are applied in the correct
 *  order and only to appropriate sources.
 *
 * Copyright (c) 1996 by The University of Toronto.
 * 
 * See the file "copyright" for information on usage and redistribution
 * of this file, and for a DISCLAIMER OF ALL WARRANTIES.
 * 
 * Author: Delve (delve@cs.toronto.edu)
 * 
 * $Id: patchlevel.h,v 1.2.2.4.2.3 1997/11/27 16:42:57 revow Exp $ */

#define DELVE_PATCH_LEVEL "1.1p3"
