/**********************************************************************************************************
Copyright  2012   The Regents of the University of California
All Rights Reserved
 
Permission to copy, modify and distribute any part of this ORION3.0 software distribution for educational, 
research and non-profit purposes, without fee, and without a written agreement is hereby granted, provided 
that the above copyright notice, this paragraph and the following three paragraphs appear in all copies.
 
Those desiring to incorporate this ORION 3.0 software distribution into commercial products or use for 
commercial purposes should contact the Technology Transfer Office.

Technology Transfer Office
University of California, San Diego 
9500 Gilman Drive 
Mail Code 0910 
La Jolla, CA 92093-0910

Ph: (858) 534-5815
FAX: (858) 534-7345
E-MAIL:invent@ucsd.edu.

 
IN NO EVENT SHALL THE UNIVERSITY OF CALIFORNIA BE LIABLE TO ANY PARTY FOR DIRECT, INDIRECT, SPECIAL, 
INCIDENTAL, OR CONSEQUENTIAL DAMAGES, INCLUDING LOST PROFITS, ARISING OUT OF THE USE OF THIS ORION 3.0 
SOFTWARE DISTRIBUTION, EVEN IF THE UNIVERSITY OF CALIFORNIA HAS BEEN ADVISED OF THE POSSIBILITY 
OF SUCH DAMAGE.
 
THE ORION 3.0 SOFTWARE DISTRIBUTION PROVIDED HEREIN IS ON AN "AS IS" BASIS, AND THE UNIVERSITY OF 
CALIFORNIA HAS NO OBLIGATION TO PROVIDE MAINTENANCE, SUPPORT, UPDATES, ENHANCEMENTS, OR MODIFICATIONS.  
THE UNIVERSITY OF CALIFORNIA MAKES NO REPRESENTATIONS AND EXTENDS NO WARRANTIES OF ANY KIND, EITHER 
IMPLIED OR EXPRESS, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY OR FITNESS 
FOR A PARTICULAR PURPOSE, OR THAT THE USE OF THE ORION 3.0 SOFTWARE DISTRIBUTION WILL NOT INFRINGE ANY 
PATENT, TRADEMARK OR OTHER RIGHTS.
**********************************************************************************************************/
#ifndef _OUTBUF_H
#define _OUTBUF_H

#include "SIM_parameter.h"
/*!
\brief A stuct defining an OUTBUF block (ORION 3.0)
*/
typedef struct{
/**
* in ports
*/
u_int p_in;

/**
* out ports
*/
u_int p_out;

/**
* virtual channels
*/
u_int v_channel; 

/**
* flit width
*/
u_int flit_width;
/**
* in buffers
*/
int buf_in;  
/**
* out buffers
*/
int buf_out; 
/**
* model
*/
int model;
/**
* frequency
*/
float clk;
/**
* toggle rate
*/
float tr;

/*private variables*/
/*!
* # of insts
*/
int insts;

/*!
Function pointer to 
OUTBUF_get_instances
*/
int (*get_instances)(void *);

/*!
Function pointer to 
OUTBUF_get_area
*/
double (*get_area)(void *);

/*!
Function pointer to 
OUTBUF_get_leakage_power
*/
double (*get_leakage_power)(void *);

/*!
Function pointer to 
OUTBUF_get_leakage_power
*/
double (*get_internal_power)(void *);

/*!
Function pointer to 
OUTBUF_get_leakage_power
*/
double (*get_switching_power)(void *);
} OUTBUF;

/*! 
\fn OUTBUF_initialize
\param OUTBUF *
\return TRUE if initialization went through without errors, FALSE otherwise
\brief This routine will initialize a OUTBUF based on supplied input 
parameters and initialize all class private variables.
@relates OUTBUF
*/
int OUTBUF_initialize(OUTBUF *outbuffer);

/*!
\fn OUTBUF_get_instances
\param OUTBUF *
\return Returns instance count of OUTBUF.
@relates OUTBUF
*/
int OUTBUF_get_instances(void *);

/*!
\fn OUTBUF_get_area
\param OUTBUF * 
\return area of OUTBUF whose type is double
\brief Calculates and returns area of OUTBUF based on instance count and 
Area of standard cells 
@relates OUTBUF
*/
double OUTBUF_get_area(void *);

/*!
\fn OUTBUF_get_leakage_power
\param OUTBUF * 
\return  leakage power of OUTBUF whose type is double
\brief Calculates leakage power based off instance count and leakage power 
of standard cells
@relates OUTBUF
*/
double OUTBUF_get_leakage_power(void *);

/*!
\fn OUTBUF_get_internal_power
\param OUTBUF * 
\return internal power OUTBUF whose type is double
\brief Calculates and returns the internal power of the OUTBUF block based 
on instance count, internal energy
of standard cells and toggle rate.
@relates OUTBUF
*/
double OUTBUF_get_internal_power(void *);

/*!
\fn OUTBUF_get_switching_power
\param OUTBUF * 
\return switching power of OUTBUF whose type is double,
\brief calculates switching power based off load capacitance, vdd(global),
 tech(global), tr, instance count, and frequency (global)
@relates OUTBUF
*/
double OUTBUF_get_switching_power(void *);

#endif
