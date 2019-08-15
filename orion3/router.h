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

#ifndef _ROUTER_H
#define _ROUTER_H

#include "SIM_parameter.h"
#include "XBAR.h"
#include "SWVC.h"
#include "CLKCTRL.h"
#include "INBUF.h"
#include "OUTBUF.h"

/*!
\brief A stuct defining a router_area_t
*
*	This struct is used to intialize mutliple classes for use in area 
*	calculations.
*/
typedef struct {
	XBAR a_crossbar;
	SWVC a_arbiter;
	INBUF a_inbuffer;
	OUTBUF a_outbuffer;
	CLKCTRL a_clockctrl;	
} router_area_t;

/*!
\brief A stuct defining a router_power_t
*
*	This struct is used to intialize mutliple classes for use in 
*	power calculations.
*/
typedef struct {
	XBAR p_crossbar;
	SWVC p_arbiter;
	INBUF p_inbuffer;
	OUTBUF p_outbuffer;
	CLKCTRL p_clockctrl;	
} router_power_t;

/*!
\brief A stuct defining a router_info_t
*
*	This struct is used to intialize mutliple classes for use in providing
*   specific router information to various calculations.
*/
typedef struct {
	u_int p_in;
	u_int p_out;
	u_int v_channel;
	u_int flit_width;
	u_int buf_in;
	u_int buf_out;
	float clk;
	float tr;
	
	/*crossbar parameters*/
	int model;
} router_info_t;


/* global variables */
extern GLOBDEF(router_power_t, router_power);
extern GLOBDEF(router_info_t, router_info);
extern GLOBDEF(router_area_t, router_area);

/*!
\fn router_initialize
\param router_info_t *info
\param router_power_t *router_power
\param router_area_t *router_area
\param int model
\return TRUE if router_initialize succeeded, FALSE otherwise
\brief Initializes a router to be used in power and area calculations 
\see	router_power_initialize(router_info_t*, router_power_t*);
\see	get_router_power(router_power_t*);
\see	router_area_initialize(router_info_t *info, router_area_t *router_area);
\see	get_router_area(router_area_t *router_area);
*/
extern int router_initialize(router_info_t *info, router_power_t *router_power, router_area_t *router_area);


/*!
\fn router_power_initialize
\param router_info_t *info
\param router_power_t *router
\return TRUE if router_initialize succeeded, FALSE otherwise
\brief Initializes a router to be used in power and area calculations
\see	XBAR_initialize(XBAR*);
\see	SWVC_initialize(SWVC*);
\see	INBUF_initialize(INBUF*);
\see	OUTBUF_initialize(OUTBUF*);
\see	CLKCTRL_initialize(CLKCTRL*);
*/
extern int router_power_initialize(router_info_t *info, router_power_t *router);

/*!
\fn get_router_power
\param router_power_t *router
\return Power consumption of the router
\brief Prints a statement describing the instance count and power
 consumption of each component block

*/
extern double get_router_power(router_power_t *router);


/*!
\fn router_area_initialize
\param router_info_t *info
\param router_power_t *router
\return TRUE if router_initialize succeeded, FALSE otherwise
\brief Initializes a router to be used in power and area calculations
\see	XBAR_initialize(XBAR*);
\see	SWVC_initialize(SWVC*);
\see	INBUF_initialize(INBUF*);
\see	OUTBUF_initialize(OUTBUF*);
\see	CLKCTRL_initialize(CLKCTRL*);
*/
extern int router_area_initialize(router_info_t *info, router_area_t *router_area);

/*!
\fn get_router_area
\param router_power_t *router
\return Power consumption of the router
\brief Prints a statement describing the instance count and power 
consumption of each component block

*/
extern double get_router_area(router_area_t *router_area);



#endif /* _ROUTER_H */

