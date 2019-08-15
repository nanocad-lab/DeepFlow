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

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "router_area.h"

int router_area_initialize(router_info_t *info, router_area_t *router_area)
{
	XBAR_initialize(&router_area->a_crossbar);
	SWVC_initialize(&router_area->a_arbiter);
	INBUF_initialize(&router_area->a_inbuffer);
	OUTBUF_initialize(&router_area->a_outbuffer);
	CLKCTRL_initialize(&router_area->a_clockctrl);
	
	return 1;
}

double get_router_area(router_area_t *router_area)
{
	double a_xbar, a_inbuffer,a_outbuffer, a_swvc, a_clkctrl, a_total;
	double insts_xbar, insts_inbuffer,insts_outbuffer, insts_swvc, insts_clkctrl;
	
	a_xbar = router_area->a_crossbar.get_area(&router_area->a_crossbar);
	a_swvc = router_area->a_arbiter.get_area(&router_area->a_arbiter);
	a_inbuffer = router_area->a_inbuffer.get_area(&router_area->a_inbuffer);
	a_outbuffer = router_area->a_outbuffer.get_area(&router_area->a_outbuffer);
	a_clkctrl = router_area->a_clockctrl.get_area(&router_area->a_clockctrl);
	
	a_total = a_xbar + a_inbuffer+ a_outbuffer + a_swvc + a_clkctrl;
	
	insts_xbar = router_area->a_crossbar.get_instances(&router_area->a_crossbar);
	insts_swvc = router_area->a_arbiter.get_instances(&router_area->a_arbiter);
	insts_inbuffer = router_area->a_inbuffer.get_instances(&router_area->a_inbuffer);
	insts_outbuffer = router_area->a_outbuffer.get_instances(&router_area->a_outbuffer);
	insts_clkctrl = router_area->a_clockctrl.get_instances(&router_area->a_clockctrl);

#if( PARM(TECH_POINT) <= 90 )
	fprintf(stdout, "INSTSinbuffer:%g\tINSTSoutbuffer:%g\t INSTScrossbar:%g\t INSTSswvc:%g\t INSTSclkctrl:%g\t Ainbuffer:%g\tAoutbuffer:%g\t Acrossbar:%g\t Aswvc:%g\t Aclkctrl:%g\t Atotal:%g\n", 
	insts_inbuffer,insts_outbuffer,insts_xbar, insts_swvc, insts_clkctrl,a_inbuffer,a_outbuffer, a_xbar, a_swvc, a_clkctrl, a_total);	

#else
    fprintf(stderr, "Router area is only supported for 65nm and 45nm\n");
#endif
	return a_total;
}
