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
#include <string.h>
#include "stdlib.h"
#include <math.h>

#include "router_power.h"



int router_power_initialize(router_info_t *info, router_power_t *router)
{		
	XBAR_initialize(&router->p_crossbar);
	SWVC_initialize(&router->p_arbiter);
	INBUF_initialize(&router->p_inbuffer);
	OUTBUF_initialize(&router->p_outbuffer);
	CLKCTRL_initialize(&router->p_clockctrl);
	
	return 1;

}


double get_router_power(router_power_t *router)
{	
	
	double insts_xbar, insts_inbuffer,insts_outbuffer, insts_swvc, insts_clkctrl;
	double p_leakage, p_internal, p_switching, p_total;
	
	insts_xbar = router->p_crossbar.get_instances(&router->p_crossbar);
	insts_swvc = router->p_arbiter.get_instances(&router->p_arbiter);
	insts_inbuffer = router->p_inbuffer.get_instances(&router->p_inbuffer);
	insts_outbuffer =router->p_outbuffer.get_instances(&router->p_outbuffer);
	insts_clkctrl = router->p_clockctrl.get_instances(&router->p_clockctrl);
	
	p_leakage = 1e-6 *(router->p_crossbar.get_leakage_power(&router->p_crossbar) + 
				router->p_arbiter.get_leakage_power(&router->p_arbiter) +
				router->p_inbuffer.get_leakage_power(&router->p_inbuffer)+
				router->p_outbuffer.get_leakage_power(&router->p_outbuffer)+ 
				router->p_clockctrl.get_leakage_power(&router->p_clockctrl));
				
	p_internal =(router->p_crossbar.get_internal_power(&router->p_crossbar) + 
				router->p_arbiter.get_internal_power(&router->p_arbiter) +
				router->p_inbuffer.get_internal_power(&router->p_inbuffer)+
				router->p_outbuffer.get_internal_power(&router->p_outbuffer)+ 
				router->p_clockctrl.get_internal_power(&router->p_clockctrl));
				
	p_switching = 1e-9*(router->p_crossbar.get_switching_power(&router->p_crossbar) + 
				router->p_arbiter.get_switching_power(&router->p_arbiter) +
				router->p_inbuffer.get_switching_power(&router->p_inbuffer)+
				router->p_outbuffer.get_switching_power(&router->p_outbuffer)+ 
				router->p_clockctrl.get_switching_power(&router->p_clockctrl));
	
	p_total = p_leakage + p_internal+ p_switching; 
	
	fprintf(stdout, "INSTSinbuffer:%g\tINSTSoutbuffer:%g\t INSTScrossbar:%g\t INSTSswvc:%g\t INSTSclkctrl:%g\tPleakage:%g\tPinternal:%g\tPswitching:%g\tPtotal:%g\n", 
	insts_inbuffer,insts_outbuffer,insts_xbar, insts_swvc, insts_clkctrl,p_leakage, p_internal, p_switching, p_total);
	
	return p_total;
}
