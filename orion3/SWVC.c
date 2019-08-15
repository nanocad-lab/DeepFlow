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

#include "SWVC.h"
#include <math.h>

int SWVC_initialize(SWVC *arbiter)
{
	arbiter->p_in = PARM(in_port);		/*in ports*/
	arbiter->p_out = PARM(out_port);		/*out ports*/
	arbiter->v_channel = PARM(v_channel);		/*virtual channels*/
	arbiter->flit_width = PARM(flit_width);		/*flit width*/
	arbiter->buf_in = PARM(in_buf_set);		/*input buffers*/
	arbiter->buf_out =  PARM(out_buf_set);		/*ouput buffers*/
	
	arbiter->clk = PARM(FREQ_Hz);
	arbiter->tr = PARM(tr); 
		
	arbiter->insts = 9 * ((pow(arbiter->p_in,2)* arbiter->v_channel * 
					arbiter->v_channel) + pow(arbiter->p_in,2)
					+ (arbiter->p_in * arbiter->v_channel) - arbiter->p_in);
	arbiter->get_instances = SWVC_get_instances;
	arbiter->get_area = SWVC_get_area;
	arbiter->get_leakage_power = SWVC_get_leakage_power;
	arbiter->get_internal_power = SWVC_get_internal_power;
	arbiter->get_switching_power = SWVC_get_switching_power;
	return 1;
}

int SWVC_get_instances(void *arbiter)
{
	return ((SWVC*)arbiter)->insts;
}

double SWVC_get_area(void *arbiter)
{
	return ((6*Area_NOR_um2 + 2*Area_INV_um2 + Area_DFF_um2)/9) *
			((SWVC*)arbiter)->insts;
}

double SWVC_get_leakage_power(void *arbiter)
{
	return ((6*NOR_leak_nW + 2*INV_leak_nW + DFF_leak_nW)/9)* 
			((SWVC*)arbiter)->insts;
}
double SWVC_get_internal_power(void *arbiter)
{
	return (6*NOR_int_J + 2*INV_int_J + DFF_int_J) * ((SWVC*)arbiter)->insts
			*((SWVC*)arbiter)->tr; 
}

double SWVC_get_switching_power(void *arbiter)
{
	return 0.5 *1.4 * (NOR_load_pF + INV_load_pF + DFF_load_pF) 
			* PARM(VDD_V)*	PARM(VDD_V) *((SWVC*)arbiter)->clk *
			((SWVC*)arbiter)->insts*((SWVC*)arbiter)->tr;
}

