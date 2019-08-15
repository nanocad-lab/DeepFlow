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
#include "CLKCTRL.h"
#include <math.h>

int CLKCTRL_initialize(CLKCTRL *clockctrl)
{
	clockctrl->p_in = PARM(in_port);		/*in ports*/
	clockctrl->p_out = PARM(out_port);		/*out ports*/
	clockctrl->v_channel = PARM(v_channel);		/*virtual channels*/
	clockctrl->flit_width = PARM(flit_width);		/*flit width*/
	clockctrl->buf_in = PARM(in_buf_set);		/*input buffers*/
	clockctrl->buf_out=  PARM(out_buf_set);		/*ouput buffers*/
	
	clockctrl->clk = PARM(FREQ_Hz);
	clockctrl->tr = PARM(tr); 
		
	clockctrl->insts = .02*((25*clockctrl->p_out + 80*clockctrl->p_out *
						clockctrl->v_channel) + (180*clockctrl->p_in*
						clockctrl->v_channel + 2*clockctrl->p_in
						*clockctrl->v_channel*clockctrl->buf_in*
						clockctrl->flit_width + 2*clockctrl->p_in*
						clockctrl->p_in*clockctrl->v_channel*
						clockctrl->buf_in +	3*clockctrl->p_in*
						clockctrl->v_channel*clockctrl->buf_in +
						5*clockctrl->p_in*clockctrl->p_in*clockctrl->buf_in +
						clockctrl->p_in*clockctrl->p_in + clockctrl->p_in
						*clockctrl->flit_width + 15*clockctrl->p_in) +
						9 * ((pow(clockctrl->p_in,2)* clockctrl->v_channel * 
						clockctrl->v_channel) + pow(clockctrl->p_in,2)
						+ (clockctrl->p_in * clockctrl->v_channel) 
						- clockctrl->p_in));
	
	clockctrl->get_instances = CLKCTRL_get_instances;
	clockctrl->get_area = CLKCTRL_get_area;
	clockctrl->get_leakage_power = CLKCTRL_get_leakage_power;
	clockctrl->get_internal_power = CLKCTRL_get_internal_power;
	clockctrl->get_switching_power = CLKCTRL_get_switching_power;
		
	return 1;
}

int CLKCTRL_get_instances(void *clockctrl)
{
	return ((CLKCTRL*)clockctrl)->insts;
}

double CLKCTRL_get_area(void *clockctrl)
{
	return ((Area_AOI_um2 + Area_INV_um2)/2) * ((CLKCTRL*)clockctrl)->insts;
}

double CLKCTRL_get_leakage_power(void *clockctrl)
{
	return ((AOI_leak_nW + INV_leak_nW)/2) * ((CLKCTRL*)clockctrl)->insts;
}
double CLKCTRL_get_internal_power(void *clockctrl)
{
	return (AOI_int_J + INV_int_J) * ((CLKCTRL*)clockctrl)->insts 
			* ((CLKCTRL*)clockctrl)->tr; 
}

double CLKCTRL_get_switching_power(void *clockctrl)
{	
	return  .5 * 1.4 *(INV_load_pF + AOI_load_pF) * PARM(VDD_V) * PARM(VDD_V) * 
			((CLKCTRL*)clockctrl)->clk *((CLKCTRL*)clockctrl)->insts 
			* ((CLKCTRL*)clockctrl)->tr;
	
}
