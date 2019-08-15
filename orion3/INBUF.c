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

#include "INBUF.h"

int INBUF_initialize(INBUF *inbuffer)
{		
	inbuffer->p_in = PARM(in_port);		/*in ports*/
	inbuffer->p_out = PARM(out_port);		/*out ports*/
	inbuffer->v_channel = PARM(v_channel);		/*virtual channels*/
	inbuffer->flit_width = PARM(flit_width);		/*flit width*/
	inbuffer->buf_in = PARM(in_buf_set);		/*input buffers*/
	inbuffer->buf_out=  PARM(out_buf_set);		/*ouput buffers*/

	inbuffer->clk = PARM(FREQ_Hz);
	inbuffer->tr = PARM(tr); 
		
	inbuffer->insts = 180*inbuffer->p_in*inbuffer->v_channel + 2*
						inbuffer->p_in*inbuffer->v_channel*inbuffer->buf_in*
						inbuffer->flit_width + 2*inbuffer->p_in*
						inbuffer->p_in*inbuffer->v_channel*inbuffer->buf_in
						+ 3*inbuffer->p_in*inbuffer->v_channel*
						inbuffer->buf_in + 5*inbuffer->p_in*inbuffer->p_in
						*inbuffer->buf_in +	inbuffer->p_in*inbuffer->p_in 
						+ inbuffer->p_in*inbuffer->flit_width + 
						15*inbuffer->p_in;
	
	inbuffer->get_instances = INBUF_get_instances;
	inbuffer->get_area = INBUF_get_area;
	inbuffer->get_leakage_power = INBUF_get_leakage_power;
	inbuffer->get_internal_power = INBUF_get_internal_power;
	inbuffer->get_switching_power = INBUF_get_switching_power;
	return 1;
}		


int INBUF_get_instances(void *inbuffer)
{
	return ((INBUF*)inbuffer)->insts;
}

double INBUF_get_area(void *inbuffer)
{
	return ((Area_AOI_um2 + Area_DFF_um2)/2) * ((INBUF*)inbuffer)->insts;
}

double INBUF_get_leakage_power(void *inbuffer)
{
	return ((AOI_leak_nW + DFF_leak_nW)/2) * ((INBUF*)inbuffer)->insts;
}


double INBUF_get_internal_power(void *inbuffer)
{
	
	return (AOI_int_J + DFF_int_J) *.5* (((INBUF*)inbuffer)->insts 
			* ((INBUF*)inbuffer)->tr + .05*((INBUF*)inbuffer)->insts); 
}

double INBUF_get_switching_power(void *inbuffer)
{

	return  0.5 *1.4 * PARM(VDD_V) * PARM(VDD_V)*((INBUF*)inbuffer)->clk*.5* 
			(((INBUF*)inbuffer)->insts * ((INBUF*)inbuffer)->tr*AOI_load_pF  
			+ .05*((INBUF*)inbuffer)->insts*DFF_load_pF);
	
}
