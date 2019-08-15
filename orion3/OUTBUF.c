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
#include "OUTBUF.h"

int OUTBUF_initialize(OUTBUF *outbuffer)
{
	outbuffer->p_in = PARM(in_port);		/*in ports*/
	outbuffer->p_out = PARM(out_port);		/*out ports*/
	outbuffer->v_channel = PARM(v_channel);		/*virtual channels*/
	outbuffer->flit_width = PARM(flit_width);		/*flit width*/
	outbuffer->buf_in = PARM(in_buf_set);		/*input buffers*/
	outbuffer->buf_out=  PARM(out_buf_set);		/*ouput buffers*/
	
	outbuffer->clk = PARM(FREQ_Hz);
	outbuffer->tr = PARM(tr); 
		
	outbuffer->insts = 25*outbuffer->p_out + 80*outbuffer->p_out*
						outbuffer->v_channel;
	
	outbuffer->get_instances = OUTBUF_get_instances;
	outbuffer->get_area = OUTBUF_get_area;
	outbuffer->get_leakage_power = OUTBUF_get_leakage_power;
	outbuffer->get_internal_power = OUTBUF_get_internal_power;
	outbuffer->get_switching_power = OUTBUF_get_switching_power;	
	return 1;
}

int OUTBUF_get_instances(void *outbuffer)
{
	return ((OUTBUF*)outbuffer)->insts;
}

double OUTBUF_get_area(void *outbuffer)
{
	return ((Area_AOI_um2 + Area_DFF_um2)/2) * ((OUTBUF*)outbuffer)->insts;
}

double OUTBUF_get_leakage_power(void *outbuffer)
{
	return ((AOI_leak_nW + DFF_leak_nW)/2) * ((OUTBUF*)outbuffer)->insts;
}


double OUTBUF_get_internal_power(void *outbuffer)
{
	
	return (AOI_int_J + DFF_int_J) *.5* (((OUTBUF*)outbuffer)->insts 
			* ((OUTBUF*)outbuffer)->tr + .05*((OUTBUF*)outbuffer)->insts); 
}


double OUTBUF_get_switching_power(void *outbuffer)
{

	return  0.5 *1.4 * PARM(VDD_V) * PARM(VDD_V)*((OUTBUF*)outbuffer)->clk*.5
			* (((OUTBUF*)outbuffer)->insts * ((OUTBUF*)outbuffer)->tr* 
			AOI_load_pF + .05*((OUTBUF*)outbuffer)->insts* DFF_load_pF);
	
}

