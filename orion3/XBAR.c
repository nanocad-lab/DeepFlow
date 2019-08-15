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
#include "XBAR.h"
#include <math.h>

int XBAR_initialize(XBAR *crsbar)
{
	crsbar->p_in = PARM(in_port);		/*in ports*/
	crsbar->p_out = PARM(out_port);		/*out ports*/
	crsbar->v_channel = PARM(v_channel);		/*virtual channels*/
	crsbar->flit_width = PARM(flit_width);		/*flit width*/
	crsbar->buf_in = PARM(in_buf_set);		/*input buffers*/
	crsbar->buf_out =  PARM(out_buf_set);		/*ouput buffers*/
	crsbar->model = PARM(crossbar_model);
	crsbar->clk = PARM(FREQ_Hz);
	crsbar->tr = PARM(tr); 
	switch(crsbar->model) {
		case MATRIX_CROSSBAR:
			crsbar->insts = crsbar->p_in * crsbar->p_out * pow(crsbar->flit_width, 2);
			break;
		case MULTREE_CROSSBAR:
			crsbar->insts = crsbar->p_in * crsbar->flit_width * crsbar->p_out; 
			break;
		case TRISTATE_CROSSBAR:
			crsbar->insts = crsbar->p_in * crsbar->p_out * crsbar->flit_width;
			break;
		
	}
	crsbar->get_instances = XBAR_get_instances;
	crsbar->get_area = XBAR_get_area;
	crsbar->get_leakage_power = XBAR_get_leakage_power;
	crsbar->get_internal_power = XBAR_get_internal_power;
	crsbar->get_switching_power = XBAR_get_switching_power;
	
	return 1;
	
}

int XBAR_get_instances(void *crsbar)
{
	return ((XBAR*)crsbar)->insts;
}

double XBAR_get_area(void *crsbar)
{
	return Area_MUX2_um2 * ((XBAR*)crsbar)->insts;
}

double XBAR_get_leakage_power(void *crsbar)
{
	return MUX2_leak_nW * ((XBAR*)crsbar)->insts;
}
double XBAR_get_internal_power(void *crsbar)
{
	return MUX2_int_J * ((XBAR*)crsbar)->tr * ((XBAR*)crsbar)->insts; 
}

double XBAR_get_switching_power(void *crsbar)
{
	return  0.5 * 1.4 * MUX2_load_pF * PARM(VDD_V) * PARM(VDD_V) * ((XBAR*)crsbar)->tr * ((XBAR*)crsbar)->clk;
}
