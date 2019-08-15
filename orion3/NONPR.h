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
#ifndef _NONPR_H
#define _NONPR_H

/*! 
\fn LSQR
\brief This function will call a matlab script which uses LSQR
to generate a model. It will then output predictions to 
int_pow.txt leak_pow.txt sw_pow.txt total_area.txt
\return Area in um^2
\return Internal Power in mW 
\return Switching Power in mW
\return Leakage Power in mW
\return Total Power in mW   
*/
void LSQR();

/*! 
\fn RBF
\brief This function will call a matlab script which uses RBF
to generate a model. It will then output predictions to 
int_pow.txt leak_pow.txt sw_pow.txt total_area.txt
\return Area in um^2
\return Internal Power in mW 
\return Switching Power in mW
\return Leakage Power in mW
\return Total Power in mW   
*/
void RBF();

/*! 
\fn KG
\brief This function will call a matlab script which uses KG
to generate a model. It will then output predictions to 
int_pow.txt leak_pow.txt sw_pow.txt total_area.txt
\return Area in um^2
\return Internal Power in mW 
\return Switching Power in mW
\return Leakage Power in mW
\return Total Power in mW   
*/
void KG();

/*! 
\fn MARS
\brief This function will call a matlab script which uses MARS
to generate a model. It will then output predictions to 
int_pow.txt leak_pow.txt sw_pow.txt total_area.txt
\return Area in um^2
\return Internal Power in mW 
\return Switching Power in mW
\return Leakage Power in mW
\return Total Power in mW   
*/
void MARS();

/*! 
\fn SVM
\brief This function will call a matlab script which uses SVM
to generate a model. It will then output predictions to 
int_pow.txt leak_pow.txt sw_pow.txt total_area.txt
\return Area in um^2
\return Internal Power in mW 
\return Switching Power in mW
\return Leakage Power in mW
\return Total Power in mW   
*/
void SVM();

#endif