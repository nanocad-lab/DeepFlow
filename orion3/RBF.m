%{
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
%}

P = strcat(pwd, '/rbf2'); 
addpath(genpath(P));

fid = fopen('params.txt', 'r');
TECH = fgetl(fid);
TECH = str2num(TECH);
new_q = fscanf(fid, '%u %u %u %u', [4 inf]);
fclose(fid);


if TECH == 65
	insts = fopen('selected_area_65.txt', 'r');
	area = fscanf(insts, '%u %u %u %u %f', [5 inf]);
	fclose(insts);
	
	
	insts = fopen('selected_power_65.txt', 'r');
	power = fscanf(insts, '%u %u %u %u %f %f %f %f', [8 inf]);
	fclose(insts);
	
	
elseif TECH == 45
	
	insts = fopen('selected_area_45.txt', 'r');
	area = fscanf(insts, '%u %u %u %u %f', [5 inf]);
	fclose(insts);
	
	
	insts = fopen('selected_power_45.txt', 'r');
	power = fscanf(insts, '%u %u %u %u %f %f %f %f', [8 inf]);
	fclose(insts);
	
	
	else
	
	%throw error
	
	end
	
X = area(1:4,:);
area = area';
Y = area(:,5);

confa.msc = 'GCV';
confa.type = 'm';
confa.scales = [.5 1 1.6];
confa.lambdas = 1;

[C, R, w, info, conf] = rbf_rr_2(X, log10(Y), confa);

confdma.type = 'm';

H = rbf_dm(new_q, C, R,confdma);

area_q= H * w;

area_q = 10.^(area_q);

newfid = fopen('total_area.txt', 'w+');
fprintf(newfid, '%f  \n', area_q);
fclose(newfid);

power = power';
int_pow = power(:,5);
sw_pow = power(:,6);
leak_pow = power(:,7);

conf.msc = 'GCV';
conf.type = 'm';
conf.scales = 11;

[int_C, int_R, int_w, int_info, int_conf] = rbf_rr_2(X, log10(int_pow), conf);
[sw_C, sw_R, sw_w, sw_info, sw_conf] = rbf_rr_2(X, log10(sw_pow), conf);
[leak_C, leak_R, leak_w, leak_info, leak_conf] = rbf_rr_2(X, log10(leak_pow), conf);

confdm.type = 'm';

H = rbf_dm(new_q, int_C, int_R,confdm);
int_powq = H * int_w;

H = rbf_dm(new_q, sw_C, sw_R,confdm);
sw_powq = H * sw_w;

H = rbf_dm(new_q, leak_C, leak_R,confdm);
leak_powq = H * leak_w;

int_powq =10.^(int_powq);
sw_powq =10.^(sw_powq) ;
leak_powq =10.^(leak_powq);

total_powq = int_powq + sw_powq + leak_powq;

newfid = fopen('int_pow.txt', 'w+');
fprintf(newfid, '%f  \n', int_powq);
fclose(newfid);

newfid1 = fopen('sw_pow.txt', 'w+');
fprintf(newfid1, '%f \n', sw_powq);
fclose(newfid1);

newfid2 = fopen('leak_pow.txt', 'w+');
fprintf(newfid2, '%f \n', leak_powq);
fclose(newfid2);

newfid3 = fopen('total_pow.txt', 'w+');
fprintf(newfid3, '%f \n', total_powq);
fprintf(newfid3, '%s', 'ABKGROUP_ORION');
fclose(newfid3);

exit
