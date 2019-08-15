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

fid = fopen('params.txt', 'r');
TECH = fgetl(fid);
TECH = str2num(TECH);
new_q = fscanf(fid, '%u %u %u %u', [4 inf]);
fclose(fid);

[q_count,q_n] = size(new_q);

if TECH == 65
	insts = fopen('selected_area_65.txt', 'r');
	area = fscanf(insts, '%u %u %u %u %f', [5 inf]);
	fclose(insts);
	area= area';
	
	insts = fopen('selected_power_65.txt', 'r');
	power = fscanf(insts, '%u %u %u %u %f %f %f %f', [8 inf]);
	fclose(insts);
	power= power';
	
elseif TECH == 45
	
	insts = fopen('selected_area_45.txt', 'r');
	area = fscanf(insts, '%u %u %u %u %f', [5 inf]);
	fclose(insts);
	area= area';
	
	insts = fopen('selected_power_45.txt', 'r');
	power = fscanf(insts, '%u %u %u %u %f %f %f %f', [8 inf]);
	fclose(insts);
	power= power';
	
	else
	
	%throw error
	
	end
	
[count,n] = size(area);

B = area(:,1);
V = area(:,2);
P = area(:,3);
F = area(:,4);

xbar = P .* P .* F;
swvc = 9 .* ((P .* P .* V .* V) + (P .* P) + (P .* V) - P);
inbuf = (180 .* P .* V) + (2 .* P .* V.* B .* F) + (2 .* P.*P.*B.*V) + (3 .* P.*V.*B) + (5 .* P.*P.*B) + (P.*P) + (P.*F) + 15.*P;
outbuf = (80 .* P .* V) + (P.*25);
clkctrl = .02* (swvc + inbuf + outbuf); 

area_total = area(:,5);

total = [xbar swvc inbuf outbuf clkctrl ones(count,1)];

options = optimset('TolX',1e-11); 
x = lsqnonneg(total, area_total);


B = new_q(1,:)';
V = new_q(2,:)';
P = new_q(3,:)';
F = new_q(4,:)';

nxbar = P .* P .* F;
nswvc = 9 .* ((P .* P .* V .* V) + (P .* P) + (P .* V) - P);
ninbuf = (180 .* P .* V) + (2 .* P .* V.* B .* F) + (2 .* P.*P.*B.*V) + (3 .* P.*V.*B) + (5 .* P.*P.*B) + (P.*P) + (P.*F) + 15.*P;
noutbuf = (80 .* P .* V) + (P.*25);
nclkctrl = .02* (nswvc + ninbuf + noutbuf); 

ntotal = [nxbar nswvc ninbuf noutbuf nclkctrl ones(q_n,1)];


for i = 1:q_n
	area_q(i,:) = sum(ntotal(i,:) .* x');
end

newfid = fopen('total_area.txt', 'w+');
fprintf(newfid, '%f  \n',area_q);
fclose(newfid);

int_pow = power(:,5);
sw_pow = power(:,6);
leak_pow = power(:,7);

int = lsqnonneg(total,int_pow);
sw = lsqnonneg(total, sw_pow);
leak = lsqnonneg(total, leak_pow);

for i = 1:q_n
	intq(i,:) = sum(ntotal(i,:) .* int');
	swq(i,:) = sum(ntotal(i,:) .* sw');
	leakq(i,:) = sum(ntotal(i,:) .* leak');
	
end

total_powq = intq + swq + leakq;

newfid = fopen('int_pow.txt', 'w+');
fprintf(newfid, '%f  \n', intq);
fclose(newfid);

newfid1 = fopen('sw_pow.txt', 'w+');
fprintf(newfid1, '%f \n', swq);
fclose(newfid1);

newfid2 = fopen('leak_pow.txt', 'w+');
fprintf(newfid2, '%f \n', leakq);
fprintf(newfid, '%s', '$$$');
fclose(newfid2);

newfid3 = fopen('total_pow.txt', 'w+');
fprintf(newfid3, '%f \n', total_powq);
fprintf(newfid3, '%s', 'ABKGROUP_ORION');
fclose(newfid3);

exit
