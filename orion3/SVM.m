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

P = strcat(pwd, '/libsvm-3.12/libsvm-3.12/matlab'); 
addpath(genpath(P));

fid = fopen('params.txt', 'r');
TECH = fgetl(fid);
TECH = str2num(TECH);
new_q = fscanf(fid, '%u %u %u %u', [4 inf]);
fclose(fid);

new_q = new_q';
[count, n] = size(new_q);

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

%Area
areaX = area(:,1:4);
areaY = log10(area(:,5));

s = 4; t = 2; C = max(areaY) - min(areaY); gset = 2.^[-7:7]; eset = 1; wt = 1;
param.libsvm = ['-s ', num2str(s), ' -t ', num2str(t), ...                    
                ' -c ', num2str(C), ' -g ', num2str(gset), ...
                ' -p ', num2str(eset),...
                ' -wi ', num2str(wt)];
modelA = svmtrain(areaY, areaX, param.libsvm); 
[YpA, Acc, proj] = svmpredict(zeros(count,1), new_q, modelA);

YpA = 10.^(YpA);

newfid = fopen('total_area.txt', 'w+');
fprintf(newfid, '%f  \n', YpA);
fclose(newfid);


trnX = power(:,1:4); trnI = log10(power(:,5)); 
trnS = log10(power(:,6));
trnL = log10(power(:,7));



%Internal power
s = 4; t = 2; C = max(trnI) - min(trnI); gset = 2.^[-7:7]; eset = 1; wt = 1;
param.libsvm = ['-s ', num2str(s), ' -t ', num2str(t), ...                    
                ' -c ', num2str(C), ' -g ', num2str(gset), ...
                ' -p ', num2str(eset),...
                ' -wi ', num2str(wt)];
modelI = svmtrain(trnI, trnX, param.libsvm); 
[YpI, Acc, proj] = svmpredict(zeros(count,1), new_q, modelI);


%Switching power
s = 4; t = 2; C = max(trnS) - min(trnS); gset = 2.^[-7:7]; eset = 1; wt = 1;
param.libsvm = ['-s ', num2str(s), ' -t ', num2str(t), ...                    
                ' -c ', num2str(C), ' -g ', num2str(gset), ...
                ' -p ', num2str(eset),...
                ' -wi ', num2str(wt)];
modelS = svmtrain(trnS, trnX, param.libsvm); 
[YpS, Acc, proj] = svmpredict(zeros(count,1), new_q, modelS);


%Leakage power
s = 4; t = 2; C = max(trnL) - min(trnL); gset = 2.^[-7:7]; eset = 1; wt = 1;
param.libsvm = ['-s ', num2str(s), ' -t ', num2str(t), ...                    
                ' -c ', num2str(C), ' -g ', num2str(gset), ...
                ' -p ', num2str(eset),...
                ' -wi ', num2str(wt)];
modelL = svmtrain(trnL, trnX, param.libsvm); 
[YpL, Acc, proj] = svmpredict(zeros(count,1), new_q, modelL);

int_powq =10.^(YpI);
sw_powq =10.^(YpS);
leak_powq =10.^(YpL);

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