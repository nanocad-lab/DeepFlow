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
#include <stdlib.h>
#include <unistd.h>
#include <string.h>
#include <time.h>


#include "SIM_parameter.h"
#include "NONPR.h"


char path[2048];
extern char *optarg;
extern int optind;

int main(int argc, char **argv)
{
	int ver = 3;
	int rmodel = 1;
	int test = 0;
	int max_flag = AVG_ENERGY;
	int plot_flag = 0;
	int str_index = 0;
	int cmd_index = 0;

	u_int print_depth = 0;
	double load = 1;
	char *name;
	char opt;
	char st[8];
	char test_path[64];
	char train_pathA[124];
	char train_pathP[124] ;
	char path_Train[256];
	char path_Test[256];
	strcpy(test_path, "default");
	strcpy(train_pathA, "default");
	strcpy(train_pathP, "default");
	
	if (argc < 2) {
	printf("No information provided. Running Orion 3.0 basic mode\n");
	} else {

		if (atoi(argv[2]) == 2) {
			printf("%s\n", "ver 2");
			if (optind >= argc) {
			fprintf(stderr, "orion_router: [-pm] [-d print_depth] [-l load] <router_name>\n");
			return 1;
			}
			ver = 2;
			while ((opt = getopt(argc, argv, "+pmd:l:")) != -1) {
			switch (opt) {
				case 'p': plot_flag = 1; break;
				case 'm': max_flag = MAX_ENERGY; break;
				case 'd': print_depth = atoi(optarg); break;
				case 'l': load = atof(optarg); break;
					}
			}
		} else {
			if (strcmp(argv[1], "-v") == 0) {
				cmd_index = 3;
			} else {
				cmd_index = 1;
			}
			if (argc <= cmd_index) {
				printf("%s\n", "No model provided. Running basic\n");			
			} else if (strcmp(argv[cmd_index], "-model") != 0) {
				printf("%s\n", "No model provided. Running basic\n");
			} else { 
				cmd_index++;
				strcpy(st, argv[cmd_index]);
				for (str_index = 0; st[str_index]; str_index++){
					st[str_index] = tolower(st[ str_index ]);
				}
				if (strcmp(st, "basic") == 0) {
					rmodel = 1;
				} else if (strcmp(st, "lsqr") == 0){
					rmodel = 2;
				} else if (strcmp(st, "rbf") == 0){
					rmodel = 3;
				} else if (strcmp(st, "kg") == 0){
					rmodel = 4;
				} else if (strcmp(st, "svm") == 0){
					rmodel = 5;
				} else if (strcmp(st, "mars") == 0){
					rmodel = 6; 
				} else {
					fprintf(stderr, "Model not supported. Use basic, lsqr, rbf, kg, mars, or svm");
				}
			}
			cmd_index++;
			if (argc > cmd_index) {
				if (strcmp(argv[cmd_index], "-test") == 0) {
					strcpy(test_path, argv[cmd_index + 1]);
					cmd_index = cmd_index + 2;
				} else if (strcmp(argv[cmd_index], "-train") == 0) {
					strcpy(train_pathA, argv[cmd_index + 1]);
					strcpy(train_pathP, argv[cmd_index + 2]);
					cmd_index = cmd_index + 3;
				}
			}
			
			if (argc > cmd_index) {
				if (strcmp(argv[cmd_index], "-test") == 0) {
					strcpy(test_path, argv[cmd_index + 1]);
					cmd_index = cmd_index + 2;
				} else if (strcmp(argv[cmd_index], "-train") == 0) {
					strcpy(train_pathA, argv[cmd_index + 1]);
					strcpy(train_pathP, argv[cmd_index + 2]);
					cmd_index = cmd_index + 3;
				}
			} 
		}	
	}
	
	if (ver == 2) {
		#include "SIM_router.h"
		name = argv[optind];
		SIM_router_init(&GLOB(SIM_router_info), &GLOB(SIM_router_power), NULL);

		SIM_router_stat_energy(&GLOB(SIM_router_info), &GLOB(SIM_router_power), 
					print_depth, name, max_flag, load, plot_flag, PARM(Freq));

		SIM_router_init(&GLOB(SIM_router_info), NULL, &GLOB(SIM_router_area));

		SIM_router_area(&GLOB(SIM_router_area));
	}
	else if(ver ==3){
		#include "router.h"
		if (rmodel ==1) {
			router_initialize(&GLOB(router_info), &GLOB(router_power), NULL);
			router_initialize(&GLOB(router_info), NULL, &GLOB(router_area));
		} else 
		{	
			strcpy(path_Test, "sh print_params.sh ");
			strcat(path_Test, test_path);
			strcpy(path_Train, train_pathA);
			strcat(path_Train, " ");
			strcat(path_Train, train_pathP);
			system(path_Test);

			if (rmodel == 2) {
				LSQR();
				system(strcat("sh LSQR.sh ", path_Train));
			} else if (rmodel == 3) {
				RBF();
				system(strcat("sh RBF.sh ", path_Train));
			} else if (rmodel == 4) {
				KG();
				system(strcat("sh KG.sh ", path_Train));
			} else if (rmodel == 5) {
				SVM();
				system(strcat("sh SVM.sh ", path_Train));
			} else if (rmodel == 6) {
				MARS();
				system(strcat("sh MARS.sh ", path_Train));
			}
		}
	} else {
		fprintf(stderr, "Version not found. Versions supported: 2, 3\n");
		return 1;
	}

return 0;
}