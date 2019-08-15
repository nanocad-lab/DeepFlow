#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <string.h>

int main() {
		FILE *fp;
		char line[20];
		float q;
		float total;
		
		fp=fopen("total_area.txt", "r");
		while(fgets(line, 20, fp) != NULL) {
			sscanf (line, "%f", &q);
			printf ("Total Area:%f\n", q);
		}
		fclose(fp);
		
		fp=fopen("int_pow.txt", "r");
		while(fgets(line, 20, fp) != NULL) {
			sscanf (line, "%f", &q);
			printf ("Internal Power %f \t", q);
		}
		fclose(fp);
		total = total +q; 
		
		fp=fopen("sw_pow.txt", "r");
		while(fgets(line, 20, fp) != NULL) {
			sscanf (line, "%f", &q);
			printf ("Switching Power:%f \t", q);
		}
		fclose(fp);
		total = total +q; 
		
		fp=fopen("leak_pow.txt", "r");
		while(fgets(line, 20, fp) != NULL) {
			sscanf (line, "%f", &q);
			printf ("Leakage Power:%f \n", q);
			
		}
		fclose(fp);
		total = total +q; 
		
		fp=fopen("leak_pow.txt", "r");
		while(fgets(line, 20, fp) != NULL) {
			sscanf (line, "%f", &q);
			printf ("Leakage Power:%f \n", q);
		}
		fclose(fp);
		total = total +q; 
		
		printf ("Total Power:%f \n", total);
  
		 
		 return 1;
		 }