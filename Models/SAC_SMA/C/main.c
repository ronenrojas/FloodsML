#include "header_file.h"
#include <string.h>
#include <stdio.h>
#include <math.h>
#include <stdlib.h>

#define EQUAL 0
#define OPTIMIZE 0
#define FIXED 1
#define NOTUSED 1

int main(int argc, char *argv[]) {
     FILE *indata, *inpars, *outflux;
     int i;
     char s[300];
	if (argc==4) {

	} else {
     	fprintf(stderr,"Wrong number of input\n usage: ./sim input_file param_file output_file\n");
		return 0;
	}

     static struct PARAMETERS parameters;
     static struct PARAMETER parameter[MAXPAR];
     static struct OUTPUT output;
     static struct OBS obs;

     /* READ NAME OF OBSERVED DATA FILE */
     fprintf(stderr,"NAME OF OBSERVED DATA FILE: %s\n", argv[1]);

     /* READ NAME OF MODEL PARAMETER FILE */
     fprintf(stderr,"NAME OF 1 ROW 23 NUMBER FIELD PARAMETER FILE: %s\n", argv[2]);

     /* READ NAME OF OUTPUT SUMMARY FILE FOR FLUXES */
     fprintf(stderr,"NAME YOUR 2 COLUMN OUTPUT FILE FOR FLUXES: %s\n", argv[3]);

     inpars = fopen(argv[2], "r");

     if (inpars == NULL) {
          fprintf(stderr,"\n File %s is not opened!", argv[2]);
          return 0;
     }

     if (fgets(s, MAXLINE, inpars) != NULL) {
          sscanf(s, "%lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf", //15 current value read 
                 &parameter[1].current,  &parameter[2].current,  &parameter[3].current,  &parameter[4].current,  &parameter[5].current,
                 &parameter[6].current,  &parameter[7].current,  &parameter[8].current,  &parameter[9].current,  &parameter[10].current, 
		 &parameter[11].current, &parameter[12].current, &parameter[13].current, &parameter[14].current, &parameter[15].current);

          for (i = 0; i < 23; i++) {
               parameter[i].calflag = FIXED;
               parameters.parameter[i] = &parameter[i];
          }
          parameters.NrOfPars = 23;
     }
     else{
          fprintf(stderr,"\n File is Opened But Cannot read the Parameters\n");
     }
     fclose(inpars);

     indata = fopen(argv[1], "r");
     if (indata == NULL) {
          fprintf(stderr,"\n File %s is not opened!", argv[1]);
          return 0;
     }

     i = 0;
     while (fgets(s, MAXLINE, indata) != NULL) {
          sscanf(s, "%lf %lf %lf %lf %lf", &obs.PET[i][0],
                 &obs.Precip[i][0], &obs.Precip[i][1], &obs.Precip[i][2],
                 &obs.Precip[i][3]);
          i++;
     }
     obs.datalength = i;
     fclose(indata);
     
     sac_sma(&parameters, &obs, &output);

     outflux = fopen(argv[3], "w"); 
     for (i = 0; i < obs.datalength; i++) {
          fprintf(outflux, "%lf \n", output.Qcomp_total[i][0]);
     }
     fclose(outflux);
     return 0;
}
