#include "header_file.h"
#include <string.h>
#include <stdio.h>
#include <math.h>
#include <stdlib.h>

#define EQUAL 0
#define OPTIMIZE 0
#define FIXED 1
#define NOTUSED 1

int main() {
     FILE *indata, *inpars, *outflux;
     int i;
     char s[300], s1[100];
     char in1[100], in2[100];

     struct PARAMETERS parameters;
     struct PARAMETER parameter[MAXPAR];
     struct OUTPUT output;
     struct OBS obs;

     /* READ NAME OF OBSERVED DATA FILE */
     printf("\nNAME OF OBSERVED DATA FILE: ");
     scanf("%s", in1);

     /* READ NAME OF MODEL PARAMETER FILE */
     printf("\nNAME OF 1 ROW 23 NUMBER FIELD PARAMETER FILE: ");
     scanf("%s", in2);

     /* READ NAME OF OUTPUT SUMMARY FILE FOR FLUXES */
     printf("\nNAME YOUR 2 COLUMN OUTPUT FILE FOR FLUXES: ");
     scanf("%s", s1);

     inpars = fopen(in2, "r");

     if (inpars == NULL) {
          printf("\n File %s is not opened!", in2);
          return 0;
     }

     if (fgets(s, MAXLINE, inpars) != NULL) {
          sscanf(s, " %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf  %lf %lf %lf %lf %lf", //23 current value read 
                 &parameter[1].current, &parameter[2].current,
                 &parameter[3].current, &parameter[4].current, &parameter[5].current,
                 &parameter[6].current, &parameter[7].current, &parameter[8].current,
                 &parameter[9].current, &parameter[10].current, &parameter[11].current,
                 &parameter[12].current, &parameter[13].current, &parameter[14].current,
                 &parameter[15].current, &parameter[16].current, &parameter[17].current,
                 &parameter[18].current, &parameter[19].current, &parameter[20].current,
                 &parameter[21].current, &parameter[22].current);

          for (i = 0; i < 23; i++) {
               parameter[i].calflag = FIXED;
               parameters.parameter[i] = &parameter[i];
          }
          parameters.NrOfPars = 23;
     }
     else{
          printf("\n File is Opened But Cannot read the Parameters\n");
     }
     fclose(inpars);

     indata = fopen(in1, "r");
     if (indata == NULL) {
          printf("\n File %s is not opened!", in1);
          return 0;
     }

     i = 0;
     while (fgets(s, MAXLINE, indata) != NULL) {
          sscanf(s, "%lf %lf %lf %lf %lf %lf", &obs.Qobs[i][0], &obs.PET[i][0],
                 &obs.Precip[i][0], &obs.Precip[i][1], &obs.Precip[i][2],
                 &obs.Precip[i][3]);
          i++;
     }
     obs.datalength = i;
     fclose(indata);
     
     sac_sma(&parameters, &obs, &output);

     outflux = fopen(s1, "w"); 
     for (i = 0; i < obs.datalength; i++) {
          fprintf(outflux, "%10f %10f \n", obs.Qobs[i][0], output.Qcomp_total[i][0]);
     }
     fclose(outflux);
}
