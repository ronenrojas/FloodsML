#include "header_file.h"
#include <string.h>       
#include <stdio.h> 
#include <math.h> 
#include <stdlib.h> 

#define NOTFEASIBLE 1


int routing(struct SMA *sma)
     
     /* This is a revised subroutine ex2.  It converts total channel
      * inflow to simulated instantaneous streamflow in cms, then
      * converts simulated instantaneous streamflow to simulated 
      * mean daily streamflow in cms.  This subroutine combines 
      * the functions of subroutines ex2 and ex6 of the NWSRFS
      * model as modified for research on the University of
      * Arizona campus.  Last revised 4/25/88 wtw. */
     
     /* THIS SUBROUTINE WAS WRITTEN IN C FROM FORTRAN CODE BY
      * PATRICE O. YAPO 7/27/93 */
     /* MODIFIED and EXTRACTED FROM MIKE WINCHEL'S DIRECTORY TO GET A PLAIN SAC
MODEL RUNNING.... Feyzan, August 2000*/
     
     
{
  int i,j,k,nno;
  
  /*  Convert total channel inflow (TCI, in mm), to simulated
   *  instantaneous streamflow (SQIN, in cms). */
  
  for (i=0;i<(TIMESPERDAY * MAXTSTEP);i++) {
    sma->tci_in[i] = 0.;
    nno = min(sma->UH_nord, (i+1));
    for (j=0;j<nno;j++) { /* Performs convolution: tci_in * uhg */
      sma->tci_in[i] += sma->tci_total[i-j] * sma->UH[j];
    }
  }

  /*  Convert simulated instantaneous streamflow to simulated mean
   * daily streamflow (SMQ, in cms). */
  for (i=0;i<MAXTSTEP;i++) {
    if (i == 0) {
      sma->tci_out[0] = sma->tci_in[0]/4.+sma->tci_in[1]/4.+sma->tci_in[2]/4.+sma->tci_in[3]/8.;
    }
    else {
      j = 4 * i - 1;
      sma->tci_out[i]= (sma->tci_in[j] + sma->tci_in[j+4])/8.; 
      for (k=1;k<=3;k++) sma->tci_out[i] += (sma->tci_in[j+k])/4.;
    }
  }
  return 0;
} 
/* %%%%%%%%%%%%%%%%%%%%%%%%%%% 80 characters wide %%%%%%%%%%%%%%%%%%%%%%%%%%% */

/* int sac_sma(struct MODELPAR *parmet) { */

int sac_sma(struct PARAMETERS *parameters,struct OBS *obs,
	  struct OUTPUT *output) {
  int i,j;
  int ii,n,gg,t_end,t,fact ;
  float kk ;
  
  struct SMA sma;
  struct FSUM1 fsum1;
  
  /* ASSIGN MODEL SPECIFIC PARAMETERS TO PARAMETERS STRUCT */
  sma.uztwm = parameters->parameter[1]->current;
  sma.uzfwm = parameters->parameter[2]->current;
  sma.uzk   = parameters->parameter[3]->current;
  sma.pctim = parameters->parameter[4]->current;
  sma.adimp = parameters->parameter[5]->current;
  sma.riva  = 0.0;
  sma.zperc = parameters->parameter[6]->current;
  sma.rexp  = parameters->parameter[7]->current;
  sma.lztwm = parameters->parameter[8]->current;
  sma.lzfsm = parameters->parameter[9]->current;
  sma.lzfpm = parameters->parameter[10]->current;
  sma.lzsk  = parameters->parameter[11]->current;
  sma.lzpk  = parameters->parameter[12]->current;
  sma.pfree = parameters->parameter[13]->current;
  sma.rserv = 0.3;
  sma.side  = 0.0;
  sma.pxmlt = 1.0;

   sma.uztwc = 0.5 * sma.uztwm;
   sma.uzfwc = 0.5 * sma.uzfwm;
   sma.lztwc = 0.5 * sma.lztwm;
   sma.lzfsc = 0.5 * sma.lzfsm;
   sma.lzfpc = 0.5 * sma.lzfpm;
   sma.adimc = 0.5 * (sma.uztwm + sma.lztwm);



 n =   parameters->parameter[15]->current;
 kk  = parameters->parameter[14]->current;

 fact = 100;
 t_end = 19;
 for ( t = 0; t < t_end; t = t + 1 )
 {
     for (gg=1,ii=1 ; ii<=n-1 ; ii++)
        gg=gg*ii;
  sma.UH[t]=    ( 1 / (kk * gg ) ) *pow( ( t / kk ),(n-1)) *exp(-1 * t / kk ) * fact ;
/*  printf("%f\n",sma.UH[t] ) ;
*/
}
 sma.UH_nord = 19;


/*DIRTY METHOD TO CANCEL THE ROUTING PROCEDURE  */ /*eylon */
/*  sma.UH[0] = 1. ;
  sma.UH_nord = 1;
*/

  /* SET SOME INITIAL VALUES TO ZERO */
  fsum1.srot=fsum1.simpvt=fsum1.srodt=fsum1.srost=0.;
  fsum1.sintft=fsum1.sgwfp=fsum1.sgwfs=fsum1.srecht=fsum1.sett=0.;
  fsum1.se1=fsum1.se3=fsum1.se4=fsum1.se5=0.;
  
 
  /* SET EVAPOTRANSPIRATION DISTRIBUTION */ 
  /* (0.25 = UNIFORM RATE/ DAY) */
  sma.epdist = 0.25;
  
  /* DT IS THE LENGTH OF EACH TIME INTERVAL IN DAYS */
  /* DT IS USED TO CALCULATE dinc IN fland1 */
  sma.dt = 0.25;  /*  */

 
  for (i=0;i<MAXTSTEP;i++) {
    /* ASSIGN AND ADJUST PET VALUE */
    sma.ep =  obs->PET[i][0];
    sma.ep *= sma.epdist;

    for (j=0;j<TIMESPERDAY;j++) {
      /* ASSIGN AND ADJUST PRECIPITATION VALUE */
      sma.pxv =obs->Precip[i][j];
      sma.pxv *= sma.pxmlt; 
      
      /* PERFORM SOIL MOISTURE ACCOUNTING OPERATIONS  */
      fland1(&sma,&fsum1);

      /* SET TOTAL CHANNEL INFLOW EQUAL TO THE EXCESS AT THE END  */
      /* OF EACH TIME PERIOD (4 TIMESTEPS/DAY) */
      sma.tci_total[4*i + j] = sma.tlci;
    }
  }


  /*  PERFORM UH ROUTING */
   routing(&sma);
  
  
  
   for (i=0;i<MAXTSTEP;i++) {
    output->Qcomp_total[i][0] = sma.tci_out[i];
   }
  
  
  
  return 0;
}


