#include "header_file.h"
#include <string.h>
#include <stdio.h>
#include <math.h>
#include <stdlib.h>
   
#define NOTFEASIBLE 1
    

int routing(struct SMA *sma)


 fact =22.5; 
 t_end =18;  
 k =  3.;
 n = 4
  
  for t=1:t_end
     gg=prod(1:n-1);
     gm(t)= (1/(k*gg))* (t/k)^(n-1) *exp(-1*t/k);
  end








/* normelizing%%%%%%%%%%%%%%%%%%%%%%%%%5
   q=zeros(t_end,1); q(1:t_end,1) = q_gama(1:t_end)';   q=q ./ max(q) .* fact;
   uh(:,i) =q;                     % save
*/

/* q= [1.4; 3.2; 4.5; 5.1; 5.2;
     5.4;6.1;6.9;7.3;7.3;6.8;6.1;5.2;4.2;3.4;2.6;1.6;0.6];  t_end=length(q);
*/


