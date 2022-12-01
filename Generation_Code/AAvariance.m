function [ Variance, VarianceCoef ] = AAvariance( Cn_2, D, L, lamda, wave, CalcType )
%calculating the variance of the AOA.
%If CalacType = 'full', then we ucalculate the varicance based on a formula
%from the paper "Closed form approximations for the AOA variance of plane
%and spherical waves prpagating through homogenous and isotoropic turbulence.
%If not, the varianve is calculated according to the known equations, which
%are good only if q>>1

beta = 0.5216;
f = sqrt(lamda*L);
q = D/f;

if strcmp(CalcType, 'full')
    VarianceCoef = sqrt(3)/16*gamma(1/6)*gamma(8/3)*((beta/2)^(-1/3));
    if strcmp(wave, 'plane')
        VarianceCoef =VarianceCoef*(1+6/5*((pi/2)^(1/6))*((beta*q)^(1/3))*((1+((pi/2)^2)*((beta*q)^4))^(5/12)...
            *sin(5/6*atan(2/(pi*(beta*q)^2)))));
    else %%sphericak wave
        F = hypergeom([1/6 17/6], 23/6, 1-0.5i*pi*(beta*q)^2);
        VarianceCoef = 3/8*VarianceCoef*(1+16/17*real((-2i/pi/((beta*q)^2))^(-1/6)*F));
    end
    
else %% requires q>>1
    if strcmp(wave, 'sphere')
        VarianceCoef = 1.09;
    else %% plane wave
        VarianceCoef = 2.91;
    end
end

Variance = VarianceCoef*Cn_2*L*(D^(-1/3));
end

