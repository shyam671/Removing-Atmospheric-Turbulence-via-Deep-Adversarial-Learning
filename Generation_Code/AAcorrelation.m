function b = AAcorrelation(gamma,wave,beta_max,nbeta,nkappa,nxi)
%Based on: Eq. (4) in Belen'ki et.al. "Turbulence-induced edge image
%waviness: theory and experiment."
%
%AA (AOA) (Angle Of Arrival) correlation scale from the AA correlation function
%b(theta) = < phi(theta0)*phi(theta0+theta)> / sigmaA_2 
%phi :the AA
%sigmaA_2 : the AA variance
%theta : angular separation between two sources
%< > : statistical average
%
%Output:
%Structure b with fields:
%beta = theta/theta_D : vector of normalized angles
%b_long : the correlation of the LONGITUDINAL AA, PARALLEL to angular displacement.
%b_lat : the correlation of the LATERAL AA, TRANSVERSE to angular displacement.
%
%Input:
%gamma = L0/D; %L0 outer scale
%wave = 'plane' or 'sphere' [default = 'sphere']
%beta_max = largest value of beta, in units of gamma [default = 2000, so largest beta is 2000*gamma]
%nbeta = number of normalized angle values [default = 1000]
%nkappa = number of kappa values between 1e-2 and 1e2 [default = 2000]
%nxi = number of xi values between 0 and 1 [default = 1000]
%
%Note:
%Cn_2 : turbulence refractive index structure function
%  - horizontal path : we assume that Cn_2 is constant, not needed

% Check inputs
if(~exist('wave', 'var')), wave = 'sphere'; end
if(~exist('beta_max', 'var')),
    switch wave,
        case 'plane', beta_max = 16*gamma;
        case 'sphere', beta_max = 1000*gamma;
    end
end
if(~exist('nbeta', 'var')), nbeta = 1000; end
if(~exist('nkappa', 'var')), nkappa = 2000; end
if(~exist('nxi', 'var')), nxi = 1000; end


%%
%kappa %vector of the spectrum variable - integrates to inf
%kappa = 2*pi/lambda
%lambda = [8:0.01:13]*1e-6; %wavelength - spectral response of imager in meters
% kappa = [0:0.05:50];
nkappa = 1000;
kappa = logspace(-2,2,nkappa);
nxi = 1000;
xi = linspace(0,1,nxi); % Normalized by L
beta = logspace(-2,log10(beta_max),nbeta);
T = round(15e-9 * nbeta * nkappa * nxi);    % Expected computing time

switch wave
case 'plane',
    Gk = ((2*gamma*kappa).^2 + 1).^(-11/6) .* (besselj(2, kappa).^2./kappa);
    I_minus = zeros(size(beta));
    I_plus = zeros(size(beta));
    for(t = 1:nbeta),
        J0 = besselj(0, 2*beta(t)*kappa);
        J2 = besselj(2, 2*beta(t)*kappa);
        I_minus(t) = trapz(kappa, Gk .* (J0 - J2));
        I_plus(t) = trapz(kappa, Gk .* (J0 + J2));
    end
case 'sphere',
    Q = (1-xi).^(5/3);
    Gk = ((2*gamma*kappa).^2 + 1).^(-11/6) .* (besselj(2, kappa).^2./kappa);
    I_minus = zeros(size(beta));
    I_plus = zeros(size(beta));
    h = waitbar(0,['Computing AA correlation (~', num2str(T), ' min)']);
    for(t = 1:nbeta),
        J0 = besselj(0, 2*beta(t)*(xi .* (1-xi))'*kappa);
        J2 = besselj(2, 2*beta(t)*(xi .* (1-xi))'*kappa);
        inner_minus = trapz(kappa, repmat(Gk, [nxi 1]) .* (J0 - J2), 2);
        inner_plus = trapz(kappa, repmat(Gk, [nxi 1]) .* (J0 + J2), 2);
        I_minus(t) = trapz(xi, inner_minus);
        I_plus(t) = trapz(xi, inner_plus);
    waitbar(t / nbeta, h)
    end
    close(h)
    
    % Fix inaccuracies
    i = find(diff(beta > 1000));
    I_minus(i:end) = 0;
end
    
b_long = I_minus / I_minus(1);
b_lat = I_plus / I_plus(1);
b = struct('beta', beta, 'long', b_long, 'lat', b_lat);

return


%% test
gamma = 4;
tic
b = AAcorrelation(gamma, 'sphere', 1000);
toc
plot(b.beta, b.long, b.beta, b.lat), axis([0 b.beta(end) -0.2 1])
xlabel('beta = theta/theta_{D}'), ylabel('b'), legend('b_{long}','b_{lat}')
semilogx(b.beta, b.long, b.beta, b.lat)
xlabel('theta/theta\_D'), ylabel('b'), legend('long', 'lat')

% Figure
% b1 = AAcorrelation(1);
b4 = AAcorrelation(4);
b10 = AAcorrelation(10);
% save('AA_gamma1', 'b1')
% save('AA_gamma4', 'b4')

load('AA_gamma1');
load('AA_gamma4');b4=b;
load('AA_gamma10'); b10=b;
%%
figure(1), hold on;
plot(b10.beta, b10.lat, 'r-', b10.beta, b10.long, 'r--','LineWidth',1.75);
plot(b4.beta, b4.lat, 'b-', b4.beta, b4.long, 'b--','LineWidth',1.75);
% plot(b1.beta, b1.lat, '-.', b1.beta, b1.long, '-.');
plot([0 100], [0 0], 'k-')
legend('b_{\perp} ({\rho} = 10)','b_{||}  ({\rho} = 10)', ...
    'b_{\perp} ({\rho} = 4)','b_{||} ({\rho} = 4)');
    % 'b_{lat} ({\it\gamma} = 1)','b_{long}  ({\it\gamma} = 1)');
set(gca, 'fontsize', 16, 'LineWidth',1.2);
box on;
xlabel('{\it\eta} = {\it\theta} /{\it\theta_{D}}'), ylabel('b');
axis([0 400 -0.1 1.001]), 
hold off;

% Use the same color for the same rho. Say blue for rho=10, red for rho=4.
% 
% * Use solid line for the perp, and dashed line for the parallel, in each of the cases.
% 
% * Use a line width much wider. Try 2pt. Now it is too deliacte to see convninetly without zoom.
% 
% * enlarge the fonts on the axes so they are closer in size to the font-size of the caption. You can delute the x-axix, showing, say 0,50,100,200, 300, 400
% 
% * Use thicker lines in the axes. Try thickness of 2 points.

