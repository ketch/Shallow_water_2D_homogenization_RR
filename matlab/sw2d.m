% 2D shallow water in primitive variable,
% wih periodic bathimetry in the y-direction
global delta kx ky B B_y g X Y x

delta = 1;
py = 4;
px = py+11;
ny = 32;% 2^py;
nx = 32000; %2^px;
dy = delta/ny;
L = 1000; %nx*dx/2;
dx = 2*L/nx; %dy;
y = 0:dy:delta-dy;
x = -L:dx:L-dx;
save("x32.txt","-ascii","x")
[X,Y] = meshgrid(x,y);

% Initialization
g = 9.81; %Gravitational acceleration in m/s^2
[h,u,v,B,Eta] = initial(X,Y);

% Define the Fourier kx and ky
ky = -ny/2:ny/2-1;
ky = fftshift(ky);
ky = ky*2*pi/delta;

kx = -nx/2:nx/2-1;
kx = fftshift(kx);
kx = kx*2*pi/(2*L);

ky = ky(:);
kx = kx(:);

B_y = dery(B,ky);

%h_x = real(ifft(1i*fft(h').*kx))';
h_x = derx(h,kx);
h_x_exa = Eta.*(-2/25.*X);

% Start cycle
t = 0;
tmax = 300;
U = zeros(ny,nx,3);
U(:,:,1) = h;
U(:,:,2) = u;
U(:,:,3) = v;
CFL = 0.7;
nt = 0;
dt = 0.005;
E0 = Energy(U)

path = "./32k/"

while t<tmax
    % c = rhomax(U);  % Adaptive time stepping
    %dt = CFL*dx/c;
    if mod(nt,100)==0
        view(t,U,0);
    end
    if mod(nt, 1000)==0
        h = U(:,:,1);
        Eta = h+B;
        Etam = mean(Eta);
        save(path+"eta_"+string(round(nt*dt))+".txt","-ascii","Etam")
        hv = U(:,:,1).*U(:,:,3);
        slice1 = hv(8,:);
        slice2 = hv(16,:);
        %save(path+"hv1_"+string(round(nt*dt))+".txt","-ascii","slice1")
        %save(path+"hv2_"+string(round(nt*dt))+".txt","-ascii","slice2")
    end
    U_new = RK4(U,dt);

    U = U_new;
    t = t+dt;
    nt = nt+1;
end

function E = Energy(U)
    global g B
    h = U(:,:,1);
    u = U(:,:,2);
    v = U(:,:,3);
    E = sum(sum(0.5*h.*(u.^2 + v.^2) + 0.5*g*h.^2 + g*h.*B));
end

function Udot = func(U)
    global kx ky g B_y
    h = U(:,:,1);
    u = U(:,:,2);
    v = U(:,:,3);
    h_x = derx(h,kx);
    h_y = dery(h,ky);
    u_x = derx(u,kx);
    u_y = dery(u,ky);
    v_x = derx(v,kx);
    v_y = dery(v,ky);
    Udot = zeros(size(U));
    Udot(:,:,1) = -h.*u_x - u.*h_x - h.*v_y - v.*h_y;
    Udot(:,:,2) = -u.*u_x - v.*u_y - g*h_x;
    Udot(:,:,3) = -u.*v_y - v.*v_y - g*h_y - g*B_y;
end


function u_new = RK4(u,dt)
    k1 = func(u);
    k2 = func(u+dt/2*k1);
    k3 = func(u+dt/2*k2);
    k4 = func(u+dt*k3);
    u_new = u+dt*(k1+2*k2+2*k3+k4)/6;
end

function u_x = derx(u,k)
    u_x = real(ifft(1i*fft(u').*k))';
end

function u_y = dery(u,k)
    u_y = real(ifft(1i*fft(u).*k));
end

function [h,u,v,B,Eta] = initial(X,Y);
    global delta
    amplib = 0.3;
    B = -1+amplib*sin(2*pi*Y/delta);
    %B = -8/5+6/5*sin(2*pi*Y/delta);
    Eta = exp(-(X/5).^2)/20;
    h = Eta - B;
    u = zeros(size(Eta));
    v = zeros(size(Eta));
end

function view(t,U,iop)
    global X Y B x
    h = U(:,:,1);
    u = U(:,:,2);
    v = U(:,:,3);
    Eta = h+B;
    if iop==0
        Etam = mean(Eta);
        figure(6)
        plot(x,Etam)
        title(['t = ',num2str(t)])
        ylabel \eta
        xlabel x
        drawnow
    else
        figure(5)
        subplot(3,1,1)
        mesh(X,Y,Eta)
        xlabel x
        ylabel y
        zlabel h
        title(['t = ',num2str(t)])
        subplot(3,1,2)
        mesh(X,Y,u)
        xlabel x
        ylabel y
        zlabel u
        subplot(3,1,3)
        mesh(X,Y,v)
        xlabel x
        ylabel y
        zlabel v
        drawnow
    end
end


function c = rhomax(U)
    global g
    h = U(:,:,1);
    u = U(:,:,2);
    v = U(:,:,3);
    c = max(max(sqrt(u.^2+v.^2)+sqrt(g*h)));
end


