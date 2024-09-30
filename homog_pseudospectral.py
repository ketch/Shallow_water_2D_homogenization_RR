import numpy as np
from scipy.integrate import quad
import matplotlib.pyplot as plt
import matplotlib.animation
from IPython.display import HTML
ifft = np.fft.ifft
fft = np.fft.fft
from ipywidgets import IntProgress
from IPython.display import display
import time

g = 9.81

def bracket(f):
    """
    Returns a function that computes the bracket of a given function f.

    Parameters:
    f (function): The function to compute the bracket of.

    Returns:
    function: A function that computes the bracket of f.
    """
    mean = quad(f,0,1)[0]     # <f>
    brace = lambda y: f(y)-mean  #  {f} = f - <f>
    brack_nzm = lambda y: quad(brace,0,y)[0]  # [[f]] + C
    mean_bracket = quad(brack_nzm,0,1)[0]   # C
    def brack(y):
        return quad(brace,0,y)[0] - mean_bracket  # [[f]]
    return brack

def spectral_representation(x0,uhat,xi):
    """
    Returns a truncated Fourier series representation of a function.

    Parameters:
    x0 (float): The left endpoint of the domain of the function.
    uhat (numpy.ndarray): The Fourier coefficients of the function.
    xi (numpy.ndarray): The vector of wavenumbers.

    Returns:
    u_fun: A vectorized function that represents the Fourier series.
    """
    u_fun = lambda y : np.real(np.sum(uhat*np.exp(1j*xi*(y+x0))))/len(uhat)
    u_fun = np.vectorize(u_fun)
    return u_fun

def fine_resolution(f, n, x, xi):
    """
    Interpolates a periodic function `f` onto a finer grid of `n` points using a Fourier series.

    Parameters:
    -----------
    f : function
        The function to be interpolated.
    n : int
        The number of points in the finer grid.
    x : array-like
        The original grid of `f`.
    xi : array-like
        The Fourier modes.

    Returns:
    --------
    x_fine : array-like
        The finer grid of `n` points.
    f_spectral : function
        The Fourier interpolation `f` on the finer grid.
    """
def fine_resolution(f,n,x,xi):
    fhat = fft(f)
    f_spectral = spectral_representation(x[0],fhat,xi)
    x_fine = np.linspace(x[0],x[-1],n)
    return x_fine, f_spectral(x_fine)

def rkm(u,xi,rhs,dt,du,fy,method,params=None):
    A = method.A
    b = method.b
    for i in range(len(b)):
        y = u.copy()
        for j in range(i):
            y += dt*A[i,j]*fy[j,:,:]
        fy[i,:,:] = rhs(y,du,xi,**params)
    #u_new = u + dt*sum([b[i]*fy[i,:,:] for i in range(len(b))])
    u_new = u + dt*np.sum(b[:,np.newaxis,np.newaxis]*fy, axis=0) # faster
    return u_new


def xxt_rhs(u, du, xi, **params):
    """
    Solves the homogenized equation for a given set of parameters.

    Args:
        u (ndarray): Array of shape (2, N) containing the values of eta and q.
        du (ndarray): Array of shape (2, N) containing the derivatives of eta and q.
        xi (ndarray): Array of shape (N,) containing the values of xi.
        **params: Dictionary containing the values of the parameters 

    Returns:
        ndarray: Array of shape (2, N) containing the derivatives of eta and q.
    """
    delta, Havg, mu = params['delta'], params['Havg'], params['mu']

    eta = u[0,:]
    q   = u[1,:]
    etahat = fft(eta)
    qhat = fft(q)
    
    eta_x = np.real(ifft(1j*xi*etahat))
    q_x = np.real(ifft(1j*xi*qhat))
    csq = g*Havg
    
    deta = -q_x - (delta/Havg)*(eta*q_x + q*eta_x)
    dq = -csq*eta_x - (delta/Havg)*q*q_x
    dqhat = fft(dq)
    dq = np.real(ifft(dqhat/( 1 - xi**2*delta**2*(-mu/Havg) )))

    du[0,:] = deta
    du[1,:] = dq
    return du


def homogenized_coefficients(btype="pwc",b_amp=1.0,eps=1.e-7):
    """
    Computes homogenized coefficients for a given periodic bathymetry profile.

    Parameters:
    H (function): A function that takes a single argument (y) and returns the unperturbed water depth at that point.
    btype (str): The type of bathymetry. Can be either 'pwc' (piecewise-constant) or 'sinusoidal'.
    b_amp (float, optional): The amplitude of the bathymetry variation. Required if btype is 'pwc'.
    delta (float, optional): The period of the bathymetry.
    eps (float, optional): The desired accuracy of the numerical integration.

    Returns:
    dict: A dictionary containing the homogenized coefficients.

    I ought to just rewrite this function so the input is the bathymetry function or H.
    """
    params = {}

    params['delta'] = 1.0
    params['Havg'] = 1.0

    if btype == "pwc":
        Ha = 1 + 0.5*b_amp
        Hb = 1 - 0.5*b_amp
        params['mu'] =  b_amp**2/(192*Ha*Hb)
    elif btype == "sinusoidal":
        b = lambda y : -1. + b_amp*np.sin(2*np.pi*y)
        H = lambda y : -b(y)  # \eta_0 = 0
        bH = bracket(H); bH = np.vectorize(bH)
        HibH2 = lambda y : (bH(y))**2 / H(y)
        params["mu"] = quad(HibH2,0,1,epsabs=eps,epsrel=eps)[0]
    elif btype == "experimental":
        Ha = 0.05
        Hb = 0.55
        #params['delta'] = 0.6
        params['Havg'] = (Ha+Hb)/2.
        params['mu'] =  (Ha-Hb)**2*(Ha+Hb)/(384*Ha*Hb)

    return params


def solve_SWH(h_amp=0.15, b_amp=1.0, width=5.0, btype='pwc',IC='pulse',L=200,tmax=100.,m=256, dtfac=0.5,
                make_anim=True,skip=128,num_plots=100):
    """
    Solve the homogenized SW equations using Fourier spectral collocation in space
    and SSPRK3 in time, on the domain (-L/2,L/2).

    Parameters:
    -----------
    h_amp : float, optional
        Amplitude of the initial wave. Default is 0.3.
    b_amp : float, optional
        Amplitude of the bathymetry. Default is 0.5.
    width : float, optional
        Width of the initial wave. Default is 5.0.
    btype : str, optional
        Type of bathymetry. Can be 'sinusoidal' or 'pwc' (piecewise-constant). Default is 'pwc'.
    IC : str, optional
        Type of initial condition. Can be 'pulse', 'step', or 'data'. Default is 'pulse'.
    L : float, optional
        Length of the domain. Default is 200.
    tmax : float, optional
        Maximum time to run the simulation. Default is 100.
    m : int, optional
        Number of grid points. Default is 256.
    dtfac : float, optional
        Time step factor. Default is 0.05.
    make_anim : bool, optional
        Whether to create an animation of the simulation. Default is True.
    skip : int, optional
        Number of data points to skip when using 'data' initial condition. Default is 128.

    Returns:
    --------
    x : numpy.ndarray
        Array of grid points.
    xi : numpy.ndarray
        Array of wavenumbers.
    momentum : list
        List of arrays of momentum at each time step.
    eta : list
        List of arrays of surface elevation at each time step.
    anim : matplotlib.animation.FuncAnimation or None
        Animation of the simulation, if make_anim is True. Otherwise, None.
    """
    params = homogenized_coefficients(btype=btype,b_amp=b_amp)

    # Grid
    x = np.arange(-m/2,m/2)*(L/m)
    xi = np.fft.fftfreq(m)*m*2*np.pi/L

    from nodepy import rk
    method = rk.loadRKM('BS5').__num__()
    dt = dtfac * 1.73/np.max(xi)
    fy = np.zeros((len(method),2,m))

    
    q0 = np.zeros_like(x)
    if IC == "pulse":
        eta0 = h_amp * np.exp(-x**2 / width**2)
    elif IC == "step":
        eta0 = h_amp*(x<0)
    else:
        data = np.loadtxt(IC)
        data = data[::skip,:]
        x = data[:,0]
        eta0 = data[:,1]
        q0 = data[:,2]
        print(m, len(x))
        assert(m==len(x))
        L = x[-1]-x[0]+x[1]-x[0]
        xi = np.fft.fftfreq(m)*m*2*np.pi/L
        
    u = np.zeros((2,len(x)))

    u[0,:] = eta0
    u[1,:] = q0

    du = np.zeros_like(u)

    plot_interval = tmax/num_plots
    steps_between_plots = int(round(plot_interval/dt))
    dt = plot_interval/steps_between_plots
    nmax = num_plots*steps_between_plots

    fig = plt.figure(figsize=(12,8))
    axes = fig.add_subplot(111)
    line, = axes.plot(x,u[0,:],lw=2)
    xi_max = np.max(np.abs(xi))
    axes.set_xlabel(r'$x$',fontsize=30)
    plt.close()

    eta = [u[0,:].copy()]
    momentum = [u[1,:].copy()]
    tt = [0]
    
    xi_max = np.max(np.abs(xi))

    f = IntProgress(min=0, max=num_plots) # instantiate the bar
    display(f) # display the bar


    for n in range(1,nmax+1):
        u_new = rkm(u,xi,xxt_rhs,dt,du,fy,method,params)
            
        u = u_new.copy()
        t = n*dt

        # Plotting
        if np.mod(n,steps_between_plots) == 0:
            f.value += 1
            eta.append(u[0,:].copy())
            momentum.append(u[1,:].copy())
            tt.append(t)
        
    def plot_frame(i):
        etahat = np.fft.fft(eta[i])
        eta_spectral = spectral_representation(x[0],etahat,xi)
        x_fine = np.linspace(x[0],x[-1],5000)
        line.set_data(x_fine,eta_spectral(x_fine))
        axes.set_title('t= %.2e' % tt[i])

    if make_anim:
        anim = matplotlib.animation.FuncAnimation(fig, plot_frame,
                                           frames=len(eta), interval=100)
        anim = HTML(anim.to_jshtml())
    else:
        anim = None

    return x, xi, momentum, eta, anim


def eta_variation(x,eta,momentum,btype,b_amp):
    m = len(x)
    L = x[-1]-x[0]
    xi = np.fft.fftfreq(m)*m*2*np.pi/L
    delta = 1
    eps = 1e-7
    if btype == 'sinusoidal':
        b = lambda y: -1 + b_amp * np.sin(2*np.pi*y)
    elif btype == 'tanh':
        s = 2000 # Smoothing parameter
        b = lambda y: -1+b_amp*(1+(np.tanh(s*(y-0.5))*(-np.tanh(s*(y-1.)))))/2
    elif btype == 'pwc': # piecewise-constant
        b = lambda y: -1 + b_amp*((y-np.floor(y))>0.5)

    eta0 = 0.
    H = lambda y: eta0-b(y)
    ih1 = lambda y: 1/H(y)
    ih2 = lambda y: 1/H(y)**2
    b1 = bracket(ih1); b1 = np.vectorize(b1)
    b2 = bracket(ih2); b2 = np.vectorize(b2)
    if btype == 'tanh': # should be pwc
        gam1 = 1.; gam2 = 1/(1-b_amp)
        bb1 = lambda y: ((y<=0.5)*((gam2-gam1)*y+2*y**2*(gam1-gam2)) + (y>0.5)*(-gam1+3*gam1*y-2*y**2*gam1+gam2-3*y*gam2+2*y**2*gam2))/8
        bb1 = np.vectorize(bb1)
        C3 = -(gam1-gam2)**2/192
    else:
        bb1 = bracket(b1); bb1 = np.vectorize(bb1)
        C3f = lambda y: b1(y)**2
        C3 = -quad(C3f,0,1,epsabs=eps,epsrel=eps)[0]

    H1 = quad(lambda y: 1/H(y),0,delta,epsabs=eps,epsrel=eps)[0]
    H2 = quad(lambda y: 1/H(y)**2,0,delta,epsabs=eps,epsrel=eps)[0]
    H3 = quad(lambda y: 1/H(y)**3,0,delta,epsabs=eps,epsrel=eps)[0]
    braceH2 = lambda y: 1/H(y)**2 - H2; braceH2 = np.vectorize(braceH2)
    braceH3 = lambda y: 1/H(y)**3 - H3; braceH3 = np.vectorize(braceH3)
    
    xx = x - np.floor(x)

    q = momentum
    etahat = fft(eta)
    qhat = fft(q)
    eta_x = np.real(ifft(1j*xi*etahat))
    eta_xx  = np.real(ifft(-xi**2*etahat))
    eta_xxx = np.real(ifft(-1j*xi**3*etahat))
    q_x   = np.real(ifft(1j*xi*qhat))

    eta2 = b1(xx)/H1 * eta_x - q**2 * braceH2(xx)/(2*g)
    eta2 += q**2*eta*braceH3(xx)/g - q*q_x*(2*b1(xx)*H2/H1-b2(xx))/g
    eta2 += eta*eta_x*(b1(xx)*H2/H1-b2(xx))/H1
    eta2 -= bb1(xx)*eta_xx/H1
    eta2 -= C3*b1(xx)*eta_xxx/H1**3
    return eta2
