import itertools
import numpy as np
import sympy as sy
from sympy import symbols
from sympy.plotting import plot, plot3d, PlotGrid
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Polygon
from scipy import integrate
from mpl_toolkits.mplot3d import axes3d
from matplotlib.colors import ListedColormap
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.collections import LineCollection
from matplotlib import patheffects
from matplotlib import cm
from colorspacious import cspace_converter
from xarray import DataArray

# =============================================================================
# Derivada
# =============================================================================

y = lambda x: x**2 - 5*x + 6

def derivada_direita(funcao, a, h=0.01):
    return (funcao(a+h) - funcao(a))/h

def derivada_esquerda(funcao, a, h=0.01):
    return (funcao(a) - funcao(a-h))/h

# y'(x) = 2x - 5

derivada_direita(y, 0) #-4.9900
derivada_esquerda(y, 0) #-5.0099

# =============================================================================
# Derivada Parcial
# =============================================================================

def derivada_parcial(f, x: float, y: float, h: float = 0.01):
    return (f(x + h / 2, y + h / 2) - f(x - h / 2, y + h / 2) - f(x + h / 2, y - h / 2) +
            f(x - h / 2, y - h / 2)) / (h ** 2)

f = lambda x, y: x * np.sin(y)

derivada_parcial(f, 0, np.pi/4) # 0.7071

# =============================================================================
# Método Central
# =============================================================================

y = lambda x: x**2 - 5*x + 6

def derivada_central(f, a: float, h: float = 0.01) -> float:
    return(f(a+h/2) - f(a-h/2)) / h

derivada_central(y, 0) #-5.0000

# =============================================================================
# Segunda Derivada
# =============================================================================

def segunda_derivada(f, a: int, h: float = 0.01) -> float:
    """ Return the second derivative formula """
    return (f(a+h) - 2 * f(a) + f(a-h)) / (h ** 2)


y = lambda x: x ** 2 - 5 * x + 6

segunda_derivada(y, 0) # 1.9999

# =============================================================================
# Integral Dupla
# =============================================================================

f = lambda x, y: x * y

integral = integrate.dblquad(f, 0, 1, lambda x: 0, lambda x: 1)

round(integral[0], 2) # 0.25

# =============================================================================
# Área embaixo da Cuva
# =============================================================================

x = np.linspace(0, 5, 100)
y = x ** 2 - 5 * x + 6

fig, (ax1, ax2) = plt.subplots(1, 2, sharex=True, sharey=True)
# Create a continuous norm to map from data points to colors

ax1.plot(x, y, label='F(x) = x^2 - 5x + 6',color='#00ced1')
ax1.set_xlabel('x')
ax1.set_ylabel('f(x)')
ax1.legend()
colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
cmap = ListedColormap(colors[:len(np.unique(y))])
ax2.fill_between(x, 0, y, 
                 label='I(x) = x^3/3 - 5x^2/2 + 6x',
                 color='#00ced1') 
ax2.set_xlabel('x')
ax2.set_ylabel('f(x)')
ax2.legend()
plt.show()

# =============================================================================
# Cuva
# =============================================================================

x = np.linspace(0, 5, 100)
def f(x):
    return x ** 2 - 5 * x + 6
y = f(x)

def multicolored_line(x, y):
    
    # Points and segments for the LineCollection
    points = np.array([x, y]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    
    # Cmap color
    viridis=cm.get_cmap('viridis', 16)
    color = (viridis(range(0,16)))
    cmap = ListedColormap(color)
    
    # Fig
    fig, ax = plt.subplots()
    # Create the line collection object, setting the colormapping parameters.
    lc = LineCollection(segments, cmap=cmap)
    lc.set_array(y)
    lc.set_linewidth(3)
    line = ax.add_collection(lc)
    fig.colorbar(line, ax=ax, label='Units')
    ax.set_xlabel('x')
    ax.set_ylabel('f(x)')
    ax.legend(title='F(x) = x^2 - 5x + 6', title_fontsize = 10, bbox_to_anchor= (1, 1))
    # ax.legend('F(x) = x^2 - 5x + 6')
    
    # Scale and Limits
    plt.xlim(x.min(), x.max())
    plt.ylim(-1, 6)
    plt.show()
      
    return lc

multicolored_line(x, y)

# =============================================================================
# Área entre duas curvas
# =============================================================================
def f(x):
    return x**2

def g(x):
    return x**(1/2)  
  
x = sy.Symbol("x")
print(sy.integrate(f(x)-g(x), (x, 0, 2)))

fig, ax = plt.subplots()

x = np.linspace(0, 2, 1000)
ax.plot(x, f(x), label="f(x)", color='#0303b5')
ax.plot(x,g(x), label="g(x)", color='#131c46')
ax.legend()
plt.fill_between(x, f(x), g(x), where=[(x > 0) and (x < 2) for x in x],color='#00ced1')
plt.show()

# =============================================================================
# Gráfico com Texto 
# =============================================================================
# Cmap color
viridis=cm.get_cmap('viridis', 16)
color = (viridis(range(0,16)))
cmap = ListedColormap(color)

#  Fuction
def func(x):
    return (x - 3) * (x - 5) * (x - 7) + 85

a, b = 2, 9  # integral limits
x = np.linspace(0, 10)
y = func(x)

# Point and Segments for the Cmap
points = np.array([x, y]).T.reshape(-1, 1, 2)
segments = np.concatenate([points[:-1], points[1:]], axis=1)

# Fiz and Axis
fig, ax = plt.subplots()

ax.plot(x, y)
ax.set_ylim(bottom=0)
lc = LineCollection(segments, cmap=cmap)
lc.set_array(y)
lc.set_linewidth(3)
line = ax.add_collection(lc)
fig.colorbar(line, ax=ax)

# Make the shaded region
ix = np.linspace(a, b)
iy = func(ix)
verts = [(a, 0), *zip(ix, iy), (b, 0)]
poly = Polygon(verts, facecolor='0.9', edgecolor='0.5')
ax.add_patch(poly)

# Text Math
ax.text(0.5 * (a + b), 30, r"$\int_a^b f(x)\mathrm{d}x$",
        horizontalalignment='center', fontsize=20)
fig.text(0.9, 0.05, '$x$')
fig.text(0.1, 0.9, '$y$')

# Ticks
ax.set_xticks([a, b], labels=['$a$', '$b$'])
ax.set_yticks([])

plt.show()

# =============================================================================
# Gráfico em 3D
# =============================================================================
fig = plt.figure()
ax = fig.add_subplot(projection='3d')

def f(x,y):
    return np.sin(x)+np.cos(y) # fuction waves
    # return np.sin(np.sqrt(x ** 2 + y ** 2)) #fuction flower

x = np.linspace(-6,6,30)
y = np.linspace(-6,6,30)
X, Y = np.meshgrid(x,y)
Z = f(X,Y)

ax.plot_wireframe(X, Y, Z, color='green')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
    
surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1,
                    cmap='viridis', edgecolor='none')
# fig.colorbar(surf, ax = ax, shrink = 0.7, aspect = 7) # colorbar
ax.set_title('Gráfico de superfícies');
    
# rotate the axes and update
def rotate(angle):
    ax.view_init(azim=angle)
    plt.draw()

print("Making animation")
rot_animation = animation.FuncAnimation(fig, rotate, frames=np.arange(0, 360, 2), interval=100)

# GIF
f = "rotation.gif" 
writergif = animation.PillowWriter(fps=1080) 
rot_animation.save(f, writer=writergif)
