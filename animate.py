
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from scipy.interpolate import interp1d

def render_anim(data, update, save = None, num = None, dt = 1, show = True):

    fig = plt.figure()
    ax = plt.axes()
    t = data[0,0]
    x = data[:,1]
    y = data[:,2]  
    u = data[:,3] 
    v = data[:,4] 
    c = u**2 + v**2 
    field = ax.quiver(x, y, u, v, c, pivot = 'mid', cmap = 'seismic', scale = 100)#, headaxislength = 0, headlength = 0)
    T_text = ax.text(0.05, 1.01, ' ', transform=ax.transAxes, fontsize = 16, color = 'k')


    # animation function.  This is called sequentially
    def animate(i, field, T_text):
        data = update()
        t = data[0,0]
        u = data[:,3] 
        v = data[:,4] 
        c = u**2 + v**2 
        # ax.remove()
        field.set_UVC(u, v, c)
        T_text.set_text('time: {:.2f} steps'.format(t))
        return field, T_text

    if num:
        frames = num
    else:
        frames = 2000
        # call the animator.  blit=True means only re-draw the parts that have changed.
    anim = animation.FuncAnimation(fig, animate,
                                frames=frames, interval=dt*1000, blit=False, fargs=(field, T_text))

        # save the animation as an mp4.  This requires ffmpeg or mencoder to be
        # installed.  The extra_args ensure that the x264 codec is used, so that
        # the video can be embedded in html5.  You may need to adjust this for
        # your system: for more information, see
        # http://matplotlib.sourceforge.net/api/animation_api.html
    if show:
        plt.show()
    if save is not None:
        print('saving...')
        anim.save(save, fps=20, extra_args=['-vcodec', 'libx264'], dpi = 300)
        print('saved')
    
