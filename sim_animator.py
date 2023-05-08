"""
Animator function to animate one trial from model simulations.
"""


from matplotlib import animation
import matplotlib.pyplot as plt
import numpy as np


"""
states: row vectors of midpoint, left hand and right hand lateral and forward coordinates
target_coords: lateral and forward coordinates of the target
file_name: file name of publishing the animation

"""
def animator(states=np.zeros((6, 100)), target_coords=(0, 0), params={}, background='w', fps=90, file_name='xyz.mp4'):
    anim_figure = plt.figure(figsize = (12, 10))
    anim_figure.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=None, hspace=None)
    anim_ax = plt.axes(xlim = (-0.15,0.15), ylim = (-0.05, 0.2))
    anim_ax.set_facecolor(background)
    anim_figure.patch.set_facecolor(background)
    anim_ax.set_frame_on(False)
    anim_ax.set_aspect('equal')
    anim_ax.axis('off')

    # anim_figure.patch.set_alpha(0.)
    anim_figure.tight_layout()

    anim_ax.plot(target_coords[0], target_coords[1], marker = 'o', markersize = 25, color=params.target_color)
    
    trace_width = 2
    trace_alpha = 0.5
    cursor_size = 12

    left_cursor, = anim_ax.plot([],[], marker = 'o', color = 'tab:orange', markersize = cursor_size)
    right_cursor, = anim_ax.plot([],[], marker = 'o', color = 'tab:blue',  markersize = cursor_size)
    lcursor_trace, = anim_ax.plot([],[], color = 'tab:orange', linewidth=trace_width, alpha=trace_alpha)
    rcursor_trace, = anim_ax.plot([],[], color = 'tab:blue', linewidth=trace_width, alpha=trace_alpha)
    lhand_trace, = anim_ax.plot([],[], color = 'tab:orange', linewidth=trace_width, linestyle='--', alpha=trace_alpha)
    rhand_trace, = anim_ax.plot([],[], color = 'tab:blue', linewidth=trace_width, linestyle='--', alpha=trace_alpha)

    midpoint, = anim_ax.plot([],[], marker = 'o', markersize = cursor_size, color='tab:grey')
    midpoint_trace, = anim_ax.plot([],[], color = 'tab:grey', linewidth=trace_width, alpha=trace_alpha)

    def update(frame):
        left_cursor.set_data(states[6, frame], states[3, frame])
        right_cursor.set_data(states[7, frame], states[5, frame])
        midpoint.set_data(states[0, frame], states[1, frame])
        midpoint_trace.set_data(states[0, :frame], states[1, :frame])
        lhand_trace.set_data(states[2, :frame], states[3, :frame])
        rhand_trace.set_data(states[4, :frame], states[5, :frame])
        lcursor_trace.set_data(states[6, :frame], states[3, :frame])
        rcursor_trace.set_data(states[7, :frame], states[5, :frame])

        return left_cursor, right_cursor, lhand_trace, rhand_trace, midpoint

    Writer = animation.writers['ffmpeg']
    writer = Writer(fps=fps, metadata=dict(artist='Me'), bitrate=3000)

    ani = animation.FuncAnimation(anim_figure, update, frames=states.shape[1], interval=100, blit=True)
    ani.save(file_name, writer = writer)

