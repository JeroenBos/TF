import itertools
from six.moves import xrange
from matplotlib.cbook import iterable
from matplotlib.animation import FuncAnimation


class PushFuncAnimation(FuncAnimation):
    '''
    Makes an animation by repeatedly calling a function ``func``.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
       The figure object that is used to get draw, resize, and any
       other needed events.

    func : callable
       The function to call at each frame.  The first argument will
       be the next value in ``frames``.   Any additional positional
       arguments can be supplied via the ``fargs`` parameter.

       The required signature is::

          def func(frame, *fargs) -> iterable_of_artists:

    init_func : callable, optional
       A function used to draw a clear frame. If not given, the
       results of drawing from the first item in the frames sequence
       will be used. This function will be called once before the
       first frame.

       If ``blit == True``, ``init_func`` must return an iterable of artists
       to be re-drawn.

       The required signature is::

          def init_func() -> iterable_of_artists:

    fargs : tuple or None, optional
       Additional arguments to pass to each call to *func*.

    save_count : int, optional
       The number of values from *frames* to cache.

    interval : number, optional
       Delay between frames in milliseconds.  Defaults to 200.

    repeat_delay : number, optional
       If the animation in repeated, adds a delay in milliseconds
       before repeating the animation.  Defaults to ``None``.

    repeat : bool, optional
       Controls whether the animation should repeat when the sequence
       of frames is completed.  Defaults to ``True``.

    blit : bool, optional
       Controls whether blitting is used to optimize drawing.  Defaults
       to ``False``.

    '''
    def __init__(self, fig, func, init_func=None, fargs=None, save_count=None, **kwargs):
        FuncAnimation.__init__(self, fig, func, None, init_func, fargs, save_count, **kwargs)
        self.__framedata = None

    def _draw_frame(self, frame_index):
        if self.__framedata:
            data = self.__framedata
            self.__framedata = None
            super(FuncAnimation, self)._draw_frame(data)

    def update(self, new_data):
        assert new_data
        self.__framedata = new_data
