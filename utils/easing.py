""" easing.py """
import math


def ease_in_linear(t, b, c, d, s=0):
    """ Linear easing in - accelerating from zero velocity
    Arguments:
        t: input time
        b: start value
        c: change in value
        d: duration
        s: start time
    Returns:
        eased value
    """
    t -= s
    return c * t / d + b


def ease_out_linear(t, b, c, d, s=0):
    """ Linear easing out - decelerating to zero velocity
    Arguments:
        t: input time
        b: start value
        c: change in value
        d: duration
        s: start time
    Returns:
        eased value
    """
    t -= s
    return c * t / d + b


def ease_in_out_linear(t, b, c, d, s=0):
    """ Linear easing in/out - accelerating until halfway, then decelerating
    Arguments:
        t: input time
        b: start value
        c: change in value
        d: duration
        s: start time
    Returns:
        eased value
    """
    t -= s
    return c * t / d + b


def ease_in_sin(t, b, c, d, s=0):
    """ Sinusoidal easing in - accelerating from zero velocity
    Arguments:
        t: input time
        b: start value
        c: change in value
        d: duration
        s: start time
    Returns:
        eased value
    """
    t -= s
    return -c * math.cos(t/d * (math.pi/2)) + c + b


def ease_out_sin(t, b, c, d, s=0):
    """ Sinusoidal easing out - decelerating to zero velocity
    Arguments:
        t: input time
        b: start value
        c: change in value
        d: duration
        s: start time
    Returns:
        eased value
    """
    t -= s
    return c * math.sin(t/d * (math.pi/2)) + b


def ease_in_out_sin(t, b, c, d, s=0):
    """ Sinusoidal easing in/out - accelerating until halfway, then decelerating
    Arguments:
        t: input time
        b: start value
        c: change in value
        d: duration
        s: start time
    Returns:
        eased value
    """
    t -= s
    return -c/2 * (math.cos(math.pi*t/d) - 1) + b


def ease_in_exp(t, b, c, d, s=0):
    """ Exponential easing in - accelerating from zero velocity
    Arguments:
        t: input time
        b: start value
        c: change in value
        d: duration
        s: start time
    Returns:
        eased value
    """
    t -= s
    return c * math.pow(2, 10 * (t/d - 1)) + b


def ease_out_exp(t, b, c, d, s=0):
    """ Expoential easing out - decelerating to zero velocity
    Arguments:
        t: input time
        b: start value
        c: change in value
        d: duration
        s: start time
    Returns:
        eased value
    """
    t -= s
    return c * (-math.pow(2, -10 * t/d) + 1) + b


def ease_in_out_exp(t, b, c, d, s=0):
    """ Exponential easing in/out - accelerating until halfway, then decelerating
    Arguments:
        t: input time
        b: start value
        c: change in value
        d: duration
        s: start time
    Returns:
        eased value
    """
    t -= s
    t = 2 * t/d
    if t < 1:
        return c/2 * math.pow(2, 10 * (t - 1)) + b
    t -= 1
    return c/2 * (-math.pow(2, -10 * t) + 2) + b
